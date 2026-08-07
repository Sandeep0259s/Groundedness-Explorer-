from collections.abc import Iterator
from pathlib import Path

from .config import settings
from .hallucination import get_scorer
from .llm import OllamaLLM
from .reranker import get_reranker
from .structured_qa import SPREADSHEET_SUFFIXES, answer_structured_question
from .vectorstore import VectorStore
from .vision import IMAGE_SUFFIXES, VisionUnavailable, get_vision_model


class RAGPipeline:
    def __init__(self):
        self.store = VectorStore()
        self.llm = OllamaLLM()
        self.scorer = get_scorer()
        self.reranker = get_reranker()

    def ask(
        self,
        question: str,
        top_k: int = settings.top_k,
        label: str | None = None,
        history: list[dict] | None = None,
        model: str | None = None,
    ) -> dict:
        candidate_k = max(top_k * settings.rerank_candidate_multiplier, top_k)
        candidates = self.store.query_hybrid(question, top_k=candidate_k, label=label)
        hits = self.reranker.rerank(question, candidates, top_k)
        context_chunks = [hit["text"] for hit in hits]

        if not context_chunks:
            return {
                "answer": "No documents have been ingested yet, so I have no context to answer from.",
                "sources": [],
                "groundedness": {"overall_score": 0.0, "label": "unknown", "sentences": []},
                "answer_mode": "text",
            }

        answer, answer_mode = self._generate(question, hits, context_chunks, history=history, model=model)
        groundedness = self._score(answer, answer_mode, context_chunks)

        return {
            "answer": answer,
            "sources": hits,
            "groundedness": groundedness,
            "answer_mode": answer_mode,
        }

    def _generate(
        self,
        question: str,
        hits: list[dict],
        context_chunks: list[str],
        history: list[dict] | None,
        model: str | None,
    ) -> tuple[str, str]:
        """Returns (answer, answer_mode). answer_mode is one of:
        - "vision": answered by actually looking at the top-matched image
        - "vision_fallback": top match was an image but the vision model
          gave no usable answer, so the text LLM answered from its caption
        - "structured": answered by computing a pandas expression over the
          top-matched spreadsheet, not by retrieving a text fragment
        - "text": none of the above — the ordinary text-only path
        """
        # If the single best-matching source is an image, answer by actually
        # looking at it through a vision model rather than only its cached
        # OCR/caption text — genuine visual QA, not just retrieval over a
        # canned description. Small vision models can be brittle and return
        # an empty/near-empty response for terse factual questions even
        # though they caption the same image fine — when that happens, fall
        # back to the text LLM, whose context already includes a rich
        # ingest-time caption of the image and often still answers it.
        top_source = Path(hits[0]["source"])
        if top_source.suffix.lower() in IMAGE_SUFFIXES:
            # Only pass *non-image* context alongside the picture itself —
            # another retrieved image's caption describes a different photo
            # entirely, and handing it to the vision model as "context"
            # measurably confused it into blending details from both images.
            extra_context = [
                hit["text"] for hit in hits[1:] if Path(hit["source"]).suffix.lower() not in IMAGE_SUFFIXES
            ]
            try:
                answer = get_vision_model().answer(top_source, question, extra_context)
                if len(answer.strip()) >= 3:
                    return answer, "vision"
            except VisionUnavailable:
                pass
            return self.llm.generate(question, context_chunks, history=history, model=model), "vision_fallback"

        # A spreadsheet row flattened to text can't answer "what's the total
        # in column X" — that needs an actual aggregation over the real
        # data, not retrieval over a fragment of it.
        if top_source.suffix.lower() in SPREADSHEET_SUFFIXES:
            structured = answer_structured_question(top_source, question, model)
            if structured:
                return structured, "structured"

        return self.llm.generate(question, context_chunks, history=history, model=model), "text"

    def _score(self, answer: str, answer_mode: str, context_chunks: list[str]) -> dict:
        if answer_mode == "structured":
            # NLI entailment compares two natural-language sentences — a
            # bare computed number like "700" against a prose chunk isn't a
            # meaningful entailment pair, and scoring it that way produced a
            # consistently near-zero "possibly hallucinated" score for
            # answers that were, in fact, deterministically computed from
            # the actual data. The real risk here is a wrong *translation*
            # of the question into code, not a hallucinated fact — a
            # different failure mode the NLI scorer can't measure at all.
            return {"overall_score": 1.0, "label": "computed", "sentences": []}
        return self.scorer.score(answer, context_chunks)

    def ask_stream(
        self,
        question: str,
        top_k: int = settings.top_k,
        label: str | None = None,
        history: list[dict] | None = None,
        model: str | None = None,
    ) -> Iterator[tuple[str, object]]:
        """Same retrieval + answer routing as ask(), but yields the answer as
        it's generated instead of only returning it once complete.

        Yields ("token", str) for each piece of text, then exactly one
        ("done", dict) with the same shape ask() returns (answer, sources,
        groundedness, answer_mode) once generation finishes.
        """
        candidate_k = max(top_k * settings.rerank_candidate_multiplier, top_k)
        candidates = self.store.query_hybrid(question, top_k=candidate_k, label=label)
        hits = self.reranker.rerank(question, candidates, top_k)
        context_chunks = [hit["text"] for hit in hits]

        if not context_chunks:
            yield "done", {
                "answer": "No documents have been ingested yet, so I have no context to answer from.",
                "sources": [],
                "groundedness": {"overall_score": 0.0, "label": "unknown", "sentences": []},
                "answer_mode": "text",
            }
            return

        answer_mode = "text"
        collected: list[str] = []

        top_source = Path(hits[0]["source"])
        if top_source.suffix.lower() in IMAGE_SUFFIXES:
            extra_context = [
                hit["text"] for hit in hits[1:] if Path(hit["source"]).suffix.lower() not in IMAGE_SUFFIXES
            ]
            try:
                for piece in get_vision_model().answer_stream(top_source, question, extra_context):
                    collected.append(piece)
                    yield "token", piece
            except VisionUnavailable:
                pass

            # A vision model that can't answer this kind of question tends to
            # emit nothing at all (see hallucination.py-style brittleness
            # notes in vision.py) rather than a wrong partial answer — so if
            # nothing came out, no tokens have reached the caller yet and
            # it's safe to fall back to the text path from scratch.
            if "".join(collected).strip():
                answer_mode = "vision"
            else:
                collected = []
                answer_mode = "vision_fallback"
                for piece in self.llm.generate_stream(question, context_chunks, history=history, model=model):
                    collected.append(piece)
                    yield "token", piece
        elif top_source.suffix.lower() in SPREADSHEET_SUFFIXES and (
            structured := answer_structured_question(top_source, question, model)
        ):
            # Not meaningfully streamable — it's a single computed result,
            # not generated token-by-token — so it arrives as one piece.
            collected.append(structured)
            yield "token", structured
            answer_mode = "structured"
        else:
            for piece in self.llm.generate_stream(question, context_chunks, history=history, model=model):
                collected.append(piece)
                yield "token", piece

        answer = "".join(collected)
        groundedness = self._score(answer, answer_mode, context_chunks)
        yield "done", {
            "answer": answer,
            "sources": hits,
            "groundedness": groundedness,
            "answer_mode": answer_mode,
        }
