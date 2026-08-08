from collections.abc import Iterator
from pathlib import Path

from .config import settings
from .hallucination import get_scorer
from .llm import OllamaLLM
from .query_rewrite import rewrite_for_retrieval
from .reranker import get_reranker
from .structured_qa import SPREADSHEET_SUFFIXES, answer_structured_question
from .vectorstore import VectorStore
from .vision import IMAGE_SUFFIXES, VisionUnavailable, get_vision_model

# A vision model that can't answer a question tends to emit nothing at all
# rather than a wrong partial answer (see vision.py's brittleness notes) —
# so "non-empty" is the right bar for "usable", not an arbitrary length.
_MIN_USABLE_VISION_ANSWER_CHARS = 1


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
        """Same retrieval + answer routing as ask_stream(), just collected
        into one return value instead of yielded incrementally. Built as a
        thin wrapper around ask_stream() on purpose — two independent
        implementations of the same image/spreadsheet/text dispatch logic
        had already silently drifted once (a vision-answer-usable threshold
        that disagreed between the streaming and non-streaming paths for
        the same question) before this was unified into one path."""
        final_result = None
        for kind, payload in self.ask_stream(question, top_k=top_k, label=label, history=history, model=model):
            if kind == "done":
                final_result = payload
        return final_result

    def ask_stream(
        self,
        question: str,
        top_k: int = settings.top_k,
        label: str | None = None,
        history: list[dict] | None = None,
        model: str | None = None,
    ) -> Iterator[tuple[str, object]]:
        """Yields ("token", str) for each piece of text as it's generated,
        then exactly one ("done", dict) with the full result (answer,
        sources, groundedness, answer_mode, retrieval_question) once
        generation finishes. For routes that don't stream naturally
        (structured QA's single computed result), the whole answer arrives
        as one "token" piece rather than being faked into a delay.
        """
        # Retrieval never sees conversation history on its own — a follow-up
        # like "which continent is it on?" has no signal about what "it" is
        # once embedded/BM25-searched in isolation. Rewriting it into a
        # standalone question *for retrieval only* fixes that without
        # changing what generation sees (still the original question + full
        # history, below).
        retrieval_question = rewrite_for_retrieval(question, history, model)

        candidate_k = max(top_k * settings.rerank_candidate_multiplier, top_k)
        candidates = self.store.query_hybrid(retrieval_question, top_k=candidate_k, label=label)
        hits = self.reranker.rerank(retrieval_question, candidates, top_k)
        context_chunks = [hit["text"] for hit in hits]

        if not context_chunks:
            yield "done", {
                "answer": "No documents have been ingested yet, so I have no context to answer from.",
                "sources": [],
                "groundedness": {"overall_score": 0.0, "label": "unknown", "sentences": []},
                "answer_mode": "text",
                "retrieval_question": None,
            }
            return

        answer_mode = "text"
        collected: list[str] = []

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
                for piece in get_vision_model().answer_stream(top_source, question, extra_context):
                    collected.append(piece)
                    yield "token", piece
            except VisionUnavailable:
                pass

            if len("".join(collected).strip()) >= _MIN_USABLE_VISION_ANSWER_CHARS:
                answer_mode = "vision"
            else:
                # A vision model that can't answer this kind of question
                # tends to emit nothing at all (see vision.py's brittleness
                # notes) rather than a wrong partial answer — so if nothing
                # usable came out, no tokens have reached the caller yet and
                # it's safe to fall back to the text path from scratch.
                collected = []
                answer_mode = "vision_fallback"
                for piece in self.llm.generate_stream(question, context_chunks, history=history, model=model):
                    collected.append(piece)
                    yield "token", piece
        elif top_source.suffix.lower() in SPREADSHEET_SUFFIXES and (
            structured := answer_structured_question(top_source, question, model)
        ):
            # A spreadsheet row flattened to text can't answer "what's the
            # total in column X" — that needs an actual aggregation over the
            # real data, not retrieval over a fragment of it. Not
            # meaningfully streamable either way — it's a single computed
            # result, not generated token-by-token — so it arrives as one piece.
            collected.append(structured)
            yield "token", structured
            answer_mode = "structured"
        else:
            for piece in self.llm.generate_stream(question, context_chunks, history=history, model=model):
                collected.append(piece)
                yield "token", piece

        answer = "".join(collected)
        groundedness = self._score(answer, answer_mode, hits, context_chunks)
        yield "done", {
            "answer": answer,
            "sources": hits,
            "groundedness": groundedness,
            "answer_mode": answer_mode,
            "retrieval_question": retrieval_question if retrieval_question != question else None,
        }

    def _score(self, answer: str, answer_mode: str, hits: list[dict], context_chunks: list[str]) -> dict:
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

        # The reranker already judges each candidate's relevance to the
        # question (that's its whole job) — if it found *nothing* relevant
        # (e.g. a greeting like "hi" retrieves only unrelated documents),
        # the answer wasn't really "about the documents" in the first
        # place. Scoring it via NLI against irrelevant chunks anyway
        # produces a misleading "possibly hallucinated" label for what's
        # actually just an off-topic reply — report "no relevant context
        # found" instead, the same honest label already used when nothing
        # is retrieved at all.
        best_relevance = max((h.get("rerank_score", 0.0) for h in hits), default=0.0)
        if best_relevance < settings.relevance_threshold:
            return {"overall_score": 0.0, "label": "unknown", "sentences": []}

        return self.scorer.score(answer, context_chunks)
