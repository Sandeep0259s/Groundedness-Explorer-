from pathlib import Path

from .config import settings
from .hallucination import get_scorer
from .llm import OllamaLLM
from .reranker import get_reranker
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
        candidates = self.store.query(question, top_k=candidate_k, label=label)
        hits = self.reranker.rerank(question, candidates, top_k)
        context_chunks = [hit["text"] for hit in hits]

        if not context_chunks:
            return {
                "answer": "No documents have been ingested yet, so I have no context to answer from.",
                "sources": [],
                "groundedness": {"overall_score": 0.0, "label": "unknown", "sentences": []},
            }

        answer = self._generate(question, hits, context_chunks, history=history, model=model)
        groundedness = self.scorer.score(answer, context_chunks)

        return {
            "answer": answer,
            "sources": hits,
            "groundedness": groundedness,
        }

    def _generate(
        self,
        question: str,
        hits: list[dict],
        context_chunks: list[str],
        history: list[dict] | None,
        model: str | None,
    ) -> str:
        # If the single best-matching source is an image, answer by actually
        # looking at it through a vision model rather than only its cached
        # OCR/caption text — genuine visual QA, not just retrieval over a
        # canned description. Falls back to the normal text path if no
        # vision-capable model is available.
        top_source = Path(hits[0]["source"])
        if top_source.suffix.lower() in IMAGE_SUFFIXES:
            try:
                return get_vision_model().answer(top_source, question, context_chunks[1:])
            except VisionUnavailable:
                pass

        return self.llm.generate(question, context_chunks, history=history, model=model)
