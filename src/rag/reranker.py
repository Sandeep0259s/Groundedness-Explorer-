from functools import lru_cache

from sentence_transformers import CrossEncoder

from .config import settings
from .device import resolve_device


class Reranker:
    """Re-scores retrieved chunks for actual relevance to the query.

    Embedding similarity (used for the initial retrieval) is fast but
    approximate; a cross-encoder that looks at the query and chunk together
    is slower but substantially more accurate, so we over-fetch candidates
    from the vector store and let this narrow them down to top_k.
    """

    def __init__(self, model_name: str = settings.reranker_model):
        self.model = CrossEncoder(model_name, device=resolve_device())

    def rerank(self, query: str, hits: list[dict], top_k: int) -> list[dict]:
        if len(hits) <= 1:
            return hits

        pairs = [(query, hit["text"]) for hit in hits]
        scores = self.model.predict(pairs)
        for hit, score in zip(hits, scores):
            hit["rerank_score"] = float(score)

        return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:top_k]


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker()
