from functools import lru_cache

from sentence_transformers import SentenceTransformer

from .config import settings
from .device import resolve_device


class Embedder:
    def __init__(self, model_name: str = settings.embedding_model):
        self.model = SentenceTransformer(model_name, device=resolve_device())

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    return Embedder()
