import chromadb

from .config import settings
from .embeddings import get_embedder
from .labels import DEFAULT_LABEL


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.vectorstore_dir)
        self.collection = self.client.get_or_create_collection(settings.collection_name)
        self.embedder = get_embedder()

    def add_chunks(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        embeddings = self.embedder.embed(texts)
        self.collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    def query(self, question: str, top_k: int = settings.top_k, label: str | None = None) -> list[dict]:
        embedding = self.embedder.embed_one(question)
        where = {"label": label} if label else None
        result = self.collection.query(query_embeddings=[embedding], n_results=top_k, where=where)

        hits = []
        for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            hits.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "label": meta.get("label", DEFAULT_LABEL),
                "distance": dist,
            })
        return hits

    def count(self) -> int:
        return self.collection.count()

    def list_sources(self, label: str | None = None) -> list[dict]:
        if self.count() == 0:
            return []

        where = {"label": label} if label else None
        result = self.collection.get(where=where, include=["metadatas"])
        counts: dict[tuple[str, str], int] = {}
        for meta in result["metadatas"]:
            key = (meta.get("source", "unknown"), meta.get("label", DEFAULT_LABEL))
            counts[key] = counts.get(key, 0) + 1
        return [
            {"source": source, "label": lbl, "chunks": n}
            for (source, lbl), n in sorted(counts.items())
        ]

    def delete_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})

    def delete_label(self, label: str) -> None:
        self.collection.delete(where={"label": label})
