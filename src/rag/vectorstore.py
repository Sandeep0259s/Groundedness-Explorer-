import re

import chromadb
from rank_bm25 import BM25Okapi

from .config import settings
from .embeddings import get_embedder
from .labels import DEFAULT_LABEL
from .memory_guard import available_mb

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


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
        ids = result["ids"][0] if result["ids"] else []
        for doc_id, doc, meta, dist in zip(ids, result["documents"][0], result["metadatas"][0], result["distances"][0]):
            hits.append({
                "id": doc_id,
                "text": doc,
                "source": meta.get("source", "unknown"),
                "label": meta.get("label", DEFAULT_LABEL),
                "distance": dist,
            })
        return hits

    def _bm25_hits(self, question: str, label: str | None, top_k: int) -> list[dict]:
        if self.count() == 0:
            return []
        # Unlike every other memory-heavy path in this codebase (uploads,
        # OCR), rebuilding BM25 pulls the *entire* matching collection into
        # Python on every single question — degrade to embeddings-only
        # instead of risking a crash when memory is tight, rather than
        # hard-failing the whole question over a keyword-search bonus.
        if available_mb() < settings.min_free_memory_mb:
            return []
        where = {"label": label} if label else None
        result = self.collection.get(where=where, include=["documents", "metadatas"])
        ids, docs, metas = result["ids"], result["documents"], result["metadatas"]
        if not docs:
            return []

        bm25 = BM25Okapi([_tokenize(d) for d in docs])
        scores = bm25.get_scores(_tokenize(question))
        ranked = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {
                "id": ids[i],
                "text": docs[i],
                "source": metas[i].get("source", "unknown"),
                "label": metas[i].get("label", DEFAULT_LABEL),
                # A BM25-only hit has no embedding distance — every hit dict
                # must carry the same keys regardless of which leg found it,
                # since downstream code (the frontend's `hit.distance`,
                # reranker.rerank) treats "source" hits as one uniform shape.
                "distance": None,
                "bm25_score": float(scores[i]),
            }
            for i in ranked
        ]

    def query_hybrid(self, question: str, top_k: int = settings.top_k, label: str | None = None) -> list[dict]:
        """Combines embedding similarity with BM25 keyword search via
        reciprocal rank fusion. Embeddings alone miss exact-token matches
        (names, part numbers, dates) that keyword search catches; BM25 alone
        misses paraphrases embeddings catch. Fusing by *rank* (their raw
        scores live on incomparable scales — cosine distance vs. BM25 term
        weight) is the standard, robust way to combine them."""
        fetch_k = max(top_k * 2, top_k)
        vector_hits = self.query(question, top_k=fetch_k, label=label)
        bm25_hits = self._bm25_hits(question, label, fetch_k)

        rrf_k = 60  # standard RRF damping constant
        scores: dict[str, float] = {}
        items: dict[str, dict] = {}
        for rank, hit in enumerate(vector_hits):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1.0 / (rrf_k + rank + 1)
            items[hit["id"]] = hit
        for rank, hit in enumerate(bm25_hits):
            scores[hit["id"]] = scores.get(hit["id"], 0.0) + 1.0 / (rrf_k + rank + 1)
            items.setdefault(hit["id"], hit)

        ranked_ids = sorted(scores, key=lambda i: scores[i], reverse=True)[:top_k]
        return [items[i] for i in ranked_ids]

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
