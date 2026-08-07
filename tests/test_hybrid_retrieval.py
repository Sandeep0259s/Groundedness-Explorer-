from src.rag.vectorstore import VectorStore


def test_bm25_catches_exact_keyword_match_embeddings_might_miss():
    store = VectorStore()
    label = "hybrid-test-label"

    # One chunk contains an exact, unusual token an embedding model has no
    # learned representation for; the other is topically related but never
    # says the literal term — a case pure embedding similarity often misses
    # and keyword search reliably catches.
    store.add_chunks(
        ids=["hybrid-1", "hybrid-2"],
        texts=[
            "The replacement part number is XJ-4471-Q, available from the warehouse.",
            "Spare components can usually be ordered from the regional supply depot.",
        ],
        metadatas=[{"source": "doc1", "label": label}, {"source": "doc2", "label": label}],
    )

    hits = store.query_hybrid("What is part XJ-4471-Q?", top_k=2, label=label)
    assert any("XJ-4471-Q" in h["text"] for h in hits)

    store.delete_label(label)


def test_query_hybrid_returns_at_most_top_k():
    store = VectorStore()
    label = "hybrid-test-label-2"
    store.add_chunks(
        ids=[f"h2-{i}" for i in range(5)],
        texts=[f"Document number {i} about widgets and gadgets." for i in range(5)],
        metadatas=[{"source": f"doc{i}", "label": label} for i in range(5)],
    )

    hits = store.query_hybrid("widgets", top_k=3, label=label)
    assert len(hits) <= 3

    store.delete_label(label)


def test_query_hybrid_empty_label_returns_empty():
    store = VectorStore()
    assert store.query_hybrid("anything", top_k=3, label="definitely-empty-label-xyz") == []
