from src.rag import answer_cache


def setup_function():
    answer_cache.clear()


def test_miss_on_empty_cache():
    assert answer_cache.get("How tall is the Eiffel Tower?", None, None, 4) is None


def test_put_then_get_roundtrip():
    result = {"answer": "330 metres.", "sources": [], "groundedness": {}, "answer_mode": "text"}
    answer_cache.put("How tall is the Eiffel Tower?", None, None, 4, result)

    hit = answer_cache.get("How tall is the Eiffel Tower?", None, None, 4)
    assert hit is not None
    assert hit["answer"] == "330 metres."
    assert "_cached_at" not in hit  # internal bookkeeping shouldn't leak out


def test_cache_key_is_case_and_whitespace_insensitive():
    result = {"answer": "330 metres.", "sources": [], "groundedness": {}, "answer_mode": "text"}
    answer_cache.put("How tall is the Eiffel Tower?", None, None, 4, result)
    assert answer_cache.get("  HOW TALL IS THE EIFFEL TOWER?  ", None, None, 4) is not None


def test_different_label_is_a_different_cache_entry():
    result = {"answer": "answer for general", "sources": [], "groundedness": {}, "answer_mode": "text"}
    answer_cache.put("same question", "general", None, 4, result)
    assert answer_cache.get("same question", "other-label", None, 4) is None


def test_different_model_is_a_different_cache_entry():
    result = {"answer": "answer for model A", "sources": [], "groundedness": {}, "answer_mode": "text"}
    answer_cache.put("same question", None, "llama3.2:3b", 4, result)
    assert answer_cache.get("same question", None, "qwen3.5:4b", 4) is None


def test_clear_empties_the_cache():
    answer_cache.put("q", None, None, 4, {"answer": "a", "sources": [], "groundedness": {}, "answer_mode": "text"})
    assert answer_cache.size() == 1
    answer_cache.clear()
    assert answer_cache.size() == 0


def test_eviction_bounds_cache_size(monkeypatch):
    monkeypatch.setattr(answer_cache, "_MAX_ENTRIES", 3)
    for i in range(5):
        answer_cache.put(f"question {i}", None, None, 4, {"answer": str(i), "sources": [], "groundedness": {}, "answer_mode": "text"})
    assert answer_cache.size() <= 3
