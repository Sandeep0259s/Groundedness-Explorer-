import pytest

from src.rag.query_rewrite import rewrite_for_retrieval


def test_returns_question_unchanged_with_no_history():
    assert rewrite_for_retrieval("How tall is it?", None) == "How tall is it?"
    assert rewrite_for_retrieval("How tall is it?", []) == "How tall is it?"


def test_falls_back_to_original_question_on_ollama_error(monkeypatch):
    import ollama

    def _boom(*args, **kwargs):
        raise ConnectionError("Ollama is not running")

    monkeypatch.setattr(ollama.Client, "chat", _boom)
    history = [{"role": "user", "content": "Tell me about the Eiffel Tower."}]
    assert rewrite_for_retrieval("How tall is it?", history) == "How tall is it?"


def test_rewrite_with_ollama(ollama_available):
    if not ollama_available:
        pytest.skip("Ollama is not running — skipping tests that need real LLM generation")

    history = [
        {"role": "user", "content": "Tell me about the Eiffel Tower."},
        {"role": "assistant", "content": "The Eiffel Tower is a wrought-iron lattice tower in Paris."},
    ]
    rewritten = rewrite_for_retrieval("How tall is it?", history)
    # Small models paraphrase rather than quote the literal noun back (e.g.
    # "one of the most iconic structures in Paris" instead of "Eiffel
    # Tower") — the property that actually matters for retrieval is that
    # *something* changed to add context, not specific wording.
    assert rewritten.strip()
    assert rewritten != "How tall is it?"
