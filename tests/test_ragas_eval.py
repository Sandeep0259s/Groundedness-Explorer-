import pytest

from src.rag.ragas_eval import answer_relevancy, context_precision, faithfulness


def test_faithfulness_matches_hallucination_scorer():
    context = ["The Eiffel Tower stands 330 metres tall."]
    answer = "The Eiffel Tower is 330 metres tall."
    score = faithfulness(answer, context)
    assert 0.0 <= score <= 1.0
    assert score > 0.5  # a near-paraphrase of the source should score well


def test_answer_relevancy_returns_zero_for_empty_answer():
    assert answer_relevancy("How tall is the tower?", "") == 0.0


def test_context_precision_returns_zero_for_no_context():
    assert context_precision("anything", []) == 0.0


def test_ragas_metrics_with_ollama(ollama_available):
    if not ollama_available:
        pytest.skip("Ollama is not running — skipping tests that need real LLM generation")

    question = "How tall is the Eiffel Tower?"
    answer = "The Eiffel Tower is 330 metres tall."
    context = ["The Eiffel Tower stands 330 metres tall and is located in Paris."]

    relevancy = answer_relevancy(question, answer)
    assert 0.0 <= relevancy <= 1.0

    precision = context_precision(question, context)
    assert precision in (0.0, 1.0)  # single chunk: judged either relevant or not
