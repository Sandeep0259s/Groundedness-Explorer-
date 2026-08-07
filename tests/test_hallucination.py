"""These load the real NLI model (slower, ~seconds on first run) — they're
the ones that actually guard the project's core value proposition, so it's
worth paying that cost rather than mocking the model away."""
from src.rag.hallucination import get_scorer

EIFFEL_CONTEXT = (
    "The Eiffel Tower is a wrought-iron lattice tower located on the Champ de Mars in Paris, France. "
    "It was designed by the engineer Gustave Eiffel and completed in 1889 as the entrance arch for the "
    "World's Fair. The tower stands 330 metres tall."
)


def test_grounded_answer_scores_high():
    scorer = get_scorer()
    result = scorer.score("The Eiffel Tower is 330 metres tall.", [EIFFEL_CONTEXT])
    assert result["label"] == "grounded"
    assert result["overall_score"] > 0.5


def test_unrelated_answer_scores_low():
    scorer = get_scorer()
    result = scorer.score("Bananas are a good source of potassium.", [EIFFEL_CONTEXT])
    assert result["label"] == "possibly hallucinated"
    assert result["overall_score"] < 0.5


def test_contradicted_answer_scores_low():
    scorer = get_scorer()
    result = scorer.score("The Eiffel Tower is 900 metres tall.", [EIFFEL_CONTEXT])
    assert result["overall_score"] < 0.5


def test_source_index_points_at_the_supporting_chunk():
    scorer = get_scorer()
    unrelated = "Photosynthesis produces oxygen as a byproduct."
    result = scorer.score("The Eiffel Tower is 330 metres tall.", [unrelated, EIFFEL_CONTEXT])
    assert result["sentences"][0]["source_index"] == 1


def test_empty_answer_or_context_is_unknown():
    scorer = get_scorer()
    assert scorer.score("", [EIFFEL_CONTEXT])["label"] == "unknown"
    assert scorer.score("Some answer.", [])["label"] == "unknown"
