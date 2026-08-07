from functools import lru_cache

from sentence_transformers import CrossEncoder

from .config import settings
from .device import resolve_device
from .text_utils import split_sentences


class GroundednessScorer:
    """Scores an answer's groundedness in retrieved context using NLI entailment.

    For each answer sentence, checks whether the best-matching context chunk
    entails it. Low entailment across sentences suggests hallucination.
    """

    def __init__(self, model_name: str = settings.nli_model):
        self.model = CrossEncoder(model_name, device=resolve_device())
        id2label = {k: v.lower() for k, v in self.model.config.id2label.items()}
        self.entailment_index = next(i for i, label in id2label.items() if label == "entailment")

    def score(self, answer: str, context_chunks: list[str]) -> dict:
        sentences = split_sentences(answer)
        if not sentences or not context_chunks:
            return {"overall_score": 0.0, "label": "unknown", "sentences": []}

        # This NLI model is surprisingly brittle to premise granularity: the
        # same fact can score ~0 as a whole-paragraph premise but ~1 as a
        # single-sentence premise, or vice versa, depending on phrasing —
        # discovered by comparing active vs. passive voice against the same
        # source text during evaluation. Checking both the whole chunk and
        # each of its sentences, and keeping the best match, makes scoring
        # robust to that brittleness instead of being at its mercy.
        candidates: list[tuple[str, int]] = []  # (premise_text, source_chunk_index)
        for idx, chunk in enumerate(context_chunks):
            candidates.append((chunk, idx))
            for chunk_sentence in split_sentences(chunk):
                if chunk_sentence != chunk:
                    candidates.append((chunk_sentence, idx))

        pairs = [(premise, sentence) for sentence in sentences for premise, _ in candidates]
        scores = self.model.predict(pairs, apply_softmax=True)

        per_sentence = []
        candidate_count = len(candidates)
        for i, sentence in enumerate(sentences):
            sentence_scores = scores[i * candidate_count : (i + 1) * candidate_count]
            entailments = [row[self.entailment_index] for row in sentence_scores]
            best_j = max(range(candidate_count), key=lambda j: entailments[j])
            per_sentence.append({
                "sentence": sentence,
                "entailment": float(entailments[best_j]),
                "source_index": candidates[best_j][1],
            })

        overall = sum(s["entailment"] for s in per_sentence) / len(per_sentence)
        label = "grounded" if overall >= settings.groundedness_threshold else "possibly hallucinated"
        return {"overall_score": overall, "label": label, "sentences": per_sentence}


@lru_cache(maxsize=1)
def get_scorer() -> GroundednessScorer:
    return GroundednessScorer()
