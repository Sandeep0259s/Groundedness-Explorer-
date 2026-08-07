"""Lightweight, dependency-free implementations of the three most-cited
RAGAS metrics, computed directly against this app's own pipeline instead of
requiring the `ragas` package (which expects an OpenAI-compatible judge by
default and a heavier dependency chain).

- **Faithfulness** is intentionally the *same* NLI-entailment technique
  `hallucination.py` already uses — RAGAS defines faithfulness as "each
  claim in the answer is supported by the context," which is exactly what
  the groundedness scorer measures. Reported under RAGAS's name here so a
  reviewer who knows the standard metric recognizes it immediately.
- **Answer relevancy** and **context precision** are genuinely new
  measurements this project didn't previously have: whether the answer
  actually addresses the question asked (not just whether its claims are
  grounded — an answer can be perfectly faithful and still not answer the
  question), and whether the retrieved chunks were actually useful, not
  just superficially similar.

Both use an LLM-as-judge, which is itself an approximation with known
limitations (judge model quality, prompt sensitivity) — reported as such,
not as ground truth.
"""
import ollama

from . import model_prefs
from .config import settings
from .embeddings import get_embedder
from .hallucination import get_scorer


def _judge_client() -> ollama.Client:
    return ollama.Client(host=settings.ollama_host)


def _active_chat_model(model: str | None) -> str:
    return model or model_prefs.load_active_model("chat", settings.ollama_model)


def faithfulness(answer: str, context_chunks: list[str]) -> float:
    """RAGAS's faithfulness == this project's existing groundedness score."""
    return get_scorer().score(answer, context_chunks)["overall_score"]


_REVERSE_QUESTION_PROMPT = """Given this answer, write {n} different questions that this answer would be a good, direct response to. Reply with ONLY the questions, one per line, no numbering, no other text.

Answer: {answer}"""


def answer_relevancy(question: str, answer: str, model: str | None = None, n: int = 3) -> float:
    """Generates candidate questions the answer would address, then measures
    how semantically close they are to the actual question asked — a low
    score means the answer wandered off-topic even if every individual
    claim in it was well-grounded in the context."""
    if not answer.strip():
        return 0.0

    prompt = _REVERSE_QUESTION_PROMPT.format(n=n, answer=answer)
    response = _judge_client().chat(
        model=_active_chat_model(model), messages=[{"role": "user", "content": prompt}]
    )
    candidates = [line.strip("-* ").strip() for line in response["message"]["content"].splitlines() if line.strip()]
    if not candidates:
        return 0.0

    import numpy as np

    embedder = get_embedder()
    q_vec = np.array(embedder.embed_one(question))
    c_vecs = [np.array(v) for v in embedder.embed(candidates)]
    sims = [float(np.dot(q_vec, c) / (np.linalg.norm(q_vec) * np.linalg.norm(c) + 1e-8)) for c in c_vecs]
    return sum(sims) / len(sims)


_RELEVANCE_JUDGE_PROMPT = """Question: {question}

Chunk: {chunk}

Does this chunk contain information that's actually useful for answering the question? Reply with only YES or NO, nothing else."""


def context_precision(question: str, context_chunks: list[str], model: str | None = None) -> float:
    """Of the retrieved chunks, what fraction does an LLM judge as actually
    relevant to the question — a low score means the retriever is pulling
    in noise even if the top hit happened to be good enough to answer from."""
    if not context_chunks:
        return 0.0

    active = _active_chat_model(model)
    client = _judge_client()
    relevant = 0
    for chunk in context_chunks:
        prompt = _RELEVANCE_JUDGE_PROMPT.format(question=question, chunk=chunk)
        response = client.chat(model=active, messages=[{"role": "user", "content": prompt}])
        if response["message"]["content"].strip().upper().startswith("Y"):
            relevant += 1
    return relevant / len(context_chunks)


def evaluate_answer(question: str, pipeline, label: str | None = None, model: str | None = None) -> dict:
    """Runs a question through the live pipeline and scores the result on
    all three metrics at once, so the eval script has one call per question."""
    result = pipeline.ask(question, label=label, model=model)
    context_chunks = [hit["text"] for hit in result["sources"]]

    return {
        "question": question,
        "answer": result["answer"],
        "answer_mode": result.get("answer_mode", "text"),
        "faithfulness": faithfulness(result["answer"], context_chunks),
        "answer_relevancy": answer_relevancy(question, result["answer"], model=model),
        "context_precision": context_precision(question, context_chunks, model=model),
    }
