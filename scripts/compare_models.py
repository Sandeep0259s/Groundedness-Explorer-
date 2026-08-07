"""Benchmark multiple local Ollama models on the same questions: answer,
groundedness, and latency — for an ablation section in a report.

Usage:
    python -m scripts.compare_models --models llama3.2:3b,qwen3.5:4b
    python -m scripts.compare_models --models llama3.2:3b,qwen3.5:4b --questions questions.txt --out results.json
"""
import argparse
import json
import time

from src.rag.config import settings
from src.rag.hallucination import get_scorer
from src.rag.llm import OllamaLLM
from src.rag.reranker import get_reranker
from src.rag.vectorstore import VectorStore

DEFAULT_QUESTIONS = [
    "How tall is the Eiffel Tower?",
    "What byproduct does photosynthesis generate?",
    "Who designed the Eiffel Tower?",
]


def load_questions(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_QUESTIONS
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Benchmark multiple Ollama models on the same questions")
    parser.add_argument("--models", required=True, help="Comma-separated Ollama model names, e.g. llama3.2:3b,qwen3.5:4b")
    parser.add_argument("--questions", default=None, help="Path to a text file with one question per line")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--label", default=None, help="Restrict retrieval to one label")
    parser.add_argument("--out", default=None, help="Optional path to write full results as JSON")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    questions = load_questions(args.questions)

    store = VectorStore()
    scorer = get_scorer()
    reranker = get_reranker()

    if store.count() == 0:
        print("No documents ingested yet — run `python -m src.rag.ingest` first.")
        return

    per_model_runs: dict[str, list[dict]] = {m: [] for m in models}
    llm = OllamaLLM()

    for question in questions:
        print(f"\n{'=' * 90}\nQ: {question}\n{'=' * 90}")
        candidate_k = max(args.top_k * settings.rerank_candidate_multiplier, args.top_k)
        candidates = store.query_hybrid(question, top_k=candidate_k, label=args.label)
        hits = reranker.rerank(question, candidates, args.top_k)
        context_chunks = [hit["text"] for hit in hits]

        for model_name in models:
            start = time.perf_counter()
            answer = llm.generate(question, context_chunks, model=model_name)
            latency = time.perf_counter() - start

            groundedness = scorer.score(answer, context_chunks)

            per_model_runs[model_name].append({
                "question": question,
                "answer": answer,
                "latency_seconds": latency,
                "groundedness_score": groundedness["overall_score"],
                "groundedness_label": groundedness["label"],
            })

            print(f"\n[{model_name}]  ({latency:.1f}s)")
            print(f"  answer: {answer.strip()[:300]}")
            print(f"  groundedness: {groundedness['label']} ({groundedness['overall_score']:.2f})")

    print(f"\n{'=' * 90}\nSummary\n{'=' * 90}")
    print(f"{'model':<20} {'avg latency':>12} {'avg groundedness':>18} {'% grounded':>12}")
    summary = {}
    for model_name, runs in per_model_runs.items():
        avg_latency = sum(r["latency_seconds"] for r in runs) / len(runs)
        avg_score = sum(r["groundedness_score"] for r in runs) / len(runs)
        pct_grounded = 100 * sum(1 for r in runs if r["groundedness_label"] == "grounded") / len(runs)
        summary[model_name] = {
            "avg_latency_seconds": avg_latency,
            "avg_groundedness_score": avg_score,
            "pct_grounded": pct_grounded,
        }
        print(f"{model_name:<20} {avg_latency:>11.1f}s {avg_score:>18.2f} {pct_grounded:>11.0f}%")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"runs": per_model_runs, "summary": summary}, f, indent=2)
        print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
