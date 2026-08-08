"""RAGAS-style evaluation: faithfulness, answer relevancy, and context
precision — computed against this app's live pipeline instead of requiring
the `ragas` package's OpenAI-judge assumption. See src/rag/ragas_eval.py
for what each metric actually measures and why.

Usage:
    python -m scripts.evaluate_ragas
    python -m scripts.evaluate_ragas --questions questions.txt --label resume --out results.json
"""
import argparse
import json
import statistics

from src.rag.pipeline import RAGPipeline
from src.rag.ragas_eval import evaluate_answer

DEFAULT_QUESTIONS = [
    "How tall is the Eiffel Tower?",
    "Who designed the Eiffel Tower?",
    "What byproduct does photosynthesis generate?",
]

METRICS = ("faithfulness", "answer_relevancy", "context_precision")


def load_questions(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_QUESTIONS
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", default=None, help="Path to a text file, one question per line")
    parser.add_argument("--label", default=None, help="Restrict retrieval to one label")
    parser.add_argument("--model", default=None, help="Override the active chat model for this run")
    parser.add_argument("--out", default=None, help="Optional path to write full results as JSON")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if not questions:
        print("No questions to evaluate (the --questions file was empty).")
        return

    pipeline = RAGPipeline()

    if pipeline.store.count() == 0:
        print("No documents ingested yet — ingest something first (see README).")
        return

    results = []
    for question in questions:
        print(f"\n{'=' * 90}\nQ: {question}\n{'=' * 90}")
        result = evaluate_answer(question, pipeline, label=args.label, model=args.model)
        results.append(result)
        print(f"answer: {result['answer'].strip()[:300]}")
        print(f"  faithfulness:      {result['faithfulness']:.2f}")
        print(f"  answer_relevancy:  {result['answer_relevancy']:.2f}")
        print(f"  context_precision: {result['context_precision']:.2f}")

    print(f"\n{'=' * 90}\nSummary ({len(results)} questions)\n{'=' * 90}")
    summary = {}
    for metric in METRICS:
        values = [r[metric] for r in results]
        summary[metric] = statistics.mean(values)
        print(f"{metric:<20} mean={summary[metric]:.3f}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"runs": results, "summary": summary}, f, indent=2)
        print(f"\nFull results written to {args.out}")


if __name__ == "__main__":
    main()
