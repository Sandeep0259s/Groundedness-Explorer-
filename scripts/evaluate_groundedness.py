"""Evaluate the groundedness scorer against a labeled (context, sentence, label) set.

Usage:
    python -m scripts.evaluate_groundedness
    python -m scripts.evaluate_groundedness --sweep
    python -m scripts.evaluate_groundedness --calibrate   # persist the best threshold as the new default
"""
import argparse
import json

from src.rag.config import CALIBRATED_THRESHOLD_FILE, settings
from src.rag.hallucination import get_scorer

DEFAULT_DATASET = "data/eval/groundedness_eval.jsonl"


def load_dataset(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metrics_at_threshold(rows: list[dict], threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for row in rows:
        predicted = "grounded" if row["overall_score"] >= threshold else "hallucinated"
        actual = row["label"]
        if predicted == "hallucinated" and actual == "hallucinated":
            tp += 1
        elif predicted == "hallucinated" and actual == "grounded":
            fp += 1
        elif predicted == "grounded" and actual == "grounded":
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(rows) if rows else 0.0
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the NLI groundedness scorer")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--sweep", action="store_true", help="Sweep thresholds 0.05-0.95 to find the best F1")
    parser.add_argument("--threshold", type=float, default=None, help="Override the threshold used for the per-example table")
    parser.add_argument(
        "--calibrate", action="store_true",
        help="Sweep thresholds and persist the best one to .groundedness_threshold.json as the new default",
    )
    args = parser.parse_args()
    if args.calibrate:
        args.sweep = True

    dataset = load_dataset(args.dataset)
    scorer = get_scorer()

    print(f"Scoring {len(dataset)} labeled (context, sentence) pairs from {args.dataset}...\n")

    for row in dataset:
        result = scorer.score(row["sentence"], [row["context"]])
        row["overall_score"] = result["overall_score"]

    threshold = args.threshold if args.threshold is not None else settings.groundedness_threshold

    print(f"{'label':<13} {'score':>6}  sentence")
    print("-" * 80)
    for row in sorted(dataset, key=lambda r: r["overall_score"]):
        predicted = "grounded" if row["overall_score"] >= threshold else "hallucinated"
        flag = "  " if predicted == row["label"] else " X"
        print(f"{row['label']:<13} {row['overall_score']:.3f}  {row['sentence'][:70]}{flag}")

    print()
    result = metrics_at_threshold(dataset, threshold)
    print(f"At threshold={threshold:.2f}: accuracy={result['accuracy']:.2f} "
          f"precision={result['precision']:.2f} recall={result['recall']:.2f} f1={result['f1']:.2f} "
          f"(tp={result['tp']} fp={result['fp']} tn={result['tn']} fn={result['fn']})")

    if args.sweep:
        print("\nThreshold sweep:")
        print(f"{'threshold':>9}  {'accuracy':>8}  {'precision':>9}  {'recall':>6}  {'f1':>5}")
        best = None
        step = 0.05
        t = 0.05
        while t < 1.0:
            m = metrics_at_threshold(dataset, t)
            marker = ""
            if best is None or m["f1"] > best["f1"]:
                best = m
            print(f"{t:>9.2f}  {m['accuracy']:>8.2f}  {m['precision']:>9.2f}  {m['recall']:>6.2f}  {m['f1']:>5.2f}")
            t += step
        print(f"\nBest F1 at threshold={best['threshold']:.2f} (f1={best['f1']:.2f}) "
              f"— current effective default is {settings.groundedness_threshold:.2f}")

        if args.calibrate:
            CALIBRATED_THRESHOLD_FILE.write_text(json.dumps({"threshold": round(best["threshold"], 2)}))
            print(f"Calibrated: wrote threshold={best['threshold']:.2f} to {CALIBRATED_THRESHOLD_FILE.name} "
                  f"— this is now the default until you recalibrate or set RAG_GROUNDEDNESS_THRESHOLD.")


if __name__ == "__main__":
    main()
