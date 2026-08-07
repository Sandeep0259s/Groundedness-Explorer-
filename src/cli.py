import argparse

from src.rag.config import settings
from src.rag.ingest import ingest
from src.rag.pipeline import RAGPipeline


def cmd_ingest(args):
    ingest(args.data_dir)


def cmd_ask(args):
    pipeline = RAGPipeline()
    result = pipeline.ask(args.question, top_k=args.top_k)

    print("\n=== Answer ===")
    print(result["answer"])

    print("\n=== Sources ===")
    for hit in result["sources"]:
        print(f"- {hit['source']} (distance={hit['distance']:.4f})")

    g = result["groundedness"]
    print(f"\n=== Groundedness: {g['label']} ({g['overall_score']:.2f}) ===")
    for s in g["sentences"]:
        print(f"  [{s['entailment']:.2f}] {s['sentence']}")


def main():
    parser = argparse.ArgumentParser(description="RAG with hallucination detection")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into the vector store")
    ingest_parser.add_argument("--data-dir", default=settings.data_dir)
    ingest_parser.set_defaults(func=cmd_ingest)

    ask_parser = subparsers.add_parser("ask", help="Ask a question against ingested documents")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=settings.top_k)
    ask_parser.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
