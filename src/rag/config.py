import json
import os
from dataclasses import dataclass
from pathlib import Path

CALIBRATED_THRESHOLD_FILE = Path(__file__).resolve().parent.parent.parent / ".groundedness_threshold.json"


def _read_calibrated_threshold(default: float = 0.5) -> float:
    """scripts/evaluate_groundedness.py --calibrate writes a tuned threshold
    here; an explicit RAG_GROUNDEDNESS_THRESHOLD env var still wins over it."""
    try:
        return float(json.loads(CALIBRATED_THRESHOLD_FILE.read_text())["threshold"])
    except (OSError, json.JSONDecodeError, KeyError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    data_dir: str = os.environ.get("RAG_DATA_DIR", "data/raw")
    vectorstore_dir: str = os.environ.get("RAG_VECTORSTORE_DIR", "vectorstore")
    collection_name: str = os.environ.get("RAG_COLLECTION", "documents")

    embedding_model: str = os.environ.get("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    nli_model: str = os.environ.get("RAG_NLI_MODEL", "cross-encoder/nli-deberta-v3-small")
    reranker_model: str = os.environ.get("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_candidate_multiplier: int = int(os.environ.get("RAG_RERANK_CANDIDATES", 4))

    ollama_model: str = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")
    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    # "" means: use whatever's set via the UI/API, falling back to the first
    # vision-capable model Ollama already has pulled — never a hardcoded name.
    vision_model: str = os.environ.get("RAG_VISION_MODEL", "")

    chunk_size: int = int(os.environ.get("RAG_CHUNK_SIZE", 800))
    chunk_overlap: int = int(os.environ.get("RAG_CHUNK_OVERLAP", 120))
    top_k: int = int(os.environ.get("RAG_TOP_K", 4))

    groundedness_threshold: float = float(
        os.environ.get("RAG_GROUNDEDNESS_THRESHOLD", _read_calibrated_threshold())
    )
    # Below this reranker score, a candidate is treated as "not actually
    # relevant" rather than just "the best available" — lets a greeting or
    # off-topic message get an honest "no relevant context" groundedness
    # label instead of a misleading NLI score against irrelevant chunks.
    relevance_threshold: float = float(os.environ.get("RAG_RELEVANCE_THRESHOLD", 0.0))

    device: str = os.environ.get("RAG_DEVICE", "")  # "", "cpu", "cuda", or "mps" — "" means auto-detect + ask

    max_upload_mb: int = int(os.environ.get("RAG_MAX_UPLOAD_MB", 300))
    min_free_memory_mb: int = int(os.environ.get("RAG_MIN_FREE_MEMORY_MB", 500))

    # "" (default) means no auth — fine for the intended "runs on your own
    # machine" use case. Set RAG_API_KEY to require an X-API-Key header on
    # every /api/* call before exposing this beyond localhost (a shared
    # network, a tunnel, a small-group demo link).
    api_key: str = os.environ.get("RAG_API_KEY", "")


settings = Settings()
