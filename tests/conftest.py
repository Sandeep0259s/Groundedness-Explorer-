"""Points the app at a throwaway data/vectorstore directory for the whole
test session, set up BEFORE any src.rag module is imported (env vars here
run at conftest collection time, ahead of test-module-level imports) — so
running tests can never touch or pollute the real app's data."""
import os
import tempfile
from pathlib import Path

import pytest

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="rag_test_"))
(_TEST_ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)

os.environ.setdefault("RAG_DATA_DIR", str(_TEST_ROOT / "data" / "raw"))
os.environ.setdefault("RAG_VECTORSTORE_DIR", str(_TEST_ROOT / "vectorstore"))
os.environ.setdefault("RAG_DEVICE", "cpu")  # never prompt for GPU permission during tests
os.environ.setdefault("RAG_MIN_FREE_MEMORY_MB", "50")  # test files are tiny; keep the guard from false-rejecting


@pytest.fixture(scope="session")
def ollama_available():
    """Shared across every test file — tests that need real LLM generation
    (chat or vision) should skip themselves when this is False rather than
    failing, so the suite still passes in CI or on a machine without Ollama."""
    import ollama

    from src.rag.config import settings

    try:
        ollama.Client(host=settings.ollama_host).list()
        return True
    except Exception:
        return False
