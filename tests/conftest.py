"""Points the app at a throwaway data/vectorstore directory for the whole
test session, set up BEFORE any src.rag module is imported (env vars here
run at conftest collection time, ahead of test-module-level imports) — so
running tests can never touch or pollute the real app's data."""
import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="rag_test_"))
(_TEST_ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)

os.environ.setdefault("RAG_DATA_DIR", str(_TEST_ROOT / "data" / "raw"))
os.environ.setdefault("RAG_VECTORSTORE_DIR", str(_TEST_ROOT / "vectorstore"))
os.environ.setdefault("RAG_DEVICE", "cpu")  # never prompt for GPU permission during tests
os.environ.setdefault("RAG_MIN_FREE_MEMORY_MB", "50")  # test files are tiny; keep the guard from false-rejecting


@pytest.fixture(scope="module")
def client():
    """Shared across every test file — a fresh TestClient per test module
    (module-scoped so the app's lifespan startup, which loads real models,
    only runs once per file instead of once per test)."""
    from src.api.main import app

    with TestClient(app) as c:
        yield c


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
