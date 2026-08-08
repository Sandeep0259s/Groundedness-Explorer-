# Run Commands

Prerequisites already set up on this machine: `.venv` with all dependencies installed, and
Ollama running locally with `llama3.2:3b` (chat) and `moondream` (vision) pulled.

## Run the app (Web UI — normal way to run this project)

```powershell
.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload
```

Open http://127.0.0.1:8000

## Alternatives / extras (not needed for normal use)

| Command | When to use it |
|---|---|
| `.venv\Scripts\python.exe -m src.cli ingest` | CLI: ingest documents instead of using the web UI |
| `.venv\Scripts\python.exe -m src.cli ask "your question"` | CLI: ask a question instead of using the web UI |
| `.venv\Scripts\python.exe -m streamlit run src/app.py` | Alternative UI (Streamlit) instead of the FastAPI web app |
| `.venv\Scripts\python.exe -m pytest -v` | Run the test suite (only needed when changing code) |
| `.venv\Scripts\python.exe -m scripts.evaluate_groundedness --sweep` | Groundedness accuracy eval / threshold sweep |
| `.venv\Scripts\python.exe -m scripts.compare_models --models llama3.2:3b,qwen3.5:4b --out results.json` | Benchmark multiple models |
| `docker compose up` | Alternative way to run the whole stack via Docker (not needed — local venv + Ollama already set up) |
