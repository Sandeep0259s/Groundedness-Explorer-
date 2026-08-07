import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.rag import device as device_module
from src.rag import labels as labels_store
from src.rag import model_prefs
from src.rag import model_registry
from src.rag import system_stats
from src.rag.config import settings
from src.rag.embeddings import get_embedder
from src.rag.hallucination import get_scorer
from src.rag.ingest import SUFFIX_READERS, ingest, ingest_path
from src.rag.labels import DEFAULT_LABEL
from src.rag.memory_guard import InsufficientMemoryError, ensure_headroom
from src.rag.pipeline import RAGPipeline
from src.rag.reranker import get_reranker
from src.rag.vision import active_vision_model

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
UPLOAD_READ_CHUNK = 1024 * 1024  # stream uploads instead of loading whole file into memory


@asynccontextmanager
async def lifespan(app: FastAPI):
    device_module.apply_saved_performance_mode()
    labels_store.ensure_default_labels()
    app.state.pipeline = RAGPipeline()
    # Ephemeral ("session") labels get wiped at the start of every run —
    # a session, for this local single-process app, is one server lifetime.
    for name in labels_store.ephemeral_label_names():
        labels_store.clear_label_contents(name)
        app.state.pipeline.store.delete_label(name)
    app.state.jobs = {}
    app.state.conversations = {}
    yield


app = FastAPI(title="RAG + Hallucination Detector", lifespan=lifespan)


MAX_HISTORY_TURNS = 6  # keep the prompt from growing unbounded in a long conversation


class AskRequest(BaseModel):
    question: str
    top_k: int | None = None
    label: str | None = None
    conversation_id: str | None = None
    model: str | None = None  # override the active model for just this question


class LabelCreateRequest(BaseModel):
    name: str
    ephemeral: bool = False


class DeviceRequest(BaseModel):
    device: str


class PerformanceRequest(BaseModel):
    mode: str


class ModelRequest(BaseModel):
    model: str
    role: str = "chat"  # "chat" or "vision" today; any new role just works


def _new_job() -> str:
    job_id = uuid.uuid4().hex
    app.state.jobs[job_id] = {"status": "queued", "results": None, "sources": None, "error": None}
    return job_id


def _run_ingest_job(job_id: str):
    job = app.state.jobs[job_id]
    pipeline: RAGPipeline = app.state.pipeline
    job["status"] = "running"
    try:
        count = ingest(settings.data_dir, store=pipeline.store)
        job["chunks_ingested"] = count
        job["sources"] = pipeline.store.list_sources()
        job["status"] = "done"
    except Exception as exc:
        job["error"] = str(exc)
        job["status"] = "error"


def _run_upload_job(job_id: str, saved_paths: list[Path]):
    job = app.state.jobs[job_id]
    pipeline: RAGPipeline = app.state.pipeline
    job["status"] = "running"
    try:
        results = [{"filename": p.name, "chunks": ingest_path(p, pipeline.store)} for p in saved_paths]
        job["results"] = results
        job["sources"] = pipeline.store.list_sources()
        job["status"] = "done"
    except Exception as exc:
        job["error"] = str(exc)
        job["status"] = "error"


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = app.state.jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/api/documents")
def documents(label: str | None = None):
    pipeline: RAGPipeline = app.state.pipeline
    return {"sources": pipeline.store.list_sources(label=label), "total_chunks": pipeline.store.count()}


@app.delete("/api/documents")
def delete_document(source: str):
    pipeline: RAGPipeline = app.state.pipeline
    pipeline.store.delete_source(source)
    try:
        Path(source).unlink(missing_ok=True)
    except OSError:
        pass
    return {"sources": pipeline.store.list_sources()}


@app.get("/api/labels")
def list_labels():
    pipeline: RAGPipeline = app.state.pipeline
    sources = pipeline.store.list_sources()

    chunk_counts: dict[str, int] = {}
    doc_counts: dict[str, int] = {}
    for s in sources:
        chunk_counts[s["label"]] = chunk_counts.get(s["label"], 0) + s["chunks"]
        doc_counts[s["label"]] = doc_counts.get(s["label"], 0) + 1

    result = [
        {**lbl, "chunk_count": chunk_counts.get(lbl["name"], 0), "document_count": doc_counts.get(lbl["name"], 0)}
        for lbl in labels_store.list_labels()
    ]
    return {"labels": result}


@app.post("/api/labels")
def api_create_label(request: LabelCreateRequest):
    try:
        return labels_store.create_label(request.name, request.ephemeral)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/labels/{name}")
def api_delete_label(name: str):
    pipeline: RAGPipeline = app.state.pipeline
    try:
        labels_store.delete_label(name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    pipeline.store.delete_label(name)
    return {"labels": labels_store.list_labels()}


@app.post("/api/labels/{name}/clear")
def api_clear_label(name: str):
    pipeline: RAGPipeline = app.state.pipeline
    labels_store.clear_label_contents(name)
    pipeline.store.delete_label(name)
    return {"sources": pipeline.store.list_sources()}


@app.post("/api/ingest")
def api_ingest(background_tasks: BackgroundTasks):
    job_id = _new_job()
    background_tasks.add_task(_run_ingest_job, job_id)
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/upload")
async def upload(
    files: list[UploadFile],
    background_tasks: BackgroundTasks,
    label: str = Form(DEFAULT_LABEL),
):
    try:
        label = labels_store.validate_name(label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not labels_store.label_exists(label):
        labels_store.create_label(label)

    data_dir = labels_store.label_dir(label)
    data_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = settings.max_upload_mb * 1024 * 1024

    saved_paths = []
    skipped = []
    for file in files:
        safe_name = Path(file.filename).name
        suffix = Path(safe_name).suffix.lower()
        if suffix not in SUFFIX_READERS:
            skipped.append({"filename": file.filename, "error": "unsupported file type"})
            continue

        try:
            ensure_headroom(context=f"uploading {safe_name}")
        except InsufficientMemoryError as exc:
            skipped.append({"filename": file.filename, "error": str(exc)})
            continue

        # Stream to disk in chunks rather than `await file.read()` all at once —
        # a single huge upload should never be able to exhaust RAM by itself.
        dest = data_dir / safe_name
        written = 0
        too_large = False
        with open(dest, "wb") as out:
            while chunk := await file.read(UPLOAD_READ_CHUNK):
                written += len(chunk)
                if written > max_bytes:
                    too_large = True
                    break
                out.write(chunk)

        if too_large:
            dest.unlink(missing_ok=True)
            skipped.append(
                {"filename": file.filename, "error": f"exceeds the {settings.max_upload_mb}MB upload limit"}
            )
            continue

        saved_paths.append(dest)

    job_id = _new_job()
    background_tasks.add_task(_run_upload_job, job_id, saved_paths)
    return {"job_id": job_id, "status": "queued", "skipped": skipped, "label": label}


@app.post("/api/ask")
def ask(request: AskRequest):
    pipeline: RAGPipeline = app.state.pipeline
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    conversation_id = request.conversation_id or uuid.uuid4().hex
    history = app.state.conversations.setdefault(conversation_id, [])

    top_k = request.top_k or settings.top_k
    result = pipeline.ask(
        request.question,
        top_k=top_k,
        label=request.label,
        history=history[-MAX_HISTORY_TURNS * 2 :],
        model=request.model,
    )

    history.append({"role": "user", "content": request.question})
    history.append({"role": "assistant", "content": result["answer"]})
    if len(history) > MAX_HISTORY_TURNS * 2:
        del history[: -MAX_HISTORY_TURNS * 2]

    result["conversation_id"] = conversation_id
    return result


@app.post("/api/conversations/{conversation_id}/clear")
def clear_conversation(conversation_id: str):
    app.state.conversations.pop(conversation_id, None)
    return {"status": "cleared"}


# ---------- system: live resource usage, device, performance, model ----------

@app.get("/api/system/stats")
def system_stats_endpoint():
    return system_stats.get_stats()


@app.get("/api/system/devices")
def system_devices():
    return {"devices": device_module.available_devices(), "active": device_module.get_current_device()}


@app.post("/api/system/device")
def system_set_device(request: DeviceRequest):
    """Switches CPU/GPU at runtime. Rebuilds the local models (embedder,
    reranker, NLI scorer) on the new device — the vector store itself is
    untouched, so no re-ingestion is needed."""
    try:
        device_module.set_device(request.device)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    get_embedder.cache_clear()
    get_reranker.cache_clear()
    get_scorer.cache_clear()
    app.state.pipeline = RAGPipeline()

    return {"active": device_module.get_current_device()}


@app.get("/api/system/performance")
def system_get_performance():
    return {"mode": device_module.get_performance_mode(), "options": list(device_module.PERFORMANCE_THREADS)}


@app.post("/api/system/performance")
def system_set_performance(request: PerformanceRequest):
    try:
        device_module.set_performance_mode(request.mode)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"mode": device_module.get_performance_mode()}


@app.get("/api/system/models")
def system_list_models():
    """Every model Ollama has pulled/created, with its capabilities (as
    reported by Ollama itself — nothing here is a hardcoded name), plus
    which one is currently active for each role."""
    try:
        models = model_registry.list_models()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"could not reach Ollama: {exc}")

    return {
        "models": models,
        "active": {
            "chat": model_prefs.load_active_model("chat", settings.ollama_model),
            "vision": active_vision_model(),
        },
    }


@app.post("/api/system/model")
def system_set_model(request: ModelRequest):
    name = request.model.strip()
    if not name:
        raise HTTPException(status_code=400, detail="model name must not be empty")
    if request.role not in ("chat", "vision"):
        raise HTTPException(status_code=400, detail="role must be 'chat' or 'vision'")
    # Both roles resolve their active model fresh on every call (see llm.py /
    # vision.py) — persisting the preference is all that's needed here, no
    # pipeline object to mutate or reload.
    model_prefs.save_active_model(request.role, name)
    return {"role": request.role, "active": name}


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
