import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.rag import answer_cache
from src.rag import conversation_store
from src.rag import device as device_module
from src.rag import labels as labels_store
from src.rag import model_prefs
from src.rag import model_registry
from src.rag import system_stats
from src.rag.config import settings
from src.rag.embeddings import get_embedder
from src.rag.hallucination import get_scorer
from src.rag.ingest import SUFFIX_READERS, ingest_path, list_ingestible_paths
from src.rag.labels import DEFAULT_LABEL
from src.rag.memory_guard import InsufficientMemoryError, ensure_headroom
from src.rag.pipeline import RAGPipeline
from src.rag.reranker import get_reranker
from src.rag.vision import IMAGE_SUFFIXES, active_vision_model

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
    role: str = "chat"  # "chat", "vision_caption", or "vision_answer" today; any new role just works


def _new_job() -> str:
    job_id = uuid.uuid4().hex
    app.state.jobs[job_id] = {
        "status": "queued",
        "results": None,
        "sources": None,
        "error": None,
        "progress": {"done": 0, "total": 0},
    }
    return job_id


def _run_ingest_job(job_id: str):
    job = app.state.jobs[job_id]
    pipeline: RAGPipeline = app.state.pipeline
    job["status"] = "running"
    try:
        paths = list_ingestible_paths(settings.data_dir)
        job["progress"] = {"done": 0, "total": len(paths)}
        total_chunks = 0
        for path in paths:
            try:
                total_chunks += ingest_path(path, pipeline.store)
            except InsufficientMemoryError as exc:
                # One oversized/problem file shouldn't abort the whole batch.
                print(f"Skipping {path.name}: {exc}")
            job["progress"]["done"] += 1
        job["chunks_ingested"] = total_chunks
        job["sources"] = pipeline.store.list_sources()
        job["status"] = "done"
        answer_cache.clear()
    except Exception as exc:
        job["error"] = str(exc)
        job["status"] = "error"


def _run_upload_job(job_id: str, saved_paths: list[Path]):
    job = app.state.jobs[job_id]
    pipeline: RAGPipeline = app.state.pipeline
    job["status"] = "running"
    job["progress"] = {"done": 0, "total": len(saved_paths)}
    try:
        results = []
        for p in saved_paths:
            try:
                results.append({"filename": p.name, "chunks": ingest_path(p, pipeline.store)})
            except InsufficientMemoryError as exc:
                results.append({"filename": p.name, "chunks": 0, "error": str(exc)})
            job["progress"]["done"] += 1
        job["results"] = results
        job["sources"] = pipeline.store.list_sources()
        job["status"] = "done"
        answer_cache.clear()
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


@app.get("/api/documents/file")
def get_document_file(source: str):
    """Serves an ingested image's raw bytes, so the UI can show the actual
    picture next to its caption instead of text alone. Restricted to image
    files under data_dir — the source path always comes from a real ingested
    document, but resolving + checking containment still guards against a
    caller passing an arbitrary path outside it."""
    resolved = Path(source).resolve()
    data_root = Path(settings.data_dir).resolve()
    try:
        resolved.relative_to(data_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid source path")
    if resolved.suffix.lower() not in IMAGE_SUFFIXES:
        raise HTTPException(status_code=415, detail="only image files can be served this way")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(resolved)


@app.delete("/api/documents")
def delete_document(source: str):
    pipeline: RAGPipeline = app.state.pipeline
    pipeline.store.delete_source(source)
    try:
        Path(source).unlink(missing_ok=True)
    except OSError:
        pass
    answer_cache.clear()
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
    answer_cache.clear()
    return {"labels": labels_store.list_labels()}


@app.post("/api/labels/{name}/clear")
def api_clear_label(name: str):
    pipeline: RAGPipeline = app.state.pipeline
    labels_store.clear_label_contents(name)
    pipeline.store.delete_label(name)
    answer_cache.clear()
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
    history = conversation_store.load_history(conversation_id)
    top_k = request.top_k or settings.top_k

    # Only cache stateless (no prior turns) questions — the same question
    # text can mean different things mid-conversation, so caching on text
    # alone would risk serving a cross-conversation wrong answer.
    cached = answer_cache.get(request.question, request.label, request.model, top_k) if not history else None
    if cached:
        result = cached
    else:
        result = pipeline.ask(
            request.question,
            top_k=top_k,
            label=request.label,
            history=history[-MAX_HISTORY_TURNS * 2 :],
            model=request.model,
        )
        if not history:
            answer_cache.put(request.question, request.label, request.model, top_k, result)
    result["cached"] = cached is not None

    history.append({"role": "user", "content": request.question})
    history.append({"role": "assistant", "content": result["answer"]})
    if len(history) > MAX_HISTORY_TURNS * 2:
        history = history[-MAX_HISTORY_TURNS * 2 :]
    conversation_store.save_history(conversation_id, history)

    result["conversation_id"] = conversation_id
    return result


@app.post("/api/ask/stream")
def ask_stream(request: AskRequest):
    """Same as /api/ask, but streams the answer as Server-Sent Events —
    an `event: token` per piece of text as it's generated, then a single
    `event: done` with the same sources/groundedness/answer_mode payload
    /api/ask returns all at once. Groundedness can only be scored once the
    full answer exists, so it's necessarily part of the final event, not
    streamed incrementally."""
    pipeline: RAGPipeline = app.state.pipeline
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    conversation_id = request.conversation_id or uuid.uuid4().hex
    history = conversation_store.load_history(conversation_id)
    top_k = request.top_k or settings.top_k

    cached = answer_cache.get(request.question, request.label, request.model, top_k) if not history else None

    def event_source():
        if cached:
            # Nothing to actually stream — send the whole answer as one
            # piece so a repeated question still renders instantly instead
            # of re-running retrieval/rerank/generation from scratch.
            final_result = {**cached, "cached": True}
            yield f"event: token\ndata: {json.dumps(final_result['answer'])}\n\n"
        else:
            final_result = None
            for kind, payload in pipeline.ask_stream(
                request.question,
                top_k=top_k,
                label=request.label,
                history=history[-MAX_HISTORY_TURNS * 2 :],
                model=request.model,
            ):
                if kind == "token":
                    yield f"event: token\ndata: {json.dumps(payload)}\n\n"
                else:
                    final_result = payload
            final_result["cached"] = False
            if not history:
                answer_cache.put(request.question, request.label, request.model, top_k, final_result)

        final_result["conversation_id"] = conversation_id
        yield f"event: done\ndata: {json.dumps(final_result)}\n\n"

        updated_history = history + [
            {"role": "user", "content": request.question},
            {"role": "assistant", "content": final_result["answer"]},
        ]
        if len(updated_history) > MAX_HISTORY_TURNS * 2:
            updated_history = updated_history[-MAX_HISTORY_TURNS * 2 :]
        conversation_store.save_history(conversation_id, updated_history)

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.post("/api/conversations/{conversation_id}/clear")
def clear_conversation(conversation_id: str):
    conversation_store.clear_history(conversation_id)
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
    answer_cache.clear()  # embeddings/reranking on the new device can reorder retrieval

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
            "vision_caption": active_vision_model("caption"),
            "vision_answer": active_vision_model("answer"),
        },
    }


VALID_MODEL_ROLES = ("chat", "vision_caption", "vision_answer")


@app.post("/api/system/model")
def system_set_model(request: ModelRequest):
    name = request.model.strip()
    if not name:
        raise HTTPException(status_code=400, detail="model name must not be empty")
    if request.role not in VALID_MODEL_ROLES:
        raise HTTPException(status_code=400, detail=f"role must be one of {VALID_MODEL_ROLES}")
    # Every role resolves its active model fresh on every call (see llm.py /
    # vision.py) — persisting the preference is all that's needed here, no
    # pipeline object to mutate or reload.
    model_prefs.save_active_model(request.role, name)
    # A cached answer generated with the *previous* default model (request.model
    # left unset) would otherwise keep being served under the new one.
    answer_cache.clear()
    return {"role": request.role, "active": name}


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
