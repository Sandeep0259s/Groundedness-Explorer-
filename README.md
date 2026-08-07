# RAG with Hallucination / Groundedness Detection

A retrieval-augmented generation system that answers questions from your own
documents — text, PDFs, spreadsheets, images, audio, and video — and scores
every answer for how well it's actually supported by the retrieved context,
instead of just trusting the LLM. Documents are organized into user-defined
**labels** (collections) so retrieval can be scoped to just the ones you want.

## How it works

1. **Ingest** — documents are chunked (recursively, along paragraph/sentence
   boundaries, not arbitrary word offsets) and embedded with a local
   sentence-transformer model, then stored in a persistent Chroma vector
   database, tagged with whichever label you filed them under.
2. **Retrieve + re-rank** — a question is embedded and the most similar
   chunks are fetched from the vector store; a second, purpose-built
   cross-encoder then re-scores those candidates for actual relevance and
   narrows them down to the final top-k, which is more accurate than
   embedding similarity alone.
3. **Ask** — a local LLM (via [Ollama](https://ollama.com)) generates an
   answer grounded in those chunks, aware of the recent conversation history
   for natural follow-up questions.
4. **Score** — the answer is split into sentences, and each is checked
   against the retrieved chunks with an NLI (natural language inference)
   cross-encoder. If a sentence isn't entailed by the context, it's flagged
   as potentially hallucinated — and the app tells you exactly *which*
   retrieved chunk it checked the sentence against (click a sentence in the
   UI to jump to its source).

Runs fine on CPU alone (no GPU required) and will automatically offer to use
one if it finds it — see "GPU usage" below.

## Supported file types

| Type | Formats | Notes |
|---|---|---|
| Documents | `.pdf`, `.txt`, `.md`, `.docx`, `.html` | |
| Spreadsheets | `.csv`, `.xlsx` | Rows converted to `key: value` text |
| Images | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp` | OCR via Tesseract + captioning via a vision model |
| Video | `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm` | Speech-to-text via faster-whisper |
| Audio | `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg` | Speech-to-text via faster-whisper |

A scanned/image-only PDF automatically falls back to OCR too.

## Setup

### 1. Install Ollama and pull a model

Download Ollama for Windows from https://ollama.com/download, install it,
then pull a small model that runs fine on CPU:

```powershell
ollama pull llama3.2:3b
```

Ollama runs as a background service on `localhost:11434` once installed.

### 2. Create a virtual environment and install dependencies

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Or just run `.\setup.ps1` (`./setup.sh` on Linux/Mac) to do this and check
for all the optional external tools in one step.

The first run also downloads the embedding, NLI, reranker, and (if you use
video/audio) speech-to-text models from Hugging Face — a one-time download,
cached locally afterward.

### 3. (Optional) OCR and video/audio support

Regular text PDFs, DOCX, CSV, XLSX, HTML, TXT, and MD work with no extra
setup. Two more tools unlock the rest:

```powershell
winget install --id UB-Mannheim.TesseractOCR   # scanned PDFs + images
winget install --id Gyan.FFmpeg                 # video + audio transcription
```

Skip either you don't need — everything else still works without them.

## Usage

Two sample documents are pre-loaded so you can test immediately. There are
three ways to run it:

### Option A — Web app (recommended, best for demos)

```powershell
python -m uvicorn src.api.main:app --reload
```

Open http://127.0.0.1:8000. From the sidebar you can:
- **Create/delete labels** (collections) — e.g. "resume", "coursework" — to
  keep unrelated documents from cluttering each other's search results.
- **Upload files into a chosen label** by drag-and-drop, or click
  **Re-ingest data/raw** to pick up files already on disk.
- **Delete a single file** (×) or **clear/delete a whole label** without
  touching the rest.
- **Scope a question to one label** ("Search within") or search everything.
- Start a **New chat** to reset the conversation while keeping your documents.

Large uploads run in the background — you can keep asking questions while a
big file is still being processed; the sidebar shows an "Ingesting…" badge
and updates automatically the moment it's done, no refresh needed.

### Option B — Command line

```powershell
python -m src.cli ingest
python -m src.cli ask "How tall is the Eiffel Tower?"
```

### Option C — Streamlit demo

```powershell
streamlit run src/app.py
```

## Labels (collections)

Every document lives under a label — a folder-backed collection you can
scope retrieval to. Two labels exist by default:

- **general** — the fallback for anything not filed under a specific label;
  can't be deleted.
- **session** — for throwaway documents you only need for one sitting: its
  contents are **wiped automatically every time the server restarts**.

Create your own with the sidebar's "+ Add" (or `POST /api/labels`), delete
one with its trash icon (or `DELETE /api/labels/{name}`), and clear a
label's contents without deleting the label itself with its clear icon (or
`POST /api/labels/{name}/clear`). Files can be added or removed one at a
time within a label, same as labels themselves.

## GPU usage & live resource panel

The embedding, reranking, and NLI groundedness models run on CPU by default.
If a CUDA GPU (or Apple Silicon MPS) is detected, the app asks once, on the
terminal, before using it:

```
GPU detected: NVIDIA GeForce RTX 3060. Use it to speed up embeddings and
groundedness scoring? [y/N]:
```

Your answer is remembered in `.device_choice.json` (gitignored) so you're
only asked once per machine/GPU. To skip the prompt entirely, set
`RAG_DEVICE=cpu` or `RAG_DEVICE=cuda`. Ollama manages its own GPU usage for
the LLM independently — this setting only affects this codebase's own models.
If there's no terminal attached (e.g. a background service), it never blocks
waiting for an answer — it just uses CPU.

**This no longer has to be a one-time, fixed choice.** The sidebar's
**Performance** panel lets you change it live, no restart needed:

- **Compute device** — switch between CPU and GPU on demand. Flipping this
  rebuilds the embedder, reranker, and groundedness scorer on the new
  device (the vector store itself isn't touched, so nothing needs
  re-ingesting).
- **CPU usage** — three modes (`eco` / `balanced` / `max`) controlling how
  many CPU threads the local models are allowed to use. `eco` leaves the
  most headroom for whatever else is running on the machine; `max` uses
  every logical core for the fastest answers. This is persisted in
  `.performance_mode.json` (gitignored) and restored on the next run.
- **Live usage bars** — actual CPU%, RAM%, and (if present) GPU utilization
  and VRAM, polled every ~2.5s, so you can see when it's actually worth
  raising or lowering the mode instead of guessing.

The same controls are plain REST endpoints if you want to script them — see
the API table below (`/api/system/*`).

## Switching models

The **Model** panel in the sidebar lists every model currently pulled/created
in Ollama and lets you pick which one answers questions — no restart, no
config edit. Nothing about which models exist is hardcoded: `model_registry.py`
asks Ollama itself what's locally available and what each one can do
(`ollama.show(name).capabilities`), so a model you pull or create shows up
automatically the next time you hit refresh (⟳) — no code change needed to
start using it.

**To use your own fine-tuned/custom model**, it needs to be Ollama-compatible
(a GGUF checkpoint, or a model merged/exported to GGUF from your training
run). Import it once:

```powershell
ollama create my-model -f Modelfile
```

where `Modelfile` points at your `.gguf` file (see the
[Ollama docs](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)).
Once created, refresh the Model panel — it shows up alongside the rest, and
selecting it makes it the active model for every subsequent question. You
can also override the model for a single question via the API (`POST
/api/ask` with a `model` field) without changing the global default.

Model choice is **role-based**, not a single global value: there's an active
model for `chat` (answering questions) and one for `vision` (see below), each
persisted independently in `.model_choice.json` (gitignored). Adding a third
role in the future (e.g. a swappable embedding model) is one function call
in `model_prefs.py` — no new storage format, no new UI plumbing pattern.

## Multimodal: understanding images, not just OCR-ing them

Plain OCR only finds images that already contain text. A photo, a diagram,
or a chart has nothing for OCR to extract — so this app also runs images
through a vision-capable Ollama model, in two places:

1. **At ingest** — every image gets OCR'd *and* captioned (`vision.py`'s
   `caption()`), and both go into the indexed text. A text-free photo that
   used to ingest as "0 chunks" now gets a real, searchable description.
2. **At ask time** — if the best-matching retrieved source for a question is
   an image, the app asks the vision model to look at the *actual image*
   and answer directly (`pipeline.py`'s `_generate`), instead of only reusing
   the cached ingest-time caption. This is genuine visual Q&A: "what color
   is the shirt in pic2.jpeg" gets answered by looking at pic2.jpeg, not by
   pattern-matching a paragraph written about it earlier.

**No vision model is hardcoded.** `active_vision_model()` in `vision.py`
resolves to (in order): an explicit `RAG_VISION_MODEL` env var → whatever
you last picked in the Model panel's "Image understanding model" dropdown →
the first vision-capable model `model_registry.py` finds already pulled.
Pull any vision-capable model — `moondream` (small, fast, best for bulk
captioning), `qwen3.5:4b` if you already have it, `llava`, `llama3.2-vision`,
whatever — and it becomes selectable with no code change. If none is pulled
yet, images still get OCR'd; captioning and visual Q&A are simply skipped
rather than failing the ingest.

**A real finding worth knowing about**: very small vision models can be
excellent at open-ended captioning ("describe this image") while being
surprisingly brittle on terse factual questions ("what color is the
shirt?") — `moondream` in this project returned an empty response for the
latter but a full, accurate description for the former, on the *same
image*. `pipeline.py` treats an empty/near-empty vision-model answer as a
signal to fall back to the text LLM using the retrieved context (which
already includes that image's rich ingest-time caption) rather than
surfacing a blank answer — another instance of the "small local models are
brittle in specific, reproducible ways" pattern already documented for the
NLI groundedness scorer below.

**Speed vs. reliability is a real, visible tradeoff on CPU.** `moondream`
captions/answers in seconds but is the model that hits the brittleness above
most often; `qwen3.5:4b` answered every direct visual question correctly in
testing but took 1–2+ minutes per question on CPU (it defaults to an
extended "thinking" mode — `vision.py` passes `think=False` to skip that,
which helps but doesn't erase the size difference). There's no universally
"right" choice here: pick `moondream` for fast bulk captioning during
ingest, switch to a larger vision model in the Model panel for a specific
question you need answered accurately, and switch back after.

## Memory safety

A best-effort safety net, not an OS-level guarantee, but designed so a
single large or malformed file can't take down your machine or other
running programs:

- **Uploads are streamed to disk**, not read into memory in one shot, and
  capped at `RAG_MAX_UPLOAD_MB` (default 300MB).
- **Before reading any file**, the app checks that enough system memory is
  free relative to the file's size and skips it with a clear error rather
  than risking an out-of-memory crash.
- **OCR** checks available memory before each page and backs off if memory
  is tight.
- Bulk re-ingest skips a problem file and continues with the rest, rather
  than aborting the whole batch.

## Evaluation & benchmarking

```powershell
# Score the groundedness model against a labeled test set, sweep thresholds
python -m scripts.evaluate_groundedness --sweep

# Persist the best threshold as the new default
python -m scripts.evaluate_groundedness --calibrate

# Compare multiple local models on the same questions: answers, groundedness, latency
python -m scripts.compare_models --models llama3.2:3b,qwen3.5:4b --out results.json
```

`data/eval/groundedness_eval.jsonl` has 20 hand-labeled (context, sentence,
grounded/hallucinated) pairs to start from — extend it with your own
documents for a report-ready accuracy number.

## Testing

```powershell
pip install -r requirements-dev.txt
pytest -v
```

38 tests covering chunking, labels, memory guards, the groundedness scorer,
and the full API (upload → ingest → ask → delete). Tests that need a live
LLM skip themselves automatically if Ollama isn't running, so the suite
still passes in CI or on a machine without it set up.

## Docker

```powershell
docker compose up --build
```

Runs the app plus an Ollama container. Pull a model into it once with
`docker compose exec ollama ollama pull llama3.2:3b`. GPU passthrough isn't
configured by default (add `--gpus all` / the NVIDIA container toolkit if
you want it) — CPU works fine for both.

## Project structure

```
src/
  rag/
    config.py           settings (models, chunk size, top_k, thresholds)
    embeddings.py        sentence-transformer embedding wrapper
    reranker.py          cross-encoder relevance re-ranking
    vectorstore.py        Chroma persistent vector store wrapper (label-aware)
    labels.py             label registry (create/delete/list, ephemeral clearing)
    ingest.py             document loading, chunking, OCR/transcription, ingestion
    llm.py                Ollama chat client wrapper (multi-turn, dynamic active model)
    vision.py              image captioning + visual Q&A via the active vision model
    hallucination.py      NLI-based groundedness/hallucination scorer
    text_utils.py         shared sentence splitting
    pipeline.py           ties retrieval + rerank + generation + scoring + vision together
    device.py             GPU detection, permission prompt, runtime device/perf switching
    system_stats.py        live CPU/RAM/GPU usage for the resource panel
    model_registry.py      discovers pulled Ollama models + their capabilities (no hardcoded names)
    model_prefs.py         persists the active model per role (chat/vision/...) across restarts
    memory_guard.py       memory headroom checks (upload cap, OCR backoff)
  api/
    main.py              FastAPI app — see endpoints below; serves frontend/
  cli.py                 command-line entrypoint (ingest / ask)
  app.py                 Streamlit demo UI
frontend/
  index.html             app layout (labels sidebar, chat, settings)
  style.css              theme (light/dark), chat bubbles, groundedness meter
  app.js                 labels, upload, chat, citation highlighting, API wiring
scripts/
  evaluate_groundedness.py   accuracy eval + threshold calibration
  compare_models.py          multi-model benchmark (answers, groundedness, latency)
tests/                    pytest suite (isolated from real data — see conftest.py)
data/raw/                 documents, organized as data/raw/<label>/<file>
data/eval/                 labeled groundedness evaluation set
vectorstore/              persisted Chroma DB (gitignored)
```

### API endpoints

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Liveness check |
| `GET /api/labels` | List labels with document/chunk counts |
| `POST /api/labels` | Create a label |
| `DELETE /api/labels/{name}` | Delete a label and its documents |
| `POST /api/labels/{name}/clear` | Empty a label without deleting it |
| `GET /api/documents` | List ingested documents (optionally `?label=`) |
| `DELETE /api/documents?source=` | Delete a single document |
| `POST /api/upload` | Upload files (background job) into a label |
| `POST /api/ingest` | Re-scan `data/raw/` (background job) |
| `GET /api/jobs/{id}` | Poll a background job's status |
| `POST /api/ask` | Ask a question (`label`, `conversation_id`, `model` optional) |
| `POST /api/conversations/{id}/clear` | Forget a conversation's history |
| `GET /api/system/stats` | Live CPU/RAM/GPU usage, active device, performance mode |
| `GET /api/system/devices` | List available compute devices (CPU + any detected GPU) |
| `POST /api/system/device` | Switch device at runtime, reloads local models |
| `GET/POST /api/system/performance` | Get/set CPU usage mode (`eco`/`balanced`/`max`) |
| `GET /api/system/models` | List Ollama models with capabilities + active model per role |
| `POST /api/system/model` | Set the active model for a role (`{"model", "role": "chat"\|"vision"}`) |

## Roadmap / further improvements

Everything below is a genuine next step, not filler — ordered roughly by
effort-to-value for a student project:

- **Streaming answers** — Ollama supports token streaming; wiring that
  through `/api/ask` (Server-Sent Events or a WebSocket) would make answers
  appear incrementally instead of all at once, which reads as much faster.
- **Query rewriting for follow-ups** — a follow-up like "what about the
  second one?" retrieves poorly on its own since it's missing the referent.
  Rewriting it into a standalone question (using the LLM + conversation
  history) before retrieval would fix multi-turn retrieval accuracy.
- **Hybrid retrieval (BM25 + embeddings)** — pure embedding search misses
  exact keyword/number matches (part numbers, names, dates). Adding a
  keyword-based BM25 pass and merging it with vector search (reciprocal
  rank fusion) is a well-known accuracy win for RAG.
- **Per-document access control / multi-user support** — right now every
  label is visible to whoever opens the app. If this ever needs to serve
  more than one person, labels would need an owner and the API would need
  auth (even a simple API key per user).
- **Answer caching** — identical questions against an unchanged label
  currently re-run the whole pipeline. A cache keyed on
  `(question, label, model)` invalidated on ingest would cut latency and
  compute for repeated demo questions.
- **Structured-data QA** — CSV/XLSX are currently flattened to text chunks,
  which loses the ability to actually compute over the data (sums, filters,
  sorts). A text-to-SQL or pandas-agent path for spreadsheet labels would
  handle "what's the total in column X" — style questions correctly, which
  today's chunk-and-retrieve approach fundamentally can't.
- **GPU-aware batch sizing** — the reranker and NLI scorer currently send
  every candidate pair to the model in one batch; on a GPU with more
  headroom, batching more aggressively (and on CPU, less) would use the
  performance-mode setting more precisely than just a thread count.
- **Quantized/distilled model options** — offering a "fast" vs. "accurate"
  model pair (e.g. a smaller reranker/NLI model) alongside the performance
  mode, so a low-end CPU machine can trade some accuracy for speed instead
  of just fewer threads.

## Notes for the writeup / evaluation

- The groundedness score is a legitimate quantitative metric: at the default
  threshold this project's own eval set scores 95% accuracy / 100% precision
  / 90% recall (F1 0.95) — worth reporting alongside *how* you got there.
- **A concrete finding worth including**: the NLI model used here
  (`cross-encoder/nli-deberta-v3-small`) is highly sensitive to premise
  granularity and sentence voice — the same fact can score near 1.0 as a
  whole-paragraph premise but near 0 as a single sentence (or vice versa),
  and a passive-voice paraphrase can score near 0 even when a near-identical
  active-voice one scores near 1.0. `hallucination.py` mitigates this by
  checking each answer sentence against both the whole chunk *and* each of
  its individual sentences and keeping the best match — which took F1 from
  0.87 to 0.95 on the eval set. This is good material for a "limitations and
  mitigations" section: it's a real, reproducible failure mode of small NLI
  models, not a made-up caveat.
- `RAG_GROUNDEDNESS_THRESHOLD` / `scripts/evaluate_groundedness.py
  --calibrate` control the grounded/hallucinated cutoff — tune it against
  your own labeled sample for your report, not just this project's demo set.
- Use `scripts/compare_models.py` to report answer quality, groundedness,
  and latency across models as an ablation.
