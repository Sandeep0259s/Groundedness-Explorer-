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
2. **Retrieve (hybrid) + re-rank** — a follow-up question is first rewritten
   into a standalone one using conversation history ("how tall is it?" →
   "how tall is the Eiffel Tower?") purely for retrieval purposes, then
   matched against the vector store two ways at once: embedding similarity
   (catches paraphrases) and BM25 keyword search (catches exact names, part
   numbers, dates that embeddings often miss), merged by reciprocal rank
   fusion. A second, purpose-built cross-encoder then re-scores those fused
   candidates for actual relevance and narrows them down to the final
   top-k — each stage catches what the previous one misses.
3. **Ask** — a local LLM (via [Ollama](https://ollama.com)) generates an
   answer grounded in those chunks, aware of the recent conversation history
   (persisted to SQLite, so an open conversation survives a restart) for
   natural follow-up questions. Answers stream token-by-token over SSE.
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
| Spreadsheets | `.csv`, `.xlsx` | Rows flattened to text for retrieval; computation questions ("what's the total in column X") are answered by running a real pandas expression instead — see below |
| Images | `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp` | OCR via Tesseract + captioning via a vision model |
| Video | `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm` | Speech-to-text (faster-whisper) + 3 keyframes captioned via the vision model |
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

## Structured-data Q&A for spreadsheets

Chunk-and-retrieve treats a spreadsheet row as a flat sentence, which works
for "what's in row 3" but not "what's the total of column X" — that needs a
real aggregation over the actual data, not a retrieved text fragment. When
the best-matching source for a question is a `.csv`/`.xlsx` file,
`structured_qa.py` instead asks the active chat model to translate the
question into a single pandas expression (given the real column names and a
few sample rows), evaluates it against the actual DataFrame, and returns
the computed result — tagged `answer_mode: "structured"` in the API and
shown as "🧮 Answered by computing over the spreadsheet" in the UI.

**Security note, stated plainly**: this executes model-generated code.
`_safe_eval` runs it in a restricted namespace — only `df` itself is
exposed, no builtins beyond a small allowlist, and **the `pandas` module
itself is deliberately left out of the namespace entirely**, not just
denylisted by name. An earlier version exposed `pd` alongside a denylist of
dangerous substrings (`import`, `os.`, etc.), which still let
`pd.read_pickle(...)` / `pd.read_html(...)` / arbitrary network reads
through — no legitimate aggregate/filter expression this feature needs
ever calls `pd.` directly, so removing the module closes that whole class
of attack in one move instead of trying to enumerate every dangerous
function by name. The denylist is now defense-in-depth on top of that
(also blocking the `df.to_pickle(...)`/`df.to_csv(...)` family, which
remain reachable through `df` itself). This is still **not a real
sandbox**: there's no process isolation or resource limit behind it — an
accepted tradeoff for a local, single-user app talking to a local model you
already trust, not something to expose to untrusted multi-tenant use
without adding real sandboxing (e.g. running the eval in a resource-limited
subprocess). `tests/test_structured_qa.py` has concrete examples of what it
blocks.

Groundedness scoring is skipped for structured answers and reported as
`{"label": "computed", "overall_score": 1.0}` instead of run through the
NLI scorer — comparing a bare number like `700` against a prose chunk isn't
a meaningful entailment pair, and doing it anyway produced a consistently
misleading "possibly hallucinated" score for answers that were, in fact,
computed correctly. The real risk with this feature is a wrong *translation*
of the question into code, which is a different failure mode than
hallucination and isn't something the NLI scorer can measure.

## Other hardening from a full code-review pass

A handful of real bugs surfaced from a systematic review of everything
added this session, beyond the ones already called out inline above:

- **Frontend XSS**: an uploaded filename or a document's own text content
  was interpolated straight into `innerHTML` in a few places (the source
  list, the image thumbnail's `alt` attribute) with no escaping — a
  crafted filename could break out of its attribute and run script in the
  browser. Fixed with a proper `escapeHtml()` used everywhere untrusted
  text reaches the DOM as markup.
- **Hybrid retrieval crash**: a BM25-only hit (no embedding match at all)
  had no `distance` key, and the frontend unconditionally called
  `hit.distance.toFixed(3)` — a `TypeError` on any question where keyword
  search alone found the answer. Fixed by giving every hit a consistent
  shape and rendering "keyword match" instead of a distance when there
  isn't one.
- **SQLite connections were never closed** in `conversation_store.py` (only
  committed), leaking a file handle on every single question; fixed with a
  proper context manager, plus a connect timeout so two near-simultaneous
  requests on the same conversation wait for each other instead of raising
  "database is locked."
- **`answer_cache.py` had no locking** around its eviction scan, so two
  concurrent requests could raise "dictionary changed size during
  iteration"; fixed with a lock around every read/write.
- **Two answer-mode thresholds had silently drifted**: `ask()` required a
  vision answer to be ≥3 characters to count as usable, `ask_stream()`
  accepted anything non-empty — meaning a correct short answer like "3" or
  "No" was accepted by the streaming endpoint and rejected by the
  non-streaming one for the same question. Unified to one shared constant.
- **`query_hybrid()` had no memory guard**, unlike every other memory-heavy
  path in this codebase — it pulls the whole matching collection into
  Python to rebuild BM25 on every question. It now checks the same
  `min_free_memory_mb` headroom the rest of the app respects and degrades
  to embeddings-only instead of risking a crash.
- Cache invalidation on ingest/upload now runs on failure too (a `finally`,
  not just the success path) — a partially-failed batch may have already
  ingested earlier files before erroring, and the old code left stale
  answers cached against documents that did land.

## Answer caching

A repeated question — common when demoing or just re-testing something —
skips the whole retrieve/rerank/generate pipeline via an in-memory cache in
`answer_cache.py`, keyed on the exact `(question, label, model, top_k)`.
Measured on the machine this was built on: **~20.5s → ~0.36s** on a cache
hit, both non-streaming and streaming (`/api/ask` and `/api/ask/stream`
both check it; a streaming cache hit sends the whole answer as one
"token" event instead of a fake delay). Every response includes a `cached`
boolean so this is visible, not silent — the UI shows a small "⚡ Cached
answer" tag when it fires.

Deliberately **not** applied to any question with conversation history —
the same question text can mean something different mid-conversation
("which continent is it on?"), and caching on text alone would risk
serving a cross-conversation wrong answer. Only stateless (first-turn)
questions are cached.

The cache is cleared wholesale (not per-document) on any ingest/upload/
delete/label mutation or chat/vision model switch — simple and always
correct for a small local single-user cache; a stale answer is a worse
failure mode than an occasional unnecessary recompute.

## Casual conversation, not just document Q&A

Early on, a plain "hi" got refused with "I don't know" and labeled
**possibly hallucinated** — a real, reproducible bug, not a hypothetical
edge case. The cause: `llm.py`'s prompt forced *every* message through
"answer using only the provided context," and the groundedness scorer then
NLI-scored the refusal against whatever documents happened to be retrieved,
regardless of whether the message was actually a question about them.

Two fixes, both addressing the actual mechanism rather than pattern-matching
a list of greeting words:

1. The system prompt now tells the model the message might be a genuine
   question *or* just a greeting/casual remark, and to answer the latter
   naturally without needing the context at all.
2. `pipeline.py` reuses a signal the reranker already computes for every
   question — its own relevance score for each candidate — and reports
   groundedness as **"no relevant context found"** (not "possibly
   hallucinated") whenever the reranker's best candidate scores below
   `RAG_RELEVANCE_THRESHOLD` (default `0.0`). A greeting reliably retrieves
   nothing the cross-encoder considers relevant, so this generalizes to any
   off-topic message, not just a hardcoded set of greeting strings.

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

# RAGAS-style: faithfulness, answer relevancy, context precision — against the live pipeline
python -m scripts.evaluate_ragas --questions questions.txt --out ragas_results.json
```

`data/eval/groundedness_eval.jsonl` has 20 hand-labeled (context, sentence,
grounded/hallucinated) pairs to start from — extend it with your own
documents for a report-ready accuracy number.

`scripts/evaluate_ragas.py` adds two RAGAS metrics `evaluate_groundedness.py`
doesn't cover — **answer relevancy** (does the answer actually address the
question, independent of whether its claims are grounded) and **context
precision** (what fraction of retrieved chunks were actually useful, not
just superficially similar) — alongside faithfulness, which is intentionally
the same technique as the groundedness scorer above, reported under RAGAS's
name for a reviewer who knows the standard metric. Run against this
project's own tiny 3-document demo corpus with `top_k=4`, context precision
comes out around 0.25 — a real, honest finding: with only a few documents
loaded, most of `top_k` gets filled with genuinely irrelevant chunks just to
reach the requested count. It's evidence the metric works, not a bug to fix
in the demo data.

## Testing

```powershell
pip install -r requirements-dev.txt
pytest -v
```

82 tests covering chunking, labels, memory guards, the groundedness scorer,
hybrid retrieval, structured-data QA safety checks, video keyframe
extraction, conversation persistence, answer caching, query rewriting,
RAGAS-style metrics, and the full API (upload → ingest → ask → delete).
Tests that need a live LLM skip themselves automatically if Ollama isn't
running, so the suite still passes in CI or on a machine without it set up.

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
    vectorstore.py        Chroma vector store wrapper (label-aware, hybrid BM25+embedding retrieval)
    labels.py             label registry (create/delete/list, ephemeral clearing)
    ingest.py             document loading, chunking, OCR/transcription, ingestion
    llm.py                Ollama chat client wrapper (multi-turn, dynamic active model)
    vision.py              image captioning + visual Q&A via the active vision model
    hallucination.py      NLI-based groundedness/hallucination scorer
    text_utils.py         shared sentence splitting
    pipeline.py           ties retrieval + rerank + generation + scoring + vision + structured QA together
    structured_qa.py        text-to-pandas Q&A for spreadsheet questions
    device.py             GPU detection, permission prompt, runtime device/perf switching
    system_stats.py        live CPU/RAM/GPU usage for the resource panel
    model_registry.py      discovers pulled Ollama models + their capabilities (no hardcoded names)
    model_prefs.py         persists the active model per role (chat/vision/...) across restarts
    memory_guard.py       memory headroom checks (upload cap, OCR backoff)
    conversation_store.py  SQLite-backed multi-turn conversation history
    answer_cache.py         in-memory cache for repeated stateless questions
    query_rewrite.py        rewrites follow-ups into standalone questions for retrieval
    ragas_eval.py           faithfulness / answer relevancy / context precision
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
  evaluate_ragas.py           faithfulness / answer relevancy / context precision eval
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
| `GET /api/documents/file?source=` | Serve an ingested image's raw bytes (for thumbnails) |
| `DELETE /api/documents?source=` | Delete a single document |
| `POST /api/upload` | Upload files (background job, with per-file `progress`) into a label |
| `POST /api/ingest` | Re-scan `data/raw/` (background job, with per-file `progress`) |
| `GET /api/jobs/{id}` | Poll a background job's status |
| `POST /api/ask` | Ask a question (`label`, `conversation_id`, `model` optional); response includes `answer_mode` (`text`/`vision`/`vision_fallback`/`structured`) and `cached` |
| `POST /api/ask/stream` | Same as `/api/ask`, streamed as Server-Sent Events (`event: token` then `event: done`) |
| `POST /api/conversations/{id}/clear` | Forget a conversation's history |
| `GET /api/system/stats` | Live CPU/RAM/GPU usage, active device, performance mode |
| `GET /api/system/devices` | List available compute devices (CPU + any detected GPU) |
| `POST /api/system/device` | Switch device at runtime, reloads local models |
| `GET/POST /api/system/performance` | Get/set CPU usage mode (`eco`/`balanced`/`max`) |
| `GET /api/system/models` | List Ollama models with capabilities + active model per role |
| `POST /api/system/model` | Set the active model for a role (`{"model", "role": "chat"\|"vision_caption"\|"vision_answer"}`) |

## Roadmap / further improvements

Everything below is a genuine next step, not filler — ordered roughly by
effort-to-value for a student project.

**Done** (kept here so the list doesn't misrepresent itself as all-future):
streaming answers over SSE, hybrid BM25+embedding retrieval, structured-data
QA for spreadsheets, persistent (SQLite-backed) conversation history, image
captioning + visual Q&A, video keyframe captioning, per-role model
switching, runtime GPU/CPU + performance control, answer caching for
repeated stateless questions (measured ~57x faster on a cache hit — 20.5s
to 0.36s — on the CPU this was built on), and query rewriting for follow-up
questions (see `query_rewrite.py` — a follow-up like "how tall is it?" gets
rewritten to something like "how tall is the Eiffel Tower?" before
retrieval only, so it actually finds the right chunk).

**Still open:**

- **Per-document access control / multi-user support** — right now every
  label is visible to whoever opens the app. If this ever needs to serve
  more than one person, labels would need an owner and the API would need
  auth (even a simple API key per user).
- **A real sandbox for structured QA** — `structured_qa.py`'s restricted
  namespace + denylist is a reasonable local-app tradeoff (see its section
  above) but isn't a real sandbox; running the eval in a resource-limited
  subprocess would be the honest next step before this ever left a
  single-user local context.
- **GPU-aware batch sizing** — the reranker and NLI scorer currently send
  every candidate pair to the model in one batch; on a GPU with more
  headroom, batching more aggressively (and on CPU, less) would use the
  performance-mode setting more precisely than just a thread count.
- **Quantized/distilled model options** — offering a "fast" vs. "accurate"
  model pair (e.g. a smaller reranker/NLI model) alongside the performance
  mode, so a low-end CPU machine can trade some accuracy for speed instead
  of just fewer threads.
- **A standardized RAG eval suite** — the custom groundedness metric is
  good, reproducible material, but adding faithfulness/answer-relevance/
  context-precision numbers alongside it (RAGAS-style) would give numbers a
  reviewer already recognizes, not just a bespoke one.

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
- **Hybrid retrieval ablation**: `vectorstore.py`'s `query()` (embeddings
  only) and `query_hybrid()` (embeddings + BM25 via reciprocal rank fusion)
  are both available, so you can report retrieval quality with and without
  the BM25 leg on your own document set — `tests/test_hybrid_retrieval.py`
  has a concrete, reproducible example (an exact part-number token that
  embedding similarity alone ranks poorly but BM25 finds immediately).
