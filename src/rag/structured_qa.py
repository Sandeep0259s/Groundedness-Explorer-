"""Text-to-pandas Q&A for CSV/XLSX-labeled questions.

Chunk-and-retrieve treats a spreadsheet as flattened text, which loses the
ability to actually compute over it — "what's the total in column X" can't
be answered correctly from a retrieved text fragment, only by running a real
aggregation against the data. This asks the active chat model to translate
the question into a single pandas expression against the real DataFrame,
then evaluates it in a restricted namespace and returns the result.

Security note: this executes model-generated code. `_safe_eval` restricts
the namespace to a small allowlist (no `pandas` module, no dangerous
builtins) and rejects anything that looks like an import/attribute-escape/
IO attempt — and `_run_sandboxed` then runs that already-restricted eval in
a separate child process with a hard timeout and (on POSIX) a memory
ceiling, so a crash or a runaway loop in a generated expression can't take
down or hang the API server itself. This is real process isolation, not
just a namespace restriction — but it's still a single-machine, same-user
sandbox (no container, no seccomp/network namespace), an accepted tradeoff
for a local, single-user app talking to a local model you already trust.
"""
import multiprocessing as mp
import re
from pathlib import Path

import pandas as pd

from . import ollama_client

_EVAL_TIMEOUT_SECONDS = 5.0
_EVAL_MEMORY_LIMIT_BYTES = 512 * 1024 * 1024

SPREADSHEET_SUFFIXES = {".csv", ".xlsx"}

_CODE_PROMPT = """You are translating a question into a single pandas expression.
The DataFrame is called `df`. Its columns are: {columns}
Sample rows:
{sample}

Question: {question}

Reply with ONLY a single-line pandas expression that computes the answer
(e.g. df['Price'].sum(), df[df['Category'] == 'Books']['Price'].mean()).
No explanation, no markdown, no assignment, no print statement — just the
bare expression, nothing else."""

_BLOCKED_TOKENS = (
    "import", "__", "open(", "exec(", "eval(", "compile(", "os.", "sys.", "subprocess",
    "globals", "locals", "getattr", "setattr", "delattr",
    # DataFrame/Series I/O methods (df.to_pickle(...), df.to_csv(...), ...) are never
    # needed to *compute* an answer, only to write/exfiltrate — block the whole family
    # rather than trying to enumerate every dangerous one-off individually.
    "to_pickle", "to_csv", "to_excel", "to_html", "to_sql", "to_json", "to_parquet",
    "to_feather", "to_hdf", "to_clipboard", "read_pickle", "read_html", "read_sql",
    "read_json", "read_parquet", "read_feather", "read_hdf", "read_clipboard",
    "http://", "https://", "ftp://", "requests", "urllib", "socket",
)
_SAFE_BUILTINS = {"len": len, "sum": sum, "min": min, "max": max, "round": round, "abs": abs, "sorted": sorted}


class StructuredQAError(RuntimeError):
    """Raised when the question can't be safely/successfully answered this
    way — callers should fall back to the normal text pipeline."""


_dataframe_cache: dict[Path, tuple[float, pd.DataFrame]] = {}


def _load_dataframe(path: Path) -> pd.DataFrame:
    """Cached by (path, mtime) — every question against the same
    unmodified spreadsheet used to re-read and re-parse the whole file from
    scratch; a repeated question, or a second question about the same
    file, now skips that entirely. The mtime check means an edited file
    still gets re-parsed rather than silently serving stale data."""
    mtime = path.stat().st_mtime
    cached = _dataframe_cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]

    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    _dataframe_cache[path] = (mtime, df)
    return df


def _extract_expression(raw: str) -> str:
    # Models often wrap code in a fenced block despite instructions not to —
    # strip that, then keep only the first non-empty line so this can never
    # become multi-statement code.
    stripped = re.sub(r"^```(?:python)?\s*|\s*```$", "", raw.strip(), flags=re.MULTILINE).strip()
    for line in stripped.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _generate_expression(df: pd.DataFrame, question: str, model: str | None = None) -> str:
    prompt = _CODE_PROMPT.format(
        columns=", ".join(str(c) for c in df.columns),
        sample=df.head(3).to_string(),
        question=question,
    )
    response = ollama_client.get_client().chat(
        model=ollama_client.active_chat_model(model), messages=[{"role": "user", "content": prompt}]
    )
    return _extract_expression(response["message"]["content"])


def _safe_eval(code: str, df: pd.DataFrame):
    if not code or any(token in code for token in _BLOCKED_TOKENS):
        raise StructuredQAError(f"generated code failed a safety check: {code!r}")
    # `pd` is deliberately NOT exposed here: every legitimate aggregate/filter
    # expression this feature needs only touches `df` itself, and leaving the
    # full pandas module out closes the whole pd.read_pickle/pd.read_html/
    # pd.io.* class of attacks in one move instead of trying to enumerate
    # every dangerous module-level function by name.
    namespace = {"df": df, "__builtins__": _SAFE_BUILTINS}
    return eval(code, namespace)  # noqa: S307 — restricted namespace, single expression, see module docstring


def _sandboxed_worker(code: str, df: pd.DataFrame, queue: "mp.Queue") -> None:
    try:
        try:
            import resource  # POSIX-only — no-ops out on Windows via the except below

            resource.setrlimit(resource.RLIMIT_AS, (_EVAL_MEMORY_LIMIT_BYTES, _EVAL_MEMORY_LIMIT_BYTES))
        except (ImportError, ValueError, OSError):
            pass  # best-effort: no hard memory ceiling on this platform, still process-isolated
        queue.put(("ok", _safe_eval(code, df)))
    except Exception as exc:
        queue.put(("error", str(exc)))


def _run_sandboxed(code: str, df: pd.DataFrame, timeout: float = _EVAL_TIMEOUT_SECONDS):
    """Runs _safe_eval in a separate process rather than this one — a real
    process boundary and a hard timeout, on top of (not instead of)
    _safe_eval's own namespace restrictions. This is what actually closes
    the gap the module docstring used to just disclose: a crash or a
    runaway loop in a generated expression can no longer take down or hang
    the API server itself, only the disposable child process running it.
    A memory ceiling is applied where the platform supports it (POSIX);
    Windows still gets process isolation and the timeout, just not a hard
    memory cap — `resource` doesn't exist there."""
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_sandboxed_worker, args=(code, df, queue))
    proc.start()
    proc.join(timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        raise StructuredQAError(f"evaluation timed out after {timeout}s: {code!r}")

    if queue.empty():
        raise StructuredQAError(f"evaluation process exited unexpectedly (code {proc.exitcode}): {code!r}")

    status, payload = queue.get()
    if status == "error":
        raise StructuredQAError(payload)
    return payload


def answer_structured_question(path: Path, question: str, model: str | None = None) -> str | None:
    """Returns an answer string, or None if this question couldn't be
    answered this way — the caller should fall back to the text pipeline."""
    try:
        df = _load_dataframe(path)
        code = _generate_expression(df, question, model)
        result = _run_sandboxed(code, df)
    except Exception as exc:
        print(f"Structured QA fell back to text for {path.name}: {exc}")
        return None

    result_str = result.to_string() if isinstance(result, (pd.Series, pd.DataFrame)) else str(result)
    return f"{result_str}\n\n(computed as `{code}` over {path.name})"
