"""Text-to-pandas Q&A for CSV/XLSX-labeled questions.

Chunk-and-retrieve treats a spreadsheet as flattened text, which loses the
ability to actually compute over it — "what's the total in column X" can't
be answered correctly from a retrieved text fragment, only by running a real
aggregation against the data. This asks the active chat model to translate
the question into a single pandas expression against the real DataFrame,
then evaluates it in a restricted namespace and returns the result.

Security note: this executes model-generated code. `_safe_eval` restricts
the namespace to a small allowlist and rejects anything that looks like an
import/attribute-escape/IO attempt, but this is NOT a real sandbox (no
process isolation, no resource limits) — an accepted tradeoff for a local,
single-user app talking to a local model, not something to expose to
untrusted multi-tenant use without adding real sandboxing (e.g. running the
eval in a separate resource-limited subprocess).
"""
import re
from pathlib import Path

import ollama
import pandas as pd

from . import model_prefs
from .config import settings

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


def _load_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


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
    active = model or model_prefs.load_active_model("chat", settings.ollama_model)
    client = ollama.Client(host=settings.ollama_host)
    response = client.chat(model=active, messages=[{"role": "user", "content": prompt}])
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


def answer_structured_question(path: Path, question: str, model: str | None = None) -> str | None:
    """Returns an answer string, or None if this question couldn't be
    answered this way — the caller should fall back to the text pipeline."""
    try:
        df = _load_dataframe(path)
        code = _generate_expression(df, question, model)
        result = _safe_eval(code, df)
    except Exception as exc:
        print(f"Structured QA fell back to text for {path.name}: {exc}")
        return None

    result_str = result.to_string() if isinstance(result, (pd.Series, pd.DataFrame)) else str(result)
    return f"{result_str}\n\n(computed as `{code}` over {path.name})"
