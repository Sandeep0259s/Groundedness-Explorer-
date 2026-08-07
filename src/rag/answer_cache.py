"""In-memory cache for /api/ask results, keyed on the question + retrieval
settings — a repeated demo question (common when presenting or just
re-testing something) skips the whole retrieve/rerank/generate pipeline
instead of re-running it, which matters on a CPU-only machine where a
single answer can take real seconds.

Deliberately **not** used for any question with conversation history: the
same question text can mean different things depending on what came before
it in a multi-turn conversation ("which continent is it on?"), and caching
purely on question text would silently serve the wrong cross-conversation
answer. Restricting this to history-less (first-turn / stateless) questions
sidesteps that correctness risk entirely rather than trying to fold history
into the cache key.

Invalidated wholesale (not per-document) on any ingest/upload/delete/label
mutation or model/device switch — simple and always correct for a small
local single-user cache; a stale answer is a worse failure mode than an
occasional unnecessary recompute.
"""
import hashlib
import json
import time

_cache: dict[str, dict] = {}
_MAX_ENTRIES = 200  # small local app — bound memory rather than never evicting


def _key(question: str, label: str | None, model: str | None, top_k: int) -> str:
    raw = json.dumps({"q": question.strip().lower(), "label": label, "model": model, "top_k": top_k}, sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()


def get(question: str, label: str | None, model: str | None, top_k: int) -> dict | None:
    entry = _cache.get(_key(question, label, model, top_k))
    if entry is None:
        return None
    return {k: v for k, v in entry.items() if k != "_cached_at"}


def put(question: str, label: str | None, model: str | None, top_k: int, result: dict) -> None:
    if len(_cache) >= _MAX_ENTRIES:
        oldest_key = min(_cache, key=lambda k: _cache[k]["_cached_at"])
        _cache.pop(oldest_key, None)
    _cache[_key(question, label, model, top_k)] = {**result, "_cached_at": time.time()}


def clear() -> None:
    _cache.clear()


def size() -> int:
    return len(_cache)
