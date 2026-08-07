"""Discovers what Ollama models are actually pulled/created locally and what
each one can do (chat, vision, tools, ...), using Ollama's own model
metadata instead of a hardcoded list of names.

This is the single source of truth for "what models exist" — nothing else
in this codebase should hardcode a model name to decide whether it can be
used for chat vs. vision. Pull or create any model with the Ollama CLI
(`ollama pull <name>` / `ollama create <name> -f Modelfile`) and it becomes
selectable everywhere in the app the next time this is queried — no code
change needed.
"""
from functools import lru_cache

import ollama

from .config import settings


def _client() -> ollama.Client:
    return ollama.Client(host=settings.ollama_host)


@lru_cache(maxsize=64)
def _capabilities(name: str) -> tuple[str, ...]:
    """Cached per model name for the life of the process — a model's
    capabilities don't change without a new pull/create under that name."""
    try:
        info = _client().show(name)
        return tuple(getattr(info, "capabilities", None) or [])
    except Exception:
        return ()


def list_models() -> list[dict]:
    """[{"name": "llama3.2:3b", "capabilities": ["completion", "tools"]}, ...]

    Raises whatever ollama.Client().list() raises (e.g. connection error) —
    callers decide how to surface that.
    """
    raw = _client().list()
    models = [{"name": m.model, "capabilities": list(_capabilities(m.model))} for m in raw.models]
    return sorted(models, key=lambda m: m["name"])


def models_with_capability(capability: str) -> list[str]:
    try:
        return [m["name"] for m in list_models() if capability in m["capabilities"]]
    except Exception:
        return []  # Ollama unreachable — treat as "nothing available" rather than crash


def first_pulled(candidates: list[str]) -> str | None:
    """First name from `candidates` that's actually pulled locally — a way
    to pick a sensible fallback without assuming any specific one exists."""
    try:
        pulled = {m["name"] for m in list_models()}
    except Exception:
        return None
    return next((n for n in candidates if n in pulled), None)
