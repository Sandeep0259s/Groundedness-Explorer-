"""Persists which model is active for each role ("chat", "vision", ...), so
a switch made in the UI survives a restart.

Roles are just string keys into one small JSON file — adding a new role
later (e.g. a swappable embedding model) is calling load_active_model /
save_active_model with a new role name, no new storage plumbing required.
"""
import json
from pathlib import Path

MODEL_PREF_FILE = Path(__file__).resolve().parent.parent.parent / ".model_choice.json"


def _load_all() -> dict:
    try:
        return json.loads(MODEL_PREF_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def load_active_model(role: str, default: str | None = None) -> str | None:
    return _load_all().get(role, default)


def save_active_model(role: str, name: str) -> None:
    data = _load_all()
    data[role] = name
    try:
        MODEL_PREF_FILE.write_text(json.dumps(data))
    except OSError:
        pass  # best-effort — worst case it resets to the default next run
