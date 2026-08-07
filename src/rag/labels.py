"""Labels are user-defined collections (like folders) documents get filed
under — e.g. "resume", "coursework" — so retrieval and browsing can be
scoped to one collection instead of searching everything at once.

A label marked "ephemeral" (the built-in "session" label) has its contents
wiped automatically every time the server starts, for throwaway documents
you only need for one sitting.
"""
import json
import shutil
from pathlib import Path

from .config import settings

DEFAULT_LABEL = "general"
SESSION_LABEL = "session"

LABELS_FILE = Path(settings.data_dir).resolve().parent / "labels.json"
_VALID_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")


def _label_dir(name: str) -> Path:
    return Path(settings.data_dir) / name


def label_dir(name: str) -> Path:
    return _label_dir(name)


def _load_registry() -> dict:
    try:
        return json.loads(LABELS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save_registry(registry: dict) -> None:
    LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
    LABELS_FILE.write_text(json.dumps(registry, indent=2))


def validate_name(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("label name must not be empty")
    if not set(name) <= _VALID_NAME_CHARS:
        raise ValueError("label names may only contain letters, numbers, '-' and '_'")
    return name


def ensure_default_labels() -> None:
    registry = _load_registry()
    changed = False
    for name, ephemeral in ((DEFAULT_LABEL, False), (SESSION_LABEL, True)):
        if name not in registry:
            registry[name] = {"ephemeral": ephemeral}
            changed = True
    if changed:
        _save_registry(registry)
    for name in registry:
        _label_dir(name).mkdir(parents=True, exist_ok=True)


def list_labels() -> list[dict]:
    return [{"name": name, **meta} for name, meta in sorted(_load_registry().items())]


def label_exists(name: str) -> bool:
    return name in _load_registry()


def create_label(name: str, ephemeral: bool = False) -> dict:
    name = validate_name(name)
    registry = _load_registry()
    if name not in registry:
        registry[name] = {"ephemeral": ephemeral}
        _save_registry(registry)
    _label_dir(name).mkdir(parents=True, exist_ok=True)
    return {"name": name, **registry[name]}


def delete_label(name: str) -> None:
    if name == DEFAULT_LABEL:
        raise ValueError(f"the '{DEFAULT_LABEL}' label can't be deleted")
    registry = _load_registry()
    registry.pop(name, None)
    _save_registry(registry)
    shutil.rmtree(_label_dir(name), ignore_errors=True)


def clear_label_contents(name: str) -> None:
    directory = _label_dir(name)
    if not directory.exists():
        return
    for child in directory.iterdir():
        if child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child, ignore_errors=True)


def ephemeral_label_names() -> list[str]:
    return [name for name, meta in _load_registry().items() if meta.get("ephemeral")]
