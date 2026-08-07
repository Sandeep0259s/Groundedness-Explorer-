"""Picks CPU or GPU for the local embedding/scoring models.

A GPU is only ever used with the user's consent: the first time one is
detected we ask on the terminal, then remember the answer (keyed to that
specific GPU) so we don't ask again. Non-interactive processes (no TTY —
e.g. a server started from a script) never block on input() and default
to CPU, since a hidden prompt would just hang forever.

The device and CPU "performance mode" can also be changed at runtime (e.g.
from the web UI's resource-usage panel) via `set_device()` /
`set_performance_mode()`, instead of only being fixed at process start.
"""
import json
import os
import sys
import threading
from pathlib import Path

from .config import settings

PREFERENCE_FILE = Path(__file__).resolve().parent.parent.parent / ".device_choice.json"
PERFORMANCE_FILE = Path(__file__).resolve().parent.parent.parent / ".performance_mode.json"

# Thread count torch uses for CPU work in each mode. "max" (0) means "use
# every logical core" — fastest, but leaves the least headroom for other
# programs running at the same time; "eco" leaves the most headroom.
PERFORMANCE_THREADS = {"eco": 2, "balanced": 4, "max": 0}

_resolved_device: str | None = None
_performance_mode = "balanced"
_lock = threading.Lock()


def _detect_gpu() -> tuple[str | None, str | None]:
    try:
        import torch
    except ImportError:
        return None, None

    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", "Apple Silicon GPU (MPS)"
    return None, None


def _load_preference() -> dict | None:
    try:
        return json.loads(PREFERENCE_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _save_preference(signature: str, use_gpu: bool) -> None:
    try:
        PREFERENCE_FILE.write_text(json.dumps({"signature": signature, "use_gpu": use_gpu}))
    except OSError:
        pass  # best-effort — worst case we ask again next run


def resolve_device() -> str:
    """Returns "cpu", "cuda", or "mps". Cached for the life of the process."""
    global _resolved_device
    if _resolved_device is not None:
        return _resolved_device

    if settings.device in ("cpu", "cuda", "mps"):
        _resolved_device = settings.device
        return _resolved_device

    device_kind, gpu_name = _detect_gpu()
    if device_kind is None:
        _resolved_device = "cpu"
        return _resolved_device

    pref = _load_preference()
    if pref and pref.get("signature") == gpu_name:
        _resolved_device = device_kind if pref["use_gpu"] else "cpu"
        return _resolved_device

    if sys.stdin.isatty():
        answer = input(
            f"\nGPU detected: {gpu_name}. Use it to speed up embeddings and "
            f"groundedness scoring? [y/N]: "
        ).strip().lower()
        use_gpu = answer == "y"
    else:
        use_gpu = False  # can't prompt — never hang a non-interactive process

    _save_preference(gpu_name, use_gpu)
    _resolved_device = device_kind if use_gpu else "cpu"
    return _resolved_device


def get_current_device() -> str:
    """Same as resolve_device(), named for symmetry with set_device()."""
    return resolve_device()


def available_devices() -> list[dict]:
    """CPU is always available; a GPU entry is added only if one was detected."""
    devices = [{"id": "cpu", "name": "CPU"}]
    device_kind, gpu_name = _detect_gpu()
    if device_kind:
        devices.append({"id": device_kind, "name": gpu_name})
    return devices


def set_device(device_id: str) -> None:
    """Switch the active device at runtime. Raises ValueError if unavailable.

    Callers must reload any already-constructed models (embedder, reranker,
    NLI scorer) afterward — this only changes what resolve_device() returns
    for models built from this point on.
    """
    global _resolved_device
    valid_ids = {d["id"] for d in available_devices()}
    if device_id not in valid_ids:
        raise ValueError(f"'{device_id}' is not available on this machine (have: {sorted(valid_ids)})")

    with _lock:
        _resolved_device = device_id

    _, gpu_name = _detect_gpu()
    if gpu_name:
        _save_preference(gpu_name, device_id != "cpu")


def get_performance_mode() -> str:
    return _performance_mode


def set_performance_mode(mode: str) -> None:
    """Adjusts how many CPU threads the local models are allowed to use.

    This is the practical lever for "use more/less CPU" on a machine with no
    GPU: fewer threads leaves more headroom for other programs (at the cost
    of slower embedding/scoring), more threads is faster but more intrusive.
    """
    global _performance_mode
    if mode not in PERFORMANCE_THREADS:
        raise ValueError(f"mode must be one of {sorted(PERFORMANCE_THREADS)}")

    with _lock:
        _performance_mode = mode

    try:
        import torch
        torch.set_num_threads(PERFORMANCE_THREADS[mode] or (os.cpu_count() or 4))
    except ImportError:
        pass

    try:
        PERFORMANCE_FILE.write_text(json.dumps({"mode": mode}))
    except OSError:
        pass  # best-effort — worst case it resets to "balanced" next run


def apply_saved_performance_mode() -> None:
    """Called once at startup to restore the last-chosen performance mode."""
    try:
        mode = json.loads(PERFORMANCE_FILE.read_text())["mode"]
    except (OSError, json.JSONDecodeError, KeyError):
        mode = "balanced"
    set_performance_mode(mode)
