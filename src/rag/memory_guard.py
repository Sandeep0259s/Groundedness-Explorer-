"""Best-effort memory safety net.

This is a Python-level guard, not an OS sandbox — it can't guarantee this
process never touches too much RAM, but it catches the common ways a single
large document could push a laptop into swapping and starving other
programs: a giant file read into memory in one shot, or a long page-by-page
OCR job that keeps allocating without checking in. Where we can, we back off
and wait a bit (memory pressure is often transient) before giving up with a
clear error instead of crashing.
"""
import gc
import time

import psutil

from .config import settings


class InsufficientMemoryError(RuntimeError):
    pass


def available_mb() -> float:
    return psutil.virtual_memory().available / (1024 * 1024)


def ensure_headroom(context: str = "") -> None:
    free = available_mb()
    if free >= settings.min_free_memory_mb:
        return

    gc.collect()
    free = available_mb()
    if free < settings.min_free_memory_mb:
        where = f" ({context})" if context else ""
        raise InsufficientMemoryError(
            f"Only {free:.0f}MB of system memory free (need at least "
            f"{settings.min_free_memory_mb}MB){where} — stopping before this risks "
            "crashing other programs. Close some applications and try again."
        )


def wait_for_headroom(context: str = "", max_wait_seconds: float = 30, poll_seconds: float = 2) -> None:
    waited = 0.0
    while available_mb() < settings.min_free_memory_mb and waited < max_wait_seconds:
        time.sleep(poll_seconds)
        waited += poll_seconds
    ensure_headroom(context)


def ensure_headroom_for_file(path_size_bytes: int, context: str = "") -> None:
    """A large file needs several times its size free (parsing overhead,
    OCR image buffers, embeddings) — check before reading it into memory."""
    required_mb = (path_size_bytes / (1024 * 1024)) * 3 + settings.min_free_memory_mb
    free = available_mb()
    if free < required_mb:
        where = f" ({context})" if context else ""
        raise InsufficientMemoryError(
            f"This file is large relative to available memory ({free:.0f}MB free, "
            f"~{required_mb:.0f}MB recommended){where} — skipping it to avoid crashing "
            "other programs. Free up memory or try a smaller file."
        )
