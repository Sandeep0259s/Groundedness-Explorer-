import pytest

from src.rag.memory_guard import InsufficientMemoryError, available_mb, ensure_headroom, ensure_headroom_for_file


def test_available_mb_is_positive():
    assert available_mb() > 0


def test_ensure_headroom_passes_under_normal_conditions():
    ensure_headroom("test")  # should not raise on a normal dev machine


def test_ensure_headroom_for_file_passes_for_small_file():
    ensure_headroom_for_file(1024, context="tiny file")  # 1KB — should not raise


def test_ensure_headroom_for_file_rejects_absurdly_large_file():
    huge = 10 * 1024 * 1024 * 1024 * 1024  # 10TB
    with pytest.raises(InsufficientMemoryError):
        ensure_headroom_for_file(huge, context="huge file")
