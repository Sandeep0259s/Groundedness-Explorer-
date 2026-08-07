"""Live CPU/RAM/GPU usage for the web UI's resource-usage panel.

Best-effort and read-only: nothing here changes behavior, it only reports
numbers so a user can decide whether to raise or lower the performance mode
or switch devices via device.py.
"""
import shutil
import subprocess

import psutil

from . import device as device_module


def _nvidia_smi_utilization() -> float | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            timeout=2,
            stderr=subprocess.DEVNULL,
        )
        return float(out.decode().strip().splitlines()[0])
    except (subprocess.SubprocessError, ValueError, OSError):
        return None


def _gpu_stats() -> dict | None:
    try:
        import torch
    except ImportError:
        return None

    if torch.cuda.is_available():
        total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        used_mb = torch.cuda.memory_reserved(0) / (1024 ** 2)
        return {
            "kind": "cuda",
            "name": torch.cuda.get_device_name(0),
            "memory_used_mb": round(used_mb, 1),
            "memory_total_mb": round(total_mb, 1),
            "memory_percent": round(used_mb / total_mb * 100, 1) if total_mb else None,
            "utilization_percent": _nvidia_smi_utilization(),
        }

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return {
            "kind": "mps",
            "name": "Apple Silicon GPU (MPS)",
            "memory_used_mb": None,
            "memory_total_mb": None,
            "memory_percent": None,
            "utilization_percent": None,  # not exposed by torch's MPS backend
        }

    return None


def get_stats() -> dict:
    vm = psutil.virtual_memory()
    return {
        "cpu": {
            "percent": psutil.cpu_percent(interval=0.1),
            "cores_logical": psutil.cpu_count(logical=True),
            "cores_physical": psutil.cpu_count(logical=False),
        },
        "memory": {
            "percent": vm.percent,
            "used_mb": round(vm.used / (1024 ** 2), 1),
            "total_mb": round(vm.total / (1024 ** 2), 1),
        },
        "gpu": _gpu_stats(),
        "active_device": device_module.get_current_device(),
        "performance_mode": device_module.get_performance_mode(),
    }
