# GPU memory counters. torch counters only see tensors of the PyTorch allocator but can be reset
# per stage; nvml reports the memory of the whole process as seen by the driver (CUDA context and
# TensorFlow included), which is the only meaningful number for OmniGlue.
# Every function returns None on CPU, so callers do not need to check for a GPU.
import os

import torch

_MB = 1024.0 * 1024.0
_nvml = None
_nvml_tried = False


def sync():
    # CUDA calls are asynchronous: synchronize before reading a timer or a memory counter.
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def allocated_mb():
    if not torch.cuda.is_available():
        return None
    sync()
    return torch.cuda.memory_allocated() / _MB


def reserved_mb():
    if not torch.cuda.is_available():
        return None
    sync()
    return torch.cuda.memory_reserved() / _MB


def reset_peak():
    """Start a new measurement window for peak_mb() and peak_reserved_mb()."""
    if torch.cuda.is_available():
        sync()
        torch.cuda.reset_peak_memory_stats()


def peak_mb():
    if not torch.cuda.is_available():
        return None
    sync()
    return torch.cuda.max_memory_allocated() / _MB


def peak_reserved_mb():
    if not torch.cuda.is_available():
        return None
    sync()
    return torch.cuda.max_memory_reserved() / _MB


def _nvml_handle():
    global _nvml, _nvml_tried
    if _nvml_tried:
        return _nvml
    _nvml_tried = True
    if not torch.cuda.is_available():
        return None
    try:
        import pynvml
        pynvml.nvmlInit()
        # Index 0 assumes a single visible GPU, as requested by the SLURM job (--gres=gpu:1).
        _nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
    except Exception:  # noqa: BLE001  pynvml missing or driver not queryable
        _nvml = None
    return _nvml


def nvml_process_mb():
    """GPU memory used by this process according to the driver, or None if unavailable."""
    h = _nvml_handle()
    if h is None:
        return None
    pynvml, handle = h
    try:
        sync()
        pid = os.getpid()
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            if p.pid == pid and p.usedGpuMemory is not None:
                return p.usedGpuMemory / _MB
    except Exception:  # noqa: BLE001
        pass
    return None


def nvml_device_mb():
    """Memory used on the whole GPU. On a shared node it includes other jobs: diagnostic only."""
    h = _nvml_handle()
    if h is None:
        return None
    pynvml, handle = h
    try:
        sync()
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used / _MB
    except Exception:  # noqa: BLE001
        return None


def delta_mb(before, after):
    if before is None or after is None:
        return None
    return after - before
