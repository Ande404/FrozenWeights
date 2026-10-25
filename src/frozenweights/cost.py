"""Cost logging: trainable parameters, peak accelerator memory, wall-clock time, checkpoint size.

On CUDA (Colab) peak memory is torch.cuda.max_memory_allocated(), as in the proposal.
On Apple MPS there is no max counter, so a Trainer callback samples
torch.mps.current_allocated_memory() every step and keeps the maximum. MPS numbers are
only for local debugging; reported results come from Colab GPUs.
"""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path

import torch
from transformers import TrainerCallback


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_name() -> str:
    dev = get_device()
    if dev == "cuda":
        return torch.cuda.get_device_name(0)
    if dev == "mps":
        return f"Apple MPS ({platform.processor() or platform.machine()})"
    return f"CPU ({platform.processor() or platform.machine()})"


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_params": total, "trainable_params": trainable, "trainable_pct": 100 * trainable / total}


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def current_memory_bytes() -> int:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated()
    if torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory()
    return 0


class PeakMemoryCallback(TrainerCallback):
    """Tracks peak allocated accelerator memory over a Trainer run."""

    def __init__(self):
        self.peak = 0

    def _sample(self):
        if torch.cuda.is_available():
            self.peak = max(self.peak, torch.cuda.max_memory_allocated())
        else:
            self.peak = max(self.peak, current_memory_bytes())

    def on_step_end(self, args, state, control, **kwargs):
        self._sample()

    def on_train_end(self, args, state, control, **kwargs):
        self._sample()

    @property
    def peak_gb(self) -> float:
        return self.peak / 1024**3


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start


def save_trainable(model, path: str | Path) -> float:
    """Save only the trainable parameters (what each method must store per task); return size in MB.

    For full fine-tuning this is the whole model; for head-only/PEFT it is the small added part.
    """
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {n: p.detach().cpu().contiguous() for n, p in model.named_parameters() if p.requires_grad}
    save_file(state, str(path))
    return path.stat().st_size / 1024**2


def dir_size_mb(path: str | Path) -> float:
    """Total size of the files that have to be stored for this task (model or adapter weights)."""
    total = 0
    for root, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    return total / 1024**2
