from __future__ import annotations

import time
from typing import Dict, Optional, Sequence

import torch


PROFILING_BUFFER_NAMES = ("total_ops", "total_params")


def _remove_profiling_buffers(model) -> None:
    for module in model.modules():
        buffer_store = getattr(module, "_buffers", None)
        if isinstance(buffer_store, dict):
            for key in PROFILING_BUFFER_NAMES:
                buffer_store.pop(key, None)
        non_persistent = getattr(module, "_non_persistent_buffers_set", None)
        if hasattr(non_persistent, "discard"):
            for key in PROFILING_BUFFER_NAMES:
                non_persistent.discard(key)


def count_parameters(model, *, trainable_only: bool = False) -> int:
    parameters = model.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def maybe_compute_flops(model, input_shape: Sequence[int], device: torch.device) -> Optional[int]:
    try:
        from thop import profile
    except ImportError:
        return None

    dummy = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
    _remove_profiling_buffers(model)
    return int(flops)


def benchmark_inference(
    model,
    *,
    input_shape: Sequence[int],
    device: torch.device,
    warmup_steps: int = 10,
    measure_steps: int = 30,
) -> Dict[str, Optional[float]]:
    dummy = torch.randn(*input_shape, device=device)
    model.eval()

    with torch.no_grad():
        for _ in range(max(0, warmup_steps)):
            model(dummy)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        for _ in range(max(1, measure_steps)):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start_time

    mean_latency = elapsed / max(1, measure_steps)
    fps = 1.0 / mean_latency if mean_latency > 0 else None
    return {
        "inference_time_seconds": mean_latency,
        "fps": fps,
    }
