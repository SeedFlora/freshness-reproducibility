from __future__ import annotations

import io
import statistics
import time

import numpy as np
import torch


def model_size_bytes(model: torch.nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.tell()


@torch.inference_mode()
def profile_model(
    model: torch.nn.Module,
    device: str | torch.device,
    *,
    image_size: int = 224,
    batch_size: int = 1,
    warmup: int = 20,
    iterations: int = 100,
    amp: bool = True,
) -> dict[str, float | int | str]:
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be >= 0 and iterations must be >= 1")
    device = torch.device(device)
    model = model.to(device).eval()
    inputs = torch.randn(batch_size, 3, image_size, image_size, device=device)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for _ in range(warmup):
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            model(inputs)
    synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        synchronize()
        start = time.perf_counter()
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            model(inputs)
        synchronize()
        samples.append((time.perf_counter() - start) * 1000.0 / batch_size)
    state_bytes = model_size_bytes(model)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "measurement_scope": "model_forward_pass_only",
        "includes_image_decoding": False,
        "includes_preprocessing": False,
        "device": str(device),
        "image_size": image_size,
        "batch_size": batch_size,
        "amp_enabled": bool(amp and device.type == "cuda"),
        "warmup_iterations": warmup,
        "timed_iterations": iterations,
        "latency_mean_ms_per_image": statistics.fmean(samples),
        "latency_median_ms_per_image": statistics.median(samples),
        "latency_p95_ms_per_image": float(np.percentile(samples, 95)),
        "parameter_count": parameters,
        "trainable_parameter_count": trainable,
        "state_dict_bytes": state_bytes,
        "state_dict_megabytes": state_bytes / (1024**2),
    }
