from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from .models import FreshnessClassifier


class GradCAM:
    """Grad-CAM for a benchmark model's final convolutional block."""

    def __init__(self, model: FreshnessClassifier) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.handle = model.gradcam_target_layer().register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output: torch.Tensor) -> None:
        self.activations = output
        if output.requires_grad:
            output.retain_grad()

    def close(self) -> None:
        self.handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def __call__(self, images: torch.Tensor, class_indices: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(images)
        logits = outputs["freshness_logits"]
        if class_indices is None:
            class_indices = logits.argmax(dim=1)
        scores = logits.gather(1, class_indices[:, None]).sum()
        scores.backward()
        if self.activations is None or self.activations.grad is None:
            raise RuntimeError("Grad-CAM target layer did not produce gradients")
        activations = self.activations
        gradients = activations.grad
        weights = gradients.mean(dim=(-2, -1), keepdim=True)
        heatmap = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        heatmap = functional.interpolate(heatmap, size=images.shape[-2:], mode="bilinear", align_corners=False)
        flat = heatmap.flatten(1)
        minima = flat.min(dim=1).values[:, None, None, None]
        maxima = flat.max(dim=1).values[:, None, None, None]
        heatmap = (heatmap - minima) / (maxima - minima).clamp_min(1e-8)
        return heatmap.detach(), logits.detach()


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    source = image.convert("RGB")
    values = np.asarray(heatmap, dtype=np.float32).squeeze()
    values = np.clip(values, 0.0, 1.0)
    red = np.uint8(values * 255)
    blue = np.uint8((1.0 - values) * 100)
    green = np.uint8(np.maximum(0.0, 1.0 - np.abs(values - 0.5) * 2.0) * 180)
    color = np.stack([red, green, blue], axis=-1)
    colored = Image.fromarray(color, mode="RGB").resize(source.size, Image.Resampling.BILINEAR)
    return Image.blend(source, colored, alpha)
