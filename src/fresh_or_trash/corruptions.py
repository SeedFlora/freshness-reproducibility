from __future__ import annotations

import io
import hashlib
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


CORRUPTIONS = ("gaussian_noise", "gaussian_blur", "brightness", "contrast", "occlusion", "jpeg")


def apply_corruption(image: Image.Image, name: str, severity: int, seed: int = 42) -> Image.Image:
    if name not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption {name!r}; choose from {CORRUPTIONS}")
    if severity not in {1, 2, 3, 4, 5}:
        raise ValueError("severity must be in 1..5")
    source = image.convert("RGB")
    content_digest = hashlib.blake2b(source.tobytes(), digest_size=8).digest()
    sample_seed = seed ^ int.from_bytes(content_digest, byteorder="big", signed=False)
    rng = np.random.default_rng(sample_seed)
    if name == "gaussian_noise":
        array = np.asarray(source, dtype=np.float32) / 255.0
        noise = rng.normal(0.0, 0.02 * severity, array.shape)
        return Image.fromarray(np.uint8(np.clip(array + noise, 0.0, 1.0) * 255.0), mode="RGB")
    if name == "gaussian_blur":
        return source.filter(ImageFilter.GaussianBlur(radius=0.5 * severity))
    if name == "brightness":
        factors = (0.85, 0.70, 0.55, 0.40, 0.25)
        return ImageEnhance.Brightness(source).enhance(factors[severity - 1])
    if name == "contrast":
        factors = (0.85, 0.70, 0.55, 0.40, 0.25)
        return ImageEnhance.Contrast(source).enhance(factors[severity - 1])
    if name == "occlusion":
        array = np.asarray(source).copy()
        height, width = array.shape[:2]
        fractions = (0.10, 0.15, 0.20, 0.25, 0.30)
        patch_height = max(1, int(round(height * fractions[severity - 1])))
        patch_width = max(1, int(round(width * fractions[severity - 1])))
        top = int(rng.integers(0, max(1, height - patch_height + 1)))
        left = int(rng.integers(0, max(1, width - patch_width + 1)))
        array[top : top + patch_height, left : left + patch_width] = 0
        return Image.fromarray(array, mode="RGB")
    qualities = (70, 50, 35, 20, 10)
    buffer = io.BytesIO()
    source.save(buffer, format="JPEG", quality=qualities[severity - 1])
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB").copy()


@dataclass(frozen=True)
class CorruptionTransform:
    name: str
    severity: int
    base_transform: Callable
    seed: int = 42

    def __call__(self, image: Image.Image):
        return self.base_transform(apply_corruption(image, self.name, self.severity, self.seed))
