from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .manifest import FRESHNESS_CLASSES, resolve_image_path

ImageFile.LOAD_TRUNCATED_IMAGES = False
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(image_size: int = 224, training: bool = False) -> Callable:
    if training:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR),
                transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15, hue=0.03),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    resize = int(round(image_size / 0.875))
    return transforms.Compose(
        [
            transforms.Resize(resize, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class ManifestDataset(Dataset):
    """Image dataset backed by a normalized manifest."""

    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: str | Path,
        *,
        task: Literal["three_stage", "binary"] = "three_stage",
        transform: Callable | None = None,
        image_size: int = 224,
        produce_classes: list[str] | None = None,
    ) -> None:
        if task not in {"three_stage", "binary"}:
            raise ValueError("task must be 'three_stage' or 'binary'")
        frame = frame.copy().reset_index(drop=True)
        if task == "binary":
            # Semi-Fresh is excluded, never folded into either endpoint class.
            frame = frame[frame["freshness_label"].isin(["Fresh", "Rotten"])].reset_index(drop=True)
            freshness_classes = ["Fresh", "Rotten"]
        else:
            freshness_classes = list(FRESHNESS_CLASSES)
        if frame.empty:
            raise ValueError(f"No samples remain for task={task}")
        self.frame = frame
        self.image_root = Path(image_root)
        self.task = task
        self.transform = transform or build_transform(image_size=image_size, training=False)
        self.freshness_classes = freshness_classes
        self.freshness_to_index = {label: index for index, label in enumerate(freshness_classes)}
        self.produce_classes = produce_classes or sorted(frame["produce_type"].astype(str).unique())
        self.produce_to_index = {label: index for index, label in enumerate(self.produce_classes)}

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.frame.iloc[index]
        path = resolve_image_path(row, self.image_root)
        with Image.open(path) as source:
            image = source.convert("RGB")
        return {
            "image": self.transform(image),
            "target": torch.tensor(self.freshness_to_index[row["freshness_label"]], dtype=torch.long),
            "produce_target": torch.tensor(self.produce_to_index[str(row["produce_type"])], dtype=torch.long),
            "sample_id": str(row["sample_id"]),
            "produce_type": str(row["produce_type"]),
            "path": str(path),
        }
