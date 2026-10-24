from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .calibration import TemperatureScaler
from .corruptions import CORRUPTIONS, CorruptionTransform
from .data import ManifestDataset, build_transform
from .metrics import classification_metrics


def _identity_transform(image):
    """Picklable transform used while the expanded dataset selects a condition."""

    return image


class _ExpandedCorruptionDataset(Dataset):
    """Expose every corruption condition through one persistent worker pool.

    The underlying sample order is condition-major, matching the former nested
    evaluator exactly.  A matching condition-aware batch sampler below also
    keeps condition boundaries intact, so inference batch shapes (including a
    final short batch) are unchanged from the previous one-loader-per-condition
    implementation.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: str | Path,
        *,
        task: str,
        image_size: int,
        produce_classes: list[str],
        conditions: Sequence[tuple[str, int]],
        base_transform: Callable,
        seed: int,
    ) -> None:
        self.base = ManifestDataset(
            frame,
            image_root,
            task=task,  # type: ignore[arg-type]
            # Defer the image transform until the condition is known.
            transform=_identity_transform,
            image_size=image_size,
            produce_classes=produce_classes,
        )
        self.conditions = tuple(conditions)
        self.transforms = {
            condition: CorruptionTransform(condition[0], condition[1], base_transform, seed)
            for condition in self.conditions
        }

    def __len__(self) -> int:
        return len(self.base) * len(self.conditions)

    def __getitem__(self, index: int) -> dict[str, object]:
        base_length = len(self.base)
        condition = self.conditions[index // base_length]
        sample = self.base[index % base_length]
        sample["image"] = self.transforms[condition](sample["image"])
        sample["corruption"], sample["severity"] = condition
        return sample


class _ConditionBatchSampler:
    """Yield ordinary sequential batches without crossing a condition boundary."""

    def __init__(self, base_length: int, condition_count: int, batch_size: int) -> None:
        if base_length < 1 or condition_count < 1 or batch_size < 1:
            raise ValueError("base_length, condition_count, and batch_size must be positive")
        self.base_length = base_length
        self.condition_count = condition_count
        self.batch_size = batch_size

    def __iter__(self):
        for condition_index in range(self.condition_count):
            start = condition_index * self.base_length
            stop = start + self.base_length
            for batch_start in range(start, stop, self.batch_size):
                yield list(range(batch_start, min(batch_start + self.batch_size, stop)))

    def __len__(self) -> int:
        return self.condition_count * ((self.base_length + self.batch_size - 1) // self.batch_size)


@torch.inference_mode()
def predict_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str | torch.device,
    temperature: TemperatureScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    model.eval()
    logits_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)["freshness_logits"]
        if temperature is not None:
            temperature = temperature.to(logits.device)
            logits = temperature(logits)
        logits_chunks.append(logits.cpu().numpy())
        targets = batch["target"].numpy()
        target_chunks.append(targets)
        for index, sample_id in enumerate(batch["sample_id"]):
            record: dict[str, object] = {
                "sample_id": sample_id,
                "target_index": int(targets[index]),
                "produce_type": batch["produce_type"][index],
                "source_path": batch["path"][index],
            }
            # The standard manifest dataset does not provide these fields, so
            # normal clean/validation/test prediction artifacts are unchanged.
            for key in ("corruption", "severity"):
                if key in batch:
                    value = batch[key][index]
                    record[key] = value.item() if isinstance(value, torch.Tensor) else value
            records.append(record)
    if not logits_chunks:
        raise ValueError("Evaluation loader is empty")
    return np.concatenate(logits_chunks), np.concatenate(target_chunks), pd.DataFrame.from_records(records)


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str | torch.device,
    *,
    class_names: list[str],
    temperature: TemperatureScaler | None = None,
    ece_bins: int = 15,
) -> tuple[dict[str, object], pd.DataFrame, np.ndarray, np.ndarray]:
    logits, targets, predictions = predict_loader(model, loader, device, temperature)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    metrics = classification_metrics(targets, probabilities=probabilities, class_names=class_names, ece_bins=ece_bins)
    predictions["predicted_index"] = probabilities.argmax(axis=1)
    predictions["predicted_label"] = [class_names[index] for index in predictions["predicted_index"]]
    predictions["confidence"] = probabilities.max(axis=1)
    for index, class_name in enumerate(class_names):
        predictions[f"probability_{class_name.lower().replace('-', '_').replace(' ', '_')}"] = probabilities[:, index]
    return metrics, predictions, logits, targets


def evaluate_corruptions(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    image_root: str | Path,
    device: str | torch.device,
    *,
    task: str,
    class_names: list[str],
    produce_classes: list[str],
    image_size: int = 224,
    batch_size: int = 24,
    num_workers: int = 4,
    corruption_names: Iterable[str] = CORRUPTIONS,
    severities: Iterable[int] = (1, 2, 3, 4, 5),
    temperature: TemperatureScaler | None = None,
    ece_bins: int = 15,
    seed: int = 42,
) -> pd.DataFrame:
    conditions = [(str(corruption), int(severity)) for corruption in corruption_names for severity in severities]
    if not conditions:
        raise ValueError("At least one corruption/severity condition is required")
    clean_transform = build_transform(image_size, training=False)
    dataset = _ExpandedCorruptionDataset(
        frame,
        image_root,
        task=task,
        image_size=image_size,
        produce_classes=produce_classes,
        conditions=conditions,
        base_transform=clean_transform,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=_ConditionBatchSampler(len(dataset.base), len(conditions), batch_size),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    logits, targets, predictions = predict_loader(model, loader, device, temperature)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predictions["corruption"] = predictions["corruption"].astype(str)
    predictions["severity"] = pd.to_numeric(predictions["severity"], errors="raise").astype(int)

    records: list[dict[str, object]] = []
    for corruption, severity in conditions:
        mask = predictions["corruption"].eq(corruption) & predictions["severity"].eq(severity)
        condition_metrics = classification_metrics(
            targets[mask.to_numpy()],
            probabilities=probabilities[mask.to_numpy()],
            class_names=class_names,
            ece_bins=ece_bins,
        )
        records.append(
            {
                "corruption": corruption,
                "severity": severity,
                **{key: value for key, value in condition_metrics.items() if isinstance(value, (int, float))},
            }
        )
    return pd.DataFrame.from_records(records)
