"""Evaluate the disclosed corruption suite and keep per-image predictions.

This mirrors ``fresh_or_trash.evaluation.evaluate_corruptions`` (same condition
order, corruption seed 42, deterministic transforms) but also writes every
corrupted prediction so the revision can report seed variation and cluster
bootstrap intervals instead of one aggregate per condition.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from fresh_or_trash.calibration import TemperatureScaler
from fresh_or_trash.corruptions import CORRUPTIONS
from fresh_or_trash.data import build_transform
from fresh_or_trash.evaluation import _ConditionBatchSampler, _ExpandedCorruptionDataset, predict_loader
from fresh_or_trash.manifest import load_manifest
from fresh_or_trash.metrics import classification_metrics
from fresh_or_trash.models import load_checkpoint
from fresh_or_trash.utils import resolve_device


CORRUPTION_SEED = 42


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, metadata = load_checkpoint(args.checkpoint, device)
    class_names = list(metadata.get("freshness_classes", ["Fresh", "Semi-Fresh", "Rotten"]))
    temperature = TemperatureScaler(float(metadata["temperature"])) if metadata.get("temperature") is not None else None
    frame = load_manifest(args.manifest, require_split=True)
    partition = frame[frame["split"] == "test"]
    image_size = int(metadata.get("image_size", 224))
    conditions = [(name, int(severity)) for name in CORRUPTIONS for severity in args.severities]
    dataset = _ExpandedCorruptionDataset(
        partition,
        args.image_root,
        task=str(metadata.get("task", "three_stage")),
        image_size=image_size,
        produce_classes=list(metadata.get("produce_classes", sorted(frame["produce_type"].astype(str).unique()))),
        conditions=conditions,
        base_transform=build_transform(image_size, training=False),
        seed=CORRUPTION_SEED,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=_ConditionBatchSampler(len(dataset.base), len(conditions), args.batch_size),
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    logits, targets, predictions = predict_loader(model, loader, device, temperature)
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    predictions["corruption"] = predictions["corruption"].astype(str)
    predictions["severity"] = pd.to_numeric(predictions["severity"], errors="raise").astype(int)
    predictions["predicted_index"] = probabilities.argmax(axis=1)
    for index, name in enumerate(class_names):
        predictions[f"probability_{name.lower().replace('-', '_')}"] = probabilities[:, index]

    records = []
    for corruption, severity in conditions:
        mask = (predictions["corruption"].eq(corruption) & predictions["severity"].eq(severity)).to_numpy()
        metrics = classification_metrics(targets[mask], probabilities=probabilities[mask], class_names=class_names)
        records.append(
            {"corruption": corruption, "severity": severity}
            | {key: value for key, value in metrics.items() if isinstance(value, (int, float))}
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.drop(columns=["source_path"]).to_csv(args.output_dir / "corruption_predictions.csv", index=False)
    pd.DataFrame.from_records(records).to_csv(args.output_dir / "corruption_metrics.csv", index=False)
    print(f"{len(conditions)} conditions x {len(partition)} images -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
