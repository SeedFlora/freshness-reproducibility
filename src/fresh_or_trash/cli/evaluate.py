from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fresh_or_trash.calibration import TemperatureScaler
from fresh_or_trash.corruptions import CORRUPTIONS
from fresh_or_trash.data import ManifestDataset, build_transform
from fresh_or_trash.evaluation import evaluate_corruptions, evaluate_loader
from fresh_or_trash.manifest import load_manifest
from fresh_or_trash.models import load_checkpoint
from fresh_or_trash.utils import resolve_device, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate clean classification, calibration, and optional corruptions")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--temperature", type=Path, help="Temperature JSON; embedded checkpoint temperature is used otherwise")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--corruptions", action="store_true")
    parser.add_argument("--corruption-names", nargs="+", choices=CORRUPTIONS, default=list(CORRUPTIONS))
    parser.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    model, metadata = load_checkpoint(args.checkpoint, device)
    frame = load_manifest(args.manifest, require_split=True)
    partition = frame[frame["split"] == args.split]
    if partition.empty:
        raise ValueError(f"No rows in split {args.split!r}")
    task = metadata.get("task", "three_stage")
    class_names = metadata.get("freshness_classes", ["Fresh", "Semi-Fresh", "Rotten"])
    produce_classes = metadata.get("produce_classes", sorted(frame["produce_type"].astype(str).unique()))
    image_size = int(metadata.get("image_size", 224))
    temperature = None
    if args.temperature:
        temperature = TemperatureScaler.load(args.temperature)
    elif metadata.get("temperature") is not None:
        temperature = TemperatureScaler(float(metadata["temperature"]))
    dataset = ManifestDataset(
        partition,
        args.image_root,
        task=task,
        transform=build_transform(image_size, training=False),
        image_size=image_size,
        produce_classes=produce_classes,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    metrics, predictions, _, _ = evaluate_loader(
        model,
        loader,
        device,
        class_names=class_names,
        temperature=temperature,
        ece_bins=args.ece_bins,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, args.output_dir / f"{args.split}_metrics.json")
    predictions.to_csv(args.output_dir / f"{args.split}_predictions.csv", index=False)
    if args.corruptions:
        if args.split != "test":
            raise ValueError("Corruption benchmarking is reserved for the locked test split")
        corruption_metrics = evaluate_corruptions(
            model,
            partition,
            args.image_root,
            device,
            task=task,
            class_names=class_names,
            produce_classes=produce_classes,
            image_size=image_size,
            batch_size=args.batch_size,
            num_workers=args.workers,
            corruption_names=args.corruption_names,
            severities=args.severities,
            temperature=temperature,
            ece_bins=args.ece_bins,
        )
        corruption_metrics.to_csv(args.output_dir / "corruption_metrics.csv", index=False)
    print(f"{args.split} accuracy={metrics['accuracy']:.6f}, macro-F1={metrics['macro_f1']:.6f} -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
