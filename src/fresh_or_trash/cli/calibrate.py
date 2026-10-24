from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fresh_or_trash.calibration import TemperatureScaler
from fresh_or_trash.data import ManifestDataset, build_transform
from fresh_or_trash.evaluation import predict_loader
from fresh_or_trash.manifest import load_manifest
from fresh_or_trash.models import load_checkpoint
from fresh_or_trash.utils import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit temperature scaling on the locked validation partition")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        help="Derived calibrated checkpoint (default: CHECKPOINT.calibrated.pt); the input is never overwritten",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    model, metadata = load_checkpoint(args.checkpoint, device)
    frame = load_manifest(args.manifest, require_split=True)
    validation = frame[frame["split"] == "validation"]
    dataset = ManifestDataset(
        validation,
        args.image_root,
        task=metadata.get("task", "three_stage"),
        transform=build_transform(int(metadata.get("image_size", 224)), training=False),
        produce_classes=metadata.get("produce_classes"),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    logits, targets, _ = predict_loader(model, loader, device)
    scaler = TemperatureScaler().to(device)
    temperature = scaler.fit(torch.from_numpy(logits).to(device), torch.from_numpy(targets).to(device))
    scaler.save(args.output, checkpoint=str(args.checkpoint))
    output_checkpoint = args.output_checkpoint or args.checkpoint.with_name(f"{args.checkpoint.stem}.calibrated.pt")
    calibrated_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    calibrated_payload["temperature"] = temperature
    calibrated_payload["calibration"] = {
        "method": "temperature_scaling",
        "temperature": temperature,
        "fit_split": "validation",
        "source_checkpoint": str(args.checkpoint),
    }
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(calibrated_payload, output_checkpoint)
    print(f"validation temperature={temperature:.6f} -> {args.output}; checkpoint -> {output_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
