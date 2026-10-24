from __future__ import annotations

import argparse
import re
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from fresh_or_trash.data import ManifestDataset, build_transform
from fresh_or_trash.evaluation import evaluate_loader
from fresh_or_trash.experiments import EXTERNAL_PRODUCE_TYPES, flatten_scalar_metrics
from fresh_or_trash.manifest import load_manifest
from fresh_or_trash.metrics import classification_metrics
from fresh_or_trash.models import MODEL_NAMES, load_checkpoint
from fresh_or_trash.utils import resolve_device, save_json


_AUGMENTED_NAME = re.compile(r"^aug[_-]?\d*", re.IGNORECASE)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a validation-selected binary checkpoint on original-only external banana/orange images"
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--allowed-produce", nargs="+", default=list(EXTERNAL_PRODUCE_TYPES))
    parser.add_argument("--original-path-component", default="Original Image")
    return parser


def _normalized_parts(value: object) -> tuple[str, ...]:
    return tuple(part.casefold() for part in PurePosixPath(str(value).replace("\\", "/")).parts)


def _validate_external_manifest(
    frame: pd.DataFrame,
    *,
    allowed_produce: list[str],
    original_path_component: str,
) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("External manifest is empty")
    allowed = {value.strip().casefold() for value in allowed_produce}
    observed_produce = set(frame["produce_type"].astype(str).str.casefold())
    if observed_produce != allowed:
        raise ValueError(f"External produce types must be exactly {sorted(allowed)}; got {sorted(observed_produce)}")
    labels = set(frame["freshness_label"].astype(str))
    if labels != {"Fresh", "Rotten"}:
        raise ValueError(f"External labels must be exactly Fresh/Rotten; got {sorted(labels)}")
    if "split" in frame and set(frame["split"].astype(str)) != {"external_test"}:
        raise ValueError("Every external sample must be marked split=external_test")
    required_component = original_path_component.casefold()
    bad_paths = [
        value
        for value in frame["source_path"].astype(str)
        if required_component not in _normalized_parts(value)
    ]
    if bad_paths:
        raise ValueError(f"External source paths are not original-only; first mismatch: {bad_paths[0]}")
    filenames = frame.get("filename", frame["source_path"].map(lambda value: Path(str(value)).name)).astype(str)
    augmented = filenames.str.match(_AUGMENTED_NAME)
    if augmented.any():
        raise ValueError(f"Pre-generated external augmentation detected: {filenames[augmented].iloc[0]}")
    return frame.reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    expected_outputs = (
        args.output_dir / "external_metrics.json",
        args.output_dir / "external_predictions.csv",
        args.output_dir / "external_per_produce_metrics.csv",
    )
    existing = [path for path in expected_outputs if path.exists()]
    if existing:
        raise FileExistsError(f"External evaluation refuses to overwrite existing artifact: {existing[0]}")

    device = resolve_device(args.device)
    model, metadata = load_checkpoint(args.checkpoint, device)
    if metadata.get("architecture") not in MODEL_NAMES:
        raise ValueError("Checkpoint architecture is outside the controlled five-model registry")
    if metadata.get("task") != "binary":
        raise ValueError("External validation requires a grouped binary checkpoint")
    class_names = list(metadata.get("freshness_classes", []))
    if class_names != ["Fresh", "Rotten"]:
        raise ValueError(f"Binary checkpoint class order must be ['Fresh', 'Rotten']; got {class_names}")

    frame = _validate_external_manifest(
        load_manifest(args.manifest),
        allowed_produce=args.allowed_produce,
        original_path_component=args.original_path_component,
    )
    produce_classes = list(metadata.get("produce_classes", []))
    unknown_produce = sorted(set(frame["produce_type"].astype(str)) - set(produce_classes))
    if unknown_produce:
        raise ValueError(f"External manifest has produce types absent from training metadata: {unknown_produce}")
    dataset = ManifestDataset(
        frame,
        args.image_root,
        task="binary",
        transform=build_transform(int(metadata.get("image_size", 224)), training=False),
        image_size=int(metadata.get("image_size", 224)),
        produce_classes=produce_classes,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    metrics, predictions, _, targets = evaluate_loader(
        model,
        loader,
        device,
        class_names=class_names,
        ece_bins=args.ece_bins,
    )
    probability_columns = ["probability_fresh", "probability_rotten"]
    probabilities = predictions[probability_columns].to_numpy(dtype=float)
    per_produce_records: list[dict[str, object]] = []
    for produce_type in sorted(frame["produce_type"].astype(str).unique()):
        mask = predictions["produce_type"].astype(str).eq(produce_type).to_numpy()
        produce_metrics = classification_metrics(
            np.asarray(targets)[mask],
            probabilities=probabilities[mask],
            class_names=class_names,
            ece_bins=args.ece_bins,
        )
        per_produce_records.append({"produce_type": produce_type, **flatten_scalar_metrics(produce_metrics)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope": "cross_dataset_external_validation",
        "fine_tuning": False,
        "original_images_only": True,
        "architecture": metadata["architecture"],
        "checkpoint": str(args.checkpoint),
        "task": "binary",
        "produce_types": sorted(frame["produce_type"].astype(str).unique()),
        "metrics": metrics,
    }
    save_json(payload, expected_outputs[0])
    predictions.assign(evaluation_scope="external_original_only").to_csv(expected_outputs[1], index=False)
    pd.DataFrame.from_records(per_produce_records).to_csv(expected_outputs[2], index=False)
    print(
        f"external original-only accuracy={metrics['accuracy']:.6f}, "
        f"macro-F1={metrics['macro_f1']:.6f} -> {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
