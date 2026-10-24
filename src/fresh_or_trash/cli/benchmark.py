from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fresh_or_trash.cli.common import add_config_argument, config_value, read_config
from fresh_or_trash.manifest import load_manifest
from fresh_or_trash.models import MODEL_NAMES
from fresh_or_trash.training import options_from_config, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the controlled five-model benchmark or the two-model naive ablation")
    add_config_argument(parser)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--setting", choices=["grouped", "naive"], default="grouped")
    parser.add_argument("--task", choices=["three_stage", "binary"], default="three_stage")
    parser.add_argument("--best-grouped-model", choices=MODEL_NAMES, help="Validation-selected grouped model; required for naive setting")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--no-pretrained", action="store_true", help="Smoke testing only")
    parser.add_argument("--no-test", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = read_config(args.config)
    configured = tuple(config_value(config, "model.names", MODEL_NAMES))
    if configured != MODEL_NAMES:
        raise ValueError(f"Controlled grouped benchmark must contain exactly {MODEL_NAMES}; config has {configured}")
    if args.setting == "naive" and not args.best_grouped_model:
        raise ValueError("--best-grouped-model is required for the naive split ablation")
    architectures = list(MODEL_NAMES) if args.setting == "grouped" else list(dict.fromkeys(["mobilenet_v2", args.best_grouped_model]))
    frame = load_manifest(args.manifest, require_split=True)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    root = args.output_dir or Path(config.get("output_dir", "results")) / args.setting / args.task
    rows: list[dict[str, object]] = []
    for architecture in architectures:
        print(f"training {architecture} ({args.setting}, {args.task})")
        run = train_model(
            frame,
            args.image_root,
            architecture=architecture,
            output_dir=root / architecture / f"seed-{seed}",
            task=args.task,
            image_size=int(config_value(config, "data.image_size", 224)),
            batch_size=int(config_value(config, "data.batch_size", 24)),
            num_workers=int(config_value(config, "data.num_workers", 4)),
            pin_memory=bool(config_value(config, "data.pin_memory", True)),
            pretrained=not args.no_pretrained and bool(config_value(config, "model.pretrained", True)),
            auxiliary_produce_head=bool(config_value(config, "model.auxiliary_produce_head", False)),
            options=options_from_config(config),
            seed=seed,
            requested_device=args.device or str(config.get("device", "auto")),
            evaluate_test=not args.no_test,
        )
        row: dict[str, object] = {
            "architecture": architecture,
            "setting": args.setting,
            "task": args.task,
            "seed": seed,
            "validation_macro_f1": run["validation_macro_f1"],
            "best_epoch": run["best_epoch"],
            "checkpoint": run["checkpoint"],
        }
        if run["test_metrics"]:
            row.update({f"test_{key}": value for key, value in run["test_metrics"].items() if isinstance(value, (int, float))})
        rows.append(row)
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows).to_csv(root / "benchmark_summary.csv", index=False)
    print(f"benchmark summary -> {root / 'benchmark_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
