from __future__ import annotations

import argparse
from pathlib import Path

from fresh_or_trash.cli.common import add_config_argument, config_value, read_config
from fresh_or_trash.manifest import load_manifest
from fresh_or_trash.models import MODEL_NAMES
from fresh_or_trash.training import options_from_config, train_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one controlled freshness benchmark model")
    add_config_argument(parser)
    parser.add_argument("manifest", type=Path, help="Split manifest containing train/validation/test")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--task", choices=["three_stage", "binary"], default=None)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--auxiliary-produce-head", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--evaluate-test", action=argparse.BooleanOptionalAction, default=True)
    return parser


def run(args: argparse.Namespace) -> dict:
    config = read_config(args.config)
    frame = load_manifest(args.manifest, require_split=True)
    task = args.task or config_value(config, "training.task", "three_stage")
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    pretrained = args.pretrained if args.pretrained is not None else bool(config_value(config, "model.pretrained", True))
    auxiliary = (
        args.auxiliary_produce_head
        if args.auxiliary_produce_head is not None
        else bool(config_value(config, "model.auxiliary_produce_head", False))
    )
    output_dir = args.output_dir or Path(config.get("output_dir", "results")) / "grouped" / task / args.model / f"seed-{seed}"
    return train_model(
        frame,
        args.image_root,
        architecture=args.model,
        output_dir=output_dir,
        task=task,
        image_size=int(config_value(config, "data.image_size", 224)),
        batch_size=int(config_value(config, "data.batch_size", 24)),
        num_workers=int(config_value(config, "data.num_workers", 4)),
        pin_memory=bool(config_value(config, "data.pin_memory", True)),
        pretrained=pretrained,
        auxiliary_produce_head=auxiliary,
        options=options_from_config(config),
        seed=seed,
        requested_device=args.device or str(config.get("device", "auto")),
        evaluate_test=args.evaluate_test,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(f"best validation macro-F1={result['validation_macro_f1']:.6f}; checkpoint={result['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
