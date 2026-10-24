from __future__ import annotations

import argparse
from pathlib import Path

from fresh_or_trash.models import load_checkpoint
from fresh_or_trash.profiling import profile_model
from fresh_or_trash.utils import resolve_device, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure local model size and inference latency")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    model, metadata = load_checkpoint(args.checkpoint, device)
    result = profile_model(
        model,
        device,
        image_size=int(metadata.get("image_size", 224)),
        batch_size=args.batch_size,
        warmup=args.warmup,
        iterations=args.iterations,
        amp=args.amp,
    )
    checkpoint_bytes = args.checkpoint.stat().st_size
    result.update(
        {
            "architecture": metadata["architecture"],
            "checkpoint": str(args.checkpoint),
            "checkpoint_file_bytes": checkpoint_bytes,
            "checkpoint_file_megabytes": checkpoint_bytes / (1024**2),
        }
    )
    save_json(result, args.output)
    print(f"median latency={result['latency_median_ms_per_image']:.3f} ms, size={result['state_dict_megabytes']:.2f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
