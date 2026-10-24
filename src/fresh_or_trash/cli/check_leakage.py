from __future__ import annotations

import argparse
from pathlib import Path

from fresh_or_trash.manifest import load_manifest
from fresh_or_trash.splits import group_leakage


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fail if duplicate or capture groups cross partitions")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--group-columns", nargs="+", default=["duplicate_group_id", "capture_group_id"])
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = load_manifest(args.manifest, require_split=True)
    leakage = group_leakage(frame, args.group_columns)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        leakage.to_csv(args.report, index=False)
    if not leakage.empty:
        print(leakage.head(20).to_string(index=False))
        raise SystemExit(f"FAILED: {len(leakage)} groups cross partitions")
    print(f"PASS: no leakage across {', '.join(args.group_columns)} ({len(frame)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
