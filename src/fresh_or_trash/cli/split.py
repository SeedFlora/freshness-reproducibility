from __future__ import annotations

import argparse
from pathlib import Path

from fresh_or_trash.manifest import load_manifest
from fresh_or_trash.splits import grouped_stratified_split, naive_stratified_split, write_split_manifests


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create deterministic 70/15/15 research partitions")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--setting", choices=["grouped", "naive", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.70, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--stratify-columns", nargs="+", default=["freshness_label", "produce_type"])
    parser.add_argument("--group-columns", nargs="+", default=["duplicate_group_id", "capture_group_id"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = load_manifest(args.manifest)
    if args.setting in {"grouped", "both"}:
        missing_group_columns = [column for column in args.group_columns if column not in frame.columns]
        blank_group_columns = [
            column
            for column in args.group_columns
            if column in frame.columns and frame[column].fillna("").astype(str).str.strip().eq("").any()
        ]
        if missing_group_columns or blank_group_columns:
            raise ValueError(
                "Grouped splitting requires complete duplicate/capture grouping. "
                f"Missing={missing_group_columns}, blank={blank_group_columns}. Run fot-group first."
            )
        grouped = grouped_stratified_split(
            frame,
            ratios=args.ratios,
            stratify_columns=args.stratify_columns,
            group_columns=args.group_columns,
            seed=args.seed,
        )
        write_split_manifests(grouped, args.output_dir, "grouped")
        print(f"grouped leakage-aware split -> {args.output_dir / 'grouped_manifest.csv'}")
    if args.setting in {"naive", "both"}:
        naive = naive_stratified_split(
            frame,
            ratios=args.ratios,
            stratify_columns=args.stratify_columns,
            seed=args.seed,
        )
        write_split_manifests(naive, args.output_dir, "naive")
        print(f"naive image-level ablation split -> {args.output_dir / 'naive_manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
