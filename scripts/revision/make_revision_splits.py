"""Build the revision-stage partitions requested in peer review.

Two families are written next to, never over, the frozen paper splits:

* ``staged``: the seed-2026 allocator rerun with progressively stricter group
  definitions (exact SHA-256 -> + near-duplicate hashes -> + embedding
  clusters -> + filename capture links). Level L4 must reproduce the frozen
  grouped manifest exactly.
* ``cv``: five capture-aware folds from the same allocator. Fold ``k`` is the
  test partition, fold ``k+1`` the validation partition, and the remaining
  three folds train, so every capture group is tested exactly once.

Leakage and composition diagnostics are written for every partition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fresh_or_trash.manifest import load_manifest
from fresh_or_trash.splits import assert_no_group_leakage, grouped_stratified_split, group_leakage


SPLIT_SEED = 2026
STRATIFY = ("freshness_label", "produce_type")
DIAGNOSTIC_COLUMNS = ("sha256", "duplicate_group_id", "embedding_cluster_id", "capture_group_id")
STAGED_LEVELS = {
    "L1_exact": ("sha256",),
    "L2_hash": ("duplicate_group_id",),
    "L3_embedding": ("duplicate_group_id", "embedding_cluster_id"),
    "L4_capture": ("duplicate_group_id", "capture_group_id"),
}
FOLD_COUNT = 5


def _base_frame(grouped_manifest: Path) -> pd.DataFrame:
    frame = load_manifest(grouped_manifest)
    return frame.drop(columns=["split"]).reset_index(drop=True)


def _crossings(frame: pd.DataFrame, setting: str) -> list[dict[str, object]]:
    leakage = group_leakage(frame, DIAGNOSTIC_COLUMNS)
    records = []
    for column in DIAGNOSTIC_COLUMNS:
        subset = leakage[leakage["group_column"] == column]
        records.append(
            {
                "setting": setting,
                "group_column": column,
                "group_count": int(frame[column].nunique()),
                "crossing_groups": int(len(subset)),
                "crossing_images": int(subset["sample_count"].sum()) if len(subset) else 0,
            }
        )
    return records


def _composition(frame: pd.DataFrame, setting: str) -> list[dict[str, object]]:
    records = []
    for split, partition in frame.groupby("split", sort=True):
        for (produce, label), cell in partition.groupby(["produce_type", "freshness_label"], sort=True):
            records.append(
                {
                    "setting": setting,
                    "split": split,
                    "produce_type": produce,
                    "freshness_label": label,
                    "images": int(len(cell)),
                    "capture_groups": int(cell["capture_group_id"].nunique()),
                }
            )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grouped-manifest", type=Path, default=Path("data/splits/grouped_manifest.csv"))
    parser.add_argument("--naive-manifest", type=Path, default=Path("data/splits/naive_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/revision_splits"))
    parser.add_argument("--report-dir", type=Path, default=Path("results/revision/splits"))
    args = parser.parse_args()

    base = _base_frame(args.grouped_manifest)
    frozen = load_manifest(args.grouped_manifest)
    naive = load_manifest(args.naive_manifest)
    crossings: list[dict[str, object]] = _crossings(naive, "L0_naive")
    composition: list[dict[str, object]] = _composition(naive, "L0_naive")
    sizes: list[dict[str, object]] = []

    staged_dir = args.output_dir / "staged"
    staged_dir.mkdir(parents=True, exist_ok=True)
    for level, columns in STAGED_LEVELS.items():
        split = grouped_stratified_split(
            base,
            stratify_columns=STRATIFY,
            group_columns=columns,
            seed=SPLIT_SEED,
        )
        assert_no_group_leakage(split, columns)
        split.to_csv(staged_dir / f"{level}_manifest.csv", index=False)
        crossings.extend(_crossings(split, level))
        composition.extend(_composition(split, level))
        for name, partition in split.groupby("split", sort=True):
            sizes.append({"setting": level, "split": name, "images": int(len(partition))})
        if level == "L4_capture":
            merged = frozen[["sample_id", "split"]].merge(
                split[["sample_id", "split"]], on="sample_id", suffixes=("_frozen", "_rebuilt")
            )
            mismatches = int((merged["split_frozen"] != merged["split_rebuilt"]).sum())
            if mismatches or len(merged) != len(frozen):
                raise AssertionError(f"L4 rebuild differs from the frozen grouped split ({mismatches} rows)")

    fold_names = [f"fold-{index}" for index in range(FOLD_COUNT)]
    folds = grouped_stratified_split(
        base,
        ratios=[1.0 / FOLD_COUNT] * FOLD_COUNT,
        split_names=fold_names,
        stratify_columns=STRATIFY,
        group_columns=("duplicate_group_id", "capture_group_id"),
        seed=SPLIT_SEED,
    )
    cv_dir = args.output_dir / "cv"
    cv_dir.mkdir(parents=True, exist_ok=True)
    folds[["sample_id", "split"]].rename(columns={"split": "fold"}).to_csv(cv_dir / "fold_assignment.csv", index=False)
    for index, test_fold in enumerate(fold_names):
        validation_fold = fold_names[(index + 1) % FOLD_COUNT]
        manifest = folds.copy()
        manifest["split"] = "train"
        manifest.loc[folds["split"] == test_fold, "split"] = "test"
        manifest.loc[folds["split"] == validation_fold, "split"] = "validation"
        assert_no_group_leakage(manifest, ("duplicate_group_id", "capture_group_id"))
        setting = f"cv_fold-{index}"
        manifest.to_csv(cv_dir / f"{setting}_manifest.csv", index=False)
        crossings.extend(_crossings(manifest, setting))
        composition.extend(_composition(manifest, setting))
        for name, partition in manifest.groupby("split", sort=True):
            sizes.append({"setting": setting, "split": name, "images": int(len(partition))})

    args.report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(crossings).to_csv(args.report_dir / "group_crossings.csv", index=False)
    pd.DataFrame(composition).to_csv(args.report_dir / "split_composition.csv", index=False)
    pd.DataFrame(sizes).to_csv(args.report_dir / "split_sizes.csv", index=False)
    (args.report_dir / "split_provenance.json").write_text(
        json.dumps(
            {
                "split_seed": SPLIT_SEED,
                "stratify_columns": list(STRATIFY),
                "staged_levels": {key: list(value) for key, value in STAGED_LEVELS.items()},
                "cv_folds": FOLD_COUNT,
                "cv_rule": "test=fold k, validation=fold (k+1) mod 5, train=remaining three folds",
                "source_manifest": str(args.grouped_manifest),
                "l4_reproduces_frozen_grouped_split": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"staged and CV manifests -> {args.output_dir}; diagnostics -> {args.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
