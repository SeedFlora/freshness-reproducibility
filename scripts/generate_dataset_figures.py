"""Generate dataset-audit figures directly from manifests and acquired images."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


AUGMENTED_PREFIX = re.compile(r"^aug_\d+_", re.IGNORECASE)
STAGE_COLORS = {"Fresh": "#2d8a57", "Semi-Fresh": "#e3a72f", "Rotten": "#b7463a"}


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def find_augmentation_pair(exclusions: pd.DataFrame, acquisition_root: Path) -> tuple[Path, Path]:
    augmented = exclusions[exclusions["filename"].astype(str).str.match(AUGMENTED_PREFIX)]
    for row in augmented.sort_values("source_path").itertuples(index=False):
        augmented_path = acquisition_root / str(row.source_path)
        original_name = AUGMENTED_PREFIX.sub("", str(row.filename))
        original_path = augmented_path.parent / original_name
        if augmented_path.is_file() and original_path.is_file():
            return original_path, augmented_path
    raise RuntimeError("No deterministic augmentation/original pair was found")


def find_conflict_pair(conflicts: pd.DataFrame, acquisition_root: Path) -> tuple[pd.Series, pd.Series]:
    for _, group in conflicts.groupby("conflict_group_id", sort=True):
        if len(group) >= 2:
            rows = group.sort_values(["freshness_label", "source_path"]).iloc[:2]
            if all((acquisition_root / path).is_file() for path in rows["source_path"].astype(str)):
                return rows.iloc[0], rows.iloc[1]
    raise RuntimeError("No exact cross-label conflict pair was found")


def find_same_label_duplicate(manifest: pd.DataFrame, acquisition_root: Path) -> tuple[pd.Series, pd.Series]:
    sizes = manifest.groupby("sha256").size()
    for sha256 in sorted(sizes[sizes > 1].index):
        group = manifest[manifest["sha256"] == sha256].sort_values("source_path")
        if group["freshness_label"].nunique() == 1 and len(group) >= 2:
            rows = group.iloc[:2]
            if all((acquisition_root / path).is_file() for path in rows["source_path"].astype(str)):
                return rows.iloc[0], rows.iloc[1]
    raise RuntimeError("No same-label exact duplicate pair was found")


def duplicate_figure(
    manifest: pd.DataFrame,
    exclusions: pd.DataFrame,
    conflicts: pd.DataFrame,
    acquisition_root: Path,
    output: Path,
) -> None:
    original_path, augmented_path = find_augmentation_pair(exclusions, acquisition_root)
    conflict_left, conflict_right = find_conflict_pair(conflicts, acquisition_root)
    duplicate_left, duplicate_right = find_same_label_duplicate(manifest, acquisition_root)

    examples = [
        (
            original_path,
            augmented_path,
            "Repository-generated variant",
            f"Original: {original_path.name}",
            f"Excluded: {augmented_path.name}",
        ),
        (
            acquisition_root / str(conflict_left["source_path"]),
            acquisition_root / str(conflict_right["source_path"]),
            "Exact bytes, conflicting folder labels",
            f"{conflict_left['freshness_label']}: {conflict_left['filename']}",
            f"{conflict_right['freshness_label']}: {conflict_right['filename']}",
        ),
        (
            acquisition_root / str(duplicate_left["source_path"]),
            acquisition_root / str(duplicate_right["source_path"]),
            "Same-label exact duplicate retained as one group",
            str(duplicate_left["filename"]),
            str(duplicate_right["filename"]),
        ),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(7.1, 8.7), constrained_layout=True)
    for row_index, (left, right, relation, left_caption, right_caption) in enumerate(examples):
        for column_index, (path, caption) in enumerate(((left, left_caption), (right, right_caption))):
            axes[row_index, column_index].imshow(load_rgb(path))
            axes[row_index, column_index].set_title(caption, fontsize=8)
            axes[row_index, column_index].axis("off")
        axes[row_index, 0].text(
            0.0,
            1.10,
            relation,
            transform=axes[row_index, 0].transAxes,
            fontsize=10,
            fontweight="bold",
        )
    fig.suptitle("Examples identified during the duplicate and label audit", fontsize=12, fontweight="bold")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def overview_figure(
    manifest: pd.DataFrame,
    exclusions: pd.DataFrame,
    conflicts: pd.DataFrame,
    output: Path,
) -> None:
    counts = (
        manifest.groupby(["produce_type", "freshness_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Fresh", "Semi-Fresh", "Rotten"])
    )
    extracted = len(manifest) + len(exclusions) + len(conflicts)
    original_candidates = len(manifest) + len(conflicts)
    clean = len(manifest)

    fig, (axis_flow, axis_counts) = plt.subplots(1, 2, figsize=(10.0, 4.2), gridspec_kw={"width_ratios": [0.85, 1.6]})
    axis_flow.axis("off")
    flow = [
        ("Extracted", extracted, "24 folders"),
        ("Remove pre-generated variants", original_candidates, f"-{len(exclusions):,}"),
        ("Remove unresolved label conflicts", clean, f"-{len(conflicts):,}"),
        ("Clean benchmark candidates", clean, "group before split"),
    ]
    y_values = [0.88, 0.62, 0.36, 0.10]
    for index, ((label, value, note), y) in enumerate(zip(flow, y_values)):
        axis_flow.text(
            0.5,
            y,
            f"{label}\n{value:,} images\n{note}",
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f4f6f7", "edgecolor": "#46535e"},
        )
        if index < len(flow) - 1:
            axis_flow.annotate("", xy=(0.5, y_values[index + 1] + 0.09), xytext=(0.5, y - 0.09), arrowprops={"arrowstyle": "->"})
    axis_flow.set_title("Archive-to-benchmark audit", fontsize=11, fontweight="bold")

    bottom = pd.Series(0, index=counts.index, dtype=float)
    for stage in counts.columns:
        axis_counts.barh(
            counts.index,
            counts[stage],
            left=bottom,
            color=STAGE_COLORS[stage],
            label=stage,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += counts[stage]
    axis_counts.set_xlabel("Clean original images")
    axis_counts.set_ylabel("Produce type")
    axis_counts.set_yticks(range(len(counts.index)), [value.replace("_", " ") for value in counts.index])
    axis_counts.set_title("Clean candidates by produce and stage", fontsize=11, fontweight="bold")
    axis_counts.grid(axis="x", alpha=0.2)
    axis_counts.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/primary_benchmark_manifest.csv"))
    parser.add_argument("--exclusions", type=Path, default=Path("data/primary_exclusions.csv"))
    parser.add_argument("--conflicts", type=Path, default=Path("data/primary_label_conflicts.csv"))
    parser.add_argument("--acquisition-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.manifest)
    exclusions = pd.read_csv(args.exclusions)
    conflicts = pd.read_csv(args.conflicts)
    duplicate_figure(manifest, exclusions, conflicts, args.acquisition_root, args.output_dir / "duplicate_examples.png")
    overview_figure(manifest, exclusions, conflicts, args.output_dir / "dataset_overview.png")
    print(f"wrote {args.output_dir / 'duplicate_examples.png'}")
    print(f"wrote {args.output_dir / 'dataset_overview.png'}")


if __name__ == "__main__":
    main()
