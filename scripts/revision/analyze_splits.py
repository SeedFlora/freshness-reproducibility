"""Split-design and label-validity analyses for the peer-review revision.

Topic ``splits`` covers the staged grouping ablation (L0 naive to L4
capture-aware), the leakage mechanism inside the non-grouped splits, split
composition with composition-standardised macro-F1, grouped five-fold
cross-validation, and capture-group sizes. Topic ``labels`` covers exact and
near-duplicate label conflicts, capture groups spanning stages, folder
day-range boundaries, model-consensus difficulty on the grouped test, and the
most-disputed test images for expert adjudication. Parts whose queue outputs
do not exist yet are recorded as pending; rerunning the script fills them in.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

from fresh_or_trash.manifest import resolve_image_path
from revision_common import (
    ANALYSIS,
    BINARY,
    BOOTSTRAP_REPLICATES,
    COLORS,
    DISPLAY_NAMES,
    DOUBLE_COLUMN_WIDTH,
    FROZEN,
    GROUPED_MANIFEST,
    MODEL_ORDER,
    NAIVE_MANIFEST,
    PAPER_FIGURES,
    PAPER_TABLES,
    REVISION,
    ROOT,
    SEEDS,
    SINGLE_COLUMN_WIDTH,
    THREE_STAGE,
    class_names,
    cluster_bootstrap,
    figure_style,
    frozen_test_predictions,
    interval,
    latex_table,
    macro_f1,
    mean_sd,
    percentile_interval,
    probability_columns,
    read_predictions,
    write_json,
)


STAGED_DIR = ROOT / "data/revision_splits/staged"
CV_DIR = ROOT / "data/revision_splits/cv"
SPLIT_REPORTS = REVISION / "splits"
ORIGINAL_MANIFEST = ROOT / "data/primary_original_manifest.csv"
LABEL_CONFLICTS = ROOT / "data/primary_label_conflicts.csv"
NEAR_DUPLICATES = ROOT / "results/near_duplicate_report.csv"
DEFAULT_IMAGE_ROOT = ROOT.parent / "data/raw"

LEVELS = ("L0_naive", "L1_exact", "L2_hash", "L3_embedding", "L4_capture")
RETRAINED_LEVELS = ("L1_exact", "L2_hash", "L3_embedding")
LEVEL_LABELS = {
    "L0_naive": "L0 naive",
    "L1_exact": "L1 SHA-256",
    "L2_hash": "L2 +hash",
    "L3_embedding": "L3 +embed.",
    "L4_capture": "L4 +capture",
}
LEVEL_TICKS = {
    "L0_naive": "L0\nnaive",
    "L1_exact": "L1\nSHA-256",
    "L2_hash": "L2\n+hash",
    "L3_embedding": "L3\n+embed.",
    "L4_capture": "L4\n+capture",
}
ABLATION_MODELS = ("mobilenet_v2", "convnext_tiny")
TASKS = ("three_stage", "binary")
TASK_LABELS = {"three_stage": "Three-stage", "binary": "Binary"}
SPLITS = ("train", "validation", "test")
FOLDS = tuple(range(5))
CV_SEED = 2026
RUN_FILES = ("test_predictions.csv", "test_metrics.json", "run.json")
MANIFEST_COLUMNS = [
    "sample_id",
    "source_path",
    "produce_type",
    "freshness_label",
    "shelf_life_label",
    "sha256",
    "duplicate_group_id",
    "embedding_cluster_id",
    "capture_group_id",
    "split",
]
CROSSING_COLUMNS = {
    "capture": "capture_group_id",
    "duplicate": "duplicate_group_id",
    "embedding": "embedding_cluster_id",
    "sha256": "sha256",
}
LINK_SUBSETS = ("all", "capture_linked", "unlinked", "duplicate_linked", "capture_only_linked")
STAGE_SHORT = {"Fresh": "F", "Semi-Fresh": "SF", "Rotten": "R"}
DISPUTE_THRESHOLD = 0.8
SHEET_SHAPE = (4, 6)
THUMBNAIL_PIXELS = 240
DAY_RANGE = re.compile(r"(\d+)\s*-\s*(\d+)")


@dataclass
class Context:
    runs: Path
    splits_dir: Path
    labels_dir: Path
    tables: Path
    figures: Path
    image_root: Path
    replicates: int
    written: list[Path] = field(default_factory=list)

    def csv(self, frame: pd.DataFrame, directory: Path, name: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        frame.to_csv(directory / name, index=False)
        self.written.append(directory / name)

    def json(self, payload: object, directory: Path, name: str) -> None:
        write_json(payload, directory / name)
        self.written.append(directory / name)

    def table(self, name: str, **kwargs: object) -> None:
        path = self.tables / f"{name}.tex"
        latex_table(path, label="tab:" + name.replace("_", "-"), **kwargs)
        self.written.append(path)

    def figure(self, figure: object, name: str) -> None:
        self.figures.mkdir(parents=True, exist_ok=True)
        for suffix in (".png", ".pdf"):
            figure.savefig(self.figures / f"{name}{suffix}")
            self.written.append(self.figures / f"{name}{suffix}")


class Status:
    """Record every analysis part as ``done`` or ``pending: <missing input>``."""

    def __init__(self) -> None:
        self.parts: dict[str, str] = {}

    def set(self, part: str, missing: Sequence[str] = (), note: str = "") -> None:
        if not missing:
            self.parts[part] = "done"
            return
        listed = ", ".join(missing[:3]) + (f" and {len(missing) - 3} more" if len(missing) > 3 else "")
        self.parts[part] = f"pending: {len(missing)} missing input(s): {listed}" + (f"; {note}" if note else "")
        print(f"WARNING {part}: {self.parts[part]}", flush=True)


def limit_threads() -> None:
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(4)


def pyplot():
    figure_style()
    import matplotlib.pyplot as plt

    return plt


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def ordered(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Sort by level, task, architecture, and seed in their canonical orders."""
    keys = {
        "level": {name: index for index, name in enumerate(LEVELS)},
        "setting": {name: index for index, name in enumerate(LEVELS)},
        "task": {name: index for index, name in enumerate(TASKS)},
        "architecture": {name: index for index, name in enumerate(MODEL_ORDER)},
        "subset": {name: index for index, name in enumerate(LINK_SUBSETS)},
    }
    return frame.sort_values(
        list(columns), key=lambda column: column.map(keys[column.name]) if column.name in keys else column, kind="stable"
    ).reset_index(drop=True)


def read_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=MANIFEST_COLUMNS, dtype=str, keep_default_na=False)
    return frame.sort_values("sample_id", kind="stable").reset_index(drop=True)


def load_level_manifests() -> dict[str, pd.DataFrame]:
    manifests = {"L0_naive": read_manifest(NAIVE_MANIFEST)}
    for level in RETRAINED_LEVELS:
        manifests[level] = read_manifest(STAGED_DIR / f"{level}_manifest.csv")
    grouped = read_manifest(GROUPED_MANIFEST)
    rebuilt = read_manifest(STAGED_DIR / "L4_capture_manifest.csv")
    if not grouped[["sample_id", "split"]].equals(rebuilt[["sample_id", "split"]]):
        raise ValueError("L4_capture manifest does not reproduce the frozen grouped split")
    manifests["L4_capture"] = grouped
    reference = grouped.drop(columns="split")
    for level, manifest in manifests.items():
        if not manifest.drop(columns="split").equals(reference):
            raise ValueError(f"{level} manifest disagrees with the grouped manifest outside the split column")
    return manifests


def load_fold_manifests() -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    assignment = pd.read_csv(CV_DIR / "fold_assignment.csv", dtype=str)
    manifests = {fold: read_manifest(CV_DIR / f"cv_fold-{fold}_manifest.csv") for fold in FOLDS}
    for fold, manifest in manifests.items():
        tested = set(assignment.loc[assignment["fold"] == f"fold-{fold}", "sample_id"])
        if set(manifest.loc[manifest["split"] == "test", "sample_id"]) != tested:
            raise ValueError(f"cv_fold-{fold} test partition differs from fold_assignment.csv")
    return manifests, assignment


def expected_test(manifest: pd.DataFrame, task: str) -> pd.DataFrame:
    test = manifest[manifest["split"] == "test"]
    if task == "binary":
        test = test[test["freshness_label"].isin(BINARY)]
    return test.reset_index(drop=True)


def targets_of(frame: pd.DataFrame, task: str) -> np.ndarray:
    lookup = {name: index for index, name in enumerate(class_names(task))}
    return frame["freshness_label"].map(lookup).to_numpy(dtype=int)


def aligned_probabilities(frame: pd.DataFrame, expected: pd.DataFrame, task: str, source: str) -> np.ndarray:
    """Probabilities in the row order of ``expected`` after checking samples and targets."""
    indexed = frame.set_index(frame["sample_id"].astype(str))
    if len(indexed) != len(expected) or set(indexed.index) != set(expected["sample_id"]):
        raise ValueError(f"{source}: predicted samples differ from the manifest test partition")
    indexed = indexed.loc[expected["sample_id"].to_numpy()]
    if not np.array_equal(indexed["target_index"].to_numpy(dtype=int), targets_of(expected, task)):
        raise ValueError(f"{source}: targets differ from the manifest labels")
    probabilities = indexed[probability_columns(task)].to_numpy(dtype=float)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def frozen_predictions(task: str, arch: str, seed: int, setting: str, calibration: str) -> pd.DataFrame:
    path = FROZEN / "evaluation" / setting / task / arch / f"seed-{seed}" / calibration / "test_predictions.csv"
    return read_predictions(path, task)


def run_predictions(run_dir: Path, task: str) -> tuple[pd.DataFrame, float] | None:
    """Test predictions of a finished revision run (``run.json`` is written last)."""
    if not all((run_dir / name).is_file() for name in RUN_FILES):
        return None
    reported = json.loads((run_dir / "test_metrics.json").read_text(encoding="utf-8"))["macro_f1"]
    return read_predictions(run_dir / "test_predictions.csv", task), float(reported)


def present_macro_f1(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Macro-F1 over the classes present in ``targets``; NaN for an empty subset."""
    if len(targets) == 0:
        return float("nan")
    labels = np.unique(targets).tolist()
    return float(f1_score(targets, predictions, labels=labels, average="macro", zero_division=0))


def fast_macro_f1(targets: np.ndarray, predictions: np.ndarray, class_count: int) -> float:
    """Confusion-matrix macro-F1, identical to ``macro_f1`` and fast enough for the bootstrap."""
    confusion = np.bincount(targets * class_count + predictions, minlength=class_count * class_count)
    confusion = confusion.reshape(class_count, class_count)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    scores = np.divide(2.0 * np.diag(confusion), denominator, out=np.zeros(class_count), where=denominator > 0)
    return float(scores.mean())


def accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean(targets == predictions)) if len(targets) else float("nan")


def summarise(frame: pd.DataFrame, keys: list[str], metrics: Sequence[str], count: str = "seeds") -> pd.DataFrame:
    records = []
    for key, group in frame.groupby(keys, sort=False):
        record = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        record[count] = len(group)
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        records.append(record)
    return pd.DataFrame(records)


def signed(value: float) -> str:
    return rf"${value:+.3f}$"


def stage_combination(labels: Sequence[str]) -> str:
    present = set(labels)
    return "+".join(stage for stage in THREE_STAGE if stage in present)


def stage_pair(first: str, second: str) -> str:
    if first == second:
        return "same stage"
    low, high = sorted((first, second), key=THREE_STAGE.index)
    return f"{low}/{high}"


def produce_name(value: str) -> str:
    return value.replace("_", " ")


def multicolumn(width: int, text: str) -> str:
    return rf"\multicolumn{{{width}}}{{@{{}}l}}{{{text}}}"


def collect_staged(
    context: Context, manifests: dict[str, pd.DataFrame]
) -> tuple[dict[tuple[str, str, int], np.ndarray], list[str], float]:
    """Argmax test predictions of both ablation models at every grouping level."""
    predictions: dict[tuple[str, str, int], np.ndarray] = {}
    missing: list[str] = []
    deviation = 0.0
    for level in LEVELS:
        expected = expected_test(manifests[level], "three_stage")
        targets = targets_of(expected, "three_stage")
        for arch in ABLATION_MODELS:
            for seed in SEEDS:
                reported = None
                if level in RETRAINED_LEVELS:
                    run_dir = context.runs / "staged" / level / arch / f"seed-{seed}"
                    loaded = run_predictions(run_dir, "three_stage")
                    if loaded is None:
                        missing.append(relative(run_dir))
                        continue
                    frame, reported = loaded
                else:
                    setting = "naive" if level == "L0_naive" else "grouped"
                    frame = frozen_predictions("three_stage", arch, seed, setting, "uncalibrated")
                labels = aligned_probabilities(frame, expected, "three_stage", f"{level}/{arch}/seed-{seed}").argmax(axis=1)
                if reported is not None:
                    deviation = max(deviation, abs(macro_f1(targets, labels, len(THREE_STAGE)) - reported))
                predictions[(level, arch, seed)] = labels
    return predictions, missing, deviation


def crossings(manifest: pd.DataFrame, column: str) -> tuple[int, int]:
    spans = manifest.groupby(column)["split"].nunique()
    crossing = spans.index[spans > 1]
    return int(len(crossing)), int(manifest[column].isin(crossing).sum())


def linkage_flags(manifest: pd.DataFrame) -> pd.DataFrame:
    """Flag test images whose capture or duplicate group also has training members."""
    train = manifest[manifest["split"] == "train"]
    test = expected_test(manifest, "three_stage").copy()
    test["capture_linked"] = test["capture_group_id"].isin(set(train["capture_group_id"]))
    test["duplicate_linked"] = test["duplicate_group_id"].isin(set(train["duplicate_group_id"]))
    group_sizes = manifest["capture_group_id"].value_counts()
    train_sizes = train["capture_group_id"].value_counts()
    test["train_share_of_capture_group"] = (
        test["capture_group_id"].map(train_sizes).fillna(0).to_numpy(dtype=float)
        / test["capture_group_id"].map(group_sizes).to_numpy(dtype=float)
    )
    if (test["duplicate_linked"] & ~test["capture_linked"]).any():
        raise ValueError("A duplicate-linked test image is not capture-linked; duplicate groups must nest in capture groups")
    return test


def link_masks(flags: pd.DataFrame) -> dict[str, np.ndarray]:
    capture = flags["capture_linked"].to_numpy(dtype=bool)
    duplicate = flags["duplicate_linked"].to_numpy(dtype=bool)
    masks = {
        "all": np.ones(len(flags), dtype=bool),
        "capture_linked": capture,
        "unlinked": ~capture,
        "duplicate_linked": duplicate,
        "capture_only_linked": capture & ~duplicate,
    }
    return {name: masks[name] for name in LINK_SUBSETS}


def level_diagnostics(manifests: dict[str, pd.DataFrame]) -> pd.DataFrame:
    reported = pd.read_csv(SPLIT_REPORTS / "group_crossings.csv")
    records = []
    for level in LEVELS:
        manifest = manifests[level]
        record: dict[str, object] = {"level": level, "label": LEVEL_LABELS[level]}
        for name, column in CROSSING_COLUMNS.items():
            row = reported[(reported["setting"] == level) & (reported["group_column"] == column)]
            if len(row) != 1:
                raise ValueError(f"group_crossings.csv lacks {level}/{column}")
            groups, images = int(row["crossing_groups"].iloc[0]), int(row["crossing_images"].iloc[0])
            if (groups, images) != crossings(manifest, column):
                raise ValueError(f"group_crossings.csv disagrees with the {level} manifest for {column}")
            record[f"crossing_{name}_groups"] = groups
            record[f"crossing_{name}_images"] = images
        test = expected_test(manifest, "three_stage")
        flags = linkage_flags(manifest)
        record["test_images"] = len(test)
        record["test_capture_groups"] = int(test["capture_group_id"].nunique())
        for stage in THREE_STAGE:
            record[f"test_{stage}_share"] = float((test["freshness_label"] == stage).mean())
        record["test_capture_linked"] = int(flags["capture_linked"].sum())
        record["test_duplicate_linked"] = int(flags["duplicate_linked"].sum())
        records.append(record)
    return pd.DataFrame(records)


def staged_ablation(
    context: Context,
    status: Status,
    manifests: dict[str, pd.DataFrame],
    staged: dict[tuple[str, str, int], np.ndarray],
    missing: list[str],
) -> None:
    diagnostics = level_diagnostics(manifests)
    records = []
    for (level, arch, seed), labels in staged.items():
        targets = targets_of(expected_test(manifests[level], "three_stage"), "three_stage")
        records.append(
            {
                "level": level,
                "architecture": arch,
                "seed": seed,
                "test_images": len(targets),
                "macro_f1": macro_f1(targets, labels, len(THREE_STAGE)),
                "accuracy": accuracy(targets, labels),
            }
        )
    seeds = ordered(pd.DataFrame(records), ["level", "architecture", "seed"])
    summary = summarise(seeds, ["level", "architecture"], ["macro_f1", "accuracy"])
    context.csv(diagnostics, context.splits_dir, "staged_split_diagnostics.csv")
    context.csv(seeds, context.splits_dir, "staged_macro_f1_seeds.csv")
    context.csv(summary, context.splits_dir, "staged_macro_f1_summary.csv")

    def score(level: str, arch: str) -> str:
        values = seeds.loc[(seeds["level"] == level) & (seeds["architecture"] == arch), "macro_f1"]
        return mean_sd(values) if len(values) == len(SEEDS) else "--"

    rows = []
    for record in diagnostics.to_dict(orient="records"):
        rows.append(
            [
                record["label"],
                f"{record['crossing_capture_groups']} ({record['crossing_capture_images']:,})",
                f"{record['crossing_duplicate_groups']} ({record['crossing_duplicate_images']:,})",
                f"{record['test_images']} ({record['test_capture_linked']})",
                f"{record['test_Fresh_share']:.3f}",
                *[score(record["level"], arch) for arch in ABLATION_MODELS],
            ]
        )
    notes = (
        "Partitions rebuilt with split seed 2026: L1 groups exact SHA-256 copies; L2 adds pHash/dHash near-duplicates; "
        "L3 adds embedding clusters; L4 adds filename capture links and equals the frozen grouped split; "
        "L0 is the image-level naive split. Cap./Dup. = capture/duplicate groups spanning more than one partition "
        "(images in them); linked = test images sharing a capture group with a training image."
    )
    if missing:
        notes += " -- = retraining runs pending."
    context.table(
        "rev_staged",
        caption=(
            "Staged grouping ablation. Three-stage test macro-F1 is the mean $\\pm$ sample SD over seeds 2026, 3407, "
            "and 777 (uncalibrated softmax; temperature scaling leaves the argmax unchanged)."
        ),
        headers=["Level", "Cap.", "Dup.", "Test (linked)", "Fresh", "MobileNetV2", "ConvNeXt-T"],
        rows=rows,
        notes=notes,
    )
    plot_staged(context, seeds, diagnostics)
    status.set("staged_ablation", missing)
    status.set("staged_ablation_figure", missing)


def plot_staged(context: Context, seeds: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    plt = pyplot()
    from matplotlib.lines import Line2D
    from matplotlib.transforms import blended_transform_factory

    figure, axis = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, 2.45))
    offsets = {"mobilenet_v2": -0.16, "convnext_tiny": 0.16}
    jitter = dict(zip(SEEDS, (-0.05, 0.0, 0.05)))
    for arch in ABLATION_MODELS:
        means: dict[int, float] = {}
        for position, level in enumerate(LEVELS):
            rows = seeds[(seeds["level"] == level) & (seeds["architecture"] == arch)]
            if rows.empty:
                continue
            centre = position + offsets[arch]
            axis.scatter(
                centre + rows["seed"].map(jitter).to_numpy(),
                rows["macro_f1"],
                s=11,
                facecolors="none",
                edgecolors=COLORS[arch],
                linewidths=0.8,
                zorder=3,
            )
            means[position] = float(rows["macro_f1"].mean())
            axis.hlines(means[position], centre - 0.11, centre + 0.11, colors=COLORS[arch], linewidth=1.8, zorder=4)
        for position in means:
            if position + 1 in means:
                axis.plot(
                    [position + offsets[arch], position + 1 + offsets[arch]],
                    [means[position], means[position + 1]],
                    color=COLORS[arch], linewidth=0.8, linestyle=":", zorder=2,
                )
    blend = blended_transform_factory(axis.transData, axis.transAxes)
    for position, level in enumerate(LEVELS):
        if seeds[seeds["level"] == level].empty:
            axis.text(position, 0.5, "pending", transform=blend, ha="center", va="center", fontsize=7, color="0.45", style="italic")
    axis.set_xlim(-0.5, len(LEVELS) - 0.5)
    axis.set_xticks(range(len(LEVELS)), [LEVEL_TICKS[level] for level in LEVELS])
    axis.set_ylabel("Three-stage test macro-F1")
    top = axis.secondary_xaxis("top")
    top.set_xticks(range(len(LEVELS)), [str(value) for value in diagnostics["crossing_capture_groups"]])
    top.set_xlabel("Capture groups crossing partitions (of 285)", fontsize=7)
    top.tick_params(length=0, labelsize=7)
    handles = [
        Line2D([], [], color=COLORS[arch], linestyle=":", linewidth=0.8, label=DISPLAY_NAMES[arch]) for arch in ABLATION_MODELS
    ]
    handles += [
        Line2D([], [], color="0.35", marker="o", markerfacecolor="none", markersize=3.5, linestyle="none", label="Seed"),
        Line2D([], [], color="0.35", linewidth=1.8, label="Mean of seeds"),
    ]
    axis.legend(handles=handles, loc="lower left", ncol=2, frameon=False, handlelength=1.6, columnspacing=1.0)
    context.figure(figure, "staged_ablation")
    plt.close(figure)


def leakage_mechanism(
    context: Context,
    status: Status,
    manifests: dict[str, pd.DataFrame],
    staged: dict[tuple[str, str, int], np.ndarray],
    missing: list[str],
) -> None:
    counts, records, flag_frames = [], [], []
    for level in LEVELS:
        flags = linkage_flags(manifests[level])
        masks = link_masks(flags)
        targets = targets_of(flags, "three_stage")
        flag_frames.append(
            flags.assign(level=level)[
                ["level", "sample_id", "produce_type", "freshness_label", "capture_group_id", "duplicate_group_id",
                 "capture_linked", "duplicate_linked", "train_share_of_capture_group"]
            ]
        )
        for subset, mask in masks.items():
            labels = flags.loc[mask, "freshness_label"].value_counts()
            counts.append(
                {
                    "level": level,
                    "subset": subset,
                    "images": int(mask.sum()),
                    "capture_groups": int(flags.loc[mask, "capture_group_id"].nunique()),
                    **{stage: int(labels.get(stage, 0)) for stage in THREE_STAGE},
                }
            )
        for arch in ABLATION_MODELS:
            for seed in SEEDS:
                predictions = staged.get((level, arch, seed))
                if predictions is None:
                    continue
                for subset, mask in masks.items():
                    records.append(
                        {
                            "level": level,
                            "architecture": arch,
                            "seed": seed,
                            "subset": subset,
                            "images": int(mask.sum()),
                            "macro_f1": present_macro_f1(targets[mask], predictions[mask]),
                            "accuracy": accuracy(targets[mask], predictions[mask]),
                        }
                    )
    seeds = ordered(pd.DataFrame(records), ["level", "architecture", "subset", "seed"])
    summary = summarise(seeds, ["level", "architecture", "subset", "images"], ["macro_f1", "accuracy"])
    context.csv(pd.DataFrame(counts), context.splits_dir, "leakage_linkage_counts.csv")
    context.csv(pd.concat(flag_frames, ignore_index=True), context.splits_dir, "leakage_linkage_flags.csv")
    context.csv(seeds, context.splits_dir, "leakage_linkage_seeds.csv")
    context.csv(summary, context.splits_dir, "leakage_linkage_summary.csv")
    status.set("leakage_mechanism", missing, "outputs written for the available levels")


def composition_frame(settings: dict[str, pd.DataFrame]) -> pd.DataFrame:
    produce_types = sorted(next(iter(settings.values()))["produce_type"].unique())
    records = []
    for setting, manifest in settings.items():
        for split in SPLITS:
            part = manifest[manifest["split"] == split]
            labels = part["freshness_label"].value_counts()
            produce = part["produce_type"].value_counts().reindex(produce_types, fill_value=0)
            record: dict[str, object] = {
                "setting": setting,
                "split": split,
                "images": len(part),
                "capture_groups": int(part["capture_group_id"].nunique()),
                "duplicate_groups": int(part["duplicate_group_id"].nunique()),
            }
            for stage in THREE_STAGE:
                record[stage] = int(labels.get(stage, 0))
            for stage in THREE_STAGE:
                record[f"{stage}_share"] = record[stage] / len(part)
            for name in produce_types:
                record[f"share_{name}"] = produce[name] / len(part)
            record["max_produce"] = produce.idxmax()
            record["max_produce_share"] = produce.max() / len(part)
            records.append(record)
    return pd.DataFrame(records)


def check_composition(settings: dict[str, pd.DataFrame]) -> None:
    reported = pd.read_csv(SPLIT_REPORTS / "split_composition.csv")
    keys = ["split", "produce_type", "freshness_label"]
    for setting, manifest in settings.items():
        expected = reported[reported["setting"] == setting].set_index(keys)["images"].sort_index()
        observed = manifest.groupby(keys).size().rename("images").sort_index()
        if not observed.equals(expected):
            raise ValueError(f"split_composition.csv disagrees with the {setting} manifest")


def split_composition(
    context: Context, status: Status, manifests: dict[str, pd.DataFrame], fold_manifests: dict[int, pd.DataFrame]
) -> None:
    settings = {level: manifests[level] for level in LEVELS}
    settings |= {f"cv_fold-{fold}": fold_manifests[fold] for fold in FOLDS}
    check_composition(settings)
    frame = composition_frame(settings)
    context.csv(frame, context.splits_dir, "split_composition.csv")

    def row(label: str, setting: str, split: str) -> list[str]:
        record = frame[(frame["setting"] == setting) & (frame["split"] == split)].iloc[0]
        return [
            label,
            f"{record['images']:,}",
            f"{record['Fresh']:,}",
            f"{record['Semi-Fresh']:,}",
            f"{record['Rotten']:,}",
            f"{record['capture_groups']}",
            f"{record['max_produce_share']:.3f}",
        ]

    width = 7
    rows = [[multicolumn(width, r"\textit{Grouped split (frozen; L4)}")]]
    rows += [row(split.capitalize(), "L4_capture", split) for split in SPLITS]
    rows += [[r"\midrule " + multicolumn(width, r"\textit{Naive image-level split (L0)}")]]
    rows += [row(split.capitalize(), "L0_naive", split) for split in SPLITS]
    rows += [[r"\midrule " + multicolumn(width, r"\textit{Grouped 5-fold CV, test folds}")]]
    rows += [row(f"Fold {fold}", f"cv_fold-{fold}", "test") for fold in FOLDS]
    total = manifests["L4_capture"]
    context.table(
        "rev_split_composition",
        caption="Class distribution per partition (images).",
        headers=["Partition", "Images", "Fresh", "Semi-Fresh", "Rotten", "Groups", "Max prod."],
        rows=rows,
        notes=(
            f"All {len(total):,} benchmark images: "
            + ", ".join(f"{stage} {int((total['freshness_label'] == stage).sum()):,}" for stage in THREE_STAGE)
            + f"; {total['capture_group_id'].nunique()} capture groups. Groups = capture groups with at least one image in "
            "the partition (naive partitions share groups). Max prod. = share of the most frequent produce type. "
            "In CV fold k, validation is test fold k+1 (mod 5) and training is the remaining three folds."
        ),
    )
    status.set("split_composition")


def standardisation_weights(test: pd.DataFrame, reference: pd.DataFrame) -> tuple[np.ndarray, float, int]:
    """Weights making the test produce x stage cell shares equal those of ``reference``."""
    keys = ["produce_type", "freshness_label"]
    reference_share = reference.groupby(keys).size() / len(reference)
    test_share = test.groupby(keys).size() / len(test)
    cells = pd.MultiIndex.from_frame(test[keys])
    weights = reference_share.reindex(cells).to_numpy() / test_share.reindex(cells).to_numpy()
    if np.isnan(weights).any():
        raise ValueError("A test cell is absent from the reference distribution")
    coverage = float(reference_share.reindex(test_share.index).sum())
    missing_cells = int((~reference_share.index.isin(test_share.index)).sum())
    return weights, coverage, missing_cells


def standardised_macro_f1(
    context: Context,
    status: Status,
    manifests: dict[str, pd.DataFrame],
    staged: dict[tuple[str, str, int], np.ndarray],
    missing: list[str],
) -> None:
    grouped = manifests["L4_capture"]
    records = []

    def record(setting: str, task: str, arch: str, seed: int, test: pd.DataFrame, reference: pd.DataFrame, labels: np.ndarray) -> None:
        weights, coverage, missing_cells = standardisation_weights(test, reference)
        targets = targets_of(test, task)
        count = len(class_names(task))
        raw, weighted = macro_f1(targets, labels, count), macro_f1(targets, labels, count, weights=weights)
        records.append(
            {
                "setting": setting,
                "task": task,
                "architecture": arch,
                "seed": seed,
                "test_images": len(test),
                "macro_f1": raw,
                "standardised_macro_f1": weighted,
                "difference": weighted - raw,
                "reference_coverage": coverage,
                "missing_cells": missing_cells,
                "min_weight": float(weights.min()),
                "max_weight": float(weights.max()),
            }
        )

    for level in LEVELS[:-1]:
        test = expected_test(manifests[level], "three_stage")
        for arch in ABLATION_MODELS:
            for seed in SEEDS:
                if (level, arch, seed) in staged:
                    record(level, "three_stage", arch, seed, test, grouped, staged[(level, arch, seed)])
    for task in TASKS:
        test = expected_test(grouped, task)
        reference = grouped if task == "three_stage" else grouped[grouped["freshness_label"].isin(BINARY)]
        for arch in MODEL_ORDER:
            for seed in SEEDS:
                frame = frozen_test_predictions(task, arch, seed)
                labels = aligned_probabilities(frame, test, task, f"grouped {task}/{arch}/seed-{seed}").argmax(axis=1)
                record("L4_capture", task, arch, seed, test, reference, labels)
    seeds = ordered(pd.DataFrame(records), ["task", "setting", "architecture", "seed"])
    summary = summarise(seeds, ["task", "setting", "architecture"], ["macro_f1", "standardised_macro_f1", "difference"])
    context.csv(seeds, context.splits_dir, "standardised_macro_f1_seeds.csv")
    context.csv(summary, context.splits_dir, "standardised_macro_f1_summary.csv")

    rows: list[list[str]] = []
    for task in TASKS:
        heading = rf"\textit{{{TASK_LABELS[task]} grouped test ({len(expected_test(grouped, task))} images)}}"
        rows.append([(r"\midrule " if rows else "") + multicolumn(4, heading)])
        for arch in MODEL_ORDER:
            subset = seeds[(seeds["task"] == task) & (seeds["setting"] == "L4_capture") & (seeds["architecture"] == arch)]
            rows.append(
                [
                    DISPLAY_NAMES[arch],
                    mean_sd(subset["macro_f1"]),
                    mean_sd(subset["standardised_macro_f1"]),
                    signed(float(subset["difference"].mean())),
                ]
            )
    context.table(
        "rev_standardised",
        caption=(
            "Composition-standardised test macro-F1: test images reweighted so that produce $\\times$ stage cell shares "
            "equal those of all 5,183 benchmark images (Fresh and Rotten images only for the binary endpoint). "
            "Mean $\\pm$ sample SD over seeds 2026, 3407, and 777."
        ),
        headers=["Model", "Raw", "Standardised", r"$\Delta$"],
        rows=rows,
        notes="Weights equal dataset cell share divided by test cell share. Per-level values (L0--L3) are in the CSV output.",
    )
    status.set("standardised_macro_f1", missing, "outputs written for the available levels")


def cv_task(
    context: Context, task: str, grouped: pd.DataFrame, fold_manifests: dict[int, pd.DataFrame]
) -> tuple[dict[str, object] | None, list[str]]:
    count = len(class_names(task))
    expected = {fold: expected_test(fold_manifests[fold], task) for fold in FOLDS}
    labels: dict[tuple[int, str], np.ndarray] = {}
    records, missing = [], []
    deviation = 0.0
    for fold in FOLDS:
        targets = targets_of(expected[fold], task)
        for arch in MODEL_ORDER:
            run_dir = context.runs / "cv" / task / f"fold-{fold}" / arch / f"seed-{CV_SEED}"
            loaded = run_predictions(run_dir, task)
            if loaded is None:
                missing.append(relative(run_dir))
                continue
            frame, reported = loaded
            predicted = aligned_probabilities(frame, expected[fold], task, f"cv {task} fold-{fold}/{arch}").argmax(axis=1)
            score = macro_f1(targets, predicted, count)
            deviation = max(deviation, abs(score - reported))
            labels[(fold, arch)] = predicted
            records.append(
                {
                    "task": task,
                    "fold": fold,
                    "architecture": arch,
                    "test_images": len(targets),
                    "capture_groups": int(expected[fold]["capture_group_id"].nunique()),
                    "macro_f1": score,
                    "accuracy": accuracy(targets, predicted),
                }
            )
    if not records:
        return None, missing
    folds = ordered(pd.DataFrame(records), ["fold", "architecture"])
    complete_folds = [fold for fold in FOLDS if all((fold, arch) in labels for arch in MODEL_ORDER)]
    folds["rank"] = np.nan
    folds["ranked_first"] = False
    for fold in complete_folds:
        mask = folds["fold"] == fold
        scores = folds.loc[mask, "macro_f1"].to_numpy()
        folds.loc[mask, "rank"] = stats.rankdata(-scores, method="average")
        folds.loc[mask, "ranked_first"] = scores == scores.max()

    summary = summarise(folds, ["architecture"], ["macro_f1"], count="folds")
    summary = ordered(summary.rename(columns={"macro_f1_mean": "fold_mean_macro_f1", "macro_f1_sd": "fold_sd_macro_f1"}), ["architecture"])
    ranked = folds[folds["fold"].isin(complete_folds)].groupby("architecture")
    summary["mean_rank"] = summary["architecture"].map(ranked["rank"].mean())
    summary["folds_ranked_first"] = summary["architecture"].map(ranked["ranked_first"].sum()).fillna(0).astype(int)
    summary["complete_folds"] = len(complete_folds)

    complete_models = [arch for arch in MODEL_ORDER if all((fold, arch) in labels for fold in FOLDS)]
    pooled_records: list[dict[str, object]] = []
    leader = None
    group_count = 0
    if complete_models:
        order = pd.concat([expected[fold] for fold in FOLDS], ignore_index=True)
        reference = expected_test(grouped.assign(split="test"), task)
        if order["sample_id"].duplicated().any() or set(order["sample_id"]) != set(reference["sample_id"]):
            raise ValueError(f"cv {task}: test folds do not predict every image exactly once")
        targets = targets_of(order, task)
        predictions = np.stack([np.concatenate([labels[(fold, arch)] for fold in FOLDS]) for arch in complete_models])
        pooled = np.array([macro_f1(targets, row, count) for row in predictions])
        if not np.allclose(pooled, [fast_macro_f1(targets, row, count) for row in predictions], rtol=0.0, atol=1e-12):
            raise AssertionError("fast_macro_f1 disagrees with macro_f1")
        groups = order["capture_group_id"].to_numpy()
        group_count = int(len(np.unique(groups)))

        def statistic(indices: np.ndarray) -> np.ndarray:
            sample = targets[indices]
            return np.array([fast_macro_f1(sample, row[indices], count) for row in predictions])

        replicates = cluster_bootstrap(groups, statistic, replicates=context.replicates).reshape(-1, len(complete_models))
        best = int(np.argmax(pooled))
        leader = complete_models[best]
        differences = replicates - replicates[:, [best]]
        for index, arch in enumerate(complete_models):
            low, high = percentile_interval(replicates[:, index])
            difference_low, difference_high = percentile_interval(differences[:, index])
            pooled_records.append(
                {
                    "architecture": arch,
                    "pooled_macro_f1": float(pooled[index]),
                    "pooled_ci_low": low,
                    "pooled_ci_high": high,
                    "difference_vs_leader": float(pooled[index] - pooled[best]),
                    "difference_ci_low": difference_low,
                    "difference_ci_high": difference_high,
                    "share_replicates_below_leader": float(np.mean(differences[:, index] < 0.0)),
                    "is_leader": arch == leader,
                }
            )
    pooled_columns = ["architecture", "pooled_macro_f1", "pooled_ci_low", "pooled_ci_high", "difference_vs_leader",
                      "difference_ci_low", "difference_ci_high", "share_replicates_below_leader", "is_leader"]
    summary = summary.merge(pd.DataFrame(pooled_records, columns=pooled_columns), on="architecture", how="left")
    tests: dict[str, object] = {
        "task": task,
        "available_runs": len(records),
        "expected_runs": len(FOLDS) * len(MODEL_ORDER),
        "complete_folds": complete_folds,
        "complete_models": complete_models,
        "leader": leader,
        "bootstrap_replicates": context.replicates,
        "bootstrap_unit": "capture_group_id",
        "bootstrap_groups": group_count,
        "max_abs_difference_vs_test_metrics_json": deviation,
        "friedman_statistic": None,
        "friedman_p_value": None,
        "kendall_w": None,
    }
    if len(complete_folds) == len(FOLDS):
        scores = folds.pivot(index="fold", columns="architecture", values="macro_f1")[list(MODEL_ORDER)].to_numpy()
        statistic_value, p_value = stats.friedmanchisquare(*scores.T)
        tests |= {
            "friedman_statistic": float(statistic_value),
            "friedman_p_value": float(p_value),
            "friedman_df": len(MODEL_ORDER) - 1,
            "kendall_w": float(statistic_value / (len(FOLDS) * (len(MODEL_ORDER) - 1))),
        }
    return {"folds": folds, "summary": summary, "tests": tests}, missing


def cross_validation(
    context: Context, status: Status, grouped: pd.DataFrame, fold_manifests: dict[int, pd.DataFrame]
) -> None:
    results: dict[str, dict[str, object]] = {}
    pending: list[str] = []
    for task in TASKS:
        result, missing = cv_task(context, task, grouped, fold_manifests)
        pending += missing
        if result is not None:
            results[task] = result
            context.csv(result["folds"], context.splits_dir, f"cv_{task}_folds.csv")
            context.csv(result["summary"], context.splits_dir, f"cv_{task}_summary.csv")
            context.json(result["tests"], context.splits_dir, f"cv_{task}_tests.json")
        note = f"partial summary over {len(FOLDS) * len(MODEL_ORDER) - len(missing)}/25 runs written" if result else ""
        status.set(f"cv_{task}", missing, note)
    if not results:
        status.set("cv_table", pending)
        status.set("cv_figure", pending)
        return
    cv_table(context, results, grouped)
    plot_cv(context, results)
    status.set("cv_table", pending)
    status.set("cv_figure", pending)


def cv_table(context: Context, results: dict[str, dict[str, object]], grouped: pd.DataFrame) -> None:
    width = 6
    rows: list[list[str]] = []
    for task in TASKS:
        test = expected_test(grouped.assign(split="test"), task)
        heading = rf"\textit{{{TASK_LABELS[task]} ({len(test):,} images, {test['capture_group_id'].nunique()} capture groups)}}"
        rows.append([(r"\midrule " if rows else "") + multicolumn(width, heading)])
        result = results.get(task)
        if result is None:
            rows.append([multicolumn(width, "pending")])
            continue
        summary = result["summary"].set_index("architecture")
        tests = result["tests"]
        for arch in MODEL_ORDER:
            if arch not in summary.index:
                rows.append([DISPLAY_NAMES[arch], "--", "--", "--", "--", "--"])
                continue
            record = summary.loc[arch]
            complete = record["folds"] == len(FOLDS)
            folds = result["folds"]
            values = folds.loc[folds["architecture"] == arch, "macro_f1"]
            if pd.isna(record["pooled_macro_f1"]):
                pooled, bounds, difference = "--", "--", "--"
            else:
                pooled = f"{record['pooled_macro_f1']:.3f}"
                bounds = interval(record["pooled_ci_low"], record["pooled_ci_high"])
                if arch == tests["leader"]:
                    difference = "leader"
                else:
                    excludes_zero = record["difference_ci_high"] < 0.0 or record["difference_ci_low"] > 0.0
                    difference = rf"${record['difference_vs_leader']:+.3f}{'^{*}' if excludes_zero else ''}$"
            wins = f"{int(record['folds_ranked_first'])}/{len(FOLDS)}" if record["complete_folds"] == len(FOLDS) else "--"
            rows.append([DISPLAY_NAMES[arch], mean_sd(values) if complete else "--", pooled, bounds, difference, wins])
        if tests["friedman_statistic"] is not None:
            rows.append(
                [
                    multicolumn(
                        width,
                        rf"Friedman $\chi^2_{{{tests['friedman_df']}}}={tests['friedman_statistic']:.2f}$, "
                        rf"$p={tests['friedman_p_value']:.3f}$; Kendall's $W={tests['kendall_w']:.3f}$",
                    )
                ]
            )
    expected_runs = len(FOLDS) * len(MODEL_ORDER)
    pending = any(task not in results or results[task]["tests"]["available_runs"] < expected_runs for task in TASKS)
    context.table(
        "rev_cv",
        caption=(
            "Grouped 5-fold cross-validation (seed 2026; uncalibrated softmax). Folds: mean $\\pm$ sample SD of the "
            "per-fold test macro-F1. Pooled: out-of-fold macro-F1 with each image predicted once, with a 95\\% "
            f"percentile cluster-bootstrap CI over capture groups ({context.replicates:,} replicates)."
        ),
        headers=["Model", "Folds", "Pooled", r"95\% CI", r"$\Delta$", "Wins"],
        rows=rows,
        notes=(
            r"$\Delta$ = pooled difference from the pooled leader; $^{*}$ paired cluster-bootstrap 95\% CI excludes zero. "
            "Wins = folds ranked first. The Friedman test (asymptotic) and Kendall's $W$ use the five folds as blocks."
            + (" -- = runs pending." if pending else "")
        ),
    )


def plot_cv(context: Context, results: dict[str, dict[str, object]]) -> None:
    plt = pyplot()
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(2, 1, figsize=(SINGLE_COLUMN_WIDTH, 3.7), sharex=True)
    offsets = dict(zip(MODEL_ORDER, np.linspace(-0.2, 0.2, len(MODEL_ORDER))))
    pooled_position = len(FOLDS) + 0.4
    for axis, task in zip(axes, TASKS):
        axis.set_ylabel(f"{TASK_LABELS[task]} test macro-F1")
        result = results.get(task)
        if result is None:
            axis.text(0.5, 0.5, "pending", transform=axis.transAxes, ha="center", va="center", color="0.45", style="italic")
            continue
        folds = result["folds"]
        summary = result["summary"].set_index("architecture")
        for arch in MODEL_ORDER:
            rows = folds[folds["architecture"] == arch].sort_values("fold")
            if rows.empty:
                continue
            axis.plot(
                rows["fold"] + offsets[arch], rows["macro_f1"], color=COLORS[arch], marker="o", markersize=3,
                linewidth=0.7, alpha=0.9,
            )
            record = summary.loc[arch]
            if not pd.isna(record["pooled_macro_f1"]):
                value = record["pooled_macro_f1"]
                axis.errorbar(
                    pooled_position + offsets[arch], value,
                    yerr=[[value - record["pooled_ci_low"]], [record["pooled_ci_high"] - value]],
                    color=COLORS[arch], marker="s", markersize=3.5, elinewidth=0.9, capsize=1.5, linestyle="none",
                )
        axis.axvline(len(FOLDS) - 0.3, color="0.75", linewidth=0.6)
        pending = result["tests"]["available_runs"] < len(FOLDS) * len(MODEL_ORDER)
        if pending:
            axis.text(0.99, 0.03, "partial: runs pending", transform=axis.transAxes, ha="right", va="bottom", fontsize=7, color="0.45", style="italic")
    axes[-1].set_xticks([*FOLDS, pooled_position], [*(f"Fold {fold}" for fold in FOLDS), "Pooled\n(95% CI)"])
    axes[-1].set_xlim(-0.5, pooled_position + 0.5)
    handles = [
        Line2D([], [], color=COLORS[arch], marker="o", markersize=3, linewidth=0.7, label=DISPLAY_NAMES[arch]) for arch in MODEL_ORDER
    ]
    figure.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.07), columnspacing=1.0)
    figure.tight_layout()
    context.figure(figure, "cv_folds")
    plt.close(figure)


def capture_group_profile(grouped: pd.DataFrame, assignment: pd.DataFrame) -> pd.DataFrame:
    counts = pd.crosstab(grouped["capture_group_id"], grouped["freshness_label"]).reindex(columns=list(THREE_STAGE), fill_value=0)
    merged = grouped.merge(assignment, on="sample_id", how="left", validate="one_to_one")
    profile = merged.groupby("capture_group_id").agg(
        produce_type=("produce_type", "first"),
        produce_types=("produce_type", "nunique"),
        images=("sample_id", "size"),
        duplicate_groups=("duplicate_group_id", "nunique"),
        grouped_split=("split", "first"),
        grouped_splits=("split", "nunique"),
        cv_fold=("fold", "first"),
        cv_folds=("fold", "nunique"),
    )
    if (profile[["produce_types", "grouped_splits", "cv_folds"]] != 1).any().any():
        raise ValueError("A capture group spans produce types, grouped partitions, or CV folds")
    profile = profile.drop(columns=["produce_types", "grouped_splits", "cv_folds"]).join(counts)
    present = profile[list(THREE_STAGE)] > 0
    profile["stage_count"] = present.sum(axis=1)
    profile["stages"] = [stage_combination([stage for stage in THREE_STAGE if row[stage]]) for _, row in present.iterrows()]
    profile = profile.reset_index()
    return profile.sort_values(["images", "capture_group_id"], ascending=[False, True], kind="stable").reset_index(drop=True)


def size_statistics(scope: str, sizes: np.ndarray) -> dict[str, object]:
    values = np.sort(np.asarray(sizes, dtype=int))[::-1]
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    top = int(values[:10].sum())
    return {
        "scope": scope,
        "groups": int(len(values)),
        "images": int(values.sum()),
        "min": int(values.min()),
        "q1": float(q1),
        "median": float(median),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "mean": float(values.mean()),
        "max": int(values.max()),
        "top10_images": top,
        "top10_share": top / int(values.sum()),
    }


def capture_group_sizes(context: Context, status: Status, profile: pd.DataFrame) -> None:
    records = [size_statistics("all", profile["images"].to_numpy())]
    for split in SPLITS:
        records.append(size_statistics(f"grouped_{split}", profile.loc[profile["grouped_split"] == split, "images"].to_numpy()))
    for fold in FOLDS:
        records.append(size_statistics(f"cv_fold-{fold}_test", profile.loc[profile["cv_fold"] == f"fold-{fold}", "images"].to_numpy()))
    context.csv(profile, context.splits_dir, "capture_group_profile.csv")
    context.csv(pd.DataFrame(records), context.splits_dir, "capture_group_size_statistics.csv")
    status.set("capture_group_sizes")


def frozen_reproduction(context: Context, status: Status, manifests: dict[str, pd.DataFrame]) -> None:
    summary = pd.read_csv(FROZEN / "aggregate/benchmark_summary.csv")
    records = []
    plans = (
        ("grouped", "three_stage", "calibrated", MODEL_ORDER),
        ("grouped", "three_stage", "uncalibrated", MODEL_ORDER),
        ("grouped", "binary", "uncalibrated", MODEL_ORDER),
        ("naive", "three_stage", "uncalibrated", ABLATION_MODELS),
    )
    argmax: dict[tuple[str, int, str], np.ndarray] = {}
    for setting, task, calibration, architectures in plans:
        expected = expected_test(manifests["L4_capture" if setting == "grouped" else "L0_naive"], task)
        targets = targets_of(expected, task)
        for arch in architectures:
            scores, mismatched = [], 0
            for seed in SEEDS:
                frame = frozen_predictions(task, arch, seed, setting, calibration)
                labels = aligned_probabilities(frame, expected, task, f"{setting}/{task}/{arch}/{seed}").argmax(axis=1)
                stored = frame.set_index("sample_id").loc[expected["sample_id"], "predicted_index"].to_numpy(dtype=int)
                mismatched += int((stored != labels).sum())
                scores.append(macro_f1(targets, labels, len(class_names(task))))
                if setting == "grouped" and task == "three_stage":
                    argmax[(arch, seed, calibration)] = labels
            frozen = summary[
                (summary["setting"] == setting) & (summary["task"] == task) & (summary["architecture"] == arch)
            ]
            reference = float(frozen["macro_f1_mean"].iloc[0]) if len(frozen) else float("nan")
            records.append(
                {
                    "setting": setting,
                    "task": task,
                    "calibration": calibration,
                    "architecture": arch,
                    "recomputed_macro_f1_mean": float(np.mean(scores)),
                    "recomputed_macro_f1_sd": float(np.std(scores, ddof=1)),
                    "frozen_macro_f1_mean": reference,
                    "absolute_difference": abs(float(np.mean(scores)) - reference),
                    "predicted_index_mismatches": mismatched,
                }
            )
    disagreements = sum(
        int((argmax[(arch, seed, "calibrated")] != argmax[(arch, seed, "uncalibrated")]).sum())
        for arch in MODEL_ORDER
        for seed in SEEDS
    )
    frame = pd.DataFrame(records)
    worst = float(frame["absolute_difference"].max())
    payload = {
        "macro_f1": frame.to_dict(orient="records"),
        "max_absolute_difference": worst,
        "calibrated_vs_uncalibrated_argmax_disagreements": disagreements,
        "reproduced": bool(worst < 1e-9 and disagreements == 0 and frame["predicted_index_mismatches"].sum() == 0),
    }
    context.json(payload, context.splits_dir, "frozen_reproduction.json")
    if not payload["reproduced"]:
        print(f"WARNING frozen reproduction mismatch: max |diff|={worst:.3g}, argmax disagreements={disagreements}", flush=True)
    status.set("frozen_reproduction")


def run_splits(context: Context, manifests: dict[str, pd.DataFrame]) -> None:
    status = Status()
    fold_manifests, assignment = load_fold_manifests()
    staged, missing, deviation = collect_staged(context, manifests)
    if deviation > 1e-6:
        print(f"WARNING staged runs: recomputed macro-F1 differs from test_metrics.json by {deviation:.3g}", flush=True)
    frozen_reproduction(context, status, manifests)
    staged_ablation(context, status, manifests, staged, missing)
    leakage_mechanism(context, status, manifests, staged, missing)
    split_composition(context, status, manifests, fold_manifests)
    standardised_macro_f1(context, status, manifests, staged, missing)
    cross_validation(context, status, manifests["L4_capture"], fold_manifests)
    capture_group_sizes(context, status, capture_group_profile(manifests["L4_capture"], assignment))
    context.json(status.parts, context.splits_dir, "status.json")


def exact_conflicts(context: Context) -> dict[str, object]:
    frame = pd.read_csv(LABEL_CONFLICTS, dtype=str, keep_default_na=False)
    groups = frame.groupby("conflict_group_id").agg(
        produce_type=("produce_type", lambda values: "+".join(sorted(set(values)))),
        stages=("freshness_label", stage_combination),
        folder_ranges=("shelf_life_label", lambda values: " | ".join(sorted(set(values), key=lambda text: int(DAY_RANGE.search(text).group(1))))),
        files=("source_path", "size"),
    )
    summary = (
        groups.groupby(["produce_type", "stages", "folder_ranges"])
        .agg(conflict_groups=("files", "size"), files=("files", "sum"))
        .reset_index()
    )
    context.csv(summary, context.labels_dir, "exact_label_conflicts.csv")
    return {
        "conflict_groups": int(len(groups)),
        "files": int(groups["files"].sum()),
        "combinations": summary.to_dict(orient="records"),
    }


def near_duplicate_labels(context: Context, grouped: pd.DataFrame) -> dict[str, object]:
    report = pd.read_csv(NEAR_DUPLICATES, dtype={"sample_id_a": str, "sample_id_b": str})
    labels = grouped.set_index("sample_id")["freshness_label"]
    original = pd.read_csv(ORIGINAL_MANIFEST, usecols=["sample_id", "freshness_label"], dtype=str).set_index("sample_id")
    lookup = labels.combine_first(original["freshness_label"])
    linked = report["linked"].astype(str).str.lower() == "true"
    same_produce = report["produce_type_a"] == report["produce_type_b"]
    pairs = report[linked & same_produce].copy()
    pairs["label_a"] = pairs["sample_id_a"].map(lookup)
    pairs["label_b"] = pairs["sample_id_b"].map(lookup)
    unresolved = int((pairs["label_a"].isna() | pairs["label_b"].isna()).sum())
    if unresolved:
        raise ValueError(f"{unresolved} near-duplicate pairs have no freshness label")
    pairs["stage_pair"] = [stage_pair(a, b) for a, b in zip(pairs["label_a"], pairs["label_b"])]
    pairs["labels_differ"] = pairs["label_a"] != pairs["label_b"]
    order = ["same stage", "Fresh/Semi-Fresh", "Semi-Fresh/Rotten", "Fresh/Rotten"]
    by_pair = pairs["stage_pair"].value_counts().reindex(order, fill_value=0)
    differing = pairs[pairs["labels_differ"]]
    by_produce = (
        differing.groupby(["produce_type_a", "stage_pair"]).size().rename("pairs").reset_index()
        .rename(columns={"produce_type_a": "produce_type"})
    )
    context.csv(
        pairs[["sample_id_a", "sample_id_b", "produce_type_a", "phash_distance", "dhash_distance", "label_a", "label_b",
               "stage_pair", "labels_differ"]].rename(columns={"produce_type_a": "produce_type"}),
        context.labels_dir,
        "near_duplicate_label_pairs.csv",
    )
    context.csv(by_produce, context.labels_dir, "near_duplicate_label_conflicts_by_produce.csv")
    return {
        "linked_within_produce_pairs": int(len(pairs)),
        "unlinked_or_cross_produce_candidates": int(len(report) - len(pairs)),
        "pairs_with_different_stages": int(len(differing)),
        "images_in_differing_pairs": int(len(set(differing["sample_id_a"]) | set(differing["sample_id_b"]))),
        "by_stage_pair": {key: int(value) for key, value in by_pair.items()},
    }


def multi_stage_groups(context: Context, profile: pd.DataFrame) -> dict[str, object]:
    spanning = profile[profile["stage_count"] > 1]
    combinations = (
        spanning.groupby("stages").agg(groups=("capture_group_id", "size"), images=("images", "sum")).reset_index()
    )
    context.csv(spanning, context.labels_dir, "multi_stage_capture_groups.csv")
    context.csv(combinations, context.labels_dir, "multi_stage_capture_group_combinations.csv")
    return {
        "capture_groups": int(len(profile)),
        "spanning_groups": int(len(spanning)),
        "spanning_images": int(spanning["images"].sum()),
        "spanning_groups_by_grouped_split": {key: int(value) for key, value in spanning["grouped_split"].value_counts().sort_index().items()},
        "spanning_images_by_grouped_split": {key: int(value) for key, value in spanning.groupby("grouped_split")["images"].sum().items()},
        "by_combination": combinations.to_dict(orient="records"),
    }


def day_range_boundaries(context: Context, grouped: pd.DataFrame) -> dict[str, object]:
    ranges = grouped.groupby(["produce_type", "freshness_label"])["shelf_life_label"].unique()
    records = []
    for produce in sorted(grouped["produce_type"].unique()):
        spans = {}
        for stage in THREE_STAGE:
            values = ranges[(produce, stage)]
            if len(values) != 1:
                raise ValueError(f"{produce}/{stage} has {len(values)} folder day ranges")
            match = DAY_RANGE.search(values[0])
            spans[stage] = (int(match.group(1)), int(match.group(2)))
        for lower, upper in zip(THREE_STAGE[:-1], THREE_STAGE[1:]):
            end, start = spans[lower][1], spans[upper][0]
            relation = "shared endpoint" if end == start else ("gap" if start > end else "overlap")
            records.append(
                {
                    "produce_type": produce,
                    "boundary": f"{lower}/{upper}",
                    "lower_range": "{}-{}".format(*spans[lower]),
                    "upper_range": "{}-{}".format(*spans[upper]),
                    "lower_end_day": end,
                    "upper_start_day": start,
                    "relation": relation,
                    "days_in_both": max(0, end - start + 1),
                }
            )
    frame = pd.DataFrame(records)
    context.csv(frame, context.labels_dir, "day_range_boundaries.csv")
    return {
        "boundaries": int(len(frame)),
        "by_relation": {key: int(value) for key, value in frame["relation"].value_counts().sort_index().items()},
    }


def consensus_difficulty(
    context: Context, grouped: pd.DataFrame, profile: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    expected = expected_test(grouped, "three_stage")
    targets = targets_of(expected, "three_stage")
    checkpoints = [(arch, seed) for arch in MODEL_ORDER for seed in SEEDS]
    probabilities = np.stack(
        [
            aligned_probabilities(frozen_test_predictions("three_stage", arch, seed), expected, "three_stage", f"{arch}/seed-{seed}")
            for arch, seed in checkpoints
        ]
    )
    predictions = probabilities.argmax(axis=2)
    wrong = predictions != targets
    votes = np.stack([(predictions == index).sum(axis=0) for index in range(len(THREE_STAGE))], axis=1)
    mean_probabilities = probabilities.mean(axis=0)
    tied = votes == votes.max(axis=1, keepdims=True)
    majority = np.where(tied, mean_probabilities, -np.inf).argmax(axis=1)
    images = expected[["sample_id", "source_path", "produce_type", "freshness_label", "shelf_life_label", "capture_group_id"]].copy()
    images["checkpoints_wrong"] = wrong.sum(axis=0)
    images["share_wrong"] = wrong.mean(axis=0)
    images["majority_prediction"] = [THREE_STAGE[index] for index in majority]
    for index, stage in enumerate(THREE_STAGE):
        images[f"votes_{STAGE_SHORT[stage]}"] = votes[:, index]
    for index, stage in enumerate(THREE_STAGE):
        images[f"mean_probability_{STAGE_SHORT[stage]}"] = mean_probabilities[:, index]
    images["mean_true_class_probability"] = mean_probabilities[np.arange(len(targets)), targets]
    images = images.sort_values(
        ["share_wrong", "mean_true_class_probability", "sample_id"], ascending=[False, True, True], kind="stable"
    ).reset_index(drop=True)
    images.insert(0, "dispute_rank", np.arange(1, len(images) + 1))
    context.csv(images, context.labels_dir, "consensus_test_images.csv")

    records = []
    for stage in [*THREE_STAGE, "all"]:
        subset = images if stage == "all" else images[images["freshness_label"] == stage]
        share = subset["share_wrong"].to_numpy()
        records.append(
            {
                "true_class": stage,
                "images": int(len(subset)),
                "mean_share_wrong": float(share.mean()),
                "q1_share_wrong": float(np.percentile(share, 25)),
                "median_share_wrong": float(np.median(share)),
                "q3_share_wrong": float(np.percentile(share, 75)),
                "never_wrong": int((share == 0).sum()),
                "majority_wrong": int((share > 0.5).sum()),
                "wrong_at_least_80pct": int((share >= DISPUTE_THRESHOLD).sum()),
                "always_wrong": int((share == 1).sum()),
            }
        )
    by_class = pd.DataFrame(records)
    hard = images[images["share_wrong"] >= DISPUTE_THRESHOLD]
    table = pd.crosstab(hard["produce_type"], hard["freshness_label"]).reindex(
        index=sorted(grouped["produce_type"].unique()), columns=list(THREE_STAGE), fill_value=0
    )
    totals = pd.crosstab(images["produce_type"], images["freshness_label"]).reindex(
        index=table.index, columns=list(THREE_STAGE), fill_value=0
    )
    produce = table.add_suffix("_hard").join(totals.add_suffix("_test")).reset_index()
    context.csv(by_class, context.labels_dir, "consensus_by_class.csv")
    context.csv(produce, context.labels_dir, "consensus_hard_by_produce.csv")

    error_records = []
    for (arch, seed), labels in zip(checkpoints, predictions):
        distance = np.abs(labels - targets)
        errors = distance > 0
        record = {
            "architecture": arch,
            "seed": seed,
            "errors": int(errors.sum()),
            "adjacent_errors": int((distance == 1).sum()),
            "fresh_rotten_errors": int((distance == 2).sum()),
            "adjacent_share": float((distance == 1).sum() / errors.sum()),
            "fresh_rotten_share": float((distance == 2).sum() / errors.sum()),
            "mean_absolute_ordinal_error": float(distance.mean()),
            "mean_ordinal_error_given_error": float(distance[errors].mean()),
        }
        for true_index, true_stage in enumerate(THREE_STAGE):
            for predicted_index, predicted_stage in enumerate(THREE_STAGE):
                if true_index != predicted_index:
                    record[f"{STAGE_SHORT[true_stage]}_to_{STAGE_SHORT[predicted_stage]}"] = int(
                        ((targets == true_index) & (labels == predicted_index)).sum()
                    )
        error_records.append(record)
    errors = pd.DataFrame(error_records)
    metrics = ["adjacent_share", "fresh_rotten_share", "mean_absolute_ordinal_error", "mean_ordinal_error_given_error", "errors"]
    context.csv(errors, context.labels_dir, "error_structure_seeds.csv")
    context.csv(summarise(errors, ["architecture"], metrics), context.labels_dir, "error_structure_summary.csv")

    model_means = errors.groupby("architecture")[["adjacent_share", "mean_absolute_ordinal_error"]].mean()
    hard_groups = hard["capture_group_id"].value_counts()
    spanning = set(profile.loc[profile["stage_count"] > 1, "capture_group_id"])
    evidence = {
        "checkpoints": len(checkpoints),
        "test_images": int(len(images)),
        "test_capture_groups": int(images["capture_group_id"].nunique()),
        "wrong_at_least_80pct": int(len(hard)),
        "wrong_at_least_80pct_by_class": {stage: int((hard["freshness_label"] == stage).sum()) for stage in THREE_STAGE},
        "wrong_at_least_80pct_capture_groups": int(len(hard_groups)),
        "wrong_at_least_80pct_in_five_largest_groups": int(hard_groups.head(5).sum()),
        "wrong_at_least_80pct_in_multi_stage_groups": int(hard["capture_group_id"].isin(spanning).sum()),
        "test_images_in_multi_stage_groups": int(images["capture_group_id"].isin(spanning).sum()),
        "always_wrong": int((images["share_wrong"] == 1).sum()),
        "always_wrong_capture_groups": int(images.loc[images["share_wrong"] == 1, "capture_group_id"].nunique()),
        "never_wrong": int((images["share_wrong"] == 0).sum()),
        "adjacent_share_range": [float(model_means["adjacent_share"].min()), float(model_means["adjacent_share"].max())],
        "mean_absolute_ordinal_error_range": [
            float(model_means["mean_absolute_ordinal_error"].min()),
            float(model_means["mean_absolute_ordinal_error"].max()),
        ],
    }
    return images, evidence


def sheet_selection(images: pd.DataFrame, count: int) -> pd.DataFrame:
    """Most-disputed images; equally disputed images are interleaved across capture groups."""
    ranked = images.assign(turn=images.groupby(["checkpoints_wrong", "capture_group_id"]).cumcount())
    ranked = ranked.sort_values(["checkpoints_wrong", "turn", "dispute_rank"], ascending=[False, True, True], kind="stable")
    return ranked.head(count).drop(columns="turn").reset_index(drop=True)


def disputed_images(context: Context, status: Status, images: pd.DataFrame) -> None:
    top = sheet_selection(images, SHEET_SHAPE[0] * SHEET_SHAPE[1])
    positions = pd.Series(np.arange(1, len(top) + 1), index=top["sample_id"].to_numpy())
    disputed = images[images["share_wrong"] >= DISPUTE_THRESHOLD].copy()
    disputed["contact_sheet_position"] = disputed["sample_id"].map(positions).astype("Int64")
    context.csv(disputed, context.labels_dir, "disputed_test_images.csv")
    status.set("disputed_images_csv")
    paths = [resolve_image_path(row, context.image_root) for _, row in top.iterrows()]
    absent = [str(path) for path in paths if not path.is_file()]
    if absent:
        status.set("disputed_examples_figure", absent, f"image root {context.image_root}")
        return
    plot_disputed(context, top, paths)
    status.set("disputed_examples_figure")


def plot_disputed(context: Context, top: pd.DataFrame, paths: Sequence[Path]) -> None:
    from PIL import Image

    plt = pyplot()
    rows, columns = SHEET_SHAPE
    figure, axes = plt.subplots(rows, columns, figsize=(DOUBLE_COLUMN_WIDTH, 6.55))
    figure.subplots_adjust(left=0.005, right=0.995, top=0.955, bottom=0.055, wspace=0.05, hspace=0.42)
    for axis in axes.flat:
        axis.set_axis_off()
    checkpoints = len(MODEL_ORDER) * len(SEEDS)
    for axis, (_, record), path in zip(axes.flat, top.iterrows(), paths):
        with Image.open(path) as source:
            thumbnail = source.convert("RGB")
        thumbnail.thumbnail((THUMBNAIL_PIXELS, THUMBNAIL_PIXELS), Image.Resampling.LANCZOS)
        axis.imshow(np.asarray(thumbnail))
        days = str(record["shelf_life_label"]).replace(" days", " d")
        caption = (
            f"#{record['dispute_rank']} {produce_name(record['produce_type'])}\n"
            f"{record['freshness_label']} → {record['majority_prediction']}\n"
            f"{days} · {record['checkpoints_wrong']}/{checkpoints} wrong"
        )
        axis.text(0.5, -0.025, caption, transform=axis.transAxes, ha="center", va="top", fontsize=7, linespacing=1.15)
    figure.suptitle(
        f"Most-disputed grouped-test images: #rank, produce; folder label → majority prediction of {checkpoints} "
        "calibrated three-stage checkpoints; folder day range, checkpoints wrong",
        fontsize=7,
    )
    context.figure(figure, "disputed_examples")
    plt.close(figure)


def label_evidence_table(context: Context, evidence: dict[str, dict[str, object]]) -> None:
    conflicts = evidence["exact_conflicts"]
    near = evidence["near_duplicates"]
    spanning = evidence["multi_stage_groups"]
    days = evidence["day_ranges"]
    consensus = evidence["consensus"]
    combination = conflicts["combinations"][0] if len(conflicts["combinations"]) == 1 else None
    conflict_detail = (
        f" ({produce_name(combination['produce_type'])}, "
        f"{combination['folder_ranges'].replace(' days', '').replace(' | ', ' vs ')} days)"
        if combination else ""
    )
    pairs = near["by_stage_pair"]
    differing = ", ".join(
        f"{'/'.join(STAGE_SHORT[stage] for stage in key.split('/'))} {value}"
        for key, value in pairs.items()
        if key != "same stage"
    )
    combos = ", ".join(
        f"{'+'.join(STAGE_SHORT[stage] for stage in record['stages'].split('+'))} {record['groups']}"
        for record in spanning["by_combination"]
    )
    hard = consensus["wrong_at_least_80pct_by_class"]
    shared = days["by_relation"].get("shared endpoint", 0)
    low, high = consensus["adjacent_share_range"]
    mae_low, mae_high = consensus["mean_absolute_ordinal_error_range"]
    rows = [
        ["Exact duplicates with conflicting folders (excluded)", f"{conflicts['conflict_groups']} pairs, {conflicts['files']} files{conflict_detail}"],
        ["Linked near-duplicate pairs with different stages", f"{near['pairs_with_different_stages']} of {near['linked_within_produce_pairs']} ({differing})"],
        ["Capture groups spanning more than one stage", f"{spanning['spanning_groups']} of {spanning['capture_groups']} ({spanning['spanning_images']:,} images; {combos})"],
        ["Adjacent-stage folder day ranges sharing an endpoint day", f"{shared} of {days['boundaries']} boundaries"],
        [
            rf"Test images wrong for $\geq$80\% of {consensus['checkpoints']} checkpoints",
            f"{consensus['wrong_at_least_80pct']} of {consensus['test_images']} ("
            + ", ".join(f"{STAGE_SHORT[stage]} {hard[stage]}" for stage in THREE_STAGE)
            + f"; {consensus['wrong_at_least_80pct_capture_groups']} of {consensus['test_capture_groups']} capture groups)",
        ],
        [
            f"Test images wrong for all {consensus['checkpoints']} checkpoints",
            f"{consensus['always_wrong']} ({consensus['always_wrong_capture_groups']} capture groups)",
        ],
        ["Adjacent-stage share of test errors (model means)", f"{low:.3f}--{high:.3f}"],
        ["Mean absolute ordinal error (model means)", f"{mae_low:.3f}--{mae_high:.3f}"],
    ]
    context.table(
        "rev_label_evidence",
        caption=(
            "Label-validity evidence for folder-derived stages. Consensus rows use the 15 calibrated three-stage "
            "checkpoints (5 models $\\times$ 3 seeds) on the 831-image grouped test."
        ),
        headers=["Evidence", "Value"],
        rows=rows,
        alignment=r">{\raggedright\arraybackslash}p{0.47\linewidth}>{\raggedright\arraybackslash}p{0.49\linewidth}",
        notes=(
            "F = Fresh, SF = Semi-Fresh, R = Rotten. Near-duplicates: pHash and dHash Hamming distance $\\leq$6 within "
            "produce. Adjacent-stage errors confuse neighbouring stages; ordinal error uses F=0, SF=1, R=2."
        ),
    )


def run_labels(context: Context, grouped: pd.DataFrame) -> None:
    status = Status()
    _, assignment = load_fold_manifests()
    profile = capture_group_profile(grouped, assignment)
    evidence: dict[str, dict[str, object]] = {}
    evidence["exact_conflicts"] = exact_conflicts(context)
    status.set("exact_label_conflicts")
    evidence["near_duplicates"] = near_duplicate_labels(context, grouped)
    status.set("near_duplicate_label_pairs")
    evidence["multi_stage_groups"] = multi_stage_groups(context, profile)
    status.set("multi_stage_capture_groups")
    evidence["day_ranges"] = day_range_boundaries(context, grouped)
    status.set("day_range_boundaries")
    images, evidence["consensus"] = consensus_difficulty(context, grouped, profile)
    status.set("consensus_difficulty")
    status.set("error_structure")
    disputed_images(context, status, images)
    context.json(evidence, context.labels_dir, "label_evidence.json")
    label_evidence_table(context, evidence)
    status.set("label_evidence_table")
    context.json(status.parts, context.labels_dir, "status.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--topics", nargs="+", choices=("splits", "labels"), default=["splits", "labels"])
    parser.add_argument("--runs-dir", type=Path, default=REVISION / "runs", help="Root of the revision training runs")
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS)
    parser.add_argument("--tables-dir", type=Path, default=PAPER_TABLES)
    parser.add_argument("--figures-dir", type=Path, default=PAPER_FIGURES)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT, help="Root of the primary dataset images")
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()

    limit_threads()
    context = Context(
        runs=args.runs_dir,
        splits_dir=args.analysis_dir / "splits",
        labels_dir=args.analysis_dir / "labels",
        tables=args.tables_dir,
        figures=args.figures_dir,
        image_root=args.image_root,
        replicates=args.replicates,
    )
    manifests = load_level_manifests()
    if "splits" in args.topics:
        run_splits(context, manifests)
    if "labels" in args.topics:
        run_labels(context, manifests["L4_capture"])
    for path in context.written:
        print(f"wrote {relative(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
