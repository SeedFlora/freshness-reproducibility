"""Robustness, external-validity, and protocol-fairness sensitivity analyses.

Each part is skipped with a "pending" status until its revision-queue inputs
exist, so the script can simply be rerun while the queue fills in:

* corruption: calibrated three-stage checkpoints under 30 corruption conditions
  for three seeds, reproduction of the frozen seed-2026 aggregate, and a paired
  capture-group bootstrap interval for the clean-to-corrupted drop;
* external: all five binary models on the original-only banana/orange external
  set, ranked by binary validation, with validation-fitted temperature scaling
  and the in-domain banana/orange reference;
* lr: validation-selected learning rate per model and the resulting ranking;
* resolution: 160, 224 (frozen), and 320 px inputs;
* scratch: ImageNet-free versus pretrained MobileNetV2.

Macro-F1 is invariant to temperature scaling, so the uncalibrated test
predictions of new training runs compare directly with the calibrated frozen
ones.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from fresh_or_trash.corruptions import CORRUPTIONS
from revision_common import (
    ANALYSIS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    COLORS,
    DISPLAY_NAMES,
    DOUBLE_COLUMN_WIDTH,
    FROZEN,
    GROUPED_MANIFEST,
    MODEL_ORDER,
    PAPER_FIGURES,
    PAPER_TABLES,
    REFERENCE_SEED,
    REVISION,
    ROOT,
    SEEDS,
    apply_temperature,
    capture_groups,
    cluster_bootstrap,
    ece,
    figure_style,
    fit_temperature,
    frozen_test_predictions,
    interval,
    latex_table,
    macro_f1,
    mean_sd,
    metrics,
    percentile_interval,
    probability_columns,
    read_predictions,
    stack_seeds,
    write_json,
)


TOPIC = "sensitivity"
PARTS = ("corruption", "external", "lr", "resolution", "scratch")
SEVERITIES = (1, 2, 3, 4, 5)
CONDITIONS = tuple((name, severity) for name in CORRUPTIONS for severity in SEVERITIES)
CORRUPTION_LABELS = {
    "gaussian_noise": "Gaussian noise",
    "gaussian_blur": "Gaussian blur",
    "brightness": "Brightness",
    "contrast": "Contrast",
    "occlusion": "Occlusion",
    "jpeg": "JPEG compression",
}
CORRUPTION_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_precision",
    "macro_recall",
    "ece",
    "brier",
    "nll",
    "roc_auc_ovr_macro",
)
MARKERS = {"mobilenet_v2": "o", "resnet50": "s", "densenet121": "^", "efficientnet_b0": "D", "convnext_tiny": "v"}
EXTERNAL_MANIFEST = ROOT / "data/external_original_overlap_manifest.csv"
EXTERNAL_FRUITS = ("banana", "orange")
EXTERNAL_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "ece", "nll", "brier", "roc_auc")
ABLATION_MODELS = ("mobilenet_v2", "convnext_tiny")
LEARNING_RATES = {"base": 3e-4, "lr_1e-4": 1e-4, "lr_1e-3": 1e-3}
LEARNING_RATE_LABELS = {"base": r"$3{\times}10^{-4}$", "lr_1e-4": r"$10^{-4}$", "lr_1e-3": r"$10^{-3}$"}
RESOLUTIONS = {"res_160": 160, "base": 224, "res_320": 320}
HEADLINE_TEST_MACRO_F1 = {
    "three_stage": {"mobilenet_v2": 0.627, "resnet50": 0.690, "densenet121": 0.561, "efficientnet_b0": 0.643, "convnext_tiny": 0.645},
    "binary": {"mobilenet_v2": 0.813, "resnet50": 0.897, "densenet121": 0.899, "efficientnet_b0": 0.797, "convnext_tiny": 0.862},
}
EXACT = 1e-6


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _missing(paths: Sequence[Path]) -> str:
    return f"missing {len(paths)} input(s), e.g. {_display(paths[0])}"


def _sd(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    return float(array.std(ddof=1)) if len(array) > 1 else float("nan")


def _ranks(scores: dict[str, float]) -> dict[str, int]:
    ordered = sorted(scores, key=lambda arch: (-scores[arch], MODEL_ORDER.index(arch)))
    return {arch: position + 1 for position, arch in enumerate(ordered)}


def _signed(value: float, digits: int = 3) -> str:
    return rf"${value:.{digits}f}$"


def _math_interval(low: float, high: float) -> str:
    return f"${interval(low, high)}$"


@dataclass
class Context:
    revision: Path
    output: Path
    tables: Path
    figures: Path
    replicates: int
    allow_partial: bool
    status: dict[str, str] = field(default_factory=dict)
    summary: dict[str, object] = field(default_factory=dict)
    written: list[str] = field(default_factory=list)

    def csv(self, frame: pd.DataFrame, name: str) -> None:
        path = self.output / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        self.written.append(_display(path))

    def table(self, name: str, **content: object) -> None:
        path = self.tables / f"{name}.tex"
        latex_table(path, label=f"tab:rev-{name.removeprefix('rev_')}", **content)  # type: ignore[arg-type]
        self.written.append(_display(path))

    def done(self, part: str) -> None:
        self.status[part] = "done"

    def pending(self, part: str, reason: str | Sequence[Path]) -> None:
        message = reason if isinstance(reason, str) else _missing(reason)
        self.status[part] = f"pending: {message}"
        print(f"WARNING {part} pending: {message}")


def _macro_f1_from_confusion(confusion: np.ndarray) -> np.ndarray:
    """Macro-F1 over trailing [true, predicted] axes, scoring an absent and unpredicted class 0 like sklearn."""
    true_positive = np.diagonal(confusion, axis1=-2, axis2=-1)
    denominator = confusion.sum(axis=-1) + confusion.sum(axis=-2)
    scores = np.divide(2.0 * true_positive, denominator, out=np.zeros(denominator.shape), where=denominator > 0)
    return scores.mean(axis=-1)


class GroupBootstrap:
    """Capture-group draws of ``cluster_bootstrap`` with vectorised macro-F1.

    Multiplicities are taken from ``cluster_bootstrap`` itself, so replicates
    are identical to (and paired with) every revision analysis that resamples
    the same samples with the same seed. Macro-F1 is then evaluated for all
    replicates at once from per-group confusion counts.
    """

    def __init__(self, sample_ids: Sequence[str], replicates: int, seed: int = BOOTSTRAP_SEED) -> None:
        groups = capture_groups(list(sample_ids))
        _, self.group_index = np.unique(groups, return_inverse=True)
        self.group_count = int(self.group_index.max()) + 1
        sizes = np.bincount(self.group_index, minlength=self.group_count)
        draws = cluster_bootstrap(
            groups,
            lambda indices: np.bincount(self.group_index[indices], minlength=self.group_count) / sizes,
            replicates=replicates,
            seed=seed,
        )
        self.weights = np.rint(draws)
        self.replicates = replicates

    def macro_f1(self, targets: np.ndarray, predictions: np.ndarray, class_count: int, *, point: bool = False) -> np.ndarray:
        """Macro-F1 of predictions shaped [..., N]: replicates [R, ...], or [...] on the original sample."""
        leading = predictions.shape[:-1]
        flat = predictions.reshape(-1, predictions.shape[-1]).astype(np.int64)
        cells = class_count * class_count
        index = (np.arange(len(flat))[:, None] * self.group_count + self.group_index[None, :]) * cells
        index = index + targets[None, :] * class_count + flat
        counts = np.bincount(index.ravel(), minlength=len(flat) * self.group_count * cells)
        counts = counts.reshape(len(flat), self.group_count, cells).astype(float)
        if point:
            return _macro_f1_from_confusion(counts.sum(axis=1).reshape(*leading, class_count, class_count))
        confusion = np.einsum("rg,kgc->rkc", self.weights, counts)
        return _macro_f1_from_confusion(confusion.reshape(self.replicates, *leading, class_count, class_count))

    def mean_difference(self, targets: np.ndarray, first: Sequence[np.ndarray], second: Sequence[np.ndarray]) -> tuple[float, float]:
        """Percentile interval of mean-over-runs macro-F1 of ``first`` minus that of ``second``."""
        values = self.macro_f1(targets, np.stack(first), 3).mean(axis=1) - self.macro_f1(targets, np.stack(second), 3).mean(axis=1)
        return percentile_interval(values)


@dataclass
class TestSet:
    """Grouped three-stage locked test with frozen calibrated probabilities [S, N, C] per model."""

    order: list[str]
    targets: np.ndarray
    clean: dict[str, np.ndarray]
    bootstrap: GroupBootstrap


@dataclass
class BinaryTestSet:
    order: list[str]
    targets: np.ndarray
    produce: np.ndarray
    probabilities: dict[str, np.ndarray]


@dataclass(frozen=True)
class RunResult:
    seed: int
    validation_macro_f1: float
    test_macro_f1: float
    best_epoch: int
    predictions: np.ndarray


def _frozen_run_json(task: str, arch: str, seed: int) -> dict[str, object]:
    path = FROZEN / "runs/grouped" / task / arch / f"seed-{seed}" / "run.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _references(ctx: Context) -> tuple[TestSet, BinaryTestSet]:
    clean: dict[str, np.ndarray] = {}
    binary: dict[str, np.ndarray] = {}
    order: list[str] | None = None
    targets = binary_order = binary_targets = produce = None
    for arch in MODEL_ORDER:
        arch_order, arch_targets, probabilities = stack_seeds(
            [frozen_test_predictions("three_stage", arch, seed) for seed in SEEDS], "three_stage"
        )
        if order is None:
            order, targets = arch_order, arch_targets
        elif arch_order != order or not np.array_equal(arch_targets, targets):
            raise ValueError(f"Frozen three-stage test predictions of {arch} cover different samples")
        clean[arch] = probabilities
        frames = [frozen_test_predictions("binary", arch, seed) for seed in SEEDS]
        arch_order, arch_targets, probabilities = stack_seeds(frames, "binary")
        if binary_order is None:
            binary_order, binary_targets = arch_order, arch_targets
            mapping = dict(zip(frames[0]["sample_id"].astype(str), frames[0]["produce_type"].astype(str)))
            produce = np.asarray([mapping[sample] for sample in binary_order])
        elif arch_order != binary_order or not np.array_equal(arch_targets, binary_targets):
            raise ValueError(f"Frozen binary test predictions of {arch} cover different samples")
        binary[arch] = probabilities
    assert order is not None and targets is not None and binary_order is not None
    test = TestSet(order, targets, clean, GroupBootstrap(order, ctx.replicates))
    return test, BinaryTestSet(binary_order, binary_targets, produce, binary)


def _clean_reference(ctx: Context, test: TestSet, binary: BinaryTestSet) -> None:
    records = []
    for arch in MODEL_ORDER:
        for index, seed in enumerate(SEEDS):
            records.append(
                {
                    "architecture": arch,
                    "seed": seed,
                    "three_stage_test_macro_f1": macro_f1(test.targets, test.clean[arch][index].argmax(-1), 3),
                    "binary_test_macro_f1": macro_f1(binary.targets, binary.probabilities[arch][index].argmax(-1), 2),
                    "three_stage_validation_macro_f1": float(_frozen_run_json("three_stage", arch, seed)["validation_macro_f1"]),
                    "binary_validation_macro_f1": float(_frozen_run_json("binary", arch, seed)["validation_macro_f1"]),
                }
            )
    frame = pd.DataFrame.from_records(records)
    ctx.csv(frame, "clean_reference")
    checks = []
    for task, expected in HEADLINE_TEST_MACRO_F1.items():
        means = frame.groupby("architecture")[f"{task}_test_macro_f1"].mean()
        for arch in MODEL_ORDER:
            checks.append(
                {
                    "task": task,
                    "architecture": arch,
                    "recomputed_mean": float(means[arch]),
                    "headline": expected[arch],
                    "matches_to_3_decimals": bool(round(float(means[arch]), 3) == expected[arch]),
                }
            )
    mismatches = [check for check in checks if not check["matches_to_3_decimals"]]
    for check in mismatches:
        print(f"WARNING headline mismatch {check['task']}/{check['architecture']}: {check['recomputed_mean']:.4f} vs {check['headline']}")
    ctx.summary["headline_check"] = {"checks": checks, "all_match": not mismatches}
    ctx.done("reference.frozen_headline_check")


def _read_corruption(path: Path, test: TestSet) -> tuple[np.ndarray, int]:
    """Probabilities [conditions, N, C] in CONDITIONS order aligned to the test order, plus argmax mismatches."""
    frame = pd.read_csv(path)
    columns = probability_columns("three_stage")
    missing = {"sample_id", "target_index", "corruption", "severity", *columns} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["corruption"] = frame["corruption"].astype(str)
    frame["severity"] = frame["severity"].astype(int)
    if frame.duplicated(["corruption", "severity", "sample_id"]).any():
        raise ValueError(f"{path} repeats a sample within a condition")
    observed = set(zip(frame["corruption"], frame["severity"]))
    if observed != set(CONDITIONS):
        raise ValueError(f"{path} covers {len(observed & set(CONDITIONS))} of {len(CONDITIONS)} conditions")
    indexed = frame.set_index(["corruption", "severity", "sample_id"]).sort_index()
    stacked = []
    mismatches = 0
    for condition in CONDITIONS:
        block = indexed.loc[condition]
        if len(block) != len(test.order) or set(block.index) != set(test.order):
            raise ValueError(f"{path} condition {condition} does not cover the grouped locked test")
        block = block.loc[test.order]
        if not np.array_equal(block["target_index"].to_numpy(dtype=int), test.targets):
            raise ValueError(f"{path} condition {condition} disagrees with the frozen targets")
        probabilities = block[columns].to_numpy(dtype=float)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        if "predicted_index" in block:
            mismatches += int((block["predicted_index"].to_numpy(dtype=int) != probabilities.argmax(axis=1)).sum())
        stacked.append(probabilities)
    return np.stack(stacked), mismatches


def _corruption_part(ctx: Context, test: TestSet) -> None:
    available: dict[str, dict[int, np.ndarray]] = {arch: {} for arch in MODEL_ORDER}
    missing: list[Path] = []
    condition_records = []
    seed_records = []
    for arch in MODEL_ORDER:
        for index, seed in enumerate(SEEDS):
            path = ctx.revision / "corruption" / arch / f"seed-{seed}" / "corruption_predictions.csv"
            if not path.is_file():
                missing.append(path)
                continue
            corrupted, mismatches = _read_corruption(path, test)
            available[arch][seed] = corrupted
            clean_f1 = macro_f1(test.targets, test.clean[arch][index].argmax(-1), 3)
            scores = []
            for (name, severity), probabilities in zip(CONDITIONS, corrupted):
                values = metrics(test.targets, probabilities, "three_stage")
                record = {"architecture": arch, "seed": seed, "corruption": name, "severity": severity}
                record |= {key: float(values[key]) for key in CORRUPTION_METRICS if key in values}
                record |= {"clean_macro_f1": clean_f1, "macro_f1_drop_from_clean": clean_f1 - float(values["macro_f1"])}
                condition_records.append(record)
                scores.append(float(values["macro_f1"]))
            scores_array = np.asarray(scores)
            worst = int(scores_array.argmin())
            corrupted_mean = float(scores_array.mean())
            seed_records.append(
                {
                    "architecture": arch,
                    "seed": seed,
                    "clean_macro_f1": clean_f1,
                    "mean_corrupted_macro_f1": corrupted_mean,
                    "mean_drop": clean_f1 - corrupted_mean,
                    "retention": corrupted_mean / clean_f1,
                    "worst_macro_f1": float(scores_array[worst]),
                    "worst_corruption": CONDITIONS[worst][0],
                    "worst_severity": CONDITIONS[worst][1],
                    "corrupted_at_least_clean": bool(corrupted_mean >= clean_f1),
                    "predicted_index_mismatches": mismatches,
                }
            )
    parts = ("per_condition", "seed2026_reproduction", "bootstrap_drop", "per_type_drop", "table_rev_corruption", "figure_corruption_severity")
    if not seed_records:
        for part in parts:
            ctx.pending(f"corruption.{part}", missing)
        return
    conditions = pd.DataFrame.from_records(condition_records)
    seeds = pd.DataFrame.from_records(seed_records)
    ctx.csv(conditions, "corruption_condition_metrics")
    ctx.csv(seeds, "corruption_seed_summary")
    if missing:
        ctx.pending("corruption.per_condition", missing)
    else:
        ctx.done("corruption.per_condition")
    _corruption_reproduction(ctx, conditions, seeds)

    complete = [arch for arch in MODEL_ORDER if len(available[arch]) == len(SEEDS)]
    usable = [arch for arch in MODEL_ORDER if available[arch] and (ctx.allow_partial or arch in complete)]
    if not usable:
        for part in parts[2:]:
            ctx.pending(f"corruption.{part}", missing)
        return
    model_records = []
    for arch in usable:
        arch_seeds = [seed for seed in SEEDS if seed in available[arch]]
        predictions = np.stack(
            [
                np.concatenate([test.clean[arch][SEEDS.index(seed)].argmax(-1)[None], available[arch][seed].argmax(-1)])
                for seed in arch_seeds
            ]
        )
        point = test.bootstrap.macro_f1(test.targets, predictions, 3, point=True)
        frame = seeds[seeds["architecture"] == arch].set_index("seed").loc[arch_seeds]
        expected = conditions[conditions["architecture"] == arch].set_index(["seed", "corruption", "severity"])["macro_f1"]
        reference = np.asarray(
            [[frame.loc[seed, "clean_macro_f1"], *[expected[(seed, *condition)] for condition in CONDITIONS]] for seed in arch_seeds]
        )
        if np.max(np.abs(point - reference)) > 1e-12:
            raise AssertionError("Vectorised macro-F1 disagrees with sklearn")
        replicates = test.bootstrap.macro_f1(test.targets, predictions, 3)
        clean_replicates = replicates[..., 0]
        corrupted_replicates = replicates[..., 1:].mean(axis=-1)
        drop = (clean_replicates - corrupted_replicates).mean(axis=-1)
        retention = np.divide(
            corrupted_replicates, clean_replicates, out=np.full(clean_replicates.shape, np.nan), where=clean_replicates > 0
        ).mean(axis=-1)
        drop_low, drop_high = percentile_interval(drop)
        corrupted_low, corrupted_high = percentile_interval(corrupted_replicates.mean(axis=-1))
        retention_low, retention_high = percentile_interval(retention[np.isfinite(retention)])
        model_records.append(
            {
                "architecture": arch,
                "seeds": ";".join(str(seed) for seed in arch_seeds),
                "seed_count": len(arch_seeds),
                "clean_macro_f1_mean": float(frame["clean_macro_f1"].mean()),
                "clean_macro_f1_sd": _sd(frame["clean_macro_f1"]),
                "mean_corrupted_macro_f1_mean": float(frame["mean_corrupted_macro_f1"].mean()),
                "mean_corrupted_macro_f1_sd": _sd(frame["mean_corrupted_macro_f1"]),
                "mean_corrupted_ci_low": corrupted_low,
                "mean_corrupted_ci_high": corrupted_high,
                "mean_drop_mean": float(frame["mean_drop"].mean()),
                "mean_drop_sd": _sd(frame["mean_drop"]),
                "mean_drop_ci_low": drop_low,
                "mean_drop_ci_high": drop_high,
                "mean_drop_bootstrap_share_nonpositive": float(np.mean(drop <= 0.0)),
                "retention_mean": float(frame["retention"].mean()),
                "retention_sd": _sd(frame["retention"]),
                "retention_ci_low": retention_low,
                "retention_ci_high": retention_high,
                "worst_macro_f1_mean": float(frame["worst_macro_f1"].mean()),
                "worst_macro_f1_sd": _sd(frame["worst_macro_f1"]),
                "worst_conditions": "; ".join(
                    f"{seed}:{row.worst_corruption}/{row.worst_severity}" for seed, row in frame.iterrows()
                ),
                "seeds_corrupted_at_least_clean": int(frame["corrupted_at_least_clean"].sum()),
                "bootstrap_replicates": test.bootstrap.replicates,
                "bootstrap_groups": test.bootstrap.group_count,
            }
        )
    models = pd.DataFrame.from_records(model_records)
    ctx.csv(models, "corruption_model_summary")
    pending_message = _missing(missing) if missing else ""
    if len(complete) == len(MODEL_ORDER):
        ctx.done("corruption.bootstrap_drop")
    else:
        ctx.pending("corruption.bootstrap_drop", f"{len(complete)}/{len(MODEL_ORDER)} models complete; {pending_message}")

    pairs = {(arch, seed) for arch in usable for seed in available[arch]}
    used = conditions[[pair in pairs for pair in zip(conditions["architecture"], conditions["seed"])]]
    type_seed = (
        used.groupby(["architecture", "seed", "corruption"], sort=False)
        .agg(type_macro_f1=("macro_f1", "mean"), clean_macro_f1=("clean_macro_f1", "first"))
        .reset_index()
    )
    type_seed["type_drop"] = type_seed["clean_macro_f1"] - type_seed["type_macro_f1"]
    types = (
        type_seed.groupby(["architecture", "corruption"], sort=False)
        .agg(
            seed_count=("seed", "size"),
            type_macro_f1_mean=("type_macro_f1", "mean"),
            type_macro_f1_sd=("type_macro_f1", "std"),
            type_drop_mean=("type_drop", "mean"),
            type_drop_sd=("type_drop", "std"),
            seeds_type_corrupted_at_least_clean=("type_drop", lambda values: int((values <= 0.0).sum())),
        )
        .reset_index()
    )
    ctx.csv(types, "corruption_type_drop")
    curves = (
        used.groupby(["architecture", "corruption", "severity"], sort=False)
        .agg(seed_count=("seed", "size"), macro_f1_mean=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"))
        .reset_index()
    )
    clean_rows = [
        {
            "architecture": row.architecture,
            "corruption": name,
            "severity": 0,
            "seed_count": row.seed_count,
            "macro_f1_mean": row.clean_macro_f1_mean,
            "macro_f1_sd": row.clean_macro_f1_sd,
        }
        for row in models.itertuples(index=False)
        for name in CORRUPTIONS
    ]
    curves = pd.concat([pd.DataFrame.from_records(clean_rows), curves], ignore_index=True)
    curves["model_rank"] = curves["architecture"].map(MODEL_ORDER.index)
    curves["corruption_rank"] = curves["corruption"].map(CORRUPTIONS.index)
    curves = curves.sort_values(["model_rank", "corruption_rank", "severity"]).drop(columns=["model_rank", "corruption_rank"])
    ctx.csv(curves, "corruption_severity_curves")
    if len(complete) == len(MODEL_ORDER):
        ctx.done("corruption.per_type_drop")
    else:
        ctx.pending("corruption.per_type_drop", f"{len(complete)}/{len(MODEL_ORDER)} models complete; {pending_message}")

    ctx.summary["corruption"] = {
        "models": {
            row["architecture"]: {key: value for key, value in row.items() if key != "architecture"} for row in model_records
        },
        "most_damaging_type": {
            arch: str(types[types["architecture"] == arch].sort_values("type_drop_mean").iloc[-1]["corruption"]) for arch in usable
        },
    }
    if len(complete) == len(MODEL_ORDER) or ctx.allow_partial:
        _corruption_table(ctx, models, seeds)
        _corruption_figure(ctx, curves, usable)
    if len(complete) == len(MODEL_ORDER):
        ctx.done("corruption.table_rev_corruption")
        ctx.done("corruption.figure_corruption_severity")
    else:
        message = f"{len(complete)}/{len(MODEL_ORDER)} models complete; {pending_message}"
        ctx.pending("corruption.table_rev_corruption", message)
        ctx.pending("corruption.figure_corruption_severity", message)


def _corruption_reproduction(ctx: Context, conditions: pd.DataFrame, seeds: pd.DataFrame) -> None:
    frozen = pd.read_csv(FROZEN / "aggregate/corruption_metrics.csv")
    frozen = frozen[frozen["seed"] == REFERENCE_SEED]
    ours = conditions[conditions["seed"] == REFERENCE_SEED]
    self_check = _corruption_self_check(ctx, conditions)
    if ours.empty:
        ctx.summary["corruption_reproduction"] = {"self_check": self_check}
        ctx.pending("corruption.seed2026_reproduction", "no seed-2026 corruption predictions yet")
        return
    compared = [metric for metric in CORRUPTION_METRICS if metric in frozen.columns]
    keys = ["architecture", "corruption", "severity"]
    merged = ours[keys + compared + ["clean_macro_f1"]].merge(
        frozen[keys + compared + ["clean_macro_f1"]], on=keys, how="inner", suffixes=("", "_frozen"), validate="one_to_one"
    )
    for metric in [*compared, "clean_macro_f1"]:
        merged[f"{metric}_abs_diff"] = (merged[metric] - merged[f"{metric}_frozen"]).abs()
    ctx.csv(merged, "corruption_reproduction")
    by_model = merged.groupby("architecture")["macro_f1_abs_diff"].max()
    frozen_summary = pd.read_csv(FROZEN / "aggregate/corruption_summary.csv").set_index("architecture")
    reference_seeds = seeds[seeds["seed"] == REFERENCE_SEED].set_index("architecture")
    summary_diff = {
        arch: {
            key: float(abs(reference_seeds.loc[arch, ours_key] - frozen_summary.loc[arch, key]))
            for ours_key, key in (
                ("clean_macro_f1", "clean_macro_f1"),
                ("mean_corrupted_macro_f1", "mean_corrupted_macro_f1"),
                ("mean_drop", "mean_macro_f1_drop_from_clean"),
                ("worst_macro_f1", "worst_corrupted_macro_f1"),
            )
        }
        for arch in reference_seeds.index
        if arch in frozen_summary.index
    }
    maximum = float(merged["macro_f1_abs_diff"].max())
    reproduced = maximum <= EXACT
    ctx.summary["corruption_reproduction"] = {
        "models_compared": sorted(by_model.index, key=MODEL_ORDER.index),
        "conditions_compared": int(len(merged)),
        "max_abs_diff_macro_f1": maximum,
        "max_abs_diff_macro_f1_by_model": {arch: float(value) for arch, value in by_model.items()},
        "max_abs_diff_by_metric": {metric: float(merged[f"{metric}_abs_diff"].max()) for metric in [*compared, "clean_macro_f1"]},
        "summary_abs_diff": summary_diff,
        "reproduced_within": EXACT,
        "reproduced": bool(reproduced),
        "self_check": self_check,
    }
    if not reproduced:
        print(f"WARNING seed-{REFERENCE_SEED} corruption macro-F1 differs from the frozen aggregate by up to {maximum:.6f}")
    if len(by_model) == len(MODEL_ORDER):
        ctx.done("corruption.seed2026_reproduction")
    else:
        ctx.pending("corruption.seed2026_reproduction", f"seed-{REFERENCE_SEED} predictions for {len(by_model)}/{len(MODEL_ORDER)} models")


def _corruption_self_check(ctx: Context, conditions: pd.DataFrame) -> dict[str, float]:
    """Largest gap between recomputed metrics and the per-run corruption_metrics.csv written by the queue."""
    gaps: dict[str, float] = {}
    for (arch, seed), frame in conditions.groupby(["architecture", "seed"]):
        path = ctx.revision / "corruption" / str(arch) / f"seed-{seed}" / "corruption_metrics.csv"
        if not path.is_file():
            continue
        written = pd.read_csv(path)
        merged = frame.merge(written, on=["corruption", "severity"], suffixes=("", "_written"), validate="one_to_one")
        for metric in CORRUPTION_METRICS:
            if f"{metric}_written" in merged:
                gap = float((merged[metric] - merged[f"{metric}_written"]).abs().max())
                gaps[metric] = max(gaps.get(metric, 0.0), gap)
    return gaps


def _corruption_table(ctx: Context, models: pd.DataFrame, seeds: pd.DataFrame) -> None:
    rows = []
    counts = []
    for row in models.itertuples(index=False):
        frame = seeds[seeds["architecture"] == row.architecture]
        rows.append(
            [
                DISPLAY_NAMES[row.architecture],
                mean_sd(frame["clean_macro_f1"]),
                mean_sd(frame["mean_corrupted_macro_f1"]),
                f"{_signed(row.mean_drop_mean)} {_math_interval(row.mean_drop_ci_low, row.mean_drop_ci_high)}",
                mean_sd(frame["worst_macro_f1"]),
            ]
        )
        counts.append(f"{DISPLAY_NAMES[row.architecture]} {row.seeds_corrupted_at_least_clean}/{row.seed_count}")
    reproduction = ctx.summary.get("corruption_reproduction", {})
    notes = "Seeds with mean corrupted F1 $\\geq$ clean F1: " + ", ".join(counts) + "."
    if isinstance(reproduction, dict) and "max_abs_diff_macro_f1" in reproduction:
        notes += (
            f" Seed {REFERENCE_SEED} reproduces the original single-seed per-condition macro-F1 "
            f"(max.\\ abs.\\ difference {reproduction['max_abs_diff_macro_f1']:.3f})."
        )
    seed_text = "/".join(str(seed) for seed in SEEDS)
    ctx.table(
        "rev_corruption",
        caption=(
            "Corruption robustness of the calibrated three-stage checkpoints on the grouped locked test (831 images), "
            f"seeds {seed_text}. Clean and mean corrupted macro-F1 (6 types $\\times$ 5 severities) and worst single "
            "condition are mean$\\pm$SD over seeds. Drop is clean minus mean corrupted F1, averaged over seeds, with a "
            f"paired 95\\% capture-group bootstrap percentile interval ({ctx.replicates:,} replicates)."
        ),
        headers=("Model", "Clean F1", "Corrupted F1", r"Drop [95\% CI]", "Worst F1"),
        rows=rows,
        notes=notes,
    )


def _corruption_figure(ctx: Context, curves: pd.DataFrame, architectures: Sequence[str]) -> None:
    figure_style()
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(DOUBLE_COLUMN_WIDTH, 3.5), sharex=True, sharey=True)
    for axis, name in zip(axes.ravel(), CORRUPTIONS):
        for arch in architectures:
            rows = curves[(curves["architecture"] == arch) & (curves["corruption"] == name)].sort_values("severity")
            x = rows["severity"].to_numpy(dtype=float)
            mean = rows["macro_f1_mean"].to_numpy(dtype=float)
            spread = rows["macro_f1_sd"].fillna(0.0).to_numpy(dtype=float)
            axis.fill_between(x, mean - spread, mean + spread, color=COLORS[arch], alpha=0.14, linewidth=0)
            axis.plot(
                x, mean, color=COLORS[arch], marker=MARKERS[arch], markersize=3.2, linewidth=1.1, label=DISPLAY_NAMES[arch]
            )
        axis.set_title(CORRUPTION_LABELS[name])
        axis.set_xticks(range(0, len(SEVERITIES) + 1))
        axis.set_xticklabels(["Clean", *[str(severity) for severity in SEVERITIES]])
    for axis in axes[-1]:
        axis.set_xlabel("Severity")
    for axis in axes[:, 0]:
        axis.set_ylabel("Macro-F1")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 1.035))
    figure.tight_layout(rect=(0, 0, 1, 0.96), h_pad=0.6, w_pad=0.6)
    ctx.figures.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = ctx.figures / f"corruption_severity.{suffix}"
        figure.savefig(path, dpi=300)
        ctx.written.append(_display(path))
    plt.close(figure)


def _binary_ranking(ctx: Context) -> pd.DataFrame:
    selection = pd.read_csv(FROZEN / "aggregate/validation_selection.csv").set_index("architecture")
    rank_changes = pd.read_csv(FROZEN / "aggregate/rank_changes.csv").set_index("architecture")
    binary_scores = {arch: [float(_frozen_run_json("binary", arch, seed)["validation_macro_f1"]) for seed in SEEDS] for arch in MODEL_ORDER}
    three_scores = {arch: [float(_frozen_run_json("three_stage", arch, seed)["validation_macro_f1"]) for seed in SEEDS] for arch in MODEL_ORDER}
    binary_ranks = _ranks({arch: float(np.mean(values)) for arch, values in binary_scores.items()})
    three_ranks = _ranks({arch: float(np.mean(values)) for arch, values in three_scores.items()})
    records = [
        {
            "architecture": arch,
            "binary_validation_macro_f1_mean": float(np.mean(binary_scores[arch])),
            "binary_validation_macro_f1_sd": _sd(binary_scores[arch]),
            "binary_validation_rank": binary_ranks[arch],
            "binary_validation_top2": binary_ranks[arch] <= 2,
            "three_stage_validation_macro_f1_mean": float(np.mean(three_scores[arch])),
            "three_stage_validation_rank": three_ranks[arch],
            "original_external_selection": bool(selection.loc[arch, "selected_for_external"]),
            "frozen_binary_rank_agrees": int(rank_changes.loc[arch, "validation_binary_rank"]) == binary_ranks[arch],
        }
        for arch in MODEL_ORDER
    ]
    frame = pd.DataFrame.from_records(records)
    if not frame["frozen_binary_rank_agrees"].all():
        print("WARNING binary validation ranking disagrees with results/experiments/aggregate/rank_changes.csv")
    ctx.csv(frame, "binary_validation_ranking")
    ctx.summary["binary_validation_ranking"] = {
        "order": sorted(MODEL_ORDER, key=binary_ranks.get),
        "protocol_top2": [arch for arch in sorted(MODEL_ORDER, key=binary_ranks.get) if binary_ranks[arch] <= 2],
        "original_external_selection": [arch for arch in MODEL_ORDER if bool(selection.loc[arch, "selected_for_external"])],
        "agrees_with_frozen_rank_changes": bool(frame["frozen_binary_rank_agrees"].all()),
    }
    ctx.done("external.binary_validation_ranking")
    return frame.set_index("architecture")


def _in_domain_reference(ctx: Context, binary: BinaryTestSet) -> pd.DataFrame:
    mask = np.isin(binary.produce, EXTERNAL_FRUITS)
    records = []
    for arch in MODEL_ORDER:
        for index, seed in enumerate(SEEDS):
            predictions = binary.probabilities[arch][index].argmax(-1)
            records.append(
                {
                    "architecture": arch,
                    "seed": seed,
                    "in_domain_banana_orange_macro_f1": macro_f1(binary.targets[mask], predictions[mask], 2),
                    "in_domain_banana_orange_images": int(mask.sum()),
                    "in_domain_banana_orange_rotten_to_fresh_rate": float(
                        np.mean(predictions[mask & (binary.targets == 1)] == 0)
                    ),
                    "in_domain_full_binary_macro_f1": macro_f1(binary.targets, predictions, 2),
                }
            )
    frame = pd.DataFrame.from_records(records)
    ctx.csv(frame, "in_domain_reference")
    ctx.done("external.in_domain_reference")
    return frame


def _external_manifest() -> pd.DataFrame:
    frame = pd.read_csv(EXTERNAL_MANIFEST, usecols=["sample_id", "produce_type", "freshness_label"])
    frame["sample_id"] = frame["sample_id"].astype(str)
    counts = frame.groupby(["produce_type", "freshness_label"]).size()
    if set(frame["produce_type"]) != set(EXTERNAL_FRUITS) or set(frame["freshness_label"]) != {"Fresh", "Rotten"}:
        raise ValueError("External manifest is not the banana/orange Fresh/Rotten set")
    if counts.nunique() != 1:
        raise ValueError(f"External manifest cells are unbalanced: {counts.to_dict()}")
    return frame.set_index("sample_id")


def _read_external(path: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    frame = read_predictions(path, "binary")
    frame["sample_id"] = frame["sample_id"].astype(str)
    if set(frame["sample_id"]) != set(manifest.index) or len(frame) != len(manifest):
        raise ValueError(f"{path} does not cover the {len(manifest)} external manifest images")
    frame = frame.sort_values("sample_id").reset_index(drop=True)
    expected = manifest.loc[frame["sample_id"]]
    if not np.array_equal(expected["freshness_label"].map({"Fresh": 0, "Rotten": 1}).to_numpy(), frame["target_index"].to_numpy(dtype=int)):
        raise ValueError(f"{path} targets disagree with the external manifest")
    frame["produce_type"] = expected["produce_type"].astype(str).to_numpy()
    return frame


@lru_cache(maxsize=1)
def _binary_validation_ids() -> frozenset[str]:
    manifest = pd.read_csv(GROUPED_MANIFEST, usecols=["sample_id", "split", "freshness_label"])
    expected = manifest[(manifest["split"] == "validation") & (manifest["freshness_label"] != "Semi-Fresh")]
    return frozenset(expected["sample_id"].astype(str))


def _read_binary_validation(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = read_predictions(path, "binary")
    if set(frame["sample_id"].astype(str)) != _binary_validation_ids():
        raise ValueError(f"{path} does not cover the grouped binary validation split")
    probabilities = frame[probability_columns("binary")].to_numpy(dtype=float)
    return frame["target_index"].to_numpy(dtype=int), probabilities / probabilities.sum(axis=1, keepdims=True)


def _external_part(ctx: Context, binary: BinaryTestSet) -> None:
    ranking = _binary_ranking(ctx)
    in_domain = _in_domain_reference(ctx, binary).set_index(["architecture", "seed"])
    manifest = _external_manifest()
    seed_records = []
    fruit_records = []
    missing_external: list[Path] = []
    missing_validation: list[Path] = []
    for arch in MODEL_ORDER:
        for seed in SEEDS:
            path = ctx.revision / "external" / arch / f"seed-{seed}" / "external_predictions.csv"
            if not path.is_file():
                missing_external.append(path)
                continue
            frame = _read_external(path, manifest)
            targets = frame["target_index"].to_numpy(dtype=int)
            probabilities = frame[probability_columns("binary")].to_numpy(dtype=float)
            probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
            predictions = probabilities.argmax(axis=1)
            values = metrics(targets, probabilities, "binary")
            record: dict[str, object] = {"architecture": arch, "seed": seed, "images": len(frame)}
            record |= {metric: float(values[metric]) for metric in EXTERNAL_METRICS if metric in values}
            record["rotten_to_fresh_rate"] = float(np.mean(predictions[targets == 1] == 0))
            record["fresh_to_rotten_rate"] = float(np.mean(predictions[targets == 0] == 1))
            for fruit in EXTERNAL_FRUITS:
                mask = frame["produce_type"].to_numpy() == fruit
                fruit_values = metrics(targets[mask], probabilities[mask], "binary")
                record[f"{fruit}_macro_f1"] = float(fruit_values["macro_f1"])
                fruit_records.append(
                    {
                        "architecture": arch,
                        "seed": seed,
                        "produce_type": fruit,
                        "images": int(mask.sum()),
                        **{metric: float(fruit_values[metric]) for metric in EXTERNAL_METRICS if metric in fruit_values},
                        "rotten_to_fresh_rate": float(np.mean(predictions[mask & (targets == 1)] == 0)),
                        "fresh_to_rotten_rate": float(np.mean(predictions[mask & (targets == 0)] == 1)),
                    }
                )
            reference = in_domain.loc[(arch, seed)]
            record["in_domain_banana_orange_macro_f1"] = float(reference["in_domain_banana_orange_macro_f1"])
            record["in_domain_to_external_change"] = record["macro_f1"] - record["in_domain_banana_orange_macro_f1"]
            validation_path = (
                ctx.revision / "predictions/grouped/binary" / arch / f"seed-{seed}" / "uncalibrated" / "validation_predictions.csv"
            )
            if validation_path.is_file():
                validation_targets, validation_probabilities = _read_binary_validation(validation_path)
                temperature = fit_temperature(validation_probabilities, validation_targets)
                scaled_validation = apply_temperature(validation_probabilities, temperature)
                scaled = metrics(targets, apply_temperature(probabilities, temperature), "binary")
                record |= {
                    "temperature": temperature,
                    "validation_ece": ece(validation_targets, validation_probabilities),
                    "validation_ece_temperature": ece(validation_targets, scaled_validation),
                    "ece_temperature": float(scaled["ece"]),
                    "nll_temperature": float(scaled["nll"]),
                    "brier_temperature": float(scaled["brier"]),
                }
            else:
                missing_validation.append(validation_path)
            seed_records.append(record)

    if not seed_records:
        for part in ("metrics", "temperature_scaling", "sanity_check_seed2026", "table_rev_external"):
            ctx.pending(f"external.{part}", missing_external)
        return
    seeds = pd.DataFrame.from_records(seed_records)
    ctx.csv(seeds, "external_seed_metrics")
    ctx.csv(pd.DataFrame.from_records(fruit_records), "external_per_fruit_seed_metrics")
    if missing_external:
        ctx.pending("external.metrics", missing_external)
    else:
        ctx.done("external.metrics")
    if missing_validation or missing_external:
        ctx.pending("external.temperature_scaling", [*missing_validation, *missing_external])
    else:
        ctx.done("external.temperature_scaling")
    _external_sanity_check(ctx, seeds, pd.DataFrame.from_records(fruit_records))

    complete = [arch for arch in MODEL_ORDER if (seeds["architecture"] == arch).sum() == len(SEEDS)]
    usable = [arch for arch in MODEL_ORDER if (seeds["architecture"] == arch).any() and (ctx.allow_partial or arch in complete)]
    summary_columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "ece",
        "ece_temperature",
        "nll",
        "nll_temperature",
        "temperature",
        "rotten_to_fresh_rate",
        "fresh_to_rotten_rate",
        *[f"{fruit}_macro_f1" for fruit in EXTERNAL_FRUITS],
        "in_domain_banana_orange_macro_f1",
        "in_domain_to_external_change",
    ]
    model_records = []
    for arch in usable:
        frame = seeds[seeds["architecture"] == arch]
        record: dict[str, object] = {"architecture": arch, "seed_count": len(frame)}
        for column in summary_columns:
            values = frame[column].dropna() if column in frame else pd.Series(dtype=float)
            record[f"{column}_mean"] = float(values.mean()) if len(values) else float("nan")
            record[f"{column}_sd"] = _sd(values) if len(values) else float("nan")
            record[f"{column}_seed_count"] = int(len(values))
        record |= {
            "binary_validation_macro_f1_mean": float(ranking.loc[arch, "binary_validation_macro_f1_mean"]),
            "binary_validation_rank": int(ranking.loc[arch, "binary_validation_rank"]),
            "binary_validation_top2": bool(ranking.loc[arch, "binary_validation_top2"]),
            "original_external_selection": bool(ranking.loc[arch, "original_external_selection"]),
        }
        model_records.append(record)
    models = pd.DataFrame.from_records(model_records)
    if not models.empty:
        ctx.csv(models, "external_model_summary")
        ctx.summary["external"] = {
            row["architecture"]: {
                key: row[key]
                for key in (
                    "seed_count",
                    "macro_f1_mean",
                    "macro_f1_sd",
                    "accuracy_mean",
                    "balanced_accuracy_mean",
                    "ece_mean",
                    "ece_temperature_mean",
                    "rotten_to_fresh_rate_mean",
                    "banana_macro_f1_mean",
                    "orange_macro_f1_mean",
                    "in_domain_banana_orange_macro_f1_mean",
                    "in_domain_to_external_change_mean",
                    "binary_validation_rank",
                    "binary_validation_top2",
                )
            }
            for row in model_records
        }
    if (len(complete) == len(MODEL_ORDER) or ctx.allow_partial) and not models.empty:
        _external_table(ctx, models, seeds)
    if len(complete) == len(MODEL_ORDER):
        ctx.done("external.table_rev_external")
    else:
        ctx.pending("external.table_rev_external", missing_external)


def _external_sanity_check(ctx: Context, seeds: pd.DataFrame, fruits: pd.DataFrame) -> None:
    frozen = pd.read_csv(FROZEN / "aggregate/external_metrics.csv")
    records = []
    for row in frozen.itertuples(index=False):
        if int(row.seed) != REFERENCE_SEED:
            continue
        if row.scope == "aggregate":
            ours = seeds[(seeds["architecture"] == row.architecture) & (seeds["seed"] == REFERENCE_SEED)]
        else:
            ours = fruits[
                (fruits["architecture"] == row.architecture) & (fruits["seed"] == REFERENCE_SEED) & (fruits["produce_type"] == row.produce_type)
            ]
        if ours.empty:
            continue
        for metric in EXTERNAL_METRICS:
            if metric in ours and hasattr(row, metric):
                revision_value = float(ours.iloc[0][metric])
                frozen_value = float(getattr(row, metric))
                records.append(
                    {
                        "architecture": row.architecture,
                        "seed": REFERENCE_SEED,
                        "scope": row.scope,
                        "produce_type": row.produce_type,
                        "metric": metric,
                        "revision_value": revision_value,
                        "frozen_value": frozen_value,
                        "abs_diff": abs(revision_value - frozen_value),
                    }
                )
    expected = frozen[(frozen["seed"] == REFERENCE_SEED)][["architecture", "scope", "produce_type"]].drop_duplicates()
    if not records:
        ctx.pending("external.sanity_check_seed2026", f"seed-{REFERENCE_SEED} external predictions for {sorted(expected['architecture'].unique())}")
        return
    frame = pd.DataFrame.from_records(records)
    ctx.csv(frame, "external_sanity_check")
    f1 = frame[frame["metric"] == "macro_f1"]
    maximum = float(f1["abs_diff"].max())
    ctx.summary["external_sanity_check"] = {
        "aggregate_macro_f1": {
            row.architecture: {"revision": row.revision_value, "frozen": row.frozen_value}
            for row in f1[f1["scope"] == "aggregate"].itertuples(index=False)
        },
        "max_abs_diff_macro_f1": maximum,
        "max_abs_diff_any_metric": float(frame["abs_diff"].max()),
        "reproduced": bool(maximum <= EXACT),
    }
    if maximum > EXACT:
        print(f"WARNING external seed-{REFERENCE_SEED} macro-F1 differs from the frozen aggregate by up to {maximum:.6f}")
    compared = set(zip(frame["architecture"], frame["scope"], frame["produce_type"]))
    if len(compared) == len(expected):
        ctx.done("external.sanity_check_seed2026")
    else:
        ctx.pending("external.sanity_check_seed2026", f"{len(compared)}/{len(expected)} frozen external rows comparable")


def _external_table(ctx: Context, models: pd.DataFrame, seeds: pd.DataFrame) -> None:
    rows = []
    for row in models.itertuples(index=False):
        frame = seeds[seeds["architecture"] == row.architecture]
        rank = f"{row.binary_validation_rank}" + (r"$^{\dagger}$" if row.binary_validation_top2 else "")
        after = f"{row.ece_temperature_mean:.3f}" if np.isfinite(row.ece_temperature_mean) else "--"
        rows.append(
            [
                DISPLAY_NAMES[row.architecture],
                rank,
                mean_sd(frame["in_domain_banana_orange_macro_f1"]),
                mean_sd(frame["macro_f1"]),
                f"{row.ece_mean:.3f}/{after}",
                mean_sd(100.0 * frame["rotten_to_fresh_rate"], digits=1),
            ]
        )
    seed_text = "/".join(str(seed) for seed in SEEDS)
    ctx.table(
        "rev_external",
        caption=(
            "External binary validation on 800 original-only banana/orange images (200 per fruit and class) without "
            f"fine-tuning: uncalibrated grouped binary checkpoints, mean$\\pm$SD over seeds {seed_text}. Rank: mean "
            "grouped binary validation macro-F1. In-domain: banana/orange images of the grouped binary test. ECE "
            "(15 bins, seed mean) before/after temperature scaling fitted on each seed's binary validation predictions. "
            "R$\\to$F: Rotten images predicted Fresh."
        ),
        headers=("Model", "Rank", r"In-domain F1", r"External F1", r"ECE raw/$T$", r"R$\to$F (\%)"),
        rows=rows,
        notes=(
            r"$^{\dagger}$Protocol-consistent top-2 by binary validation; the original submission evaluated the "
            "three-stage validation top-2 (ConvNeXt-Tiny, ResNet50). Temperature scaling leaves macro-F1 unchanged."
        ),
    )


def _frozen_run(arch: str, seed: int, test: TestSet) -> RunResult:
    payload = _frozen_run_json("three_stage", arch, seed)
    predictions = test.clean[arch][SEEDS.index(seed)].argmax(-1)
    return RunResult(seed, float(payload["validation_macro_f1"]), macro_f1(test.targets, predictions, 3), int(payload["best_epoch"]), predictions)


def _revision_run(run_dir: Path, test: TestSet) -> RunResult | None:
    run_path, prediction_path = run_dir / "run.json", run_dir / "test_predictions.csv"
    if not (run_path.is_file() and prediction_path.is_file()):
        return None
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    order, targets, probabilities = stack_seeds([read_predictions(prediction_path, "three_stage")], "three_stage")
    if order != test.order or not np.array_equal(targets, test.targets):
        raise ValueError(f"{prediction_path} does not cover the grouped locked test")
    predictions = probabilities[0].argmax(-1)
    return RunResult(
        int(payload["seed"]), float(payload["validation_macro_f1"]), macro_f1(targets, predictions, 3), int(payload["best_epoch"]), predictions
    )


def _run_record(arch: str, setting: str, value: float, result: RunResult, source: str) -> dict[str, object]:
    return {
        "architecture": arch,
        "setting": setting,
        "value": value,
        "seed": result.seed,
        "validation_macro_f1": result.validation_macro_f1,
        "test_macro_f1": result.test_macro_f1,
        "best_epoch": result.best_epoch,
        "source": source,
    }


def _summarise_runs(prefix: str, results: Sequence[RunResult]) -> dict[str, object]:
    validation = [result.validation_macro_f1 for result in results]
    tests = [result.test_macro_f1 for result in results]
    return {
        f"{prefix}_seeds": ";".join(str(result.seed) for result in results),
        f"{prefix}_seed_count": len(results),
        f"{prefix}_validation_macro_f1_mean": float(np.mean(validation)),
        f"{prefix}_validation_macro_f1_sd": _sd(validation),
        f"{prefix}_test_macro_f1_mean": float(np.mean(tests)),
        f"{prefix}_test_macro_f1_sd": _sd(tests),
    }


def _lr_part(ctx: Context, test: TestSet) -> dict[str, object]:
    shared = {arch: [_frozen_run(arch, seed, test) for seed in SEEDS] for arch in MODEL_ORDER}
    grid: dict[str, dict[str, RunResult]] = {arch: {} for arch in MODEL_ORDER}
    records = []
    missing: list[Path] = []
    for arch in MODEL_ORDER:
        for name, rate in LEARNING_RATES.items():
            if name == "base":
                result = shared[arch][SEEDS.index(REFERENCE_SEED)]
            else:
                run_dir = ctx.revision / "runs/lr" / name / arch / f"seed-{REFERENCE_SEED}"
                loaded = _revision_run(run_dir, test)
                if loaded is None:
                    missing.append(run_dir)
                    continue
                result = loaded
            grid[arch][name] = result
            records.append(_run_record(arch, name, rate, result, "frozen" if name == "base" else "revision"))
    outcome: dict[str, object] = {"shared": shared, "selected": {}, "tuned": {}}
    if not missing:
        ctx.done("lr.seed2026_grid")
    else:
        ctx.pending("lr.seed2026_grid", missing)
    if all(record["source"] == "frozen" for record in records):
        for part in ("selection", "follow_up_seeds", "ranking"):
            ctx.pending(f"lr.{part}", missing)
        return outcome
    grid_frame = pd.DataFrame.from_records(records)
    grid_frame["model_rank"] = grid_frame["architecture"].map(MODEL_ORDER.index)
    grid_frame = grid_frame.sort_values(["model_rank", "value"]).drop(columns=["model_rank"])
    ctx.csv(grid_frame, "lr_seed2026")

    selected: dict[str, str] = {}
    tuned: dict[str, list[RunResult]] = {}
    follow_missing: list[Path] = []
    selection_records = []
    for arch in MODEL_ORDER:
        if len(grid[arch]) != len(LEARNING_RATES):
            continue
        choice = max(grid[arch], key=lambda name: (grid[arch][name].validation_macro_f1, name == "base"))
        test_choice = max(grid[arch], key=lambda name: (grid[arch][name].test_macro_f1, name == "base"))
        selected[arch] = choice
        if choice == "base":
            results = list(shared[arch])
        else:
            results = [grid[arch][choice]]
            for seed in SEEDS[1:]:
                run_dir = ctx.revision / "runs/lr" / choice / arch / f"seed-{seed}"
                loaded = _revision_run(run_dir, test)
                if loaded is None:
                    follow_missing.append(run_dir)
                else:
                    results.append(loaded)
        record: dict[str, object] = {"architecture": arch}
        for name in LEARNING_RATES:
            record[f"{name}_validation_macro_f1_seed{REFERENCE_SEED}"] = grid[arch][name].validation_macro_f1
            record[f"{name}_test_macro_f1_seed{REFERENCE_SEED}"] = grid[arch][name].test_macro_f1
        record |= {
            "selected_setting": choice,
            "selected_learning_rate": LEARNING_RATES[choice],
            "test_best_setting_seed2026_diagnostic": test_choice,
            **_summarise_runs("shared", shared[arch]),
        }
        if len(results) == len(SEEDS) or ctx.allow_partial:
            tuned[arch] = results
            record |= _summarise_runs("tuned", results)
            record["tuned_minus_shared_test_macro_f1"] = record["tuned_test_macro_f1_mean"] - record["shared_test_macro_f1_mean"]
            if choice == "base":
                record["tuned_minus_shared_ci_low"] = record["tuned_minus_shared_ci_high"] = 0.0
            else:
                low, high = test.bootstrap.mean_difference(
                    test.targets, [result.predictions for result in results], [result.predictions for result in shared[arch]]
                )
                record["tuned_minus_shared_ci_low"], record["tuned_minus_shared_ci_high"] = low, high
        selection_records.append(record)
    outcome["selected"], outcome["tuned"] = selected, tuned
    if selection_records:
        ctx.csv(pd.DataFrame.from_records(selection_records), "lr_selection")
    if len(selected) == len(MODEL_ORDER):
        ctx.done("lr.selection")
    else:
        ctx.pending("lr.selection", missing)
    if len(selected) < len(MODEL_ORDER):
        ctx.pending("lr.follow_up_seeds", "learning-rate selection incomplete")
    elif follow_missing:
        ctx.pending("lr.follow_up_seeds", follow_missing)
    else:
        ctx.done("lr.follow_up_seeds")

    complete = len(selected) == len(MODEL_ORDER) and not follow_missing
    if complete or (ctx.allow_partial and len(tuned) == len(MODEL_ORDER)):
        shared_test = {arch: float(np.mean([result.test_macro_f1 for result in shared[arch]])) for arch in MODEL_ORDER}
        tuned_test = {arch: float(np.mean([result.test_macro_f1 for result in tuned[arch]])) for arch in MODEL_ORDER}
        shared_validation = {arch: float(np.mean([result.validation_macro_f1 for result in shared[arch]])) for arch in MODEL_ORDER}
        tuned_validation = {arch: float(np.mean([result.validation_macro_f1 for result in tuned[arch]])) for arch in MODEL_ORDER}
        ranks = {name: _ranks(scores) for name, scores in (
            ("shared_test", shared_test), ("tuned_test", tuned_test), ("shared_validation", shared_validation), ("tuned_validation", tuned_validation)
        )}
        ranking = pd.DataFrame.from_records(
            [
                {
                    "architecture": arch,
                    "selected_setting": selected[arch],
                    "shared_test_macro_f1_mean": shared_test[arch],
                    "shared_test_rank": ranks["shared_test"][arch],
                    "tuned_test_macro_f1_mean": tuned_test[arch],
                    "tuned_test_rank": ranks["tuned_test"][arch],
                    "shared_validation_macro_f1_mean": shared_validation[arch],
                    "shared_validation_rank": ranks["shared_validation"][arch],
                    "tuned_validation_macro_f1_mean": tuned_validation[arch],
                    "tuned_validation_rank": ranks["tuned_validation"][arch],
                }
                for arch in MODEL_ORDER
            ]
        )
        ctx.csv(ranking, "lr_ranking")
        tau_test = kendalltau([ranks["shared_test"][arch] for arch in MODEL_ORDER], [ranks["tuned_test"][arch] for arch in MODEL_ORDER])
        tau_validation = kendalltau(
            [ranks["shared_validation"][arch] for arch in MODEL_ORDER], [ranks["tuned_validation"][arch] for arch in MODEL_ORDER]
        )
        ctx.summary["lr"] = {
            "selected": selected,
            "shared_test_order": sorted(MODEL_ORDER, key=ranks["shared_test"].get),
            "tuned_test_order": sorted(MODEL_ORDER, key=ranks["tuned_test"].get),
            "shared_validation_order": sorted(MODEL_ORDER, key=ranks["shared_validation"].get),
            "tuned_validation_order": sorted(MODEL_ORDER, key=ranks["tuned_validation"].get),
            "kendall_tau_test_ranks": float(tau_test.statistic),
            "kendall_tau_validation_ranks": float(tau_validation.statistic),
            "tuned_test_macro_f1_mean": tuned_test,
            "shared_test_macro_f1_mean": shared_test,
        }
    if complete:
        ctx.done("lr.ranking")
    else:
        ctx.pending("lr.ranking", follow_missing or missing or "learning-rate selection incomplete")
    if selected and "lr" not in ctx.summary:
        ctx.summary["lr"] = {"selected": selected}
    return outcome


def _resolution_part(ctx: Context, test: TestSet) -> dict[tuple[str, str], list[RunResult]]:
    expected = {arch: SEEDS if arch in ABLATION_MODELS else (REFERENCE_SEED,) for arch in MODEL_ORDER}
    results: dict[tuple[str, str], list[RunResult]] = {}
    records = []
    missing_reference: list[Path] = []
    missing_follow: list[Path] = []
    for arch in MODEL_ORDER:
        for name, size in RESOLUTIONS.items():
            found = []
            for seed in SEEDS if name == "base" else expected[arch]:
                if name == "base":
                    result = _frozen_run(arch, seed, test)
                else:
                    run_dir = ctx.revision / "runs/res" / name / arch / f"seed-{seed}"
                    loaded = _revision_run(run_dir, test)
                    if loaded is None:
                        (missing_reference if seed == REFERENCE_SEED else missing_follow).append(run_dir)
                        continue
                    result = loaded
                found.append(result)
                records.append(_run_record(arch, name, size, result, "frozen" if name == "base" else "revision"))
            results[(arch, name)] = found
    if missing_reference:
        ctx.pending("resolution.seed2026", missing_reference)
    else:
        ctx.done("resolution.seed2026")
    if missing_follow:
        ctx.pending("resolution.three_seed_ablation_models", missing_follow)
    else:
        ctx.done("resolution.three_seed_ablation_models")
    if all(record["source"] == "frozen" for record in records):
        return {}
    ctx.csv(pd.DataFrame.from_records(records), "resolution_seed_metrics")
    summary_records = []
    for arch in MODEL_ORDER:
        base = {result.seed: result for result in results[(arch, "base")]}
        for name, size in RESOLUTIONS.items():
            found = results[(arch, name)]
            wanted = SEEDS if name == "base" else expected[arch]
            record: dict[str, object] = {"architecture": arch, "setting": name, "image_size": size, "expected_seeds": len(wanted)}
            reference = next((result for result in found if result.seed == REFERENCE_SEED), None)
            record["validation_macro_f1_seed2026"] = reference.validation_macro_f1 if reference else float("nan")
            record["test_macro_f1_seed2026"] = reference.test_macro_f1 if reference else float("nan")
            if found and (len(found) == len(wanted) or ctx.allow_partial):
                record |= _summarise_runs("runs", found)
                if name != "base":
                    paired = [base[result.seed] for result in found]
                    record["test_minus_224_macro_f1"] = record["runs_test_macro_f1_mean"] - float(
                        np.mean([result.test_macro_f1 for result in paired])
                    )
                    low, high = test.bootstrap.mean_difference(
                        test.targets, [result.predictions for result in found], [result.predictions for result in paired]
                    )
                    record["test_minus_224_ci_low"], record["test_minus_224_ci_high"] = low, high
            summary_records.append(record)
    ctx.csv(pd.DataFrame.from_records(summary_records), "resolution_summary")
    ctx.summary["resolution"] = {
        f"{record['architecture']}/{record['image_size']}": {
            key: record[key] for key in ("runs_test_macro_f1_mean", "runs_test_macro_f1_sd", "runs_seed_count", "test_macro_f1_seed2026") if key in record
        }
        for record in summary_records
    }
    return results


def _scratch_part(ctx: Context, test: TestSet) -> dict[str, object] | None:
    arch = "mobilenet_v2"
    scratch = []
    missing: list[Path] = []
    for seed in SEEDS:
        run_dir = ctx.revision / "runs/scratch/three_stage" / arch / f"seed-{seed}"
        loaded = _revision_run(run_dir, test)
        if loaded is None:
            missing.append(run_dir)
        else:
            scratch.append(loaded)
    if missing:
        ctx.pending("scratch.three_seed", missing)
    else:
        ctx.done("scratch.three_seed")
    if not scratch or (missing and not ctx.allow_partial):
        return None
    pretrained = {seed: _frozen_run(arch, seed, test) for seed in SEEDS}
    seed_frame = pd.DataFrame.from_records(
        [
            {
                "architecture": arch,
                "seed": result.seed,
                "scratch_validation_macro_f1": result.validation_macro_f1,
                "scratch_test_macro_f1": result.test_macro_f1,
                "scratch_best_epoch": result.best_epoch,
                "pretrained_validation_macro_f1": pretrained[result.seed].validation_macro_f1,
                "pretrained_test_macro_f1": pretrained[result.seed].test_macro_f1,
                "pretrained_minus_scratch_test_macro_f1": pretrained[result.seed].test_macro_f1 - result.test_macro_f1,
            }
            for result in scratch
        ]
    )
    ctx.csv(seed_frame, "scratch_seed_metrics")
    paired = [pretrained[result.seed] for result in scratch]
    low, high = test.bootstrap.mean_difference(
        test.targets, [result.predictions for result in paired], [result.predictions for result in scratch]
    )
    summary = {
        "architecture": arch,
        **_summarise_runs("scratch", scratch),
        **_summarise_runs("pretrained", paired),
        "pretrained_minus_scratch_test_macro_f1": float(seed_frame["pretrained_minus_scratch_test_macro_f1"].mean()),
        "pretrained_minus_scratch_test_macro_f1_sd": _sd(seed_frame["pretrained_minus_scratch_test_macro_f1"]),
        "pretrained_minus_scratch_ci_low": low,
        "pretrained_minus_scratch_ci_high": high,
    }
    ctx.csv(pd.DataFrame.from_records([summary]), "scratch_summary")
    ctx.summary["scratch"] = summary
    return summary


def _fairness_table(
    ctx: Context,
    test: TestSet,
    lr: dict[str, object],
    resolution: dict[tuple[str, str], list[RunResult]],
    scratch: dict[str, object] | None,
) -> None:
    required = (
        "lr.selection",
        "lr.follow_up_seeds",
        "resolution.seed2026",
        "resolution.three_seed_ablation_models",
        "scratch.three_seed",
    )
    waiting = [part for part in required if ctx.status.get(part) != "done"]
    if waiting and not ctx.allow_partial:
        ctx.pending("fairness.table_rev_fairness", "waiting for " + ", ".join(waiting))
        return
    shared = lr["shared"]
    selected = lr["selected"]
    tuned = lr["tuned"]
    assert isinstance(shared, dict) and isinstance(selected, dict) and isinstance(tuned, dict)
    expected = {arch: SEEDS if arch in ABLATION_MODELS else (REFERENCE_SEED,) for arch in MODEL_ORDER}
    rows = []
    records = []
    for arch in MODEL_ORDER:
        shared_scores = [result.test_macro_f1 for result in shared[arch]]
        tuned_scores = [result.test_macro_f1 for result in tuned.get(arch, [])]
        cells = [DISPLAY_NAMES[arch], mean_sd(shared_scores)]
        cells.append(LEARNING_RATE_LABELS[selected[arch]] if arch in selected else "--")
        cells.append(mean_sd(tuned_scores) if tuned_scores else "--")
        record: dict[str, object] = {
            "architecture": arch,
            "shared_test_macro_f1_mean": float(np.mean(shared_scores)),
            "shared_test_macro_f1_sd": _sd(shared_scores),
            "selected_setting": selected.get(arch, ""),
            "tuned_seed_count": len(tuned_scores),
            "tuned_test_macro_f1_mean": float(np.mean(tuned_scores)) if tuned_scores else float("nan"),
            "tuned_test_macro_f1_sd": _sd(tuned_scores) if tuned_scores else float("nan"),
        }
        for name in ("res_160", "res_320"):
            found = resolution.get((arch, name), [])
            scores = [result.test_macro_f1 for result in found]
            usable = scores and (len(scores) == len(expected[arch]) or ctx.allow_partial)
            cells.append(mean_sd(scores) if usable else "--")
            record[f"{name}_seed_count"] = len(scores)
            record[f"{name}_test_macro_f1_mean"] = float(np.mean(scores)) if usable else float("nan")
            record[f"{name}_test_macro_f1_sd"] = _sd(scores) if usable else float("nan")
        rows.append(cells)
        records.append(record)
    ctx.csv(pd.DataFrame.from_records(records), "fairness_summary")
    seed_text = "/".join(str(seed) for seed in SEEDS)
    if scratch is not None:
        difference = _signed(float(scratch["pretrained_minus_scratch_test_macro_f1"]))
        bounds = _math_interval(float(scratch["pretrained_minus_scratch_ci_low"]), float(scratch["pretrained_minus_scratch_ci_high"]))
        scratch_note = (
            f"ImageNet-free MobileNetV2 (same protocol, seeds {scratch['scratch_seeds'].replace(';', '/')}): "
            f"{_mean_sd_from(scratch, 'scratch')} versus {_mean_sd_from(scratch, 'pretrained')} pretrained; "
            f"pretrained minus scratch {difference} {bounds} (paired 95\\% capture-group bootstrap)."
        )
    else:
        scratch_note = "ImageNet-free MobileNetV2: pending."
    notes = (
        "Shared: frozen protocol ($3{\\times}10^{-4}$, 224 px, ImageNet initialization). LR: learning rate selected on "
        f"seed-{REFERENCE_SEED} validation macro-F1 among $10^{{-4}}$, $3{{\\times}}10^{{-4}}$, $10^{{-3}}$; Tuned is the "
        "test macro-F1 at that rate. 160/320 px: other settings as shared. " + scratch_note
    )
    if waiting:
        notes += " -- = pending revision runs."
    ctx.table(
        "rev_fairness",
        caption=(
            "Protocol-fairness sensitivity: grouped three-stage locked-test macro-F1 (831 images). Entries with $\\pm$ are "
            f"mean$\\pm$SD over seeds {seed_text}; single values are seed {REFERENCE_SEED} only."
        ),
        headers=("Model", "Shared", "LR", "Tuned", "160 px", "320 px"),
        rows=rows,
        notes=notes,
    )
    if waiting:
        ctx.pending("fairness.table_rev_fairness", "partial table written; waiting for " + ", ".join(waiting))
    else:
        ctx.done("fairness.table_rev_fairness")


def _mean_sd_from(summary: dict[str, object], prefix: str) -> str:
    mean = float(summary[f"{prefix}_test_macro_f1_mean"])
    spread = float(summary[f"{prefix}_test_macro_f1_sd"])
    if not np.isfinite(spread):
        return f"{mean:.3f}"
    return rf"${mean:.3f}\pm{spread:.3f}$"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--revision-root", type=Path, default=REVISION, help="revision queue outputs")
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS / TOPIC)
    parser.add_argument("--tables-dir", type=Path, default=PAPER_TABLES)
    parser.add_argument("--figures-dir", type=Path, default=PAPER_FIGURES)
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    parser.add_argument(
        "--allow-partial", action="store_true", help="also write tables and figures from incomplete seed or model sets"
    )
    args = parser.parse_args()

    ctx = Context(args.revision_root, args.output_dir, args.tables_dir, args.figures_dir, args.replicates, args.allow_partial)
    test, binary = _references(ctx)
    _clean_reference(ctx, test, binary)
    if "corruption" in args.parts:
        _corruption_part(ctx, test)
    if "external" in args.parts:
        _external_part(ctx, binary)
    lr = _lr_part(ctx, test) if "lr" in args.parts else None
    resolution = _resolution_part(ctx, test) if "resolution" in args.parts else None
    scratch = _scratch_part(ctx, test) if "scratch" in args.parts else None
    if lr is not None and resolution is not None and "scratch" in args.parts:
        _fairness_table(ctx, test, lr, resolution, scratch)

    status_path = ctx.output / "status.json"
    summary_path = ctx.output / "summary.json"
    parts, outputs, summary = dict(ctx.status), set(ctx.written), dict(ctx.summary)
    if len(args.parts) < len(PARTS):
        if status_path.is_file():
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            parts = previous.get("parts", {}) | parts
            outputs |= set(previous.get("outputs", []))
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8")) | summary
    outputs |= {_display(summary_path), _display(status_path)}
    write_json(summary, summary_path)
    write_json(
        {
            "topic": TOPIC,
            "revision_root": _display(ctx.revision),
            "bootstrap": {"replicates": ctx.replicates, "seed": BOOTSTRAP_SEED, "unit": "capture_group_id"},
            "allow_partial": ctx.allow_partial,
            "parts": parts,
            "outputs": sorted(outputs),
        },
        status_path,
    )
    done = sum(value == "done" for value in parts.values())
    print(f"{done}/{len(parts)} sensitivity parts done -> {_display(status_path)}")
    for part, value in parts.items():
        print(f"  {part}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
