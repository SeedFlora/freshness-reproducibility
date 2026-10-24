"""Statistical and operational evidence from the frozen grouped test predictions.

Parts: (1) full three-stage and binary metric table with capture-group cluster
bootstrap intervals; (2) paired bootstrap model comparisons and ranking
probabilities; (3) confusion matrices; (4) critical-miss and false-alarm rates;
(5) risk-coverage analysis of calibrated confidence; (6) abstention and
Fresh-gate thresholds chosen on grouped validation and applied to the locked
test; (7) per-produce macro-F1 for all five backbones. Recomputed values are
checked against the frozen aggregates. Parts whose queue inputs are missing
are reported as pending and filled in by a later rerun.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

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
    ROOT,
    SEEDS,
    SINGLE_COLUMN_WIDTH,
    THREE_STAGE,
    capture_groups,
    class_names,
    cluster_bootstrap,
    figure_style,
    frozen_test_predictions,
    interval,
    latex_table,
    mean_sd,
    metrics,
    percentile_interval,
    revision_validation_predictions,
    seed_mean_macro_f1,
    stack_seeds,
    write_json,
)


TOPIC = "operational"
TASKS = ("three_stage", "binary")
CALIBRATION_STATE = {"three_stage": "calibrated", "binary": "uncalibrated"}
PROTOCOL_SELECTED = "convnext_tiny"
HEADLINE_MACRO_F1 = {
    "three_stage": {"mobilenet_v2": 0.627, "resnet50": 0.690, "densenet121": 0.561, "efficientnet_b0": 0.643, "convnext_tiny": 0.645},
    "binary": {"mobilenet_v2": 0.813, "resnet50": 0.897, "densenet121": 0.899, "efficientnet_b0": 0.797, "convnext_tiny": 0.862},
}
SCALAR_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall", "nll", "brier", "ece", "auroc")
FROZEN_SEED_METRICS = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "macro_f1": "macro_f1", "nll": "nll", "brier": "brier", "ece": "ece"}
ERROR_RATES = {
    "three_stage": {
        "rotten_to_fresh": ("Rotten", ("Fresh",)),
        "rotten_missed": ("Rotten", ("Fresh", "Semi-Fresh")),
        "fresh_to_rotten": ("Fresh", ("Rotten",)),
        "semi_fresh_to_fresh": ("Semi-Fresh", ("Fresh",)),
    },
    "binary": {
        "rotten_to_fresh": ("Rotten", ("Fresh",)),
        "fresh_to_rotten": ("Fresh", ("Rotten",)),
    },
}
COVERAGE_GRID = np.round(np.arange(1, 101) / 100.0, 2)
REPORTED_COVERAGES = (0.5, 0.8)
MIN_ACCEPTED_ROTTEN = 10
ABSTENTION_TARGET = 0.85
ABSTENTION_SENSITIVITY = (0.80, 0.85, 0.90, 0.95)
GATE_LIMIT = 0.02
GATE_SENSITIVITY = (0.0, 0.005, 0.01, 0.02, 0.05)
LINE_STYLES = ("-", "--", "-.", ":", (0, (5, 1.5, 1, 1.5, 1, 1.5)))


@dataclass
class TaskData:
    """Seed-stacked predictions of the five backbones on one evaluation set."""

    task: str
    sample_ids: list[str]
    targets: np.ndarray
    probabilities: np.ndarray
    produce: np.ndarray
    groups: np.ndarray

    @property
    def predictions(self) -> np.ndarray:
        return self.probabilities.argmax(axis=-1)

    @property
    def class_count(self) -> int:
        return self.probabilities.shape[-1]


def load_test(task: str, manifest: Path) -> TaskData:
    order: list[str] | None = None
    targets: np.ndarray | None = None
    produce: np.ndarray | None = None
    stacked = []
    for arch in MODEL_ORDER:
        frames = [frozen_test_predictions(task, arch, seed) for seed in SEEDS]
        sample_ids, arch_targets, probabilities = stack_seeds(frames, task)
        if order is None:
            order, targets = sample_ids, arch_targets
            indexed = frames[0].set_index(frames[0]["sample_id"].astype(str))
            produce = indexed.loc[order, "produce_type"].astype(str).to_numpy()
        elif sample_ids != order or not np.array_equal(arch_targets, targets):
            raise ValueError(f"{task}/{arch} test predictions cover different samples or targets")
        stacked.append(probabilities)
    assert order is not None and targets is not None and produce is not None
    return TaskData(task, order, targets, np.stack(stacked), produce, capture_groups(order, manifest=manifest))


def expected_validation_ids(task: str, manifest: Path) -> set[str]:
    frame = pd.read_csv(manifest, usecols=["sample_id", "freshness_label", "split"])
    frame = frame[frame["split"] == "validation"]
    if task == "binary":
        frame = frame[frame["freshness_label"] != "Semi-Fresh"]
    return set(frame["sample_id"].astype(str))


def load_validation(task: str, manifest: Path) -> tuple[dict[tuple[str, int], tuple[np.ndarray, np.ndarray]], list[str]]:
    """Validation predictions per (arch, seed) plus the labels of missing or incomplete files."""
    expected = expected_validation_ids(task, manifest)
    available: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
    missing: list[str] = []
    for arch in MODEL_ORDER:
        for seed in SEEDS:
            label = f"{task}/{arch}/seed-{seed}"
            try:
                frame = revision_validation_predictions(task, arch, seed)
            except (FileNotFoundError, ValueError):
                missing.append(label)
                continue
            if set(frame["sample_id"].astype(str)) != expected:
                missing.append(f"{label} (incomplete)")
                continue
            _, targets, probabilities = stack_seeds([frame], task)
            available[(arch, seed)] = (targets, probabilities[0])
    return available, missing


def confusion_matrix(targets: np.ndarray, predictions: np.ndarray, class_count: int) -> np.ndarray:
    return np.bincount(targets * class_count + predictions, minlength=class_count**2).reshape(class_count, class_count)


def confusion_stack(data: TaskData, indices: np.ndarray | None = None) -> np.ndarray:
    """Confusion matrices [models, seeds, C, C] on all samples or on an index resample."""
    selected = np.arange(len(data.targets)) if indices is None else indices
    targets = data.targets[selected]
    predictions = data.predictions[:, :, selected]
    return np.stack(
        [[confusion_matrix(targets, seed_predictions, data.class_count) for seed_predictions in model] for model in predictions]
    )


def macro_f1_from_confusion(matrices: np.ndarray) -> np.ndarray:
    """Macro-F1 with every class counted and zero for undefined classes, as in sklearn's zero_division=0."""
    matrices = np.asarray(matrices, dtype=float)
    true_positive = np.diagonal(matrices, axis1=-2, axis2=-1)
    denominator = matrices.sum(axis=-1) + matrices.sum(axis=-2)
    f1 = np.divide(2.0 * true_positive, denominator, out=np.zeros_like(denominator), where=denominator > 0)
    return f1.mean(axis=-1)


def error_rates(matrices: np.ndarray, task: str) -> dict[str, np.ndarray]:
    """Row-normalised operational error rates plus the Rotten share of Fresh labels."""
    matrices = np.asarray(matrices, dtype=float)
    names = class_names(task)
    rates: dict[str, np.ndarray] = {}
    for key, (true_name, predicted_names) in ERROR_RATES[task].items():
        row = matrices[..., names.index(true_name), :]
        hits = row[..., [names.index(name) for name in predicted_names]].sum(axis=-1)
        total = row.sum(axis=-1)
        rates[key] = np.divide(hits, total, out=np.full(total.shape, np.nan), where=total > 0)
    fresh, rotten = names.index("Fresh"), names.index("Rotten")
    labelled_fresh = matrices[..., :, fresh].sum(axis=-1)
    rates["rotten_among_fresh_labels"] = np.divide(
        matrices[..., rotten, fresh], labelled_fresh, out=np.full(labelled_fresh.shape, np.nan), where=labelled_fresh > 0
    )
    return rates


def bootstrap_confusions(data: TaskData, replicates: int) -> np.ndarray:
    """Cluster-bootstrap confusion matrices [replicates, models, seeds, C, C] on shared resamples."""
    models, seeds, count, classes = data.probabilities.shape
    codes = (data.targets * classes + data.predictions).reshape(models * seeds, count)
    offsets = (np.arange(models * seeds) * classes**2)[:, None]

    def statistic(indices: np.ndarray) -> np.ndarray:
        return np.bincount((codes[:, indices] + offsets).ravel(), minlength=models * seeds * classes**2)

    values = cluster_bootstrap(data.groups, statistic, replicates=replicates, seed=BOOTSTRAP_SEED)
    return values.reshape(replicates, models, seeds, classes, classes)


def helper_agreement(data: TaskData) -> float:
    """Largest gap between confusion-derived and helper seed-mean macro-F1 on the full set and five resamples."""
    helpers = [seed_mean_macro_f1(data.targets, model) for model in data.probabilities]

    def gap(indices: np.ndarray) -> float:
        fast = macro_f1_from_confusion(confusion_stack(data, indices)).mean(axis=1)
        return float(np.max(np.abs(fast - np.asarray([helper(indices) for helper in helpers]))))

    resampled = cluster_bootstrap(data.groups, gap, replicates=5, seed=BOOTSTRAP_SEED)
    return max(gap(np.arange(len(data.targets))), float(resampled.max()))


def valid_interval(values: np.ndarray) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return math.nan, math.nan, 0
    low, high = percentile_interval(finite)
    return low, high, int(len(finite))


def seed_metrics(data: TaskData) -> pd.DataFrame:
    records = []
    for model_index, arch in enumerate(MODEL_ORDER):
        for seed_index, seed in enumerate(SEEDS):
            result = metrics(data.targets, data.probabilities[model_index, seed_index], data.task)
            record: dict[str, object] = {
                "task": data.task,
                "calibration_state": CALIBRATION_STATE[data.task],
                "architecture": arch,
                "seed": seed,
                "samples": result["sample_count"],
            }
            for key in SCALAR_METRICS[:-1]:
                record[key] = result[key]
            record["auroc"] = result.get("roc_auc", result.get("roc_auc_ovr_macro", math.nan))
            for name in class_names(data.task):
                slug = name.lower().replace("-", "_")
                for field, column in (("precision", "precision"), ("recall", "recall"), ("f1-score", "f1")):
                    record[f"{column}_{slug}"] = result["per_class"][name][field]
            records.append(record)
    return pd.DataFrame.from_records(records)


def metric_summary(data: TaskData, seed_frame: pd.DataFrame, f1_replicates: np.ndarray, observed_f1: np.ndarray) -> pd.DataFrame:
    value_columns = [column for column in seed_frame.columns if column not in {"task", "calibration_state", "architecture", "seed", "samples"}]
    records = []
    for model_index, arch in enumerate(MODEL_ORDER):
        subset = seed_frame[seed_frame["architecture"] == arch]
        record: dict[str, object] = {
            "task": data.task,
            "calibration_state": CALIBRATION_STATE[data.task],
            "architecture": arch,
            "seeds": int(len(subset)),
            "samples": int(len(data.targets)),
            "capture_groups": int(len(np.unique(data.groups))),
        }
        for column in value_columns:
            record[f"{column}_mean"] = float(subset[column].mean())
            record[f"{column}_sd"] = float(subset[column].std(ddof=1))
        low, high, valid = valid_interval(f1_replicates[:, model_index])
        record.update(
            {
                "macro_f1_seed_mean": float(observed_f1[model_index]),
                "macro_f1_ci_low": low,
                "macro_f1_ci_high": high,
                "macro_f1_bootstrap_sd": float(np.std(f1_replicates[:, model_index], ddof=1)),
                "bootstrap_replicates": valid,
            }
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def pairwise_differences(task: str, f1_replicates: np.ndarray, observed_f1: np.ndarray) -> pd.DataFrame:
    records = []
    for first, model_a in enumerate(MODEL_ORDER):
        for second, model_b in enumerate(MODEL_ORDER):
            if first == second:
                continue
            difference = f1_replicates[:, first] - f1_replicates[:, second]
            low, high = percentile_interval(difference)
            records.append(
                {
                    "task": task,
                    "model_a": model_a,
                    "model_b": model_b,
                    "observed_difference": float(observed_f1[first] - observed_f1[second]),
                    "bootstrap_mean_difference": float(difference.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "share_positive": float(np.mean(difference > 0)),
                    "share_negative": float(np.mean(difference < 0)),
                    "share_tied": float(np.mean(difference == 0)),
                    "replicates": int(len(difference)),
                }
            )
    return pd.DataFrame.from_records(records)


def rank_probabilities(task: str, f1_replicates: np.ndarray, observed_f1: np.ndarray) -> pd.DataFrame:
    """Rank distribution per model; tied models share the tied ranks equally."""
    replicates, models = f1_replicates.shape
    shares = np.zeros((models, models))
    for values in f1_replicates:
        order = np.argsort(-values, kind="stable")
        position = 0
        while position < models:
            tied = order[np.flatnonzero(values[order] == values[order[position]])]
            ranks = np.arange(position, position + len(tied))
            shares[np.ix_(tied, ranks)] += 1.0 / len(tied)
            position += len(tied)
    shares /= replicates
    observed_rank = np.argsort(np.argsort(-observed_f1, kind="stable"), kind="stable") + 1
    records = []
    for model_index, arch in enumerate(MODEL_ORDER):
        record: dict[str, object] = {"task": task, "architecture": arch, "observed_rank": int(observed_rank[model_index])}
        record.update({f"p_rank_{rank + 1}": float(shares[model_index, rank]) for rank in range(models)})
        record["mean_rank"] = float(np.sum(shares[model_index] * np.arange(1, models + 1)))
        record["replicates"] = int(replicates)
        records.append(record)
    return pd.DataFrame.from_records(records)


def confusion_records(data: TaskData, matrices: np.ndarray) -> pd.DataFrame:
    names = class_names(data.task)
    records = []
    for model_index, arch in enumerate(MODEL_ORDER):
        per_seed = [(str(seed), matrices[model_index, seed_index]) for seed_index, seed in enumerate(SEEDS)]
        for seed_label, matrix in [*per_seed, ("all", matrices[model_index].sum(axis=0))]:
            row_totals = matrix.sum(axis=1)
            for true_index, true_name in enumerate(names):
                for predicted_index, predicted_name in enumerate(names):
                    count = int(matrix[true_index, predicted_index])
                    records.append(
                        {
                            "task": data.task,
                            "architecture": arch,
                            "seed": seed_label,
                            "true_label": true_name,
                            "predicted_label": predicted_name,
                            "count": count,
                            "row_share": count / row_totals[true_index] if row_totals[true_index] else math.nan,
                        }
                    )
    return pd.DataFrame.from_records(records)


def operational_tables(data: TaskData, matrices: np.ndarray, replicates: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    names = class_names(data.task)
    fresh, rotten = names.index("Fresh"), names.index("Rotten")
    seed_rates = error_rates(matrices, data.task)
    # Seeds share targets, so a rate is undefined for every seed of a resample at once.
    replicate_rates = {key: value.mean(axis=2) for key, value in error_rates(replicates, data.task).items()}
    seed_records = []
    for model_index, arch in enumerate(MODEL_ORDER):
        for seed_index, seed in enumerate(SEEDS):
            matrix = matrices[model_index, seed_index]
            record: dict[str, object] = {
                "task": data.task,
                "architecture": arch,
                "seed": seed,
                "rotten_count": int(matrix[rotten].sum()),
                "rotten_to_fresh_count": int(matrix[rotten, fresh]),
                "fresh_count": int(matrix[fresh].sum()),
                "fresh_to_rotten_count": int(matrix[fresh, rotten]),
            }
            record.update({key: float(value[model_index, seed_index]) for key, value in seed_rates.items()})
            seed_records.append(record)
    summary_records = []
    for model_index, arch in enumerate(MODEL_ORDER):
        record = {"task": data.task, "architecture": arch, "seeds": len(SEEDS)}
        for key, value in seed_rates.items():
            low, high, valid = valid_interval(replicate_rates[key][:, model_index])
            record.update(
                {
                    f"{key}_mean": float(np.mean(value[model_index])),
                    f"{key}_sd": float(np.std(value[model_index], ddof=1)),
                    f"{key}_ci_low": low,
                    f"{key}_ci_high": high,
                    f"{key}_ci_replicates": valid,
                }
            )
        summary_records.append(record)
    return pd.DataFrame.from_records(seed_records), pd.DataFrame.from_records(summary_records)


def tie_averaged_cumsum(values: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """Expected sum of ``values`` over the k most confident items, k = 1..N, averaging over tie orders."""
    order = np.argsort(-confidence, kind="stable")
    ranked = confidence[order]
    cumulative = np.concatenate([[0.0], np.cumsum(np.asarray(values, dtype=float)[order])])
    starts = np.flatnonzero(np.r_[True, ranked[1:] != ranked[:-1]])
    ends = np.r_[starts[1:], len(ranked)]
    sizes = ends - starts
    group = np.repeat(np.arange(len(starts)), sizes)
    position = np.arange(1, len(ranked) + 1)
    totals = cumulative[ends] - cumulative[starts]
    return cumulative[starts][group] + (position - starts[group]) * totals[group] / sizes[group]


def selective_curve(probabilities: np.ndarray, targets: np.ndarray, task: str) -> dict[str, np.ndarray]:
    """Risk-coverage curve accepting items in order of decreasing max-probability."""
    names = class_names(task)
    fresh, rotten = names.index("Fresh"), names.index("Rotten")
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accepted = np.arange(1, len(targets) + 1, dtype=float)
    errors = tie_averaged_cumsum(predictions != targets, confidence)
    rotten_accepted = tie_averaged_cumsum(targets == rotten, confidence)
    critical = tie_averaged_cumsum((targets == rotten) & (predictions == fresh), confidence)
    return {
        "accepted": accepted,
        "coverage": accepted / len(targets),
        "selective_error": errors / accepted,
        "rotten_accepted": rotten_accepted,
        "rotten_to_fresh_accepted": np.divide(critical, rotten_accepted, out=np.full(len(targets), np.nan), where=rotten_accepted > 0),
        "rotten_to_fresh_escape": critical / max(int(np.sum(targets == rotten)), 1),
        "critical_selective_risk": critical / accepted,
    }


def coverage_index(coverage: float, count: int) -> int:
    return max(int(math.ceil(round(coverage * count, 9))), 1) - 1


def aurc(curve: dict[str, np.ndarray]) -> tuple[float, float]:
    """Area under the risk-coverage curve and its excess over the oracle ranking."""
    count = len(curve["accepted"])
    risk = curve["selective_error"]
    errors = int(round(risk[-1] * count))
    accepted = curve["accepted"]
    oracle = np.maximum(accepted - (count - errors), 0.0) / accepted
    value = float(risk.mean())
    return value, value - float(oracle.mean())


def risk_coverage(data: TaskData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    count = len(data.targets)
    grid_index = np.asarray([coverage_index(coverage, count) for coverage in COVERAGE_GRID])
    curve_records, seed_records = [], []
    for model_index, arch in enumerate(MODEL_ORDER):
        for seed_index, seed in enumerate(SEEDS):
            curve = selective_curve(data.probabilities[model_index, seed_index], data.targets, data.task)
            area, excess = aurc(curve)
            record: dict[str, object] = {"architecture": arch, "seed": seed, "aurc": area, "e_aurc": excess, "error_full": float(curve["selective_error"][-1])}
            for coverage in REPORTED_COVERAGES:
                index = coverage_index(coverage, count)
                tag = f"{round(coverage * 100):d}"
                record[f"selective_error_at_{tag}"] = float(curve["selective_error"][index])
                record[f"rotten_to_fresh_accepted_at_{tag}"] = float(curve["rotten_to_fresh_accepted"][index])
                record[f"rotten_to_fresh_escape_at_{tag}"] = float(curve["rotten_to_fresh_escape"][index])
                record[f"rotten_accepted_at_{tag}"] = float(curve["rotten_accepted"][index])
            seed_records.append(record)
            for grid_value, index in zip(COVERAGE_GRID, grid_index):
                curve_records.append(
                    {
                        "architecture": arch,
                        "seed": seed,
                        "coverage": float(grid_value),
                        "accepted": int(curve["accepted"][index]),
                        "selective_error": float(curve["selective_error"][index]),
                        "rotten_accepted": float(curve["rotten_accepted"][index]),
                        "rotten_to_fresh_accepted": float(curve["rotten_to_fresh_accepted"][index]),
                        "rotten_to_fresh_escape": float(curve["rotten_to_fresh_escape"][index]),
                        "critical_selective_risk": float(curve["critical_selective_risk"][index]),
                    }
                )
    curves = pd.DataFrame.from_records(curve_records)
    averaged = (
        curves.groupby(["architecture", "coverage"], sort=False)
        .agg(
            selective_error=("selective_error", "mean"),
            rotten_to_fresh_accepted=("rotten_to_fresh_accepted", "mean"),
            rotten_to_fresh_escape=("rotten_to_fresh_escape", "mean"),
            min_rotten_accepted=("rotten_accepted", "min"),
        )
        .reset_index()
    )
    averaged["rotten_to_fresh_plotted"] = averaged["min_rotten_accepted"] >= MIN_ACCEPTED_ROTTEN
    return curves, averaged, pd.DataFrame.from_records(seed_records)


def abstention_threshold(confidence: np.ndarray, correct: np.ndarray, target: float) -> tuple[float, bool]:
    """Lowest max-probability threshold whose selective accuracy reaches ``target`` (else the most accurate one)."""
    candidates = np.r_[0.0, np.unique(confidence)]
    order = np.argsort(confidence, kind="stable")
    ranked = confidence[order]
    suffix_correct = np.r_[np.cumsum(np.asarray(correct, dtype=float)[order][::-1])[::-1], 0.0]
    start = np.searchsorted(ranked, candidates, side="left")
    accepted = len(ranked) - start
    accuracy = suffix_correct[start] / np.maximum(accepted, 1)
    reached = np.flatnonzero(accuracy >= target)
    if len(reached):
        return float(candidates[reached[0]]), True
    return float(candidates[np.flatnonzero(accuracy == accuracy.max())[0]]), False


def abstention_outcome(probabilities: np.ndarray, targets: np.ndarray, task: str, threshold: float) -> dict[str, float]:
    names = class_names(task)
    fresh, rotten = names.index("Fresh"), names.index("Rotten")
    predictions = probabilities.argmax(axis=1)
    accepted = probabilities.max(axis=1) >= threshold
    is_rotten = targets == rotten
    critical = is_rotten & (predictions == fresh)
    rotten_accepted = int(np.sum(is_rotten & accepted))
    return {
        "coverage": float(accepted.mean()),
        "accepted": int(accepted.sum()),
        "selective_accuracy": float(np.mean(predictions[accepted] == targets[accepted])) if accepted.any() else math.nan,
        "rotten_accepted": rotten_accepted,
        "rotten_to_fresh_accepted": float(np.sum(critical & accepted) / rotten_accepted) if rotten_accepted else math.nan,
        "rotten_to_fresh_escape": float(np.sum(critical & accepted) / max(int(is_rotten.sum()), 1)),
    }


def fresh_gate_threshold(probabilities: np.ndarray, targets: np.ndarray, task: str, limit: float) -> float:
    """Smallest P(Fresh) gate whose Rotten->Fresh rate is at most ``limit`` (infinity gates every Fresh label)."""
    names = class_names(task)
    fresh, rotten = names.index("Fresh"), names.index("Rotten")
    p_fresh = probabilities[:, fresh]
    fresh_predicted = probabilities.argmax(axis=1) == fresh
    candidates = np.r_[0.0, np.unique(p_fresh[fresh_predicted]), np.inf]
    is_rotten = targets == rotten
    critical_scores = np.sort(p_fresh[fresh_predicted & is_rotten])
    counts = len(critical_scores) - np.searchsorted(critical_scores, candidates, side="left")
    rates = counts / max(int(is_rotten.sum()), 1)
    return float(candidates[np.flatnonzero(rates <= limit)[0]])


def fresh_gate_outcome(probabilities: np.ndarray, targets: np.ndarray, task: str, tau: float) -> dict[str, float]:
    """Fresh only if P(Fresh) >= tau; gated argmax-Fresh items go to manual inspection, other labels are kept."""
    names = class_names(task)
    fresh, rotten = names.index("Fresh"), names.index("Rotten")
    p_fresh = probabilities[:, fresh]
    predictions = probabilities.argmax(axis=1)
    fresh_predicted = predictions == fresh
    labelled_fresh = fresh_predicted & (p_fresh >= tau)
    manual = fresh_predicted & ~labelled_fresh
    is_fresh, is_rotten = targets == fresh, targets == rotten
    automatic = ~manual
    return {
        "rotten_to_fresh": float(np.sum(labelled_fresh & is_rotten) / max(int(is_rotten.sum()), 1)),
        "fresh_kept": float(np.sum(labelled_fresh & is_fresh) / max(int(is_fresh.sum()), 1)),
        "manual_share": float(manual.mean()),
        "fresh_routed": float(np.sum(manual & is_fresh) / max(int(is_fresh.sum()), 1)),
        "rotten_routed": float(np.sum(manual & is_rotten) / max(int(is_rotten.sum()), 1)),
        "fresh_label_precision": float(np.sum(labelled_fresh & is_fresh) / labelled_fresh.sum()) if labelled_fresh.any() else math.nan,
        "automatic_accuracy": float(np.mean(predictions[automatic] == targets[automatic])) if automatic.any() else math.nan,
    }


def operating_points(
    test: TaskData, validation: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]], *, abstention: bool
) -> pd.DataFrame:
    records = []
    for model_index, arch in enumerate(MODEL_ORDER):
        for seed_index, seed in enumerate(SEEDS):
            val_targets, val_probabilities = validation[(arch, seed)]
            test_probabilities = test.probabilities[model_index, seed_index]
            base = {"task": test.task, "architecture": arch, "seed": seed}
            if abstention:
                val_correct = val_probabilities.argmax(axis=1) == val_targets
                for target in ABSTENTION_SENSITIVITY:
                    threshold, met = abstention_threshold(val_probabilities.max(axis=1), val_correct, target)
                    val_outcome = abstention_outcome(val_probabilities, val_targets, test.task, threshold)
                    test_outcome = abstention_outcome(test_probabilities, test.targets, test.task, threshold)
                    records.append(
                        base
                        | {"rule": "abstention", "target": target, "threshold": threshold, "target_met_on_validation": met}
                        | {f"validation_{key}": value for key, value in val_outcome.items()}
                        | {f"test_{key}": value for key, value in test_outcome.items()}
                    )
            ungated = fresh_gate_outcome(test_probabilities, test.targets, test.task, 0.0)
            for limit in GATE_SENSITIVITY:
                tau = fresh_gate_threshold(val_probabilities, val_targets, test.task, limit)
                val_outcome = fresh_gate_outcome(val_probabilities, val_targets, test.task, tau)
                test_outcome = fresh_gate_outcome(test_probabilities, test.targets, test.task, tau)
                records.append(
                    base
                    | {
                        "rule": "fresh_gate",
                        "target": limit,
                        "threshold": tau,
                        "target_met_on_validation": val_outcome["rotten_to_fresh"] <= limit,
                        "gate_active": bool(tau > 0.0),
                        "degenerate": bool(not np.isfinite(tau)),
                        "test_rotten_to_fresh_ungated": ungated["rotten_to_fresh"],
                        "test_fresh_kept_ungated": ungated["fresh_kept"],
                    }
                    | {f"validation_{key}": value for key, value in val_outcome.items()}
                    | {f"test_{key}": value for key, value in test_outcome.items()}
                )
    return pd.DataFrame.from_records(records)


def summarize_operating_points(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["rule", "task", "target", "architecture"]
    numeric = [
        column
        for column in frame.columns
        if column not in {*keys, "seed"} and pd.api.types.is_numeric_dtype(frame[column]) and not pd.api.types.is_bool_dtype(frame[column])
    ]
    records = []
    for values, subset in frame.groupby(keys, sort=False):
        record: dict[str, object] = dict(zip(keys, values))
        record["seeds"] = int(len(subset))
        record["target_met_on_validation_seeds"] = int(subset["target_met_on_validation"].eq(True).sum())
        if "gate_active" in subset and subset["gate_active"].notna().any():
            record["gate_active_seeds"] = int(subset["gate_active"].eq(True).sum())
            record["degenerate_seeds"] = int(subset["degenerate"].eq(True).sum())
        for column in numeric:
            column_values = subset[column].to_numpy(dtype=float)
            record[f"{column}_mean"] = float(np.mean(column_values))
            record[f"{column}_sd"] = float(np.std(column_values, ddof=1)) if len(column_values) > 1 else math.nan
        records.append(record)
    return pd.DataFrame.from_records(records)


def per_produce(data: TaskData) -> pd.DataFrame:
    names = class_names(data.task)
    fresh, rotten = names.index("Fresh"), names.index("Rotten")
    records = []
    for model_index, arch in enumerate(MODEL_ORDER):
        for seed_index, seed in enumerate(SEEDS):
            probabilities = data.probabilities[model_index, seed_index]
            for produce in sorted(set(data.produce)):
                mask = data.produce == produce
                result = metrics(data.targets[mask], probabilities[mask], data.task)
                is_rotten = data.targets[mask] == rotten
                predictions = probabilities[mask].argmax(axis=1)
                records.append(
                    {
                        "architecture": arch,
                        "seed": seed,
                        "produce_type": produce,
                        "sample_count": int(mask.sum()),
                        "capture_groups": int(len(np.unique(data.groups[mask]))),
                        "macro_f1": result["macro_f1"],
                        "accuracy": result["accuracy"],
                        "balanced_accuracy": result["balanced_accuracy"],
                        "rotten_count": int(is_rotten.sum()),
                        "rotten_to_fresh": float(np.mean(predictions[is_rotten] == fresh)) if is_rotten.any() else math.nan,
                    }
                )
    return pd.DataFrame.from_records(records)


def frozen_checks(
    seed_frames: dict[str, pd.DataFrame], matrices: dict[str, np.ndarray], produce_frame: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare recomputed values with the frozen aggregates and the stated headline numbers."""
    checks: dict[str, object] = {}
    rows = []
    summary_path = FROZEN / "aggregate/benchmark_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.is_file() else None
    for task, frame in seed_frames.items():
        for arch in MODEL_ORDER:
            recomputed = float(frame.loc[frame["architecture"] == arch, "macro_f1"].mean())
            frozen = math.nan
            if summary is not None:
                match = summary[
                    (summary["setting"] == "grouped")
                    & (summary["task"] == task)
                    & (summary["calibration_state"] == CALIBRATION_STATE[task])
                    & (summary["architecture"] == arch)
                ]
                frozen = float(match["macro_f1_mean"].iloc[0]) if len(match) else math.nan
            rows.append(
                {
                    "task": task,
                    "architecture": arch,
                    "recomputed_macro_f1_mean": recomputed,
                    "frozen_aggregate_macro_f1_mean": frozen,
                    "headline_macro_f1": HEADLINE_MACRO_F1[task][arch],
                    "abs_difference_vs_aggregate": abs(recomputed - frozen) if math.isfinite(frozen) else math.nan,
                    "headline_reproduced": round(recomputed, 3) == HEADLINE_MACRO_F1[task][arch],
                }
            )
    table = pd.DataFrame.from_records(rows)
    checks["headline_reproduced"] = bool(table["headline_reproduced"].all())
    differences = table["abs_difference_vs_aggregate"].dropna()
    checks["aggregate_max_abs_difference"] = float(differences.max()) if len(differences) else None

    seed_path = FROZEN / "aggregate/benchmark_seed_metrics.csv"
    if seed_path.is_file():
        frozen_seed = pd.read_csv(seed_path)
        largest = 0.0
        for task, frame in seed_frames.items():
            subset = frozen_seed[
                (frozen_seed["setting"] == "grouped") & (frozen_seed["task"] == task) & (frozen_seed["calibration_state"] == CALIBRATION_STATE[task])
            ]
            merged = frame.merge(subset, on=["architecture", "seed"], suffixes=("", "_frozen"), validate="one_to_one")
            if len(merged) != len(frame):
                raise ValueError(f"Frozen seed metrics do not cover every {task} run")
            for column, frozen_column in FROZEN_SEED_METRICS.items():
                largest = max(largest, float(np.max(np.abs(merged[column] - merged[f"{frozen_column}_frozen"]))))
            auroc_column = "roc_auc" if task == "binary" else "roc_auc_ovr_macro"
            largest = max(largest, float(np.max(np.abs(merged["auroc"] - merged[auroc_column]))))
        checks["seed_metrics_max_abs_difference"] = largest

    confusion_match = True
    compared = 0
    for task, stack in matrices.items():
        for model_index, arch in enumerate(MODEL_ORDER):
            for seed_index, seed in enumerate(SEEDS):
                path = FROZEN / "evaluation/grouped" / task / arch / f"seed-{seed}" / CALIBRATION_STATE[task] / "test_metrics.json"
                if path.is_file():
                    frozen_matrix = np.asarray(json.loads(path.read_text(encoding="utf-8"))["confusion_matrix"])
                    confusion_match &= bool(np.array_equal(frozen_matrix, stack[model_index, seed_index]))
                    compared += 1
    checks["confusion_matrices_match_frozen"] = confusion_match if compared else None
    checks["confusion_matrices_compared"] = compared

    produce_path = FROZEN / "aggregate/per_produce_seed_metrics.csv"
    if produce_path.is_file():
        frozen_produce = pd.read_csv(produce_path)
        merged = produce_frame.merge(frozen_produce, on=["architecture", "seed", "produce_type"], suffixes=("", "_frozen"), validate="one_to_one")
        if len(merged) != len(produce_frame):
            raise ValueError("Frozen per-produce metrics do not cover every run and produce type")
        checks["per_produce_max_abs_difference"] = float(
            max(np.max(np.abs(merged[column] - merged[f"{column}_frozen"])) for column in ("macro_f1", "accuracy", "balanced_accuracy"))
        )
        checks["per_produce_sample_counts_match"] = bool((merged["sample_count"] == merged["sample_count_frozen"]).all())
    return table, checks


def percent_sd(values: np.ndarray) -> str:
    """Percent mean and SD; a dagger marks a mean over only the seeds where the rate is defined."""
    array = 100.0 * np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if len(finite) == 0:
        return "n/a"
    text = rf"${finite.mean():.1f}\pm{finite.std(ddof=1):.1f}$" if len(finite) > 1 else rf"${finite[0]:.1f}$"
    return text if len(finite) == len(array) else text + r"$^{\dagger}$"


def threshold_sd(values: np.ndarray) -> str:
    """Mean and SD of finite thresholds; an asterisk marks seeds whose gate removed every Fresh label."""
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if len(finite) == len(array):
        return mean_sd(array)
    return rf"{mean_sd(finite)}$^{{\ast}}$" if len(finite) else r"$\infty^{\ast}$"


def stacked_cell(top: str, bottom: str) -> str:
    return rf"\begin{{tabular}}[c]{{@{{}}c@{{}}}}{top}\\[-0.4ex]{{\scriptsize {bottom}}}\end{{tabular}}"


def signed_interval(low: float, high: float, digits: int = 3) -> str:
    return rf"$[{low:+.{digits}f}, {high:+.{digits}f}]$"


def panel_title(text: str, columns: int, first: bool) -> list[str]:
    prefix = "" if first else r"\midrule"
    return [rf"{prefix}\multicolumn{{{columns}}}{{@{{}}l@{{}}}}{{\emph{{{text}}}}}"]


def write_main_table(path: Path, summaries: dict[str, pd.DataFrame], seed_frames: dict[str, pd.DataFrame], data: dict[str, TaskData]) -> None:
    headers = ["Model", "Acc.", "Bal.\\ acc.", "Macro-F1", r"95\% CI", "NLL", "Brier", "ECE", "AUROC"]
    titles = {
        "three_stage": "(a) Three-stage (Fresh/Semi-Fresh/Rotten), temperature-scaled",
        "binary": "(b) Binary (Fresh/Rotten), uncalibrated",
    }
    rows: list[list[str]] = []
    for position, task in enumerate(TASKS):
        frame, summary = seed_frames[task], summaries[task].set_index("architecture")
        size = f"{len(data[task].targets)} images, {len(np.unique(data[task].groups))} capture groups"
        rows.append(panel_title(f"{titles[task]}; {size}", len(headers), position == 0))
        for arch in MODEL_ORDER:
            subset = frame[frame["architecture"] == arch]
            rows.append(
                [
                    DISPLAY_NAMES[arch],
                    mean_sd(subset["accuracy"]),
                    mean_sd(subset["balanced_accuracy"]),
                    mean_sd(subset["macro_f1"]),
                    interval(summary.loc[arch, "macro_f1_ci_low"], summary.loc[arch, "macro_f1_ci_high"]),
                    mean_sd(subset["nll"]),
                    mean_sd(subset["brier"]),
                    mean_sd(subset["ece"]),
                    mean_sd(subset["auroc"]),
                ]
            )
    latex_table(
        path,
        caption=(
            "Grouped locked-test results of the five backbones (mean $\\pm$ sample SD over training seeds 2026, 3407, and 777). "
            "The 95\\% CI is the percentile cluster-bootstrap interval of the seed-averaged macro-F1 (2,000 resamples of whole "
            "capture groups). Three-stage probabilities use the validation-fitted temperature; binary probabilities are uncalibrated. "
            "ECE uses 15 equal-width bins, Brier is summed over classes, and three-stage AUROC is one-vs-rest macro-averaged."
        ),
        label="tab:rev-main-metrics",
        headers=headers,
        rows=rows,
        wide=True,
    )


def write_pairwise_table(path: Path, pairwise: dict[str, pd.DataFrame], ranks: dict[str, pd.DataFrame], observed: dict[str, np.ndarray]) -> dict[str, str]:
    headers = ["Model", r"$\Delta$ vs leader", r"$\Delta$ vs selected", "P(best)"]
    rows: list[list[str]] = []
    leaders: dict[str, str] = {}
    for position, task in enumerate(TASKS):
        leader = MODEL_ORDER[int(np.argmax(observed[task]))]
        leaders[task] = leader
        name = "Three-stage" if task == "three_stage" else "Binary"
        title = f"({'ab'[position]}) {name}: leader {DISPLAY_NAMES[leader]}, selected {DISPLAY_NAMES[PROTOCOL_SELECTED]}"
        rows.append(panel_title(title, len(headers), position == 0))
        frame = pairwise[task].set_index(["model_a", "model_b"])
        rank_frame = ranks[task].set_index("architecture")
        for arch in MODEL_ORDER:
            cells = [DISPLAY_NAMES[arch]]
            for reference in (leader, PROTOCOL_SELECTED):
                if arch == reference:
                    cells.append("---")
                    continue
                row = frame.loc[(arch, reference)]
                cells.append(stacked_cell(rf"${row['observed_difference']:+.3f}$", signed_interval(row["ci_low"], row["ci_high"])))
            cells.append(f"{rank_frame.loc[arch, 'p_rank_1']:.3f}")
            rows.append(cells)
    latex_table(
        path,
        caption=(
            "Paired cluster-bootstrap comparison of seed-averaged grouped-test macro-F1. $\\Delta$ is the row model minus the "
            "reference, with its percentile 95\\% CI beneath; all models share the same 2,000 capture-group resamples. The leader "
            "is the descriptive test best (never used for selection); the selected model is the protocol choice from three-stage "
            "validation. P(best) is the share of resamples in which the model has the highest macro-F1."
        ),
        label="tab:rev-pairwise",
        headers=headers,
        rows=rows,
    )
    return leaders


def write_operational_table(path: Path, seed_rates: dict[str, pd.DataFrame], summaries: dict[str, pd.DataFrame]) -> None:
    headers = ["Three-stage", r"R$\to$F", "R missed", r"F$\to$R", r"SF$\to$F"]
    rows: list[list[str]] = []
    for task in TASKS:
        frame, summary = seed_rates[task], summaries[task].set_index("architecture")
        if task == "binary":
            rows.append([r"\midrule\textbf{Binary}", r"\textbf{R$\to$F}", r"\textbf{F$\to$R}", "", ""])
        for position, arch in enumerate(MODEL_ORDER):
            subset = frame[frame["architecture"] == arch]
            ci = interval(100 * summary.loc[arch, "rotten_to_fresh_ci_low"], 100 * summary.loc[arch, "rotten_to_fresh_ci_high"], digits=1)
            critical = stacked_cell(percent_sd(subset["rotten_to_fresh"]), ci)
            name = DISPLAY_NAMES[arch] if task == "three_stage" or position else rf"\midrule {DISPLAY_NAMES[arch]}"
            if task == "three_stage":
                rows.append([name, critical, percent_sd(subset["rotten_missed"]), percent_sd(subset["fresh_to_rotten"]), percent_sd(subset["semi_fresh_to_fresh"])])
            else:
                rows.append([name, critical, percent_sd(subset["fresh_to_rotten"]), "", ""])
    latex_table(
        path,
        caption=(
            "Operational error rates on the grouped test (\\%, mean $\\pm$ sample SD over three seeds). R$\\to$F: true Rotten "
            "predicted Fresh (critical miss), with the percentile cluster-bootstrap 95\\% CI of the seed-averaged rate beneath "
            "(2,000 capture-group resamples); R missed: true Rotten predicted not Rotten; F$\\to$R: true Fresh predicted Rotten "
            "(false alarm); SF$\\to$F: true Semi-Fresh predicted Fresh. Three-stage probabilities are temperature-scaled; binary "
            "probabilities are uncalibrated, and every binary Rotten miss is an R$\\to$F error."
        ),
        label="tab:rev-operational",
        headers=headers,
        rows=rows,
    )


def write_selective_table(path: Path, seed_frame: pd.DataFrame) -> None:
    headers = ["Model", "AURC", r"Err@50\%", r"Err@80\%", r"R$\to$F@80\%"]
    rows = []
    for arch in MODEL_ORDER:
        subset = seed_frame[seed_frame["architecture"] == arch]
        rows.append(
            [
                DISPLAY_NAMES[arch],
                mean_sd(subset["aurc"]),
                percent_sd(subset["selective_error_at_50"]),
                percent_sd(subset["selective_error_at_80"]),
                percent_sd(subset["rotten_to_fresh_accepted_at_80"]),
            ]
        )
    latex_table(
        path,
        caption=(
            "Selective classification by temperature-scaled maximum probability on the three-stage grouped test (mean $\\pm$ "
            "sample SD over three seeds). AURC is the area under the risk--coverage curve (lower is better); Err@$c$ is the error "
            "among the $c$ most confident images (\\%); R$\\to$F@80\\% is the share of accepted true-Rotten images predicted "
            "Fresh at 80\\% coverage (\\%)."
        ),
        label="tab:rev-selective",
        headers=headers,
        rows=rows,
    )


def write_operating_point_table(path: Path, frame: pd.DataFrame) -> None:
    """Three-stage abstention plus the specified and the strictest Fresh gate (binary gates stay in the CSV)."""
    frame = frame[frame["task"] == "three_stage"]
    headers = ["Abstention", "Threshold", "Coverage", r"Sel.\ acc.", r"R$\to$F"]
    rows: list[list[str]] = []
    abstention = frame[(frame["rule"] == "abstention") & (frame["target"] == ABSTENTION_TARGET)]
    for arch in MODEL_ORDER:
        subset = abstention[abstention["architecture"] == arch]
        rows.append(
            [
                DISPLAY_NAMES[arch],
                threshold_sd(subset["threshold"]),
                percent_sd(subset["test_coverage"]),
                percent_sd(subset["test_selective_accuracy"]),
                percent_sd(subset["test_rotten_to_fresh_accepted"]),
            ]
        )
    active: dict[float, int] = {}
    for limit in (GATE_LIMIT, min(GATE_SENSITIVITY)):
        gate = frame[(frame["rule"] == "fresh_gate") & (frame["target"] == limit)]
        active[limit] = int(gate["gate_active"].eq(True).sum())
        rows.append([rf"\midrule\textbf{{Gate ({limit * 100:g}\%)}}", r"\textbf{$\tau$}", r"\textbf{R$\to$F}", r"\textbf{Fresh kept}", r"\textbf{Manual}"])
        for position, arch in enumerate(MODEL_ORDER):
            subset = gate[gate["architecture"] == arch]
            name = DISPLAY_NAMES[arch] if position else rf"\midrule {DISPLAY_NAMES[arch]}"
            rows.append(
                [
                    name,
                    threshold_sd(subset["threshold"]),
                    percent_sd(subset["test_rotten_to_fresh"]),
                    percent_sd(subset["test_fresh_kept"]),
                    percent_sd(subset["test_manual_share"]),
                ]
            )
    runs = len(MODEL_ORDER) * len(SEEDS)
    strict = min(GATE_SENSITIVITY)
    notes = []
    if any(r"^{\dagger}" in cell for row in rows for cell in row):
        notes.append("$^{\\dagger}$Mean over the seeds that accepted at least one true-Rotten test image.")
    if any(r"^{\ast}" in cell for row in rows for cell in row):
        notes.append("$^{\\ast}$Mean over finite thresholds; in the other seeds the gate removed every Fresh label.")
    latex_table(
        path,
        caption=(
            "Three-stage operating points chosen per seed on grouped validation only and applied to the grouped test "
            "(temperature-scaled; mean $\\pm$ sample SD over three seeds; rates in \\%). Abstention: lowest maximum-probability "
            f"threshold with validation selective accuracy of at least {ABSTENTION_TARGET * 100:.0f}\\%; images below it go to "
            "manual inspection, and R$\\to$F is the share of accepted true-Rotten images predicted Fresh. Gate: an image keeps "
            "a Fresh label only if $P(\\mathrm{Fresh})\\geq\\tau$, with $\\tau$ the smallest value giving a validation R$\\to$F "
            f"rate of at most {GATE_LIMIT * 100:g}\\% (specified; active in {active[GATE_LIMIT]} of {runs} runs, otherwise "
            f"$\\tau=0$) or {strict * 100:g}\\% (stricter sensitivity setting; active in {active[strict]} of {runs} runs); gated "
            "images go to manual inspection and other predictions are unchanged. Here R$\\to$F is the share of all true-Rotten "
            "images labelled Fresh, Fresh kept the share of true-Fresh images still labelled Fresh, and Manual the share of "
            "all images routed." + "".join(f" {note}" for note in notes)
        ),
        label="tab:rev-operating-points",
        headers=headers,
        rows=rows,
    )


def write_produce_table(path: Path, frame: pd.DataFrame, overall: pd.DataFrame, data: TaskData) -> None:
    headers = ["Produce", "Images", *[rf"\rotatebox{{90}}{{{DISPLAY_NAMES[arch]}}}" for arch in MODEL_ORDER]]
    rows = []
    for produce in sorted(frame["produce_type"].unique()):
        subset = frame[frame["produce_type"] == produce]
        first = subset.iloc[0]
        cells = [produce.replace("_", " "), f"{int(first['sample_count'])} ({int(first['capture_groups'])})"]
        for arch in MODEL_ORDER:
            values = subset.loc[subset["architecture"] == arch, "macro_f1"].to_numpy(dtype=float)
            cells.append(stacked_cell(f"{values.mean():.3f}", rf"$\pm${values.std(ddof=1):.3f}"))
        rows.append(cells)
    cells = [r"\midrule All", f"{len(data.targets)} ({len(np.unique(data.groups))})"]
    for arch in MODEL_ORDER:
        values = overall.loc[overall["architecture"] == arch, "macro_f1"].to_numpy(dtype=float)
        cells.append(stacked_cell(f"{values.mean():.3f}", rf"$\pm${values.std(ddof=1):.3f}"))
    rows.append(cells)
    latex_table(
        path,
        caption=(
            "Three-stage grouped-test macro-F1 by produce for all five backbones (temperature-scaled; mean over three seeds with "
            "the sample SD beneath). Images gives the test support with the number of capture groups in parentheses; strata "
            "drawn from one or a few capture groups describe those captures rather than produce-level generalization."
        ),
        label="tab:rev-per-produce",
        headers=headers,
        rows=rows,
    )


def plot_confusions(matrices: np.ndarray, stem: Path) -> list[Path]:
    figure_style()
    import matplotlib.pyplot as plt

    plt.rcParams["pdf.fonttype"] = 42
    labels = [name.replace("-", "-\n") for name in THREE_STAGE]
    figure, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(DOUBLE_COLUMN_WIDTH, 1.95), constrained_layout=True)
    image = None
    for model_index, (axis, arch) in enumerate(zip(axes, MODEL_ORDER)):
        matrix = matrices[model_index].sum(axis=0)
        share = 100.0 * matrix / matrix.sum(axis=1, keepdims=True)
        image = axis.imshow(share, cmap="Blues", vmin=0.0, vmax=100.0)
        for true_index in range(matrix.shape[0]):
            for predicted_index in range(matrix.shape[1]):
                value = share[true_index, predicted_index]
                axis.text(
                    predicted_index,
                    true_index,
                    f"{value:.1f}%\n({matrix[true_index, predicted_index]})",
                    ha="center",
                    va="center",
                    fontsize=7,
                    linespacing=1.05,
                    color="white" if value >= 55.0 else "black",
                )
        axis.set_title(DISPLAY_NAMES[arch])
        axis.set_xticks(range(len(labels)), labels)
        axis.set_yticks(range(len(labels)), labels if model_index == 0 else [])
        if model_index == 0:
            axis.set_ylabel("True label")
        axis.grid(False)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.supxlabel("Predicted label", fontsize=8)
    colorbar = figure.colorbar(image, ax=list(axes), fraction=0.025, pad=0.015)
    colorbar.set_label("Row share (%)")
    colorbar.outline.set_visible(False)
    outputs = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    for output in outputs:
        figure.savefig(output)
    plt.close(figure)
    return outputs


def plot_risk_coverage(averaged: pd.DataFrame, stem: Path) -> list[Path]:
    figure_style()
    import matplotlib.pyplot as plt

    plt.rcParams["pdf.fonttype"] = 42
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(SINGLE_COLUMN_WIDTH, 3.9), sharex=True, constrained_layout=True)
    for arch, style in zip(MODEL_ORDER, LINE_STYLES):
        subset = averaged[averaged["architecture"] == arch].sort_values("coverage")
        critical = subset["rotten_to_fresh_accepted"].where(subset["rotten_to_fresh_plotted"])
        kwargs = {"color": COLORS[arch], "linestyle": style, "linewidth": 1.2, "label": DISPLAY_NAMES[arch]}
        top.plot(subset["coverage"], subset["selective_error"], **kwargs)
        bottom.plot(subset["coverage"], 100.0 * critical, **kwargs)
    for axis in (top, bottom):
        for coverage in REPORTED_COVERAGES:
            axis.axvline(coverage, color="0.6", linewidth=0.6, linestyle=":", zorder=0)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(bottom=0.0)
    top.set_ylabel("Selective error")
    top.set_title("(a) Error among accepted images", loc="left")
    bottom.set_ylabel("Rotten$\\to$Fresh (%)")
    bottom.set_title("(b) Rotten$\\to$Fresh among accepted Rotten images", loc="left")
    bottom.set_xlabel("Coverage (share of test images accepted)")
    bottom.legend(loc="upper left", frameon=False, handlelength=2.6)
    outputs = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    for output in outputs:
        figure.savefig(output)
    plt.close(figure)
    return outputs


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=GROUPED_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS / TOPIC)
    parser.add_argument("--tables-dir", type=Path, default=PAPER_TABLES)
    parser.add_argument("--figures-dir", type=Path, default=PAPER_FIGURES)
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()

    for directory in (args.output_dir, args.tables_dir, args.figures_dir):
        directory.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    outputs: dict[str, list[str]] = {}

    def save(frame: pd.DataFrame, name: str, part: str) -> None:
        path = args.output_dir / name
        frame.to_csv(path, index=False)
        outputs.setdefault(part, []).append(relative(path))

    data = {task: load_test(task, args.manifest) for task in TASKS}
    matrices = {task: confusion_stack(data[task]) for task in TASKS}
    replicates = {task: bootstrap_confusions(data[task], args.replicates) for task in TASKS}
    agreement = {task: helper_agreement(data[task]) for task in TASKS}
    if max(agreement.values()) > 1e-12:
        raise RuntimeError(f"Confusion-derived macro-F1 disagrees with the shared helper: {agreement}")
    observed_f1 = {task: macro_f1_from_confusion(matrices[task]).mean(axis=1) for task in TASKS}
    f1_replicates = {task: macro_f1_from_confusion(replicates[task]).mean(axis=2) for task in TASKS}

    seed_frames = {task: seed_metrics(data[task]) for task in TASKS}
    summaries = {task: metric_summary(data[task], seed_frames[task], f1_replicates[task], observed_f1[task]) for task in TASKS}
    save(pd.concat(seed_frames.values(), ignore_index=True), "main_metrics_seed.csv", "main_metrics")
    save(pd.concat(summaries.values(), ignore_index=True), "main_metrics_summary.csv", "main_metrics")
    replicate_frame = pd.concat(
        [
            pd.DataFrame({"task": task, "replicate": np.arange(len(values)), **{arch: values[:, index] for index, arch in enumerate(MODEL_ORDER)}})
            for task, values in f1_replicates.items()
        ],
        ignore_index=True,
    )
    save(replicate_frame, "bootstrap_macro_f1_replicates.csv", "main_metrics")
    main_table = args.tables_dir / "rev_main_metrics.tex"
    write_main_table(main_table, summaries, seed_frames, data)
    outputs["main_metrics"].append(relative(main_table))
    status["main_metrics"] = "done"

    pairwise = {task: pairwise_differences(task, f1_replicates[task], observed_f1[task]) for task in TASKS}
    ranks = {task: rank_probabilities(task, f1_replicates[task], observed_f1[task]) for task in TASKS}
    save(pd.concat(pairwise.values(), ignore_index=True), "pairwise_differences.csv", "pairwise_comparisons")
    save(pd.concat(ranks.values(), ignore_index=True), "rank_probabilities.csv", "pairwise_comparisons")
    pairwise_table = args.tables_dir / "rev_pairwise.tex"
    leaders = write_pairwise_table(pairwise_table, pairwise, ranks, observed_f1)
    outputs["pairwise_comparisons"].append(relative(pairwise_table))
    status["pairwise_comparisons"] = "done"

    save(pd.concat([confusion_records(data[task], matrices[task]) for task in TASKS], ignore_index=True), "confusion_matrices.csv", "confusion_matrices")
    figure_paths = plot_confusions(matrices["three_stage"], args.figures_dir / "confusion_matrices")
    outputs["confusion_matrices"].extend(relative(path) for path in figure_paths)
    status["confusion_matrices"] = "done"

    operational = {task: operational_tables(data[task], matrices[task], replicates[task]) for task in TASKS}
    save(pd.concat([value[0] for value in operational.values()], ignore_index=True), "operational_rates_seed.csv", "operational_rates")
    save(pd.concat([value[1] for value in operational.values()], ignore_index=True), "operational_rates_summary.csv", "operational_rates")
    operational_table = args.tables_dir / "rev_operational.tex"
    write_operational_table(operational_table, {task: value[0] for task, value in operational.items()}, {task: value[1] for task, value in operational.items()})
    outputs["operational_rates"].append(relative(operational_table))
    status["operational_rates"] = "done"

    curves, averaged, selective_seed = risk_coverage(data["three_stage"])
    save(curves, "risk_coverage_curves.csv", "risk_coverage")
    save(averaged, "risk_coverage_seed_averaged.csv", "risk_coverage")
    save(selective_seed, "selective_seed.csv", "risk_coverage")
    selective_columns = [column for column in selective_seed.columns if column not in {"architecture", "seed"}]
    selective_summary = selective_seed.groupby("architecture", sort=False)[selective_columns].agg(["mean", "std"])
    selective_summary.columns = [f"{column}_{'sd' if statistic == 'std' else statistic}" for column, statistic in selective_summary.columns]
    save(selective_summary.reset_index(), "selective_summary.csv", "risk_coverage")
    selective_table = args.tables_dir / "rev_selective.tex"
    write_selective_table(selective_table, selective_seed)
    outputs["risk_coverage"].append(relative(selective_table))
    outputs["risk_coverage"].extend(relative(path) for path in plot_risk_coverage(averaged, args.figures_dir / "risk_coverage"))
    status["risk_coverage"] = "done"

    point_frames = []
    validation_missing: dict[str, list[str]] = {}
    for task in TASKS:
        validation, missing = load_validation(task, args.manifest)
        part = "operating_points_three_stage" if task == "three_stage" else "fresh_gate_binary"
        if missing:
            validation_missing[task] = missing
            status[part] = f"pending: {len(missing)} of {len(MODEL_ORDER) * len(SEEDS)} {task} validation prediction files ({', '.join(missing[:3])}{', ...' if len(missing) > 3 else ''})"
            print(f"WARNING: {part} pending; missing {len(missing)} validation prediction files, first {missing[0]}")
            continue
        point_frames.append(operating_points(data[task], validation, abstention=task == "three_stage"))
        status[part] = "done"
    if point_frames:
        points = pd.concat(point_frames, ignore_index=True)
        save(points, "operating_points_seed.csv", "operating_points")
        save(summarize_operating_points(points), "operating_points_summary.csv", "operating_points")
        if "three_stage" not in validation_missing:
            points_table = args.tables_dir / "rev_operating_points.tex"
            write_operating_point_table(points_table, points)
            outputs["operating_points"].append(relative(points_table))

    produce_frame = per_produce(data["three_stage"])
    save(produce_frame, "per_produce_seed.csv", "per_produce")
    produce_columns = ["macro_f1", "accuracy", "balanced_accuracy", "rotten_to_fresh"]
    produce_summary = produce_frame.groupby(["produce_type", "architecture"], sort=True).agg(
        sample_count=("sample_count", "first"),
        capture_groups=("capture_groups", "first"),
        seeds=("seed", "size"),
        **{f"{column}_{name}": (column, statistic) for column in produce_columns for name, statistic in (("mean", "mean"), ("sd", "std"))},
    )
    save(produce_summary.reset_index(), "per_produce_summary.csv", "per_produce")
    produce_table = args.tables_dir / "rev_per_produce.tex"
    write_produce_table(produce_table, produce_frame, seed_frames["three_stage"], data["three_stage"])
    outputs["per_produce"].append(relative(produce_table))
    status["per_produce"] = "done"

    reproduction, checks = frozen_checks(seed_frames, matrices, produce_frame)
    save(reproduction, "frozen_reproduction.csv", "checks")
    checks["bootstrap_helper_max_abs_difference"] = max(agreement.values())
    checks["test_leaders"] = leaders
    checks["protocol_selected"] = PROTOCOL_SELECTED
    selection_path = FROZEN / "selection/grouped_three_stage_validation_selection.csv"
    if selection_path.is_file():
        selection = pd.read_csv(selection_path)
        checks["protocol_selected_matches_frozen"] = bool(selection.loc[selection["validation_rank"] == 1, "architecture"].iloc[0] == PROTOCOL_SELECTED)
    if not checks["headline_reproduced"]:
        print("WARNING: recomputed macro-F1 does not reproduce the frozen headline numbers; see frozen_reproduction.csv")

    write_json(
        {
            "topic": TOPIC,
            "parts": status,
            "outputs": outputs,
            "checks": checks,
            "bootstrap": {"replicates": args.replicates, "seed": BOOTSTRAP_SEED, "unit": "capture_group_id", "interval": "percentile 95%"},
            "groups": {task: int(len(np.unique(data[task].groups))) for task in TASKS},
        },
        args.output_dir / "status.json",
    )

    for task in TASKS:
        summary = summaries[task].set_index("architecture")
        print(f"[{task}] seed-mean macro-F1 with 95% cluster-bootstrap CI:")
        for arch in MODEL_ORDER:
            row = summary.loc[arch]
            print(f"  {DISPLAY_NAMES[arch]:15s} {row['macro_f1_mean']:.4f} +- {row['macro_f1_sd']:.4f}  [{row['macro_f1_ci_low']:.4f}, {row['macro_f1_ci_high']:.4f}]")
    print(f"checks: {json.dumps(checks, default=str)}")
    for part, state in status.items():
        print(f"{part}: {state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
