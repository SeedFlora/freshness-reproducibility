"""Shared helpers for the peer-review revision analyses.

Every revision script imports this module so model order, display names,
prediction alignment, cluster-bootstrap resampling, post-hoc temperature
scaling, LaTeX tables, and figure styling stay identical across outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import f1_score

from fresh_or_trash.metrics import classification_metrics, expected_calibration_error


ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "results/experiments"
REVISION = ROOT / "results/revision"
ANALYSIS = REVISION / "analysis"
PAPER_TABLES = ROOT / "paper/generated/revision"
PAPER_FIGURES = ROOT / "paper/figures/revision"
GROUPED_MANIFEST = ROOT / "data/splits/grouped_manifest.csv"
NAIVE_MANIFEST = ROOT / "data/splits/naive_manifest.csv"

MODEL_ORDER = ("mobilenet_v2", "resnet50", "densenet121", "efficientnet_b0", "convnext_tiny")
DISPLAY_NAMES = {
    "mobilenet_v2": "MobileNetV2",
    "resnet50": "ResNet50",
    "densenet121": "DenseNet121",
    "efficientnet_b0": "EfficientNetB0",
    "convnext_tiny": "ConvNeXt-Tiny",
}
COLORS = {
    "mobilenet_v2": "#4C78A8",
    "resnet50": "#F58518",
    "densenet121": "#54A24B",
    "efficientnet_b0": "#E45756",
    "convnext_tiny": "#7A5195",
}
SEEDS = (2026, 3407, 777)
REFERENCE_SEED = 2026
THREE_STAGE = ("Fresh", "Semi-Fresh", "Rotten")
BINARY = ("Fresh", "Rotten")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 2026


def class_names(task: str) -> tuple[str, ...]:
    return THREE_STAGE if task == "three_stage" else BINARY


def probability_columns(task: str) -> list[str]:
    return [f"probability_{name.lower().replace('-', '_')}" for name in class_names(task)]


def read_predictions(path: str | Path, task: str) -> pd.DataFrame:
    """Load a prediction CSV and check it carries one probability per class."""
    frame = pd.read_csv(path)
    missing = {"sample_id", "target_index", *probability_columns(task)} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks columns {sorted(missing)}")
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"{path} repeats sample IDs")
    return frame


def frozen_test_predictions(task: str, arch: str, seed: int, *, setting: str = "grouped") -> pd.DataFrame:
    """Locked-test predictions of the frozen protocol (calibrated for three-stage)."""
    kind = "calibrated" if (task == "three_stage" and setting == "grouped") else "uncalibrated"
    path = FROZEN / "evaluation" / setting / task / arch / f"seed-{seed}" / kind / "test_predictions.csv"
    return read_predictions(path, task)


def revision_validation_predictions(task: str, arch: str, seed: int) -> pd.DataFrame:
    kind = "calibrated" if task == "three_stage" else "uncalibrated"
    path = REVISION / "predictions/grouped" / task / arch / f"seed-{seed}" / kind / "validation_predictions.csv"
    return read_predictions(path, task)


def stack_seeds(frames: Sequence[pd.DataFrame], task: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Align per-seed predictions on sample ID.

    Returns the sample order, integer targets [N], and probabilities [S, N, C].
    """
    if not frames:
        raise ValueError("No prediction frames supplied")
    order = sorted(frames[0]["sample_id"].astype(str))
    columns = probability_columns(task)
    targets = None
    stacked = []
    for frame in frames:
        indexed = frame.set_index(frame["sample_id"].astype(str)).loc[order]
        if set(indexed.index) != set(order) or len(indexed) != len(order):
            raise ValueError("Seed prediction files cover different samples")
        current = indexed["target_index"].to_numpy(dtype=int)
        if targets is not None and not np.array_equal(targets, current):
            raise ValueError("Seed prediction files disagree on targets")
        targets = current
        probabilities = indexed[columns].to_numpy(dtype=float)
        stacked.append(probabilities / probabilities.sum(axis=1, keepdims=True))
    assert targets is not None
    return order, targets, np.stack(stacked)


def macro_f1(targets: np.ndarray, predictions: np.ndarray, class_count: int, weights: np.ndarray | None = None) -> float:
    return float(
        f1_score(targets, predictions, labels=list(range(class_count)), average="macro", zero_division=0, sample_weight=weights)
    )


def capture_groups(sample_ids: Sequence[str], manifest: Path = GROUPED_MANIFEST) -> np.ndarray:
    """Capture-group ID for each sample, the resampling unit of the cluster bootstrap."""
    frame = pd.read_csv(manifest, usecols=["sample_id", "capture_group_id"])
    mapping = dict(zip(frame["sample_id"].astype(str), frame["capture_group_id"].astype(str)))
    missing = [value for value in sample_ids if value not in mapping]
    if missing:
        raise KeyError(f"{len(missing)} samples have no capture group; first {missing[0]}")
    return np.asarray([mapping[value] for value in sample_ids])


def cluster_bootstrap(
    groups: np.ndarray,
    statistic: Callable[[np.ndarray], float | np.ndarray],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    """Resample whole capture groups with replacement and evaluate ``statistic``.

    ``statistic`` receives an index array into the original samples (members
    of a group drawn twice appear twice) and returns a scalar or vector. The
    same generator seed gives paired replicates across calls, so differences
    between models can be formed replicate by replicate.
    """
    unique, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(len(unique))]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(replicates):
        drawn = rng.integers(0, len(unique), size=len(unique))
        indices = np.concatenate([members[index] for index in drawn])
        values.append(statistic(indices))
    return np.asarray(values, dtype=float)


def percentile_interval(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0 * 100.0
    return float(np.percentile(values, tail)), float(np.percentile(values, 100.0 - tail))


def seed_mean_macro_f1(targets: np.ndarray, probabilities: np.ndarray) -> Callable[[np.ndarray], float]:
    """Statistic for the bootstrap: macro-F1 per seed on the resample, averaged over seeds."""
    class_count = probabilities.shape[-1]
    predictions = probabilities.argmax(axis=-1)

    def statistic(indices: np.ndarray) -> float:
        return float(np.mean([macro_f1(targets[indices], seed_predictions[indices], class_count) for seed_predictions in predictions]))

    return statistic


def fit_temperature(probabilities: np.ndarray, targets: np.ndarray) -> float:
    """Fit one temperature by validation NLL on saved probabilities.

    Softmax probabilities identify logits up to a per-sample constant, which
    cancels after division by T, so softmax(log(p) / T) equals scaling the
    original logits.
    """
    log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))

    def nll(log_temperature: float) -> float:
        scaled = apply_temperature_log(log_probabilities, float(np.exp(log_temperature)))
        return float(-np.mean(np.log(np.clip(scaled[np.arange(len(targets)), targets], 1e-12, 1.0))))

    result = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(result.x))


def apply_temperature_log(log_probabilities: np.ndarray, temperature: float) -> np.ndarray:
    scaled = log_probabilities / temperature
    scaled -= scaled.max(axis=-1, keepdims=True)
    exponentials = np.exp(scaled)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    return apply_temperature_log(np.log(np.clip(probabilities, 1e-12, 1.0)), temperature)


def metrics(targets: np.ndarray, probabilities: np.ndarray, task: str) -> dict[str, object]:
    return classification_metrics(targets, probabilities=probabilities, class_names=list(class_names(task)))


def ece(targets: np.ndarray, probabilities: np.ndarray) -> float:
    return expected_calibration_error(probabilities, targets, bins=15)


def mean_sd(values: Iterable[float], digits: int = 3) -> str:
    array = np.asarray(list(values), dtype=float)
    if len(array) == 1:
        return f"{array[0]:.{digits}f}"
    return rf"${array.mean():.{digits}f}\pm{array.std(ddof=1):.{digits}f}$"


def interval(low: float, high: float, digits: int = 3) -> str:
    return f"[{low:.{digits}f}, {high:.{digits}f}]"


def latex_table(
    path: str | Path,
    *,
    caption: str,
    label: str,
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
    wide: bool = False,
    alignment: str | None = None,
    notes: str | None = None,
) -> None:
    """Write a booktabs table fragment; cells are inserted verbatim (pre-formatted LaTeX)."""
    environment = "table*" if wide else "table"
    columns = alignment or ("l" + "c" * (len(headers) - 1))
    lines = [
        "% Auto-generated by scripts/revision; do not edit by hand.",
        rf"\begin{{{environment}}}[t]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\begin{{tabular}}{{@{{}}{columns}@{{}}}}",
        r"\toprule",
        " & ".join(rf"\textbf{{{header}}}" for header in headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(str(cell) for cell in row) + r" \\" for row in rows)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if notes:
        lines.append(rf"\par\smallskip\parbox{{\linewidth}}{{\scriptsize {notes}}}")
    lines.append(rf"\end{{{environment}}}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def figure_style() -> None:
    """IEEE-sized figure defaults: 8 pt text, light grid, 300 dpi output."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


SINGLE_COLUMN_WIDTH = 3.5
DOUBLE_COLUMN_WIDTH = 7.16


def write_json(payload: object, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
