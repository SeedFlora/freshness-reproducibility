from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def expected_calibration_error(probabilities: np.ndarray, targets: np.ndarray, bins: int = 15) -> float:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = predictions == targets
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        in_bin = (confidences > lower) & (confidences <= upper)
        if lower == 0.0:
            in_bin |= confidences == 0.0
        if not np.any(in_bin):
            continue
        bin_accuracy = correctness[in_bin].mean()
        bin_confidence = confidences[in_bin].mean()
        ece += float(in_bin.mean() * abs(bin_accuracy - bin_confidence))
    return float(ece)


def multiclass_brier_score(probabilities: np.ndarray, targets: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1], dtype=float)[targets]
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def classification_metrics(
    targets: np.ndarray | Sequence[int],
    *,
    logits: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
    class_names: Sequence[str] | None = None,
    ece_bins: int = 15,
) -> dict[str, object]:
    targets_array = np.asarray(targets, dtype=int)
    if probabilities is None:
        if logits is None:
            raise ValueError("Provide logits or probabilities")
        probabilities = softmax(np.asarray(logits, dtype=float))
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or len(probabilities) != len(targets_array):
        raise ValueError("probabilities must have shape [samples, classes]")
    probability_sums = probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all() or np.any(probability_sums <= 0.0):
        raise ValueError("probabilities must be finite and have a positive row sum")
    # Torch softmax values are commonly materialized as float32.  Renormalize
    # after conversion so downstream sklearn checks are not tripped by harmless
    # floating-point row-sum drift.
    probabilities = probabilities / probability_sums
    predictions = probabilities.argmax(axis=1)
    labels = list(range(probabilities.shape[1]))
    names = list(class_names) if class_names else [str(label) for label in labels]
    if len(names) != len(labels):
        raise ValueError("class_names length must equal probability columns")
    report = classification_report(
        targets_array,
        predictions,
        labels=labels,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    result: dict[str, object] = {
        "sample_count": int(len(targets_array)),
        "accuracy": float(accuracy_score(targets_array, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(targets_array, predictions)),
        "macro_f1": float(f1_score(targets_array, predictions, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets_array, predictions, labels=labels, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(targets_array, predictions, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(targets_array, predictions, labels=labels, average="macro", zero_division=0)),
        "ece": expected_calibration_error(probabilities, targets_array, bins=ece_bins),
        "brier": multiclass_brier_score(probabilities, targets_array),
        "nll": float(log_loss(targets_array, probabilities, labels=labels)),
        "confusion_matrix": confusion_matrix(targets_array, predictions, labels=labels).tolist(),
        "per_class": {name: report[name] for name in names},
    }
    try:
        if probabilities.shape[1] == 2:
            result["roc_auc"] = float(roc_auc_score(targets_array, probabilities[:, 1]))
        else:
            result["roc_auc_ovr_macro"] = float(
                roc_auc_score(targets_array, probabilities, labels=labels, multi_class="ovr", average="macro")
            )
    except ValueError:
        # Undefined when an evaluation subset lacks one or more classes.
        pass
    return result
