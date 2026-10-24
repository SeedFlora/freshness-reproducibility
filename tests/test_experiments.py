from __future__ import annotations

import math

import pandas as pd
import pytest

from fresh_or_trash.experiments import (
    TRAINING_SEEDS,
    compute_rank_changes,
    select_top_architectures,
    summarize_seed_metrics,
    validate_protocol,
)
from fresh_or_trash.models import MODEL_NAMES
from scripts.aggregate_experiments import _dynamic_seed_summary, _validate_shelf_stage_mapping


def _selection_rows(scores: dict[str, tuple[float, float, float]]) -> pd.DataFrame:
    rows = []
    for architecture in MODEL_NAMES:
        for seed, score in zip(TRAINING_SEEDS, scores[architecture]):
            rows.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "validation_macro_f1": score,
                    # A tempting locked-test column must have no effect on selection.
                    "test_macro_f1": 1.0 if architecture == "mobilenet_v2" else 0.0,
                }
            )
    return pd.DataFrame.from_records(rows)


def test_frozen_protocol_has_controlled_budget_and_seeds() -> None:
    protocol = validate_protocol("config/experiments.yaml")
    assert protocol["experiment"]["split_seed"] == 2026
    assert protocol["experiment"]["training_seeds"] == [2026, 3407, 777]
    assert protocol["training"]["epochs"] == 12
    assert protocol["training"]["head_warmup_epochs"] == 1
    assert protocol["training"]["patience"] == 3
    assert protocol["training"]["amp"] is True
    assert protocol["data"]["image_size"] == 224
    assert protocol["data"]["batch_size"] == 24
    assert protocol["model"]["auxiliary_produce_head"] is True


def test_selection_uses_mean_validation_macro_f1_not_test_metrics() -> None:
    scores = {
        "mobilenet_v2": (0.70, 0.71, 0.69),
        "resnet50": (0.80, 0.81, 0.79),
        "densenet121": (0.76, 0.77, 0.75),
        "efficientnet_b0": (0.85, 0.84, 0.86),
        "convnext_tiny": (0.82, 0.83, 0.81),
    }
    selected = select_top_architectures(_selection_rows(scores), top_k=2)
    assert selected.iloc[0]["architecture"] == "efficientnet_b0"
    assert selected.head(2)["architecture"].tolist() == ["efficientnet_b0", "convnext_tiny"]
    assert not selected["selection_uses_test_metrics"].any()


def test_selection_rejects_an_incomplete_matched_seed_grid() -> None:
    scores = {architecture: (0.7, 0.7, 0.7) for architecture in MODEL_NAMES}
    rows = _selection_rows(scores)
    rows = rows[~((rows["architecture"] == "resnet50") & (rows["seed"] == 777))]
    with pytest.raises(ValueError, match="required"):
        select_top_architectures(rows)


def test_selection_tie_break_is_seed_sd_then_registry_order() -> None:
    scores = {architecture: (0.8, 0.8, 0.8) for architecture in MODEL_NAMES}
    selected = select_top_architectures(_selection_rows(scores))
    assert selected["architecture"].tolist() == list(MODEL_NAMES)


def test_seed_aggregation_uses_sample_standard_deviation() -> None:
    rows = pd.DataFrame(
        [
            {"architecture": "a", "seed": 2026, "macro_f1": 0.7, "ece": 0.12},
            {"architecture": "a", "seed": 3407, "macro_f1": 0.8, "ece": 0.10},
            {"architecture": "a", "seed": 777, "macro_f1": 0.9, "ece": 0.08},
        ]
    )
    summary = summarize_seed_metrics(
        rows,
        group_columns=("architecture",),
        metric_columns=("macro_f1", "ece"),
    ).iloc[0]
    assert summary["seed_count"] == 3
    assert summary["macro_f1_mean"] == pytest.approx(0.8)
    assert summary["macro_f1_std"] == pytest.approx(0.1)
    assert summary["ece_mean"] == pytest.approx(0.1)
    assert not math.isnan(summary["ece_std"])


def test_seed_aggregation_rejects_duplicate_seed_weighting() -> None:
    rows = pd.DataFrame(
        [
            {"architecture": "a", "seed": 2026, "macro_f1": 0.7},
            {"architecture": "a", "seed": 2026, "macro_f1": 0.8},
        ]
    )
    with pytest.raises(ValueError, match="Duplicate seed"):
        summarize_seed_metrics(rows, group_columns=("architecture",), metric_columns=("macro_f1",))


def test_rank_change_sign_is_binary_minus_three_stage() -> None:
    rows = []
    for index, architecture in enumerate(MODEL_NAMES):
        rows.append({"architecture": architecture, "task": "three_stage", "score": 5 - index})
        rows.append({"architecture": architecture, "task": "binary", "score": index + 1})
    changes = compute_rank_changes(pd.DataFrame(rows), metric="score").set_index("architecture")
    assert changes.loc["mobilenet_v2", "three_stage_rank"] == 1
    assert changes.loc["mobilenet_v2", "binary_rank"] == 5
    assert changes.loc["mobilenet_v2", "binary_minus_three_stage_rank_change"] == 4


def test_shelf_intervals_map_to_exactly_one_target_stage() -> None:
    manifest = pd.DataFrame(
        [
            {"produce_type": "banana", "shelf_life_label": "1-4 days", "freshness_label": "Fresh"},
            {"produce_type": "banana", "shelf_life_label": "1-4 days", "freshness_label": "Fresh"},
            {"produce_type": "banana", "shelf_life_label": "4-7 days", "freshness_label": "Semi-Fresh"},
        ]
    )
    _validate_shelf_stage_mapping(manifest)


def test_shelf_intervals_reject_mixed_target_stages() -> None:
    manifest = pd.DataFrame(
        [
            {"produce_type": "banana", "shelf_life_label": "1-4 days", "freshness_label": "Fresh"},
            {"produce_type": "banana", "shelf_life_label": "1-4 days", "freshness_label": "Rotten"},
        ]
    )
    with pytest.raises(ValueError, match="exactly one freshness stage"):
        _validate_shelf_stage_mapping(manifest)


def test_shelf_seed_summary_preserves_target_stage() -> None:
    rows = pd.DataFrame(
        [
            {
                "architecture": "a",
                "produce_type": "banana",
                "shelf_life_label": "1-4 days",
                "target_stage": "Fresh",
                "seed": 2026,
                "accuracy": 0.8,
            },
            {
                "architecture": "a",
                "produce_type": "banana",
                "shelf_life_label": "1-4 days",
                "target_stage": "Fresh",
                "seed": 3407,
                "accuracy": 0.9,
            },
        ]
    )
    summary = _dynamic_seed_summary(
        rows,
        ("architecture", "produce_type", "shelf_life_label", "target_stage"),
    )
    assert summary.loc[0, "target_stage"] == "Fresh"
    assert summary.loc[0, "accuracy_mean"] == pytest.approx(0.85)
