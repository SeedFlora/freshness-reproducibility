"""Aggregate only complete, real experiment artifacts into paper-ready outputs."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fresh_or_trash.corruptions import CORRUPTIONS
from fresh_or_trash.experiments import (
    REFERENCE_SEED,
    TRAINING_SEEDS,
    compute_rank_changes,
    flatten_scalar_metrics,
    json_records,
    read_json,
    require_complete_training_runs,
    select_top_architectures,
    validate_protocol,
    validate_split_inputs,
)
from fresh_or_trash.metrics import classification_metrics
from fresh_or_trash.models import MODEL_NAMES


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


def _evaluation_dir(
    root: Path,
    setting: str,
    task: str,
    architecture: str,
    seed: int,
    calibration_state: str,
) -> Path:
    return root / "evaluation" / setting / task / architecture / f"seed-{seed}" / calibration_state


def _load_selection(results_root: Path) -> pd.DataFrame:
    path = results_root / "selection" / "grouped_three_stage_validation_selection.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    stored = pd.read_csv(path)
    runs = require_complete_training_runs(
        results_root,
        setting="grouped",
        task="three_stage",
        architectures=MODEL_NAMES,
    )
    recomputed = select_top_architectures(runs, top_k=2)
    columns = [
        "architecture",
        "validation_macro_f1_mean",
        "validation_macro_f1_std",
        "seed_count",
        "validation_rank",
    ]
    pd.testing.assert_frame_equal(
        stored[columns].reset_index(drop=True),
        recomputed[columns].reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    if stored.get("selection_uses_test_metrics", pd.Series([True])).astype(str).str.casefold().isin({"true", "1"}).any():
        raise ValueError("Selection artifact must explicitly state that test metrics were not used")
    return recomputed


def _training_records(results_root: Path, naive_architectures: Sequence[str]) -> pd.DataFrame:
    parts = []
    for task in ("three_stage", "binary"):
        parts.append(
            require_complete_training_runs(
                results_root,
                setting="grouped",
                task=task,
                architectures=MODEL_NAMES,
            )
        )
    parts.append(
        require_complete_training_runs(
            results_root,
            setting="naive",
            task="three_stage",
            architectures=naive_architectures,
        )
    )
    return pd.concat(parts, ignore_index=True)


def _collect_evaluations(
    results_root: Path,
    training_runs: pd.DataFrame,
    naive_architectures: Sequence[str],
) -> pd.DataFrame:
    specifications = [
        ("grouped", "three_stage", MODEL_NAMES, "uncalibrated"),
        ("grouped", "three_stage", MODEL_NAMES, "calibrated"),
        ("grouped", "binary", MODEL_NAMES, "uncalibrated"),
        ("naive", "three_stage", naive_architectures, "uncalibrated"),
    ]
    validation_lookup = training_runs.set_index(["setting", "task", "architecture", "seed"])[
        "validation_macro_f1"
    ].to_dict()
    records: list[dict[str, object]] = []
    for setting, task, architectures, calibration_state in specifications:
        for architecture in architectures:
            for seed in TRAINING_SEEDS:
                directory = _evaluation_dir(results_root, setting, task, architecture, seed, calibration_state)
                metrics = read_json(directory / "test_metrics.json")
                predictions = directory / "test_predictions.csv"
                if not predictions.is_file():
                    raise FileNotFoundError(predictions)
                record: dict[str, object] = {
                    "setting": setting,
                    "task": task,
                    "architecture": architecture,
                    "seed": seed,
                    "calibration_state": calibration_state,
                    "validation_macro_f1": float(validation_lookup[(setting, task, architecture, seed)]),
                }
                record.update(flatten_scalar_metrics(metrics))
                records.append(record)
    return pd.DataFrame.from_records(records)


def _dynamic_seed_summary(rows: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    metadata = {*group_columns, "seed"}
    numeric = [
        column
        for column in rows.columns
        if column not in metadata and pd.to_numeric(rows[column], errors="coerce").notna().any()
    ]
    records: list[dict[str, object]] = []
    for keys, group in rows.groupby(list(group_columns), sort=False, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        record: dict[str, object] = dict(zip(group_columns, key_values))
        record["seed_count"] = int(group["seed"].nunique())
        for metric in numeric:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                continue
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else math.nan
        records.append(record)
    return pd.DataFrame.from_records(records)


def _benchmark_rows(evaluations: pd.DataFrame) -> pd.DataFrame:
    selected = (
        ((evaluations["setting"] == "grouped") & (evaluations["task"] == "three_stage") & (evaluations["calibration_state"] == "calibrated"))
        | ((evaluations["setting"] == "grouped") & (evaluations["task"] == "binary") & (evaluations["calibration_state"] == "uncalibrated"))
        | ((evaluations["setting"] == "naive") & (evaluations["task"] == "three_stage") & (evaluations["calibration_state"] == "uncalibrated"))
    )
    return evaluations[selected].reset_index(drop=True)


def _distribution(values: pd.Series, prefix: str) -> dict[str, float]:
    numeric = pd.to_numeric(values, errors="raise")
    return {
        f"{prefix}_mean": float(numeric.mean()),
        f"{prefix}_std": float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0,
        f"{prefix}_q25": float(numeric.quantile(0.25)),
        f"{prefix}_median": float(numeric.median()),
        f"{prefix}_q75": float(numeric.quantile(0.75)),
    }


def _subgroup_record(
    predictions: pd.DataFrame,
    *,
    class_names: Sequence[str],
    ece_bins: int,
) -> dict[str, object]:
    probability_columns = [f"probability_{name.lower().replace('-', '_').replace(' ', '_')}" for name in class_names]
    missing = {"target_index", "confidence", "probability_rotten", *probability_columns} - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction artifact is missing subgroup columns: {sorted(missing)}")
    metrics = classification_metrics(
        predictions["target_index"].to_numpy(dtype=int),
        probabilities=predictions[probability_columns].to_numpy(dtype=float),
        class_names=class_names,
        ece_bins=ece_bins,
    )
    record: dict[str, object] = flatten_scalar_metrics(metrics)
    record.update(_distribution(predictions["confidence"], "confidence"))
    record.update(_distribution(predictions["probability_rotten"], "rotten_probability"))
    return record


def _shelf_bounds(label: object) -> tuple[float, float]:
    numbers = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", str(label))]
    if len(numbers) < 2:
        raise ValueError(f"Cannot parse ordered shelf-life interval: {label!r}")
    return numbers[0], numbers[1]


def _shelf_target_stage(
    rows: pd.DataFrame,
    *,
    produce_type: object,
    shelf_life_label: object,
) -> str:
    """Return the sole class-directory stage represented by a shelf interval."""

    stages = sorted(
        {
            str(value).strip()
            for value in rows["freshness_label"]
            if str(value).strip()
        }
    )
    if len(stages) != 1:
        raise ValueError(
            "Every produce + shelf-life interval must map to exactly one freshness "
            f"stage; {produce_type!r} / {shelf_life_label!r} maps to {stages!r}"
        )
    return stages[0]


def _validate_shelf_stage_mapping(test_manifest: pd.DataFrame) -> None:
    required = {"produce_type", "shelf_life_label", "freshness_label"}
    missing = required - set(test_manifest.columns)
    if missing:
        raise ValueError(f"Shelf-life aggregation manifest is missing columns: {sorted(missing)}")
    for (produce_type, shelf_label), rows in test_manifest.groupby(
        ["produce_type", "shelf_life_label"], sort=True, dropna=False
    ):
        _shelf_target_stage(
            rows,
            produce_type=produce_type,
            shelf_life_label=shelf_label,
        )


def _collect_subgroups(
    results_root: Path,
    grouped_manifest: Path,
    *,
    ece_bins: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_csv(grouped_manifest, keep_default_na=False)
    test_manifest = manifest[manifest["split"] == "test"]
    if test_manifest["shelf_life_label"].astype(str).str.strip().eq("").any():
        raise ValueError("Shelf-life aggregation requires complete class-level interval labels")
    _validate_shelf_stage_mapping(test_manifest)
    join_columns = ["sample_id", "freshness_label", "shelf_life_label"]
    produce_records: list[dict[str, object]] = []
    shelf_records: list[dict[str, object]] = []
    class_names = ["Fresh", "Semi-Fresh", "Rotten"]
    for architecture in MODEL_NAMES:
        for seed in TRAINING_SEEDS:
            predictions_path = _evaluation_dir(
                results_root, "grouped", "three_stage", architecture, seed, "calibrated"
            ) / "test_predictions.csv"
            predictions = pd.read_csv(predictions_path, keep_default_na=False)
            if predictions["sample_id"].astype(str).duplicated().any():
                raise ValueError(f"Duplicate prediction sample_id in {predictions_path}")
            merged = predictions.merge(test_manifest[join_columns], on="sample_id", how="left", validate="one_to_one")
            if len(merged) != len(test_manifest) or merged["shelf_life_label"].astype(str).str.strip().eq("").any():
                raise ValueError(f"Predictions do not cover the locked grouped test manifest: {predictions_path}")
            for produce_type, subset in merged.groupby("produce_type", sort=True):
                produce_records.append(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "produce_type": produce_type,
                        **_subgroup_record(subset, class_names=class_names, ece_bins=ece_bins),
                    }
                )
            for (produce_type, shelf_label), subset in merged.groupby(
                ["produce_type", "shelf_life_label"], sort=True
            ):
                lower, upper = _shelf_bounds(shelf_label)
                target_stage = _shelf_target_stage(
                    subset,
                    produce_type=produce_type,
                    shelf_life_label=shelf_label,
                )
                shelf_records.append(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "produce_type": produce_type,
                        "shelf_life_label": shelf_label,
                        "target_stage": target_stage,
                        "interval_lower_days": lower,
                        "interval_upper_days": upper,
                        "interval_midpoint_days": (lower + upper) / 2.0,
                        **_subgroup_record(subset, class_names=class_names, ece_bins=ece_bins),
                    }
                )
    return pd.DataFrame.from_records(produce_records), pd.DataFrame.from_records(shelf_records)


def _calibration_rows(results_root: Path, evaluations: pd.DataFrame) -> pd.DataFrame:
    indexed = evaluations.set_index(["setting", "task", "architecture", "seed", "calibration_state"])
    records: list[dict[str, object]] = []
    for architecture in MODEL_NAMES:
        for seed in TRAINING_SEEDS:
            uncalibrated = indexed.loc[("grouped", "three_stage", architecture, seed, "uncalibrated")]
            calibrated = indexed.loc[("grouped", "three_stage", architecture, seed, "calibrated")]
            temperature_path = (
                results_root
                / "runs"
                / "grouped"
                / "three_stage"
                / architecture
                / f"seed-{seed}"
                / "calibration"
                / "temperature.json"
            )
            temperature_payload = read_json(temperature_path)
            if (
                temperature_payload.get("method") != "temperature_scaling"
                or temperature_payload.get("fit_split") != "validation"
            ):
                raise ValueError(f"Calibration artifact is not validation-only temperature scaling: {temperature_path}")
            source_checkpoint = temperature_payload.get("checkpoint")
            expected_checkpoint = temperature_path.parents[1] / "best.pt"
            if not source_checkpoint or Path(str(source_checkpoint)).resolve() != expected_checkpoint.resolve():
                raise ValueError(f"Calibration artifact does not identify its source checkpoint: {temperature_path}")
            temperature = float(temperature_payload["temperature"])
            if not math.isfinite(temperature) or temperature <= 0.0:
                raise ValueError(f"Calibration temperature must be finite and positive: {temperature_path}")
            checkpoint = temperature_path.with_name("best.calibrated.pt")
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            record: dict[str, object] = {"architecture": architecture, "seed": seed, "temperature": temperature}
            for metric in ("ece", "brier", "nll", "accuracy", "macro_f1"):
                before = float(uncalibrated[metric])
                after = float(calibrated[metric])
                record[f"{metric}_uncalibrated"] = before
                record[f"{metric}_calibrated"] = after
                record[f"{metric}_calibrated_minus_uncalibrated"] = after - before
            records.append(record)
    return pd.DataFrame.from_records(records)


def _robustness_rows(results_root: Path, severities: Sequence[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    overall_records: list[dict[str, object]] = []
    expected_pairs = {(name, int(severity)) for name in CORRUPTIONS for severity in severities}
    for architecture in MODEL_NAMES:
        directory = results_root / "diagnostics" / f"reference-seed-{REFERENCE_SEED}" / architecture / "corruptions"
        frame = pd.read_csv(directory / "corruption_metrics.csv")
        observed_pairs = set(zip(frame["corruption"].astype(str), frame["severity"].astype(int)))
        if observed_pairs != expected_pairs:
            raise ValueError(f"Incomplete corruption grid for {architecture}")
        clean = read_json(directory / "test_metrics.json")
        frame.insert(0, "architecture", architecture)
        frame.insert(1, "seed", REFERENCE_SEED)
        frame["clean_macro_f1"] = float(clean["macro_f1"])
        frame["macro_f1_drop_from_clean"] = float(clean["macro_f1"]) - frame["macro_f1"].astype(float)
        rows.append(frame)
        overall_records.append(
            {
                "architecture": architecture,
                "seed": REFERENCE_SEED,
                "clean_macro_f1": float(clean["macro_f1"]),
                "mean_corrupted_macro_f1": float(frame["macro_f1"].mean()),
                "mean_macro_f1_drop_from_clean": float(frame["macro_f1_drop_from_clean"].mean()),
                "worst_corrupted_macro_f1": float(frame["macro_f1"].min()),
            }
        )
    return pd.concat(rows, ignore_index=True), pd.DataFrame.from_records(overall_records)


def _efficiency_rows(results_root: Path, *, warmup: int, iterations: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for architecture in MODEL_NAMES:
        for device in ("cpu", "cuda"):
            path = (
                results_root
                / "efficiency"
                / f"reference-seed-{REFERENCE_SEED}"
                / architecture
                / f"{device}.json"
            )
            payload = read_json(path)
            required = {
                "measurement_scope": "model_forward_pass_only",
                "includes_image_decoding": False,
                "includes_preprocessing": False,
                "architecture": architecture,
                "batch_size": 1,
                "warmup_iterations": warmup,
                "timed_iterations": iterations,
            }
            for key, expected in required.items():
                if payload.get(key) != expected:
                    raise ValueError(f"Invalid efficiency artifact {path}: {key}={payload.get(key)!r}")
            observed_device = str(payload["device"]).split(":", maxsplit=1)[0]
            if observed_device != device:
                raise ValueError(f"Expected a {device} profile in {path}; got {payload['device']}")
            record = {"architecture": architecture, "seed": REFERENCE_SEED, "profile_device": device}
            record.update(flatten_scalar_metrics(payload))
            record["measurement_scope"] = payload["measurement_scope"]
            record["includes_image_decoding"] = payload["includes_image_decoding"]
            record["includes_preprocessing"] = payload["includes_preprocessing"]
            records.append(record)
    return pd.DataFrame.from_records(records)


def _external_rows(results_root: Path, selected: Sequence[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for architecture in selected:
        directory = results_root / "external" / architecture / f"seed-{REFERENCE_SEED}"
        payload = read_json(directory / "external_metrics.json")
        if payload.get("architecture") != architecture or payload.get("fine_tuning") is not False:
            raise ValueError(f"Invalid external protocol metadata in {directory}")
        if payload.get("original_images_only") is not True or set(payload.get("produce_types", [])) != {"banana", "orange"}:
            raise ValueError(f"External evaluation is not the original-only banana/orange intersection: {directory}")
        records.append(
            {
                "architecture": architecture,
                "seed": REFERENCE_SEED,
                "scope": "aggregate",
                "produce_type": "all",
                **flatten_scalar_metrics(payload["metrics"]),
            }
        )
        per_produce = pd.read_csv(directory / "external_per_produce_metrics.csv")
        if set(per_produce["produce_type"].astype(str)) != {"banana", "orange"}:
            raise ValueError(f"External per-produce rows are incomplete for {architecture}")
        for row in per_produce.to_dict(orient="records"):
            records.append(
                {
                    "architecture": architecture,
                    "seed": REFERENCE_SEED,
                    "scope": "per_produce",
                    **row,
                }
            )
    return pd.DataFrame.from_records(records)


def _leakage_rows(benchmark_summary: pd.DataFrame, naive_architectures: Sequence[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for architecture in naive_architectures:
        grouped = benchmark_summary[
            (benchmark_summary["setting"] == "grouped")
            & (benchmark_summary["task"] == "three_stage")
            & (benchmark_summary["architecture"] == architecture)
        ].iloc[0]
        naive = benchmark_summary[
            (benchmark_summary["setting"] == "naive")
            & (benchmark_summary["task"] == "three_stage")
            & (benchmark_summary["architecture"] == architecture)
        ].iloc[0]
        records.append(
            {
                "architecture": architecture,
                "grouped_macro_f1_mean": float(grouped["macro_f1_mean"]),
                "grouped_macro_f1_std": float(grouped["macro_f1_std"]),
                "naive_macro_f1_mean": float(naive["macro_f1_mean"]),
                "naive_macro_f1_std": float(naive["macro_f1_std"]),
                "naive_minus_grouped_macro_f1": float(naive["macro_f1_mean"] - grouped["macro_f1_mean"]),
            }
        )
    return pd.DataFrame.from_records(records)


def _latex_escape(value: object) -> str:
    text = str(value).replace("_", " ")
    replacements = {"&": r"\&", "%": r"\%", "#": r"\#"}
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _mean_sd(row: pd.Series, metric: str, digits: int = 3) -> str:
    mean = float(row[f"{metric}_mean"])
    std = float(row[f"{metric}_std"])
    return rf"${mean:.{digits}f}\pm{std:.{digits}f}$"


def _table(caption: str, label: str, headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    alignment = "l" + "c" * (len(headers) - 1)
    lines = [
        "% Auto-generated from machine-readable experiment artifacts; do not edit by hand.",
        r"\begin{table*}[t]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering",
        r"\footnotesize",
        rf"\begin{{tabular}}{{@{{}}{alignment}@{{}}}}",
        r"\toprule",
        " & ".join(rf"\textbf{{{_latex_escape(header)}}}" for header in headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(" & ".join(_latex_escape(value) for value in row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    return "\n".join(lines)


def _latex_outputs(
    benchmark_summary: pd.DataFrame,
    selection: pd.DataFrame,
    rank_changes: pd.DataFrame,
    leakage: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    robustness_overall: pd.DataFrame,
    produce_summary: pd.DataFrame,
    shelf_summary: pd.DataFrame,
    external: pd.DataFrame,
    efficiency: pd.DataFrame,
) -> dict[str, str]:
    main_rows = []
    for architecture in MODEL_NAMES:
        three = benchmark_summary.query(
            "setting == 'grouped' and task == 'three_stage' and architecture == @architecture"
        ).iloc[0]
        binary = benchmark_summary.query(
            "setting == 'grouped' and task == 'binary' and architecture == @architecture"
        ).iloc[0]
        calibration = calibration_summary[calibration_summary["architecture"] == architecture].iloc[0]
        cpu = efficiency[(efficiency["architecture"] == architecture) & (efficiency["profile_device"] == "cpu")].iloc[0]
        cuda = efficiency[(efficiency["architecture"] == architecture) & (efficiency["profile_device"] == "cuda")].iloc[0]
        main_rows.append(
            (
                DISPLAY_NAMES[architecture],
                _mean_sd(three, "macro_f1"),
                _mean_sd(binary, "macro_f1"),
                _mean_sd(calibration, "ece_calibrated"),
                f"{float(cpu['state_dict_megabytes']):.1f}",
                f"{float(cpu['latency_median_ms_per_image']):.2f}",
                f"{float(cuda['latency_median_ms_per_image']):.2f}",
            )
        )
    outputs = {
        "experiment_main_benchmark_table.tex": _table(
            "Grouped Test Performance, Calibration, and Batch-One Model Forward-Pass Efficiency (mean $\\pm$ SD across seeds where applicable).",
            "tab:experiment-main-results",
            ("Model", "Three-stage F1", "Binary F1", "Cal. ECE", "State MiB", "CPU ms", "CUDA ms"),
            main_rows,
        )
    }
    outputs["experiment_rank_change_table.tex"] = _table(
        "Grouped validation rank changes between the three-stage and binary tasks.",
        "tab:experiment-rank-change",
        ("Model", "Three-stage rank", "Binary rank", "Change"),
        [
            (
                DISPLAY_NAMES[row.architecture],
                int(row.validation_three_stage_rank),
                int(row.validation_binary_rank),
                int(row.validation_binary_minus_three_stage_rank_change),
            )
            for row in rank_changes.itertuples(index=False)
        ],
    )
    outputs["experiment_leakage_ablation_table.tex"] = _table(
        "Naive-split leakage-sensitivity ablation (test macro-F1, mean $\\pm$ SD).",
        "tab:experiment-leakage",
        ("Model", "Naive F1", "Grouped F1", "Naive $-$ grouped"),
        [
            (
                DISPLAY_NAMES[row.architecture],
                rf"${row.naive_macro_f1_mean:.3f}\pm{row.naive_macro_f1_std:.3f}$",
                rf"${row.grouped_macro_f1_mean:.3f}\pm{row.grouped_macro_f1_std:.3f}$",
                f"{row.naive_minus_grouped_macro_f1:+.3f}",
            )
            for row in leakage.itertuples(index=False)
        ],
    )
    outputs["experiment_calibration_table.tex"] = _table(
        "Validation-only temperature scaling on the grouped three-stage task (test metrics, mean $\\pm$ SD).",
        "tab:experiment-calibration",
        ("Model", "Temperature", "ECE before", "ECE after", "NLL before", "NLL after"),
        [
            (
                DISPLAY_NAMES[row.architecture],
                _mean_sd(pd.Series(row._asdict()), "temperature"),
                _mean_sd(pd.Series(row._asdict()), "ece_uncalibrated"),
                _mean_sd(pd.Series(row._asdict()), "ece_calibrated"),
                _mean_sd(pd.Series(row._asdict()), "nll_uncalibrated"),
                _mean_sd(pd.Series(row._asdict()), "nll_calibrated"),
            )
            for row in calibration_summary.itertuples(index=False)
        ],
    )
    outputs["experiment_robustness_table.tex"] = _table(
        f"Corruption robustness at reference training seed {REFERENCE_SEED} across all corruption types and severities.",
        "tab:experiment-robustness",
        ("Model", "Clean F1", "Mean corrupted F1", "Mean drop", "Worst F1"),
        [
            (
                DISPLAY_NAMES[row.architecture],
                f"{row.clean_macro_f1:.3f}",
                f"{row.mean_corrupted_macro_f1:.3f}",
                f"{row.mean_macro_f1_drop_from_clean:.3f}",
                f"{row.worst_corrupted_macro_f1:.3f}",
            )
            for row in robustness_overall.itertuples(index=False)
        ],
    )
    best = str(selection.iloc[0]["architecture"])
    best_produce = produce_summary[produce_summary["architecture"] == best].sort_values("produce_type")
    outputs["experiment_produce_table.tex"] = _table(
        f"Produce-level grouped test results for validation-selected {DISPLAY_NAMES[best]} (mean $\\pm$ SD across seeds).",
        "tab:experiment-produce",
        ("Produce", "Images", "Macro-F1", "Confidence", "Rotten probability"),
        [
            (
                row.produce_type,
                f"{row.sample_count_mean:.0f}",
                _mean_sd(pd.Series(row._asdict()), "macro_f1"),
                _mean_sd(pd.Series(row._asdict()), "confidence_mean"),
                _mean_sd(pd.Series(row._asdict()), "rotten_probability_mean"),
            )
            for row in best_produce.itertuples(index=False)
        ],
    )
    best_shelf = shelf_summary[shelf_summary["architecture"] == best].sort_values(
        ["produce_type", "interval_lower_days_mean"]
    )
    outputs["experiment_shelf_interval_table.tex"] = _table(
        f"Descriptive results for folder-defined produce-specific shelf intervals using validation-selected {DISPLAY_NAMES[best]} (mean $\\pm$ SD across seeds). Each interval contains one target stage, so interval accuracy equals recall for that target stage.",
        "tab:experiment-shelf",
        ("Produce", "Folder interval", "Target stage", "Images", "Accuracy", "NLL", "Rotten probability"),
        [
            (
                row.produce_type,
                row.shelf_life_label,
                row.target_stage,
                f"{row.sample_count_mean:.0f}",
                _mean_sd(pd.Series(row._asdict()), "accuracy"),
                _mean_sd(pd.Series(row._asdict()), "nll"),
                _mean_sd(pd.Series(row._asdict()), "rotten_probability_mean"),
            )
            for row in best_shelf.itertuples(index=False)
        ],
    )
    outputs["experiment_external_validation_table.tex"] = _table(
        "Cross-dataset binary validation on original-only banana and orange images without fine-tuning.",
        "tab:experiment-external",
        ("Model", "Scope", "Images", "Accuracy", "Macro-F1", "ECE"),
        [
            (
                DISPLAY_NAMES[row.architecture],
                "aggregate" if row.scope == "aggregate" else row.produce_type,
                int(row.sample_count),
                f"{row.accuracy:.3f}",
                f"{row.macro_f1:.3f}",
                f"{row.ece:.3f}",
            )
            for row in external.itertuples(index=False)
        ],
    )
    efficiency_rows = []
    for architecture in MODEL_NAMES:
        cpu = efficiency[(efficiency["architecture"] == architecture) & (efficiency["profile_device"] == "cpu")].iloc[0]
        cuda = efficiency[(efficiency["architecture"] == architecture) & (efficiency["profile_device"] == "cuda")].iloc[0]
        efficiency_rows.append(
            (
                DISPLAY_NAMES[architecture],
                f"{float(cpu['parameter_count']) / 1_000_000:.2f}",
                f"{float(cpu['checkpoint_file_megabytes']):.1f}",
                f"{float(cpu['state_dict_megabytes']):.1f}",
                f"{float(cpu['latency_mean_ms_per_image']):.2f} / {float(cpu['latency_median_ms_per_image']):.2f} / {float(cpu['latency_p95_ms_per_image']):.2f}",
                f"{float(cuda['latency_mean_ms_per_image']):.2f} / {float(cuda['latency_median_ms_per_image']):.2f} / {float(cuda['latency_p95_ms_per_image']):.2f}",
            )
        )
    outputs["experiment_efficiency_table.tex"] = _table(
        "Reference-seed batch-one model forward-pass efficiency. Latency is mean / median / p95 and excludes image decoding and preprocessing.",
        "tab:experiment-efficiency",
        ("Model", "Parameters (M)", "Checkpoint MiB", "State MiB", "CPU ms", "CUDA ms"),
        efficiency_rows,
    )
    best_row = selection.iloc[0]
    binary_best = (
        benchmark_summary[benchmark_summary["task"] == "binary"]
        .sort_values("validation_macro_f1_mean", ascending=False)
        .iloc[0]
    )
    outputs["experiment_results_values.tex"] = "\n".join(
        [
            "% Auto-generated from validation-gated experiment artifacts; do not edit by hand.",
            rf"\newcommand{{\ExperimentBestGroupedModel}}{{{DISPLAY_NAMES[str(best_row['architecture'])]}}}",
            rf"\newcommand{{\ExperimentBestGroupedValidationFOne}}{{{float(best_row['validation_macro_f1_mean']):.3f}}}",
            rf"\newcommand{{\ExperimentBestBinaryModel}}{{{DISPLAY_NAMES[str(binary_best['architecture'])]}}}",
            rf"\newcommand{{\ExperimentReferenceSeed}}{{{REFERENCE_SEED}}}",
            "",
        ]
    )
    return outputs


def _plot_outputs(
    figure_dir: Path,
    benchmark_summary: pd.DataFrame,
    calibration_summary: pd.DataFrame,
    robustness: pd.DataFrame,
    leakage: pd.DataFrame,
    external: pd.DataFrame,
    shelf_summary: pd.DataFrame,
    best_architecture: str,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=False)
    architecture_order = list(MODEL_NAMES)
    positions = np.arange(len(architecture_order))

    fig, axis = plt.subplots(figsize=(8.4, 4.5))
    width = 0.36
    for offset, task, label in ((-width / 2, "three_stage", "Three-stage"), (width / 2, "binary", "Binary")):
        task_rows = (
            benchmark_summary[(benchmark_summary["setting"] == "grouped") & (benchmark_summary["task"] == task)]
            .set_index("architecture")
            .reindex(architecture_order)
        )
        axis.bar(
            positions + offset,
            task_rows["macro_f1_mean"],
            width,
            yerr=task_rows["macro_f1_std"],
            capsize=3,
            label=label,
        )
    axis.set_xticks(positions, [DISPLAY_NAMES[value] for value in architecture_order], rotation=18, ha="right")
    axis.set_ylabel("Locked-test macro-F1")
    axis.set_ylim(0, 1)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.4, 4.5))
    before = calibration_summary.set_index("architecture").reindex(architecture_order)
    axis.bar(positions - width / 2, before["ece_uncalibrated_mean"], width, label="Uncalibrated")
    axis.bar(positions + width / 2, before["ece_calibrated_mean"], width, label="Temperature-scaled")
    axis.set_xticks(positions, [DISPLAY_NAMES[value] for value in architecture_order], rotation=18, ha="right")
    axis.set_ylabel("Expected calibration error")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "calibration_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    severity_means = robustness.groupby(["architecture", "severity"], as_index=False)["macro_f1"].mean()
    for architecture in architecture_order:
        subset = severity_means[severity_means["architecture"] == architecture]
        axis.plot(subset["severity"], subset["macro_f1"], marker="o", label=DISPLAY_NAMES[architecture], color=COLORS[architecture])
    axis.set_xlabel("Corruption severity")
    axis.set_ylabel("Mean macro-F1 across corruption types")
    axis.set_xticks(sorted(robustness["severity"].unique()))
    axis.set_ylim(0, 1)
    axis.legend(frameon=False, ncol=2)
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "corruption_robustness.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.8, 4.3))
    x = np.arange(len(leakage))
    axis.bar(x - width / 2, leakage["grouped_macro_f1_mean"], width, yerr=leakage["grouped_macro_f1_std"], label="Grouped", capsize=3)
    axis.bar(x + width / 2, leakage["naive_macro_f1_mean"], width, yerr=leakage["naive_macro_f1_std"], label="Naive", capsize=3)
    axis.set_xticks(x, [DISPLAY_NAMES[value] for value in leakage["architecture"]])
    axis.set_ylabel("Locked-test macro-F1")
    axis.set_ylim(0, 1)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "leakage_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    per_produce = external[external["scope"] == "per_produce"]
    selected_architectures = list(dict.fromkeys(per_produce["architecture"]))
    fruits = ["banana", "orange"]
    fig, axis = plt.subplots(figsize=(6.8, 4.3))
    for index, architecture in enumerate(selected_architectures):
        values = per_produce[per_produce["architecture"] == architecture].set_index("produce_type").reindex(fruits)
        axis.bar(np.arange(len(fruits)) + (index - 0.5) * width, values["macro_f1"], width, label=DISPLAY_NAMES[architecture])
    axis.set_xticks(np.arange(len(fruits)), [value.title() for value in fruits])
    axis.set_ylabel("External macro-F1")
    axis.set_ylim(0, 1)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "external_per_fruit.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    selected_shelf = shelf_summary[shelf_summary["architecture"] == best_architecture]
    produce_types = sorted(selected_shelf["produce_type"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.8), sharey=True, constrained_layout=True)
    for axis, produce_type in zip(axes.flat, produce_types):
        subset = selected_shelf[selected_shelf["produce_type"] == produce_type].sort_values("interval_lower_days_mean")
        positions = np.arange(len(subset))
        stage_colors = {
            "Fresh": "#54A24B",
            "Semi-Fresh": "#F2CF5B",
            "Rotten": "#E45756",
        }
        axis.bar(
            positions,
            subset["rotten_probability_mean_mean"],
            yerr=subset["rotten_probability_mean_std"],
            capsize=3,
            color=[stage_colors.get(str(stage), "#9D9D9D") for stage in subset["target_stage"]],
        )
        axis.set_title(produce_type.replace("_", " ").title())
        axis.set_xticks(
            positions,
            [
                f"{interval_label}\n({target_stage})"
                for interval_label, target_stage in zip(subset["shelf_life_label"], subset["target_stage"])
            ],
            rotation=20,
            ha="right",
        )
        axis.set_xlabel("Folder-defined interval")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.2)
    for axis in axes.flat[len(produce_types):]:
        axis.set_visible(False)
    axes[0, 0].set_ylabel("Mean calibrated Rotten probability")
    axes[1, 0].set_ylabel("Mean calibrated Rotten probability")
    fig.suptitle(
        f"Descriptive folder-defined shelf intervals: {DISPLAY_NAMES[best_architecture]}"
    )
    fig.savefig(figure_dir / "shelf_interval_rotten_probability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _reserve_outputs(paths: Iterable[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"Aggregation refuses to overwrite existing output: {existing[0]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate the complete frozen experiment grid")
    parser.add_argument("--protocol", type=Path, default=Path("config/experiments.yaml"))
    parser.add_argument("--grouped-manifest", type=Path, default=Path("data/splits/grouped_manifest.csv"))
    parser.add_argument("--naive-manifest", type=Path, default=Path("data/splits/naive_manifest.csv"))
    parser.add_argument("--split-provenance", type=Path, default=Path("data/splits/split_provenance.json"))
    parser.add_argument("--results-root", type=Path, default=Path("results/experiments"))
    parser.add_argument("--paper-generated-dir", type=Path, default=Path("paper/generated"))
    parser.add_argument("--figure-dir", type=Path, default=Path("paper/figures/results"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = validate_protocol(args.protocol)
    validate_split_inputs(args.grouped_manifest, args.naive_manifest, args.split_provenance)
    results_root = args.results_root
    selection = _load_selection(results_root)
    best_architecture = str(selection.iloc[0]["architecture"])
    selected_external = selection.head(2)["architecture"].astype(str).tolist()
    naive_architectures = list(dict.fromkeys(("mobilenet_v2", best_architecture)))

    training_runs = _training_records(results_root, naive_architectures)
    evaluations = _collect_evaluations(results_root, training_runs, naive_architectures)
    benchmark_seed = _benchmark_rows(evaluations)
    benchmark_summary = _dynamic_seed_summary(
        benchmark_seed,
        ("setting", "task", "calibration_state", "architecture"),
    )
    grouped_summary = benchmark_summary[benchmark_summary["setting"] == "grouped"]
    validation_rank_change = compute_rank_changes(grouped_summary, metric="validation_macro_f1_mean")
    validation_rank_change = validation_rank_change.rename(
        columns={
            "three_stage_rank": "validation_three_stage_rank",
            "binary_rank": "validation_binary_rank",
            "binary_minus_three_stage_rank_change": "validation_binary_minus_three_stage_rank_change",
        }
    )
    test_rank_change = compute_rank_changes(grouped_summary, metric="macro_f1_mean").rename(
        columns={
            "three_stage_rank": "test_three_stage_rank",
            "binary_rank": "test_binary_rank",
            "binary_minus_three_stage_rank_change": "test_binary_minus_three_stage_rank_change",
        }
    )
    rank_changes = validation_rank_change.merge(test_rank_change, on="architecture", validate="one_to_one")
    leakage = _leakage_rows(benchmark_summary, naive_architectures)

    ece_bins = int(protocol["evaluation"]["ece_bins"])
    produce_seed, shelf_seed = _collect_subgroups(results_root, args.grouped_manifest, ece_bins=ece_bins)
    produce_summary = _dynamic_seed_summary(produce_seed, ("architecture", "produce_type"))
    shelf_summary = _dynamic_seed_summary(
        shelf_seed,
        ("architecture", "produce_type", "shelf_life_label", "target_stage"),
    )
    calibration_seed = _calibration_rows(results_root, evaluations)
    calibration_summary = _dynamic_seed_summary(calibration_seed, ("architecture",))
    severities = [int(value) for value in protocol["evaluation"]["corruption_severities"]]
    robustness, robustness_overall = _robustness_rows(results_root, severities)
    efficiency = _efficiency_rows(
        results_root,
        warmup=int(protocol["evaluation"]["latency_warmup"]),
        iterations=int(protocol["evaluation"]["latency_iterations"]),
    )
    external = _external_rows(results_root, selected_external)

    aggregate_dir = results_root / "aggregate"
    csv_outputs = {
        "training_runs.csv": training_runs,
        "evaluation_seed_metrics.csv": evaluations,
        "benchmark_seed_metrics.csv": benchmark_seed,
        "benchmark_summary.csv": benchmark_summary,
        "rank_changes.csv": rank_changes,
        "leakage_ablation.csv": leakage,
        "per_produce_seed_metrics.csv": produce_seed,
        "per_produce_summary.csv": produce_summary,
        "shelf_interval_seed_metrics.csv": shelf_seed,
        "shelf_interval_summary.csv": shelf_summary,
        "calibration_seed_metrics.csv": calibration_seed,
        "calibration_summary.csv": calibration_summary,
        "corruption_metrics.csv": robustness,
        "corruption_summary.csv": robustness_overall,
        "external_metrics.csv": external,
        "efficiency.csv": efficiency,
        "validation_selection.csv": selection,
    }
    latex_outputs = _latex_outputs(
        benchmark_summary,
        selection,
        rank_changes,
        leakage,
        calibration_summary,
        robustness_overall,
        produce_summary,
        shelf_summary,
        external,
        efficiency,
    )
    summary_json = aggregate_dir / "experiment_summary.json"
    final_paths = [summary_json, *(aggregate_dir / name for name in csv_outputs), *(args.paper_generated_dir / name for name in latex_outputs)]
    if args.figure_dir.exists():
        final_paths.append(args.figure_dir)
    _reserve_outputs(final_paths)

    aggregate_dir.mkdir(parents=True, exist_ok=False)
    for filename, frame in csv_outputs.items():
        frame.to_csv(aggregate_dir / filename, index=False, lineterminator="\n")
    payload: dict[str, Any] = {
        "protocol": {
            "split_seed": int(protocol["experiment"]["split_seed"]),
            "training_seeds": list(TRAINING_SEEDS),
            "reference_seed": REFERENCE_SEED,
            "selection_metric": "mean grouped three-stage validation macro-F1",
            "test_metrics_used_for_selection": False,
            "image_size": int(protocol["data"]["image_size"]),
            "batch_size": int(protocol["data"]["batch_size"]),
            "max_epochs": int(protocol["training"]["epochs"]),
            "head_warmup_epochs": int(protocol["training"]["head_warmup_epochs"]),
            "early_stopping_patience": int(protocol["training"]["patience"]),
            "amp": bool(protocol["training"]["amp"]),
            "auxiliary_produce_head": bool(protocol["model"]["auxiliary_produce_head"]),
        },
        "validation_selection": json_records(selection),
        "benchmark_summary": json_records(benchmark_summary),
        "rank_changes": json_records(rank_changes),
        "leakage_ablation": json_records(leakage),
        "per_produce_summary": json_records(produce_summary),
        "shelf_interval_summary": json_records(shelf_summary),
        "calibration_summary": json_records(calibration_summary),
        "corruption_summary": json_records(robustness_overall),
        "external_metrics": json_records(external),
        "efficiency": json_records(efficiency),
    }
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    args.paper_generated_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in latex_outputs.items():
        (args.paper_generated_dir / filename).write_text(content, encoding="utf-8")
    _plot_outputs(
        args.figure_dir,
        benchmark_summary,
        calibration_summary,
        robustness,
        leakage,
        external,
        shelf_summary,
        best_architecture,
    )
    print(f"aggregated real experiment artifacts -> {aggregate_dir}")
    print(f"generated LaTeX -> {args.paper_generated_dir}; figures -> {args.figure_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
