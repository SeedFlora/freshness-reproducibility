from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import load_config
from .models import MODEL_NAMES
from .splits import assert_no_group_leakage


SPLIT_SEED = 2026
TRAINING_SEEDS = (2026, 3407, 777)
REFERENCE_SEED = 2026
IMAGE_SIZE = 224
BATCH_SIZE = 24
MAX_EPOCHS = 12
HEAD_WARMUP_EPOCHS = 1
EARLY_STOPPING_PATIENCE = 3
SELECTION_METRIC = "validation_macro_f1"
EXTERNAL_PRODUCE_TYPES = ("banana", "orange")


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_protocol(path: str | Path) -> dict[str, Any]:
    """Load the paper protocol and reject any silent budget/seed drift."""

    config = load_config(path)
    expected: dict[tuple[str, ...], object] = {
        ("seed",): SPLIT_SEED,
        ("experiment", "split_seed"): SPLIT_SEED,
        ("experiment", "training_seeds"): list(TRAINING_SEEDS),
        ("experiment", "reference_seed"): REFERENCE_SEED,
        ("experiment", "selection_metric"): SELECTION_METRIC,
        ("experiment", "selection_direction"): "maximize",
        ("experiment", "top_external_architectures"): 2,
        ("experiment", "external_produce_types"): list(EXTERNAL_PRODUCE_TYPES),
        ("data", "image_size"): IMAGE_SIZE,
        ("data", "batch_size"): BATCH_SIZE,
        ("model", "names"): list(MODEL_NAMES),
        ("model", "pretrained"): True,
        ("model", "auxiliary_produce_head"): True,
        ("training", "epochs"): MAX_EPOCHS,
        ("training", "head_warmup_epochs"): HEAD_WARMUP_EPOCHS,
        ("training", "amp"): True,
        ("training", "patience"): EARLY_STOPPING_PATIENCE,
        ("training", "monitor"): SELECTION_METRIC,
        ("augmentation", "implementation"): "fresh_or_trash.data.build_transform",
        ("evaluation", "latency_batch_size"): 1,
        ("evaluation", "latency_devices"): ["cpu", "cuda"],
    }
    mismatches: list[str] = []
    for key_path, required in expected.items():
        value: Any = config
        for key in key_path:
            if not isinstance(value, Mapping) or key not in value:
                value = None
                break
            value = value[key]
        if value != required:
            mismatches.append(f"{'.'.join(key_path)}={value!r} (required {required!r})")
    if mismatches:
        raise ValueError("Controlled experiment protocol drift:\n- " + "\n- ".join(mismatches))
    return config


def _manifest_hash_from_provenance(payload: Mapping[str, Any], setting: str) -> str | None:
    manifests = payload.get("manifests", {})
    value: Any = manifests.get(setting) if isinstance(manifests, Mapping) else None
    if value is None:
        value = payload.get(f"{setting}_manifest")
    if isinstance(value, Mapping):
        value = value.get("sha256")
    if value is None:
        value = payload.get(f"{setting}_manifest_sha256")
    return str(value).lower() if value else None


def validate_split_inputs(
    grouped_manifest: str | Path,
    naive_manifest: str | Path,
    split_provenance: str | Path,
) -> dict[str, Any]:
    """Verify the locked split seed, hashes, row identity, and grouped integrity."""

    grouped_path = Path(grouped_manifest)
    naive_path = Path(naive_manifest)
    provenance_path = Path(split_provenance)
    for path in (grouped_path, naive_path, provenance_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    if int(payload.get("split_seed", -1)) != SPLIT_SEED:
        raise ValueError(f"Split provenance must record split_seed={SPLIT_SEED}")
    observed_hashes = {
        "grouped": file_sha256(grouped_path),
        "naive": file_sha256(naive_path),
    }
    for setting, observed in observed_hashes.items():
        recorded = _manifest_hash_from_provenance(payload, setting)
        if not recorded:
            raise ValueError(f"Split provenance has no SHA-256 for {setting}_manifest")
        if recorded != observed:
            raise ValueError(f"{setting} manifest SHA-256 mismatch: provenance={recorded}, observed={observed}")

    grouped = pd.read_csv(grouped_path, keep_default_na=False)
    naive = pd.read_csv(naive_path, keep_default_na=False)
    required = {"sample_id", "produce_type", "freshness_label", "split"}
    for name, frame in (("grouped", grouped), ("naive", naive)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} split manifest is missing {sorted(missing)}")
        if set(frame["split"].astype(str)) != {"train", "validation", "test"}:
            raise ValueError(f"{name} split manifest must contain train, validation, and test")
        if frame["sample_id"].astype(str).duplicated().any():
            raise ValueError(f"{name} split manifest has duplicate sample_id values")
    identity_columns = ["sample_id", "produce_type", "freshness_label"]
    grouped_identity = grouped[identity_columns].sort_values("sample_id").reset_index(drop=True)
    naive_identity = naive[identity_columns].sort_values("sample_id").reset_index(drop=True)
    if not grouped_identity.equals(naive_identity):
        raise ValueError("Grouped and naive manifests do not contain the same labeled samples")
    assert_no_group_leakage(grouped, ("duplicate_group_id", "capture_group_id"))
    return {
        "split_seed": SPLIT_SEED,
        "split_provenance": str(provenance_path),
        "grouped_manifest": str(grouped_path),
        "grouped_manifest_sha256": observed_hashes["grouped"],
        "naive_manifest": str(naive_path),
        "naive_manifest_sha256": observed_hashes["naive"],
        "sample_count": int(len(grouped)),
    }


def select_top_architectures(
    runs: pd.DataFrame,
    *,
    top_k: int = 2,
    expected_architectures: Sequence[str] = MODEL_NAMES,
    expected_seeds: Sequence[int] = TRAINING_SEEDS,
) -> pd.DataFrame:
    """Rank architectures solely by mean grouped three-stage validation macro-F1."""

    required = {"architecture", "seed", SELECTION_METRIC}
    missing = required - set(runs.columns)
    if missing:
        raise ValueError(f"Selection rows are missing columns: {sorted(missing)}")
    if top_k < 1 or top_k > len(expected_architectures):
        raise ValueError("top_k is outside the controlled architecture set")
    frame = runs[list(required)].copy()
    frame["architecture"] = frame["architecture"].astype(str)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame[SELECTION_METRIC] = pd.to_numeric(frame[SELECTION_METRIC], errors="raise")
    if not np.isfinite(frame[SELECTION_METRIC].to_numpy(dtype=float)).all():
        raise ValueError("Selection metrics must all be finite")
    duplicates = frame.duplicated(["architecture", "seed"])
    if duplicates.any():
        raise ValueError("Selection input contains duplicate architecture/seed rows")

    expected_architecture_set = set(expected_architectures)
    if set(frame["architecture"]) != expected_architecture_set:
        raise ValueError(
            f"Selection requires exactly {list(expected_architectures)}; got {sorted(frame['architecture'].unique())}"
        )
    expected_seed_set = {int(seed) for seed in expected_seeds}
    for architecture, group in frame.groupby("architecture", sort=False):
        seeds = set(group["seed"].tolist())
        if seeds != expected_seed_set:
            raise ValueError(f"{architecture} has seeds {sorted(seeds)}; required {sorted(expected_seed_set)}")

    summary = (
        frame.groupby("architecture", sort=False)[SELECTION_METRIC]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": f"{SELECTION_METRIC}_mean",
                "std": f"{SELECTION_METRIC}_std",
                "count": "seed_count",
            }
        )
    )
    order = {architecture: index for index, architecture in enumerate(expected_architectures)}
    summary["_registry_order"] = summary["architecture"].map(order)
    # The predeclared tie policy is lower seed SD, then controlled registry order.
    summary = summary.sort_values(
        [f"{SELECTION_METRIC}_mean", f"{SELECTION_METRIC}_std", "_registry_order"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    summary["validation_rank"] = np.arange(1, len(summary) + 1)
    summary["selected_for_external"] = summary["validation_rank"] <= top_k
    summary["selected_for_naive_ablation"] = summary["validation_rank"] == 1
    summary["selection_metric"] = SELECTION_METRIC
    summary["selection_uses_test_metrics"] = False
    return summary.drop(columns="_registry_order")


def summarize_seed_metrics(
    rows: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    metric_columns: Sequence[str],
) -> pd.DataFrame:
    """Compute reproducible per-seed mean/SD/count summaries for scalar metrics."""

    missing = (set(group_columns) | set(metric_columns)) - set(rows.columns)
    if missing:
        raise ValueError(f"Cannot summarize missing columns: {sorted(missing)}")
    if "seed" in rows.columns and rows.duplicated([*group_columns, "seed"]).any():
        raise ValueError("Duplicate seed rows would overweight an experiment")
    records: list[dict[str, object]] = []
    grouper: str | list[str] = list(group_columns)
    if len(group_columns) == 1:
        grouper = group_columns[0]
    for keys, group in rows.groupby(grouper, sort=False, dropna=False):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        record: dict[str, object] = dict(zip(group_columns, key_values))
        record["seed_count"] = int(group["seed"].nunique()) if "seed" in group else int(len(group))
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            if values.empty:
                raise ValueError(f"Metric {metric!r} has no numeric values for {record}")
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else math.nan
        records.append(record)
    return pd.DataFrame.from_records(records)


def compute_rank_changes(summary: pd.DataFrame, metric: str = "validation_macro_f1_mean") -> pd.DataFrame:
    """Compare grouped architecture ranks between the three-stage and binary tasks."""

    required = {"architecture", "task", metric}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Rank-change input is missing {sorted(missing)}")
    frame = summary[summary["task"].isin(["three_stage", "binary"])].copy()
    records: list[dict[str, object]] = []
    for task in ("three_stage", "binary"):
        task_frame = frame[frame["task"] == task]
        if set(task_frame["architecture"]) != set(MODEL_NAMES):
            raise ValueError(f"Rank changes require all five grouped models for {task}")
        ranks = task_frame.set_index("architecture")[metric].rank(method="min", ascending=False)
        for architecture, rank in ranks.items():
            records.append({"architecture": architecture, "task": task, "rank": int(rank)})
    pivot = pd.DataFrame.from_records(records).pivot(index="architecture", columns="task", values="rank").reset_index()
    pivot = pivot.rename(columns={"three_stage": "three_stage_rank", "binary": "binary_rank"})
    pivot["binary_minus_three_stage_rank_change"] = pivot["binary_rank"] - pivot["three_stage_rank"]
    registry_order = {architecture: index for index, architecture in enumerate(MODEL_NAMES)}
    return pivot.sort_values("architecture", key=lambda series: series.map(registry_order)).reset_index(drop=True)


def flatten_scalar_metrics(metrics: Mapping[str, Any], prefix: str = "") -> dict[str, float | int]:
    """Flatten numeric metric leaves while intentionally omitting arrays/matrices."""

    flattened: dict[str, float | int] = {}
    for key, value in metrics.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(flatten_scalar_metrics(value, name))
        elif isinstance(value, bool):
            continue
        elif isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
            flattened[name] = int(value) if isinstance(value, (int, np.integer)) else float(value)
    return flattened


def read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {source}")
    return value


def require_complete_training_runs(
    results_root: str | Path,
    *,
    setting: str,
    task: str,
    architectures: Iterable[str],
    seeds: Iterable[int] = TRAINING_SEEDS,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    root = Path(results_root)
    for architecture in architectures:
        for seed in seeds:
            run_path = root / "runs" / setting / task / architecture / f"seed-{seed}" / "run.json"
            payload = read_json(run_path)
            expected = {"architecture": architecture, "task": task, "seed": int(seed)}
            for key, value in expected.items():
                if payload.get(key) != value:
                    raise ValueError(f"{run_path} has {key}={payload.get(key)!r}; expected {value!r}")
            checkpoint = run_path.parent / "best.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            records.append(
                {
                    "setting": setting,
                    "task": task,
                    "architecture": architecture,
                    "seed": int(seed),
                    "best_epoch": int(payload["best_epoch"]),
                    SELECTION_METRIC: float(payload[SELECTION_METRIC]),
                    "elapsed_seconds": float(payload["elapsed_seconds"]),
                    "checkpoint": str(checkpoint),
                }
            )
    return pd.DataFrame.from_records(records)


def json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Return strict-JSON-safe table records (NaN/inf become null)."""

    normalized = frame.astype(object).where(pd.notna(frame), None)
    records = normalized.to_dict(orient="records")
    for record in records:
        for key, value in tuple(record.items()):
            if isinstance(value, np.generic):
                record[key] = value.item()
    return records
