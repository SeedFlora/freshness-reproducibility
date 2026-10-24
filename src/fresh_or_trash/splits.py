from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .grouping import UnionFind


DEFAULT_SPLITS = ("train", "validation", "test")
DEFAULT_RATIOS = (0.70, 0.15, 0.15)


def _validate_ratios(split_names: Sequence[str], ratios: Sequence[float]) -> np.ndarray:
    if len(split_names) != len(ratios) or len(ratios) < 2:
        raise ValueError("split_names and ratios must have the same length >= 2")
    values = np.asarray(ratios, dtype=float)
    if np.any(values <= 0) or not np.isclose(values.sum(), 1.0):
        raise ValueError(f"Split ratios must be positive and sum to 1; got {ratios}")
    return values


def joint_stratum(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing stratification columns: {sorted(missing)}")
    return frame[list(columns)].astype(str).agg("||".join, axis=1)


def resolve_group_components(frame: pd.DataFrame, group_columns: Sequence[str]) -> pd.Series:
    """Resolve possibly overlapping duplicate/capture identifiers transitively."""
    sample_ids = frame["sample_id"].astype(str).tolist()
    union = UnionFind(sample_ids)
    present = [column for column in group_columns if column in frame.columns]
    for column in present:
        valid = frame[column].fillna("").astype(str).str.strip() != ""
        for _, group in frame[valid].groupby(column, sort=False):
            members = group["sample_id"].astype(str).tolist()
            for member in members[1:]:
                union.union(members[0], member)
    components = [sorted(members) for members in union.components().values()]
    components.sort(key=lambda members: members[0])
    mapping = {
        sample_id: f"split-group-{index:06d}"
        for index, members in enumerate(components)
        for sample_id in members
    }
    return frame["sample_id"].astype(str).map(mapping)


def grouped_stratified_split(
    frame: pd.DataFrame,
    *,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    split_names: Sequence[str] = DEFAULT_SPLITS,
    stratify_columns: Sequence[str] = ("freshness_label", "produce_type"),
    group_columns: Sequence[str] = ("duplicate_group_id", "capture_group_id"),
    seed: int = 42,
) -> pd.DataFrame:
    """Deterministically approximate joint stratification while preserving groups."""
    ratio_values = _validate_ratios(split_names, ratios)
    if frame.empty:
        raise ValueError("Cannot split an empty manifest")
    result = frame.copy().reset_index(drop=True)
    result["_stratum"] = joint_stratum(result, stratify_columns)
    result["_split_group"] = resolve_group_components(result, group_columns)
    strata = sorted(result["_stratum"].unique())
    stratum_index = {value: index for index, value in enumerate(strata)}

    groups: list[tuple[str, int, np.ndarray, float]] = []
    rng = np.random.default_rng(seed)
    for group_id, group in result.groupby("_split_group", sort=True):
        counts = np.zeros(len(strata), dtype=float)
        for stratum, count in group["_stratum"].value_counts().items():
            counts[stratum_index[stratum]] = count
        rarity = float(np.sum(counts / np.maximum(1.0, result["_stratum"].value_counts().reindex(strata).to_numpy())))
        groups.append((str(group_id), len(group), counts, rarity + rng.uniform(0, 1e-9)))
    # Large and rare groups are placed first; random jitter gives seeded tie-breaking.
    groups.sort(key=lambda item: (-item[1], -item[3], item[0]))

    total_size = float(len(result))
    total_strata = np.zeros(len(strata), dtype=float)
    for stratum, count in result["_stratum"].value_counts().items():
        total_strata[stratum_index[stratum]] = count
    size_targets = ratio_values * total_size
    stratum_targets = ratio_values[:, None] * total_strata[None, :]
    assigned_sizes = np.zeros(len(split_names), dtype=float)
    assigned_strata = np.zeros((len(split_names), len(strata)), dtype=float)
    assignments: dict[str, str] = {}

    for group_id, group_size, group_counts, _ in groups:
        relevant = group_counts > 0
        candidate_scores: list[tuple[float, int]] = []
        for split_index in range(len(split_names)):
            size_progress = assigned_sizes[split_index] / max(size_targets[split_index], 1.0)
            label_progress = float(
                np.mean(
                    assigned_strata[split_index, relevant]
                    / np.maximum(stratum_targets[split_index, relevant], 1.0)
                )
            )
            new_size = assigned_sizes[split_index] + group_size
            overflow = max(0.0, new_size - size_targets[split_index]) / max(size_targets[split_index], 1.0)
            # Distribution progress dominates, with a strong penalty for avoidable overflow.
            score = size_progress + label_progress + 4.0 * overflow
            candidate_scores.append((score, split_index))
        _, chosen = min(candidate_scores, key=lambda value: (value[0], value[1]))
        assignments[group_id] = str(split_names[chosen])
        assigned_sizes[chosen] += group_size
        assigned_strata[chosen] += group_counts

    result["split"] = result["_split_group"].map(assignments)
    result = result.drop(columns=["_stratum", "_split_group"])
    assert_no_group_leakage(result, group_columns)
    return result


def naive_stratified_split(
    frame: pd.DataFrame,
    *,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    split_names: Sequence[str] = DEFAULT_SPLITS,
    stratify_columns: Sequence[str] = ("freshness_label", "produce_type"),
    seed: int = 42,
) -> pd.DataFrame:
    """Individual-image split for leakage-sensitivity ablation only."""
    ratio_values = _validate_ratios(split_names, ratios)
    if len(split_names) != 3:
        raise ValueError("naive_stratified_split currently requires three splits")
    result = frame.copy().reset_index(drop=True)
    strata = joint_stratum(result, stratify_columns)
    indices = np.arange(len(result))
    holdout_ratio = float(ratio_values[1] + ratio_values[2])
    can_stratify = strata.value_counts().min() >= 2 and len(result) >= 4
    train_indices, holdout_indices = train_test_split(
        indices,
        test_size=holdout_ratio,
        random_state=seed,
        shuffle=True,
        stratify=strata if can_stratify else None,
    )
    holdout_strata = strata.iloc[holdout_indices]
    relative_test = float(ratio_values[2] / holdout_ratio)
    holdout_can_stratify = holdout_strata.value_counts().min() >= 2 and len(holdout_indices) >= 4
    validation_indices, test_indices = train_test_split(
        holdout_indices,
        test_size=relative_test,
        random_state=seed,
        shuffle=True,
        stratify=holdout_strata if holdout_can_stratify else None,
    )
    result["split"] = ""
    result.loc[train_indices, "split"] = split_names[0]
    result.loc[validation_indices, "split"] = split_names[1]
    result.loc[test_indices, "split"] = split_names[2]
    return result


def group_leakage(frame: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    if "split" not in frame:
        raise ValueError("Manifest has no split column")
    records: list[dict[str, object]] = []
    for column in group_columns:
        if column not in frame:
            continue
        valid = frame[column].fillna("").astype(str).str.strip() != ""
        for group_id, group in frame[valid].groupby(column, sort=True):
            partitions = sorted(group["split"].astype(str).unique())
            if len(partitions) > 1:
                records.append(
                    {
                        "group_column": column,
                        "group_id": group_id,
                        "splits": ";".join(partitions),
                        "sample_count": len(group),
                    }
                )
    return pd.DataFrame.from_records(records, columns=["group_column", "group_id", "splits", "sample_count"])


def assert_no_group_leakage(frame: pd.DataFrame, group_columns: Sequence[str]) -> None:
    leakage = group_leakage(frame, group_columns)
    if not leakage.empty:
        example = leakage.iloc[0].to_dict()
        raise AssertionError(f"Group leakage detected ({len(leakage)} groups); first: {example}")


def split_summary(
    frame: pd.DataFrame,
    group_columns: Sequence[str] = ("duplicate_group_id", "capture_group_id"),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split, partition in frame.groupby("split", sort=False):
        row: dict[str, object] = {"split": split, "image_count": len(partition), "proportion": len(partition) / len(frame)}
        for column in group_columns:
            if column in partition:
                row[f"{column}_count"] = partition[column].replace("", np.nan).nunique(dropna=True)
        rows.append(row)
    return pd.DataFrame(rows)


def write_split_manifests(frame: pd.DataFrame, output_dir: str | Path, prefix: str = "grouped") -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination / f"{prefix}_manifest.csv", index=False)
    for split, partition in frame.groupby("split", sort=False):
        partition.to_csv(destination / f"{prefix}_{split}.csv", index=False)
    split_summary(frame).to_csv(destination / f"{prefix}_summary.csv", index=False)
