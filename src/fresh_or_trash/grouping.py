from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

import imagehash
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from .data import ManifestDataset, build_transform
from .manifest import file_sha256, resolve_image_path
from .models import build_model


class UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {str(item): str(item) for item in items}
        self.rank = {str(item): 0 for item in items}

    def find(self, item: str) -> str:
        item = str(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1

    def components(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for item in self.parent:
            groups[self.find(item)].append(item)
        return dict(groups)


class BKTree:
    """Small metric tree for sub-quadratic Hamming-neighbor lookup."""

    def __init__(self) -> None:
        self.root: tuple[int, dict[int, object]] | None = None

    @staticmethod
    def distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = (value, {})
            return
        node = self.root
        while True:
            current, children = node
            distance = self.distance(value, current)
            if distance == 0:
                return
            child = children.get(distance)
            if child is None:
                children[distance] = (value, {})
                return
            node = child  # type: ignore[assignment]

    def query(self, value: int, radius: int) -> Iterator[tuple[int, int]]:
        if self.root is None:
            return
        stack = [self.root]
        while stack:
            current, children = stack.pop()
            distance = self.distance(value, current)
            if distance <= radius:
                yield current, distance
            lower, upper = distance - radius, distance + radius
            stack.extend(child for edge, child in children.items() if lower <= edge <= upper)  # type: ignore[arg-type]


def compute_image_hashes(frame: pd.DataFrame, image_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute SHA-256, pHash, and dHash while retaining corrupt-file errors."""
    result = frame.copy()
    records: list[dict[str, str]] = []
    hashes: dict[str, tuple[str, str, str]] = {}
    for _, row in result.iterrows():
        sample_id = str(row["sample_id"])
        path = resolve_image_path(row, image_root)
        try:
            sha256 = file_sha256(path)
            with Image.open(path) as source:
                image = source.convert("RGB")
                phash = str(imagehash.phash(image, hash_size=8))
                dhash = str(imagehash.dhash(image, hash_size=8))
            hashes[sample_id] = (sha256, phash, dhash)
            records.append({"sample_id": sample_id, "source_path": str(path), "status": "ok", "error": ""})
        except (OSError, ValueError) as exc:
            hashes[sample_id] = ("", "", "")
            records.append(
                {"sample_id": sample_id, "source_path": str(path), "status": "invalid", "error": f"{type(exc).__name__}: {exc}"}
            )
    result["sha256"] = result["sample_id"].astype(str).map(lambda value: hashes[value][0])
    result["perceptual_hash"] = result["sample_id"].astype(str).map(lambda value: hashes[value][1])
    result["difference_hash"] = result["sample_id"].astype(str).map(lambda value: hashes[value][2])
    return result, pd.DataFrame.from_records(records)


def assign_duplicate_groups(
    frame: pd.DataFrame,
    *,
    phash_threshold: int = 6,
    dhash_threshold: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Group exact and pixel-level near duplicates without deleting images.

    A near-duplicate link requires both pHash and dHash distances to be within
    their thresholds. Exact SHA matches are always linked.
    """
    required = {"sample_id", "produce_type", "sha256", "perceptual_hash", "difference_hash"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Hash grouping requires columns: {sorted(missing)}")
    result = frame.copy()
    sample_ids = result["sample_id"].astype(str).tolist()
    union = UnionFind(sample_ids)
    exact_records: list[dict[str, object]] = []
    for sha256, group in result[result["sha256"].astype(str) != ""].groupby("sha256", sort=True):
        members = sorted(group["sample_id"].astype(str))
        if len(members) > 1:
            for member in members[1:]:
                union.union(members[0], member)
            for member in members:
                exact_records.append({"sha256": sha256, "sample_id": member, "exact_group_size": len(members)})

    hash_to_ids: dict[int, list[str]] = defaultdict(list)
    dhashes: dict[str, int] = {}
    produce_by_id = {
        str(row["sample_id"]): str(row["produce_type"]).strip().casefold()
        for _, row in result.iterrows()
    }
    tree = BKTree()
    near_records: list[dict[str, object]] = []
    valid = result[(result["perceptual_hash"].astype(str) != "") & (result["difference_hash"].astype(str) != "")]
    for _, row in valid.sort_values("sample_id").iterrows():
        sample_id = str(row["sample_id"])
        phash = int(str(row["perceptual_hash"]), 16)
        dhash = int(str(row["difference_hash"]), 16)
        for neighbor_hash, phash_distance in tree.query(phash, phash_threshold):
            for neighbor_id in hash_to_ids[neighbor_hash]:
                dhash_distance = BKTree.distance(dhash, dhashes[neighbor_id])
                if dhash_distance <= dhash_threshold:
                    same_produce = produce_by_id[sample_id] == produce_by_id[neighbor_id]
                    if same_produce:
                        union.union(sample_id, neighbor_id)
                    near_records.append(
                        {
                            "sample_id_a": neighbor_id,
                            "sample_id_b": sample_id,
                            "produce_type_a": produce_by_id[neighbor_id],
                            "produce_type_b": produce_by_id[sample_id],
                            "phash_distance": phash_distance,
                            "dhash_distance": dhash_distance,
                            "criterion": "phash_and_dhash" if same_produce else "cross_produce_skipped",
                            "linked": same_produce,
                        }
                    )
        tree.add(phash)
        hash_to_ids[phash].append(sample_id)
        dhashes[sample_id] = dhash

    component_members = [sorted(members) for members in union.components().values()]
    component_members.sort(key=lambda members: members[0])
    group_by_id = {
        sample_id: f"duplicate-{index:06d}"
        for index, members in enumerate(component_members)
        for sample_id in members
    }
    result["duplicate_group_id"] = result["sample_id"].astype(str).map(group_by_id)
    return (
        result,
        pd.DataFrame.from_records(exact_records, columns=["sha256", "sample_id", "exact_group_size"]),
        pd.DataFrame.from_records(
            near_records,
            columns=[
                "sample_id_a",
                "sample_id_b",
                "produce_type_a",
                "produce_type_b",
                "phash_distance",
                "dhash_distance",
                "criterion",
                "linked",
            ],
        ),
    )


@torch.inference_mode()
def extract_embeddings(
    frame: pd.DataFrame,
    image_root: str | Path,
    *,
    architecture: str = "convnext_tiny",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str | torch.device = "cpu",
    pretrained: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Embedding hook for capture-sequence grouping using fixed ImageNet features."""
    dataset = ManifestDataset(
        frame,
        image_root,
        task="three_stage",
        transform=build_transform(image_size, training=False),
        image_size=image_size,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model = build_model(architecture, num_freshness_classes=3, pretrained=pretrained).to(device).eval()
    chunks: list[np.ndarray] = []
    sample_ids: list[str] = []
    for batch in loader:
        features = model.extract_features(batch["image"].to(device, non_blocking=True))
        features = torch.nn.functional.normalize(features, dim=1)
        chunks.append(features.cpu().numpy().astype(np.float32))
        sample_ids.extend(batch["sample_id"])
    return np.concatenate(chunks, axis=0), sample_ids


def assign_embedding_groups(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    sample_ids: list[str],
    *,
    cosine_similarity_threshold: float = 0.985,
    neighbors: int = 10,
    filename_gap_seconds: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster highly similar embeddings into capture groups.

    The output is a grouping signal for split integrity, not a deletion list.
    """
    if embeddings.ndim != 2 or embeddings.shape[0] != len(sample_ids):
        raise ValueError("Embeddings must have shape [len(sample_ids), feature_dim]")
    if set(sample_ids) != set(frame["sample_id"].astype(str)):
        raise ValueError("Embedding sample IDs must match the manifest exactly")
    union = UnionFind(sample_ids)
    embedding_union = UnionFind(sample_ids)
    produce_by_id = {
        str(row["sample_id"]): str(row["produce_type"]).strip().casefold()
        for _, row in frame.iterrows()
    }
    records: list[dict[str, object]] = []
    if "duplicate_group_id" in frame:
        for _, group in frame.groupby("duplicate_group_id", sort=False):
            members = sorted(group["sample_id"].astype(str))
            for member in members[1:]:
                union.union(members[0], member)
                records.append(
                    {
                        "sample_id_a": members[0],
                        "sample_id_b": member,
                        "criterion": "duplicate_group",
                        "cosine_similarity": np.nan,
                    }
                )

    records.extend(_filename_capture_links(frame, union, gap_seconds=filename_gap_seconds))

    count = min(max(2, neighbors), len(sample_ids))
    index = NearestNeighbors(n_neighbors=count, metric="cosine", algorithm="brute").fit(embeddings)
    distances, indices = index.kneighbors(embeddings)
    for row_index, (row_distances, row_neighbors) in enumerate(zip(distances, indices)):
        for distance, neighbor_index in zip(row_distances, row_neighbors):
            if int(neighbor_index) == row_index:
                continue
            if distance > (1.0 - cosine_similarity_threshold):
                continue
            left, right = sample_ids[row_index], sample_ids[int(neighbor_index)]
            if left >= right:
                continue
            same_produce = produce_by_id[left] == produce_by_id[right]
            if same_produce:
                union.union(left, right)
                embedding_union.union(left, right)
            records.append(
                {
                    "sample_id_a": left,
                    "sample_id_b": right,
                    "produce_type_a": produce_by_id[left],
                    "produce_type_b": produce_by_id[right],
                    "criterion": "embedding_cosine" if same_produce else "embedding_cross_produce_skipped",
                    "linked": same_produce,
                    "cosine_similarity": 1.0 - float(distance),
                }
            )
    embedding_components = [sorted(members) for members in embedding_union.components().values()]
    embedding_components.sort(key=lambda members: members[0])
    embedding_mapping = {
        sample_id: f"embedding-{index:06d}"
        for index, members in enumerate(embedding_components)
        for sample_id in members
    }
    capture_components = [sorted(members) for members in union.components().values()]
    capture_components.sort(key=lambda members: members[0])
    capture_mapping = {
        sample_id: f"capture-{index:06d}"
        for index, members in enumerate(capture_components)
        for sample_id in members
    }
    result = frame.copy()
    result["embedding_cluster_id"] = result["sample_id"].astype(str).map(embedding_mapping)
    result["capture_group_id"] = result["sample_id"].astype(str).map(capture_mapping)
    return result, pd.DataFrame.from_records(
        records,
        columns=[
            "sample_id_a",
            "sample_id_b",
            "produce_type_a",
            "produce_type_b",
            "criterion",
            "criterion_value",
            "linked",
            "cosine_similarity",
        ],
    )


_TIMESTAMP_PATTERN = re.compile(r"(?i)^IMG[_-](\d{8})[_-](\d{6})(\d{3})?")
_WHATSAPP_PATTERN = re.compile(r"(?i)^IMG-(\d{8})-WA(\d+)")
_MANUAL_PATTERN = re.compile(r"^(.*?)[ _-]*\((\d+)\)$", re.IGNORECASE)


def _capture_scope(row: pd.Series) -> tuple[str, str]:
    parent = str(Path(str(row.get("source_path", row.get("filename", "")))).parent).replace("\\", "/").lower()
    return str(row.get("produce_type", "")).strip().lower(), parent


def _filename_capture_links(frame: pd.DataFrame, union: UnionFind, *, gap_seconds: int = 120) -> list[dict[str, object]]:
    """Link deterministic filename sequences that plausibly share a capture session.

    Rules are deliberately metadata-only: timestamped captures link within 120 s,
    WhatsApp exports link adjacent indices on the same date, and numbered manual
    series link adjacent indices with the same normalized prefix. All rules are
    scoped by produce type and source directory.
    """
    if gap_seconds < 0:
        raise ValueError("gap_seconds must be non-negative")
    timestamp_groups: dict[tuple[str, str, str], list[tuple[datetime, str]]] = defaultdict(list)
    whatsapp_groups: dict[tuple[str, str, str], list[tuple[int, str]]] = defaultdict(list)
    manual_groups: dict[tuple[str, str, str], list[tuple[int, str]]] = defaultdict(list)
    for _, row in frame.iterrows():
        sample_id = str(row["sample_id"])
        filename = str(row.get("filename") or Path(str(row.get("source_path", ""))).name)
        stem = Path(filename).stem
        produce, parent = _capture_scope(row)
        timestamp_match = _TIMESTAMP_PATTERN.match(stem)
        if timestamp_match:
            date, clock, milliseconds = timestamp_match.groups()
            try:
                timestamp = datetime.strptime(date + clock, "%Y%m%d%H%M%S")
                if milliseconds:
                    timestamp = timestamp.replace(microsecond=int(milliseconds) * 1000)
                timestamp_groups[(produce, parent, date)].append((timestamp, sample_id))
            except ValueError:
                pass
            continue
        whatsapp_match = _WHATSAPP_PATTERN.match(stem)
        if whatsapp_match:
            date, sequence = whatsapp_match.groups()
            whatsapp_groups[(produce, parent, date)].append((int(sequence), sample_id))
            continue
        manual_match = _MANUAL_PATTERN.match(stem)
        if manual_match:
            prefix, sequence = manual_match.groups()
            normalized_prefix = re.sub(r"[^a-z0-9]+", "", prefix.lower())
            if normalized_prefix:
                manual_groups[(produce, parent, normalized_prefix)].append((int(sequence), sample_id))

    records: list[dict[str, object]] = []

    def link(left: str, right: str, criterion: str, detail: object) -> None:
        union.union(left, right)
        records.append(
            {
                "sample_id_a": left,
                "sample_id_b": right,
                "criterion": criterion,
                "criterion_value": detail,
                "cosine_similarity": np.nan,
            }
        )

    for entries in timestamp_groups.values():
        entries.sort()
        for (left_time, left_id), (right_time, right_id) in zip(entries, entries[1:]):
            difference = (right_time - left_time).total_seconds()
            if difference <= gap_seconds:
                link(left_id, right_id, "filename_timestamp_gap", difference)
    for entries in whatsapp_groups.values():
        entries.sort()
        for (left_index, left_id), (right_index, right_id) in zip(entries, entries[1:]):
            if right_index - left_index == 1:
                link(left_id, right_id, "filename_whatsapp_adjacent", 1)
    for entries in manual_groups.values():
        entries.sort()
        for (left_index, left_id), (right_index, right_id) in zip(entries, entries[1:]):
            if right_index - left_index == 1:
                link(left_id, right_id, "filename_manual_series_adjacent", 1)
    return records


def assign_metadata_capture_groups(
    frame: pd.DataFrame,
    *,
    filename_gap_seconds: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build duplicate- plus filename-aware capture groups without embeddings."""
    sample_ids = frame["sample_id"].astype(str).tolist()
    union = UnionFind(sample_ids)
    records: list[dict[str, object]] = []
    if "duplicate_group_id" in frame:
        for _, group in frame.groupby("duplicate_group_id", sort=False):
            members = sorted(group["sample_id"].astype(str))
            for member in members[1:]:
                union.union(members[0], member)
                records.append(
                    {
                        "sample_id_a": members[0],
                        "sample_id_b": member,
                        "criterion": "duplicate_group",
                        "criterion_value": "",
                        "cosine_similarity": np.nan,
                    }
                )
    records.extend(_filename_capture_links(frame, union, gap_seconds=filename_gap_seconds))
    components = [sorted(members) for members in union.components().values()]
    components.sort(key=lambda members: members[0])
    mapping = {
        sample_id: f"capture-{index:06d}"
        for index, members in enumerate(components)
        for sample_id in members
    }
    result = frame.copy()
    result["embedding_cluster_id"] = ""
    result["capture_group_id"] = result["sample_id"].astype(str).map(mapping)
    return result, pd.DataFrame.from_records(
        records,
        columns=["sample_id_a", "sample_id_b", "criterion", "criterion_value", "cosine_similarity"],
    )
