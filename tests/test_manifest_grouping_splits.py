from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from fresh_or_trash.grouping import (
    assign_duplicate_groups,
    assign_embedding_groups,
    assign_metadata_capture_groups,
    compute_image_hashes,
)
from fresh_or_trash.manifest import load_manifest
from fresh_or_trash.splits import assert_no_group_leakage, grouped_stratified_split, naive_stratified_split


def _base_frame(count: int = 60) -> pd.DataFrame:
    produce = ["banana", "tomato"]
    freshness = ["Fresh", "Semi-Fresh", "Rotten"]
    rows = []
    for index in range(count):
        rows.append(
            {
                "sample_id": f"sample-{index:03d}",
                "source_path": f"images/{index}.png",
                "filename": f"IMG_20250101_{120000 + index:06d}.png",
                "source_dataset": "synthetic-test",
                "produce_type": produce[index % len(produce)],
                "freshness_label": freshness[index % len(freshness)],
                "duplicate_group_id": f"duplicate-{index // 2:03d}",
                "capture_group_id": f"capture-{index // 4:03d}",
            }
        )
    return pd.DataFrame(rows)


def test_manifest_normalizes_dhash_alias(tmp_path: Path) -> None:
    frame = _base_frame(2)
    frame["dhash"] = ["0123456789abcdef", "fedcba9876543210"]
    path = tmp_path / "manifest.csv"
    frame.to_csv(path, index=False)
    loaded = load_manifest(path)
    assert "difference_hash" in loaded
    assert loaded.loc[0, "difference_hash"] == "0123456789abcdef"


def test_hash_and_duplicate_groups(tmp_path: Path) -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :16, 0] = 255
    Image.fromarray(image).save(tmp_path / "a.png")
    Image.fromarray(image).save(tmp_path / "b.png")
    changed = image.copy()
    changed[0:8, 0:8, 1] = 255
    Image.fromarray(changed).save(tmp_path / "c.png")
    frame = pd.DataFrame(
        [
            {"sample_id": "a", "source_path": "a.png", "produce_type": "banana", "freshness_label": "Fresh"},
            {"sample_id": "b", "source_path": "b.png", "produce_type": "banana", "freshness_label": "Fresh"},
            {"sample_id": "c", "source_path": "c.png", "produce_type": "banana", "freshness_label": "Fresh"},
        ]
    )
    hashed, audit = compute_image_hashes(frame, tmp_path)
    grouped, exact, _near = assign_duplicate_groups(hashed, phash_threshold=0, dhash_threshold=0)
    assert (audit["status"] == "ok").all()
    assert len(exact) == 2
    assert grouped.loc[0, "duplicate_group_id"] == grouped.loc[1, "duplicate_group_id"]


def test_near_hash_collision_cannot_bridge_produce_types() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "tomato",
                "source_path": "tomato.jpg",
                "produce_type": "Tomato",
                "freshness_label": "Fresh",
                "sha256": "a" * 64,
                "perceptual_hash": "0123456789abcdef",
                "difference_hash": "fedcba9876543210",
            },
            {
                "sample_id": "eggplant",
                "source_path": "eggplant.jpg",
                "produce_type": "eggplant",
                "freshness_label": "Fresh",
                "sha256": "b" * 64,
                "perceptual_hash": "0123456789abcdef",
                "difference_hash": "fedcba9876543210",
            },
        ]
    )
    grouped, _exact, near = assign_duplicate_groups(frame)
    assert grouped.loc[0, "duplicate_group_id"] != grouped.loc[1, "duplicate_group_id"]
    assert near.loc[0, "criterion"] == "cross_produce_skipped"
    assert not bool(near.loc[0, "linked"])


def test_exact_hash_matches_remain_global_across_produce_types() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "tomato",
                "source_path": "tomato.jpg",
                "produce_type": "tomato",
                "freshness_label": "Fresh",
                "sha256": "a" * 64,
                "perceptual_hash": "0" * 16,
                "difference_hash": "0" * 16,
            },
            {
                "sample_id": "eggplant",
                "source_path": "eggplant.jpg",
                "produce_type": "eggplant",
                "freshness_label": "Fresh",
                "sha256": "a" * 64,
                "perceptual_hash": "f" * 16,
                "difference_hash": "f" * 16,
            },
        ]
    )
    grouped, exact, _near = assign_duplicate_groups(frame, phash_threshold=0, dhash_threshold=0)
    assert grouped.loc[0, "duplicate_group_id"] == grouped.loc[1, "duplicate_group_id"]
    assert len(exact) == 2


def test_filename_and_embedding_capture_groups() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "t1",
                "source_path": "fresh/IMG_20250101_120000000.jpg",
                "filename": "IMG_20250101_120000000.jpg",
                "produce_type": "banana",
                "freshness_label": "Fresh",
                "duplicate_group_id": "d1",
            },
            {
                "sample_id": "t2",
                "source_path": "fresh/IMG_20250101_120100000.jpg",
                "filename": "IMG_20250101_120100000.jpg",
                "produce_type": "banana",
                "freshness_label": "Fresh",
                "duplicate_group_id": "d2",
            },
            {
                "sample_id": "w1",
                "source_path": "rotten/IMG-20250102-WA0001.jpg",
                "filename": "IMG-20250102-WA0001.jpg",
                "produce_type": "tomato",
                "freshness_label": "Rotten",
                "duplicate_group_id": "d3",
            },
            {
                "sample_id": "w2",
                "source_path": "rotten/IMG-20250102-WA0002.jpg",
                "filename": "IMG-20250102-WA0002.jpg",
                "produce_type": "tomato",
                "freshness_label": "Rotten",
                "duplicate_group_id": "d4",
            },
        ]
    )
    metadata_grouped, report = assign_metadata_capture_groups(frame)
    assert metadata_grouped.loc[0, "capture_group_id"] == metadata_grouped.loc[1, "capture_group_id"]
    assert metadata_grouped.loc[2, "capture_group_id"] == metadata_grouped.loc[3, "capture_group_id"]
    assert {"filename_timestamp_gap", "filename_whatsapp_adjacent"} <= set(report["criterion"])

    embeddings = np.asarray([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]], dtype=np.float32)
    embedded, links = assign_embedding_groups(frame, embeddings, frame["sample_id"].tolist(), cosine_similarity_threshold=0.98)
    assert embedded["capture_group_id"].notna().all()
    assert "embedding_cosine" in set(links["criterion"])


def test_embedding_collision_cannot_bridge_produce_types() -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "tomato",
                "source_path": "fresh/tomato-one.jpg",
                "filename": "tomato-one.jpg",
                "produce_type": "tomato",
                "freshness_label": "Fresh",
                "duplicate_group_id": "d1",
            },
            {
                "sample_id": "eggplant",
                "source_path": "fresh/eggplant-one.jpg",
                "filename": "eggplant-one.jpg",
                "produce_type": "eggplant",
                "freshness_label": "Fresh",
                "duplicate_group_id": "d2",
            },
        ]
    )
    embeddings = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    grouped, report = assign_embedding_groups(frame, embeddings, ["tomato", "eggplant"], cosine_similarity_threshold=0.99)
    assert grouped.loc[0, "embedding_cluster_id"] != grouped.loc[1, "embedding_cluster_id"]
    assert grouped.loc[0, "capture_group_id"] != grouped.loc[1, "capture_group_id"]
    assert report.loc[0, "criterion"] == "embedding_cross_produce_skipped"
    assert not bool(report.loc[0, "linked"])


def test_grouped_split_is_deterministic_and_leakage_free() -> None:
    frame = _base_frame()
    first = grouped_stratified_split(frame, seed=7)
    second = grouped_stratified_split(frame, seed=7)
    assert first["split"].tolist() == second["split"].tolist()
    assert_no_group_leakage(first, ["duplicate_group_id", "capture_group_id"])
    assert set(first["split"]) == {"train", "validation", "test"}


def test_naive_split_assigns_every_sample() -> None:
    result = naive_stratified_split(_base_frame(), seed=11)
    assert set(result["split"]) == {"train", "validation", "test"}
    assert result["split"].ne("").all()
