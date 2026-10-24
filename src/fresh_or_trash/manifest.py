from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
from PIL import Image, UnidentifiedImageError


MANIFEST_COLUMNS = (
    "sample_id",
    "source_dataset",
    "source_path",
    "filename",
    "produce_type",
    "freshness_label",
    "shelf_life_label",
    "original_class_name",
    "width",
    "height",
    "channels",
    "file_format",
    "file_size",
    "sha256",
    "perceptual_hash",
    "difference_hash",
    "embedding_cluster_id",
    "capture_group_id",
    "duplicate_group_id",
    "split",
)

FRESHNESS_CLASSES = ("Fresh", "Semi-Fresh", "Rotten")
_LABEL_ALIASES = {
    "fresh": "Fresh",
    "semi-fresh": "Semi-Fresh",
    "semi_fresh": "Semi-Fresh",
    "semifresh": "Semi-Fresh",
    "semi fresh": "Semi-Fresh",
    "rotten": "Rotten",
    "spoiled": "Rotten",
}


def normalize_freshness_label(value: object) -> str:
    key = str(value).strip().lower()
    if key not in _LABEL_ALIASES:
        raise ValueError(f"Unsupported freshness label: {value!r}")
    return _LABEL_ALIASES[key]


def load_manifest(
    path: str | Path,
    *,
    require_split: bool = False,
    validate_paths: bool = False,
    image_root: str | Path | None = None,
) -> pd.DataFrame:
    """Load and validate a normalized CSV/Parquet manifest.

    The loader never infers labels from pixels. Label normalization is limited to
    documented spelling aliases and invalid labels fail loudly.
    """
    source = Path(path)
    if source.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    else:
        frame = pd.read_csv(source, keep_default_na=False)
    aliases = {
        "path": "source_path",
        "label": "freshness_label",
        "class": "original_class_name",
        "dhash": "difference_hash",
        "phash": "perceptual_hash",
    }
    frame = frame.rename(columns={key: value for key, value in aliases.items() if key in frame.columns})

    required = {"sample_id", "source_path", "produce_type", "freshness_label"}
    if require_split:
        required.add("split")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest {source} is missing columns: {', '.join(missing)}")
    if frame["sample_id"].astype(str).duplicated().any():
        duplicated = frame.loc[frame["sample_id"].astype(str).duplicated(), "sample_id"].head().tolist()
        raise ValueError(f"sample_id must be unique; examples: {duplicated}")
    frame = frame.copy()
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["source_path"] = frame["source_path"].astype(str)
    frame["produce_type"] = frame["produce_type"].astype(str).str.strip()
    frame["freshness_label"] = frame["freshness_label"].map(normalize_freshness_label)
    if (frame["produce_type"] == "").any():
        raise ValueError("produce_type contains empty values")
    if require_split:
        allowed = {"train", "validation", "test"}
        invalid = sorted(set(frame["split"].astype(str)) - allowed)
        if invalid:
            raise ValueError(f"Invalid split values: {invalid}; expected {sorted(allowed)}")
    if validate_paths:
        root = Path(image_root) if image_root else source.parent
        missing_paths = [str(resolve_image_path(row, root)) for _, row in frame.iterrows() if not resolve_image_path(row, root).is_file()]
        if missing_paths:
            raise FileNotFoundError(f"{len(missing_paths)} manifest images are missing; first: {missing_paths[0]}")
    return frame


def resolve_image_path(row: pd.Series | dict, image_root: str | Path) -> Path:
    path = Path(str(row["source_path"]))
    return path if path.is_absolute() else Path(image_root) / path


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: str | Path) -> dict[str, object]:
    source = Path(path)
    try:
        with Image.open(source) as image:
            image.verify()
        with Image.open(source) as image:
            return {
                "width": image.width,
                "height": image.height,
                "channels": len(image.getbands()),
                "file_format": image.format or source.suffix.lstrip(".").upper(),
                "file_size": source.stat().st_size,
                "sha256": file_sha256(source),
                "error": "",
            }
    except (OSError, UnidentifiedImageError) as exc:
        return {
            "width": "",
            "height": "",
            "channels": "",
            "file_format": source.suffix.lstrip(".").upper(),
            "file_size": source.stat().st_size if source.exists() else "",
            "sha256": file_sha256(source) if source.is_file() else "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def audit_manifest_images(frame: pd.DataFrame, image_root: str | Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        record = {"sample_id": row["sample_id"]}
        record.update(inspect_image(resolve_image_path(row, image_root)))
        records.append(record)
    return pd.DataFrame.from_records(records)


def ensure_columns(frame: pd.DataFrame, columns: Sequence[str] = MANIFEST_COLUMNS) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column not in result:
            result[column] = ""
    extras = [column for column in result.columns if column not in columns]
    return result[list(columns) + extras]


def write_manifest(frame: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ensure_columns(frame).to_csv(destination, index=False)
