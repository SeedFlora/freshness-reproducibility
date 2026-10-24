from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fresh_or_trash.grouping import (
    assign_duplicate_groups,
    assign_embedding_groups,
    assign_metadata_capture_groups,
    compute_image_hashes,
    extract_embeddings,
)
from fresh_or_trash.manifest import load_manifest, write_manifest
from fresh_or_trash.utils import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create exact, near-duplicate, and embedding/capture groups")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_manifest", type=Path)
    parser.add_argument("--image-root", type=Path, required=True, help="Root prepended to relative source_path values")
    parser.add_argument("--report-dir", type=Path, default=Path("results"))
    parser.add_argument("--phash-threshold", type=int, default=6)
    parser.add_argument("--dhash-threshold", type=int, default=6)
    parser.add_argument("--embedding-threshold", type=float, default=0.985)
    parser.add_argument("--embedding-neighbors", type=int, default=10)
    parser.add_argument("--filename-gap-seconds", type=int, default=120)
    parser.add_argument("--embedding-architecture", default="convnext_tiny", choices=["convnext_tiny"])
    parser.add_argument("--embeddings-npy", type=Path, help="Reuse a [N,D] embedding array")
    parser.add_argument("--embedding-sample-ids", type=Path, help="Text file with one sample_id per embedding row")
    parser.add_argument("--save-embeddings", type=Path, help="Save computed embeddings for reproducibility")
    parser.add_argument("--skip-embeddings", action="store_true", help="Skip semantic grouping; capture groups fall back to duplicate groups")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true", help="Testing only: do not use fixed ImageNet weights")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = load_manifest(args.manifest)
    hash_columns = {"sha256", "perceptual_hash", "difference_hash"}
    needs_hashes = not hash_columns.issubset(frame.columns) or any((frame[column].astype(str) == "").any() for column in hash_columns)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    if needs_hashes:
        frame, image_audit = compute_image_hashes(frame, args.image_root)
        image_audit.to_csv(args.report_dir / "image_hash_audit.csv", index=False)
        invalid = image_audit[image_audit["status"] != "ok"]
        if not invalid.empty:
            raise RuntimeError(f"Hashing found {len(invalid)} invalid images; inspect {args.report_dir / 'image_hash_audit.csv'}")
    frame, exact_report, near_report = assign_duplicate_groups(
        frame,
        phash_threshold=args.phash_threshold,
        dhash_threshold=args.dhash_threshold,
    )
    exact_report.to_csv(args.report_dir / "exact_duplicate_report.csv", index=False)
    near_report.to_csv(args.report_dir / "near_duplicate_report.csv", index=False)

    if args.skip_embeddings:
        frame, capture_report = assign_metadata_capture_groups(frame, filename_gap_seconds=args.filename_gap_seconds)
    else:
        if bool(args.embeddings_npy) != bool(args.embedding_sample_ids):
            raise ValueError("Provide both --embeddings-npy and --embedding-sample-ids, or neither")
        if args.embeddings_npy:
            embeddings = np.load(args.embeddings_npy)
            sample_ids = [line.strip() for line in args.embedding_sample_ids.read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            device = resolve_device(args.device)
            embeddings, sample_ids = extract_embeddings(
                frame,
                args.image_root,
                architecture=args.embedding_architecture,
                batch_size=args.batch_size,
                num_workers=args.workers,
                device=device,
                pretrained=not args.no_pretrained,
            )
            if args.save_embeddings:
                args.save_embeddings.parent.mkdir(parents=True, exist_ok=True)
                np.save(args.save_embeddings, embeddings)
                args.save_embeddings.with_suffix(".sample_ids.txt").write_text("\n".join(sample_ids) + "\n", encoding="utf-8")
        frame, capture_report = assign_embedding_groups(
            frame,
            embeddings,
            sample_ids,
            cosine_similarity_threshold=args.embedding_threshold,
            neighbors=args.embedding_neighbors,
            filename_gap_seconds=args.filename_gap_seconds,
        )
    capture_report.to_csv(args.report_dir / "capture_group_report.csv", index=False)
    write_manifest(frame, args.output_manifest)
    print(
        f"grouped {len(frame)} images into {frame['duplicate_group_id'].nunique()} duplicate groups and "
        f"{frame['capture_group_id'].nunique()} capture groups -> {args.output_manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
