from __future__ import annotations

import argparse
from pathlib import Path

from fresh_or_trash.manifest import audit_manifest_images, load_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a normalized dataset manifest and optionally audit image decoding")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--image-root", type=Path, required=True, help="Root prepended to relative source_path values")
    parser.add_argument("--audit-output", type=Path, help="Write per-image decode/hash audit CSV")
    parser.add_argument("--require-split", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frame = load_manifest(args.manifest, require_split=args.require_split)
    print(f"valid manifest: {len(frame)} rows, {frame['produce_type'].nunique()} produce types")
    if args.audit_output:
        audit = audit_manifest_images(frame, args.image_root)
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        audit.to_csv(args.audit_output, index=False)
        failures = int((audit["error"] != "").sum())
        print(f"image audit: {len(audit) - failures} valid, {failures} invalid -> {args.audit_output}")
        return 1 if failures else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
