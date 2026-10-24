# Produce freshness research reproducibility snapshot

This bounded snapshot supports the revised leakage-aware produce-freshness benchmark. It contains the executed research pipeline, frozen protocol/configurations, aggregate and per-seed numerical results, revision analyses and generated table definitions. It excludes raw images, model weights/checkpoints, per-image model predictions, embedding arrays, Grad-CAM images, application assets and execution logs.

## Data and scope

Obtain the original data from their versioned sources:

- AgriFreshNET Freshness and Shelf-Life Image Dataset v1: https://doi.org/10.17632/42m5tb7yv9.1
- Fresh and Rotten Fruits Dataset v1: https://doi.org/10.17632/bdd69gyhv8.1

The source acquisition record identifies the datasets as CC BY 4.0. These dataset terms do not automatically license the software. No new software reuse license is asserted here. The unrelated upstream dashboard and its original app/model assets are not included; its historical source is https://github.com/akmalhen/fresh-or-trash-dashboard at commit 1389409df5aefdea2b925746600d0da1d1b431bb.

The primary benchmark contains 5,183 images after excluding pre-generated augmentations and conflicting exact duplicates. The locked grouped split is 3,514 train, 838 validation and 831 test images. The reference split seed is 2026; original matched training seeds are 2026, 3407, 777. The external source contributes 800 original-only banana/orange images. Manifests carry source-archive relative paths, hashes, labels and grouping/split assignments; they contain no image pixels.

Seven portable input manifests are compressed solely to keep the public snapshot bounded. Decompression restores the exact source bytes and recorded SHA-256 values:

    python tools/unpack_manifests.py --check-only
    python tools/unpack_manifests.py

## Numerical evidence

- results/experiments/aggregate: the original three-seed benchmark, calibration, external-validation, corruption, efficiency and subgroup tables.
- results/revision/analysis/baselines: frozen-feature and histogram baselines, validation grids and paired differences.
- results/revision/analysis/splits: capture-linkage ablations, split-composition controls and five-fold grouped cross-validation.
- results/revision/analysis/operational: locked-test means/uncertainty, confusion matrices, critical misses, selective prediction and validation operating points.
- results/revision/analysis/sensitivity: learning-rate/input-size controls, random initialization, corruption and external-transfer summaries.
- results/revision/analysis/deployment: model size, FLOPs, CPU/ONNX latency, GPU profile and quantization accuracy summaries.
- results/revision/analysis/labels: aggregate label-consistency diagnostics.
- paper/generated/revision: exact numerical table definitions used by the revised paper.

Stored tables can be inspected without image downloads or training. The retained code provides the entry points that produced the experiments; it does not make omitted weights, per-image predictions, caches or all intermediate artifacts publicly available.

## Reproduction entry points

Install the declared dependencies, unpack the manifests and obtain/extract the original images outside the repository. Supply your own image roots; none is embedded in this release.

    python -m pip install -e ".[dev]"
    python scripts/run_experiments.py --help
    python scripts/revision/make_revision_splits.py --help
    python scripts/revision/run_revision_queue.py --help
    python scripts/revision/analyze_splits.py --help
    python scripts/revision/analyze_operational.py --help
    python scripts/revision/analyze_sensitivity.py --help
    python scripts/revision/build_paper_tables.py --help

Training and inference must be rerun to regenerate omitted checkpoints and per-image predictions before analyses that consume them. The baseline/revision source and protocol are preserved; this publication process did not rerun model training.

RELEASE_MANIFEST.json records each published file's SHA-256 and any compression/derivation. Numeric/code/table source copies are byte-identical to the executed local project. data/dataset_sources.json is a bounded metadata extraction with local storage details omitted.

Visual freshness estimates support manual inspection. They do not establish food safety, edibility, pathogens, toxins or internal defects.
