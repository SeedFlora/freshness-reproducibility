"""Run every GPU job needed for the peer-review revision as one resumable queue.

Jobs reuse the frozen CLIs (train, evaluate, external) plus the per-image
corruption evaluator. Nothing under ``results/experiments`` is modified; all
outputs go to ``results/revision``. A job is skipped when its expected files
exist, failures are logged and skipped, and the job list is rebuilt after
every job so dependent jobs (the learning-rate follow-up) appear only once
their inputs are complete.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from fresh_or_trash.models import MODEL_NAMES


ROOT = Path(__file__).resolve().parents[2]
FROZEN = Path("results/experiments")
OUT = Path("results/revision")
SEEDS = (2026, 3407, 777)
ABLATION_MODELS = ("mobilenet_v2", "convnext_tiny")
STAGED_LEVELS = ("L1_exact", "L2_hash", "L3_embedding")
CV_FOLDS = tuple(range(5))
LEARNING_RATES = {"lr_1e-4": 1e-4, "lr_1e-3": 1e-3}
RESOLUTIONS = {"res_160": 160, "res_320": 320}
# Small inference jobs load images in-process: on Windows, spawning eight
# workers that each re-import torch costs more than decoding ~800 images.
GROUP_ORDER = ("val_preds", "external", "cv3", "staged", "corruption", "scratch", "lr", "res", "cv2")


@dataclass
class Job:
    group: str
    name: str
    command: list[str]
    expected: list[Path]
    after: list[Path] = field(default_factory=list)

    def complete(self) -> bool:
        return all((ROOT / path).is_file() for path in self.expected)

    def ready(self) -> bool:
        return all((ROOT / path).is_file() for path in self.after)


def _module(module: str, *arguments: object) -> list[str]:
    return [sys.executable, "-m", module, *(str(value) for value in arguments)]


def _write_configs() -> dict[str, Path]:
    base = yaml.safe_load((ROOT / "config/experiments.yaml").read_text(encoding="utf-8"))
    variants: dict[str, dict] = {"base": base}
    for name, rate in LEARNING_RATES.items():
        variant = json.loads(json.dumps(base))
        variant["training"]["learning_rate"] = rate
        variants[name] = variant
    for name, size in RESOLUTIONS.items():
        variant = json.loads(json.dumps(base))
        variant["data"]["image_size"] = size
        variants[name] = variant
    paths = {}
    for name, payload in variants.items():
        path = Path("config/revision") / f"{name}.yaml"
        (ROOT / path).parent.mkdir(parents=True, exist_ok=True)
        text = "# Derived from config/experiments.yaml for the peer-review revision.\n" + yaml.safe_dump(payload, sort_keys=False)
        if not (ROOT / path).is_file() or (ROOT / path).read_text(encoding="utf-8") != text:
            (ROOT / path).write_text(text, encoding="utf-8")
        paths[name] = path
    return paths


def _train_and_eval(
    group: str,
    label: str,
    *,
    config: Path,
    manifest: Path,
    image_root: Path,
    arch: str,
    task: str,
    seed: int,
    run_dir: Path,
    pretrained: bool = True,
) -> list[Job]:
    # Test inference runs inside the training process after early stopping on
    # validation; the frozen runs deferred it only to lock the test set until
    # architecture selection, which these sensitivity runs do not perform.
    return [
        Job(
            group,
            f"train {label}",
            _module(
                "fresh_or_trash.cli.train",
                "--config", config, manifest,
                "--image-root", image_root,
                "--model", arch, "--task", task,
                "--output-dir", run_dir, "--seed", seed, "--device", "cuda",
                "--pretrained" if pretrained else "--no-pretrained",
                "--auxiliary-produce-head", "--evaluate-test",
            ),
            [run_dir / "best.pt", run_dir / "run.json", run_dir / "test_predictions.csv", run_dir / "test_metrics.json"],
        )
    ]


def _best_learning_rate(arch: str) -> str | None:
    """Validation-only choice among 1e-4, 3e-4 (frozen), and 1e-3 at seed 2026."""
    candidates = {"base": ROOT / FROZEN / "runs/grouped/three_stage" / arch / "seed-2026/run.json"}
    for name in LEARNING_RATES:
        candidates[name] = ROOT / OUT / "runs/lr" / name / arch / "seed-2026/run.json"
    if not all(path.is_file() for path in candidates.values()):
        return None
    scores = {name: json.loads(path.read_text(encoding="utf-8"))["validation_macro_f1"] for name, path in candidates.items()}
    return max(scores, key=lambda name: (scores[name], name == "base"))


def build_jobs(primary_root: Path, external_root: Path) -> list[Job]:
    configs = _write_configs()
    grouped = Path("data/splits/grouped_manifest.csv")
    jobs: list[Job] = []

    for task, kind in (("three_stage", "calibrated"), ("binary", "uncalibrated")):
        for arch in MODEL_NAMES:
            for seed in SEEDS:
                run = FROZEN / "runs/grouped" / task / arch / f"seed-{seed}"
                checkpoint = run / ("calibration/best.calibrated.pt" if kind == "calibrated" else "best.pt")
                output = OUT / "predictions/grouped" / task / arch / f"seed-{seed}" / kind
                jobs.append(
                    Job(
                        "val_preds",
                        f"validation predictions {task}/{arch}/seed-{seed}",
                        _module(
                            "fresh_or_trash.cli.evaluate", checkpoint, grouped,
                            "--image-root", primary_root, "--output-dir", output,
                            "--split", "validation", "--batch-size", 24, "--workers", 0, "--device", "cuda",
                        ),
                        [output / "validation_metrics.json", output / "validation_predictions.csv"],
                    )
                )

    for arch in MODEL_NAMES:
        for seed in SEEDS:
            output = OUT / "external" / arch / f"seed-{seed}"
            jobs.append(
                Job(
                    "external",
                    f"external {arch}/seed-{seed}",
                    _module(
                        "fresh_or_trash.cli.external",
                        FROZEN / "runs/grouped/binary" / arch / f"seed-{seed}/best.pt",
                        "data/external_original_overlap_manifest.csv",
                        "--image-root", external_root, "--output-dir", output,
                        "--batch-size", 24, "--workers", 0, "--device", "cuda",
                        "--allowed-produce", "banana", "orange",
                        "--original-path-component", "Original Image",
                    ),
                    [output / "external_metrics.json", output / "external_predictions.csv", output / "external_per_produce_metrics.csv"],
                )
            )

    for fold in CV_FOLDS:
        manifest = Path(f"data/revision_splits/cv/cv_fold-{fold}_manifest.csv")
        for arch in MODEL_NAMES:
            run = OUT / "runs/cv/three_stage" / f"fold-{fold}" / arch / "seed-2026"
            jobs += _train_and_eval(
                "cv3", f"cv3 fold-{fold}/{arch}", config=configs["base"], manifest=manifest,
                image_root=primary_root, arch=arch, task="three_stage", seed=2026, run_dir=run,
            )

    for level in STAGED_LEVELS:
        manifest = Path(f"data/revision_splits/staged/{level}_manifest.csv")
        for arch in ABLATION_MODELS:
            for seed in SEEDS:
                run = OUT / "runs/staged" / level / arch / f"seed-{seed}"
                jobs += _train_and_eval(
                    "staged", f"staged {level}/{arch}/seed-{seed}", config=configs["base"], manifest=manifest,
                    image_root=primary_root, arch=arch, task="three_stage", seed=seed, run_dir=run,
                )

    for arch in MODEL_NAMES:
        for seed in SEEDS:
            output = OUT / "corruption" / arch / f"seed-{seed}"
            jobs.append(
                Job(
                    "corruption",
                    f"corruption {arch}/seed-{seed}",
                    [
                        sys.executable, "scripts/revision/corruption_predictions.py",
                        str(FROZEN / "runs/grouped/three_stage" / arch / f"seed-{seed}/calibration/best.calibrated.pt"),
                        str(grouped), "--image-root", str(primary_root), "--output-dir", str(output),
                        "--workers", "8", "--device", "cuda",
                    ],
                    [output / "corruption_predictions.csv", output / "corruption_metrics.csv"],
                )
            )

    for seed in SEEDS:
        run = OUT / "runs/scratch/three_stage/mobilenet_v2" / f"seed-{seed}"
        jobs += _train_and_eval(
            "scratch", f"scratch mobilenet_v2/seed-{seed}", config=configs["base"], manifest=grouped,
            image_root=primary_root, arch="mobilenet_v2", task="three_stage", seed=seed, run_dir=run, pretrained=False,
        )

    for name in LEARNING_RATES:
        for arch in MODEL_NAMES:
            run = OUT / "runs/lr" / name / arch / "seed-2026"
            jobs += _train_and_eval(
                "lr", f"lr {name}/{arch}/seed-2026", config=configs[name], manifest=grouped,
                image_root=primary_root, arch=arch, task="three_stage", seed=2026, run_dir=run,
            )
    for arch in MODEL_NAMES:
        best = _best_learning_rate(arch)
        if best and best != "base":
            for seed in SEEDS[1:]:
                run = OUT / "runs/lr" / best / arch / f"seed-{seed}"
                jobs += _train_and_eval(
                    "lr", f"lr follow-up {best}/{arch}/seed-{seed}", config=configs[best], manifest=grouped,
                    image_root=primary_root, arch=arch, task="three_stage", seed=seed, run_dir=run,
                )

    for name in RESOLUTIONS:
        for arch in MODEL_NAMES:
            seeds = SEEDS if arch in ABLATION_MODELS else SEEDS[:1]
            for seed in seeds:
                run = OUT / "runs/res" / name / arch / f"seed-{seed}"
                jobs += _train_and_eval(
                    "res", f"res {name}/{arch}/seed-{seed}", config=configs[name], manifest=grouped,
                    image_root=primary_root, arch=arch, task="three_stage", seed=seed, run_dir=run,
                )

    for fold in CV_FOLDS:
        manifest = Path(f"data/revision_splits/cv/cv_fold-{fold}_manifest.csv")
        for arch in MODEL_NAMES:
            run = OUT / "runs/cv/binary" / f"fold-{fold}" / arch / "seed-2026"
            jobs += _train_and_eval(
                "cv2", f"cv2 fold-{fold}/{arch}", config=configs["base"], manifest=manifest,
                image_root=primary_root, arch=arch, task="binary", seed=2026, run_dir=run,
            )
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--external-image-root", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", choices=GROUP_ORDER, default=list(GROUP_ORDER))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), environment.get("PYTHONPATH", "")]))
    log_dir = ROOT / OUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = ROOT / OUT / "queue_status.jsonl"
    failed: set[str] = set()
    order = {name: index for index, name in enumerate(GROUP_ORDER)}
    while True:
        jobs = [job for job in build_jobs(args.image_root, args.external_image_root) if job.group in args.groups]
        jobs.sort(key=lambda job: order[job.group])
        if args.dry_run:
            for job in jobs:
                state = "done" if job.complete() else ("ready" if job.ready() else "blocked")
                print(f"[{state}] {job.group}: {job.name}")
            print(f"{sum(job.complete() for job in jobs)}/{len(jobs)} complete")
            return 0
        pending = [job for job in jobs if not job.complete() and job.ready() and job.name not in failed]
        if not pending:
            break
        job = pending[0]
        remaining = len([item for item in jobs if not item.complete()])
        print(f"RUN ({remaining} pending) {job.name}", flush=True)
        log_path = log_dir / (job.name.replace("/", "_").replace(" ", "_") + ".log")
        started = time.perf_counter()
        with log_path.open("w", encoding="utf-8") as handle:
            result = subprocess.run(job.command, cwd=ROOT, env=environment, stdout=handle, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - started
        ok = result.returncode == 0 and job.complete()
        if not ok:
            failed.add(job.name)
        with status_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"job": job.name, "group": job.group, "ok": ok, "seconds": round(elapsed, 1)}) + "\n")
        print(f"{'OK' if ok else 'FAILED'} {job.name} ({elapsed:.0f} s)", flush=True)
    print(f"queue finished; failed={sorted(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
