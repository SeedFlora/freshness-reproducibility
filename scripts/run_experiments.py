"""Run the frozen paper experiment graph without selecting on locked-test metrics.

The script is intentionally resumable at whole-artifact boundaries. It skips a
complete command, refuses partial outputs, and never overwrites a result.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import torch

from fresh_or_trash.experiments import (
    REFERENCE_SEED,
    TRAINING_SEEDS,
    json_records,
    require_complete_training_runs,
    select_top_architectures,
    validate_protocol,
    validate_split_inputs,
)
from fresh_or_trash.models import MODEL_NAMES


ALL_STAGES = (
    "grouped_train",
    "selection",
    "naive_train",
    "calibration",
    "grouped_test",
    "calibrated_test",
    "naive_test",
    "diagnostics",
    "external",
    "aggregate",
)

# Keep the orchestration completion check in lockstep with the aggregation
# deliverables.  Checking only a single table and figure could otherwise mark
# a partially interrupted aggregation as complete.
AGGREGATE_CSV_OUTPUTS = (
    "training_runs.csv",
    "evaluation_seed_metrics.csv",
    "benchmark_seed_metrics.csv",
    "benchmark_summary.csv",
    "rank_changes.csv",
    "leakage_ablation.csv",
    "per_produce_seed_metrics.csv",
    "per_produce_summary.csv",
    "shelf_interval_seed_metrics.csv",
    "shelf_interval_summary.csv",
    "calibration_seed_metrics.csv",
    "calibration_summary.csv",
    "corruption_metrics.csv",
    "corruption_summary.csv",
    "external_metrics.csv",
    "efficiency.csv",
    "validation_selection.csv",
)
AGGREGATE_LATEX_OUTPUTS = (
    "experiment_main_benchmark_table.tex",
    "experiment_rank_change_table.tex",
    "experiment_leakage_ablation_table.tex",
    "experiment_calibration_table.tex",
    "experiment_robustness_table.tex",
    "experiment_produce_table.tex",
    "experiment_shelf_interval_table.tex",
    "experiment_external_validation_table.tex",
    "experiment_efficiency_table.tex",
    "experiment_results_values.tex",
)
AGGREGATE_FIGURES = (
    "model_comparison.png",
    "calibration_comparison.png",
    "corruption_robustness.png",
    "leakage_ablation.png",
    "external_per_fruit.png",
    "shelf_interval_rotten_probability.png",
)


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_once_or_verify(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to replace a different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class CommandRunner:
    def __init__(self, project_root: Path, *, dry_run: bool) -> None:
        self.project_root = project_root
        self.dry_run = dry_run
        self.environment = os.environ.copy()
        source_path = str(project_root / "src")
        current = self.environment.get("PYTHONPATH", "")
        self.environment["PYTHONPATH"] = os.pathsep.join(value for value in (source_path, current) if value)

    def run(self, command: Sequence[str], *, expected: Iterable[Path], label: str) -> None:
        expected_paths = tuple(expected)
        present = [path.is_file() for path in expected_paths]
        if present and all(present):
            print(f"SKIP complete: {label}")
            return
        if any(present):
            found = [str(path) for path, exists in zip(expected_paths, present) if exists]
            missing = [str(path) for path, exists in zip(expected_paths, present) if not exists]
            raise RuntimeError(f"Partial output for {label}; found={found}, missing={missing}")
        rendered = subprocess.list2cmdline(list(command)) if os.name == "nt" else shlex.join(command)
        if self.dry_run:
            print(f"DRY RUN {label}: {rendered}")
            return
        print(f"RUN {label}: {rendered}", flush=True)
        subprocess.run(
            list(command),
            cwd=self.project_root,
            env=self.environment,
            check=True,
        )
        missing_after = [str(path) for path in expected_paths if not path.is_file()]
        if missing_after:
            raise RuntimeError(f"Command succeeded but did not create required artifacts: {missing_after}")


def _module_command(module: str, *arguments: object) -> list[str]:
    return [sys.executable, "-m", module, *(str(argument) for argument in arguments)]


def _training_dir(root: Path, setting: str, task: str, architecture: str, seed: int) -> Path:
    return root / "runs" / setting / task / architecture / f"seed-{seed}"


def _train(
    runner: CommandRunner,
    *,
    protocol: Path,
    manifest: Path,
    image_root: Path,
    results_root: Path,
    setting: str,
    task: str,
    architectures: Iterable[str],
    device: str,
) -> None:
    for architecture in architectures:
        for seed in TRAINING_SEEDS:
            output = _training_dir(results_root, setting, task, architecture, seed)
            command = _module_command(
                "fresh_or_trash.cli.train",
                "--config",
                protocol,
                manifest,
                "--image-root",
                image_root,
                "--model",
                architecture,
                "--task",
                task,
                "--output-dir",
                output,
                "--seed",
                seed,
                "--device",
                device,
                "--pretrained",
                "--auxiliary-produce-head",
                "--no-evaluate-test",
            )
            runner.run(
                command,
                expected=(output / "best.pt", output / "run.json"),
                label=f"train {setting}/{task}/{architecture}/seed-{seed}",
            )


def _selection(results_root: Path, *, dry_run: bool) -> pd.DataFrame | None:
    if dry_run:
        print("DRY RUN selection: rank only the 15 grouped three-stage validation_macro_f1 values")
        return None
    runs = require_complete_training_runs(
        results_root,
        setting="grouped",
        task="three_stage",
        architectures=MODEL_NAMES,
    )
    selection = select_top_architectures(runs, top_k=2)
    csv_path = results_root / "selection" / "grouped_three_stage_validation_selection.csv"
    json_path = results_root / "selection" / "grouped_three_stage_validation_selection.json"
    csv_text = selection.to_csv(index=False, lineterminator="\n")
    payload = {
        "selection_scope": "grouped_three_stage_validation_only",
        "selection_metric": "mean_validation_macro_f1_across_matched_seeds",
        "test_metrics_used": False,
        "training_seeds": list(TRAINING_SEEDS),
        "ranking": json_records(selection),
    }
    _write_once_or_verify(csv_path, csv_text)
    _write_once_or_verify(json_path, _json_text(payload))
    print(f"validation selection -> {json_path}")
    return selection


def _read_selection(results_root: Path) -> pd.DataFrame:
    path = results_root / "selection" / "grouped_three_stage_validation_selection.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Run the selection stage first: {path}")
    frame = pd.read_csv(path)
    required = {
        "architecture",
        "validation_macro_f1_mean",
        "validation_macro_f1_std",
        "seed_count",
        "validation_rank",
        "selection_uses_test_metrics",
    }
    if required - set(frame.columns):
        raise ValueError(f"Invalid selection artifact: {path}")
    uses_test = frame["selection_uses_test_metrics"].astype(str).str.casefold().isin({"true", "1"})
    if uses_test.any():
        raise ValueError("Selection artifact claims test metrics were used")
    recomputed = select_top_architectures(
        require_complete_training_runs(
            results_root,
            setting="grouped",
            task="three_stage",
            architectures=MODEL_NAMES,
        ),
        top_k=2,
    )
    columns = [
        "architecture",
        "validation_macro_f1_mean",
        "validation_macro_f1_std",
        "seed_count",
        "validation_rank",
    ]
    pd.testing.assert_frame_equal(
        frame.sort_values("validation_rank")[columns].reset_index(drop=True),
        recomputed[columns].reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    # Use the independently recomputed ranking so a stale CSV cannot steer a
    # naive ablation or external evaluation after a resumed invocation.
    return recomputed


def _calibrate(
    runner: CommandRunner,
    *,
    grouped_manifest: Path,
    image_root: Path,
    results_root: Path,
    device: str,
    batch_size: int,
    workers: int,
) -> None:
    for architecture in MODEL_NAMES:
        for seed in TRAINING_SEEDS:
            run_dir = _training_dir(results_root, "grouped", "three_stage", architecture, seed)
            checkpoint = run_dir / "best.pt"
            temperature = run_dir / "calibration" / "temperature.json"
            calibrated = run_dir / "calibration" / "best.calibrated.pt"
            command = _module_command(
                "fresh_or_trash.cli.calibrate",
                checkpoint,
                grouped_manifest,
                "--image-root",
                image_root,
                "--output",
                temperature,
                "--output-checkpoint",
                calibrated,
                "--batch-size",
                batch_size,
                "--workers",
                workers,
                "--device",
                device,
            )
            runner.run(
                command,
                expected=(temperature, calibrated),
                label=f"calibrate grouped/three_stage/{architecture}/seed-{seed}",
            )


def _evaluate_primary(
    runner: CommandRunner,
    *,
    grouped_manifest: Path,
    naive_manifest: Path,
    image_root: Path,
    results_root: Path,
    device: str,
    batch_size: int,
    workers: int,
    setting: str,
    task: str,
    architectures: Iterable[str],
    calibrated: bool,
) -> None:
    manifest = grouped_manifest if setting == "grouped" else naive_manifest
    calibration_name = "calibrated" if calibrated else "uncalibrated"
    for architecture in architectures:
        for seed in TRAINING_SEEDS:
            run_dir = _training_dir(results_root, setting, task, architecture, seed)
            checkpoint = run_dir / "best.pt"
            if calibrated:
                if setting != "grouped" or task != "three_stage":
                    raise ValueError("Temperature-scaled evaluation is reserved for grouped three-stage runs")
                checkpoint = run_dir / "calibration" / "best.calibrated.pt"
            output = results_root / "evaluation" / setting / task / architecture / f"seed-{seed}" / calibration_name
            command = _module_command(
                "fresh_or_trash.cli.evaluate",
                checkpoint,
                manifest,
                "--image-root",
                image_root,
                "--output-dir",
                output,
                "--split",
                "test",
                "--batch-size",
                batch_size,
                "--workers",
                workers,
                "--device",
                device,
            )
            runner.run(
                command,
                expected=(output / "test_metrics.json", output / "test_predictions.csv"),
                label=f"test {setting}/{task}/{architecture}/seed-{seed}/{calibration_name}",
            )


def _diagnostics(
    runner: CommandRunner,
    *,
    grouped_manifest: Path,
    image_root: Path,
    results_root: Path,
    device: str,
    batch_size: int,
    workers: int,
    severities: Sequence[int],
    warmup: int,
    iterations: int,
) -> None:
    for architecture in MODEL_NAMES:
        run_dir = _training_dir(results_root, "grouped", "three_stage", architecture, REFERENCE_SEED)
        checkpoint = run_dir / "calibration" / "best.calibrated.pt"
        corruption_dir = results_root / "diagnostics" / f"reference-seed-{REFERENCE_SEED}" / architecture / "corruptions"
        corruption_command = _module_command(
            "fresh_or_trash.cli.evaluate",
            checkpoint,
            grouped_manifest,
            "--image-root",
            image_root,
            "--output-dir",
            corruption_dir,
            "--split",
            "test",
            "--batch-size",
            batch_size,
            "--workers",
            workers,
            "--device",
            device,
            "--corruptions",
            "--severities",
            *severities,
        )
        runner.run(
            corruption_command,
            expected=(
                corruption_dir / "test_metrics.json",
                corruption_dir / "test_predictions.csv",
                corruption_dir / "corruption_metrics.csv",
            ),
            label=f"corruption benchmark {architecture}/seed-{REFERENCE_SEED}",
        )

        for profile_device in ("cpu", "cuda"):
            if profile_device == "cuda" and not torch.cuda.is_available() and not runner.dry_run:
                raise RuntimeError(
                    "The frozen protocol requires a CUDA batch-one profile, but CUDA is unavailable. "
                    "Run the diagnostics stage on a CUDA host; no substitute value will be emitted."
                )
            profile_output = (
                results_root
                / "efficiency"
                / f"reference-seed-{REFERENCE_SEED}"
                / architecture
                / f"{profile_device}.json"
            )
            amp_flag = "--amp" if profile_device == "cuda" else "--no-amp"
            profile_command = _module_command(
                "fresh_or_trash.cli.profile",
                checkpoint,
                "--output",
                profile_output,
                "--device",
                profile_device,
                "--batch-size",
                1,
                "--warmup",
                warmup,
                "--iterations",
                iterations,
                amp_flag,
            )
            runner.run(
                profile_command,
                expected=(profile_output,),
                label=f"{profile_device} forward-pass profile {architecture}/seed-{REFERENCE_SEED}",
            )


def _external(
    runner: CommandRunner,
    *,
    external_manifest: Path,
    external_image_root: Path,
    results_root: Path,
    device: str,
    batch_size: int,
    workers: int,
    selected: Sequence[str],
    original_path_component: str,
) -> None:
    if len(selected) != 2:
        raise ValueError("External validation requires exactly two validation-selected architectures")
    for architecture in selected:
        checkpoint = _training_dir(
            results_root, "grouped", "binary", architecture, REFERENCE_SEED
        ) / "best.pt"
        output = results_root / "external" / architecture / f"seed-{REFERENCE_SEED}"
        command = _module_command(
            "fresh_or_trash.cli.external",
            checkpoint,
            external_manifest,
            "--image-root",
            external_image_root,
            "--output-dir",
            output,
            "--batch-size",
            batch_size,
            "--workers",
            workers,
            "--device",
            device,
            "--allowed-produce",
            "banana",
            "orange",
            "--original-path-component",
            original_path_component,
        )
        runner.run(
            command,
            expected=(
                output / "external_metrics.json",
                output / "external_predictions.csv",
                output / "external_per_produce_metrics.csv",
            ),
            label=f"external original-only binary {architecture}/seed-{REFERENCE_SEED}",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen, validation-gated paper experiment graph")
    parser.add_argument("--protocol", type=Path, default=Path("config/experiments.yaml"))
    parser.add_argument("--grouped-manifest", type=Path, default=Path("data/splits/grouped_manifest.csv"))
    parser.add_argument("--naive-manifest", type=Path, default=Path("data/splits/naive_manifest.csv"))
    parser.add_argument("--split-provenance", type=Path, default=Path("data/splits/split_provenance.json"))
    parser.add_argument("--external-manifest", type=Path, default=Path("data/external_original_overlap_manifest.csv"))
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--external-image-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, default=Path("results/experiments"))
    parser.add_argument("--paper-generated-dir", type=Path, default=Path("paper/generated"))
    parser.add_argument("--figure-dir", type=Path, default=Path("paper/figures/results"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stages", nargs="+", choices=(*ALL_STAGES, "all"), default=["all"])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    protocol = validate_protocol(args.protocol)
    split_snapshot = validate_split_inputs(args.grouped_manifest, args.naive_manifest, args.split_provenance)
    results_root = args.results_root
    stages = list(ALL_STAGES) if "all" in args.stages else list(dict.fromkeys(args.stages))
    runner = CommandRunner(project_root, dry_run=args.dry_run)
    if not args.dry_run:
        snapshot = {
            "protocol_file": str(args.protocol),
            "protocol": protocol,
            "split_inputs": split_snapshot,
            "test_selection_policy": "validation_only; test metrics are evaluated after selection",
        }
        _write_once_or_verify(results_root / "protocol_snapshot.json", _json_text(snapshot))

    data_config = protocol["data"]
    evaluation_config = protocol["evaluation"]
    if "grouped_train" in stages:
        for task in ("three_stage", "binary"):
            _train(
                runner,
                protocol=args.protocol,
                manifest=args.grouped_manifest,
                image_root=args.image_root,
                results_root=results_root,
                setting="grouped",
                task=task,
                architectures=MODEL_NAMES,
                device=args.device,
            )
    selection: pd.DataFrame | None = None
    if "selection" in stages:
        selection = _selection(results_root, dry_run=args.dry_run)
    if selection is None and any(stage in stages for stage in ("naive_train", "naive_test", "external")):
        if args.dry_run and not (results_root / "selection" / "grouped_three_stage_validation_selection.csv").is_file():
            print("DRY RUN dependent stages require the architecture chosen after grouped validation; commands are deferred")
        else:
            selection = _read_selection(results_root)

    if "naive_train" in stages and selection is not None:
        best = str(selection.iloc[0]["architecture"])
        architectures = list(dict.fromkeys(("mobilenet_v2", best)))
        _train(
            runner,
            protocol=args.protocol,
            manifest=args.naive_manifest,
            image_root=args.image_root,
            results_root=results_root,
            setting="naive",
            task="three_stage",
            architectures=architectures,
            device=args.device,
        )
    if "calibration" in stages:
        _calibrate(
            runner,
            grouped_manifest=args.grouped_manifest,
            image_root=args.image_root,
            results_root=results_root,
            device=args.device,
            batch_size=int(data_config["batch_size"]),
            workers=int(data_config["num_workers"]),
        )
    if "grouped_test" in stages:
        for task in ("three_stage", "binary"):
            _evaluate_primary(
                runner,
                grouped_manifest=args.grouped_manifest,
                naive_manifest=args.naive_manifest,
                image_root=args.image_root,
                results_root=results_root,
                device=args.device,
                batch_size=int(data_config["batch_size"]),
                workers=int(data_config["num_workers"]),
                setting="grouped",
                task=task,
                architectures=MODEL_NAMES,
                calibrated=False,
            )
    if "calibrated_test" in stages:
        _evaluate_primary(
            runner,
            grouped_manifest=args.grouped_manifest,
            naive_manifest=args.naive_manifest,
            image_root=args.image_root,
            results_root=results_root,
            device=args.device,
            batch_size=int(data_config["batch_size"]),
            workers=int(data_config["num_workers"]),
            setting="grouped",
            task="three_stage",
            architectures=MODEL_NAMES,
            calibrated=True,
        )
    if "naive_test" in stages and selection is not None:
        best = str(selection.iloc[0]["architecture"])
        architectures = list(dict.fromkeys(("mobilenet_v2", best)))
        _evaluate_primary(
            runner,
            grouped_manifest=args.grouped_manifest,
            naive_manifest=args.naive_manifest,
            image_root=args.image_root,
            results_root=results_root,
            device=args.device,
            batch_size=int(data_config["batch_size"]),
            workers=int(data_config["num_workers"]),
            setting="naive",
            task="three_stage",
            architectures=architectures,
            calibrated=False,
        )
    if "diagnostics" in stages:
        _diagnostics(
            runner,
            grouped_manifest=args.grouped_manifest,
            image_root=args.image_root,
            results_root=results_root,
            device=args.device,
            batch_size=int(data_config["batch_size"]),
            workers=int(data_config["num_workers"]),
            severities=[int(value) for value in evaluation_config["corruption_severities"]],
            warmup=int(evaluation_config["latency_warmup"]),
            iterations=int(evaluation_config["latency_iterations"]),
        )
    if "external" in stages and selection is not None:
        selected = selection.head(2)["architecture"].astype(str).tolist()
        _external(
            runner,
            external_manifest=args.external_manifest,
            external_image_root=args.external_image_root,
            results_root=results_root,
            device=args.device,
            batch_size=int(data_config["batch_size"]),
            workers=int(data_config["num_workers"]),
            selected=selected,
            original_path_component=str(protocol["experiment"]["external_source_path_component"]),
        )
    if "aggregate" in stages:
        aggregate_command = [
            sys.executable,
            str(project_root / "scripts" / "aggregate_experiments.py"),
            "--protocol",
            str(args.protocol),
            "--grouped-manifest",
            str(args.grouped_manifest),
            "--naive-manifest",
            str(args.naive_manifest),
            "--split-provenance",
            str(args.split_provenance),
            "--results-root",
            str(results_root),
            "--paper-generated-dir",
            str(args.paper_generated_dir),
            "--figure-dir",
            str(args.figure_dir),
        ]
        runner.run(
            aggregate_command,
            expected=(
                results_root / "aggregate" / "experiment_summary.json",
                *(results_root / "aggregate" / filename for filename in AGGREGATE_CSV_OUTPUTS),
                *(args.paper_generated_dir / filename for filename in AGGREGATE_LATEX_OUTPUTS),
                *(args.figure_dir / filename for filename in AGGREGATE_FIGURES),
            ),
            label="aggregate machine-readable results, LaTeX tables, and figures",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
