"""Non-fine-tuned baselines on the frozen grouped split.

A train-prior majority classifier, an HSV colour-histogram logistic regression,
and logistic-regression probes on frozen ImageNet ResNet50 and ConvNeXt-Tiny
penultimate features are tuned on grouped validation macro-F1 only and scored
once on the locked test split for the three-stage and binary (Semi-Fresh
excluded) endpoints, next to frozen fine-tuned reference models. Image
features are cached under the output directory, so reruns only refit the
classifiers and recompute the intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from PIL import Image
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torchvision import transforms
from torchvision.models import get_model_weights

from fresh_or_trash.data import build_transform
from fresh_or_trash.manifest import load_manifest, resolve_image_path
from fresh_or_trash.models import build_model
from revision_common import (
    ANALYSIS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DISPLAY_NAMES,
    FROZEN,
    GROUPED_MANIFEST,
    PAPER_TABLES,
    REFERENCE_SEED,
    ROOT,
    SEEDS,
    apply_temperature,
    capture_groups,
    class_names,
    cluster_bootstrap,
    fit_temperature,
    frozen_test_predictions,
    interval,
    latex_table,
    macro_f1,
    metrics,
    percentile_interval,
    probability_columns,
    read_predictions,
    revision_validation_predictions,
    seed_mean_macro_f1,
    stack_seeds,
    write_json,
)


TOPIC = "baselines"
TABLE_NAME = "rev_baselines"
TASKS = ("three_stage", "binary")
PARTS = ("train", "validation", "test")
BASELINES = ("majority", "hsv_histogram", "probe_resnet50", "probe_convnext_tiny")
BASELINE_LABELS = {
    "majority": "Majority class (train prior)",
    "hsv_histogram": "HSV histogram + LR",
    "probe_resnet50": "ResNet50 linear probe",
    "probe_convnext_tiny": "ConvNeXt-Tiny linear probe",
}
HSV_FEATURES = ("hsv_joint_8x4x4", "hsv_channels_3x32")
BACKBONES = {"resnet50_imagenet": "resnet50", "convnext_tiny_imagenet": "convnext_tiny"}
FEATURE_NAMES = HSV_FEATURES + tuple(BACKBONES)
METHOD_FEATURES = {
    "hsv_histogram": HSV_FEATURES,
    "probe_resnet50": ("resnet50_imagenet",),
    "probe_convnext_tiny": ("convnext_tiny_imagenet",),
}
REFERENCE_MODELS = ("mobilenet_v2", "resnet50", "convnext_tiny")
REFERENCE_ROLES = {"mobilenet_v2": "smallest", "resnet50": "test leader", "convnext_tiny": "validation-selected"}
FROZEN_HEADLINE = {
    "three_stage": {"mobilenet_v2": 0.627, "resnet50": 0.690, "convnext_tiny": 0.645},
    "binary": {"mobilenet_v2": 0.813, "resnet50": 0.897, "convnext_tiny": 0.862},
}
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
CLASS_WEIGHTS = ("none", "balanced")
MAX_ITER = 10000
IMAGE_SIZE = 224
EMBEDDING_CACHE = ROOT / "data/cache/convnext_tiny_imagenet_embeddings.npy"
CACHE_CHECK_IMAGES = 16
CACHE_CHECK_TOLERANCE = 1e-3


class Pending(RuntimeError):
    """A required input does not exist yet; the part is reported as pending."""


class Timer:
    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - start


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_paths(directory: Path, name: str) -> tuple[Path, Path, Path]:
    return directory / f"{name}.npy", directory / f"{name}.sample_ids.txt", directory / f"{name}.json"


def load_cache(directory: Path, name: str, sample_ids: list[str]) -> tuple[np.ndarray, dict] | None:
    array_path, ids_path, metadata_path = cache_paths(directory, name)
    if not (array_path.is_file() and ids_path.is_file() and metadata_path.is_file()):
        return None
    if ids_path.read_text(encoding="utf-8").split() != sample_ids:
        print(f"warning: {array_path} was built for other samples; recomputing")
        return None
    return np.load(array_path), json.loads(metadata_path.read_text(encoding="utf-8"))


def save_cache(directory: Path, name: str, array: np.ndarray, sample_ids: list[str], metadata: dict) -> None:
    array_path, ids_path, metadata_path = cache_paths(directory, name)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(array_path, array.astype(np.float32))
    ids_path.write_text("\n".join(sample_ids) + "\n", encoding="utf-8")
    write_json(metadata, metadata_path)


def evaluation_steps(image_size: int = IMAGE_SIZE) -> tuple[transforms.Compose, transforms.Compose]:
    """Split the CNN evaluation transform into its PIL geometry and its tensor normalisation."""
    steps = list(build_transform(image_size, training=False).transforms)
    if not (isinstance(steps[0], transforms.Resize) and isinstance(steps[1], transforms.CenterCrop)):
        raise RuntimeError("build_transform no longer starts with Resize and CenterCrop")
    return transforms.Compose(steps[:2]), transforms.Compose(steps[2:])


def hsv_histograms(crop: Image.Image) -> dict[str, np.ndarray]:
    """L1-normalised joint 8x4x4 and concatenated per-channel 3x32 histograms of PIL HSV (0-255)."""
    hsv = np.asarray(crop.convert("HSV"), dtype=np.int64).reshape(-1, 3)
    joint = (hsv[:, 0] >> 5) * 16 + (hsv[:, 1] >> 6) * 4 + (hsv[:, 2] >> 6)
    channels = [np.bincount(hsv[:, channel] >> 3, minlength=32) for channel in range(3)]
    return {
        "hsv_joint_8x4x4": np.bincount(joint, minlength=128) / len(hsv),
        "hsv_channels_3x32": np.concatenate(channels) / (3 * len(hsv)),
    }


def save_checkpoint(path: Path, names: list[str], sample_ids: list[str], chunks: dict[str, list[np.ndarray]], seconds: dict[str, float]) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    arrays = {f"feature__{name}": np.concatenate(chunks[name]).astype(np.float32) for name in names}
    np.savez(temporary, names=np.asarray(names), sample_ids=np.asarray(sample_ids), seconds=json.dumps(seconds), **arrays)
    temporary.replace(path)


def load_checkpoint(path: Path, names: list[str], sample_ids: list[str]) -> tuple[int, dict[str, list[np.ndarray]], dict[str, float]] | None:
    if not path.is_file():
        return None
    with np.load(path) as saved:
        done = saved["sample_ids"].tolist()
        if saved["names"].tolist() != names or done != sample_ids[: len(done)]:
            return None
        chunks = {name: [saved[f"feature__{name}"]] for name in names}
        seconds = json.loads(str(saved["seconds"]))
    print(f"  resuming {'+'.join(names)} after {len(done)} images from {path.name}")
    return len(done), chunks, seconds


def extract(
    frame: pd.DataFrame,
    image_root: Path,
    names: list[str],
    batch_size: int,
    checkpoint: Path | None = None,
    checkpoint_every: int = 20,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """One CPU pass over the images that computes every requested feature family.

    With ``checkpoint`` the partial arrays are saved every ``checkpoint_every``
    batches and a rerun resumes from them, so a crash only loses one chunk.
    """
    geometry, to_tensor = evaluation_steps()
    models = {
        name: build_model(BACKBONES[name], pretrained=True).eval().to(memory_format=torch.channels_last)
        for name in names
        if name in BACKBONES
    }
    histogram_names = [name for name in names if name in HSV_FEATURES]
    sample_ids = frame["sample_id"].tolist()
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in names}
    seconds = {name: 0.0 for name in names} | {"decode_and_preprocess": 0.0, "pass_total": 0.0}
    first = 0
    resumed = load_checkpoint(checkpoint, names, sample_ids) if checkpoint is not None else None
    if resumed is not None:
        first, chunks, seconds = resumed
    started = time.perf_counter() - seconds["pass_total"]
    for batch_index, start in enumerate(range(first, len(frame), batch_size)):
        clock = time.perf_counter()
        crops = []
        for _, row in frame.iloc[start : start + batch_size].iterrows():
            with Image.open(resolve_image_path(row, image_root)) as source:
                crops.append(geometry(source.convert("RGB")))
        batch = torch.stack([to_tensor(crop) for crop in crops]).contiguous(memory_format=torch.channels_last) if models else None
        seconds["decode_and_preprocess"] += time.perf_counter() - clock
        if histogram_names:
            clock = time.perf_counter()
            histograms = [hsv_histograms(crop) for crop in crops]
            for name in histogram_names:
                chunks[name].append(np.stack([histogram[name] for histogram in histograms]))
                seconds[name] += (time.perf_counter() - clock) / len(histogram_names)
        for name, model in models.items():
            clock = time.perf_counter()
            with torch.inference_mode():
                chunks[name].append(model.extract_features(batch).float().numpy())
            seconds[name] += time.perf_counter() - clock
        done = min(start + batch_size, len(frame))
        seconds["pass_total"] = time.perf_counter() - started
        if checkpoint is not None and (batch_index + 1) % checkpoint_every == 0 and done < len(frame):
            save_checkpoint(checkpoint, names, sample_ids[:done], chunks, seconds)
        if batch_index % checkpoint_every == 0 or done == len(frame):
            print(f"  {'+'.join(names)}: {done}/{len(frame)} images, {seconds['pass_total']:.0f}s", flush=True)
    return {name: np.concatenate(chunks[name]).astype(np.float32) for name in names}, seconds


def l2_normalise(array: np.ndarray) -> np.ndarray:
    return array / np.linalg.norm(array, axis=1, keepdims=True)


def verify_embedding_cache(manifest: pd.DataFrame, args: argparse.Namespace) -> tuple[dict, np.ndarray | None]:
    """Recompute a few ConvNeXt-Tiny embeddings on CPU and compare them with the shared cache."""
    cache_path = args.embedding_cache
    ids_path = cache_path.with_suffix(".sample_ids.txt")
    if not (cache_path.is_file() and ids_path.is_file()):
        raise Pending(f"embedding cache {cache_path} not found")
    cached_ids = ids_path.read_text(encoding="utf-8").split()
    cache = np.load(cache_path)
    index = {sample_id: row for row, sample_id in enumerate(cached_ids)}
    sample_ids = manifest["sample_id"].tolist()
    covered = all(sample_id in index for sample_id in sample_ids)
    aligned = cache[[index[sample_id] for sample_id in sample_ids]] if covered else None
    digest = file_sha256(cache_path)
    record_path = args.output_dir / "embedding_cache_check.json"
    if record_path.is_file():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
        if previous.get("cache_sha256") == digest and previous.get("passed") and covered:
            return previous | {"reused_previous_check": True}, aligned
    if not args.image_root.is_dir():
        raise Pending(f"image root {args.image_root} not found, so the embedding cache cannot be verified")
    candidates = manifest[manifest["sample_id"].isin(index)]
    subset = candidates.sample(n=min(CACHE_CHECK_IMAGES, len(candidates)), random_state=REFERENCE_SEED).sort_values("sample_id")
    fresh, seconds = extract(subset, args.image_root, ["convnext_tiny_imagenet"], args.batch_size)
    recomputed = l2_normalise(fresh["convnext_tiny_imagenet"])
    stored = cache[[index[sample_id] for sample_id in subset["sample_id"]]]
    cosine = np.sum(recomputed * stored, axis=1) / (np.linalg.norm(recomputed, axis=1) * np.linalg.norm(stored, axis=1))
    absolute = float(np.abs(recomputed - stored).max())
    record = {
        "cache": str(cache_path.relative_to(ROOT) if cache_path.is_relative_to(ROOT) else cache_path),
        "cache_sha256": digest,
        "cache_shape": list(cache.shape),
        "covers_manifest": covered,
        "checked_sample_ids": subset["sample_id"].tolist(),
        "checked_splits": subset["split"].value_counts().sort_index().to_dict(),
        "recomputation": "build_model('convnext_tiny', pretrained=True).extract_features, build_transform(224) evaluation geometry, CPU float32, L2-normalised",
        "max_abs_difference": absolute,
        "min_cosine_similarity": float(cosine.min()),
        "tolerance_max_abs_difference": CACHE_CHECK_TOLERANCE,
        "passed": bool(absolute <= CACHE_CHECK_TOLERANCE and covered),
        "seconds": seconds["pass_total"],
        "reused_previous_check": False,
    }
    write_json(record, record_path)
    return record, aligned


def feature_metadata(name: str, seconds: dict[str, float], families: list[str], count: int, args: argparse.Namespace) -> dict:
    metadata = {
        "feature": name,
        "images": count,
        "image_size": IMAGE_SIZE,
        "geometry": "Resize(256, bicubic) + CenterCrop(224) from fresh_or_trash.data.build_transform(224, training=False)",
        "device": "cpu",
        "torch_threads": args.threads,
        "batch_size": args.batch_size,
        "pass_families": families,
        "pass_seconds": seconds["pass_total"],
        "decode_and_preprocess_seconds": seconds["decode_and_preprocess"],
        "compute_seconds": seconds[name],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
    if name in BACKBONES:
        metadata |= {
            "architecture": BACKBONES[name],
            "weights": str(get_model_weights(BACKBONES[name]).DEFAULT),
            "layer": "penultimate (FreshnessClassifier.extract_features, classifier removed), not normalised",
            "memory_format": "channels_last",
        }
    else:
        metadata["bins"] = "joint H8 x S4 x V4" if name == "hsv_joint_8x4x4" else "H32 + S32 + V32 concatenated"
    return metadata


def build_features(manifest: pd.DataFrame, args: argparse.Namespace, timer: Timer, status: dict[str, str]) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    sample_ids = manifest["sample_id"].tolist()
    directory = args.output_dir / "features"
    features: dict[str, np.ndarray] = {}
    provenance: dict[str, dict] = {}
    for name in FEATURE_NAMES:
        loaded = load_cache(directory, name, sample_ids)
        if loaded is not None:
            features[name], metadata = loaded
            provenance[name] = metadata | {"source": "feature cache", "from_cache": True}
    blocked: dict[str, str] = {}
    if "convnext_tiny_imagenet" not in features and not args.no_embedding_cache:
        try:
            with timer.stage("embedding_cache_check"):
                check, aligned = verify_embedding_cache(manifest, args)
            if check["passed"] and aligned is not None:
                features["convnext_tiny_imagenet"] = aligned
                provenance["convnext_tiny_imagenet"] = {
                    "source": check["cache"],
                    "from_cache": True,
                    "note": "L2-normalised ConvNeXt-Tiny ImageNet embeddings at 224 px, verified against CPU re-extraction",
                    "check": check,
                }
            else:
                print(f"warning: embedding cache check failed (max abs diff {check['max_abs_difference']:.2e}); extracting instead")
        except Pending as reason:
            blocked["convnext_tiny_imagenet"] = str(reason)
    missing = [name for name in FEATURE_NAMES if name not in features and name not in blocked]
    if missing and not args.image_root.is_dir():
        blocked |= {name: f"image root {args.image_root} not found" for name in missing}
        missing = []
    if missing:
        print(f"extracting {', '.join(missing)} for {len(manifest)} images on {args.threads} CPU threads")
        checkpoint = directory / "partial_extraction.npz"
        directory.mkdir(parents=True, exist_ok=True)
        with timer.stage("extract:" + "+".join(missing)):
            arrays, seconds = extract(manifest, args.image_root, missing, args.batch_size, checkpoint=checkpoint)
        for name in missing:
            metadata = feature_metadata(name, seconds, missing, len(manifest), args)
            save_cache(directory, name, arrays[name], sample_ids, metadata)
            features[name] = arrays[name]
            provenance[name] = metadata | {"source": "extracted this run", "from_cache": False}
        checkpoint.unlink(missing_ok=True)
    for name in FEATURE_NAMES:
        status[f"features/{name}"] = "done" if name in features else f"pending: {blocked.get(name, 'not computed')}"
    return features, provenance


def task_partitions(manifest: pd.DataFrame, task: str) -> dict[str, pd.DataFrame]:
    names = class_names(task)
    frame = manifest[manifest["freshness_label"].isin(names)].copy()
    frame["target_index"] = frame["freshness_label"].map({name: index for index, name in enumerate(names)}).astype(int)
    frame = frame.sort_values("sample_id", kind="stable")
    return {part: frame[frame["split"] == part].reset_index(drop=True) for part in PARTS}


def majority_baseline(parts: dict[str, pd.DataFrame], task: str) -> dict:
    prior = np.bincount(parts["train"]["target_index"], minlength=len(class_names(task))) / len(parts["train"])
    return {
        "config": {"feature": "none", "C": np.nan, "class_weight": "", "train_prior": prior.tolist()},
        "probabilities": {part: np.tile(prior, (len(frame), 1)) for part, frame in parts.items()},
        "grid": [],
    }


def logistic_baseline(
    parts: dict[str, pd.DataFrame], task: str, method: str, features: dict[str, np.ndarray], rows: dict[str, int]
) -> dict:
    """Standardised features + L2 logistic regression; C, class weights and representation by validation macro-F1."""
    count = len(class_names(task))
    targets = {part: frame["target_index"].to_numpy() for part, frame in parts.items()}
    grid: list[dict] = []
    best: dict | None = None
    for name in METHOD_FEATURES[method]:
        arrays = {part: features[name][[rows[sample_id] for sample_id in frame["sample_id"]]].astype(np.float64) for part, frame in parts.items()}
        if name in BACKBONES:
            arrays = {part: l2_normalise(array) for part, array in arrays.items()}
        scaler = StandardScaler().fit(arrays["train"])
        scaled = {part: scaler.transform(array) for part, array in arrays.items()}
        for c_value in C_GRID:
            for weight in CLASS_WEIGHTS:
                clock = time.perf_counter()
                model = LogisticRegression(C=c_value, class_weight=None if weight == "none" else weight, max_iter=MAX_ITER)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit(scaled["train"], targets["train"])
                validation = model.predict_proba(scaled["validation"])
                train_predictions = model.predict(scaled["train"])
                row = {
                    "task": task,
                    "method": method,
                    "feature": name,
                    "feature_dim": int(scaled["train"].shape[1]),
                    "C": c_value,
                    "class_weight": weight,
                    "train_macro_f1": macro_f1(targets["train"], train_predictions, count),
                    "validation_macro_f1": macro_f1(targets["validation"], validation.argmax(axis=1), count),
                    "validation_accuracy": float(accuracy_score(targets["validation"], validation.argmax(axis=1))),
                    "validation_nll": float(log_loss(targets["validation"], validation, labels=list(range(count)))),
                    "iterations": int(np.max(model.n_iter_)),
                    "converged": not any(issubclass(item.category, ConvergenceWarning) for item in caught),
                    "fit_seconds": time.perf_counter() - clock,
                    "selected": False,
                }
                grid.append(row)
                if best is None or row["validation_macro_f1"] > best["row"]["validation_macro_f1"]:
                    best = {"row": row, "model": model, "scaled": scaled}
    assert best is not None
    best["row"]["selected"] = True
    probabilities = {part: best["model"].predict_proba(array) for part, array in best["scaled"].items()}
    config = {key: best["row"][key] for key in ("feature", "C", "class_weight", "iterations", "converged")}
    return {"config": config, "probabilities": probabilities, "grid": grid}


def f1_columns(task: str) -> dict[str, str]:
    return {name: f"f1_{name.lower().replace('-', '_')}" for name in class_names(task)}


def per_class_f1(summary: dict, task: str) -> dict[str, float]:
    return {column: float(summary["per_class"][name]["f1-score"]) for name, column in f1_columns(task).items()}


def evaluate_baseline(task: str, method: str, parts: dict[str, pd.DataFrame], outcome: dict, groups: np.ndarray, replicates: int) -> tuple[dict, np.ndarray]:
    """Test metrics of one baseline; the constant train prior of the majority rule is not temperature-scaled."""
    targets = parts["test"]["target_index"].to_numpy()
    validation_targets = parts["validation"]["target_index"].to_numpy()
    raw = outcome["probabilities"]["test"]
    temperature = np.nan if method == "majority" else fit_temperature(outcome["probabilities"]["validation"], validation_targets)
    summary = metrics(targets, raw, task)
    scaled = metrics(targets, apply_temperature(raw, temperature), task) if np.isfinite(temperature) else {"ece": np.nan, "nll": np.nan}
    validation = metrics(validation_targets, outcome["probabilities"]["validation"], task)
    values = cluster_bootstrap(groups, seed_mean_macro_f1(targets, raw[np.newaxis]), replicates=replicates, seed=BOOTSTRAP_SEED)
    low, high = percentile_interval(values)
    row = {
        "task": task,
        "method": method,
        "label": BASELINE_LABELS[method],
        "kind": "baseline",
        "role": "",
        "seed_count": 1,
        "test_count": len(targets),
        "test_capture_groups": int(len(np.unique(groups))),
        "feature": outcome["config"]["feature"],
        "C": outcome["config"]["C"],
        "class_weight": outcome["config"]["class_weight"],
        "temperature": temperature,
        "validation_macro_f1": validation["macro_f1"],
        "validation_macro_f1_sd": np.nan,
        "accuracy": summary["accuracy"],
        "accuracy_sd": np.nan,
        "balanced_accuracy": summary["balanced_accuracy"],
        "balanced_accuracy_sd": np.nan,
        "macro_f1": summary["macro_f1"],
        "macro_f1_sd": np.nan,
        "macro_f1_ci_low": low,
        "macro_f1_ci_high": high,
        "ece_uncalibrated": summary["ece"],
        "ece_uncalibrated_sd": np.nan,
        "ece_temperature_scaled": scaled["ece"],
        "ece_temperature_scaled_sd": np.nan,
        "nll_uncalibrated": summary["nll"],
        "nll_temperature_scaled": scaled["nll"],
    } | per_class_f1(summary, task)
    return row, values


def frozen_uncalibrated_three_stage(arch: str, seed: int) -> pd.DataFrame:
    path = FROZEN / "evaluation/grouped/three_stage" / arch / f"seed-{seed}" / "uncalibrated/test_predictions.csv"
    return read_predictions(path, "three_stage")


def frozen_run_value(task: str, arch: str, seed: int, *parts: str) -> dict:
    return json.loads((FROZEN / "runs/grouped" / task / arch / f"seed-{seed}" / Path(*parts)).read_text(encoding="utf-8"))


def evaluate_references(
    task: str, order: list[str], targets: np.ndarray, groups: np.ndarray, validation_count: int, replicates: int
) -> tuple[list[dict], list[dict], dict[str, np.ndarray], str]:
    """Frozen fine-tuned rows; the binary validation-temperature ECE waits for the revision queue."""
    rows, seed_rows, values = [], [], {}
    calibration_pending: list[str] = []
    for arch in REFERENCE_MODELS:
        try:
            frames = [frozen_test_predictions(task, arch, seed) for seed in SEEDS]
            uncalibrated_frames = [frozen_uncalibrated_three_stage(arch, seed) for seed in SEEDS] if task == "three_stage" else frames
        except FileNotFoundError as error:
            raise Pending(f"frozen predictions missing ({error.filename})") from error
        reference_order, reference_targets, probabilities = stack_seeds(frames, task)
        _, uncalibrated_targets, uncalibrated = stack_seeds(uncalibrated_frames, task)
        if reference_order != order or not np.array_equal(reference_targets, targets) or not np.array_equal(uncalibrated_targets, targets):
            raise ValueError(f"Frozen {task} predictions of {arch} do not match the manifest test split")
        for seed_index, seed in enumerate(SEEDS):
            summary = metrics(targets, probabilities[seed_index], task)
            raw = metrics(targets, uncalibrated[seed_index], task) if task == "three_stage" else summary
            temperature, scaled = np.nan, {"ece": np.nan, "nll": np.nan}
            if task == "three_stage":
                temperature = frozen_run_value(task, arch, seed, "calibration", "temperature.json")["temperature"]
                scaled = summary
            else:
                try:
                    validation = revision_validation_predictions(task, arch, seed)
                    _, validation_targets, validation_probabilities = stack_seeds([validation], task)
                    if len(validation_targets) != validation_count:
                        raise ValueError(f"{arch} seed {seed} validation predictions cover {len(validation_targets)} images, expected {validation_count}")
                    temperature = fit_temperature(validation_probabilities[0], validation_targets)
                    scaled = metrics(targets, apply_temperature(probabilities[seed_index], temperature), task)
                except FileNotFoundError as error:
                    missing = Path(error.filename)
                    calibration_pending.append(str(missing.relative_to(ROOT) if missing.is_relative_to(ROOT) else missing))
            seed_rows.append(
                {
                    "task": task,
                    "method": f"finetuned_{arch}",
                    "architecture": arch,
                    "seed": seed,
                    "probabilities": "calibrated" if task == "three_stage" else "uncalibrated",
                    "validation_macro_f1": frozen_run_value(task, arch, seed, "run.json")["validation_macro_f1"],
                    "accuracy": summary["accuracy"],
                    "balanced_accuracy": summary["balanced_accuracy"],
                    "macro_f1": summary["macro_f1"],
                    "temperature": temperature,
                    "ece_uncalibrated": raw["ece"],
                    "ece_temperature_scaled": scaled["ece"],
                    "nll_uncalibrated": raw["nll"],
                    "nll_temperature_scaled": scaled["nll"],
                }
                | per_class_f1(summary, task)
            )
        values[f"finetuned_{arch}"] = cluster_bootstrap(groups, seed_mean_macro_f1(targets, probabilities), replicates=replicates, seed=BOOTSTRAP_SEED)
        low, high = percentile_interval(values[f"finetuned_{arch}"])
        frame = pd.DataFrame([row for row in seed_rows if row["architecture"] == arch])
        row = {
            "task": task,
            "method": f"finetuned_{arch}",
            "label": DISPLAY_NAMES[arch],
            "kind": "fine_tuned",
            "role": REFERENCE_ROLES[arch],
            "seed_count": len(SEEDS),
            "test_count": len(targets),
            "test_capture_groups": int(len(np.unique(groups))),
            "feature": "end-to-end fine-tuned",
            "C": np.nan,
            "class_weight": "",
            "temperature": float(frame["temperature"].mean(skipna=False)),
            "macro_f1_ci_low": low,
            "macro_f1_ci_high": high,
            "nll_uncalibrated": float(frame["nll_uncalibrated"].mean()),
            "nll_temperature_scaled": float(frame["nll_temperature_scaled"].mean(skipna=False)),
        }
        for column in ("validation_macro_f1", "accuracy", "balanced_accuracy", "macro_f1", "ece_uncalibrated", "ece_temperature_scaled"):
            row[column] = float(frame[column].mean(skipna=False))
            row[f"{column}_sd"] = float(frame[column].std(ddof=1, skipna=False))
        row |= {column: float(frame[column].mean()) for column in f1_columns(task).values()}
        rows.append(row)
    pending = "" if not calibration_pending else f"revision validation predictions missing ({len(calibration_pending)} files, first {calibration_pending[0]})"
    return rows, seed_rows, values, pending


def reproduction_check(rows: list[dict]) -> list[dict]:
    """Compare recomputed three-seed means with the frozen headline numbers and aggregate CSV."""
    summary_path = FROZEN / "aggregate/benchmark_summary.csv"
    aggregate = pd.read_csv(summary_path) if summary_path.is_file() else pd.DataFrame()
    records = []
    for row in rows:
        task, arch = row["task"], row["method"].removeprefix("finetuned_")
        state = "calibrated" if task == "three_stage" else "uncalibrated"
        frozen_value = np.nan
        if not aggregate.empty:
            match = aggregate[(aggregate["setting"] == "grouped") & (aggregate["task"] == task) & (aggregate["calibration_state"] == state) & (aggregate["architecture"] == arch)]
            frozen_value = float(match["macro_f1_mean"].iloc[0]) if len(match) == 1 else np.nan
        records.append(
            {
                "task": task,
                "architecture": arch,
                "recomputed_macro_f1_mean": row["macro_f1"],
                "headline_macro_f1": FROZEN_HEADLINE[task][arch],
                "matches_headline_3dp": round(row["macro_f1"], 3) == FROZEN_HEADLINE[task][arch],
                "aggregate_macro_f1_mean": frozen_value,
                "abs_difference_vs_aggregate": abs(row["macro_f1"] - frozen_value) if np.isfinite(frozen_value) else np.nan,
            }
        )
    return records


def prediction_frame(frame: pd.DataFrame, probabilities: np.ndarray, task: str) -> pd.DataFrame:
    names = class_names(task)
    predicted = probabilities.argmax(axis=1)
    output = pd.DataFrame(
        {
            "sample_id": frame["sample_id"],
            "target_index": frame["target_index"],
            "produce_type": frame["produce_type"],
            "source_path": frame["source_path"],
            "predicted_index": predicted,
            "predicted_label": [names[index] for index in predicted],
            "confidence": probabilities.max(axis=1),
        }
    )
    for column, values in zip(probability_columns(task), probabilities.T):
        output[column] = values
    return output


def paired_differences(rows: list[dict], values: dict[tuple[str, str], np.ndarray]) -> list[dict]:
    """Fine-tuned minus baseline macro-F1 on identical capture-group resamples."""
    lookup = {(row["task"], row["method"]): row for row in rows}
    records = []
    for task in TASKS:
        for arch in REFERENCE_MODELS:
            reference = (task, f"finetuned_{arch}")
            for method in BASELINES:
                if reference not in values or (task, method) not in values:
                    continue
                difference = values[reference] - values[(task, method)]
                low, high = percentile_interval(difference)
                records.append(
                    {
                        "task": task,
                        "reference": arch,
                        "baseline": method,
                        "macro_f1_difference": lookup[reference]["macro_f1"] - lookup[(task, method)]["macro_f1"],
                        "ci_low": low,
                        "ci_high": high,
                        "share_replicates_not_above_zero": float(np.mean(difference <= 0.0)),
                    }
                )
    return records


def write_table(path: Path, rows: list[dict], replicates: int) -> None:
    lookup = {(row["task"], row["method"]): row for row in rows}
    three, binary = lookup[("three_stage", "finetuned_resnet50")], lookup[("binary", "finetuned_resnet50")]
    labels = {"majority": "Majority class", "hsv_histogram": "HSV histogram", "probe_resnet50": "ResNet50 probe", "probe_convnext_tiny": "ConvNeXt-T probe"}
    body: list[list[str]] = [[r"\multicolumn{4}{@{}l}{\emph{Without fine-tuning (one fit)}}"]]
    for method in BASELINES:
        first, second = lookup[("three_stage", method)], lookup[("binary", method)]
        body.append(
            [
                labels[method],
                f"{first['macro_f1']:.3f} {interval(first['macro_f1_ci_low'], first['macro_f1_ci_high'])}",
                f"{first['balanced_accuracy']:.3f}",
                f"{second['macro_f1']:.3f}",
            ]
        )
    body.append([r"\midrule\multicolumn{4}{@{}l}{\emph{Fine-tuned (frozen runs, three seeds)}}"])
    for arch in REFERENCE_MODELS:
        first, second = lookup[("three_stage", f"finetuned_{arch}")], lookup[("binary", f"finetuned_{arch}")]
        body.append(
            [
                DISPLAY_NAMES[arch],
                f"{first['macro_f1']:.3f} {interval(first['macro_f1_ci_low'], first['macro_f1_ci_high'])}",
                rf"${first['balanced_accuracy']:.3f}\pm{first['balanced_accuracy_sd']:.3f}$",
                rf"${second['macro_f1']:.3f}\pm{second['macro_f1_sd']:.3f}$",
            ]
        )
    caption = (
        f"Baselines without fine-tuning versus fine-tuned CNNs, grouped test split ({three['test_count']} three-stage images; "
        f"{binary['test_count']} binary Fresh/Rotten images). HSV-histogram and probe baselines are logistic regressions on colour "
        "histograms or frozen ImageNet penultimate features, with features, $C$, and class weights chosen on validation macro-F1. "
        f"Brackets: 95\\% cluster-bootstrap intervals over {three['test_capture_groups']} test capture groups ({replicates:,} replicates). "
        "Fine-tuned rows: three-seed means (interval of the seed mean), otherwise mean$\\pm$SD; ResNet50 led the test, "
        "ConvNeXt-Tiny was validation-selected, MobileNetV2 is the smallest."
    )
    latex_table(
        path,
        caption=caption,
        label="tab:rev-baselines",
        headers=[
            "Method",
            r"\shortstack{3-stage macro-F1\\{[}95\% CI{]}}",
            r"\shortstack{3-stage\\bal.\ acc.}",
            r"\shortstack{Binary\\macro-F1}",
        ],
        rows=body,
    )


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def run(args: argparse.Namespace) -> int:
    timer = Timer()
    status: dict[str, str] = {}
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with timer.stage("total"):
        with timer.stage("load_manifest"):
            manifest = load_manifest(args.manifest, require_split=True)
        with timer.stage("features"):
            features, provenance = build_features(manifest, args, timer, status)
        rows_by_id = {sample_id: row for row, sample_id in enumerate(manifest["sample_id"])}
        results: list[dict] = []
        seed_rows: list[dict] = []
        grid_rows: list[dict] = []
        replicates: dict[tuple[str, str], np.ndarray] = {}
        for task in TASKS:
            parts = task_partitions(manifest, task)
            order = parts["test"]["sample_id"].tolist()
            targets = parts["test"]["target_index"].to_numpy()
            groups = capture_groups(order, manifest=args.manifest)
            for method in BASELINES:
                part = f"{task}/{method}"
                needed = METHOD_FEATURES.get(method, ())
                absent = [name for name in needed if name not in features]
                if absent:
                    status[part] = f"pending: features {', '.join(absent)}"
                    print(f"warning: {part} skipped ({status[part]})")
                    continue
                with timer.stage(f"fit:{part}"):
                    outcome = majority_baseline(parts, task) if method == "majority" else logistic_baseline(parts, task, method, features, rows_by_id)
                with timer.stage(f"evaluate:{part}"):
                    row, values = evaluate_baseline(task, method, parts, outcome, groups, args.replicates)
                grid_rows.extend(outcome["grid"])
                results.append(row)
                replicates[(task, method)] = values
                temperature = row["temperature"]
                kinds = [("uncalibrated", outcome["probabilities"]["test"])]
                if np.isfinite(temperature):
                    kinds.append(("calibrated", apply_temperature(outcome["probabilities"]["test"], temperature)))
                for kind, probabilities in kinds:
                    destination = output / "predictions" / task / method / kind
                    destination.mkdir(parents=True, exist_ok=True)
                    prediction_frame(parts["test"], probabilities, task).to_csv(destination / "test_predictions.csv", index=False)
                status[part] = "done"
                print(
                    f"{part}: validation macro-F1 {row['validation_macro_f1']:.3f} ({row['feature']}, C={row['C']}, weight={row['class_weight'] or '-'}); "
                    f"test macro-F1 {row['macro_f1']:.3f} {interval(row['macro_f1_ci_low'], row['macro_f1_ci_high'])}, "
                    f"balanced accuracy {row['balanced_accuracy']:.3f}, ECE {row['ece_uncalibrated']:.3f} raw / {row['ece_temperature_scaled']:.3f} T={temperature:.3f}"
                )
            part = f"{task}/references"
            try:
                with timer.stage(f"evaluate:{part}"):
                    reference_rows, reference_seed_rows, reference_values, calibration_pending = evaluate_references(
                        task, order, targets, groups, len(parts["validation"]), args.replicates
                    )
            except Pending as reason:
                status[part] = f"pending: {reason}"
                status[f"{task}/references_temperature_scaled_ece"] = f"pending: {reason}"
                print(f"warning: {part} pending: {reason}")
                continue
            results.extend(reference_rows)
            seed_rows.extend(reference_seed_rows)
            replicates |= {(task, method): values for method, values in reference_values.items()}
            status[part] = "done"
            status[f"{task}/references_temperature_scaled_ece"] = f"pending: {calibration_pending}" if calibration_pending else "done"
            if calibration_pending:
                print(f"warning: {task} reference temperature-scaled ECE pending: {calibration_pending}")
            for row in reference_rows:
                print(
                    f"{task}/{row['method']}: test macro-F1 {row['macro_f1']:.3f}+-{row['macro_f1_sd']:.3f} "
                    f"{interval(row['macro_f1_ci_low'], row['macro_f1_ci_high'])}, balanced accuracy {row['balanced_accuracy']:.3f}"
                )

        with timer.stage("write_outputs"):
            frame = pd.DataFrame(results)
            frame.to_csv(output / "test_metrics.csv", index=False)
            pd.DataFrame(grid_rows).to_csv(output / "validation_grid.csv", index=False)
            pd.DataFrame(seed_rows).to_csv(output / "reference_seed_metrics.csv", index=False)
            differences = paired_differences(results, replicates)
            pd.DataFrame(differences).to_csv(output / "paired_differences.csv", index=False)
            status["paired_differences"] = "done" if differences else "pending: no baseline/reference pair available"
            np.savez_compressed(
                output / "bootstrap_replicates.npz",
                **{f"{task}__{method}": values for (task, method), values in sorted(replicates.items())},
            )
            reproduction = reproduction_check([row for row in results if row["kind"] == "fine_tuned"])
            pd.DataFrame(reproduction).to_csv(output / "reproduction_check.csv", index=False)
            mismatches = [record for record in reproduction if not record["matches_headline_3dp"]]
            for record in mismatches:
                print(f"warning: {record['task']} {record['architecture']} recomputed {record['recomputed_macro_f1_mean']:.4f} != headline {record['headline_macro_f1']}")
            required = {(task, method) for task in TASKS for method in BASELINES} | {
                (task, f"finetuned_{arch}") for task in TASKS for arch in REFERENCE_MODELS
            }
            available = {(row["task"], row["method"]) for row in results}
            table_path = args.table_dir / f"{TABLE_NAME}.tex"
            if required <= available:
                write_table(table_path, results, args.replicates)
                status["table"] = "done"
            else:
                missing_rows = sorted(f"{task}/{method}" for task, method in required - available)
                status["table"] = f"pending: rows {', '.join(missing_rows)}"
                print(f"warning: table not written; missing {', '.join(missing_rows)}")
    timings = {
        "stages_seconds": dict(sorted(timer.seconds.items())),
        "feature_extraction": {
            name: {
                key: provenance[name].get(key)
                for key in ("source", "from_cache", "pass_seconds", "compute_seconds", "decode_and_preprocess_seconds", "pass_families", "images")
                if key in provenance[name]
            }
            for name in sorted(provenance)
        },
        "device": "cpu",
        "threads": args.threads,
        "note": "Wall-clock on a shared CPU while the GPU revision queue was running; treat as indicative.",
    }
    write_json(json_ready(timings), output / "timings.json")
    summary = {
        "scope": {
            "split": "frozen grouped split (data/splits/grouped_manifest.csv)",
            "images": {task: {part: len(frame) for part, frame in task_partitions(manifest, task).items()} for task in TASKS},
            "selection": "validation macro-F1 only (ties keep the earlier grid entry: smaller C, no class weights, joint HSV)",
            "C_grid": list(C_GRID),
            "class_weight_grid": list(CLASS_WEIGHTS),
            "hsv_representations": list(HSV_FEATURES),
            "probe_features": "L2-normalised penultimate ImageNet features, standardised on train",
            "logistic_regression": f"sklearn {sklearn.__version__} LogisticRegression(lbfgs, L2, max_iter={MAX_ITER})",
            "bootstrap": {"replicates": args.replicates, "seed": BOOTSTRAP_SEED, "unit": "capture_group_id", "interval": "95% percentile"},
            "ece": "15 equal-width confidence bins; temperature-scaled = validation-fitted temperature on saved probabilities",
            "references": {arch: REFERENCE_ROLES[arch] for arch in REFERENCE_MODELS},
        },
        "results": [{key: row[key] for key in ("task", "method", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "balanced_accuracy", "accuracy", "ece_uncalibrated", "ece_temperature_scaled", "validation_macro_f1")} for row in results],
        "feature_provenance": provenance,
        "reproduction_check": reproduction,
        "reproduction_mismatches": len(mismatches),
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__},
    }
    write_json(json_ready(summary), output / "summary.json")
    write_json(dict(sorted(status.items())), output / "status.json")
    pending = {part: state for part, state in status.items() if state != "done"}
    print(f"status: {len(status) - len(pending)} done, {len(pending)} pending -> {output / 'status.json'}")
    for part, state in sorted(pending.items()):
        print(f"  {part}: {state}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=GROUPED_MANIFEST)
    parser.add_argument("--image-root", type=Path, default=ROOT.parent / "data/raw")
    parser.add_argument("--output-dir", type=Path, default=ANALYSIS / TOPIC)
    parser.add_argument("--table-dir", type=Path, default=PAPER_TABLES)
    parser.add_argument("--embedding-cache", type=Path, default=EMBEDDING_CACHE)
    parser.add_argument("--no-embedding-cache", action="store_true", help="extract ConvNeXt-Tiny features instead of reusing the verified cache")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
