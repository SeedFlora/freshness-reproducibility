"""Deployment profile: compute, size, end-to-end CPU latency, INT8 quantisation, GPU energy.

Profiles the five calibrated reference-seed (2026) three-stage checkpoints:
parameters, FLOPs, and file sizes; end-to-end CPU latency on real grouped-test
JPEGs (decode, evaluation transform, forward pass, temperature-scaled softmax)
at several intra-op thread counts; ONNX export (opset 17, batch 1) with ONNX
Runtime FP32 and static INT8 (QDQ, per-channel weights, training-split
calibration) latency and locked-test accuracy; and, when CUDA is available,
batch-1 and batch-32 FP32/FP16-autocast latency with NVML energy per image.
``--quick`` runs a few images per stage into a scratch directory.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import logging
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Sequence

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.shape_inference import quant_pre_process
from PIL import Image
from torch import nn

from fresh_or_trash.data import build_transform
from fresh_or_trash.manifest import load_manifest, resolve_image_path
from fresh_or_trash.models import FreshnessClassifier, load_checkpoint
from revision_common import (
    ANALYSIS,
    DISPLAY_NAMES,
    FROZEN,
    GROUPED_MANIFEST,
    MODEL_ORDER,
    PAPER_TABLES,
    REFERENCE_SEED,
    REVISION,
    ROOT,
    THREE_STAGE,
    capture_groups,
    cluster_bootstrap,
    frozen_test_predictions,
    latex_table,
    macro_f1,
    metrics,
    percentile_interval,
    probability_columns,
    write_json,
)


TOPIC = "deployment"
TASK = "three_stage"
OUTPUT = ANALYSIS / TOPIC
MODEL_OUTPUT = REVISION / TOPIC / "onnx"
QUICK_OUTPUT = Path(tempfile.gettempdir()) / "fresh-or-trash-deployment-quick"
DEFAULT_IMAGE_ROOT = ROOT.parent / "data/raw"
TABLE_NAME = "rev_deployment"
IMAGE_SIZE = 224
OPSET = 17
INPUT_NAME = "image"
OUTPUT_NAME = "probabilities"
GPU_BATCH = 32
STAGES = ("decode", "transform", "preprocess", "forward", "total")
BACKENDS = ("pytorch_fp32", "ort_fp32", "ort_int8")
FULL = {"images": 100, "warmup": 10, "iterations": 500, "calibration_images": 200, "accuracy_images": 0, "bootstrap": 2000}
QUICK = {"images": 8, "warmup": 2, "iterations": 20, "calibration_images": 8, "accuracy_images": 24, "bootstrap": 200}
FULL_GPU = {"gpu_min_seconds": 10.0, "idle_seconds": 10.0, "settle_seconds": 5.0, "gpu_warmup": 20}
QUICK_GPU = {"gpu_min_seconds": 0.5, "idle_seconds": 1.0, "settle_seconds": 0.0, "gpu_warmup": 3}
PARTS = (
    "environment",
    "complexity_and_size",
    "cpu_latency_pytorch",
    "onnx_export",
    "int8_quantization",
    "onnx_accuracy",
    "cpu_latency_onnxruntime",
    "gpu_latency_energy",
    "latex_table",
)


@dataclass(frozen=True)
class Settings:
    quick: bool
    models: tuple[str, ...]
    threads: tuple[int, ...]
    ort_threads: tuple[int, ...]
    images: int
    warmup: int
    iterations: int
    calibration_images: int
    accuracy_images: int
    bootstrap: int
    gpu_min_seconds: float
    idle_seconds: float
    settle_seconds: float
    gpu_warmup: int
    skip_gpu: bool
    output_dir: Path
    model_dir: Path
    table_path: Path
    image_root: Path

    @property
    def work_threads(self) -> int:
        return max(self.threads)


@dataclass
class Workload:
    latency: pd.DataFrame
    calibration: pd.DataFrame
    accuracy: pd.DataFrame


class DeployedClassifier(nn.Module):
    """Backbone, freshness head, and temperature-scaled softmax: the deployed inference graph."""

    def __init__(self, model: FreshnessClassifier, temperature: float) -> None:
        super().__init__()
        self.backbone = model.backbone
        self.head = model.freshness_head
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        return torch.softmax(self.head(features) / self.temperature, dim=1)


class TrainImageReader(CalibrationDataReader):
    """Feeds evaluation-transformed training images to the ONNX Runtime calibrator."""

    def __init__(self, paths: Sequence[Path], transform: Callable) -> None:
        self.paths = list(paths)
        self.transform = transform
        self.position = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.position >= len(self.paths):
            return None
        with Image.open(self.paths[self.position]) as source:
            array = self.transform(source.convert("RGB")).unsqueeze(0).numpy()
        self.position += 1
        return {INPUT_NAME: array}

    def rewind(self) -> None:
        self.position = 0

    def __len__(self) -> int:
        return len(self.paths)


class PowerSampler(threading.Thread):
    """Samples board power at a fixed interval; ``stop`` returns the trapezoid-integrated joules."""

    def __init__(self, read_watts: Callable[[], float], interval: float = 0.01) -> None:
        super().__init__(daemon=True)
        self.read_watts = read_watts
        self.interval = interval
        self.samples: list[tuple[float, float]] = []
        self.halt = threading.Event()

    def run(self) -> None:
        while not self.halt.is_set():
            self.samples.append((time.perf_counter(), self.read_watts()))
            self.halt.wait(self.interval)

    def stop(self) -> float:
        self.halt.set()
        self.join()
        self.samples.append((time.perf_counter(), self.read_watts()))
        return integrate_power(self.samples)


class EnergyMeter:
    """NVML energy of one GPU: the cumulative counter when supported, else ~10 ms power sampling."""

    def __init__(self, nvml: object, handle: object, method: str) -> None:
        self.nvml = nvml
        self.handle = handle
        self.method = method

    @classmethod
    def open(cls, uuid: str | None = None, *, prefer_counter: bool = True) -> "EnergyMeter":
        import pynvml

        pynvml.nvmlInit()
        handle = None
        if uuid:
            with contextlib.suppress(pynvml.NVMLError):
                handle = pynvml.nvmlDeviceGetHandleByUUID(uuid if uuid.startswith("GPU-") else f"GPU-{uuid}")
        if handle is None:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        probes = [("power_sampling", pynvml.nvmlDeviceGetPowerUsage)]
        if prefer_counter:
            probes.insert(0, ("energy_counter", pynvml.nvmlDeviceGetTotalEnergyConsumption))
        method = "unsupported"
        for name, probe in probes:
            try:
                probe(handle)
            except pynvml.NVMLError:
                continue
            method = name
            break
        return cls(pynvml, handle, method)

    def _counter_joules(self) -> float:
        return self.nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000.0

    def _watts(self) -> float:
        return self.nvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0

    @contextlib.contextmanager
    def measure(self) -> Iterator[dict[str, float]]:
        reading: dict[str, float] = {}
        sampler = PowerSampler(self._watts) if self.method == "power_sampling" else None
        if sampler is not None:
            sampler.start()
        start_joules = self._counter_joules() if self.method == "energy_counter" else 0.0
        started = time.perf_counter()
        try:
            yield reading
        finally:
            seconds = time.perf_counter() - started
            if self.method == "energy_counter":
                joules = self._counter_joules() - start_joules
            elif sampler is not None:
                joules = sampler.stop()
                reading["power_samples"] = len(sampler.samples)
            else:
                joules = float("nan")
            reading.update(seconds=seconds, joules=joules, watts=joules / seconds if seconds > 0 else float("nan"))

    def idle(self, seconds: float) -> dict[str, float]:
        with self.measure() as reading:
            time.sleep(seconds)
        return reading

    def describe(self) -> dict[str, object]:
        nvml = self.nvml
        info: dict[str, object] = {"energy_method": self.method}
        for key, call in (
            ("name", lambda: nvml.nvmlDeviceGetName(self.handle)),
            ("driver", nvml.nvmlSystemGetDriverVersion),
            ("nvml_version", nvml.nvmlSystemGetNVMLVersion),
            ("memory_bytes", lambda: int(nvml.nvmlDeviceGetMemoryInfo(self.handle).total)),
            ("power_limit_w", lambda: nvml.nvmlDeviceGetEnforcedPowerLimit(self.handle) / 1000.0),
            ("utilization_percent", lambda: int(nvml.nvmlDeviceGetUtilizationRates(self.handle).gpu)),
            (
                "other_compute_processes",
                lambda: sum(item.pid != os.getpid() for item in nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)),
            ),
        ):
            try:
                value = call()
                info[key] = value.decode() if isinstance(value, bytes) else value
            except Exception as error:
                info[key] = f"unavailable: {type(error).__name__}"
        return info


def integrate_power(samples: Sequence[tuple[float, float]]) -> float:
    if len(samples) < 2:
        return float("nan")
    times, watts = np.asarray(samples, dtype=float).T
    return float(np.trapezoid(watts, times))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def physical_cores() -> int:
    with contextlib.suppress(ImportError):
        import psutil

        count = psutil.cpu_count(logical=False)
        if count:
            return int(count)
    return os.cpu_count() or 1


def cpu_name() -> str:
    if sys.platform == "win32":
        with contextlib.suppress(OSError):
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def checkpoint_path(arch: str) -> Path:
    return FROZEN / "runs/grouped" / TASK / arch / f"seed-{REFERENCE_SEED}" / "calibration/best.calibrated.pt"


def stratified_sample(pool: pd.DataFrame, count: int, seed: int, by: Sequence[str] = ("produce_type", "freshness_label")) -> pd.DataFrame:
    """Deterministic sample with produce-by-stage quotas (largest-remainder allocation)."""
    pool = pool.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    if count <= 0 or count >= len(pool):
        return pool
    sizes = pool.groupby(list(by), sort=True).size()
    quota = sizes / sizes.sum() * count
    allocation = np.floor(quota).astype(int)
    remainder = int(count - allocation.sum())
    allocation.loc[(quota - allocation).sort_values(ascending=False, kind="mergesort").index[:remainder]] += 1
    rng = np.random.default_rng(seed)
    picks = []
    for key, group in pool.groupby(list(by), sort=True):
        size = int(allocation.loc[key])
        if size:
            picks.append(group.iloc[np.sort(rng.choice(len(group), size=size, replace=False))])
    return pd.concat(picks).sort_values("sample_id", kind="mergesort").reset_index(drop=True)


def select_workload(settings: Settings) -> Workload:
    manifest = load_manifest(GROUPED_MANIFEST, require_split=True)
    test = manifest[manifest["split"] == "test"]
    train = manifest[manifest["split"] == "train"]
    calibration = stratified_sample(train, settings.calibration_images, REFERENCE_SEED)
    if set(calibration["split"]) != {"train"} or set(calibration["capture_group_id"]) & set(test["capture_group_id"]):
        raise RuntimeError("INT8 calibration images must come from training capture groups only")
    return Workload(
        latency=stratified_sample(test, settings.images, REFERENCE_SEED),
        calibration=calibration,
        accuracy=stratified_sample(test, settings.accuracy_images, REFERENCE_SEED + 1),
    )


def image_paths(frame: pd.DataFrame, image_root: Path) -> list[Path]:
    return [resolve_image_path(row, image_root) for _, row in frame.iterrows()]


def load_deployed(arch: str, device: str | torch.device = "cpu") -> tuple[DeployedClassifier, FreshnessClassifier, dict]:
    model, payload = load_checkpoint(checkpoint_path(arch), device)
    if payload.get("temperature") is None:
        raise ValueError(f"{arch}: calibrated checkpoint carries no temperature")
    if list(payload.get("freshness_classes", THREE_STAGE)) != list(THREE_STAGE):
        raise ValueError(f"{arch}: unexpected class order {payload.get('freshness_classes')}")
    if int(payload.get("image_size", IMAGE_SIZE)) != IMAGE_SIZE:
        raise ValueError(f"{arch}: expected {IMAGE_SIZE}-pixel inputs, checkpoint says {payload.get('image_size')}")
    model.eval()
    return DeployedClassifier(model, float(payload["temperature"])).to(device).eval(), model, payload


def count_flops(module: nn.Module, image_size: int) -> int:
    from torch.utils.flop_counter import FlopCounterMode

    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        module(torch.zeros(1, 3, image_size, image_size))
    return int(counter.get_total_flops())


def export_onnx(module: nn.Module, path: Path, image_size: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    example = (torch.zeros(1, 3, image_size, image_size),)
    names = {"input_names": [INPUT_NAME], "output_names": [OUTPUT_NAME], "opset_version": OPSET}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.onnx.export(module, example, str(path), dynamo=False, do_constant_folding=True, **names)
            exporter = "torchscript"
        except Exception as error:
            torch.onnx.export(module, example, str(path), dynamo=True, external_data=False, **names)
            exporter = f"dynamo (torchscript exporter failed: {type(error).__name__})"
    onnx.checker.check_model(str(path))
    return exporter


def float_head_nodes(model: onnx.ModelProto) -> list[str]:
    """Nodes from the graph output back to the classifier's linear layer, which stay FP32."""
    producers = {output: node for node in model.graph.node for output in node.output}
    names: list[str] = []
    tensor: str | None = model.graph.output[0].name
    while tensor in producers:
        node = producers[tensor]
        names.append(node.name)
        if node.op_type in {"Gemm", "MatMul"}:
            break
        tensor = next((name for name in node.input if name in producers), None)
    return names


def int8_coverage(model: onnx.ModelProto) -> str:
    producers = {output: node.op_type for node in model.graph.node for output in node.output}
    counts: dict[str, list[int]] = {}
    for node in model.graph.node:
        if node.op_type in {"Conv", "Gemm", "MatMul"}:
            quantized = len(node.input) > 1 and producers.get(node.input[1]) == "DequantizeLinear"
            tally = counts.setdefault(node.op_type, [0, 0])
            tally[0] += int(quantized)
            tally[1] += 1
    return "; ".join(f"{op} {done}/{total}" for op, (done, total) in sorted(counts.items()))


class MessageCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@contextlib.contextmanager
def collected_warnings() -> Iterator[list[str]]:
    """Route the quantiser's root-logger warnings into a list instead of stderr."""
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    collector = MessageCollector()
    root.handlers = [collector]
    root.setLevel(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield collector.messages
    finally:
        root.handlers = handlers
        root.setLevel(level)


def summarize_messages(messages: Sequence[str]) -> str:
    templates: dict[str, int] = {}
    for message in messages:
        template = re.sub(r"'[^']*'", "'*'", message.strip().splitlines()[0] if message.strip() else "")
        templates[template] = templates.get(template, 0) + 1
    return "; ".join(f"{count}x {template}" for template, count in templates.items())


@contextlib.contextmanager
def calibration_threads(threads: int) -> Iterator[None]:
    """Bound the ONNX Runtime calibration session's intra-op threads (it otherwise uses every core)."""
    from onnxruntime.quantization import calibrate

    original = calibrate.CalibraterBase.create_inference_session

    def create_inference_session(self: calibrate.CalibraterBase) -> None:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.infer_session = ort.InferenceSession(
            self.augmented_model_path, sess_options=options, providers=self.execution_providers
        )

    calibrate.CalibraterBase.create_inference_session = create_inference_session
    try:
        yield
    finally:
        calibrate.CalibraterBase.create_inference_session = original


def quantize_int8(fp32_path: Path, int8_path: Path, reader: CalibrationDataReader, threads: int) -> dict[str, object]:
    with collected_warnings() as messages, tempfile.TemporaryDirectory(prefix="deployment-int8-", ignore_cleanup_errors=True) as scratch:
        prepared = Path(scratch) / "prepared.onnx"
        try:
            quant_pre_process(str(fp32_path), str(prepared))
            preprocessing = "symbolic and ONNX shape inference, ORT basic optimisation"
        except Exception as error:
            quant_pre_process(str(fp32_path), str(prepared), skip_symbolic_shape=True)
            preprocessing = f"ONNX shape inference, ORT basic optimisation (symbolic inference failed: {type(error).__name__})"
        model = onnx.load(str(prepared))
        for index, node in enumerate(model.graph.node):
            if not node.name:
                node.name = f"{node.op_type}_{index}"
        excluded = float_head_nodes(model)
        onnx.save(model, str(prepared))
        with calibration_threads(threads):
            quantize_static(
                str(prepared),
                str(int8_path),
                reader,
                quant_format=QuantFormat.QDQ,
                per_channel=True,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                calibrate_method=CalibrationMethod.MinMax,
                nodes_to_exclude=excluded,
            )
    onnx.checker.check_model(str(int8_path))
    return {
        "int8_preprocessing": preprocessing,
        "int8_fp32_nodes": ";".join(excluded),
        "int8_coverage": int8_coverage(onnx.load(str(int8_path))),
        "int8_quantizer_warnings": summarize_messages(messages),
    }


def ort_session(path: Path, threads: int) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def torch_runner(module: nn.Module) -> Callable[[torch.Tensor], np.ndarray]:
    def infer(batch: torch.Tensor) -> np.ndarray:
        with torch.inference_mode():
            return module(batch).numpy()

    return infer


def ort_runner(session: ort.InferenceSession) -> Callable[[torch.Tensor], np.ndarray]:
    def infer(batch: torch.Tensor) -> np.ndarray:
        return session.run(None, {INPUT_NAME: batch.numpy()})[0]

    return infer


def run_pipeline(path: Path, transform: Callable, infer: Callable[[torch.Tensor], np.ndarray]) -> dict[str, float]:
    start = time.perf_counter_ns()
    with Image.open(path) as source:
        image = source.convert("RGB")
    decoded = time.perf_counter_ns()
    batch = transform(image).unsqueeze(0)
    prepared = time.perf_counter_ns()
    infer(batch)
    finished = time.perf_counter_ns()
    return {
        "decode_ms": (decoded - start) / 1e6,
        "transform_ms": (prepared - decoded) / 1e6,
        "preprocess_ms": (prepared - start) / 1e6,
        "forward_ms": (finished - prepared) / 1e6,
        "total_ms": (finished - start) / 1e6,
    }


def time_pipeline(
    paths: Sequence[Path], transform: Callable, infer: Callable[[torch.Tensor], np.ndarray], warmup: int
) -> list[dict[str, float]]:
    for index in range(warmup):
        run_pipeline(paths[index % len(paths)], transform, infer)
    return [run_pipeline(path, transform, infer) for path in paths]


def summarize_latency(samples: pd.DataFrame) -> pd.DataFrame:
    keys = ["architecture", "runtime", "precision", "threads"]
    records = []
    for key, group in samples.groupby(keys, sort=False):
        for stage in STAGES:
            values = group[f"{stage}_ms"].to_numpy(dtype=float)
            records.append(
                dict(zip(keys, key))
                | {
                    "stage": stage,
                    "n": len(values),
                    "mean_ms": float(values.mean()),
                    "median_ms": float(np.median(values)),
                    "p95_ms": float(np.percentile(values, 95)),
                    "sd_ms": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
                    "min_ms": float(values.min()),
                    "max_ms": float(values.max()),
                }
            )
    return pd.DataFrame.from_records(records)


def accuracy_pass(
    module: nn.Module, sessions: dict[str, ort.InferenceSession], paths: Sequence[Path], transform: Callable
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    runners = {"pytorch_fp32": torch_runner(module)} | {name: ort_runner(session) for name, session in sessions.items()}
    outputs: dict[str, list[np.ndarray]] = {name: [] for name in runners}
    failures: dict[str, str] = {}
    for path in paths:
        with Image.open(path) as source:
            batch = transform(source.convert("RGB")).unsqueeze(0)
        for name, infer in runners.items():
            if name in failures:
                continue
            try:
                outputs[name].append(np.asarray(infer(batch), dtype=np.float64)[0])
            except Exception as error:
                failures[name] = f"{type(error).__name__}: {error}"
    return {name: np.stack(values) for name, values in outputs.items() if name not in failures}, failures


def evaluate_backends(
    arch: str, outputs: dict[str, np.ndarray], frame: pd.DataFrame, replicates: int
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    sample_ids = frame["sample_id"].astype(str).tolist()
    targets = np.asarray([THREE_STAGE.index(label) for label in frame["freshness_label"]], dtype=int)
    frozen = frozen_test_predictions(TASK, arch, REFERENCE_SEED).set_index("sample_id").loc[sample_ids]
    if not np.array_equal(frozen["target_index"].to_numpy(dtype=int), targets):
        raise ValueError(f"{arch}: frozen predictions disagree with manifest labels")
    frozen_probabilities = frozen[probability_columns(TASK)].to_numpy(dtype=float)
    reference = outputs["pytorch_fp32"]
    reference_predictions = reference.argmax(axis=1)
    groups = capture_groups(sample_ids)
    class_count = len(THREE_STAGE)
    rows: list[dict[str, object]] = []
    predictions = []
    for backend, probabilities in [("frozen_reference", frozen_probabilities), *outputs.items()]:
        predicted = probabilities.argmax(axis=1)
        scores = metrics(targets, probabilities, TASK)
        row: dict[str, object] = {
            "architecture": arch,
            "backend": backend,
            "n": len(targets),
            "capture_groups": len(set(groups)),
            **{key: scores[key] for key in ("macro_f1", "accuracy", "balanced_accuracy", "weighted_f1", "ece", "nll", "brier")},
            "agreement_with_pytorch_fp32": float(np.mean(predicted == reference_predictions)),
            "max_abs_probability_difference": float(np.abs(probabilities - reference).max()),
            "mean_abs_probability_difference": float(np.abs(probabilities - reference).mean()),
        }
        if backend != "pytorch_fp32":
            row["macro_f1_delta_vs_pytorch_fp32"] = scores["macro_f1"] - macro_f1(targets, reference_predictions, class_count)
        if backend in BACKENDS[1:]:

            def statistic(indices: np.ndarray, predicted: np.ndarray = predicted) -> float:
                return macro_f1(targets[indices], predicted[indices], class_count) - macro_f1(
                    targets[indices], reference_predictions[indices], class_count
                )

            row["delta_ci95_low"], row["delta_ci95_high"] = percentile_interval(
                cluster_bootstrap(groups, statistic, replicates=replicates)
            )
        rows.append(row)
        if backend in BACKENDS:
            records = pd.DataFrame(probabilities, columns=probability_columns(TASK))
            records.insert(0, "predicted_index", predicted)
            records.insert(0, "target_index", targets)
            records.insert(0, "sample_id", sample_ids)
            records.insert(0, "backend", backend)
            records.insert(0, "architecture", arch)
            predictions.append(records)
    return rows, pd.concat(predictions, ignore_index=True)


def profile_cpu(arch: str, settings: Settings, workload: Workload, run_started: str) -> dict[str, object]:
    timer = time.perf_counter()
    module, model, payload = load_deployed(arch)
    image_size = IMAGE_SIZE
    transform = build_transform(image_size, training=False)
    checkpoint = checkpoint_path(arch)
    flops = count_flops(module, image_size)
    profile: dict[str, object] = {
        "architecture": arch,
        "display_name": DISPLAY_NAMES[arch],
        "run_started": run_started,
        "checkpoint": relative(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_mib": checkpoint.stat().st_size / 2**20,
        "temperature": float(payload["temperature"]),
        "image_size": image_size,
        "parameters_total": sum(parameter.numel() for parameter in model.parameters()),
        "parameters_deployed": sum(parameter.numel() for parameter in module.parameters()),
        "flops": flops,
        "macs": flops / 2,
        "gflops": flops / 1e9,
    }
    fp32_path = settings.model_dir / f"{arch}_seed-{REFERENCE_SEED}_fp32.onnx"
    int8_path = settings.model_dir / f"{arch}_seed-{REFERENCE_SEED}_int8.onnx"
    exported = quantized = False
    started = time.perf_counter()
    fp32_path.unlink(missing_ok=True)
    try:
        profile["exporter"] = export_onnx(module, fp32_path, image_size)
        profile |= {
            "export_status": "ok",
            "onnx_fp32_path": relative(fp32_path),
            "onnx_fp32_bytes": fp32_path.stat().st_size,
            "onnx_fp32_mib": fp32_path.stat().st_size / 2**20,
            "onnx_fp32_sha256": file_sha256(fp32_path),
        }
        exported = True
    except Exception as error:
        profile["export_status"] = f"failed: {type(error).__name__}: {error}"
        print(f"WARNING {arch}: ONNX export failed: {error}", flush=True)
    profile["export_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    if exported:
        int8_path.unlink(missing_ok=True)
        reader = TrainImageReader(image_paths(workload.calibration, settings.image_root), transform)
        try:
            profile |= quantize_int8(fp32_path, int8_path, reader, settings.work_threads)
            profile |= {
                "quantization_status": "ok",
                "calibration_images": len(reader),
                "onnx_int8_path": relative(int8_path),
                "onnx_int8_bytes": int8_path.stat().st_size,
                "onnx_int8_mib": int8_path.stat().st_size / 2**20,
                "onnx_int8_sha256": file_sha256(int8_path),
                "int8_size_ratio": int8_path.stat().st_size / fp32_path.stat().st_size,
            }
            quantized = True
        except Exception as error:
            profile["quantization_status"] = f"failed: {type(error).__name__}: {error}"
            int8_path.unlink(missing_ok=True)
            print(f"WARNING {arch}: INT8 quantisation failed: {error}", flush=True)
    else:
        profile["quantization_status"] = "not attempted: ONNX export failed"
    profile["quantization_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    torch.set_num_threads(settings.work_threads)
    sessions: dict[str, ort.InferenceSession] = {}
    session_failures: dict[str, str] = {}
    for name, path, available in (("ort_fp32", fp32_path, exported), ("ort_int8", int8_path, quantized)):
        if available:
            try:
                sessions[name] = ort_session(path, settings.work_threads)
            except Exception as error:
                session_failures[name] = f"session: {type(error).__name__}: {error}"
    outputs, run_failures = accuracy_pass(module, sessions, image_paths(workload.accuracy, settings.image_root), transform)
    sessions.clear()
    failures = session_failures | run_failures
    for name, message in failures.items():
        profile[f"{name}_status"] = f"failed: {message}"
        print(f"WARNING {arch}: {name} failed: {message}", flush=True)
    accuracy_rows, predictions = evaluate_backends(arch, outputs, workload.accuracy, settings.bootstrap)
    profile["accuracy_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    paths = image_paths(workload.latency, settings.image_root)
    sample_ids = workload.latency["sample_id"].astype(str).tolist()
    samples = []

    def record(runtime: str, precision: str, threads: int, timings: list[dict[str, float]]) -> None:
        for index, (sample_id, timing) in enumerate(zip(sample_ids, timings)):
            samples.append(
                {"architecture": arch, "runtime": runtime, "precision": precision, "threads": threads, "image_index": index, "sample_id": sample_id}
                | timing
            )

    for threads in settings.threads:
        torch.set_num_threads(threads)
        record("pytorch", "fp32", threads, time_pipeline(paths, transform, torch_runner(module), settings.warmup))
    torch.set_num_threads(settings.work_threads)
    for precision, path in (("fp32", fp32_path), ("int8", int8_path)):
        if f"ort_{precision}" in outputs:
            for threads in settings.ort_threads:
                session = ort_session(path, threads)
                record("onnxruntime", precision, threads, time_pipeline(paths, transform, ort_runner(session), settings.warmup))
                del session
    profile["latency_seconds"] = time.perf_counter() - started
    profile["total_seconds"] = time.perf_counter() - timer
    return {
        "profile": profile,
        "accuracy": accuracy_rows,
        "predictions": predictions,
        "samples": pd.DataFrame.from_records(samples),
    }


def measure_accelerator(
    module: nn.Module,
    inputs: torch.Tensor,
    *,
    fp16: bool,
    iterations: int,
    min_seconds: float,
    warmup: int,
    meter: EnergyMeter,
    idle_watts: float,
) -> dict[str, float]:
    """Per-iteration latency (CUDA events on GPU), throughput, and NVML energy over one timed window."""
    cuda = inputs.device.type == "cuda"
    if fp16 and not cuda:
        raise ValueError("FP16 autocast is profiled on CUDA only")
    synchronize = (lambda: torch.cuda.synchronize(inputs.device)) if cuda else (lambda: None)
    autocast = torch.autocast(device_type="cuda", dtype=torch.float16) if fp16 else contextlib.nullcontext()
    latencies: list[float] = []
    with torch.inference_mode(), autocast:
        for _ in range(warmup):
            module(inputs)
        synchronize()
        begin = torch.cuda.Event(enable_timing=True) if cuda else None
        end = torch.cuda.Event(enable_timing=True) if cuda else None
        with meter.measure() as reading:
            started = time.perf_counter()
            while len(latencies) < iterations or time.perf_counter() - started < min_seconds:
                if cuda:
                    begin.record()
                    module(inputs)
                    end.record()
                    end.synchronize()
                    latencies.append(begin.elapsed_time(end))
                else:
                    tick = time.perf_counter()
                    module(inputs)
                    latencies.append((time.perf_counter() - tick) * 1e3)
            synchronize()
    values = np.asarray(latencies, dtype=float)
    images = len(values) * len(inputs)
    return {
        "iterations": len(values),
        "images": images,
        "seconds": reading["seconds"],
        "latency_mean_ms": float(values.mean()),
        "latency_median_ms": float(np.median(values)),
        "latency_p95_ms": float(np.percentile(values, 95)),
        "throughput_images_per_second": images / reading["seconds"],
        "mean_power_w": reading["watts"],
        "gross_energy_mj_per_image": reading["joules"] * 1e3 / images,
        "net_energy_mj_per_image": (reading["joules"] - idle_watts * reading["seconds"]) * 1e3 / images,
    }


def gpu_section(settings: Settings, workload: Workload, run_started: str) -> tuple[pd.DataFrame, dict[str, object]]:
    device = torch.device("cuda", 0)
    properties = torch.cuda.get_device_properties(device)
    uuid = getattr(properties, "uuid", None)
    meter = EnergyMeter.open(str(uuid) if uuid is not None else None)
    meta: dict[str, object] = {
        "torch_device_name": torch.cuda.get_device_name(device),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "latency_timer": "CUDA events, synchronised every iteration",
        "batch_sizes": [1, GPU_BATCH],
        "min_window_seconds": settings.gpu_min_seconds,
        "warmup": settings.gpu_warmup,
    } | meter.describe()
    transform = build_transform(IMAGE_SIZE, training=False)
    paths = image_paths(workload.latency, settings.image_root)
    tensors = []
    for index in range(GPU_BATCH):
        with Image.open(paths[index % len(paths)]) as source:
            tensors.append(transform(source.convert("RGB")))
    batch = torch.stack(tensors).to(device)
    torch.cuda.synchronize(device)
    time.sleep(settings.settle_seconds)
    idle = meter.idle(settings.idle_seconds)
    meta["idle_before"] = idle
    rows = []
    for arch in settings.models:
        module = None
        try:
            module, _, _ = load_deployed(arch, device)
            for fp16 in (False, True):
                for size in (1, GPU_BATCH):
                    iterations = settings.iterations if size == 1 else max(10, settings.iterations // 10)
                    result = measure_accelerator(
                        module,
                        batch[:size],
                        fp16=fp16,
                        iterations=iterations,
                        min_seconds=settings.gpu_min_seconds,
                        warmup=settings.gpu_warmup,
                        meter=meter,
                        idle_watts=idle["watts"],
                    )
                    rows.append(
                        {"architecture": arch, "run_started": run_started, "precision": "fp16_autocast" if fp16 else "fp32", "batch_size": size, "status": "ok"}
                        | result
                        | {"idle_power_w": idle["watts"], "energy_method": meter.method}
                    )
                    print(f"  GPU {arch} {'fp16' if fp16 else 'fp32'} b{size}: {result['latency_mean_ms']:.2f} ms, {result['net_energy_mj_per_image']:.1f} mJ/img", flush=True)
        except Exception as error:
            rows.append({"architecture": arch, "run_started": run_started, "status": f"failed: {type(error).__name__}: {error}"})
            print(f"WARNING {arch}: GPU profiling failed: {error}", flush=True)
        finally:
            del module
            torch.cuda.empty_cache()
    meta["idle_after"] = meter.idle(settings.idle_seconds)
    with contextlib.suppress(Exception):
        meter.nvml.nvmlShutdown()
    return pd.DataFrame.from_records(rows), meta


def windows_power_scheme() -> str | None:
    if sys.platform != "win32":
        return None
    with contextlib.suppress(Exception):
        text = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=10).stdout.strip()
        return text[text.find("(") + 1 : text.rfind(")")].strip() if "(" in text else text or None
    return None


def describe_environment(settings: Settings, cuda_available: bool) -> dict[str, object]:
    import PIL
    import torchvision

    info: dict[str, object] = {
        "captured_at_utc": utc_now(),
        "cpu": {
            "name": cpu_name(),
            "physical_cores": physical_cores(),
            "logical_processors": os.cpu_count(),
            "torch_cpu_capability": torch.backends.cpu.get_cpu_capability(),
        },
        "operating_system": {"platform": platform.platform(), "version": platform.version(), "machine": platform.machine()},
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pillow": PIL.__version__,
            "mkldnn_available": torch.backends.mkldnn.is_available(),
            "onnxruntime_providers": ort.get_available_providers(),
        },
        "torch_parallel": {
            "intra_op_threads_at_start": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
        },
        "windows_power_scheme": windows_power_scheme(),
        "cuda_visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_available": cuda_available,
    }
    with contextlib.suppress(ImportError):
        import psutil

        battery = psutil.sensors_battery()
        info["memory_bytes"] = int(psutil.virtual_memory().total)
        info["load_at_start"] = {
            "cpu_percent_1s": psutil.cpu_percent(interval=1.0),
            "available_memory_bytes": int(psutil.virtual_memory().available),
        }
        info["power_source"] = None if battery is None else {"ac_power": battery.power_plugged, "battery_percent": battery.percent}
    with contextlib.suppress(Exception):
        meter = EnergyMeter.open()
        info["gpu_nvml"] = meter.describe()
        meter.nvml.nvmlShutdown()
    return info


def merge_rows(path: Path, frame: pd.DataFrame, models: Sequence[str]) -> None:
    if path.is_file():
        previous = pd.read_csv(path)
        frame = pd.concat([previous[~previous["architecture"].isin(models)], frame], ignore_index=True)
    order = {name: index for index, name in enumerate(MODEL_ORDER)}
    frame = frame.sort_values("architecture", key=lambda column: column.map(order), kind="mergesort")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame(columns=["architecture"])


def pick(frame: pd.DataFrame, column: str, **conditions: object) -> float:
    mask = np.ones(len(frame), dtype=bool)
    for key, value in conditions.items():
        if key not in frame:
            return float("nan")
        mask &= (frame[key] == value).to_numpy()
    values = frame.loc[mask, column] if column in frame else pd.Series(dtype=float)
    return float(values.iloc[0]) if len(values) else float("nan")


def build_summary(output_dir: Path) -> pd.DataFrame:
    profile = read_optional(output_dir / "model_profile.csv")
    latency = read_optional(output_dir / "cpu_latency.csv")
    accuracy = read_optional(output_dir / "onnx_accuracy.csv")
    gpu = read_optional(output_dir / "gpu_profile.csv")
    rows = []
    for arch in MODEL_ORDER:
        own = {name: frame[frame["architecture"] == arch] for name, frame in (("profile", profile), ("latency", latency), ("accuracy", accuracy), ("gpu", gpu))}
        if all(frame.empty for frame in own.values()):
            continue
        row: dict[str, object] = {"architecture": arch, "display_name": DISPLAY_NAMES[arch]}
        for column in ("gflops", "macs", "parameters_total", "parameters_deployed", "checkpoint_mib", "onnx_fp32_mib", "onnx_int8_mib", "int8_size_ratio", "quantization_status"):
            if column in own["profile"] and len(own["profile"]):
                row[column] = own["profile"][column].iloc[0]
        for (runtime, precision, threads), group in own["latency"].groupby(["runtime", "precision", "threads"], sort=False) if len(own["latency"]) else []:
            prefix = f"{'torch' if runtime == 'pytorch' else 'ort'}_{precision}_{int(threads)}t"
            for stage in ("preprocess", "forward", "total"):
                row[f"{prefix}_{stage}_mean_ms"] = pick(group, "mean_ms", stage=stage)
            row[f"{prefix}_total_median_ms"] = pick(group, "median_ms", stage="total")
            row[f"{prefix}_total_p95_ms"] = pick(group, "p95_ms", stage="total")
        for backend in ("frozen_reference", *BACKENDS):
            row[f"{backend}_macro_f1"] = pick(own["accuracy"], "macro_f1", backend=backend)
        for backend in ("ort_fp32", "ort_int8"):
            row[f"{backend}_agreement"] = pick(own["accuracy"], "agreement_with_pytorch_fp32", backend=backend)
            for column in ("macro_f1_delta_vs_pytorch_fp32", "delta_ci95_low", "delta_ci95_high"):
                row[f"{backend}_{column}"] = pick(own["accuracy"], column, backend=backend)
        row["pytorch_agreement_with_frozen"] = pick(own["accuracy"], "agreement_with_pytorch_fp32", backend="frozen_reference")
        for (precision, size), group in own["gpu"].groupby(["precision", "batch_size"], sort=False) if "batch_size" in own["gpu"] and len(own["gpu"]) else []:
            prefix = f"gpu_{precision}_b{int(size)}"
            for column in ("latency_mean_ms", "latency_median_ms", "latency_p95_ms", "throughput_images_per_second", "mean_power_w", "gross_energy_mj_per_image", "net_energy_mj_per_image"):
                row[f"{prefix}_{column}"] = float(group[column].iloc[0])
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def number(value: object, digits: int, *, signed: bool = False) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(value):
        return "--"
    return f"${value:+.{digits}f}$" if signed else f"{value:.{digits}f}"


def short_hardware_name(name: object) -> str:
    text = str(name or "")
    for token in ("(R)", "(TM)", "12th Gen ", "13th Gen ", "14th Gen ", "NVIDIA ", "GeForce ", " CPU", "@"):
        text = text.replace(token, "")
    return " ".join(text.split())


def write_table(settings: Settings, summary: pd.DataFrame, environment: dict[str, object]) -> bool:
    if summary.empty:
        return False
    low, mid = (1, 4)
    rows = []
    for _, row in summary.iterrows():
        int8 = row.get("onnx_int8_mib") if str(row.get("quantization_status", "")).startswith("ok") else float("nan")
        rows.append(
            [
                row["display_name"],
                number(row.get("gflops"), 2),
                number(float(row.get("parameters_total", float("nan"))) / 1e6, 2),
                number(row.get("onnx_fp32_mib"), 1),
                number(int8, 1),
                f"{number(row.get(f'torch_fp32_{low}t_total_mean_ms'), 1)} / {number(row.get(f'torch_fp32_{mid}t_total_mean_ms'), 1)}",
                f"{number(row.get(f'ort_fp32_{mid}t_total_mean_ms'), 1)} / {number(row.get(f'ort_int8_{mid}t_total_mean_ms'), 1)}",
                number(row.get("ort_int8_macro_f1_delta_vs_pytorch_fp32"), 3, signed=True),
                number(row.get("gpu_fp16_autocast_b1_latency_mean_ms"), 2),
                number(row.get("gpu_fp16_autocast_b1_net_energy_mj_per_image"), 0),
            ]
        )
    cpu = short_hardware_name(environment.get("cpu", {}).get("name") if isinstance(environment.get("cpu"), dict) else "")
    gpu_info = environment.get("gpu_nvml", {})
    gpu = short_hardware_name(gpu_info.get("name") if isinstance(gpu_info, dict) else "")
    accuracy_scope = "all 831" if settings.accuracy_images <= 0 else f"{settings.accuracy_images}"
    caption = (
        f"Deployment profile of the calibrated reference-seed ({REFERENCE_SEED}) three-stage checkpoints "
        f"(one checkpoint per model; CPU: {cpu}; GPU: {gpu}). "
        f"CPU columns are mean end-to-end latency over {settings.images} grouped-test JPEGs at batch 1."
    )
    notes = (
        r"GFLOPs: $2\times$ multiply--accumulates of convolution and linear layers for one $224\times224$ image "
        r"(torch.utils.flop\_counter). Params include the auxiliary produce head. MiB: ONNX opset-17 FP32 and static INT8 files "
        f"(QDQ, per-channel weights, MinMax calibration on {settings.calibration_images} training images; classifier head, "
        r"temperature, and softmax kept in FP32). CPU e2e: JPEG decode, evaluation transform, forward pass, and "
        r"temperature-scaled softmax in PyTorch FP32 with 1 / 4 intra-op threads; ORT: the same pipeline in ONNX Runtime "
        r"with 4 threads. INT8 $\Delta$F1: ORT INT8 minus PyTorch FP32 macro-F1 on " + accuracy_scope + r" grouped-test "
        r"images (95\% capture-group bootstrap intervals in the CSV). GPU: batch-1 FP16-autocast forward latency (CUDA events) "
        r"and NVML energy per image above the idle baseline. --: not measured or quantisation failed."
    )
    latex_table(
        settings.table_path,
        caption=caption,
        label=f"tab:{TABLE_NAME.replace('_', '-')}",
        headers=[
            "Model",
            "GFLOPs",
            r"\shortstack{Params\\(M)}",
            r"\shortstack{FP32\\MiB}",
            r"\shortstack{INT8\\MiB}",
            r"\shortstack{CPU e2e ms\\1 / 4 thr}",
            r"\shortstack{ORT ms, 4 thr\\FP32 / INT8}",
            r"\shortstack{INT8\\$\Delta$F1}",
            r"\shortstack{GPU FP16\\ms}",
            r"\shortstack{GPU\\mJ/img}",
        ],
        rows=rows,
        wide=True,
        notes=notes,
    )
    return True


def part_status(settings: Settings) -> dict[str, str]:
    directory = settings.output_dir
    profile = read_optional(directory / "model_profile.csv")
    latency = read_optional(directory / "cpu_latency.csv")
    accuracy = read_optional(directory / "onnx_accuracy.csv")
    gpu = read_optional(directory / "gpu_profile.csv")

    def coverage(frame: pd.DataFrame, label: str, column: str | None = None) -> str:
        present = set(frame["architecture"].astype(str))
        missing = [arch for arch in MODEL_ORDER if arch not in present]
        if missing:
            return f"pending: {label} not yet measured for {', '.join(missing)}"
        if column and column in frame:
            failed = sorted({str(arch) for arch, value in zip(frame["architecture"], frame[column]) if not str(value).startswith("ok")})
            if failed:
                return f"done; failed for {', '.join(failed)} (see {column} in model_profile.csv)"
        return "done"

    def runtime(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        return frame[frame["runtime"] == name] if "runtime" in frame else frame.iloc[0:0]

    parts = {
        "environment": "done" if (directory / "environment.json").is_file() else "pending: no run yet",
        "complexity_and_size": coverage(profile, "parameters, FLOPs, and sizes"),
        "cpu_latency_pytorch": coverage(runtime(latency, "pytorch"), "PyTorch end-to-end CPU latency"),
        "onnx_export": coverage(profile, "ONNX export", "export_status"),
        "int8_quantization": coverage(profile, "INT8 quantisation", "quantization_status"),
        "onnx_accuracy": coverage(accuracy, "ONNX accuracy"),
        "cpu_latency_onnxruntime": coverage(runtime(latency, "onnxruntime"), "ONNX Runtime CPU latency"),
        "gpu_latency_energy": coverage(gpu, "GPU latency and energy (needs CUDA and no --skip-gpu)", "status"),
        "latex_table": "done" if settings.table_path.is_file() else "pending: no profiled model yet",
    }
    if parts["latex_table"] == "done" and any(not parts[name].startswith("done") for name in PARTS[:-1]):
        parts["latex_table"] = "pending: written with missing cells; rerun after the pending parts"
    if settings.quick:
        parts = {name: f"quick smoke test only ({value})" for name, value in parts.items()}
    return parts


def resolve_settings(args: argparse.Namespace) -> Settings:
    defaults = QUICK if args.quick else FULL
    gpu = QUICK_GPU if args.quick else FULL_GPU
    output_dir = (args.output_dir or (QUICK_OUTPUT if args.quick else OUTPUT)).resolve()
    if args.quick and output_dir == OUTPUT.resolve():
        raise SystemExit("--quick writes smoke-test numbers; choose a scratch --output-dir, not the final location")
    threads = tuple(args.threads or ((1, 4) if args.quick else sorted({1, 4, physical_cores()})))
    if min(threads) < 1:
        raise SystemExit("--threads must be positive")
    return Settings(
        quick=args.quick,
        models=tuple(dict.fromkeys(args.models)),
        threads=threads,
        ort_threads=tuple(args.ort_threads or (1, 4)),
        images=args.images or defaults["images"],
        warmup=defaults["warmup"] if args.warmup is None else args.warmup,
        iterations=args.iterations or defaults["iterations"],
        calibration_images=args.calibration_images or defaults["calibration_images"],
        accuracy_images=defaults["accuracy_images"] if args.accuracy_images is None else args.accuracy_images,
        bootstrap=args.bootstrap or defaults["bootstrap"],
        skip_gpu=args.skip_gpu,
        output_dir=output_dir,
        model_dir=(args.model_dir or (output_dir / "onnx" if args.quick else MODEL_OUTPUT)).resolve(),
        table_path=output_dir / f"{TABLE_NAME}.tex" if args.quick else PAPER_TABLES / f"{TABLE_NAME}.tex",
        image_root=args.image_root,
        **gpu,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="Smoke test with a few images and iterations into a scratch directory")
    parser.add_argument("--skip-gpu", action="store_true", help="Skip the CUDA latency and NVML energy section")
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=list(MODEL_ORDER))
    parser.add_argument("--threads", nargs="+", type=int, help="PyTorch intra-op threads (default: 1, 4, all physical cores)")
    parser.add_argument("--ort-threads", nargs="+", type=int, help="ONNX Runtime intra-op threads (default: 1, 4)")
    parser.add_argument("--images", type=int, help="Grouped-test images for end-to-end CPU latency (default 100)")
    parser.add_argument("--warmup", type=int, help="Untimed warm-up images per CPU configuration (default 10)")
    parser.add_argument("--iterations", type=int, help="Timed GPU batch-1 iterations; batch 32 uses a tenth (default 500)")
    parser.add_argument("--calibration-images", type=int, help="Training images for INT8 calibration (default 200)")
    parser.add_argument("--accuracy-images", type=int, help="Grouped-test images for ONNX accuracy; 0 means all 831")
    parser.add_argument("--bootstrap", type=int, help="Capture-group bootstrap replicates for INT8 macro-F1 changes")
    parser.add_argument("--output-dir", type=Path, help=f"Default {relative(OUTPUT)} (quick: system temp directory)")
    parser.add_argument("--model-dir", type=Path, help=f"ONNX files (default {relative(MODEL_OUTPUT)})")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--status-only", action="store_true", help="Only rewrite status.json from existing outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = resolve_settings(args)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    if args.status_only:
        write_json(part_status(settings), settings.output_dir / "status.json")
        print(f"status -> {settings.output_dir / 'status.json'}")
        return 0
    run_started = utc_now()
    clock = time.perf_counter()
    cuda_available = (not settings.skip_gpu) and torch.cuda.is_available()
    environment = describe_environment(settings, cuda_available)
    write_json(environment, settings.output_dir / "environment.json")
    torch.set_num_threads(settings.work_threads)
    workload = select_workload(settings)
    for name, frame in (("latency", workload.latency), ("calibration", workload.calibration), ("accuracy", workload.accuracy)):
        frame[["sample_id", "split", "produce_type", "freshness_label", "capture_group_id", "source_path"]].to_csv(
            settings.output_dir / f"{name}_images.csv", index=False
        )
    for path in image_paths(workload.latency, settings.image_root):
        path.read_bytes()
    print(
        f"{'QUICK ' if settings.quick else ''}deployment profile: models={list(settings.models)}, torch threads={list(settings.threads)}, "
        f"ORT threads={list(settings.ort_threads)}, latency images={len(workload.latency)}, calibration images={len(workload.calibration)}, "
        f"accuracy images={len(workload.accuracy)} -> {settings.output_dir}",
        flush=True,
    )
    reproduction = {}
    for arch in settings.models:
        print(f"[{arch}] CPU profile", flush=True)
        result = profile_cpu(arch, settings, workload, run_started)
        profile = result["profile"]
        merge_rows(settings.output_dir / "model_profile.csv", pd.DataFrame([profile]), [arch])
        merge_rows(settings.output_dir / "onnx_accuracy.csv", pd.DataFrame(result["accuracy"]), [arch])
        merge_rows(settings.output_dir / "deployment_predictions.csv", result["predictions"], [arch])
        merge_rows(settings.output_dir / "cpu_latency_samples.csv", result["samples"], [arch])
        merge_rows(settings.output_dir / "cpu_latency.csv", summarize_latency(result["samples"]), [arch])
        scores = {row["backend"]: row for row in result["accuracy"]}
        reproduction[arch] = {
            "pytorch_cpu_macro_f1": scores["pytorch_fp32"]["macro_f1"],
            "frozen_reference_macro_f1": scores["frozen_reference"]["macro_f1"],
            "prediction_agreement": scores["frozen_reference"]["agreement_with_pytorch_fp32"],
            "max_abs_probability_difference": scores["frozen_reference"]["max_abs_probability_difference"],
        }
        print(
            f"  {profile['gflops']:.3f} GFLOPs, {profile['parameters_total'] / 1e6:.2f} M params, export {profile['export_status']}, "
            f"INT8 {str(profile['quantization_status'])[:60]}; macro-F1 "
            + ", ".join(f"{name}={row['macro_f1']:.4f}" for name, row in scores.items())
            + f" ({profile['total_seconds']:.0f} s)",
            flush=True,
        )
        if abs(reproduction[arch]["pytorch_cpu_macro_f1"] - reproduction[arch]["frozen_reference_macro_f1"]) > 1e-9:
            print(f"WARNING {arch}: PyTorch CPU macro-F1 differs from the frozen seed-{REFERENCE_SEED} predictions", flush=True)
    gpu_meta: dict[str, object] = {}
    if cuda_available:
        print("GPU section", flush=True)
        gpu_rows, gpu_meta = gpu_section(settings, workload, run_started)
        merge_rows(settings.output_dir / "gpu_profile.csv", gpu_rows, list(settings.models))
    else:
        reason = "--skip-gpu" if settings.skip_gpu else "CUDA unavailable"
        print(f"WARNING: GPU latency/energy skipped ({reason}); status stays pending", flush=True)
        gpu_meta = {"skipped": reason}
    summary = build_summary(settings.output_dir)
    summary.to_csv(settings.output_dir / "deployment_summary.csv", index=False)
    write_table(settings, summary, environment)
    write_json(
        {
            "topic": TOPIC,
            "mode": "quick" if settings.quick else "full",
            "run_started_utc": run_started,
            "run_finished_utc": utc_now(),
            "elapsed_seconds": time.perf_counter() - clock,
            "settings": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(settings).items()},
            "frozen_reproduction": reproduction,
            "gpu": gpu_meta,
            "definitions": {
                "flops": "torch.utils.flop_counter.FlopCounterMode on one zero 224x224 image through the deployed graph (2 x MACs of conv/linear)",
                "cpu_end_to_end": "PIL open+decode, build_transform(224, training=False), batch-1 forward, temperature-scaled softmax; warm OS file cache",
                "onnx": f"torch.onnx.export opset {OPSET}, static batch 1, output = temperature-scaled softmax",
                "int8": "onnxruntime quantize_static after quant_pre_process: QDQ, S8 activations and per-channel S8 weights, MinMax calibration",
                "int8_delta": "ORT INT8 macro-F1 minus PyTorch CPU FP32 macro-F1; percentile interval from paired capture-group bootstrap",
                "gpu_energy": "NVML total energy counter (else ~10 ms power sampling) over the timed window minus idle power x window",
            },
        },
        settings.output_dir / "deployment_run.json",
    )
    status = part_status(settings)
    write_json(status, settings.output_dir / "status.json")
    for name, value in status.items():
        print(f"  {name}: {value}")
    print(f"finished in {time.perf_counter() - clock:.0f} s -> {settings.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
