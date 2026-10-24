from __future__ import annotations

import gc

import numpy as np
import pytest
import torch
from PIL import Image
import pandas as pd
from torch.utils.data import DataLoader

from fresh_or_trash.calibration import TemperatureScaler
from fresh_or_trash.corruptions import CORRUPTIONS, CorruptionTransform, apply_corruption
from fresh_or_trash.data import ManifestDataset, build_transform
from fresh_or_trash.evaluation import evaluate_corruptions, evaluate_loader
from fresh_or_trash.gradcam import GradCAM
from fresh_or_trash.metrics import classification_metrics
from fresh_or_trash.models import MODEL_NAMES, build_model
from fresh_or_trash.profiling import profile_model


def test_exact_controlled_model_registry() -> None:
    assert MODEL_NAMES == (
        "mobilenet_v2",
        "resnet50",
        "densenet121",
        "efficientnet_b0",
        "convnext_tiny",
    )


@pytest.mark.parametrize("architecture", MODEL_NAMES)
def test_all_models_forward_without_weights(architecture: str) -> None:
    model = build_model(architecture, num_freshness_classes=3, pretrained=False, num_produce_classes=4).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 64, 64))
    assert output["freshness_logits"].shape == (1, 3)
    assert output["produce_logits"].shape == (1, 4)
    del model
    gc.collect()


def test_metrics_include_calibration_scores_and_nll() -> None:
    targets = np.asarray([0, 1, 2, 0])
    logits = np.asarray([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0], [2.0, 1.0, 0.0]])
    metrics = classification_metrics(targets, logits=logits, class_names=["Fresh", "Semi-Fresh", "Rotten"])
    assert {"ece", "brier", "nll", "macro_f1", "accuracy"} <= set(metrics)
    assert metrics["nll"] >= 0


def test_temperature_scaler_fits_positive_temperature() -> None:
    logits = torch.tensor([[8.0, -2.0], [-2.0, 8.0], [6.0, -1.0], [1.0, 2.0]])
    targets = torch.tensor([0, 1, 1, 0])
    scaler = TemperatureScaler()
    value = scaler.fit(logits, targets, max_iter=10)
    assert 0.0 < value <= 100.0


def test_binary_dataset_excludes_semi_fresh() -> None:
    frame = pd.DataFrame(
        [
            {"sample_id": "a", "source_path": "a.jpg", "produce_type": "banana", "freshness_label": "Fresh"},
            {"sample_id": "b", "source_path": "b.jpg", "produce_type": "banana", "freshness_label": "Semi-Fresh"},
            {"sample_id": "c", "source_path": "c.jpg", "produce_type": "banana", "freshness_label": "Rotten"},
        ]
    )
    dataset = ManifestDataset(frame, ".", task="binary")
    assert len(dataset) == 2
    assert dataset.freshness_classes == ["Fresh", "Rotten"]
    assert set(dataset.frame["freshness_label"]) == {"Fresh", "Rotten"}


def test_corruptions_are_deterministic_and_sample_stable() -> None:
    first = Image.fromarray(np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3))
    second = Image.fromarray(np.flipud(np.asarray(first)).copy())
    for name in CORRUPTIONS:
        one = np.asarray(apply_corruption(first, name, 3, seed=42))
        again = np.asarray(apply_corruption(first, name, 3, seed=42))
        assert np.array_equal(one, again)
    noise_one = np.asarray(apply_corruption(first, "gaussian_noise", 3, seed=42))
    noise_two = np.asarray(apply_corruption(second, "gaussian_noise", 3, seed=42))
    assert not np.array_equal(noise_one, noise_two)


class _ToyFreshnessModel(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        signal = images.mean(dim=(1, 2, 3))
        return {"freshness_logits": torch.stack((signal, -signal, 0.25 * signal), dim=1)}


def _legacy_corruption_metrics(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    image_root: str,
    *,
    corruption_names: list[str],
    severities: list[int],
) -> pd.DataFrame:
    """Reference implementation mirroring the pre-batched evaluator."""

    records = []
    clean_transform = build_transform(32, training=False)
    for corruption in corruption_names:
        for severity in severities:
            dataset = ManifestDataset(
                frame,
                image_root,
                task="three_stage",
                transform=CorruptionTransform(corruption, severity, clean_transform, seed=42),
                image_size=32,
                produce_classes=["banana"],
            )
            metrics, _, _, _ = evaluate_loader(
                model,
                DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0),
                "cpu",
                class_names=["Fresh", "Semi-Fresh", "Rotten"],
            )
            records.append(
                {
                    "corruption": corruption,
                    "severity": severity,
                    **{key: value for key, value in metrics.items() if isinstance(value, (int, float))},
                }
            )
    return pd.DataFrame.from_records(records)


def test_expanded_corruption_sweep_matches_legacy_condition_metrics(tmp_path) -> None:
    labels = ["Fresh", "Semi-Fresh", "Rotten", "Fresh", "Rotten"]
    rows = []
    for index, label in enumerate(labels):
        filename = f"sample-{index}.jpg"
        pixels = np.full((40, 40, 3), index * 37, dtype=np.uint8)
        Image.fromarray(pixels).save(tmp_path / filename)
        rows.append(
            {
                "sample_id": f"sample-{index}",
                "source_path": filename,
                "produce_type": "banana",
                "freshness_label": label,
            }
        )
    frame = pd.DataFrame.from_records(rows)
    model = _ToyFreshnessModel().eval()
    names = ["brightness", "gaussian_noise"]
    severities = [1, 3]
    expected = _legacy_corruption_metrics(
        model, frame, str(tmp_path), corruption_names=names, severities=severities
    )
    actual = evaluate_corruptions(
        model,
        frame,
        tmp_path,
        "cpu",
        task="three_stage",
        class_names=["Fresh", "Semi-Fresh", "Rotten"],
        produce_classes=["banana"],
        image_size=32,
        batch_size=2,
        num_workers=0,
        corruption_names=names,
        severities=severities,
        seed=42,
    )
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    spawned_worker_result = evaluate_corruptions(
        model,
        frame,
        tmp_path,
        "cpu",
        task="three_stage",
        class_names=["Fresh", "Semi-Fresh", "Rotten"],
        produce_classes=["banana"],
        image_size=32,
        batch_size=2,
        num_workers=2,
        corruption_names=names,
        severities=severities,
        seed=42,
    )
    pd.testing.assert_frame_equal(spawned_worker_result, expected, check_exact=True)


def test_gradcam_and_profile_smoke() -> None:
    model = build_model("mobilenet_v2", num_freshness_classes=3, pretrained=False)
    images = torch.randn(1, 3, 64, 64)
    with GradCAM(model) as gradcam:
        heatmap, logits = gradcam(images)
    assert heatmap.shape == (1, 1, 64, 64)
    assert logits.shape == (1, 3)
    profile = profile_model(model, "cpu", image_size=64, warmup=0, iterations=1, amp=False)
    assert profile["parameter_count"] > 0
    assert profile["state_dict_bytes"] > 0
    assert profile["measurement_scope"] == "model_forward_pass_only"
    assert profile["includes_image_decoding"] is False
    assert profile["includes_preprocessing"] is False
