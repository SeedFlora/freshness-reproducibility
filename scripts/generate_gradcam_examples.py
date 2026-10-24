"""Generate deterministic, auditable qualitative Grad-CAM paper examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fresh_or_trash.data import build_transform  # noqa: E402
from fresh_or_trash.gradcam import GradCAM, overlay_heatmap  # noqa: E402
from fresh_or_trash.manifest import FRESHNESS_CLASSES, file_sha256  # noqa: E402
from fresh_or_trash.models import load_checkpoint  # noqa: E402
from fresh_or_trash.utils import resolve_device, save_json, seed_everything  # noqa: E402


ARCHITECTURE = "convnext_tiny"
REFERENCE_SEED = 2026
CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "runs"
    / "grouped"
    / "three_stage"
    / ARCHITECTURE
    / f"seed-{REFERENCE_SEED}"
    / "calibration"
    / "best.calibrated.pt"
)
PREDICTIONS = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "evaluation"
    / "grouped"
    / "three_stage"
    / ARCHITECTURE
    / f"seed-{REFERENCE_SEED}"
    / "calibrated"
    / "test_predictions.csv"
)
TEST_MANIFEST = PROJECT_ROOT / "data" / "splits" / "grouped_test.csv"
SELECTION_RECORD = (
    PROJECT_ROOT
    / "results"
    / "experiments"
    / "selection"
    / "grouped_three_stage_validation_selection.json"
)
IMAGE_ROOT = PROJECT_ROOT.parent / "data" / "raw"
DEFAULT_FIGURE = PROJECT_ROOT / "paper" / "figures" / "gradcam_examples.png"
DEFAULT_SIDECAR = PROJECT_ROOT / "results" / "gradcam_examples.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_and_validate_sources() -> tuple[pd.DataFrame, dict]:
    for source in (CHECKPOINT, PREDICTIONS, TEST_MANIFEST, SELECTION_RECORD):
        if not source.is_file():
            raise FileNotFoundError(source)

    selection = json.loads(SELECTION_RECORD.read_text(encoding="utf-8"))
    ranking = selection.get("ranking", [])
    if not ranking or ranking[0].get("architecture") != ARCHITECTURE:
        raise ValueError("ConvNeXt-Tiny is not first in the frozen validation-only ranking")
    if selection.get("test_metrics_used") is not False:
        raise ValueError("The architecture-selection record does not exclude test metrics")

    predictions = pd.read_csv(PREDICTIONS)
    manifest = pd.read_csv(TEST_MANIFEST)
    if manifest["split"].astype(str).ne("test").any():
        raise ValueError("The grouped test manifest contains a non-test row")
    if set(predictions["sample_id"]) != set(manifest["sample_id"]):
        raise ValueError("Prediction and grouped test sample IDs do not match exactly")
    if predictions["sample_id"].duplicated().any() or manifest["sample_id"].duplicated().any():
        raise ValueError("Expected unique sample IDs in predictions and test manifest")

    joined = predictions.merge(
        manifest[["sample_id", "freshness_label", "produce_type", "source_path"]],
        on="sample_id",
        how="inner",
        validate="one_to_one",
        suffixes=("_prediction", "_manifest"),
    )
    if len(joined) != len(manifest):
        raise ValueError("Prediction/test join changed the grouped test sample count")
    if joined["produce_type_prediction"].astype(str).ne(joined["produce_type_manifest"].astype(str)).any():
        raise ValueError("Prediction and test manifest produce labels differ")
    # The locked manifest is the canonical source for the image path and label.
    joined["produce_type"] = joined["produce_type_manifest"]
    joined["source_path"] = joined["source_path_manifest"]
    return joined, selection


def deterministic_selection(frame: pd.DataFrame) -> list[dict]:
    """Select one median-confidence correct case and top-confidence error per true stage."""
    selected: list[dict] = []
    for kind in ("representative_correct", "highest_confidence_error"):
        for stage in FRESHNESS_CLASSES:
            stage_rows = frame[frame["freshness_label"] == stage].copy()
            if kind == "representative_correct":
                candidates = stage_rows[stage_rows["predicted_label"] == stage].copy()
                if candidates.empty:
                    raise ValueError(f"No correct grouped-test prediction for {stage}")
                reference = float(candidates["confidence"].median())
                candidates["selection_distance"] = (candidates["confidence"] - reference).abs()
                chosen = candidates.sort_values(
                    ["selection_distance", "sample_id"], ascending=[True, True], kind="mergesort"
                ).iloc[0]
                detail = {
                    "criterion": "minimum absolute distance to the within-stage median calibrated confidence among correct predictions",
                    "within_stage_correct_median_confidence": reference,
                    "absolute_distance_to_median": float(chosen["selection_distance"]),
                    "tie_breaker": "sample_id ascending",
                }
            else:
                candidates = stage_rows[stage_rows["predicted_label"] != stage].copy()
                if candidates.empty:
                    raise ValueError(f"No grouped-test misclassification for {stage}")
                chosen = candidates.sort_values(
                    ["confidence", "sample_id"], ascending=[False, True], kind="mergesort"
                ).iloc[0]
                detail = {
                    "criterion": "maximum calibrated predicted-class confidence among misclassifications with this true stage",
                    "tie_breaker": "sample_id ascending",
                }
            selected.append({"kind": kind, "true_stage": stage, "row": chosen, "selection_detail": detail})
    return selected


def probability_record(row: pd.Series) -> dict[str, float]:
    return {
        "Fresh": float(row["probability_fresh"]),
        "Semi-Fresh": float(row["probability_semi_fresh"]),
        "Rotten": float(row["probability_rotten"]),
    }


def main() -> int:
    args = parse_args()
    seed_everything(REFERENCE_SEED, deterministic=True)
    frame, selection_record = load_and_validate_sources()
    selected = deterministic_selection(frame)

    device = resolve_device(args.device)
    model, metadata = load_checkpoint(CHECKPOINT, device)
    if metadata.get("architecture") != ARCHITECTURE or metadata.get("task") != "three_stage":
        raise ValueError("Checkpoint metadata does not describe the frozen ConvNeXt-Tiny three-stage model")
    class_names = list(metadata.get("freshness_classes", []))
    if class_names != list(FRESHNESS_CLASSES):
        raise ValueError(f"Unexpected freshness class order: {class_names}")
    temperature = float(metadata.get("temperature", 0.0))
    calibration = metadata.get("calibration", {})
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Calibrated checkpoint has no valid embedded temperature")
    if calibration.get("fit_split") != "validation":
        raise ValueError("Checkpoint temperature was not documented as validation-fitted")

    image_size = int(metadata.get("image_size", 224))
    display_transform = transforms.Compose(
        [
            transforms.Resize(int(round(image_size / 0.875)), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
        ]
    )
    tensor_transform = build_transform(image_size=image_size, training=False)
    cropped_images: list[Image.Image] = []
    tensors: list[torch.Tensor] = []
    class_indices: list[int] = []
    image_paths: list[Path] = []
    for item in selected:
        row = item["row"]
        image_path = IMAGE_ROOT / str(row["source_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        cropped_images.append(display_transform(image))
        tensors.append(tensor_transform(image))
        class_indices.append(int(row["predicted_index"]))
        image_paths.append(image_path)

    batch = torch.stack(tensors).to(device)
    targets = torch.tensor(class_indices, dtype=torch.long, device=device)
    with GradCAM(model) as gradcam:
        heatmaps, logits = gradcam(batch, targets)
    recalculated = torch.softmax(logits / temperature, dim=1).cpu().numpy()
    heatmaps_array = heatmaps[:, 0].cpu().numpy()

    for index, item in enumerate(selected):
        stored = np.array(list(probability_record(item["row"]).values()))
        # CPU/GPU convolution kernels can differ in the last few decimal places;
        # this tolerance is far below the precision displayed in the figure.
        if not np.allclose(recalculated[index], stored, atol=2e-5, rtol=2e-5):
            raise ValueError(f"Recomputed calibrated probabilities differ for {item['row']['sample_id']}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titleweight": "bold",
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(10.8, 6.7), dpi=180)
    figure.patch.set_facecolor("white")
    for index, (axis, item, cropped, heatmap) in enumerate(
        zip(axes.flat, selected, cropped_images, heatmaps_array, strict=True)
    ):
        row = item["row"]
        overlay = overlay_heatmap(cropped, heatmap, alpha=0.42)
        axis.imshow(overlay)
        axis.set_title(
            f"{str(row['produce_type']).replace('_', ' ').title()}\n"
            f"True: {row['freshness_label']}  →  Pred: {row['predicted_label']} ({float(row['confidence']):.1%})",
            fontsize=9.2,
            pad=7,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#D1D5DB")
            spine.set_linewidth(0.8)

    figure.suptitle(
        "ConvNeXt-Tiny (reference seed 2026): qualitative Grad-CAM examples",
        fontsize=13,
        fontweight="bold",
        y=0.982,
    )
    figure.text(
        0.018,
        0.685,
        "Representative correct\n(stage-median confidence)",
        rotation=90,
        va="center",
        ha="center",
        fontsize=9,
        color="#374151",
    )
    figure.text(
        0.018,
        0.292,
        "Highest-confidence error\n(per true stage)",
        rotation=90,
        va="center",
        ha="center",
        fontsize=9,
        color="#374151",
    )
    figure.text(
        0.5,
        0.018,
        "Heatmaps are qualitative and normalized within each image; color intensity is not a calibrated importance score.",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#4B5563",
    )
    figure.subplots_adjust(left=0.075, right=0.99, top=0.90, bottom=0.075, wspace=0.11, hspace=0.31)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.figure, dpi=300, facecolor="white")
    plt.close(figure)

    examples: list[dict] = []
    for panel_index, (item, image_path) in enumerate(zip(selected, image_paths, strict=True), start=1):
        row = item["row"]
        examples.append(
            {
                "panel": panel_index,
                "row_kind": item["kind"],
                "sample_id": str(row["sample_id"]),
                "produce_type": str(row["produce_type"]),
                "true_label": str(row["freshness_label"]),
                "predicted_label": str(row["predicted_label"]),
                "predicted_index": int(row["predicted_index"]),
                "correct": bool(row["freshness_label"] == row["predicted_label"]),
                "calibrated_confidence": float(row["confidence"]),
                "calibrated_probabilities": probability_record(row),
                "source_path": str(row["source_path"]).replace("\\", "/"),
                "image_sha256": file_sha256(image_path),
                "gradcam_target": "predicted freshness logit",
                "selection_detail": item["selection_detail"],
            }
        )

    sidecar = {
        "artifact": str(args.figure.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "artifact_sha256": file_sha256(args.figure),
        "architecture": ARCHITECTURE,
        "reference_seed": REFERENCE_SEED,
        "checkpoint": str(CHECKPOINT.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "checkpoint_sha256": file_sha256(CHECKPOINT),
        "calibration": {
            "method": calibration.get("method", "temperature_scaling"),
            "fit_split": calibration["fit_split"],
            "temperature": temperature,
        },
        "partition": "grouped clean test",
        "test_sample_count": int(len(frame)),
        "prediction_source": str(PREDICTIONS.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "prediction_source_sha256": file_sha256(PREDICTIONS),
        "test_manifest": str(TEST_MANIFEST.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "test_manifest_sha256": file_sha256(TEST_MANIFEST),
        "architecture_selection": {
            "source": str(SELECTION_RECORD.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "scope": selection_record["selection_scope"],
            "metric": selection_record["selection_metric"],
            "validation_rank": int(selection_record["ranking"][0]["validation_rank"]),
            "test_metrics_used": bool(selection_record["test_metrics_used"]),
        },
        "selection_rule": {
            "version": 1,
            "stage_order": list(FRESHNESS_CLASSES),
            "representative_correct": "For each true stage, select the correct prediction with calibrated confidence closest to the within-stage median among correct predictions.",
            "misclassification": "For each true stage, select the misclassification with the highest calibrated predicted-class confidence.",
            "tie_breaker": "sample_id ascending",
            "image_content_used_for_selection": False,
        },
        "interpretation": "Grad-CAM overlays are qualitative, target the predicted freshness logit, and are independently normalized within each image; they are not calibrated importance scores or causal explanations.",
        "examples": examples,
    }
    save_json(sidecar, args.sidecar)
    print(f"Wrote {args.figure}")
    print(f"Wrote {args.sidecar}")
    for example in examples:
        print(
            f"panel {example['panel']}: {example['sample_id']} | {example['produce_type']} | "
            f"{example['true_label']} -> {example['predicted_label']} | {example['calibrated_confidence']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
