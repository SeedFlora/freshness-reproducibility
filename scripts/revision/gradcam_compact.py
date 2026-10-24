"""Render the frozen Grad-CAM selection at IEEE single-column width.

The six examples, checkpoint, and selection rule are those of
``scripts/generate_gradcam_examples.py``; this script only re-lays them out
with legible labels and never rewrites the frozen figure or its sidecar.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from revision_common import PAPER_FIGURES, ROOT, SINGLE_COLUMN_WIDTH


SHORT = {"Fresh": "F", "Semi-Fresh": "SF", "Rotten": "R"}


def _frozen_module():
    spec = importlib.util.spec_from_file_location("generate_gradcam_examples", ROOT / "scripts/generate_gradcam_examples.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=PAPER_FIGURES / "gradcam_compact")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    frozen = _frozen_module()
    frozen.seed_everything(frozen.REFERENCE_SEED, deterministic=True)
    frame, _ = frozen.load_and_validate_sources()
    selected = frozen.deterministic_selection(frame)
    sidecar = json.loads((ROOT / "results/gradcam_examples.json").read_text(encoding="utf-8"))
    expected = [example["sample_id"] for example in sidecar["examples"]]
    if [str(item["row"]["sample_id"]) for item in selected] != expected:
        raise ValueError("Selection differs from the frozen Grad-CAM sidecar")

    model, metadata = frozen.load_checkpoint(frozen.CHECKPOINT, args.device)
    image_size = int(metadata.get("image_size", 224))
    display = transforms.Compose(
        [transforms.Resize(int(round(image_size / 0.875)), interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(image_size)]
    )
    tensor_transform = frozen.build_transform(image_size=image_size, training=False)
    crops, tensors, targets = [], [], []
    for item in selected:
        with Image.open(frozen.IMAGE_ROOT / str(item["row"]["source_path"])) as handle:
            image = handle.convert("RGB")
        crops.append(display(image))
        tensors.append(tensor_transform(image))
        targets.append(int(item["row"]["predicted_index"]))
    with frozen.GradCAM(model) as gradcam:
        heatmaps, _ = gradcam(torch.stack(tensors).to(args.device), torch.tensor(targets, device=args.device))
    heatmaps = heatmaps[:, 0].cpu().numpy()

    plt.rcParams.update({"font.size": 6, "font.family": "DejaVu Sans"})
    figure, axes = plt.subplots(2, 3, figsize=(SINGLE_COLUMN_WIDTH, 2.62))
    for axis, item, crop, heatmap in zip(axes.flat, selected, crops, heatmaps, strict=True):
        row = item["row"]
        axis.imshow(frozen.overlay_heatmap(crop, heatmap, alpha=0.42))
        produce = str(row["produce_type"]).replace("_", " ").capitalize()
        axis.set_title(
            f"{produce}: {SHORT[row['freshness_label']]}$\\rightarrow${SHORT[row['predicted_label']]} ({float(row['confidence']):.2f})",
            fontsize=6,
            pad=2,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_linewidth(0.4)
            spine.set_color("#9CA3AF")
    axes[0, 0].set_ylabel("Correct", fontsize=6)
    axes[1, 0].set_ylabel("Error", fontsize=6)
    figure.subplots_adjust(left=0.05, right=0.995, top=0.94, bottom=0.01, wspace=0.05, hspace=0.16)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        figure.savefig(args.output.with_suffix(suffix), dpi=300, facecolor="white")
    plt.close(figure)
    print(f"compact Grad-CAM -> {args.output}.png/.pdf ({len(selected)} panels, ids match the frozen sidecar)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
