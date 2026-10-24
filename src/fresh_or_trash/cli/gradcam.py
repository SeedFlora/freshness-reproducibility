from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from fresh_or_trash.data import build_transform
from fresh_or_trash.gradcam import GradCAM, overlay_heatmap
from fresh_or_trash.models import load_checkpoint
from fresh_or_trash.utils import resolve_device, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a Grad-CAM overlay for one image")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--class-index", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--alpha", type=float, default=0.45)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    model, metadata = load_checkpoint(args.checkpoint, device)
    image_size = int(metadata.get("image_size", 224))
    with Image.open(args.image) as handle:
        image = handle.convert("RGB")
    tensor = build_transform(image_size, training=False)(image).unsqueeze(0).to(device)
    class_indices = torch.tensor([args.class_index], device=device) if args.class_index is not None else None
    with GradCAM(model) as gradcam:
        heatmaps, logits = gradcam(tensor, class_indices)
    probabilities = torch.softmax(logits, dim=1)[0]
    predicted_index = int(probabilities.argmax())
    class_names = metadata.get("freshness_classes", [str(index) for index in range(len(probabilities))])
    overlay = overlay_heatmap(image, heatmaps[0, 0].cpu().numpy(), alpha=args.alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(args.output)
    save_json(
        {
            "checkpoint": str(args.checkpoint),
            "image": str(args.image),
            "predicted_index": predicted_index,
            "predicted_label": class_names[predicted_index],
            "probabilities": {name: float(probabilities[index]) for index, name in enumerate(class_names)},
        },
        args.output.with_suffix(".json"),
    )
    print(f"Grad-CAM for {class_names[predicted_index]} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
