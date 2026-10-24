"""Run the Streamlit inference path without launching a browser."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from app.app import (  # noqa: E402
    DISCLAIMER,
    RECOMMENDATIONS,
    checkpoint_temperature,
    gradcam,
    model_input,
    predict,
)
from fresh_or_trash.manifest import file_sha256  # noqa: E402
from fresh_or_trash.models import load_checkpoint  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    temperature, embedded = checkpoint_temperature(payload)
    with Image.open(args.image) as source:
        tensor, _ = model_input(source.convert("RGB"), int(payload["image_size"]))
    output, probabilities, latency_ms = predict(model, tensor, temperature, device)
    predicted_index = int(probabilities.argmax(dim=1).item())
    predicted_label = payload["freshness_classes"][predicted_index]
    heatmap = gradcam(model, tensor, predicted_index, device)

    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "image": str(args.image.resolve()),
        "image_sha256": file_sha256(args.image),
        "device": str(device),
        "architecture": payload["architecture"],
        "task": payload["task"],
        "temperature": temperature,
        "embedded_temperature": embedded,
        "freshness_classes": payload["freshness_classes"],
        "probabilities": [float(value) for value in probabilities[0]],
        "prediction": predicted_label,
        "produce_head_classes": len(payload.get("produce_classes", [])),
        "produce_logits_present": isinstance(output.get("produce_logits"), torch.Tensor),
        "latency_ms": latency_ms,
        "latency_scope": "single cold application forward pass; not the warmed benchmark profile",
        "heatmap_shape": list(heatmap.shape),
        "heatmap_finite": bool(np.isfinite(heatmap).all()),
        "recommendation": RECOMMENDATIONS[predicted_label],
        "disclaimer_exact": DISCLAIMER
        == (
            "This prototype performs visual freshness screening only.\n"
            "It does not determine food safety, microbial contamination,\n"
            "chemical contamination, or whether an item is safe to consume."
        ),
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
