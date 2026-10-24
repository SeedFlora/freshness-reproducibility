from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models


MODEL_NAMES = (
    "mobilenet_v2",
    "resnet50",
    "densenet121",
    "efficientnet_b0",
    "convnext_tiny",
)


@dataclass(frozen=True)
class BackboneSpec:
    constructor: Any
    weights: Any
    feature_dim: int


_SPECS = {
    "mobilenet_v2": BackboneSpec(models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT, 1280),
    "resnet50": BackboneSpec(models.resnet50, models.ResNet50_Weights.DEFAULT, 2048),
    "densenet121": BackboneSpec(models.densenet121, models.DenseNet121_Weights.DEFAULT, 1024),
    "efficientnet_b0": BackboneSpec(models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT, 1280),
    "convnext_tiny": BackboneSpec(models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT, 768),
}


class FreshnessClassifier(nn.Module):
    """A torchvision backbone with freshness and optional produce heads."""

    def __init__(
        self,
        architecture: str,
        num_freshness_classes: int,
        *,
        pretrained: bool = True,
        num_produce_classes: int | None = None,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        if architecture not in MODEL_NAMES:
            raise ValueError(f"Unknown architecture {architecture!r}; choose from {MODEL_NAMES}")
        if num_freshness_classes < 2:
            raise ValueError("num_freshness_classes must be at least 2")
        spec = _SPECS[architecture]
        self.architecture = architecture
        self.num_freshness_classes = num_freshness_classes
        self.num_produce_classes = num_produce_classes
        self.backbone = spec.constructor(weights=spec.weights if pretrained else None)
        self._remove_classifier()
        self.feature_dim = spec.feature_dim
        self.freshness_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, num_freshness_classes))
        self.produce_head = (
            nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_dim, num_produce_classes))
            if num_produce_classes
            else None
        )

    def _remove_classifier(self) -> None:
        if self.architecture == "resnet50":
            self.backbone.fc = nn.Identity()
        else:
            self.backbone.classifier = nn.Identity()

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        return features

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor | None]:
        features = self.extract_features(images)
        return {
            "freshness_logits": self.freshness_head(features),
            "produce_logits": self.produce_head(features) if self.produce_head is not None else None,
            "features": features,
        }

    def freeze_backbone(self, frozen: bool = True) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not frozen

    def gradcam_target_layer(self) -> nn.Module:
        if self.architecture == "resnet50":
            return self.backbone.layer4[-1]
        if self.architecture == "densenet121":
            return self.backbone.features.denseblock4
        return self.backbone.features[-1]


def build_model(
    architecture: str,
    num_freshness_classes: int = 3,
    *,
    pretrained: bool = True,
    num_produce_classes: int | None = None,
    dropout: float = 0.20,
) -> FreshnessClassifier:
    """Build one of the five controlled benchmark architectures."""
    return FreshnessClassifier(
        architecture=architecture,
        num_freshness_classes=num_freshness_classes,
        pretrained=pretrained,
        num_produce_classes=num_produce_classes,
        dropout=dropout,
    )


def checkpoint_payload(
    model: FreshnessClassifier,
    *,
    freshness_classes: list[str],
    produce_classes: list[str],
    image_size: int,
    task: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "architecture": model.architecture,
        "num_freshness_classes": model.num_freshness_classes,
        "num_produce_classes": model.num_produce_classes,
        "freshness_classes": freshness_classes,
        "produce_classes": produce_classes,
        "image_size": image_size,
        "task": task,
        "state_dict": model.state_dict(),
        "extra": extra or {},
    }


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[FreshnessClassifier, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    required = {"architecture", "num_freshness_classes", "state_dict"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Checkpoint is missing metadata: {sorted(missing)}")
    model = build_model(
        payload["architecture"],
        num_freshness_classes=int(payload["num_freshness_classes"]),
        pretrained=False,
        num_produce_classes=payload.get("num_produce_classes"),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    return model, payload
