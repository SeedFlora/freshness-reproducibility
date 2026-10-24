from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import ManifestDataset, build_transform
from .evaluation import evaluate_loader
from .manifest import load_manifest
from .models import MODEL_NAMES, FreshnessClassifier, build_model, checkpoint_payload
from .utils import resolve_device, save_json, seed_everything


@dataclass
class TrainingOptions:
    epochs: int = 30
    head_warmup_epochs: int = 2
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    auxiliary_loss_weight: float = 0.20
    amp: bool = True
    gradient_accumulation_steps: int = 1
    patience: int = 6


def make_loaders(
    frame: pd.DataFrame,
    image_root: str | Path,
    *,
    task: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
) -> tuple[dict[str, DataLoader], list[str], list[str]]:
    produce_classes = sorted(frame["produce_type"].astype(str).unique())
    loaders: dict[str, DataLoader] = {}
    freshness_classes: list[str] | None = None
    generator = torch.Generator().manual_seed(seed)
    for split in ("train", "validation", "test"):
        partition = frame[frame["split"] == split]
        if partition.empty:
            raise ValueError(f"Manifest split {split!r} is empty")
        dataset = ManifestDataset(
            partition,
            image_root,
            task=task,
            transform=build_transform(image_size, training=split == "train"),
            image_size=image_size,
            produce_classes=produce_classes,
        )
        freshness_classes = dataset.freshness_classes
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,
            generator=generator if split == "train" else None,
        )
    assert freshness_classes is not None
    return loaders, freshness_classes, produce_classes


def _train_epoch(
    model: FreshnessClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    *,
    options: TrainingOptions,
) -> dict[str, float]:
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=options.label_smoothing)
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        produce_targets = batch["produce_target"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            outputs = model(images)
            loss = criterion(outputs["freshness_logits"], targets)
            if outputs["produce_logits"] is not None:
                loss = loss + options.auxiliary_loss_weight * criterion(outputs["produce_logits"], produce_targets)
            scaled_loss = loss / options.gradient_accumulation_steps
        scaler.scale(scaled_loss).backward()
        should_step = (step + 1) % options.gradient_accumulation_steps == 0 or (step + 1) == len(loader)
        if should_step:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        batch_size = len(targets)
        loss_sum += float(loss.detach()) * batch_size
        correct += int((outputs["freshness_logits"].argmax(dim=1) == targets).sum())
        sample_count += batch_size
    return {"loss": loss_sum / sample_count, "accuracy": correct / sample_count}


def train_model(
    frame: pd.DataFrame,
    image_root: str | Path,
    *,
    architecture: str,
    output_dir: str | Path,
    task: str = "three_stage",
    image_size: int = 224,
    batch_size: int = 24,
    num_workers: int = 4,
    pin_memory: bool = True,
    pretrained: bool = True,
    auxiliary_produce_head: bool = False,
    options: TrainingOptions | None = None,
    seed: int = 42,
    requested_device: str = "auto",
    evaluate_test: bool = True,
) -> dict[str, Any]:
    if architecture not in MODEL_NAMES:
        raise ValueError(f"architecture must be one of {MODEL_NAMES}")
    if task not in {"three_stage", "binary"}:
        raise ValueError("task must be three_stage or binary")
    options = options or TrainingOptions()
    seed_everything(seed)
    device = resolve_device(requested_device)
    loaders, freshness_classes, produce_classes = make_loaders(
        frame,
        image_root,
        task=task,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory and device.type == "cuda",
        seed=seed,
    )
    model = build_model(
        architecture,
        num_freshness_classes=len(freshness_classes),
        pretrained=pretrained,
        num_produce_classes=len(produce_classes) if auxiliary_produce_head else None,
    ).to(device)
    if options.head_warmup_epochs > 0:
        model.freeze_backbone(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.2, patience=2)
    scaler = torch.amp.GradScaler(device.type, enabled=options.amp and device.type == "cuda")

    history: list[dict[str, float | int]] = []
    best_score = float("-inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    started = time.perf_counter()
    for epoch in range(options.epochs):
        if epoch == options.head_warmup_epochs:
            model.freeze_backbone(False)
        train_metrics = _train_epoch(model, loaders["train"], optimizer, scaler, device, options=options)
        validation_metrics, _, _, _ = evaluate_loader(
            model,
            loaders["validation"],
            device,
            class_names=freshness_classes,
        )
        score = float(validation_metrics["macro_f1"])
        scheduler.step(score)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "validation_accuracy": float(validation_metrics["accuracy"]),
                "validation_macro_f1": score,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= options.patience:
                break
    if best_state is None:
        raise RuntimeError("Training completed without a checkpoint")
    model.load_state_dict(best_state)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "best.pt"
    torch.save(
        checkpoint_payload(
            model,
            freshness_classes=freshness_classes,
            produce_classes=produce_classes,
            image_size=image_size,
            task=task,
            extra={"best_epoch": best_epoch, "seed": seed, "validation_macro_f1": best_score},
        ),
        checkpoint_path,
    )
    test_metrics: dict[str, object] | None = None
    if evaluate_test:
        test_metrics, predictions, _, _ = evaluate_loader(
            model,
            loaders["test"],
            device,
            class_names=freshness_classes,
        )
        predictions.to_csv(output / "test_predictions.csv", index=False)
        save_json(test_metrics, output / "test_metrics.json")
    run = {
        "architecture": architecture,
        "task": task,
        "seed": seed,
        "device": str(device),
        "checkpoint": str(checkpoint_path),
        "best_epoch": best_epoch,
        "validation_macro_f1": best_score,
        "elapsed_seconds": time.perf_counter() - started,
        "test_metrics": test_metrics,
        "history": history,
    }
    save_json(run, output / "run.json")
    return run


def options_from_config(config: dict[str, Any]) -> TrainingOptions:
    training = config.get("training", {})
    fields = TrainingOptions.__dataclass_fields__
    return TrainingOptions(**{key: value for key, value in training.items() if key in fields})
