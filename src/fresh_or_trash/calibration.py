from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class TemperatureScaler(nn.Module):
    """Single positive temperature fit on validation logits only."""

    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        self.log_temperature = nn.Parameter(torch.tensor(float(np.log(initial_temperature))))

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp(1e-3, 100.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(self, logits: torch.Tensor, targets: torch.Tensor, max_iter: int = 50) -> float:
        if logits.ndim != 2 or targets.ndim != 1 or len(logits) != len(targets):
            raise ValueError("Expected logits [N, C] and targets [N]")
        logits = logits.detach().float()
        targets = targets.detach().long()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.05, max_iter=max_iter, line_search_fn="strong_wolfe")

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss = criterion(self(logits), targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.detach().cpu())

    def save(self, path: str | Path, *, checkpoint: str | None = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {"method": "temperature_scaling", "temperature": float(self.temperature.detach()), "fit_split": "validation"}
        if checkpoint:
            payload["checkpoint"] = checkpoint
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TemperatureScaler":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(float(payload["temperature"]))
