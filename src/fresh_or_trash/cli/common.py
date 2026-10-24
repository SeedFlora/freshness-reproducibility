from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from fresh_or_trash.config import load_config


DEFAULT_CONFIG = Path("config/benchmark.yaml")


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Benchmark YAML configuration")


def config_value(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def read_config(path: Path) -> dict[str, Any]:
    return load_config(path)
