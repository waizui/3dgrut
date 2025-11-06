"""Utility script for launching multiple 3DGRt training runs.

The script wraps :mod:`train.py` and executes it repeatedly with different
Hydra override sets.  Update :data:`EXPERIMENTS` with the scenes and optional
overrides you would like to sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = PROJECT_ROOT / "train.py"
DATA_ROOT = PROJECT_ROOT.parent / "dataset" / "assets"

DEFAULT_CONFIG_NAME = "apps/colmap_3dgrt.yaml"
DEFAULT_OVERRIDES: dict[str, Any] = {
    "out_dir": "runs",
    "dataset.downsample_factor": 1,
}


@dataclass(frozen=True)
class Experiment:
    """Definition of a single training configuration."""

    name: str
    overrides: Mapping[str, Any]
    config_name: str | None = None

    def resolved_overrides(self, base_overrides: Mapping[str, Any]) -> dict[str, Any]:
        merged = {**base_overrides, **self.overrides}

        merged.setdefault("path", resolve_dataset_path(self.name))
        merged.setdefault("experiment_name", f"{slugify(self.name)}_3dgrt")

        return merged

    def cli_args(self, base_overrides: Mapping[str, Any]) -> list[str]:
        config_argument = self.config_name or DEFAULT_CONFIG_NAME
        cli: list[str] = ["--config-name", config_argument]
        for key, value in self.resolved_overrides(base_overrides).items():
            if value is None:
                continue
            cli.append(format_override(key, value))
        return cli


def slugify(value: str) -> str:
    """Normalize experiment names so they can be used as folder names."""

    return re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower() or "run"


def resolve_dataset_path(dataset: str) -> str:
    dataset_path = Path(dataset)
    if not dataset_path.is_absolute():
        dataset_path = DATA_ROOT / dataset_path
    return str(dataset_path)


def format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def format_override(key: str, value: Any) -> str:
    if isinstance(value, (list, tuple)):
        payload = ",".join(format_scalar(item) for item in value)
        return f"{key}=[{payload}]"

    payload = format_scalar(value)
    if isinstance(value, str) and re.search(r"\s", value):
        payload = f'"{value}"'
    return f"{key}={payload}"


def parse_cli_overrides(raw_args: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for raw in raw_args:
        if "=" not in raw:
            raise ValueError(f"Override '{raw}' must be in key=value form.")
        key, value = raw.split("=", 1)
        overrides[key] = interpret_value(value)
    return overrides


def interpret_value(value: str) -> Any:
    lowered = value.strip().lower()
    value = value.strip()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1]
        if not inner:
            return []
    return [interpret_value(part.strip()) for part in inner.split(",")]
    return value


EXPERIMENTS: list[Experiment] = [
    Experiment(
        name="bicycle",
        overrides={
            "dataset.downsample_factor": 4,
        },
    ),
    Experiment(
        name="garden",
        overrides={
            "dataset.downsample_factor": 4,
        },
    ),
    Experiment(
        name="stump",
        overrides={
            "dataset.downsample_factor": 4,
        },
    ),
    Experiment(
        name="bonsai",
        overrides={
            "dataset.downsample_factor": 2,
        },
    ),
    Experiment(
        name="counter",
        overrides={
            "dataset.downsample_factor": 2,
        },
    ),
    Experiment(
        name="room",
        overrides={
            "dataset.downsample_factor": 2,
        },
    ),
    Experiment(
        name="kitchen",
        overrides={
            "dataset.downsample_factor": 2,
        },
    ),
    Experiment(
        name="train",
        overrides={
            "dataset.downsample_factor": 1,
        },
    ),
    Experiment(
        name="truck",
        overrides={
            "dataset.downsample_factor": 1,
        },
    ),
]


def run_experiment(exp: Experiment, base_overrides: Mapping[str, Any]) -> None:
    command = [sys.executable, str(TRAIN_SCRIPT), *exp.cli_args(base_overrides)]

    print(f"\n=== Running experiment: {exp.name} ===")
    print("CLI:", " ".join(command))

    subprocess.run(command, check=True)


def run_all(
    experiments: Iterable[Experiment], base_overrides: Mapping[str, Any]
) -> None:
    for exp in experiments:
        run_experiment(exp, base_overrides)


if __name__ == "__main__":
    try:
        cli_overrides = parse_cli_overrides(sys.argv[1:])
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    combined_overrides = {**DEFAULT_OVERRIDES, **cli_overrides}
    run_all(EXPERIMENTS, combined_overrides)
