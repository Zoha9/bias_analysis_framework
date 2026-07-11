from __future__ import annotations

from pathlib import Path
from typing import Any


class InputValidationError(ValueError):
    """Raised when expected user input files or columns are missing."""


def validate_existing_file(path: str | Path, label: str) -> Path:
    """Validate that a configured input file exists."""
    resolved = Path(path).expanduser().resolve()

    if not resolved.exists():
        raise InputValidationError(f"{label} does not exist: {resolved}")

    if not resolved.is_file():
        raise InputValidationError(f"{label} is not a file: {resolved}")

    return resolved


def validate_existing_dir(path: str | Path, label: str) -> Path:
    """Validate that a configured input directory exists."""
    resolved = Path(path).expanduser().resolve()

    if not resolved.exists():
        raise InputValidationError(f"{label} does not exist: {resolved}")

    if not resolved.is_dir():
        raise InputValidationError(f"{label} is not a directory: {resolved}")

    return resolved


def target_names(config: dict[str, Any]) -> list[str]:
    """Return target biomarker names from config."""
    return [target["name"] for target in config.get("targets", [])]


def bias_factor_names(config: dict[str, Any]) -> list[str]:
    """Return bias factor names from config."""
    return [factor["name"] for factor in config.get("bias_factors", [])]