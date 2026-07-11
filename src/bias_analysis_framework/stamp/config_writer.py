"""Configuration generation for STAMP preprocessing."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_stamp_config_template(template_path: str | Path) -> dict[str, Any]:
    """Load a STAMP YAML configuration template.

    Parameters
    ----------
    template_path:
        Path to an existing STAMP-compatible YAML configuration file.

    Returns
    -------
    dict
        Parsed YAML configuration.

    Raises
    ------
    FileNotFoundError
        If the template file does not exist.
    ValueError
        If the template is empty, is not a mapping, or lacks a preprocessing
        section.
    """
    path = Path(template_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"STAMP config template does not exist: {path}")

    if not path.is_file():
        raise ValueError(f"STAMP config template is not a file: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"STAMP config template is empty: {path}")

    if not isinstance(config, dict):
        raise ValueError(
            "STAMP config template must contain a top-level YAML mapping."
        )

    preprocessing = config.get("preprocessing")
    if preprocessing is None:
        config["preprocessing"] = {}
    elif not isinstance(preprocessing, dict):
        raise ValueError(
            "The 'preprocessing' section of the STAMP template must be a mapping."
        )

    return config


def build_stamp_preprocess_config(
    template_config: dict[str, Any],
    *,
    wsi_dir: str | Path,
    output_dir: str | Path,
    extractor: str,
    cache_dir: str | Path | None = None,
    default_slide_mpp: float | None = None,
    max_workers: int | None = None,
    preprocessing_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one STAMP preprocessing config for one feature extractor.

    Values supplied here override matching values in the template's
    ``preprocessing`` section.
    """
    config = deepcopy(template_config)
    preprocessing = config.setdefault("preprocessing", {})

    preprocessing["wsi_dir"] = str(Path(wsi_dir).expanduser().resolve())
    preprocessing["output_dir"] = str(Path(output_dir).expanduser().resolve())
    preprocessing["extractor"] = str(extractor)

    if cache_dir is not None:
        preprocessing["cache_dir"] = str(
            Path(cache_dir).expanduser().resolve()
        )

    if default_slide_mpp is not None:
        preprocessing["default_slide_mpp"] = float(default_slide_mpp)

    if max_workers is not None:
        if int(max_workers) < 1:
            raise ValueError("STAMP max_workers must be at least 1.")
        preprocessing["max_workers"] = int(max_workers)

    if preprocessing_overrides:
        preprocessing.update(preprocessing_overrides)

    return config


def write_stamp_config(
    config: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Write a generated STAMP YAML configuration."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
            default_flow_style=False,
        )

    return path


def safe_config_name(value: object) -> str:
    """Return a safe filename component for a generated config."""
    text = str(value).strip()

    for character in [
        " ",
        "/",
        "\\",
        ":",
        ";",
        ",",
        "|",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
    ]:
        text = text.replace(character, "_")

    while "__" in text:
        text = text.replace("__", "_")

    return text.strip("_") or "extractor"