from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml


ConfigDict = dict[str, Any]


class ConfigError(ValueError):
    """Raised when the user config is missing required information."""


def load_config(config_path: str | Path) -> ConfigDict:
    """Load a YAML config file and expand user/environment variables.

    Parameters
    ----------
    config_path:
        Path to the user config YAML file.

    Returns
    -------
    dict
        Parsed and normalized config dictionary.
    """
    path = Path(config_path).expanduser().resolve()

    if not path.exists():
        raise ConfigError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ConfigError(f"Config file must contain a YAML mapping at the top level: {path}")

    config["_config_path"] = str(path)
    config["_config_dir"] = str(path.parent)

    config = _expand_strings(config)
    _validate_minimal_config(config)

    return config


def load_config_from_args(argv: list[str] | None = None) -> ConfigDict:
    """Load config using a standard --config command-line argument."""
    parser = argparse.ArgumentParser(description="Bias Analysis Framework workflow runner")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the bias analysis YAML config file.",
    )
    args = parser.parse_args(argv)
    return load_config(args.config)


def get_workflow_config(config: ConfigDict, workflow_name: str) -> ConfigDict:
    """Return config for one workflow, or an empty dictionary."""
    return (
        config.get("workflows", {})
        .get(workflow_name, {})
        or {}
    )


def workflow_enabled(config: ConfigDict, workflow_name: str) -> bool:
    """Return whether a workflow is enabled in the config."""
    workflow_config = get_workflow_config(config, workflow_name)
    return bool(workflow_config.get("run", False))


def require_section(config: ConfigDict, section_name: str) -> ConfigDict:
    """Return a required top-level config section."""
    section = config.get(section_name)
    if not isinstance(section, dict):
        raise ConfigError(f"Missing or invalid required config section: '{section_name}'")
    return section


def require_list(config: ConfigDict, section_name: str) -> list[Any]:
    """Return a required top-level list config section."""
    section = config.get(section_name)
    if not isinstance(section, list):
        raise ConfigError(f"Missing or invalid required list section: '{section_name}'")
    return section


def _expand_strings(value: Any) -> Any:
    """Recursively expand ~ and environment variables in strings."""
    if isinstance(value, dict):
        return {key: _expand_strings(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_expand_strings(item) for item in value]

    if isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))

    return value


def _validate_minimal_config(config: ConfigDict) -> None:
    """Validate fields required by all workflows."""
    require_section(config, "project")
    require_section(config, "data")
    require_section(config, "cohort")
    require_list(config, "targets")
    require_list(config, "bias_factors")

    project = config["project"]
    data = config["data"]
    cohort = config["cohort"]

    for key in ["name", "output_root"]:
        if not project.get(key):
            raise ConfigError(f"Missing required config value: project.{key}")

    for key in ["clinical_table", "slide_table"]:
        if not data.get(key):
            raise ConfigError(f"Missing required config value: data.{key}")

    for key in ["organ", "dataset"]:
        if not cohort.get(key):
            raise ConfigError(f"Missing required config value: cohort.{key}")

    for idx, target in enumerate(config["targets"]):
        if not isinstance(target, dict) or not target.get("name"):
            raise ConfigError(f"Each target must have a name. Problem at targets[{idx}]")

    for idx, factor in enumerate(config["bias_factors"]):
        if not isinstance(factor, dict) or not factor.get("name"):
            raise ConfigError(f"Each bias factor must have a name. Problem at bias_factors[{idx}]")