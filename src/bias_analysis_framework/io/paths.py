from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkflowPaths:
    """Standard output paths for one workflow step."""

    output_root: Path
    workflow_root: Path
    tables_dir: Path
    figures_dir: Path
    logs_dir: Path


def get_output_root(config: dict[str, Any]) -> Path:
    """Return project output root from config."""
    output_root = config.get("project", {}).get("output_root")

    if not output_root:
        raise ValueError("Missing required config value: project.output_root")

    return Path(output_root).expanduser().resolve()


def get_cohort_label(config: dict[str, Any]) -> str:
    """Return a stable organ/dataset label."""
    cohort = config.get("cohort", {})
    organ = str(cohort.get("organ", "unknown_organ")).strip()
    dataset = str(cohort.get("dataset", "unknown_dataset")).strip()
    return f"{organ}__{dataset}"


def get_workflow_paths(
    config: dict[str, Any],
    workflow_name: str,
    create: bool = True,
) -> WorkflowPaths:
    """Create and return standard directories for a workflow.

    Directory layout:
        output_root/workflow_name/organ__dataset/
            tables/
            figures/
            logs/
    """
    output_root = get_output_root(config)
    cohort_label = get_cohort_label(config)

    workflow_root = output_root / workflow_name / cohort_label
    tables_dir = workflow_root / "tables"
    figures_dir = workflow_root / "figures"
    logs_dir = workflow_root / "logs"

    paths = WorkflowPaths(
        output_root=output_root,
        workflow_root=workflow_root,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        logs_dir=logs_dir,
    )

    if create:
        for directory in [
            paths.output_root,
            paths.workflow_root,
            paths.tables_dir,
            paths.figures_dir,
            paths.logs_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    return paths


def safe_name(value: str) -> str:
    """Create a filesystem-safe label."""
    cleaned = str(value).strip()
    for char in [" ", "/", "\\", ":", ";", ",", "|", "(", ")", "[", "]"]:
        cleaned = cleaned.replace(char, "_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")