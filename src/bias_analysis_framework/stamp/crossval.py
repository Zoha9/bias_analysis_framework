"""Config-driven STAMP cross-validation training."""

from __future__ import annotations

import shlex
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from bias_analysis_framework.stamp.config_writer import (
    load_stamp_config_template,
    safe_config_name,
    write_stamp_config,
)
from bias_analysis_framework.stamp.discovery import (
    count_feature_files,
    discover_model_feature_directory,
)
from bias_analysis_framework.stamp.preprocess import (
    build_environment_from_config,
    get_stamp_config,
)
from bias_analysis_framework.stamp.runner import (
    check_stamp_command_available,
    normalize_stamp_command,
)


@dataclass(frozen=True)
class StampCrossvalJob:
    """One STAMP cross-validation job."""

    model_name: str
    extractor: str
    label_name: str
    label_type: str
    feature_dir: Path
    output_dir: Path
    config_path: Path
    log_path: Path
    expected_splits: int


def run_stamp_crossval_training(
    *,
    config: dict[str, Any],
    workflow_output_dir: str | Path,
) -> pd.DataFrame:
    """Prepare and optionally run all configured STAMP crossval jobs."""
    workflow_output_dir = (
        Path(workflow_output_dir)
        .expanduser()
        .resolve()
    )
    workflow_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    workflow_config = get_crossval_workflow_config(
        config
    )
    stamp_config = get_stamp_config(config)

    validate_crossval_config(
        config=config,
        workflow_config=workflow_config,
        stamp_config=stamp_config,
    )

    jobs = build_crossval_jobs(
        config=config,
        workflow_output_dir=workflow_output_dir,
    )

    template_path = get_crossval_template_path(
        stamp_config
    )
    template_config = load_stamp_config_template(
        template_path
    )

    execute = bool(
        workflow_config.get(
            "execute",
            False,
        )
    )
    skip_existing = bool(
        workflow_config.get(
            "skip_existing",
            True,
        )
    )
    continue_on_error = bool(
        workflow_config.get(
            "continue_on_error",
            False,
        )
    )

    environment = (
        build_environment_from_config(
            stamp_config
        )
        if execute
        else None
    )

    rows: list[dict[str, Any]] = []

    for job in jobs:
        existing_splits = discover_split_directories(
            job.output_dir
        )

        if (
            skip_existing
            and len(existing_splits)
            >= job.expected_splits
        ):
            rows.append(
                build_crossval_summary_row(
                    job=job,
                    status="skipped_existing",
                    message=(
                        f"Found {len(existing_splits)} "
                        "existing split directories."
                    ),
                    executed=False,
                    return_code=pd.NA,
                    split_count=len(existing_splits),
                )
            )
            continue

        job.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        generated_config = build_stamp_crossval_config(
            template_config=template_config,
            job=job,
            config=config,
            workflow_config=workflow_config,
        )

        write_stamp_config(
            config=generated_config,
            output_path=job.config_path,
        )

        try:
            return_code = run_crossval_command(
                command=stamp_config.get(
                    "command",
                    "stamp",
                ),
                config_path=job.config_path,
                log_path=job.log_path,
                environment=environment,
                working_directory=stamp_config.get(
                    "working_directory"
                ),
                execute=execute,
            )

            split_directories = discover_split_directories(
                job.output_dir
            )

            if not execute:
                status = "dry_run"
                message = (
                    "STAMP crossval config and command were "
                    "generated but not executed."
                )
            elif len(split_directories) >= job.expected_splits:
                status = "completed"
                message = (
                    f"Found {len(split_directories)} "
                    "cross-validation split directories."
                )
            else:
                status = "completed_with_warning"
                message = (
                    f"STAMP returned successfully, but only "
                    f"{len(split_directories)} of "
                    f"{job.expected_splits} expected split "
                    "directories were found."
                )

            rows.append(
                build_crossval_summary_row(
                    job=job,
                    status=status,
                    message=message,
                    executed=execute,
                    return_code=return_code,
                    split_count=len(split_directories),
                )
            )

        except Exception as error:
            rows.append(
                build_crossval_summary_row(
                    job=job,
                    status="failed",
                    message=(
                        f"{type(error).__name__}: {error}"
                    ),
                    executed=execute,
                    return_code=getattr(
                        error,
                        "returncode",
                        pd.NA,
                    ),
                    split_count=len(
                        discover_split_directories(
                            job.output_dir
                        )
                    ),
                )
            )

            write_crossval_summary(
                rows=rows,
                output_path=(
                    workflow_output_dir
                    / "tables"
                    / "stamp_crossval_jobs.xlsx"
                ),
            )

            if not continue_on_error:
                raise

    summary_df = write_crossval_summary(
        rows=rows,
        output_path=(
            workflow_output_dir
            / "tables"
            / "stamp_crossval_jobs.xlsx"
        ),
    )

    return summary_df


def get_crossval_workflow_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return Step 4 workflow config."""
    value = (
        config.get("workflows", {})
        .get("stamp_crossval_training", {})
    )

    if not isinstance(value, dict):
        raise ValueError(
            "workflows.stamp_crossval_training "
            "must be a mapping."
        )

    return value


def validate_crossval_config(
    *,
    config: dict[str, Any],
    workflow_config: dict[str, Any],
    stamp_config: dict[str, Any],
) -> None:
    """Validate required Step 4 settings."""
    data_config = config.get("data", {})

    for key in (
        "clinical_table",
        "slide_table",
        "feature_root",
    ):
        if not data_config.get(key):
            raise ValueError(
                f"Missing required config value: data.{key}"
            )

    clinical_path = Path(
        data_config["clinical_table"]
    ).expanduser().resolve()
    slide_path = Path(
        data_config["slide_table"]
    ).expanduser().resolve()
    feature_root = Path(
        data_config["feature_root"]
    ).expanduser().resolve()

    if not clinical_path.is_file():
        raise FileNotFoundError(
            f"Clinical table does not exist: "
            f"{clinical_path}"
        )

    if not slide_path.is_file():
        raise FileNotFoundError(
            f"Slide table does not exist: "
            f"{slide_path}"
        )

    if not feature_root.is_dir():
        raise FileNotFoundError(
            f"Feature root does not exist: "
            f"{feature_root}"
        )

    template_path = get_crossval_template_path(
        stamp_config
    )

    if not template_path.is_file():
        raise FileNotFoundError(
            f"STAMP crossval template does not exist: "
            f"{template_path}"
        )

    n_splits = int(
        workflow_config.get(
            "n_splits",
            5,
        )
    )

    if n_splits < 2:
        raise ValueError(
            "STAMP cross-validation n_splits "
            "must be at least 2."
        )

    if not get_configured_models(config):
        raise ValueError(
            "No models are configured."
        )

    if not get_training_labels(config):
        raise ValueError(
            "No cross-validation labels are configured."
        )


def get_crossval_template_path(
    stamp_config: dict[str, Any],
) -> Path:
    """Return the STAMP crossval template path."""
    template_value = (
        stamp_config.get(
            "crossval_config_template"
        )
        or stamp_config.get(
            "stamp_config_template"
        )
    )

    if not template_value:
        raise ValueError(
            "Missing stamp.crossval_config_template."
        )

    return (
        Path(template_value)
        .expanduser()
        .resolve()
    )


def build_crossval_jobs(
    *,
    config: dict[str, Any],
    workflow_output_dir: Path,
) -> list[StampCrossvalJob]:
    """Build all model × label cross-validation jobs."""
    workflow_config = get_crossval_workflow_config(
        config
    )
    data_config = config["data"]
    cohort_config = config.get("cohort", {})

    feature_root = Path(
        data_config["feature_root"]
    ).expanduser().resolve()

    organ = str(
        cohort_config.get(
            "organ",
            "",
        )
    ).strip()
    dataset = str(
        cohort_config.get(
            "dataset",
            "",
        )
    ).strip()

    expected_splits = int(
        workflow_config.get(
            "n_splits",
            5,
        )
    )

    config_dir = (
        workflow_output_dir
        / "configs"
    )
    log_dir = (
        workflow_output_dir
        / "logs"
    )
    result_dir = (
        workflow_output_dir
        / "crossval_results"
    )

    for directory in (
        config_dir,
        log_dir,
        result_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    jobs: list[StampCrossvalJob] = []

    for model_name, extractor in get_configured_models(
        config
    ):
        feature_dir = discover_model_feature_directory(
            feature_root=feature_root,
            model_name=extractor,
            organ=organ,
            dataset=dataset,
        )

        for label_name, label_type in get_training_labels(
            config
        ):
            safe_model = safe_config_name(
                extractor
            )
            safe_label = safe_config_name(
                label_name
            )

            output_dir = (
                result_dir
                / safe_label
                / safe_model
            )

            jobs.append(
                StampCrossvalJob(
                    model_name=model_name,
                    extractor=extractor,
                    label_name=label_name,
                    label_type=label_type,
                    feature_dir=feature_dir,
                    output_dir=output_dir,
                    config_path=(
                        config_dir
                        / (
                            "stamp_crossval__"
                            f"{safe_model}__"
                            f"{safe_label}.yaml"
                        )
                    ),
                    log_path=(
                        log_dir
                        / (
                            "stamp_crossval__"
                            f"{safe_model}__"
                            f"{safe_label}.log"
                        )
                    ),
                    expected_splits=expected_splits,
                )
            )

    return jobs


def get_configured_models(
    config: dict[str, Any],
) -> list[tuple[str, str]]:
    """Return display name and STAMP extractor name."""
    output: list[tuple[str, str]] = []

    for model in config.get("models", []):
        if isinstance(model, str):
            display_name = model
            extractor = model
        elif isinstance(model, dict):
            extractor = str(
                model.get("extractor")
                or model.get("name")
                or ""
            ).strip()
            display_name = str(
                model.get("name")
                or extractor
            ).strip()
        else:
            raise ValueError(
                "Each models entry must be a string "
                "or mapping."
            )

        if extractor:
            pair = (
                display_name,
                extractor,
            )

            if pair not in output:
                output.append(pair)

    return output


def get_training_labels(
    config: dict[str, Any],
) -> list[tuple[str, str]]:
    """Return labels selected for STAMP training."""
    workflow_config = get_crossval_workflow_config(
        config
    )

    label_sources = workflow_config.get(
        "label_sources",
        [
            "targets",
            "bias_factors",
        ],
    )

    explicit_labels = workflow_config.get(
        "labels"
    )

    output: list[tuple[str, str]] = []

    if explicit_labels:
        for label in explicit_labels:
            if isinstance(label, str):
                pair = (
                    label,
                    "explicit",
                )
            elif isinstance(label, dict):
                name = str(
                    label.get("name", "")
                ).strip()
                label_type = str(
                    label.get(
                        "type",
                        "explicit",
                    )
                ).strip()

                if not name:
                    continue

                pair = (
                    name,
                    label_type,
                )
            else:
                raise ValueError(
                    "Crossval labels must be strings "
                    "or mappings."
                )

            if pair not in output:
                output.append(pair)

        return output

    if "targets" in label_sources:
        for target in config.get(
            "targets",
            [],
        ):
            name = target.get("name")

            if name:
                pair = (
                    str(name),
                    "target",
                )

                if pair not in output:
                    output.append(pair)

    if "bias_factors" in label_sources:
        for factor in config.get(
            "bias_factors",
            [],
        ):
            name = (
                factor.get("column")
                or factor.get("name")
            )

            if name:
                pair = (
                    str(name),
                    "bias_factor",
                )

                if pair not in output:
                    output.append(pair)

    return output


def build_stamp_crossval_config(
    *,
    template_config: dict[str, Any],
    job: StampCrossvalJob,
    config: dict[str, Any],
    workflow_config: dict[str, Any],
) -> dict[str, Any]:
    """Create one STAMP crossval config."""
    generated = deepcopy(
        template_config
    )
    crossval = generated.setdefault(
        "crossval",
        {},
    )

    data_config = config["data"]

    crossval["output_dir"] = str(
        job.output_dir
    )
    crossval["feature_dir"] = str(
        job.feature_dir
    )
    crossval["clini_table"] = str(
        Path(
            data_config["clinical_table"]
        ).expanduser().resolve()
    )
    crossval["slide_table"] = str(
        Path(
            data_config["slide_table"]
        ).expanduser().resolve()
    )
    crossval["ground_truth_label"] = (
        job.label_name
    )
    crossval["n_splits"] = int(
        job.expected_splits
    )

    optional_keys = (
        "num_workers",
        "patient_column",
        "filename_column",
        "categorical",
        "balance",
    )

    for key in optional_keys:
        if key in workflow_config:
            crossval[key] = (
                workflow_config[key]
            )

    overrides = workflow_config.get(
        "crossval_overrides",
        {},
    )

    if overrides:
        if not isinstance(overrides, dict):
            raise ValueError(
                "crossval_overrides must be a mapping."
            )

        crossval.update(overrides)

    return generated


def run_crossval_command(
    *,
    command: str | list[str],
    config_path: Path,
    log_path: Path,
    environment: dict[str, str] | None,
    working_directory: str | Path | None,
    execute: bool,
) -> int:
    """Run or preview a STAMP crossval command."""
    command_args = [
        *normalize_stamp_command(command),
        "--config",
        str(config_path),
        "crossval",
    ]

    printable = shlex.join(
        command_args
    )
    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not execute:
        log_path.write_text(
            "DRY RUN\n"
            f"Command: {printable}\n"
            f"Config: {config_path}\n",
            encoding="utf-8",
        )
        return 0

    check_stamp_command_available(
        command
    )

    cwd = (
        str(
            Path(
                working_directory
            ).expanduser().resolve()
        )
        if working_directory
        else None
    )

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_file:
        log_file.write(
            f"Command: {printable}\n"
        )
        log_file.write(
            f"Config: {config_path}\n\n"
        )
        log_file.flush()

        completed = subprocess.run(
            command_args,
            cwd=cwd,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            command_args,
        )

    return completed.returncode


def discover_split_directories(
    output_dir: str | Path,
) -> list[Path]:
    """Return STAMP split directories."""
    path = Path(output_dir)

    if not path.exists():
        return []

    return sorted(
        child
        for child in path.iterdir()
        if child.is_dir()
        and child.name.startswith(
            "split-"
        )
    )


def build_crossval_summary_row(
    *,
    job: StampCrossvalJob,
    status: str,
    message: str,
    executed: bool,
    return_code: Any,
    split_count: int,
) -> dict[str, Any]:
    """Create one crossval summary row."""
    return {
        "Model": job.model_name,
        "Extractor": job.extractor,
        "Label": job.label_name,
        "LabelType": job.label_type,
        "Status": status,
        "Executed": executed,
        "ReturnCode": return_code,
        "FeatureDir": str(
            job.feature_dir
        ),
        "FeatureFileCount": count_feature_files(
            job.feature_dir
        ),
        "OutputDir": str(
            job.output_dir
        ),
        "GeneratedConfig": str(
            job.config_path
        ),
        "LogFile": str(
            job.log_path
        ),
        "ExpectedSplits": (
            job.expected_splits
        ),
        "FoundSplits": split_count,
        "Message": message,
    }


def write_crossval_summary(
    *,
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> pd.DataFrame:
    """Write the Step 4 crossval job workbook."""
    output_path = (
        Path(output_path)
        .expanduser()
        .resolve()
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df = pd.DataFrame(
        rows
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:
        summary_df.to_excel(
            writer,
            sheet_name="crossval_jobs",
            index=False,
        )

    return summary_df