"""Config-driven regular STAMP model deployment."""

from __future__ import annotations

import shlex
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from bias_analysis_framework.io.stamp_outputs import (
    StampCheckpoint,
    count_prediction_files,
    discover_stamp_checkpoints,
)
from bias_analysis_framework.stamp.config_writer import (
    load_stamp_config_template,
    safe_config_name,
    write_stamp_config,
)
from bias_analysis_framework.stamp.crossval import (
    get_configured_models,
    get_training_labels,
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
class StampRegularDeploymentJob:
    """One regular STAMP deployment job."""

    model_name: str
    extractor: str
    label_name: str
    label_type: str
    split_number: int
    feature_dir: Path
    checkpoint_path: Path
    output_dir: Path
    config_path: Path
    log_path: Path


def run_regular_stamp_deployment(
    *,
    config: dict[str, Any],
    workflow_output_dir: str | Path,
) -> pd.DataFrame:
    """Prepare and optionally execute regular STAMP deployment jobs."""
    workflow_output_dir = (
        Path(workflow_output_dir)
        .expanduser()
        .resolve()
    )
    workflow_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    workflow_config = get_regular_deployment_config(
        config
    )
    stamp_config = get_stamp_config(config)

    validate_regular_deployment_config(
        config=config,
        workflow_config=workflow_config,
        stamp_config=stamp_config,
    )

    jobs = build_regular_deployment_jobs(
        config=config,
        workflow_output_dir=workflow_output_dir,
    )

    template_path = get_deployment_template_path(
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
        existing_prediction_files = (
            count_prediction_files(
                job.output_dir
            )
        )

        if (
            skip_existing
            and existing_prediction_files > 0
        ):
            rows.append(
                build_deployment_summary_row(
                    job=job,
                    status="skipped_existing",
                    executed=False,
                    return_code=pd.NA,
                    prediction_file_count=(
                        existing_prediction_files
                    ),
                    message=(
                        "Existing prediction output files "
                        "were found."
                    ),
                )
            )
            continue

        job.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        generated_config = (
            build_stamp_deployment_config(
                template_config=template_config,
                job=job,
                config=config,
                workflow_config=workflow_config,
            )
        )

        write_stamp_config(
            config=generated_config,
            output_path=job.config_path,
        )

        try:
            return_code = run_deployment_command(
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

            prediction_file_count = (
                count_prediction_files(
                    job.output_dir
                )
            )

            if not execute:
                status = "dry_run"
                message = (
                    "STAMP deployment config and command "
                    "were generated but not executed."
                )
            elif prediction_file_count > 0:
                status = "completed"
                message = (
                    f"Found {prediction_file_count} likely "
                    "prediction output file(s)."
                )
            else:
                status = "completed_with_warning"
                message = (
                    "STAMP returned successfully, but no "
                    "CSV, Excel, Parquet, or JSON prediction "
                    "files were found."
                )

            rows.append(
                build_deployment_summary_row(
                    job=job,
                    status=status,
                    executed=execute,
                    return_code=return_code,
                    prediction_file_count=(
                        prediction_file_count
                    ),
                    message=message,
                )
            )

        except Exception as error:
            rows.append(
                build_deployment_summary_row(
                    job=job,
                    status="failed",
                    executed=execute,
                    return_code=getattr(
                        error,
                        "returncode",
                        pd.NA,
                    ),
                    prediction_file_count=(
                        count_prediction_files(
                            job.output_dir
                        )
                    ),
                    message=(
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                )
            )

            write_regular_deployment_summary(
                rows=rows,
                output_path=(
                    workflow_output_dir
                    / "tables"
                    / "regular_deployment_jobs.xlsx"
                ),
            )

            if not continue_on_error:
                raise

    return write_regular_deployment_summary(
        rows=rows,
        output_path=(
            workflow_output_dir
            / "tables"
            / "regular_deployment_jobs.xlsx"
        ),
    )


def get_regular_deployment_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return the regular prediction workflow config."""
    value = (
        config.get("workflows", {})
        .get(
            "regular_model_prediction_analysis",
            {},
        )
    )

    if not isinstance(value, dict):
        raise ValueError(
            "workflows.regular_model_prediction_analysis "
            "must be a mapping."
        )

    return value


def validate_regular_deployment_config(
    *,
    config: dict[str, Any],
    workflow_config: dict[str, Any],
    stamp_config: dict[str, Any],
) -> None:
    """Validate regular deployment settings."""
    data_config = config.get("data", {})

    for key in (
        "clinical_table",
        "slide_table",
        "feature_root",
    ):
        if not data_config.get(key):
            raise ValueError(
                f"Missing required config value: "
                f"data.{key}"
            )

    clinical_path = (
        Path(data_config["clinical_table"])
        .expanduser()
        .resolve()
    )
    slide_path = (
        Path(data_config["slide_table"])
        .expanduser()
        .resolve()
    )
    feature_root = (
        Path(data_config["feature_root"])
        .expanduser()
        .resolve()
    )

    if not clinical_path.is_file():
        raise FileNotFoundError(
            f"Deployment clinical table does not "
            f"exist: {clinical_path}"
        )

    if not slide_path.is_file():
        raise FileNotFoundError(
            f"Deployment slide table does not "
            f"exist: {slide_path}"
        )

    if not feature_root.is_dir():
        raise FileNotFoundError(
            f"Deployment feature root does not "
            f"exist: {feature_root}"
        )

    crossval_root = get_crossval_results_root(
        config
    )

    if not crossval_root.is_dir():
        raise FileNotFoundError(
            f"Cross-validation results root does "
            f"not exist: {crossval_root}"
        )

    template_path = get_deployment_template_path(
        stamp_config
    )

    if not template_path.is_file():
        raise FileNotFoundError(
            f"STAMP deployment template does not "
            f"exist: {template_path}"
        )

    if not get_configured_models(config):
        raise ValueError(
            "No models are configured."
        )

    if not get_deployment_labels(config):
        raise ValueError(
            "No regular deployment labels are "
            "configured."
        )

    selected_splits = workflow_config.get(
        "splits"
    )

    if selected_splits is not None:
        if not isinstance(
            selected_splits,
            list,
        ):
            raise ValueError(
                "regular_model_prediction_analysis."
                "splits must be a list or null."
            )

        for split_number in selected_splits:
            if int(split_number) < 0:
                raise ValueError(
                    "Deployment split numbers must "
                    "be zero or greater."
                )


def get_deployment_template_path(
    stamp_config: dict[str, Any],
) -> Path:
    """Return the STAMP deployment template."""
    template_value = (
        stamp_config.get(
            "deployment_config_template"
        )
        or stamp_config.get(
            "stamp_config_template"
        )
    )

    if not template_value:
        raise ValueError(
            "Missing stamp.deployment_config_template."
        )

    return (
        Path(template_value)
        .expanduser()
        .resolve()
    )


def get_crossval_results_root(
    config: dict[str, Any],
) -> Path:
    """Return the root containing trained split checkpoints.

    This can point to:

    - Step 4 outputs for the same dataset;
    - Step 4 outputs from a different training dataset;
    - a legacy STAMP cross-validation result directory.
    """
    workflow_config = (
        get_regular_deployment_config(
            config
        )
    )

    configured_root = workflow_config.get(
        "crossval_results_root"
    )

    if configured_root:
        return (
            Path(configured_root)
            .expanduser()
            .resolve()
        )

    project_output_root = (
        Path(
            config.get(
                "project",
                {},
            ).get(
                "output_root",
                "outputs",
            )
        )
        .expanduser()
        .resolve()
    )

    training_cohort = workflow_config.get(
        "training_cohort",
        {},
    )

    organ = str(
        training_cohort.get(
            "organ",
            config.get(
                "cohort",
                {},
            ).get(
                "organ",
                "",
            ),
        )
    ).strip()

    dataset = str(
        training_cohort.get(
            "dataset",
            config.get(
                "cohort",
                {},
            ).get(
                "dataset",
                "",
            ),
        )
    ).strip()

    cohort_folder = f"{organ}__{dataset}"

    return (
        project_output_root
        / "stamp_crossval_training"
        / cohort_folder
        / "crossval_results"
    )


def get_deployment_labels(
    config: dict[str, Any],
) -> list[tuple[str, str]]:
    """Return labels selected for regular deployment."""
    workflow_config = (
        get_regular_deployment_config(
            config
        )
    )

    explicit_labels = workflow_config.get(
        "labels"
    )

    if explicit_labels:
        output: list[tuple[str, str]] = []

        for label in explicit_labels:
            if isinstance(label, str):
                pair = (
                    label,
                    "explicit",
                )
            elif isinstance(label, dict):
                label_name = resolve_explicit_label_name(
                    label=label,
                    config=config,
                )
                label_type = str(
                    label.get(
                        "type",
                        "explicit",
                    )
                ).strip()

                if not label_name:
                    continue

                pair = (
                    label_name,
                    label_type,
                )
            else:
                raise ValueError(
                    "Deployment labels must be "
                    "strings or mappings."
                )

            if pair not in output:
                output.append(pair)

        return output

    return get_training_labels(config)


def resolve_explicit_label_name(
    *,
    label: dict[str, Any],
    config: dict[str, Any],
) -> str:
    """Resolve an explicit display label to its clinical column."""
    requested_name = str(
        label.get("name", "")
    ).strip()

    label_type = str(
        label.get(
            "type",
            "explicit",
        )
    ).strip()

    if (
        label_type == "bias_factor"
        and requested_name
    ):
        for factor in config.get(
            "bias_factors",
            [],
        ):
            display_name = str(
                factor.get(
                    "name",
                    "",
                )
            ).strip()

            column_name = str(
                factor.get("column")
                or display_name
            ).strip()

            if requested_name in {
                display_name,
                column_name,
            }:
                return column_name

    return requested_name


def build_regular_deployment_jobs(
    *,
    config: dict[str, Any],
    workflow_output_dir: Path,
) -> list[StampRegularDeploymentJob]:
    """Build model × label × split deployment jobs."""
    workflow_config = (
        get_regular_deployment_config(
            config
        )
    )
    data_config = config["data"]
    cohort = config.get("cohort", {})

    feature_root = (
        Path(data_config["feature_root"])
        .expanduser()
        .resolve()
    )

    deployment_organ = str(
        cohort.get(
            "organ",
            "",
        )
    ).strip()
    deployment_dataset = str(
        cohort.get(
            "dataset",
            "",
        )
    ).strip()

    crossval_results_root = (
        get_crossval_results_root(
            config
        )
    )

    selected_splits = workflow_config.get(
        "splits"
    )

    if selected_splits is not None:
        selected_split_set = {
            int(value)
            for value in selected_splits
        }
    else:
        selected_split_set = None

    config_dir = (
        workflow_output_dir
        / "configs"
    )
    log_dir = (
        workflow_output_dir
        / "logs"
    )
    prediction_root = (
        workflow_output_dir
        / "predictions"
    )

    for directory in (
        config_dir,
        log_dir,
        prediction_root,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    jobs: list[
        StampRegularDeploymentJob
    ] = []

    for model_name, extractor in (
        get_configured_models(config)
    ):
        feature_dir = (
            discover_model_feature_directory(
                feature_root=feature_root,
                model_name=extractor,
                organ=deployment_organ,
                dataset=deployment_dataset,
            )
        )

        for label_name, label_type in (
            get_deployment_labels(config)
        ):
            checkpoints = (
                discover_stamp_checkpoints(
                    crossval_results_root=(
                        crossval_results_root
                    ),
                    label_name=label_name,
                    model_name=extractor,
                    checkpoint_filename=str(
                        workflow_config.get(
                            "checkpoint_filename",
                            "model.ckpt",
                        )
                    ),
                )
            )

            if selected_split_set is not None:
                checkpoints = [
                    checkpoint
                    for checkpoint in checkpoints
                    if checkpoint.split_number
                    in selected_split_set
                ]

            if not checkpoints:
                if bool(
                    workflow_config.get(
                        "require_all_checkpoints",
                        True,
                    )
                ):
                    raise FileNotFoundError(
                        "No checkpoints were found for "
                        f"model={extractor!r}, "
                        f"label={label_name!r} under "
                        f"{crossval_results_root}."
                    )

                continue

            for checkpoint in checkpoints:
                jobs.append(
                    build_regular_deployment_job(
                        workflow_output_dir=(
                            workflow_output_dir
                        ),
                        prediction_root=(
                            prediction_root
                        ),
                        model_name=model_name,
                        extractor=extractor,
                        label_name=label_name,
                        label_type=label_type,
                        feature_dir=feature_dir,
                        checkpoint=checkpoint,
                    )
                )

    return jobs


def build_regular_deployment_job(
    *,
    workflow_output_dir: Path,
    prediction_root: Path,
    model_name: str,
    extractor: str,
    label_name: str,
    label_type: str,
    feature_dir: Path,
    checkpoint: StampCheckpoint,
) -> StampRegularDeploymentJob:
    """Build one deployment job and its paths."""
    safe_model = safe_config_name(
        extractor
    )
    safe_label = safe_config_name(
        label_name
    )
    safe_split = safe_config_name(
        checkpoint.split_name
    )

    output_dir = (
        prediction_root
        / safe_label
        / safe_model
        / safe_split
    )

    config_path = (
        workflow_output_dir
        / "configs"
        / (
            "stamp_deploy__"
            f"{safe_model}__"
            f"{safe_label}__"
            f"{safe_split}.yaml"
        )
    )

    log_path = (
        workflow_output_dir
        / "logs"
        / (
            "stamp_deploy__"
            f"{safe_model}__"
            f"{safe_label}__"
            f"{safe_split}.log"
        )
    )

    return StampRegularDeploymentJob(
        model_name=model_name,
        extractor=extractor,
        label_name=label_name,
        label_type=label_type,
        split_number=checkpoint.split_number,
        feature_dir=feature_dir,
        checkpoint_path=(
            checkpoint.checkpoint_path
        ),
        output_dir=output_dir,
        config_path=config_path,
        log_path=log_path,
    )


def build_stamp_deployment_config(
    *,
    template_config: dict[str, Any],
    job: StampRegularDeploymentJob,
    config: dict[str, Any],
    workflow_config: dict[str, Any],
) -> dict[str, Any]:
    """Create one STAMP regular deployment config."""
    generated = deepcopy(
        template_config
    )

    deployment = generated.setdefault(
        "deployment",
        {},
    )

    data_config = config["data"]

    deployment["output_dir"] = str(
        job.output_dir
    )
    deployment["feature_dir"] = str(
        job.feature_dir
    )
    deployment["clini_table"] = str(
        Path(
            data_config["clinical_table"]
        ).expanduser().resolve()
    )
    deployment["slide_table"] = str(
        Path(
            data_config["slide_table"]
        ).expanduser().resolve()
    )
    deployment["ground_truth_label"] = (
        job.label_name
    )
    deployment["patient_label"] = str(
        data_config.get(
            "patient_column",
            "PATIENT",
        )
    )
    deployment["filename_label"] = str(
        data_config.get(
            "slide_filename_column",
            "FILENAME",
        )
    )
    deployment["checkpoint_paths"] = [
        str(job.checkpoint_path)
    ]

    overrides = workflow_config.get(
        "deployment_overrides",
        {},
    )

    if overrides:
        if not isinstance(overrides, dict):
            raise ValueError(
                "deployment_overrides must be "
                "a mapping."
            )

        deployment.update(overrides)

    return generated


def run_deployment_command(
    *,
    command: str | list[str],
    config_path: Path,
    log_path: Path,
    environment: dict[str, str] | None,
    working_directory: str | Path | None,
    execute: bool,
) -> int:
    """Run or preview one STAMP deploy command."""
    command_args = [
        *normalize_stamp_command(command),
        "--config",
        str(config_path),
        "deploy",
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


def build_deployment_summary_row(
    *,
    job: StampRegularDeploymentJob,
    status: str,
    executed: bool,
    return_code: Any,
    prediction_file_count: int,
    message: str,
) -> dict[str, Any]:
    """Build one deployment job summary row."""
    return {
        "Model": job.model_name,
        "Extractor": job.extractor,
        "Label": job.label_name,
        "LabelType": job.label_type,
        "Split": job.split_number,
        "Status": status,
        "Executed": executed,
        "ReturnCode": return_code,
        "FeatureDir": str(
            job.feature_dir
        ),
        "FeatureFileCount": count_feature_files(
            job.feature_dir
        ),
        "CheckpointPath": str(
            job.checkpoint_path
        ),
        "OutputDir": str(
            job.output_dir
        ),
        "PredictionFileCount": (
            prediction_file_count
        ),
        "GeneratedConfig": str(
            job.config_path
        ),
        "LogFile": str(
            job.log_path
        ),
        "Message": message,
    }


def write_regular_deployment_summary(
    *,
    rows: list[dict[str, Any]],
    output_path: str | Path,
) -> pd.DataFrame:
    """Write the regular deployment job workbook."""
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
            sheet_name="deployment_jobs",
            index=False,
        )

    return summary_df