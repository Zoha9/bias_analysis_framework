"""Config-driven STAMP preprocessing and feature-extraction workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from bias_analysis_framework.stamp.config_writer import (
    build_stamp_preprocess_config,
    load_stamp_config_template,
    safe_config_name,
    write_stamp_config,
)
from bias_analysis_framework.stamp.runner import (
    StampCommandResult,
    build_stamp_environment,
    run_stamp_preprocess_command,
)


@dataclass(frozen=True)
class StampFeatureJob:
    """One STAMP feature-extraction job."""

    extractor: str
    wsi_dir: Path
    output_dir: Path
    config_path: Path
    log_path: Path
    max_workers: int | None


def run_stamp_feature_extraction(
    config: dict[str, Any],
    *,
    workflow_output_dir: str | Path,
) -> pd.DataFrame:
    """Prepare and optionally execute STAMP feature-extraction jobs.

    Parameters
    ----------
    config:
        Framework YAML configuration.
    workflow_output_dir:
        Output directory for generated STAMP configs, logs, and job summaries.

    Returns
    -------
    pandas.DataFrame
        One row per configured feature extractor.
    """
    workflow_output_dir = Path(workflow_output_dir).expanduser().resolve()
    workflow_output_dir.mkdir(parents=True, exist_ok=True)

    stamp_config = get_stamp_config(config)
    workflow_config = get_stamp_workflow_config(config)

    validate_stamp_feature_extraction_config(
        config=config,
        stamp_config=stamp_config,
        workflow_config=workflow_config,
    )

    jobs = build_stamp_feature_jobs(
        config=config,
        workflow_output_dir=workflow_output_dir,
    )

    template_path = Path(
        stamp_config["stamp_config_template"]
    ).expanduser().resolve()

    template_config = load_stamp_config_template(template_path)

    execute = bool(workflow_config.get("execute", False))
    skip_existing = bool(workflow_config.get("skip_existing", True))

    environment = (
        build_environment_from_config(stamp_config)
        if execute
        else None
    )

    job_rows: list[dict[str, Any]] = []

    for job in jobs:
        status = "pending"
        message = ""
        result: StampCommandResult | None = None

        existing_extractor_dir = find_existing_extractor_output(
            output_dir=job.output_dir,
            extractor=job.extractor,
        )

        if skip_existing and existing_extractor_dir is not None:
            status = "skipped_existing"
            message = (
                f"Existing extractor output found: "
                f"{existing_extractor_dir}"
            )

            job_rows.append(
                build_job_summary_row(
                    job=job,
                    status=status,
                    message=message,
                    execute=execute,
                    result=None,
                    existing_output=existing_extractor_dir,
                )
            )
            continue

        job.output_dir.mkdir(parents=True, exist_ok=True)

        generated_config = build_stamp_preprocess_config(
            template_config=template_config,
            wsi_dir=job.wsi_dir,
            output_dir=job.output_dir,
            extractor=job.extractor,
            cache_dir=get_optional_path(stamp_config, "cache_dir"),
            default_slide_mpp=get_default_slide_mpp(
                stamp_config=stamp_config,
                workflow_config=workflow_config,
            ),
            max_workers=job.max_workers,
            preprocessing_overrides=get_preprocessing_overrides(
                stamp_config=stamp_config,
                workflow_config=workflow_config,
                extractor=job.extractor,
            ),
        )

        write_stamp_config(
            config=generated_config,
            output_path=job.config_path,
        )

        try:
            result = run_stamp_preprocess_command(
                command=stamp_config.get("command", "stamp"),
                config_path=job.config_path,
                log_path=job.log_path,
                environment=environment,
                working_directory=stamp_config.get("working_directory"),
                execute=execute,
            )

            if result.executed:
                status = "completed"
                message = "STAMP preprocessing completed successfully."
            else:
                status = "dry_run"
                message = "STAMP command prepared but not executed."

        except Exception as error:
            status = "failed"
            message = f"{type(error).__name__}: {error}"

            job_rows.append(
                build_job_summary_row(
                    job=job,
                    status=status,
                    message=message,
                    execute=execute,
                    result=result,
                    existing_output=None,
                )
            )

            if not bool(workflow_config.get("continue_on_error", False)):
                summary_df = pd.DataFrame(job_rows)
                write_stamp_job_summary(
                    summary_df=summary_df,
                    output_path=workflow_output_dir / "stamp_feature_extraction_jobs.xlsx",
                )
                raise

            continue

        job_rows.append(
            build_job_summary_row(
                job=job,
                status=status,
                message=message,
                execute=execute,
                result=result,
                existing_output=None,
            )
        )

    summary_df = pd.DataFrame(job_rows)

    write_stamp_job_summary(
        summary_df=summary_df,
        output_path=workflow_output_dir / "tables" /"stamp_feature_extraction_jobs.xlsx",
    )

    return summary_df


def get_stamp_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return the top-level STAMP configuration section."""
    stamp_config = config.get("stamp", {})

    if not isinstance(stamp_config, dict):
        raise ValueError("The top-level 'stamp' config section must be a mapping.")

    return stamp_config


def get_stamp_workflow_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return the Step 2 workflow configuration."""
    workflows = config.get("workflows", {})

    if not isinstance(workflows, dict):
        raise ValueError("The top-level 'workflows' config section must be a mapping.")

    workflow_config = workflows.get("stamp_feature_extraction", {})

    if not isinstance(workflow_config, dict):
        raise ValueError(
            "'workflows.stamp_feature_extraction' must be a mapping."
        )

    return workflow_config


def validate_stamp_feature_extraction_config(
    *,
    config: dict[str, Any],
    stamp_config: dict[str, Any],
    workflow_config: dict[str, Any],
) -> None:
    """Validate required Step 2 configuration values."""
    data_config = config.get("data", {})

    if not isinstance(data_config, dict):
        raise ValueError("The top-level 'data' config section must be a mapping.")

    required_data_fields = [
        "wsi_root",
        "feature_root",
    ]

    missing_data_fields = [
        field
        for field in required_data_fields
        if not data_config.get(field)
    ]

    if missing_data_fields:
        raise ValueError(
            "Missing required Step 2 data config values: "
            + ", ".join(missing_data_fields)
        )

    if not stamp_config.get("stamp_config_template"):
        raise ValueError(
            "Missing required config value: stamp.stamp_config_template"
        )

    template_path = Path(
        stamp_config["stamp_config_template"]
    ).expanduser().resolve()

    if not template_path.exists():
        raise FileNotFoundError(
            f"STAMP config template does not exist: {template_path}"
        )

    wsi_root = Path(data_config["wsi_root"]).expanduser().resolve()

    if not wsi_root.exists():
        raise FileNotFoundError(f"WSI root directory does not exist: {wsi_root}")

    if not wsi_root.is_dir():
        raise ValueError(f"WSI root path is not a directory: {wsi_root}")

    models = get_configured_extractors(config)

    if not models:
        raise ValueError(
            "No feature extractors were configured. "
            "Add model names under the top-level 'models' list."
        )

    if "execute" in workflow_config and not isinstance(
        workflow_config["execute"],
        bool,
    ):
        raise ValueError(
            "workflows.stamp_feature_extraction.execute must be true or false."
        )


def get_configured_extractors(
    config: dict[str, Any],
) -> list[str]:
    """Return configured STAMP feature extractor names."""
    raw_models = config.get("models", [])

    if raw_models is None:
        return []

    if not isinstance(raw_models, list):
        raise ValueError("The top-level 'models' config value must be a list.")

    extractors: list[str] = []

    for model in raw_models:
        if isinstance(model, str):
            extractor = model.strip()
        elif isinstance(model, dict):
            extractor = str(
                model.get("extractor")
                or model.get("name")
                or ""
            ).strip()
        else:
            raise ValueError(
                "Each models entry must be a string or mapping."
            )

        if extractor and extractor not in extractors:
            extractors.append(extractor)

    return extractors


def build_stamp_feature_jobs(
    *,
    config: dict[str, Any],
    workflow_output_dir: str | Path,
) -> list[StampFeatureJob]:
    """Build one STAMP preprocessing job per configured extractor."""
    workflow_output_dir = Path(workflow_output_dir).expanduser().resolve()

    data_config = config.get("data", {})
    stamp_config = get_stamp_config(config)
    workflow_config = get_stamp_workflow_config(config)

    wsi_root = Path(data_config["wsi_root"]).expanduser().resolve()
    feature_root = Path(data_config["feature_root"]).expanduser().resolve()

    cohort = config.get("cohort", {})
    organ = safe_config_name(cohort.get("organ", "UNKNOWN"))
    dataset = safe_config_name(cohort.get("dataset", "DATASET"))

    config_dir = workflow_output_dir / "configs"
    log_dir = workflow_output_dir / "logs"

    config_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    feature_root.mkdir(parents=True, exist_ok=True)

    jobs: list[StampFeatureJob] = []

    for extractor in get_configured_extractors(config):
        safe_extractor = safe_config_name(extractor)

        output_dir = build_extractor_output_dir(
            feature_root=feature_root,
            organ=organ,
            dataset=dataset,
            extractor=extractor,
            workflow_config=workflow_config,
        )

        max_workers = get_max_workers_for_extractor(
            extractor=extractor,
            stamp_config=stamp_config,
            workflow_config=workflow_config,
        )

        jobs.append(
            StampFeatureJob(
                extractor=extractor,
                wsi_dir=wsi_root,
                output_dir=output_dir,
                config_path=(
                    config_dir
                    / f"stamp_preprocess__{organ}__{dataset}__{safe_extractor}.yaml"
                ),
                log_path=(
                    log_dir
                    / f"stamp_preprocess__{organ}__{dataset}__{safe_extractor}.log"
                ),
                max_workers=max_workers,
            )
        )

    return jobs


def build_extractor_output_dir(
    *,
    feature_root: Path,
    organ: str,
    dataset: str,
    extractor: str,
    workflow_config: dict[str, Any],
) -> Path:
    """Build the output directory for one feature extractor.

    By default, all STAMP extractor outputs are written below:

    feature_root / organ / dataset

    STAMP may create extractor-specific subdirectories inside this directory.
    """
    separate_extractor_directories = bool(
        workflow_config.get("separate_extractor_directories", False)
    )

    base_output_dir = feature_root / organ / dataset

    if separate_extractor_directories:
        return base_output_dir / safe_config_name(extractor)

    return base_output_dir


def get_max_workers_for_extractor(
    *,
    extractor: str,
    stamp_config: dict[str, Any],
    workflow_config: dict[str, Any],
) -> int | None:
    """Return configured worker count for one extractor."""
    extractor_workers = workflow_config.get("extractor_max_workers")

    if extractor_workers is None:
        extractor_workers = stamp_config.get("extractor_max_workers", {})

    if extractor_workers and not isinstance(extractor_workers, dict):
        raise ValueError(
            "extractor_max_workers must be a mapping of extractor names "
            "to worker counts."
        )

    if isinstance(extractor_workers, dict):
        value = extractor_workers.get(extractor)

        if value is not None:
            workers = int(value)

            if workers < 1:
                raise ValueError(
                    f"max_workers for extractor {extractor!r} must be at least 1."
                )

            return workers

    default_workers = workflow_config.get(
        "max_workers",
        stamp_config.get("max_workers"),
    )

    if default_workers is None:
        return None

    workers = int(default_workers)

    if workers < 1:
        raise ValueError("STAMP max_workers must be at least 1.")

    return workers


def get_default_slide_mpp(
    *,
    stamp_config: dict[str, Any],
    workflow_config: dict[str, Any],
) -> float | None:
    """Return the configured fallback slide MPP."""
    value = workflow_config.get(
        "default_slide_mpp",
        stamp_config.get("default_slide_mpp"),
    )

    if value is None:
        return None

    mpp = float(value)

    if mpp <= 0:
        raise ValueError("default_slide_mpp must be greater than zero.")

    return mpp


def get_preprocessing_overrides(
    *,
    stamp_config: dict[str, Any],
    workflow_config: dict[str, Any],
    extractor: str,
) -> dict[str, Any]:
    """Return global and extractor-specific preprocessing overrides."""
    output: dict[str, Any] = {}

    global_overrides = stamp_config.get("preprocessing_overrides", {})
    workflow_overrides = workflow_config.get("preprocessing_overrides", {})

    if global_overrides:
        if not isinstance(global_overrides, dict):
            raise ValueError(
                "stamp.preprocessing_overrides must be a mapping."
            )
        output.update(global_overrides)

    if workflow_overrides:
        if not isinstance(workflow_overrides, dict):
            raise ValueError(
                "workflows.stamp_feature_extraction.preprocessing_overrides "
                "must be a mapping."
            )
        output.update(workflow_overrides)

    extractor_overrides = workflow_config.get(
        "extractor_overrides",
        stamp_config.get("extractor_overrides", {}),
    )

    if extractor_overrides:
        if not isinstance(extractor_overrides, dict):
            raise ValueError(
                "extractor_overrides must be a mapping."
            )

        selected = extractor_overrides.get(extractor, {})

        if selected:
            if not isinstance(selected, dict):
                raise ValueError(
                    f"Overrides for extractor {extractor!r} must be a mapping."
                )
            output.update(selected)

    return output


def build_environment_from_config(
    stamp_config: dict[str, Any],
) -> dict[str, str]:
    """Build the environment used by STAMP subprocesses."""
    cache_environment: dict[str, str | Path] = {}

    cache_mapping = {
        "HF_HOME": "hf_home",
        "HUGGINGFACE_HUB_CACHE": "huggingface_hub_cache",
        "XDG_CACHE_HOME": "xdg_cache_home",
        "TORCH_HOME": "torch_home",
    }

    for environment_variable, config_key in cache_mapping.items():
        value = stamp_config.get(config_key)

        if value:
            cache_environment[environment_variable] = value

    extra_environment = stamp_config.get("environment", {})

    if extra_environment and not isinstance(extra_environment, dict):
        raise ValueError("stamp.environment must be a mapping.")

    return build_stamp_environment(
        token_file=stamp_config.get("hf_token_file"),
        cache_environment=cache_environment,
        extra_environment=extra_environment,
    )


def find_existing_extractor_output(
    *,
    output_dir: str | Path,
    extractor: str,
) -> Path | None:
    """Find an existing STAMP output directory for an extractor.

    This preserves the behavior of the old script, which considered an
    extractor complete when a directory beginning with ``extractor-`` existed.
    It also recognizes a directory matching the extractor name exactly.
    """
    output_dir = Path(output_dir).expanduser().resolve()

    if not output_dir.exists() or not output_dir.is_dir():
        return None

    extractor_name = str(extractor).strip()

    for child in output_dir.iterdir():
        if not child.is_dir():
            continue

        if child.name == extractor_name:
            return child

        if child.name.startswith(f"{extractor_name}-"):
            return child

    return None


def get_optional_path(
    config: dict[str, Any],
    key: str,
) -> Path | None:
    """Return an optional configured path."""
    value = config.get(key)

    if value in {None, ""}:
        return None

    return Path(value).expanduser().resolve()


def build_job_summary_row(
    *,
    job: StampFeatureJob,
    status: str,
    message: str,
    execute: bool,
    result: StampCommandResult | None,
    existing_output: Path | None,
) -> dict[str, Any]:
    """Build one summary row for a STAMP job."""
    return {
        "extractor": job.extractor,
        "status": status,
        "execute_requested": execute,
        "command_executed": bool(result.executed) if result else False,
        "return_code": result.return_code if result else pd.NA,
        "wsi_dir": str(job.wsi_dir),
        "output_dir": str(job.output_dir),
        "existing_output": (
            str(existing_output)
            if existing_output is not None
            else ""
        ),
        "generated_config": str(job.config_path),
        "log_file": str(job.log_path),
        "max_workers": (
            job.max_workers
            if job.max_workers is not None
            else pd.NA
        ),
        "message": message,
    }


def write_stamp_job_summary(
    *,
    summary_df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Write the STAMP feature-extraction job summary workbook."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(
            writer,
            sheet_name="feature_extraction_jobs",
            index=False,
        )

    return output_path