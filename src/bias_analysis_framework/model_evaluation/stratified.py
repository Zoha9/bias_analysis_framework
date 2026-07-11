"""Stratified subgroup evaluation of regular STAMP predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from bias_analysis_framework.io.predictions import (
    choose_prediction_file,
    discover_prediction_files,
    discover_probability_columns,
    read_prediction_table,
    validate_prediction_table,
)
from bias_analysis_framework.io.stamp_outputs import (
    normalize_directory_name,
)
from bias_analysis_framework.io.tables import (
    read_table,
)
from bias_analysis_framework.metrics.model_performance_metrics import (
    compute_prediction_metrics,
)
from bias_analysis_framework.metrics.subgroup_performance_metrics import (
    aggregate_subgroup_split_metrics,
    build_pairwise_subgroup_gaps,
    build_subgroup_disparity_summary,
)
from bias_analysis_framework.model_evaluation.regular import (
    resolve_prediction_label,
    resolve_prediction_model,
)


@dataclass(frozen=True)
class StratifiedPerformanceResult:
    """All tables from subgroup performance analysis."""

    subgroup_split_metrics: pd.DataFrame
    subgroup_aggregate_metrics: pd.DataFrame
    disparity_summary: pd.DataFrame
    pairwise_gaps: pd.DataFrame
    subgroup_counts: pd.DataFrame
    file_status: pd.DataFrame


def analyze_stratified_predictions(
    *,
    config: dict[str, Any],
    predictions_root: str | Path,
) -> StratifiedPerformanceResult:
    """Evaluate regular predictions within configured clinical subgroups."""
    clinical_df = load_stratification_clinical_table(
        config
    )

    patient_column = str(
        config.get(
            "data",
            {},
        ).get(
            "patient_column",
            "PATIENT",
        )
    )

    if patient_column not in clinical_df.columns:
        raise ValueError(
            f"Clinical table does not contain patient column "
            f"{patient_column!r}."
        )

    if clinical_df[
        patient_column
    ].duplicated().any():
        duplicated = (
            clinical_df.loc[
                clinical_df[
                    patient_column
                ].duplicated(),
                patient_column,
            ]
            .astype(str)
            .head(10)
            .tolist()
        )

        raise ValueError(
            "Clinical table contains duplicate patient IDs. "
            f"Examples: {duplicated}"
        )

    prediction_files = choose_prediction_file(
        discover_prediction_files(
            predictions_root=predictions_root,
        )
    )

    if not prediction_files:
        return empty_stratified_result(
            message=(
                "No patient-preds.csv or "
                "patient-preds-0.csv files were found."
            )
        )

    subgroup_specs = get_subgroup_specs(
        config
    )

    if not subgroup_specs:
        return empty_stratified_result(
            message=(
                "No configured bias factors are available "
                "for subgroup evaluation."
            )
        )

    workflow_config = get_stratified_workflow_config(
        config
    )

    minimum_subgroup_size = int(
        workflow_config.get(
            "minimum_subgroup_size",
            5,
        )
    )

    subgroup_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []

    for prediction_file in prediction_files:
        label_spec = resolve_prediction_label(
            directory_label=(
                prediction_file.label_name
            ),
            config=config,
        )

        if (
            label_spec["type"] != "target"
            and not bool(
                workflow_config.get(
                    "evaluate_bias_factor_models",
                    False,
                )
            )
        ):
            continue

        target_column = label_spec[
            "column"
        ]
        target_name = label_spec[
            "display_name"
        ]

        model_spec = resolve_prediction_model(
            directory_model=(
                prediction_file.model_name
            ),
            config=config,
        )

        try:
            prediction_df = read_prediction_table(
                prediction_file.path
            )

            validate_prediction_table(
                dataframe=prediction_df,
                target_column=target_column,
            )

            prediction_patient_column = (
                resolve_prediction_patient_column(
                    prediction_df=prediction_df,
                    configured_patient_column=(
                        patient_column
                    ),
                )
            )

            merged = prediction_df.merge(
                clinical_df,
                left_on=prediction_patient_column,
                right_on=patient_column,
                how="left",
                validate="many_to_one",
                suffixes=(
                    "",
                    "__clinical",
                ),
                indicator="__clinical_merge_status",
            )

            missing_clinical = int(
                (
                    merged["__clinical_merge_status"]
                    != "both"
                ).sum()
            )

            merged.drop(
                columns=["__clinical_merge_status"],
                inplace=True,
            )

            probability_columns = (
                discover_probability_columns(
                    dataframe=merged,
                    target_column=target_column,
                )
            )

            for subgroup_spec in subgroup_specs:
                factor_name = subgroup_spec[
                    "display_name"
                ]
                factor_column = subgroup_spec[
                    "column"
                ]

                resolved_factor_column = (
                    resolve_merged_clinical_column(
                        dataframe=merged,
                        requested_column=factor_column,
                    )
                )

                if resolved_factor_column is None:
                    status_rows.append(
                        {
                            "Target": target_name,
                            "Model": model_spec[
                                "display_name"
                            ],
                            "Split": (
                                prediction_file.split_number
                            ),
                            "BiasFactor": factor_name,
                            "BiasFactorColumn": factor_column,
                            "PredictionFile": str(
                                prediction_file.path
                            ),
                            "Status": (
                                "missing_bias_column"
                            ),
                            "Message": (
                                f"Column {factor_column!r} "
                                "was not found after merging."
                            ),
                        }
                    )
                    continue

                factor_values = clean_subgroup_series(
                    merged[
                        resolved_factor_column
                    ]
                )

                configured_subgroups = subgroup_spec.get(
                    "subgroups"
                )

                observed_subgroups = (
                    get_subgroups_to_evaluate(
                        factor_values=factor_values,
                        configured_subgroups=(
                            configured_subgroups
                        ),
                    )
                )

                for subgroup in observed_subgroups:
                    subgroup_mask = (
                        factor_values.str.casefold()
                        == str(subgroup).strip().casefold()
                    )

                    subgroup_df = merged.loc[
                        subgroup_mask
                    ].copy()

                    sample_count = len(
                        subgroup_df
                    )

                    count_rows.append(
                        {
                            "Target": target_name,
                            "TargetColumn": target_column,
                            "Model": model_spec[
                                "display_name"
                            ],
                            "Extractor": model_spec[
                                "extractor"
                            ],
                            "Split": (
                                prediction_file.split_number
                            ),
                            "BiasFactor": factor_name,
                            "BiasFactorColumn": factor_column,
                            "Subgroup": subgroup,
                            "Num_Patients": sample_count,
                        }
                    )

                    if (
                        sample_count
                        < minimum_subgroup_size
                    ):
                        status_rows.append(
                            {
                                "Target": target_name,
                                "Model": model_spec[
                                    "display_name"
                                ],
                                "Split": (
                                    prediction_file.split_number
                                ),
                                "BiasFactor": factor_name,
                                "BiasFactorColumn": (
                                    factor_column
                                ),
                                "Subgroup": subgroup,
                                "PredictionFile": str(
                                    prediction_file.path
                                ),
                                "Status": (
                                    "insufficient_subgroup_size"
                                ),
                                "Message": (
                                    f"Found {sample_count} patients; "
                                    f"minimum is "
                                    f"{minimum_subgroup_size}."
                                ),
                            }
                        )
                        continue

                    try:
                        metrics, _ = (
                            compute_prediction_metrics(
                                dataframe=subgroup_df,
                                target_column=(
                                    target_column
                                ),
                                probability_columns=(
                                    probability_columns
                                ),
                                positive_label=(
                                    label_spec.get(
                                        "positive_label"
                                    )
                                ),
                            )
                        )

                        subgroup_rows.append(
                            {
                                "Target": target_name,
                                "TargetColumn": (
                                    target_column
                                ),
                                "Model": model_spec[
                                    "display_name"
                                ],
                                "Extractor": model_spec[
                                    "extractor"
                                ],
                                "Split": (
                                    prediction_file.split_number
                                ),
                                "BiasFactor": factor_name,
                                "BiasFactorColumn": (
                                    factor_column
                                ),
                                "Subgroup": subgroup,
                                "PredictionFile": str(
                                    prediction_file.path
                                ),
                                **metrics,
                            }
                        )

                        status_rows.append(
                            {
                                "Target": target_name,
                                "Model": model_spec[
                                    "display_name"
                                ],
                                "Split": (
                                    prediction_file.split_number
                                ),
                                "BiasFactor": factor_name,
                                "BiasFactorColumn": (
                                    factor_column
                                ),
                                "Subgroup": subgroup,
                                "PredictionFile": str(
                                    prediction_file.path
                                ),
                                "Status": "completed",
                                "Message": "",
                            }
                        )

                    except Exception as metric_error:
                        status_rows.append(
                            {
                                "Target": target_name,
                                "Model": model_spec[
                                    "display_name"
                                ],
                                "Split": (
                                    prediction_file.split_number
                                ),
                                "BiasFactor": factor_name,
                                "BiasFactorColumn": (
                                    factor_column
                                ),
                                "Subgroup": subgroup,
                                "PredictionFile": str(
                                    prediction_file.path
                                ),
                                "Status": (
                                    "metric_failure"
                                ),
                                "Message": (
                                    f"{type(metric_error).__name__}: "
                                    f"{metric_error}"
                                ),
                            }
                        )

            status_rows.append(
                {
                    "Target": target_name,
                    "Model": model_spec[
                        "display_name"
                    ],
                    "Split": (
                        prediction_file.split_number
                    ),
                    "BiasFactor": pd.NA,
                    "BiasFactorColumn": pd.NA,
                    "Subgroup": pd.NA,
                    "PredictionFile": str(
                        prediction_file.path
                    ),
                    "Status": "file_completed",
                    "Message": (
                        f"Rows={len(prediction_df)}; "
                        f"missing clinical matches="
                        f"{missing_clinical}."
                    ),
                }
            )

        except Exception as file_error:
            status_rows.append(
                {
                    "Target": target_name,
                    "Model": model_spec[
                        "display_name"
                    ],
                    "Split": (
                        prediction_file.split_number
                    ),
                    "BiasFactor": pd.NA,
                    "BiasFactorColumn": pd.NA,
                    "Subgroup": pd.NA,
                    "PredictionFile": str(
                        prediction_file.path
                    ),
                    "Status": "file_failure",
                    "Message": (
                        f"{type(file_error).__name__}: "
                        f"{file_error}"
                    ),
                }
            )

    subgroup_split_metrics = pd.DataFrame(
        subgroup_rows
    )

    subgroup_aggregate_metrics = (
        aggregate_subgroup_split_metrics(
            subgroup_split_metrics
        )
    )

    disparity_summary = (
        build_subgroup_disparity_summary(
            subgroup_aggregate_metrics
        )
    )

    pairwise_gaps = (
        build_pairwise_subgroup_gaps(
            subgroup_aggregate_metrics
        )
    )

    return StratifiedPerformanceResult(
        subgroup_split_metrics=(
            subgroup_split_metrics
        ),
        subgroup_aggregate_metrics=(
            subgroup_aggregate_metrics
        ),
        disparity_summary=disparity_summary,
        pairwise_gaps=pairwise_gaps,
        subgroup_counts=pd.DataFrame(
            count_rows
        ),
        file_status=pd.DataFrame(
            status_rows
        ),
    )


def get_stratified_workflow_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return Step 6 workflow config."""
    value = (
        config.get("workflows", {})
        .get(
            "stratified_subgroup_performance_analysis",
            {},
        )
    )

    if not isinstance(value, dict):
        raise ValueError(
            "workflows."
            "stratified_subgroup_performance_analysis "
            "must be a mapping."
        )

    return value


def get_subgroup_specs(
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return subgroup factors selected for Step 6."""
    workflow_config = get_stratified_workflow_config(
        config
    )

    selected_names = workflow_config.get(
        "bias_factors"
    )

    selected_normalized = (
        {
            normalize_directory_name(
                value
            )
            for value in selected_names
        }
        if selected_names
        else None
    )

    output: list[dict[str, Any]] = []

    for factor in config.get(
        "bias_factors",
        [],
    ):
        display_name = str(
            factor.get("name")
            or factor.get("column")
            or ""
        ).strip()

        column_name = str(
            factor.get("column")
            or display_name
        ).strip()

        if not display_name or not column_name:
            continue

        if selected_normalized is not None:
            if not (
                {
                    normalize_directory_name(
                        display_name
                    ),
                    normalize_directory_name(
                        column_name
                    ),
                }
                & selected_normalized
            ):
                continue

        output.append(
            {
                "display_name": display_name,
                "column": column_name,
                "subgroups": (
                    [
                        str(value)
                        for value in factor.get(
                            "subgroups",
                            [],
                        )
                    ]
                    or None
                ),
            }
        )

    return output


def load_stratification_clinical_table(
    config: dict[str, Any],
) -> pd.DataFrame:
    """Load the clinical table used for subgroup membership."""
    workflow_config = get_stratified_workflow_config(
        config
    )

    configured_path = workflow_config.get(
        "clinical_table"
    )

    if configured_path:
        path = Path(
            configured_path
        ).expanduser().resolve()
    else:
        path = Path(
            config["data"]["clinical_table"]
        ).expanduser().resolve()

    return read_table(path)


def resolve_prediction_patient_column(
    *,
    prediction_df: pd.DataFrame,
    configured_patient_column: str,
) -> str:
    """Resolve the patient ID column in a STAMP prediction file."""
    candidates = [
        configured_patient_column,
        "PATIENT",
        "patient",
        "Patient",
        "patient_id",
        "Patient_ID",
    ]

    for candidate in candidates:
        if candidate in prediction_df.columns:
            return candidate

    raise ValueError(
        "Prediction file does not contain a recognizable "
        "patient ID column. Available columns: "
        f"{list(prediction_df.columns)}"
    )


def resolve_merged_clinical_column(
    *,
    dataframe: pd.DataFrame,
    requested_column: str,
) -> str | None:
    """Resolve clinical columns after merge suffixes are applied."""
    candidates = [
        requested_column,
        f"{requested_column}__clinical",
    ]

    requested_casefold = requested_column.casefold()

    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    for column in dataframe.columns:
        plain_column = str(column).removesuffix(
            "__clinical"
        )

        if (
            plain_column.strip().casefold()
            == requested_casefold
        ):
            return str(column)

    return None


def get_subgroups_to_evaluate(
    *,
    factor_values: pd.Series,
    configured_subgroups: list[str] | None,
) -> list[str]:
    """Return configured or observed subgroup values."""
    observed = (
        factor_values
        .dropna()
        .astype(str)
        .value_counts()
        .index
        .tolist()
    )

    if not configured_subgroups:
        return observed

    observed_casefold = {
        value.casefold(): value
        for value in observed
    }

    output: list[str] = []

    for configured in configured_subgroups:
        matched = observed_casefold.get(
            str(configured).strip().casefold()
        )

        if matched is not None:
            output.append(matched)

    return output


def clean_subgroup_series(
    series: pd.Series,
) -> pd.Series:
    """Clean subgroup labels and convert NA-like values to missing."""
    output = (
        series.astype("string")
        .str.strip()
    )

    unknown_values = {
        "",
        "na",
        "nan",
        "n/a",
        "none",
        "null",
        "unknown",
        "<na>",
    }

    return output.mask(
        output.isna()
        | output.str.casefold().isin(
            unknown_values
        )
    )


def write_stratified_performance_workbook(
    *,
    result: StratifiedPerformanceResult,
    output_path: str | Path,
) -> Path:
    """Write all Step 6 result tables."""
    output_path = Path(
        output_path
    ).expanduser().resolve()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:
        result.subgroup_aggregate_metrics.to_excel(
            writer,
            sheet_name="subgroup_aggregate",
            index=False,
        )
        result.subgroup_split_metrics.to_excel(
            writer,
            sheet_name="subgroup_by_split",
            index=False,
        )
        result.disparity_summary.to_excel(
            writer,
            sheet_name="disparity_summary",
            index=False,
        )
        result.pairwise_gaps.to_excel(
            writer,
            sheet_name="pairwise_gaps",
            index=False,
        )
        result.subgroup_counts.to_excel(
            writer,
            sheet_name="subgroup_counts",
            index=False,
        )
        result.file_status.to_excel(
            writer,
            sheet_name="file_status",
            index=False,
        )

    return output_path


def empty_stratified_result(
    *,
    message: str,
) -> StratifiedPerformanceResult:
    """Return an empty Step 6 result with a diagnostic message."""
    return StratifiedPerformanceResult(
        subgroup_split_metrics=pd.DataFrame(),
        subgroup_aggregate_metrics=pd.DataFrame(),
        disparity_summary=pd.DataFrame(),
        pairwise_gaps=pd.DataFrame(),
        subgroup_counts=pd.DataFrame(),
        file_status=pd.DataFrame(
            [
                {
                    "Status": "no_results",
                    "Message": message,
                }
            ]
        ),
    )