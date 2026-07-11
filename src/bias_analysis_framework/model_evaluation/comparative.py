"""Comparative pretraining and posttraining bias analysis."""

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
from bias_analysis_framework.io.tables import (
    read_table,
)
from bias_analysis_framework.metrics.posttraining_bias_metrics import (
    compute_bias_propagation_metrics,
)
from bias_analysis_framework.model_evaluation.regular import (
    resolve_prediction_label,
    resolve_prediction_model,
)
from bias_analysis_framework.model_evaluation.stratified import (
    clean_subgroup_series,
    get_subgroup_specs,
    resolve_merged_clinical_column,
    resolve_prediction_patient_column,
)


@dataclass(frozen=True)
class ComparativeBiasResult:
    """Tables produced by comparative bias analysis."""

    split_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    amplification_summary: pd.DataFrame
    file_status: pd.DataFrame


def analyze_comparative_bias(
    *,
    config: dict[str, Any],
    predictions_root: str | Path,
) -> ComparativeBiasResult:
    """Compare ground-truth and prediction disparities."""
    workflow_config = get_comparative_workflow_config(
        config
    )

    clinical_path = workflow_config.get(
        "clinical_table"
    ) or config["data"]["clinical_table"]

    clinical_df = read_table(
        Path(clinical_path)
        .expanduser()
        .resolve()
    )

    patient_column = str(
        config.get("data", {}).get(
            "patient_column",
            "PATIENT",
        )
    )

    if patient_column not in clinical_df.columns:
        raise ValueError(
            f"Clinical table does not contain "
            f"{patient_column!r}."
        )

    if clinical_df[
        patient_column
    ].duplicated().any():
        raise ValueError(
            "Clinical table contains duplicate patient IDs."
        )

    subgroup_specs = get_subgroup_specs(
        config
    )

    prediction_files = choose_prediction_file(
        discover_prediction_files(
            predictions_root=predictions_root,
        )
    )

    minimum_group_size = int(
        workflow_config.get(
            "minimum_group_size",
            5,
        )
    )
    laplace_alpha = float(
        workflow_config.get(
            "laplace_alpha",
            1.0,
        )
    )

    split_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []

    for prediction_file in prediction_files:
        label_spec = resolve_prediction_label(
            directory_label=(
                prediction_file.label_name
            ),
            config=config,
        )

        if label_spec["type"] != "target":
            continue

        target_name = label_spec[
            "display_name"
        ]
        target_column = label_spec[
            "column"
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

            merged = merged[
                merged["__clinical_merge_status"]
                == "both"
            ].copy()

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

            target_classes = resolve_target_classes(
                dataframe=merged,
                target_column=target_column,
                label_spec=label_spec,
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
                            "Status": (
                                "missing_bias_column"
                            ),
                            "Message": (
                                f"Column {factor_column!r} "
                                "was not found."
                            ),
                        }
                    )
                    continue

                merged[
                    resolved_factor_column
                ] = clean_subgroup_series(
                    merged[
                        resolved_factor_column
                    ]
                )

                for positive_class in target_classes:
                    probability_column = (
                        resolve_probability_column(
                            probability_columns=(
                                probability_columns
                            ),
                            positive_class=(
                                positive_class
                            ),
                        )
                    )

                    metrics = (
                        compute_bias_propagation_metrics(
                            dataframe=merged,
                            target_column=target_column,
                            prediction_column="pred",
                            subgroup_column=(
                                resolved_factor_column
                            ),
                            positive_class=positive_class,
                            probability_column=(
                                probability_column
                            ),
                            minimum_group_size=(
                                minimum_group_size
                            ),
                            laplace_alpha=laplace_alpha,
                        )
                    )

                    split_rows.append(
                        {
                            "Target": target_name,
                            "TargetColumn": target_column,
                            "TargetClass": positive_class,
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
                    "BiasFactor": pd.NA,
                    "Status": "file_completed",
                    "Message": (
                        f"Rows={len(prediction_df)}; "
                        f"unmatched clinical patients="
                        f"{missing_clinical}."
                    ),
                }
            )

        except Exception as error:
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
                    "Status": "file_failure",
                    "Message": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                }
            )

    split_metrics = pd.DataFrame(
        split_rows
    )

    aggregate_metrics = (
        aggregate_comparative_metrics(
            split_metrics
        )
    )

    amplification_summary = (
        build_amplification_summary(
            aggregate_metrics
        )
    )

    return ComparativeBiasResult(
        split_metrics=split_metrics,
        aggregate_metrics=aggregate_metrics,
        amplification_summary=(
            amplification_summary
        ),
        file_status=pd.DataFrame(
            status_rows
        ),
    )


def get_comparative_workflow_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return Step 7 workflow settings."""
    value = (
        config.get("workflows", {})
        .get(
            "comparative_bias_propagation_analysis",
            {},
        )
    )

    if not isinstance(value, dict):
        raise ValueError(
            "comparative_bias_propagation_analysis "
            "must be a mapping."
        )

    return value

def get_comparative_subgroup_specs(
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return bias factors selected for comparative analysis."""
    workflow_config = get_comparative_workflow_config(
        config
    )
    selected = workflow_config.get(
        "bias_factors"
    )

    selected_names = (
        {
            str(value).strip().casefold()
            for value in selected
        }
        if selected
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

        if selected_names is not None:
            if (
                display_name.casefold()
                not in selected_names
                and column_name.casefold()
                not in selected_names
            ):
                continue

        output.append(
            {
                "display_name": display_name,
                "column": column_name,
                "subgroups": factor.get(
                    "subgroups"
                ),
            }
        )

    return output

def resolve_target_classes(
    *,
    dataframe: pd.DataFrame,
    target_column: str,
    label_spec: dict[str, Any],
) -> list[str]:
    """Resolve binary or one-vs-rest target classes."""
    configured_positive = label_spec.get(
        "positive_label"
    )

    if configured_positive is not None:
        observed = (
            dataframe[target_column]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
            .tolist()
        )

        if len(observed) <= 2:
            return [
                str(configured_positive)
            ]

    return (
        dataframe[target_column]
        .dropna()
        .astype(str)
        .str.strip()
        .value_counts()
        .index
        .tolist()
    )


def resolve_probability_column(
    *,
    probability_columns: dict[str, str],
    positive_class: str,
) -> str | None:
    """Resolve a class probability column case-insensitively."""
    positive_casefold = (
        str(positive_class)
        .strip()
        .casefold()
    )

    for class_name, column_name in (
        probability_columns.items()
    ):
        if (
            str(class_name)
            .strip()
            .casefold()
            == positive_casefold
        ):
            return column_name

    return None


def aggregate_comparative_metrics(
    split_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate comparative metrics across splits."""
    if split_metrics.empty:
        return pd.DataFrame()

    group_columns = [
        "Target",
        "TargetColumn",
        "TargetClass",
        "Model",
        "Extractor",
        "BiasFactor",
        "BiasFactorColumn",
    ]

    metric_columns = [
        "Pretraining_PositiveRateGap",
        "Posttraining_PositiveRateGap",
        "Delta_PositiveRateGap",
        "Pretraining_DI",
        "Posttraining_DI",
        "Pretraining_DI_Distance",
        "Posttraining_DI_Distance",
        "Delta_DI_Distance",
        "AccuracyGap",
        "RecallGap",
        "AUCGap",
    ]

    rows: list[dict[str, Any]] = []

    for group_values, group_df in (
        split_metrics.groupby(
            group_columns,
            dropna=False,
        )
    ):
        row = dict(
            zip(
                group_columns,
                group_values,
            )
        )

        row["NumSplits"] = int(
            group_df["Split"].nunique()
        )
        row["NumPatients_mean"] = (
            pd.to_numeric(
                group_df["NumPatients"],
                errors="coerce",
            ).mean()
        )
        row["NumGroups_mean"] = (
            pd.to_numeric(
                group_df["NumGroups"],
                errors="coerce",
            ).mean()
        )

        for metric in metric_columns:
            values = pd.to_numeric(
                group_df[metric],
                errors="coerce",
            )

            row[f"{metric}_mean"] = (
                values.mean()
            )
            row[f"{metric}_std"] = (
                values.std(ddof=1)
            )
            row[f"{metric}_count"] = int(
                values.notna().sum()
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_amplification_summary(
    aggregate_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Add readable amplification classifications."""
    if aggregate_metrics.empty:
        return pd.DataFrame()

    output = aggregate_metrics.copy()

    output[
        "PositiveRateGapInterpretation"
    ] = output[
        "Delta_PositiveRateGap_mean"
    ].map(classify_delta)

    output[
        "DIInterpretation"
    ] = output[
        "Delta_DI_Distance_mean"
    ].map(classify_delta)

    output[
        "OverallPropagation"
    ] = output.apply(
        classify_overall_propagation,
        axis=1,
    )

    return output


def classify_delta(
    value: object,
    tolerance: float = 1e-9,
) -> str:
    """Classify a disparity change."""
    numeric = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(numeric):
        return "not_available"

    if numeric > tolerance:
        return "amplified"

    if numeric < -tolerance:
        return "attenuated"

    return "unchanged"


def classify_overall_propagation(
    row: pd.Series,
) -> str:
    """Combine positive-rate and DI conclusions."""
    conclusions = {
        row.get(
            "PositiveRateGapInterpretation"
        ),
        row.get(
            "DIInterpretation"
        ),
    }

    conclusions.discard(
        "not_available"
    )

    if not conclusions:
        return "not_available"

    if conclusions == {
        "amplified"
    }:
        return "amplified"

    if conclusions == {
        "attenuated"
    }:
        return "attenuated"

    if conclusions == {
        "unchanged"
    }:
        return "unchanged"

    return "mixed"


def write_comparative_bias_workbook(
    *,
    result: ComparativeBiasResult,
    output_path: str | Path,
) -> Path:
    """Write all comparative analysis tables."""
    output_path = (
        Path(output_path)
        .expanduser()
        .resolve()
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:
        result.amplification_summary.to_excel(
            writer,
            sheet_name="amplification_summary",
            index=False,
        )
        result.aggregate_metrics.to_excel(
            writer,
            sheet_name="aggregate_metrics",
            index=False,
        )
        result.split_metrics.to_excel(
            writer,
            sheet_name="split_metrics",
            index=False,
        )
        result.file_status.to_excel(
            writer,
            sheet_name="file_status",
            index=False,
        )

    return output_path