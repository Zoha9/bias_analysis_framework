"""Regular STAMP prediction-performance analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from bias_analysis_framework.io.predictions import (
    PredictionFile,
    choose_prediction_file,
    discover_prediction_files,
    discover_probability_columns,
    read_prediction_table,
    validate_prediction_table,
)
from bias_analysis_framework.io.stamp_outputs import (
    normalize_directory_name,
)
from bias_analysis_framework.metrics.model_performance_metrics import (
    compute_prediction_metrics,
)


@dataclass(frozen=True)
class RegularPredictionAnalysisResult:
    """Outputs from regular prediction evaluation."""

    split_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    patient_scores: pd.DataFrame
    file_status: pd.DataFrame


def analyze_regular_predictions(
    *,
    config: dict[str, Any],
    predictions_root: str | Path,
) -> RegularPredictionAnalysisResult:
    """Analyze all regular STAMP prediction outputs."""
    prediction_files = choose_prediction_file(
        discover_prediction_files(
            predictions_root=predictions_root,
        )
    )

    if not prediction_files:
        return RegularPredictionAnalysisResult(
            split_metrics=pd.DataFrame(),
            aggregate_metrics=pd.DataFrame(),
            patient_scores=pd.DataFrame(),
            file_status=pd.DataFrame(
                [
                    {
                        "Status": "no_prediction_files",
                        "Message": (
                            "No patient-preds.csv or "
                            "patient-preds-0.csv files were found."
                        ),
                    }
                ]
            ),
        )

    split_rows: list[dict[str, Any]] = []
    patient_score_tables: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []

    for prediction_file in prediction_files:
        label_spec = resolve_prediction_label(
            directory_label=prediction_file.label_name,
            config=config,
        )

        target_column = label_spec["column"]
        display_name = label_spec["display_name"]
        label_type = label_spec["type"]
        positive_label = label_spec.get(
            "positive_label"
        )

        model_spec = resolve_prediction_model(
            directory_model=prediction_file.model_name,
            config=config,
        )

        try:
            dataframe = read_prediction_table(
                prediction_file.path
            )

            validate_prediction_table(
                dataframe=dataframe,
                target_column=target_column,
            )

            probability_columns = (
                discover_probability_columns(
                    dataframe=dataframe,
                    target_column=target_column,
                )
            )

            metrics, patient_scores = (
                compute_prediction_metrics(
                    dataframe=dataframe,
                    target_column=target_column,
                    probability_columns=(
                        probability_columns
                    ),
                    positive_label=positive_label,
                )
            )

            row = {
                "Target": display_name,
                "TargetColumn": target_column,
                "LabelType": label_type,
                "Model": model_spec["display_name"],
                "Extractor": model_spec["extractor"],
                "Split": prediction_file.split_number,
                "PredictionFile": str(
                    prediction_file.path
                ),
                **metrics,
            }
            split_rows.append(row)

            if (
                patient_scores is not None
                and not patient_scores.empty
            ):
                patient_scores = (
                    patient_scores.copy()
                )
                patient_scores["Target"] = (
                    display_name
                )
                patient_scores["TargetColumn"] = (
                    target_column
                )
                patient_scores["LabelType"] = (
                    label_type
                )
                patient_scores["Model"] = (
                    model_spec["display_name"]
                )
                patient_scores["Extractor"] = (
                    model_spec["extractor"]
                )
                patient_scores["Split"] = (
                    prediction_file.split_number
                )
                patient_scores["PredictionFile"] = str(
                    prediction_file.path
                )

                patient_score_tables.append(
                    patient_scores
                )

            status_rows.append(
                {
                    "Target": display_name,
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
                    "PredictionFile": str(
                        prediction_file.path
                    ),
                    "Status": "completed",
                    "Rows": len(dataframe),
                    "Message": "",
                }
            )

        except Exception as error:
            status_rows.append(
                {
                    "Target": display_name,
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
                    "PredictionFile": str(
                        prediction_file.path
                    ),
                    "Status": "failed",
                    "Rows": pd.NA,
                    "Message": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                }
            )

    split_metrics = pd.DataFrame(
        split_rows
    )

    patient_scores = (
        pd.concat(
            patient_score_tables,
            ignore_index=True,
        )
        if patient_score_tables
        else pd.DataFrame()
    )

    aggregate_metrics = aggregate_split_metrics(
        split_metrics
    )

    return RegularPredictionAnalysisResult(
        split_metrics=split_metrics,
        aggregate_metrics=aggregate_metrics,
        patient_scores=patient_scores,
        file_status=pd.DataFrame(
            status_rows
        ),
    )


def aggregate_split_metrics(
    split_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate metrics across cross-validation splits."""
    if split_metrics.empty:
        return pd.DataFrame()

    metric_columns = [
        "Num_Patients",
        "Pos_Prevalence",
        "Baseline_Accuracy",
        "Accuracy",
        "Balanced_Accuracy",
        "F1_macro",
        "F1_weighted",
        "F1_micro",
        "AUROC",
        "AUPRC",
        "Sensitivity",
        "Specificity",
        "Precision",
        "NPV",
    ]

    available_metrics = [
        column
        for column in metric_columns
        if column in split_metrics.columns
    ]

    group_columns = [
        "Target",
        "TargetColumn",
        "LabelType",
        "Model",
        "Extractor",
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

        row["Num_Splits"] = int(
            group_df["Split"].nunique()
        )

        for metric in available_metrics:
            numeric = pd.to_numeric(
                group_df[metric],
                errors="coerce",
            )

            row[f"{metric}_mean"] = numeric.mean()
            row[f"{metric}_std"] = numeric.std(
                ddof=1
            )
            row[f"{metric}_median"] = (
                numeric.median()
            )
            row[f"{metric}_min"] = numeric.min()
            row[f"{metric}_max"] = numeric.max()
            row[f"{metric}_count"] = int(
                numeric.notna().sum()
            )

        rows.append(row)

    return pd.DataFrame(rows)


def write_regular_prediction_analysis(
    *,
    result: RegularPredictionAnalysisResult,
    output_path: str | Path,
) -> Path:
    """Write the complete regular prediction workbook."""
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
        result.patient_scores.to_excel(
            writer,
            sheet_name="patient_scores",
            index=False,
        )
        result.file_status.to_excel(
            writer,
            sheet_name="file_status",
            index=False,
        )

    return output_path


def resolve_prediction_label(
    *,
    directory_label: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Resolve a filesystem label to display and clinical names."""
    normalized = normalize_directory_name(
        directory_label
    )

    for target in config.get(
        "targets",
        [],
    ):
        column_name = str(
            target.get("column")
            or target.get("name")
            or ""
        ).strip()

        display_name = str(
            target.get("name")
            or column_name
        ).strip()

        if normalized in {
            normalize_directory_name(column_name),
            normalize_directory_name(display_name),
        }:
            return {
                "display_name": display_name,
                "column": column_name,
                "type": "target",
                "positive_label": target.get(
                    "positive_label"
                ),
            }

    for factor in config.get(
        "bias_factors",
        [],
    ):
        column_name = str(
            factor.get("column")
            or factor.get("name")
            or ""
        ).strip()

        display_name = str(
            factor.get("name")
            or column_name
        ).strip()

        if normalized in {
            normalize_directory_name(column_name),
            normalize_directory_name(display_name),
        }:
            return {
                "display_name": display_name,
                "column": column_name,
                "type": "bias_factor",
                "positive_label": factor.get(
                    "positive_label"
                ),
            }

    return {
        "display_name": directory_label,
        "column": directory_label,
        "type": "unknown",
        "positive_label": None,
    }


def resolve_prediction_model(
    *,
    directory_model: str,
    config: dict[str, Any],
) -> dict[str, str]:
    """Resolve a filesystem model directory to display/extractor names."""
    normalized = normalize_directory_name(
        directory_model
    )

    for model in config.get(
        "models",
        [],
    ):
        if isinstance(model, str):
            display_name = model
            extractor = model
        else:
            display_name = str(
                model.get("name")
                or model.get("extractor")
                or ""
            )
            extractor = str(
                model.get("extractor")
                or model.get("name")
                or ""
            )

        if normalized in {
            normalize_directory_name(display_name),
            normalize_directory_name(extractor),
        }:
            return {
                "display_name": display_name,
                "extractor": extractor,
            }

    return {
        "display_name": directory_model,
        "extractor": directory_model,
    }