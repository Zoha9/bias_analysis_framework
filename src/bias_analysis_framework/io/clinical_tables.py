from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bias_analysis_framework.io.tables import read_table, validate_columns


DEFAULT_PATIENT_COLUMN = "PATIENT"


class ClinicalTableError(ValueError):
    """Raised when the clinical input table is missing required structure."""


def get_target_names(config: dict[str, Any]) -> list[str]:
    """Return target biomarker column names from config."""
    return [target["name"] for target in config.get("targets", [])]


def get_bias_factor_names(config: dict[str, Any]) -> list[str]:
    """Return actual clinical-table columns for configured bias factors."""
    return [
        str(
            factor.get("column")
            or factor.get("name")
        )
        for factor in config.get("bias_factors", [])
        if factor.get("column") or factor.get("name")
    ]

def get_required_clinical_columns(config: dict[str, Any]) -> list[str]:
    """Return required clinical table columns for configured analysis.

    Required by default:
    - PATIENT
    - all configured target biomarker columns
    - all configured bias factor columns
    """
    patient_column = config.get("data", {}).get("patient_column", DEFAULT_PATIENT_COLUMN)

    required = [patient_column]
    required.extend(get_target_names(config))
    required.extend(get_bias_factor_names(config))

    return _deduplicate_preserve_order(required)


def read_clinical_table(config: dict[str, Any]) -> pd.DataFrame:
    """Read and validate the clinical table configured by the user."""
    clinical_path = config.get("data", {}).get("clinical_table")

    if not clinical_path:
        raise ClinicalTableError("Missing required config value: data.clinical_table")

    clinical_df = read_table(Path(clinical_path))

    validate_clinical_table(clinical_df, config)

    return clinical_df


def validate_clinical_table(
    clinical_df: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """Validate that the clinical table has required columns."""
    required_columns = get_required_clinical_columns(config)

    validate_columns(
        dataframe=clinical_df,
        required_columns=required_columns,
        table_name="Clinical table",
    )

    patient_column = config.get("data", {}).get("patient_column", DEFAULT_PATIENT_COLUMN)

    if clinical_df[patient_column].isna().any():
        raise ClinicalTableError(f"Clinical table contains missing values in '{patient_column}'.")

    duplicated_patients = clinical_df[clinical_df[patient_column].duplicated()][patient_column].unique()

    if len(duplicated_patients) > 0:
        preview = ", ".join(map(str, duplicated_patients[:10]))
        raise ClinicalTableError(
            f"Clinical table contains duplicate patient IDs in '{patient_column}'. "
            f"Examples: {preview}"
        )


def summarize_clinical_table(
    clinical_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Create a basic summary of the clinical input table."""
    patient_column = config.get("data", {}).get("patient_column", DEFAULT_PATIENT_COLUMN)
    targets = get_target_names(config)
    bias_factors = get_bias_factor_names(config)

    rows = [
        {"item": "n_rows", "value": len(clinical_df)},
        {"item": "n_columns", "value": len(clinical_df.columns)},
        {"item": "patient_column", "value": patient_column},
        {"item": "n_unique_patients", "value": clinical_df[patient_column].nunique()},
        {"item": "targets", "value": ", ".join(targets)},
        {"item": "bias_factors", "value": ", ".join(bias_factors)},
        {"item": "organ", "value": config.get("cohort", {}).get("organ", "")},
        {"item": "dataset", "value": config.get("cohort", {}).get("dataset", "")},
    ]

    return pd.DataFrame(rows)


def summarize_configured_column_values(
    clinical_df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Summarize value counts for configured target and bias columns.

    Returns a dictionary of DataFrames that can be written as Excel sheets.
    """
    output: dict[str, pd.DataFrame] = {}

    for column in get_target_names(config) + get_bias_factor_names(config):
        counts = (
            clinical_df[column]
            .fillna("MISSING")
            .astype(str)
            .value_counts(dropna=False)
            .rename_axis(column)
            .reset_index(name="count")
        )
        counts["fraction"] = counts["count"] / counts["count"].sum()
        output[_excel_safe_sheet_name(column)] = counts

    return output


def _deduplicate_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []

    for value in values:
        if value not in seen:
            output.append(value)
            seen.add(value)

    return output


def _excel_safe_sheet_name(value: str) -> str:
    sheet_name = str(value)
    for char in [":", "\\", "/", "?", "*", "[", "]"]:
        sheet_name = sheet_name.replace(char, "_")
    return sheet_name[:31]