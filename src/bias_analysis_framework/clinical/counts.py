from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd

from bias_analysis_framework.metrics.clinical_pretraining_metrics import (
    TISSUE_BIAS_NAME,
    casefold_value,
    get_bias_factors_from_config,
    get_targets_from_config,
    standardize_column_aliases,
)


NA_TOKENS = {"", "na", "nan", "n/a", "null", "none", "unknown", "<na>"}


def build_clinical_count_tables(
    clinical_df: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Build Step 1 clinical count tables.

    Produces count tables used for:
    - direct Excel reporting
    - later count heatmap visualizations

    The logic is the config-driven replacement for the count portions of the
    old pretraining_metrics_visualization_and_counts.py script.
    """
    df = standardize_column_aliases(clinical_df.copy())

    tables: dict[str, pd.DataFrame] = OrderedDict()

    target_counts = build_target_class_counts(df, config)
    if not target_counts.empty:
        tables["target_class_counts"] = target_counts

    bias_counts = build_bias_factor_subgroup_counts(df, config)
    if not bias_counts.empty:
        tables["bias_factor_subgroup_counts"] = bias_counts

    cross_counts = build_bias_factor_by_target_class_counts(df, config)
    if not cross_counts.empty:
        tables["bias_factor_by_target_class_counts"] = cross_counts

    age_counts = build_age_counts(df, config)
    if not age_counts.empty:
        tables["age_counts"] = age_counts

    tissue_counts = build_tissue_source_site_counts(df)
    if not tissue_counts.empty:
        tables["tissue_source_site_counts"] = tissue_counts

    patient_summary = build_patient_count_summary(df, config)
    tables["patient_count_summary"] = patient_summary

    return tables


def build_patient_count_summary(
    clinical_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Summarize patient-level counts."""
    patient_column = config.get("data", {}).get("patient_column", "PATIENT")

    rows = [
        {"metric": "n_rows", "value": len(clinical_df)},
        {
            "metric": "n_unique_patients",
            "value": clinical_df[patient_column].nunique()
            if patient_column in clinical_df.columns
            else pd.NA,
        },
        {
            "metric": "n_duplicate_patient_rows",
            "value": int(clinical_df[patient_column].duplicated().sum())
            if patient_column in clinical_df.columns
            else pd.NA,
        },
    ]

    return pd.DataFrame(rows)


def build_target_class_counts(
    clinical_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Count target biomarker class labels."""
    rows = []

    target_map = get_targets_from_config(config, clinical_df)

    for target_name, configured_classes in target_map.items():
        if target_name not in clinical_df.columns:
            continue

        cleaned = clean_series(clinical_df[target_name])
        counts = cleaned.value_counts(dropna=False)

        for class_label, count in counts.items():
            rows.append(
                {
                    "target": target_name,
                    "class": class_label,
                    "count": int(count),
                    "configured_class": str(class_label) in [str(x) for x in configured_classes],
                }
            )

    return pd.DataFrame(rows)


def build_bias_factor_subgroup_counts(
    clinical_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Count subgroup labels for each configured non-tissue bias factor."""
    rows = []

    factor_specs = get_bias_factors_from_config(
        config,
        clinical_df,
    )

    for factor_name, factor_column in factor_specs:
        cleaned = clean_series(
            clinical_df[factor_column]
        )
        counts = cleaned.value_counts(
            dropna=False
        )

        for subgroup, count in counts.items():
            rows.append(
                {
                    "bias_factor": factor_name,
                    "bias_factor_column": factor_column,
                    "subgroup": subgroup,
                    "count": int(count),
                }
            )

    return pd.DataFrame(rows)


def build_bias_factor_by_target_class_counts(
    clinical_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Count rows by bias-factor subgroup and target class.

    Output columns:
    - bias_factor
    - subgroup
    - target
    - class
    - count

    This is the tidy equivalent of the old large matrix:
    (BiasFactor → Subgroup) × (Organ → Dataset → Biomarker → Class).
    """
    rows = []
    target_map = get_targets_from_config(config, clinical_df)
    bias_factors = get_bias_factors_from_config(config, clinical_df)

    for factor_name, factor_column in bias_factors:
        factor_values = clean_series(
            clinical_df[factor_column]
            )

        for target_name, classes in target_map.items():
            if target_name not in clinical_df.columns:
                continue

            target_values = clean_series(clinical_df[target_name])

            for subgroup in factor_values.dropna().unique():
                subgroup_mask = factor_values == subgroup

                for class_label in classes:
                    class_mask = target_values.map(casefold_value) == casefold_value(class_label)

                    rows.append(
                        {
                            "bias_factor": factor_name,
                            "bias_factor_column": factor_column,
                            "subgroup": subgroup,
                            "target": target_name,
                            "class": class_label,
                            "count": int((subgroup_mask & class_mask).sum()),
                        }
                    )

    return pd.DataFrame(rows)


def build_age_counts(
    clinical_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Count rows by configured or discovered age column."""
    age_column = get_age_column(config, clinical_df)

    if age_column is None:
        return pd.DataFrame()

    ages = pd.to_numeric(clinical_df[age_column], errors="coerce").dropna()

    if ages.empty:
        return pd.DataFrame()

    counts = ages.astype(int).value_counts().sort_index()

    return pd.DataFrame(
        [
            {
                "age_column": age_column,
                "age": int(age),
                "count": int(count),
            }
            for age, count in counts.items()
        ]
    )


def build_tissue_source_site_counts(
    clinical_df: pd.DataFrame,
) -> pd.DataFrame:
    """Count Tissue Source Site labels if available."""
    if TISSUE_BIAS_NAME not in clinical_df.columns:
        return pd.DataFrame()

    cleaned = clean_series(clinical_df[TISSUE_BIAS_NAME])
    counts = cleaned.value_counts(dropna=False)

    return pd.DataFrame(
        [
            {
                "tissue_source_site": site,
                "count": int(count),
            }
            for site, count in counts.items()
        ]
    )


def write_clinical_count_tables_excel(
    count_tables: dict[str, pd.DataFrame],
    output_path: str | Path,
) -> Path:
    """Write Step 1 clinical count tables to an Excel workbook."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, table in count_tables.items():
            safe_sheet = safe_excel_sheet_name(sheet_name)
            table.to_excel(writer, sheet_name=safe_sheet, index=False)

    return output_path


def clean_series(series: pd.Series) -> pd.Series:
    """Clean string-like labels and convert NA-like tokens to missing values."""
    def clean_value(value: object) -> object:
        if pd.isna(value):
            return pd.NA

        text = str(value).strip()
        if text.casefold() in NA_TOKENS:
            return pd.NA

        return text

    return series.map(clean_value)


def get_age_column(
    config: dict[str, Any],
    clinical_df: pd.DataFrame,
) -> str | None:
    """Return configured or discovered age column."""
    configured = (
        config.get("workflows", {})
        .get("clinical_distribution_analysis", {})
        .get("age_column")
    )

    if configured and configured in clinical_df.columns:
        return configured

    for candidate in ["Age", "Diagnosis Age", "Age_Onset_Continuous"]:
        if candidate in clinical_df.columns:
            return candidate

    return None


def safe_excel_sheet_name(value: str) -> str:
    """Return Excel-compatible sheet name."""
    sheet_name = str(value)
    for char in [":", "\\", "/", "?", "*", "[", "]"]:
        sheet_name = sheet_name.replace(char, "_")
    return sheet_name[:31]