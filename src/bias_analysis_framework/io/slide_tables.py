from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from bias_analysis_framework.io.tables import read_table, validate_columns


DEFAULT_PATIENT_COLUMN = "PATIENT"
DEFAULT_FILENAME_COLUMN = "FILENAME"


class SlideTableError(ValueError):
    """Raised when the slide table is missing required STAMP-style structure."""


def get_patient_column(config: dict[str, Any]) -> str:
    """Return configured patient column name."""
    return config.get("data", {}).get("patient_column", DEFAULT_PATIENT_COLUMN)


def get_slide_filename_column(config: dict[str, Any]) -> str:
    """Return configured slide/feature filename column name."""
    return config.get("data", {}).get("slide_filename_column", DEFAULT_FILENAME_COLUMN)


def get_required_slide_columns(config: dict[str, Any]) -> list[str]:
    """Return required slide table columns.

    Default required structure:
    - FILENAME
    - PATIENT

    This matches STAMP-style feature tables where FILENAME points to a slide
    or extracted .h5 feature file and PATIENT links back to the clinical table.
    """
    return [
        get_slide_filename_column(config),
        get_patient_column(config),
    ]


def read_slide_table(config: dict[str, Any]) -> pd.DataFrame:
    """Read and validate the configured slide table."""
    slide_path = config.get("data", {}).get("slide_table")

    if not slide_path:
        raise SlideTableError("Missing required config value: data.slide_table")

    slide_df = read_table(Path(slide_path))

    validate_slide_table(slide_df, config)

    return slide_df


def validate_slide_table(
    slide_df: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """Validate that the slide table has required STAMP-compatible columns."""
    required_columns = get_required_slide_columns(config)

    validate_columns(
        dataframe=slide_df,
        required_columns=required_columns,
        table_name="Slide table",
    )

    patient_column = get_patient_column(config)
    filename_column = get_slide_filename_column(config)

    if slide_df[patient_column].isna().any():
        raise SlideTableError(f"Slide table contains missing values in '{patient_column}'.")

    if slide_df[filename_column].isna().any():
        raise SlideTableError(f"Slide table contains missing values in '{filename_column}'.")


def validate_clinical_slide_patient_overlap(
    clinical_df: pd.DataFrame,
    slide_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Summarize overlap between clinical patients and slide-table patients.

    This does not require perfect overlap, because users may intentionally analyze
    subsets. It returns a summary table so mismatches are visible.
    """
    patient_column = get_patient_column(config)

    clinical_patients = set(clinical_df[patient_column].dropna().astype(str))
    slide_patients = set(slide_df[patient_column].dropna().astype(str))

    shared = clinical_patients & slide_patients
    clinical_only = clinical_patients - slide_patients
    slide_only = slide_patients - clinical_patients

    return pd.DataFrame(
        [
            {"item": "clinical_unique_patients", "value": len(clinical_patients)},
            {"item": "slide_unique_patients", "value": len(slide_patients)},
            {"item": "shared_patients", "value": len(shared)},
            {"item": "clinical_only_patients", "value": len(clinical_only)},
            {"item": "slide_only_patients", "value": len(slide_only)},
            {
                "item": "clinical_only_examples",
                "value": ", ".join(sorted(clinical_only)[:10]),
            },
            {
                "item": "slide_only_examples",
                "value": ", ".join(sorted(slide_only)[:10]),
            },
        ]
    )


def summarize_slide_table(
    slide_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Create a basic summary of the slide table."""
    patient_column = get_patient_column(config)
    filename_column = get_slide_filename_column(config)

    rows = [
        {"item": "n_rows", "value": len(slide_df)},
        {"item": "n_columns", "value": len(slide_df.columns)},
        {"item": "patient_column", "value": patient_column},
        {"item": "filename_column", "value": filename_column},
        {"item": "n_unique_patients", "value": slide_df[patient_column].nunique()},
        {"item": "n_unique_filenames", "value": slide_df[filename_column].nunique()},
        {
            "item": "n_h5_files",
            "value": slide_df[filename_column].astype(str).str.lower().str.endswith(".h5").sum(),
        },
    ]

    return pd.DataFrame(rows)