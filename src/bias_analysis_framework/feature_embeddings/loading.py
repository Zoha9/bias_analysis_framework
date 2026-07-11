"""Loading and matching of STAMP feature embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bias_analysis_framework.io.feature_files import (
    aggregate_slide_embedding,
    discover_feature_files,
    load_feature_array,
)


@dataclass(frozen=True)
class LoadedEmbeddings:
    """Slide-level embeddings with matched metadata."""

    embeddings: np.ndarray
    metadata: pd.DataFrame
    feature_dimension: int


def load_slide_level_embeddings(
    *,
    clinical_df: pd.DataFrame,
    slide_df: pd.DataFrame,
    config: dict[str, Any],
    model_name: str,
) -> LoadedEmbeddings:
    """Load and aggregate STAMP features for one configured model.

    The function:

    1. discovers HDF5 feature files;
    2. matches them to slide-table filenames;
    3. averages tile embeddings into one embedding per slide;
    4. attaches patient-level clinical labels.
    """
    data_config = config.get("data", {})
    workflow_config = (
        config.get("workflows", {})
        .get("feature_embedding_analysis", {})
    )

    feature_root = Path(data_config["feature_root"]).expanduser().resolve()
    patient_column = data_config.get("patient_column", "PATIENT")
    filename_column = data_config.get("slide_filename_column", "FILENAME")

    aggregation = workflow_config.get("slide_aggregation", "mean")
    configured_dataset_name = workflow_config.get("h5_dataset_name")

    model_root = resolve_model_feature_root(
        feature_root=feature_root,
        config=config,
        model_name=model_name,
    )

    feature_files = discover_feature_files(model_root)

    if not feature_files:
        raise FileNotFoundError(
            f"No HDF5 feature files were found for model {model_name!r} "
            f"under {model_root}."
        )

    if patient_column not in clinical_df.columns:
        raise ValueError(
            f"Clinical table is missing patient column: {patient_column}"
        )

    if patient_column not in slide_df.columns:
        raise ValueError(
            f"Slide table is missing patient column: {patient_column}"
        )

    if filename_column not in slide_df.columns:
        raise ValueError(
            f"Slide table is missing filename column: {filename_column}"
        )

    clinical_unique = (
        clinical_df.drop_duplicates(subset=[patient_column], keep="first")
        .copy()
    )

    feature_lookup = build_feature_lookup(feature_files)

    metadata_rows: list[dict[str, Any]] = []
    embedding_rows: list[np.ndarray] = []

    expected_dimension: int | None = None

    for _, slide_row in slide_df.iterrows():
        raw_filename = slide_row.get(filename_column)
        patient_id = slide_row.get(patient_column)

        if pd.isna(raw_filename) or pd.isna(patient_id):
            continue

        feature_path = match_feature_file(
            filename=str(raw_filename),
            feature_lookup=feature_lookup,
        )

        if feature_path is None:
            continue

        feature_array = load_feature_array(
            feature_path,
            dataset_name=configured_dataset_name,
        )

        slide_embedding = aggregate_slide_embedding(
            feature_array,
            method=aggregation,
        )

        if expected_dimension is None:
            expected_dimension = int(slide_embedding.shape[0])
        elif slide_embedding.shape[0] != expected_dimension:
            raise ValueError(
                f"Inconsistent feature dimensions for model {model_name!r}. "
                f"Expected {expected_dimension}, found "
                f"{slide_embedding.shape[0]} in {feature_path}."
            )

        clinical_match = clinical_unique[
            clinical_unique[patient_column].astype(str) == str(patient_id)
        ]

        if clinical_match.empty:
            continue

        clinical_row = clinical_match.iloc[0].to_dict()

        metadata = {
            **clinical_row,
            "MODEL": model_name,
            "FEATURE_FILE": str(feature_path),
            "SLIDE_FILENAME": str(raw_filename),
            "N_TILES": int(feature_array.shape[0]),
            "FEATURE_DIMENSION": int(feature_array.shape[1]),
        }

        metadata_rows.append(metadata)
        embedding_rows.append(slide_embedding)

    if not embedding_rows:
        raise ValueError(
            f"No feature files for model {model_name!r} could be matched "
            "to both the slide table and clinical table."
        )

    embeddings = np.vstack(embedding_rows).astype(np.float32)
    metadata_df = pd.DataFrame(metadata_rows).reset_index(drop=True)

    return LoadedEmbeddings(
        embeddings=embeddings,
        metadata=metadata_df,
        feature_dimension=int(embeddings.shape[1]),
    )


def resolve_model_feature_root(
    *,
    feature_root: Path,
    config: dict[str, Any],
    model_name: str,
) -> Path:
    """Resolve the most likely feature directory for one model."""
    cohort = config.get("cohort", {})
    organ = str(cohort.get("organ", "")).strip()
    dataset = str(cohort.get("dataset", "")).strip()

    candidates = [
        feature_root / organ / dataset / model_name,
        feature_root / organ / dataset,
        feature_root / model_name,
        feature_root,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            model_specific_files = [
                path
                for path in candidate.rglob("*.h5")
                if model_name.casefold() in str(path).casefold()
            ]

            if model_specific_files:
                common_parent = model_specific_files[0].parent
                return common_parent

            if candidate.name.casefold() == model_name.casefold():
                return candidate

    cohort_root = feature_root / organ / dataset
    if cohort_root.exists():
        return cohort_root

    return feature_root


def build_feature_lookup(
    feature_files: list[Path],
) -> dict[str, Path]:
    """Build filename and stem lookup keys for feature files."""
    lookup: dict[str, Path] = {}

    for path in feature_files:
        keys = {
            path.name.casefold(),
            path.stem.casefold(),
        }

        for key in keys:
            lookup.setdefault(key, path)

    return lookup


def match_feature_file(
    *,
    filename: str,
    feature_lookup: dict[str, Path],
) -> Path | None:
    """Match a slide-table filename to a discovered feature file."""
    filename_path = Path(filename)

    candidates = [
        filename_path.name.casefold(),
        filename_path.stem.casefold(),
    ]

    if filename_path.suffix.casefold() not in {".h5", ".hdf5"}:
        candidates.extend(
            [
                f"{filename_path.name}.h5".casefold(),
                f"{filename_path.stem}.h5".casefold(),
            ]
        )

    for candidate in candidates:
        if candidate in feature_lookup:
            return feature_lookup[candidate]

    return None