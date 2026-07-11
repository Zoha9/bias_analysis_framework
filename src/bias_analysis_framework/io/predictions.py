"""Loading and discovery of STAMP patient prediction files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from bias_analysis_framework.io.stamp_outputs import (
    normalize_directory_name,
    parse_split_number,
)


DEFAULT_PREDICTION_FILENAMES = (
    "patient-preds.csv",
    "patient-preds-0.csv",
)


@dataclass(frozen=True)
class PredictionFile:
    """Metadata describing one STAMP prediction file."""

    label_name: str
    model_name: str
    split_number: int
    path: Path


def discover_prediction_files(
    *,
    predictions_root: str | Path,
    filenames: Iterable[str] = DEFAULT_PREDICTION_FILENAMES,
) -> list[PredictionFile]:
    """Discover STAMP patient-level prediction CSVs.

    Expected structure:

    predictions/
        label/
            model/
                split-0/
                    patient-preds.csv
    """
    root = Path(predictions_root).expanduser().resolve()

    if not root.exists():
        return []

    allowed_names = {
        str(filename).casefold()
        for filename in filenames
    }

    discovered: list[PredictionFile] = []

    for candidate in root.rglob("*.csv"):
        if candidate.name.casefold() not in allowed_names:
            continue

        relative = candidate.relative_to(root)

        if len(relative.parts) < 4:
            continue

        label_directory = relative.parts[0]
        model_directory = relative.parts[1]
        split_directory = relative.parts[2]

        split_number = parse_split_number(
            split_directory
        )

        if split_number is None:
            continue

        discovered.append(
            PredictionFile(
                label_name=label_directory,
                model_name=model_directory,
                split_number=split_number,
                path=candidate.resolve(),
            )
        )

    return sorted(
        discovered,
        key=lambda item: (
            normalize_directory_name(item.label_name),
            normalize_directory_name(item.model_name),
            item.split_number,
            item.path.name,
        ),
    )


def choose_prediction_file(
    prediction_files: list[PredictionFile],
) -> list[PredictionFile]:
    """Keep one preferred prediction file per label, model, and split.

    ``patient-preds.csv`` is preferred over ``patient-preds-0.csv``.
    """
    grouped: dict[
        tuple[str, str, int],
        list[PredictionFile],
    ] = {}

    for prediction_file in prediction_files:
        key = (
            prediction_file.label_name,
            prediction_file.model_name,
            prediction_file.split_number,
        )
        grouped.setdefault(key, []).append(
            prediction_file
        )

    output: list[PredictionFile] = []

    priority = {
        "patient-preds.csv": 0,
        "patient-preds-0.csv": 1,
    }

    for candidates in grouped.values():
        selected = sorted(
            candidates,
            key=lambda item: priority.get(
                item.path.name.casefold(),
                99,
            ),
        )[0]

        output.append(selected)

    return sorted(
        output,
        key=lambda item: (
            item.label_name,
            item.model_name,
            item.split_number,
        ),
    )


def read_prediction_table(
    prediction_path: str | Path,
) -> pd.DataFrame:
    """Read one STAMP patient prediction CSV."""
    path = Path(prediction_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Prediction file does not exist: {path}"
        )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(
            f"Prediction file is empty: {path}"
        )

    return dataframe


def discover_probability_columns(
    *,
    dataframe: pd.DataFrame,
    target_column: str,
) -> dict[str, str]:
    """Return mapping of class label to probability-column name.

    STAMP typically emits columns such as:

    ``MSI_Status_MSI-H``
    ``MSI_Status_MSS``
    """
    prefix = f"{target_column}_"

    output: dict[str, str] = {}

    for column in dataframe.columns:
        column_text = str(column)

        if not column_text.startswith(prefix):
            continue

        class_name = column_text[
            len(prefix):
        ].strip()

        if class_name:
            output[class_name] = column_text

    return output


def validate_prediction_table(
    *,
    dataframe: pd.DataFrame,
    target_column: str,
    prediction_column: str = "pred",
) -> None:
    """Validate the minimum columns required for model evaluation."""
    missing = [
        column
        for column in (
            target_column,
            prediction_column,
        )
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            "Prediction table is missing required columns "
            f"{missing}. Available columns: "
            f"{list(dataframe.columns)}"
        )