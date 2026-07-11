"""Pretraining and posttraining bias-propagation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
)


def compute_bias_propagation_metrics(
    *,
    dataframe: pd.DataFrame,
    target_column: str,
    prediction_column: str,
    subgroup_column: str,
    positive_class: str,
    probability_column: str | None = None,
    minimum_group_size: int = 5,
    laplace_alpha: float = 1.0,
) -> dict[str, Any]:
    """Compare subgroup disparities in labels and predictions.

    The target and prediction are converted to one-vs-rest binary indicators
    for ``positive_class``. This supports binary and multiclass targets.
    """
    required_columns = [
        target_column,
        prediction_column,
        subgroup_column,
    ]

    working = dataframe[
        required_columns
    ].copy()

    if probability_column is not None:
        working[probability_column] = pd.to_numeric(
            dataframe[probability_column],
            errors="coerce",
        )

    working = working.dropna(
        subset=required_columns
    )

    working[target_column] = (
        working[target_column]
        .astype(str)
        .str.strip()
    )
    working[prediction_column] = (
        working[prediction_column]
        .astype(str)
        .str.strip()
    )
    working[subgroup_column] = (
        working[subgroup_column]
        .astype(str)
        .str.strip()
    )

    positive_class_casefold = (
        str(positive_class)
        .strip()
        .casefold()
    )

    working["__y_true_binary"] = (
        working[target_column]
        .str.casefold()
        == positive_class_casefold
    ).astype(int)

    working["__y_pred_binary"] = (
        working[prediction_column]
        .str.casefold()
        == positive_class_casefold
    ).astype(int)

    group_sizes = (
        working.groupby(
            subgroup_column,
            dropna=False,
        )
        .size()
    )

    valid_groups = group_sizes[
        group_sizes >= minimum_group_size
    ].index

    working = working[
        working[subgroup_column].isin(
            valid_groups
        )
    ].copy()

    if working.empty:
        return empty_bias_metric_result(
            positive_class=positive_class,
        )

    groups = (
        working[subgroup_column]
        .dropna()
        .unique()
        .tolist()
    )

    if len(groups) < 2:
        return empty_bias_metric_result(
            positive_class=positive_class,
            num_groups=len(groups),
            num_patients=len(working),
        )

    true_rates = compute_smoothed_positive_rates(
        dataframe=working,
        subgroup_column=subgroup_column,
        binary_column="__y_true_binary",
        laplace_alpha=laplace_alpha,
    )

    predicted_rates = compute_smoothed_positive_rates(
        dataframe=working,
        subgroup_column=subgroup_column,
        binary_column="__y_pred_binary",
        laplace_alpha=laplace_alpha,
    )

    pretraining_gap = rate_range(
        true_rates
    )
    posttraining_gap = rate_range(
        predicted_rates
    )

    pretraining_di = disparate_impact(
        true_rates
    )
    posttraining_di = disparate_impact(
        predicted_rates
    )

    accuracy_by_group = compute_accuracy_by_group(
        dataframe=working,
        subgroup_column=subgroup_column,
        target_column="__y_true_binary",
        prediction_column="__y_pred_binary",
    )

    recall_by_group = compute_recall_by_group(
        dataframe=working,
        subgroup_column=subgroup_column,
        target_column="__y_true_binary",
        prediction_column="__y_pred_binary",
    )

    auc_by_group = compute_auc_by_group(
        dataframe=working,
        subgroup_column=subgroup_column,
        target_column="__y_true_binary",
        probability_column=probability_column,
    )

    pretraining_di_distance = (
        absolute_di_distance(
            pretraining_di
        )
    )
    posttraining_di_distance = (
        absolute_di_distance(
            posttraining_di
        )
    )

    return {
        "PositiveClass": positive_class,
        "NumPatients": int(len(working)),
        "NumGroups": int(len(groups)),
        "Groups": "; ".join(
            sorted(
                str(group)
                for group in groups
            )
        ),
        "Pretraining_PositiveRateGap": (
            pretraining_gap
        ),
        "Posttraining_PositiveRateGap": (
            posttraining_gap
        ),
        "Delta_PositiveRateGap": (
            safe_difference(
                posttraining_gap,
                pretraining_gap,
            )
        ),
        "Pretraining_DI": pretraining_di,
        "Posttraining_DI": posttraining_di,
        "Pretraining_DI_Distance": (
            pretraining_di_distance
        ),
        "Posttraining_DI_Distance": (
            posttraining_di_distance
        ),
        "Delta_DI_Distance": (
            safe_difference(
                posttraining_di_distance,
                pretraining_di_distance,
            )
        ),
        "AccuracyGap": rate_range(
            accuracy_by_group
        ),
        "RecallGap": rate_range(
            recall_by_group
        ),
        "AUCGap": rate_range(
            auc_by_group
        ),
        "Pretraining_PositiveRate_Min": (
            safe_min(true_rates)
        ),
        "Pretraining_PositiveRate_Max": (
            safe_max(true_rates)
        ),
        "Posttraining_PositiveRate_Min": (
            safe_min(predicted_rates)
        ),
        "Posttraining_PositiveRate_Max": (
            safe_max(predicted_rates)
        ),
        "Accuracy_Min": safe_min(
            accuracy_by_group
        ),
        "Accuracy_Max": safe_max(
            accuracy_by_group
        ),
        "Recall_Min": safe_min(
            recall_by_group
        ),
        "Recall_Max": safe_max(
            recall_by_group
        ),
        "AUC_Min": safe_min(
            auc_by_group
        ),
        "AUC_Max": safe_max(
            auc_by_group
        ),
        "BiasAmplified_PositiveRateGap": (
            bool(
                np.isfinite(
                    safe_difference(
                        posttraining_gap,
                        pretraining_gap,
                    )
                )
                and safe_difference(
                    posttraining_gap,
                    pretraining_gap,
                ) > 0
            )
        ),
        "BiasAmplified_DI": (
            bool(
                np.isfinite(
                    safe_difference(
                        posttraining_di_distance,
                        pretraining_di_distance,
                    )
                )
                and safe_difference(
                    posttraining_di_distance,
                    pretraining_di_distance,
                ) > 0
            )
        ),
    }


def compute_smoothed_positive_rates(
    *,
    dataframe: pd.DataFrame,
    subgroup_column: str,
    binary_column: str,
    laplace_alpha: float,
) -> pd.Series:
    """Compute Laplace-smoothed positive rates by subgroup."""
    grouped = dataframe.groupby(
        subgroup_column
    )[binary_column]

    positives = grouped.sum()
    totals = grouped.count()

    rates = (
        positives + laplace_alpha
    ) / (
        totals + 2.0 * laplace_alpha
    )

    return rates.astype(float)


def compute_accuracy_by_group(
    *,
    dataframe: pd.DataFrame,
    subgroup_column: str,
    target_column: str,
    prediction_column: str,
) -> pd.Series:
    """Compute binary accuracy separately for each subgroup."""
    values: dict[str, float] = {}

    for subgroup, subgroup_df in dataframe.groupby(
        subgroup_column
    ):
        values[str(subgroup)] = float(
            accuracy_score(
                subgroup_df[target_column],
                subgroup_df[prediction_column],
            )
        )

    return pd.Series(
        values,
        dtype=float,
    )


def compute_recall_by_group(
    *,
    dataframe: pd.DataFrame,
    subgroup_column: str,
    target_column: str,
    prediction_column: str,
) -> pd.Series:
    """Compute positive-class recall separately by subgroup."""
    values: dict[str, float] = {}

    for subgroup, subgroup_df in dataframe.groupby(
        subgroup_column
    ):
        true_values = subgroup_df[
            target_column
        ]

        if true_values.sum() == 0:
            values[str(subgroup)] = np.nan
            continue

        values[str(subgroup)] = float(
            recall_score(
                true_values,
                subgroup_df[
                    prediction_column
                ],
                zero_division=0,
            )
        )

    return pd.Series(
        values,
        dtype=float,
    )


def compute_auc_by_group(
    *,
    dataframe: pd.DataFrame,
    subgroup_column: str,
    target_column: str,
    probability_column: str | None,
) -> pd.Series:
    """Compute binary AUROC separately by subgroup."""
    if (
        probability_column is None
        or probability_column
        not in dataframe.columns
    ):
        return pd.Series(
            dtype=float
        )

    values: dict[str, float] = {}

    for subgroup, subgroup_df in dataframe.groupby(
        subgroup_column
    ):
        usable = subgroup_df[
            [
                target_column,
                probability_column,
            ]
        ].dropna()

        usable[probability_column] = (
            pd.to_numeric(
                usable[probability_column],
                errors="coerce",
            )
        )
        usable = usable.dropna(
            subset=[probability_column]
        )

        if (
            usable.empty
            or usable[target_column].nunique()
            < 2
        ):
            values[str(subgroup)] = np.nan
            continue

        try:
            values[str(subgroup)] = float(
                roc_auc_score(
                    usable[target_column],
                    usable[probability_column],
                )
            )
        except Exception:
            values[str(subgroup)] = np.nan

    return pd.Series(
        values,
        dtype=float,
    )


def disparate_impact(
    rates: pd.Series,
) -> float:
    """Compute min/max disparate impact."""
    usable = pd.to_numeric(
        rates,
        errors="coerce",
    ).dropna()

    if len(usable) < 2:
        return np.nan

    maximum = float(
        usable.max()
    )

    if maximum <= 0:
        return np.nan

    return float(
        usable.min() / maximum
    )


def absolute_di_distance(
    disparate_impact_value: float,
) -> float:
    """Express DI disparity as distance from ideal value one."""
    if not np.isfinite(
        disparate_impact_value
    ):
        return np.nan

    return float(
        abs(
            1.0
            - disparate_impact_value
        )
    )


def rate_range(
    values: pd.Series,
) -> float:
    """Return maximum minus minimum for valid subgroup values."""
    usable = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if len(usable) < 2:
        return np.nan

    return float(
        usable.max()
        - usable.min()
    )


def safe_difference(
    after: float,
    before: float,
) -> float:
    """Return after minus before when both values are finite."""
    if (
        not np.isfinite(after)
        or not np.isfinite(before)
    ):
        return np.nan

    return float(
        after - before
    )


def safe_min(
    values: pd.Series,
) -> float:
    """Return the finite minimum."""
    usable = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if usable.empty:
        return np.nan

    return float(
        usable.min()
    )


def safe_max(
    values: pd.Series,
) -> float:
    """Return the finite maximum."""
    usable = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if usable.empty:
        return np.nan

    return float(
        usable.max()
    )


def empty_bias_metric_result(
    *,
    positive_class: str,
    num_groups: int = 0,
    num_patients: int = 0,
) -> dict[str, Any]:
    """Return a stable all-NaN metric result."""
    return {
        "PositiveClass": positive_class,
        "NumPatients": int(
            num_patients
        ),
        "NumGroups": int(
            num_groups
        ),
        "Groups": "",
        "Pretraining_PositiveRateGap": np.nan,
        "Posttraining_PositiveRateGap": np.nan,
        "Delta_PositiveRateGap": np.nan,
        "Pretraining_DI": np.nan,
        "Posttraining_DI": np.nan,
        "Pretraining_DI_Distance": np.nan,
        "Posttraining_DI_Distance": np.nan,
        "Delta_DI_Distance": np.nan,
        "AccuracyGap": np.nan,
        "RecallGap": np.nan,
        "AUCGap": np.nan,
        "Pretraining_PositiveRate_Min": np.nan,
        "Pretraining_PositiveRate_Max": np.nan,
        "Posttraining_PositiveRate_Min": np.nan,
        "Posttraining_PositiveRate_Max": np.nan,
        "Accuracy_Min": np.nan,
        "Accuracy_Max": np.nan,
        "Recall_Min": np.nan,
        "Recall_Max": np.nan,
        "AUC_Min": np.nan,
        "AUC_Max": np.nan,
        "BiasAmplified_PositiveRateGap": False,
        "BiasAmplified_DI": False,
    }