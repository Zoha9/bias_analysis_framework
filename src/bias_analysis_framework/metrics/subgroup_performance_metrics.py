"""Metrics and disparity summaries for subgroup model performance."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_SUBGROUP_METRICS = (
    "Accuracy",
    "Balanced_Accuracy",
    "F1_macro",
    "AUROC",
    "AUPRC",
    "Sensitivity",
    "Specificity",
    "Precision",
    "NPV",
)


def aggregate_subgroup_split_metrics(
    subgroup_split_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate subgroup performance across model splits."""
    if subgroup_split_metrics.empty:
        return pd.DataFrame()

    metric_columns = [
        column
        for column in DEFAULT_SUBGROUP_METRICS
        if column in subgroup_split_metrics.columns
    ]

    group_columns = [
        "Target",
        "TargetColumn",
        "Model",
        "Extractor",
        "BiasFactor",
        "BiasFactorColumn",
        "Subgroup",
    ]

    rows: list[dict[str, Any]] = []

    grouped = subgroup_split_metrics.groupby(
        group_columns,
        dropna=False,
    )

    for group_values, group_df in grouped:
        row = dict(
            zip(
                group_columns,
                group_values,
            )
        )

        row["Num_Splits"] = int(
            group_df["Split"].nunique()
        )

        sample_sizes = pd.to_numeric(
            group_df["Num_Patients"],
            errors="coerce",
        )

        row["Num_Patients_mean"] = sample_sizes.mean()
        row["Num_Patients_min"] = sample_sizes.min()
        row["Num_Patients_max"] = sample_sizes.max()

        prevalence = pd.to_numeric(
            group_df.get(
                "Pos_Prevalence",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        )
        row["Pos_Prevalence_mean"] = prevalence.mean()

        for metric in metric_columns:
            values = pd.to_numeric(
                group_df[metric],
                errors="coerce",
            )

            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std(
                ddof=1
            )
            row[f"{metric}_median"] = values.median()
            row[f"{metric}_min"] = values.min()
            row[f"{metric}_max"] = values.max()
            row[f"{metric}_count"] = int(
                values.notna().sum()
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_subgroup_disparity_summary(
    subgroup_aggregate_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate best-to-worst subgroup disparities.

    For each target, model, bias factor, and metric, reports:

    - best subgroup;
    - worst subgroup;
    - absolute range;
    - ratio of worst to best;
    - mean subgroup score;
    - standard deviation across subgroup means.
    """
    if subgroup_aggregate_metrics.empty:
        return pd.DataFrame()

    metric_mean_columns = [
        column
        for column in subgroup_aggregate_metrics.columns
        if column.endswith("_mean")
        and column.removesuffix("_mean")
        in DEFAULT_SUBGROUP_METRICS
    ]

    group_columns = [
        "Target",
        "TargetColumn",
        "Model",
        "Extractor",
        "BiasFactor",
        "BiasFactorColumn",
    ]

    rows: list[dict[str, Any]] = []

    for group_values, group_df in (
        subgroup_aggregate_metrics.groupby(
            group_columns,
            dropna=False,
        )
    ):
        group_metadata = dict(
            zip(
                group_columns,
                group_values,
            )
        )

        for metric_column in metric_mean_columns:
            metric_name = metric_column.removesuffix(
                "_mean"
            )

            working = group_df[
                [
                    "Subgroup",
                    metric_column,
                ]
            ].copy()

            working[metric_column] = pd.to_numeric(
                working[metric_column],
                errors="coerce",
            )
            working = working.dropna(
                subset=[metric_column]
            )

            if working.empty:
                continue

            best_index = working[
                metric_column
            ].idxmax()
            worst_index = working[
                metric_column
            ].idxmin()

            best_row = working.loc[
                best_index
            ]
            worst_row = working.loc[
                worst_index
            ]

            best_value = float(
                best_row[metric_column]
            )
            worst_value = float(
                worst_row[metric_column]
            )

            rows.append(
                {
                    **group_metadata,
                    "Metric": metric_name,
                    "BestSubgroup": str(
                        best_row["Subgroup"]
                    ),
                    "BestValue": best_value,
                    "WorstSubgroup": str(
                        worst_row["Subgroup"]
                    ),
                    "WorstValue": worst_value,
                    "AbsoluteGap": (
                        best_value - worst_value
                    ),
                    "WorstToBestRatio": (
                        safe_ratio(
                            worst_value,
                            best_value,
                        )
                    ),
                    "SubgroupMean": float(
                        working[
                            metric_column
                        ].mean()
                    ),
                    "SubgroupStd": float(
                        working[
                            metric_column
                        ].std(ddof=1)
                    )
                    if len(working) > 1
                    else np.nan,
                    "NumSubgroups": int(
                        len(working)
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_pairwise_subgroup_gaps(
    subgroup_aggregate_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build pairwise absolute metric differences between subgroups."""
    if subgroup_aggregate_metrics.empty:
        return pd.DataFrame()

    metric_mean_columns = [
        column
        for column in subgroup_aggregate_metrics.columns
        if column.endswith("_mean")
        and column.removesuffix("_mean")
        in DEFAULT_SUBGROUP_METRICS
    ]

    group_columns = [
        "Target",
        "TargetColumn",
        "Model",
        "Extractor",
        "BiasFactor",
        "BiasFactorColumn",
    ]

    rows: list[dict[str, Any]] = []

    for group_values, group_df in (
        subgroup_aggregate_metrics.groupby(
            group_columns,
            dropna=False,
        )
    ):
        group_metadata = dict(
            zip(
                group_columns,
                group_values,
            )
        )

        subgroup_rows = list(
            group_df.to_dict(
                orient="records"
            )
        )

        for first_index in range(
            len(subgroup_rows)
        ):
            for second_index in range(
                first_index + 1,
                len(subgroup_rows),
            ):
                first = subgroup_rows[
                    first_index
                ]
                second = subgroup_rows[
                    second_index
                ]

                for metric_column in metric_mean_columns:
                    first_value = pd.to_numeric(
                        pd.Series(
                            [first.get(metric_column)]
                        ),
                        errors="coerce",
                    ).iloc[0]

                    second_value = pd.to_numeric(
                        pd.Series(
                            [second.get(metric_column)]
                        ),
                        errors="coerce",
                    ).iloc[0]

                    if (
                        pd.isna(first_value)
                        or pd.isna(second_value)
                    ):
                        continue

                    rows.append(
                        {
                            **group_metadata,
                            "Metric": (
                                metric_column.removesuffix(
                                    "_mean"
                                )
                            ),
                            "SubgroupA": first[
                                "Subgroup"
                            ],
                            "SubgroupB": second[
                                "Subgroup"
                            ],
                            "ValueA": float(
                                first_value
                            ),
                            "ValueB": float(
                                second_value
                            ),
                            "SignedDifference_A_minus_B": float(
                                first_value
                                - second_value
                            ),
                            "AbsoluteDifference": float(
                                abs(
                                    first_value
                                    - second_value
                                )
                            ),
                        }
                    )

    return pd.DataFrame(rows)


def safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    """Return a finite ratio or NaN."""
    if denominator == 0:
        return np.nan

    return float(
        numerator / denominator
    )