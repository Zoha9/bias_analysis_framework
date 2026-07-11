"""Visualizations for subgroup-stratified model performance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUBGROUP_COLORS = [
    "#2AB7CA",
    "#E94E77",
    "#FDB515",
    "#76B041",
    "#6A5ACD",
    "#A950B4",
    "#4A5FE1",
    "#964B36",
    "#7F7F7F",
]


def set_stratified_plot_style() -> None:
    """Apply compact, high-resolution plotting defaults."""
    mpl.rcParams.update(
        {
            "figure.dpi": 350,
            "savefig.dpi": 350,
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
        }
    )


def generate_stratified_performance_figures(
    *,
    aggregate_metrics: pd.DataFrame,
    disparity_summary: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    """Generate subgroup bar, line, disparity, and radar plots."""
    set_stratified_plot_style()

    output_dir = Path(
        output_dir
    ).expanduser().resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if aggregate_metrics.empty:
        return []

    workflow_config = (
        config.get("workflows", {})
        .get(
            "stratified_subgroup_performance_analysis",
            {},
        )
    )

    visual_config = workflow_config.get(
        "visualizations",
        {},
    )

    metric_names = visual_config.get(
        "metrics",
        [
            "AUROC",
            "AUPRC",
            "Accuracy",
            "F1_macro",
        ],
    )

    written: list[Path] = []

    grouped = aggregate_metrics.groupby(
        [
            "Target",
            "BiasFactor",
        ],
        dropna=False,
    )

    for (
        target_name,
        bias_factor,
    ), group_df in grouped:
        for metric_name in metric_names:
            metric_column = (
                f"{metric_name}_mean"
            )

            if metric_column not in group_df.columns:
                continue

            usable = pd.to_numeric(
                group_df[metric_column],
                errors="coerce",
            ).notna().any()

            if not usable:
                continue

            bar_path = (
                output_dir
                / "subgroup_performance"
                / safe_name(target_name)
                / (
                    f"{safe_name(bias_factor)}__"
                    f"{safe_name(metric_name)}__bar.png"
                )
            )

            plot_subgroup_metric_bars(
                dataframe=group_df,
                target_name=str(target_name),
                bias_factor=str(bias_factor),
                metric_name=str(metric_name),
                output_path=bar_path,
            )
            written.append(bar_path)

            line_path = (
                output_dir
                / "subgroup_performance"
                / safe_name(target_name)
                / (
                    f"{safe_name(bias_factor)}__"
                    f"{safe_name(metric_name)}__line.png"
                )
            )

            plot_subgroup_metric_lines(
                dataframe=group_df,
                target_name=str(target_name),
                bias_factor=str(bias_factor),
                metric_name=str(metric_name),
                output_path=line_path,
            )
            written.append(line_path)

        if visual_config.get(
            "make_starplots",
            True,
        ):
            star_path = (
                output_dir
                / "starplots"
                / (
                    f"{safe_name(target_name)}__"
                    f"{safe_name(bias_factor)}.png"
                )
            )

            generated = plot_subgroup_starplot(
                dataframe=group_df,
                target_name=str(target_name),
                bias_factor=str(bias_factor),
                metrics=[
                    str(value)
                    for value in visual_config.get(
                        "starplot_metrics",
                        [
                            "AUROC",
                            "AUPRC",
                            "Accuracy",
                            "F1_macro",
                            "Sensitivity",
                            "Specificity",
                        ],
                    )
                ],
                output_path=star_path,
            )

            if generated:
                written.append(star_path)

    if (
        not disparity_summary.empty
        and visual_config.get(
            "disparity_plots",
            True,
        )
    ):
        for target_name, target_df in (
            disparity_summary.groupby(
                "Target",
                dropna=False,
            )
        ):
            disparity_path = (
                output_dir
                / "disparities"
                / (
                    f"{safe_name(target_name)}__"
                    "absolute_gaps.png"
                )
            )

            plot_disparity_gaps(
                dataframe=target_df,
                target_name=str(target_name),
                output_path=disparity_path,
            )
            written.append(disparity_path)

    return written


def plot_subgroup_metric_bars(
    *,
    dataframe: pd.DataFrame,
    target_name: str,
    bias_factor: str,
    metric_name: str,
    output_path: str | Path,
) -> Path:
    """Plot subgroup metric means with split standard deviations."""
    metric_mean = f"{metric_name}_mean"
    metric_std = f"{metric_name}_std"

    output_path = Path(
        output_path
    ).expanduser().resolve()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    models = (
        dataframe["Model"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    subgroups = (
        dataframe["Subgroup"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    x_positions = np.arange(
        len(models),
        dtype=float,
    )

    total_width = 0.82
    bar_width = total_width / max(
        1,
        len(subgroups),
    )

    fig, ax = plt.subplots(
        figsize=(
            max(7.0, 1.6 * len(models)),
            5.3,
        )
    )

    for subgroup_index, subgroup in enumerate(
        subgroups
    ):
        subgroup_df = dataframe[
            dataframe["Subgroup"].astype(str)
            == subgroup
        ]

        means = []
        errors = []

        for model in models:
            model_rows = subgroup_df[
                subgroup_df["Model"].astype(str)
                == model
            ]

            means.append(
                pd.to_numeric(
                    model_rows.get(
                        metric_mean,
                        pd.Series(dtype=float),
                    ),
                    errors="coerce",
                ).mean()
            )

            errors.append(
                pd.to_numeric(
                    model_rows.get(
                        metric_std,
                        pd.Series(dtype=float),
                    ),
                    errors="coerce",
                ).mean()
            )

        offset = (
            subgroup_index
            - (len(subgroups) - 1) / 2
        ) * bar_width

        ax.bar(
            x_positions + offset,
            means,
            width=bar_width * 0.92,
            yerr=errors,
            capsize=3,
            color=SUBGROUP_COLORS[
                subgroup_index
                % len(SUBGROUP_COLORS)
            ],
            alpha=0.86,
            label=subgroup,
        )

    add_metric_references(
        ax=ax,
        dataframe=dataframe,
        metric_name=metric_name,
    )

    ax.set_ylim(0, 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        models,
        rotation=25,
        ha="right",
    )
    ax.set_ylabel(metric_name)
    ax.set_xlabel("Model")
    ax.set_title(
        f"{target_name}: {metric_name} by {bias_factor}"
    )
    ax.legend(
        frameon=False,
        title=bias_factor,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=350,
        bbox_inches="tight",
    )
    plt.close(fig)

    return output_path


def plot_subgroup_metric_lines(
    *,
    dataframe: pd.DataFrame,
    target_name: str,
    bias_factor: str,
    metric_name: str,
    output_path: str | Path,
) -> Path:
    """Plot subgroup trajectories across models."""
    metric_column = f"{metric_name}_mean"

    output_path = Path(
        output_path
    ).expanduser().resolve()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    models = (
        dataframe["Model"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    subgroups = (
        dataframe["Subgroup"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    fig, ax = plt.subplots(
        figsize=(
            max(7.0, 1.4 * len(models)),
            5.2,
        )
    )

    x_positions = np.arange(
        len(models)
    )

    for subgroup_index, subgroup in enumerate(
        subgroups
    ):
        subgroup_df = dataframe[
            dataframe["Subgroup"].astype(str)
            == subgroup
        ]

        values = []

        for model in models:
            model_values = pd.to_numeric(
                subgroup_df.loc[
                    subgroup_df["Model"].astype(str)
                    == model,
                    metric_column,
                ],
                errors="coerce",
            )

            values.append(
                model_values.mean()
            )

        ax.plot(
            x_positions,
            values,
            marker="o",
            linewidth=1.6,
            markersize=5,
            color=SUBGROUP_COLORS[
                subgroup_index
                % len(SUBGROUP_COLORS)
            ],
            label=subgroup,
        )

    add_metric_references(
        ax=ax,
        dataframe=dataframe,
        metric_name=metric_name,
    )

    ax.set_ylim(0, 1)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        models,
        rotation=25,
        ha="right",
    )
    ax.set_ylabel(metric_name)
    ax.set_xlabel("Model")
    ax.set_title(
        f"{target_name}: subgroup trajectories by {bias_factor}"
    )
    ax.legend(
        frameon=False,
        title=bias_factor,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=350,
        bbox_inches="tight",
    )
    plt.close(fig)

    return output_path


def plot_subgroup_starplot(
    *,
    dataframe: pd.DataFrame,
    target_name: str,
    bias_factor: str,
    metrics: list[str],
    output_path: str | Path,
) -> bool:
    """Create radar/star plots for each model and subgroup."""
    available_metrics = [
        metric
        for metric in metrics
        if f"{metric}_mean"
        in dataframe.columns
    ]

    if len(available_metrics) < 3:
        return False

    models = (
        dataframe["Model"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not models:
        return False

    output_path = Path(
        output_path
    ).expanduser().resolve()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    n_columns = min(
        3,
        len(models),
    )
    n_rows = int(
        np.ceil(
            len(models) / n_columns
        )
    )

    angles = np.linspace(
        0,
        2 * np.pi,
        len(available_metrics),
        endpoint=False,
    )
    closed_angles = np.concatenate(
        [angles, angles[:1]]
    )

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(
            4.5 * n_columns,
            4.4 * n_rows,
        ),
        subplot_kw={
            "projection": "polar"
        },
        squeeze=False,
    )

    for model_index, model in enumerate(
        models
    ):
        row = model_index // n_columns
        column = model_index % n_columns
        ax = axes[row][column]

        model_df = dataframe[
            dataframe["Model"].astype(str)
            == model
        ]

        subgroups = (
            model_df["Subgroup"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        for subgroup_index, subgroup in enumerate(
            subgroups
        ):
            subgroup_df = model_df[
                model_df["Subgroup"].astype(str)
                == subgroup
            ]

            values = [
                pd.to_numeric(
                    subgroup_df[
                        f"{metric}_mean"
                    ],
                    errors="coerce",
                ).mean()
                for metric in available_metrics
            ]

            if all(
                pd.isna(value)
                for value in values
            ):
                continue

            values = [
                0.0
                if pd.isna(value)
                else float(value)
                for value in values
            ]

            closed_values = values + [
                values[0]
            ]

            color = SUBGROUP_COLORS[
                subgroup_index
                % len(SUBGROUP_COLORS)
            ]

            ax.plot(
                closed_angles,
                closed_values,
                linewidth=1.5,
                color=color,
                label=subgroup,
            )
            ax.fill(
                closed_angles,
                closed_values,
                color=color,
                alpha=0.10,
            )

        ax.set_xticks(angles)
        ax.set_xticklabels(
            available_metrics,
            fontsize=7,
        )
        ax.set_ylim(0, 1)
        ax.set_yticks(
            [0.25, 0.5, 0.75, 1.0]
        )
        ax.set_title(
            model,
            pad=15,
        )
        ax.legend(
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(1.35, 1.15),
        )

    for model_index in range(
        len(models),
        n_rows * n_columns,
    ):
        row = model_index // n_columns
        column = model_index % n_columns
        axes[row][column].set_visible(False)

    fig.suptitle(
        f"{target_name}: subgroup starplots by {bias_factor}",
        fontsize=11,
        y=0.995,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.97]
    )
    fig.savefig(
        output_path,
        dpi=350,
        bbox_inches="tight",
    )
    plt.close(fig)

    return True


def plot_disparity_gaps(
    *,
    dataframe: pd.DataFrame,
    target_name: str,
    output_path: str | Path,
) -> Path:
    """Plot best-minus-worst subgroup gaps."""
    output_path = Path(
        output_path
    ).expanduser().resolve()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_df = dataframe.copy()
    plot_df["Label"] = (
        plot_df["Model"].astype(str)
        + " | "
        + plot_df["BiasFactor"].astype(str)
        + " | "
        + plot_df["Metric"].astype(str)
    )

    plot_df = plot_df.sort_values(
        "AbsoluteGap",
        ascending=True,
    )

    fig_height = max(
        5.0,
        0.32 * len(plot_df),
    )

    fig, ax = plt.subplots(
        figsize=(9.5, fig_height)
    )

    colors = [
        SUBGROUP_COLORS[
            index % len(SUBGROUP_COLORS)
        ]
        for index in range(len(plot_df))
    ]

    ax.barh(
        plot_df["Label"],
        plot_df["AbsoluteGap"],
        color=colors,
        alpha=0.86,
    )

    ax.set_xlim(
        0,
        max(
            0.05,
            min(
                1.0,
                float(
                    plot_df[
                        "AbsoluteGap"
                    ].max()
                )
                * 1.1,
            ),
        ),
    )
    ax.set_xlabel(
        "Best subgroup minus worst subgroup"
    )
    ax.set_ylabel("")
    ax.set_title(
        f"{target_name}: subgroup performance gaps"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=350,
        bbox_inches="tight",
    )
    plt.close(fig)

    return output_path


def add_metric_references(
    *,
    ax: plt.Axes,
    dataframe: pd.DataFrame,
    metric_name: str,
) -> None:
    """Add old-workflow metric reference lines."""
    ax.axhline(
        0,
        linestyle="--",
        linewidth=0.8,
        color="black",
    )
    ax.axhline(
        0.5,
        linestyle="--",
        linewidth=0.8,
        color="gray",
        label="Reference = 0.5",
    )

    if metric_name == "AUPRC":
        prevalence = pd.to_numeric(
            dataframe.get(
                "Pos_Prevalence_mean",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        ).mean()

        if np.isfinite(prevalence):
            ax.axhline(
                prevalence,
                linestyle="--",
                linewidth=1.0,
                color="#E94E77",
                label=(
                    "Mean subgroup prevalence "
                    f"= {prevalence:.2f}"
                ),
            )


def safe_name(
    value: object,
) -> str:
    """Return a filesystem-safe component."""
    text = str(value).strip()

    for character in (
        " ",
        "/",
        "\\",
        ":",
        ";",
        ",",
        "|",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
    ):
        text = text.replace(
            character,
            "_",
        )

    while "__" in text:
        text = text.replace(
            "__",
            "_",
        )

    return text.strip("_") or "unnamed"