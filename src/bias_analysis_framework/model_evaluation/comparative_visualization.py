"""Visualizations for comparative bias propagation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_comparative_plot_style() -> None:
    """Apply project-consistent plotting defaults."""
    mpl.rcParams.update(
        {
            "figure.dpi": 350,
            "savefig.dpi": 350,
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )


def generate_comparative_bias_figures(
    *,
    aggregate_metrics: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    """Generate delta heatmaps and before-after comparison plots."""
    set_comparative_plot_style()

    output_dir = (
        Path(output_dir)
        .expanduser()
        .resolve()
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if aggregate_metrics.empty:
        return []

    workflow_config = (
        config.get("workflows", {})
        .get(
            "comparative_bias_propagation_analysis",
            {},
        )
    )

    visual_config = workflow_config.get(
        "visualizations",
        {},
    )

    delta_metrics = visual_config.get(
        "delta_metrics",
        [
            "Delta_PositiveRateGap_mean",
            "Delta_DI_Distance_mean",
        ],
    )

    written: list[Path] = []

    for target_name, target_df in (
        aggregate_metrics.groupby(
            "Target",
            dropna=False,
        )
    ):
        for metric_column in delta_metrics:
            if metric_column not in target_df.columns:
                continue

            output_path = (
                output_dir
                / "delta_heatmaps"
                / (
                    f"{safe_name(target_name)}__"
                    f"{safe_name(metric_column)}.png"
                )
            )

            generated = plot_delta_heatmap(
                dataframe=target_df,
                target_name=str(target_name),
                metric_column=str(metric_column),
                output_path=output_path,
            )

            if generated:
                written.append(output_path)

        comparison_path = (
            output_dir
            / "before_after"
            / (
                f"{safe_name(target_name)}__"
                "positive_rate_gaps.png"
            )
        )

        plot_before_after_gaps(
            dataframe=target_df,
            target_name=str(target_name),
            output_path=comparison_path,
        )
        written.append(comparison_path)

    return written


def plot_delta_heatmap(
    *,
    dataframe: pd.DataFrame,
    target_name: str,
    metric_column: str,
    output_path: str | Path,
) -> bool:
    """Plot model-by-factor disparity changes."""
    pivot = dataframe.pivot_table(
        index="BiasFactor",
        columns="Model",
        values=metric_column,
        aggfunc="mean",
    )

    if pivot.empty:
        return False

    output_path = (
        Path(output_path)
        .expanduser()
        .resolve()
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    values = pivot.to_numpy(
        dtype=float
    )

    finite_values = values[
        np.isfinite(values)
    ]

    if finite_values.size == 0:
        return False

    maximum = max(
        float(
            np.nanmax(
                np.abs(finite_values)
            )
        ),
        0.01,
    )

    fig, ax = plt.subplots(
        figsize=(
            max(6.0, 1.15 * len(pivot.columns)),
            max(4.0, 0.65 * len(pivot.index)),
        )
    )

    image = ax.imshow(
        values,
        cmap="coolwarm",
        vmin=-maximum,
        vmax=maximum,
        aspect="auto",
    )

    ax.set_xticks(
        range(len(pivot.columns))
    )
    ax.set_xticklabels(
        pivot.columns,
        rotation=30,
        ha="right",
    )
    ax.set_yticks(
        range(len(pivot.index))
    )
    ax.set_yticklabels(
        pivot.index
    )

    for row_index in range(
        values.shape[0]
    ):
        for column_index in range(
            values.shape[1]
        ):
            value = values[
                row_index,
                column_index,
            ]

            if not np.isfinite(value):
                label = "NA"
            else:
                label = f"{value:.3f}"

            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=7,
            )

    ax.set_title(
        f"{target_name}: {metric_column}"
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Bias factor")

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    colorbar.set_label(
        "Posttraining minus pretraining\n"
        "Positive = amplification"
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=350,
        bbox_inches="tight",
    )
    plt.close(fig)

    return True


def plot_before_after_gaps(
    *,
    dataframe: pd.DataFrame,
    target_name: str,
    output_path: str | Path,
) -> Path:
    """Plot pretraining and posttraining positive-rate gaps."""
    output_path = (
        Path(output_path)
        .expanduser()
        .resolve()
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_df = dataframe.copy()

    plot_df["Label"] = (
        plot_df["Model"].astype(str)
        + " | "
        + plot_df["BiasFactor"].astype(str)
    )

    plot_df = plot_df.sort_values(
        "Label"
    ).reset_index(
        drop=True
    )

    y_positions = np.arange(
        len(plot_df)
    )

    before = pd.to_numeric(
        plot_df[
            "Pretraining_PositiveRateGap_mean"
        ],
        errors="coerce",
    )
    after = pd.to_numeric(
        plot_df[
            "Posttraining_PositiveRateGap_mean"
        ],
        errors="coerce",
    )

    fig, ax = plt.subplots(
        figsize=(
            9.0,
            max(
                5.0,
                0.38 * len(plot_df),
            ),
        )
    )

    for index in range(
        len(plot_df)
    ):
        if (
            pd.isna(before.iloc[index])
            or pd.isna(after.iloc[index])
        ):
            continue

        ax.plot(
            [
                before.iloc[index],
                after.iloc[index],
            ],
            [
                y_positions[index],
                y_positions[index],
            ],
            color="gray",
            linewidth=1.0,
        )

    ax.scatter(
        before,
        y_positions,
        label="Pretraining",
        marker="o",
        s=30,
    )
    ax.scatter(
        after,
        y_positions,
        label="Posttraining",
        marker="D",
        s=30,
    )

    ax.set_xlim(0, 1)
    ax.set_yticks(
        y_positions
    )
    ax.set_yticklabels(
        plot_df["Label"]
    )
    ax.set_xlabel(
        "Subgroup positive-rate gap"
    )
    ax.set_ylabel("")
    ax.set_title(
        f"{target_name}: before and after training"
    )
    ax.legend(
        frameon=False
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


def safe_name(
    value: object,
) -> str:
    """Create a filesystem-safe path component."""
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