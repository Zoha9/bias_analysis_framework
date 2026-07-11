"""Visualizations for regular prediction performance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_COLORS = [
    "#2AB7CA",
    "#E94E77",
    "#FDB515",
    "#76B041",
    "#6A5ACD",
    "#A950B4",
    "#4A5FE1",
    "#964B36",
]


def set_model_evaluation_style() -> None:
    """Apply plotting settings aligned with the old workflow."""
    mpl.rcParams.update(
        {
            "figure.dpi": 350,
            "savefig.dpi": 350,
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.7,
        }
    )


def generate_regular_performance_figures(
    *,
    split_metrics: pd.DataFrame,
    patient_scores: pd.DataFrame,
    config: dict[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    """Generate regular model-performance figures."""
    set_model_evaluation_style()

    output_dir = (
        Path(output_dir)
        .expanduser()
        .resolve()
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if split_metrics.empty:
        return []

    workflow_config = (
        config.get("workflows", {})
        .get(
            "regular_model_prediction_analysis",
            {},
        )
    )

    visualization_config = workflow_config.get(
        "visualizations",
        {},
    )

    metric_names = visualization_config.get(
        "metrics",
        [
            "Accuracy",
            "F1_macro",
            "AUROC",
            "AUPRC",
        ],
    )

    written: list[Path] = []

    for target_name in (
        split_metrics["Target"]
        .dropna()
        .unique()
    ):
        target_df = split_metrics[
            split_metrics["Target"]
            == target_name
        ].copy()

        if target_df.empty:
            continue

        for metric_name in metric_names:
            if metric_name not in target_df.columns:
                continue

            if pd.to_numeric(
                target_df[metric_name],
                errors="coerce",
            ).notna().sum() == 0:
                continue

            output_path = (
                output_dir
                / "target_performance"
                / (
                    f"{safe_name(target_name)}__"
                    f"{safe_name(metric_name)}.png"
                )
            )

            plot_metric_across_models(
                dataframe=target_df,
                target_name=str(target_name),
                metric_name=str(metric_name),
                output_path=output_path,
            )
            written.append(output_path)

    if (
        not patient_scores.empty
        and visualization_config.get(
            "patient_probability_plots",
            True,
        )
    ):
        for (
            target_name,
            model_name,
        ), group_df in patient_scores.groupby(
            [
                "Target",
                "Model",
            ],
            dropna=False,
        ):
            output_path = (
                output_dir
                / "patient_scores"
                / (
                    f"{safe_name(target_name)}__"
                    f"{safe_name(model_name)}.png"
                )
            )

            plot_patient_probabilities(
                dataframe=group_df,
                target_name=str(target_name),
                model_name=str(model_name),
                output_path=output_path,
            )
            written.append(output_path)

    return written


def plot_metric_across_models(
    *,
    dataframe: pd.DataFrame,
    target_name: str,
    metric_name: str,
    output_path: str | Path,
) -> Path:
    """Plot split-level metric distributions across models."""
    output_path = (
        Path(output_path)
        .expanduser()
        .resolve()
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_names = (
        dataframe["Model"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    fig, ax = plt.subplots(
        figsize=(
            max(6.0, 1.25 * len(model_names)),
            5.0,
        )
    )

    rng = np.random.default_rng(42)

    for model_index, model_name in enumerate(
        model_names
    ):
        model_values = pd.to_numeric(
            dataframe.loc[
                dataframe["Model"]
                == model_name,
                metric_name,
            ],
            errors="coerce",
        ).dropna()

        if model_values.empty:
            continue

        x_values = np.full(
            len(model_values),
            model_index,
            dtype=float,
        )
        x_values += rng.normal(
            0.0,
            0.045,
            size=len(model_values),
        )

        color = MODEL_COLORS[
            model_index % len(MODEL_COLORS)
        ]

        ax.scatter(
            x_values,
            model_values,
            s=30,
            alpha=0.82,
            color=color,
            edgecolors="none",
            label=model_name,
        )

        mean_value = float(
            model_values.mean()
        )
        std_value = float(
            model_values.std(ddof=1)
        )

        ax.errorbar(
            model_index,
            mean_value,
            yerr=(
                std_value
                if np.isfinite(std_value)
                else None
            ),
            marker="D",
            markersize=5,
            color="black",
            capsize=4,
            linewidth=1.1,
        )

    ax.set_ylim(
        0.0,
        1.0,
    )
    ax.axhline(
        0.5,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Reference = 0.5",
    )

    if metric_name == "AUPRC":
        prevalence = pd.to_numeric(
            dataframe.get(
                "Pos_Prevalence",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        ).mean()

        if np.isfinite(prevalence):
            ax.axhline(
                prevalence,
                color="#E94E77",
                linestyle="--",
                linewidth=1,
                label=(
                    "Baseline prevalence "
                    f"= {prevalence:.2f}"
                ),
            )

    if metric_name in {
        "Accuracy",
        "Balanced_Accuracy",
    }:
        baseline = pd.to_numeric(
            dataframe.get(
                "Baseline_Accuracy",
                pd.Series(dtype=float),
            ),
            errors="coerce",
        ).mean()

        if np.isfinite(baseline):
            ax.axhline(
                baseline,
                color="#FDB515",
                linestyle=":",
                linewidth=1,
                label=(
                    "Majority baseline "
                    f"= {baseline:.2f}"
                ),
            )

    ax.set_xticks(
        range(len(model_names))
    )
    ax.set_xticklabels(
        model_names,
        rotation=30,
        ha="right",
    )
    ax.set_ylabel(metric_name)
    ax.set_xlabel("Model")
    ax.set_title(
        f"{target_name}: {metric_name}"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles, labels = (
        ax.get_legend_handles_labels()
    )

    reference_handles = [
        (handle, label)
        for handle, label in zip(
            handles,
            labels,
        )
        if "Reference" in label
        or "Baseline" in label
        or "Majority" in label
    ]

    if reference_handles:
        ax.legend(
            handles=[
                item[0]
                for item in reference_handles
            ],
            labels=[
                item[1]
                for item in reference_handles
            ],
            frameon=False,
            loc="lower right",
        )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=350,
        bbox_inches="tight",
    )
    plt.close(fig)

    return output_path


def plot_patient_probabilities(
    *,
    dataframe: pd.DataFrame,
    target_name: str,
    model_name: str,
    output_path: str | Path,
) -> Path:
    """Plot patient-level positive-class probabilities."""
    output_path = (
        Path(output_path)
        .expanduser()
        .resolve()
    )
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_df = dataframe.dropna(
        subset=[
            "True",
            "PosProb",
        ]
    ).copy()

    class_names = (
        plot_df["True"]
        .astype(str)
        .value_counts()
        .index
        .tolist()
    )

    fig, ax = plt.subplots(
        figsize=(
            max(6.0, 1.2 * len(class_names)),
            5.0,
        )
    )

    rng = np.random.default_rng(42)

    for class_index, class_name in enumerate(
        class_names
    ):
        class_df = plot_df[
            plot_df["True"].astype(str)
            == class_name
        ]

        probabilities = pd.to_numeric(
            class_df["PosProb"],
            errors="coerce",
        ).dropna()

        x_values = np.full(
            len(probabilities),
            class_index,
            dtype=float,
        )
        x_values += rng.normal(
            0.0,
            0.055,
            size=len(probabilities),
        )

        ax.scatter(
            x_values,
            probabilities,
            s=16,
            alpha=0.6,
            color=MODEL_COLORS[
                class_index
                % len(MODEL_COLORS)
            ],
            edgecolors="none",
        )

        if not probabilities.empty:
            ax.plot(
                [
                    class_index - 0.18,
                    class_index + 0.18,
                ],
                [
                    probabilities.median(),
                    probabilities.median(),
                ],
                color="black",
                linewidth=1.4,
            )

    ax.axhline(
        0.5,
        color="gray",
        linestyle="--",
        linewidth=1,
        label="Decision threshold = 0.5",
    )

    ax.set_ylim(
        0,
        1,
    )
    ax.set_xticks(
        range(len(class_names))
    )
    ax.set_xticklabels(
        [
            (
                f"{class_name}\n"
                f"(n={int((plot_df['True'].astype(str) == class_name).sum())})"
            )
            for class_name in class_names
        ]
    )
    ax.set_ylabel(
        "Positive-class probability"
    )
    ax.set_xlabel(
        "True class"
    )
    ax.set_title(
        f"{target_name}: patient scores, {model_name}"
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=False,
        loc="best",
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=350,
        bbox_inches="tight",
    )
    plt.close(fig)

    return output_path


def safe_name(value: object) -> str:
    """Create a filesystem-safe component."""
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