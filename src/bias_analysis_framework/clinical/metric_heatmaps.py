from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bias_analysis_framework.metrics.clinical_pretraining_metrics import parse_section_title


MetricTransform = Callable[[float], float]


METRIC_SPECS: dict[str, dict[str, Any]] = {
    "statistical_parity_difference_abs": {
        "metric_names": ["Statistical Parity Difference"],
        "title": "Absolute statistical parity difference",
        "colorbar_label": "|SPD|",
        "filename": "heatmap__absolute_statistical_parity_difference.png",
        "transform": "abs",
        "vmin": 0,
        "vmax": 1,
    },
    "conditional_demographic_disparity": {
        "metric_names": ["Conditional Demographic Disparity"],
        "title": "Conditional demographic disparity",
        "colorbar_label": "CDDL",
        "filename": "heatmap__conditional_demographic_disparity.png",
        "transform": "identity",
        "vmin": 0,
        "vmax": 1,
    },
    "difference_in_positive_proportions": {
        "metric_names": ["Difference in Positive Proportions"],
        "title": "Difference in positive proportions",
        "colorbar_label": "DPL",
        "filename": "heatmap__difference_in_positive_proportions.png",
        "transform": "identity",
        "vmin": 0,
        "vmax": 1,
    },
    "max_pairwise_spd_abs": {
        "metric_names": ["Max Pairwise SPD (abs)"],
        "title": "Maximum pairwise absolute SPD",
        "colorbar_label": "Max pairwise |SPD|",
        "filename": "heatmap__max_pairwise_spd_abs.png",
        "transform": "identity",
        "vmin": 0,
        "vmax": 1,
    },
    "disparate_impact_ratio": {
        "metric_names": ["Disparate Impact (min/max)"],
        "title": "Disparate impact ratio",
        "colorbar_label": "DI ratio",
        "filename": "heatmap__disparate_impact_ratio.png",
        "transform": "identity",
        "vmin": 0,
        "vmax": 1,
    },
    "disparate_impact_log2_deviation": {
        "metric_names": ["Disparate Impact (min/max)"],
        "title": "Disparate impact deviation from parity",
        "colorbar_label": "|log2(DI)|",
        "filename": "heatmap__disparate_impact_log2_deviation.png",
        "transform": "di_log2_abs",
        "vmin": 0,
        "vmax": None,
    },
    "coefficient_of_variation_positive_rates": {
        "metric_names": ["Coefficient of Variation (pos rates)"],
        "title": "Coefficient of variation across positive rates",
        "colorbar_label": "CV",
        "filename": "heatmap__coefficient_of_variation_positive_rates.png",
        "transform": "identity",
        "vmin": 0,
        "vmax": None,
    },
    "range_of_positive_rates": {
        "metric_names": ["Range of Positive Rates"],
        "title": "Range of positive rates",
        "colorbar_label": "Range",
        "filename": "heatmap__range_of_positive_rates.png",
        "transform": "identity",
        "vmin": 0,
        "vmax": 1,
    },
    "gini_across_groups_positive_rates": {
        "metric_names": ["Gini across Groups (pos rates)"],
        "title": "Gini across subgroup positive rates",
        "colorbar_label": "Gini",
        "filename": "heatmap__gini_across_groups_positive_rates.png",
        "transform": "identity",
        "vmin": 0,
        "vmax": 1,
    },
    "cramers_v": {
        "metric_names": ["Cramér's V (group x label)"],
        "title": "Cramér's V association",
        "colorbar_label": "Cramér's V",
        "filename": "heatmap__cramers_v.png",
        "transform": "identity",
        "vmin": 0,
        "vmax": 1,
    },
    "chi_square_significance": {
        "metric_names": ["Chi-square p-value"],
        "title": "Chi-square significance",
        "colorbar_label": "-log10(p)",
        "filename": "heatmap__chi_square_significance.png",
        "transform": "negative_log10",
        "vmin": 0,
        "vmax": None,
    },
    "jsd_to_centroid": {
        "metric_names": ["JSD_mean_to_centroid", "JSD mean to centroid"],
        "title": "Jensen-Shannon divergence to centroid",
        "colorbar_label": "JSD to centroid",
        "filename": "heatmap__jsd_to_centroid.png",
        "transform": "identity",
        "vmin": 0,
        "vmax": None,
    },
    "tvd_to_centroid": {
        "metric_names": ["TVD_mean_to_centroid", "TVD mean to centroid"],
        "title": "Total variation distance to centroid",
        "colorbar_label": "TVD to centroid",
        "filename": "heatmap__tvd_to_centroid.png",
        "transform": "identity",
        "vmin": 0,
        "vmax": None,
    },
}


def generate_clinical_metric_heatmaps(
    section_map: dict[str, pd.DataFrame],
    output_dir: str | Path,
    config: dict[str, Any],
) -> list[Path]:
    """Generate heatmaps from Step 1 clinical pretraining metric tables.

    Parameters
    ----------
    section_map:
        Output from compute_clinical_pretraining_metrics().
    output_dir:
        Directory where metric heatmaps should be written.
    config:
        User workflow config.

    Returns
    -------
    list[pathlib.Path]
        Paths of generated heatmap figures.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    heatmap_dir = output_dir / "clinical_pretraining_bias_metric_heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    metric_long_df = extract_metric_long_table(section_map)

    if metric_long_df.empty:
        return []

    written: list[Path] = []

    for metric_key, metric_spec in METRIC_SPECS.items():
        matrix = build_metric_matrix(
            metric_long_df=metric_long_df,
            metric_names=metric_spec["metric_names"],
            transform=metric_spec["transform"],
            config=config,
        )

        if matrix.empty:
            continue

        output_path = heatmap_dir / metric_spec["filename"]

        plot_metric_heatmap(
            matrix=matrix,
            title=metric_spec["title"],
            colorbar_label=metric_spec["colorbar_label"],
            output_path=output_path,
            vmin=metric_spec.get("vmin"),
            vmax=metric_spec.get("vmax"),
        )

        written.append(output_path)

    return written


def extract_metric_long_table(
    section_map: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Extract metric rows from sectioned pretraining metric tables.

    The returned table has one row per:
    target, class, factor, section, metric, value.

    Subgroup count/rate rows are intentionally excluded because these heatmaps
    are metric heatmaps. Subgroup counts are visualized by clinical/visualization.py.
    """
    rows = []

    for section_title, section_df in section_map.items():
        if not isinstance(section_df, pd.DataFrame):
            continue

        if "Metric" not in section_df.columns or "Value" not in section_df.columns:
            continue

        parsed = parse_section_title(section_title)

        usable = section_df.copy()
        usable = usable[usable["Metric"].notna()]
        usable = usable[usable["Metric"].astype(str).str.strip() != ""]
        usable = usable[usable["Metric"].astype(str).str.strip() != "—"]

        if "Subgroup" in usable.columns:
            usable = usable[usable["Subgroup"].isna()]

        for _, row in usable.iterrows():
            metric_name = str(row["Metric"]).strip()
            value = pd.to_numeric(row["Value"], errors="coerce")

            if pd.isna(value):
                continue

            rows.append(
                {
                    "target": parsed.get("Target", ""),
                    "class": parsed.get("Class", "") or "ALL",
                    "factor": parsed.get("Factor", ""),
                    "section": parsed.get("Section", ""),
                    "metric": metric_name,
                    "value": float(value),
                }
            )

    return pd.DataFrame(
        rows,
        columns=["target", "class", "factor", "section", "metric", "value"],
    )


def build_metric_matrix(
    metric_long_df: pd.DataFrame,
    metric_names: list[str],
    transform: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Build a heatmap matrix for one metric family.

    Rows are bias factors.
    Columns are target/class combinations.
    """
    metric_df = metric_long_df[metric_long_df["metric"].isin(metric_names)].copy()

    if metric_df.empty:
        return pd.DataFrame()

    metric_df["value"] = metric_df["value"].map(lambda value: transform_metric_value(value, transform))
    metric_df = metric_df.replace([np.inf, -np.inf], np.nan)
    metric_df = metric_df.dropna(subset=["value"])

    if metric_df.empty:
        return pd.DataFrame()

    metric_df["column_label"] = metric_df.apply(make_target_class_label, axis=1)

    matrix = metric_df.pivot_table(
        index="factor",
        columns="column_label",
        values="value",
        aggfunc="max",
    )

    matrix = order_metric_matrix(matrix, config)

    return matrix


def transform_metric_value(
    value: float,
    transform: str,
) -> float:
    """Apply metric-specific transformations for visualization."""
    if pd.isna(value):
        return np.nan

    value = float(value)

    if transform == "identity":
        return value

    if transform == "abs":
        return abs(value)

    if transform == "negative_log10":
        if value <= 0:
            value = 1e-300
        return -math.log10(value)

    if transform == "di_log2_abs":
        if value <= 0:
            return np.nan
        return abs(math.log2(value))

    raise ValueError(f"Unknown metric transform: {transform}")


def make_target_class_label(row: pd.Series) -> str:
    """Create a readable target/class column label."""
    target = str(row.get("target", "")).strip()
    class_name = str(row.get("class", "")).strip()

    if class_name in {"", "ALL"}:
        return f"{target} [ALL]"

    if class_name == "MULTICLASS":
        return f"{target} [MULTICLASS]"

    return f"{target} [{class_name}]"


def order_metric_matrix(
    matrix: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Order heatmap rows and columns according to config."""
    if matrix.empty:
        return matrix

    row_order = []
    for factor in config.get("bias_factors", []):
        factor_name = factor.get("name")
        if factor_name and factor_name in matrix.index:
            row_order.append(factor_name)

    for extra_factor in ["Tissue Source Site", "Age (continuous)"]:
        if extra_factor in matrix.index and extra_factor not in row_order:
            row_order.append(extra_factor)

    remaining_rows = [row for row in matrix.index if row not in row_order]

    column_order = []
    for target in config.get("targets", []):
        target_name = target.get("name")
        if not target_name:
            continue

        if target.get("labels"):
            labels = target["labels"]
            for label in labels:
                column_order.append(f"{target_name} [{label}]")
            column_order.append(f"{target_name} [MULTICLASS]")

        elif target.get("negative_label") is not None and target.get("positive_label") is not None:
            column_order.append(f"{target_name} [ALL]")

        else:
            column_order.append(f"{target_name} [ALL]")

    existing_columns = [column for column in column_order if column in matrix.columns]
    remaining_columns = [column for column in matrix.columns if column not in existing_columns]

    return matrix.loc[row_order + remaining_rows, existing_columns + remaining_columns]


def plot_metric_heatmap(
    matrix: pd.DataFrame,
    title: str,
    colorbar_label: str,
    output_path: str | Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    """Plot and save one metric heatmap."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if matrix.empty:
        return output_path

    matrix = matrix.copy()
    matrix = matrix.replace([np.inf, -np.inf], np.nan)

    if vmax is None:
        vmax = robust_vmax(matrix)

    fig_width = max(9, min(30, 1.1 * matrix.shape[1] + 5))
    fig_height = max(5.5, min(24, 0.65 * matrix.shape[0] + 4))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    masked = np.ma.masked_invalid(matrix.values.astype(float))

    image = ax.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))

    ax.set_xticklabels(
        [wrap_label(label, width=16) for label in matrix.columns],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_yticklabels(
        [wrap_label(label, width=24) for label in matrix.index],
        fontsize=9,
    )

    ax.set_xlabel("Target biomarker class")
    ax.set_ylabel("Bias factor")
    ax.set_title(title)

    add_heatmap_grid(ax, matrix)
    annotate_metric_heatmap(ax, matrix)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def robust_vmax(
    matrix: pd.DataFrame,
    percentile: float = 95,
) -> float | None:
    """Choose a robust heatmap upper bound when a fixed vmax is not provided."""
    values = matrix.to_numpy(dtype=float).ravel()
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None

    vmax = float(np.nanpercentile(values, percentile))

    if vmax <= 0:
        return None

    return vmax


def annotate_metric_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    max_cells: int = 180,
) -> None:
    """Annotate heatmap cells with compact numeric values."""
    n_rows, n_cols = matrix.shape

    if n_rows * n_cols > max_cells:
        return

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            value = matrix.iat[row_idx, col_idx]

            if pd.isna(value):
                continue

            ax.text(
                col_idx,
                row_idx,
                format_metric_value(value),
                ha="center",
                va="center",
                fontsize=7,
            )


def format_metric_value(value: float) -> str:
    """Format metric values for heatmap annotations."""
    if pd.isna(value):
        return ""

    value = float(value)

    if abs(value) >= 100:
        return f"{value:.0f}"

    if abs(value) >= 10:
        return f"{value:.1f}"

    return f"{value:.2f}"


def add_heatmap_grid(
    ax: plt.Axes,
    matrix: pd.DataFrame,
) -> None:
    """Add subtle grid lines to heatmap cells."""
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)


def wrap_label(
    label: object,
    width: int = 18,
) -> str:
    """Wrap long labels for plots."""
    return "\n".join(
        textwrap.wrap(
            str(label),
            width=width,
            break_long_words=False,
            replace_whitespace=False,
        )
    )