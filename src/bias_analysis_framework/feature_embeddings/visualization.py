"""Visualizations for feature-embedding bias analysis."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


BASE_CATEGORICAL_PALETTE = [
    "#2AB7CA",
    "#E94E77",
    "#FDB515",
    "#7F7F7F",
    "#76B041",
    "#6A5ACD",
    "#0C635E",
    "#A950B4",
    "#964B36",
    "#C68797",
    "#4A5FE1",
    "#5A8A8B",
]

MODEL_PALETTE = [
    "#2AB7CA",
    "#E94E77",
    "#FDB515",
    "#76B041",
    "#6A5ACD",
    "#A950B4",
    "#4A5FE1",
    "#964B36",
]


def set_embedding_plot_style() -> None:
    """Apply the compact publication-style settings used by the old workflow."""
    mpl.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
        }
    )


def generate_model_embedding_figures(
    *,
    projections: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    model_name: str,
    config: dict[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    """Generate all configured embedding figures for one model.

    Generated outputs include:

    - global projections colored by configured targets;
    - global projections colored by configured bias factors;
    - per-target projections;
    - per-target projections colored by each bias factor;
    - target panel grids;
    - bias-factor panel grids.
    """
    set_embedding_plot_style()

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    workflow_config = get_workflow_config(config)
    visual_config = workflow_config.get("visualizations", {})

    target_specs = get_target_specs(config)
    bias_specs = get_bias_specs(config)

    written: list[Path] = []

    for projection_name, projection_df in projections.items():
        joined = combine_projection_and_metadata(
            projection_df=projection_df,
            metadata=metadata,
        )

        projection_dir = output_dir / projection_name
        projection_dir.mkdir(parents=True, exist_ok=True)

        if visual_config.get("global_target_plots", True):
            for target_name, allowed_labels in target_specs.items():
                if target_name not in joined.columns:
                    continue

                filtered = filter_labels(
                    dataframe=joined,
                    column=target_name,
                    allowed_values=allowed_labels,
                )

                if filtered.empty:
                    continue

                path = (
                    projection_dir
                    / "global_targets"
                    / f"{safe_name(target_name)}.png"
                )

                plot_embedding_dataframe(
                    dataframe=filtered,
                    projection_name=projection_name,
                    color_column=target_name,
                    title=(
                        f"{model_name} - {target_name} - "
                        f"{projection_display_name(projection_name)}"
                    ),
                    output_path=path,
                    config=config,
                )
                written.append(path)

        if visual_config.get("global_bias_plots", True):
            for bias_name, bias_spec in bias_specs.items():
                bias_column = bias_spec["column"]

                if bias_column not in joined.columns:
                    continue

                filtered = prepare_bias_plot_dataframe(
                    dataframe=joined,
                    bias_column=bias_column,
                    bias_spec=bias_spec,
                )

                if filtered.empty:
                    continue

                path = (
                    projection_dir
                    / "global_bias_factors"
                    / f"{safe_name(bias_name)}.png"
                )

                plot_embedding_dataframe(
                    dataframe=filtered,
                    projection_name=projection_name,
                    color_column=bias_column,
                    title=(
                        f"{model_name} - global - {bias_name} - "
                        f"{projection_display_name(projection_name)}"
                    ),
                    output_path=path,
                    config=config,
                )
                written.append(path)

        if visual_config.get("per_target_bias_plots", True):
            for target_name, allowed_labels in target_specs.items():
                if target_name not in joined.columns:
                    continue

                target_subset = filter_labels(
                    dataframe=joined,
                    column=target_name,
                    allowed_values=allowed_labels,
                )

                if target_subset.empty:
                    continue

                target_dir = (
                    projection_dir
                    / "by_target"
                    / safe_name(target_name)
                )

                target_path = target_dir / "colored_by_target.png"

                plot_embedding_dataframe(
                    dataframe=target_subset,
                    projection_name=projection_name,
                    color_column=target_name,
                    title=(
                        f"{model_name} - {target_name} - "
                        f"{projection_display_name(projection_name)}"
                    ),
                    output_path=target_path,
                    config=config,
                )
                written.append(target_path)

                for bias_name, bias_spec in bias_specs.items():
                    bias_column = bias_spec["column"]

                    if bias_column not in target_subset.columns:
                        continue

                    bias_subset = prepare_bias_plot_dataframe(
                        dataframe=target_subset,
                        bias_column=bias_column,
                        bias_spec=bias_spec,
                    )

                    if bias_subset.empty:
                        continue

                    bias_path = (
                        target_dir
                        / f"colored_by_{safe_name(bias_name)}.png"
                    )

                    plot_embedding_dataframe(
                        dataframe=bias_subset,
                        projection_name=projection_name,
                        color_column=bias_column,
                        title=(
                            f"{model_name} - {target_name} - "
                            f"{projection_display_name(projection_name)} "
                            f"by {bias_name}"
                        ),
                        output_path=bias_path,
                        config=config,
                    )
                    written.append(bias_path)

        if visual_config.get("target_panel_grid", True):
            panel_path = (
                projection_dir
                / "panels"
                / "targets_panel.png"
            )

            generated = plot_label_panel_grid(
                dataframe=joined,
                projection_name=projection_name,
                panel_specs=[
                    {
                        "display_name": target_name,
                        "column": target_name,
                        "allowed_values": allowed_labels,
                        "minimum_count": 1,
                        "other_label": None,
                    }
                    for target_name, allowed_labels in target_specs.items()
                    if target_name in joined.columns
                ],
                panel_title=(
                    f"{model_name} - target biomarkers - "
                    f"{projection_display_name(projection_name)}"
                ),
                output_path=panel_path,
                config=config,
            )

            if generated:
                written.append(panel_path)

        if visual_config.get("bias_panel_grid", True):
            selected_bias_names = get_selected_bias_factors_for_model(
                model_name=model_name,
                bias_specs=bias_specs,
                config=config,
            )

            panel_specs = []

            for bias_name in selected_bias_names:
                if bias_name not in bias_specs:
                    continue

                bias_spec = bias_specs[bias_name]
                bias_column = bias_spec["column"]

                if bias_column not in joined.columns:
                    continue

                panel_specs.append(
                    {
                        "display_name": bias_name,
                        "column": bias_column,
                        "allowed_values": bias_spec.get("subgroups"),
                        "minimum_count": bias_spec.get(
                            "minimum_count",
                            1,
                        ),
                        "other_label": bias_spec.get("other_label"),
                    }
                )

            panel_path = (
                projection_dir
                / "panels"
                / "bias_factors_panel.png"
            )

            generated = plot_label_panel_grid(
                dataframe=joined,
                projection_name=projection_name,
                panel_specs=panel_specs,
                panel_title=(
                    f"{model_name} - bias factors - "
                    f"{projection_display_name(projection_name)}"
                ),
                output_path=panel_path,
                config=config,
            )

            if generated:
                written.append(panel_path)

    return written


def combine_projection_and_metadata(
    *,
    projection_df: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Combine aligned projection coordinates and metadata."""
    projection_df = projection_df.reset_index(drop=True)
    metadata = metadata.reset_index(drop=True)

    if len(projection_df) != len(metadata):
        raise ValueError(
            "Projection coordinates and metadata must have equal lengths. "
            f"Found {len(projection_df)} projection rows and "
            f"{len(metadata)} metadata rows."
        )

    duplicate_columns = [
        column
        for column in projection_df.columns
        if column in metadata.columns
    ]

    if duplicate_columns:
        projection_df = projection_df.drop(
            columns=duplicate_columns
        )

    return pd.concat(
        [metadata, projection_df],
        axis=1,
    )


def plot_embedding_dataframe(
    *,
    dataframe: pd.DataFrame,
    projection_name: str,
    color_column: str,
    title: str,
    output_path: str | Path,
    config: dict[str, Any],
) -> Path:
    """Plot one publication-style embedding scatter plot."""
    x_column, y_column = projection_columns(
        dataframe,
        projection_name,
    )

    labels = clean_plot_labels(
        dataframe[color_column]
    )

    usable_mask = labels.notna()
    plot_df = dataframe.loc[usable_mask].reset_index(drop=True)
    labels = labels.loc[usable_mask].reset_index(drop=True)

    if plot_df.empty:
        raise ValueError(
            f"No usable labels remain for plot column {color_column!r}."
        )

    visual_config = (
        get_workflow_config(config)
        .get("visualizations", {})
    )

    point_size = float(
        visual_config.get("point_size", 9)
    )
    point_alpha = float(
        visual_config.get("point_alpha", 0.78)
    )
    zoom_percentiles = visual_config.get(
        "zoom_percentiles",
        [1, 99],
    )

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(
        figsize=tuple(
            visual_config.get(
                "single_plot_figsize",
                [7.5, 5.5],
            )
        )
    )

    draw_embedding_axis(
        ax=ax,
        x_values=plot_df[x_column].to_numpy(),
        y_values=plot_df[y_column].to_numpy(),
        labels=labels,
        title=title,
        projection_name=projection_name,
        point_size=point_size,
        point_alpha=point_alpha,
        zoom_percentiles=zoom_percentiles,
        show_legend=True,
        legend_ncol=int(
            visual_config.get("legend_ncol", 1)
        ),
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=350,
        bbox_inches="tight",
    )
    plt.close(fig)

    return output_path


def plot_label_panel_grid(
    *,
    dataframe: pd.DataFrame,
    projection_name: str,
    panel_specs: list[dict[str, Any]],
    panel_title: str,
    output_path: str | Path,
    config: dict[str, Any],
) -> bool:
    """Plot several target or bias-factor colorings in one panel grid."""
    if not panel_specs:
        return False

    x_column, y_column = projection_columns(
        dataframe,
        projection_name,
    )

    visual_config = (
        get_workflow_config(config)
        .get("visualizations", {})
    )

    configured_columns = int(
        visual_config.get("panel_columns", 3)
    )
    n_columns = max(
        1,
        min(configured_columns, len(panel_specs)),
    )
    n_rows = int(
        np.ceil(len(panel_specs) / n_columns)
    )

    fig_width = float(
        visual_config.get("panel_cell_width", 4.0)
    ) * n_columns
    fig_height = float(
        visual_config.get("panel_cell_height", 3.6)
    ) * n_rows

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    point_size = float(
        visual_config.get("panel_point_size", 5)
    )
    point_alpha = float(
        visual_config.get("panel_point_alpha", 0.78)
    )
    zoom_percentiles = visual_config.get(
        "zoom_percentiles",
        [1, 99],
    )

    for index, spec in enumerate(panel_specs):
        row_index = index // n_columns
        column_index = index % n_columns
        ax = axes[row_index][column_index]

        panel_df = filter_labels(
            dataframe=dataframe,
            column=spec["column"],
            allowed_values=spec.get("allowed_values"),
        )

        panel_df = pool_or_drop_small_groups_for_plot(
            dataframe=panel_df,
            column=spec["column"],
            minimum_count=int(
                spec.get("minimum_count", 1)
            ),
            other_label=spec.get("other_label"),
        )

        if panel_df.empty:
            ax.set_visible(False)
            continue

        labels = clean_plot_labels(
            panel_df[spec["column"]]
        )

        usable_mask = labels.notna()
        panel_df = panel_df.loc[
            usable_mask
        ].reset_index(drop=True)
        labels = labels.loc[
            usable_mask
        ].reset_index(drop=True)

        draw_embedding_axis(
            ax=ax,
            x_values=panel_df[x_column].to_numpy(),
            y_values=panel_df[y_column].to_numpy(),
            labels=labels,
            title=str(spec["display_name"]),
            projection_name=projection_name,
            point_size=point_size,
            point_alpha=point_alpha,
            zoom_percentiles=zoom_percentiles,
            show_legend=True,
            legend_ncol=int(
                visual_config.get(
                    "row_legend_ncol",
                    3,
                )
            ),
        )

    for index in range(
        len(panel_specs),
        n_rows * n_columns,
    ):
        row_index = index // n_columns
        column_index = index % n_columns
        axes[row_index][column_index].set_visible(False)

    fig.suptitle(
        panel_title,
        fontsize=11,
        y=0.995,
    )

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(
        output_path,
        dpi=350,
        bbox_inches="tight",
    )
    plt.close(fig)

    return True


def plot_cross_model_target_panel(
    *,
    model_projection_tables: dict[str, pd.DataFrame],
    projection_name: str,
    target_name: str,
    allowed_values: list[str] | None,
    output_path: str | Path,
    config: dict[str, Any],
    dataset_label: str | None = None,
) -> bool:
    """Plot one target across multiple models.

    One panel is generated per model. The same label-to-color mapping is used
    across all models.
    """
    usable_models: OrderedDict[str, pd.DataFrame] = OrderedDict()

    for model_name, dataframe in model_projection_tables.items():
        if target_name not in dataframe.columns:
            continue

        filtered = filter_labels(
            dataframe=dataframe,
            column=target_name,
            allowed_values=allowed_values,
        )

        if not filtered.empty:
            usable_models[model_name] = filtered

    if not usable_models:
        return False

    shared_labels = pd.concat(
        [
            clean_plot_labels(
                dataframe[target_name]
            )
            for dataframe in usable_models.values()
        ],
        ignore_index=True,
    )

    ordered_classes = ordered_label_classes(
        shared_labels
    )
    color_map = build_label_color_map(
        ordered_classes
    )

    visual_config = (
        get_workflow_config(config)
        .get("visualizations", {})
    )

    n_models = len(usable_models)
    n_columns = min(
        int(
            visual_config.get(
                "cross_model_columns",
                3,
            )
        ),
        n_models,
    )
    n_rows = int(
        np.ceil(n_models / n_columns)
    )

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(
            4.1 * n_columns,
            3.7 * n_rows,
        ),
        squeeze=False,
    )

    for index, (model_name, dataframe) in enumerate(
        usable_models.items()
    ):
        row_index = index // n_columns
        column_index = index % n_columns
        ax = axes[row_index][column_index]

        x_column, y_column = projection_columns(
            dataframe,
            projection_name,
        )

        draw_embedding_axis(
            ax=ax,
            x_values=dataframe[x_column].to_numpy(),
            y_values=dataframe[y_column].to_numpy(),
            labels=clean_plot_labels(
                dataframe[target_name]
            ),
            title=model_name,
            projection_name=projection_name,
            point_size=float(
                visual_config.get(
                    "panel_point_size",
                    5,
                )
            ),
            point_alpha=float(
                visual_config.get(
                    "panel_point_alpha",
                    0.78,
                )
            ),
            zoom_percentiles=visual_config.get(
                "zoom_percentiles",
                [1, 99],
            ),
            show_legend=False,
            color_map=color_map,
        )

    for index in range(
        n_models,
        n_rows * n_columns,
    ):
        row_index = index // n_columns
        column_index = index % n_columns
        axes[row_index][column_index].set_visible(False)

    legend_handles = build_legend_handles(
        classes=ordered_classes,
        color_map=color_map,
        label_counts=shared_labels.value_counts(),
    )

    title_prefix = (
        f"{dataset_label} - "
        if dataset_label
        else ""
    )

    fig.suptitle(
        f"{title_prefix}{target_name} across models - "
        f"{projection_display_name(projection_name)}",
        fontsize=11,
        y=0.995,
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        frameon=False,
        ncol=min(
            int(
                visual_config.get(
                    "row_legend_ncol",
                    5,
                )
            ),
            max(1, len(legend_handles)),
        ),
    )

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout(rect=[0, 0.07, 1, 0.96])
    fig.savefig(
        output_path,
        dpi=350,
        bbox_inches="tight",
    )
    plt.close(fig)

    return True


def plot_cross_model_bias_panel(
    *,
    model_projection_tables: dict[str, pd.DataFrame],
    projection_name: str,
    selected_bias_factors: dict[str, list[str]],
    bias_specs: OrderedDict[str, dict[str, Any]],
    output_path: str | Path,
    config: dict[str, Any],
    target_filter: tuple[str, list[str] | None] | None = None,
    dataset_label: str | None = None,
) -> bool:
    """Plot selected bias factors across multiple models.

    Rows correspond to models. Columns correspond to the configured selected
    bias factors for each model. This reproduces the old configurable
    BIAS_PANEL_SELECTION behavior.
    """
    panel_items: list[
        tuple[str, str, pd.DataFrame, str]
    ] = []

    for model_name, factor_names in selected_bias_factors.items():
        dataframe = model_projection_tables.get(model_name)

        if dataframe is None:
            continue

        model_df = dataframe.copy()

        if target_filter is not None:
            target_name, allowed_values = target_filter

            if target_name not in model_df.columns:
                continue

            model_df = filter_labels(
                dataframe=model_df,
                column=target_name,
                allowed_values=allowed_values,
            )

        for factor_name in factor_names:
            factor_spec = bias_specs.get(factor_name)

            if factor_spec is None:
                continue

            factor_column = factor_spec["column"]

            if factor_column not in model_df.columns:
                continue

            filtered = prepare_bias_plot_dataframe(
                dataframe=model_df,
                bias_column=factor_column,
                bias_spec=factor_spec,
            )

            if filtered.empty:
                continue

            panel_items.append(
                (
                    model_name,
                    factor_name,
                    filtered,
                    factor_column,
                )
            )

    if not panel_items:
        return False

    model_order = list(selected_bias_factors)
    max_columns = max(
        len(selected_bias_factors.get(model, []))
        for model in model_order
    )

    fig, axes = plt.subplots(
        len(model_order),
        max_columns,
        figsize=(
            4.2 * max_columns,
            3.7 * len(model_order),
        ),
        squeeze=False,
    )

    visual_config = (
        get_workflow_config(config)
        .get("visualizations", {})
    )

    for row_index, model_name in enumerate(model_order):
        model_items = [
            item
            for item in panel_items
            if item[0] == model_name
        ]

        for column_index in range(max_columns):
            ax = axes[row_index][column_index]

            if column_index >= len(model_items):
                ax.set_visible(False)
                continue

            _, factor_name, dataframe, factor_column = (
                model_items[column_index]
            )

            x_column, y_column = projection_columns(
                dataframe,
                projection_name,
            )

            draw_embedding_axis(
                ax=ax,
                x_values=dataframe[x_column].to_numpy(),
                y_values=dataframe[y_column].to_numpy(),
                labels=clean_plot_labels(
                    dataframe[factor_column]
                ),
                title=f"{model_name} - {factor_name}",
                projection_name=projection_name,
                point_size=float(
                    visual_config.get(
                        "panel_point_size",
                        5,
                    )
                ),
                point_alpha=float(
                    visual_config.get(
                        "panel_point_alpha",
                        0.78,
                    )
                ),
                zoom_percentiles=visual_config.get(
                    "zoom_percentiles",
                    [1, 99],
                ),
                show_legend=True,
                legend_ncol=int(
                    visual_config.get(
                        "row_legend_ncol",
                        3,
                    )
                ),
            )

    title_prefix = (
        f"{dataset_label} - "
        if dataset_label
        else ""
    )

    fig.suptitle(
        f"{title_prefix}selected bias factors across models - "
        f"{projection_display_name(projection_name)}",
        fontsize=11,
        y=0.995,
    )

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(
        output_path,
        dpi=350,
        bbox_inches="tight",
    )
    plt.close(fig)

    return True


def plot_cross_dataset_model_target_panel(
    *,
    dataset_model_tables: dict[
        str,
        dict[str, pd.DataFrame],
    ],
    projection_name: str,
    target_name: str,
    allowed_values: list[str] | None,
    output_path: str | Path,
    config: dict[str, Any],
) -> bool:
    """Plot dataset × model panels for one target.

    Rows are datasets and columns are models. This function is intended for a
    later aggregate call after separate dataset configs have been run.
    """
    dataset_order = list(dataset_model_tables)

    model_order: list[str] = []
    for model_tables in dataset_model_tables.values():
        for model_name in model_tables:
            if model_name not in model_order:
                model_order.append(model_name)

    if not dataset_order or not model_order:
        return False

    usable_labels: list[pd.Series] = []

    for model_tables in dataset_model_tables.values():
        for dataframe in model_tables.values():
            if target_name in dataframe.columns:
                filtered = filter_labels(
                    dataframe=dataframe,
                    column=target_name,
                    allowed_values=allowed_values,
                )
                if not filtered.empty:
                    usable_labels.append(
                        clean_plot_labels(
                            filtered[target_name]
                        )
                    )

    if not usable_labels:
        return False

    combined_labels = pd.concat(
        usable_labels,
        ignore_index=True,
    )
    ordered_classes = ordered_label_classes(
        combined_labels
    )
    color_map = build_label_color_map(
        ordered_classes
    )

    fig, axes = plt.subplots(
        len(dataset_order),
        len(model_order),
        figsize=(
            4.0 * len(model_order),
            3.6 * len(dataset_order),
        ),
        squeeze=False,
    )

    visual_config = (
        get_workflow_config(config)
        .get("visualizations", {})
    )

    for row_index, dataset_name in enumerate(dataset_order):
        model_tables = dataset_model_tables[dataset_name]

        for column_index, model_name in enumerate(model_order):
            ax = axes[row_index][column_index]
            dataframe = model_tables.get(model_name)

            if dataframe is None or target_name not in dataframe.columns:
                ax.set_visible(False)
                continue

            filtered = filter_labels(
                dataframe=dataframe,
                column=target_name,
                allowed_values=allowed_values,
            )

            if filtered.empty:
                ax.set_visible(False)
                continue

            x_column, y_column = projection_columns(
                filtered,
                projection_name,
            )

            draw_embedding_axis(
                ax=ax,
                x_values=filtered[x_column].to_numpy(),
                y_values=filtered[y_column].to_numpy(),
                labels=clean_plot_labels(
                    filtered[target_name]
                ),
                title=f"{dataset_name} - {model_name}",
                projection_name=projection_name,
                point_size=float(
                    visual_config.get(
                        "panel_point_size",
                        5,
                    )
                ),
                point_alpha=float(
                    visual_config.get(
                        "panel_point_alpha",
                        0.78,
                    )
                ),
                zoom_percentiles=visual_config.get(
                    "zoom_percentiles",
                    [1, 99],
                ),
                show_legend=False,
                color_map=color_map,
            )

    legend_handles = build_legend_handles(
        classes=ordered_classes,
        color_map=color_map,
        label_counts=combined_labels.value_counts(),
    )

    fig.suptitle(
        f"{target_name}: datasets × models - "
        f"{projection_display_name(projection_name)}",
        fontsize=11,
        y=0.995,
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        frameon=False,
        ncol=min(5, max(1, len(legend_handles))),
    )

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(
        output_path,
        dpi=350,
        bbox_inches="tight",
    )
    plt.close(fig)

    return True


def draw_embedding_axis(
    *,
    ax: Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    labels: pd.Series,
    title: str,
    projection_name: str,
    point_size: float,
    point_alpha: float,
    zoom_percentiles: Iterable[float],
    show_legend: bool,
    legend_ncol: int = 1,
    color_map: dict[str, Any] | None = None,
) -> None:
    """Draw one embedding scatter plot on an existing axis."""
    labels = clean_plot_labels(labels)

    valid_mask = (
        labels.notna()
        & np.isfinite(x_values)
        & np.isfinite(y_values)
    )

    x_values = np.asarray(x_values)[
        valid_mask.to_numpy()
    ]
    y_values = np.asarray(y_values)[
        valid_mask.to_numpy()
    ]
    labels = labels.loc[
        valid_mask
    ].reset_index(drop=True)

    ordered_classes = ordered_label_classes(
        labels
    )

    if color_map is None:
        color_map = build_label_color_map(
            ordered_classes
        )

    for class_name in ordered_classes:
        class_mask = (
            labels.astype(str)
            == str(class_name)
        ).to_numpy()

        ax.scatter(
            x_values[class_mask],
            y_values[class_mask],
            s=point_size,
            alpha=point_alpha,
            color=color_map[str(class_name)],
            edgecolors="none",
            rasterized=True,
            label=(
                f"{class_name} "
                f"(n={int(class_mask.sum())})"
            ),
        )

    lower, upper = list(zoom_percentiles)

    apply_percentile_limits(
        ax=ax,
        x_values=x_values,
        y_values=y_values,
        lower=float(lower),
        upper=float(upper),
        margin=0.08,
    )

    ax.set_title(title)
    ax.set_xlabel(
        f"Component 1 ({projection_display_name(projection_name)})"
    )
    ax.set_ylabel(
        f"Component 2 ({projection_display_name(projection_name)})"
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    if show_legend:
        ax.legend(
            frameon=False,
            loc="best",
            markerscale=1.6,
            ncol=max(1, legend_ncol),
            handletextpad=0.3,
            columnspacing=0.8,
        )


def apply_percentile_limits(
    *,
    ax: Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    lower: float,
    upper: float,
    margin: float,
) -> None:
    """Apply robust percentile zooming, matching the old compact plots."""
    if len(x_values) == 0:
        return

    x_low, x_high = np.nanpercentile(
        x_values,
        [lower, upper],
    )
    y_low, y_high = np.nanpercentile(
        y_values,
        [lower, upper],
    )

    x_span = max(
        float(x_high - x_low),
        1e-9,
    )
    y_span = max(
        float(y_high - y_low),
        1e-9,
    )

    ax.set_xlim(
        x_low - margin * x_span,
        x_high + margin * x_span,
    )
    ax.set_ylim(
        y_low - margin * y_span,
        y_high + margin * y_span,
    )


def prepare_bias_plot_dataframe(
    *,
    dataframe: pd.DataFrame,
    bias_column: str,
    bias_spec: dict[str, Any],
) -> pd.DataFrame:
    """Filter and optionally pool small bias groups for plotting."""
    filtered = filter_labels(
        dataframe=dataframe,
        column=bias_column,
        allowed_values=bias_spec.get("subgroups"),
    )

    return pool_or_drop_small_groups_for_plot(
        dataframe=filtered,
        column=bias_column,
        minimum_count=int(
            bias_spec.get("minimum_count", 1)
        ),
        other_label=bias_spec.get("other_label"),
    )


def filter_labels(
    *,
    dataframe: pd.DataFrame,
    column: str,
    allowed_values: list[str] | None,
) -> pd.DataFrame:
    """Filter unknown labels and optional configured classes."""
    if column not in dataframe.columns:
        return dataframe.iloc[0:0].copy()

    labels = clean_plot_labels(
        dataframe[column]
    )

    mask = labels.notna()

    if allowed_values:
        allowed_casefold = {
            str(value).strip().casefold()
            for value in allowed_values
        }

        mask &= labels.str.casefold().isin(
            allowed_casefold
        )

    output = dataframe.loc[
        mask
    ].copy()
    output[column] = labels.loc[mask].to_numpy()

    return output.reset_index(drop=True)


def pool_or_drop_small_groups_for_plot(
    *,
    dataframe: pd.DataFrame,
    column: str,
    minimum_count: int,
    other_label: str | None,
) -> pd.DataFrame:
    """Pool or drop low-count plot categories."""
    if dataframe.empty or minimum_count <= 1:
        return dataframe.reset_index(drop=True)

    output = dataframe.copy()
    labels = clean_plot_labels(
        output[column]
    )

    counts = labels.value_counts()
    small_groups = set(
        counts[counts < minimum_count].index
    )

    if not small_groups:
        return output.reset_index(drop=True)

    if other_label:
        output[column] = labels.where(
            ~labels.isin(small_groups),
            str(other_label),
        )
        return output.reset_index(drop=True)

    keep_mask = ~labels.isin(small_groups)

    return output.loc[
        keep_mask
    ].reset_index(drop=True)


def clean_plot_labels(
    labels: pd.Series,
) -> pd.Series:
    """Clean labels and turn missing/unknown values into NaN."""
    output = (
        labels
        .astype("string")
        .str.strip()
    )

    unknown_tokens = {
        "",
        "unknown",
        "na",
        "nan",
        "n/a",
        "none",
        "null",
        "<na>",
    }

    unknown_mask = (
        output.isna()
        | output.str.casefold().isin(
            unknown_tokens
        )
    )

    return output.mask(unknown_mask)


def ordered_label_classes(
    labels: pd.Series,
) -> list[str]:
    """Order classes by descending observed count."""
    return (
        labels
        .dropna()
        .astype(str)
        .value_counts()
        .index
        .tolist()
    )


def build_label_color_map(
    classes: list[str],
) -> dict[str, Any]:
    """Build the old categorical palette, expanding when needed."""
    colors: list[Any] = list(
        BASE_CATEGORICAL_PALETTE
    )

    if len(classes) > len(colors):
        set2 = mpl.colormaps.get_cmap("Set2")
        tab20 = mpl.colormaps.get_cmap("tab20")

        colors.extend(
            [
                set2(index / max(1, len(classes) - 1))
                for index in range(len(classes))
            ]
        )
        colors.extend(
            [
                tab20(index / max(1, len(classes) - 1))
                for index in range(len(classes))
            ]
        )

    while len(colors) < len(classes):
        colors.extend(
            BASE_CATEGORICAL_PALETTE
        )

    return {
        str(class_name): colors[index]
        for index, class_name in enumerate(classes)
    }


def build_legend_handles(
    *,
    classes: list[str],
    color_map: dict[str, Any],
    label_counts: pd.Series,
) -> list[Line2D]:
    """Create shared legend handles for panel figures."""
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=5,
            markerfacecolor=color_map[str(class_name)],
            markeredgecolor="none",
            label=(
                f"{class_name} "
                f"(n={int(label_counts.get(class_name, 0))})"
            ),
        )
        for class_name in classes
    ]


def projection_columns(
    dataframe: pd.DataFrame,
    projection_name: str,
) -> tuple[str, str]:
    """Return coordinate columns for a projection table."""
    mapping = {
        "pca": ("PCA_1", "PCA_2"),
        "tsne": ("TSNE_1", "TSNE_2"),
        "umap": ("UMAP_1", "UMAP_2"),
    }

    if projection_name not in mapping:
        raise ValueError(
            f"Unsupported projection: {projection_name}"
        )

    x_column, y_column = mapping[projection_name]

    missing = [
        column
        for column in (x_column, y_column)
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"Projection {projection_name!r} is missing columns: {missing}"
        )

    return x_column, y_column


def projection_display_name(
    projection_name: str,
) -> str:
    """Return human-readable projection name."""
    return {
        "pca": "PCA",
        "tsne": "t-SNE",
        "umap": "UMAP",
    }.get(
        projection_name,
        projection_name.upper(),
    )


def get_workflow_config(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return Step 3 workflow config."""
    value = (
        config.get("workflows", {})
        .get("feature_embedding_analysis", {})
    )

    if not isinstance(value, dict):
        raise ValueError(
            "workflows.feature_embedding_analysis must be a mapping."
        )

    return value


def get_target_specs(
    config: dict[str, Any],
) -> OrderedDict[str, list[str] | None]:
    """Return targets and configured labels."""
    output: OrderedDict[
        str,
        list[str] | None,
    ] = OrderedDict()

    for target in config.get("targets", []):
        name = target.get("name")

        if not name:
            continue

        if target.get("labels"):
            labels = [
                str(value)
                for value in target["labels"]
            ]
        elif (
            target.get("negative_label") is not None
            and target.get("positive_label") is not None
        ):
            labels = [
                str(target["negative_label"]),
                str(target["positive_label"]),
            ]
        else:
            labels = None

        output[str(name)] = labels

    return output


def get_bias_specs(
    config: dict[str, Any],
) -> OrderedDict[str, dict[str, Any]]:
    """Return configured bias-factor plotting specifications."""
    workflow_config = get_workflow_config(
        config
    )

    minimum_counts = workflow_config.get(
        "factor_minimum_counts",
        {},
    )
    default_minimum = int(
        workflow_config.get(
            "minimum_group_size_for_metrics",
            2,
        )
    )

    output: OrderedDict[
        str,
        dict[str, Any],
    ] = OrderedDict()

    for factor in config.get("bias_factors", []):
        name = factor.get("name")

        if not name:
            continue

        output[str(name)] = {
            "column": str(
                factor.get("column", name)
            ),
            "subgroups": (
                [
                    str(value)
                    for value in factor["subgroups"]
                ]
                if factor.get("subgroups")
                else None
            ),
            "minimum_count": int(
                factor.get(
                    "minimum_count",
                    minimum_counts.get(
                        name,
                        default_minimum,
                    ),
                )
            ),
            "other_label": factor.get(
                "other_label"
            ),
        }

    return output


def get_selected_bias_factors_for_model(
    *,
    model_name: str,
    bias_specs: OrderedDict[str, dict[str, Any]],
    config: dict[str, Any],
) -> list[str]:
    """Return selected bias factors for a model panel."""
    visual_config = (
        get_workflow_config(config)
        .get("visualizations", {})
    )

    selection = visual_config.get(
        "bias_panel_selection",
        {},
    )

    if isinstance(selection, dict):
        selected = selection.get(model_name)

        if selected:
            return [
                str(value)
                for value in selected
            ]

    max_factors = int(
        visual_config.get(
            "max_bias_factors_per_panel",
            6,
        )
    )

    return list(bias_specs)[:max_factors]


def safe_name(value: object) -> str:
    """Create a filesystem-safe label."""
    text = str(value).strip()

    for character in [
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
    ]:
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