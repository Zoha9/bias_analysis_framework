from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bias_analysis_framework.config.loader import (
    load_config_from_args,
    workflow_enabled,
)
from bias_analysis_framework.feature_embeddings.analysis import (
    analyze_model_feature_embeddings,
    write_feature_embedding_metric_workbook,
)
from bias_analysis_framework.feature_embeddings.loading import (
    load_slide_level_embeddings,
)
from bias_analysis_framework.feature_embeddings.projections import (
    compute_embedding_projections,
)
from bias_analysis_framework.feature_embeddings.visualization import (
    combine_projection_and_metadata,
    generate_model_embedding_figures,
    get_bias_specs,
    get_target_specs,
    plot_cross_model_bias_panel,
    plot_cross_model_target_panel,
    safe_name,
)
from bias_analysis_framework.io.clinical_tables import (
    read_clinical_table,
)
from bias_analysis_framework.io.paths import (
    get_workflow_paths,
)
from bias_analysis_framework.io.slide_tables import (
    read_slide_table,
)


WORKFLOW_NAME = "feature_embedding_analysis"


def run(config: dict[str, Any]) -> None:
    """Run Step 3 feature-embedding bias analysis."""
    if not workflow_enabled(
        config,
        WORKFLOW_NAME,
    ):
        print(
            f"Workflow '{WORKFLOW_NAME}' is disabled in config. "
            "Nothing to run."
        )
        return

    paths = get_workflow_paths(
        config=config,
        workflow_name=WORKFLOW_NAME,
        create=True,
    )

    workflow_config = (
        config.get("workflows", {})
        .get(WORKFLOW_NAME, {})
    )

    clinical_df = read_clinical_table(config)
    slide_df = read_slide_table(config)

    models = get_configured_models(config)

    if not models:
        raise ValueError(
            "No models are configured under the top-level models list."
        )

    print("Starting Step 3 feature-embedding bias analysis.")
    print(f"Configured models: {', '.join(models)}")
    print(f"Clinical rows: {len(clinical_df)}")
    print(f"Slide rows: {len(slide_df)}")

    all_metrics: list[pd.DataFrame] = []
    all_counts: list[pd.DataFrame] = []
    model_projection_tables: dict[
        str,
        dict[str, pd.DataFrame],
    ] = {}
    model_status_rows: list[dict[str, Any]] = []

    continue_on_error = bool(
        workflow_config.get(
            "continue_on_error",
            False,
        )
    )

    for model_name in models:
        print()
        print(f"[{model_name}] Loading feature embeddings.")

        try:
            loaded = load_slide_level_embeddings(
                clinical_df=clinical_df,
                slide_df=slide_df,
                config=config,
                model_name=model_name,
            )

            print(
                f"[{model_name}] Loaded {len(loaded.metadata)} "
                f"slide embeddings with dimension "
                f"{loaded.feature_dimension}."
            )

            analysis_result = analyze_model_feature_embeddings(
                loaded_embeddings=loaded,
                config=config,
                model_name=model_name,
            )

            patient_embeddings = (
                analysis_result.patient_embeddings
            )

            print(
                f"[{model_name}] Pooled to "
                f"{len(patient_embeddings.metadata)} patient embeddings."
            )

            projections = compute_embedding_projections(
                patient_embeddings.embeddings,
                config=config,
            )

            model_table_dir = (
                paths.tables_dir
                / safe_name(model_name)
            )
            model_figure_dir = (
                paths.figures_dir
                / safe_name(model_name)
            )

            model_table_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            model_figure_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            metric_workbook = (
                model_table_dir
                / f"{safe_name(model_name)}_embedding_bias_metrics.xlsx"
            )

            write_feature_embedding_metric_workbook(
                result=analysis_result,
                output_path=metric_workbook,
            )

            projection_tables = (
                write_projection_tables(
                    projections=projections,
                    metadata=patient_embeddings.metadata,
                    output_dir=model_table_dir,
                    model_name=model_name,
                )
            )

            model_projection_tables[
                model_name
            ] = projection_tables

            figure_paths = (
                generate_model_embedding_figures(
                    projections=projections,
                    metadata=patient_embeddings.metadata,
                    model_name=model_name,
                    config=config,
                    output_dir=model_figure_dir,
                )
            )

            if not analysis_result.metrics.empty:
                all_metrics.append(
                    analysis_result.metrics
                )

            if not analysis_result.counts.empty:
                all_counts.append(
                    analysis_result.counts
                )

            model_status_rows.append(
                {
                    "Model": model_name,
                    "Status": "completed",
                    "N_Slides": len(
                        loaded.metadata
                    ),
                    "N_Patients": len(
                        patient_embeddings.metadata
                    ),
                    "FeatureDimension": (
                        patient_embeddings.embeddings.shape[1]
                    ),
                    "N_MetricRows": len(
                        analysis_result.metrics
                    ),
                    "N_Figures": len(
                        figure_paths
                    ),
                    "MetricWorkbook": str(
                        metric_workbook
                    ),
                    "Message": "",
                }
            )

            print(
                f"[{model_name}] Wrote metric workbook: "
                f"{metric_workbook}"
            )
            print(
                f"[{model_name}] Wrote "
                f"{len(figure_paths)} figure(s)."
            )

        except Exception as error:
            model_status_rows.append(
                {
                    "Model": model_name,
                    "Status": "failed",
                    "N_Slides": pd.NA,
                    "N_Patients": pd.NA,
                    "FeatureDimension": pd.NA,
                    "N_MetricRows": pd.NA,
                    "N_Figures": pd.NA,
                    "MetricWorkbook": "",
                    "Message": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                }
            )

            print(
                f"[{model_name}] Failed: "
                f"{type(error).__name__}: {error}"
            )

            if not continue_on_error:
                write_combined_outputs(
                    all_metrics=all_metrics,
                    all_counts=all_counts,
                    status_rows=model_status_rows,
                    output_path=(
                        paths.tables_dir
                        / "feature_embedding_bias_analysis_summary.xlsx"
                    ),
                )
                raise

    combined_output_path = (
        paths.tables_dir
        / "feature_embedding_bias_analysis_summary.xlsx"
    )

    write_combined_outputs(
        all_metrics=all_metrics,
        all_counts=all_counts,
        status_rows=model_status_rows,
        output_path=combined_output_path,
    )

    cross_model_paths = generate_cross_model_panels(
        model_projection_tables=model_projection_tables,
        config=config,
        output_dir=(
            paths.figures_dir
            / "cross_model_panels"
        ),
    )

    print()
    print("Step 3 feature-embedding bias analysis completed.")
    print(
        f"Combined summary workbook: "
        f"{combined_output_path}"
    )
    print(
        f"Cross-model panels written: "
        f"{len(cross_model_paths)}"
    )
    print(
        f"Tables directory: "
        f"{paths.tables_dir}"
    )
    print(
        f"Figures directory: "
        f"{paths.figures_dir}"
    )


def write_projection_tables(
    *,
    projections: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    output_dir: str | Path,
    model_name: str,
) -> dict[str, pd.DataFrame]:
    """Write aligned projection coordinates and metadata."""
    output_dir = Path(
        output_dir
    ).expanduser().resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output: dict[str, pd.DataFrame] = {}

    for projection_name, projection_df in projections.items():
        joined = combine_projection_and_metadata(
            projection_df=projection_df,
            metadata=metadata,
        )

        output_path = (
            output_dir
            / (
                f"{safe_name(model_name)}_"
                f"{projection_name}_coordinates.csv"
            )
        )

        joined.to_csv(
            output_path,
            index=False,
        )

        output[projection_name] = joined

    return output


def write_combined_outputs(
    *,
    all_metrics: list[pd.DataFrame],
    all_counts: list[pd.DataFrame],
    status_rows: list[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Write combined Step 3 tables."""
    output_path = Path(
        output_path
    ).expanduser().resolve()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df = (
        pd.concat(
            all_metrics,
            ignore_index=True,
        )
        if all_metrics
        else pd.DataFrame()
    )

    counts_df = (
        pd.concat(
            all_counts,
            ignore_index=True,
        )
        if all_counts
        else pd.DataFrame()
    )

    status_df = pd.DataFrame(
        status_rows
    )

    metric_summary = build_metric_summary(
        metrics_df
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:
        status_df.to_excel(
            writer,
            sheet_name="model_status",
            index=False,
        )
        metrics_df.to_excel(
            writer,
            sheet_name="all_metrics",
            index=False,
        )
        metric_summary.to_excel(
            writer,
            sheet_name="metric_summary",
            index=False,
        )
        counts_df.to_excel(
            writer,
            sheet_name="class_counts",
            index=False,
        )

    return output_path


def build_metric_summary(
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create cross-model mean metric summaries."""
    if metrics_df.empty:
        return pd.DataFrame()

    metric_columns = [
        "kNN_acc_biomarker",
        "kNN_acc_biasfactor",
        "Robustness_Index",
        "Bio_Silhouette(cosine)",
        "Bias_Silhouette(cosine)",
        "Bio_Calinski_Harabasz",
        "Bias_Calinski_Harabasz",
        "Bio_Davies_Bouldin",
        "Bias_Davies_Bouldin",
        "Delta_kNN_Biomarker_minus_Bias",
        "Delta_Silhouette_Biomarker_minus_Bias",
    ]

    available_metric_columns = [
        column
        for column in metric_columns
        if column in metrics_df.columns
    ]

    if not available_metric_columns:
        return pd.DataFrame()

    return (
        metrics_df.groupby(
            ["Model"],
            dropna=False,
        )[available_metric_columns]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )


def generate_cross_model_panels(
    *,
    model_projection_tables: dict[
        str,
        dict[str, pd.DataFrame],
    ],
    config: dict[str, Any],
    output_dir: str | Path,
) -> list[Path]:
    """Generate configured cross-model target and bias panels."""
    output_dir = Path(
        output_dir
    ).expanduser().resolve()
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    workflow_config = (
        config.get("workflows", {})
        .get(WORKFLOW_NAME, {})
    )
    visual_config = workflow_config.get(
        "visualizations",
        {},
    )

    if not visual_config.get(
        "cross_model_panels",
        True,
    ):
        return []

    target_specs = get_target_specs(
        config
    )
    bias_specs = get_bias_specs(
        config
    )

    dataset_label = build_dataset_label(
        config
    )

    projection_names = sorted(
        {
            projection_name
            for model_tables
            in model_projection_tables.values()
            for projection_name
            in model_tables
        }
    )

    written: list[Path] = []

    for projection_name in projection_names:
        per_model_tables = {
            model_name: tables[projection_name]
            for model_name, tables
            in model_projection_tables.items()
            if projection_name in tables
        }

        for target_name, allowed_values in target_specs.items():
            output_path = (
                output_dir
                / projection_name
                / "targets"
                / f"{safe_name(target_name)}.png"
            )

            generated = plot_cross_model_target_panel(
                model_projection_tables=per_model_tables,
                projection_name=projection_name,
                target_name=target_name,
                allowed_values=allowed_values,
                output_path=output_path,
                config=config,
                dataset_label=dataset_label,
            )

            if generated:
                written.append(
                    output_path
                )

        selected_bias_factors = visual_config.get(
            "bias_panel_selection",
            {},
        )

        if not isinstance(
            selected_bias_factors,
            dict,
        ):
            selected_bias_factors = {}

        if not selected_bias_factors:
            selected_bias_factors = {
                model_name: list(
                    bias_specs
                )[
                    : int(
                        visual_config.get(
                            "max_bias_factors_per_panel",
                            2,
                        )
                    )
                ]
                for model_name in per_model_tables
            }

        target_filter = None
        panel_target = visual_config.get(
            "bias_panel_target"
        )

        if panel_target:
            target_filter = (
                str(panel_target),
                target_specs.get(
                    str(panel_target)
                ),
            )

        output_path = (
            output_dir
            / projection_name
            / "bias_factors"
            / "selected_bias_factors.png"
        )

        generated = plot_cross_model_bias_panel(
            model_projection_tables=per_model_tables,
            projection_name=projection_name,
            selected_bias_factors={
                str(model): [
                    str(factor)
                    for factor in factors
                ]
                for model, factors
                in selected_bias_factors.items()
            },
            bias_specs=bias_specs,
            output_path=output_path,
            config=config,
            target_filter=target_filter,
            dataset_label=dataset_label,
        )

        if generated:
            written.append(
                output_path
            )

    return written


def get_configured_models(
    config: dict[str, Any],
) -> list[str]:
    """Return configured model/extractor names."""
    output: list[str] = []

    for model in config.get(
        "models",
        [],
    ):
        if isinstance(model, str):
            name = model.strip()
        elif isinstance(model, dict):
            name = str(
                model.get("extractor")
                or model.get("name")
                or ""
            ).strip()
        else:
            raise ValueError(
                "Each models entry must be a string or mapping."
            )

        if name and name not in output:
            output.append(name)

    return output


def build_dataset_label(
    config: dict[str, Any],
) -> str:
    """Return organ and dataset label."""
    cohort = config.get(
        "cohort",
        {},
    )

    organ = str(
        cohort.get(
            "organ",
            "",
        )
    ).strip()
    dataset = str(
        cohort.get(
            "dataset",
            "",
        )
    ).strip()

    values = [
        value
        for value in [organ, dataset]
        if value
    ]

    return " / ".join(values)


def main() -> None:
    """Load config and run Step 3."""
    config = load_config_from_args()
    run(config)


if __name__ == "__main__":
    main()