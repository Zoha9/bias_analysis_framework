from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without requiring installation first.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bias_analysis_framework.config.loader import load_config_from_args, workflow_enabled
from bias_analysis_framework.io.paths import get_workflow_paths

from bias_analysis_framework.io.clinical_tables import (
    get_bias_factor_names,
    get_target_names,
    read_clinical_table,
    summarize_clinical_table,
    summarize_configured_column_values,
)
from bias_analysis_framework.io.slide_tables import (
    read_slide_table,
    summarize_slide_table,
    validate_clinical_slide_patient_overlap,
)
from bias_analysis_framework.io.tables import write_excel_sheets

from bias_analysis_framework.metrics.clinical_pretraining_metrics import (
    compute_clinical_pretraining_metrics,
    write_pretraining_summary_excel,
    write_sectioned_pretraining_metrics_excel,
)

from bias_analysis_framework.clinical.counts import (
    build_clinical_count_tables,
    write_clinical_count_tables_excel,
)

from bias_analysis_framework.clinical.visualization import (
    generate_clinical_distribution_figures,
)

from bias_analysis_framework.clinical.metric_heatmaps import (
    generate_clinical_metric_heatmaps,
)

WORKFLOW_NAME = "clinical_distribution_analysis"


def run(config: dict) -> None:
    """Run Step 1 clinical distribution analysis.

    Current implementation:
    - reads user config
    - validates clinical and slide tables
    - checks target/bias columns
    - checks clinical/slide patient overlap
    - creates standard output folders
    - writes an input validation workbook

    Next implementation:
    - refactor pretraining_metrics.py into metrics/clinical_pretraining_metrics.py
    - refactor counts + heatmaps into clinical/counts.py and clinical/visualization.py
    """
    if not workflow_enabled(config, WORKFLOW_NAME):
        print(f"Workflow '{WORKFLOW_NAME}' is disabled in config. Nothing to run.")
        return

    clinical_df = read_clinical_table(config)
    slide_df = read_slide_table(config)

    paths = get_workflow_paths(config, WORKFLOW_NAME, create=True)

    clinical_summary_df = summarize_clinical_table(clinical_df, config)
    slide_summary_df = summarize_slide_table(slide_df, config)
    overlap_summary_df = validate_clinical_slide_patient_overlap(
        clinical_df=clinical_df,
        slide_df=slide_df,
        config=config,
    )
    value_count_sheets = summarize_configured_column_values(clinical_df, config)

    output_path = paths.tables_dir / "clinical_input_data_summary.xlsx"

    sheets = {
        "clinical_summary": clinical_summary_df,
        "slide_summary": slide_summary_df,
        "patient_overlap": overlap_summary_df,
        "clinical_preview": clinical_df.head(20),
        "slide_preview": slide_df.head(20),
    }
    sheets.update(value_count_sheets)

    write_excel_sheets(
        output_path=output_path,
        sheets=sheets,
        index=False,
    )

    print("Step 1 clinical distribution input validation completed.")
    print(f"Clinical table rows: {len(clinical_df)}")
    print(f"Slide table rows: {len(slide_df)}")
    print(f"Validated target columns: {', '.join(get_target_names(config))}")
    print(f"Validated bias factor columns: {', '.join(get_bias_factor_names(config))}")
    print(f"Wrote clinical input summary workbook: {output_path}")

    workflow_config = config.get("workflows", {}).get(WORKFLOW_NAME, {})
    section_map = None

    if workflow_config.get("compute_metrics", True):
        section_map = compute_clinical_pretraining_metrics(
            clinical_df=clinical_df,
            config=config,
        )

        dataset_name = config.get("cohort", {}).get("dataset", "dataset")

        metrics_output_path = paths.tables_dir / "clinical_pretraining_bias_metrics.xlsx"
        summary_output_path = paths.tables_dir / "clinical_pretraining_bias_metrics_summary.xlsx"

        write_sectioned_pretraining_metrics_excel(
            section_map=section_map,
            output_path=metrics_output_path,
            dataset_name=dataset_name,
        )

        write_pretraining_summary_excel(
            section_map=section_map,
            output_path=summary_output_path,
            config=config,
        )

        print(f"Wrote pretraining metric workbook: {metrics_output_path}")
        print(f"Wrote pretraining summary workbook: {summary_output_path}")
    

    if workflow_config.get("compute_counts", True):
        count_tables = build_clinical_count_tables(
            clinical_df=clinical_df,
            config=config,
        )

        count_output_path = paths.tables_dir / "clinical_distribution_count_tables.xlsx"

        write_clinical_count_tables_excel(
            count_tables=count_tables,
            output_path=count_output_path,
        )

        print(f"Wrote clinical count workbook: {count_output_path}")

    if workflow_config.get("make_heatmaps", True):
        if count_tables is None:
            count_tables = build_clinical_count_tables(
                clinical_df=clinical_df,
                config=config,
            )

        figure_paths = generate_clinical_distribution_figures(
            count_tables=count_tables,
            output_dir=paths.figures_dir,
            config=config,
        )

        print(f"Wrote {len(figure_paths)} clinical distribution figure(s) to: {paths.figures_dir}")
    
    if workflow_config.get("make_metric_heatmaps", True):
        if section_map is None:
            section_map = compute_clinical_pretraining_metrics(
                clinical_df=clinical_df,
                config=config,
            )

        metric_heatmap_paths = generate_clinical_metric_heatmaps(
            section_map=section_map,
            output_dir=paths.figures_dir,
            config=config,
        )

        print(
            f"Wrote {len(metric_heatmap_paths)} clinical pretraining metric heatmap(s) "
            f"to: {paths.figures_dir}"
        )


def main() -> None:
    config = load_config_from_args()
    run(config)


if __name__ == "__main__":
    main()