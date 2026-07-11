from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bias_analysis_framework.config.loader import (
    load_config_from_args,
    workflow_enabled,
)
from bias_analysis_framework.io.paths import (
    get_workflow_paths,
)
from bias_analysis_framework.model_evaluation.stratified import (
    analyze_stratified_predictions,
    write_stratified_performance_workbook,
)
from bias_analysis_framework.model_evaluation.stratified_visualization import (
    generate_stratified_performance_figures,
)


WORKFLOW_NAME = (
    "stratified_subgroup_performance_analysis"
)


def run(config: dict[str, Any]) -> None:
    """Run Step 6 subgroup-stratified performance analysis."""
    if not workflow_enabled(
        config,
        WORKFLOW_NAME,
    ):
        print(
            f"Workflow '{WORKFLOW_NAME}' is disabled. "
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

    predictions_root = (
        resolve_predictions_root(
            config=config,
            workflow_config=workflow_config,
        )
    )

    print(
        "Starting subgroup-stratified "
        "performance analysis."
    )
    print(
        f"Prediction root: {predictions_root}"
    )

    result = analyze_stratified_predictions(
        config=config,
        predictions_root=predictions_root,
    )

    workbook_path = (
        paths.tables_dir
        / "stratified_subgroup_performance.xlsx"
    )

    write_stratified_performance_workbook(
        result=result,
        output_path=workbook_path,
    )

    figure_paths = (
        generate_stratified_performance_figures(
            aggregate_metrics=(
                result.subgroup_aggregate_metrics
            ),
            disparity_summary=(
                result.disparity_summary
            ),
            config=config,
            output_dir=paths.figures_dir,
        )
    )

    print()
    print(
        "Step 6 subgroup analysis completed."
    )
    print(
        f"Split-level subgroup metric rows: "
        f"{len(result.subgroup_split_metrics)}"
    )
    print(
        f"Aggregate subgroup metric rows: "
        f"{len(result.subgroup_aggregate_metrics)}"
    )
    print(
        f"Disparity summary rows: "
        f"{len(result.disparity_summary)}"
    )
    print(
        f"Workbook: {workbook_path}"
    )
    print(
        f"Figures written: {len(figure_paths)}"
    )

    failures = (
        result.file_status[
            result.file_status["Status"].isin(
                [
                    "file_failure",
                    "metric_failure",
                ]
            )
        ]
        if (
            not result.file_status.empty
            and "Status"
            in result.file_status.columns
        )
        else result.file_status.iloc[0:0]
    )

    continue_on_error = bool(
        workflow_config.get(
            "continue_on_error",
            True,
        )
    )

    if (
        not failures.empty
        and not continue_on_error
    ):
        raise RuntimeError(
            f"Step 6 encountered "
            f"{len(failures)} failed analysis item(s). "
            f"See {workbook_path}."
        )


def resolve_predictions_root(
    *,
    config: dict[str, Any],
    workflow_config: dict[str, Any],
) -> Path:
    """Resolve regular Step 5 prediction outputs."""
    configured = workflow_config.get(
        "predictions_root"
    )

    if configured:
        return Path(
            configured
        ).expanduser().resolve()

    output_root = Path(
        config["project"]["output_root"]
    ).expanduser().resolve()

    cohort = config.get(
        "cohort",
        {},
    )
    organ = str(
        cohort.get(
            "organ",
            "unknown_organ",
        )
    ).strip()
    dataset = str(
        cohort.get(
            "dataset",
            "unknown_dataset",
        )
    ).strip()

    return (
        output_root
        / "regular_model_prediction_analysis"
        / f"{organ}__{dataset}"
        / "predictions"
    )


def main() -> None:
    """Load config and run Step 6."""
    config = load_config_from_args()
    run(config)


if __name__ == "__main__":
    main()