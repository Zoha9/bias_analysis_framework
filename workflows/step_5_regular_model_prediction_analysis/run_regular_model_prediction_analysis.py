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
from bias_analysis_framework.stamp.deploy_regular import (
    run_regular_stamp_deployment,
)

from bias_analysis_framework.model_evaluation.regular import (
    analyze_regular_predictions,
    write_regular_prediction_analysis,
)
from bias_analysis_framework.model_evaluation.visualization import (
    generate_regular_performance_figures,
)

WORKFLOW_NAME = (
    "regular_model_prediction_analysis"
)


def run(config: dict[str, Any]) -> None:
    """Run regular STAMP deployment for Step 5."""
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

    execute = bool(
        workflow_config.get(
            "execute",
            False,
        )
    )

    if execute:
        print(
            "Starting regular STAMP deployment."
        )
        print(
            "STAMP deployment commands will be "
            "executed."
        )
    else:
        print(
            "Starting regular STAMP deployment "
            "dry run."
        )
        print(
            "Deployment configs and commands will "
            "be generated without running STAMP."
        )

    summary_df = run_regular_stamp_deployment(
        config=config,
        workflow_output_dir=(
            paths.workflow_root
        ),
    )

    analyze_predictions = bool(
        workflow_config.get(
            "analyze_predictions",
            True,
        )
    )

    analysis_result = None
    analysis_workbook = None
    figure_paths: list[Path] = []

    if analyze_predictions:
        predictions_root = (
            paths.workflow_root
            / "predictions"
        )

        analysis_result = (
            analyze_regular_predictions(
                config=config,
                predictions_root=predictions_root,
            )
        )

        analysis_workbook = (
            paths.tables_dir
            / "regular_prediction_performance.xlsx"
        )

        write_regular_prediction_analysis(
            result=analysis_result,
            output_path=analysis_workbook,
        )

        figure_paths = (
            generate_regular_performance_figures(
                split_metrics=(
                    analysis_result.split_metrics
                ),
                patient_scores=(
                    analysis_result.patient_scores
                ),
                config=config,
                output_dir=paths.figures_dir,
            )
        )

    summary_path = (
        paths.tables_dir
        / "regular_deployment_jobs.xlsx"
    )

    print()
    print(
        "Regular STAMP deployment workflow "
        "completed."
    )
    print(
        f"Configured deployment jobs: "
        f"{len(summary_df)}"
    )
    print(
        f"Job summary workbook: "
        f"{summary_path}"
    )
    print(
        f"Generated configs: "
        f"{paths.workflow_root / 'configs'}"
    )
    print(
        f"Deployment logs: "
        f"{paths.logs_dir}"
    )
    print(
        f"Prediction outputs: "
        f"{paths.workflow_root / 'predictions'}"
    )

    if analysis_workbook is not None:
        print(
            f"Prediction performance workbook: "
            f"{analysis_workbook}"
        )
        print(
            f"Performance figures written: "
            f"{len(figure_paths)}"
        )

        if analysis_result is not None:
            print(
                f"Analyzed prediction files: "
                f"{len(analysis_result.split_metrics)}"
            )

    if (
        not summary_df.empty
        and "Status" in summary_df.columns
    ):
        print("Job statuses:")

        for status, count in (
            summary_df["Status"]
            .value_counts(dropna=False)
            .items()
        ):
            print(
                f"  {status}: {count}"
            )

    failed = (
        summary_df[
            summary_df["Status"]
            == "failed"
        ]
        if "Status" in summary_df.columns
        else summary_df.iloc[0:0]
    )

    if not failed.empty:
        failed_descriptions = [
            (
                f"{row['Extractor']} / "
                f"{row['Label']} / "
                f"split-{row['Split']}"
            )
            for _, row in failed.iterrows()
        ]

        raise RuntimeError(
            "One or more regular STAMP "
            "deployment jobs failed: "
            + ", ".join(
                failed_descriptions
            )
        )


def main() -> None:
    """Load config and run Step 5."""
    config = load_config_from_args()
    run(config)


if __name__ == "__main__":
    main()