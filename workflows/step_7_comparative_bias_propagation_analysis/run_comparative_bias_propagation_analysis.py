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
from bias_analysis_framework.model_evaluation.comparative import (
    analyze_comparative_bias,
    write_comparative_bias_workbook,
)
from bias_analysis_framework.model_evaluation.comparative_visualization import (
    generate_comparative_bias_figures,
)


WORKFLOW_NAME = (
    "comparative_bias_propagation_analysis"
)


def run(config: dict[str, Any]) -> None:
    """Run comparative pretraining/posttraining bias analysis."""
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

    predictions_root = resolve_predictions_root(
        config=config,
        workflow_config=workflow_config,
    )

    print(
        "Starting comparative bias propagation analysis."
    )
    print(
        f"Prediction root: {predictions_root}"
    )

    result = analyze_comparative_bias(
        config=config,
        predictions_root=predictions_root,
    )

    workbook_path = (
        paths.tables_dir
        / "comparative_bias_propagation.xlsx"
    )

    write_comparative_bias_workbook(
        result=result,
        output_path=workbook_path,
    )

    figure_paths = (
        generate_comparative_bias_figures(
            aggregate_metrics=(
                result.aggregate_metrics
            ),
            config=config,
            output_dir=paths.figures_dir,
        )
    )

    print()
    print(
        "Comparative bias propagation analysis completed."
    )
    print(
        f"Split metric rows: "
        f"{len(result.split_metrics)}"
    )
    print(
        f"Aggregate metric rows: "
        f"{len(result.aggregate_metrics)}"
    )
    print(
        f"Amplification summary rows: "
        f"{len(result.amplification_summary)}"
    )
    print(
        f"Workbook: {workbook_path}"
    )
    print(
        f"Figures written: {len(figure_paths)}"
    )

    failures = (
        result.file_status[
            result.file_status["Status"]
            == "file_failure"
        ]
        if (
            not result.file_status.empty
            and "Status"
            in result.file_status.columns
        )
        else result.file_status.iloc[0:0]
    )

    if (
        not failures.empty
        and not bool(
            workflow_config.get(
                "continue_on_error",
                True,
            )
        )
    ):
        raise RuntimeError(
            f"Step 7 encountered "
            f"{len(failures)} failed prediction file(s). "
            f"See {workbook_path}."
        )


def resolve_predictions_root(
    *,
    config: dict[str, Any],
    workflow_config: dict[str, Any],
) -> Path:
    """Resolve Step 5 prediction outputs."""
    configured = workflow_config.get(
        "predictions_root"
    )

    if configured:
        return (
            Path(configured)
            .expanduser()
            .resolve()
        )

    output_root = (
        Path(
            config["project"]["output_root"]
        )
        .expanduser()
        .resolve()
    )

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
    """Load config and run Step 7."""
    config = load_config_from_args()
    run(config)


if __name__ == "__main__":
    main()