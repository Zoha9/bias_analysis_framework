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
from bias_analysis_framework.stamp.crossval import (
    run_stamp_crossval_training,
)


WORKFLOW_NAME = "stamp_crossval_training"


def run(config: dict[str, Any]) -> None:
    """Run Step 4 STAMP cross-validation training."""
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
            "Starting STAMP cross-validation training."
        )
        print(
            "STAMP commands will be executed."
        )
    else:
        print(
            "Starting STAMP cross-validation dry run."
        )
        print(
            "Configs and commands will be generated "
            "without running STAMP."
        )

    summary_df = run_stamp_crossval_training(
        config=config,
        workflow_output_dir=paths.workflow_root,
    )

    summary_path = (
        paths.tables_dir
        / "stamp_crossval_jobs.xlsx"
    )

    print()
    print(
        "Step 4 STAMP cross-validation workflow completed."
    )
    print(
        f"Configured jobs: {len(summary_df)}"
    )
    print(
        f"Job summary workbook: {summary_path}"
    )
    print(
        f"Generated configs: "
        f"{paths.workflow_root / 'configs'}"
    )
    print(
        f"Logs: {paths.logs_dir}"
    )
    print(
        f"Crossval results: "
        f"{paths.workflow_root / 'crossval_results'}"
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
        failed_jobs = [
            (
                f"{row['Extractor']} / "
                f"{row['Label']}"
            )
            for _, row in failed.iterrows()
        ]

        raise RuntimeError(
            "One or more STAMP cross-validation "
            "jobs failed: "
            + ", ".join(failed_jobs)
        )


def main() -> None:
    """Load config and run Step 4."""
    config = load_config_from_args()
    run(config)


if __name__ == "__main__":
    main()