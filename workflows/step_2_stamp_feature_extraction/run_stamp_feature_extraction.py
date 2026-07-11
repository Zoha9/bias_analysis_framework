from __future__ import annotations

import sys
from pathlib import Path

# Allow running the workflow directly from the repository root without
# requiring the package to be installed first.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from bias_analysis_framework.config.loader import (
    load_config_from_args,
    workflow_enabled,
)
from bias_analysis_framework.io.paths import get_workflow_paths
from bias_analysis_framework.stamp.preprocess import (
    run_stamp_feature_extraction as execute_stamp_feature_extraction,
)


WORKFLOW_NAME = "stamp_feature_extraction"


def run(config: dict) -> None:
    """Run Step 2 STAMP feature extraction.

    The workflow:

    1. Reads the framework YAML config.
    2. Validates the WSI root, feature root, STAMP template, and extractors.
    3. Generates one STAMP preprocessing YAML file per extractor.
    4. Writes one command log per extractor.
    5. Optionally executes STAMP preprocessing.
    6. Writes an Excel job summary.

    By default, users should first run with ``execute: false``. This verifies
    paths and generated STAMP configs without starting WSI processing.
    """
    if not workflow_enabled(config, WORKFLOW_NAME):
        print(f"Workflow '{WORKFLOW_NAME}' is disabled in config. Nothing to run.")
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

    execute_requested = bool(workflow_config.get("execute", False))

    if execute_requested:
        print("Starting STAMP feature extraction.")
        print("STAMP commands will be executed.")
    else:
        print("Starting STAMP feature extraction dry run.")
        print("STAMP configs and commands will be generated but not executed.")

    summary_df = execute_stamp_feature_extraction(
        config=config,
        workflow_output_dir=paths.workflow_root,
    )

    summary_path = (
        paths.tables_dir
        / "stamp_feature_extraction_jobs.xlsx"
    )

    print()
    print("Step 2 STAMP feature extraction workflow completed.")
    print(f"Configured jobs: {len(summary_df)}")
    print(f"Job summary workbook: {summary_path}")
    print(f"Generated STAMP configs: {paths.workflow_root / 'configs'}")
    print(f"STAMP command logs: {paths.logs_dir}")

    if summary_df.empty:
        print("No STAMP feature-extraction jobs were created.")
        return

    if "status" in summary_df.columns:
        status_counts = summary_df["status"].value_counts(dropna=False)

        print("Job statuses:")
        for status, count in status_counts.items():
            print(f"  {status}: {count}")

    if "output_dir" in summary_df.columns:
        output_directories = (
            summary_df["output_dir"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        if output_directories:
            print("Expected STAMP feature output directories:")
            for output_directory in output_directories:
                print(f"  {output_directory}")

    failed_jobs = (
        summary_df[summary_df["status"] == "failed"]
        if "status" in summary_df.columns
        else summary_df.iloc[0:0]
    )

    if not failed_jobs.empty:
        failed_extractors = failed_jobs["extractor"].astype(str).tolist()

        raise RuntimeError(
            "One or more STAMP feature-extraction jobs failed: "
            + ", ".join(failed_extractors)
            + f". Check logs under: {paths.logs_dir}"
        )


def main() -> None:
    """Load the user config and run Step 2."""
    config = load_config_from_args()
    run(config)


if __name__ == "__main__":
    main()