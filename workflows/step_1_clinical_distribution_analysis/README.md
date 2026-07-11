# Step 1: Clinical Distribution Analysis

This workflow analyzes the clinical input data before model training. It checks that the configured clinical and slide tables are readable, verifies that required columns exist, summarizes target biomarkers and bias factors, computes pretraining bias metrics, and generates count and metric heatmap figures.

## Purpose

Step 1 answers these questions:

1. Are the clinical and slide input files correctly formatted?
2. Do the configured target biomarker columns exist?
3. Do the configured bias factor columns exist?
4. How many patients, slides, target classes, and bias-factor subgroups are present?
5. Are target labels unevenly distributed across population subgroups? (count/distribution plots)
6. Which pretraining bias metrics show the strongest subgroup differences? (pretraining metric workbooks and metric heatmaps)

This step should be run before STAMP feature extraction, model training, or downstream bias evaluation.

## Required input files

The workflow requires:

- A clinical table
- A slide table

The paths are configured in the YAML config file:

```yaml
data:
      clinical_table: "/path/to/clinical_table.xlsx"
      slide_table: "/path/to/slide_table.csv"
      patient_column: "PATIENT"
      slide_filename_column: "FILENAME"
```
## Clinical table format

The clinical table must contain:

- One patient identifier column, usually `PATIENT`
- All configured target biomarker columns
- All configured bias factor columns

Example clinical columns:

    PATIENT
    MSI_Status
    PIK3CA
    BRAF
    KRAS
    Gender
    Age_MedianSplit
    Age_Group
    Race_Binary

## Slide table format

The slide table must contain:

- One slide or feature filename column, usually `FILENAME`
- One patient identifier column, usually `PATIENT`

Example slide table:

    FILENAME,PATIENT
    slide_001.h5,patient_001
    slide_002.h5,patient_002

The patient column is used to check overlap between the clinical and slide tables.

## Target biomarker configuration

Targets are configured in the YAML configuration file.

Binary target example:

    targets:
      - name: "MSI_Status"
        negative_label: "MSS"
        positive_label: "MSI-H"

Mutation target example:

    targets:
      - name: "KRAS"
        negative_label: "WT"
        positive_label: "MUT"

Multiclass target example:

    targets:
      - name: "BRCA_Subtype_PAM50"
        labels:
          - "LumA"
          - "LumB"
          - "Basal"
          - "Her2"
          - "Normal"

## Bias factor configuration

Bias factors are configured in the YAML configuration file.

Example:

    bias_factors:
      - name: "Gender"
      - name: "Race_Binary"
      - name: "Age_MedianSplit"
      - name: "Age_Group"

## Workflow toggles

This workflow is controlled by the `clinical_distribution_analysis` section of the configuration.

Example:

    workflows:
      clinical_distribution_analysis:
        run: true
        compute_metrics: true
        compute_counts: true
        make_heatmaps: true
        make_metric_heatmaps: true

Toggle meanings:

- `run`: Runs or skips the workflow.
- `compute_metrics`: Writes pretraining bias metric workbooks.
- `compute_counts`: Writes count summary tables.
- `make_heatmaps`: Writes count and distribution figures.
- `make_metric_heatmaps`: Writes pretraining metric heatmaps.

## How to run

From the repository root:

    python workflows/step_1_clinical_distribution_analysis/run_clinical_distribution_analysis.py \
      --config configs/local_test_config.yaml

## Output location

Outputs are written to:

    {project.output_root}/clinical_distribution_analysis/{organ}__{dataset}/

Example:

    results/clinical_distribution_analysis/CRC__CPTAC/

## Main outputs

The workflow writes tables to:

    tables/

It writes figures to:

    figures/

See [`outputs.md`](outputs.md) for the complete output list.

## Notes

This workflow does not train models and does not run STAMP.

It analyzes the input clinical data and pretraining label distributions. STAMP feature extraction and model training are handled by later workflow steps.

