# Output Structure

All workflow outputs are organized below the configured project output root.

```yaml
project:
  output_root: "/path/to/results"
```

## Shared path convention

Each workflow writes to:

```text
{project.output_root}/{workflow_name}/{organ}__{dataset}/
```

Example:

```text
/path/to/results/clinical_distribution_analysis/CRC__TCGA/
```

The cohort folder uses a double underscore between organ and dataset.

## Common subdirectories

Not every workflow creates every directory.

```text
configs/           generated workflow or STAMP configurations
logs/              commands, standard output, standard error, and status logs
tables/            CSV and Excel summaries
figures/           plots and visual summaries
predictions/       patient-level deployment outputs
crossval_results/  trained split directories and checkpoints
```

## Typical repository-wide layout

```text
{output_root}/
├── clinical_distribution_analysis/
│   └── {organ}__{dataset}/
│       ├── tables/
│       └── figures/
├── stamp_feature_extraction/
│   └── {organ}__{dataset}/
│       ├── configs/
│       └── logs/
├── feature_embedding_analysis/
│   └── {organ}__{dataset}/
│       ├── tables/
│       └── figures/
├── stamp_crossval_training/
│   └── {organ}__{dataset}/
│       ├── configs/
│       ├── logs/
│       └── crossval_results/
├── regular_model_prediction_analysis/
│   └── {organ}__{dataset}/
│       ├── configs/
│       ├── logs/
│       ├── predictions/
│       ├── tables/
│       └── figures/
├── stratified_subgroup_performance_analysis/
│   └── {organ}__{dataset}/
│       ├── tables/
│       └── figures/
└── comparative_bias_propagation_analysis/
    └── {organ}__{dataset}/
        ├── tables/
        └── figures/
```

## Job summary workbooks

STAMP-facing workflows produce job summaries that record generated configurations, commands, execution status, and output locations. Important examples include:

```text
stamp_feature_extraction_jobs.xlsx
stamp_crossval_jobs.xlsx
regular_deployment_jobs.xlsx
```

These files should be retained even for dry runs because they document the generated work plan.

## Analysis workbooks

Major analysis outputs include:

```text
clinical_input_data_summary.xlsx
clinical_pretraining_bias_metrics.xlsx
regular_prediction_performance.xlsx
stratified_subgroup_performance.xlsx
comparative_bias_propagation.xlsx
```

Workbook sheets separate aggregate metrics, split-level metrics, patient-level scores, subgroup counts, disparity summaries, and file-status information as appropriate.

## Checkpoints

Step 4 stores split-specific checkpoints under a label and model hierarchy. A typical checkpoint path is:

```text
crossval_results/{label}/{model}/split-{n}/model.ckpt
```

Directory discovery supports normalized matching for labels and models, but output producers should retain stable names to avoid ambiguity.

## Predictions

STAMP deployment commonly produces patient-level files such as:

```text
patient-preds.csv
patient-preds-0.csv
```

Downstream workflows discover these files rather than requiring them to be manually copied.

## Exact step outputs

The shared structure above explains where artifacts live. Exact filenames, workbook sheets, figures, status values, and interpretation belong in each workflow's `outputs.md` file.