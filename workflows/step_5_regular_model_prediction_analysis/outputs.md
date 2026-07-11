# Step 5 Outputs

Outputs are written under:

```text
{project.output_root}/regular_model_prediction_analysis/{organ}__{dataset}/
```

## Folder structure

Expected structure:

```text
regular_model_prediction_analysis/
└── {organ}__{dataset}/
    ├── configs/
    ├── logs/
    ├── predictions/
    │   └── {target}/{model}/split-{n}/
    ├── tables/
    │   ├── regular_deployment_jobs.xlsx
    │   └── regular_prediction_performance.xlsx
    └── figures/
        ├── target_performance/
        └── patient_scores/
```

## Main outputs

### `regular_deployment_jobs.xlsx`

Summarizes deployment configs, checkpoints, output paths, prediction file counts, and status.

### Prediction files

STAMP normally writes:

### `patient-preds.csv` 

 and

### `patient-preds-0.csv` 

Expected sheets:

aggregate_metrics
split_metrics
patient_scores
file_status

### `regular_prediction_performance.xlsx`

Metrics include accuracy, balanced accuracy, F1, AUROC, AUPRC, sensitivity, specificity, precision, and NPV.

### Figures

Target-performance figures compare models and splits. Patient-score figures show positive-class probabilities by true class. 
