# Step 4 Outputs

Outputs are written under:

```text
{project.output_root}/stamp_crossval_training/{organ}__{dataset}/
```

## Folder structure

Expected structure:

```text
stamp_crossval_training/
└── {organ}__{dataset}/
    ├── configs/
    ├── logs/
    ├── tables/
    │   └── stamp_crossval_jobs.xlsx
    └── crossval_results/
        └── {label}/{model}/split-{n}/
```

## Main outputs

The workflow writes tables to:

### `configs/`

Contains one generated STAMP cross-validation YAML per model and label.

### `logs/`

Contains one training log per model and label.


### `stamp_crossval_jobs.xlsx`

Summarizes feature paths, labels, execution status, split counts, configs, and logs.

 ### `crossval_results/`

Contains STAMP split outputs and trained checkpoints, normally:

### `model.ckpt`

A missing split or checkpoint should be investigated before Step 5 deployment.