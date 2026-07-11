# Step 7 Outputs

Outputs are written under:

```text
{project.output_root}/comparative_bias_propagation_analysis/{organ}__{dataset}/
```

## Folder structure

Expected structure:

```text
comparative_bias_propagation_analysis/
└── {organ}__{dataset}/
    ├── tables/
    │   └── comparative_bias_propagation.xlsx
    └── figures/
        ├── delta_heatmaps/
        └── before_after/
```

## Main outputs

Workbook sheets

### `amplification_summary`

Provides readable classifications:

### `- amplified`
### `- attenuated`
### `- unchanged`
### `- mixed`
### `- not_available`

### `aggregate_metrics`

Contains means, standard deviations, and valid split counts for pretraining, posttraining, and delta metrics.

### `split_metrics`

Contains one row per target, class, model, split, and bias factor.

### `file_status`

Records successfully analyzed and failed prediction files.

## Main comparative metrics

### `- Pretraining_PositiveRateGap`
### `- Posttraining_PositiveRateGap`
### `- Delta_PositiveRateGap`
### `- Pretraining_DI`
### `- Posttraining_DI`
### `- Delta_DI_Distance`
### `- AccuracyGap`
### `- RecallGap`
### `- AUCGap`

Positive delta values indicate greater posttraining disparity. Negative values indicate reduced disparity.

Figures include model-by-factor delta heatmaps and before-versus-after positive-rate gap plots.