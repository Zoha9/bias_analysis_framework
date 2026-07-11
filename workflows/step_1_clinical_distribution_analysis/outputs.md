# Step 1 Outputs

Outputs are written under:

```text
{project.output_root}/clinical_distribution_analysis/{organ}__{dataset}/
```

Example:

```text
results/clinical_distribution_analysis/CRC__CPTAC/
```

## Folder structure

Expected structure:

```text
clinical_distribution_analysis/
└── {organ}__{dataset}/
    ├── tables/
    │   ├── clinical_input_data_summary.xlsx
    │   ├── clinical_pretraining_bias_metrics.xlsx
    │   ├── clinical_pretraining_bias_metrics_summary.xlsx
    │   └── clinical_distribution_count_tables.xlsx
    └── figures/
        ├── target_class_counts/
        ├── bias_factor_subgroup_counts/
        ├── age_counts/
        ├── tissue_source_site_counts/
        ├── bias_factor_by_target_class_counts/
        └── clinical_pretraining_bias_metric_heatmaps/
```

Some figure folders may be absent when the corresponding data columns are unavailable. For example, `tissue_source_site_counts/` will not be created if the clinical table does not contain a tissue source site column.

## Tables

### `clinical_input_data_summary.xlsx`

This workbook summarizes the input clinical and slide tables.

Expected sheets include:

- `clinical_summary`
- `slide_summary`
- `patient_overlap`
- `clinical_preview`
- `slide_preview`
- One value-count sheet for each configured target and bias factor

Review this workbook first. It confirms that the configured input files and columns were read correctly.

### `clinical_pretraining_bias_metrics.xlsx`

This is the detailed pretraining bias metric workbook.

It may include:

- Summary metrics
- Subgroup counts
- Subgroup positive rates
- Pairwise SPD matrices
- Pairwise DI matrices
- JSD tables
- TVD tables
- Multiclass distribution tables
- Continuous-age association metrics

This workbook is detailed and may contain many sections.

### `clinical_pretraining_bias_metrics_summary.xlsx`

This is the compact pretraining bias metric summary workbook.

Expected sheets:

- `summary_long`
- `summary_wide`

Use this workbook for easier review, filtering, and downstream comparisons.

### `clinical_distribution_count_tables.xlsx`

This workbook contains count-focused tables.

Expected sheets may include:

- `target_class_counts`
- `bias_factor_subgroup_counts`
- `bias_factor_by_target_class_counts`
- `age_counts`
- `tissue_source_site_counts`
- `patient_count_summary`

Some sheets may be absent when the corresponding columns are unavailable.

## Figures

### `target_class_counts/`

Contains one bar plot per target biomarker showing class counts.

Examples:

```text
MSI_Status__class_counts.png
KRAS__class_counts.png
```

### `bias_factor_subgroup_counts/`

Contains one bar plot per bias factor showing subgroup counts.

Examples:

```text
Gender__subgroup_counts.png
Race_Binary__subgroup_counts.png
```

### `age_counts/`

Contains age-count distribution figures when an age column is available.

Example:

```text
age_counts.png
```

### `tissue_source_site_counts/`

Contains tissue source site count figures when a tissue source site column is available.

Example:

```text
tissue_source_site_counts.png
```

### `bias_factor_by_target_class_counts/`

Contains count heatmaps showing bias-factor subgroups across target biomarker classes.

Example:

```text
bias_factor_by_target_class_counts_heatmap.png
```

### `clinical_pretraining_bias_metric_heatmaps/`

Contains heatmaps generated from pretraining bias metrics.

Possible heatmaps include:

- Absolute statistical parity difference
- Conditional demographic disparity
- Difference in positive proportions
- Maximum pairwise SPD
- Disparate impact ratio
- Disparate impact deviation from parity
- Coefficient of variation across positive rates
- Range of positive rates
- Gini coefficient across subgroup positive rates
- Cramér's V
- Chi-square significance
- JSD to centroid
- TVD to centroid

Some heatmaps may be absent when a metric cannot be computed for the configured data.