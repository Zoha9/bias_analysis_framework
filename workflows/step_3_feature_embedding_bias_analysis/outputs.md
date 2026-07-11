# Step 3 Outputs

Outputs are written under:

```text
{project.output_root}/feature_embedding_analysis/{organ}__{dataset}/
```

## Folder structure

Expected structure:

```text
feature_embedding_analysis/
└── {organ}__{dataset}/
    ├── tables/
    │   ├── model_status.xlsx
    │   ├── combined_embedding_bias_metrics.xlsx
    │   └── {model}/
    │       ├── {model}_embedding_bias_metrics.xlsx
    │       └── projection tables
    └── figures/
        ├── {model}/
        └── cross_model/
```

## Main outputs

### Model metric workbook

Contains patient counts, subgroup counts, robustness metrics, k-nearest-neighbor accuracy, and clustering metrics.

### Projection tables

Contain patient metadata joined with PCA, t-SNE, and UMAP coordinates.

### Model figures

Show target and bias-factor labels on embedding projections.

### Cross-model figures

Compare the same target or bias factor across feature extractors.

Missing figures usually indicate unavailable columns, insufficient group size, or a disabled visualization option.