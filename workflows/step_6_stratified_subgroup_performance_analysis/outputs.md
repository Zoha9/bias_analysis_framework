# Step 6 Outputs

Outputs are written under:

```text
{project.output_root}/stratified_subgroup_performance_analysis/{organ}__{dataset}/
```

## Folder structure

Expected structure:

```text
stratified_subgroup_performance_analysis/
└── {organ}__{dataset}/
    ├── tables/
    │   └── stratified_subgroup_performance.xlsx
    └── figures/
        ├── subgroup_performance/
        ├── starplots/
        └── disparities/
```

## Main outputs

Workbook sheets

### `subgroup_aggregate`

Performance summarized across splits for each target, model, bias factor, and subgroup.

### `subgroup_by_split`

One row per target, model, split, bias factor, and subgroup.

### `disparity_summary`

Contains best subgroup, worst subgroup, absolute gap, subgroup mean, and subgroup variability.

### `pairwise_gaps`

Contains pairwise subgroup differences for factors with two or more groups.

### `subgroup_counts`

Shows the number of evaluated patients per subgroup and split.

### `file_status`

Records missing columns, insufficient subgroup sizes, metric failures, and completed files.

Figures include grouped bars, subgroup trajectories, radar plots, and absolute disparity summaries.
