# Step 6: Stratified Subgroup Performance Analysis

This workflow evaluates Step 5 predictions separately within configured population subgroups.

## Purpose

Step 6 answers these questions:

1. Does model performance differ across demographic or clinical subgroups?
2. Which subgroup has the highest and lowest performance?
3. How large are best-to-worst and pairwise subgroup gaps?
4. Are subgroup differences consistent across models and splits?

The workflow reuses existing predictions. It does not redeploy STAMP for each subgroup.

## Required inputs

- Step 5 prediction files
- Clinical table with patient IDs and bias-factor columns
- Configured bias factors and subgroup labels

Relevant YAML section:

```yaml
workflows:
  stratified_subgroup_performance_analysis:
    run: true
    predictions_root: null
    minimum_subgroup_size: 5
    bias_factors: null
```

## How to run
    
    python workflows/step_6_stratified_subgroup_performance_analysis/run_stratified_subgroup_performance_analysis.py \
        --config configs/local_test_config.yaml
    
## Output location

    {project.output_root}/stratified_subgroup_performance_analysis/{organ}__{dataset}/

See [`outputs.md`](outputs.md) for the generated files.