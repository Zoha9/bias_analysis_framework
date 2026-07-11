# Step 7: Comparative Bias Propagation Analysis

This workflow compares subgroup disparities in ground-truth labels with disparities in model predictions.

## Purpose

Step 7 answers these questions:

1. Was subgroup disparity already present before training?
2. Did model predictions amplify or reduce that disparity?
3. How did positive-rate gaps and disparate impact change?
4. Which targets, models, and bias factors show the largest propagation?

For multiclass targets, metrics are calculated one class versus the remaining classes.

## Required inputs

- Step 5 prediction files
- Clinical table
- Configured targets and positive labels
- Configured bias factors

Relevant YAML section:

```yaml
workflows:
  comparative_bias_propagation_analysis:
    run: true
    predictions_root: null
    minimum_group_size: 5
    laplace_alpha: 1.0
    bias_factors: null
```

## Interpretation

### ` delta = posttraining disparity - pretraining disparity `

A positive delta means amplification. A negative delta means attenuation.

Disparate impact is converted to distance from parity:

### ` DI distance = |1 - DI| `

    
## How to run
    
    python workflows/step_7_comparative_bias_propagation_analysis/run_comparative_bias_propagation_analysis.py \
        --config configs/local_test_config.yaml
    
## Output location

    {project.output_root}/comparative_bias_propagation_analysis/{organ}__{dataset}/

See [`outputs.md`](outputs.md) for the generated files.
