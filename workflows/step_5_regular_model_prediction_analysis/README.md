# Step 5: Regular Model Prediction Analysis

This workflow deploys trained STAMP checkpoints and evaluates regular target prediction performance.

## Purpose

Step 5 has two parts:

1. Deploy each trained model and split on the configured cohort.
2. Calculate split-level and aggregate prediction metrics.

It answers:

- Were checkpoints and deployment features found?
- Were patient prediction files generated?
- How well does each model predict each target?
- How stable is performance across splits?

## Required inputs

- Step 4 checkpoints
- Deployment feature embeddings
- Clinical and slide tables
- STAMP deployment template

Relevant YAML section:

```yaml
workflows:
  regular_model_prediction_analysis:
    run: true
    execute: false
    splits: null
    analyze_predictions: true
```
`splits: null` deploys all available checkpoints. A list such as [0, 2] deploys selected splits.

## How to run

    python workflows/step_5_regular_model_prediction_analysis/run_regular_model_prediction_analysis.py \
        --config configs/local_test_config.yaml
    
## Output location

    {project.output_root}/regular_model_prediction_analysis/{organ}__{dataset}/

See [`outputs.md`](outputs.md) for the generated files.