# Step 4: STAMP Cross-Validation Training

This workflow trains STAMP classifiers for configured targets and optional bias-factor labels using extracted feature embeddings.

## Purpose

Step 4 answers these questions:

1. Are feature directories available for each model?
2. Which target and bias-factor labels should be trained?
3. Can one STAMP cross-validation config be generated per model and label?
4. Were the expected split directories and checkpoints produced?

Cross-validation is kept separate from deployment so trained checkpoints can be reused on other cohorts.

## Required inputs

- Step 2 feature embeddings
- Clinical table
- Slide table
- STAMP cross-validation template
- Configured models and labels

Relevant YAML section:

```yaml
workflows:
  stamp_crossval_training:
    run: true
    execute: false
    n_splits: 5
    label_sources:
      - targets
      - bias_factors
```


## How to run
    
    python workflows/step_4_stamp_model_training/run_stamp_model_training.py \
        --config configs/local_test_config.yaml

Run with execute: false before enabling training.

## Output location

    {project.output_root}/stamp_crossval_training/{organ}__{dataset}/

See [`outputs.md`](outputs.md) for the generated files.