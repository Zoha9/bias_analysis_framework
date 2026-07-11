# Workflow Guide

Run workflow commands from the repository root with a local copy of the example configuration.

```bash
cp configs/bias_analysis_config.example.yaml configs/local_config.yaml
```

Do not commit local paths, credentials, protected data locations, or access tokens.

## Recommended execution order

| Step | Directory | Depends on |
|---|---|---|
| 1 | `workflows/step_1_clinical_distribution_analysis/` | Clinical and slide tables |
| 2 | `workflows/step_2_stamp_feature_extraction/` | Slide table, WSIs, STAMP |
| 3 | `workflows/step_3_feature_embedding_bias_analysis/` | Clinical data and feature embeddings |
| 4 | `workflows/step_4_stamp_model_training/` | Feature embeddings, clinical labels, STAMP |
| 5 | `workflows/step_5_regular_model_prediction_analysis/` | Step 4 checkpoints and deployment inputs |
| 6 | `workflows/step_6_stratified_subgroup_performance_analysis/` | Step 5 predictions and clinical subgroup columns |
| 7 | `workflows/step_7_comparative_bias_propagation_analysis/` | Pretraining information and posttraining prediction analyses |

Steps should normally be run in numerical order. Existing compatible artifacts can be reused when their paths and naming conventions match the current configuration.

## Configure workflow selection

Each workflow has a section below `workflows` in the YAML file. Use its `run` setting to make the intended state explicit.

```yaml
workflows:
  clinical_distribution_analysis:
    run: true
```

A workflow's own `README.md` documents its relevant configuration keys and command.

## Start STAMP workflows in dry-run mode

Steps 2, 4, and the deployment portion of Step 5 can invoke STAMP. Begin with:

```yaml
execute: false
```

Dry-run mode should generate configurations, commands, logs, and job summaries without starting expensive processing. Inspect the generated files before changing the value to `true`.

## Cohort naming

Workflow outputs use:

```text
{organ}__{dataset}
```

For example:

```text
CRC__TCGA
```

The double underscore is part of the directory convention and must not be normalized to a single underscore.

## Step 1: clinical distribution analysis

Use Step 1 to validate cohort inputs, summarize patient and slide counts, and compute configured pretraining bias metrics.

Run:

```bash
python workflows/step_1_clinical_distribution_analysis/run_clinical_distribution_analysis.py \
  --config configs/local_config.yaml
```

Review its summary and file-status outputs before proceeding.

## Step 2: STAMP feature extraction

Step 2 generates one preprocessing job per configured extractor and writes embeddings below `data.feature_root`.

First run with `execute: false`. Confirm the generated STAMP configuration, WSI paths, extractor names, cache locations, and worker settings.

## Step 3: feature embedding bias analysis

Step 3 loads slide embeddings, aggregates them to patient level, computes enabled projections, and evaluates target and bias-factor structure.

Projection settings are independent:

```yaml
projections:
  pca: true
  tsne: true
  umap: true
```

Documentation and interpretation should refer to enabled projections because any method can be disabled.

## Step 4: STAMP cross-validation training

Step 4 discovers compatible feature directories and generates model-by-label cross-validation jobs.

Use `labels: null` to rely on the configured label sources, or provide an explicit label list. Begin with one model, one binary target, and one split-compatible test configuration.

## Step 5: deployment and regular performance analysis

Step 5 first discovers Step 4 checkpoints and deploys them. It then evaluates patient-level predictions.

To deploy every discovered split:

```yaml
splits: null
```

To deploy selected splits:

```yaml
splits:
  - 0
  - 2
```

The current schema expects a list or `null`, not a single integer.

## Step 6: stratified subgroup performance analysis

Step 6 joins Step 5 predictions to clinical subgroup columns and recomputes target performance within each subgroup. It does not rerun STAMP for every subgroup.

Check subgroup counts and file-status sheets before interpreting performance gaps, especially for small or single-class subgroups.

## Step 7: comparative bias propagation analysis

Step 7 compares pretraining disparity with posttraining prediction disparity.

```text
delta = posttraining disparity - pretraining disparity
```

A positive delta indicates amplification; a negative delta indicates attenuation. For disparate impact, comparison uses distance from parity:

```text
DI distance = |1 - DI|
```

Step 7 may define its own optional bias-factor selection and should not silently inherit Step 6's selection.

## Minimal end-to-end smoke run

Use a deliberately small configuration:

```text
one cohort
one model
one binary target
one bias factor
one split
execute: false where supported
```

Run Steps 1 through 7 in order and inspect each status workbook before expanding the experiment.

## Failure handling

When a workflow records partial or failed jobs, preserve the generated configs and logs. They are part of the reproducibility trail and usually provide the fastest route to the broken path, missing label, incompatible checkpoint, or STAMP configuration.