# Project Overview

## Purpose

The Bias Analysis Framework is a configuration-driven toolkit for examining how population-related differences appear and change across computational pathology workflows.

It supports analysis at three connected stages:

1. **Before model training:** clinical distributions, representation, and pretraining bias metrics.
2. **Within learned representations:** structure in feature embeddings produced by foundation models.
3. **After model training:** overall performance, subgroup performance, and bias propagation in predictions.

The framework is intended to make these analyses repeatable across cohorts, targets, bias factors, feature extractors, and trained models without rewriting workflow scripts for every experiment.

## Design principles

- **One shared configuration:** project paths, cohort identifiers, targets, bias factors, models, STAMP settings, and workflow options are defined in a YAML file.
- **Explicit naming:** bias factors can use a readable report name and a separate clinical-table column.
- **Stepwise execution:** workflows are organized as seven numbered steps with clear dependencies.
- **Dry-run support:** STAMP workflows can generate and validate commands with `execute: false` before expensive processing begins.
- **Traceable outputs:** generated configurations, logs, tables, figures, predictions, and status workbooks are stored below a predictable workflow-specific output root.
- **Separation of concerns:** repository-level guides explain shared behavior; each workflow directory documents step-specific inputs and outputs.

## Workflow architecture

| Step | Workflow | Main responsibility |
|---|---|---|
| 1 | Clinical distribution analysis | Validate cohort inputs and quantify pretraining representation and disparity |
| 2 | STAMP feature extraction | Generate slide-level feature embeddings for configured extractors |
| 3 | Feature embedding bias analysis | Pool embeddings and evaluate target and bias-factor structure using enabled projections |
| 4 | STAMP model training | Generate and optionally execute cross-validation training jobs |
| 5 | Regular model prediction analysis | Deploy trained checkpoints and evaluate overall predictive performance |
| 6 | Stratified subgroup performance analysis | Compare target performance across configured subgroups |
| 7 | Comparative bias propagation analysis | Compare pretraining and posttraining disparity to identify amplification or attenuation |

Later workflows depend on artifacts from earlier steps. Step 6 reuses Step 5 predictions rather than rerunning deployment for each subgroup. Step 7 combines pretraining information with prediction-based disparity summaries.

## Shared configuration model

The example configuration is located at:

```text
configs/bias_analysis_config.example.yaml
```

Its principal sections are:

```yaml
project:
data:
cohort:
stamp:
models:
targets:
bias_factors:
workflows:
```

Bias factors support separate display and source-column names:

```yaml
bias_factors:
  - name: "Race"
    column: "Race Category"
```

`name` is used in reports, workbooks, and figures. `column` is used for clinical-table access. When `column` is omitted, workflows fall back to `name`.

## Output model

Each workflow writes below:

```text
{project.output_root}/{workflow_name}/{organ}__{dataset}/
```

The double underscore in `{organ}__{dataset}` is intentional and must be preserved.

Depending on the step, the cohort directory may contain:

```text
configs/
logs/
tables/
figures/
predictions/
crossval_results/
```

See [Output Structure](output_structure.md) for shared conventions and each workflow's `outputs.md` for exact files and workbook sheets.

## Scope and interpretation

The framework measures representation, association, predictive performance, and subgroup disparity. These results can identify patterns that require investigation, but they do not independently establish causality, fairness, clinical utility, or suitability for deployment.

Users remain responsible for cohort design, label quality, subgroup definitions, statistical interpretation, data governance, and domain-specific validation.