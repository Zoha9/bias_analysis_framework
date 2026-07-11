# Configuration Guide

The framework is controlled through a shared YAML configuration. Start from:

```text
configs/bias_analysis_config.example.yaml
```

Create a local copy:

```bash
cp configs/bias_analysis_config.example.yaml configs/local_config.yaml
```

Keep machine-specific paths, credentials, and protected data locations out of version control.

## Top-level sections

```yaml
project:
stamp:
data:
cohort:
targets:
bias_factors:
models:
workflows:
```

## `project`

Defines shared project behavior.

```yaml
project:
  name: "CRC bias analysis"
  output_root: "/path/to/results"
  random_seed: 42
```

- `name`: readable project label.
- `output_root`: root directory for workflow outputs.
- `random_seed`: seed used by supported stochastic operations.

## `data`

Defines cohort inputs and shared identifier columns.

```yaml
data:
  clinical_table: "/path/to/clinical_table.xlsx"
  slide_table: "/path/to/slide_table.csv"
  wsi_root: "/path/to/wsi_folder"
  feature_root: "/path/to/features"
  patient_column: "PATIENT"
  slide_filename_column: "FILENAME"
```

See [Input Data Requirements](input_data_requirements.md) for expected contents.

## `cohort`

Defines the cohort identifier used in output paths.

```yaml
cohort:
  organ: "CRC"
  dataset: "TCGA"
```

The output cohort folder is:

```text
CRC__TCGA
```

The double underscore is intentional.

## `stamp`

Defines the STAMP command, templates, working directory, cache locations, and optional environment values.

```yaml
stamp:
  required: true
  command: "stamp"
  working_directory: null
  stamp_config_template: "/path/to/stamp/config.yaml"
  crossval_config_template: "/path/to/stamp/crossval_config.yaml"
  deployment_config_template: "/path/to/stamp/deployment_config.yaml"
```

The framework modifies valid STAMP templates. It does not replace STAMP's own configuration schema.

Use `execute: false` in STAMP-facing workflow sections for initial validation.

## `models`

Defines readable model names and STAMP extractor identifiers.

```yaml
models:
  - name: "UNI"
    extractor: "uni"
  - name: "CONCH"
    extractor: "conch"
```

- `name`: label used in reports and output directories.
- `extractor`: identifier used for feature discovery or STAMP configuration.

## `targets`

Defines prediction targets.

```yaml
targets:
  - name: "MSI_Status"
    positive_label: "MSI-H"
    negative_label: "MSS"
```

For binary targets, `positive_label` controls sensitivity, specificity, precision, NPV, AUROC, AUPRC, and related interpretation.

A multiclass target may define its allowed labels:

```yaml
targets:
  - name: "BRCA_Subtype_PAM50"
    labels:
      - "LumA"
      - "LumB"
      - "Basal"
      - "Her2"
      - "Normal"
```

Do not add a binary `positive_label` to multiclass targets unless an explicit one-versus-rest analysis is intended and supported.

## `bias_factors`

Defines population or cohort factors used in reports and subgroup analyses.

```yaml
bias_factors:
  - name: "Race"
    column: "Race Category"
    subgroups:
      - "WHITE"
      - "BLACK OR AFRICAN AMERICAN"
      - "ASIAN"
```

- `name`: display name used in reports, workbooks, and figures.
- `column`: source column in the clinical table.
- `subgroups`: optional explicit subgroup selection.
- `minimum_count`: optional threshold for grouping or filtering small categories.
- `other_label`: optional label for combined low-frequency categories.

Clinical-column resolution follows:

```python
factor.get("column") or factor.get("name")
```

A binary bias factor may define a positive class when binary metrics require it:

```yaml
bias_factors:
  - name: "Gender"
    column: "Gender"
    positive_label: "Male"
```

## `workflows`

Each workflow has its own mapping below `workflows`.

```yaml
workflows:
  clinical_distribution_analysis:
    run: true
  stamp_feature_extraction:
    run: true
    execute: false
```

Shared patterns include:

- `run`: whether the workflow is selected.
- `execute`: whether an external STAMP command should actually run.
- `continue_on_error`: whether later jobs continue after a failed job.
- `skip_existing`: whether compatible existing outputs are reused.

Exact settings are documented in each workflow's `README.md`.

## Split selection

For Step 5 deployment:

```yaml
splits: null
```

means all discovered splits.

```yaml
splits:
  - 0
  - 2
```

means selected splits only. A single integer such as `splits: 5` is not valid for the current schema.

## Optional selections

A `null` selection usually means automatic discovery or use of the workflow's default configured sources. An explicit list narrows the analysis.

Example:

```yaml
labels: null
```

or:

```yaml
labels:
  - name: "MSI_Status"
    type: "target"
```

Step 7 may define its own optional `bias_factors` selection. It should not silently depend on Step 6's selection.

## Validation

Confirm that the YAML parses before running workflows:

```bash
python - <<'PY'
from pathlib import Path
import yaml

path = Path("configs/local_config.yaml")
with path.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

print("Configuration loaded successfully")
print("Workflow keys:", list(config["workflows"]))
PY
```

Also verify that target names are unique, bias-factor names and columns are unambiguous, required paths exist, and selected labels occur in the clinical table.