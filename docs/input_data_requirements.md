# Input Data Requirements

This guide describes the shared inputs expected by the framework. Individual workflows may use only a subset of these inputs.

## Clinical table

The clinical table is configured with:

```yaml
data:
  clinical_table: "/path/to/clinical_table.xlsx"
  patient_column: "PATIENT"
```

Supported tabular formats depend on the workflow implementation, but the example project uses Excel and CSV files.

The table should contain:

- one stable patient identifier column;
- every configured target column;
- every configured bias-factor source column;
- one row per patient, unless the workflow documentation explicitly permits repeated patients.

Target names are used as clinical-table columns. Bias factors may use a separate source column:

```yaml
bias_factors:
  - name: "Race"
    column: "Race Category"
```

Here, reports use `Race`, while the clinical table must contain `Race Category`.

### Recommended validation

Before running workflows, confirm that:

- patient identifiers are non-empty and consistently formatted;
- target and subgroup labels use consistent spelling and capitalization;
- duplicate patient rows are intentional and handled appropriately;
- missing values are understood and not encoded as accidental string categories;
- binary target labels match any configured `positive_label` and `negative_label` values.

## Slide table

The slide table is configured with:

```yaml
data:
  slide_table: "/path/to/slide_table.csv"
  patient_column: "PATIENT"
  slide_filename_column: "FILENAME"
```

It should map each slide to a patient identifier. Multiple slides per patient are supported where the relevant workflow performs patient-level aggregation.

Recommended columns are:

| Column | Purpose |
|---|---|
| Patient identifier | Joins slides, embeddings, predictions, and clinical data |
| Slide filename | Resolves WSI or embedding files |

Slide identifiers and filenames must be consistent with the paths generated or expected by STAMP.

## Whole-slide images

Whole-slide images are referenced through:

```yaml
data:
  wsi_root: "/path/to/wsi_folder"
```

WSI files are required for Step 2 feature extraction. They are not stored in this repository.

Before execution, verify that:

- the configured root exists;
- slide filenames in the slide table resolve to real files;
- the active STAMP installation supports the image format;
- magnification or microns-per-pixel settings are appropriate for the cohort.

## Feature embeddings

Feature embeddings are referenced through:

```yaml
data:
  feature_root: "/path/to/features"
```

They are produced by Step 2 or may be supplied from a compatible previous run. Step 3 and Step 4 rely on the expected extractor and cohort directory structure.

Embedding files must be readable by the configured loader. When HDF5 files contain multiple datasets, configure the expected dataset name where supported.

## STAMP templates

STAMP-related workflows require valid template configurations:

```yaml
stamp:
  stamp_config_template: "/path/to/stamp/config.yaml"
  crossval_config_template: "/path/to/stamp/crossval_config.yaml"
  deployment_config_template: "/path/to/stamp/deployment_config.yaml"
```

Each template must already contain the corresponding valid STAMP section. The framework copies and modifies these templates rather than replacing STAMP's schema.

## Trained checkpoints and predictions

Step 5 discovers checkpoints produced by Step 4. The expected default checkpoint filename is:

```text
model.ckpt
```

Step 5 performance analysis, Step 6, and Step 7 consume patient-level prediction files, typically:

```text
patient-preds.csv
patient-preds-0.csv
```

When more than one compatible file exists, consult the workflow documentation for discovery and precedence rules.

## Privacy and governance

Clinical tables, WSIs, embeddings, checkpoints, and predictions may contain sensitive or linkable information. Keep them outside version control and follow applicable institutional approvals, access controls, retention policies, and data-sharing agreements.