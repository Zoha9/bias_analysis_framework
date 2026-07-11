# Step 2 Outputs

Outputs are written under:

```text
{project.output_root}/stamp_feature_extraction/{organ}__{dataset}/
```

## Folder structure

Expected structure:

```text
stamp_feature_extraction/
└── {organ}__{dataset}/
    ├── configs/
    ├── logs/
    └── tables/
        └── stamp_feature_extraction_jobs.xlsx
```

Feature embeddings are written to the configured data.feature_root

## Main outputs

The workflow writes tables to:

### `configs/`

Contains one generated STAMP preprocessing YAML per extractor.

### `logs/`

Contains the command or execution log for each extractor.


### `stamp_feature_extraction_jobs.xlsx`

Summarizes extractor name, generated config, output directory, status, execution mode, and errors.
