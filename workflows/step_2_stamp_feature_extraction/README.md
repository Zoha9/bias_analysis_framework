# Step 2: STAMP Feature Extraction

This workflow converts whole-slide images into feature embeddings using the configured STAMP extractors.

## Purpose

Step 2 answers these questions:

1. Are the WSI, STAMP template, cache, and output paths valid?
2. Which feature extractors should be run?
3. Can valid STAMP preprocessing configs be generated?
4. Were feature files produced for each configured model?

Feature extraction is separated from downstream analysis so that embeddings can be reused across later steps.

## Required inputs

- Whole-slide image directory
- STAMP configuration template
- Configured feature extractors
- Cache and feature output paths

Relevant YAML sections:
```yaml
data:
  wsi_root: "/path/to/wsi"
  feature_root: "/path/to/features"

models:
  - name: "UNI"
    extractor: "uni"

workflows:
  stamp_feature_extraction:
    run: true
    execute: false
```

Run with execute: false first. This generates configs and commands without processing slides.

## How to run
    
    python workflows/step_2_stamp_feature_extraction/run_stamp_feature_extraction.py \
        --config configs/local_test_config.yaml

## Output location

    {project.output_root}/stamp_feature_extraction/{organ}__{dataset}/

See [`outputs.md`](outputs.md) for the generated files.
