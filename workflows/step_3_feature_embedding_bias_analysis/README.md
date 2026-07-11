# Step 3: Feature Embedding Bias Analysis

This workflow examines whether patient feature embeddings contain structure associated with targets or population bias factors before classifier training.

## Purpose

Step 3 answers these questions:

1. Can slide embeddings be loaded and pooled to patient level?
2. Do PCA, t-SNE, or UMAP projections separate target classes?
3. Do embeddings separate demographic or clinical subgroups?
4. How strongly can target or subgroup labels be recovered from embeddings?
5. Are the observed patterns consistent across feature extractors?

This step studies representational bias before supervised model training.

## Required inputs

- Step 2 feature embeddings
- Clinical table
- Slide table
- Configured targets, bias factors, and models

Relevant YAML section:

```yaml
workflows:
  feature_embedding_analysis:
    run: true
    projection_preprocessing: "l2_normalize"

    projections:
      pca: true
      tsne: true
      umap: true
```

## How to run
    
    python workflows/step_3_feature_embedding_bias_analysis/run_feature_embedding_bias_analysis.py \
        --config configs/local_test_config.yaml

## Output location

    {project.output_root}/feature_embedding_analysis/{organ}__{dataset}/

See [`outputs.md`](outputs.md) for the generated files.
