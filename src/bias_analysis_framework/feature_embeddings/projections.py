"""Dimensionality-reduction projections for feature embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, normalize


def compute_embedding_projections(
    embeddings: np.ndarray,
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Compute configured PCA, t-SNE, and UMAP projections."""
    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix, found shape {embeddings.shape}."
        )

    if embeddings.shape[0] < 3:
        raise ValueError(
            "At least three embeddings are required for projections."
        )

    workflow_config = (
        config.get("workflows", {})
        .get("feature_embedding_analysis", {})
    )
    projection_config = workflow_config.get("projections", {})

    random_seed = int(
        config.get("project", {}).get("random_seed", 42)
    )
    
    preprocessing_method = str(
        workflow_config.get(
            "projection_preprocessing",
            "l2_normalize",
        )
    ).strip().casefold()

    if preprocessing_method == "l2_normalize":
        transformed = normalize(
            embeddings,
            norm="l2",
        )
    elif preprocessing_method == "standardize":
        transformed = StandardScaler().fit_transform(
            embeddings
        )
    elif preprocessing_method in {"none", "raw"}:
        transformed = embeddings.copy()
    else:
        raise ValueError(
            "Unsupported projection_preprocessing value. "
            "Use l2_normalize, standardize, or none."
        )

    outputs: dict[str, pd.DataFrame] = {}

    if projection_config.get("pca", True):
        outputs["pca"] = compute_pca_projection(
            transformed,
            random_seed=random_seed,
        )

    if projection_config.get("tsne", True):
        outputs["tsne"] = compute_tsne_projection(
            transformed,
            random_seed=random_seed,
            config=workflow_config,
        )

    if projection_config.get("umap", True):
        outputs["umap"] = compute_umap_projection(
            transformed,
            random_seed=random_seed,
            config=workflow_config,
        )

    return outputs


def compute_pca_projection(
    embeddings: np.ndarray,
    *,
    random_seed: int,
) -> pd.DataFrame:
    """Compute a two-dimensional PCA projection."""
    reducer = PCA(
        n_components=2,
        random_state=random_seed,
    )

    projected = reducer.fit_transform(embeddings)

    return pd.DataFrame(
        {
            "PCA_1": projected[:, 0],
            "PCA_2": projected[:, 1],
            "PCA_explained_variance_1": reducer.explained_variance_ratio_[0],
            "PCA_explained_variance_2": reducer.explained_variance_ratio_[1],
        }
    )


def compute_tsne_projection(
    embeddings: np.ndarray,
    *,
    random_seed: int,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Compute a two-dimensional t-SNE projection."""
    n_samples = embeddings.shape[0]

    configured_perplexity = float(
        config.get("tsne_perplexity", 30)
    )

    max_valid_perplexity = max(1.0, float(n_samples - 1))
    perplexity = max(1.0, min(configured_perplexity, max_valid_perplexity),)
    max_iter = int(config.get("tsne_max_iter", 1000))

    reducer = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_seed,
        max_iter=max_iter,
    )

    projected = reducer.fit_transform(embeddings)

    return pd.DataFrame(
        {
            "TSNE_1": projected[:, 0],
            "TSNE_2": projected[:, 1],
            "TSNE_perplexity": perplexity,
            "TSNE_max_iter": max_iter,
        }
    )


def compute_umap_projection(
    embeddings: np.ndarray,
    *,
    random_seed: int,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Compute a two-dimensional UMAP projection."""
    try:
        import umap
    except ImportError as error:
        raise ImportError(
            "UMAP projection requires the 'umap-learn' package."
        ) from error

    n_samples = embeddings.shape[0]

    configured_neighbors = int(
        config.get("umap_n_neighbors", 30)
    )
    n_neighbors = min(
        max(2, configured_neighbors),
        n_samples - 1,
    )

    min_dist = float(config.get("umap_min_dist", 0.1))

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_seed,
    )

    projected = reducer.fit_transform(embeddings)

    return pd.DataFrame(
        {
            "UMAP_1": projected[:, 0],
            "UMAP_2": projected[:, 1],
            "UMAP_n_neighbors": n_neighbors,
            "UMAP_min_dist": min_dist,
        }
    )