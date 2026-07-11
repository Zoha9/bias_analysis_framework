"""Metrics for feature-embedding bias analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler, normalize


@dataclass(frozen=True)
class ClusteringScores:
    """Clustering-separation scores for one label grouping."""

    silhouette_cosine: float
    calinski_harabasz: float
    davies_bouldin: float


@dataclass(frozen=True)
class RobustnessIndexResult:
    """Robustness Index and its neighbor-match counts."""

    robustness_index: float
    same_target_neighbors: int
    same_bias_neighbors: int
    effective_k: int


def encode_labels(
    labels: pd.Series | np.ndarray | list[Any],
) -> tuple[np.ndarray, LabelEncoder]:
    """Encode categorical labels as integer values."""
    series = pd.Series(labels).astype(str)
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(series)
    return encoded, encoder


def valid_grouping(
    labels: pd.Series | np.ndarray | list[Any],
    *,
    min_groups: int = 2,
    min_group_size: int = 2,
) -> bool:
    """Check whether labels are suitable for embedding metrics."""
    series = pd.Series(labels).dropna()

    if series.empty:
        return False

    counts = series.value_counts()

    if len(counts) < min_groups:
        return False

    if counts.min() < min_group_size:
        return False

    return True


def robustness_index(
    embeddings: np.ndarray,
    target_labels: pd.Series | np.ndarray | list[Any],
    bias_labels: pd.Series | np.ndarray | list[Any],
    *,
    k: int = 50,
    min_group_size: int = 2,
) -> RobustnessIndexResult:
    """Compute the feature-embedding Robustness Index.

    The calculation preserves the original project definition:

    RI = number of same-target nearest-neighbor matches
         divided by
         number of same-bias-factor nearest-neighbor matches

    Cosine nearest neighbors are calculated after L2 normalization.
    """
    embeddings = validate_embedding_matrix(embeddings)

    target_series = pd.Series(target_labels).reset_index(drop=True)
    bias_series = pd.Series(bias_labels).reset_index(drop=True)

    if len(embeddings) != len(target_series) or len(embeddings) != len(bias_series):
        raise ValueError(
            "Embeddings, target labels, and bias labels must have equal lengths."
        )

    if not valid_grouping(
        target_series,
        min_group_size=min_group_size,
    ):
        return RobustnessIndexResult(
            robustness_index=np.nan,
            same_target_neighbors=0,
            same_bias_neighbors=0,
            effective_k=0,
        )

    if not valid_grouping(
        bias_series,
        min_group_size=min_group_size,
    ):
        return RobustnessIndexResult(
            robustness_index=np.nan,
            same_target_neighbors=0,
            same_bias_neighbors=0,
            effective_k=0,
        )

    n_samples = len(embeddings)

    if n_samples < 2:
        return RobustnessIndexResult(
            robustness_index=np.nan,
            same_target_neighbors=0,
            same_bias_neighbors=0,
            effective_k=0,
        )

    effective_k = min(int(k), n_samples - 1)

    if effective_k < 1:
        return RobustnessIndexResult(
            robustness_index=np.nan,
            same_target_neighbors=0,
            same_bias_neighbors=0,
            effective_k=0,
        )

    normalized_embeddings = normalize(embeddings, norm="l2")

    neighbors = NearestNeighbors(
        n_neighbors=effective_k + 1,
        metric="cosine",
    )
    neighbors.fit(normalized_embeddings)

    neighbor_indices = neighbors.kneighbors(
        normalized_embeddings,
        return_distance=False,
    )[:, 1:]

    target_values = target_series.astype(str).to_numpy()
    bias_values = bias_series.astype(str).to_numpy()

    same_target = (
        target_values[neighbor_indices]
        == target_values[:, None]
    )
    same_bias = (
        bias_values[neighbor_indices]
        == bias_values[:, None]
    )

    same_target_count = int(same_target.sum())
    same_bias_count = int(same_bias.sum())

    if same_bias_count == 0:
        score = np.nan
    else:
        score = float(same_target_count / same_bias_count)

    return RobustnessIndexResult(
        robustness_index=score,
        same_target_neighbors=same_target_count,
        same_bias_neighbors=same_bias_count,
        effective_k=effective_k,
    )


def knn_cross_validated_accuracy(
    embeddings: np.ndarray,
    labels: pd.Series | np.ndarray | list[Any],
    *,
    k: int = 5,
    n_splits: int = 5,
    random_seed: int = 42,
    min_group_size: int = 2,
) -> float:
    """Compute stratified cross-validated kNN accuracy.

    This preserves the old workflow's cosine-distance kNN calculation while
    safely reducing the number of folds when a class contains fewer than five
    samples.
    """
    embeddings = validate_embedding_matrix(embeddings)
    label_series = pd.Series(labels).reset_index(drop=True)

    if len(embeddings) != len(label_series):
        raise ValueError(
            "Embeddings and labels must have equal lengths."
        )

    if not valid_grouping(
        label_series,
        min_group_size=min_group_size,
    ):
        return np.nan

    encoded_labels, _ = encode_labels(label_series)
    class_counts = pd.Series(encoded_labels).value_counts()

    effective_splits = min(
        int(n_splits),
        int(class_counts.min()),
    )

    if effective_splits < 2:
        return np.nan

    normalized_embeddings = normalize(
        embeddings,
        norm="l2",
    )

    splitter = StratifiedKFold(
        n_splits=effective_splits,
        shuffle=True,
        random_state=random_seed,
    )

    fold_accuracies: list[float] = []

    for train_indices, test_indices in splitter.split(
        normalized_embeddings,
        encoded_labels,
    ):
        effective_k = min(
            int(k),
            len(train_indices),
        )

        if effective_k < 1:
            continue

        classifier = KNeighborsClassifier(
            n_neighbors=effective_k,
            metric="cosine",
        )

        classifier.fit(
            normalized_embeddings[train_indices],
            encoded_labels[train_indices],
        )

        predictions = classifier.predict(
            normalized_embeddings[test_indices]
        )

        fold_accuracies.append(
            accuracy_score(
                encoded_labels[test_indices],
                predictions,
            )
        )

    if not fold_accuracies:
        return np.nan

    return float(np.mean(fold_accuracies))


def clustering_scores(
    embeddings: np.ndarray,
    labels: pd.Series | np.ndarray | list[Any],
    *,
    min_group_size: int = 2,
) -> ClusteringScores:
    """Compute silhouette, Calinski-Harabasz, and Davies-Bouldin scores.

    This preserves the original processing:

    - cosine silhouette on L2-normalized embeddings;
    - Calinski-Harabasz on standardized Euclidean embeddings;
    - Davies-Bouldin on standardized Euclidean embeddings.
    """
    embeddings = validate_embedding_matrix(embeddings)
    label_series = pd.Series(labels).reset_index(drop=True)

    if len(embeddings) != len(label_series):
        raise ValueError(
            "Embeddings and labels must have equal lengths."
        )

    if not valid_grouping(
        label_series,
        min_group_size=min_group_size,
    ):
        return ClusteringScores(
            silhouette_cosine=np.nan,
            calinski_harabasz=np.nan,
            davies_bouldin=np.nan,
        )

    encoded_labels, _ = encode_labels(label_series)

    normalized_embeddings = normalize(
        embeddings,
        norm="l2",
    )

    try:
        silhouette = float(
            silhouette_score(
                normalized_embeddings,
                encoded_labels,
                metric="cosine",
            )
        )
    except Exception:
        silhouette = np.nan

    try:
        standardized_embeddings = StandardScaler(
            with_mean=True,
            with_std=True,
        ).fit_transform(
            np.asarray(embeddings, dtype=np.float64)
        )
    except Exception:
        standardized_embeddings = np.asarray(
            embeddings,
            dtype=np.float64,
        )

    try:
        calinski_harabasz = float(
            calinski_harabasz_score(
                standardized_embeddings,
                encoded_labels,
            )
        )
    except Exception:
        calinski_harabasz = np.nan

    try:
        davies_bouldin = float(
            davies_bouldin_score(
                standardized_embeddings,
                encoded_labels,
            )
        )
    except Exception:
        davies_bouldin = np.nan

    return ClusteringScores(
        silhouette_cosine=silhouette,
        calinski_harabasz=calinski_harabasz,
        davies_bouldin=davies_bouldin,
    )


def class_count_table(
    labels: pd.Series | np.ndarray | list[Any],
    *,
    label_type: str,
    label_name: str,
) -> pd.DataFrame:
    """Create a class/subgroup count table."""
    counts = (
        pd.Series(labels)
        .astype(str)
        .value_counts(dropna=False)
        .rename_axis("Class")
        .reset_index(name="Count")
    )

    counts.insert(0, "LabelName", label_name)
    counts.insert(0, "LabelType", label_type)

    return counts


def validate_embedding_matrix(
    embeddings: np.ndarray,
) -> np.ndarray:
    """Validate and return a numeric two-dimensional embedding matrix."""
    array = np.asarray(
        embeddings,
        dtype=np.float64,
    )

    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D embedding matrix, found shape {array.shape}."
        )

    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("Embedding matrix is empty.")

    if not np.isfinite(array).all():
        raise ValueError(
            "Embedding matrix contains NaN or infinite values."
        )

    return array