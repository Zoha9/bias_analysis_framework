"""Feature-embedding bias metric orchestration."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bias_analysis_framework.feature_embeddings.loading import LoadedEmbeddings
from bias_analysis_framework.metrics.feature_embedding_metrics import (
    class_count_table,
    clustering_scores,
    knn_cross_validated_accuracy,
    robustness_index,
)


UNKNOWN_LABELS = {
    "",
    "unknown",
    "na",
    "nan",
    "n/a",
    "none",
    "null",
    "<na>",
}


@dataclass(frozen=True)
class PatientEmbeddings:
    """Patient-level embeddings and aligned clinical metadata."""

    embeddings: np.ndarray
    metadata: pd.DataFrame


@dataclass(frozen=True)
class FeatureEmbeddingAnalysisResult:
    """Outputs from one model's feature-embedding analysis."""

    metrics: pd.DataFrame
    counts: pd.DataFrame
    patient_embeddings: PatientEmbeddings


def analyze_model_feature_embeddings(
    *,
    loaded_embeddings: LoadedEmbeddings,
    config: dict[str, Any],
    model_name: str,
) -> FeatureEmbeddingAnalysisResult:
    """Compute feature-embedding metrics for one model.

    Metrics are calculated for every configured:

    - target biomarker;
    - bias factor;
    - target biomarker × bias factor combination.

    Slide embeddings are first pooled to one mean embedding per patient to
    preserve the patient-level behavior of the original workflow.
    """
    patient_column = (
        config.get("data", {})
        .get("patient_column", "PATIENT")
    )

    workflow_config = (
        config.get("workflows", {})
        .get("feature_embedding_analysis", {})
    )

    patient_embeddings = pool_slide_embeddings_by_patient(
        loaded_embeddings=loaded_embeddings,
        patient_column=patient_column,
        method=workflow_config.get(
            "patient_aggregation",
            "mean",
        ),
    )

    random_seed = int(
        config.get("project", {})
        .get("random_seed", 42)
    )

    robustness_k = int(
        workflow_config.get("robustness_k", 50)
    )
    knn_k = int(
        workflow_config.get("knn_k", 5)
    )
    knn_folds = int(
        workflow_config.get("knn_folds", 5)
    )
    minimum_group_size = int(
        workflow_config.get("minimum_group_size_for_metrics", 2)
    )

    target_specs = get_target_specs(config)
    bias_specs = get_bias_factor_specs(config)

    metrics_rows: list[dict[str, Any]] = []
    count_tables: list[pd.DataFrame] = []

    metadata = patient_embeddings.metadata
    embeddings = patient_embeddings.embeddings

    for target_name, allowed_target_labels in target_specs.items():
        if target_name not in metadata.columns:
            continue

        target_mask = valid_label_mask(
            metadata[target_name],
            allowed_values=allowed_target_labels,
        )

        target_metadata = metadata.loc[target_mask].reset_index(drop=True)
        target_embeddings = embeddings[target_mask.to_numpy()]

        if target_metadata.empty:
            continue

        target_labels = clean_labels(
            target_metadata[target_name]
        )

        target_counts = class_count_table(
            target_labels,
            label_type="Target",
            label_name=target_name,
        )
        target_counts.insert(0, "Model", model_name)
        count_tables.append(target_counts)

        target_knn = knn_cross_validated_accuracy(
            target_embeddings,
            target_labels,
            k=knn_k,
            n_splits=knn_folds,
            random_seed=random_seed,
            min_group_size=minimum_group_size,
        )

        target_clustering = clustering_scores(
            target_embeddings,
            target_labels,
            min_group_size=minimum_group_size,
        )

        for bias_name, bias_spec in bias_specs.items():
            bias_column = bias_spec["column"]

            if bias_column not in target_metadata.columns:
                continue

            combination_mask = valid_label_mask(
                target_metadata[bias_column],
                allowed_values=bias_spec["subgroups"],
            )

            combination_metadata = (
                target_metadata.loc[combination_mask]
                .reset_index(drop=True)
            )
            combination_embeddings = target_embeddings[
                combination_mask.to_numpy()
            ]

            if combination_metadata.empty:
                continue

            combination_target_labels = clean_labels(
                combination_metadata[target_name]
            )
            combination_bias_labels = clean_labels(
                combination_metadata[bias_column]
            )

            minimum_count = int(
                bias_spec.get(
                    "minimum_count",
                    minimum_group_size,
                )
            )

            combination_metadata, combination_embeddings = filter_small_bias_groups(
                metadata=combination_metadata,
                embeddings=combination_embeddings,
                bias_column=bias_column,
                minimum_count=minimum_count,
                other_label=bias_spec.get("other_label"),
            )

            if combination_metadata.empty:
                continue

            combination_target_labels = clean_labels(
                combination_metadata[target_name]
            )
            combination_bias_labels = clean_labels(
                combination_metadata[bias_column]
            )

            bias_counts = class_count_table(
                combination_bias_labels,
                label_type="BiasFactor",
                label_name=bias_name,
            )
            bias_counts.insert(0, "Target", target_name)
            bias_counts.insert(0, "Model", model_name)
            count_tables.append(bias_counts)

            bias_knn = knn_cross_validated_accuracy(
                combination_embeddings,
                combination_bias_labels,
                k=knn_k,
                n_splits=knn_folds,
                random_seed=random_seed,
                min_group_size=minimum_group_size,
            )

            bias_clustering = clustering_scores(
                combination_embeddings,
                combination_bias_labels,
                min_group_size=minimum_group_size,
            )

            target_combination_knn = knn_cross_validated_accuracy(
                combination_embeddings,
                combination_target_labels,
                k=knn_k,
                n_splits=knn_folds,
                random_seed=random_seed,
                min_group_size=minimum_group_size,
            )

            target_combination_clustering = clustering_scores(
                combination_embeddings,
                combination_target_labels,
                min_group_size=minimum_group_size,
            )

            robustness = robustness_index(
                combination_embeddings,
                combination_target_labels,
                combination_bias_labels,
                k=robustness_k,
                min_group_size=minimum_group_size,
            )

            metrics_rows.append(
                {
                    "Model": model_name,
                    "Biomarker": target_name,
                    "BiasFactor": bias_name,
                    "BiasFactorColumn": bias_column,
                    "N": len(combination_metadata),
                    "N_BiomarkerClasses": combination_target_labels.nunique(),
                    "N_BiasGroups": combination_bias_labels.nunique(),
                    "BiomarkerClasses": join_label_counts(
                        combination_target_labels
                    ),
                    "BiasGroups": join_label_counts(
                        combination_bias_labels
                    ),
                    "kNN_acc_biomarker": target_combination_knn,
                    "kNN_acc_biasfactor": bias_knn,
                    "Robustness_Index": robustness.robustness_index,
                    "RI_num_same_biomarker": robustness.same_target_neighbors,
                    "RI_num_same_biasfactor": robustness.same_bias_neighbors,
                    "RI_effective_k": robustness.effective_k,
                    "Bio_Silhouette(cosine)": (
                        target_combination_clustering.silhouette_cosine
                    ),
                    "Bio_Calinski_Harabasz": (
                        target_combination_clustering.calinski_harabasz
                    ),
                    "Bio_Davies_Bouldin": (
                        target_combination_clustering.davies_bouldin
                    ),
                    "Bias_Silhouette(cosine)": (
                        bias_clustering.silhouette_cosine
                    ),
                    "Bias_Calinski_Harabasz": (
                        bias_clustering.calinski_harabasz
                    ),
                    "Bias_Davies_Bouldin": (
                        bias_clustering.davies_bouldin
                    ),
                    "Delta_kNN_Biomarker_minus_Bias": (
                        target_combination_knn - bias_knn
                        if pd.notna(target_combination_knn)
                        and pd.notna(bias_knn)
                        else np.nan
                    ),
                    "Delta_Silhouette_Biomarker_minus_Bias": (
                        target_combination_clustering.silhouette_cosine
                        - bias_clustering.silhouette_cosine
                        if pd.notna(
                            target_combination_clustering.silhouette_cosine
                        )
                        and pd.notna(
                            bias_clustering.silhouette_cosine
                        )
                        else np.nan
                    ),
                    "FullTarget_kNN_acc_biomarker": target_knn,
                    "FullTarget_Bio_Silhouette(cosine)": (
                        target_clustering.silhouette_cosine
                    ),
                    "FullTarget_Bio_Calinski_Harabasz": (
                        target_clustering.calinski_harabasz
                    ),
                    "FullTarget_Bio_Davies_Bouldin": (
                        target_clustering.davies_bouldin
                    ),
                }
            )

    metrics_df = pd.DataFrame(metrics_rows)

    if count_tables:
        counts_df = pd.concat(
            count_tables,
            ignore_index=True,
        )
    else:
        counts_df = pd.DataFrame(
            columns=[
                "Model",
                "Target",
                "LabelType",
                "LabelName",
                "Class",
                "Count",
            ]
        )

    return FeatureEmbeddingAnalysisResult(
        metrics=metrics_df,
        counts=counts_df,
        patient_embeddings=patient_embeddings,
    )


def pool_slide_embeddings_by_patient(
    *,
    loaded_embeddings: LoadedEmbeddings,
    patient_column: str,
    method: str = "mean",
) -> PatientEmbeddings:
    """Pool slide-level embeddings to one embedding per patient."""
    metadata = loaded_embeddings.metadata.reset_index(drop=True)
    embeddings = np.asarray(loaded_embeddings.embeddings)

    if patient_column not in metadata.columns:
        raise ValueError(
            f"Embedding metadata is missing patient column: {patient_column}"
        )

    if len(metadata) != len(embeddings):
        raise ValueError(
            "Embedding metadata and embedding matrix have different lengths."
        )

    pooled_embeddings: list[np.ndarray] = []
    pooled_metadata: list[dict[str, Any]] = []

    grouped_indices = metadata.groupby(
        patient_column,
        sort=False,
    ).indices

    for patient_id, indices in grouped_indices.items():
        patient_indices = np.asarray(indices, dtype=int)
        patient_slide_embeddings = embeddings[patient_indices]

        if method == "mean":
            pooled = np.mean(
                patient_slide_embeddings,
                axis=0,
            )
        elif method == "median":
            pooled = np.median(
                patient_slide_embeddings,
                axis=0,
            )
        elif method == "max":
            pooled = np.max(
                patient_slide_embeddings,
                axis=0,
            )
        else:
            raise ValueError(
                "Unsupported patient aggregation method. "
                "Use mean, median, or max."
            )

        first_metadata = metadata.iloc[
            patient_indices[0]
        ].to_dict()

        first_metadata["N_SLIDES_FOR_PATIENT"] = len(
            patient_indices
        )
        first_metadata["PATIENT_AGGREGATION"] = method

        pooled_embeddings.append(
            pooled.astype(np.float32)
        )
        pooled_metadata.append(first_metadata)

    return PatientEmbeddings(
        embeddings=np.vstack(pooled_embeddings),
        metadata=pd.DataFrame(
            pooled_metadata
        ).reset_index(drop=True),
    )


def get_target_specs(
    config: dict[str, Any],
) -> OrderedDict[str, list[str] | None]:
    """Return configured targets and allowed class labels."""
    output: OrderedDict[str, list[str] | None] = OrderedDict()

    for target in config.get("targets", []):
        name = target.get("name")

        if not name:
            continue

        if target.get("labels"):
            labels = [
                str(value)
                for value in target["labels"]
            ]
        elif (
            target.get("negative_label") is not None
            and target.get("positive_label") is not None
        ):
            labels = [
                str(target["negative_label"]),
                str(target["positive_label"]),
            ]
        else:
            labels = None

        output[str(name)] = labels

    return output


def get_bias_factor_specs(
    config: dict[str, Any],
) -> OrderedDict[str, dict[str, Any]]:
    """Return configured bias-factor analysis specifications."""
    output: OrderedDict[str, dict[str, Any]] = OrderedDict()

    workflow_config = (
        config.get("workflows", {})
        .get("feature_embedding_analysis", {})
    )

    minimum_counts = workflow_config.get(
        "factor_minimum_counts",
        {},
    )

    default_minimum = int(
        workflow_config.get(
            "minimum_group_size_for_metrics",
            2,
        )
    )

    for factor in config.get("bias_factors", []):
        name = factor.get("name")

        if not name:
            continue

        column = factor.get(
            "column",
            name,
        )

        minimum_count = factor.get(
            "minimum_count",
            minimum_counts.get(
                name,
                default_minimum,
            ),
        )

        output[str(name)] = {
            "column": str(column),
            "subgroups": (
                [str(value) for value in factor["subgroups"]]
                if factor.get("subgroups")
                else None
            ),
            "minimum_count": int(minimum_count),
            "other_label": factor.get("other_label"),
        }

    return output


def valid_label_mask(
    labels: pd.Series,
    *,
    allowed_values: list[str] | None,
) -> pd.Series:
    """Return mask for non-missing, non-unknown, configured labels."""
    cleaned = clean_labels(labels)
    mask = ~cleaned.str.casefold().isin(
        UNKNOWN_LABELS
    )

    if allowed_values:
        allowed_casefold = {
            str(value).strip().casefold()
            for value in allowed_values
        }
        mask &= cleaned.str.casefold().isin(
            allowed_casefold
        )

    return mask


def clean_labels(
    labels: pd.Series,
) -> pd.Series:
    """Normalize labels for metric processing."""
    return (
        labels
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )


def filter_small_bias_groups(
    *,
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    bias_column: str,
    minimum_count: int,
    other_label: str | None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Drop or combine bias groups below a minimum sample count."""
    if minimum_count <= 1:
        return metadata.reset_index(drop=True), embeddings

    output_metadata = metadata.copy()
    labels = clean_labels(
        output_metadata[bias_column]
    )
    counts = labels.value_counts()
    small_groups = set(
        counts[counts < minimum_count].index
    )

    if not small_groups:
        return output_metadata.reset_index(drop=True), embeddings

    if other_label:
        output_metadata[bias_column] = labels.where(
            ~labels.isin(small_groups),
            other_label,
        )
        return output_metadata.reset_index(drop=True), embeddings

    keep_mask = ~labels.isin(
        small_groups
    )

    return (
        output_metadata.loc[
            keep_mask
        ].reset_index(drop=True),
        embeddings[keep_mask.to_numpy()],
    )


def join_label_counts(
    labels: pd.Series,
) -> str:
    """Format label counts for the metrics workbook."""
    counts = labels.value_counts()

    return "; ".join(
        f"{label}={int(count)}"
        for label, count in counts.items()
    )


def write_feature_embedding_metric_workbook(
    *,
    result: FeatureEmbeddingAnalysisResult,
    output_path: str | Path,
) -> Path:
    """Write one model's feature-embedding metric workbook."""
    output_path = Path(
        output_path
    ).expanduser().resolve()
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    patient_metadata = (
        result.patient_embeddings.metadata.copy()
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:
        result.metrics.to_excel(
            writer,
            sheet_name="metrics",
            index=False,
        )
        result.counts.to_excel(
            writer,
            sheet_name="class_counts",
            index=False,
        )
        patient_metadata.to_excel(
            writer,
            sheet_name="patient_embedding_metadata",
            index=False,
        )

    return output_path