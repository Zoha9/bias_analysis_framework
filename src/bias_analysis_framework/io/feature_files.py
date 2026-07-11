"""Discovery and low-level loading of STAMP feature files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


DEFAULT_FEATURE_DATASET_NAMES = (
    "feats",
    "features",
    "embeddings",
    "embedding",
)


def discover_feature_files(
    feature_root: str | Path,
    *,
    suffixes: Iterable[str] = (".h5", ".hdf5"),
) -> list[Path]:
    """Recursively discover feature files below a root directory."""
    root = Path(feature_root).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Feature root does not exist: {root}")

    if not root.is_dir():
        raise ValueError(f"Feature root is not a directory: {root}")

    normalized_suffixes = {suffix.casefold() for suffix in suffixes}

    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.casefold() in normalized_suffixes
    ]

    return sorted(files)


def inspect_h5_datasets(feature_path: str | Path) -> list[str]:
    """Return all dataset paths found in an HDF5 feature file."""
    path = Path(feature_path).expanduser().resolve()
    dataset_names: list[str] = []

    with h5py.File(path, "r") as handle:

        def visitor(name: str, item: h5py.Dataset | h5py.Group) -> None:
            if isinstance(item, h5py.Dataset):
                dataset_names.append(name)

        handle.visititems(visitor)

    return dataset_names


def find_embedding_dataset(
    handle: h5py.File,
    preferred_names: Iterable[str] = DEFAULT_FEATURE_DATASET_NAMES,
) -> str:
    """Find the most likely embedding dataset in an open HDF5 file."""
    dataset_names: list[str] = []

    def visitor(name: str, item: h5py.Dataset | h5py.Group) -> None:
        if isinstance(item, h5py.Dataset):
            dataset_names.append(name)

    handle.visititems(visitor)

    if not dataset_names:
        raise ValueError("No datasets were found in the HDF5 feature file.")

    preferred_casefold = [name.casefold() for name in preferred_names]

    for preferred in preferred_casefold:
        for dataset_name in dataset_names:
            leaf_name = dataset_name.split("/")[-1].casefold()
            if leaf_name == preferred:
                return dataset_name

    numeric_candidates = []

    for dataset_name in dataset_names:
        dataset = handle[dataset_name]

        if not np.issubdtype(dataset.dtype, np.number):
            continue

        if dataset.ndim in {1, 2}:
            numeric_candidates.append(dataset_name)

    if len(numeric_candidates) == 1:
        return numeric_candidates[0]

    two_dimensional = [
        name
        for name in numeric_candidates
        if handle[name].ndim == 2
    ]

    if len(two_dimensional) == 1:
        return two_dimensional[0]

    raise ValueError(
        "Could not determine the embedding dataset automatically. "
        f"Available datasets: {dataset_names}"
    )


def load_feature_array(
    feature_path: str | Path,
    *,
    dataset_name: str | None = None,
) -> np.ndarray:
    """Load an embedding array from an HDF5 feature file."""
    path = Path(feature_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Feature file does not exist: {path}")

    with h5py.File(path, "r") as handle:
        selected_dataset = (
            dataset_name
            if dataset_name is not None
            else find_embedding_dataset(handle)
        )

        if selected_dataset not in handle:
            raise KeyError(
                f"Dataset {selected_dataset!r} was not found in {path}. "
                f"Available datasets: {inspect_h5_datasets(path)}"
            )

        features = np.asarray(handle[selected_dataset])

    if features.ndim == 1:
        features = features.reshape(1, -1)

    if features.ndim != 2:
        raise ValueError(
            f"Expected a one- or two-dimensional embedding array in {path}, "
            f"but found shape {features.shape}."
        )

    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(f"Feature array is empty in {path}.")

    return features.astype(np.float32, copy=False)


def aggregate_slide_embedding(
    features: np.ndarray,
    method: str = "mean",
) -> np.ndarray:
    """Aggregate tile embeddings into one slide-level embedding."""
    if features.ndim != 2:
        raise ValueError(
            f"Expected a 2D tile-feature matrix, found shape {features.shape}."
        )

    if method == "mean":
        embedding = np.nanmean(features, axis=0)
    elif method == "median":
        embedding = np.nanmedian(features, axis=0)
    elif method == "max":
        embedding = np.nanmax(features, axis=0)
    else:
        raise ValueError(
            "Unsupported slide aggregation method. "
            "Use one of: mean, median, max."
        )

    if not np.isfinite(embedding).all():
        raise ValueError(
            "The aggregated slide embedding contains NaN or infinite values."
        )

    return embedding.astype(np.float32, copy=False)


def feature_inventory(
    feature_files: list[Path],
) -> pd.DataFrame:
    """Create a basic inventory table for discovered feature files."""
    return pd.DataFrame(
        [
            {
                "feature_file": str(path),
                "filename": path.name,
                "stem": path.stem,
                "suffix": path.suffix,
                "size_bytes": path.stat().st_size,
            }
            for path in feature_files
        ]
    )