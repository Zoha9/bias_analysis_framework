"""Discovery helpers for STAMP feature and result directories."""

from __future__ import annotations

from pathlib import Path


FEATURE_SUFFIXES = {
    ".h5",
    ".hdf5",
}


def discover_model_feature_directory(
    *,
    feature_root: str | Path,
    model_name: str,
    organ: str | None = None,
    dataset: str | None = None,
) -> Path:
    """Find the STAMP feature directory for one model.

    STAMP commonly creates directories such as:

    ``uni-<version>/data/``
    ``conch-<version>/data/``
    ``virchow2-<version>/data/``

    The function also supports explicitly separated model directories.
    """
    root = Path(feature_root).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(
            f"Feature root does not exist: {root}"
        )

    search_roots = build_feature_search_roots(
        feature_root=root,
        organ=organ,
        dataset=dataset,
    )

    candidates: list[Path] = []

    for search_root in search_roots:
        if not search_root.exists():
            continue

        direct_model_dir = search_root / model_name
        if direct_model_dir.is_dir():
            candidates.append(direct_model_dir)

        for child in search_root.iterdir():
            if not child.is_dir():
                continue

            child_name = child.name.casefold()
            model_casefold = model_name.casefold()

            if child_name == model_casefold:
                candidates.append(child)
            elif child_name.startswith(
                f"{model_casefold}-"
            ):
                candidates.append(child)
            elif child_name.startswith(
                f"{model_casefold}_"
            ):
                candidates.append(child)

    candidates = deduplicate_paths(candidates)

    usable_candidates: list[Path] = []

    for candidate in candidates:
        feature_dir = select_feature_data_directory(
            candidate
        )

        if contains_feature_files(feature_dir):
            usable_candidates.append(feature_dir)

    if not usable_candidates:
        searched = ", ".join(
            str(path)
            for path in search_roots
        )

        raise FileNotFoundError(
            f"No usable feature directory was found for model "
            f"{model_name!r}. Searched: {searched}"
        )

    if len(usable_candidates) > 1:
        exact_matches = [
            candidate
            for candidate in usable_candidates
            if candidate.parent.name.casefold()
            == model_name.casefold()
            or candidate.name.casefold()
            == model_name.casefold()
        ]

        if len(exact_matches) == 1:
            return exact_matches[0]

        candidates_text = "\n".join(
            f"  - {candidate}"
            for candidate in usable_candidates
        )

        raise ValueError(
            f"Multiple feature directories match model "
            f"{model_name!r}:\n{candidates_text}\n"
            "Remove duplicate directories or configure a unique "
            "feature root."
        )

    return usable_candidates[0]


def build_feature_search_roots(
    *,
    feature_root: Path,
    organ: str | None,
    dataset: str | None,
) -> list[Path]:
    """Build likely feature-root locations in priority order."""
    output: list[Path] = []

    if organ and dataset:
        output.extend(
            [
                feature_root / organ / dataset,
                feature_root / f"{organ}_{dataset}",
                feature_root / f"{organ}-{dataset}",
            ]
        )

    if dataset:
        output.extend(
            [
                feature_root / dataset,
            ]
        )

    output.append(feature_root)

    return deduplicate_paths(output)


def select_feature_data_directory(
    candidate: Path,
) -> Path:
    """Select STAMP's nested data directory when present."""
    data_dir = candidate / "data"

    if data_dir.is_dir():
        return data_dir

    return candidate


def contains_feature_files(
    directory: str | Path,
) -> bool:
    """Return whether a directory contains HDF5 feature files."""
    path = Path(directory)

    if not path.exists() or not path.is_dir():
        return False

    return any(
        child.is_file()
        and child.suffix.casefold() in FEATURE_SUFFIXES
        for child in path.iterdir()
    )


def count_feature_files(
    directory: str | Path,
) -> int:
    """Count HDF5 feature files in a directory."""
    path = Path(directory)

    if not path.exists() or not path.is_dir():
        return 0

    return sum(
        1
        for child in path.iterdir()
        if child.is_file()
        and child.suffix.casefold() in FEATURE_SUFFIXES
    )


def deduplicate_paths(
    paths: list[Path],
) -> list[Path]:
    """Deduplicate paths while preserving order."""
    output: list[Path] = []
    seen: set[Path] = set()

    for path in paths:
        resolved = path.expanduser().resolve()

        if resolved not in seen:
            output.append(resolved)
            seen.add(resolved)

    return output