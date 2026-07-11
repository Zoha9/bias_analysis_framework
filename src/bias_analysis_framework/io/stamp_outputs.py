"""Discovery helpers for STAMP training and deployment outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StampCheckpoint:
    """One trained STAMP cross-validation checkpoint."""

    label_name: str
    model_name: str
    split_name: str
    split_number: int
    checkpoint_path: Path


def discover_stamp_checkpoints(
    *,
    crossval_results_root: str | Path,
    label_name: str,
    model_name: str,
    checkpoint_filename: str = "model.ckpt",
) -> list[StampCheckpoint]:
    """Discover trained checkpoints for one label and model.

    Expected structure:

    crossval_results_root/
        label/
            model/
                split-0/
                    model.ckpt
                split-1/
                    model.ckpt
    """
    root = (
        Path(crossval_results_root)
        .expanduser()
        .resolve()
    )

    label_dir = resolve_named_directory(
        parent=root,
        requested_name=label_name,
    )

    if label_dir is None:
        return []

    model_dir = resolve_named_directory(
        parent=label_dir,
        requested_name=model_name,
    )

    if model_dir is None:
        return []

    checkpoints: list[StampCheckpoint] = []

    for split_dir in sorted(model_dir.iterdir()):
        if not split_dir.is_dir():
            continue

        split_number = parse_split_number(
            split_dir.name
        )

        if split_number is None:
            continue

        checkpoint_path = (
            split_dir
            / checkpoint_filename
        )

        if not checkpoint_path.is_file():
            continue

        checkpoints.append(
            StampCheckpoint(
                label_name=label_name,
                model_name=model_name,
                split_name=split_dir.name,
                split_number=split_number,
                checkpoint_path=checkpoint_path.resolve(),
            )
        )

    return sorted(
        checkpoints,
        key=lambda checkpoint: checkpoint.split_number,
    )


def resolve_named_directory(
    *,
    parent: str | Path,
    requested_name: str,
) -> Path | None:
    """Resolve a child directory using exact or normalized naming."""
    parent = Path(parent)

    if not parent.is_dir():
        return None

    exact = parent / requested_name

    if exact.is_dir():
        return exact.resolve()

    requested_normalized = normalize_directory_name(
        requested_name
    )

    matches = [
        child
        for child in parent.iterdir()
        if child.is_dir()
        and normalize_directory_name(
            child.name
        )
        == requested_normalized
    ]

    if len(matches) == 1:
        return matches[0].resolve()

    if len(matches) > 1:
        match_text = "\n".join(
            f"  - {match}"
            for match in matches
        )

        raise ValueError(
            f"Multiple directories match "
            f"{requested_name!r} under {parent}:\n"
            f"{match_text}"
        )

    return None


def parse_split_number(
    split_name: str,
) -> int | None:
    """Parse a STAMP split directory such as split-0."""
    prefix = "split-"

    if not split_name.startswith(prefix):
        return None

    raw_number = split_name[
        len(prefix):
    ]

    try:
        return int(raw_number)
    except ValueError:
        return None


def normalize_directory_name(
    value: object,
) -> str:
    """Normalize labels used in STAMP result-directory names."""
    text = str(value).strip().casefold()

    for character in (
        " ",
        "-",
        "/",
        "\\",
        ":",
        ";",
        ",",
        "|",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
    ):
        text = text.replace(
            character,
            "_",
        )

    while "__" in text:
        text = text.replace(
            "__",
            "_",
        )

    return text.strip("_")


def count_prediction_files(
    output_dir: str | Path,
    *,
    suffixes: tuple[str, ...] = (
        ".csv",
        ".xlsx",
        ".parquet",
        ".json",
    ),
) -> int:
    """Count likely prediction output files recursively."""
    path = Path(output_dir)

    if not path.is_dir():
        return 0

    normalized_suffixes = {
        suffix.casefold()
        for suffix in suffixes
    }

    return sum(
        1
        for candidate in path.rglob("*")
        if candidate.is_file()
        and candidate.suffix.casefold()
        in normalized_suffixes
    )