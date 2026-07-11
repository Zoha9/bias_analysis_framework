from __future__ import annotations

from pathlib import Path

import pandas as pd


class TableReadError(ValueError):
    """Raised when an input table cannot be read."""


def read_table(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read CSV, TSV, or Excel table based on file extension."""
    table_path = Path(path).expanduser().resolve()

    if not table_path.exists():
        raise TableReadError(f"Table does not exist: {table_path}")

    suffix = table_path.suffix.lower()

    try:
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(table_path, **kwargs)

        if suffix == ".csv":
            return pd.read_csv(table_path, **kwargs)

        if suffix in {".tsv", ".txt"}:
            return pd.read_csv(table_path, sep="\t", **kwargs)

    except Exception as exc:
        raise TableReadError(f"Failed to read table {table_path}: {exc}") from exc

    raise TableReadError(
        f"Unsupported table format for {table_path}. "
        "Supported formats: .xlsx, .xls, .csv, .tsv, .txt"
    )


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    table_name: str,
) -> None:
    """Validate that required columns exist in a dataframe."""
    missing = [column for column in required_columns if column not in dataframe.columns]

    if missing:
        available_preview = ", ".join(map(str, dataframe.columns[:30]))
        raise ValueError(
            f"{table_name} is missing required columns: {missing}. "
            f"Available columns include: {available_preview}"
        )


def write_excel_sheets(
    output_path: str | Path,
    sheets: dict[str, pd.DataFrame],
    index: bool = False,
) -> Path:
    """Write multiple dataframes to an Excel workbook."""
    resolved = Path(output_path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(resolved, engine="openpyxl") as writer:
        for sheet_name, dataframe in sheets.items():
            safe_sheet = str(sheet_name)[:31]
            dataframe.to_excel(writer, sheet_name=safe_sheet, index=index)

    return resolved