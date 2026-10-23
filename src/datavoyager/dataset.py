"""Dataset input/output utilities — CSV, Parquet, Excel, JSON ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from datavoyager.exceptions import DatasetEmptyError


def _validate_path(path: Path) -> None:
    """Raise FileNotFoundError if path does not point to an existing file."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")


def _validate_df(df: pd.DataFrame, path: Path) -> None:
    """Raise DatasetEmptyError if the loaded DataFrame has no data."""
    if df.empty:
        raise DatasetEmptyError(f"Dataset at '{path}' is empty (0 rows or columns).")


def read_csv(
    filepath: str | Path,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a CSV file into a Pandas DataFrame with validation.

    Args:
        filepath: Path to the CSV file.
        **kwargs: Optional arguments passed directly to ``pd.read_csv``.

    Returns:
        pd.DataFrame loaded from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        DatasetEmptyError: If the resulting DataFrame has 0 rows or columns.
    """
    path = Path(filepath)
    _validate_path(path)
    # Set default low_memory to False for consistent type inference unless overridden
    kwargs.setdefault("low_memory", False)
    df = pd.read_csv(path, **kwargs)
    _validate_df(df, path)
    return df


def read_parquet(
    filepath: str | Path,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a Parquet file into a Pandas DataFrame with validation.

    Requires ``pyarrow`` or ``fastparquet`` to be installed (standard Parquet engines).

    Args:
        filepath: Path to the ``.parquet`` file.
        **kwargs: Optional arguments passed directly to ``pd.read_parquet``.

    Returns:
        pd.DataFrame loaded from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        DatasetEmptyError: If the resulting DataFrame has 0 rows or columns.
        ImportError: If no Parquet engine (``pyarrow`` or ``fastparquet``) is installed.
    """
    path = Path(filepath)
    _validate_path(path)
    try:
        df = pd.read_parquet(path, **kwargs)
    except ImportError as exc:
        raise ImportError(
            "Reading Parquet files requires 'pyarrow' or 'fastparquet'. "
            "Install one via: pip install pyarrow"
        ) from exc
    _validate_df(df, path)
    return df


def read_excel(
    filepath: str | Path,
    sheet_name: str | int | None = 0,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read an Excel file (.xlsx / .xls / .ods) into a Pandas DataFrame with validation.

    Requires ``openpyxl`` (xlsx) or ``xlrd`` (xls) to be installed.

    Args:
        filepath: Path to the Excel file.
        sheet_name: Sheet to load. Defaults to the first sheet (index 0).
            Pass a sheet name string or integer index.
        **kwargs: Optional arguments passed directly to ``pd.read_excel``.

    Returns:
        pd.DataFrame loaded from the specified sheet.

    Raises:
        FileNotFoundError: If the file does not exist.
        DatasetEmptyError: If the resulting DataFrame has 0 rows or columns.
        ImportError: If the required Excel engine is not installed.
    """
    path = Path(filepath)
    _validate_path(path)
    try:
        df = pd.read_excel(path, sheet_name=sheet_name, **kwargs)
    except ImportError as exc:
        raise ImportError(
            "Reading Excel files requires 'openpyxl' (for .xlsx) or 'xlrd' (for .xls). "
            "Install via: pip install openpyxl"
        ) from exc
    if not isinstance(df, pd.DataFrame):
        # pd.read_excel returns a dict when sheet_name=None; guard against that
        raise TypeError(
            "read_excel returned multiple sheets. Pass a specific sheet_name to load a single sheet."
        )
    _validate_df(df, path)
    return df


def read_json(
    filepath: str | Path,
    orient: str | None = None,
    lines: bool = False,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a JSON or JSON Lines file into a Pandas DataFrame with validation.

    Args:
        filepath: Path to the ``.json`` or ``.jsonl`` file.
        orient: JSON orientation string passed to ``pd.read_json``
            (e.g. ``'records'``, ``'index'``, ``'columns'``).
            Auto-detected by pandas when ``None``.
        lines: If ``True``, reads the file as newline-delimited JSON (JSON Lines /
            ``.jsonl`` format). Defaults to ``False``.
        **kwargs: Optional arguments passed directly to ``pd.read_json``.

    Returns:
        pd.DataFrame loaded from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        DatasetEmptyError: If the resulting DataFrame has 0 rows or columns.
    """
    path = Path(filepath)
    _validate_path(path)
    read_kwargs: dict[str, Any] = {"lines": lines, **kwargs}
    if orient is not None:
        read_kwargs["orient"] = orient
    df = pd.read_json(path, **read_kwargs)
    _validate_df(df, path)
    return df


def read_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Accept a pre-built Pandas DataFrame, validating it is non-empty.

    This is a convenience alias for passing a DataFrame directly into the
    DataVoyager pipeline without any file I/O.

    Args:
        df: An existing ``pd.DataFrame``.

    Returns:
        The same DataFrame (unmodified), after validation.

    Raises:
        TypeError: If the input is not a ``pd.DataFrame``.
        DatasetEmptyError: If the DataFrame has 0 rows or columns.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__!r}.")
    if df.empty:
        raise DatasetEmptyError("The provided DataFrame is empty (0 rows or columns).")
    return df
