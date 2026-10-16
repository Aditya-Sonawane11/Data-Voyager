"""Dataset input/output utilities and CSV ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from datax.exceptions import DatasetEmptyError


def read_csv(
    filepath: str | Path,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read a CSV file into a Pandas DataFrame with validation.

    Args:
        filepath: Path to the CSV file.
        **kwargs: Optional arguments passed directly to `pd.read_csv`.

    Returns:
        pd.DataFrame loaded from the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        DatasetEmptyError: If the resulting DataFrame has 0 rows or columns.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    # Set default low_memory to False for consistent type inference unless overridden
    kwargs.setdefault("low_memory", False)

    df = pd.read_csv(path, **kwargs)

    if df.empty:
        raise DatasetEmptyError(f"Dataset at '{path}' is empty (0 rows or columns).")

    return df
