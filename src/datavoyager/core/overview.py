"""Dataset overview analysis and column type inference."""

from __future__ import annotations

import pandas as pd

from datavoyager.config import Settings
from datavoyager.exceptions import DatasetEmptyError
from datavoyager.profile import ColumnType, OverviewProfile


def format_bytes(num_bytes: int) -> str:
    """Format memory bytes into human-readable string (KB, MB, GB)."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    for unit in ["KB", "MB", "GB", "TB"]:
        num_bytes /= 1024.0
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
    return f"{num_bytes:.2f} PB"


def infer_column_type(series: pd.Series) -> ColumnType:
    """Infer the semantic analytical type for a given Pandas series."""
    dtype_str = str(series.dtype).lower()

    # Boolean
    if "bool" in dtype_str:
        return ColumnType.BOOLEAN

    # Datetime / Timedelta
    if "datetime" in dtype_str or "timedelta" in dtype_str or "period" in dtype_str:
        return ColumnType.DATETIME

    # Numerical (integers, floats, decimal)
    if "int" in dtype_str or "float" in dtype_str:
        return ColumnType.NUMERICAL

    # Complex numbers
    if "complex" in dtype_str:
        return ColumnType.COMPLEX

    # Categorical dtype explicitly
    if "category" in dtype_str:
        return ColumnType.CATEGORICAL

    # If it is object or string, examine non-null values
    non_null = series.dropna()
    if len(non_null) == 0:
        return ColumnType.CATEGORICAL

    # Check if object series contains boolean-like values
    sample = non_null.iloc[:100]
    if all(isinstance(val, (bool,)) for val in sample):
        return ColumnType.BOOLEAN

    # Check if string/object might actually be parsable datetime (heuristically)
    # Only if sample looks like ISO dates or datetimes
    if all(isinstance(val, str) for val in sample):
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pd.to_datetime(sample, errors="raise")
            return ColumnType.DATETIME
        except Exception:
            pass

    return ColumnType.CATEGORICAL


def analyze_overview(
    df: pd.DataFrame,
    config: Settings,
    is_sampled: bool = False,
    sampled_rows: int | None = None,
) -> tuple[OverviewProfile, dict[str, ColumnType]]:
    """Analyze dataset-level metrics, memory consumption, and column types."""
    n_rows, n_cols = df.shape
    if n_rows == 0 or n_cols == 0:
        raise DatasetEmptyError(f"Dataset has no data to profile (shape: {n_rows} rows x {n_cols} columns).")

    total_cells = n_rows * n_cols

    # Calculate memory usage
    try:
        memory_bytes = int(df.memory_usage(deep=True).sum())
    except Exception:
        memory_bytes = int(df.memory_usage(deep=False).sum())

    memory_formatted = format_bytes(memory_bytes)

    # Infer column types
    inferred_types: dict[str, ColumnType] = {}
    type_counts: dict[str, int] = {}

    for col in df.columns:
        col_type = infer_column_type(df[col])
        inferred_types[str(col)] = col_type
        type_name = col_type.value
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

    overview = OverviewProfile(
        n_rows=n_rows,
        n_cols=n_cols,
        total_cells=total_cells,
        memory_bytes=memory_bytes,
        memory_formatted=memory_formatted,
        column_types={k: v.value for k, v in inferred_types.items()},
        type_counts=type_counts,
        is_sampled=is_sampled,
        sampled_rows=sampled_rows,
    )

    return overview, inferred_types
