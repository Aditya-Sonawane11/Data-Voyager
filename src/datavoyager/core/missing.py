"""Missing-value diagnostics and pattern analysis."""

from __future__ import annotations

from typing import Any

import pandas as pd

from datavoyager.profile import MissingProfile


def analyze_missing(df: pd.DataFrame) -> MissingProfile:
    """Analyze missing values across all columns and overall dataset."""
    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols

    missing_series = df.isna().sum()
    total_missing_cells = int(missing_series.sum())
    overall_missing_ratio = float(total_missing_cells / total_cells) if total_cells > 0 else 0.0

    missing_cells_by_col: dict[str, int] = {}
    missing_ratios_by_col: dict[str, float] = {}
    cols_with_missing: list[dict[str, Any]] = []

    for col, count in missing_series.items():
        col_str = str(col)
        cnt = int(count)
        ratio = float(cnt / n_rows) if n_rows > 0 else 0.0

        missing_cells_by_col[col_str] = cnt
        missing_ratios_by_col[col_str] = ratio

        if cnt > 0:
            cols_with_missing.append({
                "column": col_str,
                "missing_count": cnt,
                "missing_ratio": ratio,
            })

    # Sort descending by missing count
    cols_with_missing.sort(key=lambda x: x["missing_count"], reverse=True)

    return MissingProfile(
        total_missing_cells=total_missing_cells,
        overall_missing_ratio=overall_missing_ratio,
        columns_with_missing=cols_with_missing,
        missing_cells_by_column=missing_cells_by_col,
        missing_ratios_by_column=missing_ratios_by_col,
    )
