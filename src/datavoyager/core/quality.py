"""Dataset quality diagnostics, duplicate detection, and anomaly checks."""

from __future__ import annotations

import pandas as pd

from datavoyager.profile import ColumnProfile, QualityProfile


def analyze_quality(
    df: pd.DataFrame,
    column_profiles: dict[str, ColumnProfile],
) -> QualityProfile:
    """Analyze dataset quality issues, duplicates, constants, and anomalies."""
    n_rows = len(df)

    # Duplicate rows
    try:
        duplicate_rows = int(df.duplicated().sum())
    except Exception:
        duplicate_rows = 0
    duplicate_ratio = float(duplicate_rows / n_rows) if n_rows > 0 else 0.0

    constant_cols: list[str] = []
    quasi_constant_cols: list[str] = []
    potential_id_cols: list[str] = []
    columns_with_outliers: dict[str, int] = {}
    columns_with_infinities: dict[str, int] = {}

    for col_name, profile in column_profiles.items():
        if profile.is_constant:
            constant_cols.append(col_name)
        elif profile.is_quasi_constant:
            quasi_constant_cols.append(col_name)

        if profile.is_potential_id:
            potential_id_cols.append(col_name)

        if profile.numerical_stats is not None:
            if profile.numerical_stats.outliers_count > 0:
                columns_with_outliers[col_name] = profile.numerical_stats.outliers_count
            if profile.numerical_stats.infinite_count > 0:
                columns_with_infinities[col_name] = profile.numerical_stats.infinite_count

    return QualityProfile(
        duplicate_rows=duplicate_rows,
        duplicate_ratio=duplicate_ratio,
        constant_columns=constant_cols,
        quasi_constant_columns=quasi_constant_cols,
        potential_id_columns=potential_id_cols,
        columns_with_outliers=columns_with_outliers,
        columns_with_infinities=columns_with_infinities,
    )
