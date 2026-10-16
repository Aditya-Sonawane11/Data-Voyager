"""Detailed column-level statistical profiling."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from datax.config import Settings
from datax.profile import (
    CategoricalStats,
    ColumnProfile,
    ColumnType,
    DatetimeStats,
    NumericalStats,
)


def profile_numerical_column(
    series: pd.Series,
    config: Settings,
    total_rows: int,
) -> tuple[NumericalStats, list[str]]:
    """Compute comprehensive distribution and outlier statistics for a numerical column."""
    warnings: list[str] = []

    # Drop NaNs
    valid = series.dropna()
    valid_count = len(valid)

    if valid_count == 0:
        warnings.append("All values in this numerical column are missing.")
        return NumericalStats(count=0), warnings

    # Check for infinities
    inf_mask = np.isinf(valid.values) if valid.dtype.kind in "fc" else np.zeros(valid_count, dtype=bool)
    infinite_count = int(np.sum(inf_mask))
    if infinite_count > 0:
        warnings.append(f"Contains {infinite_count} non-finite/infinite value(s).")
        finite_valid = valid[~inf_mask]
    else:
        finite_valid = valid

    finite_count = len(finite_valid)
    if finite_count == 0:
        warnings.append("All non-null values are infinite.")
        return (
            NumericalStats(
                count=valid_count,
                infinite_count=infinite_count,
            ),
            warnings,
        )

    # Convert to float array for robust calculation
    arr = finite_valid.to_numpy(dtype=float)

    # Zeros
    zeros_count = int(np.sum(arr == 0.0))
    zeros_ratio = float(zeros_count / total_rows) if total_rows > 0 else 0.0

    # Summary statistics
    c_mean = float(np.mean(arr))
    c_std = float(np.std(arr, ddof=1)) if finite_count > 1 else 0.0
    c_min = float(np.min(arr))
    c_max = float(np.max(arr))

    # Quartiles
    p25 = float(np.percentile(arr, 25))
    median = float(np.percentile(arr, 50))
    p75 = float(np.percentile(arr, 75))
    iqr = float(p75 - p25)

    # Skewness (Fisher-Pearson coefficient of skewness)
    skewness: float | None = None
    if finite_count >= 3 and c_std > 0:
        try:
            m3 = float(np.mean((arr - c_mean) ** 3))
            skewness = float(m3 / (c_std**3))
            if math.isnan(skewness) or math.isinf(skewness):
                skewness = None
        except Exception:
            skewness = None

    # Outlier detection using IQR rule
    outliers_count = 0
    outliers_ratio = 0.0
    if iqr > 0:
        lower_bound = p25 - config.iqr_multiplier * iqr
        upper_bound = p75 + config.iqr_multiplier * iqr
        outlier_mask = (arr < lower_bound) | (arr > upper_bound)
        outliers_count = int(np.sum(outlier_mask))
        outliers_ratio = float(outliers_count / finite_count)
        if outliers_count > 0:
            warnings.append(
                f"{outliers_count} ({outliers_ratio:.1%}) outlier(s) detected via IQR x {config.iqr_multiplier} rule."
            )
    elif c_std == 0.0:
        warnings.append("Column has zero variance (all non-null finite values are identical).")

    stats = NumericalStats(
        count=valid_count,
        mean=c_mean,
        std=c_std,
        min=c_min,
        p25=p25,
        median=median,
        p75=p75,
        max=c_max,
        skewness=skewness,
        zeros_count=zeros_count,
        zeros_ratio=zeros_ratio,
        infinite_count=infinite_count,
        iqr=iqr,
        outliers_count=outliers_count,
        outliers_ratio=outliers_ratio,
    )

    return stats, warnings


def profile_categorical_column(
    series: pd.Series,
    config: Settings,
    total_rows: int,
) -> tuple[CategoricalStats, list[str]]:
    """Compute frequency distribution and cardinality for a categorical/string column."""
    warnings: list[str] = []

    valid = series.dropna()
    valid_count = len(valid)

    if valid_count == 0:
        warnings.append("All values in this categorical column are missing.")
        return CategoricalStats(count=0, unique_count=0, unique_ratio=0.0), warnings

    # Value counts
    val_counts = valid.value_counts()
    unique_count = len(val_counts)
    unique_ratio = float(unique_count / total_rows) if total_rows > 0 else 0.0

    mode = val_counts.index[0] if unique_count > 0 else None

    # Top frequencies
    top_entries: list[dict[str, Any]] = []
    top_n = val_counts.iloc[: config.categorical_top_n]
    for val, count in top_n.items():
        top_entries.append({
            "value": str(val),
            "count": int(count),
            "ratio": float(count / total_rows) if total_rows > 0 else 0.0,
        })

    if unique_ratio > 0.8 and total_rows > 50:
        warnings.append("High cardinality: over 80% distinct values.")

    stats = CategoricalStats(
        count=valid_count,
        unique_count=unique_count,
        unique_ratio=unique_ratio,
        top_frequencies=top_entries,
        mode=str(mode) if mode is not None else None,
    )

    return stats, warnings


def profile_datetime_column(
    series: pd.Series,
    total_rows: int,
) -> tuple[DatetimeStats, list[str]]:
    """Compute temporal range and unique counts for a datetime column."""
    warnings: list[str] = []

    # Ensure datetime parsing if needed
    if not pd.api.types.is_datetime64_any_dtype(series):
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                converted = pd.to_datetime(series, errors="coerce")
        except Exception:
            converted = series
    else:
        converted = series

    valid = converted.dropna()
    valid_count = len(valid)

    if valid_count == 0:
        warnings.append("All values in this datetime column are missing or unparseable.")
        return DatetimeStats(count=0), warnings

    min_date = str(valid.min())
    max_date = str(valid.max())

    range_days: float | None = None
    try:
        delta = valid.max() - valid.min()
        range_days = float(delta.total_seconds() / 86400.0)
    except Exception:
        pass

    unique_count = int(valid.nunique())

    stats = DatetimeStats(
        count=valid_count,
        min_date=min_date,
        max_date=max_date,
        range_days=range_days,
        unique_count=unique_count,
    )

    return stats, warnings


def analyze_columns(
    df: pd.DataFrame,
    inferred_types: dict[str, ColumnType],
    config: Settings,
) -> dict[str, ColumnProfile]:
    """Profile every column in the dataset according to its analytical category."""
    n_rows = len(df)
    profiles: dict[str, ColumnProfile] = {}

    for col in df.columns:
        col_str = str(col)
        series = df[col]
        col_type = inferred_types.get(col_str, ColumnType.CATEGORICAL)

        total_count = n_rows
        missing_count = int(series.isna().sum())
        missing_ratio = float(missing_count / total_count) if total_count > 0 else 0.0

        distinct_count = int(series.nunique(dropna=True))
        distinct_ratio = float(distinct_count / total_count) if total_count > 0 else 0.0

        is_constant = distinct_count <= 1
        is_quasi_constant = False
        is_potential_id = (distinct_count == total_count) and (total_count > 10) and (missing_count == 0)

        # Check quasi-constant
        if not is_constant and total_count > 0:
            top_freq_count = series.value_counts(dropna=True).max() if distinct_count > 0 else 0
            if (top_freq_count / total_count) >= config.quasi_constant_threshold:
                is_quasi_constant = True

        column_warnings: list[str] = []

        if missing_ratio >= config.missing_warning_threshold:
            column_warnings.append(
                f"Missing rate is {missing_ratio:.1%} (>= {config.missing_warning_threshold:.1%})."
            )
        if is_constant:
            column_warnings.append("Constant column (single unique value).")
        elif is_quasi_constant:
            column_warnings.append(
                f"Quasi-constant column (>= {config.quasi_constant_threshold:.1%} identical values)."
            )
        if is_potential_id:
            column_warnings.append("Potential primary key or record identifier (all values unique).")

        num_stats: NumericalStats | None = None
        cat_stats: CategoricalStats | None = None
        dt_stats: DatetimeStats | None = None

        if col_type == ColumnType.NUMERICAL:
            num_stats, extra_warnings = profile_numerical_column(series, config, total_count)
            column_warnings.extend(extra_warnings)
        elif col_type == ColumnType.DATETIME:
            dt_stats, extra_warnings = profile_datetime_column(series, total_count)
            column_warnings.extend(extra_warnings)
        elif col_type in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN, ColumnType.COMPLEX, ColumnType.OTHER):
            cat_stats, extra_warnings = profile_categorical_column(series, config, total_count)
            column_warnings.extend(extra_warnings)

        profiles[col_str] = ColumnProfile(
            name=col_str,
            inferred_type=col_type,
            dtype=str(series.dtype),
            total_count=total_count,
            missing_count=missing_count,
            missing_ratio=missing_ratio,
            distinct_count=distinct_count,
            distinct_ratio=distinct_ratio,
            is_constant=is_constant,
            is_quasi_constant=is_quasi_constant,
            is_potential_id=is_potential_id,
            numerical_stats=num_stats,
            categorical_stats=cat_stats,
            datetime_stats=dt_stats,
            warnings=column_warnings,
        )

    return profiles
