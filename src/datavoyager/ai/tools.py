"""Verified deterministic analytical query tools for dataset interrogation."""

from __future__ import annotations

from typing import Any

from datavoyager.profile import ColumnType, DatasetProfile


def tool_get_overview(profile: DatasetProfile) -> dict[str, Any]:
    """Retrieve verified dataset dimensions, cell count, and memory usage."""
    ov = profile.overview
    return {
        "rows": ov.n_rows,
        "columns": ov.n_cols,
        "total_cells": ov.total_cells,
        "memory_formatted": ov.memory_formatted,
        "type_counts": ov.type_counts,
    }


def tool_get_missing_summary(profile: DatasetProfile, top_n: int = 5) -> dict[str, Any]:
    """Retrieve columns with highest missing value rates."""
    ms = profile.missing
    return {
        "total_missing_cells": ms.total_missing_cells,
        "overall_missing_ratio": ms.overall_missing_ratio,
        "top_missing_columns": ms.columns_with_missing[:top_n],
    }


def tool_get_duplicate_summary(profile: DatasetProfile) -> dict[str, Any]:
    """Retrieve duplicate row count and ratio."""
    ql = profile.quality
    return {
        "duplicate_rows": ql.duplicate_rows,
        "duplicate_ratio": ql.duplicate_ratio,
    }


def tool_get_strong_correlations(profile: DatasetProfile, top_n: int = 5) -> dict[str, Any]:
    """Retrieve verified strong correlation pairs."""
    corr = profile.correlations
    pairs = [
        {
            "col1": p.col1,
            "col2": p.col2,
            "coefficient": p.coefficient,
            "method": p.method,
        }
        for p in corr.strong_correlations[:top_n]
    ]
    return {
        "method": corr.method,
        "skipped": corr.skipped,
        "skip_reason": corr.skip_reason,
        "pairs": pairs,
    }


def tool_get_column_details(profile: DatasetProfile, column_name: str) -> dict[str, Any] | None:
    """Retrieve verified statistics for a specific column."""
    # Case-insensitive column matching
    match = None
    for col_key, col_prof in profile.columns.items():
        if col_key.lower() == column_name.lower().strip():
            match = col_prof
            break

    if match is None:
        return None

    res: dict[str, Any] = {
        "name": match.name,
        "type": match.inferred_type.value,
        "missing_count": match.missing_count,
        "missing_ratio": match.missing_ratio,
        "distinct_count": match.distinct_count,
        "warnings": match.warnings,
    }

    if match.numerical_stats is not None:
        ns = match.numerical_stats
        res["numerical"] = {
            "mean": ns.mean,
            "std": ns.std,
            "min": ns.min,
            "median": ns.median,
            "max": ns.max,
            "skewness": ns.skewness,
            "outliers_count": ns.outliers_count,
            "outliers_ratio": ns.outliers_ratio,
        }
    elif match.categorical_stats is not None:
        cs = match.categorical_stats
        res["categorical"] = {
            "unique_count": cs.unique_count,
            "mode": cs.mode,
            "top_frequencies": cs.top_frequencies[:5],
        }

    return res


def tool_get_outliers_summary(profile: DatasetProfile) -> dict[str, Any]:
    """Retrieve columns that have detected outliers."""
    return {
        "columns_with_outliers": profile.quality.columns_with_outliers,
    }
