"""Pairwise correlation analysis for numerical columns."""

from __future__ import annotations

import pandas as pd

from datavoyager.config import Settings
from datavoyager.profile import ColumnProfile, ColumnType, CorrelationPair, CorrelationProfile


def analyze_correlations(
    df: pd.DataFrame,
    column_profiles: dict[str, ColumnProfile],
    config: Settings,
) -> CorrelationProfile:
    """Compute pairwise numerical correlations and highlight strong associations."""
    # Identify non-constant numerical columns
    num_cols = [
        col_name
        for col_name, prof in column_profiles.items()
        if prof.inferred_type == ColumnType.NUMERICAL
        and not prof.is_constant
        and (prof.numerical_stats is not None and prof.numerical_stats.count > 1)
    ]

    if len(num_cols) < 2:
        return CorrelationProfile(
            method=config.correlation_method,
            skipped=True,
            skip_reason="Fewer than two valid non-constant numerical columns available.",
        )

    # Check maximum column limit
    if len(num_cols) > config.max_correlation_columns:
        analyzed_cols = num_cols[: config.max_correlation_columns]
        skip_reason = (
            f"Dataset has {len(num_cols)} numerical columns; restricted to first "
            f"{config.max_correlation_columns} to prevent excessive computational overhead."
        )
    else:
        analyzed_cols = num_cols
        skip_reason = None

    # Subset DataFrame with numeric dtype
    sub_df = df[analyzed_cols].apply(pd.to_numeric, errors="coerce")

    # Compute correlation matrix
    try:
        corr_df = sub_df.corr(method=config.correlation_method)
    except Exception as e:
        return CorrelationProfile(
            method=config.correlation_method,
            skipped=True,
            skip_reason=f"Failed to calculate correlation matrix: {e}",
        )

    # Convert to nested dictionary, replacing NaN with 0.0 or None
    matrix_dict: dict[str, dict[str, float]] = {}
    for r in analyzed_cols:
        matrix_dict[r] = {}
        for c in analyzed_cols:
            val = corr_df.loc[r, c] if (r in corr_df.index and c in corr_df.columns) else 0.0
            matrix_dict[r][c] = 0.0 if pd.isna(val) else round(float(val), 4)

    # Extract distinct strong correlation pairs (upper triangle)
    strong_pairs: list[CorrelationPair] = []
    for i in range(len(analyzed_cols)):
        for j in range(i + 1, len(analyzed_cols)):
            col1 = analyzed_cols[i]
            col2 = analyzed_cols[j]
            coeff = matrix_dict[col1][col2]
            abs_coeff = abs(coeff)

            if abs_coeff >= config.correlation_threshold:
                strong_pairs.append(
                    CorrelationPair(
                        col1=col1,
                        col2=col2,
                        coefficient=coeff,
                        abs_coefficient=round(abs_coeff, 4),
                        method=config.correlation_method,
                    )
                )

    # Sort descending by absolute strength
    strong_pairs.sort(key=lambda x: x.abs_coefficient, reverse=True)

    return CorrelationProfile(
        method=config.correlation_method,
        matrix=matrix_dict,
        strong_correlations=strong_pairs,
        analyzed_columns=analyzed_cols,
        skipped=False,
        skip_reason=skip_reason,
    )
