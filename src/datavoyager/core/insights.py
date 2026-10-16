"""Automated analytical insight generation engine."""

from __future__ import annotations

from datavoyager.config import Settings
from datavoyager.profile import (
    ColumnProfile,
    ColumnType,
    CorrelationProfile,
    Insight,
    MissingProfile,
    OverviewProfile,
    QualityProfile,
)


def generate_insights(
    overview: OverviewProfile,
    columns: dict[str, ColumnProfile],
    missing: MissingProfile,
    quality: QualityProfile,
    correlations: CorrelationProfile,
    config: Settings,
) -> list[Insight]:
    """Inspect all analytical profiles and formulate structured actionable insights."""
    insights: list[Insight] = []

    # 1. Missingness Insights
    if missing.overall_missing_ratio > 0.30:
        insights.append(
            Insight(
                severity="critical",
                category="missing",
                title="High Dataset Incompleteness",
                message=(
                    f"{missing.overall_missing_ratio:.1%} of all cells in the dataset are missing "
                    f"({missing.total_missing_cells:,} missing cells)."
                ),
                details={"overall_ratio": missing.overall_missing_ratio},
            )
        )
    elif missing.overall_missing_ratio > 0.05:
        insights.append(
            Insight(
                severity="warning",
                category="missing",
                title="Moderate Missing Values",
                message=(
                    f"Overall missingness is {missing.overall_missing_ratio:.1%}. "
                    f"{len(missing.columns_with_missing)} column(s) contain missing values."
                ),
                details={"columns_affected": len(missing.columns_with_missing)},
            )
        )

    # Columns with > 50% missing values
    for item in missing.columns_with_missing:
        if item["missing_ratio"] >= 0.50:
            insights.append(
                Insight(
                    severity="warning",
                    category="missing",
                    title=f"Severely Incomplete Column: '{item['column']}'",
                    message=(
                        f"Column '{item['column']}' is missing {item['missing_ratio']:.1%} of its values "
                        f"({item['missing_count']:,} rows). Consider imputation or dropping if uninformative."
                    ),
                    details=item,
                )
            )

    # 2. Duplicate Row Insights
    if quality.duplicate_rows > 0:
        sev = "critical" if quality.duplicate_ratio > 0.10 else "warning"
        insights.append(
            Insight(
                severity=sev,
                category="duplicates",
                title="Duplicate Rows Detected",
                message=(
                    f"Found {quality.duplicate_rows:,} duplicate rows ({quality.duplicate_ratio:.1%}). "
                    f"Verify whether these represent repeated events or redundant recordings."
                ),
                details={
                    "duplicate_rows": quality.duplicate_rows,
                    "duplicate_ratio": quality.duplicate_ratio,
                },
            )
        )

    # 3. Constant and Quasi-Constant Columns
    for col_name in quality.constant_columns:
        insights.append(
            Insight(
                severity="warning",
                category="cardinality",
                title=f"Zero Variance in '{col_name}'",
                message=(
                    f"Column '{col_name}' is constant (contains only a single unique value). "
                    f"It provides zero predictive or analytical variance."
                ),
                details={"column": col_name},
            )
        )

    for col_name in quality.quasi_constant_columns:
        insights.append(
            Insight(
                severity="info",
                category="cardinality",
                title=f"Quasi-Constant Column '{col_name}'",
                message=(
                    f"Column '{col_name}' has >= {config.quasi_constant_threshold:.1%} identical values. "
                    f"Information content is very low."
                ),
                details={"column": col_name},
            )
        )

    # 4. Potential Identifiers
    for col_name in quality.potential_id_columns:
        insights.append(
            Insight(
                severity="info",
                category="cardinality",
                title=f"Candidate Identifier: '{col_name}'",
                message=(
                    f"Column '{col_name}' has 100% distinct values across {overview.n_rows:,} rows with no missing data. "
                    f"Likely functions as a primary key or entity identifier."
                ),
                details={"column": col_name},
            )
        )

    # 5. Outliers and Skewness
    for col_name, prof in columns.items():
        if prof.inferred_type == ColumnType.NUMERICAL and prof.numerical_stats is not None:
            stats = prof.numerical_stats
            # Outliers
            if stats.outliers_ratio > 0.05:
                insights.append(
                    Insight(
                        severity="warning",
                        category="outliers",
                        title=f"Elevated Outlier Rate in '{col_name}'",
                        message=(
                            f"{stats.outliers_count:,} values ({stats.outliers_ratio:.1%}) lie beyond "
                            f"{config.iqr_multiplier}x IQR bounds (Q1={stats.p25}, Q3={stats.p75})."
                        ),
                        details={
                            "column": col_name,
                            "outliers_count": stats.outliers_count,
                            "outliers_ratio": stats.outliers_ratio,
                        },
                    )
                )

            # Severe Skewness
            if stats.skewness is not None and abs(stats.skewness) >= 2.0:
                direction = "right/positive" if stats.skewness > 0 else "left/negative"
                insights.append(
                    Insight(
                        severity="info",
                        category="distribution",
                        title=f"Heavily Skewed Distribution: '{col_name}'",
                        message=(
                            f"Column '{col_name}' displays substantial {direction} skewness ({stats.skewness:.2f}). "
                            f"Consider a power or log transformation if using linear modeling."
                        ),
                        details={"column": col_name, "skewness": stats.skewness},
                    )
                )

            # Non-finite values
            if stats.infinite_count > 0:
                insights.append(
                    Insight(
                        severity="critical",
                        category="outliers",
                        title=f"Non-Finite / Infinite Values in '{col_name}'",
                        message=(
                            f"Column '{col_name}' contains {stats.infinite_count:,} infinite values. "
                            f"These must be handled before numerical model training."
                        ),
                        details={"column": col_name, "infinite_count": stats.infinite_count},
                    )
                )

    # 6. Strong Correlations
    for pair in correlations.strong_correlations[:5]:
        direction = "positive" if pair.coefficient > 0 else "negative"
        insights.append(
            Insight(
                severity="info",
                category="correlation",
                title=f"Strong Association: '{pair.col1}' & '{pair.col2}'",
                message=(
                    f"Strong {direction} {pair.method} correlation (r = {pair.coefficient:.3f}). "
                    f"Note: High correlation indicates linear or monotonic co-movement, not causal dependence."
                ),
                details={
                    "col1": pair.col1,
                    "col2": pair.col2,
                    "coefficient": pair.coefficient,
                },
            )
        )

    # Sort insights by severity: critical -> warning -> info
    order = {"critical": 0, "warning": 1, "info": 2}
    insights.sort(key=lambda x: order.get(x.severity, 3))

    return insights
