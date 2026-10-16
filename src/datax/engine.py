"""Core orchestration engine for running automated exploratory data analysis."""

from __future__ import annotations

import pandas as pd

from datax.config import Settings
from datax.core.columns import analyze_columns
from datax.core.correlations import analyze_correlations
from datax.core.insights import generate_insights
from datax.core.missing import analyze_missing
from datax.core.overview import analyze_overview
from datax.core.quality import analyze_quality
from datax.exceptions import AnalysisError, DatasetEmptyError
from datax.profile import DatasetProfile


def analyze(
    df: pd.DataFrame,
    config: Settings | None = None,
    title: str | None = None,
    **kwargs: object,
) -> DatasetProfile:
    """Perform automated exploratory data analysis on a Pandas DataFrame.

    This function never mutates the original DataFrame.

    Args:
        df: Input Pandas DataFrame to analyze.
        config: Optional custom Settings configuration.
        title: Optional title override for the report.
        **kwargs: Overrides for individual Settings parameters (e.g. sample_size=1000).

    Returns:
        DatasetProfile: Structured profile containing overview, column statistics,
            quality indicators, missingness, correlations, and automated insights.

    Raises:
        DatasetEmptyError: If the DataFrame contains no rows or columns.
        AnalysisError: If an unexpected error occurs during profiling.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a valid pandas.DataFrame.")

    if df.empty or len(df.columns) == 0:
        raise DatasetEmptyError("Cannot analyze an empty DataFrame (shape is 0 rows or 0 columns).")

    # Build and validate configuration
    if config is None:
        cfg = Settings(**kwargs)  # type: ignore[arg-type]
    else:
        cfg = config
        for k, v in kwargs.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

    if title is not None:
        cfg.title = title

    cfg.validate()

    # Determine if sampling applies
    is_sampled = False
    sampled_rows: int | None = None
    work_df = df

    if cfg.sample_size is not None and len(df) > cfg.sample_size:
        is_sampled = True
        sampled_rows = cfg.sample_size
        work_df = df.sample(n=cfg.sample_size, random_state=cfg.random_state)

    try:
        # 1. Overview & Type inference (performed on full df for accurate shape/memory)
        overview, inferred_types = analyze_overview(
            df=df,
            config=cfg,
            is_sampled=is_sampled,
            sampled_rows=sampled_rows,
        )

        # 2. Column-level statistics
        columns = analyze_columns(
            df=work_df,
            inferred_types=inferred_types,
            config=cfg,
        )

        # 3. Missingness analysis
        missing = analyze_missing(df=work_df)

        # 4. Data quality diagnostics
        quality = analyze_quality(
            df=work_df,
            column_profiles=columns,
        )

        # 5. Correlations
        correlations = analyze_correlations(
            df=work_df,
            column_profiles=columns,
            config=cfg,
        )

        # 6. Automated insights
        insights = generate_insights(
            overview=overview,
            columns=columns,
            missing=missing,
            quality=quality,
            correlations=correlations,
            config=cfg,
        )

        return DatasetProfile(
            overview=overview,
            columns=columns,
            missing=missing,
            quality=quality,
            correlations=correlations,
            insights=insights,
            config=cfg,
            _df=df,
        )

    except (DatasetEmptyError, TypeError):
        raise
    except Exception as exc:
        raise AnalysisError(f"Failed to profile dataset: {exc}") from exc
