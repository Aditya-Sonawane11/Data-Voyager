"""Core orchestration engine for running automated exploratory data analysis."""

from __future__ import annotations

import dataclasses
from typing import Callable

import pandas as pd

from datavoyager.config import Settings
from datavoyager.core.columns import analyze_columns
from datavoyager.core.correlations import analyze_correlations
from datavoyager.core.insights import generate_insights
from datavoyager.core.missing import analyze_missing
from datavoyager.core.overview import analyze_overview
from datavoyager.core.quality import analyze_quality
from datavoyager.exceptions import AnalysisError, DatasetEmptyError
from datavoyager.profile import DatasetProfile

_ANALYSIS_STEPS = [
    "Overview & type inference",
    "Column statistics",
    "Missingness analysis",
    "Data quality diagnostics",
    "Correlation analysis",
    "Insight generation",
]


def _make_progress_reporter(
    progress: bool,
    on_progress: Callable[[int, int, str], None] | None,
) -> Callable[[int, str], None]:
    """Return a callable that reports step progress, optionally via tqdm."""
    total = len(_ANALYSIS_STEPS)

    if on_progress is not None:
        def reporter(step: int, label: str) -> None:
            on_progress(step, total, label)
        return reporter

    if progress:
        try:
            from tqdm import tqdm  # type: ignore[import]
            pbar = tqdm(total=total, desc="DataVoyager", unit="step", ncols=70)

            def reporter(step: int, label: str) -> None:  # noqa: F811
                pbar.set_postfix_str(label, refresh=True)
                pbar.update(1)
                if step == total:
                    pbar.close()
            return reporter
        except ImportError:
            # tqdm not installed — fall back to print
            def reporter(step: int, label: str) -> None:  # noqa: F811
                print(f"  [{step}/{total}] {label}")
            return reporter

    # No progress reporting
    return lambda step, label: None


def analyze(
    df: pd.DataFrame,
    config: Settings | None = None,
    title: str | None = None,
    progress: bool = False,
    on_progress: Callable[[int, int, str], None] | None = None,
    **kwargs: object,
) -> DatasetProfile:
    """Perform automated exploratory data analysis on a Pandas DataFrame.

    This function never mutates the original DataFrame or the provided ``config``.

    Args:
        df: Input Pandas DataFrame to analyze.
        config: Optional custom Settings configuration.
        title: Optional title override for the report.
        progress: If ``True``, display a tqdm progress bar during analysis.
            Requires ``tqdm`` to be installed (``pip install tqdm``). Falls back
            to simple print statements if tqdm is not available.
        on_progress: Optional callable ``(step, total, label) -> None`` invoked
            after each analysis phase completes. Overrides ``progress`` when set.
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
        # Make a safe copy so we never mutate the caller's Settings instance
        cfg = dataclasses.replace(config, **{k: v for k, v in kwargs.items() if hasattr(config, k)})  # type: ignore[arg-type]

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
        report = _make_progress_reporter(progress, on_progress)

        # 1. Overview & Type inference (performed on full df for accurate shape/memory)
        overview, inferred_types = analyze_overview(
            df=df,
            config=cfg,
            is_sampled=is_sampled,
            sampled_rows=sampled_rows,
        )
        report(1, _ANALYSIS_STEPS[0])

        # 2. Column-level statistics
        columns = analyze_columns(
            df=work_df,
            inferred_types=inferred_types,
            config=cfg,
        )
        report(2, _ANALYSIS_STEPS[1])

        # 3. Missingness analysis
        missing = analyze_missing(df=work_df)
        report(3, _ANALYSIS_STEPS[2])

        # 4. Data quality diagnostics
        quality = analyze_quality(
            df=work_df,
            column_profiles=columns,
        )
        report(4, _ANALYSIS_STEPS[3])

        # 5. Correlations
        correlations = analyze_correlations(
            df=work_df,
            column_profiles=columns,
            config=cfg,
        )
        report(5, _ANALYSIS_STEPS[4])

        # 6. Automated insights
        insights = generate_insights(
            overview=overview,
            columns=columns,
            missing=missing,
            quality=quality,
            correlations=correlations,
            config=cfg,
        )
        report(6, _ANALYSIS_STEPS[5])

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
