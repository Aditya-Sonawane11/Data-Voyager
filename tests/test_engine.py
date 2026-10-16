"""Unit tests for the core DataVoyager profiling and analysis engine."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

import datavoyager
from datavoyager.exceptions import ConfigurationError, DatasetEmptyError
from datavoyager.profile import ColumnType


def test_analyze_basic_dataframe():
    """Verify basic analysis pipeline on standard mixed data."""
    df = pd.DataFrame({
        "id": range(1, 101),
        "age": np.random.RandomState(42).randint(18, 65, size=100),
        "salary": np.random.RandomState(42).normal(50000, 15000, size=100),
        "department": np.random.RandomState(42).choice(["Engineering", "Sales", "HR"], size=100),
        "is_active": np.random.RandomState(42).choice([True, False], size=100),
    })

    report = datavoyager.analyze(df, title="Employee Profile")

    assert report.overview.n_rows == 100
    assert report.overview.n_cols == 5
    assert report.overview.total_cells == 500
    assert "salary" in report.columns
    assert report.columns["salary"].inferred_type == ColumnType.NUMERICAL
    assert report.columns["department"].inferred_type == ColumnType.CATEGORICAL
    assert report.columns["is_active"].inferred_type == ColumnType.BOOLEAN

    # Salary numerical stats
    sal_stats = report.columns["salary"].numerical_stats
    assert sal_stats is not None
    assert sal_stats.count == 100
    assert sal_stats.mean is not None and sal_stats.mean > 0
    assert sal_stats.std is not None and sal_stats.std > 0

    # Potential ID check on 'id'
    assert report.columns["id"].is_potential_id is True
    assert "id" in report.quality.potential_id_columns


def test_dataframe_immutability():
    """Ensure that datavoyager.analyze never modifies the user's original DataFrame."""
    original_data = {
        "x": [1.0, 2.0, np.nan, 4.0],
        "y": ["a", "b", "b", "c"],
    }
    df = pd.DataFrame(original_data)
    df_copy = df.copy(deep=True)

    _ = datavoyager.analyze(df)

    # Check identical values, index, columns, and types
    pd.testing.assert_frame_equal(df, df_copy)


def test_empty_dataframe_raises():
    """Verify that empty DataFrames raise DatasetEmptyError."""
    empty_df = pd.DataFrame()
    with pytest.raises(DatasetEmptyError):
        datavoyager.analyze(empty_df)

    zero_rows_df = pd.DataFrame(columns=["a", "b", "c"])
    with pytest.raises(DatasetEmptyError):
        datavoyager.analyze(zero_rows_df)


def test_single_row_dataframe():
    """Ensure single-row datasets are processed cleanly without division errors."""
    df = pd.DataFrame({
        "num": [42.0],
        "cat": ["single_item"],
    })

    report = datavoyager.analyze(df)
    assert report.overview.n_rows == 1
    assert report.columns["num"].numerical_stats.count == 1
    assert report.columns["num"].numerical_stats.std == 0.0
    assert report.columns["cat"].categorical_stats.unique_count == 1


def test_all_missing_column():
    """Verify behavior when a column is entirely NaNs."""
    df = pd.DataFrame({
        "all_null": [np.nan, np.nan, np.nan, np.nan, np.nan],
        "valid_col": [1, 2, 3, 4, 5],
    })

    report = datavoyager.analyze(df)
    col_prof = report.columns["all_null"]
    assert col_prof.missing_count == 5
    assert col_prof.missing_ratio == 1.0
    assert any("missing" in w.lower() for w in col_prof.warnings)


def test_constant_and_quasi_constant_columns():
    """Verify detection of zero-variance constant and quasi-constant columns."""
    df = pd.DataFrame({
        "constant_num": [7] * 100,
        "constant_str": ["FIXED"] * 100,
        "quasi_constant": ["A"] * 99 + ["B"],
        "normal": range(100),
    })

    report = datavoyager.analyze(df)
    assert "constant_num" in report.quality.constant_columns
    assert "constant_str" in report.quality.constant_columns
    assert "quasi_constant" in report.quality.quasi_constant_columns
    assert "normal" not in report.quality.constant_columns


def test_infinite_and_non_finite_values():
    """Verify detection and isolation of infinite values."""
    df = pd.DataFrame({
        "with_inf": [1.0, 2.0, np.inf, -np.inf, 5.0, 6.0, 7.0, 8.0],
        "label": ["a", "b", "c", "d", "e", "f", "g", "h"],
    })

    report = datavoyager.analyze(df)
    inf_col = report.columns["with_inf"]
    assert inf_col.numerical_stats is not None
    assert inf_col.numerical_stats.infinite_count == 2
    # Mean should be calculated over finite values (1, 2, 5, 6, 7, 8) = 29/6 = 4.833
    assert math.isclose(inf_col.numerical_stats.mean, 29 / 6, rel_tol=1e-3)
    assert "with_inf" in report.quality.columns_with_infinities


def test_outlier_detection_iqr():
    """Verify that extreme outliers are caught by the IQR rule."""
    # Normal cluster with a single massive outlier
    values = [10.0, 11.0, 10.5, 9.8, 10.2, 10.1, 9.9, 10.3, 10.0, 1000.0]
    df = pd.DataFrame({"metric": values})

    report = datavoyager.analyze(df, iqr_multiplier=1.5)
    stats = report.columns["metric"].numerical_stats
    assert stats is not None
    assert stats.outliers_count == 1
    assert "metric" in report.quality.columns_with_outliers


def test_correlation_matrix_and_threshold():
    """Verify correlation calculation and strong pair identification."""
    x = np.linspace(1, 100, 100)
    y = 2.0 * x + 5.0  # Perfect positive correlation (r = 1.0)
    z = -3.0 * x + 1.0  # Perfect negative correlation (r = -1.0)
    w = np.random.RandomState(42).randn(100)  # Low correlation

    df = pd.DataFrame({"x": x, "y": y, "z": z, "w": w})
    report = datavoyager.analyze(df, correlation_threshold=0.8)

    strong_pairs = report.correlations.strong_correlations
    assert len(strong_pairs) >= 2

    # Check pair x and y
    xy_pair = next(
        (p for p in strong_pairs if (p.col1 == "x" and p.col2 == "y") or (p.col1 == "y" and p.col2 == "x")),
        None,
    )
    assert xy_pair is not None
    assert math.isclose(xy_pair.coefficient, 1.0, abs_tol=1e-3)


def test_invalid_settings():
    """Verify that invalid configuration raises ConfigurationError."""
    with pytest.raises(ConfigurationError):
        datavoyager.analyze(pd.DataFrame({"a": [1, 2]}), sample_size=-5)

    with pytest.raises(ConfigurationError):
        datavoyager.analyze(pd.DataFrame({"a": [1, 2]}), correlation_method="invalid_method")
