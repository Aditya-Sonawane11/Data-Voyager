"""Unit tests for Plotly visualizations and interactive report generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import datax
from datax.config import Settings
from datax.visualization.distributions import (
    create_boxplot,
    create_category_bar,
    create_histogram,
)
from datax.visualization.missing import create_missing_bar
from datax.visualization.relationships import (
    create_cat_num_box,
    create_correlation_heatmap,
    create_scatter_plot,
)


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "num_a": np.random.randn(n),
        "num_b": np.random.randn(n) * 2 + 1,
        "cat_x": np.random.choice(["Red", "Green", "Blue"], size=n),
        "cat_y": np.random.choice(["Alpha", "Beta"], size=n),
    })


def test_create_histogram(sample_df):
    fig = create_histogram(sample_df["num_a"], title="Test Hist")
    assert fig is not None
    assert len(fig.data) == 1
    assert fig.data[0].type == "histogram"

    # Empty series
    empty_fig = create_histogram(pd.Series([], dtype=float))
    assert empty_fig is not None


def test_create_boxplot(sample_df):
    fig = create_boxplot(sample_df["num_a"], title="Test Box")
    assert fig is not None
    assert len(fig.data) == 1
    assert fig.data[0].type == "box"

    # Empty series
    empty_fig = create_boxplot(pd.Series([], dtype=float))
    assert empty_fig is not None


def test_create_category_bar(sample_df):
    fig = create_category_bar(sample_df["cat_x"], title="Test Cat Bar", top_n=5)
    assert fig is not None
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"

    # All NaNs
    nan_fig = create_category_bar(pd.Series([None, None]))
    assert nan_fig is not None


def test_create_missing_bar():
    ratios = {"col_a": 0.25, "col_b": 0.05, "col_c": 0.0}
    fig = create_missing_bar(ratios)
    assert fig is not None
    assert len(fig.data) == 1
    assert fig.data[0].type == "bar"

    empty_fig = create_missing_bar({})
    assert empty_fig is not None


def test_create_correlation_heatmap():
    matrix = {
        "x": {"x": 1.0, "y": 0.8},
        "y": {"x": 0.8, "y": 1.0},
    }
    fig = create_correlation_heatmap(matrix)
    assert fig is not None
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"

    empty_fig = create_correlation_heatmap({})
    assert empty_fig is not None


def test_create_scatter_plot(sample_df):
    fig = create_scatter_plot(sample_df, "num_a", "num_b")
    assert fig is not None
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"

    # Missing column
    err_fig = create_scatter_plot(sample_df, "num_a", "nonexistent")
    assert err_fig is not None


def test_create_cat_num_box(sample_df):
    fig = create_cat_num_box(sample_df, "cat_x", "num_a")
    assert fig is not None
    assert len(fig.data) == 3  # 3 categories


def test_report_builder_offline_html(tmp_path, sample_df):
    report = datax.analyze(sample_df, title="Visualization Integration Test")
    html_output = report.to_html(tmp_path / "report_offline.html")

    assert "<!DOCTYPE html>" in html_output
    assert "DATAX" in html_output
    assert "Visualization Integration Test" in html_output
    # Check that Plotly JS bundle is embedded offline
    assert "Plotly.newPlot" in html_output
    # Check that sections are present
    assert "tab-overview" in html_output
    assert "tab-distributions" in html_output
    assert "tab-relationships" in html_output
    assert "tab-quality" in html_output
    assert (tmp_path / "report_offline.html").exists()


def test_report_builder_online_mode(sample_df):
    cfg = Settings(offline_mode=False, title="CDN Report")
    report = datax.analyze(sample_df, config=cfg)
    html_output = report.to_html()

    assert "<!DOCTYPE html>" in html_output
    # Should link to CDN script instead of embedding 4MB script
    assert "cdn.plot.ly" in html_output
