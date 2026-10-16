"""Unit tests for report generation and serialization."""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import pytest

import datax


@pytest.fixture
def test_df():
    return pd.DataFrame({
        "id": range(1, 21),
        "name": [f"user_{i}" for i in range(1, 21)],
        "score": [10.5, 20.1, 15.2, 33.4, 18.9] * 4,
    })


def test_to_json_and_dict(tmp_path: Path, test_df: pd.DataFrame):
    report = datax.analyze(test_df, title="JSON Test")

    # To dict
    dict_repr = report.to_dict()
    assert isinstance(dict_repr, dict)
    assert dict_repr["overview"]["n_rows"] == 20
    assert dict_repr["overview"]["n_cols"] == 3

    # To JSON string
    json_str = report.to_json()
    parsed = json.loads(json_str)
    assert parsed["overview"]["n_rows"] == 20

    # To JSON file
    file_path = tmp_path / "profile.json"
    report.to_json(file_path)
    assert file_path.exists()
    content = json.loads(file_path.read_text(encoding="utf-8"))
    assert content["overview"]["n_cols"] == 3


def test_to_html_generation(tmp_path: Path, test_df: pd.DataFrame):
    report = datax.analyze(test_df, title="HTML Export Test")

    # Direct generate
    html_str = report.to_html()
    assert "<!DOCTYPE html>" in html_str
    assert "HTML Export Test" in html_str
    assert "Feature Profiles" in html_str

    # File export
    html_file = tmp_path / "report.html"
    report.to_html(html_file)
    assert html_file.exists()
    assert "<html" in html_file.read_text(encoding="utf-8")


def test_summary_output(test_df: pd.DataFrame):
    report = datax.analyze(test_df, title="Summary Test")
    summary = report.summary()
    assert "Dimensions" in summary
    assert "20 rows x 3 columns" in summary
    assert "Memory Usage" in summary
