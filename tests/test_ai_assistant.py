"""Unit tests for the Layer 1 deterministic AI Assistant."""

from __future__ import annotations

import pandas as pd
import pytest

import datax


@pytest.fixture
def sample_report():
    df = pd.DataFrame({
        "id": range(1, 21),
        "age": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 99],
        "department": ["IT"] * 10 + ["HR"] * 10,
        "score": [95.0, 88.0, None, 72.0, None, 85.0, 90.0, 77.0, None, 82.0] * 2,
    })
    return datax.analyze(df, title="Test Dataset")


def test_assistant_missing_question(sample_report):
    assistant = sample_report.chat(mode="offline")
    answer = assistant.ask("Which columns have the most missing values?")
    assert "score" in answer.lower()
    assert "missing" in answer.lower()


def test_assistant_duplicate_question(sample_report):
    assistant = sample_report.chat(mode="offline")
    answer = assistant.ask("How many duplicate rows exist?")
    assert "duplicate" in answer.lower()


def test_assistant_correlations(sample_report):
    assistant = sample_report.chat(mode="offline")
    answer = assistant.ask("Which columns are strongly correlated?")
    assert "correlation" in answer.lower() or "no pairs exceed" in answer.lower()


def test_assistant_column_inquiry(sample_report):
    assistant = sample_report.chat(mode="offline")
    answer = assistant.ask("Tell me about the age column.")
    assert "age" in answer.lower()
    assert "mean" in answer.lower()
    assert "median" in answer.lower()


def test_assistant_overview(sample_report):
    assistant = sample_report.chat(mode="offline")
    answer = assistant.ask("Give me an overview of the dataset dimensions.")
    assert "rows" in answer.lower()
    assert "columns" in answer.lower()


def test_assistant_fallback(sample_report):
    assistant = sample_report.chat(mode="offline")
    answer = assistant.ask("What is the meaning of life?")
    assert "could not match your question" in answer.lower()
    assert "Try asking questions such as" in answer
