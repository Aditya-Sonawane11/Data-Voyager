"""Datax: An offline-first Python EDA library with interactive reports and AI assistance."""

from __future__ import annotations

from datax.config import Settings
from datax.dataset import read_csv
from datax.engine import analyze
from datax.exceptions import (
    AnalysisError,
    ConfigurationError,
    DataxError,
    DatasetEmptyError,
)
from datax.profile import ColumnType, DatasetProfile

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "analyze",
    "read_csv",
    "Settings",
    "DatasetProfile",
    "ColumnType",
    "DataxError",
    "DatasetEmptyError",
    "ConfigurationError",
    "AnalysisError",
]
