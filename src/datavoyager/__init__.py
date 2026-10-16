"""DataVoyager: An offline-first Python EDA library with interactive reports and AI assistance."""

from __future__ import annotations

from datavoyager.config import Settings
from datavoyager.dataset import read_csv
from datavoyager.engine import analyze
from datavoyager.exceptions import (
    AnalysisError,
    ConfigurationError,
    DatasetEmptyError,
    DataVoyagerError,
)
from datavoyager.profile import ColumnType, DatasetProfile

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "analyze",
    "read_csv",
    "Settings",
    "DatasetProfile",
    "ColumnType",
    "DataVoyagerError",
    "DatasetEmptyError",
    "ConfigurationError",
    "AnalysisError",
]
