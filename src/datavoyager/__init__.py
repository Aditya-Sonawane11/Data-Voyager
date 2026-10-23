"""DataVoyager: An offline-first Python EDA library with interactive reports and AI assistance."""

from __future__ import annotations

from datavoyager.config import Settings
from datavoyager.dataset import read_csv, read_dataframe, read_excel, read_json, read_parquet
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
    "read_parquet",
    "read_excel",
    "read_json",
    "read_dataframe",
    "Settings",
    "DatasetProfile",
    "ColumnType",
    "DataVoyagerError",
    "DatasetEmptyError",
    "ConfigurationError",
    "AnalysisError",
]
