"""Custom exceptions for DataVoyager."""

from __future__ import annotations


class DataVoyagerError(Exception):
    """Base exception for all DataVoyager errors."""



class DatasetEmptyError(DataVoyagerError):
    """Raised when an empty DataFrame or CSV is provided."""


class ConfigurationError(DataVoyagerError):
    """Raised when invalid configuration settings are provided."""


class AnalysisError(DataVoyagerError):
    """Raised when statistical profiling or analysis fails."""


class ReportGenerationError(DataVoyagerError):
    """Raised when HTML report construction fails."""


class AIAssistantError(DataVoyagerError):
    """Raised when an AI assistant query or provider fails."""
