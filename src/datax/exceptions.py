"""Custom exceptions for Datax."""

from __future__ import annotations


class DataxError(Exception):
    """Base exception for all Datax errors."""


class DatasetEmptyError(DataxError):
    """Raised when an empty DataFrame or CSV is provided."""


class ConfigurationError(DataxError):
    """Raised when invalid configuration settings are provided."""


class AnalysisError(DataxError):
    """Raised when statistical profiling or analysis fails."""


class ReportGenerationError(DataxError):
    """Raised when HTML report construction fails."""


class AIAssistantError(DataxError):
    """Raised when an AI assistant query or provider fails."""
