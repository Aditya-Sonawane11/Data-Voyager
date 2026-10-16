"""Configuration settings for Datax profiling and analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from datax.exceptions import ConfigurationError


@dataclass(slots=True)
class Settings:
    """Configurable settings for Datax dataset analysis.

    Attributes:
        title: Title of the generated report or profile.
        sample_size: Maximum rows to sample for expensive computations (None to disable).
        random_state: Seed for reproducible sampling.
        iqr_multiplier: Multiplier for IQR outlier boundaries (typically 1.5).
        correlation_method: Correlation algorithm ('pearson' or 'spearman').
        correlation_threshold: Absolute coefficient to flag as strong correlation.
        max_correlation_columns: Upper limit on numerical columns before pairwise correlation
            is down-selected or skipped to prevent O(N^2) hangs.
        categorical_top_n: Number of most frequent categories to capture per column.
        quasi_constant_threshold: Dominant frequency ratio to flag as quasi-constant (e.g. 0.98).
        missing_warning_threshold: Missing ratio to flag as significant missing data (e.g. 0.10).
        offline_mode: Enforce offline operations without outbound network requests.
    """

    title: str = "Datax Profile Report"
    sample_size: int | None = None
    random_state: int = 42
    iqr_multiplier: float = 1.5
    correlation_method: Literal["pearson", "spearman"] = "pearson"
    correlation_threshold: float = 0.70
    max_correlation_columns: int = 50
    categorical_top_n: int = 10
    quasi_constant_threshold: float = 0.98
    missing_warning_threshold: float = 0.10
    offline_mode: bool = True

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.sample_size is not None and self.sample_size <= 0:
            raise ConfigurationError(f"sample_size must be positive, got {self.sample_size}")
        if self.iqr_multiplier <= 0:
            raise ConfigurationError(f"iqr_multiplier must be positive, got {self.iqr_multiplier}")
        if self.correlation_method not in ("pearson", "spearman"):
            raise ConfigurationError(
                f"correlation_method must be 'pearson' or 'spearman', got '{self.correlation_method}'"
            )
        if not (0.0 < self.correlation_threshold <= 1.0):
            raise ConfigurationError(
                f"correlation_threshold must be in (0.0, 1.0], got {self.correlation_threshold}"
            )
        if self.max_correlation_columns < 2:
            raise ConfigurationError(
                f"max_correlation_columns must be >= 2, got {self.max_correlation_columns}"
            )
        if self.categorical_top_n < 1:
            raise ConfigurationError(
                f"categorical_top_n must be >= 1, got {self.categorical_top_n}"
            )
        if not (0.5 <= self.quasi_constant_threshold <= 1.0):
            raise ConfigurationError(
                f"quasi_constant_threshold must be in [0.5, 1.0], got {self.quasi_constant_threshold}"
            )
        if not (0.0 <= self.missing_warning_threshold <= 1.0):
            raise ConfigurationError(
                f"missing_warning_threshold must be in [0.0, 1.0], got {self.missing_warning_threshold}"
            )
