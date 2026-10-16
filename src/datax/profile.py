"""Data structures and representations for Datax dataset profiles."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from datax.config import Settings


class ColumnType(str, Enum):
    """Categorization of column data types."""

    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    COMPLEX = "complex"
    OTHER = "other"


@dataclass(slots=True)
class OverviewProfile:
    """Summary of overall dataset dimensions, memory, and inferred types."""

    n_rows: int
    n_cols: int
    total_cells: int
    memory_bytes: int
    memory_formatted: str
    column_types: dict[str, str]
    type_counts: dict[str, int]
    is_sampled: bool = False
    sampled_rows: int | None = None


@dataclass(slots=True)
class NumericalStats:
    """Descriptive statistics for numerical columns."""

    count: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    p25: float | None = None
    median: float | None = None
    p75: float | None = None
    max: float | None = None
    skewness: float | None = None
    zeros_count: int = 0
    zeros_ratio: float = 0.0
    infinite_count: int = 0
    iqr: float | None = None
    outliers_count: int = 0
    outliers_ratio: float = 0.0


@dataclass(slots=True)
class CategoricalStats:
    """Summary metrics for categorical or string columns."""

    count: int
    unique_count: int
    unique_ratio: float
    top_frequencies: list[dict[str, Any]] = field(default_factory=list)
    mode: Any | None = None


@dataclass(slots=True)
class DatetimeStats:
    """Summary metrics for temporal/datetime columns."""

    count: int
    min_date: str | None = None
    max_date: str | None = None
    range_days: float | None = None
    unique_count: int = 0


@dataclass(slots=True)
class ColumnProfile:
    """Complete profile for a single column."""

    name: str
    inferred_type: ColumnType
    dtype: str
    total_count: int
    missing_count: int
    missing_ratio: float
    distinct_count: int
    distinct_ratio: float
    is_constant: bool = False
    is_quasi_constant: bool = False
    is_potential_id: bool = False
    numerical_stats: NumericalStats | None = None
    categorical_stats: CategoricalStats | None = None
    datetime_stats: DatetimeStats | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class MissingProfile:
    """Comprehensive missingness metrics across the dataset."""

    total_missing_cells: int
    overall_missing_ratio: float
    columns_with_missing: list[dict[str, Any]] = field(default_factory=list)
    missing_cells_by_column: dict[str, int] = field(default_factory=dict)
    missing_ratios_by_column: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class QualityProfile:
    """Data quality diagnostics, duplicate metrics, and anomalies."""

    duplicate_rows: int
    duplicate_ratio: float
    constant_columns: list[str] = field(default_factory=list)
    quasi_constant_columns: list[str] = field(default_factory=list)
    potential_id_columns: list[str] = field(default_factory=list)
    columns_with_outliers: dict[str, int] = field(default_factory=dict)
    columns_with_infinities: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class CorrelationPair:
    """A pair of columns exhibiting notable correlation."""

    col1: str
    col2: str
    coefficient: float
    abs_coefficient: float
    method: str


@dataclass(slots=True)
class CorrelationProfile:
    """Pairwise correlation matrix and flagged associations."""

    method: str
    matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    strong_correlations: list[CorrelationPair] = field(default_factory=list)
    analyzed_columns: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None


@dataclass(slots=True)
class Insight:
    """Automated finding or observation discovered by the engine."""

    severity: str  # 'info', 'warning', 'critical'
    category: str  # 'missing', 'duplicates', 'correlation', 'outliers', 'cardinality', 'general'
    title: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


class _ProfileJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder handling enums, paths, and floating point corner cases."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        return super().default(obj)


@dataclass(slots=True)
class DatasetProfile:
    """Root profile object encapsulating the full analysis of a dataset."""

    overview: OverviewProfile
    columns: dict[str, ColumnProfile]
    missing: MissingProfile
    quality: QualityProfile
    correlations: CorrelationProfile
    insights: list[Insight]
    config: Settings
    _df: Any = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to a nested Python dictionary."""
        d = asdict(self)
        d.pop("_df", None)
        return d

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        """Serialize profile to a JSON string or save to file."""
        data = self.to_dict()
        serialized = json.dumps(data, indent=indent, cls=_ProfileJSONEncoder)
        if path is not None:
            out_path = Path(path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(serialized, encoding="utf-8")
        return serialized

    def summary(self) -> str:
        """Return a formatted terminal text summary of the dataset."""
        ov = self.overview
        ql = self.quality
        ms = self.missing

        lines = [
            f"=== Datax Profile: {self.config.title} ===",
            f"Dimensions     : {ov.n_rows:,} rows x {ov.n_cols:,} columns ({ov.total_cells:,} cells)",
            f"Memory Usage   : {ov.memory_formatted}",
            f"Column Types   : " + ", ".join(f"{k}: {v}" for k, v in ov.type_counts.items()),
            f"Duplicates     : {ql.duplicate_rows:,} rows ({ql.duplicate_ratio:.1%})",
            f"Missing Values : {ms.total_missing_cells:,} cells ({ms.overall_missing_ratio:.1%})",
        ]

        if ql.constant_columns:
            lines.append(f"Constant Cols  : {', '.join(ql.constant_columns)}")
        if ql.potential_id_columns:
            lines.append(f"Potential IDs  : {', '.join(ql.potential_id_columns)}")

        if self.correlations.strong_correlations:
            lines.append(
                f"Strong Corrs   : {len(self.correlations.strong_correlations)} pair(s) (|r| >= {self.config.correlation_threshold:.2f})"
            )
            for pair in self.correlations.strong_correlations[:3]:
                lines.append(f"  * {pair.col1} <-> {pair.col2}: r = {pair.coefficient:.3f}")

        if self.insights:
            lines.append(f"\nTop Insights ({len(self.insights)} total):")
            for insight in self.insights[:5]:
                prefix = {"critical": "[!]", "warning": "[*]", "info": "[i]"}.get(
                    insight.severity, "[-]"
                )
                lines.append(f"  {prefix} {insight.title}: {insight.message}")

        summary_text = "\n".join(lines)
        print(summary_text)
        return summary_text

    def to_html(self, path: str | Path | None = None) -> str:
        """Generate standalone HTML report."""
        from datax.report.builder import ReportBuilder

        builder = ReportBuilder(self, df=self._df)
        return builder.generate(output_path=path)

    def chat(self, mode: str = "offline", **kwargs: Any) -> Any:
        """Create an AI assistant session for conversational dataset exploration."""
        from datax.ai.assistant import DatasetAssistant

        return DatasetAssistant(profile=self, mode=mode, **kwargs)
