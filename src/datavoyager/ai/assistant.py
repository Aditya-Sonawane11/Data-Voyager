"""Dataset AI assistant providing deterministic offline and LLM-assisted Q&A."""

from __future__ import annotations

import re
from typing import Any

from datavoyager.ai.tools import (
    tool_get_column_details,
    tool_get_duplicate_summary,
    tool_get_missing_summary,
    tool_get_outliers_summary,
    tool_get_overview,
    tool_get_strong_correlations,
)
from datavoyager.profile import DatasetProfile


class DatasetAssistant:
    """Conversational assistant for querying and interpreting dataset profiles.

    Supports Layer 1 deterministic Q&A without network access or LLM models,
    as well as optional local/cloud LLM backends.
    """

    def __init__(
        self,
        profile: DatasetProfile,
        mode: str = "offline",
        **kwargs: Any,
    ) -> None:
        self.profile = profile
        self.mode = mode
        self.kwargs = kwargs

    def ask(self, question: str) -> str:
        """Answer a natural language question about the profiled dataset."""
        if not question or not question.strip():
            return "Please provide a valid question about your dataset."

        q = question.strip().lower()

        # 1. Missingness Intent
        if any(w in q for w in ["missing", "null", "nan", "empty cells"]):
            data = tool_get_missing_summary(self.profile)
            if not data["top_missing_columns"]:
                return "Good news! There are 0 missing values across all columns in this dataset."

            lines = [
                f"The dataset has {data['total_missing_cells']:,} missing cells "
                f"({data['overall_missing_ratio']:.1%} overall missingness).",
                "Columns with the highest missing values:",
            ]
            for item in data["top_missing_columns"]:
                lines.append(
                    f"  - '{item['column']}': {item['missing_count']:,} missing ({item['missing_ratio']:.1%})"
                )
            return "\n".join(lines)

        # 2. Duplicate Rows Intent
        if any(w in q for w in ["duplicate", "repeated rows", "duplicated"]):
            data = tool_get_duplicate_summary(self.profile)
            if data["duplicate_rows"] == 0:
                return "There are no duplicate rows in this dataset (0 duplicates found)."
            return (
                f"There are {data['duplicate_rows']:,} duplicate rows in this dataset, "
                f"accounting for {data['duplicate_ratio']:.1%} of total rows."
            )

        # 3. Correlation Intent
        if any(w in q for w in ["correlat", "relation", "association"]):
            data = tool_get_strong_correlations(self.profile)
            if data.get("skipped"):
                return f"Correlation analysis was skipped: {data.get('skip_reason')}"
            if not data["pairs"]:
                return (
                    f"No pairs exceed the strong correlation threshold "
                    f"(|r| >= {self.profile.config.correlation_threshold:.2f})."
                )

            lines = [f"Strongest numerical correlations ({data['method']}):"]
            for p in data["pairs"]:
                direction = "positive" if p["coefficient"] > 0 else "negative"
                lines.append(
                    f"  - '{p['col1']}' & '{p['col2']}': r = {p['coefficient']:.3f} ({direction})"
                )
            lines.append(
                "\nNote: High correlation indicates monotonic association, not causal influence."
            )
            return "\n".join(lines)

        # 4. Outlier Intent
        if any(w in q for w in ["outlier", "anomaly", "extreme value"]):
            data = tool_get_outliers_summary(self.profile)
            outliers = data["columns_with_outliers"]
            if not outliers:
                return "No notable outliers detected under the 1.5x IQR rule."

            lines = ["Columns with detected outliers (1.5x IQR rule):"]
            for col, count in outliers.items():
                lines.append(f"  - '{col}': {count:,} outlier(s)")
            return "\n".join(lines)

        # 5. Specific Column / Feature Inquiry
        # Check if any column name is mentioned in the question
        found_col = None
        for col in self.profile.columns:
            # Word boundary search for column name
            pattern = r"\b" + re.escape(col.lower()) + r"\b"
            if re.search(pattern, q):
                found_col = col
                break

        if found_col is not None:
            details = tool_get_column_details(self.profile, found_col)
            if details:
                col_name = details["name"]
                col_type = details["type"]
                lines = [
                    f"Column '{col_name}' ({col_type}):",
                    f"  - Missing values: {details['missing_count']:,} ({details['missing_ratio']:.1%})",
                    f"  - Distinct values: {details['distinct_count']:,}",
                ]

                if "numerical" in details:
                    num = details["numerical"]
                    lines.extend([
                        f"  - Mean: {num['mean']:.3f} | Median: {num['median']:.3f} | Std: {num['std']:.3f}",
                        f"  - Range: [{num['min']:.3f}, {num['max']:.3f}]",
                    ])
                    if num.get("skewness") is not None:
                        lines.append(f"  - Skewness: {num['skewness']:.3f}")
                    if num.get("outliers_count", 0) > 0:
                        lines.append(
                            f"  - Outliers: {num['outliers_count']:,} ({num['outliers_ratio']:.1%})"
                        )
                elif "categorical" in details:
                    cat = details["categorical"]
                    lines.append(f"  - Mode: {cat['mode']}")
                    if cat.get("top_frequencies"):
                        lines.append("  - Top categories:")
                        for freq in cat["top_frequencies"][:3]:
                            lines.append(
                                f"      * {freq['value']}: {freq['count']:,} ({freq['ratio']:.1%})"
                            )

                if details.get("warnings"):
                    lines.append(f"  - Warnings: {'; '.join(details['warnings'])}")

                return "\n".join(lines)

        # 6. Overview / Dimensions Intent
        if any(w in q for w in ["overview", "dimension", "shape", "how many rows", "how many columns", "size"]):
            data = tool_get_overview(self.profile)
            return (
                f"The dataset consists of {data['rows']:,} rows and {data['columns']:,} columns "
                f"({data['total_cells']:,} total cells), occupying {data['memory_formatted']}.\n"
                f"Data types: " + ", ".join(f"{k}: {v}" for k, v in data["type_counts"].items())
            )

        # 7. Insights Intent
        if any(w in q for w in ["insight", "finding", "problem", "issue"]):
            if not self.profile.insights:
                return "No critical warnings or anomalies were flagged for this dataset."

            lines = [f"Top automated insights ({len(self.profile.insights)} total):"]
            for ins in self.profile.insights[:5]:
                lines.append(f"  [{ins.severity.upper()}] {ins.title}: {ins.message}")
            return "\n".join(lines)

        # Default fallback for Layer 1
        return (
            "I could not match your question to a deterministic analysis rule in offline mode.\n\n"
            "Try asking questions such as:\n"
            "  * 'Which columns have the most missing values?'\n"
            "  * 'How many duplicate rows exist?'\n"
            "  * 'Which numerical columns are most strongly correlated?'\n"
            "  * 'Give me an overview of the dataset.'\n"
            "  * 'Tell me about <column_name>.'\n"
            "  * 'What are the key insights?'"
        )
