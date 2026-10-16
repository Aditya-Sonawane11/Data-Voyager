"""HTML report generator for DataVoyager profiles using Jinja2 and Plotly."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import jinja2
import pandas as pd
import plotly.io as pio
import plotly.offline

from datavoyager.exceptions import ReportGenerationError
from datavoyager.profile import ColumnType, DatasetProfile
from datavoyager.visualization.distributions import (
    create_boxplot,
    create_category_bar,
    create_histogram,
)
from datavoyager.visualization.missing import create_missing_bar
from datavoyager.visualization.relationships import (
    create_correlation_heatmap,
    create_scatter_plot,
)


class ReportBuilder:
    """Constructs self-contained, interactive HTML reports from DatasetProfiles."""

    def __init__(
        self,
        profile: DatasetProfile,
        df: pd.DataFrame | None = None,
    ) -> None:
        self.profile = profile
        self.df = df if df is not None else getattr(profile, "_df", None)

    def generate(self, output_path: str | Path | None = None) -> str:
        """Render the complete HTML document and optionally write to file."""
        try:
            rendered = self._render_template()
            if output_path is not None:
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(rendered, encoding="utf-8")
            return rendered
        except Exception as exc:
            raise ReportGenerationError(f"Failed to generate HTML report: {exc}") from exc

    def _render_template(self) -> str:
        p = self.profile
        cfg = p.config

        # 0. Load logo as base64 data URI for offline embedding
        logo_data_uri: str | None = None
        try:
            # Look for logo relative to the package assets folder or project root
            candidates = [
                Path(__file__).parent.parent.parent.parent / "assets" / "logo.png",
                Path(__file__).parent.parent.parent.parent.parent / "assets" / "logo.png",
            ]
            for logo_path in candidates:
                if logo_path.exists():
                    raw = logo_path.read_bytes()
                    b64 = base64.b64encode(raw).decode("ascii")
                    logo_data_uri = f"data:image/png;base64,{b64}"
                    break
        except Exception:
            logo_data_uri = None

        # 1. Plotly JS asset bundling
        plotly_js = plotly.offline.get_plotlyjs() if cfg.offline_mode else None

        # 2. Generate Missingness chart
        missing_chart_html = None
        try:
            if p.missing.missing_ratios_by_column:
                missing_fig = create_missing_bar(p.missing.missing_ratios_by_column)
                missing_chart_html = pio.to_html(missing_fig, full_html=False, include_plotlyjs=False)
        except Exception:
            missing_chart_html = None

        # 3. Generate Correlation Heatmap
        heatmap_chart_html = None
        try:
            if p.correlations.matrix and not p.correlations.skipped:
                heatmap_fig = create_correlation_heatmap(p.correlations.matrix)
                heatmap_chart_html = pio.to_html(heatmap_fig, full_html=False, include_plotlyjs=False)
        except Exception:
            heatmap_chart_html = None

        # 4. Generate Feature Distribution charts
        distribution_charts: list[str] = []
        scatter_charts: list[str] = []

        if self.df is not None:
            # Numerical charts
            num_cols = [
                c for c, prof in p.columns.items()
                if prof.inferred_type == ColumnType.NUMERICAL
            ][:10]  # Cap at top 10 to keep report fast & responsive

            for col in num_cols:
                try:
                    hist_fig = create_histogram(self.df[col])
                    box_fig = create_boxplot(self.df[col])
                    hist_html = pio.to_html(hist_fig, full_html=False, include_plotlyjs=False)
                    box_html = pio.to_html(box_fig, full_html=False, include_plotlyjs=False)
                    distribution_charts.append(f'<div class="charts-row">{hist_html}{box_html}</div>')
                except Exception:
                    pass

            # Categorical charts
            cat_cols = [
                c for c, prof in p.columns.items()
                if prof.inferred_type in (ColumnType.CATEGORICAL, ColumnType.BOOLEAN)
            ][:8]  # Cap at top 8

            for col in cat_cols:
                try:
                    bar_fig = create_category_bar(self.df[col], top_n=cfg.categorical_top_n)
                    bar_html = pio.to_html(bar_fig, full_html=False, include_plotlyjs=False)
                    distribution_charts.append(bar_html)
                except Exception:
                    pass

            # Scatter plots for top strong correlations
            for pair in p.correlations.strong_correlations[:4]:
                if pair.col1 in self.df.columns and pair.col2 in self.df.columns:
                    try:
                        sc_fig = create_scatter_plot(self.df, pair.col1, pair.col2)
                        sc_html = pio.to_html(sc_fig, full_html=False, include_plotlyjs=False)
                        scatter_charts.append(sc_html)
                    except Exception:
                        pass

        # 5. Load Jinja2 Template
        template_dir = Path(__file__).parent / "templates"
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(["html", "xml"]),
        )

        # Template helper filters
        def format_num_filter(val: Any) -> str:
            try:
                return f"{int(val):,}"
            except Exception:
                return str(val)

        env.filters["format_num"] = format_num_filter

        template = env.get_template("report.html.j2")
        return template.render(
            profile=p,
            title=cfg.title,
            plotly_js=plotly_js,
            logo_data_uri=logo_data_uri,
            missing_chart_html=missing_chart_html,
            heatmap_chart_html=heatmap_chart_html,
            distribution_charts=distribution_charts,
            scatter_charts=scatter_charts,
        )
