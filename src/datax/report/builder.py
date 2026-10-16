"""HTML report generator for Datax profiles."""

from __future__ import annotations

import html
from pathlib import Path

from datax.exceptions import ReportGenerationError
from datax.profile import DatasetProfile


class ReportBuilder:
    """Constructs self-contained, offline-compatible HTML reports from DatasetProfiles."""

    def __init__(self, profile: DatasetProfile) -> None:
        self.profile = profile

    def generate(self, output_path: str | Path | None = None) -> str:
        """Render the complete HTML document and optionally write to file."""
        try:
            rendered = self._render_html()
            if output_path is not None:
                path = Path(output_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(rendered, encoding="utf-8")
            return rendered
        except Exception as exc:
            raise ReportGenerationError(f"Failed to generate HTML report: {exc}") from exc

    def _render_html(self) -> str:
        p = self.profile
        ov = p.overview
        ql = p.quality
        ms = p.missing
        cfg = p.config

        # Build Insights HTML
        insights_html = ""
        for ins in p.insights:
            sev_class = {
                "critical": "badge-critical",
                "warning": "badge-warning",
                "info": "badge-info",
            }.get(ins.severity, "badge-info")
            insights_html += f"""
            <div class="insight-card">
                <div class="insight-header">
                    <span class="badge {sev_class}">{html.escape(ins.severity.upper())}</span>
                    <span class="insight-category">{html.escape(ins.category.title())}</span>
                    <h3 class="insight-title">{html.escape(ins.title)}</h3>
                </div>
                <p class="insight-message">{html.escape(ins.message)}</p>
            </div>
            """

        # Build Column Rows HTML
        col_rows_html = ""
        for name, col in p.columns.items():
            num_desc = "-"
            if col.numerical_stats is not None:
                ns = col.numerical_stats
                mean_str = f"{ns.mean:.2f}" if ns.mean is not None else "-"
                std_str = f"{ns.std:.2f}" if ns.std is not None else "-"
                min_str = f"{ns.min:.2f}" if ns.min is not None else "-"
                max_str = f"{ns.max:.2f}" if ns.max is not None else "-"
                num_desc = f"Mean: {mean_str} | Std: {std_str} | Range: [{min_str}, {max_str}]"
            elif col.categorical_stats is not None:
                cs = col.categorical_stats
                num_desc = f"Mode: {html.escape(str(cs.mode))} ({cs.unique_count:,} unique)"

            warn_badge = ""
            if col.warnings:
                warn_badge = f'<span class="badge badge-warning" title="{html.escape("; ".join(col.warnings))}">{len(col.warnings)} Warn</span>'

            col_rows_html += f"""
            <tr>
                <td><strong>{html.escape(name)}</strong> {warn_badge}</td>
                <td><span class="type-pill">{html.escape(col.inferred_type.value)}</span></td>
                <td>{col.missing_count:,} ({col.missing_ratio:.1%})</td>
                <td>{col.distinct_count:,}</td>
                <td class="stats-col">{num_desc}</td>
            </tr>
            """

        # Build Correlations HTML
        corr_html = ""
        if p.correlations.strong_correlations:
            corr_html = '<div class="table-container"><table class="data-table"><thead><tr><th>Feature 1</th><th>Feature 2</th><th>Correlation (r)</th><th>Strength</th></tr></thead><tbody>'
            for pair in p.correlations.strong_correlations:
                corr_html += f"""
                <tr>
                    <td>{html.escape(pair.col1)}</td>
                    <td>{html.escape(pair.col2)}</td>
                    <td><strong>{pair.coefficient:.3f}</strong></td>
                    <td>{pair.abs_coefficient:.1%}</td>
                </tr>
                """
            corr_html += "</tbody></table></div>"
        else:
            reason = p.correlations.skip_reason or "No feature pairs exceeded correlation threshold."
            corr_html = f'<p class="empty-state">{html.escape(reason)}</p>'

        doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{html.escape(cfg.title)} - Datax Report</title>
    <style>
        :root {{
            --bg-color: #0f172a;
            --surface-color: #1e293b;
            --surface-hover: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-primary: #38bdf8;
            --accent-success: #34d399;
            --accent-warning: #fbbf24;
            --accent-critical: #f87171;
            --border-color: #334155;
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg-color);
            color: var(--text-primary);
            font-family: var(--font-family);
            line-height: 1.5;
            padding: 2rem 1.5rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1.5rem;
            margin-bottom: 2rem;
        }}
        .brand {{ display: flex; align-items: baseline; gap: 0.5rem; }}
        .brand h1 {{ font-size: 1.75rem; font-weight: 700; color: var(--accent-primary); }}
        .brand span {{ color: var(--text-secondary); font-size: 0.9rem; }}
        .meta-tag {{ background: var(--surface-color); padding: 0.35rem 0.75rem; border-radius: 6px; font-size: 0.85rem; border: 1px solid var(--border-color); }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .metric-card {{
            background: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1.25rem;
        }}
        .metric-title {{ font-size: 0.85rem; color: var(--text-secondary); text-transform: uppercase; margin-bottom: 0.5rem; }}
        .metric-value {{ font-size: 1.6rem; font-weight: 700; color: var(--text-primary); }}
        .metric-subtitle {{ font-size: 0.8rem; color: var(--text-secondary); margin-top: 0.25rem; }}
        
        section {{ margin-bottom: 2.5rem; }}
        h2 {{ font-size: 1.35rem; font-weight: 600; margin-bottom: 1rem; color: var(--text-primary); border-left: 4px solid var(--accent-primary); padding-left: 0.5rem; }}
        
        .insight-card {{
            background: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem 1.25rem;
            margin-bottom: 0.75rem;
        }}
        .insight-header {{ display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem; }}
        .insight-title {{ font-size: 1rem; font-weight: 600; }}
        .insight-category {{ font-size: 0.8rem; color: var(--text-secondary); }}
        .insight-message {{ color: var(--text-secondary); font-size: 0.92rem; }}
        
        .badge {{ padding: 0.2rem 0.5rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; }}
        .badge-critical {{ background: rgba(248, 113, 113, 0.2); color: var(--accent-critical); border: 1px solid var(--accent-critical); }}
        .badge-warning {{ background: rgba(251, 191, 36, 0.2); color: var(--accent-warning); border: 1px solid var(--accent-warning); }}
        .badge-info {{ background: rgba(56, 189, 248, 0.2); color: var(--accent-primary); border: 1px solid var(--accent-primary); }}
        .type-pill {{ background: #0f172a; padding: 0.2rem 0.45rem; border-radius: 4px; font-size: 0.8rem; color: var(--accent-primary); }}
        
        .table-container {{
            background: var(--surface-color);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow-x: auto;
        }}
        table.data-table {{ width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9rem; }}
        table.data-table th {{ background: #162032; padding: 0.85rem 1rem; color: var(--text-secondary); font-weight: 600; border-bottom: 1px solid var(--border-color); }}
        table.data-table td {{ padding: 0.85rem 1rem; border-bottom: 1px solid var(--border-color); }}
        table.data-table tr:last-child td {{ border-bottom: none; }}
        table.data-table tr:hover td {{ background: var(--surface-hover); }}
        .stats-col {{ color: var(--text-secondary); font-size: 0.85rem; }}
        .empty-state {{ color: var(--text-secondary); font-style: italic; padding: 1rem 0; }}
        
        footer {{
            border-top: 1px solid var(--border-color);
            padding-top: 1.5rem;
            color: var(--text-secondary);
            font-size: 0.85rem;
            display: flex;
            justify-content: space-between;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="brand">
                <h1>Datax</h1>
                <span>/ {html.escape(cfg.title)}</span>
            </div>
            <div class="meta-tag">Offline Standalone Report</div>
        </header>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">Observations</div>
                <div class="metric-value">{ov.n_rows:,}</div>
                <div class="metric-subtitle">{ov.n_cols:,} Columns</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Memory Footprint</div>
                <div class="metric-value">{ov.memory_formatted}</div>
                <div class="metric-subtitle">{ov.total_cells:,} Total Cells</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Missing Values</div>
                <div class="metric-value">{ms.total_missing_cells:,}</div>
                <div class="metric-subtitle">{ms.overall_missing_ratio:.1%} Missing Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Duplicate Rows</div>
                <div class="metric-value">{ql.duplicate_rows:,}</div>
                <div class="metric-subtitle">{ql.duplicate_ratio:.1%} Duplication Rate</div>
            </div>
        </div>

        <section>
            <h2>Automated Insights ({len(p.insights)})</h2>
            {insights_html if insights_html else '<p class="empty-state">No anomalous patterns identified.</p>'}
        </section>

        <section>
            <h2>Column Profile Summary</h2>
            <div class="table-container">
                <table class="data-table">
                    <thead>
                        <tr>
                            <th>Column Name</th>
                            <th>Inferred Type</th>
                            <th>Missing Count</th>
                            <th>Distinct Count</th>
                            <th>Summary Statistics</th>
                        </tr>
                    </thead>
                    <tbody>
                        {col_rows_html}
                    </tbody>
                </table>
            </div>
        </section>

        <section>
            <h2>Strong Correlations (|r| &ge; {cfg.correlation_threshold:.2f})</h2>
            {corr_html}
        </section>

        <footer>
            <span>Generated by Datax v0.1.0 (Offline-First EDA)</span>
            <span>Deterministic Ground-Truth Analytics</span>
        </footer>
    </div>
</body>
</html>"""
        return doc
