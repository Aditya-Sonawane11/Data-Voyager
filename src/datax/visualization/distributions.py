"""Interactive distribution visualizations using Plotly."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


# Consistent styling theme
PLOT_THEME = {
    "paper_bgcolor": "#1e293b",
    "plot_bgcolor": "#0f172a",
    "font": {"color": "#f8fafc", "family": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"},
    "gridcolor": "#334155",
}


def create_histogram(
    series: pd.Series,
    title: str | None = None,
    n_bins: int = 30,
) -> go.Figure:
    """Create an interactive histogram for a numerical series."""
    valid = series.dropna()
    col_name = str(series.name or "Value")
    fig_title = title or f"Distribution: {col_name}"

    fig = go.Figure()

    if valid.empty:
        fig.add_annotation(
            text="No valid non-null numerical data to display",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": "#94a3b8"},
        )
    else:
        fig.add_trace(
            go.Histogram(
                x=valid,
                nbinsx=n_bins,
                marker=dict(
                    color="#38bdf8",
                    line=dict(color="#0284c7", width=1),
                ),
                opacity=0.85,
                name=col_name,
                hovertemplate="Range: %{x}<br>Count: %{y:,}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=fig_title, font=dict(size=14, color="#f8fafc")),
        margin=dict(l=40, r=20, t=40, b=40),
        height=320,
        paper_bgcolor=PLOT_THEME["paper_bgcolor"],
        plot_bgcolor=PLOT_THEME["plot_bgcolor"],
        font=PLOT_THEME["font"],
        xaxis=dict(
            title=col_name,
            gridcolor=PLOT_THEME["gridcolor"],
            zerolinecolor=PLOT_THEME["gridcolor"],
        ),
        yaxis=dict(
            title="Frequency",
            gridcolor=PLOT_THEME["gridcolor"],
            zerolinecolor=PLOT_THEME["gridcolor"],
        ),
        showlegend=False,
    )
    return fig


def create_boxplot(
    series: pd.Series,
    title: str | None = None,
) -> go.Figure:
    """Create an interactive horizontal box plot for outlier and quartile inspection."""
    valid = series.dropna()
    col_name = str(series.name or "Value")
    fig_title = title or f"Boxplot: {col_name}"

    fig = go.Figure()

    if valid.empty:
        fig.add_annotation(
            text="No valid non-null numerical data to display",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": "#94a3b8"},
        )
    else:
        fig.add_trace(
            go.Box(
                x=valid,
                name=col_name,
                boxpoints="outliers",
                jitter=0.3,
                pointpos=-1.8,
                marker=dict(color="#f43f5e", size=5),
                line=dict(color="#38bdf8", width=1.5),
                fillcolor="rgba(56, 189, 248, 0.25)",
                hovertemplate="%{x}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=fig_title, font=dict(size=14, color="#f8fafc")),
        margin=dict(l=40, r=20, t=40, b=30),
        height=220,
        paper_bgcolor=PLOT_THEME["paper_bgcolor"],
        plot_bgcolor=PLOT_THEME["plot_bgcolor"],
        font=PLOT_THEME["font"],
        xaxis=dict(
            gridcolor=PLOT_THEME["gridcolor"],
            zerolinecolor=PLOT_THEME["gridcolor"],
        ),
        yaxis=dict(showticklabels=False),
        showlegend=False,
    )
    return fig


def create_category_bar(
    series: pd.Series,
    title: str | None = None,
    top_n: int = 15,
) -> go.Figure:
    """Create an interactive horizontal bar chart for categorical frequency distribution."""
    valid = series.dropna()
    col_name = str(series.name or "Category")
    fig_title = title or f"Frequencies: {col_name}"

    fig = go.Figure()

    if valid.empty:
        fig.add_annotation(
            text="No non-null categorical data to display",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": "#94a3b8"},
        )
    else:
        counts = valid.value_counts().iloc[:top_n]
        # Invert order for horizontal bar chart (top at top)
        categories = [str(c) for c in counts.index[::-1]]
        values = counts.values[::-1]
        total = len(valid)

        ratios = [v / total for v in values]
        custom_data = [f"{r:.1%}" for r in ratios]

        fig.add_trace(
            go.Bar(
                x=values,
                y=categories,
                orientation="h",
                marker=dict(
                    color="#818cf8",
                    line=dict(color="#6366f1", width=1),
                ),
                customdata=custom_data,
                hovertemplate="<b>%{y}</b><br>Count: %{x:,} (%{customdata})<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=fig_title, font=dict(size=14, color="#f8fafc")),
        margin=dict(l=100, r=20, t=40, b=40),
        height=max(260, min(500, 30 * (len(counts) if not valid.empty else 5) + 80)),
        paper_bgcolor=PLOT_THEME["paper_bgcolor"],
        plot_bgcolor=PLOT_THEME["plot_bgcolor"],
        font=PLOT_THEME["font"],
        xaxis=dict(
            title="Frequency",
            gridcolor=PLOT_THEME["gridcolor"],
            zerolinecolor=PLOT_THEME["gridcolor"],
        ),
        yaxis=dict(
            gridcolor=PLOT_THEME["gridcolor"],
        ),
        showlegend=False,
    )
    return fig
