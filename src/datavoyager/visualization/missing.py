"""Interactive missingness visualizations using Plotly."""

from __future__ import annotations

import plotly.graph_objects as go

from datavoyager.visualization.distributions import PLOT_THEME


def create_missing_bar(
    missing_ratios: dict[str, float],
    title: str = "Missing Values Rate by Column",
) -> go.Figure:
    """Create an interactive horizontal bar chart depicting missing percentages per column."""
    fig = go.Figure()

    # Filter to columns that actually have missing values or show all if few
    sorted_items = sorted(missing_ratios.items(), key=lambda x: x[1], reverse=True)
    with_missing = [item for item in sorted_items if item[1] > 0]
    display_items = with_missing if with_missing else sorted_items[:15]

    if not display_items:
        fig.add_annotation(
            text="No columns present",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": "#94a3b8"},
        )
        return fig

    # Invert for horizontal display
    cols = [item[0] for item in display_items[::-1]]
    ratios = [item[1] * 100.0 for item in display_items[::-1]]
    bar_colors = ["#f87171" if r >= 20.0 else "#fbbf24" if r > 0.0 else "#34d399" for r in ratios]

    fig.add_trace(
        go.Bar(
            x=ratios,
            y=cols,
            orientation="h",
            marker=dict(color=bar_colors),
            hovertemplate="<b>%{y}</b><br>Missing: %{x:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#f8fafc")),
        margin=dict(l=100, r=30, t=40, b=40),
        height=max(220, min(500, 25 * len(cols) + 70)),
        paper_bgcolor=PLOT_THEME["paper_bgcolor"],
        plot_bgcolor=PLOT_THEME["plot_bgcolor"],
        font=PLOT_THEME["font"],
        xaxis=dict(
            title="Missing (%)",
            range=[0, 100],
            gridcolor=PLOT_THEME["gridcolor"],
            zerolinecolor=PLOT_THEME["gridcolor"],
        ),
        yaxis=dict(gridcolor=PLOT_THEME["gridcolor"]),
        showlegend=False,
    )
    return fig
