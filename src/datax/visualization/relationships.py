"""Interactive relationship and correlation visualizations using Plotly."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from datax.visualization.distributions import PLOT_THEME


def create_correlation_heatmap(
    matrix: dict[str, dict[str, float]],
    title: str = "Numerical Correlation Matrix",
) -> go.Figure:
    """Create an interactive heatmap from a correlation matrix dictionary."""
    fig = go.Figure()

    cols = list(matrix.keys())
    if not cols:
        fig.add_annotation(
            text="No correlation data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": "#94a3b8"},
        )
        fig.update_layout(
            paper_bgcolor=PLOT_THEME["paper_bgcolor"],
            plot_bgcolor=PLOT_THEME["plot_bgcolor"],
            font=PLOT_THEME["font"],
            height=300,
        )
        return fig

    z_values = [[matrix[r].get(c, 0.0) for c in cols] for r in cols]

    fig.add_trace(
        go.Heatmap(
            z=z_values,
            x=cols,
            y=cols,
            colorscale=[
                [0.0, "#0284c7"],    # Negative: blue
                [0.5, "#0f172a"],    # Neutral: dark
                [1.0, "#f43f5e"],    # Positive: crimson/rose
            ],
            zmin=-1.0,
            zmax=1.0,
            colorbar=dict(
                title=dict(text="r", font=dict(color="#f8fafc")),
                tickvals=[-1.0, -0.5, 0.0, 0.5, 1.0],
                tickfont=dict(color="#f8fafc"),
            ),
            hovertemplate="Feature X: %{x}<br>Feature Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>",
        )
    )

    size = max(380, min(750, 35 * len(cols) + 120))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#f8fafc")),
        margin=dict(l=80, r=40, t=50, b=80),
        width=size + 80,
        height=size,
        paper_bgcolor=PLOT_THEME["paper_bgcolor"],
        plot_bgcolor=PLOT_THEME["plot_bgcolor"],
        font=PLOT_THEME["font"],
        xaxis=dict(tickangle=-45, gridcolor=PLOT_THEME["gridcolor"]),
        yaxis=dict(gridcolor=PLOT_THEME["gridcolor"]),
    )
    return fig


def create_scatter_plot(
    df: pd.DataFrame,
    col_x: str,
    col_y: str,
    title: str | None = None,
    sample_size: int = 1500,
) -> go.Figure:
    """Create an interactive bivariate scatter plot with automatic sampling disclosure."""
    fig = go.Figure()

    if col_x not in df.columns or col_y not in df.columns:
        fig.add_annotation(
            text="Selected columns not found in dataset",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": "#94a3b8"},
        )
        return fig

    sub = df[[col_x, col_y]].dropna()
    is_sampled = len(sub) > sample_size
    plot_data = sub.sample(n=sample_size, random_state=42) if is_sampled else sub

    sample_note = f" (Sample of {sample_size:,} points)" if is_sampled else ""
    fig_title = title or f"{col_x} vs. {col_y}{sample_note}"

    fig.add_trace(
        go.Scatter(
            x=plot_data[col_x],
            y=plot_data[col_y],
            mode="markers",
            marker=dict(
                size=6,
                color="#38bdf8",
                opacity=0.6,
                line=dict(color="#0284c7", width=0.5),
            ),
            hovertemplate=f"{col_x}: %{{x}}<br>{col_y}: %{{y}}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=fig_title, font=dict(size=14, color="#f8fafc")),
        margin=dict(l=60, r=30, t=50, b=50),
        height=380,
        paper_bgcolor=PLOT_THEME["paper_bgcolor"],
        plot_bgcolor=PLOT_THEME["plot_bgcolor"],
        font=PLOT_THEME["font"],
        xaxis=dict(
            title=col_x,
            gridcolor=PLOT_THEME["gridcolor"],
            zerolinecolor=PLOT_THEME["gridcolor"],
        ),
        yaxis=dict(
            title=col_y,
            gridcolor=PLOT_THEME["gridcolor"],
            zerolinecolor=PLOT_THEME["gridcolor"],
        ),
        showlegend=False,
    )
    return fig


def create_cat_num_box(
    df: pd.DataFrame,
    cat_col: str,
    num_col: str,
    title: str | None = None,
    top_n: int = 8,
) -> go.Figure:
    """Create grouped box plots comparing a numerical variable across categorical segments."""
    fig = go.Figure()

    if cat_col not in df.columns or num_col not in df.columns:
        fig.add_annotation(
            text="Selected columns not found in dataset",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 14, "color": "#94a3b8"},
        )
        return fig

    sub = df[[cat_col, num_col]].dropna()
    top_categories = sub[cat_col].value_counts().iloc[:top_n].index
    filtered = sub[sub[cat_col].isin(top_categories)]

    for cat in top_categories:
        cat_data = filtered[filtered[cat_col] == cat][num_col]
        fig.add_trace(
            go.Box(
                y=cat_data,
                name=str(cat),
                boxpoints="outliers",
                marker=dict(size=4),
            )
        )

    fig_title = title or f"{num_col} by {cat_col}"
    fig.update_layout(
        title=dict(text=fig_title, font=dict(size=14, color="#f8fafc")),
        margin=dict(l=60, r=30, t=50, b=50),
        height=380,
        paper_bgcolor=PLOT_THEME["paper_bgcolor"],
        plot_bgcolor=PLOT_THEME["plot_bgcolor"],
        font=PLOT_THEME["font"],
        xaxis=dict(gridcolor=PLOT_THEME["gridcolor"]),
        yaxis=dict(title=num_col, gridcolor=PLOT_THEME["gridcolor"]),
        showlegend=False,
    )
    return fig
