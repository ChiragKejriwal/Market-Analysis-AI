"""
Interactive Visualization Engine using Plotly.
Modern, high-contrast, executive-ready charts built on a consistent dark theme (#0f172a)
with vibrant accents (#38bdf8, #818cf8, #34d399, #f59e0b, #f87171).
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Exact Theme Palette
DARK_BG = "#0f172a"
PAPER_BG = "rgba(15, 23, 42, 0.75)"
PLOT_BG = "rgba(15, 23, 42, 0.5)"
GRID_COLOR = "rgba(255, 255, 255, 0.08)"
FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, sans-serif"

VIBRANT_PALETTE = [
    "#38bdf8",  # Sky Blue
    "#818cf8",  # Indigo
    "#34d399",  # Emerald Green
    "#f59e0b",  # Amber Gold
    "#f87171",  # Coral Red
    "#a78bfa",  # Violet
    "#2dd4bf",  # Teal
    "#fb7185",  # Rose
]


def plot_executive_kpis(kpi_dict: Dict[str, Any]) -> go.Figure:
    """
    Renders styled executive indicator cards with delta tags.
    """
    fig = make_subplots(
        rows=1, cols=4,
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]],
        horizontal_spacing=0.05
    )

    fig.add_trace(
        go.Indicator(
            mode="number",
            value=kpi_dict.get("Total Spend", kpi_dict.get("total_spend", 0.0)),
            number={"prefix": "$", "valueformat": ",.0f", "font": {"color": "#38bdf8", "family": FONT_FAMILY}},
            title={"text": "<b>TOTAL SPEND</b>", "font": {"color": "#94a3b8", "size": 12, "family": FONT_FAMILY}}
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Indicator(
            mode="number",
            value=kpi_dict.get("Total Revenue", kpi_dict.get("total_revenue", 0.0)),
            number={"prefix": "$", "valueformat": ",.0f", "font": {"color": "#34d399", "family": FONT_FAMILY}},
            title={"text": "<b>TOTAL REVENUE</b>", "font": {"color": "#94a3b8", "size": 12, "family": FONT_FAMILY}}
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Indicator(
            mode="number",
            value=kpi_dict.get("Blended ROAS", kpi_dict.get("blended_roas", 0.0)),
            number={"suffix": "x", "valueformat": ".2f", "font": {"color": "#818cf8", "family": FONT_FAMILY}},
            title={"text": "<b>BLENDED ROAS</b>", "font": {"color": "#94a3b8", "size": 12, "family": FONT_FAMILY}}
        ),
        row=1, col=3
    )

    fig.add_trace(
        go.Indicator(
            mode="number",
            value=kpi_dict.get("Blended CAC", kpi_dict.get("blended_cpa", 0.0)),
            number={"prefix": "$", "valueformat": ".2f", "font": {"color": "#f59e0b", "family": FONT_FAMILY}},
            title={"text": "<b>BLENDED CAC</b>", "font": {"color": "#94a3b8", "size": 12, "family": FONT_FAMILY}}
        ),
        row=1, col=4
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        height=140
    )
    return fig


def plot_interactive_funnel(funnel_df: pd.DataFrame) -> go.Figure:
    """
    Renders a modern Plotly Funnel chart with gradient steps and drop-off annotations.
    """
    if funnel_df is None or funnel_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Funnel Data Available", template="plotly_dark")
        return fig

    gradient_colors = ["#38bdf8", "#818cf8", "#a78bfa", "#34d399"]

    fig = go.Figure(
        go.Funnel(
            y=funnel_df["Stage"],
            x=funnel_df["Count"],
            textinfo="value+percent initial+percent previous",
            textposition="inside",
            marker=dict(
                color=gradient_colors[:len(funnel_df)],
                line=dict(width=1.5, color="rgba(255, 255, 255, 0.4)")
            ),
            connector=dict(
                fillcolor="rgba(129, 140, 248, 0.15)",
                line=dict(color="rgba(255,255,255,0.2)", width=1)
            )
        )
    )

    fig.update_layout(
        title="<b>4-Stage Marketing Conversion Funnel & Stage Retention</b>",
        title_font=dict(size=16, color="#f8fafc", family=FONT_FAMILY),
        template="plotly_dark",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_FAMILY, color="#e2e8f0"),
        margin=dict(l=40, r=40, t=60, b=30)
    )
    return fig


def plot_cohort_heatmap(cohort_df: pd.DataFrame) -> go.Figure:
    """
    Renders a 2D heatmap of cohort conversion/retention over time.
    """
    if cohort_df is None or cohort_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Cohort Data Available", template="plotly_dark")
        return fig

    # If tabular DataFrame without pivot
    if "CohortMonth" in cohort_df.columns and "Period_Index" in cohort_df.columns:
        from src.cohort import generate_cohort_matrix
        matrix = generate_cohort_matrix(cohort_df, metric="retention")
    else:
        matrix = cohort_df.copy()

    if matrix.empty:
        fig = go.Figure()
        fig.update_layout(title="No Cohort Retention Data Available", template="plotly_dark")
        return fig

    # Build custom text with % symbol for non-null cells
    text_matrix = matrix.map(lambda v: f"{v:.1f}%" if pd.notnull(v) else "")

    fig = px.imshow(
        matrix,
        labels=dict(x="Elapsed Acquisition Months", y="Acquisition Cohort (Launch Month)", color="Retention Rate (%)"),
        x=matrix.columns,
        y=matrix.index,
        color_continuous_scale="Tealgrn",
        aspect="auto",
        title="<b>Monthly Acquisition Cohort Retention Matrix (M+0 to M+6)</b>"
    )

    fig.update_traces(
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(family=FONT_FAMILY, size=12, color="#FFFFFF"),
        hovertemplate="<b>Cohort:</b> %{y}<br><b>Period:</b> %{x}<br><b>Retention Rate:</b> %{z:.1f}%<extra></extra>"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_FAMILY, color="#e2e8f0"),
        margin=dict(l=40, r=40, t=60, b=40),
        coloraxis_colorbar=dict(
            title="Retention %",
            ticksuffix="%",
            thickness=16
        )
    )
    return fig


def plot_ab_test_ci(test_results: Union[Dict[str, Any], pd.DataFrame]) -> go.Figure:
    """
    Renders an executive-grade Forest Plot showing statistical effect size with 95% Confidence Interval error bars.
    """
    fig = go.Figure()

    if isinstance(test_results, dict):
        if "error" in test_results:
            fig.update_layout(title="A/B Test Incomplete", template="plotly_dark")
            return fig

        abs_lift = float(test_results.get("absolute_lift_pct", 0.0))
        rel_lift = float(test_results.get("relative_lift_pct", 0.0))
        ci = test_results.get("confidence_interval_95_pct", (abs_lift - 0.2, abs_lift + 0.2))
        ci_low, ci_high = float(ci[0]), float(ci[1])
        is_sig = bool(test_results.get("is_significant", False))
        p_val = float(test_results.get("p_value", 1.0))

        color = "#34D399" if (is_sig and abs_lift > 0) else ("#FB7185" if (is_sig and abs_lift < 0) else "#818CF8")

        # Forest plot marker with error bars
        fig.add_trace(
            go.Scatter(
                x=[abs_lift],
                y=["Treatment Effect (Δ CVR)"],
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[max(0.0001, ci_high - abs_lift)],
                    arrayminus=[max(0.0001, abs_lift - ci_low)],
                    color=color,
                    thickness=3.5,
                    width=12
                ),
                mode="markers",
                marker=dict(size=14, color=color, symbol="diamond", line=dict(color="#FFFFFF", width=1.5)),
                hovertemplate=(
                    f"<b>Absolute Rate Delta:</b> {abs_lift:+.3f}%<br>"
                    f"<b>Relative Lift:</b> {rel_lift:+.2f}%<br>"
                    f"<b>95% CI:</b> [{ci_low:+.3f}%, {ci_high:+.3f}%]<br>"
                    f"<b>p-value:</b> {p_val:.5f}<extra></extra>"
                ),
                showlegend=False
            )
        )

        # Compute dynamic x-axis bounds
        max_dist = max(abs(ci_low), abs(ci_high), abs(abs_lift), 0.1) * 1.6
        x_range = [-max_dist, max_dist]

    elif isinstance(test_results, pd.DataFrame) and not test_results.empty:
        # Multiple experiments forest plot
        for idx, row in test_results.iterrows():
            lift = float(row.get("Relative_Lift_Pct", 0.0))
            label = f"{row.get('Channel', '')} ({row.get('Treatment_Group', '')})"
            p_val = float(row.get("P_Value", 1.0))
            is_sig = p_val < 0.05
            color = "#34D399" if (is_sig and lift > 0) else ("#FB7185" if is_sig else "#818CF8")
            err = max(0.2, abs(lift) * 0.25)

            fig.add_trace(
                go.Scatter(
                    x=[lift],
                    y=[label],
                    error_x=dict(type="data", array=[err], color=color, thickness=2.5, width=8),
                    mode="markers",
                    marker=dict(size=10, color=color, symbol="diamond"),
                    name=label,
                    hovertemplate=f"<b>{label}</b><br>Lift: {lift:+.2f}%<br>p-value: {p_val:.4f}<extra></extra>"
                )
            )
        x_range = None
    else:
        x_range = None

    # Zero-Effect Reference Line
    fig.add_vline(
        x=0.0,
        line_width=2,
        line_dash="dash",
        line_color="#F59E0B",
        annotation_text="Zero Effect (H₀: No Lift)",
        annotation_position="top right",
        annotation_font=dict(color="#F59E0B", size=11, family=FONT_FAMILY)
    )

    fig.update_layout(
        title="<b>Statistical Effect Size & 95% Confidence Interval</b>",
        template="plotly_dark",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_FAMILY, color="#E2E8F0"),
        xaxis=dict(
            title="Absolute Conversion Rate Difference: Variant B − Variant A (% pts)",
            showgrid=True,
            gridcolor=GRID_COLOR,
            zeroline=False,
            range=x_range if x_range else None
        ),
        yaxis=dict(title="", showgrid=False),
        margin=dict(l=40, r=40, t=60, b=40),
        showlegend=False,
        height=320
    )
    return fig


def plot_response_curves(
    response_dict: Dict[str, Dict[str, float]],
    current_spend: Optional[Dict[str, float]] = None,
    optimal_spend: Optional[Dict[str, float]] = None,
    max_spend: Optional[float] = None
) -> go.Figure:
    """
    Plots diminishing returns response curves with distinct markers for Current vs. Optimal channel allocation.
    """
    fig = go.Figure()

    if not response_dict:
        fig.update_layout(title="No Response Curves Available", template="plotly_dark")
        return fig

    # Determine max spend span
    if max_spend is None:
        all_spends = []
        if current_spend:
            all_spends.extend(current_spend.values())
        if optimal_spend:
            all_spends.extend(optimal_spend.values())
        max_spend = max(all_spends) * 1.5 if all_spends else 100000.0

    spend_range = np.linspace(0, max_spend, 200)

    for idx, (channel, params) in enumerate(response_dict.items()):
        alpha = params["alpha"]
        beta = params["beta"]
        rev_curve = alpha * np.log1p(beta * spend_range)

        color = VIBRANT_PALETTE[idx % len(VIBRANT_PALETTE)]

        # Response curve line
        fig.add_trace(
            go.Scatter(
                x=spend_range,
                y=rev_curve,
                name=channel,
                mode="lines",
                line=dict(color=color, width=2.5)
            )
        )

        # Plot Current allocation point
        if current_spend and channel in current_spend:
            c_s = float(current_spend[channel])
            c_r = float(alpha * np.log1p(beta * c_s))
            fig.add_trace(
                go.Scatter(
                    x=[c_s],
                    y=[c_r],
                    mode="markers",
                    marker=dict(symbol="circle", size=9, color=color, line=dict(color="#ffffff", width=1.5)),
                    name=f"{channel} (Current)",
                    showlegend=False,
                    hoverinfo="text",
                    text=[f"<b>{channel} [Current]</b><br>Spend: ${c_s:,.0f}<br>Projected Rev: ${c_r:,.0f}"]
                )
            )

        # Plot Optimal allocation point
        if optimal_spend and channel in optimal_spend:
            o_s = float(optimal_spend[channel])
            o_r = float(alpha * np.log1p(beta * o_s))
            fig.add_trace(
                go.Scatter(
                    x=[o_s],
                    y=[o_r],
                    mode="markers",
                    marker=dict(symbol="star", size=13, color=color, line=dict(color="#ffffff", width=1.5)),
                    name=f"{channel} (Optimal)",
                    showlegend=False,
                    hoverinfo="text",
                    text=[f"<b>{channel} [Optimal ★]</b><br>Spend: ${o_s:,.0f}<br>Projected Rev: ${o_r:,.0f}"]
                )
            )

    fig.update_layout(
        title="<b>Channel Saturation Response Curves (Current ● vs. Optimal ★ Allocation)</b>",
        template="plotly_dark",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_FAMILY, color="#e2e8f0"),
        xaxis=dict(title="Marketing Spend ($)", showgrid=True, gridcolor=GRID_COLOR),
        yaxis=dict(title="Projected Revenue ($)", showgrid=True, gridcolor=GRID_COLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=30, r=30, t=60, b=30)
    )
    return fig


def plot_simulation_distribution(
    sim_results: Dict[str, Any],
    baseline_revenue: Optional[float] = None
) -> go.Figure:
    """
    Renders empirical Monte Carlo revenue distribution with VaR (95%), CVaR (95%),
    and Baseline cutoff vertical lines.
    """
    totals = sim_results.get("simulated_totals", np.array([]))
    if len(totals) == 0:
        fig = go.Figure()
        fig.update_layout(title="No Simulation Data Available", template="plotly_dark")
        return fig

    mean_rev = sim_results.get("mean_revenue", float(np.mean(totals)))
    p05 = sim_results.get("ci_90", (float(np.percentile(totals, 5)), float(np.percentile(totals, 95))))[0]
    p95 = sim_results.get("ci_90", (float(np.percentile(totals, 5)), float(np.percentile(totals, 95))))[1]
    var_95 = sim_results.get("var_95", max(0.0, mean_rev - p05))
    cvar_95 = sim_results.get("cvar_95", var_95)

    fig = go.Figure()

    # Empirical histogram
    fig.add_trace(
        go.Histogram(
            x=totals,
            name="Simulated Revenue Trajectories",
            marker_color="#818cf8",
            opacity=0.8,
            nbinsx=45
        )
    )

    # 1. Expected Mean Line
    fig.add_vline(
        x=mean_rev,
        line_width=2.5,
        line_color="#34d399",
        line_dash="solid",
        annotation_text=f"Expected Mean: ${mean_rev:,.0f}",
        annotation_position="top right",
        annotation_font=dict(color="#34d399", size=11, family=FONT_FAMILY)
    )

    # 2. VaR 95% Line (5th percentile)
    fig.add_vline(
        x=p05,
        line_width=2,
        line_color="#f87171",
        line_dash="dash",
        annotation_text=f"95% VaR: -${var_95:,.0f} (${p05:,.0f})",
        annotation_position="top left",
        annotation_font=dict(color="#f87171", size=10, family=FONT_FAMILY)
    )

    # 3. CVaR 95% Cutoff
    fig.add_vline(
        x=mean_rev - cvar_95,
        line_width=2,
        line_color="#f59e0b",
        line_dash="dot",
        annotation_text=f"95% CVaR: -${cvar_95:,.0f}",
        annotation_position="bottom left",
        annotation_font=dict(color="#f59e0b", size=10, family=FONT_FAMILY)
    )

    # 4. Optional Baseline Cutoff Line
    if baseline_revenue and baseline_revenue > 0:
        fig.add_vline(
            x=baseline_revenue,
            line_width=2,
            line_color="#38bdf8",
            line_dash="dashdot",
            annotation_text=f"Historical Baseline: ${baseline_revenue:,.0f}",
            annotation_position="bottom right",
            annotation_font=dict(color="#38bdf8", size=10, family=FONT_FAMILY)
        )

    fig.update_layout(
        title="<b>Monte Carlo Portfolio Revenue Risk Distribution (5,000 Iterations)</b>",
        title_font=dict(size=15, color="#f8fafc", family=FONT_FAMILY),
        template="plotly_dark",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_FAMILY, color="#e2e8f0"),
        xaxis=dict(title="Simulated Portfolio Revenue ($)", showgrid=True, gridcolor=GRID_COLOR),
        yaxis=dict(title="Trajectory Frequency", showgrid=True, gridcolor=GRID_COLOR),
        margin=dict(l=30, r=30, t=60, b=30),
        showlegend=False
    )
    return fig


def plot_revenue_spend_dual_axis(ts_df: pd.DataFrame) -> go.Figure:
    """Renders dual-axis interactive time-series for Spend and Revenue."""
    if ts_df is None or ts_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Time-Series Data", template="plotly_dark")
        return fig

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=ts_df["Date"],
            y=ts_df["Spend"],
            name="Marketing Spend ($)",
            marker_color="rgba(99, 102, 241, 0.65)",
            marker_line=dict(width=1, color="#818cf8")
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=ts_df["Date"],
            y=ts_df["Revenue"],
            name="Attributed Revenue ($)",
            mode="lines+markers",
            line=dict(color="#34d399", width=3),
            marker=dict(size=6, color="#34d399")
        ),
        secondary_y=True
    )

    fig.update_layout(
        title="<b>Marketing Spend vs. Generated Revenue Velocity (Dual-Axis)</b>",
        template="plotly_dark",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_FAMILY, color="#e2e8f0"),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor=GRID_COLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=30, r=30, t=60, b=30)
    )
    fig.update_yaxes(title_text="Marketing Spend ($)", showgrid=True, gridcolor=GRID_COLOR, secondary_y=False)
    fig.update_yaxes(title_text="Revenue ($)", showgrid=False, secondary_y=True)

    return fig


def plot_channel_share_donuts(channel_df: pd.DataFrame) -> go.Figure:
    """Renders side-by-side donut charts comparing Share of Spend vs. Share of Revenue."""
    if channel_df is None or channel_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Channel Data", template="plotly_dark")
        return fig

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=("<b>Share of Total Spend</b>", "<b>Share of Total Revenue</b>")
    )

    fig.add_trace(
        go.Pie(
            labels=channel_df["Channel_Used"],
            values=channel_df["Total_Spend"] if "Total_Spend" in channel_df.columns else channel_df["Spend"],
            hole=0.45,
            marker_colors=VIBRANT_PALETTE,
            textinfo="percent+label",
            name="Spend Share"
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Pie(
            labels=channel_df["Channel_Used"],
            values=channel_df["Total_Revenue"] if "Total_Revenue" in channel_df.columns else channel_df["Revenue"],
            hole=0.45,
            marker_colors=VIBRANT_PALETTE,
            textinfo="percent+label",
            name="Revenue Share"
        ),
        row=1, col=2
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PAPER_BG,
        font=dict(family=FONT_FAMILY, color="#e2e8f0"),
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    return fig


# Backward compatibility aliases
plot_revenue_spend_trend = plot_revenue_spend_dual_axis
plot_funnel_chart = plot_interactive_funnel
plot_diminishing_returns_curves = plot_response_curves
plot_monte_carlo_distribution = plot_simulation_distribution
plot_ab_test_distribution = plot_ab_test_ci


def plot_channel_roas_bar(channel_df: pd.DataFrame) -> go.Figure:
    """Renders grouped bar chart of Channel Spend, Revenue, and ROAS."""
    if channel_df is None or channel_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Channel Data Available", template="plotly_dark")
        return fig

    fig = go.Figure()

    spend_col = "Total_Spend" if "Total_Spend" in channel_df.columns else "Spend"
    rev_col = "Total_Revenue" if "Total_Revenue" in channel_df.columns else "Revenue"

    fig.add_trace(
        go.Bar(
            x=channel_df["Channel_Used"],
            y=channel_df[rev_col],
            name="Revenue ($)",
            marker_color="#34d399",
            text=[f"${v:,.0f}" for v in channel_df[rev_col]],
            textposition="auto"
        )
    )

    fig.add_trace(
        go.Bar(
            x=channel_df["Channel_Used"],
            y=channel_df[spend_col],
            name="Spend ($)",
            marker_color="#818cf8",
            text=[f"${v:,.0f}" for v in channel_df[spend_col]],
            textposition="auto"
        )
    )

    fig.update_layout(
        title="<b>Channel Revenue vs. Spend Breakdown</b>",
        barmode="group",
        template="plotly_dark",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_FAMILY, color="#e2e8f0"),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Amount ($)", showgrid=True, gridcolor=GRID_COLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=30, r=30, t=60, b=30)
    )
    return fig


def plot_budget_optimization_comparison(comparison_df: pd.DataFrame) -> go.Figure:
    """Plots side-by-side comparison of Current vs. Optimized channel allocations."""
    if comparison_df is None or comparison_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Allocation Data", template="plotly_dark")
        return fig

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=comparison_df["Channel"],
            y=comparison_df["Current_Spend"],
            name="Current Spend ($)",
            marker_color="#818cf8",
            text=[f"${v:,.0f}" for v in comparison_df["Current_Spend"]],
            textposition="auto"
        )
    )

    fig.add_trace(
        go.Bar(
            x=comparison_df["Channel"],
            y=comparison_df["Optimized_Spend"],
            name="Optimized Spend ($)",
            marker_color="#34d399",
            text=[f"${v:,.0f}" for v in comparison_df["Optimized_Spend"]],
            textposition="auto"
        )
    )

    fig.update_layout(
        title="<b>Current vs. Recommended Budget Allocation</b>",
        barmode="group",
        template="plotly_dark",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_FAMILY, color="#e2e8f0"),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Budget Allocated ($)", showgrid=True, gridcolor=GRID_COLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=30, r=30, t=60, b=30)
    )
    return fig


def plot_channel_reallocation_delta(comparison_df: pd.DataFrame) -> go.Figure:
    """
    Renders a bar chart of spend deltas showing which channels get funded (positive delta) 
    and which get defunded (negative delta).
    """
    if comparison_df is None or comparison_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No Allocation Data", template="plotly_dark")
        return fig

    deltas = comparison_df["Spend_Delta"] if "Spend_Delta" in comparison_df.columns else (comparison_df["Optimized_Spend"] - comparison_df["Current_Spend"])
    colors = ["#34d399" if d >= 0 else "#f87171" for d in deltas]

    fig = go.Figure(
        go.Bar(
            x=comparison_df["Channel"],
            y=deltas,
            marker_color=colors,
            text=[f"{'+' if d > 0 else ''}${d:,.0f}" for d in deltas],
            textposition="auto"
        )
    )

    fig.add_hline(y=0, line_width=1.5, line_color="rgba(255,255,255,0.4)")

    fig.update_layout(
        title="<b>Channel Capital Reallocation Shifts (Funded vs. Defunded)</b>",
        template="plotly_dark",
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family=FONT_FAMILY, color="#e2e8f0"),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Net Spend Delta ($)", showgrid=True, gridcolor=GRID_COLOR),
        margin=dict(l=30, r=30, t=60, b=30),
        showlegend=False
    )
    return fig

