"""
Unit tests for visualization module (Plotly charting functions).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from src.visualization import (
    plot_executive_kpis,
    plot_interactive_funnel,
    plot_cohort_heatmap,
    plot_ab_test_ci,
    plot_response_curves,
    plot_simulation_distribution,
    plot_revenue_spend_dual_axis,
    plot_channel_share_donuts,
    plot_channel_roas_bar,
    plot_budget_optimization_comparison,
    plot_channel_reallocation_delta
)


def test_plot_executive_kpis():
    kpi_dict = {
        "Total Spend": 100000.0,
        "Total Revenue": 250000.0,
        "Blended ROAS": 2.5,
        "Blended CAC": 45.0
    }
    fig = plot_executive_kpis(kpi_dict)
    assert isinstance(fig, go.Figure)


def test_plot_interactive_funnel():
    funnel_df = pd.DataFrame([
        {"Stage": "1. Impressions", "Count": 100000},
        {"Stage": "2. Clicks", "Count": 5000},
        {"Stage": "3. Conversions", "Count": 250},
        {"Stage": "4. Retention", "Count": 100}
    ])
    fig = plot_interactive_funnel(funnel_df)
    assert isinstance(fig, go.Figure)


def test_plot_cohort_heatmap():
    matrix = pd.DataFrame(
        [[100.0, 45.0, 30.0], [100.0, 50.0, np.nan], [100.0, np.nan, np.nan]],
        index=["2024-01", "2024-02", "2024-03"],
        columns=["M+0", "M+1", "M+2"]
    )
    fig = plot_cohort_heatmap(matrix)
    assert isinstance(fig, go.Figure)


def test_plot_ab_test_ci():
    test_results = {
        "metric_name": "Conversion Rate",
        "relative_lift_pct": 12.5,
        "confidence_interval_95_pct": (2.1, 22.9),
        "is_significant": True,
        "p_value": 0.015,
        "z_statistic": 2.43
    }
    fig = plot_ab_test_ci(test_results)
    assert isinstance(fig, go.Figure)


def test_plot_response_curves():
    response_dict = {
        "Google Ads": {"alpha": 12000.0, "beta": 0.0003},
        "Facebook": {"alpha": 8500.0, "beta": 0.0005}
    }
    current_spend = {"Google Ads": 30000.0, "Facebook": 20000.0}
    optimal_spend = {"Google Ads": 35000.0, "Facebook": 15000.0}
    fig = plot_response_curves(response_dict, current_spend, optimal_spend)
    assert isinstance(fig, go.Figure)


def test_plot_simulation_distribution():
    sim_results = {
        "mean_revenue": 300000.0,
        "ci_90": (270000.0, 330000.0),
        "ci_95": (260000.0, 340000.0),
        "var_95": 40000.0,
        "cvar_95": 55000.0,
        "simulated_totals": np.random.normal(300000, 20000, 1000)
    }
    fig = plot_simulation_distribution(sim_results, baseline_revenue=280000.0)
    assert isinstance(fig, go.Figure)


def test_plot_revenue_spend_dual_axis():
    ts_df = pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=10, freq="W"),
        "Spend": [1000.0 * i for i in range(1, 11)],
        "Revenue": [2500.0 * i for i in range(1, 11)]
    })
    fig = plot_revenue_spend_dual_axis(ts_df)
    assert isinstance(fig, go.Figure)


def test_plot_channel_share_donuts():
    ch_df = pd.DataFrame({
        "Channel_Used": ["Google Ads", "Facebook", "Email"],
        "Spend": [5000, 3000, 2000],
        "Revenue": [15000, 7000, 8000]
    })
    fig = plot_channel_share_donuts(ch_df)
    assert isinstance(fig, go.Figure)


def test_plot_channel_reallocation_delta():
    comp_df = pd.DataFrame({
        "Channel": ["Google Ads", "Facebook"],
        "Current_Spend": [30000.0, 20000.0],
        "Optimized_Spend": [35000.0, 15000.0],
        "Spend_Delta": [5000.0, -5000.0]
    })
    fig = plot_channel_reallocation_delta(comp_df)
    assert isinstance(fig, go.Figure)
