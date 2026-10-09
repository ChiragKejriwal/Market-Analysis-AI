"""
Unit tests for stochastic Monte Carlo risk simulation and sensitivity module.
"""

import pandas as pd
import numpy as np
import pytest
from src.simulation import (
    BudgetRiskSimulator,
    simulate_budget_allocation,
    run_budget_sensitivity,
    run_monte_carlo_budget_simulation
)
from src.roi import fit_channel_response_curves
from src.data_loader import generate_synthetic_marketing_data
from src.preprocessing import clean_and_enrich_data


@pytest.fixture
def simulation_fixture():
    raw = generate_synthetic_marketing_data(n_rows=600, seed=42)
    df = clean_and_enrich_data(raw)
    curves = fit_channel_response_curves(df)
    alloc = {c: 20000.0 for c in curves.keys()}
    return df, curves, alloc


def test_simulate_budget_allocation(simulation_fixture):
    df, curves, alloc = simulation_fixture
    sim = BudgetRiskSimulator(response_curves=curves)

    res = sim.simulate_budget_allocation(
        allocation_dict=alloc,
        n_simulations=500,
        volatility=0.12,
        random_seed=42
    )

    assert "mean_revenue" in res
    assert "ci_95" in res
    assert "var_95" in res
    assert "cvar_95" in res
    assert "prob_outperforming_baseline_pct" in res
    assert res["mean_revenue"] > 0
    assert res["ci_95"][0] < res["mean_revenue"] < res["ci_95"][1]
    assert res["cvar_95"] >= res["var_95"] >= 0.0


def test_run_budget_sensitivity(simulation_fixture):
    df, curves, alloc = simulation_fixture
    sim = BudgetRiskSimulator(response_curves=curves)

    sens_df = sim.run_budget_sensitivity(
        base_budget=100000.0,
        allocation_strategy=alloc,
        budget_steps=[-0.20, -0.10, 0.0, 0.10, 0.20]
    )

    assert len(sens_df) == 5
    assert "Scenario" in sens_df.columns
    assert "Total_Budget" in sens_df.columns
    assert "Projected_Revenue" in sens_df.columns
    assert "Marginal_ROAS_on_Step" in sens_df.columns
    # Higher budget should yield higher projected revenue (monotonic increasing)
    assert sens_df["Projected_Revenue"].is_monotonic_increasing
