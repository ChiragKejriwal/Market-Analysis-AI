"""
Unit tests for prescriptive budget optimization engine.
Ensures mathematical budget constraints are strictly respected and optimal revenue outperforms baselines.
"""

import numpy as np
import pandas as pd
import pytest
from src.optimizer import (
    MarketingBudgetOptimizer,
    optimize_budget_continuous,
    optimize_budget_pulp,
    optimize_budget_allocation
)
from src.data_loader import generate_synthetic_marketing_data
from src.preprocessing import clean_and_enrich_data


@pytest.fixture
def sample_marketing_df():
    raw = generate_synthetic_marketing_data(n_rows=600, seed=42)
    return clean_and_enrich_data(raw)


def test_budget_constraint_strictly_respected(sample_marketing_df):
    """
    Validates that:
    1. sum(S_i) <= Total_Budget (strictly respected).
    2. Optimal Revenue >= Baseline 1 (Historical Spend) Revenue.
    3. Optimal Revenue >= Baseline 2 (Equal Split) Revenue.
    """
    total_budget = 150000.0
    optimizer = MarketingBudgetOptimizer().fit(sample_marketing_df)

    res = optimizer.optimize_budget_continuous(total_budget=total_budget)

    assert res["success"] is True
    optimal_spends = res["optimal_allocation"]
    total_allocated = sum(optimal_spends.values())

    # 1. Budget Constraint: sum(spend) <= Total_Budget
    assert total_allocated <= total_budget + 1e-3
    assert np.isclose(total_allocated, total_budget, atol=1.0)

    # 2. Optimal Revenue >= Baselines
    opt_revenue = res["optimized_revenue"]
    base1_rev = res["baseline1_historical_revenue"]
    base2_rev = res["baseline2_equal_split_revenue"]

    assert opt_revenue >= base1_rev - 1e-3, f"Optimal {opt_revenue} must be >= Baseline1 {base1_rev}"
    assert opt_revenue >= base2_rev - 1e-3, f"Optimal {opt_revenue} must be >= Baseline2 {base2_rev}"
    assert res["revenue_lift_vs_historical"] >= 0.0


def test_channel_bounds_enforced(sample_marketing_df):
    """
    Validates that channel spend floors and ceilings are strictly respected for each channel.
    """
    total_budget = 100000.0
    optimizer = MarketingBudgetOptimizer().fit(sample_marketing_df)

    # Enforce strict custom bounds
    min_spend = 10000.0
    max_spend = 35000.0
    channel_bounds = {ch: (min_spend, max_spend) for ch in optimizer.response_curves.keys()}

    res = optimizer.optimize_budget_continuous(
        total_budget=total_budget,
        channel_bounds=channel_bounds
    )

    assert res["success"] is True
    for ch, spend in res["optimal_allocation"].items():
        assert spend >= min_spend - 1e-3, f"Channel {ch} spend {spend} violated lower bound {min_spend}"
        assert spend <= max_spend + 1e-3, f"Channel {ch} spend {spend} violated upper bound {max_spend}"


def test_pulp_discrete_optimizer_budget_constraint(sample_marketing_df):
    """Verifies that PuLP discrete piecewise linear optimization obeys total budget."""
    total_budget = 80000.0
    optimizer = MarketingBudgetOptimizer().fit(sample_marketing_df)
    res = optimizer.optimize_budget_pulp(total_budget=total_budget)

    assert res["status"] in ["Optimal", "1", "Not Solved"]
    total_pulp_spend = sum(res["optimal_allocation"].values())
    assert total_pulp_spend <= total_budget + 1e-2
    assert res["projected_revenue"] > 0


def test_optimize_budget_allocation_wrapper(sample_marketing_df):
    """Tests high-level wrapper API consistency."""
    res = optimize_budget_allocation(sample_marketing_df, total_budget=50000.0)
    assert res["success"] is True
    assert res["total_budget"] == 50000.0
    assert sum(res["optimal_allocation"].values()) <= 50000.01
