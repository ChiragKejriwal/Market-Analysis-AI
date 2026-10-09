"""
Unit tests for causal A/B testing and statistical hypothesis inference module.
Validates two-proportion z-test edge cases and compares outputs against scipy.stats / statsmodels.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
from src.ab_testing import (
    run_two_proportion_test,
    compute_statistical_power,
    evaluate_campaign_experiments,
    run_proportion_ab_test
)
from src.data_loader import generate_synthetic_marketing_data
from src.preprocessing import clean_and_enrich_data


def test_two_proportion_identical_rates():
    """
    Edge case: Identical conversion rates (e.g., 50/1,000 vs 50/1,000).
    Z-statistic must be 0, p-value = 1.0, relative lift = 0.0%, verdict = Inconclusive.
    """
    res = run_two_proportion_test(
        control_conversions=50,
        control_impressions=1000,
        variant_conversions=50,
        variant_impressions=1000,
        alpha=0.05
    )

    assert res["z_statistic"] == 0.0
    assert res["p_value"] == 1.0
    assert res["relative_lift_pct"] == 0.0
    assert res["is_significant"] is False
    assert res["verdict"] == "Inconclusive"
    # CI should be centered around 0
    ci_low, ci_high = res["confidence_interval_95_pct"]
    assert ci_low < 0 < ci_high


def test_two_proportion_zero_conversions():
    """
    Edge case: 0 conversions in control and/or treatment.
    """
    # 0 conversions in control, 50 in variant
    res = run_two_proportion_test(
        control_conversions=0,
        control_impressions=1000,
        variant_conversions=50,
        variant_impressions=1000,
        alpha=0.05
    )

    assert res["p_control_pct"] == 0.0
    assert res["p_variant_pct"] == 5.0
    assert res["p_value"] < 0.001
    assert res["is_significant"] is True
    assert res["verdict"] == "Statistically Significant Winner"

    # 0 conversions in both
    res_zero = run_two_proportion_test(
        control_conversions=0,
        control_impressions=1000,
        variant_conversions=0,
        variant_impressions=1000,
        alpha=0.05
    )
    assert res_zero["z_statistic"] == 0.0
    assert res_zero["p_value"] == 1.0
    assert res_zero["is_significant"] is False


def test_two_proportion_highly_significant_lift_scipy_crossval():
    """
    Verifies p-value, z-statistic, and confidence intervals against direct scipy / statsmodels calculations.
    Control: 200/10,000 (2.0%), Variant: 400/10,000 (4.0%).
    """
    n_ctrl, conv_ctrl = 10000, 200
    n_var, conv_var = 10000, 400

    res = run_two_proportion_test(
        control_conversions=conv_ctrl,
        control_impressions=n_ctrl,
        variant_conversions=conv_var,
        variant_impressions=n_var,
        alpha=0.05
    )

    # Direct statsmodels computation
    count = np.array([conv_var, conv_ctrl])
    nobs = np.array([n_var, n_ctrl])
    expected_z, expected_p = proportions_ztest(count, nobs, alternative="two-sided")

    assert pytest.approx(res["z_statistic"], rel=1e-2) == expected_z
    assert pytest.approx(res["p_value"], rel=1e-2) == expected_p
    assert res["is_significant"] is True
    assert res["verdict"] == "Statistically Significant Winner"
    assert res["relative_lift_pct"] == pytest.approx(100.0, rel=1e-2)  # +100% lift

    # Direct confidence interval cross-validation
    ci_low_direct, ci_high_direct = confint_proportions_2indep(
        conv_var, n_var, conv_ctrl, n_ctrl, compare="diff", alpha=0.05
    )
    assert pytest.approx(res["confidence_interval_95_pct"][0], abs=0.1) == ci_low_direct * 100.0
    assert pytest.approx(res["confidence_interval_95_pct"][1], abs=0.1) == ci_high_direct * 100.0


def test_underperforming_treatment():
    """Verifies that statistically significant negative lift is categorized as Underperforming."""
    res = run_two_proportion_test(
        control_conversions=600,
        control_impressions=10000,
        variant_conversions=300,
        variant_impressions=10000,
        alpha=0.05
    )
    assert res["is_significant"] is True
    assert res["relative_lift_pct"] < 0
    assert res["verdict"] == "Underperforming"


def test_evaluate_campaign_experiments():
    """Tests automated pairwise experiment ranking across campaign types."""
    raw_df = generate_synthetic_marketing_data(n_rows=600, seed=42)
    df = clean_and_enrich_data(raw_df)

    eval_df = evaluate_campaign_experiments(df, dimension="Campaign_Type", alpha=0.05)
    assert not eval_df.empty
    assert "Channel" in eval_df.columns
    assert "Control_Group" in eval_df.columns
    assert "Treatment_Group" in eval_df.columns
    assert "P_Value" in eval_df.columns
    assert "Verdict" in eval_df.columns
    assert "Statistical_Power" in eval_df.columns
