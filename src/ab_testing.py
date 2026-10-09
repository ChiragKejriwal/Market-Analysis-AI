"""
Causal A/B Testing and Statistical Hypothesis Inference Module.
Performs Two-Sample Proportion Z-tests, Statistical Power estimations, and automated experiment evaluations.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
from statsmodels.stats.power import NormalIndPower


def compute_statistical_power(
    p_control: float,
    p_variant: float,
    n_control: int,
    n_variant: int,
    alpha: float = 0.05
) -> float:
    """
    Estimates two-sample proportion statistical power using Cohen's h effect size.
    """
    if n_control <= 0 or n_variant <= 0 or p_control <= 0 or p_variant <= 0:
        return 0.0

    p_c = max(1e-5, min(1.0 - 1e-5, p_control))
    p_v = max(1e-5, min(1.0 - 1e-5, p_variant))

    # Cohen's h = 2 * (arcsin(sqrt(p_v)) - arcsin(sqrt(p_c)))
    cohens_h = 2.0 * (np.arcsin(np.sqrt(p_v)) - np.arcsin(np.sqrt(p_c)))
    effect_size = abs(cohens_h)

    if effect_size < 1e-6:
        return float(alpha)

    ratio = float(n_variant) / float(n_control)
    try:
        power_calc = NormalIndPower()
        power = power_calc.power(
            effect_size=effect_size,
            nobs1=n_control,
            alpha=alpha,
            ratio=ratio,
            alternative="two-sided"
        )
        return float(min(1.0, max(0.0, power)))
    except Exception:
        return 0.50


def run_two_proportion_test(
    control_conversions: int,
    control_impressions: int,
    variant_conversions: int,
    variant_impressions: int,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Performs a two-sided Two-Proportion Z-Test comparing Control vs. Treatment variant.

    Computes:
    - Observed rates (p_control, p_variant)
    - Absolute lift (p_variant - p_control)
    - Relative percentage lift ((p_variant - p_control) / p_control * 100)
    - 95% Confidence Interval for the lift
    - Z-statistic and p-value
    - Significance verdict ('Statistically Significant Winner', 'Inconclusive', or 'Underperforming')
    - Statistical Power estimate at alpha=0.05

    Args:
        control_conversions: Number of successes in control group.
        control_impressions: Total sample size in control group.
        variant_conversions: Number of successes in treatment variant.
        variant_impressions: Total sample size in treatment variant.
        alpha: Significance threshold (default 0.05).

    Returns:
        Dict[str, Any]: Detailed hypothesis test and inference summary.
    """
    if control_impressions <= 0 or variant_impressions <= 0:
        return {"error": "Impression/visitor counts must be strictly positive."}

    p_control = control_conversions / control_impressions
    p_variant = variant_conversions / variant_impressions

    count = np.array([variant_conversions, control_conversions])
    nobs = np.array([variant_impressions, control_impressions])

    # Two-sided proportions z-test
    if variant_conversions == 0 and control_conversions == 0:
        z_stat = 0.0
        p_val = 1.0
    else:
        z_stat, p_val = proportions_ztest(count, nobs, alternative="two-sided")
        if np.isnan(z_stat):
            z_stat = 0.0
        if np.isnan(p_val):
            p_val = 1.0

    # Absolute and Relative Lift
    absolute_lift = (p_variant - p_control)
    relative_lift_pct = ((p_variant - p_control) / p_control * 100.0) if p_control > 0 else 0.0

    # Confidence Interval of difference (p_variant - p_control)
    ci_low, ci_high = confint_proportions_2indep(
        variant_conversions, variant_impressions, control_conversions, control_impressions,
        compare="diff", alpha=alpha
    )

    # Statistical Power Calculation
    stat_power = compute_statistical_power(p_control, p_variant, control_impressions, variant_impressions, alpha=alpha)

    is_significant = bool(p_val < alpha)

    # Significance Verdict
    if is_significant:
        if absolute_lift > 0:
            verdict = "Statistically Significant Winner"
        elif absolute_lift < 0:
            verdict = "Underperforming"
        else:
            verdict = "Inconclusive"
    else:
        verdict = "Inconclusive"

    return {
        "p_control": float(p_control),
        "p_variant": float(p_variant),
        "p_control_pct": round(float(p_control * 100.0), 3),
        "p_variant_pct": round(float(p_variant * 100.0), 3),
        "absolute_lift": float(absolute_lift),
        "absolute_lift_pct": round(float(absolute_lift * 100.0), 3),
        "relative_lift_pct": round(float(relative_lift_pct), 2),
        "confidence_interval_95_pct": (round(float(ci_low * 100.0), 3), round(float(ci_high * 100.0), 3)),
        "z_statistic": round(float(z_stat), 3),
        "p_value": round(float(p_val), 5),
        "is_significant": is_significant,
        "verdict": verdict,
        "statistical_power": round(float(stat_power), 3)
    }


def evaluate_campaign_experiments(
    df: pd.DataFrame,
    dimension: str = "Campaign_Type",
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Runs automated pairwise A/B statistical comparisons across Campaign_Types or Target_Audiences
    within each marketing channel to identify winning creatives and segments.

    Args:
        df: Enriched marketing campaign DataFrame.
        dimension: Column dimension to compare ('Campaign_Type' or 'Target_Audience').
        alpha: Significance level threshold (default 0.05).

    Returns:
        pd.DataFrame: Summary table of all pairwise experiments, lifts, p-values, and verdicts.
    """
    if df is None or df.empty or dimension not in df.columns or "Channel_Used" not in df.columns:
        return pd.DataFrame()

    results = []

    for channel, ch_group in df.groupby("Channel_Used"):
        unique_groups = ch_group[dimension].dropna().unique().tolist()
        if len(unique_groups) < 2:
            continue

        # Use the largest volume group as baseline control
        group_vols = ch_group.groupby(dimension)["Clicks"].sum()
        sorted_groups = group_vols.sort_values(ascending=False).index.tolist()
        control_group = sorted_groups[0]

        ctrl_data = ch_group[ch_group[dimension] == control_group]
        ctrl_clicks = int(ctrl_data["Clicks"].sum())
        ctrl_convs = int(ctrl_data["Conversions"].sum())

        if ctrl_clicks < 10:
            continue

        for treatment_group in sorted_groups[1:]:
            treat_data = ch_group[ch_group[dimension] == treatment_group]
            treat_clicks = int(treat_data["Clicks"].sum())
            treat_convs = int(treat_data["Conversions"].sum())

            if treat_clicks < 10:
                continue

            test_res = run_two_proportion_test(
                control_conversions=ctrl_convs,
                control_impressions=ctrl_clicks,
                variant_conversions=treat_convs,
                variant_impressions=treat_clicks,
                alpha=alpha
            )

            if "error" in test_res:
                continue

            results.append({
                "Channel": channel,
                "Comparison_Dimension": dimension,
                "Control_Group": str(control_group),
                "Treatment_Group": str(treatment_group),
                "Control_CVR_Pct": test_res["p_control_pct"],
                "Treatment_CVR_Pct": test_res["p_variant_pct"],
                "Relative_Lift_Pct": test_res["relative_lift_pct"],
                "P_Value": test_res["p_value"],
                "Z_Statistic": test_res["z_statistic"],
                "Verdict": test_res["verdict"],
                "Statistical_Power": test_res["statistical_power"]
            })

    res_df = pd.DataFrame(results)
    if not res_df.empty:
        return res_df.sort_values(by=["Channel", "Relative_Lift_Pct"], ascending=[True, False]).reset_index(drop=True)
    return res_df


# Alias for backward compatibility
run_proportion_ab_test = lambda conversions_a, visitors_a, conversions_b, visitors_b, alpha=0.05, metric_name="Conversion Rate": {
    "metric_name": metric_name,
    "variant_a_rate": run_two_proportion_test(conversions_a, visitors_a, conversions_b, visitors_b, alpha)["p_control_pct"],
    "variant_b_rate": run_two_proportion_test(conversions_a, visitors_a, conversions_b, visitors_b, alpha)["p_variant_pct"],
    "relative_lift_pct": run_two_proportion_test(conversions_a, visitors_a, conversions_b, visitors_b, alpha)["relative_lift_pct"],
    "absolute_diff_pct": run_two_proportion_test(conversions_a, visitors_a, conversions_b, visitors_b, alpha)["absolute_lift_pct"],
    "z_statistic": run_two_proportion_test(conversions_a, visitors_a, conversions_b, visitors_b, alpha)["z_statistic"],
    "p_value": run_two_proportion_test(conversions_a, visitors_a, conversions_b, visitors_b, alpha)["p_value"],
    "is_significant": run_two_proportion_test(conversions_a, visitors_a, conversions_b, visitors_b, alpha)["is_significant"],
    "confidence_interval_pct": run_two_proportion_test(conversions_a, visitors_a, conversions_b, visitors_b, alpha)["confidence_interval_95_pct"],
    "decision": f"✅ Statistically Significant Winner (Lift: +{run_two_proportion_test(conversions_a, visitors_a, conversions_b, visitors_b, alpha)['relative_lift_pct']}%)" if run_two_proportion_test(conversions_a, visitors_a, conversions_b, visitors_b, alpha)["verdict"] == "Statistically Significant Winner" else "Inconclusive Result",
    "outcome_status": run_two_proportion_test(conversions_a, visitors_a, conversions_b, visitors_b, alpha)["verdict"]
}


def compare_campaign_variants(
    df: pd.DataFrame,
    group_col: str,
    control_val: str,
    treatment_val: str,
    metric_type: str = "conversion"
) -> Dict[str, Any]:
    """Compatibility wrapper for comparing two specific variants."""
    if df is None or df.empty or group_col not in df.columns:
        return {"error": "Invalid dataset or grouping column."}

    df_a = df[df[group_col].astype(str) == str(control_val)]
    df_b = df[df[group_col].astype(str) == str(treatment_val)]

    if df_a.empty or df_b.empty:
        return {"error": f"One or both variant groups are empty: '{control_val}', '{treatment_val}'."}

    if metric_type == "conversion":
        return run_proportion_ab_test(
            conversions_a=int(df_a["Conversions"].sum()),
            visitors_a=int(df_a["Clicks"].sum()),
            conversions_b=int(df_b["Conversions"].sum()),
            visitors_b=int(df_b["Clicks"].sum()),
            metric_name="Conversion Rate (CVR)"
        )
    elif metric_type == "ctr":
        return run_proportion_ab_test(
            conversions_a=int(df_a["Clicks"].sum()),
            visitors_a=int(df_a["Impressions"].sum()),
            conversions_b=int(df_b["Clicks"].sum()),
            visitors_b=int(df_b["Impressions"].sum()),
            metric_name="Click-Through Rate (CTR)"
        )
    else:
        rpc_a = df_a["Revenue"] / np.maximum(df_a["Clicks"], 1)
        rpc_b = df_b["Revenue"] / np.maximum(df_b["Clicks"], 1)
        t_stat, p_val = stats.ttest_ind(rpc_b, rpc_a, equal_var=False)
        mean_a = float(rpc_a.mean())
        mean_b = float(rpc_b.mean())
        lift = ((mean_b - mean_a) / mean_a * 100.0) if mean_a > 0 else 0.0

        return {
            "metric_name": "Revenue Per Click (RPC $)",
            "variant_a_rate": mean_a,
            "variant_b_rate": mean_b,
            "relative_lift_pct": lift,
            "absolute_diff_pct": mean_b - mean_a,
            "z_statistic": float(t_stat),
            "p_value": float(p_val),
            "is_significant": bool(p_val < 0.05),
            "confidence_interval_pct": (0.0, 0.0),
            "decision": "Statistically Significant" if p_val < 0.05 else "Inconclusive",
            "outcome_status": "Statistically Significant Winner" if (p_val < 0.05 and lift > 0) else "Inconclusive"
        }
