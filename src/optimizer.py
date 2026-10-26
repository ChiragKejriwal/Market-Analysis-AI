"""
Prescriptive Marketing Budget Allocation Optimizer Module.
Solves constrained non-linear (SciPy SLSQP) and discrete piecewise linear (PuLP)
mathematical programs to maximize multi-channel revenue under diminishing returns.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import pulp
from src.roi import fit_channel_response_curves, log_response_curve


class MarketingBudgetOptimizer:
    """
    Mathematical optimization engine for multi-channel marketing budget allocation.
    """

    def __init__(self, response_curves: Optional[Dict[str, Dict[str, float]]] = None):
        self.response_curves = response_curves or {}

    def fit(self, df: pd.DataFrame, target_metric: str = "Revenue") -> "MarketingBudgetOptimizer":
        """Fits response curves directly from an enriched marketing DataFrame."""
        self.response_curves = fit_channel_response_curves(df, target_metric=target_metric)
        return self

    def optimize_budget_continuous(
        self,
        total_budget: float,
        channel_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        response_curves: Optional[Dict[str, Dict[str, float]]] = None,
        historical_spend_dict: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Solves the non-linear continuous optimization problem using SciPy SLSQP.
        """
        curves = response_curves or self.response_curves
        if not curves or total_budget <= 0:
            return {"error": "Response curves not available or total budget is non-positive."}

        channels = sorted(list(curves.keys()))
        n = len(channels)

        bounds = []
        for c in channels:
            if channel_bounds and c in channel_bounds:
                b_min, b_max = channel_bounds[c]
            else:
                b_min = total_budget * 0.05
                b_max = total_budget * 0.50
            bounds.append((max(0.0, float(b_min)), float(b_max)))

        if historical_spend_dict:
            hist_raw = np.array([max(0.0, historical_spend_dict.get(c, 0.0)) for c in channels])
        else:
            hist_raw = np.array([curves[c].get("historical_spend", 1.0) for c in channels])
        
        hist_total = float(hist_raw.sum()) or 1.0
        baseline1_spend = (hist_raw / hist_total) * total_budget
        baseline2_spend = np.full(n, total_budget / n)

        x0 = np.clip(baseline1_spend, [b[0] for b in bounds], [b[1] for b in bounds])
        x0 = x0 * (total_budget / np.sum(x0))

        def objective(x):
            tot_rev = 0.0
            for i, c in enumerate(channels):
                alpha = curves[c]["alpha"]
                beta = curves[c]["beta"]
                tot_rev += log_response_curve(x[i], alpha, beta)
            return -tot_rev

        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - total_budget}]

        res = minimize(
            objective,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-8, "maxiter": 1000}
        )

        optimal_spend = res.x if res.success else x0
        optimal_spend = np.maximum(optimal_spend, 0.0)
        optimal_spend = optimal_spend * (total_budget / np.sum(optimal_spend))

        opt_revenue = float(-objective(optimal_spend))
        base1_revenue = float(-objective(baseline1_spend))
        base2_revenue = float(-objective(baseline2_spend))

        lift_vs_base1 = opt_revenue - base1_revenue
        lift_pct_vs_base1 = (lift_vs_base1 / base1_revenue * 100.0) if base1_revenue > 0 else 0.0

        lift_vs_base2 = opt_revenue - base2_revenue
        lift_pct_vs_base2 = (lift_vs_base2 / base2_revenue * 100.0) if base2_revenue > 0 else 0.0

        comparison_records = []
        for i, c in enumerate(channels):
            opt_s = float(optimal_spend[i])
            b1_s = float(baseline1_spend[i])
            b2_s = float(baseline2_spend[i])

            alpha = curves[c]["alpha"]
            beta = curves[c]["beta"]

            opt_r = float(log_response_curve(opt_s, alpha, beta))
            b1_r = float(log_response_curve(b1_s, alpha, beta))

            comparison_records.append({
                "Channel": c,
                "Current_Spend": round(b1_s, 2),
                "Current_Share_Pct": round(b1_s / total_budget * 100.0, 2),
                "Optimized_Spend": round(opt_s, 2),
                "Optimized_Share_Pct": round(opt_s / total_budget * 100.0, 2),
                "Spend_Delta": round(opt_s - b1_s, 2),
                "Equal_Split_Spend": round(b2_s, 2),
                "Current_Revenue": round(b1_r, 2),
                "Projected_Revenue": round(opt_r, 2),
                "Revenue_Delta": round(opt_r - b1_r, 2),
                "Projected_ROAS": round(opt_r / max(opt_s, 1.0), 2)
            })

        comparison_df = pd.DataFrame(comparison_records)
        optimal_dict = {c: round(float(optimal_spend[i]), 2) for i, c in enumerate(channels)}

        return {
            "success": bool(res.success),
            "solver": "SciPy SLSQP (Continuous Non-Linear)",
            "total_budget": total_budget,
            "optimal_allocation": optimal_dict,
            "channel_allocations": optimal_dict,
            "optimal_revenue": round(opt_revenue, 2),
            "optimized_revenue": round(opt_revenue, 2),
            "baseline1_historical_revenue": round(base1_revenue, 2),
            "baseline2_equal_split_revenue": round(base2_revenue, 2),
            "revenue_lift_vs_historical": round(lift_vs_base1, 2),
            "lift_pct_vs_historical": round(lift_pct_vs_base1, 2),
            "revenue_lift_vs_equal_split": round(lift_vs_base2, 2),
            "lift_pct_vs_equal_split": round(lift_pct_vs_base2, 2),
            "comparison_table": comparison_df,
            "channels": channels,
            "optimal_spends": optimal_spend.tolist(),
            "revenue_lift": round(lift_vs_base1, 2),
            "lift_pct": round(lift_pct_vs_base1, 2),
            "current_revenue": round(base1_revenue, 2)
        }

    def optimize_budget_pulp(
        self,
        total_budget: float,
        channel_candidates: Optional[List[str]] = None,
        channel_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        response_curves: Optional[Dict[str, Dict[str, float]]] = None,
        n_segments: int = 5
    ) -> Dict[str, Any]:
        """
        Solves a piecewise linear approximation of the diminishing returns budget allocation problem using PuLP.
        """
        curves = response_curves or self.response_curves
        if not curves or total_budget <= 0:
            return {"error": "Response curves not available or total budget invalid."}

        channels = sorted(channel_candidates) if channel_candidates else sorted(list(curves.keys()))

        prob = pulp.LpProblem("Marketing_Budget_Allocation_PuLP", pulp.LpMaximize)

        spend_vars = {}
        segment_vars = {}

        for c in channels:
            if channel_bounds and c in channel_bounds:
                b_min, b_max = channel_bounds[c]
            else:
                b_min = total_budget * 0.05
                b_max = total_budget * 0.50

            spend_vars[c] = pulp.LpVariable(f"Spend_{c}", lowBound=b_min, upBound=b_max, cat=pulp.LpContinuous)

            segment_width = (b_max - b_min) / n_segments
            segment_vars[c] = []

            for seg_idx in range(n_segments):
                s_start = b_min + seg_idx * segment_width
                s_end = s_start + segment_width
                alpha = curves[c]["alpha"]
                beta = curves[c]["beta"]

                r_start = log_response_curve(s_start, alpha, beta)
                r_end = log_response_curve(s_end, alpha, beta)
                marginal_slope = (r_end - r_start) / max(segment_width, 1e-5)

                var_seg = pulp.LpVariable(f"Seg_{c}_{seg_idx}", lowBound=0.0, upBound=segment_width, cat=pulp.LpContinuous)
                segment_vars[c].append((var_seg, marginal_slope, r_start))

        objective_terms = []
        for c in channels:
            for var_seg, slope, _ in segment_vars[c]:
                objective_terms.append(slope * var_seg)
        prob += pulp.lpSum(objective_terms)

        prob += pulp.lpSum([spend_vars[c] for c in channels]) <= total_budget

        for c in channels:
            b_min = channel_bounds[c][0] if (channel_bounds and c in channel_bounds) else (total_budget * 0.05)
            prob += spend_vars[c] == b_min + pulp.lpSum([var_seg for var_seg, _, _ in segment_vars[c]])

        solver = pulp.PULP_CBC_CMD(msg=False)
        status = prob.solve(solver)

        alloc_dict = {}
        tot_proj_rev = 0.0

        for c in channels:
            s_val = float(pulp.value(spend_vars[c]))
            alloc_dict[c] = round(s_val, 2)
            alpha = curves[c]["alpha"]
            beta = curves[c]["beta"]
            tot_proj_rev += log_response_curve(s_val, alpha, beta)

        return {
            "status": pulp.LpStatus[status],
            "solver": "PuLP Linear Approximation",
            "total_budget": total_budget,
            "optimal_allocation": alloc_dict,
            "projected_revenue": round(tot_proj_rev, 2)
        }


def optimize_budget_continuous(
    total_budget: float,
    channel_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    response_curves: Optional[Dict[str, Dict[str, float]]] = None,
    historical_spend_dict: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Continuous SLSQP budget optimizer helper function."""
    optimizer = MarketingBudgetOptimizer(response_curves=response_curves)
    return optimizer.optimize_budget_continuous(
        total_budget=total_budget,
        channel_bounds=channel_bounds,
        response_curves=response_curves,
        historical_spend_dict=historical_spend_dict
    )


def optimize_budget_pulp(
    total_budget: float,
    channel_candidates: Optional[List[str]] = None,
    channel_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    response_curves: Optional[Dict[str, Dict[str, float]]] = None
) -> Dict[str, Any]:
    """PuLP piecewise linear budget optimizer helper function."""
    optimizer = MarketingBudgetOptimizer(response_curves=response_curves)
    return optimizer.optimize_budget_pulp(
        total_budget=total_budget,
        channel_candidates=channel_candidates,
        channel_bounds=channel_bounds,
        response_curves=response_curves
    )


def optimize_budget_allocation(
    df: pd.DataFrame,
    total_budget: float,
    min_alloc_pct: float = 0.05,
    max_alloc_pct: float = 0.50,
    custom_bounds: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, Any]:
    """End-to-end DataFrame budget optimizer wrapper function."""
    if df is None or df.empty or total_budget <= 0:
        return {"error": "Invalid dataset or total budget."}

    curves = fit_channel_response_curves(df)
    hist_spends = dict(zip(df.groupby("Channel_Used")["Spend"].sum().index, df.groupby("Channel_Used")["Spend"].sum().values))

    bounds = {}
    for c in curves.keys():
        if custom_bounds and c in custom_bounds:
            bounds[c] = custom_bounds[c]
        else:
            bounds[c] = (total_budget * min_alloc_pct, total_budget * max_alloc_pct)

    return optimize_budget_continuous(
        total_budget=total_budget,
        channel_bounds=bounds,
        response_curves=curves,
        historical_spend_dict=hist_spends
    )
