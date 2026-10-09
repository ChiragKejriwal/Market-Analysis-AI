"""
Stochastic Monte Carlo Risk Simulation and Budget Sensitivity Module.
Quantifies portfolio revenue distributions, Value at Risk (VaR 95%), Expected Shortfall (CVaR),
and macroeconomic budget sensitivity scenarios.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.roi import fit_channel_response_curves, log_response_curve


class BudgetRiskSimulator:
    """
    Stochastic risk simulator modeling uncertainty and macro volatility in marketing returns.
    """

    def __init__(self, response_curves: Optional[Dict[str, Dict[str, float]]] = None):
        self.response_curves = response_curves or {}

    def fit(self, df: pd.DataFrame) -> "BudgetRiskSimulator":
        """Fits response curves from marketing DataFrame."""
        self.response_curves = fit_channel_response_curves(df)
        return self

    def simulate_budget_allocation(
        self,
        allocation_dict: Dict[str, float],
        response_curves: Optional[Dict[str, Dict[str, float]]] = None,
        n_simulations: int = 5000,
        volatility: float = 0.12,
        random_seed: int = 42,
        baseline_allocation: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Simulates portfolio returns under stochastic lognormal market noise:
            Return_sim = Return(S) * exp(N(-sigma^2 / 2, sigma^2))

        Quantifies:
        - Expected total revenue & 95% Confidence Interval
        - 95% Value at Risk (VaR)
        - 95% Conditional VaR (CVaR / Expected Shortfall)
        - Probability of outperforming baseline allocation

        Args:
            allocation_dict: Dict of channel -> spend ($).
            response_curves: Dict of fitted channel curve parameters.
            n_simulations: Number of Monte Carlo draws (default 5,000).
            volatility: Volatility standard deviation factor sigma (default 0.12 or 12%).
            random_seed: Reproducibility seed.
            baseline_allocation: Optional dict of channel -> baseline spend ($) for win-rate computation.

        Returns:
            Dict[str, Any]: Simulated risk and probability metrics.
        """
        curves = response_curves or self.response_curves
        if not curves or not allocation_dict:
            return {"error": "Response curves or budget allocation missing."}

        rng = np.random.default_rng(random_seed)

        simulated_opt_totals = np.zeros(n_simulations)
        simulated_base_totals = np.zeros(n_simulations)
        channel_simulations: Dict[str, np.ndarray] = {}

        # Precompute baseline if not provided: equal split or historical
        if baseline_allocation is None:
            tot_spend = sum(allocation_dict.values())
            n_ch = len(curves)
            baseline_allocation = {c: tot_spend / n_ch for c in curves.keys()}

        for channel, spend in allocation_dict.items():
            if channel in curves and spend > 0:
                alpha = curves[channel]["alpha"]
                beta = curves[channel]["beta"]

                base_rev = log_response_curve(spend, alpha, beta)
                
                # Uncorrelated channel shocks with mean 1.0
                shocks = rng.lognormal(mean=-0.5 * (volatility ** 2), sigma=volatility, size=n_simulations)
                channel_rev = base_rev * shocks

                channel_simulations[channel] = channel_rev
                simulated_opt_totals += channel_rev

        # Simulate baseline portfolio under identical market conditions
        for channel, b_spend in baseline_allocation.items():
            if channel in curves and b_spend > 0:
                alpha = curves[channel]["alpha"]
                beta = curves[channel]["beta"]
                base_rev = log_response_curve(b_spend, alpha, beta)
                b_shocks = rng.lognormal(mean=-0.5 * (volatility ** 2), sigma=volatility, size=n_simulations)
                simulated_base_totals += base_rev * b_shocks

        mean_rev = float(np.mean(simulated_opt_totals))
        std_rev = float(np.std(simulated_opt_totals))
        median_rev = float(np.percentile(simulated_opt_totals, 50))

        # 95% Confidence Interval (2.5th to 97.5th percentile)
        p025 = float(np.percentile(simulated_opt_totals, 2.5))
        p975 = float(np.percentile(simulated_opt_totals, 97.5))

        # 90% Confidence Interval (5th to 95th percentile)
        p05 = float(np.percentile(simulated_opt_totals, 5.0))
        p95 = float(np.percentile(simulated_opt_totals, 95.0))

        # 95% Value at Risk (Max expected revenue shortfall at 5% tail relative to mean)
        var_95 = max(0.0, mean_rev - p05)

        # 95% Conditional VaR (Expected Shortfall - average loss beyond 5th percentile)
        tail_draws = simulated_opt_totals[simulated_opt_totals <= p05]
        cvar_95 = max(0.0, mean_rev - float(np.mean(tail_draws))) if len(tail_draws) > 0 else var_95

        # Probability of outperforming baseline spend allocation
        prob_outperforming = float(np.mean(simulated_opt_totals > simulated_base_totals) * 100.0)

        total_budget = sum(allocation_dict.values())
        expected_roas = (mean_rev / total_budget) if total_budget > 0 else 0.0

        return {
            "mean_revenue": round(mean_rev, 2),
            "median_revenue": round(median_rev, 2),
            "std_revenue": round(std_rev, 2),
            "ci_95": (round(p025, 2), round(p975, 2)),
            "ci_90": (round(p05, 2), round(p95, 2)),
            "var_95": round(var_95, 2),
            "cvar_95": round(cvar_95, 2),
            "prob_outperforming_baseline_pct": round(prob_outperforming, 2),
            "expected_roas": round(expected_roas, 2),
            "expected_roi": round(expected_roas - 1.0, 2),
            "simulated_totals": simulated_opt_totals,
            "channel_simulations": channel_simulations
        }

    def run_budget_sensitivity(
        self,
        base_budget: float,
        allocation_strategy: Dict[str, float],
        response_curves: Optional[Dict[str, Dict[str, float]]] = None,
        budget_steps: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Evaluates portfolio resilience to macroeconomic marketing budget cuts or expansions.

        Args:
            base_budget: Benchmark total budget ($).
            allocation_strategy: Dict of channel -> baseline spend ($).
            response_curves: Dict of fitted channel curve parameters.
            budget_steps: List of percentage adjustments (e.g. [-0.20, -0.10, 0.0, +0.10, +0.20]).

        Returns:
            pd.DataFrame: Budget scaling scenario table with projected revenue and incremental ROAS.
        """
        curves = response_curves or self.response_curves
        if not curves or not allocation_strategy:
            return pd.DataFrame()

        if budget_steps is None:
            budget_steps = [-0.20, -0.10, 0.0, +0.10, +0.20]

        total_alloc = sum(allocation_strategy.values()) or 1.0
        shares = {c: s / total_alloc for c, s in allocation_strategy.items()}

        scenarios = []

        # Baseline revenue at 0.0 delta
        base_rev = sum(
            log_response_curve(shares[c] * base_budget, curves[c]["alpha"], curves[c]["beta"])
            for c in shares if c in curves
        )

        for step in sorted(budget_steps):
            adj_budget = base_budget * (1.0 + step)
            step_revenue = sum(
                log_response_curve(shares[c] * adj_budget, curves[c]["alpha"], curves[c]["beta"])
                for c in shares if c in curves
            )

            rev_delta = step_revenue - base_rev
            spend_delta = adj_budget - base_budget
            marginal_roas = (rev_delta / spend_delta) if spend_delta != 0 else (step_revenue / adj_budget)

            scenarios.append({
                "Scenario": f"{step * 100:+.0f}% Budget" if step != 0 else "Baseline (0%)",
                "Budget_Multiplier": round(1.0 + step, 2),
                "Total_Budget": round(adj_budget, 2),
                "Projected_Revenue": round(step_revenue, 2),
                "Projected_ROAS": round(step_revenue / max(adj_budget, 1.0), 2),
                "Revenue_Delta_vs_Base": round(rev_delta, 2),
                "Marginal_ROAS_on_Step": round(marginal_roas, 2)
            })

        return pd.DataFrame(scenarios)


# Module-level functions
def simulate_budget_allocation(
    allocation_dict: Dict[str, float],
    response_curves: Dict[str, Dict[str, float]],
    n_simulations: int = 5000,
    volatility: float = 0.12,
    random_seed: int = 42,
    baseline_allocation: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """Helper function to execute Monte Carlo risk simulation."""
    sim = BudgetRiskSimulator(response_curves=response_curves)
    return sim.simulate_budget_allocation(
        allocation_dict=allocation_dict,
        response_curves=response_curves,
        n_simulations=n_simulations,
        volatility=volatility,
        random_seed=random_seed,
        baseline_allocation=baseline_allocation
    )


def run_budget_sensitivity(
    base_budget: float,
    allocation_strategy: Dict[str, float],
    response_curves: Dict[str, Dict[str, float]],
    budget_steps: Optional[List[float]] = None
) -> pd.DataFrame:
    """Helper function to evaluate budget scaling sensitivity."""
    sim = BudgetRiskSimulator(response_curves=response_curves)
    return sim.run_budget_sensitivity(
        base_budget=base_budget,
        allocation_strategy=allocation_strategy,
        response_curves=response_curves,
        budget_steps=budget_steps
    )


def run_monte_carlo_budget_simulation(
    df: pd.DataFrame,
    budget_allocations: Dict[str, float],
    n_simulations: int = 1000,
    volatility: float = 0.15,
    seed: int = 42
) -> Dict[str, Any]:
    """End-to-end DataFrame Monte Carlo simulation wrapper."""
    curves = fit_channel_response_curves(df)
    return simulate_budget_allocation(
        allocation_dict=budget_allocations,
        response_curves=curves,
        n_simulations=n_simulations,
        volatility=volatility,
        random_seed=seed
    )
