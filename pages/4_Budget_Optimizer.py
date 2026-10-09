"""
Page 4: Prescriptive Budget Allocation Optimizer & Stochastic Risk Simulator
"""

import streamlit as st
import pandas as pd
from src.data_loader import load_marketing_data
from src.preprocessing import clean_and_enrich_data
from src.roi import fit_channel_response_curves
from src.optimizer import MarketingBudgetOptimizer
from src.simulation import BudgetRiskSimulator
from src.visualization import (
    plot_budget_optimization_comparison,
    plot_channel_reallocation_delta,
    plot_response_curves,
    plot_simulation_distribution
)

from src.ui_components import render_sidebar_header, ensure_page_data_loaded

st.set_page_config(
    page_title="Budget Optimizer | Marketing Analytics",
    page_icon="⚙️",
    layout="wide"
)

# Ensure dataset is cleanly loaded with animated loader
df, raw_df = ensure_page_data_loaded(page_title="Prescriptive Budget Optimizer")

# Render Sidebar Branding, Live Telemetry & CSV Ingestion
render_sidebar_header(active_page_title="Budget Optimizer")

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .kpi-card-opt {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
    }
    .kpi-card-opt:hover {
        border-color: rgba(52, 211, 153, 0.5);
    }
    .kpi-title {
        color: #94A3B8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }
    .kpi-val {
        color: #F8FAFC;
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚙️ Prescriptive Budget Optimizer & Risk Simulator")
st.markdown("""
Optimize marketing capital allocation across portfolio channels using constrained non-linear mathematical optimization 
(SciPy SLSQP / PuLP) and evaluate capital risk under uncertainty with 5,000-draw Monte Carlo simulations.
""")

@st.cache_resource(show_spinner="Fitting non-linear channel response curves...")
def get_cached_optimizer(df_records_hash: int, _dataframe: pd.DataFrame) -> MarketingBudgetOptimizer:
    return MarketingBudgetOptimizer().fit(_dataframe)

# Fit Optimizer & Learn Curves
optimizer = get_cached_optimizer(len(df), df)
curves = optimizer.response_curves
channels_list = list(curves.keys())

# Sidebar Optimization Controls
st.sidebar.header("🎯 Prescriptive Budget Controls")

solver_choice = st.sidebar.selectbox(
    "Mathematical Solver:",
    ["SciPy SLSQP (Continuous Non-Linear)", "PuLP (Piecewise Linear Approximation)"],
    index=0
)

current_total_spend = float(df["Spend"].sum()) if "Spend" in df.columns and df["Spend"].sum() > 0 else 100000.0

if current_total_spend > 1000000:
    min_budget = float(round(current_total_spend * 0.2, -4))
    max_budget = float(round(current_total_spend * 2.5, -4))
    step_val = float(round((max_budget - min_budget) / 100, -3))
else:
    min_budget = 10000.0
    max_budget = max(500000.0, float(round(current_total_spend * 2.5, -3)))
    step_val = 5000.0

total_budget = st.sidebar.slider(
    "Total Marketing Budget ($):",
    min_value=min_budget,
    max_value=max_budget,
    value=float(round(current_total_spend, -2)),
    step=max(1000.0, step_val)
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔒 Channel Spend Floors & Ceilings")

channel_bounds = {}
use_advanced_bounds = st.sidebar.checkbox("Customize Per-Channel Floors & Ceilings", value=False)

if use_advanced_bounds:
    for ch in channels_list:
        with st.sidebar.expander(f"⚙️ {ch} Bounds"):
            min_s = st.slider(f"Min Spend ({ch})", min_value=0.0, max_value=total_budget * 0.4, value=total_budget * 0.05, step=max(100.0, total_budget * 0.01), key=f"min_{ch}")
            max_s = st.slider(f"Max Spend ({ch})", min_value=min_s, max_value=total_budget * 0.8, value=total_budget * 0.45, step=max(100.0, total_budget * 0.01), key=f"max_{ch}")
            channel_bounds[ch] = (min_s, max_s)
else:
    min_pct = st.sidebar.slider("Global Min Channel Share (%)", min_value=1.0, max_value=20.0, value=5.0, step=0.5) / 100.0
    max_pct = st.sidebar.slider("Global Max Channel Share (%)", min_value=20.0, max_value=80.0, value=45.0, step=5.0) / 100.0
    for ch in channels_list:
        channel_bounds[ch] = (total_budget * min_pct, total_budget * max_pct)

# Solve Optimization
with st.spinner("Solving non-linear mathematical optimization program..."):
    if "SLSQP" in solver_choice:
        opt_result = optimizer.optimize_budget_continuous(
            total_budget=total_budget,
            channel_bounds=channel_bounds
        )
    else:
        opt_result = optimizer.optimize_budget_pulp(
            total_budget=total_budget,
            channel_bounds=channel_bounds
        )
        if "comparison_table" not in opt_result:
            opt_res_cont = optimizer.optimize_budget_continuous(total_budget=total_budget, channel_bounds=channel_bounds)
            opt_result["comparison_table"] = opt_res_cont["comparison_table"]
            opt_result["revenue_lift_vs_historical"] = opt_res_cont["revenue_lift_vs_historical"]
            opt_result["lift_pct_vs_historical"] = opt_res_cont["lift_pct_vs_historical"]
            opt_result["baseline1_historical_revenue"] = opt_res_cont["baseline1_historical_revenue"]
            opt_result["baseline2_equal_split_revenue"] = opt_res_cont["baseline2_equal_split_revenue"]
            opt_result["revenue_lift_vs_equal_split"] = opt_res_cont["revenue_lift_vs_equal_split"]
            opt_result["lift_pct_vs_equal_split"] = opt_res_cont["lift_pct_vs_equal_split"]

if "error" in opt_result:
    st.error(opt_result["error"])
else:
    comp_df = opt_result["comparison_table"]
    opt_rev = opt_result.get("projected_revenue", opt_result.get("optimized_revenue", 0.0))
    base1_rev = opt_result.get("baseline1_historical_revenue", opt_rev * 0.9)
    lift_hist = opt_result.get("revenue_lift_vs_historical", opt_rev - base1_rev)
    lift_hist_pct = opt_result.get("lift_pct_vs_historical", (lift_hist / max(base1_rev, 1) * 100))

    optimal_alloc_dict = dict(zip(comp_df["Channel"], comp_df["Optimized_Spend"]))
    hist_alloc_dict = dict(zip(comp_df["Channel"], comp_df["Current_Spend"]))

    # 1. Side-by-Side KPI Comparison: Current vs. Optimal Allocation
    st.subheader("🚀 Optimization Impact: Current vs. Optimal Allocation")

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="kpi-card-opt">
            <div class="kpi-title">Current / Baseline Revenue</div>
            <div class="kpi-val" style="color: #818CF8;">${base1_rev:,.0f}</div>
            <span style="color: #94A3B8; font-size: 0.8rem;">Historical allocation baseline</span>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card-opt">
            <div class="kpi-title">Optimized Projected Revenue</div>
            <div class="kpi-val" style="color: #34D399;">${opt_rev:,.0f}</div>
            <span style="color: #94A3B8; font-size: 0.8rem;">Non-linear response maximum</span>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi-card-opt">
            <div class="kpi-title">Projected Revenue Lift ($)</div>
            <div class="kpi-val" style="color: #38BDF8;">+${lift_hist:,.0f}</div>
            <span style="color: #94A3B8; font-size: 0.8rem;">Net incremental dollar gain</span>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card-opt">
            <div class="kpi-title">Relative Lift (%)</div>
            <div class="kpi-val" style="color: #F59E0B;">+{lift_hist_pct:.2f}%</div>
            <span style="color: #94A3B8; font-size: 0.8rem;">Portfolio ROAS expansion</span>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # 2. Channel Reallocation & Shifts
    st.subheader("📊 Capital Reallocation Schedule & Funded vs. Defunded Shifts")

    col_c1, col_c2 = st.columns([1, 1])

    with col_c1:
        fig_alloc = plot_budget_optimization_comparison(comp_df)
        st.plotly_chart(fig_alloc, use_container_width=True)

    with col_c2:
        fig_delta = plot_channel_reallocation_delta(comp_df)
        st.plotly_chart(fig_delta, use_container_width=True)

    # Reallocation Data Table & CSV Download
    display_tbl = comp_df[["Channel", "Current_Spend", "Optimized_Spend", "Spend_Delta", "Projected_Revenue", "Projected_ROAS"]]
    st.dataframe(
        display_tbl.style.format({
            "Current_Spend": "${:,.2f}",
            "Optimized_Spend": "${:,.2f}",
            "Spend_Delta": "${:+,.2f}",
            "Projected_Revenue": "${:,.2f}",
            "Projected_ROAS": "{:.2f}x"
        }),
        use_container_width=True
    )

    csv_data = comp_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Prescribed Budget Allocation Plan (CSV)",
        data=csv_data,
        file_name="prescribed_marketing_budget_plan.csv",
        mime="text/csv",
        type="primary"
    )

    st.markdown("---")

    # 3. Diminishing Returns Response Curves Plot
    st.subheader("📈 Channel Saturation Response Curves (Current ● vs. Optimal ★)")
    st.markdown("Mathematical diminishing returns curves ($R_i(S_i) = \\alpha_i \\ln(1 + \\beta_i S_i)$). Markers indicate where each channel lands.")

    fig_curves = plot_response_curves(
        response_dict=curves,
        current_spend=hist_alloc_dict,
        optimal_spend=optimal_alloc_dict,
        max_spend=total_budget * 0.65
    )
    st.plotly_chart(fig_curves, use_container_width=True)

    st.markdown("---")

    # 4. 5,000-Trial Monte Carlo Risk Simulation (VaR & CVaR)
    st.subheader("🎲 Monte Carlo Risk Simulator (5,000 Stochastic Iterations)")
    st.markdown("Empirical revenue distribution under channel execution uncertainty, with 95% Value at Risk (VaR), Conditional VaR (CVaR), and Baseline cutoffs.")

    col_sim_ctl, col_sim_view = st.columns([1, 2.5])

    with col_sim_ctl:
        sim_vol = st.slider("Market Volatility (σ):", min_value=0.05, max_value=0.30, value=0.12, step=0.01)
        sim_iters = st.selectbox("Simulation Trials:", [1000, 2500, 5000], index=2)

    risk_sim = BudgetRiskSimulator(response_curves=curves)
    sim_results = risk_sim.simulate_budget_allocation(
        allocation_dict=optimal_alloc_dict,
        n_simulations=sim_iters,
        volatility=sim_vol,
        baseline_allocation=hist_alloc_dict
    )

    with col_sim_view:
        fig_sim = plot_simulation_distribution(sim_results, baseline_revenue=base1_rev)
        st.plotly_chart(fig_sim, use_container_width=True)

    # Risk Metrics Summary
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Expected Mean Revenue", f"${sim_results['mean_revenue']:,.0f}")
    ci95_low, ci95_high = sim_results["ci_95"]
    r2.metric("95% Empirical CI", f"${ci95_low:,.0f} – ${ci95_high:,.0f}")
    r3.metric("95% Value at Risk (VaR)", f"-${sim_results['var_95']:,.0f}")
    r4.metric("95% Conditional VaR (CVaR)", f"-${sim_results['cvar_95']:,.0f}")

    st.markdown("---")

    # 5. Macroeconomic Budget Sensitivity Analysis
    st.subheader("🌐 Macroeconomic Budget Sensitivity Stress Test")
    st.markdown("Projected revenue and marginal returns across portfolio budget scaling scenarios (-20% to +20%).")

    sens_df = risk_sim.run_budget_sensitivity(
        base_budget=total_budget,
        allocation_strategy=optimal_alloc_dict,
        budget_steps=[-0.20, -0.10, 0.0, 0.10, 0.20]
    )

    st.dataframe(
        sens_df.style.format({
            "Budget_Multiplier": "{:.2f}x",
            "Total_Budget": "${:,.2f}",
            "Projected_Revenue": "${:,.2f}",
            "Projected_ROAS": "{:.2f}x",
            "Revenue_Delta_vs_Base": "${:+,.2f}",
            "Marginal_ROAS_on_Step": "{:.2f}x"
        }),
        use_container_width=True
    )
