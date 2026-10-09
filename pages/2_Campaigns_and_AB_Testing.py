"""
Page 2: Campaign Drill-Down & Causal A/B Experimentation Playground
"""

import streamlit as st
import pandas as pd
from src.data_loader import load_marketing_data
from src.preprocessing import clean_and_enrich_data
from src.ab_testing import (
    run_two_proportion_test,
    run_proportion_ab_test,
    evaluate_campaign_experiments,
    compare_campaign_variants
)
from src.visualization import plot_ab_test_ci

from src.ui_components import render_sidebar_header, ensure_page_data_loaded

st.set_page_config(
    page_title="Campaigns & A/B Testing | Marketing Analytics",
    page_icon="🔬",
    layout="wide"
)

# Ensure dataset is cleanly loaded with animated loader
df, raw_df = ensure_page_data_loaded(page_title="Campaigns & A/B Testing Lab")

# Render Sidebar Branding, Live Telemetry & CSV Ingestion
render_sidebar_header(active_page_title="Campaigns & A/B Testing")

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .badge-win {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        background: rgba(52, 211, 153, 0.15);
        border: 1px solid #34D399;
        color: #34D399;
    }
    .badge-inconclusive {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        background: rgba(129, 140, 248, 0.15);
        border: 1px solid #818CF8;
        color: #818CF8;
    }
    .badge-loss {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        background: rgba(248, 113, 113, 0.15);
        border: 1px solid #F87171;
        color: #F87171;
    }
    .stat-card {
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔬 Campaign Explorer & Causal A/B Testing Playground")
st.markdown("Evaluate campaign performance granularly and test treatment effects with statistical significance testing.")

# 1. Campaign Drill-Down Table
st.subheader("🔍 Campaign Drill-Down & Benchmarking")

f_col1, f_col2, f_col3 = st.columns(3)

with f_col1:
    types = ["All"] + sorted(df["Campaign_Type"].dropna().unique().tolist()) if "Campaign_Type" in df.columns else ["All"]
    sel_type = st.selectbox("Filter Campaign Type:", types)

with f_col2:
    channels = ["All"] + sorted(df["Channel_Used"].dropna().unique().tolist()) if "Channel_Used" in df.columns else ["All"]
    sel_channel = st.selectbox("Filter Channel:", channels)

with f_col3:
    audiences = ["All"] + sorted(df["Target_Audience"].dropna().unique().tolist()) if "Target_Audience" in df.columns else ["All"]
    sel_audience = st.selectbox("Filter Target Audience:", audiences)

drill_df = df.copy()
if sel_type != "All":
    drill_df = drill_df[drill_df["Campaign_Type"] == sel_type]
if sel_channel != "All":
    drill_df = drill_df[drill_df["Channel_Used"] == sel_channel]
if sel_audience != "All" and "Target_Audience" in drill_df.columns:
    drill_df = drill_df[drill_df["Target_Audience"] == sel_audience]

# Sortable table with CTR, CPC, CPA, and ROAS using native high-performance column_config
display_drill_cols = [c for c in ["Campaign_ID", "Campaign_Type", "Channel_Used", "Target_Audience", "Spend", "Revenue", "Impressions", "Clicks", "Conversions", "CTR", "CPC", "CPA", "ROAS"] if c in drill_df.columns]

st.dataframe(
    drill_df[display_drill_cols].sort_values(by="ROAS", ascending=False),
    use_container_width=True,
    column_config={
        "Spend": st.column_config.NumberColumn("Spend", format="$%.2f"),
        "Revenue": st.column_config.NumberColumn("Revenue", format="$%.2f"),
        "Impressions": st.column_config.NumberColumn("Impressions", format="%d"),
        "Clicks": st.column_config.NumberColumn("Clicks", format="%d"),
        "Conversions": st.column_config.NumberColumn("Conversions", format="%d"),
        "CTR": st.column_config.NumberColumn("CTR", format="%.2f%%"),
        "CPC": st.column_config.NumberColumn("CPC", format="$%.2f"),
        "CPA": st.column_config.NumberColumn("CPA", format="$%.2f"),
        "ROAS": st.column_config.NumberColumn("ROAS", format="%.2fx")
    }
)

st.markdown("---")

# 2. Interactive A/B Experimentation Playground
st.subheader("🧪 Interactive A/B Experimentation Playground")
st.markdown("Rigorous two-proportion statistical significance testing to validate true incremental lift across creatives, audiences, and channels.")

# Mode & Dimension Selection
ctl_col1, ctl_col2, ctl_col3 = st.columns([1.5, 1.2, 1])

with ctl_col1:
    ab_mode = st.radio(
        "Experiment Mode:",
        ["Select from Active Campaigns / Channels", "Manual Custom Input Calculator"],
        horizontal=True
    )

if ab_mode == "Select from Active Campaigns / Channels":
    with ctl_col2:
        dim_choice = st.selectbox("Comparison Dimension:", ["Channel_Used", "Campaign_Type", "Target_Audience"])
        unique_vals = sorted(df[dim_choice].dropna().unique().tolist()) if dim_choice in df.columns else ["Default"]

    with ctl_col3:
        alpha_level = st.selectbox(
            "Confidence Level (1 - α):",
            [0.05, 0.01, 0.10],
            format_func=lambda a: f"{(1 - a) * 100:.0f}% Confidence (α={a})",
            index=0
        )

    # Variant Selection Cards
    v_col1, v_col2 = st.columns(2)

    with v_col1:
        st.markdown("""
        <div class="ab-variant-card" style="border-top: 3px solid #38BDF8;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 0.25rem;">
                🅰️ Control Variant (Baseline)
            </div>
        """, unsafe_allow_html=True)
        val_ctrl = st.selectbox("Select Control Variant (A):", unique_vals, index=0, key="ctrl_select", label_visibility="collapsed")
        
        df_ctrl = df[df[dim_choice].astype(str) == str(val_ctrl)]
        ctrl_clicks = int(df_ctrl["Clicks"].sum()) if "Clicks" in df_ctrl.columns else 1000
        ctrl_convs = int(df_ctrl["Conversions"].sum()) if "Conversions" in df_ctrl.columns else 50
        ctrl_rate = (ctrl_convs / ctrl_clicks * 100.0) if ctrl_clicks > 0 else 0.0

        st.markdown(f"""
            <div style="display: flex; gap: 8px; margin-top: 0.4rem;">
                <span class="card-badge badge-blue">Clicks: {ctrl_clicks:,}</span>
                <span class="card-badge badge-blue">Conversions: {ctrl_convs:,}</span>
                <span class="card-badge badge-blue">CVR: {ctrl_rate:.2f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with v_col2:
        st.markdown("""
        <div class="ab-variant-card" style="border-top: 3px solid #818CF8;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 0.25rem;">
                🅱️ Treatment Variant (Challenger)
            </div>
        """, unsafe_allow_html=True)
        default_treat_idx = min(1, len(unique_vals) - 1) if len(unique_vals) > 1 else 0
        val_treat = st.selectbox("Select Treatment Variant (B):", unique_vals, index=default_treat_idx, key="treat_select", label_visibility="collapsed")
        
        df_treat = df[df[dim_choice].astype(str) == str(val_treat)]
        treat_clicks = int(df_treat["Clicks"].sum()) if "Clicks" in df_treat.columns else 1000
        treat_convs = int(df_treat["Conversions"].sum()) if "Conversions" in df_treat.columns else 65
        treat_rate = (treat_convs / treat_clicks * 100.0) if treat_clicks > 0 else 0.0

        st.markdown(f"""
            <div style="display: flex; gap: 8px; margin-top: 0.4rem;">
                <span class="card-badge badge-indigo">Clicks: {treat_clicks:,}</span>
                <span class="card-badge badge-indigo">Conversions: {treat_convs:,}</span>
                <span class="card-badge badge-indigo">CVR: {treat_rate:.2f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    test_res = run_two_proportion_test(
        control_conversions=ctrl_convs,
        control_impressions=ctrl_clicks,
        variant_conversions=treat_convs,
        variant_impressions=treat_clicks,
        alpha=alpha_level
    )

else:
    with ctl_col2:
        alpha_level = st.selectbox(
            "Confidence Level (1 - α):",
            [0.05, 0.01, 0.10],
            format_func=lambda a: f"{(1 - a) * 100:.0f}% Confidence (α={a})",
            index=0
        )
    with ctl_col3:
        st.write("")

    v_col1, v_col2 = st.columns(2)
    with v_col1:
        st.markdown("""
        <div class="ab-variant-card" style="border-top: 3px solid #38BDF8;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 0.5rem;">
                🅰️ Control Variant (A) Custom Metrics
            </div>
        """, unsafe_allow_html=True)
        ctrl_clicks = st.number_input("Control Impressions / Clicks (N_A):", min_value=10, value=25000, step=500)
        ctrl_convs = st.number_input("Control Conversions (X_A):", min_value=0, value=1250, step=25)
        st.markdown("</div>", unsafe_allow_html=True)

    with v_col2:
        st.markdown("""
        <div class="ab-variant-card" style="border-top: 3px solid #818CF8;">
            <div style="font-size: 0.8rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 0.5rem;">
                🅱️ Treatment Variant (B) Custom Metrics
            </div>
        """, unsafe_allow_html=True)
        treat_clicks = st.number_input("Treatment Impressions / Clicks (N_B):", min_value=10, value=25000, step=500)
        treat_convs = st.number_input("Treatment Conversions (X_B):", min_value=0, value=1450, step=25)
        st.markdown("</div>", unsafe_allow_html=True)

    test_res = run_two_proportion_test(
        control_conversions=ctrl_convs,
        control_impressions=ctrl_clicks,
        variant_conversions=treat_convs,
        variant_impressions=treat_clicks,
        alpha=alpha_level
    )

# Render Hypothesis Results
if "error" in test_res:
    st.error(test_res["error"])
else:
    verdict = test_res["verdict"]
    rel_lift = test_res["relative_lift_pct"]
    abs_lift = test_res.get("absolute_lift_pct", 0.0)
    p_val = test_res["p_value"]
    z_stat = test_res["z_statistic"]
    ci_low, ci_high = test_res["confidence_interval_95_pct"]
    power = test_res["statistical_power"]
    conf_pct = (1.0 - alpha_level) * 100

    # Executive Verdict Card
    if verdict == "Statistically Significant Winner":
        v_class = "ab-verdict-win"
        v_icon = "🚀"
        v_title = f"Statistically Significant Winner (Lift: +{rel_lift:.2f}%)"
        v_desc = (
            f"Treatment Variant B generated a statistically significant relative lift of <strong>+{rel_lift:.2f}%</strong> "
            f"(p = <code>{p_val:.5f}</code> &lt; α={alpha_level}). "
            f"There is a {conf_pct:.0f}% probability that this performance advantage is genuine and reproducible."
        )
    elif verdict == "Underperforming":
        v_class = "ab-verdict-loss"
        v_icon = "⚠️"
        v_title = f"Statistically Significant Underperformance (Lift: {rel_lift:.2f}%)"
        v_desc = (
            f"Treatment Variant B underperformed Control by <strong>{rel_lift:.2f}%</strong> "
            f"(p = <code>{p_val:.5f}</code> &lt; α={alpha_level}). "
            f"Recommendation: Maintain Control Variant A and discontinue Treatment."
        )
    else:
        v_class = "ab-verdict-inconclusive"
        v_icon = "⚖️"
        v_title = "Inconclusive / No Statistically Significant Difference"
        v_desc = (
            f"The observed lift of <strong>{rel_lift:+.2f}%</strong> is within expected random sample variation "
            f"(p = <code>{p_val:.5f}</code> &ge; α={alpha_level}). "
            f"We fail to reject the null hypothesis (H₀). Both variants perform equivalently."
        )

    st.markdown(f"""
    <div class="ab-verdict-card {v_class}">
        <div style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC; margin-bottom: 0.35rem; display: flex; align-items: center; gap: 8px;">
            <span>{v_icon}</span> {v_title}
        </div>
        <div style="color: #CBD5E1; font-size: 0.95rem; line-height: 1.5;">
            {v_desc}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Key Metric Panels
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid #38BDF8;">
            <div class="kpi-title">🅰️ Control CVR</div>
            <div class="kpi-val">{test_res['p_control_pct']:.2f}%</div>
            <div style="color: #94A3B8; font-size: 0.78rem; margin-top: 4px;">{ctrl_convs:,} / {ctrl_clicks:,} clicks</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid #818CF8;">
            <div class="kpi-title">🅱️ Treatment CVR</div>
            <div class="kpi-val">{test_res['p_variant_pct']:.2f}%</div>
            <div style="color: #94A3B8; font-size: 0.78rem; margin-top: 4px;">{treat_convs:,} / {treat_clicks:,} clicks</div>
        </div>
        """, unsafe_allow_html=True)

    lift_color = "#34D399" if (test_res['is_significant'] and rel_lift > 0) else ("#FB7185" if (test_res['is_significant'] and rel_lift < 0) else "#94A3B8")
    with k3:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid {lift_color};">
            <div class="kpi-title">📈 Relative Lift</div>
            <div class="kpi-val" style="color: {lift_color};">{rel_lift:+.2f}%</div>
            <div style="color: #94A3B8; font-size: 0.78rem; margin-top: 4px;">Δ {abs_lift:+.3f}% pts absolute</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi-card" style="border-top: 3px solid #F59E0B;">
            <div class="kpi-title">🎯 p-value &amp; Power</div>
            <div class="kpi-val" style="font-size: 1.35rem;">p = {p_val:.4f}</div>
            <div style="color: #94A3B8; font-size: 0.78rem; margin-top: 4px;">Power: {power * 100:.1f}% &bull; Z: {z_stat:+.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    # Visual Forest Plot & Decision Metrics Split
    res_col1, res_col2 = st.columns([1.4, 1])

    with res_col1:
        fig_ci = plot_ab_test_ci(test_res)
        st.plotly_chart(fig_ci, use_container_width=True)

    with res_col2:
        st.markdown("#### 🔬 Statistical Evidence Summary")
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.75); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 1.25rem;">
            <div style="margin-bottom: 0.75rem;">
                <span style="color: #94A3B8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">Confidence Interval ({conf_pct:.0f}%)</span>
                <div style="color: #F8FAFC; font-weight: 700; font-size: 1.1rem; margin-top: 2px;">
                    [{ci_low:+.3f}%, {ci_high:+.3f}%]
                </div>
                <div style="color: #64748B; font-size: 0.75rem;">Range of true absolute rate difference</div>
            </div>
            <div style="margin-bottom: 0.75rem;">
                <span style="color: #94A3B8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">Hypothesis Rejection Threshold</span>
                <div style="color: #F8FAFC; font-weight: 700; font-size: 1.1rem; margin-top: 2px;">
                    α = {alpha_level} ({'Passed p < α' if p_val < alpha_level else 'Failed p ≥ α'})
                </div>
                <div style="color: #64748B; font-size: 0.75rem;">Required statistical significance margin</div>
            </div>
            <div>
                <span style="color: #94A3B8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase;">Z-Score Standard Normal</span>
                <div style="color: #F8FAFC; font-weight: 700; font-size: 1.1rem; margin-top: 2px;">
                    Z = {z_stat:+.3f}
                </div>
                <div style="color: #64748B; font-size: 0.75rem;">Critical boundary |Z| &gt; 1.96</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# 3. Automated Pairwise Experiment Evaluations
st.subheader("📋 Automated Cross-Creative & Cross-Segment Experiment Matrix")
st.markdown("Systematically evaluates all pairwise treatment variants across marketing channels.")

exp_dim = st.selectbox("Automated Evaluation Dimension:", ["Campaign_Type", "Target_Audience"])
exp_df = evaluate_campaign_experiments(df, dimension=exp_dim)

if not exp_df.empty:
    st.dataframe(
        exp_df,
        use_container_width=True,
        column_config={
            "Control_CVR_Pct": st.column_config.NumberColumn("Control CVR", format="%.2f%%"),
            "Treatment_CVR_Pct": st.column_config.NumberColumn("Treatment CVR", format="%.2f%%"),
            "Relative_Lift_Pct": st.column_config.NumberColumn("Relative Lift", format="%+.2f%%"),
            "P_Value": st.column_config.NumberColumn("p-value", format="%.4f"),
            "Z_Statistic": st.column_config.NumberColumn("Z-Score", format="%+.2f"),
            "Statistical_Power": st.column_config.NumberColumn("Power", format="%.1%%")
        }
    )
else:
    st.info("No pairwise experiments met the minimum sample size threshold.")
