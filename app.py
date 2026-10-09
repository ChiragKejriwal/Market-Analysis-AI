import streamlit as st
import pandas as pd
from src.metrics import calculate_global_kpis
from src.ui_components import apply_custom_css, render_sidebar_header, ensure_page_data_loaded

st.set_page_config(
    page_title="Causal Marketing Analytics & Budget Optimization",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure dataset is cleanly loaded with animated loader
df, raw_df = ensure_page_data_loaded(page_title="Executive Overview Hub")

# Render Sidebar Branding, Live Telemetry & CSV Ingestion
render_sidebar_header(active_page_title="Overview Hub")

# Compute high-level summary metrics
kpis = calculate_global_kpis(df)
total_campaigns = int(df["Campaign_ID"].nunique()) if "Campaign_ID" in df.columns else len(df)
total_channels = int(df["Channel_Used"].nunique()) if "Channel_Used" in df.columns else 0
total_spend = kpis["Total Spend"]
overall_roas = kpis["Blended ROAS"]

# Hero Section
st.markdown("""
<div class="hero-container">
    <div>
        <span class="badge">📊 Causal Attribution</span>
        <span class="badge">🔬 A/B Hypothesis Testing</span>
        <span class="badge">🔻 4-Stage Conversion Funnel</span>
        <span class="badge">⚙️ Non-Linear Budget Optimizer</span>
        <span class="badge">🎲 Monte Carlo Risk Simulator</span>
    </div>
    <h1 class="hero-title">📈 Causal Marketing Analytics & Budget Optimizer</h1>
    <p class="hero-subtitle">
        An enterprise analytics platform leveraging causal statistical inference, multi-stage conversion funnel modeling, 
        and constrained non-linear mathematical optimization to maximize portfolio ROAS under diminishing channel returns.
    </p>
</div>
""", unsafe_allow_html=True)

# 1. High-Level Summary Metric Cards
m_col1, m_col2, m_col3, m_col4 = st.columns(4)

with m_col1:
    st.markdown(f"""
    <div class="glow-metric-card">
        <div class="metric-label">📌 Total Campaigns</div>
        <div class="metric-value">{total_campaigns:,}</div>
        <span style="color: #38BDF8; font-size: 0.8rem; font-weight: 500;">Across active portfolio</span>
    </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown(f"""
    <div class="glow-metric-card">
        <div class="metric-label">🌐 Marketing Channels</div>
        <div class="metric-value">{total_channels}</div>
        <span style="color: #818CF8; font-size: 0.8rem; font-weight: 500;">Diversified channels</span>
    </div>
    """, unsafe_allow_html=True)

with m_col3:
    st.markdown(f"""
    <div class="glow-metric-card">
        <div class="metric-label">💰 Total Spend</div>
        <div class="metric-value">${total_spend:,.0f}</div>
        <span style="color: #F59E0B; font-size: 0.8rem; font-weight: 500;">Deployed capital</span>
    </div>
    """, unsafe_allow_html=True)

with m_col4:
    st.markdown(f"""
    <div class="glow-metric-card">
        <div class="metric-label">📈 Overall ROAS</div>
        <div class="metric-value" style="color: #34D399;">{overall_roas:.2f}x</div>
        <span style="color: #34D399; font-size: 0.8rem; font-weight: 500;">Portfolio efficiency</span>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# 2. Prominent Methodology Callout Note
st.markdown("""
<div class="methodology-box">
    <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.75rem;">
        <span style="font-size: 1.5rem;">⚖️</span>
        <h4 style="margin: 0; color: #F59E0B; font-weight: 700; font-size: 1.15rem;">
            Core Methodology: Incremental Lift vs. Observational Correlation
        </h4>
    </div>
    <p style="color: #E2E8F0; font-size: 0.95rem; line-height: 1.6; margin: 0;">
        Standard observational reporting attributes conversions to the last touchpoint, confounding true channel incrementality with organic user intent (selection bias). 
        This platform separates <strong>correlational revenue</strong> from <strong>true causal incremental lift</strong> through rigorous two-proportion hypothesis testing ($\alpha=0.05$), 
        and models non-linear diminishing marginal returns ($R_i(S_i) = \alpha_i \ln(1 + \beta_i S_i)$) to prevent over-allocation into saturated channels.
    </p>
</div>
""", unsafe_allow_html=True)

# 3. Global Architecture Diagram
st.subheader("🏛️ System Architecture & Analytical Pipeline")

st.markdown("""
```mermaid
flowchart LR
    A["Raw Campaign Data (CSV)"] --> B["Data Validation & Normalization"]
    B --> C["Feature Engineering (CTR, CPC, CPA, ROAS)"]
    
    C --> D1["1. Executive Overview<br/>• Dual-Axis Velocity<br/>• Donut Share Breakdown"]
    C --> D2["2. Causal A/B Testing<br/>• Proportion Z-Tests<br/>• 95% CI Forest Plots"]
    C --> D3["3. Conversion Funnel<br/>• 4-Stage Drop-Off<br/>• Cohort Retention Matrix"]
    C --> D4["4. Prescriptive Optimizer<br/>• Non-Linear SLSQP / PuLP<br/>• Monte Carlo Risk & VaR"]
    
    classDef dark fill:#1e293b,stroke:#6366f1,stroke-width:1.5px,color:#f8fafc;
    classDef glow fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#38bdf8;
    class A,B,C dark;
    class D1,D2,D3,D4 glow;
```
""")

st.write("")

# 4. Core Intelligence Modules Cards
st.subheader("⚡ Core Intelligence Modules")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: #38BDF8; margin-top: 0;">📊 1. Executive Performance & ROI Overview</h4>
        <p style="color: #94A3B8; font-size: 0.95rem; line-height: 1.5;">
            Portfolio-wide blended ROAS, CAC, spend velocity, dual-axis time-series trajectories, 
            and share of spend vs. share of revenue donut comparisons.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: #818CF8; margin-top: 0;">🔬 2. Causal A/B Testing & Hypothesis Lab</h4>
        <p style="color: #94A3B8; font-size: 0.95rem; line-height: 1.5;">
            Two-sample Proportion Z-tests, statistical power estimations, 95% confidence interval forest plots, 
            and automated pairwise experiment rankings across creatives and audiences.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: #34D399; margin-top: 0;">🔻 3. Multi-Stage Funnel & Cohort Retention</h4>
        <p style="color: #94A3B8; font-size: 0.95rem; line-height: 1.5;">
            Tracks full-journey progression across Impressions ➔ Clicks ➔ Conversions ➔ Retention, 
            pinpoints critical drop-off bottlenecks, and displays monthly retention heatmaps.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: #F59E0B; margin-top: 0;">⚙️ 4. Prescriptive Optimizer & Monte Carlo</h4>
        <p style="color: #94A3B8; font-size: 0.95rem; line-height: 1.5;">
            Fits logarithmic diminishing-returns response curves, solves SLSQP / PuLP non-linear programs for optimal capital allocation, 
            and runs 5,000-trial Monte Carlo risk simulations (VaR & CVaR).
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.subheader("🧭 Getting Started")
st.info("👈 **Select any module in the left sidebar** to begin analyzing marketing campaigns and running prescriptive optimization.")
