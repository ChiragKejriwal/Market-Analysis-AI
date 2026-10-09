"""
Page 1: Executive Overview & Portfolio KPIs
High-level marketing portfolio efficiency, revenue trajectory, and cross-channel benchmarks.
"""

import streamlit as st
import pandas as pd
from src.data_loader import load_marketing_data
from src.preprocessing import clean_and_enrich_data
from src.metrics import calculate_global_kpis, calculate_channel_kpis, compute_time_series_metrics
from src.visualization import (
    plot_executive_kpis,
    plot_revenue_spend_dual_axis,
    plot_channel_share_donuts,
    plot_channel_roas_bar
)

from src.ui_components import render_sidebar_header, ensure_page_data_loaded

st.set_page_config(
    page_title="Executive Overview | Marketing Analytics",
    page_icon="📊",
    layout="wide"
)

# Ensure data is cleanly loaded with animated loader
df, raw_df = ensure_page_data_loaded(page_title="Executive Overview")

# Render Sidebar Branding, Live Telemetry & CSV Ingestion
render_sidebar_header(active_page_title="Executive Overview")

st.title("📊 Executive Portfolio Performance & ROI Overview")
st.markdown("Comprehensive view of portfolio capital deployment, revenue velocity, and channel efficiency.")

# 1. Executive Top Metric Cards
kpis = calculate_global_kpis(df)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-title">💰 Total Spend</div>
        <div class="kpi-val" style="color: #38BDF8;">${kpis['Total Spend']:,.0f}</div>
        <span style="color: #94A3B8; font-size: 0.8rem;">Portfolio marketing capital</span>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-title">💵 Total Revenue</div>
        <div class="kpi-val" style="color: #34D399;">${kpis['Total Revenue']:,.0f}</div>
        <span style="color: #94A3B8; font-size: 0.8rem;">Attributed top-line revenue</span>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-title">📈 Blended ROAS</div>
        <div class="kpi-val" style="color: #818CF8;">{kpis['Blended ROAS']:.2f}x</div>
        <span style="color: #94A3B8; font-size: 0.8rem;">Revenue / Marketing Spend</span>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-title">🏷️ Blended CAC</div>
        <div class="kpi-val" style="color: #F59E0B;">${kpis['Blended CAC']:.2f}</div>
        <span style="color: #94A3B8; font-size: 0.8rem;">Cost per acquired customer</span>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# 2. Interactive Timeline of Spend vs. Revenue over time (Dual-Axis Plotly Chart)
st.subheader("📈 Spend vs. Revenue Velocity Timeline")
freq_choice = st.radio("Aggregation Interval:", ["Weekly", "Monthly", "Daily"], horizontal=True, index=0)
freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}

ts_df = compute_time_series_metrics(df, freq=freq_map[freq_choice])
fig_ts = plot_revenue_spend_dual_axis(ts_df)
st.plotly_chart(fig_ts, use_container_width=True)

st.markdown("---")

# 3. Channel Share & Efficiency
st.subheader("🌐 Channel Share of Spend vs. Share of Revenue")
st.markdown("Identify disproportionate return channels where revenue share exceeds spend share.")

channel_df = calculate_channel_kpis(df)

col_donut, col_bar = st.columns([1, 1])

with col_donut:
    fig_donuts = plot_channel_share_donuts(channel_df)
    st.plotly_chart(fig_donuts, use_container_width=True)

with col_bar:
    fig_bar = plot_channel_roas_bar(channel_df)
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# 4. Channel KPI Summary Table
st.subheader("📋 Cross-Channel Financial & Unit Economics Breakdown")

display_cols = ["Channel_Used", "Spend", "Revenue", "Conversions", "CAC", "ROAS", "ROI", "AOV"]
formatted_df = channel_df[display_cols].copy()

st.dataframe(
    formatted_df.style.format({
        "Spend": "${:,.2f}",
        "Revenue": "${:,.2f}",
        "Conversions": "{:,}",
        "CAC": "${:,.2f}",
        "ROAS": "{:.2f}x",
        "ROI": "{:+.2f}x",
        "AOV": "${:,.2f}"
    }),
    use_container_width=True
)

st.markdown("---")
with st.expander("🔍 Preview Raw Enriched Campaign Data (Top 100 Records)"):
    st.dataframe(df.head(100), use_container_width=True)
