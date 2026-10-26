"""
Page 3: Conversion Funnel & Customer Acquisition Cohorts
"""

import streamlit as st
from src.funnel import build_funnel_stages, get_channel_funnel_comparison
from src.cohort import build_acquisition_cohorts
from src.visualization import plot_interactive_funnel, plot_cohort_heatmap
from src.ui_components import render_sidebar_header, ensure_page_data_loaded

st.set_page_config(
    page_title="Conversion Funnel & Cohorts | Marketing Analytics",
    page_icon="🔻",
    layout="wide"
)

# Ensure dataset is cleanly loaded with animated loader
df, raw_df = ensure_page_data_loaded(page_title="Conversion Funnel & Cohorts")
render_sidebar_header(active_page_title="Conversion Funnel")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .bottleneck-warning {
        background: linear-gradient(135deg, rgba(248, 113, 113, 0.1) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(248, 113, 113, 0.35);
        border-left: 5px solid #F87171;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-top: 1rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔻 Multi-Stage Conversion Funnel & Cohort Retention")
st.markdown("""
Track customer progression across the 4-stage funnel from initial Impression to High-Value Retention, 
diagnose critical drop-off bottlenecks, and analyze monthly retention matrices.
""")

# Optional Funnel Channel Filter
filter_channel = st.selectbox(
    "Filter Funnel by Marketing Channel:",
    ["All Channels"] + sorted(df["Channel_Used"].dropna().unique().tolist()) if "Channel_Used" in df.columns else ["All Channels"]
)

filter_dict = {"Channel_Used": filter_channel} if filter_channel != "All Channels" else None

# 1. 4-Stage Conversion Funnel
funnel_df = build_funnel_stages(df, filter_dict=filter_dict)

st.subheader("📊 4-Stage Interactive Conversion Funnel")

f_col1, f_col2 = st.columns([1.1, 1])

with f_col1:
    fig_funnel = plot_interactive_funnel(funnel_df)
    st.plotly_chart(fig_funnel, use_container_width=True)

with f_col2:
    st.markdown("#### Stage-by-Stage Drop-Off Breakdown")
    st.dataframe(
        funnel_df.style.format({
            "Count": "{:,}",
            "Conversion_From_Top": "{:.4f}%",
            "Step_Conversion_Rate": "{:.2f}%",
            "Drop_Off_Rate": "{:.2f}%",
            "Velocity_Index": "{:.5f}"
        }),
        use_container_width=True
    )

    if len(funnel_df) > 1:
        drop_steps = funnel_df.iloc[1:].copy()
        max_drop_row = drop_steps.loc[drop_steps["Drop_Off_Rate"].idxmax()]
        stage_name = max_drop_row["Stage"]
        drop_pct = max_drop_row["Drop_Off_Rate"]
        
        st.markdown(f"""
        <div class="bottleneck-warning">
            <h5 style="color: #F87171; margin: 0 0 0.5rem 0; font-weight: 700;">
                ⚠️ Critical Bottleneck Detected: {stage_name}
            </h5>
            <p style="color: #E2E8F0; font-size: 0.9rem; margin: 0; line-height: 1.5;">
                This transition experiences the highest customer attrition with a 
                <strong>{drop_pct:.1f}% drop-off rate</strong>. Recommended interventions: landing page UX optimization, 
                streamlined checkout flows, and targeted retargeting.
            </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# 2. Channel Funnel Efficiency Comparison
st.subheader("🌐 Cross-Channel Funnel Throughput Comparison")
ch_funnel = get_channel_funnel_comparison(df)

st.dataframe(
    ch_funnel.style.format({
        "Impressions": "{:,}",
        "Clicks": "{:,}",
        "Conversions": "{:,}",
        "Spend": "${:,.2f}",
        "Revenue": "${:,.2f}",
        "Impression_to_Click_CTR": "{:.2f}%",
        "Click_to_Conversion_CVR": "{:.2f}%",
        "End_to_End_CVR": "{:.4f}%",
        "CAC": "${:,.2f}",
        "ROAS": "{:.2f}x"
    }),
    use_container_width=True
)

st.markdown("---")

# 3. Acquisition Cohort Retention Heatmap
st.subheader("📅 Monthly Acquisition Cohort Retention Heatmap")
st.markdown("Tracks conversion retention and repeat engagement for customer cohorts over elapsed calendar months ($M+0$ to $M+6$).")

cohort_df = build_acquisition_cohorts(df)
if not cohort_df.empty:
    fig_cohort = plot_cohort_heatmap(cohort_df)
    st.plotly_chart(fig_cohort, use_container_width=True)
else:
    st.info("Insufficient historical date span to build multi-month cohort retention matrix.")
