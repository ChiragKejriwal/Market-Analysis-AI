"""
Shared UI Components & Styling Module for Causal Marketing Analytics Platform.
Provides sidebar branding, custom CSS, CSV schema modal, and animated loaders.
"""

from typing import Dict, List, Optional, Tuple
import streamlit as st
import pandas as pd
from src.data_loader import load_marketing_data
from src.preprocessing import clean_and_enrich_data
from src.validator import validate_marketing_data, REQUIRED_COLUMNS

MANDATORY_COLUMNS: Dict[str, Dict[str, str]] = {
    "Date": {"type": "YYYY-MM-DD (Date)", "desc": "Transaction or campaign attribution date.", "example": "2024-03-15"},
    "Campaign_Type": {"type": "String", "desc": "Strategy format (Search, Social Media, Email, Display, Influencer, Video).", "example": "Search"},
    "Channel_Used": {"type": "String", "desc": "Marketing channel (Google Ads, Facebook, Instagram, YouTube, Email, Organic).", "example": "Google Ads"},
    "Clicks": {"type": "Integer (>= 0)", "desc": "Total user clicks recorded.", "example": "1250"},
    "Impressions": {"type": "Integer (>= 0)", "desc": "Total ad impressions or reach (Clicks <= Impressions).", "example": "45000"},
    "Acquisition_Cost": {"type": "Numeric / Currency ($)", "desc": "Average cost per acquisition (CPA/CAC).", "example": "35.50"},
    "ROI": {"type": "Numeric (>= 0)", "desc": "Return on investment multiple (e.g. 2.50 = 250% return).", "example": "2.40"},
    "Conversion_Rate": {"type": "Numeric (0.01-1.0 or 1-100%)", "desc": "Conversion rate from click to purchase.", "example": "0.045"}
}

OPTIONAL_COLUMNS: Dict[str, Dict[str, str]] = {
    "Campaign_ID": {"type": "String / Integer", "desc": "Unique campaign identifier code.", "example": "CMP_00101"},
    "Company": {"type": "String", "desc": "Brand or operating entity name.", "example": "Nexus Commerce"},
    "Target_Audience": {"type": "String", "desc": "Target demographic or persona segment.", "example": "Tech Professionals"},
    "Duration": {"type": "String", "desc": "Campaign execution window.", "example": "30 Days"},
    "Location": {"type": "String", "desc": "Geographic targeting location.", "example": "California"},
    "Language": {"type": "String", "desc": "Campaign creative language.", "example": "English"},
    "Engagement_Score": {"type": "Numeric (1-10)", "desc": "Content engagement score index.", "example": "8.4"}
}

COLUMN_SYNONYMS: Dict[str, List[str]] = {
    "Date": ["date", "day", "timestamp", "campaign_date", "ad_date"],
    "Campaign_Type": ["campaign_type", "campaigntype", "type", "ad_type", "format"],
    "Channel_Used": ["channel_used", "channel", "marketing_channel", "media_channel", "platform"],
    "Clicks": ["clicks", "click_count", "num_clicks", "ad_clicks", "total_clicks"],
    "Impressions": ["impressions", "views", "impressions_count", "reach", "total_impressions"],
    "Acquisition_Cost": ["acquisition_cost", "cost", "cpa", "cac", "spend", "ad_spend", "cost_per_acquisition"],
    "ROI": ["roi", "return_on_investment", "roas", "return_on_ad_spend"],
    "Conversion_Rate": ["conversion_rate", "cvr", "conv_rate", "cr", "conversion_pct", "conversion_ratio"],
    "Campaign_ID": ["campaign_id", "id", "campaign", "campaign_code", "ad_id"],
    "Company": ["company", "brand", "client", "business"],
    "Target_Audience": ["target_audience", "audience", "segment", "demographic", "target_group"],
    "Duration": ["duration", "timeframe", "flight_dates", "period"],
    "Location": ["location", "region", "geo", "country", "state", "city"],
    "Language": ["language", "locale", "lang"],
    "Engagement_Score": ["engagement_score", "engagement", "quality_score", "relevance_score"]
}


def get_sample_template_df() -> pd.DataFrame:
    """Generates a sample template DataFrame matching standard schema."""
    return pd.DataFrame([
        {
            "Campaign_ID": "CMP_00001",
            "Company": "Nexus Commerce",
            "Campaign_Type": "Search",
            "Target_Audience": "Tech Professionals",
            "Duration": "30 Days",
            "Channel_Used": "Google Ads",
            "Conversion_Rate": 0.042,
            "Acquisition_Cost": 42.50,
            "ROI": 2.85,
            "Location": "California",
            "Language": "English",
            "Clicks": 2150,
            "Impressions": 58000,
            "Engagement_Score": 8.2,
            "Date": "2024-01-15"
        },
        {
            "Campaign_ID": "CMP_00002",
            "Company": "Apex Retail",
            "Campaign_Type": "Social Media",
            "Target_Audience": "Women 25-34",
            "Duration": "15 Days",
            "Channel_Used": "Instagram",
            "Conversion_Rate": 0.031,
            "Acquisition_Cost": 28.00,
            "ROI": 3.10,
            "Location": "New York",
            "Language": "English",
            "Clicks": 1840,
            "Impressions": 62000,
            "Engagement_Score": 9.1,
            "Date": "2024-02-01"
        },
        {
            "Campaign_ID": "CMP_00003",
            "Company": "Veritas Health",
            "Campaign_Type": "Email",
            "Target_Audience": "All 35-50",
            "Duration": "45 Days",
            "Channel_Used": "Email",
            "Conversion_Rate": 0.058,
            "Acquisition_Cost": 18.20,
            "ROI": 4.50,
            "Location": "Texas",
            "Language": "English",
            "Clicks": 950,
            "Impressions": 19000,
            "Engagement_Score": 7.8,
            "Date": "2024-02-18"
        },
        {
            "Campaign_ID": "CMP_00004",
            "Company": "Pulse Fashion",
            "Campaign_Type": "Video",
            "Target_Audience": "Gen Z Students",
            "Duration": "60 Days",
            "Channel_Used": "YouTube",
            "Conversion_Rate": 0.021,
            "Acquisition_Cost": 55.00,
            "ROI": 1.95,
            "Location": "Illinois",
            "Language": "English",
            "Clicks": 3400,
            "Impressions": 145000,
            "Engagement_Score": 8.9,
            "Date": "2024-03-05"
        },
        {
            "Campaign_ID": "CMP_00005",
            "Company": "Horizon Tech",
            "Campaign_Type": "Display",
            "Target_Audience": "Affluent Homeowners",
            "Duration": "30 Days",
            "Channel_Used": "Facebook",
            "Conversion_Rate": 0.029,
            "Acquisition_Cost": 34.80,
            "ROI": 2.40,
            "Location": "Florida",
            "Language": "English",
            "Clicks": 1420,
            "Impressions": 48000,
            "Engagement_Score": 7.4,
            "Date": "2024-03-20"
        }
    ])


def inspect_uploaded_df(uploaded_df: pd.DataFrame) -> Tuple[bool, Optional[pd.DataFrame], List[str], List[str], Dict[str, str]]:
    """
    Inspects an uploaded DataFrame, resolves column synonyms, and validates structure.
    """
    if uploaded_df is None or uploaded_df.empty:
        return False, None, ["Uploaded file is empty."], [], {}

    actual_cols = list(uploaded_df.columns)
    col_mapping = {}
    normalized_actual = {str(c).strip().lower().replace(" ", "_").replace("-", "_"): c for c in actual_cols}

    for canonical, synonyms in COLUMN_SYNONYMS.items():
        if canonical in actual_cols:
            col_mapping[canonical] = canonical
            continue
        for syn in synonyms:
            syn_norm = syn.lower().replace(" ", "_").replace("-", "_")
            if syn_norm in normalized_actual:
                col_mapping[normalized_actual[syn_norm]] = canonical
                break

    mapped_df = uploaded_df.rename(columns=col_mapping)
    mapped_cols = list(mapped_df.columns)
    missing_mandatory = [col for col in MANDATORY_COLUMNS if col not in mapped_cols]

    if missing_mandatory:
        errors = [f"Missing required columns: {', '.join(missing_mandatory)}"]
        return False, mapped_df, errors, [], col_mapping

    is_valid, hard_errs, soft_warns = validate_marketing_data(mapped_df, raise_on_error=False)
    return is_valid, mapped_df, hard_errs, soft_warns, col_mapping


def apply_custom_css():
    """Injects high-contrast, modern glassmorphism CSS across the application."""
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0B0F19 0%, #0F172A 50%, #0B0F19 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        .sidebar-brand-box {
            background: linear-gradient(135deg, rgba(30, 27, 75, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
            padding: 1.15rem 1.25rem;
            border-radius: 14px;
            border: 1px solid rgba(99, 102, 241, 0.35);
            margin-bottom: 1.25rem;
            box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.2);
        }

        .sidebar-status-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 0.35rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 600;
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: #34D399;
            margin-bottom: 0.6rem;
        }

        .glass-card {
            background: rgba(30, 41, 59, 0.6);
            backdrop-filter: blur(14px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.5rem;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        }

        .glass-card:hover {
            border-color: rgba(99, 102, 241, 0.4);
            box-shadow: 0 8px 30px rgba(99, 102, 241, 0.18);
            transform: translateY(-2px);
        }

        .kpi-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2);
            transition: border-color 0.2s ease;
        }

        .kpi-title {
            color: #94A3B8;
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.35rem;
        }

        .kpi-val {
            color: #F8FAFC;
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .card-badge {
            display: inline-block;
            padding: 0.25rem 0.65rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 0.35rem;
            margin-bottom: 0.35rem;
        }
        .badge-green { background: rgba(16, 185, 129, 0.18); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.4); }
        .badge-blue { background: rgba(56, 189, 248, 0.18); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.4); }
        .badge-indigo { background: rgba(99, 102, 241, 0.18); color: #818CF8; border: 1px solid rgba(99, 102, 241, 0.4); }
        .badge-rose { background: rgba(244, 63, 94, 0.18); color: #FB7185; border: 1px solid rgba(244, 63, 94, 0.4); }
        .badge-amber { background: rgba(245, 158, 11, 0.18); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.4); }

        .ab-verdict-card {
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
            backdrop-filter: blur(14px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .ab-verdict-win {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(15, 23, 42, 0.95) 100%);
            border-left: 5px solid #34D399;
        }

        .ab-verdict-loss {
            background: linear-gradient(135deg, rgba(244, 63, 94, 0.15) 0%, rgba(15, 23, 42, 0.95) 100%);
            border-left: 5px solid #FB7185;
        }

        .ab-verdict-inconclusive {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(15, 23, 42, 0.95) 100%);
            border-left: 5px solid #818CF8;
        }

        .ab-variant-card {
            background: rgba(30, 41, 59, 0.65);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.15rem;
            margin-bottom: 0.75rem;
        }

        @keyframes spinRing {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @keyframes pulseGlow {
            0%, 100% { opacity: 0.85; box-shadow: 0 10px 30px rgba(99, 102, 241, 0.35); }
            50% { opacity: 1; box-shadow: 0 15px 50px rgba(56, 189, 248, 0.55); }
        }

        .loader-card {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 27, 75, 0.9) 50%, rgba(15, 23, 42, 0.98) 100%);
            border: 1px solid rgba(99, 102, 241, 0.45);
            border-radius: 20px;
            padding: 3rem 2.5rem;
            text-align: center;
            max-width: 680px;
            margin: 3.5rem auto;
            animation: pulseGlow 3s infinite ease-in-out;
        }

        .loader-spinner-ring {
            display: inline-block;
            width: 56px;
            height: 56px;
            border: 4px solid rgba(99, 102, 241, 0.2);
            border-top: 4px solid #38BDF8;
            border-right: 4px solid #818CF8;
            border-radius: 50%;
            animation: spinRing 0.9s linear infinite;
            margin-bottom: 1.25rem;
        }

        .loader-step-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 0.4rem 1rem;
            border-radius: 9999px;
            background: rgba(99, 102, 241, 0.15);
            border: 1px solid rgba(99, 102, 241, 0.35);
            color: #C7D2FE;
            font-size: 0.85rem;
            font-weight: 600;
            margin-top: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)


def ensure_page_data_loaded(page_title: str = "") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensures that cleaned and raw marketing datasets are preloaded before rendering.
    """
    apply_custom_css()

    if "df" in st.session_state and st.session_state["df"] is not None and not st.session_state["df"].empty:
        raw_df = st.session_state.get("raw_df", st.session_state["df"])
        return st.session_state["df"], raw_df

    loader_placeholder = st.empty()
    with loader_placeholder.container():
        st.markdown(f"""
        <div class="loader-card">
            <div class="loader-spinner-ring"></div>
            <h2 style="color: #F8FAFC; margin-bottom: 0.5rem; font-weight: 800;">
                Initializing Marketing Intelligence Engine
            </h2>
            <p style="color: #94A3B8; font-size: 0.95rem; margin-bottom: 1.25rem; line-height: 1.6;">
                Preparing marketing attribution models, calculating ROAS benchmarks, and synchronizing channel metrics for <strong>{page_title or 'Executive Suite'}</strong>...
            </p>
            <div>
                <span class="loader-step-pill">
                    ⚡ Loading Dataset &amp; Normalizing Channels...
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        raw = load_marketing_data()
        clean = clean_and_enrich_data(raw)
        st.session_state["raw_df"] = raw
        st.session_state["df"] = clean

    loader_placeholder.empty()
    return clean, raw


@st.dialog("📋 Marketing Campaign CSV Schema & Format Guide", width="large")
def show_schema_modal():
    """Renders the comprehensive CSV Schema Guide and downloadable sample template."""
    st.markdown("""
    ### 📁 Marketing Data Ingestion Standard
    Upload standard CSV files matching the schema specification below.
    """)

    sample_df = get_sample_template_df()
    csv_bytes = sample_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Standard CSV Template (.csv)",
        data=csv_bytes,
        file_name="marketing_campaign_template.csv",
        mime="text/csv",
        use_container_width=True,
        type="primary"
    )

    st.markdown("---")
    st.subheader("🔴 Mandatory Columns (Required)")
    mand_rows = []
    for col, meta in MANDATORY_COLUMNS.items():
        syns = ", ".join(COLUMN_SYNONYMS.get(col, []))
        mand_rows.append({
            "Column Name": f"**`{col}`**",
            "Expected Type": meta["type"],
            "Description": meta["desc"],
            "Accepted Synonyms": f"`{syns}`",
            "Example": meta["example"]
        })
    st.dataframe(pd.DataFrame(mand_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🟢 Optional Enrichment Columns")
    opt_rows = []
    for col, meta in OPTIONAL_COLUMNS.items():
        syns = ", ".join(COLUMN_SYNONYMS.get(col, []))
        opt_rows.append({
            "Column Name": f"`{col}`",
            "Expected Type": meta["type"],
            "Description": meta["desc"],
            "Accepted Synonyms": f"`{syns}`",
            "Example": meta["example"]
        })
    st.dataframe(pd.DataFrame(opt_rows), use_container_width=True, hide_index=True)


def render_sidebar_header(active_page_title: str = ""):
    """Renders the unified sidebar brand header, template button, and CSV uploader."""
    st.sidebar.markdown("""
    <div class="sidebar-brand-box">
        <div class="sidebar-status-pill">
            <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#34D399;"></span>
            System Operational
        </div>
        <div style="color: #F8FAFC; font-weight: 800; font-size: 1.15rem; letter-spacing: -0.01em;">
            📈 Causal Marketing Suite
        </div>
        <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 2px;">
            Attribution &bull; A/B Tests &bull; Optimizer
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Data Ingestion & Schema")

    if st.sidebar.button("📋 CSV Schema Guide & Template", use_container_width=True):
        show_schema_modal()

    uploaded_file = st.sidebar.file_uploader(
        "Upload Custom Marketing CSV",
        type=["csv"],
        help="Upload marketing records matching the required schema."
    )

    if uploaded_file is not None:
        try:
            raw_uploaded = pd.read_csv(uploaded_file)
            is_valid, mapped_df, hard_errs, soft_warns, col_map = inspect_uploaded_df(raw_uploaded)

            if not is_valid:
                st.sidebar.error("❌ Schema Validation Failed:")
                for err in hard_errs:
                    st.sidebar.caption(f"• {err}")
            else:
                cleaned_uploaded = clean_and_enrich_data(mapped_df)
                st.session_state["raw_df"] = mapped_df
                st.session_state["df"] = cleaned_uploaded
                st.sidebar.success(f"✅ Loaded `{uploaded_file.name}` ({len(cleaned_uploaded):,} rows)!")

                if col_map:
                    remapped = {k: v for k, v in col_map.items() if k != v}
                    if remapped:
                        st.sidebar.info(f"Auto-mapped synonyms: {', '.join([f'`{k}` ➔ `{v}`' for k, v in remapped.items()])}")

                if soft_warns:
                    with st.sidebar.expander("⚠️ Data Quality Notes"):
                        for w in soft_warns:
                            st.caption(f"• {w}")
        except Exception as e:
            st.sidebar.error(f"Error parsing uploaded file: {e}")
