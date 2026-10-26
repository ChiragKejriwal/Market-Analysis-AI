"""
Data Preprocessing and Feature Enrichment Module.
Standardizes marketing channels, derives cost/revenue metrics, and computes KPI efficiency ratios.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd

CHANNEL_NORMALIZATION_MAP: Dict[str, str] = {
    "google": "Google Ads",
    "google ads": "Google Ads",
    "googleads": "Google Ads",
    "adwords": "Google Ads",
    "search": "Google Ads",
    "g_ads": "Google Ads",
    "facebook": "Facebook",
    "fb": "Facebook",
    "meta": "Facebook",
    "facebook ads": "Facebook",
    "instagram": "Instagram",
    "ig": "Instagram",
    "insta": "Instagram",
    "youtube": "YouTube",
    "yt": "YouTube",
    "video": "YouTube",
    "email": "Email",
    "newsletter": "Email",
    "edm": "Email",
    "organic": "Organic Search",
    "organic search": "Organic Search",
    "seo": "Organic Search",
    "linkedin": "LinkedIn",
    "li": "LinkedIn",
    "tiktok": "TikTok",
    "tt": "TikTok",
    "twitter": "X (Twitter)",
    "x": "X (Twitter)",
    "influencer": "Influencer",
    "display": "Display Network",
    "banner": "Display Network"
}


def normalize_channel_name(channel_str: Optional[str]) -> str:
    """Normalizes raw channel strings into canonical industry names."""
    if not channel_str or pd.isna(channel_str):
        return "Other / Unknown"
    
    cleaned = str(channel_str).strip().lower()
    return CHANNEL_NORMALIZATION_MAP.get(cleaned, str(channel_str).strip().title())


def clean_and_enrich_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw marketing campaign data and enriches it with derived financial and performance metrics:
    1. Standardizes 'Channel_Used' naming conventions.
    2. Computes normalized 'Conversion_Rate_Pct' and 'Conversion_Rate_Dec'.
    3. Derives 'Conversions' = round(Clicks * Conversion_Rate_Dec).
    4. Computes 'Spend' = Acquisition_Cost * max(Conversions, 1) or Acquisition_Cost * Conversions.
    5. Computes 'Revenue' = Spend * (1 + ROI).
    6. Derives KPI Ratios: CTR, CPC, CPA, ROAS, and Profit.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    enriched = df.copy()

    if "Channel_Used" in enriched.columns:
        enriched["Channel_Used"] = enriched["Channel_Used"].apply(normalize_channel_name)

    if "Campaign_Type" in enriched.columns:
        enriched["Campaign_Type"] = enriched["Campaign_Type"].astype(str).str.strip().str.title()

    def _clean_numeric(s: pd.Series) -> pd.Series:
        if s.dtype == object or isinstance(s.iloc[0] if len(s) > 0 else None, str):
            cleaned = s.astype(str).str.replace(r"[\$,€£]", "", regex=True).str.replace(",", "", regex=False).str.strip()
            return pd.to_numeric(cleaned, errors="coerce")
        return pd.to_numeric(s, errors="coerce")

    cr_series = _clean_numeric(enriched["Conversion_Rate"]).fillna(0.0)
    max_cr = cr_series.max() if not cr_series.empty else 0.0

    if max_cr > 1.0:
        enriched["Conversion_Rate_Pct"] = cr_series.round(2)
        enriched["Conversion_Rate_Dec"] = (cr_series / 100.0).round(4)
    else:
        enriched["Conversion_Rate_Dec"] = cr_series.round(4)
        enriched["Conversion_Rate_Pct"] = (cr_series * 100.0).round(2)

    clicks = _clean_numeric(enriched["Clicks"]).fillna(0).astype(int)
    impressions = _clean_numeric(enriched["Impressions"]).fillna(0).astype(int)
    acq_cost = _clean_numeric(enriched["Acquisition_Cost"]).fillna(0.0)
    roi = _clean_numeric(enriched["ROI"]).fillna(0.0)

    if "Conversions" not in enriched.columns or enriched["Conversions"].isnull().all():
        raw_conversions = clicks * enriched["Conversion_Rate_Dec"]
        enriched["Conversions"] = np.maximum(1, np.round(raw_conversions)).astype(int)
    else:
        enriched["Conversions"] = _clean_numeric(enriched["Conversions"]).fillna(0).astype(int)

    if "Spend" not in enriched.columns:
        if acq_cost.median() > 500:
            enriched["Spend"] = acq_cost.round(2)
        else:
            enriched["Spend"] = (acq_cost * enriched["Conversions"]).round(2)
    else:
        enriched["Spend"] = _clean_numeric(enriched["Spend"]).fillna(0.0).round(2)

    if "Revenue" not in enriched.columns:
        enriched["Revenue"] = (enriched["Spend"] * (1.0 + roi)).round(2)
    else:
        enriched["Revenue"] = _clean_numeric(enriched["Revenue"]).fillna(0.0).round(2)

    enriched["Profit"] = (enriched["Revenue"] - enriched["Spend"]).round(2)
    enriched["CTR"] = (clicks / np.maximum(impressions, 1)).round(4)
    enriched["CTR_Pct"] = (enriched["CTR"] * 100.0).round(2)
    enriched["CPC"] = (enriched["Spend"] / np.maximum(clicks, 1)).round(2)
    enriched["CPA"] = (enriched["Spend"] / np.maximum(enriched["Conversions"], 1)).round(2)
    enriched["ROAS"] = (enriched["Revenue"] / np.maximum(enriched["Spend"], 1.0)).round(2)

    return enriched
