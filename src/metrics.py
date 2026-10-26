"""
Marketing Performance Metrics and Financial Analytics Module.
Calculates global blended KPIs, channel efficiency tables, Customer Acquisition Cost (CAC),
and Customer Lifetime Value (CLV) models.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd


def calculate_global_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes global portfolio-level marketing financial and efficiency metrics.
    """
    if df is None or df.empty:
        return {
            "Total Spend": 0.0,
            "Total Revenue": 0.0,
            "Blended ROAS": 0.0,
            "Blended ROI": 0.0,
            "Blended CAC": 0.0,
            "Overall CTR": 0.0,
            "Conversion Rate": 0.0
        }

    total_spend = float(df["Spend"].sum()) if "Spend" in df.columns else 0.0
    total_rev = float(df["Revenue"].sum()) if "Revenue" in df.columns else 0.0
    total_clicks = float(df["Clicks"].sum()) if "Clicks" in df.columns else 0.0
    total_imp = float(df["Impressions"].sum()) if "Impressions" in df.columns else 0.0
    total_conv = float(df["Conversions"].sum()) if "Conversions" in df.columns else 0.0

    blended_roas = (total_rev / total_spend) if total_spend > 0 else 0.0
    blended_roi = ((total_rev - total_spend) / total_spend) if total_spend > 0 else 0.0
    blended_cac = (total_spend / total_conv) if total_conv > 0 else 0.0
    overall_ctr = (total_clicks / total_imp) if total_imp > 0 else 0.0
    overall_cvr = (total_conv / total_clicks) if total_clicks > 0 else 0.0

    return {
        "Total Spend": round(total_spend, 2),
        "Total Revenue": round(total_rev, 2),
        "Blended ROAS": round(blended_roas, 2),
        "Blended ROI": round(blended_roi, 2),
        "Blended CAC": round(blended_cac, 2),
        "Overall CTR": round(overall_ctr, 4),
        "Conversion Rate": round(overall_cvr, 4)
    }


def calculate_channel_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates metrics by Channel_Used: Spend, Conversions, Revenue, CAC, ROAS, ROI, and Avg Order Value (AOV).
    """
    if df is None or df.empty or "Channel_Used" not in df.columns:
        return pd.DataFrame(columns=["Channel_Used", "Spend", "Conversions", "Revenue", "CAC", "ROAS", "ROI", "AOV"])

    grouped = df.groupby("Channel_Used").agg(
        Spend=("Spend", "sum"),
        Conversions=("Conversions", "sum"),
        Revenue=("Revenue", "sum"),
        Clicks=("Clicks", "sum") if "Clicks" in df.columns else ("Spend", "count"),
        Impressions=("Impressions", "sum") if "Impressions" in df.columns else ("Spend", "count")
    ).reset_index()

    grouped["CAC"] = np.where(grouped["Conversions"] > 0, (grouped["Spend"] / grouped["Conversions"]).round(2), 0.0)
    grouped["ROAS"] = np.where(grouped["Spend"] > 0, (grouped["Revenue"] / grouped["Spend"]).round(2), 0.0)
    grouped["ROI"] = np.where(grouped["Spend"] > 0, ((grouped["Revenue"] - grouped["Spend"]) / grouped["Spend"]).round(2), 0.0)
    grouped["AOV"] = np.where(grouped["Conversions"] > 0, (grouped["Revenue"] / grouped["Conversions"]).round(2), 0.0)

    return grouped.sort_values(by="Revenue", ascending=False).reset_index(drop=True)


def calculate_clv(
    df: pd.DataFrame,
    margin_rate: float = 0.40,
    churn_rate: float = 0.15,
    purchase_frequency: Optional[float] = None
) -> pd.DataFrame:
    """
    Estimates Customer Lifetime Value (CLV) and the CLV/CAC ratio per channel:
    CLV = (AOV * Purchase_Frequency * Margin) / Churn_Rate
    """
    channel_kpis = calculate_channel_kpis(df)
    if channel_kpis.empty:
        return pd.DataFrame()

    clv_df = channel_kpis.copy()

    channel_freq_map = {
        "Email": 3.2,
        "Organic Search": 2.6,
        "Google Ads": 2.4,
        "Facebook": 1.9,
        "Instagram": 1.8,
        "YouTube": 1.6,
    }

    if purchase_frequency is not None:
        clv_df["Purchase_Frequency"] = purchase_frequency
    else:
        clv_df["Purchase_Frequency"] = clv_df["Channel_Used"].map(channel_freq_map).fillna(2.0)

    clv_df["Margin_Rate"] = margin_rate
    clv_df["Churn_Rate"] = churn_rate

    clv_df["CLV"] = np.where(
        churn_rate > 0,
        ((clv_df["AOV"] * clv_df["Purchase_Frequency"] * margin_rate) / churn_rate).round(2),
        0.0
    )

    clv_df["CLV_to_CAC_Ratio"] = np.where(
        clv_df["CAC"] > 0,
        (clv_df["CLV"] / clv_df["CAC"]).round(2),
        0.0
    )

    clv_df["Unit_Economics_Health"] = np.where(
        clv_df["CLV_to_CAC_Ratio"] >= 3.0,
        "Strong (>= 3.0x)",
        np.where(clv_df["CLV_to_CAC_Ratio"] >= 1.5, "Moderate (1.5x - 3.0x)", "At Risk (< 1.5x)")
    )

    cols = [
        "Channel_Used", "Spend", "Conversions", "Revenue", "AOV",
        "Purchase_Frequency", "CLV", "CAC", "CLV_to_CAC_Ratio", "Unit_Economics_Health"
    ]
    return clv_df[cols].sort_values(by="CLV_to_CAC_Ratio", ascending=False).reset_index(drop=True)


# Aliases for backward compatibility
compute_executive_kpis = lambda df: {
    "total_spend": calculate_global_kpis(df)["Total Spend"],
    "total_revenue": calculate_global_kpis(df)["Total Revenue"],
    "total_profit": calculate_global_kpis(df)["Total Revenue"] - calculate_global_kpis(df)["Total Spend"],
    "blended_roi": calculate_global_kpis(df)["Blended ROI"],
    "blended_roas": calculate_global_kpis(df)["Blended ROAS"],
    "total_impressions": int(df["Impressions"].sum()) if "Impressions" in df.columns else 0,
    "total_clicks": int(df["Clicks"].sum()) if "Clicks" in df.columns else 0,
    "total_conversions": int(df["Conversions"].sum()) if "Conversions" in df.columns else 0,
    "blended_ctr": calculate_global_kpis(df)["Overall CTR"] * 100.0,
    "blended_cpc": (calculate_global_kpis(df)["Total Spend"] / max(df["Clicks"].sum(), 1)) if "Clicks" in df.columns else 0.0,
    "blended_cpa": calculate_global_kpis(df)["Blended CAC"],
    "total_campaigns": int(df["Campaign_ID"].nunique()) if "Campaign_ID" in df.columns else len(df),
}

compute_channel_performance = calculate_channel_kpis


def compute_time_series_metrics(
    df: pd.DataFrame,
    freq: str = "W",
    group_col: Optional[str] = None
) -> pd.DataFrame:
    """Aggregates performance metrics over time."""
    if df is None or df.empty or "Date" not in df.columns:
        return pd.DataFrame()

    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"])

    group_keys = [pd.Grouper(key="Date", freq=freq)]
    if group_col and group_col in data.columns:
        group_keys.append(group_col)

    res = data.groupby(group_keys).agg(
        Spend=("Spend", "sum"),
        Revenue=("Revenue", "sum"),
        Profit=("Profit", "sum") if "Profit" in data.columns else ("Revenue", "sum"),
        Clicks=("Clicks", "sum") if "Clicks" in data.columns else ("Spend", "count"),
        Impressions=("Impressions", "sum") if "Impressions" in data.columns else ("Spend", "count"),
        Conversions=("Conversions", "sum") if "Conversions" in data.columns else ("Spend", "count")
    ).reset_index()

    res["ROAS"] = np.where(res["Spend"] > 0, (res["Revenue"] / res["Spend"]).round(2), 0.0)
    res["CTR_Pct"] = np.where(res["Impressions"] > 0, ((res["Clicks"] / res["Impressions"]) * 100).round(2), 0.0)
    res["CPA"] = np.where(res["Conversions"] > 0, (res["Spend"] / res["Conversions"]).round(2), 0.0)

    return res
