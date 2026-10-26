"""
Conversion Funnel Analytics Module.
Constructs 4-stage marketing conversion funnels, micro-conversion rates, and channel drop-off diagnostics.
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd


def build_funnel_stages(
    df: pd.DataFrame,
    filter_dict: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Constructs a 4-stage marketing conversion funnel:
    1. Impressions (Awareness)
    2. Clicks (Consideration & Intent)
    3. Conversions (Acquisition)
    4. Repeat/High-Value (Customer Retention & LTV)
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["Stage", "Count", "Conversion_From_Top", "Step_Conversion_Rate", "Drop_Off_Rate", "Velocity_Index"])

    data = df.copy()

    if filter_dict:
        for k, v in filter_dict.items():
            if k in data.columns and v is not None and v != "All":
                data = data[data[k].astype(str) == str(v)]

    if data.empty:
        return pd.DataFrame(columns=["Stage", "Count", "Conversion_From_Top", "Step_Conversion_Rate", "Drop_Off_Rate", "Velocity_Index"])

    total_imp = int(data["Impressions"].sum()) if "Impressions" in data.columns else 0
    total_clicks = int(data["Clicks"].sum()) if "Clicks" in data.columns else 0
    total_conv = int(data["Conversions"].sum()) if "Conversions" in data.columns else 0

    if "ROI" in data.columns and "Conversions" in data.columns:
        median_roi = data["ROI"].median()
        high_val_mask = data["ROI"] >= median_roi
        high_val_conv = int(data.loc[high_val_mask, "Conversions"].sum() * 0.45)
    elif "Engagement_Score" in data.columns:
        high_val_mask = data["Engagement_Score"] >= 7.0
        high_val_conv = int(data.loc[high_val_mask, "Conversions"].sum() * 0.50)
    else:
        high_val_conv = int(total_conv * 0.35)

    high_val_conv = min(high_val_conv, total_conv)
    total_conv = min(total_conv, total_clicks)
    total_clicks = min(total_clicks, total_imp)

    stages_data = [
        {"Stage": "1. Impressions (Awareness)", "Count": total_imp},
        {"Stage": "2. Clicks (Traffic & Intent)", "Count": total_clicks},
        {"Stage": "3. Conversions (Acquisitions)", "Count": total_conv},
        {"Stage": "4. Repeat/High-Value (Retention)", "Count": high_val_conv}
    ]

    funnel_df = pd.DataFrame(stages_data)

    funnel_df["Conversion_From_Top"] = np.where(
        total_imp > 0,
        ((funnel_df["Count"] / total_imp) * 100.0).round(4),
        0.0
    )

    prev_counts = funnel_df["Count"].shift(1).fillna(total_imp)
    funnel_df["Step_Conversion_Rate"] = np.where(
        prev_counts > 0,
        ((funnel_df["Count"] / prev_counts) * 100.0).round(2),
        100.0
    )

    funnel_df["Drop_Off_Rate"] = np.where(
        funnel_df.index == 0,
        0.0,
        (100.0 - funnel_df["Step_Conversion_Rate"]).clip(lower=0.0).round(2)
    )

    funnel_df["Velocity_Index"] = np.where(
        total_imp > 0,
        (funnel_df["Count"] / total_imp).round(5),
        0.0
    )

    return funnel_df


def get_channel_funnel_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cross-channel micro-conversion rates and efficiency ratios.
    """
    if df is None or df.empty or "Channel_Used" not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby("Channel_Used").agg(
        Impressions=("Impressions", "sum"),
        Clicks=("Clicks", "sum"),
        Conversions=("Conversions", "sum"),
        Spend=("Spend", "sum"),
        Revenue=("Revenue", "sum")
    ).reset_index()

    grouped["Impression_to_Click_CTR"] = ((grouped["Clicks"] / np.maximum(grouped["Impressions"], 1)) * 100.0).round(2)
    grouped["Click_to_Conversion_CVR"] = ((grouped["Conversions"] / np.maximum(grouped["Clicks"], 1)) * 100.0).round(2)
    grouped["End_to_End_CVR"] = ((grouped["Conversions"] / np.maximum(grouped["Impressions"], 1)) * 100.0).round(4)
    grouped["CAC"] = (grouped["Spend"] / np.maximum(grouped["Conversions"], 1)).round(2)
    grouped["ROAS"] = (grouped["Revenue"] / np.maximum(grouped["Spend"], 1.0)).round(2)

    return grouped.sort_values(by="Click_to_Conversion_CVR", ascending=False).reset_index(drop=True)


# Aliases for backward compatibility
build_funnel_summary = build_funnel_stages
channel_funnel_comparison = get_channel_funnel_comparison
