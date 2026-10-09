"""
Unit tests for data preprocessing module.
"""

import pandas as pd
import numpy as np
import pytest
from src.preprocessing import clean_and_enrich_data, normalize_channel_name


def test_normalize_channel_name():
    assert normalize_channel_name("google ads") == "Google Ads"
    assert normalize_channel_name("adwords") == "Google Ads"
    assert normalize_channel_name("fb") == "Facebook"
    assert normalize_channel_name("ig") == "Instagram"
    assert normalize_channel_name("yt") == "YouTube"


def test_clean_and_enrich_data():
    raw_df = pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02"],
        "Campaign_Type": ["search", "email"],
        "Channel_Used": ["g_ads", "newsletter"],
        "Clicks": [100, 200],
        "Impressions": [2000, 5000],
        "Acquisition_Cost": [50.0, 25.0],
        "ROI": [2.0, 3.0],
        "Conversion_Rate": [0.05, 0.10]
    })

    enriched = clean_and_enrich_data(raw_df)

    assert "Spend" in enriched.columns
    assert "Revenue" in enriched.columns
    assert "Profit" in enriched.columns
    assert "CTR" in enriched.columns
    assert "CPC" in enriched.columns
    assert "CPA" in enriched.columns
    assert "ROAS" in enriched.columns

    # Channel names normalized
    assert enriched["Channel_Used"].iloc[0] == "Google Ads"
    assert enriched["Channel_Used"].iloc[1] == "Email"

    # Conversions: 100 * 0.05 = 5; 200 * 0.10 = 20
    assert enriched["Conversions"].iloc[0] == 5
    assert enriched["Conversions"].iloc[1] == 20

    # Spend = Acq_Cost * Conversions -> 50 * 5 = 250; 25 * 20 = 500
    assert enriched["Spend"].iloc[0] == 250.0
    assert enriched["Spend"].iloc[1] == 500.0

    # Revenue = Spend * (1 + ROI) -> 250 * 3.0 = 750; 500 * 4.0 = 2000
    assert enriched["Revenue"].iloc[0] == 750.0
    assert enriched["Revenue"].iloc[1] == 2000.0

    # CTR = Clicks / Impressions -> 100 / 2000 = 0.05 (5%)
    assert enriched["CTR"].iloc[0] == 0.05
