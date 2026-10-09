"""
Unit tests for conversion funnel module.
"""

import pandas as pd
import pytest
from src.funnel import build_funnel_stages, get_channel_funnel_comparison


@pytest.fixture
def funnel_test_data():
    return pd.DataFrame({
        "Campaign_ID": ["CMP_A", "CMP_B"],
        "Channel_Used": ["Google Ads", "Facebook"],
        "Campaign_Type": ["Search", "Social"],
        "Impressions": [50000, 30000],
        "Clicks": [2500, 1200],
        "Conversions": [200, 90],
        "Spend": [3000.0, 1500.0],
        "Revenue": [9000.0, 4000.0],
        "ROI": [2.0, 1.66],
        "Engagement_Score": [8.5, 6.2]
    })


def test_build_funnel_stages(funnel_test_data):
    funnel = build_funnel_stages(funnel_test_data)
    assert len(funnel) == 4
    assert funnel["Stage"].iloc[0] == "1. Impressions (Awareness)"
    assert funnel["Stage"].iloc[1] == "2. Clicks (Traffic & Intent)"
    assert funnel["Stage"].iloc[2] == "3. Conversions (Acquisitions)"
    assert funnel["Stage"].iloc[3] == "4. Repeat/High-Value (Retention)"

    # Monotonic decreasing counts
    assert funnel["Count"].iloc[0] >= funnel["Count"].iloc[1]
    assert funnel["Count"].iloc[1] >= funnel["Count"].iloc[2]
    assert funnel["Count"].iloc[2] >= funnel["Count"].iloc[3]

    assert "Step_Conversion_Rate" in funnel.columns
    assert "Drop_Off_Rate" in funnel.columns


def test_build_funnel_stages_with_filter(funnel_test_data):
    funnel = build_funnel_stages(funnel_test_data, filter_dict={"Channel_Used": "Google Ads"})
    assert funnel["Count"].iloc[0] == 50000
    assert funnel["Count"].iloc[1] == 2500


def test_get_channel_funnel_comparison(funnel_test_data):
    comparison = get_channel_funnel_comparison(funnel_test_data)
    assert len(comparison) == 2
    assert "Impression_to_Click_CTR" in comparison.columns
    assert "Click_to_Conversion_CVR" in comparison.columns
    assert "End_to_End_CVR" in comparison.columns
