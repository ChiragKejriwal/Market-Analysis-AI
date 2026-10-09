"""
Unit tests for marketing metrics, KPI calculations, and Customer Lifetime Value (CLV) models.
Verifies analytical formulas against exact manual calculations.
"""

import numpy as np
import pandas as pd
import pytest
from src.metrics import calculate_global_kpis, calculate_channel_kpis, calculate_clv, compute_time_series_metrics


@pytest.fixture
def manual_verification_df():
    """
    Deterministic dataset designed with known manual financial and conversion metrics:
    Channel A: Spend $10,000, Revenue $30,000, Conversions 200, Clicks 4,000, Impressions 100,000
    Channel B: Spend $5,000, Revenue $10,000, Conversions 100, Clicks 2,000, Impressions 50,000
    Totals: Spend $15,000, Revenue $40,000, Conversions 300, Clicks 6,000, Impressions 150,000
    """
    return pd.DataFrame({
        "Campaign_ID": ["CMP_A", "CMP_B"],
        "Channel_Used": ["Google Ads", "Facebook"],
        "Spend": [10000.0, 5000.0],
        "Revenue": [30000.0, 10000.0],
        "Conversions": [200, 100],
        "Clicks": [4000, 2000],
        "Impressions": [100000, 50000],
        "Date": ["2024-01-01", "2024-01-08"]
    })


def test_calculate_global_kpis_manual_formulas(manual_verification_df):
    """Verifies global KPI calculations match exact manual mathematical formulas."""
    kpis = calculate_global_kpis(manual_verification_df)

    # 1. Total Spend = 10000 + 5000 = 15000.00
    assert kpis["Total Spend"] == 15000.0

    # 2. Total Revenue = 30000 + 10000 = 40000.00
    assert kpis["Total Revenue"] == 40000.0

    # 3. Blended ROAS = 40000 / 15000 = 2.6666... -> 2.67
    assert kpis["Blended ROAS"] == pytest.approx(40000.0 / 15000.0, rel=1e-2)

    # 4. Blended ROI = (40000 - 15000) / 15000 = 1.666... -> 1.67
    assert kpis["Blended ROI"] == pytest.approx(25000.0 / 15000.0, rel=1e-2)

    # 5. Blended CAC = 15000 / 300 = 50.00
    assert kpis["Blended CAC"] == 50.0

    # 6. Overall CTR = 6000 / 150000 = 0.04 (4.0%)
    assert kpis["Overall CTR"] == 0.04

    # 7. Conversion Rate = 300 / 6000 = 0.05 (5.0%)
    assert kpis["Conversion Rate"] == 0.05


def test_calculate_channel_kpis_manual_formulas(manual_verification_df):
    """Verifies channel-specific aggregation, CAC, ROAS, ROI, and AOV formulas."""
    ch_kpis = calculate_channel_kpis(manual_verification_df)

    # Google Ads verification
    g_row = ch_kpis[ch_kpis["Channel_Used"] == "Google Ads"].iloc[0]
    assert g_row["Spend"] == 10000.0
    assert g_row["Revenue"] == 30000.0
    assert g_row["Conversions"] == 200
    assert g_row["CAC"] == 10000.0 / 200.0  # $50.00
    assert g_row["ROAS"] == 30000.0 / 10000.0  # 3.00x
    assert g_row["ROI"] == (30000.0 - 10000.0) / 10000.0  # 2.00x
    assert g_row["AOV"] == 30000.0 / 200.0  # $150.00

    # Facebook verification
    f_row = ch_kpis[ch_kpis["Channel_Used"] == "Facebook"].iloc[0]
    assert f_row["Spend"] == 5000.0
    assert f_row["Revenue"] == 10000.0
    assert f_row["Conversions"] == 100
    assert f_row["CAC"] == 5000.0 / 100.0  # $50.00
    assert f_row["ROAS"] == 10000.0 / 5000.0  # 2.00x
    assert f_row["ROI"] == 1.00  # 1.00x
    assert f_row["AOV"] == 10000.0 / 100.0  # $100.00


def test_calculate_clv_manual_formula(manual_verification_df):
    """
    Verifies CLV = (AOV * Purchase_Frequency * Margin) / Churn
    and CLV / CAC ratio against manual calculation.
    """
    margin = 0.40
    churn = 0.15
    fixed_freq = 2.5

    clv_df = calculate_clv(
        manual_verification_df,
        margin_rate=margin,
        churn_rate=churn,
        purchase_frequency=fixed_freq
    )

    g_clv = clv_df[clv_df["Channel_Used"] == "Google Ads"].iloc[0]
    # AOV = 150.00
    # Expected CLV = (150 * 2.5 * 0.40) / 0.15 = 150.0 / 0.15 = 1000.00
    expected_g_clv = (150.0 * 2.5 * 0.40) / 0.15
    assert g_clv["CLV"] == pytest.approx(expected_g_clv, rel=1e-2)

    # CAC = 50.00
    # Expected CLV/CAC = 1000 / 50 = 20.00
    assert g_clv["CLV_to_CAC_Ratio"] == pytest.approx(expected_g_clv / 50.0, rel=1e-2)
    assert g_clv["Unit_Economics_Health"] == "Strong (>= 3.0x)"


def test_empty_dataframe_kpis():
    """Ensures empty dataframe handling is robust and returns zeroed metrics."""
    empty_df = pd.DataFrame()
    kpis = calculate_global_kpis(empty_df)
    assert kpis["Total Spend"] == 0.0
    assert kpis["Blended ROAS"] == 0.0
    assert kpis["Blended CAC"] == 0.0

    ch_kpis = calculate_channel_kpis(empty_df)
    assert ch_kpis.empty
