"""
Unit tests for data validator module.
"""

import pandas as pd
import numpy as np
import pytest
from src.validator import DataValidationError, validate_marketing_data


@pytest.fixture
def valid_marketing_df():
    return pd.DataFrame({
        "Date": ["2023-01-01", "2023-01-02"],
        "Campaign_Type": ["Search", "Email"],
        "Channel_Used": ["Google Ads", "Email"],
        "Clicks": [150, 300],
        "Impressions": [5000, 8000],
        "Acquisition_Cost": [45.0, 20.0],
        "ROI": [2.5, 3.2],
        "Conversion_Rate": [0.05, 0.08]
    })


def test_validate_valid_dataset(valid_marketing_df):
    is_valid, hard_errs, soft_warns = validate_marketing_data(valid_marketing_df)
    assert is_valid is True
    assert len(hard_errs) == 0


def test_validate_missing_columns(valid_marketing_df):
    df_missing = valid_marketing_df.drop(columns=["Clicks", "ROI"])
    with pytest.raises(DataValidationError):
        validate_marketing_data(df_missing, raise_on_error=True)

    is_valid, hard_errs, _ = validate_marketing_data(df_missing, raise_on_error=False)
    assert is_valid is False
    assert any("Missing required columns" in err for err in hard_errs)


def test_validate_negative_values(valid_marketing_df):
    df_neg = valid_marketing_df.copy()
    df_neg.loc[0, "Clicks"] = -10
    with pytest.raises(DataValidationError):
        validate_marketing_data(df_neg, raise_on_error=True)


def test_validate_conversion_rate_bounds(valid_marketing_df):
    df_bad_cr = valid_marketing_df.copy()
    df_bad_cr.loc[0, "Conversion_Rate"] = 150.0  # Exceeds 100%
    with pytest.raises(DataValidationError):
        validate_marketing_data(df_bad_cr, raise_on_error=True)


def test_validate_soft_warnings(valid_marketing_df):
    df_warn = valid_marketing_df.copy()
    df_warn.loc[0, "Clicks"] = 6000  # Clicks > Impressions (5000)
    df_warn.loc[1, "Acquisition_Cost"] = 0.0  # Zero cost
    is_valid, hard_errs, soft_warns = validate_marketing_data(df_warn, raise_on_error=False)
    assert is_valid is True
    assert len(soft_warns) >= 2
