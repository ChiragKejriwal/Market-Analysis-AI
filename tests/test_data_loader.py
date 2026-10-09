"""
Unit tests for data loader module.
"""

import pandas as pd
import pytest
from src.data_loader import load_marketing_data, generate_synthetic_marketing_data
from src.validator import validate_marketing_data


def test_generate_synthetic_data():
    df = generate_synthetic_marketing_data(n_rows=500, seed=123)
    assert len(df) == 500
    is_valid, hard_errs, _ = validate_marketing_data(df)
    assert is_valid is True
    assert (df["Clicks"] <= df["Impressions"]).all()
    assert (df["Conversion_Rate"] >= 0).all()
    assert (df["Conversion_Rate"] <= 1.0).all()


def test_load_marketing_data_fallback():
    df = load_marketing_data()
    assert not df.empty
    assert len(df) >= 500
    assert "Date" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    # Check chronological ordering
    assert df["Date"].is_monotonic_increasing
