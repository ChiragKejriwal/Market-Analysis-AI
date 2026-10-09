import pytest
import pandas as pd
from src.ui_components import get_sample_template_df, inspect_uploaded_df, MANDATORY_COLUMNS


def test_get_sample_template_df():
    df = get_sample_template_df()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 5
    for col in MANDATORY_COLUMNS:
        assert col in df.columns


def test_inspect_uploaded_df_valid():
    df = get_sample_template_df()
    is_valid, mapped_df, hard_errs, soft_warns, col_map = inspect_uploaded_df(df)
    assert is_valid is True
    assert len(hard_errs) == 0
    assert mapped_df is not None


def test_inspect_uploaded_df_synonyms():
    df = get_sample_template_df().rename(columns={
        "Channel_Used": "media_channel",
        "Acquisition_Cost": "cpa",
        "Conversion_Rate": "cvr",
        "Date": "timestamp"
    })
    is_valid, mapped_df, hard_errs, soft_warns, col_map = inspect_uploaded_df(df)
    assert is_valid is True
    assert "Channel_Used" in mapped_df.columns
    assert "Acquisition_Cost" in mapped_df.columns
    assert "Conversion_Rate" in mapped_df.columns
    assert "Date" in mapped_df.columns


def test_inspect_uploaded_df_missing_mandatory():
    df = pd.DataFrame({
        "Campaign_Type": ["Search"],
        "Clicks": [100]
    })
    is_valid, mapped_df, hard_errs, soft_warns, col_map = inspect_uploaded_df(df)
    assert is_valid is False
    assert len(hard_errs) > 0
