"""
Unit tests for diminishing returns response modeling and marginal ROI module.
"""

import pandas as pd
import numpy as np
import pytest
from src.roi import fit_channel_response_curves, estimate_marginal_roi, log_response_curve
from src.data_loader import generate_synthetic_marketing_data
from src.preprocessing import clean_and_enrich_data


@pytest.fixture
def sample_roi_data():
    raw_df = generate_synthetic_marketing_data(n_rows=800, seed=42)
    return clean_and_enrich_data(raw_df)


def test_fit_channel_response_curves(sample_roi_data):
    curves = fit_channel_response_curves(sample_roi_data)
    assert len(curves) > 0
    for channel, params in curves.items():
        assert "alpha" in params
        assert "beta" in params
        assert "historical_spend" in params
        assert "r_squared" in params
        assert params["alpha"] > 0
        assert params["beta"] > 0


def test_estimate_marginal_roi(sample_roi_data):
    curves = fit_channel_response_curves(sample_roi_data)
    channels = list(curves.keys())
    assert len(channels) > 0

    first_channel = channels[0]
    m_roi_low = estimate_marginal_roi(first_channel, current_spend=1000.0, curve_params=curves)
    m_roi_high = estimate_marginal_roi(first_channel, current_spend=50000.0, curve_params=curves)

    assert m_roi_low > 0
    # Diminishing returns: Marginal ROI at low spend is greater than at high spend
    assert m_roi_low >= m_roi_high


def test_log_response_curve():
    val = log_response_curve(1000.0, alpha=500.0, beta=0.002)
    assert val > 0
    assert np.isclose(val, 500.0 * np.log1p(0.002 * 1000.0))
