"""
Unit tests for acquisition cohorts module.
"""

import pandas as pd
import pytest
from src.cohort import build_acquisition_cohorts, generate_cohort_matrix


@pytest.fixture
def cohort_test_data():
    return pd.DataFrame({
        "Campaign_ID": [f"CMP_{i}" for i in range(6)],
        "Company": ["TechCorp", "TechCorp", "TechCorp", "FashionInc", "FashionInc", "FashionInc"],
        "Date": [
            "2023-01-15", "2023-02-10", "2023-03-05",
            "2023-02-20", "2023-03-15", "2023-04-10"
        ],
        "Clicks": [500, 400, 350, 600, 550, 500],
        "Conversions": [50, 40, 30, 60, 50, 45],
        "Spend": [1000.0, 800.0, 700.0, 1200.0, 1000.0, 900.0],
        "Revenue": [3000.0, 2400.0, 2100.0, 3600.0, 3000.0, 2700.0]
    })


def test_build_acquisition_cohorts(cohort_test_data):
    cohort_df = build_acquisition_cohorts(cohort_test_data)
    assert not cohort_df.empty
    assert "CohortMonth" in cohort_df.columns
    assert "Period_Index" in cohort_df.columns
    assert "Cumulative_Revenue" in cohort_df.columns
    assert "Conversion_Rate" in cohort_df.columns


def test_generate_cohort_matrix(cohort_test_data):
    rev_matrix = generate_cohort_matrix(cohort_test_data, metric="revenue")
    assert not rev_matrix.empty
    assert "M+0" in rev_matrix.columns

    ret_matrix = generate_cohort_matrix(cohort_test_data, metric="retention")
    assert not ret_matrix.empty
    assert "M+0" in ret_matrix.columns
    assert (ret_matrix["M+0"] == 100.0).all()
