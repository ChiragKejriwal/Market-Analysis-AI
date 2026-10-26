"""
Data Validation Module for Causal Marketing Analytics Platform.
Validates required columns, value boundaries, and business integrity rules.
"""

from typing import List, Tuple
import pandas as pd


class DataValidationError(Exception):
    """Raised when marketing dataset fails critical structural or domain validation checks."""
    pass


REQUIRED_COLUMNS = [
    "Date",
    "Campaign_Type",
    "Channel_Used",
    "Clicks",
    "Impressions",
    "Acquisition_Cost",
    "ROI",
    "Conversion_Rate"
]


def validate_marketing_data(
    df: pd.DataFrame,
    raise_on_error: bool = True
) -> Tuple[bool, List[str], List[str]]:
    """
    Validates marketing campaign data against schema, data integrity, and logical domain rules.
    """
    hard_errors: List[str] = []
    soft_warnings: List[str] = []

    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        hard_errors.append("Dataset is empty or not a valid pandas DataFrame.")
        if raise_on_error:
            raise DataValidationError("; ".join(hard_errors))
        return False, hard_errors, soft_warnings

    df_cols = [str(c).strip() for c in df.columns]
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df_cols]
    if missing_cols:
        hard_errors.append(f"Missing required columns: {missing_cols}")
        if raise_on_error:
            raise DataValidationError("; ".join(hard_errors))
        return False, hard_errors, soft_warnings

    cols_map = {str(c).strip(): c for c in df.columns}

    def _to_clean_numeric(s: pd.Series) -> pd.Series:
        if s.dtype == object or isinstance(s.iloc[0] if len(s) > 0 else None, str):
            cleaned = s.astype(str).str.replace(r"[\$,€£]", "", regex=True).str.replace(",", "", regex=False).str.strip()
            return pd.to_numeric(cleaned, errors="coerce")
        return pd.to_numeric(s, errors="coerce")

    for col in ["Clicks", "Impressions", "Acquisition_Cost"]:
        actual_col = cols_map[col]
        numeric_series = _to_clean_numeric(df[actual_col])
        neg_count = (numeric_series < 0).sum()
        if neg_count > 0:
            hard_errors.append(f"Found {neg_count} negative value(s) in '{col}'. Values must be non-negative.")

    cr_col = cols_map["Conversion_Rate"]
    cr_series = _to_clean_numeric(df[cr_col])
    neg_cr_count = (cr_series < 0).sum()
    if neg_cr_count > 0:
        hard_errors.append(f"Found {neg_cr_count} negative value(s) in 'Conversion_Rate'.")

    max_cr = cr_series.dropna().max() if not cr_series.dropna().empty else 0.0
    if max_cr > 100.0:
        hard_errors.append(f"Maximum Conversion_Rate is {max_cr:.2f}, exceeding the absolute upper bound of 100%.")
    elif max_cr > 1.0:
        invalid_cr_count = (cr_series > 100.0).sum()
        if invalid_cr_count > 0:
            hard_errors.append(f"Found {invalid_cr_count} records where Conversion_Rate > 100.0%.")

    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    if total_nulls > 0:
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        soft_warnings.append(f"Dataset contains {total_nulls} missing/null values across columns: {cols_with_nulls}")

    clicks = _to_clean_numeric(df[cols_map["Clicks"]]).fillna(0)
    impressions = _to_clean_numeric(df[cols_map["Impressions"]]).fillna(0)
    overclick_count = (clicks > impressions).sum()
    if overclick_count > 0:
        soft_warnings.append(f"Found {overclick_count} record(s) where Clicks exceed Impressions (CTR > 100%).")

    costs = _to_clean_numeric(df[cols_map["Acquisition_Cost"]]).fillna(0)
    zero_cost_count = (costs == 0).sum()
    if zero_cost_count > 0:
        soft_warnings.append(f"Found {zero_cost_count} record(s) with zero Acquisition_Cost.")

    is_valid = len(hard_errors) == 0
    if not is_valid and raise_on_error:
        raise DataValidationError("; ".join(hard_errors))

    return is_valid, hard_errors, soft_warnings
