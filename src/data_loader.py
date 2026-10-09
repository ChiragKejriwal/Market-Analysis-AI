"""
Data Loader Module for Causal Marketing Analytics Platform.
Handles ingestion, fallback dataset discovery, realistic synthetic generation, and validation.
"""

from pathlib import Path
from typing import Optional, Union
import logging
import numpy as np
import pandas as pd
from src.validator import validate_marketing_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_marketing_data(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic marketing campaign dataset matching the Kaggle/industry schema.
    Ensures realistic CTR, conversion rates, channel costs, and seasonal trends.

    Args:
        n_rows: Number of campaign records to generate (default 5,000).
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame: Synthetic marketing dataset.
    """
    rng = np.random.default_rng(seed)

    companies = ["Nexus Commerce", "Apex Retail", "Veritas Health", "Starlight Media", "Horizon Tech", "Pulse Fashion"]
    campaign_types = ["Search", "Social Media", "Email", "Display", "Influencer", "Video"]
    channels = ["Google Ads", "Facebook", "Email", "YouTube", "Organic Search", "Instagram"]
    audiences = ["Men 18-24", "Women 25-34", "All 35-50", "Tech Professionals", "Gen Z Students", "Affluent Homeowners"]
    durations = ["15 Days", "30 Days", "45 Days", "60 Days"]
    locations = ["New York", "California", "Texas", "Florida", "Illinois", "London", "Toronto", "Sydney"]
    languages = ["English", "Spanish", "French", "German"]

    # Generate dates across 18 months
    start_date = pd.Timestamp("2023-01-01")
    date_offsets = rng.integers(0, 540, size=n_rows)
    dates = [start_date + pd.Timedelta(days=int(d)) for d in date_offsets]

    # Impressions: Lognormal distribution
    impressions = rng.lognormal(mean=9.2, sigma=0.8, size=n_rows).astype(int)
    impressions = np.clip(impressions, 500, 150000)

    # Click-through rate: Channel dependent
    channel_choices = rng.choice(channels, size=n_rows, p=[0.30, 0.25, 0.15, 0.15, 0.08, 0.07])
    
    ctr_map = {
        "Google Ads": (0.035, 0.010),
        "Facebook": (0.025, 0.008),
        "Email": (0.050, 0.015),
        "YouTube": (0.018, 0.006),
        "Organic Search": (0.040, 0.012),
        "Instagram": (0.028, 0.009),
    }

    clicks = []
    for imp, ch in zip(impressions, channel_choices):
        mean_ctr, std_ctr = ctr_map.get(ch, (0.03, 0.01))
        sim_ctr = max(0.005, min(0.15, rng.normal(mean_ctr, std_ctr)))
        click_count = max(5, int(imp * sim_ctr))
        click_count = min(click_count, imp)  # Guarantee clicks <= impressions
        clicks.append(click_count)
    clicks = np.array(clicks)

    # Conversion Rate: Decimal representation (0.01 to 0.12)
    conv_rates = rng.beta(a=2.5, b=35, size=n_rows)
    conv_rates = np.clip(conv_rates, 0.01, 0.15).round(4)

    # Acquisition Cost: Average cost per acquisition ($10 - $180)
    acq_costs = rng.gamma(shape=3.0, scale=18.0, size=n_rows).round(2)
    acq_costs = np.clip(acq_costs, 8.50, 220.00)

    # ROI: Return on marketing investment (0.2 to 5.0)
    base_rois = rng.normal(loc=2.4, scale=0.9, size=n_rows)
    rois = np.clip(base_rois, 0.15, 6.50).round(2)

    # Engagement score (1.0 to 10.0)
    engagement = rng.uniform(2.5, 9.8, size=n_rows).round(1)

    df = pd.DataFrame({
        "Campaign_ID": [f"CMP_{i:05d}" for i in range(1, n_rows + 1)],
        "Company": rng.choice(companies, size=n_rows),
        "Campaign_Type": rng.choice(campaign_types, size=n_rows),
        "Target_Audience": rng.choice(audiences, size=n_rows),
        "Duration": rng.choice(durations, size=n_rows),
        "Channel_Used": channel_choices,
        "Conversion_Rate": conv_rates,
        "Acquisition_Cost": acq_costs,
        "ROI": rois,
        "Location": rng.choice(locations, size=n_rows),
        "Language": rng.choice(languages, size=n_rows),
        "Clicks": clicks,
        "Impressions": impressions,
        "Engagement_Score": engagement,
        "Date": dates
    })

    return df


def resolve_data_path(custom_path: Optional[Union[str, Path]] = None) -> Optional[Path]:
    """Resolves data file path by checking custom path, default filename, or looking for any .csv in data/."""
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            return p

    search_dirs = [
        Path(__file__).resolve().parent.parent / "data",
        Path("data").resolve(),
        Path("marketing-analytics/data").resolve(),
    ]

    for d in search_dirs:
        if d.is_dir():
            # Check user dataset first
            user_dataset = d / "marketing_campaign_dataset.csv"
            if user_dataset.is_file():
                return user_dataset

            default_file = d / "marketing_campaign_data.csv"
            if default_file.is_file():
                return default_file

            # Look for any .csv in directory
            csv_files = list(d.glob("*.csv"))
            if csv_files:
                return csv_files[0]

    return None


def load_marketing_data(filepath: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """
    Loads, validates, and prepares marketing campaign data.

    Workflow:
    1. Loads dataset from specified filepath, default 'data/marketing_campaign_dataset.csv' or 'data/marketing_campaign_data.csv'.
    2. If no dataset is found on disk, generates a 5,000-row synthetic dataset and saves it to data/.
    3. Cleans currency/string formatting in numeric columns.
    4. Validates structure and domain rules via validate_marketing_data.
    5. Parses 'Date' column to datetime and sorts chronologically.

    Args:
        filepath: Optional path to dataset CSV.

    Returns:
        pd.DataFrame: Cleaned, validated, and chronologically sorted DataFrame.
    """
    resolved_path = resolve_data_path(filepath)

    if resolved_path and resolved_path.is_file():
        logger.info(f"Loading marketing dataset from: {resolved_path}")
        try:
            df = pd.read_csv(resolved_path)
        except Exception as e:
            logger.warning(f"Error reading CSV at {resolved_path}: {e}. Generating fallback dataset.")
            df = None
    else:
        df = None

    # Fallback to synthetic dataset generation
    if df is None or df.empty:
        logger.info("No valid CSV found on disk. Generating 5,000-row realistic synthetic marketing dataset.")
        df = generate_synthetic_marketing_data(n_rows=5000)

        # Attempt to save to disk for future sessions
        target_dir = Path(__file__).resolve().parent.parent / "data"
        target_dir.mkdir(parents=True, exist_ok=True)
        save_path = target_dir / "marketing_campaign_dataset.csv"
        try:
            df.to_csv(save_path, index=False)
            logger.info(f"Saved synthetic dataset to: {save_path}")
        except Exception as e:
            logger.warning(f"Could not write synthetic dataset to {save_path}: {e}")

    df = df.copy()

    # Clean currency / string symbols in numeric columns if present
    if "Acquisition_Cost" in df.columns and (df["Acquisition_Cost"].dtype == object or isinstance(df["Acquisition_Cost"].iloc[0] if len(df) > 0 else None, str)):
        cleaned_acq = df["Acquisition_Cost"].astype(str).str.replace(r"[\$,€£]", "", regex=True).str.replace(",", "", regex=False).str.strip()
        df["Acquisition_Cost"] = pd.to_numeric(cleaned_acq, errors="coerce").fillna(0.0)

    # Validate dataset structure and integrity
    validate_marketing_data(df, raise_on_error=True)

    # Parse 'Date' column to datetime and sort chronologically
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Sort chronologically and reset index
    df = df.sort_values(by="Date").reset_index(drop=True)

    logger.info(f"Successfully loaded and validated {len(df):,} marketing campaign records.")
    return df
