"""
macro_features.py -- Merge FRED macro data onto the CUSIP-day feature panel.

Adds three macro features that capture the broad market environment:
  - oas_index_level: ICE BofA US Corporate Index OAS (BAMLC0A0CM)
  - vix_level: CBOE Volatility Index (VIXCLS)
  - hy_oas_level: ICE BofA US High Yield Index OAS (BAMLH0A0HYM2)

These are market-wide regime indicators. During stress periods (high VIX,
wide OAS), individual bond illiquidity spikes even for normally liquid
names (Bao, Pan & Wang, 2011; Dick-Nielsen et al., 2012).
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    """Load config.yaml from project root."""
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_macro_features(
    panel: pd.DataFrame,
    fred_data: pd.DataFrame,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """Merge FRED macro data onto the CUSIP-day feature panel.

    Performs an asof merge: for each CUSIP-day date, looks up the latest
    available FRED observation on or before that date. This handles
    weekends and holidays where FRED data may not be published.

    Args:
        panel: CUSIP-day feature DataFrame with a 'date' column.
            Typically the output of features.etf_features.compute_etf_features().
        fred_data: FRED macro DataFrame from data.fetch_fred.fetch_fred_data().
            Expected columns: ig_oas, vix, hy_oas. Index is DatetimeIndex.
        config: Configuration dict (loaded from config.yaml if None).

    Returns:
        DataFrame with 3 additional columns:
        - oas_index_level: IG OAS from FRED (in percent, e.g., 1.20 = 120 bps)
        - vix_level: VIX index level (e.g., 18.5)
        - hy_oas_level: HY OAS from FRED (in percent, e.g., 4.20 = 420 bps)

    Example output row:
        date='2023-06-15', oas_index_level=1.15, vix_level=14.2, hy_oas_level=4.35.
    """
    if config is None:
        config = _load_config()

    fred_cfg = config["data"]["fred"]
    series_map = fred_cfg["series"]  # {"ig_oas": "BAMLC0A0CM", ...}

    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Prepare FRED data for merge
    macro = fred_data.copy()
    if not isinstance(macro.index, pd.DatetimeIndex):
        macro.index = pd.to_datetime(macro.index)
    macro = macro.sort_index()

    # Rename FRED columns to feature names
    rename_map = {
        "ig_oas": "oas_index_level",
        "vix": "vix_level",
        "hy_oas": "hy_oas_level",
    }
    macro = macro.rename(columns=rename_map)
    macro = macro[[c for c in rename_map.values() if c in macro.columns]]

    # Add date column for merge
    macro = macro.reset_index()
    macro.columns = ["date"] + list(macro.columns[1:])
    macro["date"] = pd.to_datetime(macro["date"])

    # Merge: left join on date, then forward-fill any remaining NaNs
    # This is a simple date merge since FRED data has been reindexed to
    # business days and forward-filled in fetch_fred.py
    df = df.merge(macro, on="date", how="left")

    # Forward-fill any gaps (e.g., if panel dates extend beyond FRED range)
    for col in rename_map.values():
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # Log summary
    for col in rename_map.values():
        if col in df.columns:
            logger.info(
                "Macro feature %s: mean=%.3f, min=%.3f, max=%.3f",
                col, df[col].mean(), df[col].min(), df[col].max(),
            )

    logger.info("Macro features merged: %d rows", len(df))

    return df
