"""
etf_features.py -- Compute ETF-based features for each CUSIP-day.

For each CUSIP-day in the panel, computes:
  - etf_basket_flag: 1 if the CUSIP is held in LQD or HYG, 0 otherwise
  - etf_weight: weight in the ETF (0 if not held)
  - etf_overlap_count: number of ETFs holding this bond (from overlap list)

These features capture the "ETF liquidity channel": bonds held in large ETFs
tend to be more liquid due to creation/redemption arbitrage flows
(Ben-David, Franzoni & Moussawi, 2018; Pan & Zeng, 2019).
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


def compute_etf_features(
    panel: pd.DataFrame,
    holdings: pd.DataFrame,
    overlap: pd.DataFrame,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """Merge ETF holding information onto the CUSIP-day feature panel.

    Args:
        panel: CUSIP-day feature DataFrame with at least column 'cusip'.
            Typically the output of features.trace_features.build_trace_feature_panel().
        holdings: ETF holdings DataFrame with columns [cusip, ticker, weight].
            From data.fetch_etf_holdings.fetch_etf_holdings().
        overlap: ETF overlap DataFrame with columns [cusip, etf_overlap_count].
            From data.fetch_etf_holdings.compute_overlap_counts().
        config: Configuration dict (loaded from config.yaml if None).

    Returns:
        DataFrame with 3 additional columns:
        - etf_basket_flag: 1 if cusip in LQD or HYG, else 0
        - etf_weight: total weight across ETFs (e.g., 0.0032 if in LQD with that weight)
        - etf_overlap_count: number of ETFs holding this bond (0-6)

    Example output row:
        cusip='38259P508', etf_basket_flag=1, etf_weight=0.0032,
        etf_overlap_count=3.
    """
    if config is None:
        config = _load_config()

    df = panel.copy()

    # Aggregate holdings: total weight per CUSIP across all ETFs
    # A bond in both LQD and HYG gets the sum of its weights
    cusip_weights = (
        holdings.groupby("cusip")["weight"]
        .sum()
        .reset_index()
        .rename(columns={"weight": "etf_weight"})
    )

    # ETF basket flag: 1 if in any ETF
    cusip_in_etf = set(holdings["cusip"].unique())

    # Merge weight
    df = df.merge(cusip_weights, on="cusip", how="left")
    df["etf_weight"] = df["etf_weight"].fillna(0.0)

    # Basket flag
    df["etf_basket_flag"] = df["cusip"].isin(cusip_in_etf).astype(int)

    # Merge overlap count
    df = df.merge(overlap[["cusip", "etf_overlap_count"]], on="cusip", how="left")
    df["etf_overlap_count"] = df["etf_overlap_count"].fillna(0).astype(int)

    n_in_etf = df["etf_basket_flag"].sum()
    n_total = len(df)
    logger.info(
        "ETF features: %d / %d CUSIP-day rows in ETF basket (%.1f%%), "
        "mean overlap count %.1f",
        n_in_etf, n_total, n_in_etf / n_total * 100,
        df["etf_overlap_count"].mean(),
    )

    return df
