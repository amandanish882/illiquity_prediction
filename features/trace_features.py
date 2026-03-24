"""
trace_features.py -- Engineer TRACE-based features and target variables.

Computes 10 CUSIP-day features from TRACE data plus 4 target variables
(continuous and binary illiquidity measures at 1-day and 5-day horizons).

Features:
  1. spread_bps: volume-weighted average spread vs mid proxy
  2. spread_volatility_5d: rolling 5-day std of spread
  3. log_daily_volume: log(1 + daily notional)
  4. trade_count: number of trades per day
  5. avg_trade_size: average notional per trade
  6. days_since_last_trade: business days since last observed trade
  7. pct_dealer_trades: fraction of interdealer trades
  8. time_to_maturity: years to maturity from maturity_date
  9. coupon: bond coupon rate
  10. is_ig: investment-grade flag

Targets:
  - illiquidity_t1: next-day Amihud ratio (|return|/volume)
  - illiquid_t1: binary flag if illiquidity_t1 > 75th percentile
  - illiquidity_t5: 5-day-ahead average Amihud ratio
  - illiquid_t5: binary flag if illiquidity_t5 > 75th percentile

Reference: Amihud (2002), "Illiquidity and stock returns"
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def _load_config() -> dict:
    """Load config.yaml from project root."""
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _compute_mid_proxy(cusip_day: pd.DataFrame) -> pd.DataFrame:
    """Compute a mid-price proxy for each CUSIP-day.

    Uses the volume-weighted average price (VWAP) as the mid proxy,
    which is standard in TRACE literature when actual bid/ask quotes
    are not available (e.g., Dick-Nielsen et al. 2012).

    Args:
        cusip_day: CUSIP-day aggregated DataFrame with columns
            [cusip, date, price, volume].

    Returns:
        Same DataFrame with added column 'mid_proxy'.
        Example: cusip='38259P508', date='2023-06-15', price=100.25,
        mid_proxy=100.20 (rolling 5-day median for that CUSIP).
    """
    # Mid proxy = rolling 5-day median price per CUSIP
    # This smooths out individual trade noise
    cusip_day = cusip_day.sort_values(["cusip", "date"])
    cusip_day["mid_proxy"] = (
        cusip_day.groupby("cusip")["price"]
        .transform(lambda x: x.rolling(5, min_periods=1).median())
    )
    return cusip_day


def compute_trace_features(
    cusip_day: pd.DataFrame,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """Compute all 10 TRACE-based features for each CUSIP-day.

    Args:
        cusip_day: CUSIP-day aggregated DataFrame from
            data.fetch_trace.aggregate_cusip_day(). Required columns:
            cusip, date, price, yield_pct, volume, trade_count,
            pct_dealer, maturity_date, coupon, rating, sector, is_ig.
        config: Configuration dict (loaded from config.yaml if None).

    Returns:
        DataFrame with original columns plus 10 engineered features:
        spread_bps, spread_volatility_5d, log_daily_volume, trade_count,
        avg_trade_size, days_since_last_trade, pct_dealer_trades,
        time_to_maturity, coupon, is_ig.

    Example output row:
        cusip='38259P508', date='2023-06-15',
        spread_bps=12.5, spread_volatility_5d=3.2, log_daily_volume=14.1,
        trade_count=8, avg_trade_size=625000.0, days_since_last_trade=0,
        pct_dealer_trades=0.25, time_to_maturity=7.3, coupon=4.5, is_ig=1.
    """
    if config is None:
        config = _load_config()

    feat_cfg = config["features"]
    rolling_window = feat_cfg["spread_rolling_window"]  # 5
    staleness_cap = feat_cfg["staleness_cap"]            # 30

    df = cusip_day.copy()
    df = df.sort_values(["cusip", "date"]).reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    logger.info("Computing TRACE features for %d CUSIP-day rows...", len(df))

    # ── Feature 1: spread_bps ────────────────────────────────────────────
    # Spread = |price - mid_proxy| in price points, converted to bps
    # 1 price point ≈ 100 bps for a par bond
    df = _compute_mid_proxy(df)
    df["spread_bps"] = np.abs(df["price"] - df["mid_proxy"]) * 100.0
    # Example: price=100.25, mid=100.20 -> |0.05| * 100 = 5.0 bps

    # ── Feature 2: spread_volatility_5d ──────────────────────────────────
    # Rolling 5-day std of spread_bps per CUSIP
    df["spread_volatility_5d"] = (
        df.groupby("cusip")["spread_bps"]
        .transform(lambda x: x.rolling(rolling_window, min_periods=2).std())
    )
    # Fill NaN for first few days with the CUSIP's overall std
    cusip_spread_std = df.groupby("cusip")["spread_bps"].transform("std")
    df["spread_volatility_5d"] = df["spread_volatility_5d"].fillna(cusip_spread_std)

    # ── Feature 3: log_daily_volume ──────────────────────────────────────
    # log(1 + total daily notional volume)
    # Example: volume=$2M -> log(1+2000000) = 14.51
    df["log_daily_volume"] = np.log1p(df["volume"])

    # ── Feature 4: trade_count ───────────────────────────────────────────
    # Already in cusip_day; rename for clarity
    # (column already named 'trade_count' from aggregation)

    # ── Feature 5: avg_trade_size ────────────────────────────────────────
    # Average notional per trade = total_volume / trade_count
    df["avg_trade_size"] = df["volume"] / df["trade_count"].clip(lower=1)

    # ── Feature 6: days_since_last_trade ─────────────────────────────────
    # Number of business days since the CUSIP last appeared in the panel
    # For the first observation, set to 0
    df["prev_trade_date"] = df.groupby("cusip")["date"].shift(1)
    # np.busday_count cannot handle NaT, so fill NaT with the current date
    # (which produces 0 business days for the first observation per CUSIP)
    prev_dates = df["prev_trade_date"].copy()
    nat_mask = prev_dates.isna()
    prev_dates = prev_dates.fillna(df["date"])
    df["days_since_last_trade"] = np.busday_count(
        prev_dates.values.astype("datetime64[D]"),
        df["date"].values.astype("datetime64[D]"),
    )
    # First observation per CUSIP -> 0 days
    df.loc[nat_mask, "days_since_last_trade"] = 0
    df["days_since_last_trade"] = (
        df["days_since_last_trade"]
        .clip(lower=0, upper=staleness_cap)
        .astype(int)
    )
    df.drop(columns=["prev_trade_date"], inplace=True)

    # ── Feature 7: pct_dealer_trades ─────────────────────────────────────
    # Fraction of interdealer trades
    # Use pct_dealer_proxy from aggregate (based on trade count heuristic)
    # or pct_dealer if available from real TRACE data
    dealer_col = "pct_dealer_proxy" if "pct_dealer_proxy" in df.columns else "pct_dealer"
    if dealer_col in df.columns:
        df["pct_dealer_trades"] = df[dealer_col].fillna(0.25)
    else:
        df["pct_dealer_trades"] = 0.25  # default when no dealer info available

    # ── Feature 8: time_to_maturity ──────────────────────────────────────
    # Years to maturity: use years_to_maturity if available (from OSBAP),
    # otherwise compute from maturity_date
    if "years_to_maturity" in df.columns:
        df["time_to_maturity"] = df["years_to_maturity"].clip(lower=0.0)
    elif "maturity_date" in df.columns:
        mat = pd.to_datetime(df["maturity_date"])
        df["time_to_maturity"] = (mat - df["date"]).dt.days / 365.25
        df["time_to_maturity"] = df["time_to_maturity"].clip(lower=0.0)
    else:
        df["time_to_maturity"] = 5.0  # fallback
    # Example: maturity='2030-06-15', date='2023-06-15' -> 7.0 years

    # ── Feature 9: coupon ────────────────────────────────────────────────
    # Already present from bond static data (no transformation needed)

    # ── Feature 10: is_ig ────────────────────────────────────────────────
    # Already present (1 for IG, 0 for HY)

    # Clean up helper columns
    if "mid_proxy" in df.columns:
        df.drop(columns=["mid_proxy"], inplace=True)
    for col in ["pct_dealer", "pct_dealer_proxy"]:
        if col in df.columns and "pct_dealer_trades" in df.columns:
            df.drop(columns=[col], inplace=True)

    logger.info(
        "TRACE features computed. Spread mean=%.1f bps, volume mean=%.1f (log), "
        "staleness mean=%.1f days",
        df["spread_bps"].mean(),
        df["log_daily_volume"].mean(),
        df["days_since_last_trade"].mean(),
    )

    return df


def compute_target_variables(
    df: pd.DataFrame,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """Compute illiquidity target variables for supervised learning.

    Constructs the Amihud (2002) illiquidity ratio and derives both
    continuous and binary targets at 1-day and 5-day horizons.

    Amihud ratio = |daily return| / daily dollar volume
    This measures price impact per unit of trading volume. Higher values
    indicate that a given trade volume causes larger price movements,
    implying worse liquidity.

    Args:
        df: CUSIP-day DataFrame with at least columns:
            cusip, date, price, volume. Should have TRACE features already.
        config: Configuration dict (loaded from config.yaml if None).

    Returns:
        DataFrame with 4 additional target columns:
        - illiquidity_t1: next-day Amihud ratio (continuous, shifted forward 1 day)
        - illiquid_t1: 1 if illiquidity_t1 > 75th percentile, else 0
        - illiquidity_t5: 5-day-ahead mean Amihud ratio
        - illiquid_t5: 1 if illiquidity_t5 > 75th percentile, else 0

    Example:
        A bond with daily return 0.5% and volume $2M has
        Amihud = 0.005 / 2,000,000 = 2.5e-9.
        If the 75th percentile is 5e-9, this bond is liquid (illiquid_t1=0).
    """
    if config is None:
        config = _load_config()

    binary_pctile = config["model"]["binary_percentile"]  # 75

    out = df.copy()
    out = out.sort_values(["cusip", "date"]).reset_index(drop=True)

    logger.info("Computing Amihud illiquidity targets...")

    # Daily return per CUSIP: |ln(P_t / P_{t-1})|
    out["daily_return"] = (
        out.groupby("cusip")["price"]
        .transform(lambda x: np.abs(np.log(x / x.shift(1))))
    )

    # Amihud ratio: |return| / volume
    # Add small epsilon to volume to avoid division by zero
    out["amihud"] = out["daily_return"] / (out["volume"] + 1.0)

    # ── illiquidity_t1: next-day Amihud ──────────────────────────────────
    # Shift forward by 1 within each CUSIP (we predict tomorrow's illiquidity)
    out["illiquidity_t1"] = (
        out.groupby("cusip")["amihud"]
        .shift(-1)
    )

    # ── illiquidity_t5: 5-day-ahead average Amihud ──────────────────────
    # Rolling mean of next 5 days' Amihud ratios
    out["illiquidity_t5"] = (
        out.groupby("cusip")["amihud"]
        .transform(
            lambda x: x[::-1].rolling(5, min_periods=1).mean()[::-1]
        )
        .groupby(out["cusip"]).shift(-1)
    )

    # Drop rows where targets are NaN (last days of each CUSIP)
    n_before = len(out)
    valid_mask = out["illiquidity_t1"].notna()
    logger.info(
        "Target coverage: %d / %d rows have valid t1 target (%.1f%%)",
        valid_mask.sum(), n_before, valid_mask.sum() / n_before * 100,
    )

    # ── Binary targets: percentile thresholds ────────────────────────────
    # Compute threshold cross-sectionally (across all CUSIP-days)
    t1_threshold = out["illiquidity_t1"].quantile(binary_pctile / 100.0)
    t5_threshold = out["illiquidity_t5"].quantile(binary_pctile / 100.0)

    out["illiquid_t1"] = (out["illiquidity_t1"] > t1_threshold).astype(int)
    out["illiquid_t5"] = (out["illiquidity_t5"] > t5_threshold).astype(int)

    # Clean up intermediate columns
    out.drop(columns=["daily_return", "amihud"], inplace=True)

    logger.info(
        "Targets: t1 threshold=%.2e (75th pctile), t5 threshold=%.2e. "
        "Positive rate: t1=%.1f%%, t5=%.1f%%",
        t1_threshold, t5_threshold,
        out["illiquid_t1"].mean() * 100,
        out["illiquid_t5"].mean() * 100,
    )

    return out


def build_trace_feature_panel(
    cusip_day: pd.DataFrame,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """End-to-end pipeline: compute all TRACE features + targets.

    This is the main entry point. Takes aggregated CUSIP-day TRACE data
    and returns a full feature panel ready for model training.

    Args:
        cusip_day: CUSIP-day aggregated DataFrame from
            data.fetch_trace.aggregate_cusip_day().
        config: Configuration dict.

    Returns:
        DataFrame with 10 features + 4 targets per CUSIP-day row.
    """
    if config is None:
        config = _load_config()

    df = compute_trace_features(cusip_day, config)
    df = compute_target_variables(df, config)

    logger.info(
        "TRACE feature panel: %d rows, %d columns, %d CUSIPs",
        len(df), len(df.columns), df["cusip"].nunique(),
    )

    return df
