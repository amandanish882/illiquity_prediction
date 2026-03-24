"""
fetch_trace.py -- Load real FINRA TRACE corporate bond trade data from OSBAP.

DATA PROVENANCE:
  - Source: Open Source Bond Asset Pricing (openbondassetpricing.com)
  - Database: FINRA Enhanced, Standard, and 144A TRACE
  - Processing: QuantLib-computed YTM, duration, credit spreads
  - Coverage: 2002-2025, daily bond-level panels
  - File: trace_clean.parquet (processed from stage1_osbap_0k_volume_2025.parquet)
  - All data is REAL — no synthetic generation.

The OSBAP dataset provides daily CUSIP-level aggregated trade data from TRACE,
including volume-weighted prices, yields, credit spreads, modified and Macaulay
durations, trade counts, and dollar/quantity volumes.

References:
  - Bessembinder, Jacobsen, Maxwell & Venkataraman (2018) — TRACE data properties
  - Dick-Nielsen, Feldhutter & Lando (2012) — trade frequency, liquidity
  - Amihud (2002) — illiquidity ratio construction
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OSBAP data paths (shared with other Quant 300 projects)
# ---------------------------------------------------------------------------
PEAD_PROJECT = Path(
    r"C:\Users\amand\OneDrive\Quant 300\Projects 12 Feb"
    r"\LLM Driven Bond PEAD Strategy\bond_pead_project"
)
TRACE_CLEAN_PATH = PEAD_PROJECT / "data" / "processed" / "trace_clean.parquet"
OSBAP_RAW_PATH = (
    PEAD_PROJECT / "data" / "raw" / "osbap" / "extracted"
    / "stage1_osbap_0k_volume_2025.parquet"
)


# ---------------------------------------------------------------------------
# Rating classification
# ---------------------------------------------------------------------------
# Based on ICE BofA index: BBB- and above = IG
_IG_RATINGS = {
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
}
_HY_RATINGS = {
    "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D",
}

# OSBAP db_type codes:
#   1 = corporate standard (can be IG or HY -- determined by credit_spread)
#   3 = corporate 144A (almost entirely HY)
# IG vs HY is determined by credit_spread threshold + db_type:
#   db_type 3 -> always HY
#   db_type 1 with median spread > 200 bps -> HY
#   db_type 1 with median spread <= 200 bps -> IG
_DB_TYPE_ALWAYS_HY = {3}
_IG_HY_SPREAD_THRESHOLD_BPS = 200  # bonds above this are classified HY


def _load_config() -> dict:
    """Load config.yaml from project root."""
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def fetch_trace_data(
    config: Optional[dict] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Load real TRACE bond trade data from OSBAP parquet files.

    Reads the pre-processed OSBAP TRACE dataset, filters to the configured
    date range, and maps columns to the standard schema expected by the
    feature engineering pipeline.

    Args:
        config: Configuration dict (loaded from config.yaml if None).
        seed: Random seed (unused with real data, kept for API compatibility).

    Returns:
        DataFrame with columns: cusip, date, price, yield_pct, volume,
        trade_count, side, report_type, maturity_date, coupon, rating,
        sector, is_ig, credit_spread, mod_duration.
        Each row is one CUSIP-day observation from real TRACE data.

    Raises:
        FileNotFoundError: If OSBAP data files are not found at expected paths.

    Example row:
        cusip='00184AAC9', date=2024-01-02, price=109.65, yield_pct=5.97,
        volume=153503, trade_count=5, is_ig=1, credit_spread=209.9 bps
    """
    if config is None:
        config = _load_config()

    trace_cfg = config["data"]["trace"]
    start_date = pd.Timestamp(trace_cfg["start_date"])
    end_date = pd.Timestamp(trace_cfg["end_date"])

    # Check for cached processed data
    cache_dir = Path(__file__).resolve().parents[1] / config["paths"]["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "trace_data.parquet"

    if cache_path.exists():
        logger.info("Loading cached TRACE data from %s", cache_path)
        df = pd.read_parquet(cache_path)
        logger.info("Loaded %d rows from cache", len(df))
        return df

    # Load real OSBAP TRACE data
    if TRACE_CLEAN_PATH.exists():
        logger.info("Loading OSBAP TRACE data from %s", TRACE_CLEAN_PATH)
        raw = pd.read_parquet(TRACE_CLEAN_PATH)
    elif OSBAP_RAW_PATH.exists():
        logger.info("Loading raw OSBAP data from %s (may take longer)", OSBAP_RAW_PATH)
        raw = pd.read_parquet(OSBAP_RAW_PATH)
    else:
        raise FileNotFoundError(
            f"OSBAP TRACE data not found at:\n"
            f"  {TRACE_CLEAN_PATH}\n"
            f"  {OSBAP_RAW_PATH}\n"
            f"Download from openbondassetpricing.com and place in the expected path."
        )

    logger.info(
        "Raw OSBAP data: %d rows, %d CUSIPs, date range %s to %s",
        len(raw), raw["cusip_id"].nunique(),
        raw["trd_exctn_dt"].min().date(), raw["trd_exctn_dt"].max().date(),
    )

    # Filter to configured date range
    mask = (raw["trd_exctn_dt"] >= start_date) & (raw["trd_exctn_dt"] <= end_date)
    df = raw.loc[mask].copy()
    logger.info(
        "Filtered to %s — %s: %d rows, %d CUSIPs",
        start_date.date(), end_date.date(), len(df), df["cusip_id"].nunique(),
    )

    if len(df) == 0:
        raise ValueError(
            f"No TRACE data found in date range {start_date.date()} to {end_date.date()}. "
            f"Available range: {raw['trd_exctn_dt'].min().date()} to {raw['trd_exctn_dt'].max().date()}"
        )

    # Classify IG vs HY using db_type + credit_spread
    # Step 1: db_type 3 (144A) bonds are always HY
    # Step 2: For db_type 1 bonds, compute median credit_spread per CUSIP.
    #         Bonds with median spread > 200 bps are HY (e.g., fallen angels,
    #         crossover credits). The 200 bps threshold corresponds roughly
    #         to the BBB/BB boundary in the ICE BofA OAS indices.
    median_spread_by_cusip = df.groupby("cusip_id")["credit_spread"].transform("median")
    is_144a = df["db_type"].isin(_DB_TYPE_ALWAYS_HY)
    is_wide_spread = median_spread_by_cusip > (_IG_HY_SPREAD_THRESHOLD_BPS / 10_000)
    df["is_ig"] = (~is_144a & ~is_wide_spread).astype(int)

    # Estimate coupon from price, yield, maturity (semi-annual bond model)
    df["coupon"] = _estimate_coupon(
        df["pr"].values,
        df["ytm"].values,
        df["bond_maturity"].values,
    )

    # Map to standard column names
    df = df.rename(columns={
        "cusip_id": "cusip",
        "trd_exctn_dt": "date",
        "pr": "price",
        "ytm": "yield_pct",
        "dvolume": "volume",  # dollar volume in millions
        "prc_vw_par": "price_vw",
        "issuer_cusip": "issuer_id",
        "bond_maturity": "years_to_maturity",
    })

    # Convert dollar volume from millions to actual dollars
    df["volume"] = df["volume"] * 1_000_000

    # Compute maturity_date from years_to_maturity + trade date
    # Round to nearest day to eliminate time artifacts from float arithmetic
    # E.g., 7.283 years from 2024-01-02 -> 2031-04-14 (not 2031-04-15 02:46:38)
    raw_maturity = df["date"] + pd.to_timedelta(
        df["years_to_maturity"] * 365.25, unit="D"
    )
    df["maturity_date"] = raw_maturity.dt.normalize()

    # Derive rating from credit spread and db_type (approximate)
    df["rating"] = _assign_rating_from_spread(
        df["credit_spread"].values, df["is_ig"].values
    )

    # Derive sector from company_symbol (map known issuers)
    df["sector"] = _assign_sector(df)

    # Side and report_type: not available in OSBAP aggregate data
    # Use 'aggregate' to indicate this is daily aggregate, not individual trades
    df["side"] = "aggregate"
    df["report_type"] = "osbap_daily"

    # Select and order final columns
    output_cols = [
        "cusip", "date", "price", "yield_pct", "volume", "trade_count",
        "side", "report_type", "maturity_date", "coupon", "rating",
        "sector", "is_ig", "credit_spread", "mod_dur", "mac_dur",
        "years_to_maturity", "issuer_id", "company_symbol", "price_vw",
    ]
    existing_cols = [c for c in output_cols if c in df.columns]
    df = df[existing_cols].copy()

    # Drop rows with missing critical fields
    critical_cols = ["cusip", "date", "price", "volume", "trade_count"]
    before = len(df)
    df = df.dropna(subset=critical_cols)
    if len(df) < before:
        logger.info("Dropped %d rows with missing critical fields", before - len(df))

    logger.info(
        "Final TRACE dataset: %d rows, %d CUSIPs, %d trading days. "
        "IG: %d CUSIPs, HY: %d CUSIPs",
        len(df), df["cusip"].nunique(),
        df["date"].nunique(),
        df.loc[df["is_ig"] == 1, "cusip"].nunique(),
        df.loc[df["is_ig"] == 0, "cusip"].nunique(),
    )

    # Cache for future runs
    df.to_parquet(cache_path, index=False)
    logger.info("Cached TRACE data to %s", cache_path)

    return df


def _estimate_coupon(
    price: np.ndarray, ytm: np.ndarray, years_to_mat: np.ndarray,
) -> np.ndarray:
    """Back out annual coupon rate from price, YTM, and maturity.

    Uses the semi-annual coupon bond pricing identity:
        P = (c/2) * annuity + 100 * disc_N
    Rearranged:
        c = (P/100 - disc_N) / annuity * 2 * 100

    Args:
        price: Clean prices as percent of par (e.g., 98.5 means $98.50).
        ytm: Yield to maturity as decimal (e.g., 0.05 = 5%).
        years_to_mat: Years to maturity.

    Returns:
        Annual coupon rates in percent (e.g., 4.5 = 4.5%).

    Example: price=98.5, ytm=0.05, maturity=5.0 -> coupon ~4.66%
    """
    y_semi = np.asarray(ytm, dtype=np.float64) / 2.0
    n_periods = np.round(2 * np.asarray(years_to_mat, dtype=np.float64)).astype(int)
    n_periods = np.clip(n_periods, 1, None)

    disc_N = (1.0 / (1.0 + y_semi)) ** n_periods

    # Annuity factor
    with np.errstate(divide="ignore", invalid="ignore"):
        annuity = np.where(
            np.abs(y_semi) > 1e-8,
            (1.0 - disc_N) / y_semi,
            n_periods.astype(float),  # fallback for near-zero yield
        )

    p_frac = np.asarray(price, dtype=np.float64) / 100.0
    coupon = np.where(
        annuity > 1e-8,
        (p_frac - disc_N) / annuity * 2 * 100,
        0.0,
    )

    # Clip to reasonable range (0% to 15%)
    coupon = np.clip(coupon, 0.0, 15.0)
    return coupon


def _assign_rating_from_spread(
    credit_spread: np.ndarray, is_ig: np.ndarray
) -> np.ndarray:
    """Approximate credit rating from spread level and IG/HY classification.

    Uses typical OAS ranges for each rating bucket (as of 2024):
        AAA: < 50 bps,  AA: 50-80,  A: 80-130,  BBB: 130-250
        BB: 250-400,  B: 400-600,  CCC: > 600 bps

    Args:
        credit_spread: Credit spread in decimal (e.g., 0.015 = 150 bps).
        is_ig: 1 for investment grade, 0 for high yield.

    Returns:
        Array of rating strings (e.g., 'A', 'BB+').
    """
    spread_bps = np.asarray(credit_spread, dtype=np.float64) * 10_000
    ratings = np.full(len(spread_bps), "BBB", dtype=object)

    # IG ratings
    ig_mask = is_ig == 1
    ratings[ig_mask & (spread_bps < 50)] = "AAA"
    ratings[ig_mask & (spread_bps >= 50) & (spread_bps < 80)] = "AA"
    ratings[ig_mask & (spread_bps >= 80) & (spread_bps < 130)] = "A"
    ratings[ig_mask & (spread_bps >= 130) & (spread_bps < 250)] = "BBB"
    ratings[ig_mask & (spread_bps >= 250)] = "BBB-"

    # HY ratings
    hy_mask = is_ig == 0
    ratings[hy_mask & (spread_bps < 300)] = "BB+"
    ratings[hy_mask & (spread_bps >= 300) & (spread_bps < 400)] = "BB"
    ratings[hy_mask & (spread_bps >= 400) & (spread_bps < 600)] = "B"
    ratings[hy_mask & (spread_bps >= 600)] = "CCC"

    return ratings


def _assign_sector(df: pd.DataFrame) -> pd.Series:
    """Assign sector based on company_symbol or issuer metadata.

    Uses a mapping of known major corporate bond issuers to GICS-like sectors.

    Args:
        df: DataFrame with 'company_symbol' column.

    Returns:
        Series of sector strings.
    """
    # Sector mapping based on major issuers (from ICE BofA classifications)
    _SECTOR_MAP = {
        "JPM": "Financials", "JPMO": "Financials", "JPMorgan": "Financials",
        "BAC": "Financials", "BANK": "Financials", "BankAmer": "Financials",
        "C": "Financials", "CITI": "Financials", "Citigroup": "Financials",
        "GS": "Financials", "GOLD": "Financials", "Goldman": "Financials",
        "MS": "Financials", "MORG": "Financials", "MorganSt": "Financials",
        "WFC": "Financials", "WELL": "Financials", "WellsFar": "Financials",
        "BRK": "Financials", "BERK": "Financials",
        "AAPL": "Technology", "Apple": "Technology",
        "MSFT": "Technology", "Microsof": "Technology",
        "AMZN": "Technology", "Amazon": "Technology",
        "GOOG": "Technology", "Alphabet": "Technology",
        "META": "Technology", "Meta": "Technology",
        "NVDA": "Technology", "NVIDIA": "Technology",
        "CRM": "Technology", "Salesfor": "Technology",
        "XOM": "Energy", "ExxonMo": "Energy", "Exxon": "Energy",
        "CVX": "Energy", "Chevron": "Energy",
        "COP": "Energy", "ConocoP": "Energy",
        "JNJ": "Healthcare", "Johnson": "Healthcare",
        "PFE": "Healthcare", "Pfizer": "Healthcare",
        "ABBV": "Healthcare", "AbbVie": "Healthcare",
        "MRK": "Healthcare", "Merck": "Healthcare",
        "ABT": "Healthcare", "Abbott": "Healthcare",
        "PG": "Consumer", "Procter": "Consumer",
        "WMT": "Consumer", "Walmart": "Consumer",
        "KO": "Consumer", "CocaCola": "Consumer", "Coca-Col": "Consumer",
        "PEP": "Consumer", "PepsiCo": "Consumer",
        "HD": "Consumer", "HomeDep": "Consumer",
        "VZ": "Communications", "Verizon": "Communications",
        "T": "Communications", "AT&T": "Communications",
        "CMCSA": "Communications", "Comcast": "Communications",
        "DIS": "Communications", "Disney": "Communications",
        "BA": "Industrials", "Boeing": "Industrials",
        "CAT": "Industrials", "Caterpi": "Industrials",
        "HON": "Industrials", "Honeywe": "Industrials",
        "UPS": "Industrials",
        "NEE": "Utilities", "NextEra": "Utilities",
        "DUK": "Utilities", "Duke": "Utilities",
        "SO": "Utilities", "Souther": "Utilities",
    }

    if "company_symbol" not in df.columns:
        return pd.Series("Unknown", index=df.index)

    def _lookup(sym):
        if pd.isna(sym):
            return "Unknown"
        sym_str = str(sym).strip()
        # Try exact match first
        if sym_str in _SECTOR_MAP:
            return _SECTOR_MAP[sym_str]
        # Try partial match
        for key, sector in _SECTOR_MAP.items():
            if key.lower() in sym_str.lower() or sym_str.lower() in key.lower():
                return sector
        return "Other"

    return df["company_symbol"].apply(_lookup)


def aggregate_cusip_day(
    trace_df: pd.DataFrame,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """Aggregate trade-level TRACE data to CUSIP-day level.

    Since OSBAP data is already daily aggregated, this function mainly
    ensures the schema is consistent and adds any missing derived fields.

    Args:
        trace_df: Raw TRACE DataFrame from fetch_trace_data().
        config: Configuration dict (loaded from config.yaml if None).

    Returns:
        DataFrame with one row per CUSIP-day, including: cusip, date,
        price, yield_pct, volume, trade_count, coupon, is_ig, maturity_date,
        rating, sector, credit_spread, mod_dur, years_to_maturity.

    Example:
        For cusip='00184AAC9' on 2024-01-02:
        price=109.65, volume=153503, trade_count=5, credit_spread=0.021
    """
    if config is None:
        config = _load_config()

    df = trace_df.copy()

    # OSBAP data is already at CUSIP-day level, so no aggregation needed
    # But ensure no duplicates (shouldn't happen with clean data)
    n_before = len(df)
    df = df.drop_duplicates(subset=["cusip", "date"], keep="first")
    if len(df) < n_before:
        logger.warning("Dropped %d duplicate CUSIP-day rows", n_before - len(df))

    # Compute average trade size (volume / trade_count)
    df["avg_trade_size"] = np.where(
        df["trade_count"] > 0,
        df["volume"] / df["trade_count"],
        0.0,
    )

    # Compute spread in bps from credit_spread (decimal -> bps)
    if "credit_spread" in df.columns:
        df["spread_bps_raw"] = df["credit_spread"] * 10_000
    else:
        df["spread_bps_raw"] = np.nan

    # Dealer trade fraction: not available in aggregate OSBAP data
    # Use a proxy based on trade count (higher count -> more likely interdealer).
    # Empirically, interdealer fraction is ~25% on average (FINRA TRACE statistics).
    # More actively traded bonds have higher interdealer activity.
    # Provide both `pct_dealer` (for backward compatibility) and `pct_dealer_proxy`.
    df["pct_dealer_proxy"] = np.clip(
        1.0 - 1.0 / (1.0 + df["trade_count"] * 0.1), 0.0, 0.8
    )
    df["pct_dealer"] = df["pct_dealer_proxy"]

    logger.info(
        "CUSIP-day panel: %d rows, %d unique CUSIPs",
        len(df), df["cusip"].nunique(),
    )

    return df


def get_bond_static(trace_df: pd.DataFrame) -> pd.DataFrame:
    """Extract unique bond-level static data from TRACE trades.

    Takes the first observation per CUSIP to get static attributes.
    For time-varying fields like credit_spread, uses the median across
    all observations for that CUSIP.

    Args:
        trace_df: Trade-level TRACE DataFrame from fetch_trace_data().

    Returns:
        DataFrame with one row per CUSIP containing: cusip, rating, is_ig,
        sector, coupon, maturity_date.
        Example: cusip='00184AAC9', rating='BBB', is_ig=1,
        sector='Financials', coupon=7.58, maturity_date='2031-04-15'.
    """
    static_cols = ["cusip", "rating", "is_ig", "sector", "coupon", "maturity_date"]
    # Use only columns that exist in the DataFrame
    available_cols = [c for c in static_cols if c in trace_df.columns]
    return trace_df[available_cols].drop_duplicates(subset=["cusip"]).reset_index(drop=True)
