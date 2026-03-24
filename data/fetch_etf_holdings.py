"""
fetch_etf_holdings.py -- Fetch real ETF holdings for LQD and HYG from iShares.

DATA PROVENANCE:
  - Real data downloaded from iShares website CSV endpoints.
    LQD: https://www.ishares.com/us/products/239566/.../1467271812596.ajax?fileType=csv
    HYG: https://www.ishares.com/us/products/239565/.../1467271812596.ajax?fileType=csv
  - CSV format: ~9 metadata rows, then header row with columns including
    Name, Sector, Asset Class, Market Value, Weight (%), CUSIP, etc.
  - Filtered to Asset Class == "Fixed Income" to exclude cash/derivatives.
  - No synthetic fallback -- raises on failure.
"""

import io
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yaml

logger = logging.getLogger(__name__)

# Exact iShares CSV download URLs per ticker.
# Each ETF has a different product slug in the URL path.
ISHARES_URLS = {
    "LQD": (
        "https://www.ishares.com/us/products/239566/"
        "ishares-iboxx-investment-grade-corporate-bond-etf/"
        "1467271812596.ajax?fileType=csv&fileName=LQD_holdings&dataType=fund"
    ),
    "HYG": (
        "https://www.ishares.com/us/products/239565/"
        "ishares-iboxx-high-yield-corporate-bond-etf/"
        "1467271812596.ajax?fileType=csv&fileName=HYG_holdings&dataType=fund"
    ),
}

# Browser-like headers to avoid 403/429 blocks.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}


def _load_config() -> dict:
    """Load config.yaml from project root."""
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _fetch_ishares_csv(
    ticker: str,
    product_id: str,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Download and parse iShares holdings CSV for a single ETF.

    Args:
        ticker: ETF ticker, e.g. 'LQD' or 'HYG'.
        product_id: iShares product ID (e.g. '239566'). Used as fallback
            to construct alternative URL patterns if the primary URL fails.
        max_retries: Number of retry attempts on transient failures.

    Returns:
        DataFrame with columns [cusip, name, weight, sector, market_value].
        Only bond holdings (Asset Class == "Fixed Income").
        Weight is fractional (e.g. 0.0019 for 0.19%).

    Raises:
        RuntimeError: If download or parsing fails after all retries.
    """
    # Build list of URLs to try: primary (known exact URL), then fallback
    # using product_id with both IG and HY slug patterns.
    urls_to_try = []
    if ticker in ISHARES_URLS:
        urls_to_try.append(ISHARES_URLS[ticker])
    # Fallback: construct URL from product_id with both slug patterns
    for slug in [
        "ishares-iboxx-investment-grade-corporate-bond-etf",
        "ishares-iboxx-high-yield-corporate-bond-etf",
    ]:
        fallback = (
            f"https://www.ishares.com/us/products/{product_id}/"
            f"{slug}/1467271812596.ajax"
            f"?fileType=csv&fileName={ticker}_holdings&dataType=fund"
        )
        if fallback not in urls_to_try:
            urls_to_try.append(fallback)

    last_error = None
    for url in urls_to_try:
        for attempt in range(1, max_retries + 1):
            try:
                # Set Referer to the product page for this ticker
                headers = dict(_HEADERS)
                headers["Referer"] = (
                    f"https://www.ishares.com/us/products/{product_id}/"
                )

                logger.info(
                    "Fetching %s holdings (attempt %d/%d): %s",
                    ticker, attempt, max_retries, url[:90] + "...",
                )
                resp = requests.get(url, timeout=30, headers=headers)

                if resp.status_code == 429:
                    wait = 5 * attempt
                    logger.warning(
                        "Rate limited (429) for %s, waiting %ds", ticker, wait
                    )
                    time.sleep(wait)
                    continue

                if resp.status_code == 403:
                    logger.warning("Forbidden (403) for %s at %s", ticker, url[:80])
                    break  # Try next URL, don't retry same one

                if resp.status_code != 200:
                    logger.warning(
                        "HTTP %d for %s at %s", resp.status_code, ticker, url[:80]
                    )
                    break

                # Parse the CSV
                df = _parse_ishares_csv(resp.text, ticker)
                if df is not None and len(df) > 0:
                    return df

                logger.warning("Parsed 0 rows from %s for %s", url[:80], ticker)
                break  # Don't retry if parsing found nothing

            except requests.RequestException as e:
                last_error = e
                logger.warning(
                    "Request error for %s (attempt %d): %s", ticker, attempt, e
                )
                if attempt < max_retries:
                    time.sleep(2 * attempt)
            except Exception as e:
                last_error = e
                logger.warning(
                    "Parse error for %s (attempt %d): %s", ticker, attempt, e
                )
                break  # Don't retry parse errors

    raise RuntimeError(
        f"Failed to fetch iShares holdings for {ticker} after trying "
        f"{len(urls_to_try)} URLs. Last error: {last_error}"
    )


def _parse_ishares_csv(text: str, ticker: str) -> Optional[pd.DataFrame]:
    """Parse the raw iShares CSV text into a cleaned DataFrame.

    The iShares CSV format:
      - Lines 0-8: metadata (fund name, date, shares outstanding, etc.)
      - Line 9: column headers (Name, Sector, Asset Class, ..., CUSIP, ...)
      - Lines 10+: data rows
      - After data: blank line, then disclaimer text

    Args:
        text: Raw CSV text from iShares.
        ticker: ETF ticker for logging.

    Returns:
        DataFrame with columns [cusip, name, weight, sector, market_value]
        filtered to bonds only, or None if parsing fails.
    """
    lines = text.strip().split("\n")

    # Find the header row: contains both "Name" and "CUSIP" and "Weight"
    header_idx = None
    for i, line in enumerate(lines[:20]):  # Header is always in first 20 lines
        if "Name" in line and "CUSIP" in line and "Weight" in line:
            header_idx = i
            break

    if header_idx is None:
        logger.warning("Could not find header row in %s CSV", ticker)
        return None

    # Find where the data ends: look for a blank line or a line that doesn't
    # parse as CSV data (disclaimer text starts after data).
    # The data section ends when we hit a line with very few commas compared
    # to the header, or a blank/whitespace-only line.
    header_comma_count = lines[header_idx].count(",")
    end_idx = len(lines)
    for i in range(header_idx + 1, len(lines)):
        line = lines[i].strip()
        # Blank or near-blank line signals end of data
        if not line or line.startswith("\ufeff"):
            end_idx = i
            break
        # Disclaimer lines have far fewer commas than data rows
        comma_count = line.count(",")
        if comma_count < header_comma_count * 0.3:
            end_idx = i
            break

    csv_text = "\n".join(lines[header_idx:end_idx])
    df = pd.read_csv(io.StringIO(csv_text))

    logger.info(
        "Parsed %s CSV: %d rows, columns: %s",
        ticker, len(df), list(df.columns),
    )

    # Filter to Fixed Income only (excludes cash, money market, derivatives)
    if "Asset Class" in df.columns:
        df = df[df["Asset Class"] == "Fixed Income"].copy()
        logger.info("After filtering to Fixed Income: %d rows", len(df))
    else:
        logger.warning("No 'Asset Class' column found, keeping all rows")

    # Standardize column names
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl == "cusip":
            col_map[col] = "cusip"
        elif "weight" in cl:
            col_map[col] = "weight"
        elif cl == "name":
            col_map[col] = "name"
        elif "market" in cl and "value" in cl:
            col_map[col] = "market_value"
        elif cl == "sector":
            col_map[col] = "sector"
    df = df.rename(columns=col_map)

    if "cusip" not in df.columns:
        logger.warning("No CUSIP column found in %s CSV after rename", ticker)
        return None

    # Clean CUSIPs: must be exactly 9 characters (standard CUSIP length)
    df["cusip"] = df["cusip"].astype(str).str.strip()
    df = df[df["cusip"].str.len() == 9].copy()

    # Convert weight from percentage to fraction: 0.19 -> 0.0019
    if "weight" in df.columns:
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce") / 100.0

    # Keep only the columns we need
    keep_cols = ["cusip", "name", "weight", "sector", "market_value"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    df = df.dropna(subset=["cusip"])

    logger.info(
        "Final %s holdings: %d bonds, weight sum %.4f, "
        "top weight %.4f, sectors: %d unique",
        ticker, len(df),
        df["weight"].sum() if "weight" in df.columns else 0,
        df["weight"].max() if "weight" in df.columns else 0,
        df["sector"].nunique() if "sector" in df.columns else 0,
    )

    return df


def fetch_etf_holdings(
    trace_cusips: pd.DataFrame,
    config: Optional[dict] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Fetch real ETF holdings for LQD and HYG from iShares.

    Downloads holdings CSVs from iShares website, filters to bonds that
    overlap with the TRACE universe, and normalizes weights.

    Args:
        trace_cusips: DataFrame with at least columns [cusip, is_ig].
            Used to filter holdings to bonds in our TRACE universe.
        config: Configuration dict (loaded from config.yaml if None).
        seed: Unused, kept for API compatibility.

    Returns:
        DataFrame with columns: cusip, ticker, weight, sector.
        Contains holdings for LQD and HYG.
        Weights sum to ~1.0 within each ticker (after filtering to TRACE universe).
    """
    if config is None:
        config = _load_config()

    etf_cfg = config["data"]["etf"]

    # Check cache
    cache_dir = Path(__file__).resolve().parents[1] / config["paths"]["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "etf_holdings.parquet"

    if cache_path.exists():
        logger.info("Loading cached ETF holdings from %s", cache_path)
        return pd.read_parquet(cache_path)

    all_holdings: List[pd.DataFrame] = []
    trace_cusip_set = set(trace_cusips["cusip"].values)

    for etf_info in etf_cfg["tickers"]:
        ticker = etf_info["ticker"]
        product_id = etf_info["product_id"]

        # Fetch real holdings from iShares
        real_df = _fetch_ishares_csv(ticker, product_id)

        logger.info(
            "Downloaded %d bond holdings for %s from iShares", len(real_df), ticker
        )

        # Filter to bonds that exist in our TRACE universe
        matched = real_df[real_df["cusip"].isin(trace_cusip_set)].copy()
        logger.info(
            "%s: %d of %d holdings matched TRACE universe (%.1f%%)",
            ticker, len(matched), len(real_df),
            100.0 * len(matched) / len(real_df) if len(real_df) > 0 else 0,
        )

        # If very few match, keep all holdings anyway -- the TRACE universe
        # may be smaller than the ETF but the real CUSIPs are still valid.
        if len(matched) < 20:
            logger.warning(
                "%s: Only %d TRACE matches, using all %d real holdings",
                ticker, len(matched), len(real_df),
            )
            matched = real_df.copy()

        matched["ticker"] = ticker

        # Normalize weights within this ETF so they sum to 1.0
        if "weight" in matched.columns:
            w_sum = matched["weight"].sum()
            if w_sum > 0:
                matched["weight"] = matched["weight"] / w_sum
        else:
            matched["weight"] = 1.0 / len(matched)

        # Keep cusip, ticker, weight, and sector (sector is useful for basket optimization)
        keep = ["cusip", "ticker", "weight"]
        if "sector" in matched.columns:
            keep.append("sector")
        all_holdings.append(matched[keep])

    holdings = pd.concat(all_holdings, ignore_index=True)

    # Cache
    holdings.to_parquet(cache_path, index=False)
    logger.info("Cached ETF holdings to %s (%d total rows)", cache_path, len(holdings))

    return holdings


def compute_overlap_counts(
    holdings: pd.DataFrame,
    overlap_etf_list: Optional[List[str]] = None,
    config: Optional[dict] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute how many ETFs hold each CUSIP based on real holdings data.

    Counts actual overlap from the ETFs we have real holdings for (LQD, HYG).
    The overlap count will be 1 (held by one ETF) or 2 (held by both).

    Args:
        holdings: DataFrame with columns [cusip, ticker, weight] from fetch_etf_holdings().
        overlap_etf_list: Unused, kept for API compatibility.
        config: Configuration dict.
        seed: Unused, kept for API compatibility.

    Returns:
        DataFrame with columns: cusip, etf_overlap_count.
        Example: cusip='38259P508' held by both LQD and HYG -> etf_overlap_count=2.
    """
    # Count actual ETF overlap from real holdings only (no simulation)
    overlap = holdings.groupby("cusip")["ticker"].nunique().reset_index()
    overlap.columns = ["cusip", "etf_overlap_count"]

    logger.info(
        "ETF overlap (real holdings only): mean %.2f ETFs per bond, max %d, "
        "tickers with data: %s",
        overlap["etf_overlap_count"].mean(),
        overlap["etf_overlap_count"].max(),
        list(holdings["ticker"].unique()),
    )

    return overlap
