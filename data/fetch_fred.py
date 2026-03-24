"""
fetch_fred.py -- Fetch macro data from FRED (Federal Reserve Economic Data).

DATA PROVENANCE:
  - All series are REAL public data from FRED via the fredapi library:
      * BAMLC0A0CM: ICE BofA US Corporate Index OAS (bps)
      * VIXCLS: CBOE Volatility Index
      * BAMLH0A0HYM2: ICE BofA US High Yield Index OAS (bps)
  - Requires FRED_API_KEY in .env or environment variable.
    Free API keys: https://fred.stlouisfed.org/docs/api/api_key.html
  - No synthetic data. No fallbacks. Fails loudly if API is unreachable.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# FRED series configuration
FRED_SERIES = {
    "ig_oas": {
        "series_id": "BAMLC0A0CM",
        "name": "ICE BofA US Corporate Index OAS",
        "units": "percent (e.g., 1.20 = 120 bps)",
    },
    "vix": {
        "series_id": "VIXCLS",
        "name": "CBOE Volatility Index",
        "units": "index level",
    },
    "hy_oas": {
        "series_id": "BAMLH0A0HYM2",
        "name": "ICE BofA US High Yield Index OAS",
        "units": "percent (e.g., 4.20 = 420 bps)",
    },
}


def _load_config() -> dict:
    """Load config.yaml from project root."""
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _load_api_key() -> str:
    """Load FRED API key from environment or .env file.

    Checks:
      1. FRED_API_KEY environment variable
      2. .env file in project root (illiquid pricing/.env)
      3. .env file in engine root (illiquidity_engine/.env)

    Returns:
        The API key string.

    Raises:
        RuntimeError: If no API key is found.
    """
    # Check env var first
    key = os.getenv("FRED_API_KEY", "")
    if key:
        return key

    # Try loading from .env files
    try:
        from dotenv import load_dotenv
        # Project root .env
        project_env = Path(__file__).resolve().parents[2] / ".env"
        if project_env.exists():
            load_dotenv(project_env)
            key = os.getenv("FRED_API_KEY", "")
            if key:
                return key

        # Engine root .env
        engine_env = Path(__file__).resolve().parents[1] / ".env"
        if engine_env.exists():
            load_dotenv(engine_env)
            key = os.getenv("FRED_API_KEY", "")
            if key:
                return key
    except ImportError:
        # python-dotenv not installed, try manual parse
        for env_path in [
            Path(__file__).resolve().parents[2] / ".env",
            Path(__file__).resolve().parents[1] / ".env",
        ]:
            if env_path.exists():
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("FRED_API_KEY="):
                            key = line.split("=", 1)[1].strip()
                            if key:
                                os.environ["FRED_API_KEY"] = key
                                return key

    raise RuntimeError(
        "FRED_API_KEY not found. Set it in .env or as an environment variable. "
        "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
    )


def _fetch_series(
    series_id: str,
    start_date: str,
    end_date: str,
    api_key: str,
) -> pd.Series:
    """Fetch a single FRED series via the fredapi library.

    Args:
        series_id: FRED series identifier, e.g., 'BAMLC0A0CM'.
        start_date: Start date string, e.g., '2024-01-01'.
        end_date: End date string, e.g., '2024-12-31'.
        api_key: FRED API key.

    Returns:
        pandas Series with DatetimeIndex and the series values.
        Example for BAMLC0A0CM: values like 1.08, 1.12, 1.15 (percent).

    Raises:
        RuntimeError: If fetch fails or returns no data.
    """
    from fredapi import Fred

    fred = Fred(api_key=api_key)
    logger.info("Fetching %s from FRED API (%s to %s)...", series_id, start_date, end_date)

    data = fred.get_series(
        series_id,
        observation_start=start_date,
        observation_end=end_date,
    )
    data = data.dropna()

    if len(data) == 0:
        raise RuntimeError(
            f"FRED API returned no data for {series_id} "
            f"({start_date} to {end_date})"
        )

    logger.info(
        "FRED API: fetched %s — %d observations, range [%.2f, %.2f], "
        "mean=%.2f, last=%s: %.2f",
        series_id, len(data), data.min(), data.max(),
        data.mean(), data.index[-1].strftime("%Y-%m-%d"), data.iloc[-1],
    )

    return data


def fetch_fred_data(
    config: Optional[dict] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Fetch macro data from FRED using the fredapi library.

    Downloads real ICE BofA OAS indices and VIX from FRED.
    No synthetic data. No fallbacks. Raises on failure.

    Args:
        config: Configuration dict (loaded from config.yaml if None).
        seed: Unused, kept for API compatibility with run_all.py.

    Returns:
        DataFrame with columns: ig_oas, vix, hy_oas.
        DatetimeIndex (business days), forward-filled for gaps.
        ig_oas and hy_oas are in percent (e.g., 1.20 = 120 bps).
        Example row: date=2024-06-15, ig_oas=1.08, vix=13.20, hy_oas=3.45

    Raises:
        RuntimeError: If FRED API key is missing or any series fetch fails.
    """
    if config is None:
        config = _load_config()

    fred_cfg = config["data"]["fred"]
    start_date = fred_cfg["start_date"]
    end_date = fred_cfg["end_date"]
    series_map = fred_cfg["series"]  # e.g., {"ig_oas": "BAMLC0A0CM", ...}

    # Check cache
    cache_dir = Path(__file__).resolve().parents[1] / config["paths"]["cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "fred_macro.parquet"

    if cache_path.exists():
        logger.info("Loading cached FRED data from %s", cache_path)
        return pd.read_parquet(cache_path)

    # Load API key (raises if not found)
    api_key = _load_api_key()
    logger.info("FRED API key loaded (ends with ...%s)", api_key[-4:])

    trading_days = pd.bdate_range(start=start_date, end=end_date)

    results: Dict[str, pd.Series] = {}

    for label, series_id in series_map.items():
        data = _fetch_series(series_id, start_date, end_date, api_key)
        results[label] = data

    # Combine into DataFrame
    df = pd.DataFrame(results)
    df.index.name = "date"

    # Reindex to full business-day grid and forward-fill (FRED has gaps on holidays)
    df = df.reindex(trading_days).ffill().bfill()
    df.index.name = "date"

    logger.info(
        "FRED macro data: %d rows, columns=%s",
        len(df), list(df.columns),
    )
    for col in df.columns:
        logger.info(
            "  %s: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
            col, df[col].mean(), df[col].std(), df[col].min(), df[col].max(),
        )

    # Cache
    df.to_parquet(cache_path)
    logger.info("Cached FRED data to %s", cache_path)

    return df
