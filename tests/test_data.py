"""Tests for the data-fetching layer."""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Ensure the project root is on sys.path so that `data.*` imports work.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# TRACE data
# ---------------------------------------------------------------------------


def test_fetch_trace_returns_dataframe_with_expected_columns() -> None:
    """fetch_trace_data should return a DataFrame with the documented columns."""
    from data.fetch_trace import fetch_trace_data

    df = fetch_trace_data(seed=0)

    assert isinstance(df, pd.DataFrame)
    expected_cols = {
        "cusip", "date", "price", "volume", "side",
        "report_type", "maturity_date", "coupon", "rating",
        "sector", "is_ig", "yield_pct", "trade_count",
    }
    assert expected_cols.issubset(set(df.columns)), (
        f"Missing columns: {expected_cols - set(df.columns)}"
    )
    assert len(df) > 0


# ---------------------------------------------------------------------------
# ETF holdings
# ---------------------------------------------------------------------------


def test_fetch_etf_returns_holdings_with_cusip_weight() -> None:
    """fetch_etf_holdings should include cusip and weight columns."""
    from data.fetch_etf_holdings import fetch_etf_holdings
    from data.fetch_trace import fetch_trace_data, get_bond_static

    trace_df = fetch_trace_data(seed=0)
    bond_static = get_bond_static(trace_df)
    holdings = fetch_etf_holdings(trace_cusips=bond_static, seed=0)

    assert isinstance(holdings, pd.DataFrame)
    assert "cusip" in holdings.columns
    assert "weight" in holdings.columns
    assert len(holdings) > 0


# ---------------------------------------------------------------------------
# Data registry
# ---------------------------------------------------------------------------


def test_data_registry_has_expected_fields() -> None:
    """DATA_REGISTRY should document all key fields."""
    from data.data_registry import DATA_REGISTRY

    expected_keys = {
        "cusip", "date", "price", "volume", "trade_count",
        "rating", "sector", "is_ig", "coupon",
        "etf_basket_flag", "etf_weight", "etf_overlap_count",
        "oas_index_level", "vix_level", "hy_oas_level",
        "spread_bps", "illiquidity_t1", "illiquid_t1",
    }
    actual_keys = set(DATA_REGISTRY.keys())
    missing = expected_keys - actual_keys
    assert not missing, f"DATA_REGISTRY missing keys: {missing}"
