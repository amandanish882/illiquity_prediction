"""Tests for the strategy layer (spread policy + backtest)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_config() -> dict:
    cfg_path = PROJECT_ROOT / "config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg


# ---------------------------------------------------------------------------
# Spread policy
# ---------------------------------------------------------------------------


def test_linear_spread_returns_reasonable_values() -> None:
    """Linear spread policy should increase with illiquidity score."""
    from strategy.spread_policy import linear_spread_policy

    config = _load_config()

    spread_low = linear_spread_policy(0.0, config)
    spread_mid = linear_spread_policy(0.5, config)
    spread_high = linear_spread_policy(1.0, config)

    # base_spread_bps=5.0, sensitivity=20.0
    # score=0 -> 5, score=0.5 -> 15, score=1.0 -> 25
    assert spread_low == pytest.approx(5.0, abs=0.1)
    assert spread_mid == pytest.approx(15.0, abs=0.1)
    assert spread_high == pytest.approx(25.0, abs=0.1)
    assert spread_low < spread_mid < spread_high


def test_piecewise_spread_policy() -> None:
    """Piecewise policy should respect bucket boundaries."""
    from strategy.spread_policy import piecewise_spread_policy

    config = _load_config()

    # Liquid bucket: score < 0.3 -> liquid_spread_bps = 3.0
    assert piecewise_spread_policy(0.1, config) == pytest.approx(3.0, abs=0.1)
    # Illiquid bucket: score >= 0.7 -> illiquid_spread_bps = 25.0
    assert piecewise_spread_policy(0.9, config) == pytest.approx(25.0, abs=0.1)
    # Medium bucket should interpolate between 8.0 and 25.0
    mid_val = piecewise_spread_policy(0.5, config)
    assert 8.0 <= mid_val <= 25.0


def test_size_policy_reduces_for_illiquid() -> None:
    """Size policy should reduce offer size for more illiquid bonds."""
    from strategy.spread_policy import size_policy

    config = _load_config()

    size_liquid = size_policy(0.0, config)
    size_illiquid = size_policy(1.0, config)

    # max_notional=5M, reduction_factor=0.6
    # score=0 -> 5M, score=1 -> 5M * 0.4 = 2M
    assert size_liquid == pytest.approx(5_000_000, abs=100)
    assert size_illiquid == pytest.approx(2_000_000, abs=100)
    assert size_liquid > size_illiquid


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


def test_backtest_produces_markout_results() -> None:
    """run_backtest should return dicts with fill_rate and markout metrics."""
    from strategy.markout_backtest import run_backtest

    config = _load_config()
    rng = np.random.default_rng(42)
    n = 200

    # Minimal DataFrame that run_backtest can work with
    df = pd.DataFrame({
        "date": pd.bdate_range("2024-01-02", periods=n),
        "price": rng.normal(100, 2, n),
        "illiquidity_score": rng.beta(2, 5, n),
    })

    results = run_backtest(features_df=df, model_scores=None, config=config)

    assert "flat" in results
    assert "dynamic" in results
    assert "oracle" in results

    for regime in ["flat", "dynamic", "oracle"]:
        m = results[regime]["metrics"]
        assert "fill_rate" in m
        assert 0.0 <= m["fill_rate"] <= 1.0
        assert "avg_markout_t1_bps" in m
