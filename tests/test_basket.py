"""Tests for the redemption basket optimisation."""

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


def _make_holdings(n: int = 50) -> pd.DataFrame:
    """Build a small synthetic holdings DataFrame for testing."""
    rng = np.random.default_rng(77)
    sectors = ["Financials", "Technology", "Healthcare", "Energy", "Utilities"]
    weights = rng.dirichlet(np.ones(n))
    return pd.DataFrame({
        "cusip": [f"TST{i:05d}" for i in range(n)],
        "weight": weights,
        "duration": rng.normal(8.0, 2.0, n).clip(1, 20),
        "oas": rng.normal(120, 40, n).clip(30, 400),
        "sector": rng.choice(sectors, n),
        "illiquidity_score": rng.beta(2, 5, n),
    })


def test_optimized_basket_returns_dataframe() -> None:
    """construct_optimized_basket should return a DataFrame."""
    from strategy.redemption_basket import construct_optimized_basket

    config = _load_config()
    holdings = _make_holdings(50)

    basket = construct_optimized_basket(
        etf_holdings=holdings,
        illiquidity_scores=holdings["illiquidity_score"],
        bond_features=holdings,
        config=config,
    )

    assert isinstance(basket, pd.DataFrame)
    assert "weight" in basket.columns
    assert "notional" in basket.columns
    assert len(basket) == len(holdings)


def test_weights_sum_to_one() -> None:
    """Optimised basket weights should sum to approximately 1.0."""
    from strategy.redemption_basket import construct_optimized_basket

    config = _load_config()
    holdings = _make_holdings(50)

    basket = construct_optimized_basket(
        etf_holdings=holdings,
        illiquidity_scores=holdings["illiquidity_score"],
        bond_features=holdings,
        config=config,
    )

    assert basket["weight"].sum() == pytest.approx(1.0, abs=0.01)


def test_min_bonds_constraint() -> None:
    """Optimised basket should have at least min_bonds non-zero weights."""
    from strategy.redemption_basket import construct_optimized_basket

    config = _load_config()
    min_bonds = config["basket"]["min_bonds"]  # 20
    holdings = _make_holdings(60)

    basket = construct_optimized_basket(
        etf_holdings=holdings,
        illiquidity_scores=holdings["illiquidity_score"],
        bond_features=holdings,
        config=config,
    )

    n_active = int((basket["weight"] > 1e-4).sum())
    assert n_active >= min_bonds, (
        f"Expected >= {min_bonds} active bonds, got {n_active}"
    )
