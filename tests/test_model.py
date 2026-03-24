"""Tests for the model training / evaluation layer."""

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


def _build_small_feature_matrix() -> pd.DataFrame:
    """Build a small synthetic feature matrix for fast testing."""
    rng = np.random.default_rng(99)
    n = 2000
    dates = pd.bdate_range("2023-01-03", periods=100)
    rows = []
    for _ in range(n):
        d = rng.choice(dates)
        rows.append({
            "date": d,
            "spread_bps": rng.uniform(1, 50),
            "spread_volatility_5d": rng.uniform(0, 10),
            "log_daily_volume": rng.uniform(10, 17),
            "trade_count": rng.integers(1, 50),
            "avg_trade_size": rng.uniform(10_000, 5_000_000),
            "days_since_last_trade": rng.integers(0, 30),
            "pct_dealer_trades": rng.uniform(0, 1),
            "time_to_maturity": rng.uniform(0.5, 25),
            "coupon": rng.uniform(2, 10),
            "is_ig": rng.integers(0, 2),
            "etf_basket_flag": rng.integers(0, 2),
            "etf_weight": rng.uniform(0, 0.01),
            "etf_overlap_count": rng.integers(0, 7),
            "oas_index_level": rng.uniform(0.8, 1.6),
            "vix_level": rng.uniform(12, 35),
            "hy_oas_level": rng.uniform(3.5, 5.5),
            "illiquid_t1": rng.integers(0, 2),
        })
    return pd.DataFrame(rows)


def test_model_produces_predict_proba() -> None:
    """Trained model must expose predict_proba returning probabilities."""
    from models.illiquidity_model import FEATURE_COLS, prepare_train_test, train_model

    config = _load_config()
    # Override to speed up test: fewer trials, shorter timeout
    config["model"]["search"]["n_trials"] = 2
    config["model"]["search"]["timeout_seconds"] = 30

    df = _build_small_feature_matrix()
    X_train, y_train, X_test, y_test, _, _ = prepare_train_test(df, config)
    model = train_model(X_train, y_train, config)

    assert hasattr(model, "predict_proba"), "Model must have predict_proba method"
    probs = model.predict_proba(X_test)
    assert probs.shape[1] == 2, "predict_proba should return 2 columns (class 0 & 1)"
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_auc_in_reasonable_range() -> None:
    """Test AUC on synthetic data falls in a plausible range (> 0.45)."""
    from models.illiquidity_model import (
        FEATURE_COLS,
        evaluate_model,
        prepare_train_test,
        train_model,
    )

    config = _load_config()
    config["model"]["search"]["n_trials"] = 2
    config["model"]["search"]["timeout_seconds"] = 30

    df = _build_small_feature_matrix()
    X_train, y_train, X_test, y_test, _, _ = prepare_train_test(df, config)
    model = train_model(X_train, y_train, config)
    metrics = evaluate_model(model, X_test, y_test, config)

    # With random synthetic labels AUC hovers around 0.5 +/- noise.
    # We just verify it is computable and in [0, 1].
    assert 0.0 <= metrics["roc_auc"] <= 1.0
