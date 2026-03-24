"""Spread and size policies driven by calibrated illiquidity scores.

Maps illiquidity probabilities (0-1) to actionable dealer spread
adjustments (bps) and maximum offer sizes ($).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def _load_config() -> dict:
    """Load config.yaml from the project root."""
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


def linear_spread_policy(illiquidity_score: float, config: dict) -> float:
    """Compute spread adjustment via linear policy.

    Args:
        illiquidity_score: Calibrated probability in [0, 1].
        config: Full config dict (uses strategy.spread_policy keys).

    Returns:
        Spread adjustment in basis points.

    Example:
        With base_spread_bps=5.0 and sensitivity=20.0, a score of 0.6
        yields 5.0 + 20.0 * 0.6 = 17.0 bps.
    """
    sp = config["strategy"]["spread_policy"]
    base_spread: float = sp["base_spread_bps"]
    sensitivity: float = sp["sensitivity"]
    return base_spread + sensitivity * illiquidity_score


def piecewise_spread_policy(illiquidity_score: float, config: dict) -> float:
    """Compute spread adjustment via piecewise-linear (3-bucket) policy.

    Buckets:
        - Liquid  (score < 0.3): liquid_spread_bps   (e.g. 3.0 bps)
        - Medium  (0.3 <= score < 0.7): medium_spread_bps  (e.g. 8.0 bps)
        - Illiquid (score >= 0.7): illiquid_spread_bps (e.g. 25.0 bps)

    Args:
        illiquidity_score: Calibrated probability in [0, 1].
        config: Full config dict.

    Returns:
        Spread adjustment in basis points.
    """
    pw = config["strategy"]["spread_policy"]["piecewise"]
    liquid_thresh: float = pw["liquid_threshold"]       # 0.3
    illiquid_thresh: float = pw["illiquid_threshold"]   # 0.7
    liquid_bps: float = pw["liquid_spread_bps"]         # 3.0
    medium_bps: float = pw["medium_spread_bps"]         # 8.0
    illiquid_bps: float = pw["illiquid_spread_bps"]     # 25.0

    if illiquidity_score < liquid_thresh:
        # Interpolate linearly within the liquid bucket: 0 -> liquid_bps at threshold
        return liquid_bps
    elif illiquidity_score < illiquid_thresh:
        # Linearly interpolate between liquid and illiquid bps within the medium zone
        frac = (illiquidity_score - liquid_thresh) / (illiquid_thresh - liquid_thresh)
        return medium_bps + frac * (illiquid_bps - medium_bps)
    else:
        return illiquid_bps


def size_policy(illiquidity_score: float, config: dict) -> float:
    """Compute maximum offer size given illiquidity score.

    Formula:
        max_offer = max_notional * (1 - reduction_factor * illiquidity_score)

    Args:
        illiquidity_score: Calibrated probability in [0, 1].
        config: Full config dict.

    Returns:
        Maximum offer size in dollars.

    Example:
        With max_notional=$5M and reduction_factor=0.6, a score of 0.8
        yields 5_000_000 * (1 - 0.6 * 0.8) = 5_000_000 * 0.52 = $2,600,000.
    """
    sp = config["strategy"]["size_policy"]
    max_notional: float = sp["max_notional"]
    reduction: float = sp["illiquidity_reduction_factor"]
    return max_notional * (1.0 - reduction * illiquidity_score)


def apply_spread_policy(
    df: pd.DataFrame,
    config: dict,
    method: str = "piecewise",
) -> pd.DataFrame:
    """Add spread_adjustment_bps and max_offer_size columns to DataFrame.

    Args:
        df: DataFrame with an ``illiquidity_score`` column (values in [0, 1]).
        config: Full config dict.
        method: One of 'linear' or 'piecewise'.

    Returns:
        Copy of df with ``spread_adjustment_bps`` and ``max_offer_size`` added.
    """
    if "illiquidity_score" not in df.columns:
        raise ValueError("DataFrame must contain an 'illiquidity_score' column.")

    result = df.copy()

    if method == "linear":
        policy_fn = linear_spread_policy
    elif method == "piecewise":
        policy_fn = piecewise_spread_policy
    else:
        raise ValueError(f"Unknown spread policy method: {method!r}")

    result["spread_adjustment_bps"] = result["illiquidity_score"].apply(
        lambda s: policy_fn(s, config)
    )
    result["max_offer_size"] = result["illiquidity_score"].apply(
        lambda s: size_policy(s, config)
    )

    logger.info(
        "Applied %s spread policy: mean spread=%.2f bps, mean size=$%.0f",
        method,
        result["spread_adjustment_bps"].mean(),
        result["max_offer_size"].mean(),
    )
    return result
