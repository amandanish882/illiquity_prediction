"""Dealer-quoting simulation and markout backtest engine.

Compares three spread regimes — flat, model-driven dynamic, and oracle —
on fill rate, markout P&L, and adverse selection.
"""

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


def _load_config() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _ensure_dirs(config: dict) -> tuple[Path, Path]:
    """Return (figures_dir, results_dir) and create them if needed."""
    fig_dir = _PROJECT_ROOT / config["paths"]["figures_dir"]
    res_dir = _PROJECT_ROOT / config["paths"]["results_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, res_dir


# ---------------------------------------------------------------------------
# Core mechanics
# ---------------------------------------------------------------------------


def compute_fill_probability(
    spread: float,
    market_spread: float,
    alpha: float,
) -> float:
    """Logistic fill probability: tighter spread -> higher fill rate.

    Formula:
        P(fill) = 1 / (1 + exp(alpha * (spread - market_spread)))

    Args:
        spread: Dealer's quoted half-spread in bps.
        market_spread: Prevailing market half-spread in bps.
        alpha: Sensitivity parameter (from config, e.g. 2.0).

    Returns:
        Fill probability in [0, 1].

    Example:
        spread=10, market_spread=8, alpha=2.0
        -> 1 / (1 + exp(2.0 * 2)) = 1 / (1 + exp(4)) ≈ 0.018
        spread=8, market_spread=8, alpha=2.0
        -> 1 / (1 + exp(0)) = 0.5
    """
    z = alpha * (spread - market_spread)
    # Clamp to avoid overflow
    z = np.clip(z, -30.0, 30.0)
    return float(1.0 / (1.0 + np.exp(z)))


def simulate_dealer_quotes(
    df: pd.DataFrame,
    spread_col: str,
    config: dict,
) -> pd.DataFrame:
    """Simulate dealer quote fills for each bond-day row.

    For every row the dealer quotes at mid +/- spread/2.  A Bernoulli
    draw determines whether the trade fills based on the logistic fill
    probability.

    Args:
        df: Must contain columns ``spread_col``, ``market_spread_bps``,
            ``mid_price``, and ``direction`` (+1 buy, -1 sell from
            dealer's perspective).
        spread_col: Name of the column with the dealer's quoted spread (bps).
        config: Full config dict.

    Returns:
        Copy of df with added columns:
            ``fill_prob``, ``filled`` (bool), ``execution_price``.
    """
    alpha: float = config["strategy"]["backtest"]["fill_rate_alpha"]
    result = df.copy()

    # Vectorised fill probability
    z = alpha * (result[spread_col] - result["market_spread_bps"])
    z = np.clip(z, -30.0, 30.0)
    result["fill_prob"] = 1.0 / (1.0 + np.exp(z))

    rng = np.random.default_rng(42)
    result["filled"] = rng.random(len(result)) < result["fill_prob"]

    # Execution price: dealer buys at mid - half_spread, sells at mid + half_spread
    # direction = +1 means dealer buys (client sells) -> exec = mid - spread/2
    # direction = -1 means dealer sells (client buys) -> exec = mid + spread/2
    half_spread_price = result[spread_col] / 10000.0 * result["mid_price"]
    result["execution_price"] = (
        result["mid_price"] - result["direction"] * half_spread_price / 2.0
    )

    fill_rate = result["filled"].mean()
    logger.info(
        "Simulated fills for '%s': fill_rate=%.1f%%, mean_spread=%.2f bps",
        spread_col,
        fill_rate * 100,
        result[spread_col].mean(),
    )
    return result


def compute_markouts(
    df: pd.DataFrame,
    horizons: list[int],
    config: dict,
) -> pd.DataFrame:
    """Compute markout P&L at specified horizons for filled trades.

    Markout at horizon k (in bps):
        markout_k = (mid_{t+k} - execution_price) / mid_price * 10_000 * direction

    Positive markout = dealer made money on the trade.

    Args:
        df: DataFrame with ``filled``, ``execution_price``, ``mid_price``,
            ``direction``, and ``mid_price_t{k}`` for each horizon k.
        horizons: List of horizon days, e.g. [1, 5].
        config: Full config dict.

    Returns:
        Copy of df with ``markout_t{k}_bps`` columns added.
    """
    txn_cost: float = config["strategy"]["backtest"]["transaction_cost_bps"]
    result = df.copy()

    for k in horizons:
        future_col = f"mid_price_t{k}"
        if future_col not in result.columns:
            logger.warning("Column %s not found — skipping horizon %d", future_col, k)
            continue

        # Raw markout in bps
        raw_markout = (
            (result[future_col] - result["execution_price"])
            / result["mid_price"]
            * 10_000
            * result["direction"]
        )
        # Net of transaction cost
        result[f"markout_t{k}_bps"] = raw_markout - txn_cost
        # Only meaningful for filled trades
        result.loc[~result["filled"], f"markout_t{k}_bps"] = np.nan

    return result


def compute_adverse_selection(df: pd.DataFrame) -> float:
    """Average markout (bps) on trades where the market moved against the dealer.

    Adverse selection = mean of markout for filled trades where markout < 0.
    A higher (more negative) number indicates worse adverse selection.

    Args:
        df: DataFrame with ``filled`` and ``markout_t1_bps`` columns.

    Returns:
        Mean adverse-selection markout in bps (negative value means loss).
    """
    filled = df[df["filled"] & df["markout_t1_bps"].notna()]
    adverse = filled[filled["markout_t1_bps"] < 0]["markout_t1_bps"]
    if len(adverse) == 0:
        return 0.0
    return float(adverse.mean())


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------


def _prepare_market_spread(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Ensure ``market_spread_bps`` column exists.

    If missing, we estimate it from the data:  median spread for the bond's
    rating bucket, or a config-based default.
    """
    out = df.copy()
    if "market_spread_bps" not in out.columns:
        # Use the mid-point of flat IG and HY spreads as a rough market spread
        ig_flat: float = config["strategy"]["flat_spread"]["ig_bps"]
        hy_flat: float = config["strategy"]["flat_spread"]["hy_bps"]
        if "is_ig" in out.columns:
            # is_ig=1 means IG -> ig_flat; is_ig=0 means HY -> hy_flat
            out["market_spread_bps"] = np.where(out["is_ig"] == 1, ig_flat, hy_flat)
        elif "is_hy" in out.columns:
            out["market_spread_bps"] = np.where(out["is_hy"], hy_flat, ig_flat)
        else:
            out["market_spread_bps"] = ig_flat
    return out


def _prepare_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``direction`` from the ``side`` column if available.

    OSBAP TRACE side codes:
        'S' = dealer sell / client buy  -> direction = -1
        'B' = dealer buy / client sell  -> direction = +1
        'D' = dealer-to-dealer          -> randomly assigned +1/-1

    Falls back to random assignment only when no ``side`` column exists.
    """
    out = df.copy()
    if "direction" not in out.columns:
        if "side" in out.columns:
            rng = np.random.default_rng(99)
            direction = pd.Series(np.nan, index=out.index, dtype=float)
            direction[out["side"] == "B"] = 1.0
            direction[out["side"] == "S"] = -1.0
            d_mask = out["side"] == "D"
            if d_mask.any():
                direction[d_mask] = rng.choice(
                    [1.0, -1.0], size=int(d_mask.sum())
                )
            # Any remaining unmapped values get random assignment
            still_nan = direction.isna()
            if still_nan.any():
                direction[still_nan] = rng.choice(
                    [1.0, -1.0], size=int(still_nan.sum())
                )
            out["direction"] = direction.astype(int)
        else:
            rng = np.random.default_rng(99)
            out["direction"] = rng.choice([1, -1], size=len(out))
    return out


def _prepare_future_mids(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Look up real future mid prices from the CUSIP-day panel.

    For each row (cusip, date), the future mid at horizon k is the price
    observed for the same CUSIP k business days later in the panel.  This
    is computed via ``groupby('cusip')['mid_price'].shift(-k)``.

    Rows where the future price is unavailable (last k observations per
    CUSIP) are dropped so that markout P&L is only computed on rows with
    real observed outcomes.
    """
    out = df.copy()

    # Ensure proper sort order for the shift to be meaningful
    if "cusip" in out.columns and "date" in out.columns:
        out = out.sort_values(["cusip", "date"]).reset_index(drop=True)

    for k in horizons:
        col = f"mid_price_t{k}"
        if col not in out.columns:
            if "cusip" in out.columns:
                out[col] = out.groupby("cusip")["mid_price"].shift(-k)
            else:
                # Single-asset fallback: just shift the whole series
                out[col] = out["mid_price"].shift(-k)

    # Drop rows where any future-mid column is NaN (no observed outcome)
    future_cols = [f"mid_price_t{k}" for k in horizons]
    existing = [c for c in future_cols if c in out.columns]
    if existing:
        out = out.dropna(subset=existing).reset_index(drop=True)

    return out


def _prepare_mid_price(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``mid_price`` column exists."""
    out = df.copy()
    if "mid_price" not in out.columns:
        # Use par = 100 as default for bonds priced near par
        if "price" in out.columns:
            out["mid_price"] = out["price"]
        else:
            out["mid_price"] = 100.0
    return out


def _prepare_illiquidity_score(
    df: pd.DataFrame,
    model_scores: np.ndarray | pd.Series | None,
) -> pd.DataFrame:
    """Attach model illiquidity scores to the DataFrame."""
    out = df.copy()
    if model_scores is not None:
        out["illiquidity_score"] = np.asarray(model_scores)[: len(out)]
    elif "illiquidity_score" not in out.columns:
        # Generate synthetic scores from a Beta(2,5) so mean ≈ 0.29
        rng = np.random.default_rng(123)
        out["illiquidity_score"] = rng.beta(2, 5, size=len(out))
    return out


def _prepare_oracle_score(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Oracle 'knows' future realised illiquidity.

    Proxy: if mid moved more than 1 std-dev against dealer, oracle flags
    as illiquid.  We use the future mid to create a realised score in [0,1].
    """
    out = df.copy()
    if "oracle_score" not in out.columns:
        if "mid_price_t1" in out.columns:
            ret = np.abs(out["mid_price_t1"] / out["mid_price"] - 1.0)
            # Scale returns to [0,1] using a sigmoid-like transform
            # 50 bps move -> ~0.5, 100 bps -> ~0.73
            out["oracle_score"] = 1.0 / (1.0 + np.exp(-ret * 10_000 / 50.0 + 2.0))
        else:
            out["oracle_score"] = out.get("illiquidity_score", 0.5)
    return out


def run_backtest(
    features_df: pd.DataFrame,
    model_scores: np.ndarray | pd.Series | None,
    config: dict,
) -> dict[str, Any]:
    """Run the full three-regime backtest.

    Regimes:
        1. **flat** — constant spread (ig_bps / hy_bps from config)
        2. **dynamic** — piecewise-linear spread from illiquidity model
        3. **oracle** — piecewise-linear spread from realised future illiquidity

    Args:
        features_df: Bond-day panel with features. Must have at least a
            ``date`` (or index) column.  Optional: ``mid_price``,
            ``market_spread_bps``, ``direction``, ``is_hy``.
        model_scores: Array-like of calibrated illiquidity probabilities
            aligned to features_df rows. If None, the function looks for
            an ``illiquidity_score`` column or generates synthetic scores.
        config: Full config dict.

    Returns:
        Dict with keys ``flat``, ``dynamic``, ``oracle``, each containing
        metrics and the underlying DataFrames.
    """
    from strategy.spread_policy import piecewise_spread_policy, size_policy

    horizons: list[int] = config["strategy"]["backtest"]["markout_horizons"]
    ig_flat: float = config["strategy"]["flat_spread"]["ig_bps"]
    hy_flat: float = config["strategy"]["flat_spread"]["hy_bps"]

    # ---- Data preparation ----
    df = features_df.copy()
    df = _prepare_mid_price(df)
    df = _prepare_market_spread(df, config)
    df = _prepare_direction(df)
    df = _prepare_future_mids(df, horizons)
    df = _prepare_illiquidity_score(df, model_scores)
    df = _prepare_oracle_score(df, horizons)

    # ---- Flat spread ----
    if "is_ig" in df.columns:
        df["flat_spread_bps"] = np.where(df["is_ig"] == 1, ig_flat, hy_flat)
    elif "is_hy" in df.columns:
        df["flat_spread_bps"] = np.where(df["is_hy"], hy_flat, ig_flat)
    else:
        df["flat_spread_bps"] = ig_flat

    # ---- Dynamic spread (model-based) ----
    df["dynamic_spread_bps"] = df["illiquidity_score"].apply(
        lambda s: piecewise_spread_policy(s, config)
    )

    # ---- Oracle spread ----
    df["oracle_spread_bps"] = df["oracle_score"].apply(
        lambda s: piecewise_spread_policy(s, config)
    )

    results: dict[str, Any] = {}

    for regime, spread_col in [
        ("flat", "flat_spread_bps"),
        ("dynamic", "dynamic_spread_bps"),
        ("oracle", "oracle_spread_bps"),
    ]:
        regime_df = simulate_dealer_quotes(df, spread_col, config)
        regime_df = compute_markouts(regime_df, horizons, config)

        fill_rate = float(regime_df["filled"].mean())
        filled_mask = regime_df["filled"]

        metrics: dict[str, Any] = {"fill_rate": fill_rate}
        for k in horizons:
            mcol = f"markout_t{k}_bps"
            if mcol in regime_df.columns:
                filled_markouts = regime_df.loc[filled_mask, mcol].dropna()
                metrics[f"avg_markout_t{k}_bps"] = float(filled_markouts.mean()) if len(filled_markouts) > 0 else 0.0

        metrics["adverse_selection_bps"] = compute_adverse_selection(regime_df)

        # Net P&L per bond-day (only for filled)
        if "markout_t1_bps" in regime_df.columns:
            filled_pnl = regime_df.loc[filled_mask, "markout_t1_bps"].dropna()
            metrics["net_pnl_per_bondday_bps"] = float(filled_pnl.mean()) if len(filled_pnl) > 0 else 0.0
        else:
            metrics["net_pnl_per_bondday_bps"] = 0.0

        metrics["n_filled"] = int(filled_mask.sum())
        metrics["n_total"] = len(regime_df)
        metrics["mean_spread_bps"] = float(regime_df[spread_col].mean())

        results[regime] = {"metrics": metrics, "df": regime_df}
        logger.info(
            "Regime=%s  fill_rate=%.1f%%  markout_t1=%.2f bps  adverse=%.2f bps",
            regime,
            fill_rate * 100,
            metrics.get("avg_markout_t1_bps", 0),
            metrics["adverse_selection_bps"],
        )

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def generate_backtest_report(
    results: dict[str, Any],
    config: dict,
) -> None:
    """Save backtest comparison table and charts.

    Outputs:
        - outputs/results/backtest_results.json
        - outputs/figures/backtest_regime_comparison.png

    Args:
        results: Dict returned by ``run_backtest``.
        config: Full config dict.
    """
    fig_dir, res_dir = _ensure_dirs(config)

    # ---- JSON summary ----
    summary: dict[str, Any] = {}
    for regime in ["flat", "dynamic", "oracle"]:
        summary[regime] = results[regime]["metrics"]

    json_path = res_dir / "backtest_results.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved backtest results to %s", json_path)

    # ---- Comparison bar chart ----
    plot_cfg = config.get("plotting", {})
    sns.set_theme(style=plot_cfg.get("style", "whitegrid"))
    sns.set_context(plot_cfg.get("context", "paper"), font_scale=plot_cfg.get("font_scale", 1.2))

    regimes = ["flat", "dynamic", "oracle"]
    metric_names = [
        ("fill_rate", "Fill Rate", "{:.1%}"),
        ("avg_markout_t1_bps", "Avg Markout t+1 (bps)", "{:.2f}"),
        ("adverse_selection_bps", "Adverse Selection (bps)", "{:.2f}"),
        ("net_pnl_per_bondday_bps", "Net P&L / bond-day (bps)", "{:.2f}"),
    ]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(16, 5))
    colors = ["#4c72b0", "#55a868", "#c44e52"]

    for ax, (key, label, fmt) in zip(axes, metric_names):
        vals = [summary[r].get(key, 0) for r in regimes]
        bars = ax.bar(regimes, vals, color=colors)
        ax.set_title(label, fontsize=11)
        ax.set_ylabel("")
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Backtest Regime Comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    chart_path = fig_dir / "backtest_regime_comparison.png"
    fig.savefig(chart_path, dpi=plot_cfg.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved backtest chart to %s", chart_path)
