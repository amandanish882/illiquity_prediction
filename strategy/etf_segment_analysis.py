"""ETF-segment analysis: ETF-basket-eligible vs off-the-run bonds.

Demonstrates that dynamic (model-driven) spread policies generate
disproportionately more improvement for off-the-run names, while
ETF-eligible bonds are already well-served by flat spreads.
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
    fig_dir = _PROJECT_ROOT / config["paths"]["figures_dir"]
    res_dir = _PROJECT_ROOT / config["paths"]["results_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, res_dir


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def segment_by_etf(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split bond universe into ETF-eligible and off-the-run segments.

    If the DataFrame contains an ``etf_eligible`` boolean column, that is
    used directly.  Otherwise the function falls back to ``in_lqd`` or
    ``in_hyg`` flags, and as a last resort treats any bond with
    ``etf_basket_count >= 1`` as ETF-eligible.

    Args:
        df: Bond-day panel.

    Returns:
        Tuple of (etf_eligible_df, off_the_run_df).

    Example:
        A bond appearing in LQD holdings has etf_eligible=True and lands
        in the first DataFrame; a bond in neither LQD nor HYG is
        off-the-run.
    """
    if "etf_eligible" in df.columns:
        mask = df["etf_eligible"].astype(bool)
    elif "etf_basket_flag" in df.columns:
        mask = df["etf_basket_flag"].astype(bool)
    elif "in_lqd" in df.columns or "in_hyg" in df.columns:
        mask = df.get("in_lqd", False) | df.get("in_hyg", False)
    elif "etf_basket_count" in df.columns:
        mask = df["etf_basket_count"] >= 1
    else:
        # Synthetic fallback: treat top 40% by volume as ETF-eligible
        logger.warning(
            "No ETF-eligibility column found — using volume-based proxy "
            "(top 40%% of volume treated as ETF-eligible)."
        )
        if "volume" in df.columns:
            threshold = df["volume"].quantile(0.6)
            mask = df["volume"] >= threshold
        else:
            rng = np.random.default_rng(55)
            mask = pd.Series(rng.random(len(df)) < 0.4, index=df.index)

    etf_df = df[mask].copy()
    otr_df = df[~mask].copy()

    logger.info(
        "Segmented: %d ETF-eligible bond-days, %d off-the-run bond-days",
        len(etf_df),
        len(otr_df),
    )
    return etf_df, otr_df


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _segment_metrics(
    backtest_results: dict[str, Any],
    mask: pd.Series,
) -> dict[str, Any]:
    """Extract per-regime metrics for a subset of bond-days defined by mask."""
    out: dict[str, Any] = {}
    for regime in ["flat", "dynamic", "oracle"]:
        rdf = backtest_results[regime]["df"]
        sub = rdf.loc[mask]
        filled = sub[sub["filled"]]

        avg_markout = 0.0
        if "markout_t1_bps" in filled.columns:
            m = filled["markout_t1_bps"].dropna()
            avg_markout = float(m.mean()) if len(m) > 0 else 0.0

        adverse = 0.0
        if "markout_t1_bps" in filled.columns:
            neg = filled.loc[filled["markout_t1_bps"] < 0, "markout_t1_bps"]
            adverse = float(neg.mean()) if len(neg) > 0 else 0.0

        out[regime] = {
            "fill_rate": float(sub["filled"].mean()) if len(sub) > 0 else 0.0,
            "avg_markout_t1_bps": avg_markout,
            "adverse_selection_bps": adverse,
            "n_filled": int(filled["filled"].sum()) if len(filled) > 0 else 0,
            "n_total": len(sub),
        }
    return out


def compare_segments(
    backtest_results: dict[str, Any],
    config: dict,
) -> dict[str, Any]:
    """Compute per-segment, per-regime metrics.

    Args:
        backtest_results: Dict returned by ``run_backtest`` (must contain
            DataFrames under each regime key).
        config: Full config dict.

    Returns:
        Dict with keys ``etf_eligible`` and ``off_the_run``, each mapping
        regime names to metric dicts.
    """
    # Use the flat regime's DataFrame as reference for segmentation
    ref_df = backtest_results["flat"]["df"]
    etf_df, otr_df = segment_by_etf(ref_df)

    etf_idx = etf_df.index
    otr_idx = otr_df.index

    etf_mask = ref_df.index.isin(etf_idx)
    otr_mask = ref_df.index.isin(otr_idx)

    comparison = {
        "etf_eligible": _segment_metrics(backtest_results, etf_mask),
        "off_the_run": _segment_metrics(backtest_results, otr_mask),
    }

    # Compute improvement of dynamic over flat for each segment
    for seg in ["etf_eligible", "off_the_run"]:
        flat_m = comparison[seg]["flat"]["avg_markout_t1_bps"]
        dyn_m = comparison[seg]["dynamic"]["avg_markout_t1_bps"]
        comparison[seg]["dynamic_vs_flat_improvement_bps"] = dyn_m - flat_m

    logger.info(
        "ETF-eligible dynamic vs flat: %.2f bps | Off-the-run: %.2f bps",
        comparison["etf_eligible"]["dynamic_vs_flat_improvement_bps"],
        comparison["off_the_run"]["dynamic_vs_flat_improvement_bps"],
    )
    return comparison


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_segment_comparison(
    results: dict[str, Any],
    save_path: Path | str,
) -> None:
    """Create grouped bar chart of markout improvement by segment and regime.

    X-axis: {ETF-eligible, Off-the-run}
    Groups: {Flat, Dynamic, Oracle}
    Y-axis: Average markout (bps)

    Args:
        results: Dict from ``compare_segments``.
        save_path: Path for the output PNG.
    """
    config = _load_config()
    plot_cfg = config.get("plotting", {})
    sns.set_theme(style=plot_cfg.get("style", "whitegrid"))
    sns.set_context(plot_cfg.get("context", "paper"), font_scale=plot_cfg.get("font_scale", 1.2))

    segments = ["etf_eligible", "off_the_run"]
    segment_labels = ["ETF-Eligible", "Off-the-Run"]
    regimes = ["flat", "dynamic", "oracle"]
    regime_labels = ["Flat", "Dynamic", "Oracle"]
    colors = ["#4c72b0", "#55a868", "#c44e52"]

    # Build data for grouped bar chart
    x = np.arange(len(segments))
    width = 0.25

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Panel 1: Average markout ----
    ax = axes[0]
    for i, (regime, label, color) in enumerate(zip(regimes, regime_labels, colors)):
        vals = [results[seg][regime]["avg_markout_t1_bps"] for seg in segments]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_xticks(x + width)
    ax.set_xticklabels(segment_labels)
    ax.set_ylabel("Avg Markout t+1 (bps)")
    ax.set_title("Markout by Segment & Regime")
    ax.legend()

    # ---- Panel 2: Fill rate ----
    ax2 = axes[1]
    for i, (regime, label, color) in enumerate(zip(regimes, regime_labels, colors)):
        vals = [results[seg][regime]["fill_rate"] * 100 for seg in segments]
        bars = ax2.bar(x + i * width, vals, width, label=label, color=color)
        for bar, v in zip(bars, vals):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(segment_labels)
    ax2.set_ylabel("Fill Rate (%)")
    ax2.set_title("Fill Rate by Segment & Regime")
    ax2.legend()

    fig.suptitle("ETF-Eligible vs Off-the-Run: Spread Policy Impact", fontsize=13, y=1.02)
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=plot_cfg.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ETF segment comparison chart to %s", save_path)


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


def run_etf_segment_analysis(
    features_df: pd.DataFrame,
    backtest_results: dict[str, Any],
    config: dict,
) -> dict[str, Any]:
    """Full ETF-segment analysis pipeline.

    Args:
        features_df: Bond-day panel (used for segmentation metadata).
        backtest_results: Dict from ``run_backtest``.
        config: Full config dict.

    Returns:
        Comparison dict with per-segment, per-regime metrics.
    """
    fig_dir, res_dir = _ensure_dirs(config)

    comparison = compare_segments(backtest_results, config)

    # Save JSON
    json_path = res_dir / "etf_segment_results.json"
    # Strip DataFrames for serialisation
    serialisable = {}
    for seg in ["etf_eligible", "off_the_run"]:
        serialisable[seg] = {
            k: v for k, v in comparison[seg].items() if not isinstance(v, pd.DataFrame)
        }
    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info("Saved ETF segment results to %s", json_path)

    # Plot
    chart_path = fig_dir / "etf_segment_comparison.png"
    plot_segment_comparison(comparison, chart_path)

    return comparison
