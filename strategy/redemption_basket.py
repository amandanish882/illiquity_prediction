"""Tracking-error-minimising redemption basket construction.

Uses constrained optimisation (scipy SLSQP) to build a redemption basket
that minimises a combined objective of tracking error and illiquidity cost,
subject to weight, sector, duration, and OAS constraints.
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
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

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
# Pro-rata basket (benchmark)
# ---------------------------------------------------------------------------


def construct_prorata_basket(
    etf_holdings: pd.DataFrame,
    redemption_notional: float,
    config: dict,
) -> pd.DataFrame:
    """Build a pro-rata redemption basket proportional to ETF weights.

    Each bond receives a notional proportional to its weight in the ETF.

    Args:
        etf_holdings: DataFrame with columns ``cusip`` (or ``isin``),
            ``weight``, ``duration``, ``oas``, ``sector``,
            ``illiquidity_score``.  Weights should sum to ~1.
        redemption_notional: Total redemption amount in dollars
            (e.g. 25_000_000).
        config: Full config dict.

    Returns:
        DataFrame with ``cusip``, ``weight``, ``notional``, and original
        bond features.

    Example:
        A bond with weight=0.005 in a $25M redemption gets
        notional = 0.005 * 25_000_000 = $125,000.
    """
    basket = etf_holdings.copy()
    # Normalise weights to sum to 1
    total_weight = basket["weight"].sum()
    if total_weight > 0:
        basket["weight"] = basket["weight"] / total_weight
    basket["notional"] = basket["weight"] * redemption_notional
    basket["basket_type"] = "prorata"

    logger.info(
        "Pro-rata basket: %d bonds, avg illiquidity=%.3f, total=$%.0f",
        len(basket),
        basket["illiquidity_score"].mean(),
        basket["notional"].sum(),
    )
    return basket


# ---------------------------------------------------------------------------
# Optimised basket
# ---------------------------------------------------------------------------


def construct_optimized_basket(
    etf_holdings: pd.DataFrame,
    illiquidity_scores: np.ndarray | pd.Series,
    bond_features: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Build a tracking-error-minimising basket via constrained optimisation.

    Objective:
        min  TE(w) + illiquidity_penalty * sum(w_i * illiquidity_i)

    where TE(w) captures deviations in duration, OAS, and sector
    composition relative to the index.

    Constraints:
        - sum(w) = 1
        - 0 <= w_i <= max_weight_per_bond  (e.g. 10%)
        - Number of non-zero weights >= min_bonds  (e.g. 20)
        - Sector weights within +/- sector_tolerance of index
        - Portfolio duration within +/- duration_tolerance of index
        - Portfolio OAS within +/- oas_tolerance of index

    Args:
        etf_holdings: Index/ETF composition with columns ``cusip``,
            ``weight``, ``duration``, ``oas``, ``sector``.
        illiquidity_scores: Illiquidity scores aligned to etf_holdings rows.
        bond_features: Bond-level features (same index as etf_holdings).
        config: Full config dict.

    Returns:
        DataFrame with optimised weights, notionals, and features.
    """
    bkt_cfg = config["basket"]
    redemption_notional: float = bkt_cfg["redemption_notional"]
    max_w: float = bkt_cfg["max_weight_per_bond"]
    min_bonds: int = bkt_cfg["min_bonds"]
    sector_tol: float = bkt_cfg["sector_tolerance"]
    dur_tol: float = bkt_cfg["duration_tolerance"]
    oas_tol: float = bkt_cfg["oas_tolerance"]
    illiq_penalty: float = bkt_cfg["illiquidity_penalty"]

    n = len(etf_holdings)
    scores = np.asarray(illiquidity_scores, dtype=np.float64)

    # Index characteristics (target)
    idx_weights = etf_holdings["weight"].values / etf_holdings["weight"].sum()
    durations = etf_holdings["duration"].values.astype(np.float64)
    oas_vals = etf_holdings["oas"].values.astype(np.float64)

    idx_duration = float(np.dot(idx_weights, durations))
    idx_oas = float(np.dot(idx_weights, oas_vals))

    # Sector composition: build sector indicator matrix
    sectors = etf_holdings["sector"].values
    unique_sectors = np.unique(sectors)
    # S[k, i] = 1 if bond i is in sector k
    sector_matrix = np.zeros((len(unique_sectors), n))
    for k, sec in enumerate(unique_sectors):
        sector_matrix[k, :] = (sectors == sec).astype(float)

    idx_sector_weights = sector_matrix @ idx_weights  # target sector weights

    # ---- Objective function ----
    def objective(w: np.ndarray) -> float:
        # Tracking error components
        port_dur = np.dot(w, durations)
        port_oas = np.dot(w, oas_vals)
        port_sectors = sector_matrix @ w

        # Duration TE (years squared, scaled)
        dur_te = (port_dur - idx_duration) ** 2

        # OAS TE (bps squared, scaled)
        oas_te = ((port_oas - idx_oas) / 10.0) ** 2  # scale so 10 bps deviation ~ 1

        # Sector TE (sum of squared deviations)
        sector_te = float(np.sum((port_sectors - idx_sector_weights) ** 2))

        # Weight deviation from index (encourages diversification)
        weight_te = float(np.sum((w - idx_weights) ** 2))

        # Combined TE
        te = dur_te + oas_te + 10.0 * sector_te + weight_te

        # Illiquidity penalty
        illiq_cost = illiq_penalty * float(np.dot(w, scores))

        return te + illiq_cost

    # ---- Constraints ----
    # 1. Weights sum to 1
    eq_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    constraints = [eq_constraint]

    # 2. Duration within tolerance
    constraints.append({
        "type": "ineq",
        "fun": lambda w: dur_tol - abs(np.dot(w, durations) - idx_duration),
    })

    # 3. OAS within tolerance
    constraints.append({
        "type": "ineq",
        "fun": lambda w: oas_tol - abs(np.dot(w, oas_vals) - idx_oas),
    })

    # 4. Sector weights within tolerance
    for k in range(len(unique_sectors)):
        target_k = idx_sector_weights[k]
        row_k = sector_matrix[k, :]
        constraints.append({
            "type": "ineq",
            "fun": lambda w, r=row_k, t=target_k: sector_tol - abs(np.dot(r, w) - t),
        })

    # 5. Minimum number of bonds with non-zero weight
    # Enforced via a soft penalty: we add a constraint that sum(w > threshold) >= min_bonds
    # Since SLSQP doesn't handle cardinality constraints natively, we use
    # a threshold-based approach: count bonds with w >= 0.001 (0.1%)
    min_w_threshold = 1e-3
    constraints.append({
        "type": "ineq",
        "fun": lambda w: np.sum(w >= min_w_threshold) - min_bonds,
    })

    # ---- Bounds: 0 <= w_i <= max_weight ----
    bounds = [(0.0, max_w) for _ in range(n)]

    # ---- Initial guess: start from index weights (clamped) ----
    w0 = np.clip(idx_weights, 0.0, max_w)
    w0 = w0 / w0.sum()  # renormalise

    # ---- Solve ----
    logger.info(
        "Optimising basket: %d bonds, min_bonds=%d, max_weight=%.1f%%, "
        "illiq_penalty=%.2f",
        n,
        min_bonds,
        max_w * 100,
        illiq_penalty,
    )

    result = minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10, "disp": False},
    )

    if not result.success:
        logger.warning("Optimisation did not converge: %s", result.message)

    opt_weights = result.x
    # Zero out very small weights
    opt_weights[opt_weights < min_w_threshold] = 0.0
    # Renormalise
    if opt_weights.sum() > 0:
        opt_weights = opt_weights / opt_weights.sum()

    # Build output DataFrame
    basket = etf_holdings.copy()
    basket["weight"] = opt_weights
    basket["notional"] = opt_weights * redemption_notional
    basket["illiquidity_score"] = scores
    basket["basket_type"] = "optimized"

    active_bonds = int((opt_weights > 0).sum())
    port_dur = float(np.dot(opt_weights, durations))
    port_oas = float(np.dot(opt_weights, oas_vals))
    avg_illiq = float(np.dot(opt_weights, scores))

    logger.info(
        "Optimised basket: %d active bonds (of %d), duration=%.2f (idx=%.2f), "
        "OAS=%.1f (idx=%.1f), avg_illiq=%.3f",
        active_bonds,
        n,
        port_dur,
        idx_duration,
        port_oas,
        idx_oas,
        avg_illiq,
    )
    return basket


# ---------------------------------------------------------------------------
# Tracking error estimation
# ---------------------------------------------------------------------------


def estimate_tracking_error(
    basket: pd.DataFrame,
    index: pd.DataFrame,
    bond_features: pd.DataFrame,
) -> float:
    """Estimate annualised tracking error of basket vs index.

    Uses a simplified factor model: TE^2 = (dur_diff)^2 * rate_vol^2
    + (oas_diff)^2 * spread_vol^2 + sector_mismatch_penalty.

    Rate vol ~ 80 bps/year, spread vol ~ 50 bps/year for IG.

    Args:
        basket: Basket DataFrame with ``weight``, ``duration``, ``oas``,
            ``sector``.
        index: Index DataFrame with same columns.
        bond_features: Not used in simplified model but reserved for
            full covariance approach.

    Returns:
        Estimated annualised tracking error in bps.

    Example:
        Duration diff of 0.1 yr with 80 bps rate vol -> 8 bps TE
        contribution. OAS diff of 5 bps with 50 bps spread vol -> 2.5 bps.
    """
    rate_vol_annual_bps = 80.0    # typical IG rate vol
    spread_vol_annual_bps = 50.0  # typical IG spread vol

    # Basket characteristics
    bw = basket["weight"].values
    bw = bw / bw.sum() if bw.sum() > 0 else bw

    iw = index["weight"].values
    iw = iw / iw.sum() if iw.sum() > 0 else iw

    basket_dur = float(np.dot(bw, basket["duration"].values))
    index_dur = float(np.dot(iw, index["duration"].values))
    dur_diff = basket_dur - index_dur

    basket_oas = float(np.dot(bw, basket["oas"].values))
    index_oas = float(np.dot(iw, index["oas"].values))
    oas_diff = basket_oas - index_oas

    # Sector mismatch
    sectors = np.unique(np.concatenate([basket["sector"].values, index["sector"].values]))
    sector_mismatch = 0.0
    for sec in sectors:
        b_sec_w = float(bw[basket["sector"].values == sec].sum()) if sec in basket["sector"].values else 0.0
        i_sec_w = float(iw[index["sector"].values == sec].sum()) if sec in index["sector"].values else 0.0
        sector_mismatch += (b_sec_w - i_sec_w) ** 2

    # TE in bps (annualised)
    te_squared = (
        (dur_diff * rate_vol_annual_bps) ** 2
        + (oas_diff / 10.0 * spread_vol_annual_bps) ** 2
        + sector_mismatch * (30.0 ** 2)  # 30 bps penalty per unit sector mismatch
    )
    te_bps = float(np.sqrt(te_squared))
    return te_bps


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_baskets(
    prorata: pd.DataFrame,
    optimized: pd.DataFrame,
    config: dict,
) -> dict[str, Any]:
    """Compare pro-rata and optimised baskets on key metrics.

    Args:
        prorata: Pro-rata basket DataFrame.
        optimized: Optimised basket DataFrame.
        config: Full config dict.

    Returns:
        Dict with side-by-side metrics for both baskets.
    """
    index_df = prorata.copy()  # pro-rata mirrors the index composition

    te_prorata = estimate_tracking_error(prorata, index_df, prorata)
    te_optimized = estimate_tracking_error(optimized, index_df, optimized)

    pw = prorata["weight"].values
    pw = pw / pw.sum() if pw.sum() > 0 else pw
    ow = optimized["weight"].values
    ow = ow / ow.sum() if ow.sum() > 0 else ow

    avg_illiq_prorata = float(np.dot(pw, prorata["illiquidity_score"].values))
    avg_illiq_optimized = float(np.dot(ow, optimized["illiquidity_score"].values))

    # Estimated execution cost: proportional to illiquidity score
    # Use the piecewise spread policy to estimate cost
    from strategy.spread_policy import piecewise_spread_policy

    exec_cost_prorata = float(np.dot(
        pw,
        prorata["illiquidity_score"].apply(
            lambda s: piecewise_spread_policy(s, config)
        ).values,
    ))
    exec_cost_optimized = float(np.dot(
        ow,
        optimized["illiquidity_score"].apply(
            lambda s: piecewise_spread_policy(s, config)
        ).values,
    ))

    comparison = {
        "prorata": {
            "tracking_error_bps": te_prorata,
            "avg_illiquidity_score": avg_illiq_prorata,
            "estimated_execution_cost_bps": exec_cost_prorata,
            "n_bonds": int((pw > 0).sum()),
            "avg_duration": float(np.dot(pw, prorata["duration"].values)),
            "avg_oas": float(np.dot(pw, prorata["oas"].values)),
        },
        "optimized": {
            "tracking_error_bps": te_optimized,
            "avg_illiquidity_score": avg_illiq_optimized,
            "estimated_execution_cost_bps": exec_cost_optimized,
            "n_bonds": int((ow > 0).sum()),
            "avg_duration": float(np.dot(ow, optimized["duration"].values)),
            "avg_oas": float(np.dot(ow, optimized["oas"].values)),
        },
    }

    logger.info(
        "Basket comparison: prorata TE=%.1f bps illiq=%.3f | "
        "optimized TE=%.1f bps illiq=%.3f",
        te_prorata,
        avg_illiq_prorata,
        te_optimized,
        avg_illiq_optimized,
    )
    return comparison


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_basket_comparison(
    comparison: dict[str, Any],
    save_path: Path | str,
) -> None:
    """Create side-by-side bar charts comparing pro-rata and optimised baskets.

    Args:
        comparison: Dict from ``compare_baskets``.
        save_path: Path for the output PNG.
    """
    config = _load_config()
    plot_cfg = config.get("plotting", {})
    sns.set_theme(style=plot_cfg.get("style", "whitegrid"))
    sns.set_context(plot_cfg.get("context", "paper"), font_scale=plot_cfg.get("font_scale", 1.2))

    baskets = ["prorata", "optimized"]
    basket_labels = ["Pro-Rata", "Optimised"]
    colors = ["#4c72b0", "#55a868"]

    metrics = [
        ("tracking_error_bps", "Tracking Error (bps)", "{:.1f}"),
        ("avg_illiquidity_score", "Avg Illiquidity Score", "{:.3f}"),
        ("estimated_execution_cost_bps", "Est. Execution Cost (bps)", "{:.1f}"),
        ("n_bonds", "Number of Bonds", "{:.0f}"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))

    for ax, (key, label, fmt) in zip(axes, metrics):
        vals = [comparison[b][key] for b in baskets]
        bars = ax.bar(basket_labels, vals, color=colors)
        ax.set_title(label, fontsize=11)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle(
        f"Redemption Basket Comparison — {config['basket']['target_etf']} "
        f"(${config['basket']['redemption_notional'] / 1e6:.0f}M)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=plot_cfg.get("dpi", 150), bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved basket comparison chart to %s", save_path)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def _generate_synthetic_holdings(config: dict) -> pd.DataFrame:
    """Synthetic holdings are no longer supported.

    Real ETF holdings must be fetched first via the ETF holdings pipeline.
    """
    raise RuntimeError(
        "Real ETF holdings required. Run the ETF holdings fetch step first. "
        "Synthetic generation has been removed to prevent silent use of fake data."
    )


def run_basket_optimization(config: dict) -> dict[str, Any]:
    """Full redemption basket optimisation pipeline.

    Loads (or synthesises) ETF holdings, constructs pro-rata and optimised
    baskets, compares them, and saves results and charts.

    Args:
        config: Full config dict.

    Returns:
        Comparison dict with metrics for both baskets.
    """
    fig_dir, res_dir = _ensure_dirs(config)
    redemption_notional: float = config["basket"]["redemption_notional"]

    # Load real ETF holdings — no synthetic fallback
    holdings_path = _PROJECT_ROOT / config["paths"]["cache_dir"] / "etf_holdings.parquet"
    if not holdings_path.exists():
        raise RuntimeError(
            f"No ETF holdings file at {holdings_path}. "
            "Run the ETF holdings fetch step first."
        )

    etf_holdings = pd.read_parquet(holdings_path)
    logger.info("Loaded ETF holdings from %s (%d rows)", holdings_path, len(etf_holdings))

    # Filter to target ETF if 'ticker' column exists
    target_etf = config["basket"].get("target_etf", "LQD")
    if "ticker" in etf_holdings.columns:
        etf_subset = etf_holdings[etf_holdings["ticker"] == target_etf].copy()
        if len(etf_subset) > 0:
            etf_holdings = etf_subset
        else:
            raise RuntimeError(
                f"No holdings found for target ETF '{target_etf}' in {holdings_path}."
            )

    # ---- Join real bond-level data from feature matrix ----
    feature_path = _PROJECT_ROOT / config["paths"]["features_dir"] / "feature_matrix.csv"
    if not feature_path.exists():
        raise RuntimeError(
            f"Feature matrix not found at {feature_path}. "
            "Run the feature engineering step first."
        )

    fm = pd.read_csv(
        feature_path,
        usecols=["cusip", "date", "mod_dur", "credit_spread", "sector", "illiquidity_t1"],
    )
    logger.info("Loaded feature matrix: %d rows, %d unique CUSIPs", len(fm), fm["cusip"].nunique())

    # Take the most recent observation per CUSIP that has a valid illiquidity score.
    # The last day per CUSIP has NaN illiquidity_t1 (forward-shifted target),
    # so we drop NaN rows first, then take the latest remaining observation.
    fm["date"] = pd.to_datetime(fm["date"])
    fm = fm.dropna(subset=["illiquidity_t1"])
    fm = fm.sort_values("date").groupby("cusip").tail(1).reset_index(drop=True)

    # Convert credit_spread from decimal to bps (e.g. 0.0093 -> 93 bps)
    fm["oas_bps"] = fm["credit_spread"] * 10_000

    # Normalize illiquidity_t1 to [0, 1] percentile rank.
    # Raw Amihud ratios are ~1e-8 scale, too small for the optimizer's penalty term.
    # Percentile rank preserves the ordering while making the penalty meaningful.
    fm["illiquidity_t1"] = fm["illiquidity_t1"].rank(pct=True)

    # Merge onto holdings
    etf_holdings = etf_holdings.merge(
        fm[["cusip", "mod_dur", "oas_bps", "sector", "illiquidity_t1"]].rename(
            columns={
                "mod_dur": "duration",
                "oas_bps": "oas",
                "sector": "sector_fm",
                "illiquidity_t1": "illiquidity_score",
            }
        ),
        on="cusip",
        how="left",
    )

    # Prefer feature-matrix sector over holdings sector where available
    if "sector_fm" in etf_holdings.columns:
        if "sector" in etf_holdings.columns:
            etf_holdings["sector"] = etf_holdings["sector_fm"].fillna(etf_holdings["sector"])
        else:
            etf_holdings["sector"] = etf_holdings["sector_fm"]
        etf_holdings.drop(columns=["sector_fm"], inplace=True)

    # Check coverage and warn loudly about unmatched CUSIPs
    missing_mask = etf_holdings["duration"].isna() | etf_holdings["oas"].isna()
    n_missing = int(missing_mask.sum())
    if n_missing > 0:
        missing_cusips = etf_holdings.loc[missing_mask, "cusip"].tolist()
        logger.warning(
            "WARNING: %d of %d ETF holdings CUSIPs have no TRACE/feature data "
            "and will be DROPPED from basket optimisation. Missing CUSIPs (first 20): %s",
            n_missing,
            len(etf_holdings),
            missing_cusips[:20],
        )
        etf_holdings = etf_holdings[~missing_mask].copy()

    if len(etf_holdings) == 0:
        raise RuntimeError(
            "No ETF holdings could be matched to feature matrix data. "
            "Check that CUSIP formats match between etf_holdings.parquet and feature_matrix.csv."
        )

    # Ensure sector and illiquidity_score are present
    if "sector" not in etf_holdings.columns or etf_holdings["sector"].isna().all():
        raise RuntimeError(
            "No sector data available after joining feature matrix. "
            "Check the feature_matrix.csv for a 'sector' column."
        )
    if "illiquidity_score" not in etf_holdings.columns or etf_holdings["illiquidity_score"].isna().all():
        raise RuntimeError(
            "No illiquidity scores available after joining feature matrix. "
            "Check the feature_matrix.csv for an 'illiquidity_t1' column."
        )

    # Fill any remaining NaN illiquidity scores with the median (rare edge case)
    if etf_holdings["illiquidity_score"].isna().any():
        median_illiq = etf_holdings["illiquidity_score"].median()
        n_filled = int(etf_holdings["illiquidity_score"].isna().sum())
        logger.warning(
            "Filling %d NaN illiquidity_score values with median (%.6f)",
            n_filled,
            median_illiq,
        )
        etf_holdings["illiquidity_score"] = etf_holdings["illiquidity_score"].fillna(median_illiq)

    # Re-normalise weights after dropping unmatched CUSIPs
    etf_holdings["weight"] = etf_holdings["weight"] / etf_holdings["weight"].sum()

    logger.info(
        "Holdings ready for optimisation: %d bonds, avg duration=%.2f yr, "
        "avg OAS=%.1f bps, avg illiquidity=%.6f",
        len(etf_holdings),
        etf_holdings["duration"].mean(),
        etf_holdings["oas"].mean(),
        etf_holdings["illiquidity_score"].mean(),
    )

    # Build baskets
    prorata = construct_prorata_basket(etf_holdings, redemption_notional, config)
    optimized = construct_optimized_basket(
        etf_holdings,
        etf_holdings["illiquidity_score"],
        etf_holdings,
        config,
    )

    # Compare
    comparison = compare_baskets(prorata, optimized, config)

    # Save results
    json_path = res_dir / "basket_results.json"
    with open(json_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info("Saved basket results to %s", json_path)

    # Plot
    chart_path = fig_dir / "basket_comparison.png"
    plot_basket_comparison(comparison, chart_path)

    return comparison
