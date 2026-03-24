"""
run_all.py -- Master orchestrator for the Bond Illiquidity Prediction Engine.

Runs all pipeline steps end-to-end:
  1. Fetch/generate TRACE data
  2. Fetch ETF holdings
  3. Fetch FRED macro data
  4. Engineer TRACE features + targets
  5. Merge ETF features
  6. Merge macro features
  7. Train LightGBM model (Optuna search)
  8. Calibrate model (Platt scaling)
  9. Run dealer markout backtest
  10. ETF segment analysis
  11. Redemption basket optimisation
  12. SHAP / feature importance plots
  13. Save timing + results summary

Usage:
    python run_all.py
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_all")

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def _load_config() -> Dict[str, Any]:
    """Load config.yaml and attach _project_root for downstream modules."""
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["_project_root"] = str(PROJECT_ROOT)
    return cfg


def _ensure_dirs(config: Dict[str, Any]) -> None:
    """Create output directories if they do not exist."""
    for key in ["outputs_dir", "figures_dir", "results_dir", "cache_dir"]:
        d = PROJECT_ROOT / config["paths"][key]
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def step_fetch_trace(config: Dict[str, Any]) -> pd.DataFrame:
    """Step 1: Fetch or generate TRACE trade-level data."""
    from data.fetch_trace import fetch_trace_data

    logger.info("=" * 60)
    logger.info("STEP 1 — Fetch TRACE data")
    try:
        trace_df = fetch_trace_data(config=config, seed=42)
    except Exception as exc:
        logger.warning("TRACE fetch failed (%s); retrying with defaults", exc)
        trace_df = fetch_trace_data(seed=42)
    logger.info("TRACE data: %d rows, %d columns", len(trace_df), len(trace_df.columns))
    return trace_df


def step_aggregate_trace(trace_df: pd.DataFrame) -> pd.DataFrame:
    """Step 1b: Aggregate trade-level TRACE to CUSIP-day panel."""
    from data.fetch_trace import aggregate_cusip_day

    cusip_day = aggregate_cusip_day(trace_df)
    logger.info("CUSIP-day panel: %d rows", len(cusip_day))
    return cusip_day


def step_fetch_etf(
    trace_df: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:
    """Step 2: Fetch ETF holdings."""
    from data.fetch_etf_holdings import fetch_etf_holdings

    logger.info("=" * 60)
    logger.info("STEP 2 — Fetch ETF holdings")
    # trace_df already has 'cusip' and 'is_ig' columns needed by fetch_etf_holdings
    holdings = fetch_etf_holdings(trace_cusips=trace_df[["cusip", "is_ig"]].drop_duplicates(), config=config, seed=42)
    logger.info("ETF holdings: %d rows", len(holdings))
    return holdings


def step_fetch_fred(config: Dict[str, Any]) -> pd.DataFrame:
    """Step 3: Fetch FRED macro data."""
    from data.fetch_fred import fetch_fred_data

    logger.info("=" * 60)
    logger.info("STEP 3 — Fetch FRED macro data")
    fred_df = fetch_fred_data(config=config, seed=42)
    logger.info("FRED data: %d rows, %d columns", len(fred_df), len(fred_df.columns))
    return fred_df


def step_trace_features(
    cusip_day: pd.DataFrame, config: Dict[str, Any]
) -> pd.DataFrame:
    """Step 4: Compute TRACE features + target variables."""
    from features.trace_features import build_trace_feature_panel

    logger.info("=" * 60)
    logger.info("STEP 4 — Engineer TRACE features")
    panel = build_trace_feature_panel(cusip_day, config=config)
    logger.info("TRACE feature panel: %d rows, %d columns", len(panel), len(panel.columns))
    return panel


def step_etf_features(
    panel: pd.DataFrame,
    holdings: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Step 5: Merge ETF features onto panel."""
    from data.fetch_etf_holdings import compute_overlap_counts
    from features.etf_features import compute_etf_features

    logger.info("=" * 60)
    logger.info("STEP 5 — Merge ETF features")
    overlap = compute_overlap_counts(holdings, config=config, seed=42)
    panel = compute_etf_features(panel, holdings, overlap, config=config)
    logger.info("Panel with ETF features: %d rows, %d columns", len(panel), len(panel.columns))
    return panel


def step_macro_features(
    panel: pd.DataFrame,
    fred_df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Step 6: Merge macro features onto panel."""
    from features.macro_features import compute_macro_features

    logger.info("=" * 60)
    logger.info("STEP 6 — Merge macro features")
    panel = compute_macro_features(panel, fred_df, config=config)
    logger.info("Panel with macro features: %d rows, %d columns", len(panel), len(panel.columns))
    return panel


def step_save_features(panel: pd.DataFrame, config: Dict[str, Any]) -> None:
    """Save the full feature matrix to disk for the model step."""
    features_dir = PROJECT_ROOT / config["paths"]["features_dir"]
    features_dir.mkdir(parents=True, exist_ok=True)
    csv_path = features_dir / "feature_matrix.csv"
    panel.to_csv(csv_path, index=False)
    logger.info("Saved feature matrix to %s (%d rows)", csv_path, len(panel))


def step_train_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """Step 7: Train LightGBM model with Optuna search."""
    from models.illiquidity_model import run_model_pipeline

    logger.info("=" * 60)
    logger.info("STEP 7 — Train illiquidity model")
    result = run_model_pipeline(config)
    logger.info(
        "Model trained. Test AUC=%.4f, AP=%.4f",
        result["metrics"]["roc_auc"],
        result["metrics"]["average_precision"],
    )
    return result


def step_calibrate(
    model_result: Dict[str, Any], config: Dict[str, Any]
) -> Any:
    """Step 8: Platt-scaling calibration."""
    from models.calibration import run_calibration_pipeline

    logger.info("=" * 60)
    logger.info("STEP 8 — Calibrate model (Platt scaling)")
    calibrated = run_calibration_pipeline(
        model=model_result["model"],
        X_val=model_result["X_test"],
        y_val=model_result["y_test"],
        config=config,
    )
    logger.info("Calibration complete.")
    return calibrated


def step_backtest(
    panel: pd.DataFrame,
    model_result: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Step 9: Dealer markout backtest."""
    from strategy.markout_backtest import generate_backtest_report, run_backtest

    logger.info("=" * 60)
    logger.info("STEP 9 — Run dealer markout backtest")

    # Use test-set predictions as illiquidity scores
    predictions = model_result["predictions"]
    test_idx = model_result["test_idx"]
    test_panel = panel.iloc[test_idx].copy().reset_index(drop=True)

    backtest_results = run_backtest(
        features_df=test_panel,
        model_scores=predictions,
        config=config,
    )
    generate_backtest_report(backtest_results, config)
    return backtest_results


def step_etf_segment(
    panel: pd.DataFrame,
    backtest_results: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Step 10: ETF segment analysis."""
    from strategy.etf_segment_analysis import run_etf_segment_analysis

    logger.info("=" * 60)
    logger.info("STEP 10 — ETF segment analysis")
    comparison = run_etf_segment_analysis(
        features_df=panel,
        backtest_results=backtest_results,
        config=config,
    )
    return comparison


def step_basket(config: Dict[str, Any]) -> Dict[str, Any]:
    """Step 11: Redemption basket optimisation."""
    from strategy.redemption_basket import run_basket_optimization

    logger.info("=" * 60)
    logger.info("STEP 11 — Redemption basket optimisation")
    basket_results = run_basket_optimization(config)
    return basket_results


def step_shap(
    model_result: Dict[str, Any], config: Dict[str, Any]
) -> None:
    """Step 12: SHAP / feature importance analysis."""
    from models.feature_importance import run_feature_importance_pipeline

    logger.info("=" * 60)
    logger.info("STEP 12 — SHAP & feature importance")
    run_feature_importance_pipeline(
        model=model_result["model"],
        X_test=model_result["X_test"],
        feature_names=model_result["feature_names"],
        config=config,
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _save_timing(timings: Dict[str, float], config: Dict[str, Any]) -> None:
    """Save step-level timing to outputs/results/timing.json."""
    results_dir = PROJECT_ROOT / config["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    timing_path = results_dir / "timing.json"
    with open(timing_path, "w") as f:
        json.dump(timings, f, indent=2)
    logger.info("Timing summary saved to %s", timing_path)


def _print_summary(
    timings: Dict[str, float],
    model_result: Dict[str, Any],
    basket_results: Dict[str, Any],
    backtest_results: Dict[str, Any],
) -> None:
    """Print a concise results summary to the console."""
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)

    # Timing
    total = timings.get("total_seconds", 0.0)
    logger.info("Total runtime: %.1f seconds (%.1f minutes)", total, total / 60.0)
    for step, secs in timings.items():
        if step != "total_seconds":
            logger.info("  %-30s %6.1f s", step, secs)

    # Model metrics
    metrics = model_result.get("metrics", {})
    logger.info("Model performance:")
    logger.info("  ROC-AUC          : %.4f", metrics.get("roc_auc", 0))
    logger.info("  Average Precision : %.4f", metrics.get("average_precision", 0))
    logger.info("  Precision@k       : %.4f", metrics.get("precision_at_k", 0))

    # Backtest
    for regime in ["flat", "dynamic", "oracle"]:
        if regime in backtest_results:
            m = backtest_results[regime]["metrics"]
            logger.info(
                "Backtest [%s]: fill=%.1f%%, markout_t1=%.2f bps, adverse=%.2f bps",
                regime,
                m["fill_rate"] * 100,
                m.get("avg_markout_t1_bps", 0),
                m["adverse_selection_bps"],
            )

    # Basket
    if basket_results:
        for btype in ["prorata", "optimized"]:
            if btype in basket_results:
                b = basket_results[btype]
                logger.info(
                    "Basket [%s]: TE=%.1f bps, illiq=%.3f, exec_cost=%.1f bps, n_bonds=%d",
                    btype,
                    b["tracking_error_bps"],
                    b["avg_illiquidity_score"],
                    b["estimated_execution_cost_bps"],
                    b["n_bonds"],
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the full illiquidity prediction pipeline end-to-end."""
    pipeline_start = time.time()
    config = _load_config()
    _ensure_dirs(config)

    timings: Dict[str, float] = {}
    model_result: Dict[str, Any] = {}
    backtest_results: Dict[str, Any] = {}
    basket_results: Dict[str, Any] = {}

    # Helper to time each step
    def timed(name: str, fn, *args, **kwargs):
        t0 = time.time()
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:
            logger.error("Step '%s' FAILED: %s", name, exc, exc_info=True)
            timings[name] = time.time() - t0
            return None
        timings[name] = time.time() - t0
        logger.info("Step '%s' completed in %.1f s", name, timings[name])
        return result

    # Step 1: TRACE data
    trace_df = timed("fetch_trace", step_fetch_trace, config)
    if trace_df is None:
        logger.error("Cannot proceed without TRACE data. Exiting.")
        return

    # Step 1b: Aggregate
    cusip_day = timed("aggregate_trace", step_aggregate_trace, trace_df)
    if cusip_day is None:
        logger.error("Cannot proceed without CUSIP-day panel. Exiting.")
        return

    # Step 2: ETF holdings
    holdings = timed("fetch_etf", step_fetch_etf, trace_df, config)

    # Step 3: FRED data
    fred_df = timed("fetch_fred", step_fetch_fred, config)

    # Step 4: TRACE features
    panel = timed("trace_features", step_trace_features, cusip_day, config)
    if panel is None:
        logger.error("Cannot proceed without features. Exiting.")
        return

    # Step 5: ETF features
    if holdings is not None:
        panel = timed("etf_features", step_etf_features, panel, holdings, config)
        if panel is None:
            logger.error("ETF feature merge failed. Exiting.")
            return
    else:
        logger.warning("Skipping ETF features (no holdings data).")
        # Add placeholder ETF columns so the model can still run
        panel["etf_basket_flag"] = 0
        panel["etf_weight"] = 0.0
        panel["etf_overlap_count"] = 0

    # Step 6: Macro features
    if fred_df is not None:
        panel = timed("macro_features", step_macro_features, panel, fred_df, config)
        if panel is None:
            logger.error("Macro feature merge failed. Exiting.")
            return
    else:
        logger.warning("Skipping macro features (no FRED data).")
        panel["oas_index_level"] = 1.20
        panel["vix_level"] = 18.0
        panel["hy_oas_level"] = 4.20

    # Save feature matrix
    timed("save_features", step_save_features, panel, config)

    # Step 7: Train model
    model_result = timed("train_model", step_train_model, config)
    if model_result is None:
        logger.error("Model training failed. Exiting.")
        return

    # Step 8: Calibrate
    timed("calibrate_model", step_calibrate, model_result, config)

    # Step 9: Backtest
    backtest_results = timed("backtest", step_backtest, panel, model_result, config)
    if backtest_results is None:
        backtest_results = {}

    # Step 10: ETF segment analysis
    if backtest_results:
        timed("etf_segment", step_etf_segment, panel, backtest_results, config)

    # Step 11: Basket optimisation
    basket_results = timed("basket_optimization", step_basket, config)
    if basket_results is None:
        basket_results = {}

    # Step 12: SHAP
    timed("shap_analysis", step_shap, model_result, config)

    # Final timing
    timings["total_seconds"] = time.time() - pipeline_start
    _save_timing(timings, config)

    # Summary
    _print_summary(timings, model_result, basket_results, backtest_results)


if __name__ == "__main__":
    main()
