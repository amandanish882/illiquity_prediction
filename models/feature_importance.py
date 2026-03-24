"""
Feature importance analysis using SHAP (TreeExplainer).

Generates:
    1. SHAP summary (beeswarm) plot
    2. SHAP dependence plot for etf_basket_flag
    3. Traditional feature-importance bar chart (LightGBM split-based)

Key finding to verify: Meli & Todorova (2017, Bank of England) show that bonds
held in ETF baskets are structurally more liquid because the creation/redemption
mechanism provides a continuous secondary-market channel.  We expect
etf_basket_flag (and etf_weight / etf_overlap_count) to carry negative SHAP
values for the illiquidity prediction — i.e., ETF-basket membership *reduces*
predicted illiquidity.  If etf_basket_flag has negligible importance, the ETF
liquidity channel may already be captured by spread or volume features, or the
sample may lack sufficient variation.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from lightgbm import LGBMClassifier

logger = logging.getLogger(__name__)


def compute_shap_values(
    model: LGBMClassifier,
    X: pd.DataFrame,
    feature_names: List[str],
) -> shap.Explanation:
    """Compute SHAP values for a LightGBM model using TreeExplainer.

    TreeExplainer runs in O(TLD) time (T trees, L leaves, D depth) which is
    far faster than the model-agnostic KernelExplainer.  For a 500-tree model
    on 50 000 rows, this typically completes in under 30 seconds.

    Args:
        model: Fitted LGBMClassifier.
        X: Feature matrix (test set recommended to avoid train-set bias).
        feature_names: Ordered list of feature column names.

    Returns:
        shap.Explanation object with .values, .base_values, .data, and
        .feature_names populated.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # For binary classification, TreeExplainer may return values for both
    # classes.  We want the positive-class (illiquid = 1) SHAP values.
    if isinstance(shap_values.values, list):
        # Older SHAP versions return a list [class_0, class_1]
        shap_values = shap_values[..., 1]
    elif shap_values.values.ndim == 3:
        # Shape (n_samples, n_features, n_classes) — take class 1
        shap_values = shap_values[..., 1]

    shap_values.feature_names = feature_names
    logger.info(
        "SHAP values computed: %d samples x %d features.",
        shap_values.values.shape[0],
        shap_values.values.shape[1],
    )
    return shap_values


def plot_shap_summary(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    save_path: Path,
) -> None:
    """Generate and save a SHAP beeswarm summary plot.

    Each dot is one observation; the x-axis is the SHAP value (impact on
    log-odds of being illiquid), and the colour encodes the feature value.
    For example, a high spread_bps value (red dot) far to the right means
    wide spreads strongly increase predicted illiquidity.

    Args:
        shap_values: SHAP Explanation from compute_shap_values().
        X: Original feature matrix (used for colour encoding).
        save_path: PNG destination path.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    shap.plots.beeswarm(shap_values, show=False, max_display=16)

    plt.title("SHAP Summary — Feature Impact on Illiquidity Prediction")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    logger.info("SHAP summary plot saved to %s", save_path)


def plot_shap_dependence(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    feature_name: str,
    save_path: Path,
) -> None:
    """Generate a SHAP dependence plot for a single feature.

    For etf_basket_flag (binary 0/1), this produces a strip-plot showing
    how ETF-basket membership shifts the predicted illiquidity.  Per
    Meli & Todorova, we expect the SHAP value for etf_basket_flag = 1 to
    be predominantly negative (reducing illiquidity).

    Args:
        shap_values: SHAP Explanation from compute_shap_values().
        X: Original feature matrix.
        feature_name: The feature to plot (e.g. "etf_basket_flag").
        save_path: PNG destination path.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    feature_idx = list(shap_values.feature_names).index(feature_name)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    shap.plots.scatter(
        shap_values[:, feature_idx],
        color=shap_values,
        show=False,
    )

    plt.title(f"SHAP Dependence — {feature_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    # Log interpretation
    shap_vals = shap_values.values[:, feature_idx]
    feature_vals = np.asarray(X[feature_name])
    mask_1 = feature_vals == 1
    mask_0 = feature_vals == 0
    if mask_1.sum() > 0 and mask_0.sum() > 0:
        mean_shap_1 = float(np.mean(shap_vals[mask_1]))
        mean_shap_0 = float(np.mean(shap_vals[mask_0]))
        logger.info(
            "SHAP dependence for %s: mean SHAP when flag=1 is %.4f, flag=0 is %.4f.  "
            "Difference: %.4f.  %s",
            feature_name,
            mean_shap_1,
            mean_shap_0,
            mean_shap_1 - mean_shap_0,
            (
                "ETF-basket bonds show LOWER predicted illiquidity (consistent with "
                "Meli & Todorova)."
                if mean_shap_1 < mean_shap_0
                else "ETF-basket effect is weak or offset by other features — the "
                "liquidity channel may already be captured by spread/volume."
            ),
        )

    logger.info("SHAP dependence plot saved to %s", save_path)


def plot_feature_importance(
    model: LGBMClassifier,
    feature_names: List[str],
    save_path: Path,
) -> None:
    """Generate a horizontal bar chart of LightGBM split-based feature importance.

    Args:
        model: Fitted LGBMClassifier.
        feature_names: Feature column names matching model training order.
        save_path: PNG destination path.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)

    names_sorted = [feature_names[i] for i in sorted_idx]
    vals_sorted = importances[sorted_idx]

    ax.barh(range(len(names_sorted)), vals_sorted, color=sns.color_palette("viridis", len(names_sorted)))
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted)
    ax.set_xlabel("Split-based importance (number of splits)")
    ax.set_title("LightGBM Feature Importance")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Feature importance bar chart saved to %s", save_path)


def run_feature_importance_pipeline(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
    feature_names: List[str],
    config: Dict[str, Any],
) -> None:
    """Run the full feature-importance analysis and save all outputs.

    Generates three plots:
        1. SHAP beeswarm summary
        2. SHAP dependence for etf_basket_flag
        3. Split-based feature importance bar chart

    Also saves a JSON summary of mean absolute SHAP values per feature
    to outputs/results/shap_importance.json.

    Args:
        model: Fitted LGBMClassifier.
        X_test: Test-set feature matrix.
        feature_names: Ordered list of feature column names.
        config: Parsed config dict.
    """
    project_root = Path(config.get("_project_root", "."))
    figures_dir = project_root / config["paths"]["figures_dir"]
    results_dir = project_root / config["paths"]["results_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Compute SHAP values
    shap_values = compute_shap_values(model, X_test, feature_names)

    # 2. Summary plot
    plot_shap_summary(shap_values, X_test, figures_dir / "shap_summary.png")

    # 3. Dependence plot for etf_basket_flag
    plot_shap_dependence(
        shap_values, X_test, "etf_basket_flag", figures_dir / "shap_etf_dependence.png"
    )

    # 4. Traditional feature importance
    plot_feature_importance(model, feature_names, figures_dir / "feature_importance.png")

    # 5. Save numeric SHAP importance summary
    mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
    shap_importance = {
        name: round(float(val), 6)
        for name, val in sorted(
            zip(feature_names, mean_abs_shap), key=lambda x: -x[1]
        )
    }

    shap_path = results_dir / "shap_importance.json"
    with open(shap_path, "w") as f:
        json.dump(shap_importance, f, indent=2)
    logger.info("SHAP importance summary saved to %s", shap_path)

    # 6. Log ETF finding
    etf_rank = list(shap_importance.keys()).index("etf_basket_flag") + 1
    logger.info(
        "etf_basket_flag ranks #%d out of %d features by mean |SHAP|.  "
        "Meli & Todorova predict ETF-basket bonds are structurally more liquid; "
        "%s",
        etf_rank,
        len(feature_names),
        (
            "this feature is highly influential — consistent with the ETF liquidity channel."
            if etf_rank <= 5
            else "this feature has moderate-to-low importance — the ETF effect may be "
            "subsumed by spread and volume features that already reflect ETF-driven liquidity."
        ),
    )
