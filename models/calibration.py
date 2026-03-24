"""
Probability calibration for the illiquidity classifier.

Applies Platt scaling (logistic sigmoid) via CalibratedClassifierCV so that
predicted probabilities match observed frequencies.  Generates before/after
calibration curves and saves the plot to outputs/figures/.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


def calibrate_model(
    model: BaseEstimator,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
) -> CalibratedClassifierCV:
    """Wrap a trained classifier with Platt scaling (sigmoid calibration).

    The calibration is fit on (X_val, y_val), which should be a held-out
    validation set that was *not* used for training.  The underlying model's
    predict_proba is then replaced by a logistic-regression mapping that
    improves probability reliability.

    Args:
        model: A fitted classifier exposing predict_proba (e.g. LGBMClassifier).
        X_val: Validation feature matrix.
        y_val: Validation binary target.
        config: Parsed config dict (uses model.calibration.method).

    Returns:
        A CalibratedClassifierCV instance whose predict_proba output is
        calibrated.

    Example:
        Before calibration, model.predict_proba(X)[:, 1] might output 0.72
        for a group where only 55 % are actually positive.  After Platt
        scaling, the same group would receive a probability closer to 0.55.
    """
    method = config["model"]["calibration"]["method"]  # "sigmoid"
    logger.info("Calibrating model with method='%s' on %d samples.", method, len(y_val))

    # sklearn >= 1.6 removed cv="prefit"; use FrozenEstimator wrapper
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(model),
            method=method,
            cv=3,
        )
    except ImportError:
        # Older sklearn: cv="prefit" still works
        calibrated = CalibratedClassifierCV(
            estimator=model,
            method=method,
            cv="prefit",
        )
    calibrated.fit(X_val, y_val)

    logger.info("Calibration complete.")
    return calibrated


def plot_calibration(
    y_true: np.ndarray,
    prob_before: np.ndarray,
    prob_after: np.ndarray,
    save_path: Path,
    n_bins: int = 10,
) -> None:
    """Generate a calibration curve comparing raw vs calibrated probabilities.

    Plots two reliability diagrams (before and after Platt scaling) plus the
    perfectly-calibrated diagonal.  Saved as PNG.

    Args:
        y_true: Binary ground truth labels.
        prob_before: Raw predicted probabilities (before calibration).
        prob_after: Calibrated predicted probabilities (after Platt scaling).
        save_path: Destination PNG path (e.g. outputs/figures/calibration_curve.png).
        n_bins: Number of bins for the reliability diagram (default 10).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute calibration curves
    frac_pos_before, mean_pred_before = calibration_curve(
        y_true, prob_before, n_bins=n_bins, strategy="uniform"
    )
    frac_pos_after, mean_pred_after = calibration_curve(
        y_true, prob_after, n_bins=n_bins, strategy="uniform"
    )

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
    ax.plot(
        mean_pred_before,
        frac_pos_before,
        marker="o",
        linewidth=1.5,
        label="Before calibration (raw LightGBM)",
    )
    ax.plot(
        mean_pred_after,
        frac_pos_after,
        marker="s",
        linewidth=1.5,
        label="After calibration (Platt scaling)",
    )

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve: Raw vs Platt-Scaled Probabilities")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Calibration plot saved to %s", save_path)


def run_calibration_pipeline(
    model: BaseEstimator,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
) -> CalibratedClassifierCV:
    """Convenience wrapper: calibrate the model and save the calibration plot.

    Args:
        model: Fitted classifier.
        X_val: Validation features.
        y_val: Validation binary target.
        config: Parsed config dict.

    Returns:
        The calibrated model.
    """
    project_root = Path(config.get("_project_root", "."))
    figures_dir = project_root / config["paths"]["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Raw probabilities
    prob_before = model.predict_proba(X_val)[:, 1]

    # Calibrate
    calibrated_model = calibrate_model(model, X_val, y_val, config)

    # Calibrated probabilities
    prob_after = calibrated_model.predict_proba(X_val)[:, 1]

    # Plot
    plot_calibration(
        y_true=np.asarray(y_val),
        prob_before=prob_before,
        prob_after=prob_after,
        save_path=figures_dir / "calibration_curve.png",
    )

    return calibrated_model
