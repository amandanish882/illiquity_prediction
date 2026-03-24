"""
Illiquidity prediction model — LightGBM with Optuna hyperparameter search.

Trains a binary classifier to predict whether a bond's next-period illiquidity
exceeds the 75th percentile (illiquid_t1 = 1).  Uses time-series-aware splitting
(last 20 % of calendar dates as the test set, no shuffling) and 3-fold
TimeSeriesSplit cross-validation inside Optuna.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

logger = logging.getLogger(__name__)

# ── Feature columns expected by the model ────────────────────────────────────
FEATURE_COLS: List[str] = [
    "spread_bps",
    "spread_volatility_5d",
    "log_daily_volume",
    "trade_count",
    "avg_trade_size",
    "days_since_last_trade",
    "pct_dealer_trades",
    "time_to_maturity",
    "coupon",
    "is_ig",
    "etf_basket_flag",
    "etf_weight",
    "etf_overlap_count",
    "oas_index_level",
    "vix_level",
    "hy_oas_level",
]

TARGET_COL: str = "illiquid_t1"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _resolve_path(config: Dict[str, Any], key: str) -> Path:
    """Resolve a path relative to the project root stored in config."""
    base = Path(config.get("_project_root", "."))
    return base / config["paths"][key]


def _load_config(path: Path) -> Dict[str, Any]:
    """Read config.yaml and attach project root for path resolution."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["_project_root"] = str(path.parent)
    return cfg


# ── Public API ───────────────────────────────────────────────────────────────


def load_features(config: Dict[str, Any]) -> pd.DataFrame:
    """Load the merged feature matrix produced by the data/features layer.

    Args:
        config: Parsed config dict (must include paths.features_dir).

    Returns:
        DataFrame with feature columns, target column, and a 'date' column.

    Raises:
        FileNotFoundError: If the features CSV has not been built yet.
    """
    features_dir = _resolve_path(config, "features_dir")
    csv_path = features_dir / "feature_matrix.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Feature matrix not found at {csv_path}.  "
            "Run the data/features pipeline first to build it."
        )

    df = pd.read_csv(csv_path, parse_dates=["date"])
    logger.info("Loaded feature matrix: %d rows, %d columns", len(df), len(df.columns))

    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Feature matrix is missing required columns: {missing}"
        )

    return df


def prepare_train_test(
    df: pd.DataFrame, config: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
    """Time-series aware train/test split — last 20 % of *dates* form the test set.

    No shuffling.  This guarantees that training data is strictly before test
    data, preventing any future-information leakage.

    Args:
        df: Feature matrix with a 'date' column.
        config: Parsed config dict.

    Returns:
        (X_train, y_train, X_test, y_test, train_idx, test_idx)
        where train_idx / test_idx are integer positional indices into df.
    """
    test_frac = config["model"]["test_fraction"]  # 0.20

    # Sort by date to enforce temporal ordering
    df = df.sort_values("date").reset_index(drop=True)

    unique_dates = np.sort(df["date"].unique())
    n_dates = len(unique_dates)
    cutoff_idx = int(n_dates * (1 - test_frac))
    cutoff_date = unique_dates[cutoff_idx]

    train_mask = df["date"] < cutoff_date
    test_mask = df["date"] >= cutoff_date

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    X_train = df.loc[train_mask, FEATURE_COLS]
    y_train = df.loc[train_mask, TARGET_COL]
    X_test = df.loc[test_mask, FEATURE_COLS]
    y_test = df.loc[test_mask, TARGET_COL]

    logger.info(
        "Train: %d rows (%d dates) | Test: %d rows (%d dates) | cutoff=%s",
        len(X_train),
        len(df.loc[train_mask, "date"].unique()),
        len(X_test),
        len(df.loc[test_mask, "date"].unique()),
        str(cutoff_date)[:10],
    )

    return X_train, y_train, X_test, y_test, train_idx, test_idx


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Dict[str, Any],
) -> LGBMClassifier:
    """Train LightGBM with Optuna hyperparameter search.

    Uses TimeSeriesSplit CV (3 folds) inside the Optuna objective so that
    the validation fold is always later than the training folds.  The search
    runs for a maximum of 20 trials or 180 seconds, whichever comes first.

    After search, a final model is trained on the full training set with
    the best parameters.

    Args:
        X_train: Training features.
        y_train: Binary training target.
        config: Parsed config dict with model.lightgbm.param_space.

    Returns:
        Fitted LGBMClassifier with the best hyperparameters.
    """
    search_cfg = config["model"]["search"]
    param_space = config["model"]["lightgbm"]["param_space"]

    n_trials = search_cfg["n_trials"]       # 20
    cv_folds = search_cfg["cv_folds"]       # 3
    timeout = search_cfg["timeout_seconds"]  # 180

    tscv = TimeSeriesSplit(n_splits=cv_folds)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_categorical(
                "n_estimators", param_space["n_estimators"]
            ),
            "max_depth": trial.suggest_categorical(
                "max_depth", param_space["max_depth"]
            ),
            "learning_rate": trial.suggest_categorical(
                "learning_rate", param_space["learning_rate"]
            ),
            "num_leaves": trial.suggest_categorical(
                "num_leaves", param_space["num_leaves"]
            ),
            "min_child_samples": trial.suggest_categorical(
                "min_child_samples", param_space["min_child_samples"]
            ),
            "subsample": trial.suggest_categorical(
                "subsample", param_space["subsample"]
            ),
            "colsample_bytree": trial.suggest_categorical(
                "colsample_bytree", param_space["colsample_bytree"]
            ),
            "reg_alpha": trial.suggest_categorical(
                "reg_alpha", param_space["reg_alpha"]
            ),
            "reg_lambda": trial.suggest_categorical(
                "reg_lambda", param_space["reg_lambda"]
            ),
        }

        auc_scores: List[float] = []
        for train_ix, val_ix in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_ix], X_train.iloc[val_ix]
            y_tr, y_val = y_train.iloc[train_ix], y_train.iloc[val_ix]

            clf = LGBMClassifier(
                **params,
                objective="binary",
                n_jobs=-1,
                verbosity=-1,
                random_state=42,
            )
            clf.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[],
            )
            preds = clf.predict_proba(X_val)[:, 1]
            auc_scores.append(roc_auc_score(y_val, preds))

        return float(np.mean(auc_scores))

    # Suppress Optuna's internal logging to keep output clean
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", study_name="lgbm_illiquidity")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)

    best_params = study.best_params
    best_auc = study.best_value
    logger.info("Optuna best CV AUC: %.4f  |  params: %s", best_auc, best_params)

    # Sanity check
    if best_auc < 0.55:
        logger.warning(
            "CV AUC is %.4f (< 0.55) — the model may not be learning meaningful signal. "
            "Check feature engineering and target construction.",
            best_auc,
        )
    if best_auc > 0.90:
        logger.warning(
            "CV AUC is %.4f (> 0.90) — possible data leakage. "
            "Verify that no future information enters the features.",
            best_auc,
        )

    # Final model on full training set
    final_model = LGBMClassifier(
        **best_params,
        objective="binary",
        n_jobs=-1,
        verbosity=-1,
        random_state=42,
    )
    final_model.fit(X_train, y_train)
    logger.info("Final model trained on %d rows with best hyperparameters.", len(X_train))

    return final_model


def evaluate_model(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Evaluate the trained model on the hold-out test set.

    Computes:
        - ROC-AUC
        - Average precision (PR-AUC)
        - Precision @ k (k = number of actual positives)
        - Recall @ k

    Args:
        model: Fitted LGBMClassifier.
        X_test: Test features.
        y_test: Binary test target.
        config: Parsed config dict.

    Returns:
        Dictionary of metric names to values.
    """
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    roc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)

    # Precision @ k: pick top-k most-illiquid predictions where k = # actual positives
    k = int(y_test.sum())
    if k > 0:
        top_k_idx = np.argsort(y_prob)[::-1][:k]
        precision_at_k = float(y_test.iloc[top_k_idx].mean())
        recall_at_k = float(y_test.iloc[top_k_idx].sum() / y_test.sum())
    else:
        precision_at_k = 0.0
        recall_at_k = 0.0

    # Precision-recall curve for full thresholds
    prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(rec_arr, prec_arr)

    metrics = {
        "roc_auc": round(roc, 4),
        "average_precision": round(ap, 4),
        "pr_auc": round(pr_auc, 4),
        "precision_at_k": round(precision_at_k, 4),
        "recall_at_k": round(recall_at_k, 4),
        "k": k,
        "test_n": len(y_test),
        "test_positive_rate": round(float(y_test.mean()), 4),
    }

    logger.info("Test metrics: %s", metrics)

    # Economic sanity check
    if roc < 0.55:
        logger.warning(
            "Test AUC %.4f < 0.55 — model is barely better than random.", roc
        )
    elif roc > 0.90:
        logger.warning(
            "Test AUC %.4f > 0.90 — investigate possible data leakage.", roc
        )
    else:
        logger.info("Test AUC %.4f is in the expected 0.60–0.85 range.", roc)

    return metrics


def save_model(model: LGBMClassifier, path: Path) -> None:
    """Persist a trained model to disk as a pickle file.

    Args:
        model: Fitted LGBMClassifier.
        path: Destination file path (e.g. outputs/model.pkl).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", path)


def load_model(path: Path) -> LGBMClassifier:
    """Load a previously saved model from a pickle file.

    Args:
        path: Path to the pickle file.

    Returns:
        The deserialized LGBMClassifier.

    Raises:
        FileNotFoundError: If the pickle file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from %s", path)
    return model


def run_model_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """End-to-end pipeline: load features, split, train, evaluate, save.

    Args:
        config: Parsed config dict (or path will be resolved internally).

    Returns:
        Dictionary containing:
            - model: trained LGBMClassifier
            - predictions: np.ndarray of predicted probabilities on test set
            - feature_names: list of feature column names
            - train_idx, test_idx: integer arrays
            - metrics: evaluation metrics dict
    """
    project_root = Path(config.get("_project_root", "."))
    results_dir = project_root / config["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load features
    df = load_features(config)

    # 2. Train/test split
    X_train, y_train, X_test, y_test, train_idx, test_idx = prepare_train_test(
        df, config
    )

    # 3. Train with Optuna
    model = train_model(X_train, y_train, config)

    # 4. Evaluate
    metrics = evaluate_model(model, X_test, y_test, config)

    # 5. Predictions on test set
    predictions = model.predict_proba(X_test)[:, 1]

    # 6. Save model
    model_path = project_root / config["paths"]["outputs_dir"] / "model.pkl"
    save_model(model, model_path)

    # 7. Save metrics
    metrics_path = results_dir / "model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    return {
        "model": model,
        "predictions": predictions,
        "feature_names": FEATURE_COLS,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "metrics": metrics,
        "X_test": X_test,
        "y_test": y_test,
    }
