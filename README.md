# Bond Illiquidity Prediction Engine

A machine-learning pipeline that predicts next-day corporate bond illiquidity using **real FINRA TRACE data** (OSBAP), iShares ETF holdings, and FRED macro indicators. Calibrated illiquidity scores drive dynamic dealer spread policies, a three-regime markout backtest, ETF-segment analysis, and a tracking-error-minimising redemption basket optimizer.

---

## Key Results

| Metric | Value |
|--------|-------|
| Test ROC-AUC | **0.807** |
| Test Average Precision | **0.545** |
| Dynamic fill rate improvement | **+20 pp** (50% → 69%) |
| Optimised basket illiquidity reduction | **88%** (0.457 → 0.055) |
| Optimised basket execution cost reduction | **78%** (13.9 → 3.0 bps) |
| Tracking error | **0.24 bps** |

### SHAP Feature Importance
![SHAP Summary](outputs/figures/shap_summary.png)

### Model Calibration
![Calibration Curve](outputs/figures/calibration_curve.png)

### Backtest Regime Comparison
![Backtest](outputs/figures/backtest_regime_comparison.png)

### ETF Segment Analysis
![ETF Segments](outputs/figures/etf_segment_comparison.png)

### Redemption Basket Comparison
![Basket](outputs/figures/basket_comparison.png)

### ETF Liquidity Channel (SHAP Dependence)
![ETF SHAP](outputs/figures/shap_etf_dependence.png)

### Feature Importance (LightGBM splits)
![Feature Importance](outputs/figures/feature_importance.png)

---

## Setup

### 1. Python environment

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
```

### 2. Required API key

| Variable | Purpose | Where to get it |
|----------|---------|-----------------|
| `FRED_API_KEY` | FRED macro data (IG OAS, VIX, HY OAS) | https://fred.stlouisfed.org/docs/api/api_key.html |

Set in a `.env` file in the project root:
```
FRED_API_KEY=your_key_here
```

### 3. Data sources

| Source | Type | Details |
|--------|------|---------|
| **OSBAP TRACE** | Real FINRA data | 520,237 bond-day observations, 2,621 CUSIPs (2024) |
| **iShares** | Real ETF holdings | LQD (1,028 bonds) + HYG (39 bonds) from iShares CSV |
| **FRED API** | Real macro data | IG OAS, VIX, HY OAS (522 daily observations) |

No synthetic data. No fallbacks. All data is real.

---

## How to run

```bash
cd illiquidity_engine
python run_all.py
```

This executes 12 pipeline steps end-to-end and produces:
- Trained LightGBM model (`outputs/model.pkl`)
- SHAP and feature importance plots (`outputs/figures/`)
- Backtest regime comparison and JSON results (`outputs/results/`)
- ETF segment analysis and basket optimisation results
- Timing summary (`outputs/results/timing.json`)

Expected runtime: 5-8 minutes.

---

## Data Provenance

All fields documented in `data/data_registry.py`. Every data field traces to a real source:

| Field | Source | Reference |
|-------|--------|-----------|
| `cusip`, `price`, `volume`, `trade_count` | OSBAP / real FINRA TRACE | openbondassetpricing.com |
| `credit_spread`, `mod_dur`, `mac_dur` | OSBAP / QuantLib from real TRACE | OSBAP methodology |
| `coupon` | Back-calculated from real price/YTM/maturity | Semi-annual bond pricing identity |
| `rating` | Derived from real credit spreads | ICE BofA OAS breakpoints |
| `is_ig` | db_type + credit spread threshold (200 bps) | Standard BBB/BB boundary |
| `etf_basket_flag`, `etf_weight` | Real iShares LQD/HYG holdings | iShares website CSV |
| `oas_index_level` | FRED BAMLC0A0CM | ICE BofA US Corporate Index OAS |
| `vix_level` | FRED VIXCLS | CBOE Volatility Index |
| `hy_oas_level` | FRED BAMLH0A0HYM2 | ICE BofA US High Yield Index OAS |
| `illiquid_t1` | Amihud ratio > 75th percentile | Amihud (2002) |

---

## Architecture

```
illiquidity_engine/
|
|-- config.yaml                 # All hyperparameters and paths
|-- run_all.py                  # Master orchestrator (12 steps)
|-- requirements.txt            # Python dependencies
|-- README.md
|
|-- data/
|   |-- fetch_trace.py          # Real OSBAP TRACE data loader
|   |-- fetch_etf_holdings.py   # Real iShares ETF holdings fetcher
|   |-- fetch_fred.py           # Real FRED API macro data
|   |-- data_registry.py        # Provenance registry for every field
|
|-- features/
|   |-- trace_features.py       # 10 TRACE features + Amihud targets
|   |-- etf_features.py         # ETF basket flag, weight, overlap count
|   |-- macro_features.py       # OAS, VIX, HY OAS merge
|
|-- models/
|   |-- illiquidity_model.py    # LightGBM + Optuna hyperparameter search
|   |-- calibration.py          # Platt scaling (CalibratedClassifierCV)
|   |-- feature_importance.py   # SHAP TreeExplainer + importance plots
|
|-- strategy/
|   |-- spread_policy.py        # Linear + piecewise spread & size policies
|   |-- markout_backtest.py     # 3-regime dealer backtest (flat/dynamic/oracle)
|   |-- etf_segment_analysis.py # ETF-eligible vs off-the-run comparison
|   |-- redemption_basket.py    # SLSQP-optimised redemption basket
|
|-- tests/                      # Unit tests for each module
|-- outputs/
    |-- figures/                # SHAP, calibration, backtest, basket charts
    |-- results/                # JSON metrics and timing summary
```

### Pipeline Flow

```
TRACE data ──> Aggregate ──> TRACE features ──> ETF features ──> Macro features
                                                                       |
                                                                Feature Matrix
                                                                       |
                                                             Train LightGBM
                                                                       |
                                                            Platt Calibration
                                                                       |
                                             ┌─────────────────────────┼──────────────────┐
                                             v                         v                  v
                                        Backtest               SHAP Analysis        Basket Opt.
                                             |                                            |
                                    ETF Segment Analysis                          Basket Comparison
```

---

## Methodology

### Model
- **Algorithm**: LightGBM gradient-boosted decision trees
- **Target**: Binary illiquidity flag (Amihud ratio > 75th percentile, next day)
- **Features**: 16 features across microstructure (spread, volume, trade count), bond characteristics (maturity, coupon, IG/HY), ETF exposure (basket flag, weight, overlap), and macro conditions (IG OAS, VIX, HY OAS)
- **Hyperparameter search**: Optuna Bayesian optimization (20 trials, 3-fold TimeSeriesSplit CV)
- **Calibration**: Platt scaling via CalibratedClassifierCV

### Backtest
- **Fill model**: Logistic function `P(fill) = 1 / (1 + exp(alpha * (spread - market_spread)))`
- **Markout**: Real future TRACE prices at t+1 and t+5 horizons
- **Regimes**: Flat (constant spread), Dynamic (model-driven piecewise), Oracle (realized future illiquidity)

### Basket Optimization
- **Objective**: Minimize tracking error + illiquidity penalty
- **Method**: scipy SLSQP constrained optimization
- **Constraints**: Duration matching, OAS matching, sector tolerance, min bonds, max per-bond weight

---

## Limitations

- **OSBAP daily aggregates**: No trade-level side information, so markout direction is randomized
- **ETF holdings are point-in-time**: Single snapshot, not time-varying
- **Simplified tracking error**: Factor approximation rather than full covariance matrix
- **Calm market period**: 2024 data covers low-VIX regime; stress generalization unknown
- **Single model**: Only LightGBM; ensemble methods could improve robustness

## References

- Amihud, Y. (2002). "Illiquidity and stock returns: cross-section and time-series effects." *Journal of Financial Markets*.
- Meli, J. & Todorova, M. (2018). "ETFs and the liquidity of corporate bonds." Barclays Research.
- Dick-Nielsen, J., Feldhutter, P., & Lando, D. (2012). "Corporate bond liquidity before and after the onset of the subprime crisis." *Journal of Financial Economics*.
- Bao, J., Pan, J., & Wang, J. (2011). "The illiquidity of corporate bonds." *Journal of Finance*.
