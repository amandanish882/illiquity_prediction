"""
data_registry.py -- Central registry documenting every data field, its source,
and whether it is real or synthetic.

Every field used in the illiquidity prediction engine is listed here with
provenance metadata so that downstream consumers can audit data lineage.
"""

DATA_REGISTRY = {
    # ── TRACE trade-level fields (REAL from OSBAP) ────────────────────────
    "cusip": {
        "source": "OSBAP / FINRA TRACE",
        "url_or_method": (
            "Open Source Bond Asset Pricing (openbondassetpricing.com). "
            "Original source: FINRA Enhanced, Standard, and 144A TRACE. "
            "File: trace_clean.parquet processed from stage1_osbap_0k_volume_2025.parquet"
        ),
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "Real 9-character CUSIPs; ~2500 IG + ~130 HY unique bonds in 2024 data",
    },
    "date": {
        "source": "OSBAP / FINRA TRACE",
        "url_or_method": "trd_exctn_dt field from OSBAP TRACE data (trade execution date)",
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "252 NYSE trading days in 2024 (2024-01-02 to 2024-12-31)",
    },
    "price": {
        "source": "OSBAP / FINRA TRACE",
        "url_or_method": "pr field from OSBAP: volume-weighted average clean price from TRACE reports",
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "Real TRACE prices as percent of par. IG typically 90-120, HY wider range.",
    },
    "yield_pct": {
        "source": "OSBAP / FINRA TRACE",
        "url_or_method": "ytm field from OSBAP: QuantLib-computed yield to maturity (decimal, e.g., 0.05 = 5%)",
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "Real YTM computed via QuantLib bond pricing from TRACE trade data",
    },
    "volume": {
        "source": "OSBAP / FINRA TRACE",
        "url_or_method": "dvolume field from OSBAP: daily dollar volume in millions, converted to actual dollars",
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "Real daily dollar trading volume per CUSIP from TRACE dissemination",
    },
    "trade_count": {
        "source": "OSBAP / FINRA TRACE",
        "url_or_method": "trade_count field from OSBAP: number of TRACE-reported trades per CUSIP-day",
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "Real trade counts. Zipf-distributed: top bonds ~50+ trades/day, many bonds 1-3",
    },
    "side": {
        "source": "model-derived",
        "url_or_method": "Set to 'aggregate' — OSBAP provides daily aggregated data, not individual trade sides",
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "Not available at individual trade level in OSBAP daily aggregates",
    },
    "report_type": {
        "source": "model-derived",
        "url_or_method": "Set to 'osbap_daily' — indicates daily aggregate from OSBAP",
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "Indicates data is from OSBAP daily aggregation pipeline",
    },
    "maturity_date": {
        "source": "OSBAP / FINRA TRACE",
        "url_or_method": "Derived from bond_maturity (years to maturity) + trade date, normalized to date",
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "Real bond maturities from TRACE reference data",
    },
    "coupon": {
        "source": "OSBAP / derived",
        "url_or_method": (
            "Back-calculated from real price, YTM, and maturity using semi-annual bond model: "
            "c = (P/100 - disc_N) / annuity * 2 * 100"
        ),
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "Derived from real TRACE data via bond math. IG: ~2-7%, HY: ~4-11%",
    },
    "rating": {
        "source": "model-derived from OSBAP",
        "url_or_method": (
            "Approximated from real credit_spread + db_type. "
            "Thresholds: AAA <50bp, AA 50-80, A 80-130, BBB 130-250, "
            "BB 250-400, B 400-600, CCC >600 bps"
        ),
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "Derived from real credit spreads using standard OAS-rating breakpoints",
    },
    "sector": {
        "source": "model-derived from OSBAP",
        "url_or_method": "Mapped from company_symbol field via known issuer-sector lookup table",
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "Major issuers correctly mapped; smaller issuers classified as 'Other'",
    },
    "is_ig": {
        "source": "model-derived from OSBAP",
        "url_or_method": (
            "db_type 3 (144A) = HY. db_type 1 with median credit_spread > 200 bps = HY. "
            "All others = IG. The 200 bps threshold matches the BBB/BB OAS boundary."
        ),
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "~95% IG, ~5% HY in this dataset (OSBAP focuses on standard corporate bonds)",
    },
    "credit_spread": {
        "source": "OSBAP / FINRA TRACE",
        "url_or_method": "credit_spread field from OSBAP: QuantLib-computed spread over Treasury curve (decimal)",
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "Real credit spreads in decimal (e.g., 0.0078 = 78 bps). IG median ~80 bps, HY ~300+ bps",
    },
    "mod_dur": {
        "source": "OSBAP / FINRA TRACE",
        "url_or_method": "mod_dur field from OSBAP: QuantLib-computed modified duration (years)",
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "Real modified duration. Typical range 1-20 years for corporate bonds",
    },

    # ── ETF holdings fields ──────────────────────────────────────────────
    "etf_basket_flag": {
        "source": "iShares / synthetic fallback",
        "url_or_method": (
            "Primary: iShares CSV download from "
            "https://www.ishares.com/us/products/{product_id}/ishares-*/1467271812596.ajax?"
            "fileType=csv. Fallback: synthetic overlap with TRACE CUSIPs"
        ),
        "is_synthetic": True,
        "synthetic_reason": "iShares CSV download blocked by anti-bot protections in most environments",
        "realistic_properties": (
            "~60% of IG CUSIPs in LQD, ~50% of HY CUSIPs in HYG. "
            "Overlap probability higher for larger-issue, more liquid bonds"
        ),
    },
    "etf_weight": {
        "source": "iShares / synthetic fallback",
        "url_or_method": "Real: parsed from iShares holdings CSV 'Weight (%)' column. Synthetic: Dirichlet-drawn weights",
        "is_synthetic": True,
        "synthetic_reason": "iShares CSV download typically blocked",
        "realistic_properties": (
            "Weights sum to 1.0 per ETF. Individual weights ~0.01-0.5% for LQD (~2500 holdings), "
            "~0.05-1.0% for HYG (~1200 holdings). Dirichlet(alpha=1) produces realistic concentration"
        ),
    },
    "etf_overlap_count": {
        "source": "synthetic",
        "url_or_method": "Count of ETFs from overlap list holding each CUSIP",
        "is_synthetic": True,
        "synthetic_reason": "Would require holdings data from 6 ETFs (LQD, HYG, VCIT, VCSH, JNK, USIG)",
        "realistic_properties": "Range 0-6; most IG bonds in 2-3 ETFs, most HY bonds in 1-2 ETFs",
    },

    # ── FRED macro fields ────────────────────────────────────────────────
    "oas_index_level": {
        "source": "FRED",
        "url_or_method": "FRED series BAMLC0A0CM via fredapi or CSV fallback",
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": (
            "ICE BofA US Corporate Index OAS (bps). "
            "Range ~80-160 bps in 2023-2024. Daily frequency, forward-filled for holidays"
        ),
    },
    "vix_level": {
        "source": "FRED",
        "url_or_method": "FRED series VIXCLS via fredapi or CSV fallback",
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "CBOE VIX index. Range ~12-35 in 2023-2024. Daily frequency",
    },
    "hy_oas_level": {
        "source": "FRED",
        "url_or_method": "FRED series BAMLH0A0HYM2 via fredapi or CSV fallback",
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "ICE BofA US HY Index OAS (bps). Range ~350-500 bps in 2023-2024",
    },

    # ── Engineered TRACE features (derived from REAL data) ────────────────
    "spread_bps": {
        "source": "derived from real OSBAP TRACE",
        "url_or_method": "Volume-weighted average of (price - mid_proxy) * 100, per CUSIP-day",
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "Computed from real TRACE prices. IG: 5-30 bps; HY: 20-200 bps",
    },
    "spread_volatility_5d": {
        "source": "derived from real OSBAP TRACE",
        "url_or_method": "Rolling 5-day standard deviation of spread_bps per CUSIP",
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "Captures short-term spread instability from real price movements",
    },
    "log_daily_volume": {
        "source": "derived from real OSBAP TRACE",
        "url_or_method": "log(1 + daily_notional_volume) per CUSIP-day",
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "Computed from real TRACE dollar volumes",
    },
    "avg_trade_size": {
        "source": "derived from real OSBAP TRACE",
        "url_or_method": "daily_notional_volume / trade_count per CUSIP-day",
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "Real average trade size from TRACE",
    },
    "days_since_last_trade": {
        "source": "derived from real OSBAP TRACE",
        "url_or_method": "Business days since last observed trade for each CUSIP",
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "Computed from real TRACE trade dates. Capped at 30 days per config",
    },
    "pct_dealer_trades": {
        "source": "model-derived proxy",
        "url_or_method": (
            "Proxy based on trade count: pct_dealer = clip(1 - 1/(1 + trade_count*0.1), 0, 0.8). "
            "OSBAP daily aggregates do not include individual trade-level buy/sell/dealer side."
        ),
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "Proxy; more actively traded bonds assumed to have higher interdealer fraction",
    },
    "time_to_maturity": {
        "source": "derived from real OSBAP TRACE",
        "url_or_method": "bond_maturity field from OSBAP or (maturity_date - current_date).days / 365.25",
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "Real years to maturity from TRACE reference data",
    },

    # ── Target variables (derived from REAL data) ─────────────────────────
    "illiquidity_t1": {
        "source": "derived from real OSBAP TRACE",
        "url_or_method": "Next-day Amihud ratio: |daily_return| / daily_volume, shifted forward 1 day",
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": (
            "Amihud (2002) illiquidity measure computed from real TRACE prices and volumes. "
            "Continuous, right-skewed. Higher values = more illiquid."
        ),
    },
    "illiquid_t1": {
        "source": "derived from real OSBAP TRACE",
        "url_or_method": "Binary: 1 if illiquidity_t1 > 75th percentile cross-sectionally",
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "Binary classification target; ~25% positive rate by construction",
    },
    "illiquidity_t5": {
        "source": "derived from real OSBAP TRACE",
        "url_or_method": "5-day-ahead average Amihud ratio: mean of next 5 days' |return|/volume",
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "Smoother than t1; captures medium-term illiquidity regime from real data",
    },
    "illiquid_t5": {
        "source": "derived from real OSBAP TRACE",
        "url_or_method": "Binary: 1 if illiquidity_t5 > 75th percentile cross-sectionally",
        "is_synthetic": False,
        "synthetic_reason": None,
        "realistic_properties": "Binary classification target; ~25% positive rate by construction",
    },
}
