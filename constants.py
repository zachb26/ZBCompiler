# -*- coding: utf-8 -*-
"""
constants.py — Application-wide constants and static configuration.

All top-level constants used throughout the application live here.
No imports from other application modules. No streamlit imports.
The get_sector_benchmarks() helper accepts an optional settings dict
so callers can apply the valuation_benchmark_scale without depending
on settings.py.
"""

import os
import threading
from pathlib import Path

# ---------------------------------------------------------------------------
# Application identity
# ---------------------------------------------------------------------------
DB_FILENAME = "stocks_data.db"
APP_DIR = Path(__file__).resolve().parent
CONFIGURED_DB_PATH = Path(os.environ.get("STOCKS_DB_PATH", DB_FILENAME)).expanduser()
DB_PATH = CONFIGURED_DB_PATH if CONFIGURED_DB_PATH.is_absolute() else (APP_DIR / CONFIGURED_DB_PATH).resolve()
DATABASE_URL = os.environ.get("STOCKS_DATABASE_URL", os.environ.get("DATABASE_URL", "")).strip()
RUN_STARTUP_REFRESH = os.environ.get("STOCK_ENGINE_RUN_STARTUP_REFRESH", "").strip() == "1"
APP_VERSION = "1.0.1"
README_USAGE_TEXT = """
"""

# ---------------------------------------------------------------------------
# Trading / timing
# ---------------------------------------------------------------------------
TRADING_DAYS = 252
DEFAULT_BENCHMARK_TICKER = "SPY"
DEFAULT_PORTFOLIO_TICKERS = "AAPL, MSFT, NVDA, JNJ, XOM"
PORTFOLIO_OPTIONS = ["Large-Cap", "Global", "DADCO"]
SENIOR_ANALYST_PASSWORD_SECRET = "SENIOR_ANALYST_PASSWORD"
PORTFOLIO_MANAGER_PASSWORD_SECRET = "PORTFOLIO_MANAGER_PASSWORD"

# ---------------------------------------------------------------------------
# Scoring normalisation constants
# ---------------------------------------------------------------------------
TECH_SCORE_MAX = 14.0
FUND_SCORE_MAX = 7.0
VAL_SCORE_MAX = 8.0
SENT_SCORE_MAX = 4.0

# ---------------------------------------------------------------------------
# Fetch / cache
# ---------------------------------------------------------------------------
FETCH_CACHE_TTL_SECONDS = 300
FETCH_STALE_FALLBACK_TTL_SECONDS = 1800
FETCH_CACHE = {
    "ticker_history": {},
    "ticker_info": {},
    "ticker_news": {},
    "batch_history": {},
    "peer_group": {},
    "sec_ticker_map": {},
    "sec_companyfacts": {},
    "sec_submissions": {},
    "sec_filing_text": {},
    "treasury_yield": {},
    "earnings_trend": {},
    "options_data":   {},
}
FETCH_CACHE_LOCK = threading.RLock()
FETCH_CACHE_MAX_ENTRIES = {
    "ticker_history":    200,
    "ticker_info":       300,
    "ticker_news":       200,
    "batch_history":      20,
    "peer_group":        150,
    "sec_ticker_map":      5,
    "sec_companyfacts":   80,
    "sec_submissions":    80,
    "sec_filing_text":    60,
    "treasury_yield":      5,
    "earnings_trend":    300,
    "options_data":      200,
}

# ---------------------------------------------------------------------------
# Startup refresh
# ---------------------------------------------------------------------------
AUTO_REFRESH_STATUS_UPDATE_INTERVAL = 25
AUTO_REFRESH_STALE_AFTER_HOURS = 12
AUTO_REFRESH_FAILURE_STREAK_LIMIT = 6
AUTO_REFRESH_REQUEST_DELAY_SECONDS = 0.2
STARTUP_REFRESH_LOCK = threading.RLock()
STARTUP_REFRESH_STATE = {
    "started": False,
    "running": False,
    "complete": False,
    "total": 0,
    "processed": 0,
    "updated": 0,
    "failed": 0,
    "error": None,
    "started_at": None,
    "finished_at": None,
}

# ---------------------------------------------------------------------------
# SEC / DCF
# ---------------------------------------------------------------------------
SEC_REQUEST_DELAY_SECONDS = 0.15
DCF_PROJECTION_YEARS = 5
DCF_TERMINAL_GROWTH_RATE = 0.025
DCF_GROWTH_HAIRCUT = 0.85
DCF_MAX_GROWTH_RATE = 0.30
DCF_MIN_GROWTH_RATE = -0.05
# Mean-reversion fade: rates above FADE_CAP collapse to FADE_TARGET by FADE_YEARS
DCF_GROWTH_FADE_CAP = 0.15
DCF_GROWTH_FADE_TARGET = 0.06
DCF_GROWTH_FADE_YEARS = 3
DCF_DEFAULT_RISK_FREE_RATE = 0.043
DCF_DEFAULT_MARKET_RISK_PREMIUM = 0.055
DCF_DEFAULT_AFTER_TAX_COST_OF_DEBT = 0.035
DEFAULT_DCF_SETTINGS = {
    "projection_years": DCF_PROJECTION_YEARS,
    "terminal_growth_rate": DCF_TERMINAL_GROWTH_RATE,
    "growth_haircut": DCF_GROWTH_HAIRCUT,
    "max_growth_rate": DCF_MAX_GROWTH_RATE,
    "min_growth_rate": DCF_MIN_GROWTH_RATE,
    "market_risk_premium": DCF_DEFAULT_MARKET_RISK_PREMIUM,
    "default_after_tax_cost_of_debt": DCF_DEFAULT_AFTER_TAX_COST_OF_DEBT,
    "risk_free_rate_override": None,
    "manual_growth_rate": None,
}
SEC_ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}
SEC_FILING_SEARCH_FORMS = ["10-K", "10-Q"]
SEC_GUIDANCE_PATTERNS = [
    "we expect",
    "we anticipate",
    "guidance",
    "outlook",
    "forecast",
    "growth of",
    "increase of",
    "revenue of approximately",
    "we project",
]
SEC_EDGAR_ORGANIZATION = os.environ.get("SEC_EDGAR_ORGANIZATION", f"ZB Compiler/{APP_VERSION}").strip()
SEC_EDGAR_CONTACT_EMAIL = os.environ.get("SEC_EDGAR_CONTACT_EMAIL", "").strip()
SEC_USER_AGENT = os.environ.get("SEC_EDGAR_USER_AGENT", "").strip()

# ---------------------------------------------------------------------------
# Peer universe
# ---------------------------------------------------------------------------
PEER_GROUP_SIZE = 5
PEER_SEARCH_CANDIDATE_LIMIT = 140
PEER_MIN_REQUIRED = 3
PEER_UNIVERSE_FILENAME = "sp500_tickers.txt"
PEER_METRIC_MAP = {
    "PE": "trailingPE",
    "Forward_PE": "forwardPE",
    "PEG": "pegRatio",
    "PS": "priceToSalesTrailing12Months",
    "PB": "priceToBook",
    "EV_EBITDA": "enterpriseToEbitda",
    "ROE": "returnOnEquity",
    "Profit_Margins": "profitMargins",
    "Debt_to_Equity": "debtToEquity",
    "Revenue_Growth": "revenueGrowth",
    "Current_Ratio": "currentRatio",
    "Equity_Beta": "beta",
}
BENCHMARK_RELATIVE_STRENGTH_WINDOWS = {
    "Relative_Strength_3M": 63,
    "Relative_Strength_6M": 126,
    "Relative_Strength_1Y": 252,
}

# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------
FUNDAMENTAL_EVENT_KEYWORDS = {
    "earnings",
    "revenue",
    "guidance",
    "outlook",
    "forecast",
    "margin",
    "dividend",
    "buyback",
    "acquisition",
    "merger",
    "filing",
    "10-k",
    "10-q",
    "profit",
    "cash flow",
    "capex",
    "debt",
}
FILING_TAKEAWAY_PATTERNS = [
    "guidance",
    "outlook",
    "forecast",
    "expect",
    "anticipate",
    "demand",
    "margin",
    "cash flow",
    "liquidity",
    "capital",
    "debt",
    "backlog",
    "inventory",
    "customer",
    "pricing",
    "supply",
]
POSITIVE_SENTIMENT_TERMS = {
    "beat", "beats", "growth", "surge", "surges", "strong", "bullish", "buy",
    "outperform", "upgrade", "record", "expands", "expansion", "profit",
    "profits", "optimistic", "momentum", "gain", "gains",
}
NEGATIVE_SENTIMENT_TERMS = {
    "miss", "misses", "lawsuit", "drop", "drops", "fall", "falls", "weak",
    "bearish", "sell", "downgrade", "cut", "cuts", "risk", "risks", "probe",
    "investigation", "decline", "declines", "loss", "losses", "warning",
}

# ---------------------------------------------------------------------------
# Sector / stock classification
# ---------------------------------------------------------------------------
CYCLICAL_SECTORS = {
    "Consumer Cyclical",
    "Industrials",
    "Energy",
    "Basic Materials",
    "Financial Services",
    "Real Estate",
}
DEFENSIVE_SECTORS = {
    "Consumer Defensive",
    "Healthcare",
    "Utilities",
}
INCOME_SECTORS = {
    "Utilities",
    "Real Estate",
    "Consumer Defensive",
    "Energy",
    "Communication Services",
}
QUALITY_SECTORS = {
    "Technology",
    "Healthcare",
    "Consumer Defensive",
    "Communication Services",
}
STOCK_TYPE_STRATEGIES = {
    "Growth Stocks": "Favor long-term accumulation, durable earnings growth, and trend persistence rather than demanding cheap valuation at every entry.",
    "Value Stocks": "Lean hardest on valuation and balance-sheet strength, buying discounts to intrinsic value and waiting for the market to re-rate them.",
    "Dividend / Income Stocks": "Focus on sustainable yield, payout discipline, and steady compounding instead of chasing aggressive price appreciation.",
    "Cyclical Stocks": "Treat timing as critical, adding more confidently only when momentum and regime support the economic upswing.",
    "Defensive Stocks": "Use as resilient portfolio ballast, tolerating slower upside in exchange for steadier earnings during uncertainty.",
    "Blue-Chip Stocks": "Treat as core long-term holdings where quality and durability matter more than squeezing every short-term signal.",
    "Small-Cap Stocks": "Require tighter risk control and stronger confirmation because upside can be large but drawdowns and liquidity risk are higher.",
    "Mid-Cap Stocks": "Blend growth and stability, rewarding names where fundamentals and trend align without demanding mega-cap defensiveness.",
    "Large-Cap Stocks": "Use as steadier core exposure, giving more room to established market leaders and less tolerance for broken fundamentals.",
    "Speculative / Penny Stocks": "If held at all, treat as small tactical positions that demand strong confirmation and strict downside control.",
}

# ---------------------------------------------------------------------------
# Sector valuation benchmarks
# ---------------------------------------------------------------------------
SECTOR_BENCHMARKS = {
    "Technology":             {"PE": 30, "PS": 6.0, "PB": 8.0, "EV_EBITDA": 20},
    "Healthcare":             {"PE": 25, "PS": 4.0, "PB": 4.0, "EV_EBITDA": 15},
    "Financial Services":     {"PE": 14, "PS": 3.0, "PB": 1.5, "EV_EBITDA": 10},
    "Energy":                 {"PE": 10, "PS": 1.5, "PB": 1.8, "EV_EBITDA":  6},
    "Consumer Cyclical":      {"PE": 20, "PS": 2.5, "PB": 4.0, "EV_EBITDA": 14},
    "Industrials":            {"PE": 20, "PS": 2.0, "PB": 3.5, "EV_EBITDA": 12},
    "Utilities":              {"PE": 18, "PS": 2.5, "PB": 2.0, "EV_EBITDA": 10},
    "Consumer Defensive":     {"PE": 22, "PS": 2.0, "PB": 4.0, "EV_EBITDA": 15},
    "Real Estate":            {"PE": 35, "PS": 6.0, "PB": 3.0, "EV_EBITDA": 18},
    "Communication Services": {"PE": 20, "PS": 4.0, "PB": 3.0, "EV_EBITDA": 12},
    "Basic Materials":        {"PE": 15, "PS": 1.5, "PB": 2.0, "EV_EBITDA":  8},
}
DEFAULT_BENCHMARKS = {"PE": 20, "PS": 3.0, "PB": 3.0, "EV_EBITDA": 12}


def get_sector_benchmarks(sector, settings=None):
    """Return sector valuation benchmarks scaled by the optional settings dict.

    Parameters
    ----------
    sector:
        The sector name string (e.g. "Technology").
    settings:
        Optional model-settings dict.  Only the ``valuation_benchmark_scale``
        key is used; it defaults to 1.0 when absent.

    Returns
    -------
    dict
        Benchmark multiples with the scale factor applied.
    """
    scale = (settings or {}).get("valuation_benchmark_scale", 1.0)
    base_benchmarks = SECTOR_BENCHMARKS.get(sector, DEFAULT_BENCHMARKS)
    return {metric: value * scale for metric, value in base_benchmarks.items()}


# ---------------------------------------------------------------------------
# Database schema
# ---------------------------------------------------------------------------
ANALYSIS_COLUMNS = {
    "Ticker": "TEXT PRIMARY KEY",
    "Price": "REAL",
    "Verdict_Overall": "TEXT",
    "Verdict_Technical": "TEXT",
    "Verdict_Fundamental": "TEXT",
    "Verdict_Valuation": "TEXT",
    "Verdict_Sentiment": "TEXT",
    "Market_Regime": "TEXT",
    "Decision_Confidence": "REAL",
    "Decision_Notes": "TEXT",
    "Score_Tech": "INTEGER",
    "Score_Fund": "INTEGER",
    "Score_Val": "INTEGER",
    "Score_Sentiment": "REAL",
    "Sector": "TEXT",
    "Industry": "TEXT",
    "Stock_Type": "TEXT",
    "Cap_Bucket": "TEXT",
    "Style_Tags": "TEXT",
    "Type_Strategy": "TEXT",
    "Type_Confidence": "REAL",
    "Engine_Weight_Profile": "TEXT",
    "Peer_Count": "INTEGER",
    "Peer_Group_Label": "TEXT",
    "Peer_Tickers": "TEXT",
    "Peer_Summary": "TEXT",
    "Peer_Comparison": "TEXT",
    "Market_Cap": "REAL",
    "Dividend_Yield": "REAL",
    "Payout_Ratio": "REAL",
    "Equity_Beta": "REAL",
    "Relative_Strength_3M": "REAL",
    "Relative_Strength_6M": "REAL",
    "Relative_Strength_1Y": "REAL",
    "Trend_Strength": "REAL",
    "Range_Position_52W": "REAL",
    "Distance_52W_High": "REAL",
    "Distance_52W_Low": "REAL",
    "Volatility_1M": "REAL",
    "Volatility_1Y": "REAL",
    "Momentum_1M_Risk_Adjusted": "REAL",
    "Quality_Score": "REAL",
    "Piotroski_F_Score": "INTEGER",
    "Altman_Z_Score": "REAL",
    "Dividend_Safety_Score": "REAL",
    "Valuation_Signal_Count": "INTEGER",
    "Valuation_Confidence": "REAL",
    "Sentiment_Conviction": "REAL",
    "Risk_Flags": "TEXT",
    "PE_Ratio": "REAL",
    "Forward_PE": "REAL",
    "PEG_Ratio": "REAL",
    "PS_Ratio": "REAL",
    "PB_Ratio": "REAL",
    "EV_EBITDA": "REAL",
    "Graham_Number": "REAL",
    "Intrinsic_Value": "REAL",
    "DCF_Intrinsic_Value": "REAL",
    "DCF_Upside": "REAL",
    "DCF_WACC": "REAL",
    "DCF_Risk_Free_Rate": "REAL",
    "DCF_Beta": "REAL",
    "DCF_Cost_of_Equity": "REAL",
    "DCF_Cost_of_Debt": "REAL",
    "DCF_Equity_Weight": "REAL",
    "DCF_Debt_Weight": "REAL",
    "DCF_Growth_Rate": "REAL",
    "DCF_Terminal_Growth": "REAL",
    "DCF_Base_FCF": "REAL",
    "DCF_Enterprise_Value": "REAL",
    "DCF_Equity_Value": "REAL",
    "DCF_Historical_FCF_Growth": "REAL",
    "DCF_Historical_Revenue_Growth": "REAL",
    "DCF_Guidance_Growth": "REAL",
    "DCF_Source": "TEXT",
    "DCF_Confidence": "TEXT",
    "DCF_History": "TEXT",
    "DCF_Projection": "TEXT",
    "DCF_Sensitivity": "TEXT",
    "DCF_Guidance_Excerpts": "TEXT",
    "DCF_Guidance_Summary": "TEXT",
    "DCF_Filing_Form": "TEXT",
    "DCF_Filing_Date": "TEXT",
    "DCF_Last_Updated": "TEXT",
    "Profit_Margins": "REAL",
    "ROE": "REAL",
    "Debt_to_Equity": "REAL",
    "Revenue_Growth": "REAL",
    "Current_Ratio": "REAL",
    "Target_Mean_Price": "REAL",
    "Recommendation_Key": "TEXT",
    "Analyst_Opinions": "REAL",
    "Sentiment_Headline_Count": "INTEGER",
    "Sentiment_Summary": "TEXT",
    "Event_Study_Count": "INTEGER",
    "Event_Study_Avg_Abnormal_1D": "REAL",
    "Event_Study_Avg_Abnormal_5D": "REAL",
    "Event_Study_Summary": "TEXT",
    "Event_Study_Events": "TEXT",
    "RSI": "REAL",
    "MACD_Value": "REAL",
    "MACD_Signal": "TEXT",
    "SMA_Status": "TEXT",
    "Momentum_1M": "REAL",
    "Momentum_1Y": "REAL",
    "Last_Data_Update": "TEXT",
    "Last_Updated": "TEXT",
    "Overall_Score": "REAL",
    "Assumption_Profile": "TEXT",
    "Assumption_Fingerprint": "TEXT",
    "Assumption_Drift": "REAL",
    "Assumption_Snapshot": "TEXT",
    "DCF_Assumptions": "TEXT",
    "DCF_Bull_Fair_Value": "REAL",
    "DCF_Bull_Upside": "REAL",
    "DCF_Bull_Assumptions": "TEXT",
    "DCF_Bear_Fair_Value": "REAL",
    "DCF_Bear_Upside": "REAL",
    "DCF_Bear_Assumptions": "TEXT",
    "DCF_Scenario_Probs": "TEXT",
    "DCF_Blended_Fair_Value": "REAL",
    "EPS_Revision_4W":          "REAL",
    "EPS_Revision_12W":         "REAL",
    "EPS_Revision_Breadth_4W":  "REAL",
    "Short_Interest":           "REAL",
    "Short_Ratio":              "REAL",
    "Short_Float_Pct":          "REAL",
    "Options_IV_Rank":          "REAL",
    "Options_Skew":             "REAL",
    "Options_PC_Ratio":         "REAL",
    "Options_IV_Term":          "REAL",
    "Data_Completeness": "REAL",
    "Missing_Metric_Count": "INTEGER",
    "Data_Quality": "TEXT",
}
ANALYSIS_NUMERIC_COLUMNS = [
    name for name, definition in ANALYSIS_COLUMNS.items() if definition in {"REAL", "INTEGER"}
]
DCF_ANALYSIS_COLUMNS = [name for name in ANALYSIS_COLUMNS if name.startswith("DCF_")]

# ---------------------------------------------------------------------------
# Default model settings and presets
# ---------------------------------------------------------------------------
DEFAULT_MODEL_SETTINGS = {
    "weight_technical": 1.0,
    "weight_fundamental": 1.0,
    "weight_valuation": 1.0,
    "weight_sentiment": 1.0,
    "tech_rsi_oversold": 30.0,
    "tech_rsi_overbought": 70.0,
    "tech_momentum_threshold": 0.05,
    "tech_trend_tolerance": 0.02,
    "tech_extension_limit": 0.08,
    "fund_roe_threshold": 0.15,
    "fund_profit_margin_threshold": 0.20,
    "fund_debt_good_threshold": 100.0,
    "fund_debt_bad_threshold": 200.0,
    "fund_revenue_growth_threshold": 0.10,
    "fund_current_ratio_good": 1.2,
    "fund_current_ratio_bad": 1.0,
    "valuation_benchmark_scale": 1.0,
    "valuation_peg_threshold": 1.5,
    "valuation_graham_overpriced_multiple": 1.5,
    "valuation_under_score_threshold": 5.0,
    "valuation_fair_score_threshold": 2.0,
    "sentiment_analyst_boost": 2.0,
    "sentiment_upside_high": 0.15,
    "sentiment_upside_mid": 0.05,
    "sentiment_downside_high": 0.15,
    "sentiment_downside_mid": 0.05,
    "overall_buy_threshold": 3.0,
    "overall_strong_buy_threshold": 8.0,
    "overall_sell_threshold": -3.0,
    "overall_strong_sell_threshold": -8.0,
    "decision_hold_buffer": 1.0,
    "decision_min_confidence": 55.0,
    "backtest_cooldown_days": 8.0,
    "backtest_transaction_cost_bps": 10.0,
    "trading_days_per_year": 252.0,
}
MODEL_PRESETS = {
    "Balanced": DEFAULT_MODEL_SETTINGS.copy(),
    "Conservative": {
        **DEFAULT_MODEL_SETTINGS,
        "weight_technical": 0.7,
        "weight_fundamental": 1.3,
        "weight_valuation": 1.4,
        "weight_sentiment": 0.6,
        "tech_rsi_oversold": 27.0,
        "tech_rsi_overbought": 74.0,
        "tech_momentum_threshold": 0.07,
        "tech_trend_tolerance": 0.03,
        "tech_extension_limit": 0.05,
        "fund_roe_threshold": 0.20,
        "fund_profit_margin_threshold": 0.24,
        "fund_debt_good_threshold": 75.0,
        "fund_debt_bad_threshold": 160.0,
        "fund_revenue_growth_threshold": 0.13,
        "fund_current_ratio_good": 1.6,
        "fund_current_ratio_bad": 0.9,
        "valuation_benchmark_scale": 0.90,
        "valuation_peg_threshold": 1.2,
        "valuation_graham_overpriced_multiple": 1.35,
        "valuation_under_score_threshold": 6.0,
        "valuation_fair_score_threshold": 3.0,
        "sentiment_analyst_boost": 1.0,
        "sentiment_upside_high": 0.20,
        "sentiment_upside_mid": 0.10,
        "sentiment_downside_high": 0.10,
        "sentiment_downside_mid": 0.05,
        "overall_buy_threshold": 4.0,
        "overall_strong_buy_threshold": 10.0,
        "overall_sell_threshold": -4.0,
        "overall_strong_sell_threshold": -10.0,
        "decision_hold_buffer": 2.0,
        "decision_min_confidence": 65.0,
        "backtest_cooldown_days": 12.0,
    },
    "Aggressive": {
        **DEFAULT_MODEL_SETTINGS,
        "weight_technical": 1.4,
        "weight_fundamental": 0.8,
        "weight_valuation": 0.9,
        "weight_sentiment": 1.3,
        "tech_rsi_oversold": 34.0,
        "tech_rsi_overbought": 66.0,
        "tech_momentum_threshold": 0.03,
        "tech_trend_tolerance": 0.01,
        "tech_extension_limit": 0.12,
        "fund_roe_threshold": 0.10,
        "fund_profit_margin_threshold": 0.15,
        "fund_debt_good_threshold": 130.0,
        "fund_debt_bad_threshold": 260.0,
        "fund_revenue_growth_threshold": 0.06,
        "fund_current_ratio_good": 1.0,
        "fund_current_ratio_bad": 0.8,
        "valuation_benchmark_scale": 1.10,
        "valuation_peg_threshold": 1.9,
        "valuation_graham_overpriced_multiple": 1.7,
        "valuation_under_score_threshold": 4.0,
        "valuation_fair_score_threshold": 1.0,
        "sentiment_analyst_boost": 3.0,
        "sentiment_upside_high": 0.10,
        "sentiment_upside_mid": 0.03,
        "sentiment_downside_high": 0.20,
        "sentiment_downside_mid": 0.07,
        "overall_buy_threshold": 2.0,
        "overall_strong_buy_threshold": 5.0,
        "overall_sell_threshold": -2.0,
        "overall_strong_sell_threshold": -5.0,
        "decision_hold_buffer": 0.0,
        "decision_min_confidence": 45.0,
        "backtest_cooldown_days": 3.0,
    },
}
PRESET_DESCRIPTIONS = {
    "Balanced": "Balanced keeps the four engines close to equal and is the best general starting point.",
    "Conservative": "Conservative leans on fundamentals and valuation, demands stronger confirmation, and slows trading re-entry.",
    "Aggressive": "Aggressive leans on technicals and sentiment, accepts looser valuation, and reacts faster to price action.",
}

# ---------------------------------------------------------------------------
# Regime-conditional engine-weight modifiers
# ---------------------------------------------------------------------------
# Three coarse regimes × four engines = 12 multipliers.
# These stack multiplicatively with the stock-type modifiers already applied
# in get_type_adjusted_engine_weights.
#
#   Bullish Trend  — trending / low-vol: momentum leads; valuation matters less
#   Early Recovery — Transition or Range-bound: value and quality shine
#   Risk-off       — Bearish Trend: defensives dominate; sentiment unreliable
#
# "Unclear" falls through to neutral (all 1.0).
REGIME_ENGINE_WEIGHTS: dict[str, dict[str, float]] = {
    "Bullish Trend": {
        "technical":   1.15,   # momentum reliable in trending markets
        "fundamental": 0.95,   # fundamentals matter less when trend is clear
        "valuation":   0.88,   # don't fight expensive growth in a bull run
        "sentiment":   1.10,   # analyst and news follow-through is meaningful
    },
    "Early Recovery": {        # mapped from "Transition" and "Range-bound"
        "technical":   0.88,   # price signals noisy / whippy during turn
        "fundamental": 1.15,   # quality and earnings durability lead
        "valuation":   1.18,   # value unlocks before trend clears
        "sentiment":   0.92,   # analyst coverage lags the turn
    },
    "Risk-off": {              # mapped from "Bearish Trend"
        "technical":   0.85,   # trend is down; technicals confirm but don't add much
        "fundamental": 1.20,   # defensive quality is the primary screen
        "valuation":   1.08,   # cheap-enough still matters in risk-off
        "sentiment":   0.82,   # sentiment deteriorates fast; down-weight noise
    },
}

# ---------------------------------------------------------------------------
# UI help text dictionaries
# ---------------------------------------------------------------------------
ANALYSIS_HELP_TEXT = {
    "Stock Type": "The model's best-fit stock category, such as Growth, Value, Dividend, Defensive, or Speculative.",
    "Cap Bucket": "A simple size label based on market value: Small-Cap, Mid-Cap, or Large-Cap.",
    "Type Confidence": "How confidently the model thinks this stock fits its assigned type.",
    "Market Cap": "The company's total market value based on share price times shares outstanding.",
    "Technical": "A score based on price trend, momentum, RSI, MACD, and moving averages. Higher is more constructive.",
    "Fundamental": "A score based on profitability, growth, leverage, and liquidity. Higher usually means a stronger business profile.",
    "Valuation": "A score showing whether the stock looks cheap, fair, or expensive relative to a small peer set, with sector benchmarks used only as a fallback and an optional manual DCF snapshot when you create one.",
    "Sentiment": "A context layer that surfaces analyst coverage, targets, and company-related headlines without making a separate good-or-bad call.",
    "Updated": "The last time this saved analysis was refreshed.",
    "Last Data Update": "The freshest market or news timestamp the app could confirm for this saved snapshot.",
    "Overall Score": "The model's combined score after blending technical, fundamental, valuation, and sentiment inputs.",
    "Data Quality": "A quick read on how complete and usable the underlying data was for this analysis.",
    "Assumption Profile": "The preset or custom settings used when this analysis was generated.",
    "Missing Metrics": "How many important data fields were unavailable during the analysis.",
    "Confidence": "How consistently the model's inputs agree with each other after accounting for trend context, data completeness, and guardrails.",
    "Consistency": "How consistently the model's inputs agree with each other after accounting for trend context, data completeness, and guardrails.",
    "Regime": "The current market backdrop the model sees in the stock: bullish trend, bearish trend, transition, or range-bound.",
    "Trend Strength": "A blended measure of long-term price trend quality using moving averages and one-year momentum.",
    "Quality Score": "A business-quality read based on returns, margins, balance-sheet strength, and growth consistency.",
    "Piotroski F-Score": "Nine binary tests on profitability, balance-sheet quality, and operating efficiency. Scores 0–9; 7+ is strong, 3 or below is weak.",
    "Altman Z-Score": "A distress predictor built from five balance-sheet ratios. Above 2.99 is safe; below 1.81 signals financial distress risk.",
    "Dividend Safety": "A rough check on whether the dividend looks sustainable based on payout ratio, profitability, liquidity, and debt.",
    "Valuation Confidence": "How much valuation evidence the model has available. More usable valuation inputs means higher confidence.",
    "Sentiment Conviction": "How much analyst and headline context was available for the company, not whether that context was positive or negative.",
    "Peer Group": "The five closest companies the app could find using industry, sector, size, profitability, growth, and risk characteristics.",
    "Peer Summary": "A quick description of the peer set used for relative valuation comparisons.",
    "Relative Strength": "How the stock's return has compared with the benchmark over the selected lookback window.",
    "Event Study": "A simple reaction table showing how the stock moved after recent company events, plus the move relative to the benchmark.",
    "Event Study 1D": "The average one-trading-day move after recent company events after subtracting the benchmark move over the same window.",
    "Event Study 5D": "The average five-trading-day move after recent company events after subtracting the benchmark move over the same window.",
    "Graham Fair Value": "A conservative Graham-style fair value estimate based on earnings and book value when those inputs are available. It is most useful for profitable, asset-heavy businesses and less reliable for modern growth companies.",
    "Graham Discount": "Shows how far the current price sits below or above the Graham fair value estimate. Positive is cheaper. It is most useful for profitable, asset-heavy businesses and less reliable for modern growth companies.",
    "DCF Fair Value": "An on-demand five-year discounted cash flow estimate built from SEC filing history, a selected growth assumption, and a discounted terminal value.",
    "DCF Upside": "The percentage gap between the current stock price and the manual DCF fair value estimate. Positive means the DCF points to upside.",
    "DCF WACC": "The discount rate used in the DCF. It blends the estimated cost of equity and cost of debt into a weighted average cost of capital.",
    "DCF Growth Rate": "The starting growth rate used for projected free cash flow before it fades toward the terminal growth assumption by year five.",
    "DCF Source": "Shows whether the DCF growth assumption came from your manual override or from the company's own historical cash-flow and revenue trend.",
    "Historical FCF Growth": "The recent compound growth rate in free cash flow based on SEC filing history.",
    "Historical Revenue Growth": "The recent compound growth rate in revenue based on SEC filing history.",
    "Terminal Growth": "The long-run growth rate used in the DCF terminal value once the explicit five-year forecast ends.",
    "Base FCF": "A normalized free cash flow starting point built from recent SEC operating cash flow, capex history, and maintenance-capex smoothing when the latest year looks distorted.",
    "Cost of Equity": "The return shareholders are assumed to require, estimated with a CAPM-style approach.",
    "Cost of Debt": "The company's after-tax borrowing cost used inside the DCF discount rate.",
    "Equity Weight": "The share of the company's capital structure represented by market equity.",
    "Debt Weight": "The share of the company's capital structure represented by long-term debt.",
    "Free Cash Flow": "Cash generated after operating cash flow minus capital spending. This is the stream the DCF is projecting forward.",
    "P/E Ratio": "Price divided by trailing earnings. Lower than peers can suggest cheaper valuation, but not always better quality.",
    "Forward P/E": "Price divided by expected forward earnings. Useful for seeing what the market is paying for next year's profit.",
    "PEG Ratio": "P/E adjusted for growth. Lower is often more attractive because you are paying less for each unit of growth.",
    "P/S Ratio": "Price divided by sales. Helpful when earnings are noisy or thin, especially for growth businesses.",
    "EV/EBITDA": "Enterprise value divided by EBITDA. A common valuation multiple that includes debt, not just equity price.",
    "P/B Ratio": "Price divided by book value. More useful for asset-heavy businesses than for software or other asset-light companies.",
    "ROE": "Return on equity measures how efficiently the company turns shareholder capital into profit.",
    "Profit Margin": "Shows how much of each dollar of revenue becomes profit after costs.",
    "Debt/Equity": "A leverage measure. Lower usually means the business is carrying less financial risk.",
    "Revenue Growth": "How quickly the company's sales are growing. Stronger growth can support a higher valuation.",
    "Current Ratio": "A liquidity check that compares short-term assets to short-term liabilities.",
    "Dividend Yield": "Annual dividend income as a percentage of the stock price.",
    "Payout Ratio": "The share of earnings paid out as dividends. Very high payout ratios can make dividends less durable.",
    "Equity Beta": "How sensitive the stock has been to broad market moves. Above 1 is usually more volatile than the market.",
    "RSI (14)": "A momentum oscillator that helps show whether a stock may be overbought, oversold, or in a healthier middle range.",
    "RSI": "A short-term momentum reading. Very low or very high values can signal stretched conditions.",
    "Trend": "A quick read on whether the stock's moving-average structure looks bullish, bearish, or neutral.",
    "200-Day Trend": "A longer-term trend check. Many investors use the 200-day trend to separate stronger setups from weaker ones.",
    "MACD": "A momentum indicator based on moving averages that helps show whether trend acceleration is improving or fading.",
    "MACD Signal": "Tells you whether momentum currently looks bullish, bearish, or neutral based on MACD crossover behavior.",
    "1M Momentum": "The stock's recent one-month move. Strong positive momentum can confirm trend strength.",
    "1Y Momentum": "The stock's one-year move. It helps show whether the long-term trend has been strong or weak.",
    "Headlines": "The number of recent company-related headlines the app pulled into the context view.",
    "Analyst View": "The current analyst recommendation label the app could retrieve, shown as context rather than a scored signal.",
    "Target Mean": "The average analyst target price, shown as context rather than a direct verdict input.",
    "Highest Conviction": "The top-ranked stock in the current comparison after blending the active analysis layers with the current settings.",
    "Average Composite Score": "The average blended model score across the current comparison list. Higher means the group looks stronger overall.",
    "Average Target Upside": "The average gap between current price and analyst target price across the current list.",
    "Average DCF Upside": "The average gap between current price and the DCF fair value estimate across the current list.",
    "Sectors Covered": "How many unique sectors are represented in the current comparison set.",
    "Composite Score": "A weighted blend of the technical, fundamental, valuation, and sentiment engine scores.",
    "Freshness": "How recently the saved analysis was updated.",
    "Expected Return": "The portfolio's annualized return estimate based on the historical return sample in the selected lookback window.",
    "Volatility": "The annualized variability of returns. Higher volatility usually means a bumpier ride.",
    "Sharpe": "Return earned per unit of total volatility after subtracting the risk-free rate. Higher is generally better.",
    "Sortino": "Return earned per unit of downside volatility. It focuses more on harmful drawdowns than total volatility does.",
    "Treynor": "Return earned per unit of market beta after subtracting the risk-free rate.",
    "Portfolio Beta": "How sensitive the recommended portfolio has been to the benchmark's moves. Around 1 means market-like sensitivity.",
    "Downside Vol": "Volatility calculated only from downside return swings, not all swings.",
    "Min-Vol Return": "The annualized return of the lowest-volatility portfolio found in the simulation set.",
    "Effective Names": "An estimate of how diversified the portfolio really is after accounting for concentration, not just how many tickers it holds.",
    "Robustness": "How stable the model's directional view stays when nearby guarded assumption sets are tested.",
    "Dominant Bias": "The direction that showed up most often across the sensitivity scenarios: bullish, bearish, or neutral.",
    "Scenario Count": "How many assumption scenarios were tested in the sensitivity run.",
    "Verdict Variety": "How many different final verdicts appeared across the tested scenarios.",
    "Bias": "A simplified label for direction: bullish, bearish, or neutral.",
    "Assumption Drift": "How far a scenario's settings sit from the model's default baseline, shown as a rough percentage drift.",
    "Fingerprint": "A short ID that represents the exact assumption set used for that run.",
    "Strategy Return": "The total return produced by the technical-rule backtest over the selected history window. This replay uses only the Tech Score — fundamentals, valuation, and sentiment are not included.",
    "Benchmark Return": "The total return from simply holding the stock over the same window with no trading rules.",
    "Relative vs Benchmark": "How much the technical-rule strategy outperformed or underperformed simple buy-and-hold. This does NOT reflect the performance of the full composite model.",
    "Strategy Sharpe": "The backtest strategy's annualized return per unit of volatility.",
    "Trading Costs": "Estimated transaction costs deducted from the backtest whenever exposure changes.",
    "Average Exposure": "The average fraction of the portfolio the replay kept invested. Lower exposure can protect downside but can also create cash drag versus buy-and-hold. Values above 100% mean the replay tactically overweighted its strongest setups.",
    "Upside Capture": "When the benchmark finished positive, this shows how much of that upside the strategy captured. Around 100% means it roughly kept pace with buy-and-hold, and above 100% means it outpaced that upside.",
    "Win Rate": "The share of closed trades that ended with a positive return.",
    "Max Drawdown": "The deepest peak-to-trough decline the strategy experienced during the backtest.",
    "Position Changes": "How many times the strategy changed exposure, including entries, adds, reductions, and exits.",
    "Closed Trades": "How many completed round-trip trades were finished in the backtest window.",
    "Avg Trade Return": "The average return across the strategy's closed trades.",
    "Records": "How many saved analyses are currently shown in the filtered library view.",
    "Buy / Strong Buy": "How many names in the current library view have a bullish final verdict.",
    "Fresh in 24h": "How many saved rows were refreshed within the last 24 hours.",
    "Tracked Sectors": "How many sectors are represented in the current library view.",
    "Avg Composite Score": "The average blended model score across the names in that group.",
    "Avg Target Upside": "The average analyst target upside across the names in that group.",
    "Avg DCF Upside": "The average DCF upside across the names in that group.",
    "Short Interest": "The number of shares currently sold short. Source: yfinance (~2-week lag). High short interest means the market has a significant bearish position — which can be a risk flag on expensive names or a squeeze setup on beaten-down ones.",
    "Short Ratio": "Days-to-cover: shares short divided by average daily volume. A ratio above 5 means it would take more than a week of average volume for shorts to cover — raising squeeze risk if the stock rallies.",
    "Short % of Float": "Shares short as a percentage of the public float. Above 10% is elevated; above 20% is high and frequently associated with squeeze risk or concentrated bearish conviction.",
}
OPTIONS_HELP_TEXT = {
    "load_preset": "Load a bundled assumption profile. Balanced is neutral, Conservative is stricter, and Aggressive is faster and more risk-tolerant.",
    "weight_technical": "Higher values make price trend, RSI, MACD, and momentum matter more in the final score.",
    "weight_fundamental": "Higher values make profitability, growth, leverage, and liquidity matter more in the final score.",
    "weight_valuation": "Higher values make cheap-versus-expensive signals matter more in the final score.",
    "weight_sentiment": "This control is currently parked because the sentiment view is context-only and no longer adds a directional score.",
    "tech_rsi_oversold": "Lower values wait for deeper weakness before treating RSI as oversold.",
    "tech_rsi_overbought": "Lower values start treating strong rallies as overbought sooner.",
    "tech_momentum_threshold": "Higher values require a bigger recent move before momentum adds or subtracts technical points.",
    "fund_roe_threshold": "Higher values make the model demand stronger return on equity before awarding a quality point.",
    "fund_profit_margin_threshold": "Higher values make the model stricter about operating profitability.",
    "fund_debt_good_threshold": "Lower values make the model treat leverage as risky sooner.",
    "fund_debt_bad_threshold": "Lower values make the model penalize debt more aggressively.",
    "fund_revenue_growth_threshold": "Higher values make the model demand faster top-line growth before rewarding it.",
    "fund_current_ratio_good": "Higher values make the model ask for more liquidity before awarding a balance-sheet point.",
    "fund_current_ratio_bad": "Higher values make the model flag weak liquidity sooner.",
    "tech_trend_tolerance": "Higher values create a wider neutral zone around moving averages so small wiggles do not flip the trend score.",
    "tech_extension_limit": "Lower values make the model call a move stretched sooner when price runs away from the 50-day average.",
    "decision_hold_buffer": "Higher values make mixed or transitional setups need more evidence before becoming Buy or Sell.",
    "decision_min_confidence": "Higher values make the final verdict stay on Hold unless the model's signals remain more internally consistent.",
    "valuation_benchmark_scale": "Adjusts the fallback sector benchmarks that are only used when the peer set is too thin to stand on its own.",
    "valuation_peg_threshold": "Lower values make the model harsher on expensive PEG ratios.",
    "valuation_graham_overpriced_multiple": "Lower values make the model call a stock overpriced sooner versus Graham value.",
    "trading_days_per_year": "Adjusts annualized portfolio and backtest metrics such as return, volatility, and Sharpe.",
    "valuation_fair_score_threshold": "Higher values require more valuation evidence before a name earns Fair Value instead of Overvalued.",
    "valuation_under_score_threshold": "Higher values require more valuation evidence before a name earns Undervalued.",
    "sentiment_analyst_boost": "Kept for future directional sentiment work. The current context-only sentiment view does not use this setting.",
    "sentiment_upside_mid": "Kept for future directional sentiment work. The current context-only sentiment view does not use this setting.",
    "sentiment_upside_high": "Kept for future directional sentiment work. The current context-only sentiment view does not use this setting.",
    "overall_buy_threshold": "Higher values require a larger combined score before the model upgrades from Hold to Buy.",
    "overall_strong_buy_threshold": "Higher values make Strong Buy rarer.",
    "overall_sell_threshold": "Higher values make the model wait for weaker negative evidence before switching from Hold to Sell.",
    "overall_strong_sell_threshold": "Higher values make Strong Sell rarer.",
    "sentiment_downside_mid": "Kept for future directional sentiment work. The current context-only sentiment view does not use this setting.",
    "sentiment_downside_high": "Kept for future directional sentiment work. The current context-only sentiment view does not use this setting.",
    "backtest_cooldown_days": "Higher values force the replay to wait longer before re-entering after a position change.",
    "backtest_transaction_cost_bps": "Estimated trading cost in basis points charged whenever the backtest changes exposure.",
}

# ---------------------------------------------------------------------------
# Changelog
# ---------------------------------------------------------------------------
CHANGELOG_ENTRIES = [
    {
        "Date": "2026-04-05",
        "Area": "Backtest Refinement",
        "Update": "Retuned the replay to keep blue-chip, large-cap, value, and defensive names invested as true core holdings, while growth names still get tactical overweighting and deeper trend-damage exits.",
        "Impact": "Benchmark-relative backtests improved across the cached universe, average relative return moved above +5%, and the replay now shows more clearly when exposure is acting like a deliberate core stake instead of accidental cash drag.",
    },
    {
        "Date": "2026-04-02",
        "Area": "DCF Valuation",
        "Update": "Added a five-year SEC filing-based DCF model with companyfacts history, optional filing-guidance extraction, WACC estimation, terminal value math, and a sensitivity table.",
        "Impact": "The valuation engine can now use a cash-flow-based fair value estimate alongside relative multiples and Graham-style checks.",
    },
    {
        "Date": "2026-04-02",
        "Area": "Refinement Pass",
        "Update": "Added ten new refinements spanning trend strength, 52-week context, volatility-adjusted momentum, quality scoring, dividend safety, valuation breadth, sentiment conviction, risk flags, dynamic engine weights, and profile-aware trailing stops.",
        "Impact": "The model now uses a richer set of diagnostics before deciding how much to trust trend, quality, value, and sentiment for each stock type.",
    },
    {
        "Date": "2026-04-02",
        "Area": "Stock Types",
        "Update": "Added a stock-type classifier that labels names such as Growth, Value, Dividend, Cyclical, Defensive, Blue-Chip, Cap-based, or Speculative and applies style-aware decision logic.",
        "Impact": "The model now treats different kinds of stocks less uniformly and makes its style assumptions visible in research and backtests.",
    },
    {
        "Date": "2026-04-02",
        "Area": "Backtest Strategy",
        "Update": "Shifted the replay toward a core-plus-tactical trend approach so strong uptrends keep some exposure and only fully exit after a deeper breakdown. Added win-rate and average-trade diagnostics.",
        "Impact": "Secular winners are less likely to be traded out too early, and the backtest now shows whether its closed trades are actually improving.",
    },
    {
        "Date": "2026-04-02",
        "Area": "Library",
        "Update": "Added database and CSV download actions so the shared research library can be exported, reviewed, versioned, and seeded into the repo.",
        "Impact": "You can now snapshot the research store and publish a populated seed library more easily.",
    },
    {
        "Date": "2026-04-02",
        "Area": "Startup Refresh",
        "Update": "The app now refreshes every saved ticker automatically on launch and shows a bottom-right compiling badge while the refresh runs.",
        "Impact": "Shared research opens with a live refresh pass instead of relying only on stale cached rows.",
    },
    {
        "Date": "2026-04-02",
        "Area": "Options UX",
        "Update": "Added a dedicated Changelog tab, expanded Methodology explanations, and attached ? help text to every model control.",
        "Impact": "The assumption set is easier to understand before changing live research behavior.",
    },
    {
        "Date": "2026-04-02",
        "Area": "Presets",
        "Update": "Rebalanced Balanced, Conservative, and Aggressive presets so each one changes weights, thresholds, confidence rules, and backtest pacing more noticeably.",
        "Impact": "Loading a preset now creates a clearer shift in model personality instead of a barely visible tweak.",
    },
    {
        "Date": "2026-04-02",
        "Area": "Decision Engine",
        "Update": "Added market regime classification, cross-engine agreement checks, confidence scoring, and decision notes before final verdict guardrails are applied.",
        "Impact": "Mixed evidence and weak data are pushed toward hold more consistently.",
    },
    {
        "Date": "2026-04-02",
        "Area": "Backtest",
        "Update": "Changed the technical replay to allow partial positions, slower re-entries, and trade log actions such as Add and Reduce.",
        "Impact": "Backtests now behave more like staged decision-making instead of instant all-in or all-out flips.",
    },
    {
        "Date": "2026-04-02",
        "Area": "Transparency",
        "Update": "Surfaced regime, confidence, data quality, fingerprint, and assumption drift across research, sensitivity, and library views.",
        "Impact": "It is easier to see why a verdict was produced and whether saved rows are directly comparable.",
    },
]

# ---------------------------------------------------------------------------
# AI / Skill report types
# ---------------------------------------------------------------------------
SKILL_REPORT_TYPES = {
    "equity_research": "Equity Research Report",
    "comps": "Comparable Companies Analysis",
    "thesis": "Investment Thesis",
    "ic_memo": "IC Memo",
}
