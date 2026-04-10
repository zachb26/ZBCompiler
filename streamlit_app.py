# -*- coding: utf-8 -*-
import datetime
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import copy
import tempfile
import shutil
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import psycopg
import streamlit as st
import yfinance as yf

DB_FILENAME = "stocks_data.db"
APP_DIR = Path(__file__).resolve().parent
CONFIGURED_DB_PATH = Path(os.environ.get("STOCKS_DB_PATH", DB_FILENAME)).expanduser()
DB_PATH = CONFIGURED_DB_PATH if CONFIGURED_DB_PATH.is_absolute() else (APP_DIR / CONFIGURED_DB_PATH).resolve()
DATABASE_URL = os.environ.get("STOCKS_DATABASE_URL", os.environ.get("DATABASE_URL", "")).strip()
RUN_STARTUP_REFRESH = os.environ.get("STOCK_ENGINE_RUN_STARTUP_REFRESH", "").strip() == "1"
# Increase by 0.0.1 after each major update pass that includes 10+ meaningful changes.
APP_VERSION = "1.0.1"
README_USAGE_TEXT = """
"""
TRADING_DAYS = 252
DEFAULT_BENCHMARK_TICKER = "SPY"
DEFAULT_PORTFOLIO_TICKERS = "AAPL, MSFT, NVDA, JNJ, XOM"
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
}
FETCH_CACHE_LOCK = threading.RLock()
AUTO_REFRESH_STATUS_UPDATE_INTERVAL = 25
AUTO_REFRESH_STALE_AFTER_HOURS = 12
AUTO_REFRESH_FAILURE_STREAK_LIMIT = 6
AUTO_REFRESH_REQUEST_DELAY_SECONDS = 0.2
SEC_REQUEST_DELAY_SECONDS = 0.15
DCF_PROJECTION_YEARS = 5
DCF_TERMINAL_GROWTH_RATE = 0.025
DCF_GROWTH_HAIRCUT = 0.85
DCF_MAX_GROWTH_RATE = 0.30
DCF_MIN_GROWTH_RATE = -0.05
DCF_DEFAULT_RISK_FREE_RATE = 0.043
DCF_DEFAULT_MARKET_RISK_PREMIUM = 0.055
DCF_DEFAULT_AFTER_TAX_COST_OF_DEBT = 0.035
PEER_GROUP_SIZE = 5
PEER_SEARCH_CANDIDATE_LIMIT = 140
PEER_MIN_REQUIRED = 3
PEER_UNIVERSE_FILENAME = "sp500_tickers.txt"
BENCHMARK_RELATIVE_STRENGTH_WINDOWS = {
    "Relative_Strength_3M": 63,
    "Relative_Strength_6M": 126,
    "Relative_Strength_1Y": 252,
}
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
    "Basic Materials": {"PE": 15, "PS": 1.5, "PB": 2.0, "EV_EBITDA": 8},
}
DEFAULT_BENCHMARKS = {"PE": 20, "PS": 3.0, "PB": 3.0, "EV_EBITDA": 12}
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
    "Score_Sentiment": "INTEGER",
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
    "Data_Completeness": "REAL",
    "Missing_Metric_Count": "INTEGER",
    "Data_Quality": "TEXT",
}
ANALYSIS_NUMERIC_COLUMNS = [
    name for name, definition in ANALYSIS_COLUMNS.items() if definition in {"REAL", "INTEGER"}
]
DCF_ANALYSIS_COLUMNS = [name for name in ANALYSIS_COLUMNS if name.startswith("DCF_")]
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
    "Strategy Return": "The total return produced by the technical backtest strategy over the selected history window.",
    "Benchmark Return": "The total return from simply holding the stock over the same window with no trading rules.",
    "Relative vs Benchmark": "How much the strategy outperformed or underperformed simple buy-and-hold.",
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
}
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


def safe_num(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value.lower() in ["n/a", "none", "nan", "inf"]:
            return None
        try:
            return float(value.replace("%", "").replace(",", "").strip())
        except ValueError:
            return None
    return None


def get_color(verdict):
    if "STRONG BUY" in verdict or verdict in {"BUY", "STRONG", "UNDERVALUED", "POSITIVE"}:
        return "green"
    if "STRONG SELL" in verdict or verdict in {"SELL", "WEAK", "OVERVALUED", "NEGATIVE"}:
        return "red"
    return "gray"


def escape_markdown_text(value):
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def colorize_markdown_text(value, color):
    safe_text = escape_markdown_text(value)
    if color == "green":
        return f":green[{safe_text}]"
    if color == "red":
        return f":red[{safe_text}]"
    if color in {"gray", "grey"}:
        return f":gray[{safe_text}]"
    return safe_text


def tone_to_color(tone):
    return {
        "good": "green",
        "bad": "red",
        "neutral": "gray",
    }.get(str(tone or "neutral"), "gray")


def format_value(value, fmt="{:,.2f}", suffix=""):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{fmt.format(value)}{suffix}"


def format_percent(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.1f}%"


def format_int(value):
    if value is None or pd.isna(value):
        return "N/A"
    return str(int(value))


def format_market_cap(value):
    if value is None or pd.isna(value):
        return "N/A"
    value = float(value)
    if abs(value) >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"${value:,.0f}"


def parse_last_updated(value):
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def format_age(value):
    stamp = parse_last_updated(value)
    if stamp is None:
        return "Unknown"

    elapsed = datetime.datetime.now() - stamp
    total_minutes = max(int(elapsed.total_seconds() // 60), 0)
    if total_minutes == 0:
        return "Just now"
    if total_minutes < 60:
        return f"{total_minutes}m ago"

    total_hours = total_minutes // 60
    if total_hours < 24:
        return f"{total_hours}h ago"

    total_days = total_hours // 24
    return f"{total_days}d ago"


def parse_any_datetime(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime.datetime):
        if value.tzinfo is not None:
            return value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return value
    if isinstance(value, datetime.date):
        return datetime.datetime.combine(value, datetime.time.min)
    if isinstance(value, (int, float)):
        raw_value = float(value)
        if raw_value <= 0:
            return None
        if raw_value > 1_000_000_000_000:
            raw_value = raw_value / 1000
        try:
            return datetime.datetime.fromtimestamp(raw_value)
        except (OverflowError, OSError, ValueError):
            return None

    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        parsed = pd.to_datetime(text, utc=False)
        if pd.isna(parsed):
            return None
        if isinstance(parsed, pd.Timestamp):
            return parse_any_datetime(parsed.to_pydatetime())
    except Exception:
        return None
    return None


def format_datetime_value(value, fallback="Unknown"):
    stamp = parse_any_datetime(value)
    if stamp is None:
        return fallback
    if stamp.hour == 0 and stamp.minute == 0 and stamp.second == 0:
        return stamp.strftime("%Y-%m-%d")
    return stamp.strftime("%Y-%m-%d %H:%M")


def approximate_trading_days_for_period(period):
    return {
        "3mo": 63,
        "6mo": 126,
        "1y": 252,
        "18mo": 378,
        "2y": 504,
        "3y": 756,
        "5y": 1260,
        "10y": 2520,
    }.get(str(period).lower())


def trim_history_to_period(hist, period):
    if hist is None or hist.empty:
        return pd.DataFrame()
    trading_days = approximate_trading_days_for_period(period)
    if trading_days is None or len(hist) <= trading_days:
        return hist.copy()
    return hist.tail(trading_days).copy()


def extract_news_publish_time(item):
    if not isinstance(item, dict):
        return None
    for key in ["providerPublishTime", "publishTime", "published", "published_at"]:
        parsed = parse_any_datetime(item.get(key))
        if parsed is not None:
            return parsed
    content = item.get("content")
    if isinstance(content, dict):
        return extract_news_publish_time(content)
    return None


def extract_news_title(item):
    if not isinstance(item, dict):
        return ""
    title = str(item.get("title") or "").strip()
    if title:
        return title
    content = item.get("content")
    if isinstance(content, dict):
        return str(content.get("title") or "").strip()
    return ""


def build_news_context_lines(news, max_items=5):
    lines = []
    for item in news or []:
        title = extract_news_title(item)
        if not title:
            continue
        published = format_datetime_value(extract_news_publish_time(item), fallback="")
        publisher = ""
        if isinstance(item, dict):
            publisher = str(item.get("publisher") or item.get("source") or "").strip()
            if not publisher and isinstance(item.get("content"), dict):
                publisher = str(item["content"].get("publisher") or "").strip()
        prefix_parts = [part for part in [published, publisher] if part]
        prefix = " | ".join(prefix_parts)
        lines.append(f"{prefix}: {title}" if prefix else title)
        if len(lines) >= max_items:
            break
    return lines


def compute_relative_strength(close, benchmark_close, window):
    if close is None or benchmark_close is None or window <= 0:
        return None
    close = close.dropna()
    benchmark_close = benchmark_close.dropna()
    aligned = pd.concat([close.rename("stock"), benchmark_close.rename("benchmark")], axis=1, join="inner").dropna()
    if len(aligned) <= window:
        return None
    stock_return = safe_divide(aligned["stock"].iloc[-1] - aligned["stock"].iloc[-window - 1], aligned["stock"].iloc[-window - 1])
    benchmark_return = safe_divide(
        aligned["benchmark"].iloc[-1] - aligned["benchmark"].iloc[-window - 1],
        aligned["benchmark"].iloc[-window - 1],
    )
    if stock_return is None or benchmark_return is None:
        return None
    return float(stock_return - benchmark_return)


def escape_html_text(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_help_legend(items):
    legend_items = [(label, help_text) for label, help_text in items if help_text]
    if not legend_items:
        return

    st.caption("Hover the labels below for quick definitions.")
    legend_columns = st.columns(min(4, len(legend_items)))
    for idx, (label, help_text) in enumerate(legend_items):
        with legend_columns[idx % len(legend_columns)]:
            st.caption(label, help=help_text)


def tone_from_metric_threshold(value, *, good_min=None, good_max=None, bad_min=None, bad_max=None):
    if not has_numeric_value(value):
        return "neutral"
    value = float(value)
    has_good_rule = good_min is not None or good_max is not None
    has_bad_rule = bad_min is not None or bad_max is not None

    is_good = True
    if good_min is not None and value < good_min:
        is_good = False
    if good_max is not None and value > good_max:
        is_good = False
    if has_good_rule and is_good:
        return "good"

    is_bad = True
    if bad_min is not None and value < bad_min:
        is_bad = False
    if bad_max is not None and value > bad_max:
        is_bad = False
    if has_bad_rule and is_bad:
        return "bad"

    return "neutral"


def tone_from_balanced_band(value, healthy_min, healthy_max, caution_low, caution_high):
    if not has_numeric_value(value):
        return "neutral"
    value = float(value)
    if healthy_min <= value <= healthy_max:
        return "good"
    if value <= caution_low or value >= caution_high:
        return "bad"
    return "neutral"


def tone_from_signal_text(value, positives=None, negatives=None):
    normalized = str(value or "").strip().upper()
    positive_values = set(positives or [])
    negative_values = set(negatives or [])
    if normalized in positive_values:
        return "good"
    if normalized in negative_values:
        return "bad"
    return "neutral"


def tone_from_quality_label(value):
    normalized = str(value or "").strip().title()
    if normalized == "High":
        return "good"
    if normalized == "Low":
        return "bad"
    return "neutral"


def tone_from_regime(value):
    normalized = str(value or "").strip()
    if normalized == "Bullish Trend":
        return "good"
    if normalized == "Bearish Trend":
        return "bad"
    return "neutral"


def tone_from_relative_multiple(value, benchmark):
    score = score_relative_multiple(value, benchmark)
    if score > 0:
        return "good"
    if score < 0:
        return "bad"
    return "neutral"


def render_analysis_signal_cards(items, columns=4):
    if not items:
        return

    cols = st.columns(columns)
    for idx, item in enumerate(items):
        tone = item.get("tone", "neutral")
        label = str(item.get("label", ""))
        value = str(item.get("value", ""))
        note = str(item.get("note", "")).strip()
        badge_label = "Constructive" if tone == "good" else "Caution" if tone == "bad" else "Mixed"
        with cols[idx % columns]:
            with st.container(border=True):
                st.caption(label, help=item.get("help"))
                st.markdown(f"### {colorize_markdown_text(value, tone_to_color(tone))}")
                st.badge(badge_label, color=tone_to_color(tone))
                if note:
                    st.caption(note)


def render_analysis_signal_table(rows, reference_label="Reference"):
    if not rows:
        return

    with st.container(border=True):
        header_cols = st.columns([1.4, 0.9, 0.9, 0.8])
        header_cols[0].caption("Metric")
        header_cols[1].caption("Value")
        header_cols[2].caption(str(reference_label))
        header_cols[3].caption("Read")

        for idx, row in enumerate(rows):
            st.divider()
            tone = row.get("tone", "neutral")
            metric = str(row.get("metric", ""))
            value = str(row.get("value", ""))
            reference = str(row.get("reference", ""))
            status = str(row.get("status", tone.title()))
            row_cols = st.columns([1.4, 0.9, 0.9, 0.8])
            row_cols[0].caption(metric, help=row.get("help"))
            row_cols[1].markdown(colorize_markdown_text(value, tone_to_color(tone)))
            row_cols[2].write(reference)
            row_cols[3].badge(status, color=tone_to_color(tone))


def build_library_csv_bytes(df):
    export_df = df.copy()
    export_df = export_df.drop(columns=["Last_Updated_Parsed"], errors="ignore")
    return export_df.to_csv(index=False).encode("utf-8")


def build_database_download_bytes(db_path):
    if not db_path:
        return b""

    source_path = Path(db_path)
    if not source_path.exists():
        return b""

    source_conn = sqlite3.connect(source_path, timeout=30, check_same_thread=False)
    temp_path = None
    try:
        fd, temp_name = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        temp_path = Path(temp_name)
        backup_conn = sqlite3.connect(temp_path, timeout=30, check_same_thread=False)
        try:
            source_conn.backup(backup_conn)
        finally:
            backup_conn.close()
        return temp_path.read_bytes()
    finally:
        source_conn.close()
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def is_postgres_database_url(value):
    text = str(value or "").strip().lower()
    return text.startswith("postgresql://") or text.startswith("postgres://")


def build_postgres_connection_error_message(dsn, exc):
    message = summarize_fetch_error(exc)
    lowered_message = message.lower()
    dsn_text = str(dsn or "").strip()
    dsn_lower = dsn_text.lower()
    hints = []

    if "pooler.supabase.com" in dsn_lower:
        hints.append(
            "Supabase pooler connections should use the exact connection string from Connect in the Supabase dashboard."
        )
        if "password authentication failed for user" in lowered_message:
            if "for user \"postgres\"" in message or "for user 'postgres'" in message:
                hints.append(
                    "For Supabase session pooler on port 5432, the username is usually 'postgres.<project-ref>' rather than plain 'postgres'."
                )
            hints.append(
                "Make sure you are using the database password from Project Settings -> Database, not your Supabase login password or an API key."
            )
            hints.append("If needed, reset the database password in Supabase and update STOCKS_DATABASE_URL.")

    if not hints:
        return message
    return f"{message} {' '.join(hints)}"


def extract_dcf_fields(record):
    if record is None:
        return {}

    extracted = {}
    for field_name in DCF_ANALYSIS_COLUMNS:
        value = record.get(field_name) if hasattr(record, "get") else None
        if isinstance(value, float) and pd.isna(value):
            value = None
        extracted[field_name] = value
    return extracted


def has_dcf_snapshot(record):
    if record is None or not hasattr(record, "get"):
        return False

    if has_numeric_value(record.get("DCF_Intrinsic_Value")):
        return True

    last_updated = str(record.get("DCF_Last_Updated") or "").strip()
    if last_updated:
        return True

    for field_name in ["DCF_History", "DCF_Projection", "DCF_Sensitivity"]:
        payload = str(record.get(field_name) or "").strip()
        if payload and payload not in {"[]", "{}"}:
            return True
    return False


def build_dcf_download_bytes(record):
    if not has_dcf_snapshot(record):
        return b""

    payload = {
        "ticker": str(record.get("Ticker", "")).strip().upper(),
        "price": safe_num(record.get("Price")),
        "assumption_profile": record.get("Assumption_Profile"),
        "analysis_last_updated": record.get("Last_Updated"),
        "dcf_last_updated": record.get("DCF_Last_Updated") or record.get("Last_Updated"),
        "dcf_assumptions": safe_json_loads(record.get("DCF_Assumptions"), default={}),
        "dcf_summary": {
            "intrinsic_value_per_share": safe_num(record.get("DCF_Intrinsic_Value")),
            "upside": safe_num(record.get("DCF_Upside")),
            "wacc": safe_num(record.get("DCF_WACC")),
            "risk_free_rate": safe_num(record.get("DCF_Risk_Free_Rate")),
            "beta": safe_num(record.get("DCF_Beta")),
            "cost_of_equity": safe_num(record.get("DCF_Cost_of_Equity")),
            "cost_of_debt": safe_num(record.get("DCF_Cost_of_Debt")),
            "equity_weight": safe_num(record.get("DCF_Equity_Weight")),
            "debt_weight": safe_num(record.get("DCF_Debt_Weight")),
            "growth_rate": safe_num(record.get("DCF_Growth_Rate")),
            "terminal_growth": safe_num(record.get("DCF_Terminal_Growth")),
            "base_fcf": safe_num(record.get("DCF_Base_FCF")),
            "enterprise_value": safe_num(record.get("DCF_Enterprise_Value")),
            "equity_value": safe_num(record.get("DCF_Equity_Value")),
            "historical_fcf_growth": safe_num(record.get("DCF_Historical_FCF_Growth")),
            "historical_revenue_growth": safe_num(record.get("DCF_Historical_Revenue_Growth")),
            "guidance_growth": safe_num(record.get("DCF_Guidance_Growth")),
            "source": record.get("DCF_Source"),
            "confidence": record.get("DCF_Confidence"),
            "filing_form": record.get("DCF_Filing_Form"),
            "filing_date": record.get("DCF_Filing_Date"),
            "guidance_summary": record.get("DCF_Guidance_Summary"),
        },
        "history": safe_json_loads(record.get("DCF_History"), default=[]),
        "projection": safe_json_loads(record.get("DCF_Projection"), default=[]),
        "sensitivity": safe_json_loads(record.get("DCF_Sensitivity"), default=[]),
        "guidance_excerpts": safe_json_loads(record.get("DCF_Guidance_Excerpts"), default=[]),
    }
    return json.dumps(payload, indent=2).encode("utf-8")


def get_startup_refresh_snapshot():
    with STARTUP_REFRESH_LOCK:
        return STARTUP_REFRESH_STATE.copy()


def render_compiling_badge(placeholder, message):
    if placeholder is None:
        return
    safe_message = (
        str(message)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
    placeholder.markdown(
        f"""
        <style>
            .compile-badge {{
                position: fixed;
                right: 1rem;
                bottom: 1rem;
                z-index: 99999;
                padding: 0.8rem 1rem;
                border-radius: 14px;
                background: rgba(15, 23, 42, 0.94);
                color: #f8fafc;
                border: 1px solid rgba(148, 163, 184, 0.35);
                box-shadow: 0 16px 36px rgba(15, 23, 42, 0.24);
                font-size: 0.92rem;
                line-height: 1.35;
                max-width: 320px;
            }}
        </style>
        <div class="compile-badge">{safe_message}</div>
        """,
        unsafe_allow_html=True,
    )


def format_startup_refresh_message(state):
    message = "Compiling... (May take a while.)"
    if state.get("running"):
        total = int(state.get("total", 0))
        processed = int(state.get("processed", 0))
        if total > 0:
            message += f"\nRefreshing stale analyses {processed}/{total}"
        else:
            message += "\nRefreshing stale analyses"
    return message


def collect_stale_analysis_tickers(db, stale_after_hours=AUTO_REFRESH_STALE_AFTER_HOURS):
    saved_rows = db.get_all_analyses()
    if saved_rows.empty or "Ticker" not in saved_rows.columns:
        return []

    refresh_candidates = saved_rows.copy()
    if "Last_Updated" in refresh_candidates.columns:
        refresh_candidates["Last_Updated_Parsed"] = refresh_candidates["Last_Updated"].map(parse_last_updated)
        stale_cutoff = datetime.datetime.now() - datetime.timedelta(hours=stale_after_hours)
        refresh_candidates = refresh_candidates[
            refresh_candidates["Last_Updated_Parsed"].isna()
            | (refresh_candidates["Last_Updated_Parsed"] < stale_cutoff)
        ]
        refresh_candidates = refresh_candidates.sort_values("Last_Updated_Parsed", ascending=True, na_position="first")

    return (
        refresh_candidates["Ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .drop_duplicates()
        .tolist()
    )


def refresh_saved_analyses_on_launch(db, settings, badge_placeholder=None):
    while True:
        with STARTUP_REFRESH_LOCK:
            if STARTUP_REFRESH_STATE["complete"]:
                return STARTUP_REFRESH_STATE.copy()
            if not STARTUP_REFRESH_STATE["started"]:
                STARTUP_REFRESH_STATE.update(
                    {
                        "started": True,
                        "running": True,
                        "complete": False,
                        "total": 0,
                        "processed": 0,
                        "updated": 0,
                        "failed": 0,
                        "error": None,
                        "started_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "finished_at": None,
                    }
                )
                is_leader = True
            else:
                is_leader = False

        if is_leader:
            try:
                tickers = collect_stale_analysis_tickers(db, stale_after_hours=AUTO_REFRESH_STALE_AFTER_HOURS)
                if not tickers:
                    with STARTUP_REFRESH_LOCK:
                        STARTUP_REFRESH_STATE.update(
                            {
                                "running": False,
                                "complete": True,
                                "total": 0,
                                "processed": 0,
                                "updated": 0,
                                "failed": 0,
                                "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            }
                        )
                    return get_startup_refresh_snapshot()
                analyst = StockAnalyst(db)
                total = len(tickers)

                with STARTUP_REFRESH_LOCK:
                    STARTUP_REFRESH_STATE["total"] = total

                render_compiling_badge(badge_placeholder, format_startup_refresh_message(get_startup_refresh_snapshot()))

                updated_count = 0
                failed_count = 0
                failure_streak = 0
                for idx, ticker in enumerate(tickers, start=1):
                    try:
                        record = analyst.analyze(ticker, settings=settings, persist=True)
                    except Exception as exc:
                        record = None
                        analyst.last_error = summarize_fetch_error(exc)

                    if record is None:
                        failed_count += 1
                        failure_streak += 1
                    else:
                        updated_count += 1
                        failure_streak = 0

                    if idx == 1 or idx % AUTO_REFRESH_STATUS_UPDATE_INTERVAL == 0 or idx == total:
                        with STARTUP_REFRESH_LOCK:
                            STARTUP_REFRESH_STATE["processed"] = idx
                            STARTUP_REFRESH_STATE["updated"] = updated_count
                            STARTUP_REFRESH_STATE["failed"] = failed_count
                        render_compiling_badge(
                            badge_placeholder,
                            format_startup_refresh_message(get_startup_refresh_snapshot()),
                        )
                    if failure_streak >= AUTO_REFRESH_FAILURE_STREAK_LIMIT:
                        with STARTUP_REFRESH_LOCK:
                            STARTUP_REFRESH_STATE["error"] = (
                                "Launch refresh paused after repeated upstream fetch failures. "
                                "Manual analysis still works and saved rows remain available."
                            )
                            STARTUP_REFRESH_STATE["processed"] = idx
                            STARTUP_REFRESH_STATE["updated"] = updated_count
                            STARTUP_REFRESH_STATE["failed"] = failed_count
                        break
                    time.sleep(AUTO_REFRESH_REQUEST_DELAY_SECONDS)

                with STARTUP_REFRESH_LOCK:
                    STARTUP_REFRESH_STATE.update(
                        {
                            "running": False,
                            "complete": True,
                            "processed": updated_count + failed_count,
                            "updated": updated_count,
                            "failed": failed_count,
                            "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )
                return get_startup_refresh_snapshot()
            except Exception as exc:
                with STARTUP_REFRESH_LOCK:
                    STARTUP_REFRESH_STATE.update(
                        {
                            "running": False,
                            "complete": True,
                            "error": summarize_fetch_error(exc),
                            "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )
                return get_startup_refresh_snapshot()

        snapshot = get_startup_refresh_snapshot()
        if snapshot["running"]:
            render_compiling_badge(badge_placeholder, format_startup_refresh_message(snapshot))
            time.sleep(0.25)
            continue
        return snapshot


def summarize_fetch_error(exc):
    if exc is None:
        return "Unknown upstream fetch error."
    message = str(exc).strip() or exc.__class__.__name__
    message = " ".join(message.split())
    return message[:220]


def clone_cached_payload(payload):
    if isinstance(payload, pd.DataFrame):
        return payload.copy(deep=True)
    if isinstance(payload, pd.Series):
        return payload.copy(deep=True)
    return copy.deepcopy(payload)


def get_cached_fetch_payload(bucket, key, max_age_seconds=FETCH_CACHE_TTL_SECONDS):
    with FETCH_CACHE_LOCK:
        cache_entry = FETCH_CACHE[bucket].get(key)
    if not cache_entry:
        return None
    age_seconds = time.time() - cache_entry["timestamp"]
    if age_seconds > max_age_seconds:
        return None
    return clone_cached_payload(cache_entry["payload"])


def set_cached_fetch_payload(bucket, key, payload):
    with FETCH_CACHE_LOCK:
        FETCH_CACHE[bucket][key] = {
            "timestamp": time.time(),
            "payload": clone_cached_payload(payload),
        }


def normalize_history_frame(raw_history):
    if raw_history is None:
        return pd.DataFrame()

    hist = raw_history.copy()
    if isinstance(hist, pd.Series):
        hist = hist.to_frame(name="Close")

    if isinstance(hist.columns, pd.MultiIndex):
        if "Close" in hist.columns.get_level_values(0):
            hist = hist["Close"].copy()
            if isinstance(hist, pd.Series):
                hist = hist.to_frame(name="Close")
        else:
            hist.columns = [
                " ".join(str(part) for part in col if str(part) != "").strip()
                for col in hist.columns.to_flat_index()
            ]

    renamed_columns = {column: str(column).strip().title() for column in hist.columns}
    hist = hist.rename(columns=renamed_columns)

    if "Close" not in hist.columns and "Adj Close" in hist.columns:
        hist["Close"] = hist["Adj Close"]

    if "Close" not in hist.columns:
        return pd.DataFrame()

    hist = hist.sort_index()
    hist = hist[~hist.index.duplicated(keep="last")]
    if getattr(hist.index, "tz", None) is not None:
        hist.index = hist.index.tz_localize(None)
    hist["Close"] = pd.to_numeric(hist["Close"], errors="coerce")
    hist = hist.dropna(subset=["Close"])
    return hist


def normalize_info_payload(info):
    if not isinstance(info, dict):
        return {}
    cleaned = {}
    for key, value in info.items():
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        cleaned[key] = value
    return cleaned


def normalize_fast_info_payload(fast_info):
    if fast_info is None:
        return {}
    if hasattr(fast_info, "items"):
        payload = dict(fast_info.items())
    else:
        return {}
    field_map = {
        "marketCap": ["marketCap", "market_cap"],
        "beta": ["beta"],
        "sharesOutstanding": ["shares", "sharesOutstanding"],
        "lastPrice": ["lastPrice", "last_price"],
    }
    normalized = {}
    for output_key, candidate_keys in field_map.items():
        for candidate_key in candidate_keys:
            value = payload.get(candidate_key)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            normalized[output_key] = value
            break
    return normalized


def normalize_news_payload(news):
    if not isinstance(news, list):
        return []
    normalized = []
    for item in news:
        if isinstance(item, dict):
            normalized.append(item)
    return normalized


def build_info_fallback_from_saved_analysis(saved_row):
    if saved_row is None or isinstance(saved_row, pd.DataFrame) and saved_row.empty:
        return {}

    row = saved_row.iloc[0] if isinstance(saved_row, pd.DataFrame) else saved_row

    fallback_map = {
        "sector": row.get("Sector"),
        "industry": row.get("Industry"),
        "marketCap": row.get("Market_Cap"),
        "dividendYield": row.get("Dividend_Yield"),
        "payoutRatio": row.get("Payout_Ratio"),
        "beta": row.get("Equity_Beta"),
        "trailingPE": row.get("PE_Ratio"),
        "forwardPE": row.get("Forward_PE"),
        "pegRatio": row.get("PEG_Ratio"),
        "priceToSalesTrailing12Months": row.get("PS_Ratio"),
        "priceToBook": row.get("PB_Ratio"),
        "enterpriseToEbitda": row.get("EV_EBITDA"),
        "returnOnEquity": row.get("ROE"),
        "profitMargins": row.get("Profit_Margins"),
        "debtToEquity": row.get("Debt_to_Equity"),
        "revenueGrowth": row.get("Revenue_Growth"),
        "currentRatio": row.get("Current_Ratio"),
        "targetMeanPrice": row.get("Target_Mean_Price"),
        "recommendationKey": row.get("Recommendation_Key"),
        "numberOfAnalystOpinions": row.get("Analyst_Opinions"),
    }
    cleaned = {}
    for key, value in fallback_map.items():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        cleaned[key] = value
    return cleaned


def get_peer_universe_tickers(db=None):
    tickers = []
    seen = set()

    def add_many(values):
        for raw_value in values or []:
            ticker = str(raw_value or "").strip().upper()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            tickers.append(ticker)

    if db is not None:
        try:
            saved_rows = db.get_all_analyses()
        except Exception:
            saved_rows = pd.DataFrame()
        if not saved_rows.empty and "Ticker" in saved_rows.columns:
            add_many(saved_rows["Ticker"].dropna().tolist())

    peer_file = APP_DIR / PEER_UNIVERSE_FILENAME
    if peer_file.exists():
        try:
            add_many(peer_file.read_text(encoding="utf-8").splitlines())
        except OSError:
            pass

    add_many(parse_ticker_list(DEFAULT_PORTFOLIO_TICKERS))
    return tickers


def score_peer_similarity(target_info, candidate_info):
    if not isinstance(candidate_info, dict) or not candidate_info:
        return None, None

    target_sector = str(target_info.get("sector") or "").strip().lower()
    candidate_sector = str(candidate_info.get("sector") or "").strip().lower()
    target_industry = str(target_info.get("industry") or "").strip().lower()
    candidate_industry = str(candidate_info.get("industry") or "").strip().lower()

    if target_industry and candidate_industry and target_industry == candidate_industry:
        priority = 0
    elif target_sector and candidate_sector and target_sector == candidate_sector:
        priority = 1
    else:
        priority = 2

    score = 0.0
    if priority == 0:
        score += 6.0
    elif priority == 1:
        score += 3.5
    else:
        score -= 4.0

    target_market_cap = safe_num(target_info.get("marketCap"))
    candidate_market_cap = safe_num(candidate_info.get("marketCap"))
    if has_numeric_value(target_market_cap) and has_numeric_value(candidate_market_cap) and target_market_cap > 0 and candidate_market_cap > 0:
        score -= abs(np.log(candidate_market_cap / target_market_cap))

    for key, weight in [
        ("beta", 0.8),
        ("revenueGrowth", 1.0),
        ("profitMargins", 0.7),
        ("returnOnEquity", 0.6),
    ]:
        target_value = safe_num(target_info.get(key))
        candidate_value = safe_num(candidate_info.get(key))
        if has_numeric_value(target_value) and has_numeric_value(candidate_value):
            score -= min(abs(candidate_value - target_value) * weight * 4, 2.5)

    return priority, float(score)


def build_peer_candidate_info(candidate_ticker, db=None):
    cached_info = get_cached_fetch_payload("ticker_info", str(candidate_ticker).strip().upper())
    if cached_info:
        return cached_info

    fallback_info = {}
    if db is not None:
        saved_row = db.get_analysis(candidate_ticker)
        if not saved_row.empty:
            fallback_info = build_info_fallback_from_saved_analysis(saved_row.iloc[0])
            if fallback_info:
                return fallback_info

    info, _ = fetch_ticker_info_with_retry(candidate_ticker, attempts=1)
    if info:
        return info
    return fallback_info


def find_closest_peer_group(ticker, info, db=None, peer_count=PEER_GROUP_SIZE):
    cache_key = (str(ticker).strip().upper(), str(info.get("sector") or "").strip(), str(info.get("industry") or "").strip())
    cached_group = get_cached_fetch_payload("peer_group", cache_key, max_age_seconds=FETCH_STALE_FALLBACK_TTL_SECONDS)
    if cached_group:
        return cached_group

    target_ticker = str(ticker).strip().upper()
    target_info = info or {}
    universe = get_peer_universe_tickers(db)
    target_sector = str(target_info.get("sector") or "").strip()
    target_industry = str(target_info.get("industry") or "").strip()

    candidates = []
    scanned = 0
    for candidate_ticker in universe:
        candidate_ticker = str(candidate_ticker).strip().upper()
        if not candidate_ticker or candidate_ticker == target_ticker:
            continue

        candidate_info = build_peer_candidate_info(candidate_ticker, db=db)
        if not candidate_info:
            continue

        priority, similarity_score = score_peer_similarity(target_info, candidate_info)
        if similarity_score is None:
            continue

        candidates.append(
            {
                "Ticker": candidate_ticker,
                "Name": str(candidate_info.get("shortName") or candidate_info.get("longName") or candidate_ticker),
                "Sector": str(candidate_info.get("sector") or ""),
                "Industry": str(candidate_info.get("industry") or ""),
                "Similarity": similarity_score,
                "Priority": priority,
                "marketCap": safe_num(candidate_info.get("marketCap")),
                "trailingPE": safe_num(candidate_info.get("trailingPE")),
                "forwardPE": safe_num(candidate_info.get("forwardPE")),
                "pegRatio": safe_num(candidate_info.get("pegRatio")),
                "priceToSalesTrailing12Months": safe_num(candidate_info.get("priceToSalesTrailing12Months")),
                "priceToBook": safe_num(candidate_info.get("priceToBook")),
                "enterpriseToEbitda": safe_num(candidate_info.get("enterpriseToEbitda")),
                "returnOnEquity": safe_num(candidate_info.get("returnOnEquity")),
                "profitMargins": safe_num(candidate_info.get("profitMargins")),
                "debtToEquity": safe_num(candidate_info.get("debtToEquity")),
                "revenueGrowth": safe_num(candidate_info.get("revenueGrowth")),
                "currentRatio": safe_num(candidate_info.get("currentRatio")),
                "beta": safe_num(candidate_info.get("beta")),
            }
        )
        scanned += 1
        if scanned >= PEER_SEARCH_CANDIDATE_LIMIT and len(candidates) >= peer_count:
            enough_close_matches = len([row for row in candidates if row["Priority"] <= 1]) >= peer_count
            if enough_close_matches:
                break

    candidates = sorted(
        candidates,
        key=lambda row: (row["Priority"], -row["Similarity"], row["Ticker"]),
    )
    selected = candidates[:peer_count]

    metric_map = {
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
    averages = {}
    for output_key, input_key in metric_map.items():
        values = [row[input_key] for row in selected if has_numeric_value(row.get(input_key))]
        averages[output_key] = float(np.mean(values)) if values else None

    group_label = target_industry or target_sector or "Closest peers"
    peer_names = ", ".join(row["Ticker"] for row in selected[:peer_count]) if selected else "None found"
    summary = (
        f"{len(selected)} closest peers from {group_label}: {peer_names}."
        if selected
        else "Not enough comparable peers were available in the current universe."
    )
    payload = {
        "count": len(selected),
        "group_label": group_label,
        "tickers": [row["Ticker"] for row in selected],
        "summary": summary,
        "averages": averages,
        "rows": selected,
    }
    set_cached_fetch_payload("peer_group", cache_key, payload)
    return payload


def build_relative_peer_benchmarks(ticker, info, db=None, settings=None):
    peer_group = find_closest_peer_group(ticker, info, db=db, peer_count=PEER_GROUP_SIZE)
    sector_fallback = get_sector_benchmarks(info.get("sector", "Unknown"), settings=settings)
    averages = peer_group.get("averages", {})
    benchmarks = {
        "PE": averages.get("PE"),
        "PS": averages.get("PS"),
        "PB": averages.get("PB"),
        "EV_EBITDA": averages.get("EV_EBITDA"),
    }
    usable_peer_metrics = sum(1 for value in benchmarks.values() if has_numeric_value(value))
    if peer_group.get("count", 0) < PEER_MIN_REQUIRED or usable_peer_metrics < 2:
        for metric_name, fallback_value in sector_fallback.items():
            if not has_numeric_value(benchmarks.get(metric_name)):
                benchmarks[metric_name] = fallback_value
        source = "Peer group with sector fallback"
    else:
        source = "Closest peer group"
    peer_group["benchmark_source"] = source
    peer_group["benchmarks"] = benchmarks
    return benchmarks, peer_group


def classify_event_category(title):
    lowered = str(title or "").lower()
    if any(token in lowered for token in ["earnings", "eps", "quarter", "guidance"]):
        return "Earnings / Guidance"
    if any(token in lowered for token in ["dividend", "buyback", "repurchase"]):
        return "Capital Return"
    if any(token in lowered for token in ["acquisition", "merger", "deal"]):
        return "M&A"
    if any(token in lowered for token in ["filing", "10-k", "10-q", "sec"]):
        return "Filing"
    if any(token in lowered for token in ["debt", "liquidity", "cash flow", "margin"]):
        return "Balance Sheet / Cash Flow"
    return "Company Event"


def compute_event_study(news, hist, benchmark_ticker=DEFAULT_BENCHMARK_TICKER):
    close = pd.Series(dtype=float)
    if hist is not None and not hist.empty and "Close" in hist.columns:
        close = hist["Close"].dropna().astype(float)
    if close.empty:
        return {
            "count": 0,
            "avg_abnormal_1d": None,
            "avg_abnormal_5d": None,
            "summary": "No usable price history was available for an event study.",
            "events": [],
        }

    benchmark_hist, _ = fetch_ticker_history_with_retry(benchmark_ticker, period="1y", attempts=2)
    benchmark_close = (
        trim_history_to_period(benchmark_hist, "1y")["Close"].dropna().astype(float)
        if benchmark_hist is not None and not benchmark_hist.empty and "Close" in benchmark_hist.columns
        else pd.Series(dtype=float)
    )
    aligned = pd.concat(
        [close.rename("stock"), benchmark_close.rename("benchmark")],
        axis=1,
        join="outer",
    ).sort_index()
    aligned = aligned.reindex(close.index)
    if "benchmark" in aligned.columns:
        aligned["benchmark"] = aligned["benchmark"].ffill(limit=3)

    fundamental_events = []
    seen_titles = set()
    for item in news or []:
        title = extract_news_title(item)
        lowered = title.lower()
        if not title or title.lower() in seen_titles:
            continue
        if not any(keyword in lowered for keyword in FUNDAMENTAL_EVENT_KEYWORDS):
            continue
        seen_titles.add(title.lower())
        published = extract_news_publish_time(item)
        if published is None:
            continue
        normalized_published = datetime.datetime.combine(published.date(), datetime.time.min)
        event_loc = aligned.index.searchsorted(normalized_published)
        if event_loc >= len(aligned):
            continue
        stock_1d = safe_divide(aligned["stock"].iloc[min(event_loc + 1, len(aligned) - 1)] - aligned["stock"].iloc[event_loc], aligned["stock"].iloc[event_loc])
        stock_5d = safe_divide(aligned["stock"].iloc[min(event_loc + 5, len(aligned) - 1)] - aligned["stock"].iloc[event_loc], aligned["stock"].iloc[event_loc])
        bench_1d = None
        bench_5d = None
        if "benchmark" in aligned.columns and has_numeric_value(aligned["benchmark"].iloc[event_loc]):
            bench_1d = safe_divide(
                aligned["benchmark"].iloc[min(event_loc + 1, len(aligned) - 1)] - aligned["benchmark"].iloc[event_loc],
                aligned["benchmark"].iloc[event_loc],
            )
            bench_5d = safe_divide(
                aligned["benchmark"].iloc[min(event_loc + 5, len(aligned) - 1)] - aligned["benchmark"].iloc[event_loc],
                aligned["benchmark"].iloc[event_loc],
            )
        fundamental_events.append(
            {
                "Date": format_datetime_value(published),
                "Category": classify_event_category(title),
                "Headline": title,
                "Return_1D": stock_1d,
                "Return_5D": stock_5d,
                "Abnormal_1D": None if bench_1d is None or stock_1d is None else stock_1d - bench_1d,
                "Abnormal_5D": None if bench_5d is None or stock_5d is None else stock_5d - bench_5d,
            }
        )
        if len(fundamental_events) >= 5:
            break

    avg_abnormal_1d = None
    avg_abnormal_5d = None
    if fundamental_events:
        abnormal_1d_values = [event["Abnormal_1D"] for event in fundamental_events if has_numeric_value(event.get("Abnormal_1D"))]
        abnormal_5d_values = [event["Abnormal_5D"] for event in fundamental_events if has_numeric_value(event.get("Abnormal_5D"))]
        if abnormal_1d_values:
            avg_abnormal_1d = float(np.mean(abnormal_1d_values))
        if abnormal_5d_values:
            avg_abnormal_5d = float(np.mean(abnormal_5d_values))

    if fundamental_events:
        summary = f"{len(fundamental_events)} recent company events captured for reaction study."
        if has_numeric_value(avg_abnormal_5d):
            summary += f" Average 5D move versus {benchmark_ticker}: {avg_abnormal_5d * 100:.1f}%."
    else:
        summary = "No recent company events with usable dates were available for an event study."

    return {
        "count": len(fundamental_events),
        "avg_abnormal_1d": avg_abnormal_1d,
        "avg_abnormal_5d": avg_abnormal_5d,
        "summary": summary,
        "events": fundamental_events,
    }


def extract_filing_takeaways_from_text(filing_text, max_takeaways=5):
    clean_text = strip_html_to_text(filing_text)
    if not clean_text:
        return []

    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", clean_text) if sentence.strip()]
    takeaways = []
    seen = set()
    for sentence in sentences:
        lowered = sentence.lower()
        if not any(pattern in lowered for pattern in FILING_TAKEAWAY_PATTERNS):
            continue
        cleaned_sentence = " ".join(sentence.split())
        if len(cleaned_sentence) < 60:
            continue
        dedupe_key = cleaned_sentence.lower()[:260]
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        takeaways.append(cleaned_sentence)
        if len(takeaways) >= max_takeaways:
            break
    return takeaways


def fetch_batch_history_via_individual_tickers(tickers, period):
    close_frames = []
    failure_messages = []

    for ticker in tickers:
        hist, error = fetch_ticker_history_with_retry(ticker, period, attempts=2)
        if hist is None or hist.empty or "Close" not in hist.columns:
            if error:
                failure_messages.append(f"{ticker}: {error}")
            continue
        close_frames.append(hist["Close"].rename(ticker))
        time.sleep(AUTO_REFRESH_REQUEST_DELAY_SECONDS)

    if not close_frames:
        return pd.DataFrame(), " | ".join(failure_messages[:4]) if failure_messages else None

    combined = pd.concat(close_frames, axis=1, join="outer").sort_index()
    return combined, None


def fetch_ticker_history_with_retry(ticker, period, attempts=3):
    cache_key = (ticker.upper(), period)
    cached_history = get_cached_fetch_payload("ticker_history", cache_key)
    if cached_history is not None and not cached_history.empty:
        return cached_history, None

    last_error = None
    for attempt in range(attempts):
        try:
            hist = normalize_history_frame(yf.Ticker(ticker).history(period=period, auto_adjust=True))
            if not hist.empty:
                set_cached_fetch_payload("ticker_history", cache_key, hist)
                return hist, None
            download_fallback = yf.download(
                ticker.upper(),
                period=period,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            hist = normalize_history_frame(download_fallback)
            if not hist.empty:
                set_cached_fetch_payload("ticker_history", cache_key, hist)
                return hist, None
            last_error = f"Yahoo returned no usable {period} price history for {ticker}."
        except Exception as exc:
            last_error = summarize_fetch_error(exc)
        if attempt < attempts - 1:
            time.sleep(0.5 * (attempt + 1))

    alternate_periods = []
    if str(period).lower() == "1y":
        alternate_periods = ["18mo", "2y"]
    for alternate_period in alternate_periods:
        try:
            hist = normalize_history_frame(yf.download(
                ticker.upper(),
                period=alternate_period,
                auto_adjust=True,
                progress=False,
                threads=False,
            ))
            hist = trim_history_to_period(hist, period)
            if not hist.empty:
                set_cached_fetch_payload("ticker_history", cache_key, hist)
                return hist, None
        except Exception as exc:
            last_error = summarize_fetch_error(exc)

    stale_history = get_cached_fetch_payload(
        "ticker_history",
        cache_key,
        max_age_seconds=FETCH_STALE_FALLBACK_TTL_SECONDS,
    )
    if stale_history is not None and not stale_history.empty:
        return stale_history, None
    return pd.DataFrame(), last_error


def fetch_batch_history_with_retry(tickers, period, attempts=3):
    normalized_tickers = tuple(dict.fromkeys(ticker.upper() for ticker in tickers))
    cache_key = (normalized_tickers, period)
    cached_history = get_cached_fetch_payload("batch_history", cache_key)
    if cached_history is not None and not cached_history.empty:
        return cached_history, None

    last_error = None
    for attempt in range(attempts):
        try:
            raw = yf.download(
                list(normalized_tickers),
                period=period,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if raw is not None and not raw.empty:
                set_cached_fetch_payload("batch_history", cache_key, raw)
                return raw.copy(), None
            last_error = "Yahoo returned no portfolio price history for the selected basket."
        except Exception as exc:
            last_error = summarize_fetch_error(exc)
        if attempt < attempts - 1:
            time.sleep(0.5 * (attempt + 1))

    individual_history, fallback_error = fetch_batch_history_via_individual_tickers(normalized_tickers, period)
    if individual_history is not None and not individual_history.empty:
        set_cached_fetch_payload("batch_history", cache_key, individual_history)
        return individual_history.copy(), None
    if fallback_error:
        last_error = fallback_error

    stale_history = get_cached_fetch_payload(
        "batch_history",
        cache_key,
        max_age_seconds=FETCH_STALE_FALLBACK_TTL_SECONDS,
    )
    if stale_history is not None and not stale_history.empty:
        return stale_history, None
    return pd.DataFrame(), last_error


def fetch_ticker_info_with_retry(ticker, attempts=3):
    cache_key = ticker.upper()
    cached_info = get_cached_fetch_payload("ticker_info", cache_key)
    if cached_info:
        return cached_info, None

    last_error = None
    ticker_handle = yf.Ticker(ticker)
    for attempt in range(attempts):
        try:
            info = normalize_info_payload(ticker_handle.info or {})
            if info:
                fast_info = normalize_fast_info_payload(getattr(ticker_handle, "fast_info", None))
                info = {**fast_info, **info}
                set_cached_fetch_payload("ticker_info", cache_key, info)
                return info, None
            alt_info = normalize_info_payload(getattr(ticker_handle, "get_info", lambda: {})() or {})
            if alt_info:
                fast_info = normalize_fast_info_payload(getattr(ticker_handle, "fast_info", None))
                alt_info = {**fast_info, **alt_info}
                set_cached_fetch_payload("ticker_info", cache_key, alt_info)
                return alt_info, None
            fast_info = normalize_fast_info_payload(getattr(ticker_handle, "fast_info", None))
            if fast_info:
                set_cached_fetch_payload("ticker_info", cache_key, fast_info)
                return fast_info, None
            last_error = f"Yahoo returned no company profile data for {ticker}."
        except Exception as exc:
            last_error = summarize_fetch_error(exc)
        if attempt < attempts - 1:
            time.sleep(0.35 * (attempt + 1))

    stale_info = get_cached_fetch_payload(
        "ticker_info",
        cache_key,
        max_age_seconds=FETCH_STALE_FALLBACK_TTL_SECONDS,
    )
    if stale_info:
        return stale_info, None
    return {}, last_error


def fetch_ticker_news_with_retry(ticker, attempts=2):
    cache_key = ticker.upper()
    cached_news = get_cached_fetch_payload("ticker_news", cache_key)
    if cached_news:
        return cached_news, None

    last_error = None
    for attempt in range(attempts):
        try:
            news = normalize_news_payload(getattr(yf.Ticker(ticker), "news", []) or [])
            if news:
                set_cached_fetch_payload("ticker_news", cache_key, news)
                return news, None
            info_fallback = normalize_info_payload(fetch_ticker_info_with_retry(ticker, attempts=1)[0] or {})
            company_name = str(info_fallback.get("shortName") or info_fallback.get("longName") or ticker).strip()
            if company_name:
                last_error = f"Yahoo returned no recent news items for {ticker} ({company_name})."
            else:
                last_error = f"Yahoo returned no recent news items for {ticker}."
        except Exception as exc:
            last_error = summarize_fetch_error(exc)
        if attempt < attempts - 1:
            time.sleep(0.25 * (attempt + 1))

    stale_news = get_cached_fetch_payload(
        "ticker_news",
        cache_key,
        max_age_seconds=FETCH_STALE_FALLBACK_TTL_SECONDS,
    )
    if stale_news:
        return stale_news, None
    return [], last_error


def has_numeric_value(value):
    return value is not None and not pd.isna(value)


def calculate_rsi(close, period=14):
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100)
    rsi = rsi.where(avg_gain != 0, 0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)
    return rsi


def score_relative_multiple(value, benchmark, cheap_buffer=0.9, expensive_buffer=1.25):
    if not has_numeric_value(value):
        return 0
    if value <= 0:
        return -1
    if not has_numeric_value(benchmark) or benchmark <= 0:
        return 0
    if value <= benchmark * cheap_buffer:
        return 1
    if value >= benchmark * expensive_buffer:
        return -1
    return 0


def extract_sentiment_tokens(text):
    return set(re.findall(r"[a-z]+", (text or "").lower()))


def safe_divide(numerator, denominator):
    if denominator is None or pd.isna(denominator) or abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def safe_json_loads(value, default=None):
    fallback = {} if default is None else default
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return copy.deepcopy(fallback)
    if isinstance(value, (dict, list)):
        return copy.deepcopy(value)
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return copy.deepcopy(fallback)


def get_sec_request_headers():
    user_agent = SEC_USER_AGENT or (
        f"{SEC_EDGAR_ORGANIZATION} {SEC_EDGAR_CONTACT_EMAIL}"
        if SEC_EDGAR_CONTACT_EMAIL
        else f"{SEC_EDGAR_ORGANIZATION} (set SEC_EDGAR_CONTACT_EMAIL for SEC EDGAR access)"
    )
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "en-US,en;q=0.8",
    }
    if SEC_EDGAR_CONTACT_EMAIL:
        headers["From"] = SEC_EDGAR_CONTACT_EMAIL
    return headers


def get_sec_access_hint():
    return (
        "Set SEC_EDGAR_CONTACT_EMAIL=you@example.com before starting the app. "
        "You can also set SEC_EDGAR_ORGANIZATION='Your App Name' or provide a full SEC_EDGAR_USER_AGENT value."
    )


def explain_upstream_fetch_error(url, exc):
    message = summarize_fetch_error(exc)
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if "sec.gov" in str(url).lower() and (status_code == 403 or "403" in message):
        return f"SEC EDGAR returned HTTP 403. {get_sec_access_hint()}"
    return message


def fetch_json_url_with_retry(url, *, headers=None, attempts=3, timeout=15):
    try:
        import requests
    except ImportError:
        return None, "The requests library is not installed, so SEC and Treasury data could not be fetched."

    last_error = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
            time.sleep(SEC_REQUEST_DELAY_SECONDS)
            return payload, None
        except Exception as exc:
            last_error = explain_upstream_fetch_error(url, exc)
            if "SEC EDGAR returned HTTP 403" in str(last_error):
                break
        if attempt < attempts - 1:
            time.sleep(0.35 * (attempt + 1))
    return None, last_error


def fetch_text_url_with_retry(url, *, headers=None, attempts=3, timeout=20):
    try:
        import requests
    except ImportError:
        return None, "The requests library is not installed, so filing text could not be fetched."

    last_error = None
    for attempt in range(attempts):
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            time.sleep(SEC_REQUEST_DELAY_SECONDS)
            return response.text, None
        except Exception as exc:
            last_error = explain_upstream_fetch_error(url, exc)
            if "SEC EDGAR returned HTTP 403" in str(last_error):
                break
        if attempt < attempts - 1:
            time.sleep(0.35 * (attempt + 1))
    return None, last_error


def load_sec_ticker_map():
    cached = get_cached_fetch_payload("sec_ticker_map", "normalized")
    if cached:
        return cached, None

    payload, error = fetch_json_url_with_retry(
        "https://www.sec.gov/files/company_tickers.json",
        headers=get_sec_request_headers(),
        attempts=2,
        timeout=15,
    )
    if not isinstance(payload, dict):
        return {}, error

    normalized = {}
    for item in payload.values():
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker", "")).strip().upper()
        cik_value = item.get("cik_str")
        if not ticker or cik_value in {None, ""}:
            continue
        normalized[ticker] = {
            "cik": str(int(cik_value)).zfill(10),
            "title": str(item.get("title", ticker)).strip(),
        }

    if normalized:
        set_cached_fetch_payload("sec_ticker_map", "normalized", normalized)
    return normalized, error


def lookup_company_cik(ticker):
    ticker_map, error = load_sec_ticker_map()
    payload = ticker_map.get(str(ticker).strip().upper())
    if payload:
        return payload["cik"], payload.get("title"), None
    return None, None, error or f"SEC ticker mapping did not contain {ticker}."


def fetch_sec_submissions(cik_padded):
    cached = get_cached_fetch_payload("sec_submissions", cik_padded)
    if cached:
        return cached, None

    payload, error = fetch_json_url_with_retry(
        f"https://data.sec.gov/submissions/CIK{cik_padded}.json",
        headers=get_sec_request_headers(),
        attempts=2,
        timeout=15,
    )
    if isinstance(payload, dict):
        set_cached_fetch_payload("sec_submissions", cik_padded, payload)
        return payload, None
    return None, error


def fetch_sec_companyfacts(cik_padded):
    cached = get_cached_fetch_payload("sec_companyfacts", cik_padded)
    if cached:
        return cached, None

    payload, error = fetch_json_url_with_retry(
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded}.json",
        headers=get_sec_request_headers(),
        attempts=2,
        timeout=20,
    )
    if isinstance(payload, dict):
        set_cached_fetch_payload("sec_companyfacts", cik_padded, payload)
        return payload, None
    return None, error


def parse_sec_filing_metadata(submissions, preferred_forms=None):
    recent = submissions.get("filings", {}).get("recent", {})
    forms = list(recent.get("form", []) or [])
    accession_numbers = list(recent.get("accessionNumber", []) or [])
    primary_docs = list(recent.get("primaryDocument", []) or [])
    filing_dates = list(recent.get("filingDate", []) or [])

    for target_form in preferred_forms or SEC_FILING_SEARCH_FORMS:
        for idx, form in enumerate(forms):
            if form != target_form:
                continue
            accession_number = accession_numbers[idx] if idx < len(accession_numbers) else None
            primary_doc = primary_docs[idx] if idx < len(primary_docs) else None
            filing_date = filing_dates[idx] if idx < len(filing_dates) else None
            if accession_number and primary_doc:
                return {
                    "form": form,
                    "accession_number": accession_number,
                    "primary_document": primary_doc,
                    "filing_date": filing_date,
                }
    return None


def fetch_sec_filing_text(cik_padded, filing_metadata):
    if not filing_metadata:
        return None, "No recent 10-K or 10-Q filing metadata was available."

    accession_number = filing_metadata.get("accession_number", "")
    accession_no_dashes = accession_number.replace("-", "")
    primary_document = filing_metadata.get("primary_document", "")
    cik_numeric = str(int(cik_padded))
    cache_key = (cik_padded, accession_no_dashes, primary_document)
    cached = get_cached_fetch_payload("sec_filing_text", cache_key)
    if cached:
        return cached, None

    filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_numeric}/{accession_no_dashes}/{primary_document}"
    payload, error = fetch_text_url_with_retry(
        filing_url,
        headers=get_sec_request_headers(),
        attempts=2,
        timeout=20,
    )
    if payload:
        set_cached_fetch_payload("sec_filing_text", cache_key, payload)
        return payload, None
    return None, error


def strip_html_to_text(html_text):
    if not html_text:
        return ""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html_text, "html.parser")
        text = soup.get_text(" ")
    except Exception:
        text = re.sub(r"<[^>]+>", " ", str(html_text))
    return " ".join(str(text).split())


def extract_guidance_excerpts_from_text(filing_text, *, max_excerpts=3, window_sentences=4):
    clean_text = strip_html_to_text(filing_text)
    if not clean_text:
        return []

    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", clean_text) if sentence.strip()]
    excerpts = []
    seen = set()
    for idx, sentence in enumerate(sentences):
        lowered = sentence.lower()
        if not any(pattern in lowered for pattern in SEC_GUIDANCE_PATTERNS):
            continue
        start = max(0, idx - window_sentences)
        end = min(len(sentences), idx + window_sentences + 1)
        excerpt = " ".join(sentences[start:end]).strip()
        excerpt = " ".join(excerpt.split())
        if len(excerpt) < 80:
            continue
        dedupe_key = excerpt.lower()[:320]
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        excerpts.append(excerpt)
        if len(excerpts) >= max_excerpts:
            break
    return excerpts


def extract_json_object_from_text(text):
    if not text:
        return None
    match = re.search(r"\{.*\}", str(text), flags=re.DOTALL)
    return match.group(0) if match else None


def extract_guidance_with_anthropic(excerpts):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or not excerpts:
        return None, None

    try:
        from anthropic import Anthropic
    except Exception:
        return None, "Anthropic guidance extraction was skipped because the anthropic package is unavailable."

    prompt = (
        "You are a financial analyst. From these excerpts of an SEC filing, extract any specific "
        "numerical forward guidance the company provides, including expected revenue growth %, "
        "earnings growth %, margin targets, or any other quantitative forward-looking statements. "
        "Return a JSON object with keys: revenue_growth_pct, earnings_growth_pct, margin_target_pct, "
        "other_guidance (list of strings), confidence (low/medium/high). "
        "If a value is not found, return null for that key.\n\n"
        "Filing excerpts:\n"
        + "\n\n".join(f"{idx + 1}. {excerpt}" for idx, excerpt in enumerate(excerpts))
    )

    try:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = ""
        for block in getattr(response, "content", []):
            if getattr(block, "type", "") == "text":
                response_text += getattr(block, "text", "")
        json_payload = extract_json_object_from_text(response_text)
        parsed = safe_json_loads(json_payload, default={}) if json_payload else {}
        if isinstance(parsed, dict) and parsed:
            return parsed, None
    except Exception as exc:
        return None, summarize_fetch_error(exc)
    return None, "Anthropic guidance extraction returned no usable JSON payload."


def parse_percentage_range(text):
    if not text:
        return None

    range_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:to|-|\u2013)\s*(\d+(?:\.\d+)?)\s*(?:%|percent)",
        text,
        flags=re.IGNORECASE,
    )
    if range_match:
        low = safe_num(range_match.group(1))
        high = safe_num(range_match.group(2))
        if has_numeric_value(low) and has_numeric_value(high):
            return (float(low) + float(high)) / 2 / 100

    single_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent)", text, flags=re.IGNORECASE)
    if single_match:
        single_value = safe_num(single_match.group(1))
        if has_numeric_value(single_value):
            return float(single_value) / 100
    return None


def extract_regex_guidance(excerpts):
    if not excerpts:
        return {}

    result = {
        "revenue_growth_pct": None,
        "earnings_growth_pct": None,
        "margin_target_pct": None,
        "other_guidance": [],
        "confidence": "low",
    }
    for excerpt in excerpts:
        lowered = excerpt.lower()
        pct = parse_percentage_range(excerpt)
        if pct is None:
            continue
        if "revenue" in lowered and result["revenue_growth_pct"] is None:
            result["revenue_growth_pct"] = pct * 100
        elif any(term in lowered for term in ["earnings", "eps", "profit"]) and result["earnings_growth_pct"] is None:
            result["earnings_growth_pct"] = pct * 100
        elif "margin" in lowered and result["margin_target_pct"] is None:
            result["margin_target_pct"] = pct * 100
        else:
            result["other_guidance"].append(excerpt[:220])
    if any(result[key] is not None for key in ["revenue_growth_pct", "earnings_growth_pct", "margin_target_pct"]):
        return result
    return {}


def parse_year_from_date(value):
    if not value:
        return None
    try:
        return int(str(value)[:4])
    except ValueError:
        return None


def sec_entry_priority(entry):
    return (
        1 if str(entry.get("fp", "")).upper() == "FY" else 0,
        1 if str(entry.get("form", "")) in SEC_ANNUAL_FORMS else 0,
        str(entry.get("filed", "")),
        str(entry.get("end", "")),
    )


def extract_company_fact_entries(companyfacts, concepts, *, preferred_units=None, forms=None):
    facts = companyfacts.get("facts", {}).get("us-gaap", {})
    preferred_units = preferred_units or ["USD"]
    allowed_forms = set(forms or SEC_ANNUAL_FORMS)
    best_entries = []
    best_score = None

    for concept in concepts:
        concept_payload = facts.get(concept)
        if not isinstance(concept_payload, dict):
            continue

        unit_entries = concept_payload.get("units", {})
        entries = None
        for unit_name in preferred_units:
            if unit_name in unit_entries:
                entries = unit_entries[unit_name]
                break
        if entries is None and unit_entries:
            entries = next(iter(unit_entries.values()))
        if not entries:
            continue

        normalized = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            form = str(item.get("form", "")).strip()
            if allowed_forms and form not in allowed_forms:
                continue
            value = safe_num(item.get("val"))
            if value is None:
                continue
            year = item.get("fy")
            year = int(year) if str(year).isdigit() else parse_year_from_date(item.get("end"))
            if year is None:
                continue
            normalized.append(
                {
                    "concept": concept,
                    "value": float(value),
                    "year": year,
                    "end": item.get("end"),
                    "filed": item.get("filed"),
                    "form": form,
                    "fp": item.get("fp"),
                }
            )

        if not normalized:
            continue

        deduped = {}
        for entry in normalized:
            current = deduped.get(entry["year"])
            if current is None or sec_entry_priority(entry) > sec_entry_priority(current):
                deduped[entry["year"]] = entry

        selected_entries = [deduped[year] for year in sorted(deduped)]
        latest_year = max(entry["year"] for entry in selected_entries)
        nonzero_count = sum(abs(entry["value"]) > 1e-9 for entry in selected_entries)
        fy_count = sum(str(entry.get("fp", "")).upper() == "FY" for entry in selected_entries)
        latest_value = safe_num(selected_entries[-1].get("value"))
        concept_score = (
            latest_year,
            len(selected_entries),
            nonzero_count,
            fy_count,
            1 if has_numeric_value(latest_value) and abs(latest_value) > 1e-9 else 0,
        )
        if best_score is None or concept_score > best_score:
            best_entries = selected_entries
            best_score = concept_score

    return best_entries


def latest_sec_metric_value(entries):
    if not entries:
        return None
    return entries[-1].get("value")


def build_sec_financial_dataset(companyfacts):
    metric_config = {
        "Revenue": {
            "concepts": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
            "preferred_units": ["USD"],
        },
        "OperatingCF": {
            "concepts": [
                "NetCashProvidedByUsedInOperatingActivities",
                "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
            ],
            "preferred_units": ["USD"],
        },
        "CapEx": {
            "concepts": [
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "CapitalExpendituresIncurredButNotYetPaid",
                "PropertyPlantAndEquipmentAdditions",
            ],
            "preferred_units": ["USD"],
            "absolute_value": True,
        },
        "NetIncome": {"concepts": ["NetIncomeLoss"], "preferred_units": ["USD"]},
        "OperatingIncome": {"concepts": ["OperatingIncomeLoss"], "preferred_units": ["USD"]},
        "DebtBalance": {
            "concepts": [
                "DebtLongtermAndShorttermCombinedAmount",
                "LongTermDebtAndCapitalLeaseObligations",
                "LongTermDebtNoncurrent",
                "LongTermDebt",
            ],
            "preferred_units": ["USD"],
        },
        "Cash": {"concepts": ["CashAndCashEquivalentsAtCarryingValue"], "preferred_units": ["USD"]},
        "SharesOutstanding": {
            "concepts": ["CommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding"],
            "preferred_units": ["shares"],
        },
        "Depreciation": {
            "concepts": [
                "DepreciationDepletionAndAmortization",
                "DepreciationAndAmortization",
                "Depreciation",
            ],
            "preferred_units": ["USD"],
        },
        "Amortization": {
            "concepts": [
                "AmortizationOfIntangibleAssets",
                "FiniteLivedIntangibleAssetsAmortizationExpense",
            ],
            "preferred_units": ["USD"],
        },
        "TaxExpense": {"concepts": ["IncomeTaxExpenseBenefit"], "preferred_units": ["USD"]},
        "InterestExpense": {"concepts": ["InterestExpenseAndDebtExpense", "InterestExpense"], "preferred_units": ["USD"]},
        "PretaxIncome": {
            "concepts": [
                "IncomeBeforeTaxExpenseBenefit",
                "PretaxIncome",
                "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
            ],
            "preferred_units": ["USD"],
        },
    }

    metric_entries = {}
    year_map = {}
    for metric_name, config in metric_config.items():
        entries = extract_company_fact_entries(
            companyfacts,
            config["concepts"],
            preferred_units=config.get("preferred_units"),
            forms=SEC_ANNUAL_FORMS,
        )
        if config.get("absolute_value"):
            for entry in entries:
                entry["value"] = abs(entry["value"])
        metric_entries[metric_name] = entries
        for entry in entries:
            year_map.setdefault(entry["year"], {})[metric_name] = entry["value"]

    history_years = sorted(year_map)[-5:]
    history_rows = []
    for year in history_years:
        operating_cf = year_map[year].get("OperatingCF")
        capex = year_map[year].get("CapEx")
        free_cash_flow = (
            operating_cf - capex
            if has_numeric_value(operating_cf) and has_numeric_value(capex)
            else None
        )
        history_rows.append(
            {
                "Year": year,
                "Revenue": year_map[year].get("Revenue"),
                "OperatingCF": operating_cf,
                "CapEx": capex,
                "FreeCashFlow": free_cash_flow,
            }
        )

    latest = {metric_name: latest_sec_metric_value(entries) for metric_name, entries in metric_entries.items()}
    return {
        "history": history_rows,
        "latest": latest,
        "metric_entries": metric_entries,
    }


def calculate_growth_rate_from_series(history_rows, field_name, lookback_years=3):
    valid_rows = [row for row in history_rows if has_numeric_value(row.get(field_name))]
    if len(valid_rows) < 2:
        return None

    end_row = valid_rows[-1]
    start_index = max(0, len(valid_rows) - lookback_years - 1)
    start_row = valid_rows[start_index]
    year_span = max(int(end_row["Year"]) - int(start_row["Year"]), 1)
    start_value = safe_num(start_row.get(field_name))
    end_value = safe_num(end_row.get(field_name))

    if has_numeric_value(start_value) and has_numeric_value(end_value) and start_value > 0 and end_value > 0:
        return float((end_value / start_value) ** (1 / year_span) - 1)

    pct_changes = []
    for previous_row, current_row in zip(valid_rows[:-1], valid_rows[1:]):
        previous_value = safe_num(previous_row.get(field_name))
        current_value = safe_num(current_row.get(field_name))
        if has_numeric_value(previous_value) and has_numeric_value(current_value) and abs(previous_value) > 1e-9:
            pct_changes.append((current_value - previous_value) / abs(previous_value))
    if pct_changes:
        return float(np.mean(pct_changes[-lookback_years:]))
    return None


def build_growth_schedule(initial_growth_rate, terminal_growth_rate, years):
    if years <= 1:
        return [float(terminal_growth_rate)]
    return [float(value) for value in np.linspace(initial_growth_rate, terminal_growth_rate, years)]


def fetch_treasury_10y_yield():
    now = datetime.datetime.now()
    month_candidates = [now, now - datetime.timedelta(days=35), now - datetime.timedelta(days=70)]
    last_error = None
    for candidate in month_candidates:
        cache_key = f"{candidate.year}-{candidate.month:02d}"
        cached = get_cached_fetch_payload("treasury_yield", cache_key, max_age_seconds=86400)
        if cached is not None:
            return float(cached), None

        url = (
            "https://data.treasury.gov/feed.svc/DailyTreasuryYieldCurveRateData"
            f"?$filter=month(NEW_DATE)%20eq%20{candidate.month}%20and%20year(NEW_DATE)%20eq%20{candidate.year}"
            "&$select=NEW_DATE,BC_10YEAR&$orderby=NEW_DATE%20desc&$format=json"
        )
        payload, error = fetch_json_url_with_retry(url, attempts=2, timeout=15)
        if error:
            last_error = error
        results = payload.get("d", {}).get("results", []) if isinstance(payload, dict) else []
        for item in results:
            rate = safe_num(item.get("BC_10YEAR"))
            if has_numeric_value(rate):
                decimal_rate = float(rate) / 100
                set_cached_fetch_payload("treasury_yield", cache_key, decimal_rate)
                return decimal_rate, None
    return DCF_DEFAULT_RISK_FREE_RATE, last_error or "Treasury yield fetch failed; used fallback."


def compute_wacc_components(ticker, info, sec_dataset, dcf_settings):
    latest = sec_dataset.get("latest", {})
    manual_risk_free_rate = safe_num((dcf_settings or {}).get("risk_free_rate_override"))
    if has_numeric_value(manual_risk_free_rate):
        risk_free_rate = float(manual_risk_free_rate)
        rf_error = None
    else:
        risk_free_rate, rf_error = fetch_treasury_10y_yield()
    beta = safe_num((info or {}).get("beta"))
    if not has_numeric_value(beta):
        beta = 1.0

    market_cap = safe_num((info or {}).get("marketCap"))
    debt_balance = safe_num(latest.get("DebtBalance"))
    if not has_numeric_value(debt_balance):
        debt_balance = safe_num((info or {}).get("totalDebt"))
    cash = safe_num(latest.get("Cash"))
    shares_outstanding = safe_num(latest.get("SharesOutstanding")) or safe_num((info or {}).get("sharesOutstanding"))

    market_risk_premium = float((dcf_settings or {}).get("market_risk_premium", DCF_DEFAULT_MARKET_RISK_PREMIUM))
    cost_of_equity = float(risk_free_rate + beta * market_risk_premium)

    pretax_income = safe_num(latest.get("PretaxIncome"))
    tax_expense = safe_num(latest.get("TaxExpense"))
    effective_tax_rate = safe_divide(tax_expense, pretax_income)
    if not has_numeric_value(effective_tax_rate):
        effective_tax_rate = 0.21
    effective_tax_rate = float(np.clip(effective_tax_rate, 0.0, 0.45))

    interest_expense = safe_num(latest.get("InterestExpense"))
    pre_tax_cost_of_debt = (
        safe_divide(abs(interest_expense), abs(debt_balance))
        if has_numeric_value(interest_expense) and has_numeric_value(debt_balance)
        else None
    )
    if not has_numeric_value(pre_tax_cost_of_debt) or pre_tax_cost_of_debt <= 0 or pre_tax_cost_of_debt > 0.20:
        after_tax_cost_of_debt = float(
            (dcf_settings or {}).get("default_after_tax_cost_of_debt", DCF_DEFAULT_AFTER_TAX_COST_OF_DEBT)
        )
    else:
        after_tax_cost_of_debt = float(pre_tax_cost_of_debt * (1 - effective_tax_rate))

    if has_numeric_value(market_cap) and has_numeric_value(debt_balance) and (market_cap + debt_balance) > 0:
        equity_weight = float(market_cap / (market_cap + debt_balance))
        debt_weight = float(debt_balance / (market_cap + debt_balance))
    elif has_numeric_value(market_cap) and market_cap > 0:
        equity_weight = 1.0
        debt_weight = 0.0
    elif has_numeric_value(debt_balance) and debt_balance > 0:
        equity_weight = 0.0
        debt_weight = 1.0
    else:
        equity_weight = 1.0
        debt_weight = 0.0

    wacc = (equity_weight * cost_of_equity) + (debt_weight * after_tax_cost_of_debt)
    wacc = float(np.clip(wacc, 0.06, 0.20))

    return {
        "risk_free_rate": float(risk_free_rate),
        "beta": float(beta),
        "market_risk_premium": market_risk_premium,
        "cost_of_equity": cost_of_equity,
        "after_tax_cost_of_debt": after_tax_cost_of_debt,
        "equity_weight": equity_weight,
        "debt_weight": debt_weight,
        "wacc": wacc,
        "cash": cash,
        "debt_balance": debt_balance,
        "shares_outstanding": shares_outstanding,
        "market_cap": market_cap,
        "tax_rate": effective_tax_rate,
        "notes": [rf_error] if rf_error else [],
    }


def determine_growth_assumptions(history_rows, dcf_settings=None):
    dcf_settings = normalize_dcf_settings(dcf_settings or {})
    historical_fcf_growth = calculate_growth_rate_from_series(history_rows, "FreeCashFlow", lookback_years=3)
    historical_revenue_growth = calculate_growth_rate_from_series(history_rows, "Revenue", lookback_years=3)
    if has_numeric_value(historical_fcf_growth) and has_numeric_value(historical_revenue_growth):
        growth_gap = abs(float(historical_fcf_growth) - float(historical_revenue_growth))
        if growth_gap >= 0.12:
            historical_growth_estimate = float(
                historical_fcf_growth * 0.35 + historical_revenue_growth * 0.65
            )
        else:
            historical_growth_estimate = float(np.mean([historical_fcf_growth, historical_revenue_growth]))
    elif has_numeric_value(historical_revenue_growth):
        historical_growth_estimate = float(historical_revenue_growth)
    elif has_numeric_value(historical_fcf_growth):
        historical_growth_estimate = float(historical_fcf_growth)
    else:
        historical_growth_estimate = 0.04

    manual_growth_rate = safe_num(dcf_settings.get("manual_growth_rate"))
    selected_source_rate = manual_growth_rate if has_numeric_value(manual_growth_rate) else historical_growth_estimate
    guidance_source = "Manual override" if has_numeric_value(manual_growth_rate) else "Historical trend fallback"
    guidance_confidence = "manual" if has_numeric_value(manual_growth_rate) else "historical"
    guidance_summary = (
        "Used the manual growth override supplied in the DCF tab."
        if has_numeric_value(manual_growth_rate)
        else "Used recent SEC cash-flow and revenue history as the base growth assumption."
    )

    selected_pre_haircut = selected_source_rate
    selected_growth_rate = float(
        np.clip(
            selected_pre_haircut * dcf_settings["growth_haircut"],
            dcf_settings["min_growth_rate"],
            dcf_settings["max_growth_rate"],
        )
    )

    return {
        "historical_fcf_growth": historical_fcf_growth,
        "historical_revenue_growth": historical_revenue_growth,
        "historical_growth_estimate": historical_growth_estimate,
        "guidance_rate": manual_growth_rate,
        "selected_growth_rate": selected_growth_rate,
        "source": guidance_source,
        "confidence": guidance_confidence,
        "summary": guidance_summary,
        "growth_schedule": build_growth_schedule(
            selected_growth_rate,
            dcf_settings["terminal_growth_rate"],
            dcf_settings["projection_years"],
        ),
    }


def extract_recent_metric_values(history_rows, field_name, limit=5):
    values = []
    for row in history_rows or []:
        value = safe_num(row.get(field_name))
        if has_numeric_value(value):
            values.append(float(value))
    return values[-limit:]


def calculate_normalized_base_fcf(history_rows, latest_metrics=None):
    latest_metrics = latest_metrics or {}
    valid_rows = [
        row for row in history_rows or []
        if has_numeric_value(row.get("OperatingCF")) or has_numeric_value(row.get("FreeCashFlow"))
    ]
    if not valid_rows:
        return {
            "base_fcf": None,
            "latest_fcf": None,
            "normalized_fcf": None,
            "maintenance_capex": None,
            "used_normalized_capex": False,
            "capex_source": "Unavailable",
        }

    latest_row = valid_rows[-1]
    latest_operating_cf = safe_num(latest_row.get("OperatingCF"))
    latest_capex = safe_num(latest_row.get("CapEx"))
    latest_fcf = safe_num(latest_row.get("FreeCashFlow"))
    depreciation = safe_num(latest_metrics.get("Depreciation"))

    recent_capex_values = extract_recent_metric_values(valid_rows, "CapEx", limit=5)
    recent_fcf_values = extract_recent_metric_values(valid_rows, "FreeCashFlow", limit=5)
    recent_capex_window = recent_capex_values[-3:] if recent_capex_values else []
    recent_fcf_window = recent_fcf_values[-3:] if recent_fcf_values else []
    median_recent_capex = float(np.median(recent_capex_window)) if recent_capex_window else None
    median_recent_fcf = float(np.median(recent_fcf_window)) if recent_fcf_window else None

    maintenance_floor = depreciation if has_numeric_value(depreciation) else median_recent_capex
    maintenance_capex = latest_capex
    capex_source = "Latest reported capex"
    used_normalized_capex = False

    if has_numeric_value(median_recent_capex):
        spike_threshold = max(
            median_recent_capex * 1.35,
            (maintenance_floor if has_numeric_value(maintenance_floor) else median_recent_capex) * 1.75,
        )
        if not has_numeric_value(maintenance_capex):
            maintenance_capex = max(
                median_recent_capex,
                maintenance_floor if has_numeric_value(maintenance_floor) else 0.0,
            )
            capex_source = "Recent median capex"
            used_normalized_capex = True
        elif maintenance_capex > spike_threshold:
            maintenance_capex = max(
                median_recent_capex,
                maintenance_floor if has_numeric_value(maintenance_floor) else 0.0,
            )
            capex_source = "Normalized capex from recent median"
            used_normalized_capex = True
    elif not has_numeric_value(maintenance_capex) and has_numeric_value(maintenance_floor):
        maintenance_capex = float(maintenance_floor)
        capex_source = "Depreciation proxy"
        used_normalized_capex = True

    normalized_fcf = (
        latest_operating_cf - maintenance_capex
        if has_numeric_value(latest_operating_cf) and has_numeric_value(maintenance_capex)
        else latest_fcf
    )

    candidate_fcfs = []
    if has_numeric_value(normalized_fcf):
        candidate_fcfs.append(float(normalized_fcf))
    if has_numeric_value(median_recent_fcf):
        candidate_fcfs.append(float(median_recent_fcf))
    if len(recent_fcf_window) >= 2:
        weights = np.arange(1, len(recent_fcf_window) + 1)
        candidate_fcfs.append(float(np.average(recent_fcf_window, weights=weights)))

    positive_recent_count = sum(value > 0 for value in recent_fcf_window)
    if (
        has_numeric_value(latest_fcf)
        and latest_fcf > 0
        and positive_recent_count >= 2
        and not used_normalized_capex
    ):
        candidate_fcfs.append(float(latest_fcf))

    if not candidate_fcfs:
        base_fcf = latest_fcf
    elif positive_recent_count == 0 and (not has_numeric_value(normalized_fcf) or normalized_fcf <= 0):
        base_fcf = max(candidate_fcfs)
    else:
        positive_candidates = [value for value in candidate_fcfs if value > 0]
        base_fcf = float(np.mean(positive_candidates or candidate_fcfs))

    return {
        "base_fcf": base_fcf,
        "latest_fcf": latest_fcf,
        "normalized_fcf": normalized_fcf,
        "maintenance_capex": maintenance_capex,
        "used_normalized_capex": used_normalized_capex,
        "capex_source": capex_source,
    }


def estimate_terminal_exit_value(info, sec_dataset, growth_schedule, wacc, projection_years, peer_benchmarks=None):
    latest = sec_dataset.get("latest", {})
    latest_revenue = safe_num(latest.get("Revenue"))
    if not has_numeric_value(latest_revenue) or latest_revenue <= 0:
        return {}

    projected_revenue = float(latest_revenue)
    for growth_rate in growth_schedule or []:
        projected_revenue *= (1 + growth_rate)

    current_ebitda = safe_num((info or {}).get("ebitda"))
    if not has_numeric_value(current_ebitda):
        operating_income = safe_num(latest.get("OperatingIncome"))
        depreciation = safe_num(latest.get("Depreciation"))
        amortization = safe_num(latest.get("Amortization"))
        if has_numeric_value(operating_income):
            current_ebitda = float(operating_income + (depreciation or 0.0) + (amortization or 0.0))

    exit_candidates = []
    year5_ebitda = None
    current_ev_ebitda = safe_num((info or {}).get("enterpriseToEbitda"))
    peer_ev_ebitda = safe_num((peer_benchmarks or {}).get("EV_EBITDA"))
    ebitda_multiple_candidates = []
    if has_numeric_value(current_ev_ebitda) and current_ev_ebitda > 0:
        ebitda_multiple_candidates.append(float(np.clip(current_ev_ebitda * 0.90, 6.0, 20.0)))
    if has_numeric_value(peer_ev_ebitda) and peer_ev_ebitda > 0:
        ebitda_multiple_candidates.append(float(np.clip(peer_ev_ebitda, 6.0, 18.0)))

    if has_numeric_value(current_ebitda) and current_ebitda > 0 and has_numeric_value(latest_revenue):
        ebitda_margin = float(current_ebitda / latest_revenue)
        if 0.02 <= ebitda_margin <= 0.65 and ebitda_multiple_candidates:
            year5_ebitda = float(projected_revenue * ebitda_margin)
            exit_candidates.append(
                {
                    "method": "EV/EBITDA",
                    "terminal_value": year5_ebitda * float(np.mean(ebitda_multiple_candidates)),
                }
            )

    current_ev_revenue = safe_num((info or {}).get("enterpriseToRevenue"))
    if has_numeric_value(current_ev_revenue) and current_ev_revenue > 0:
        revenue_multiple = float(np.clip(current_ev_revenue * 0.85, 1.0, 12.0))
        exit_candidates.append(
            {
                "method": "EV/Sales",
                "terminal_value": projected_revenue * revenue_multiple,
            }
        )

    if not exit_candidates:
        return {}

    terminal_value = float(np.mean([candidate["terminal_value"] for candidate in exit_candidates]))
    present_value = terminal_value / ((1 + wacc) ** projection_years)
    return {
        "terminal_value": terminal_value,
        "present_value": present_value,
        "methods": [candidate["method"] for candidate in exit_candidates],
        "projected_revenue": projected_revenue,
        "projected_ebitda": year5_ebitda,
    }


def build_dcf_sensitivity_grid(projected_fcfs, wacc, cash, debt, shares_outstanding, dcf_settings):
    sensitivity_rows = []
    if not projected_fcfs or not has_numeric_value(shares_outstanding) or shares_outstanding <= 0:
        return sensitivity_rows

    wacc_range = [round(wacc + delta, 4) for delta in np.arange(-0.02, 0.0201, 0.005)]
    center_terminal_growth = float((dcf_settings or {}).get("terminal_growth_rate", DCF_TERMINAL_GROWTH_RATE))
    terminal_growth_range = sorted(
        {
            round(max(0.0, center_terminal_growth - 0.01), 3),
            round(max(0.0, center_terminal_growth - 0.005), 3),
            round(center_terminal_growth, 3),
            round(center_terminal_growth + 0.005, 3),
            round(center_terminal_growth + 0.01, 3),
        }
    )

    for wacc_candidate in wacc_range:
        row = {"WACC": wacc_candidate}
        for growth_candidate in terminal_growth_range:
            if wacc_candidate <= growth_candidate + 0.0025:
                row[f"TG_{growth_candidate:.3f}"] = None
                continue
            pv_fcfs = sum(
                projected_fcf / ((1 + wacc_candidate) ** year)
                for year, projected_fcf in enumerate(projected_fcfs, start=1)
            )
            terminal_value = projected_fcfs[-1] * (1 + growth_candidate) / (wacc_candidate - growth_candidate)
            pv_terminal = terminal_value / ((1 + wacc_candidate) ** len(projected_fcfs))
            enterprise_value = pv_fcfs + pv_terminal
            equity_value = enterprise_value - (debt or 0) + (cash or 0)
            row[f"TG_{growth_candidate:.3f}"] = safe_divide(equity_value, shares_outstanding)
        sensitivity_rows.append(row)
    return sensitivity_rows


def build_sec_dcf_model(ticker, price, info, dcf_settings=None, peer_benchmarks=None):
    dcf_settings = normalize_dcf_settings(dcf_settings or {})
    cik_padded, company_title, cik_error = lookup_company_cik(ticker)
    if not cik_padded:
        return {"available": False, "error": cik_error or f"Could not find an SEC CIK for {ticker}."}

    companyfacts, facts_error = fetch_sec_companyfacts(cik_padded)
    if not companyfacts:
        return {
            "available": False,
            "error": facts_error or f"SEC company facts were unavailable for {ticker}.",
            "cik": cik_padded,
        }

    submissions, submissions_error = fetch_sec_submissions(cik_padded)
    sec_dataset = build_sec_financial_dataset(companyfacts)
    history_rows = sec_dataset.get("history", [])
    latest = sec_dataset.get("latest", {})

    base_fcf_inputs = calculate_normalized_base_fcf(history_rows, latest_metrics=latest)
    base_fcf = safe_num(base_fcf_inputs.get("base_fcf"))
    if not has_numeric_value(base_fcf):
        return {
            "available": False,
            "error": "SEC filing history did not provide enough operating cash flow and capex data to build a DCF.",
            "cik": cik_padded,
            "company_name": companyfacts.get("entityName") or company_title or ticker,
        }

    filing_metadata = parse_sec_filing_metadata(submissions or {}, preferred_forms=SEC_FILING_SEARCH_FORMS)
    filing_text, filing_text_error = (
        fetch_sec_filing_text(cik_padded, filing_metadata) if filing_metadata else (None, submissions_error)
    )
    filing_takeaways = extract_filing_takeaways_from_text(filing_text) if filing_text else []
    growth_inputs = determine_growth_assumptions(history_rows, dcf_settings=dcf_settings)
    if base_fcf_inputs.get("used_normalized_capex"):
        growth_inputs["summary"] += (
            f" Base FCF was normalized using {base_fcf_inputs.get('capex_source', 'recent capex history')}."
        )

    wacc_inputs = compute_wacc_components(ticker, info, sec_dataset, dcf_settings=dcf_settings)
    shares_outstanding = safe_num(wacc_inputs.get("shares_outstanding")) or safe_num((info or {}).get("sharesOutstanding"))
    if not has_numeric_value(shares_outstanding) or shares_outstanding <= 0:
        return {
            "available": False,
            "error": "Shares outstanding were unavailable, so the DCF could not be converted into a per-share value.",
            "cik": cik_padded,
            "company_name": companyfacts.get("entityName") or company_title or ticker,
        }

    current_fcf = float(base_fcf)
    projected_fcfs = []
    projection_rows = []
    for year, growth_rate in enumerate(growth_inputs["growth_schedule"], start=1):
        current_fcf = current_fcf * (1 + growth_rate)
        discount_factor = 1 / ((1 + wacc_inputs["wacc"]) ** year)
        pv = current_fcf * discount_factor
        projected_fcfs.append(current_fcf)
        projection_rows.append(
            {
                "Year": year,
                "GrowthRate": growth_rate,
                "FreeCashFlow": current_fcf,
                "DiscountFactor": discount_factor,
                "PresentValue": pv,
            }
        )

    if wacc_inputs["wacc"] <= dcf_settings["terminal_growth_rate"] + 0.0025:
        return {
            "available": False,
            "error": "The DCF could not be stabilized because the discount rate was too close to terminal growth.",
            "cik": cik_padded,
            "company_name": companyfacts.get("entityName") or company_title or ticker,
        }

    terminal_value = projected_fcfs[-1] * (1 + dcf_settings["terminal_growth_rate"]) / (
        wacc_inputs["wacc"] - dcf_settings["terminal_growth_rate"]
    )
    present_value_of_terminal = terminal_value / ((1 + wacc_inputs["wacc"]) ** dcf_settings["projection_years"])
    exit_terminal = estimate_terminal_exit_value(
        info,
        sec_dataset,
        growth_inputs["growth_schedule"],
        wacc_inputs["wacc"],
        dcf_settings["projection_years"],
        peer_benchmarks=peer_benchmarks,
    )
    if has_numeric_value(exit_terminal.get("present_value")):
        exit_weight = 0.35
        if (
            base_fcf_inputs.get("used_normalized_capex")
            and has_numeric_value(base_fcf_inputs.get("latest_fcf"))
            and base_fcf_inputs["latest_fcf"] <= 0
        ):
            exit_weight = 0.50
        present_value_of_terminal = (
            present_value_of_terminal * (1 - exit_weight)
            + exit_terminal["present_value"] * exit_weight
        )
        terminal_value = terminal_value * (1 - exit_weight) + exit_terminal["terminal_value"] * exit_weight
    sum_of_projected_pv = float(sum(row["PresentValue"] for row in projection_rows))
    enterprise_value = sum_of_projected_pv + present_value_of_terminal
    equity_value = enterprise_value - (wacc_inputs.get("debt_balance") or 0) + (wacc_inputs.get("cash") or 0)
    intrinsic_value_per_share = safe_divide(equity_value, shares_outstanding)
    dcf_upside = safe_divide(intrinsic_value_per_share - price, price) if has_numeric_value(price) else None

    notes = []
    if cik_error:
        notes.append(cik_error)
    if facts_error:
        notes.append(facts_error)
    if submissions_error:
        notes.append(submissions_error)
    if filing_text_error:
        notes.append(filing_text_error)
    notes.extend(wacc_inputs.get("notes", []))
    if base_fcf_inputs.get("used_normalized_capex"):
        notes.append(
            f"Base FCF normalized from recent operating cash flow and {base_fcf_inputs.get('capex_source', 'capex history')}."
        )
    if has_numeric_value(exit_terminal.get("present_value")):
        notes.append(
            f"Terminal value blended with {', '.join(exit_terminal.get('methods', []))} cross-check."
        )

    return {
        "available": True,
        "ticker": ticker,
        "company_name": companyfacts.get("entityName") or company_title or ticker,
        "cik": cik_padded,
        "filing_form": filing_metadata.get("form") if filing_metadata else None,
        "filing_date": filing_metadata.get("filing_date") if filing_metadata else None,
        "history": history_rows,
        "projection": projection_rows,
        "sensitivity": build_dcf_sensitivity_grid(
            projected_fcfs,
            wacc_inputs["wacc"],
            wacc_inputs.get("cash"),
            wacc_inputs.get("debt_balance"),
            shares_outstanding,
            dcf_settings,
        ),
        "guidance_excerpts": filing_takeaways,
        "guidance_summary": growth_inputs["summary"],
        "guidance_payload": {},
        "historical_fcf_growth": growth_inputs["historical_fcf_growth"],
        "historical_revenue_growth": growth_inputs["historical_revenue_growth"],
        "guidance_growth_rate": growth_inputs["guidance_rate"],
        "selected_growth_rate": growth_inputs["selected_growth_rate"],
        "growth_source": growth_inputs["source"],
        "growth_confidence": growth_inputs["confidence"],
        "growth_schedule": growth_inputs["growth_schedule"],
        "base_fcf": base_fcf,
        "terminal_growth_rate": dcf_settings["terminal_growth_rate"],
        "risk_free_rate": wacc_inputs["risk_free_rate"],
        "beta": wacc_inputs["beta"],
        "cost_of_equity": wacc_inputs["cost_of_equity"],
        "after_tax_cost_of_debt": wacc_inputs["after_tax_cost_of_debt"],
        "equity_weight": wacc_inputs["equity_weight"],
        "debt_weight": wacc_inputs["debt_weight"],
        "wacc": wacc_inputs["wacc"],
        "cash": wacc_inputs.get("cash"),
        "long_term_debt": wacc_inputs.get("debt_balance"),
        "shares_outstanding": shares_outstanding,
        "enterprise_value": enterprise_value,
        "equity_value": equity_value,
        "sum_of_projected_pv": sum_of_projected_pv,
        "terminal_value": terminal_value,
        "present_value_of_terminal": present_value_of_terminal,
        "intrinsic_value_per_share": intrinsic_value_per_share,
        "upside": dcf_upside,
        "notes": [note for note in notes if note],
        "latest_sec_values": latest,
        "dcf_settings": dcf_settings,
    }


def calculate_realized_volatility(close, window):
    if close is None or len(close) < max(window, 2):
        return None
    returns = close.pct_change().dropna()
    if len(returns) < window:
        return None
    return float(returns.tail(window).std() * np.sqrt(252))


def calculate_trend_strength(price, sma50, sma200, momentum_1y=None):
    components = []
    if has_numeric_value(price) and has_numeric_value(sma200) and sma200 > 0:
        components.append(np.clip((price / sma200 - 1) * 100, -25, 25))
    if has_numeric_value(sma50) and has_numeric_value(sma200) and sma200 > 0:
        components.append(np.clip((sma50 / sma200 - 1) * 120, -25, 25))
    if has_numeric_value(momentum_1y):
        components.append(np.clip(momentum_1y * 80, -25, 25))
    if not components:
        return None
    return float(np.clip(sum(components), -100, 100))


def calculate_52w_context(close):
    if close is None or close.empty:
        return None, None, None
    window = close.tail(min(len(close), 252))
    rolling_high = safe_num(window.max())
    rolling_low = safe_num(window.min())
    price = safe_num(window.iloc[-1])
    if not has_numeric_value(price) or not has_numeric_value(rolling_high) or not has_numeric_value(rolling_low):
        return None, None, None
    range_position = safe_divide(price - rolling_low, rolling_high - rolling_low)
    distance_high = safe_divide(price - rolling_high, rolling_high)
    distance_low = safe_divide(price - rolling_low, rolling_low)
    return range_position, distance_high, distance_low


def calculate_quality_score(roe, margins, debt_eq, revenue_growth, earnings_growth, current_ratio, settings):
    score = 0.0
    if has_numeric_value(roe):
        score += 1.25 if roe >= settings["fund_roe_threshold"] else (-1 if roe < 0 else 0)
    if has_numeric_value(margins):
        score += 1.25 if margins >= settings["fund_profit_margin_threshold"] else (-1 if margins < 0 else 0)
    if has_numeric_value(debt_eq):
        if 0 <= debt_eq < settings["fund_debt_good_threshold"]:
            score += 1.0
        elif debt_eq > settings["fund_debt_bad_threshold"]:
            score -= 1.0
    if has_numeric_value(revenue_growth) and has_numeric_value(earnings_growth):
        if revenue_growth > 0 and earnings_growth > 0:
            score += 1.0
        elif revenue_growth < 0 and earnings_growth < 0:
            score -= 1.0
    elif has_numeric_value(revenue_growth):
        score += 0.5 if revenue_growth > 0 else -0.5
    elif has_numeric_value(earnings_growth):
        score += 0.5 if earnings_growth > 0 else -0.5
    if has_numeric_value(current_ratio):
        if current_ratio >= settings["fund_current_ratio_good"]:
            score += 0.5
        elif current_ratio < settings["fund_current_ratio_bad"]:
            score -= 0.5
    return float(np.clip(score, -4, 5))


def calculate_dividend_safety_score(dividend_yield, payout_ratio, margins, current_ratio, debt_eq):
    score = 0.0
    if has_numeric_value(dividend_yield):
        if dividend_yield >= 0.06:
            score += 0.5
        elif dividend_yield >= 0.025:
            score += 1.0
    if has_numeric_value(payout_ratio):
        if 0 < payout_ratio <= 0.65:
            score += 2.0
        elif payout_ratio <= 0.85:
            score += 1.0
        elif payout_ratio > 1.0:
            score -= 2.0
        elif payout_ratio > 0.85:
            score -= 1.0
    if has_numeric_value(margins) and margins > 0:
        score += 0.5
    if has_numeric_value(current_ratio):
        score += 0.5 if current_ratio >= 1.0 else -0.5
    if has_numeric_value(debt_eq):
        if debt_eq < 120:
            score += 0.5
        elif debt_eq > 220:
            score -= 0.5
    return float(np.clip(score, -3, 4))


def calculate_valuation_confidence(signal_count):
    if not has_numeric_value(signal_count):
        return 25.0
    signal_count = float(signal_count)
    return float(np.clip(20 + signal_count * 12, 20, 95))


def calculate_sentiment_conviction(sentiment_score, analyst_opinions, recommendation_key, target_mean_price, price, headline_count):
    conviction = min(abs(float(sentiment_score)) * 2, 10)
    if has_numeric_value(analyst_opinions):
        conviction += min(float(analyst_opinions) * 2.5, 25)
    if recommendation_key and recommendation_key != "N/A":
        conviction += 10
    if has_numeric_value(target_mean_price) and has_numeric_value(price) and price > 0:
        conviction += min(abs((target_mean_price - price) / price) * 100, 20)
    conviction += min((headline_count or 0) * 3, 10)
    return float(np.clip(conviction, 10, 95))


def get_type_adjusted_engine_weights(stock_profile, settings):
    primary_type = stock_profile.get("primary_type", "")
    modifiers = {
        "technical": 1.0,
        "fundamental": 1.0,
        "valuation": 1.0,
        "sentiment": 1.0,
    }
    if primary_type == "Growth Stocks":
        modifiers.update({"technical": 1.2, "fundamental": 1.05, "valuation": 0.75, "sentiment": 1.15})
    elif primary_type == "Value Stocks":
        modifiers.update({"technical": 0.85, "fundamental": 1.2, "valuation": 1.35, "sentiment": 0.8})
    elif primary_type == "Dividend / Income Stocks":
        modifiers.update({"technical": 0.75, "fundamental": 1.25, "valuation": 1.1, "sentiment": 0.7})
    elif primary_type == "Cyclical Stocks":
        modifiers.update({"technical": 1.25, "fundamental": 0.95, "valuation": 0.9, "sentiment": 1.0})
    elif primary_type == "Defensive Stocks":
        modifiers.update({"technical": 0.8, "fundamental": 1.25, "valuation": 1.0, "sentiment": 0.7})
    elif primary_type == "Blue-Chip Stocks":
        modifiers.update({"technical": 0.9, "fundamental": 1.2, "valuation": 0.95, "sentiment": 0.8})
    elif primary_type == "Small-Cap Stocks":
        modifiers.update({"technical": 1.15, "fundamental": 1.0, "valuation": 0.8, "sentiment": 1.05})
    elif primary_type == "Mid-Cap Stocks":
        modifiers.update({"technical": 1.05, "fundamental": 1.05, "valuation": 0.95, "sentiment": 0.95})
    elif primary_type == "Large-Cap Stocks":
        modifiers.update({"technical": 0.95, "fundamental": 1.1, "valuation": 1.0, "sentiment": 0.85})
    elif primary_type == "Speculative / Penny Stocks":
        modifiers.update({"technical": 1.35, "fundamental": 0.7, "valuation": 0.55, "sentiment": 1.25})

    weights = {
        "technical": settings["weight_technical"] * modifiers["technical"],
        "fundamental": settings["weight_fundamental"] * modifiers["fundamental"],
        "valuation": settings["weight_valuation"] * modifiers["valuation"],
        "sentiment": settings["weight_sentiment"] * modifiers["sentiment"],
    }
    weight_profile = (
        f"T {weights['technical']:.2f} | F {weights['fundamental']:.2f} | "
        f"V {weights['valuation']:.2f} | S {weights['sentiment']:.2f}"
    )
    return weights, weight_profile


def build_risk_flags(
    *,
    eps,
    debt_eq,
    current_ratio,
    overextended,
    distance_52w_high,
    range_position,
    volatility_1y,
    stock_profile,
):
    flags = []
    if has_numeric_value(eps) and eps <= 0:
        flags.append("Negative EPS")
    if has_numeric_value(debt_eq) and debt_eq > 220:
        flags.append("High Debt")
    if has_numeric_value(current_ratio) and current_ratio < 1.0:
        flags.append("Weak Liquidity")
    if overextended:
        flags.append("Overextended")
    if has_numeric_value(distance_52w_high) and distance_52w_high <= -0.20:
        flags.append("Far Below 52W High")
    if has_numeric_value(range_position) and range_position <= 0.15:
        flags.append("Near 52W Low")
    if has_numeric_value(volatility_1y) and volatility_1y >= 0.55:
        flags.append("High Volatility")
    if stock_profile.get("primary_type") == "Speculative / Penny Stocks":
        flags.append("Speculative")
    return flags


def classify_cap_bucket(market_cap):
    if not has_numeric_value(market_cap) or market_cap <= 0:
        return "Unknown"
    if market_cap < 2_000_000_000:
        return "Small-Cap"
    if market_cap < 10_000_000_000:
        return "Mid-Cap"
    return "Large-Cap"


def build_stock_type_strategy(primary_type):
    return STOCK_TYPE_STRATEGIES.get(
        primary_type,
        "Use the stock's blend of quality, valuation, and regime signals instead of forcing it into a one-size-fits-all strategy.",
    )


def classify_stock_profile(
    *,
    sector,
    price,
    market_cap,
    dividend_yield,
    payout_ratio,
    equity_beta,
    analyst_opinions,
    pe,
    forward_pe,
    peg_ratio,
    ps_ratio,
    pb,
    bench,
    f_score,
    v_val,
    revenue_growth,
    earnings_growth,
    margins,
    roe,
    current_ratio,
    debt_eq,
    momentum_1y,
):
    cap_bucket = classify_cap_bucket(market_cap)
    scores = {
        "Growth Stocks": 0.0,
        "Value Stocks": 0.0,
        "Dividend / Income Stocks": 0.0,
        "Cyclical Stocks": 0.0,
        "Defensive Stocks": 0.0,
        "Blue-Chip Stocks": 0.0,
        "Small-Cap Stocks": 0.0,
        "Mid-Cap Stocks": 0.0,
        "Large-Cap Stocks": 0.0,
        "Speculative / Penny Stocks": 0.0,
    }
    reasons = {name: [] for name in scores}

    if sector in {"Technology", "Communication Services"}:
        scores["Growth Stocks"] += 1
        reasons["Growth Stocks"].append("innovation-heavy sector")
    if has_numeric_value(revenue_growth) and revenue_growth >= 0.12:
        scores["Growth Stocks"] += 2
        reasons["Growth Stocks"].append("double-digit revenue growth")
    if has_numeric_value(earnings_growth) and earnings_growth >= 0.12:
        scores["Growth Stocks"] += 2
        reasons["Growth Stocks"].append("double-digit earnings growth")
    if has_numeric_value(momentum_1y) and momentum_1y >= 0.20:
        scores["Growth Stocks"] += 1
        reasons["Growth Stocks"].append("strong long-term momentum")
    if (
        (has_numeric_value(pe) and has_numeric_value(bench.get("PE")) and pe >= bench["PE"] * 1.10)
        or (has_numeric_value(ps_ratio) and has_numeric_value(bench.get("PS")) and ps_ratio >= bench["PS"] * 1.10)
        or (has_numeric_value(peg_ratio) and peg_ratio >= 1.2)
    ):
        scores["Growth Stocks"] += 1
        reasons["Growth Stocks"].append("premium valuation profile")
    if not has_numeric_value(dividend_yield) or dividend_yield < 0.015:
        scores["Growth Stocks"] += 1
        reasons["Growth Stocks"].append("low payout orientation")

    if v_val == "UNDERVALUED":
        scores["Value Stocks"] += 2
        reasons["Value Stocks"].append("undervalued by the current valuation engine")
    if has_numeric_value(pe) and has_numeric_value(bench.get("PE")) and pe <= bench["PE"] * 0.85:
        scores["Value Stocks"] += 1
        reasons["Value Stocks"].append("discounted earnings multiple")
    if has_numeric_value(pb) and has_numeric_value(bench.get("PB")) and pb <= bench["PB"] * 0.85:
        scores["Value Stocks"] += 1
        reasons["Value Stocks"].append("discounted price-to-book")
    if has_numeric_value(ps_ratio) and has_numeric_value(bench.get("PS")) and ps_ratio <= bench["PS"] * 0.85:
        scores["Value Stocks"] += 1
        reasons["Value Stocks"].append("discounted price-to-sales")
    if f_score >= 1:
        scores["Value Stocks"] += 1
        reasons["Value Stocks"].append("fundamentals are at least stable")

    if has_numeric_value(dividend_yield) and dividend_yield >= 0.03:
        scores["Dividend / Income Stocks"] += 2
        reasons["Dividend / Income Stocks"].append("meaningful dividend yield")
    if has_numeric_value(dividend_yield) and dividend_yield >= 0.05:
        scores["Dividend / Income Stocks"] += 1
        reasons["Dividend / Income Stocks"].append("high income orientation")
    if has_numeric_value(payout_ratio) and 0 < payout_ratio <= 0.80:
        scores["Dividend / Income Stocks"] += 1
        reasons["Dividend / Income Stocks"].append("payout looks more sustainable")
    if sector in INCOME_SECTORS:
        scores["Dividend / Income Stocks"] += 1
        reasons["Dividend / Income Stocks"].append("income-heavy sector")
    if has_numeric_value(equity_beta) and equity_beta < 1.0:
        scores["Dividend / Income Stocks"] += 1
        reasons["Dividend / Income Stocks"].append("below-market beta")

    if sector in CYCLICAL_SECTORS:
        scores["Cyclical Stocks"] += 2
        reasons["Cyclical Stocks"].append("economically sensitive sector")
    if has_numeric_value(equity_beta) and equity_beta > 1.10:
        scores["Cyclical Stocks"] += 1
        reasons["Cyclical Stocks"].append("higher beta profile")
    if has_numeric_value(momentum_1y) and abs(momentum_1y) >= 0.20:
        scores["Cyclical Stocks"] += 1
        reasons["Cyclical Stocks"].append("large cycle-like price swings")

    if sector in DEFENSIVE_SECTORS:
        scores["Defensive Stocks"] += 2
        reasons["Defensive Stocks"].append("defensive sector exposure")
    if has_numeric_value(equity_beta) and equity_beta < 1.0:
        scores["Defensive Stocks"] += 1
        reasons["Defensive Stocks"].append("lower beta profile")
    if has_numeric_value(margins) and margins > 0:
        scores["Defensive Stocks"] += 1
        reasons["Defensive Stocks"].append("positive margins")
    if f_score >= 1:
        scores["Defensive Stocks"] += 1
        reasons["Defensive Stocks"].append("stable operating fundamentals")

    if cap_bucket == "Large-Cap":
        scores["Large-Cap Stocks"] += 3
        reasons["Large-Cap Stocks"].append("market cap above $10B")
    elif cap_bucket == "Mid-Cap":
        scores["Mid-Cap Stocks"] += 3
        reasons["Mid-Cap Stocks"].append("market cap between $2B and $10B")
    elif cap_bucket == "Small-Cap":
        scores["Small-Cap Stocks"] += 3
        reasons["Small-Cap Stocks"].append("market cap below $2B")

    if has_numeric_value(market_cap) and market_cap >= 100_000_000_000:
        scores["Blue-Chip Stocks"] += 2
        reasons["Blue-Chip Stocks"].append("very large market capitalization")
    if has_numeric_value(market_cap) and market_cap >= 300_000_000_000:
        scores["Blue-Chip Stocks"] += 1
        reasons["Blue-Chip Stocks"].append("mega-cap scale")
    if sector in QUALITY_SECTORS:
        scores["Blue-Chip Stocks"] += 1
        reasons["Blue-Chip Stocks"].append("quality-heavy sector")
    if f_score >= 2:
        scores["Blue-Chip Stocks"] += 1
        reasons["Blue-Chip Stocks"].append("solid fundamental footing")
    if has_numeric_value(analyst_opinions) and analyst_opinions >= 10:
        scores["Blue-Chip Stocks"] += 1
        reasons["Blue-Chip Stocks"].append("broad analyst coverage")
    if has_numeric_value(roe) and roe > 0 and has_numeric_value(current_ratio) and current_ratio >= 1.0:
        scores["Blue-Chip Stocks"] += 1
        reasons["Blue-Chip Stocks"].append("quality balance-sheet profile")

    if has_numeric_value(price) and price < 5:
        scores["Speculative / Penny Stocks"] += 2
        reasons["Speculative / Penny Stocks"].append("low share price")
    if has_numeric_value(market_cap) and market_cap < 500_000_000:
        scores["Speculative / Penny Stocks"] += 2
        reasons["Speculative / Penny Stocks"].append("micro-cap scale")
    elif cap_bucket == "Small-Cap":
        scores["Speculative / Penny Stocks"] += 1
        reasons["Speculative / Penny Stocks"].append("small-cap risk profile")
    if f_score <= 0:
        scores["Speculative / Penny Stocks"] += 1
        reasons["Speculative / Penny Stocks"].append("weak or unproven fundamentals")
    if has_numeric_value(equity_beta) and equity_beta >= 1.8:
        scores["Speculative / Penny Stocks"] += 1
        reasons["Speculative / Penny Stocks"].append("very high beta")
    if has_numeric_value(analyst_opinions) and analyst_opinions < 3:
        scores["Speculative / Penny Stocks"] += 1
        reasons["Speculative / Penny Stocks"].append("thin coverage")
    if has_numeric_value(debt_eq) and debt_eq > 250:
        scores["Speculative / Penny Stocks"] += 1
        reasons["Speculative / Penny Stocks"].append("high leverage")

    style_tags = []
    for tag_name in [
        "Growth Stocks",
        "Value Stocks",
        "Dividend / Income Stocks",
        "Cyclical Stocks",
        "Defensive Stocks",
        "Blue-Chip Stocks",
    ]:
        if scores[tag_name] >= 3:
            style_tags.append(tag_name.replace(" Stocks", ""))
    if cap_bucket != "Unknown":
        style_tags.append(cap_bucket)

    top_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_type, top_score = top_scores[0]
    second_score = top_scores[1][1] if len(top_scores) > 1 else 0

    if scores["Speculative / Penny Stocks"] >= 4:
        primary_type = "Speculative / Penny Stocks"
    elif scores["Growth Stocks"] >= 5 and scores["Growth Stocks"] >= scores["Value Stocks"] + 1:
        primary_type = "Growth Stocks"
    elif scores["Value Stocks"] >= 5 and scores["Value Stocks"] >= scores["Growth Stocks"]:
        primary_type = "Value Stocks"
    elif scores["Dividend / Income Stocks"] >= 4 and scores["Dividend / Income Stocks"] >= scores["Defensive Stocks"]:
        primary_type = "Dividend / Income Stocks"
    elif scores["Defensive Stocks"] >= 4 and scores["Defensive Stocks"] >= scores["Cyclical Stocks"]:
        primary_type = "Defensive Stocks"
    elif scores["Cyclical Stocks"] >= 4:
        primary_type = "Cyclical Stocks"
    elif scores["Blue-Chip Stocks"] >= 5 and cap_bucket == "Large-Cap":
        primary_type = "Blue-Chip Stocks"
    elif top_score >= 4:
        primary_type = top_type
    elif cap_bucket == "Small-Cap":
        primary_type = "Small-Cap Stocks"
    elif cap_bucket == "Mid-Cap":
        primary_type = "Mid-Cap Stocks"
    else:
        primary_type = "Large-Cap Stocks" if cap_bucket == "Large-Cap" else "Blue-Chip Stocks"

    confidence = float(np.clip(45 + top_score * 8 + (top_score - second_score) * 6, 35, 95))
    top_reasons = reasons.get(primary_type, [])
    summary = ", ".join(top_reasons[:3]) if top_reasons else "mixed factor profile"
    return {
        "primary_type": primary_type,
        "cap_bucket": cap_bucket,
        "style_tags": " | ".join(style_tags) if style_tags else primary_type.replace(" Stocks", ""),
        "type_strategy": build_stock_type_strategy(primary_type),
        "type_confidence": confidence,
        "classification_summary": summary,
        "market_cap": market_cap,
        "dividend_yield": dividend_yield,
        "payout_ratio": payout_ratio,
        "equity_beta": equity_beta,
    }


def extract_stock_profile_from_saved_row(saved_row):
    if saved_row is None:
        return {}
    return {
        "primary_type": saved_row.get("Stock_Type", "Unknown"),
        "cap_bucket": saved_row.get("Cap_Bucket", "Unknown"),
        "style_tags": saved_row.get("Style_Tags", ""),
        "type_strategy": saved_row.get("Type_Strategy", ""),
        "type_confidence": safe_num(saved_row.get("Type_Confidence")),
        "classification_summary": "",
        "market_cap": safe_num(saved_row.get("Market_Cap")),
        "dividend_yield": safe_num(saved_row.get("Dividend_Yield")),
        "payout_ratio": safe_num(saved_row.get("Payout_Ratio")),
        "equity_beta": safe_num(saved_row.get("Equity_Beta")),
    }


def infer_stock_profile_from_snapshot(info, hist, settings=None, db=None, ticker=None):
    active_settings = get_model_settings() if settings is None else settings
    info = info or {}
    if hist is None or hist.empty or "Close" not in hist.columns:
        return {}

    close = hist["Close"].dropna().astype(float)
    if close.empty:
        return {}
    price = float(close.iloc[-1])
    momentum_1y = (price / close.iloc[0] - 1) if len(close) > 1 else None
    sector = info.get("sector", "Unknown")
    bench, _ = build_relative_peer_benchmarks(
        ticker or info.get("symbol") or "",
        info,
        db=db,
        settings=active_settings,
    )

    roe = safe_num(info.get("returnOnEquity"))
    margins = safe_num(info.get("profitMargins"))
    debt_eq = safe_num(info.get("debtToEquity"))
    revenue_growth = safe_num(info.get("revenueGrowth"))
    earnings_growth = safe_num(info.get("earningsGrowth"))
    current_ratio = safe_num(info.get("currentRatio"))

    f_score = 0
    if has_numeric_value(roe):
        f_score += 1 if roe >= active_settings["fund_roe_threshold"] else (-1 if roe < 0 else 0)
    if has_numeric_value(margins):
        f_score += 1 if margins >= active_settings["fund_profit_margin_threshold"] else (-1 if margins < 0 else 0)
    if has_numeric_value(debt_eq):
        f_score += 1 if 0 <= debt_eq < active_settings["fund_debt_good_threshold"] else (-1 if debt_eq > active_settings["fund_debt_bad_threshold"] else 0)
    if has_numeric_value(revenue_growth):
        f_score += 1 if revenue_growth >= active_settings["fund_revenue_growth_threshold"] else (-1 if revenue_growth < 0 else 0)
    if has_numeric_value(earnings_growth):
        f_score += 1 if earnings_growth >= active_settings["fund_revenue_growth_threshold"] else (-1 if earnings_growth < 0 else 0)
    if has_numeric_value(current_ratio):
        f_score += 1 if current_ratio >= active_settings["fund_current_ratio_good"] else (-1 if current_ratio < active_settings["fund_current_ratio_bad"] else 0)

    pe = safe_num(info.get("trailingPE"))
    forward_pe = safe_num(info.get("forwardPE"))
    peg_ratio = safe_num(info.get("pegRatio"))
    ps_ratio = safe_num(info.get("priceToSalesTrailing12Months"))
    pb = safe_num(info.get("priceToBook"))
    ev_ebitda = safe_num(info.get("enterpriseToEbitda"))
    v_score = 0
    valuation_signal_count = 0
    for metric_value, benchmark_value in [
        (pe, bench["PE"]),
        (forward_pe, bench["PE"]),
        (ps_ratio, bench["PS"]),
        (pb, bench["PB"]),
        (ev_ebitda, bench["EV_EBITDA"]),
    ]:
        multiple_score = score_relative_multiple(metric_value, benchmark_value)
        v_score += multiple_score
        if has_numeric_value(metric_value):
            valuation_signal_count += 1
    if has_numeric_value(peg_ratio):
        valuation_signal_count += 1
        if peg_ratio <= active_settings["valuation_peg_threshold"] * 0.9:
            v_score += 1
        elif peg_ratio >= active_settings["valuation_peg_threshold"] * 1.35:
            v_score -= 1

    if valuation_signal_count < 2 and v_score < active_settings["valuation_fair_score_threshold"]:
        v_val = "FAIR VALUE"
    elif v_score >= active_settings["valuation_under_score_threshold"]:
        v_val = "UNDERVALUED"
    elif v_score >= active_settings["valuation_fair_score_threshold"]:
        v_val = "FAIR VALUE"
    else:
        v_val = "OVERVALUED"

    return classify_stock_profile(
        sector=sector,
        price=price,
        market_cap=safe_num(info.get("marketCap")),
        dividend_yield=safe_num(info.get("dividendYield")),
        payout_ratio=safe_num(info.get("payoutRatio")),
        equity_beta=safe_num(info.get("beta")),
        analyst_opinions=safe_num(info.get("numberOfAnalystOpinions")),
        pe=pe,
        forward_pe=forward_pe,
        peg_ratio=peg_ratio,
        ps_ratio=ps_ratio,
        pb=pb,
        bench=bench,
        f_score=f_score,
        v_val=v_val,
        revenue_growth=revenue_growth,
        earnings_growth=earnings_growth,
        margins=margins,
        roe=roe,
        current_ratio=current_ratio,
        debt_eq=debt_eq,
        momentum_1y=momentum_1y,
    )


def apply_stock_type_framework(
    *,
    stock_profile,
    overall_score,
    tech_score,
    f_score,
    v_score,
    sentiment_score,
    v_fund,
    v_val,
    regime,
    bullish_trend,
    bearish_trend,
    data_quality,
    momentum_1y,
    settings,
):
    primary_type = stock_profile.get("primary_type", "")
    cap_bucket = stock_profile.get("cap_bucket", "Unknown")
    dividend_yield = stock_profile.get("dividend_yield")
    payout_ratio = stock_profile.get("payout_ratio")
    adjusted_score = overall_score
    local_settings = dict(settings)
    notes = []
    cap_verdict_to = None

    if primary_type == "Growth Stocks":
        local_settings["decision_hold_buffer"] = max(0.0, settings["decision_hold_buffer"] - 0.5)
        local_settings["overall_buy_threshold"] = max(1.0, settings["overall_buy_threshold"] - 1.0)
        if bullish_trend and (f_score >= 1 or sentiment_score >= 1):
            adjusted_score += 1.0
            notes.append("Growth profile rewards durable trend strength")
        if v_val == "OVERVALUED" and bullish_trend and has_numeric_value(momentum_1y) and momentum_1y > 0.25:
            adjusted_score += 0.75
            notes.append("Premium valuation is tolerated more for proven compounders")
        if bearish_trend and sentiment_score <= 0:
            adjusted_score -= 1.0
            notes.append("Growth setups lose conviction faster in downtrends")

    elif primary_type == "Value Stocks":
        local_settings["decision_hold_buffer"] = settings["decision_hold_buffer"] + 0.5
        if v_val == "UNDERVALUED" and f_score >= 1:
            adjusted_score += 1.25
            notes.append("Value profile leans harder on valuation plus stable fundamentals")
        if bearish_trend and tech_score <= -2:
            adjusted_score -= 0.75
            notes.append("Deep value still waits for price damage to stabilize")
        if sentiment_score <= -2:
            adjusted_score -= 0.5

    elif primary_type == "Dividend / Income Stocks":
        local_settings["overall_sell_threshold"] = settings["overall_sell_threshold"] - 1.0
        local_settings["overall_strong_sell_threshold"] = settings["overall_strong_sell_threshold"] - 1.0
        if has_numeric_value(dividend_yield) and dividend_yield >= 0.03:
            adjusted_score += 0.75
            notes.append("Income profile rewards durable shareholder yield")
        if has_numeric_value(payout_ratio) and payout_ratio > 0.95:
            adjusted_score -= 1.0
            notes.append("An overstretched payout ratio weakens the income case")
        if regime == "Range-bound" and f_score >= 1 and v_val != "OVERVALUED":
            adjusted_score += 0.5

    elif primary_type == "Cyclical Stocks":
        local_settings["decision_hold_buffer"] = settings["decision_hold_buffer"] + 1.0
        if regime == "Bullish Trend" and tech_score >= 2 and sentiment_score >= 0:
            adjusted_score += 1.25
            notes.append("Cyclicals want momentum and regime confirmation")
        if regime in {"Transition", "Range-bound"}:
            adjusted_score -= 0.75
            notes.append("Cycle timing risk pushes mixed setups toward hold")
        if bearish_trend or (has_numeric_value(momentum_1y) and momentum_1y < 0):
            adjusted_score -= 1.0

    elif primary_type == "Defensive Stocks":
        local_settings["overall_sell_threshold"] = settings["overall_sell_threshold"] - 1.0
        local_settings["overall_strong_sell_threshold"] = settings["overall_strong_sell_threshold"] - 1.0
        if f_score >= 1 and v_val != "OVERVALUED":
            adjusted_score += 0.75
            notes.append("Defensives get more credit for steady fundamentals")
        if regime in {"Range-bound", "Transition"} and f_score >= 1:
            adjusted_score += 0.5
        if bearish_trend and f_score <= 0:
            adjusted_score -= 0.5

    elif primary_type == "Blue-Chip Stocks":
        local_settings["decision_hold_buffer"] = max(0.0, settings["decision_hold_buffer"] - 0.5)
        if f_score >= 1 and sentiment_score >= 0:
            adjusted_score += 0.75
            notes.append("Blue-chip profile gives more room to durable quality")
        if v_val == "OVERVALUED" and bullish_trend and f_score >= 2:
            adjusted_score += 0.5
        if bearish_trend and f_score <= 0:
            adjusted_score -= 0.75

    elif primary_type == "Small-Cap Stocks":
        local_settings["decision_hold_buffer"] = settings["decision_hold_buffer"] + 0.5
        local_settings["overall_buy_threshold"] = settings["overall_buy_threshold"] + 1.0
        if tech_score >= 3 and sentiment_score >= 1 and bullish_trend:
            adjusted_score += 0.75
            notes.append("Small caps need stronger confirmation before leaning in")
        if data_quality == "Low" or bearish_trend:
            adjusted_score -= 1.25

    elif primary_type == "Mid-Cap Stocks":
        if f_score >= 1 and bullish_trend:
            adjusted_score += 0.5
            notes.append("Mid-cap profile likes balanced trend plus fundamentals")
        if bearish_trend and f_score <= 0:
            adjusted_score -= 0.75

    elif primary_type == "Large-Cap Stocks":
        if f_score >= 1:
            adjusted_score += 0.5
            notes.append("Large caps get a modest durability benefit")
        if bearish_trend and f_score <= 0:
            adjusted_score -= 0.5

    elif primary_type == "Speculative / Penny Stocks":
        local_settings["decision_hold_buffer"] = settings["decision_hold_buffer"] + 1.0
        local_settings["overall_buy_threshold"] = settings["overall_buy_threshold"] + 1.0
        local_settings["overall_strong_buy_threshold"] = settings["overall_strong_buy_threshold"] + 2.0
        cap_verdict_to = "BUY"
        if tech_score >= 4 and sentiment_score >= 2 and bullish_trend:
            adjusted_score += 0.5
            notes.append("Speculative profile only rewards strong confirmation")
        else:
            adjusted_score -= 1.0
        if data_quality == "Low" or f_score <= 0:
            adjusted_score -= 1.0
            notes.append("Weak data or fundamentals are penalized harder")

    if cap_bucket == "Small-Cap" and primary_type != "Speculative / Penny Stocks":
        adjusted_score -= 0.25
    elif cap_bucket == "Large-Cap" and primary_type not in {"Blue-Chip Stocks", "Large-Cap Stocks"} and f_score >= 1:
        adjusted_score += 0.25

    verdict = resolve_overall_verdict(
        overall_score=adjusted_score,
        tech_score=tech_score,
        f_score=f_score,
        v_score=v_score,
        sentiment_score=sentiment_score,
        v_fund=v_fund,
        v_val=v_val,
        regime=regime,
        bullish_trend=bullish_trend,
        bearish_trend=bearish_trend,
        settings=local_settings,
    )

    if cap_verdict_to == "BUY" and verdict == "STRONG BUY":
        verdict = "BUY"

    return adjusted_score, verdict, local_settings, notes


def adjust_type_based_confidence(confidence, stock_profile, data_quality):
    primary_type = stock_profile.get("primary_type", "")
    adjusted_confidence = float(confidence)
    if primary_type in {"Blue-Chip Stocks", "Defensive Stocks"} and data_quality != "Low":
        adjusted_confidence += 4
    elif primary_type in {"Small-Cap Stocks", "Speculative / Penny Stocks"}:
        adjusted_confidence -= 6 if primary_type == "Speculative / Penny Stocks" else 3
    return float(np.clip(round(adjusted_confidence, 1), 5.0, 95.0))


def parse_ticker_list(raw_text):
    tickers = []
    seen = set()
    normalized = raw_text.replace("\n", ",").replace(" ", ",")
    for token in normalized.split(","):
        ticker = token.strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        tickers.append(ticker)
    return tickers


def cap_weights(weights, max_weight):
    capped = weights / weights.sum()
    if max_weight >= 1:
        return capped

    for _ in range(10):
        over_limit = capped > max_weight
        if not over_limit.any():
            break

        excess = (capped[over_limit] - max_weight).sum()
        capped[over_limit] = max_weight
        under_limit = capped < max_weight
        if not under_limit.any():
            break
        capped[under_limit] += excess * (capped[under_limit] / capped[under_limit].sum())
        capped = capped / capped.sum()

    return capped / capped.sum()


def score_to_signal(score, strong_buy=4, buy=2, sell=-2, strong_sell=-4):
    if score >= strong_buy:
        return "STRONG BUY"
    if score >= buy:
        return "BUY"
    if score <= strong_sell:
        return "STRONG SELL"
    if score <= sell:
        return "SELL"
    return "HOLD"


def score_to_sentiment(score):
    if score >= 3:
        return "POSITIVE"
    if score <= -3:
        return "NEGATIVE"
    return "MIXED"


def score_trend_distance(value, baseline, tolerance=0.02):
    if not has_numeric_value(value) or not has_numeric_value(baseline) or baseline <= 0:
        return 0
    if value >= baseline * (1 + tolerance):
        return 1
    if value <= baseline * (1 - tolerance):
        return -1
    return 0


def step_signal_toward_neutral(signal):
    transitions = {
        "STRONG BUY": "BUY",
        "BUY": "HOLD",
        "HOLD": "HOLD",
        "SELL": "HOLD",
        "STRONG SELL": "SELL",
    }
    return transitions.get(signal, "HOLD")


def has_bullish_trend(price, sma50, sma200, momentum_1y=None):
    if not has_numeric_value(price) or not has_numeric_value(sma200):
        return False
    long_term_ok = (
        has_numeric_value(sma50) and sma50 >= sma200
    ) or (
        has_numeric_value(momentum_1y) and momentum_1y >= 0
    )
    return price >= sma200 and long_term_ok


def has_bearish_trend(price, sma50, sma200, momentum_1y=None):
    if not has_numeric_value(price) or not has_numeric_value(sma200):
        return False
    long_term_weak = (
        has_numeric_value(sma50) and sma50 <= sma200
    ) or (
        has_numeric_value(momentum_1y) and momentum_1y <= 0
    )
    return price <= sma200 and long_term_weak


def classify_market_regime(price, sma50, sma200, momentum_1y=None, tolerance=0.02):
    if not has_numeric_value(price) or not has_numeric_value(sma200):
        return "Unclear"
    if has_bullish_trend(price, sma50, sma200, momentum_1y) and price >= sma200 * (1 + tolerance):
        return "Bullish Trend"
    if has_bearish_trend(price, sma50, sma200, momentum_1y) and price <= sma200 * (1 - tolerance):
        return "Bearish Trend"
    if abs((price - sma200) / sma200) <= tolerance and (momentum_1y is None or abs(momentum_1y) <= 0.10):
        return "Range-bound"
    return "Transition"


def summarize_engine_biases(tech_score, f_score, v_score, sentiment_score, v_val, bullish_trend, bearish_trend):
    tech_bullish = tech_score >= 3 or (tech_score >= 2 and bullish_trend)
    tech_bearish = tech_score <= -4 or (tech_score <= -3 and bearish_trend)
    fund_bullish = f_score >= 2
    fund_bearish = f_score <= -2
    valuation_bullish = v_val == "UNDERVALUED"
    valuation_bearish = v_val == "OVERVALUED" and v_score <= -2
    sentiment_bullish = sentiment_score >= 2
    sentiment_bearish = sentiment_score <= -2
    bullish_count = sum([tech_bullish, fund_bullish, valuation_bullish, sentiment_bullish])
    bearish_count = sum([tech_bearish, fund_bearish, valuation_bearish, sentiment_bearish])
    return {
        "tech_bullish": tech_bullish,
        "tech_bearish": tech_bearish,
        "fund_bullish": fund_bullish,
        "fund_bearish": fund_bearish,
        "valuation_bullish": valuation_bullish,
        "valuation_bearish": valuation_bearish,
        "sentiment_bullish": sentiment_bullish,
        "sentiment_bearish": sentiment_bearish,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "mixed": bullish_count >= 2 and bearish_count >= 2,
    }


def compute_decision_confidence(overall_score, bias_summary, regime, completeness):
    dominant_count = max(bias_summary["bullish_count"], bias_summary["bearish_count"])
    opposing_count = min(bias_summary["bullish_count"], bias_summary["bearish_count"])
    confidence = 28 + dominant_count * 14 - opposing_count * 10 + min(abs(overall_score) * 2.0, 18)

    if regime in {"Bullish Trend", "Bearish Trend"}:
        confidence += 8
    elif regime == "Transition":
        confidence -= 6
    elif regime == "Range-bound":
        confidence -= 10

    if completeness is not None:
        confidence += (float(completeness) - 0.5) * 35

    return float(np.clip(round(confidence, 1), 5.0, 95.0))


def apply_confidence_guard(verdict, confidence, data_quality, settings):
    guarded = verdict
    strong_floor = settings["decision_min_confidence"] + 10
    base_floor = settings["decision_min_confidence"]

    if guarded in {"STRONG BUY", "STRONG SELL"} and confidence < strong_floor:
        guarded = step_signal_toward_neutral(guarded)
    if guarded in {"BUY", "SELL"} and confidence < base_floor:
        guarded = "HOLD"
    if data_quality == "Low" and guarded in {"BUY", "SELL", "STRONG BUY", "STRONG SELL"}:
        guarded = "HOLD"
    return guarded


def build_decision_notes(
    verdict,
    regime,
    bias_summary,
    confidence,
    data_quality,
    current_rsi,
    v_val,
    v_fund,
    bullish_trend,
    bearish_trend,
    overextended,
    pullback_recovery,
):
    notes = [f"Regime: {regime}"]
    if bias_summary["mixed"] and verdict == "HOLD":
        notes.append("Conflicting engine signals pushed the model toward hold")
    elif verdict in {"BUY", "STRONG BUY"} and bullish_trend:
        notes.append("Trend and engine alignment lean constructive")
    elif verdict in {"SELL", "STRONG SELL"} and bearish_trend:
        notes.append("Trend and engine alignment lean defensive")

    if overextended and verdict in {"BUY", "HOLD"}:
        notes.append("Price looks stretched above the short-term trend")
    if pullback_recovery:
        notes.append("RSI has recovered from oversold conditions")
    if v_val == "UNDERVALUED":
        notes.append("Valuation provides some downside support")
    elif v_val == "OVERVALUED":
        notes.append("Valuation still looks stretched")

    if v_fund == "STRONG":
        notes.append("Fundamentals remain supportive")
    elif v_fund == "WEAK":
        notes.append("Fundamentals are a headwind")

    if has_numeric_value(current_rsi) and current_rsi >= 70 and verdict in {"BUY", "STRONG BUY", "HOLD"}:
        notes.append("Momentum is hot enough to warrant patience on entries")
    if data_quality == "Low":
        notes.append("Low data completeness reduced conviction")
    elif confidence < 60:
        notes.append("Consistency is moderate, so the model stayed cautious")

    deduped = []
    for note in notes:
        if note not in deduped:
            deduped.append(note)
    return " | ".join(deduped[:4])


def resolve_overall_verdict(
    overall_score,
    tech_score,
    f_score,
    v_score,
    sentiment_score,
    v_fund,
    v_val,
    regime,
    bullish_trend,
    bearish_trend,
    settings,
):
    bias_summary = summarize_engine_biases(
        tech_score,
        f_score,
        v_score,
        sentiment_score,
        v_val,
        bullish_trend,
        bearish_trend,
    )
    buy_threshold = settings["overall_buy_threshold"]
    strong_buy_threshold = settings["overall_strong_buy_threshold"]
    sell_threshold = settings["overall_sell_threshold"]
    strong_sell_threshold = settings["overall_strong_sell_threshold"]
    if bias_summary["mixed"] or regime in {"Range-bound", "Transition"}:
        hold_buffer = settings["decision_hold_buffer"]
        buy_threshold += hold_buffer
        strong_buy_threshold += hold_buffer
        sell_threshold -= hold_buffer
        strong_sell_threshold -= hold_buffer

    if v_val == "UNDERVALUED" and v_fund == "STRONG" and sentiment_score >= 2:
        verdict = "STRONG BUY" if tech_score >= 2 and not bearish_trend else "BUY"
    elif v_val == "OVERVALUED" and sentiment_score <= -2 and bearish_trend and f_score <= 0:
        verdict = "STRONG SELL" if tech_score <= -3 else "SELL"
    else:
        verdict = score_to_signal(
            overall_score,
            strong_buy=strong_buy_threshold,
            buy=buy_threshold,
            sell=sell_threshold,
            strong_sell=strong_sell_threshold,
        )

    if bias_summary["mixed"]:
        return "HOLD"

    if verdict in {"BUY", "STRONG BUY"}:
        if bias_summary["bearish_count"] >= 2:
            verdict = "HOLD"
        elif bearish_trend and f_score <= 0 and sentiment_score <= 0:
            verdict = "HOLD"
        elif v_val == "OVERVALUED" and f_score < 2:
            verdict = step_signal_toward_neutral(verdict)

    if verdict in {"SELL", "STRONG SELL"}:
        if bias_summary["bullish_count"] >= 2:
            verdict = "HOLD"
        elif bullish_trend and f_score >= 1:
            verdict = "HOLD"
        elif v_val == "UNDERVALUED" and f_score >= 0 and sentiment_score > -3:
            verdict = step_signal_toward_neutral(verdict)

    if verdict == "STRONG BUY" and (bias_summary["bearish_count"] > 0 or bearish_trend):
        verdict = "BUY"
    if verdict == "STRONG SELL" and (bias_summary["bullish_count"] > 0 or bullish_trend):
        verdict = "SELL"

    if verdict == "HOLD":
        if (
            bias_summary["bullish_count"] >= 3
            and bias_summary["bearish_count"] == 0
            and regime == "Bullish Trend"
            and overall_score >= buy_threshold - 1
        ):
            verdict = "BUY"
        elif (
            bias_summary["bearish_count"] >= 3
            and bias_summary["bullish_count"] == 0
            and regime == "Bearish Trend"
            and overall_score <= sell_threshold + 1
        ):
            verdict = "SELL"

    return verdict


def derive_backtest_positions(analysis, settings=None, stock_profile=None):
    active_settings = get_model_settings() if settings is None else settings
    if analysis is None or analysis.empty:
        return pd.Series(dtype=float)

    primary_type = (stock_profile or {}).get("primary_type", "")
    trend_tolerance = active_settings["tech_trend_tolerance"]
    cooldown_days = int(round(active_settings["backtest_cooldown_days"]))
    full_target = 1.0
    core_target = 0.5
    danger_floor = 0.25
    core_reentry_cooldown = max(1, cooldown_days // 2)
    full_reentry_cooldown = cooldown_days
    entry_score_floor = 3
    hard_exit_score_floor = -5
    exit_break_multiplier = 1.5
    trailing_stop_threshold = -0.16
    allow_core_outside_bullish = False
    trim_to_core_on_non_bullish = False
    long_term_momentum_floor = max(active_settings["tech_momentum_threshold"] * 2, 0.08)
    initial_floor_if_not_bearish = 0.0
    growth_compounder_types = {"Growth Stocks"}
    overweight_target = None
    disable_bearish_reduction = False
    disable_exit = False
    ignore_initial_bearish_gate = False
    danger_score_floor = -2
    danger_momentum_multiplier = 1.0
    danger_requires_price_below_sma50 = False
    exit_requires_sma50_cross = False
    exit_momentum_floor = -long_term_momentum_floor

    if primary_type in growth_compounder_types:
        # Let leading compounders stay fully involved and briefly overweight the
        # strongest trend/momentum combinations so the replay can keep pace with
        # benchmark winners instead of constantly lagging them.
        core_target = 1.0
        full_target = 1.35
        danger_floor = 0.75
        allow_core_outside_bullish = True
        entry_score_floor = 1
        hard_exit_score_floor = -7
        exit_break_multiplier = 2.5
        trailing_stop_threshold = -0.30
        initial_floor_if_not_bearish = 1.0
    elif primary_type in {"Blue-Chip Stocks", "Large-Cap Stocks"}:
        # Treat established leaders more like core holdings: keep them invested,
        # allow a modest tactical add, and stop trading them out on ordinary
        # weakness where benchmark lag tends to come from.
        core_target = 1.0
        full_target = 1.15
        danger_floor = 1.15
        allow_core_outside_bullish = True
        entry_score_floor = 0
        initial_floor_if_not_bearish = 1.0
        disable_bearish_reduction = True
        disable_exit = True
        ignore_initial_bearish_gate = True
    elif primary_type == "Value Stocks":
        core_target = 1.0
        full_target = 1.0
        danger_floor = 1.0
        allow_core_outside_bullish = True
        entry_score_floor = 0
        initial_floor_if_not_bearish = 1.0
        disable_bearish_reduction = True
        disable_exit = True
        ignore_initial_bearish_gate = True
    elif primary_type in {"Dividend / Income Stocks", "Defensive Stocks"}:
        core_target = 1.0
        full_target = 1.0
        danger_floor = 1.0
        allow_core_outside_bullish = True
        entry_score_floor = 0
        initial_floor_if_not_bearish = 1.0
        disable_bearish_reduction = True
        disable_exit = True
        ignore_initial_bearish_gate = True
    elif primary_type == "Cyclical Stocks":
        core_target = 0.5
        danger_floor = 0.0
        hard_exit_score_floor = -4
        exit_break_multiplier = 1.2
        trailing_stop_threshold = -0.12
        trim_to_core_on_non_bullish = True
    elif primary_type == "Mid-Cap Stocks":
        core_target = 0.5
        danger_floor = 0.25
        exit_break_multiplier = 1.4
        trailing_stop_threshold = -0.14
    elif primary_type == "Small-Cap Stocks":
        core_target = 0.25
        full_target = 0.75
        danger_floor = 0.0
        entry_score_floor = 4
        hard_exit_score_floor = -4
        exit_break_multiplier = 1.2
        trailing_stop_threshold = -0.10
        trim_to_core_on_non_bullish = True
    elif primary_type == "Speculative / Penny Stocks":
        core_target = 0.0
        full_target = 0.5
        danger_floor = 0.0
        entry_score_floor = 4
        hard_exit_score_floor = -4
        exit_break_multiplier = 1.0
        trailing_stop_threshold = -0.08
        trim_to_core_on_non_bullish = True

    bullish_regime = (
        analysis["Close"].ge(analysis["SMA_200"] * (1 + trend_tolerance))
        & (
            analysis["SMA_50"].ge(analysis["SMA_200"] * (1 + trend_tolerance / 2))
            | analysis["Momentum_1Y"].gt(0)
        )
    )
    bearish_regime = (
        analysis["Close"].le(analysis["SMA_200"] * (1 - trend_tolerance))
        & (
            analysis["SMA_50"].le(analysis["SMA_200"] * (1 - trend_tolerance / 2))
            | analysis["Momentum_1Y"].lt(0)
        )
    )
    bullish_regime = bullish_regime.fillna(False)
    bearish_regime = bearish_regime.fillna(False)
    macd_bearish = analysis["MACD"].lt(analysis["MACD_Signal_Line"]).fillna(False)
    macd_bullish = analysis["MACD"].ge(analysis["MACD_Signal_Line"]).fillna(False)
    core_regime = bullish_regime | (~bearish_regime if allow_core_outside_bullish else False)
    trailing_stop_breach = analysis.get("Trailing_Drawdown_Quarter", pd.Series(index=analysis.index, dtype=float)).le(
        trailing_stop_threshold
    ).fillna(False)
    strong_bullish = (
        bullish_regime
        & macd_bullish
        & analysis["Tech Score"].ge(4)
        & analysis["Momentum_1Y"].gt(0.15).fillna(False)
        & analysis["Close"].ge(analysis["SMA_50"]).fillna(False)
    )
    entry_signal = core_regime & analysis["Tech Score"].ge(entry_score_floor)
    add_signal = bullish_regime & analysis["Tech Score"].ge(max(entry_score_floor, 2))
    danger_reduce = pd.Series(False, index=analysis.index, dtype=bool)
    if not disable_bearish_reduction:
        danger_reduce = (
            (bearish_regime | trailing_stop_breach)
            & (
                analysis["Tech Score"].le(danger_score_floor)
                | (
                    macd_bearish
                    & analysis["Momentum_1M"].lt(
                        -active_settings["tech_momentum_threshold"] * danger_momentum_multiplier
                    ).fillna(False)
                    & (
                        analysis["Close"].lt(analysis["SMA_50"]).fillna(False)
                        if danger_requires_price_below_sma50
                        else True
                    )
                )
            )
        )
    exit_signal = pd.Series(False, index=analysis.index, dtype=bool)
    if not disable_exit:
        exit_signal = (
            (analysis["Tech Score"].le(hard_exit_score_floor) & macd_bearish)
            | (
                bearish_regime
                & analysis["Close"].le(analysis["SMA_200"] * (1 - trend_tolerance * exit_break_multiplier)).fillna(False)
                & macd_bearish
                & (
                    analysis["SMA_50"].le(analysis["SMA_200"]).fillna(False)
                    if exit_requires_sma50_cross
                    else True
                )
                & analysis["Momentum_1Y"].lt(exit_momentum_floor).fillna(False)
            )
        )

    if primary_type in growth_compounder_types:
        overweight_target = 1.45
        danger_score_floor = -4
        danger_momentum_multiplier = 1.5
        danger_requires_price_below_sma50 = True
        exit_requires_sma50_cross = True
        exit_momentum_floor = -0.22
        danger_reduce = (
            (bearish_regime | trailing_stop_breach)
            & (
                analysis["Tech Score"].le(danger_score_floor)
                | (
                    macd_bearish
                    & analysis["Close"].lt(analysis["SMA_50"]).fillna(False)
                    & analysis["Momentum_1M"].lt(
                        -active_settings["tech_momentum_threshold"] * danger_momentum_multiplier
                    ).fillna(False)
                )
            )
        )
        exit_signal = (
            (analysis["Tech Score"].le(hard_exit_score_floor) & macd_bearish)
            | (
                bearish_regime
                & analysis["SMA_50"].le(analysis["SMA_200"]).fillna(False)
                & analysis["Close"].le(analysis["SMA_200"] * (1 - trend_tolerance * exit_break_multiplier)).fillna(False)
                & analysis["Momentum_1Y"].lt(exit_momentum_floor).fillna(False)
            )
        )

    positions = []
    first_bearish = bool(bearish_regime.iloc[0]) if len(bearish_regime) else False
    current_position = 0.0 if first_bearish and not ignore_initial_bearish_gate else initial_floor_if_not_bearish
    days_since_change = full_reentry_cooldown
    for is_bullish, is_bearish, enter_now, add_now, danger_now, exit_now, strong_now in zip(
        bullish_regime,
        bearish_regime,
        entry_signal,
        add_signal,
        danger_reduce,
        exit_signal,
        strong_bullish,
    ):
        target_position = current_position
        if exit_now:
            target_position = 0.0
        elif is_bullish:
            if current_position < core_target and enter_now and days_since_change >= core_reentry_cooldown:
                target_position = core_target
            if add_now and days_since_change >= full_reentry_cooldown:
                target_position = full_target
            if overweight_target is not None and strong_now and days_since_change >= core_reentry_cooldown:
                target_position = max(target_position, overweight_target)
        elif is_bearish:
            if danger_now and current_position > danger_floor:
                target_position = danger_floor
        else:
            if trim_to_core_on_non_bullish and current_position > core_target:
                target_position = core_target
            elif allow_core_outside_bullish and enter_now and current_position < core_target and days_since_change >= core_reentry_cooldown:
                target_position = core_target

        if target_position != current_position:
            current_position = target_position
            days_since_change = 0
        else:
            days_since_change += 1
        positions.append(current_position)

    return pd.Series(positions, index=analysis.index, dtype=float)


def summarize_backtest_trades(analysis):
    if analysis is None or analysis.empty or "Close" not in analysis.columns or "Position" not in analysis.columns:
        return pd.DataFrame(), {
            "Closed Trades": 0,
            "Win Rate": None,
            "Average Trade Return": None,
        }

    open_lots = []
    closed_trades = []
    previous_position = 0.0

    for date, row in analysis.iterrows():
        close = safe_num(row.get("Close"))
        current_position = safe_num(row.get("Position")) or 0.0
        if close is None:
            previous_position = current_position
            continue

        position_change = round(current_position - previous_position, 6)
        if position_change > 0:
            open_lots.append(
                {
                    "entry_date": date,
                    "entry_price": close,
                    "size": position_change,
                }
            )
        elif position_change < 0:
            remaining_to_close = abs(position_change)
            while remaining_to_close > 1e-9 and open_lots:
                lot = open_lots[0]
                closed_size = min(lot["size"], remaining_to_close)
                closed_trades.append(
                    {
                        "Entry Date": lot["entry_date"],
                        "Exit Date": date,
                        "Entry Price": lot["entry_price"],
                        "Exit Price": close,
                        "Position Size": closed_size,
                        "Return": safe_divide(close - lot["entry_price"], lot["entry_price"]),
                        "Holding Days": (date - lot["entry_date"]).days if hasattr(date, "__sub__") else None,
                    }
                )
                lot["size"] -= closed_size
                remaining_to_close -= closed_size
                if lot["size"] <= 1e-9:
                    open_lots.pop(0)

        previous_position = current_position

    if not closed_trades:
        return pd.DataFrame(), {
            "Closed Trades": 0,
            "Win Rate": None,
            "Average Trade Return": None,
        }

    closed_trades_df = pd.DataFrame(closed_trades)
    return closed_trades_df, {
        "Closed Trades": len(closed_trades_df),
        "Win Rate": (closed_trades_df["Return"] > 0).mean(),
        "Average Trade Return": closed_trades_df["Return"].mean(),
    }


def get_default_model_settings():
    return DEFAULT_MODEL_SETTINGS.copy()


def get_default_preset_name():
    return "Balanced"


def get_default_dcf_settings():
    return copy.deepcopy(DEFAULT_DCF_SETTINGS)


def normalize_dcf_settings(settings):
    normalized = get_default_dcf_settings()
    normalized.update(settings or {})

    normalized["projection_years"] = int(min(max(int(round(float(normalized["projection_years"]))), 3), 10))
    normalized["terminal_growth_rate"] = float(min(max(float(normalized["terminal_growth_rate"]), 0.0), 0.05))
    normalized["growth_haircut"] = float(min(max(float(normalized["growth_haircut"]), 0.5), 1.2))
    normalized["max_growth_rate"] = float(min(max(float(normalized["max_growth_rate"]), 0.05), 0.50))
    normalized["min_growth_rate"] = float(min(max(float(normalized["min_growth_rate"]), -0.20), 0.10))
    if normalized["min_growth_rate"] >= normalized["max_growth_rate"]:
        normalized["min_growth_rate"] = min(normalized["max_growth_rate"] - 0.01, -0.01)

    normalized["market_risk_premium"] = float(min(max(float(normalized["market_risk_premium"]), 0.03), 0.09))
    normalized["default_after_tax_cost_of_debt"] = float(
        min(max(float(normalized["default_after_tax_cost_of_debt"]), 0.01), 0.10)
    )

    for optional_key in ["risk_free_rate_override", "manual_growth_rate"]:
        optional_value = safe_num(normalized.get(optional_key))
        if optional_value is None:
            normalized[optional_key] = None
        elif optional_key == "risk_free_rate_override":
            normalized[optional_key] = float(min(max(optional_value, 0.0), 0.10))
        else:
            normalized[optional_key] = float(min(max(optional_value, -0.20), 0.50))

    return normalized


def get_dcf_settings():
    if "dcf_settings" not in st.session_state:
        st.session_state.dcf_settings = normalize_dcf_settings(get_default_dcf_settings())
        return st.session_state.dcf_settings

    normalized = normalize_dcf_settings(st.session_state.dcf_settings)
    st.session_state.dcf_settings = normalized
    return normalized


def normalize_model_settings(settings):
    normalized = get_default_model_settings()
    normalized.update(settings or {})
    notes = []

    normalized["tech_rsi_oversold"] = min(max(float(normalized["tech_rsi_oversold"]), 20.0), 45.0)
    normalized["tech_rsi_overbought"] = min(max(float(normalized["tech_rsi_overbought"]), 55.0), 85.0)
    if normalized["tech_rsi_oversold"] >= normalized["tech_rsi_overbought"]:
        normalized["tech_rsi_overbought"] = min(85.0, normalized["tech_rsi_oversold"] + 5.0)
        notes.append("The RSI overbought threshold was nudged above the oversold threshold.")

    normalized["tech_momentum_threshold"] = min(max(float(normalized["tech_momentum_threshold"]), 0.01), 0.12)
    normalized["tech_trend_tolerance"] = min(max(float(normalized["tech_trend_tolerance"]), 0.0), 0.05)
    normalized["tech_extension_limit"] = min(max(float(normalized["tech_extension_limit"]), 0.03), 0.15)
    normalized["fund_roe_threshold"] = min(max(float(normalized["fund_roe_threshold"]), 0.05), 0.35)
    normalized["fund_profit_margin_threshold"] = min(max(float(normalized["fund_profit_margin_threshold"]), 0.05), 0.35)
    normalized["fund_debt_good_threshold"] = min(max(float(normalized["fund_debt_good_threshold"]), 25.0), 200.0)
    normalized["fund_debt_bad_threshold"] = min(max(float(normalized["fund_debt_bad_threshold"]), 75.0), 400.0)
    if normalized["fund_debt_bad_threshold"] <= normalized["fund_debt_good_threshold"]:
        normalized["fund_debt_bad_threshold"] = min(400.0, normalized["fund_debt_good_threshold"] + 50.0)
        notes.append("The high-debt threshold was lifted above the low-debt threshold.")

    normalized["fund_revenue_growth_threshold"] = min(max(float(normalized["fund_revenue_growth_threshold"]), 0.0), 0.30)
    normalized["fund_current_ratio_good"] = min(max(float(normalized["fund_current_ratio_good"]), 1.0), 3.0)
    normalized["fund_current_ratio_bad"] = min(max(float(normalized["fund_current_ratio_bad"]), 0.5), 1.5)
    if normalized["fund_current_ratio_bad"] >= normalized["fund_current_ratio_good"]:
        normalized["fund_current_ratio_bad"] = max(0.5, normalized["fund_current_ratio_good"] - 0.2)
        notes.append("The weak current-ratio threshold was moved below the healthy threshold.")

    normalized["valuation_benchmark_scale"] = min(max(float(normalized["valuation_benchmark_scale"]), 0.8), 1.2)
    normalized["valuation_peg_threshold"] = min(max(float(normalized["valuation_peg_threshold"]), 0.8), 2.5)
    normalized["valuation_graham_overpriced_multiple"] = min(
        max(float(normalized["valuation_graham_overpriced_multiple"]), 1.2), 2.0
    )
    normalized["valuation_under_score_threshold"] = min(
        max(float(normalized["valuation_under_score_threshold"]), 3.0), 8.0
    )
    normalized["valuation_fair_score_threshold"] = min(
        max(float(normalized["valuation_fair_score_threshold"]), 1.0), 4.0
    )
    if normalized["valuation_under_score_threshold"] <= normalized["valuation_fair_score_threshold"]:
        normalized["valuation_under_score_threshold"] = min(
            8.0, normalized["valuation_fair_score_threshold"] + 1.0
        )
        notes.append("The undervalued score floor was kept above the fair-value score floor.")

    normalized["sentiment_analyst_boost"] = min(max(float(normalized["sentiment_analyst_boost"]), 0.0), 4.0)
    normalized["sentiment_upside_mid"] = min(max(float(normalized["sentiment_upside_mid"]), 0.02), 0.15)
    normalized["sentiment_upside_high"] = min(max(float(normalized["sentiment_upside_high"]), 0.08), 0.30)
    if normalized["sentiment_upside_high"] <= normalized["sentiment_upside_mid"]:
        normalized["sentiment_upside_high"] = min(0.30, normalized["sentiment_upside_mid"] + 0.05)
        notes.append("The high-upside threshold was kept above the moderate-upside threshold.")

    normalized["sentiment_downside_mid"] = min(max(float(normalized["sentiment_downside_mid"]), 0.02), 0.15)
    normalized["sentiment_downside_high"] = min(max(float(normalized["sentiment_downside_high"]), 0.08), 0.30)
    if normalized["sentiment_downside_high"] <= normalized["sentiment_downside_mid"]:
        normalized["sentiment_downside_high"] = min(0.30, normalized["sentiment_downside_mid"] + 0.05)
        notes.append("The deep-downside threshold was kept above the moderate-downside threshold.")

    normalized["overall_buy_threshold"] = min(max(float(normalized["overall_buy_threshold"]), 1.0), 6.0)
    normalized["overall_strong_buy_threshold"] = min(
        max(float(normalized["overall_strong_buy_threshold"]), 4.0), 12.0
    )
    if normalized["overall_strong_buy_threshold"] <= normalized["overall_buy_threshold"]:
        normalized["overall_strong_buy_threshold"] = min(12.0, normalized["overall_buy_threshold"] + 2.0)
        notes.append("The strong-buy threshold was kept above the buy threshold.")

    normalized["overall_sell_threshold"] = -min(max(abs(float(normalized["overall_sell_threshold"])), 1.0), 6.0)
    normalized["overall_strong_sell_threshold"] = -min(
        max(abs(float(normalized["overall_strong_sell_threshold"])), 4.0), 12.0
    )
    if abs(normalized["overall_strong_sell_threshold"]) <= abs(normalized["overall_sell_threshold"]):
        normalized["overall_strong_sell_threshold"] = -min(12.0, abs(normalized["overall_sell_threshold"]) + 2.0)
        notes.append("The strong-sell threshold was kept below the sell threshold.")

    normalized["decision_hold_buffer"] = min(max(float(normalized["decision_hold_buffer"]), 0.0), 3.0)
    normalized["decision_min_confidence"] = min(max(float(normalized["decision_min_confidence"]), 35.0), 80.0)
    normalized["backtest_cooldown_days"] = float(min(max(float(normalized["backtest_cooldown_days"]), 0.0), 20.0))
    normalized["backtest_transaction_cost_bps"] = float(
        min(max(float(normalized["backtest_transaction_cost_bps"]), 0.0), 50.0)
    )

    for key in ["weight_technical", "weight_fundamental", "weight_valuation", "weight_sentiment"]:
        normalized[key] = min(max(float(normalized[key]), 0.5), 1.5)

    normalized["trading_days_per_year"] = float(min(max(float(normalized["trading_days_per_year"]), 240.0), 260.0))
    return normalized, notes


def serialize_model_settings(settings):
    normalized, _ = normalize_model_settings(settings)
    rounded = {}
    for key, value in normalized.items():
        rounded[key] = round(float(value), 6) if isinstance(value, float) else value
    return json.dumps(rounded, sort_keys=True)


def serialize_dcf_settings(settings):
    normalized = normalize_dcf_settings(settings)
    rounded = {}
    for key, value in normalized.items():
        if isinstance(value, float):
            rounded[key] = round(value, 6)
        else:
            rounded[key] = value
    return json.dumps(rounded, sort_keys=True)


def get_model_presets():
    return {name: normalize_model_settings(values)[0] for name, values in MODEL_PRESETS.items()}


def get_assumption_fingerprint(settings=None):
    active_settings = get_model_settings() if settings is None else settings
    payload = serialize_model_settings(active_settings)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def detect_matching_preset(settings=None):
    active_settings = get_model_settings() if settings is None else settings
    for name, preset in get_model_presets().items():
        matches = all(np.isclose(active_settings[key], preset[key]) for key in preset)
        if matches:
            return name
    return "Custom"


def get_model_settings():
    if "model_settings" not in st.session_state:
        normalized, _ = normalize_model_settings(get_default_model_settings())
        st.session_state.model_settings = normalized
        st.session_state.model_preset_name = get_default_preset_name()
        return normalized

    normalized, _ = normalize_model_settings(st.session_state.model_settings)
    st.session_state.model_settings = normalized
    st.session_state.model_preset_name = detect_matching_preset(normalized)
    return normalized


def calculate_assumption_drift(settings=None):
    baseline = get_default_model_settings()
    active_settings = get_model_settings() if settings is None else settings
    deviations = []

    for key, default_value in baseline.items():
        current_value = float(active_settings.get(key, default_value))
        scale = max(abs(float(default_value)), 0.05)
        deviations.append(abs(current_value - float(default_value)) / scale)

    return np.mean(deviations) * 100 if deviations else 0.0


def get_sector_benchmarks(sector, settings=None):
    active_settings = get_model_settings() if settings is None else settings
    scale = active_settings["valuation_benchmark_scale"]
    base_benchmarks = SECTOR_BENCHMARKS.get(sector, DEFAULT_BENCHMARKS)
    return {metric: value * scale for metric, value in base_benchmarks.items()}


def rate_data_quality(completeness):
    if completeness >= 0.85:
        return "High"
    if completeness >= 0.60:
        return "Medium"
    return "Low"


def assess_record_quality(record):
    quality_fields = [
        "Price",
        "Sector",
        "PE_Ratio",
        "Forward_PE",
        "Profit_Margins",
        "ROE",
        "Debt_to_Equity",
        "Revenue_Growth",
        "Current_Ratio",
        "RSI",
        "MACD_Value",
        "Momentum_1M",
        "Target_Mean_Price",
        "Recommendation_Key",
    ]
    missing_count = 0
    for field in quality_fields:
        value = record.get(field)
        if value is None or (isinstance(value, float) and pd.isna(value)) or value == "N/A":
            missing_count += 1
    completeness = 1 - (missing_count / len(quality_fields))
    return completeness, missing_count, rate_data_quality(completeness)


def map_verdict_bias(verdict):
    if verdict in {"BUY", "STRONG BUY"}:
        return "Bullish"
    if verdict in {"SELL", "STRONG SELL"}:
        return "Bearish"
    return "Neutral"


def prepare_analysis_dataframe(df, settings=None):
    if df is None or df.empty:
        return pd.DataFrame()

    active_settings = get_model_settings() if settings is None else settings
    enriched = df.copy()
    for column in ANALYSIS_NUMERIC_COLUMNS:
        if column in enriched.columns:
            enriched[column] = pd.to_numeric(enriched[column], errors="coerce")

    score_columns = [
        column for column in ["Score_Tech", "Score_Fund", "Score_Val", "Score_Sentiment"]
        if column in enriched.columns
    ]
    if "Composite Score" not in enriched.columns:
        enriched["Composite Score"] = np.nan
    if score_columns:
        score_weights = {
            "Score_Tech": active_settings["weight_technical"],
            "Score_Fund": active_settings["weight_fundamental"],
            "Score_Val": active_settings["weight_valuation"],
            "Score_Sentiment": active_settings["weight_sentiment"],
        }
        composite_score = pd.Series(0.0, index=enriched.index)
        for column in score_columns:
            composite_score = composite_score + enriched[column].fillna(0) * score_weights.get(column, 1.0)
        enriched["Composite Score"] = composite_score

    if "Target Upside" not in enriched.columns:
        enriched["Target Upside"] = np.nan
    if {"Price", "Target_Mean_Price"}.issubset(enriched.columns):
        enriched["Target Upside"] = np.where(
            enriched["Price"].notna() & (enriched["Price"] != 0),
            (enriched["Target_Mean_Price"] - enriched["Price"]) / enriched["Price"],
            np.nan,
        )

    if "Graham Discount" not in enriched.columns:
        enriched["Graham Discount"] = np.nan
    if {"Price", "Graham_Number"}.issubset(enriched.columns):
        enriched["Graham Discount"] = np.where(
            enriched["Price"].notna() & (enriched["Price"] != 0) & enriched["Graham_Number"].notna(),
            (enriched["Graham_Number"] - enriched["Price"]) / enriched["Price"],
            np.nan,
        )

    if "DCF Upside" not in enriched.columns:
        enriched["DCF Upside"] = np.nan
    if {"Price", "DCF_Intrinsic_Value"}.issubset(enriched.columns):
        enriched["DCF Upside"] = np.where(
            enriched["Price"].notna() & (enriched["Price"] != 0) & enriched["DCF_Intrinsic_Value"].notna(),
            (enriched["DCF_Intrinsic_Value"] - enriched["Price"]) / enriched["Price"],
            np.nan,
        )

    if "Assumption_Profile" not in enriched.columns:
        enriched["Assumption_Profile"] = "Legacy"
    else:
        enriched["Assumption_Profile"] = enriched["Assumption_Profile"].fillna("Legacy")
    if "Assumption_Fingerprint" not in enriched.columns:
        enriched["Assumption_Fingerprint"] = "Legacy"
    else:
        enriched["Assumption_Fingerprint"] = enriched["Assumption_Fingerprint"].fillna("Legacy")
    if "Market_Regime" not in enriched.columns:
        enriched["Market_Regime"] = "Unknown"
    else:
        enriched["Market_Regime"] = enriched["Market_Regime"].fillna("Unknown")
    if "Decision_Notes" not in enriched.columns:
        enriched["Decision_Notes"] = ""
    else:
        enriched["Decision_Notes"] = enriched["Decision_Notes"].fillna("")
    if "Decision_Confidence" not in enriched.columns:
        enriched["Decision_Confidence"] = np.nan
    if "Industry" not in enriched.columns:
        enriched["Industry"] = "Unknown"
    else:
        enriched["Industry"] = enriched["Industry"].fillna("Unknown")
    if "Stock_Type" not in enriched.columns:
        enriched["Stock_Type"] = "Legacy"
    else:
        enriched["Stock_Type"] = enriched["Stock_Type"].fillna("Legacy")
    if "Cap_Bucket" not in enriched.columns:
        enriched["Cap_Bucket"] = "Unknown"
    else:
        enriched["Cap_Bucket"] = enriched["Cap_Bucket"].fillna("Unknown")
    if "Style_Tags" not in enriched.columns:
        enriched["Style_Tags"] = ""
    else:
        enriched["Style_Tags"] = enriched["Style_Tags"].fillna("")
    if "Type_Strategy" not in enriched.columns:
        enriched["Type_Strategy"] = ""
    else:
        enriched["Type_Strategy"] = enriched["Type_Strategy"].fillna("")
    if "Type_Confidence" not in enriched.columns:
        enriched["Type_Confidence"] = np.nan
    if "Engine_Weight_Profile" not in enriched.columns:
        enriched["Engine_Weight_Profile"] = ""
    else:
        enriched["Engine_Weight_Profile"] = enriched["Engine_Weight_Profile"].fillna("")
    if "Peer_Count" not in enriched.columns:
        enriched["Peer_Count"] = np.nan
    if "Peer_Group_Label" not in enriched.columns:
        enriched["Peer_Group_Label"] = ""
    else:
        enriched["Peer_Group_Label"] = enriched["Peer_Group_Label"].fillna("")
    if "Peer_Tickers" not in enriched.columns:
        enriched["Peer_Tickers"] = ""
    else:
        enriched["Peer_Tickers"] = enriched["Peer_Tickers"].fillna("")
    if "Peer_Summary" not in enriched.columns:
        enriched["Peer_Summary"] = ""
    else:
        enriched["Peer_Summary"] = enriched["Peer_Summary"].fillna("")
    if "Peer_Comparison" not in enriched.columns:
        enriched["Peer_Comparison"] = ""
    else:
        enriched["Peer_Comparison"] = enriched["Peer_Comparison"].fillna("")
    if "Risk_Flags" not in enriched.columns:
        enriched["Risk_Flags"] = ""
    else:
        enriched["Risk_Flags"] = enriched["Risk_Flags"].fillna("")
    for metric_column in ["Relative_Strength_3M", "Relative_Strength_6M", "Relative_Strength_1Y"]:
        if metric_column not in enriched.columns:
            enriched[metric_column] = np.nan
    if "DCF_Source" not in enriched.columns:
        enriched["DCF_Source"] = "Unavailable"
    else:
        enriched["DCF_Source"] = enriched["DCF_Source"].fillna("Unavailable")
    if "DCF_Confidence" not in enriched.columns:
        enriched["DCF_Confidence"] = "low"
    else:
        enriched["DCF_Confidence"] = enriched["DCF_Confidence"].fillna("low")
    if "DCF_Last_Updated" not in enriched.columns:
        enriched["DCF_Last_Updated"] = ""
    else:
        enriched["DCF_Last_Updated"] = enriched["DCF_Last_Updated"].fillna("")
    if "DCF_Assumptions" not in enriched.columns:
        enriched["DCF_Assumptions"] = ""
    else:
        enriched["DCF_Assumptions"] = enriched["DCF_Assumptions"].fillna("")
    for text_column in [
        "DCF_History",
        "DCF_Projection",
        "DCF_Sensitivity",
        "DCF_Guidance_Excerpts",
        "DCF_Guidance_Summary",
    ]:
        if text_column not in enriched.columns:
            enriched[text_column] = ""
        else:
            enriched[text_column] = enriched[text_column].fillna("")
    if "Sentiment_Summary" not in enriched.columns:
        enriched["Sentiment_Summary"] = ""
    else:
        enriched["Sentiment_Summary"] = enriched["Sentiment_Summary"].fillna("")
    if "Event_Study_Count" not in enriched.columns:
        enriched["Event_Study_Count"] = np.nan
    if "Event_Study_Avg_Abnormal_1D" not in enriched.columns:
        enriched["Event_Study_Avg_Abnormal_1D"] = np.nan
    if "Event_Study_Avg_Abnormal_5D" not in enriched.columns:
        enriched["Event_Study_Avg_Abnormal_5D"] = np.nan
    if "Event_Study_Summary" not in enriched.columns:
        enriched["Event_Study_Summary"] = ""
    else:
        enriched["Event_Study_Summary"] = enriched["Event_Study_Summary"].fillna("")
    if "Event_Study_Events" not in enriched.columns:
        enriched["Event_Study_Events"] = ""
    else:
        enriched["Event_Study_Events"] = enriched["Event_Study_Events"].fillna("")
    if "Last_Data_Update" not in enriched.columns:
        enriched["Last_Data_Update"] = ""
    else:
        enriched["Last_Data_Update"] = enriched["Last_Data_Update"].fillna("")

    if (
        "Data_Completeness" not in enriched.columns
        or "Missing_Metric_Count" not in enriched.columns
        or "Data_Quality" not in enriched.columns
        or enriched["Data_Completeness"].isna().any()
        or enriched["Missing_Metric_Count"].isna().any()
        or enriched["Data_Quality"].isna().any()
    ):
        quality_stats = enriched.apply(
            lambda row: assess_record_quality(row.to_dict()),
            axis=1,
            result_type="expand",
        )
        quality_stats.columns = ["_Data_Completeness", "_Missing_Metric_Count", "_Data_Quality"]
        enriched["Data_Completeness"] = pd.to_numeric(
            enriched.get("Data_Completeness", quality_stats["_Data_Completeness"]),
            errors="coerce",
        ).fillna(quality_stats["_Data_Completeness"])
        enriched["Missing_Metric_Count"] = pd.to_numeric(
            enriched.get("Missing_Metric_Count", quality_stats["_Missing_Metric_Count"]),
            errors="coerce",
        ).fillna(quality_stats["_Missing_Metric_Count"])
        enriched["Data_Quality"] = enriched.get("Data_Quality", quality_stats["_Data_Quality"]).fillna(
            quality_stats["_Data_Quality"]
        )

    if "Last_Updated" in enriched.columns:
        enriched["Last_Updated_Parsed"] = enriched["Last_Updated"].map(parse_last_updated)
        enriched["Freshness"] = enriched["Last_Updated"].map(format_age)
        sort_columns = ["Last_Updated_Parsed"]
        ascending = [False]
        if "Composite Score" in enriched.columns:
            sort_columns.append("Composite Score")
            ascending.append(False)
        if "Ticker" in enriched.columns:
            sort_columns.append("Ticker")
            ascending.append(True)
        enriched = enriched.sort_values(sort_columns, ascending=ascending, na_position="last")
    elif "Composite Score" in enriched.columns:
        enriched = enriched.sort_values(["Composite Score", "Ticker"], ascending=[False, True], na_position="last")
    if "Freshness" not in enriched.columns:
        enriched["Freshness"] = "Unknown"

    return enriched.reset_index(drop=True)


def collect_analysis_rows(analyst, db, tickers, refresh_live=False):
    rows = []
    failed = []
    failure_reasons = {}
    refreshed = []
    cached = []

    for ticker in tickers:
        existing = db.get_analysis(ticker)
        if refresh_live or existing.empty:
            record = analyst.analyze(ticker)
            if record is None:
                failed.append(ticker)
                failure_reasons[ticker] = analyst.last_error or "No usable market data was returned for this ticker."
                continue
            rows.append(record)
            refreshed.append(ticker)
        else:
            rows.append(existing.iloc[0].to_dict())
            cached.append(ticker)

    return prepare_analysis_dataframe(pd.DataFrame(rows)), failed, failure_reasons, refreshed, cached


def build_sensitivity_scenarios(base_settings):
    conservative = get_model_presets()["Conservative"]
    balanced = get_model_presets()["Balanced"]
    aggressive = get_model_presets()["Aggressive"]

    valuation_strict, _ = normalize_model_settings(
        {
            **base_settings,
            "valuation_benchmark_scale": base_settings["valuation_benchmark_scale"] * 0.95,
            "valuation_under_score_threshold": base_settings["valuation_under_score_threshold"] + 1,
            "overall_buy_threshold": base_settings["overall_buy_threshold"] + 1,
            "overall_strong_buy_threshold": base_settings["overall_strong_buy_threshold"] + 1,
        }
    )
    sentiment_light, _ = normalize_model_settings(
        {
            **base_settings,
            "weight_sentiment": max(0.5, base_settings["weight_sentiment"] - 0.3),
            "sentiment_analyst_boost": max(0.0, base_settings["sentiment_analyst_boost"] - 0.5),
        }
    )

    return [
        ("Current", base_settings),
        ("Balanced", balanced),
        ("Conservative", conservative),
        ("Aggressive", aggressive),
        ("Valuation Strict", valuation_strict),
        ("Low Sentiment Weight", sentiment_light),
    ]


def run_sensitivity_analysis(analyst, ticker, settings=None):
    active_settings = get_model_settings() if settings is None else settings
    hist, info, news = analyst.get_data(ticker)
    if hist is None or hist.empty:
        return None, None

    scenario_rows = []
    for scenario_name, scenario_settings in build_sensitivity_scenarios(active_settings):
        record = analyst.analyze(
            ticker,
            settings=scenario_settings,
            persist=False,
            preloaded=(hist, info, news),
        )
        if record is None:
            continue
        scenario_rows.append(
            {
                "Scenario": scenario_name,
                "Preset": detect_matching_preset(scenario_settings),
                "Verdict": record["Verdict_Overall"],
                "Bias": map_verdict_bias(record["Verdict_Overall"]),
                "Overall Score": record["Overall_Score"],
                "Technical": record["Score_Tech"],
                "Fundamental": record["Score_Fund"],
                "Valuation": record["Score_Val"],
                "Sentiment": record["Score_Sentiment"],
                "Consistency": record.get("Decision_Confidence"),
                "Regime": record.get("Market_Regime"),
                "Assumption Drift": calculate_assumption_drift(scenario_settings),
                "Fingerprint": record["Assumption_Fingerprint"],
            }
        )

    if not scenario_rows:
        return None, None

    scenarios_df = pd.DataFrame(scenario_rows)
    bias_counts = scenarios_df["Bias"].value_counts()
    dominant_bias = bias_counts.index[0]
    robustness_ratio = bias_counts.iloc[0] / len(scenarios_df)
    if robustness_ratio >= 0.85:
        robustness_label = "High"
    elif robustness_ratio >= 0.60:
        robustness_label = "Medium"
    else:
        robustness_label = "Low"

    summary = {
        "dominant_bias": dominant_bias,
        "robustness_ratio": robustness_ratio,
        "robustness_label": robustness_label,
        "verdict_count": scenarios_df["Verdict"].nunique(),
        "scenario_count": len(scenarios_df),
    }
    return scenarios_df, summary


def compute_technical_backtest(hist, settings=None, stock_profile=None):
    active_settings = get_model_settings() if settings is None else settings
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None

    close = hist["Close"].dropna().copy()
    if len(close) < 250:
        return None

    analysis = pd.DataFrame(index=close.index)
    analysis["Close"] = close
    analysis["RSI"] = calculate_rsi(close)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    analysis["MACD"] = ema12 - ema26
    analysis["MACD_Signal_Line"] = analysis["MACD"].ewm(span=9, adjust=False).mean()
    analysis["SMA_50"] = close.rolling(50).mean()
    analysis["SMA_200"] = close.rolling(200).mean()
    analysis["Momentum_1M"] = close / close.shift(22) - 1
    analysis["Momentum_1Y"] = close / close.shift(252) - 1
    analysis["Volatility_1M"] = close.pct_change().rolling(22).std() * np.sqrt(252)
    analysis["Volatility_1Y"] = close.pct_change().rolling(252).std() * np.sqrt(252)
    analysis["Momentum_1M_Risk_Adjusted"] = analysis["Momentum_1M"] / (analysis["Volatility_1M"] / np.sqrt(12))
    analysis["Rolling_High_252"] = close.rolling(252, min_periods=20).max()
    analysis["Rolling_Low_252"] = close.rolling(252, min_periods=20).min()
    analysis["Range_Position_252"] = (close - analysis["Rolling_Low_252"]) / (
        analysis["Rolling_High_252"] - analysis["Rolling_Low_252"]
    )
    analysis["Distance_52W_High"] = (close - analysis["Rolling_High_252"]) / analysis["Rolling_High_252"]
    analysis["Trend_Strength"] = (
        ((close / analysis["SMA_200"]) - 1).clip(-0.25, 0.25).fillna(0) * 100
        + ((analysis["SMA_50"] / analysis["SMA_200"]) - 1).clip(-0.25, 0.25).fillna(0) * 120
        + analysis["Momentum_1Y"].clip(-0.3125, 0.3125).fillna(0) * 80
    )
    analysis["Quarter_High"] = close.rolling(63, min_periods=20).max()
    analysis["Trailing_Drawdown_Quarter"] = (close / analysis["Quarter_High"]) - 1

    trend_tolerance = active_settings["tech_trend_tolerance"]
    extension_limit = active_settings["tech_extension_limit"]
    tech_score = pd.Series(0, index=analysis.index, dtype=float)
    tech_score += np.where(
        analysis["SMA_200"].notna(),
        np.where(
            analysis["Close"] >= analysis["SMA_200"] * (1 + trend_tolerance),
            1,
            np.where(analysis["Close"] <= analysis["SMA_200"] * (1 - trend_tolerance), -1, 0),
        ),
        0,
    )
    tech_score += np.where(
        analysis["SMA_50"].notna() & analysis["SMA_200"].notna(),
        np.where(
            analysis["SMA_50"] >= analysis["SMA_200"] * (1 + trend_tolerance / 2),
            1,
            np.where(analysis["SMA_50"] <= analysis["SMA_200"] * (1 - trend_tolerance / 2), -1, 0),
        ),
        0,
    )
    tech_score += np.where(
        analysis["SMA_50"].notna(),
        np.where(
            analysis["Close"] >= analysis["SMA_50"] * (1 + trend_tolerance / 2),
            1,
            np.where(analysis["Close"] <= analysis["SMA_50"] * (1 - trend_tolerance / 2), -1, 0),
        ),
        0,
    )
    tech_score += np.where(
        analysis["RSI"] < active_settings["tech_rsi_oversold"],
        np.where(analysis["Close"] >= analysis["SMA_200"] * 0.95, 2, 1),
        0,
    )
    tech_score += np.where(
        analysis["RSI"] > active_settings["tech_rsi_overbought"],
        np.where(analysis["Close"] <= analysis["SMA_50"] * 1.02, -2, -1),
        0,
    )
    tech_score += np.where(
        analysis["MACD"].notna() & analysis["MACD_Signal_Line"].notna(),
        np.where(analysis["MACD"] > analysis["MACD_Signal_Line"], 1, -1),
        0,
    )
    tech_score += np.where(analysis["MACD"] > 0, 1, np.where(analysis["MACD"] < 0, -1, 0))
    tech_score += np.where(analysis["Momentum_1M"] > active_settings["tech_momentum_threshold"], 1, 0)
    tech_score += np.where(analysis["Momentum_1M"] < -active_settings["tech_momentum_threshold"], -1, 0)
    tech_score += np.where(analysis["Momentum_1M_Risk_Adjusted"] > 0.75, 1, 0)
    tech_score += np.where(analysis["Momentum_1M_Risk_Adjusted"] < -0.75, -1, 0)
    long_term_momentum_threshold = max(active_settings["tech_momentum_threshold"] * 3, 0.10)
    tech_score += np.where(analysis["Momentum_1Y"] > long_term_momentum_threshold, 1, 0)
    tech_score += np.where(analysis["Momentum_1Y"] < -long_term_momentum_threshold, -1, 0)
    tech_score += np.where(analysis["Trend_Strength"] > 30, 1, 0)
    tech_score += np.where(analysis["Trend_Strength"] < -30, -1, 0)
    tech_score += np.where(
        analysis["Range_Position_252"].ge(0.80) & analysis["Close"].ge(analysis["SMA_200"]),
        1,
        0,
    )
    tech_score += np.where(
        analysis["Range_Position_252"].le(0.20) & analysis["Close"].le(analysis["SMA_200"]),
        -1,
        0,
    )
    tech_score += np.where(
        analysis["RSI"].shift(1).le(active_settings["tech_rsi_oversold"])
        & analysis["RSI"].gt(active_settings["tech_rsi_oversold"])
        & analysis["MACD"].ge(analysis["MACD_Signal_Line"]),
        1,
        0,
    )
    tech_score += np.where(
        analysis["Close"].ge(analysis["SMA_50"] * (1 + extension_limit))
        & analysis["RSI"].ge(active_settings["tech_rsi_overbought"] - 5),
        -1,
        0,
    )
    tech_score += np.where(
        analysis["Close"].le(analysis["SMA_50"] * (1 - extension_limit))
        & analysis["RSI"].le(active_settings["tech_rsi_oversold"] + 5),
        1,
        0,
    )

    analysis["Tech Score"] = tech_score
    analysis["Signal"] = analysis["Tech Score"].apply(score_to_signal)
    analysis["Position"] = derive_backtest_positions(analysis, active_settings, stock_profile=stock_profile)
    analysis["Benchmark Return"] = analysis["Close"].pct_change().fillna(0.0)
    trade_points = analysis["Position"].diff().fillna(analysis["Position"])
    trading_cost_rate = active_settings.get("backtest_transaction_cost_bps", 0.0) / 10000
    analysis["Trading Cost"] = trade_points.abs() * trading_cost_rate
    analysis["Strategy Return"] = (
        analysis["Position"].shift(1).fillna(0.0) * analysis["Benchmark Return"] - analysis["Trading Cost"]
    )
    analysis["Benchmark Equity"] = (1 + analysis["Benchmark Return"]).cumprod()
    analysis["Strategy Equity"] = (1 + analysis["Strategy Return"]).cumprod()
    strategy_total_return = analysis["Strategy Equity"].iloc[-1] - 1
    benchmark_total_return = analysis["Benchmark Equity"].iloc[-1] - 1
    average_exposure = analysis["Position"].mean()
    upside_capture = (
        safe_divide(analysis["Strategy Equity"].iloc[-1], analysis["Benchmark Equity"].iloc[-1])
        if benchmark_total_return > 0
        else None
    )

    trading_days = active_settings["trading_days_per_year"]
    strategy_ann_return = analysis["Strategy Return"].mean() * trading_days
    strategy_vol = analysis["Strategy Return"].std() * np.sqrt(trading_days)
    benchmark_ann_return = analysis["Benchmark Return"].mean() * trading_days
    benchmark_vol = analysis["Benchmark Return"].std() * np.sqrt(trading_days)

    strategy_drawdown = analysis["Strategy Equity"] / analysis["Strategy Equity"].cummax() - 1
    benchmark_drawdown = analysis["Benchmark Equity"] / analysis["Benchmark Equity"].cummax() - 1

    trade_log = pd.DataFrame(
        {
            "Date": analysis.index,
            "Action": np.select(
                [
                    trade_points >= 0.75,
                    trade_points > 0,
                    trade_points <= -0.75,
                    trade_points < 0,
                ],
                ["Enter", "Add", "Exit", "Reduce"],
                default=None,
            ),
            "Close": analysis["Close"],
            "Signal": analysis["Signal"],
            "Tech Score": analysis["Tech Score"],
            "Position": analysis["Position"],
            "Trading Cost": analysis["Trading Cost"],
        }
    ).dropna(subset=["Action"])
    closed_trades_df, trade_summary = summarize_backtest_trades(analysis)

    metrics = {
        "Strategy Total Return": strategy_total_return,
        "Benchmark Total Return": benchmark_total_return,
        "Relative Return": analysis["Strategy Equity"].iloc[-1] - analysis["Benchmark Equity"].iloc[-1],
        "Strategy Annual Return": strategy_ann_return,
        "Benchmark Annual Return": benchmark_ann_return,
        "Strategy Volatility": strategy_vol,
        "Benchmark Volatility": benchmark_vol,
        "Strategy Sharpe": safe_divide(strategy_ann_return, strategy_vol),
        "Benchmark Sharpe": safe_divide(benchmark_ann_return, benchmark_vol),
        "Strategy Max Drawdown": strategy_drawdown.min(),
        "Benchmark Max Drawdown": benchmark_drawdown.min(),
        "Trading Costs": analysis["Trading Cost"].sum(),
        "Average Exposure": average_exposure,
        "Upside Capture": upside_capture,
        "Position Changes": len(trade_log),
        "Closed Trades": trade_summary["Closed Trades"],
        "Win Rate": trade_summary["Win Rate"],
        "Average Trade Return": trade_summary["Average Trade Return"],
    }

    return {
        "history": analysis.reset_index().rename(columns={"index": "Date"}),
        "trade_log": trade_log.reset_index(drop=True),
        "closed_trades": closed_trades_df,
        "metrics": metrics,
        "stock_profile": stock_profile or {},
    }


class DatabaseManager:
    def __init__(self, db_name):
        self._raw_db_name = str(db_name).strip()
        self.db_path = None
        self._write_lock = threading.RLock()
        self._backend = "sqlite"
        self._sqlite_target = None
        self._sqlite_uri = False
        self._anchor_connection = None
        self._fallback_notice = None
        self._storage_mode = "disk"
        self._postgres_dsn = None
        self._initialize_storage_target()
        self.create_tables()

    def _initialize_storage_target(self):
        if is_postgres_database_url(self._raw_db_name):
            self._backend = "postgres"
            self._postgres_dsn = self._raw_db_name
            self.db_path = None
            self._storage_mode = "server"
            return

        if self._raw_db_name.lower() == ":memory:":
            self._activate_in_memory_mode(
                "Research storage is running in memory only. Library changes will reset when the app stops."
            )
            return

        self.db_path = Path(self._raw_db_name)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._sqlite_target = str(self.db_path)
        self._sqlite_uri = False
        self._storage_mode = "disk"

    def _activate_in_memory_mode(self, notice):
        self.db_path = None
        self._backend = "sqlite"
        self._sqlite_target = "file:zb_compiler_shared_memory?mode=memory&cache=shared"
        self._sqlite_uri = True
        self._storage_mode = "memory"
        self._fallback_notice = notice
        if self._anchor_connection is None:
            self._anchor_connection = sqlite3.connect(
                self._sqlite_target,
                timeout=30,
                check_same_thread=False,
                uri=True,
            )
            self._configure_connection(self._anchor_connection)

    def _configure_connection(self, conn):
        if self._backend != "sqlite":
            return
        conn.execute("PRAGMA busy_timeout = 30000")
        try:
            conn.execute("PRAGMA synchronous = NORMAL")
        except sqlite3.DatabaseError:
            pass

        journal_modes = ["MEMORY"] if self._storage_mode == "memory" else ["WAL", "TRUNCATE", "DELETE"]
        for mode in journal_modes:
            try:
                conn.execute(f"PRAGMA journal_mode={mode}")
                return
            except sqlite3.DatabaseError:
                continue

    def _connect(self, allow_recover=True):
        if self._backend == "postgres":
            try:
                conn = psycopg.connect(self._postgres_dsn)
            except psycopg.OperationalError as exc:
                raise psycopg.OperationalError(
                    build_postgres_connection_error_message(self._postgres_dsn, exc)
                ) from None
            conn.autocommit = False
            return conn

        # Open a fresh connection per operation so each session sees other users' commits.
        conn = None
        try:
            conn = sqlite3.connect(
                self._sqlite_target,
                timeout=30,
                check_same_thread=False,
                uri=self._sqlite_uri,
            )
            self._configure_connection(conn)
            conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
            return conn
        except sqlite3.DatabaseError as exc:
            if conn is not None:
                conn.close()
            if allow_recover and self._storage_mode == "disk" and self._recover_database_file(exc):
                return self._connect(allow_recover=False)
            if allow_recover and self._storage_mode == "disk" and self._enable_in_memory_fallback(exc):
                return self._connect(allow_recover=False)
            raise

    def _enable_in_memory_fallback(self, exc):
        if self._backend != "sqlite" or self._storage_mode == "memory":
            return False
        message = summarize_fetch_error(exc)
        self._activate_in_memory_mode(
            "Persistent research storage was unavailable, so the app fell back to in-memory mode. "
            f"SQLite reported: {message}"
        )
        return True

    def _recover_database_file(self, exc):
        if self._backend != "sqlite":
            return False
        if self.db_path is None or not self.db_path.exists():
            return False

        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            if self.db_path.stat().st_size == 0:
                self.db_path.unlink()
            else:
                backup_path = self.db_path.with_name(f"{self.db_path.stem}.corrupt-{timestamp}{self.db_path.suffix}")
                shutil.move(str(self.db_path), str(backup_path))
            return True
        except OSError:
            return False

    @contextmanager
    def _connection(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_tables(self):
        with self._write_lock:
            if self._backend == "postgres":
                with self._connection() as conn:
                    column_sql = ",\n                ".join(
                        f'"{name}" {self._postgres_column_definition(definition)}'
                        for name, definition in ANALYSIS_COLUMNS.items()
                    )
                    conn.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS analysis (
                            {column_sql}
                        )
                        """
                    )
                    existing_columns = {
                        row[0] for row in conn.execute(
                            """
                            SELECT column_name
                            FROM information_schema.columns
                            WHERE table_schema = 'public' AND table_name = 'analysis'
                            """
                        ).fetchall()
                    }
                    for name, definition in ANALYSIS_COLUMNS.items():
                        if name not in existing_columns:
                            conn.execute(
                                f'ALTER TABLE analysis ADD COLUMN "{name}" {self._postgres_column_definition(definition)}'
                            )
                return
            try:
                with self._connection() as conn:
                    column_sql = ",\n                ".join(
                        f"{name} {definition}" for name, definition in ANALYSIS_COLUMNS.items()
                    )
                    conn.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS analysis (
                            {column_sql}
                        )
                        """
                    )
                    existing_columns = {
                        row[1] for row in conn.execute("PRAGMA table_info(analysis)").fetchall()
                    }
                    for name, definition in ANALYSIS_COLUMNS.items():
                        if name not in existing_columns:
                            conn.execute(f"ALTER TABLE analysis ADD COLUMN {name} {definition}")
            except sqlite3.DatabaseError as exc:
                if self._storage_mode == "disk" and self._enable_in_memory_fallback(exc):
                    self.create_tables()
                    return
                raise

    @property
    def storage_notice(self):
        return self._fallback_notice

    @property
    def uses_persistent_storage(self):
        if self._backend == "postgres":
            return True
        return self._storage_mode == "disk" and self.db_path is not None

    @property
    def storage_label(self):
        if self._backend == "postgres":
            return self._redacted_postgres_label()
        if self.uses_persistent_storage:
            return str(self.db_path)
        return "In-memory session store"

    @property
    def storage_backend(self):
        return self._backend

    @property
    def supports_database_download(self):
        return self._backend == "sqlite" and self.uses_persistent_storage and self.db_path is not None

    def _postgres_column_definition(self, definition):
        mapping = {
            "TEXT PRIMARY KEY": "TEXT PRIMARY KEY",
            "TEXT": "TEXT",
            "REAL": "DOUBLE PRECISION",
            "INTEGER": "INTEGER",
        }
        return mapping[definition]

    def _redacted_postgres_label(self):
        try:
            parsed = urlparse(self._postgres_dsn or "")
        except ValueError:
            return "postgresql://configured"
        host = parsed.hostname or "unknown-host"
        port = parsed.port or 5432
        database = parsed.path.lstrip("/") or "unknown-db"
        user = parsed.username or "unknown-user"
        return f"postgresql://{user}@{host}:{port}/{database}"

    def _read_dataframe(self, conn, query, params=None):
        if self._backend == "postgres":
            with conn.cursor() as cursor:
                cursor.execute(query, params or ())
                rows = cursor.fetchall()
                columns = [desc.name for desc in cursor.description] if cursor.description else []
            return pd.DataFrame(rows, columns=columns)
        return pd.read_sql_query(query, conn, params=params)

    def save_analysis(self, data):
        keys = list(data.keys())
        if self._backend == "postgres":
            placeholders = ", ".join(["%s"] * len(keys))
            columns = ", ".join(f'"{key}"' for key in keys)
            update_clause = ", ".join(
                f'"{key}"=EXCLUDED."{key}"' for key in keys if key != "Ticker"
            )
            sql = (
                f'INSERT INTO analysis ({columns}) VALUES ({placeholders}) '
                f'ON CONFLICT("Ticker") DO UPDATE SET {update_clause}'
            )
        else:
            placeholders = ", ".join(["?"] * len(keys))
            columns = ", ".join(keys)
            update_clause = ", ".join(
                f"{key}=excluded.{key}" for key in keys if key != "Ticker"
            )
            sql = (
                f"INSERT INTO analysis ({columns}) VALUES ({placeholders}) "
                f"ON CONFLICT(Ticker) DO UPDATE SET {update_clause}"
            )
        with self._write_lock:
            try:
                with self._connection() as conn:
                    conn.execute(sql, list(data.values()))
            except sqlite3.DatabaseError as exc:
                if self._storage_mode == "disk" and self._enable_in_memory_fallback(exc):
                    with self._connection() as conn:
                        conn.execute(sql, list(data.values()))
                    return
                raise

    def get_analysis(self, ticker):
        query = 'SELECT * FROM analysis WHERE "Ticker"=%s' if self._backend == "postgres" else "SELECT * FROM analysis WHERE Ticker=?"
        try:
            with self._connection() as conn:
                return self._read_dataframe(conn, query, params=(ticker,))
        except (pd.errors.DatabaseError, sqlite3.DatabaseError, psycopg.Error):
            self.create_tables()
            with self._connection() as conn:
                return self._read_dataframe(conn, query, params=(ticker,))

    def get_all_analyses(self):
        try:
            with self._connection() as conn:
                return self._read_dataframe(conn, "SELECT * FROM analysis")
        except (pd.errors.DatabaseError, sqlite3.DatabaseError, psycopg.Error):
            self.create_tables()
            with self._connection() as conn:
                return self._read_dataframe(conn, "SELECT * FROM analysis")


@st.cache_resource
def get_database_manager():
    return DatabaseManager(DATABASE_URL or DB_PATH)


class StockAnalyst:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.last_error = None

    def get_data(self, ticker):
        self.last_error = None
        hist, hist_error = fetch_ticker_history_with_retry(ticker, period="1y")
        if hist is None or hist.empty:
            self.last_error = hist_error or f"Unable to load price history for {ticker}."
            return None, {}, []

        info, info_error = fetch_ticker_info_with_retry(ticker)
        news, news_error = fetch_ticker_news_with_retry(ticker)

        if not info:
            saved = self.db.get_analysis(ticker)
            if not saved.empty:
                info = build_info_fallback_from_saved_analysis(saved.iloc[0])
                if info:
                    if info_error:
                        self.last_error = f"Live profile data was unavailable for {ticker}; reused saved fundamentals."
                elif info_error:
                    self.last_error = info_error
            elif info_error:
                self.last_error = info_error

        if not news and news_error and self.last_error is None:
            self.last_error = news_error

        return hist, info, news

    def analyze_sentiment(self, info, news, price, settings=None):
        info = info or {}
        news = news or []
        recommendation_key = (info.get("recommendationKey") or "").lower()
        analyst_opinions = safe_num(info.get("numberOfAnalystOpinions"))
        target_mean_price = safe_num(info.get("targetMeanPrice"))
        headlines = build_news_context_lines(news, max_items=6)
        context_parts = []
        if recommendation_key:
            context_parts.append(f"Analyst view: {recommendation_key.upper()}")
        if has_numeric_value(analyst_opinions):
            context_parts.append(f"Analyst count: {int(round(analyst_opinions))}")
        if has_numeric_value(target_mean_price):
            context_parts.append(f"Target mean: ${target_mean_price:,.2f}")
        context_parts.extend(headlines)
        summary = " | ".join(context_parts[:6]) if context_parts else "No recent news or analyst context was available."

        return {
            "score": 0,
            "verdict": "CONTEXT ONLY",
            "recommendation_key": recommendation_key.upper() if recommendation_key else "N/A",
            "analyst_opinions": analyst_opinions,
            "target_mean_price": target_mean_price,
            "headline_count": len(headlines),
            "summary": summary,
        }

    def build_record_from_market_data(self, ticker, hist, info, news, settings=None, compute_dcf=False, dcf_settings=None):
        settings = get_model_settings() if settings is None else settings
        dcf_settings = get_dcf_settings() if dcf_settings is None else normalize_dcf_settings(dcf_settings)
        ticker = ticker.strip().upper()
        info = info or {}
        news = news or []
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None

        close = hist["Close"].dropna().astype(float)
        if close.empty:
            return None
        price = float(close.iloc[-1])

        rsi_series = calculate_rsi(close)
        current_rsi = safe_num(rsi_series.iloc[-1])
        previous_rsi = safe_num(rsi_series.iloc[-2]) if len(rsi_series) > 1 else None

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_signal_line = macd_line.ewm(span=9, adjust=False).mean()
        current_macd = safe_num(macd_line.iloc[-1])
        current_macd_signal = safe_num(macd_signal_line.iloc[-1])

        sma50 = safe_num(close.rolling(50, min_periods=50).mean().iloc[-1])
        sma200 = safe_num(close.rolling(200, min_periods=200).mean().iloc[-1])
        momentum_1m = (price / close.iloc[-22] - 1) if len(close) > 22 else None
        momentum_1y = (price / close.iloc[0] - 1) if len(close) > 1 else None
        benchmark_hist, _ = fetch_ticker_history_with_retry(DEFAULT_BENCHMARK_TICKER, period="1y", attempts=2)
        benchmark_close = (
            benchmark_hist["Close"].dropna().astype(float)
            if benchmark_hist is not None and not benchmark_hist.empty and "Close" in benchmark_hist.columns
            else pd.Series(dtype=float)
        )
        relative_strength_metrics = {
            label: compute_relative_strength(close, benchmark_close, window)
            for label, window in BENCHMARK_RELATIVE_STRENGTH_WINDOWS.items()
        }
        volatility_1m = calculate_realized_volatility(close, 22)
        volatility_1y = calculate_realized_volatility(close, 252)
        momentum_1m_risk_adjusted = safe_divide(momentum_1m, (volatility_1m / np.sqrt(12)) if has_numeric_value(volatility_1m) else None)
        range_position_52w, distance_52w_high, distance_52w_low = calculate_52w_context(close)
        trend_strength = calculate_trend_strength(price, sma50, sma200, momentum_1y)
        latest_price_timestamp = close.index[-1] if len(close.index) else None
        latest_news_timestamp = max(
            [extract_news_publish_time(item) for item in news if extract_news_publish_time(item) is not None],
            default=None,
        )
        latest_data_timestamp = max(
            [stamp for stamp in [latest_price_timestamp, latest_news_timestamp] if stamp is not None],
            default=latest_price_timestamp,
        )
        trend_tolerance = settings["tech_trend_tolerance"]
        extension_limit = settings["tech_extension_limit"]

        tech_score = 0
        if has_numeric_value(sma200):
            tech_score += score_trend_distance(price, sma200, trend_tolerance)
        if has_numeric_value(sma50) and has_numeric_value(sma200):
            tech_score += score_trend_distance(sma50, sma200, trend_tolerance / 2)
        if has_numeric_value(sma50):
            tech_score += score_trend_distance(price, sma50, trend_tolerance / 2)
        if has_numeric_value(current_rsi):
            if current_rsi < settings["tech_rsi_oversold"]:
                if has_numeric_value(sma200) and price >= sma200 * 0.95:
                    tech_score += 2
                else:
                    tech_score += 1
            elif current_rsi > settings["tech_rsi_overbought"]:
                if has_numeric_value(sma50) and price <= sma50 * 1.02:
                    tech_score -= 2
                else:
                    tech_score -= 1
        macd_signal = "Neutral"
        if has_numeric_value(current_macd) and has_numeric_value(current_macd_signal):
            if current_macd > current_macd_signal:
                tech_score += 1
                macd_signal = "Bullish Crossover"
            else:
                tech_score -= 1
                macd_signal = "Bearish Crossover"
            if current_macd > 0:
                tech_score += 1
            elif current_macd < 0:
                tech_score -= 1
        if has_numeric_value(momentum_1m):
            if momentum_1m > settings["tech_momentum_threshold"]:
                tech_score += 1
            elif momentum_1m < -settings["tech_momentum_threshold"]:
                tech_score -= 1
        if has_numeric_value(momentum_1m_risk_adjusted):
            if momentum_1m_risk_adjusted >= 0.75:
                tech_score += 1
            elif momentum_1m_risk_adjusted <= -0.75:
                tech_score -= 1
        long_term_momentum_threshold = max(settings["tech_momentum_threshold"] * 3, 0.10)
        if has_numeric_value(momentum_1y):
            if momentum_1y > long_term_momentum_threshold:
                tech_score += 1
            elif momentum_1y < -long_term_momentum_threshold:
                tech_score -= 1
        relative_strength_6m = relative_strength_metrics.get("Relative_Strength_6M")
        if has_numeric_value(relative_strength_6m):
            if relative_strength_6m >= 0.05:
                tech_score += 1
            elif relative_strength_6m <= -0.05:
                tech_score -= 1
        bullish_trend = has_bullish_trend(price, sma50, sma200, momentum_1y)
        bearish_trend = has_bearish_trend(price, sma50, sma200, momentum_1y)
        regime = classify_market_regime(price, sma50, sma200, momentum_1y, tolerance=trend_tolerance)
        overextended = has_numeric_value(sma50) and price >= sma50 * (1 + extension_limit)
        washed_out = has_numeric_value(sma50) and price <= sma50 * (1 - extension_limit)
        pullback_recovery = (
            bullish_trend
            and has_numeric_value(previous_rsi)
            and has_numeric_value(current_rsi)
            and previous_rsi <= settings["tech_rsi_oversold"]
            and current_rsi > settings["tech_rsi_oversold"]
            and has_numeric_value(current_macd)
            and has_numeric_value(current_macd_signal)
            and current_macd >= current_macd_signal
        )
        if pullback_recovery:
            tech_score += 1
        if has_numeric_value(trend_strength):
            if trend_strength >= 30:
                tech_score += 1
            elif trend_strength <= -30:
                tech_score -= 1
        if has_numeric_value(range_position_52w):
            if range_position_52w >= 0.80 and bullish_trend:
                tech_score += 1
            elif range_position_52w <= 0.20 and bearish_trend:
                tech_score -= 1
        if overextended and bullish_trend and has_numeric_value(current_rsi) and current_rsi >= settings["tech_rsi_overbought"] - 5:
            tech_score -= 1
        if washed_out and bearish_trend and has_numeric_value(current_rsi) and current_rsi <= settings["tech_rsi_oversold"] + 5:
            tech_score += 1
        v_tech = score_to_signal(tech_score)

        f_score = 0
        roe = safe_num(info.get("returnOnEquity"))
        margins = safe_num(info.get("profitMargins"))
        debt_eq = safe_num(info.get("debtToEquity"))
        revenue_growth = safe_num(info.get("revenueGrowth"))
        earnings_growth = safe_num(info.get("earningsGrowth"))
        current_ratio = safe_num(info.get("currentRatio"))
        market_cap = safe_num(info.get("marketCap"))
        dividend_yield = safe_num(info.get("dividendYield"))
        payout_ratio = safe_num(info.get("payoutRatio"))
        equity_beta = safe_num(info.get("beta"))
        if has_numeric_value(roe):
            if roe >= settings["fund_roe_threshold"]:
                f_score += 1
            elif roe < 0 or roe < settings["fund_roe_threshold"] * 0.5:
                f_score -= 1
        if has_numeric_value(margins):
            if margins >= settings["fund_profit_margin_threshold"]:
                f_score += 1
            elif margins < 0 or margins < settings["fund_profit_margin_threshold"] * 0.5:
                f_score -= 1
        if has_numeric_value(debt_eq):
            if 0 <= debt_eq < settings["fund_debt_good_threshold"]:
                f_score += 1
            elif debt_eq > settings["fund_debt_bad_threshold"]:
                f_score -= 1
        if has_numeric_value(revenue_growth):
            if revenue_growth >= settings["fund_revenue_growth_threshold"]:
                f_score += 1
            elif revenue_growth < 0:
                f_score -= 1
        if has_numeric_value(earnings_growth):
            if earnings_growth >= settings["fund_revenue_growth_threshold"]:
                f_score += 1
            elif earnings_growth < 0:
                f_score -= 1
        if has_numeric_value(current_ratio):
            if current_ratio >= settings["fund_current_ratio_good"]:
                f_score += 1
            elif current_ratio < settings["fund_current_ratio_bad"]:
                f_score -= 1
        quality_score = calculate_quality_score(
            roe,
            margins,
            debt_eq,
            revenue_growth,
            earnings_growth,
            current_ratio,
            settings,
        )
        if quality_score >= 3:
            f_score += 1
        elif quality_score <= -1.5:
            f_score -= 1
        if f_score >= 4:
            v_fund = "STRONG"
        elif f_score >= 1:
            v_fund = "STABLE"
        else:
            v_fund = "WEAK"

        v_score = 0
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")
        bench, peer_group = build_relative_peer_benchmarks(ticker, info, db=self.db, settings=settings)
        pe = safe_num(info.get("trailingPE"))
        forward_pe = safe_num(info.get("forwardPE"))
        peg_ratio = safe_num(info.get("pegRatio"))
        ps_ratio = safe_num(info.get("priceToSalesTrailing12Months"))
        ev_ebitda = safe_num(info.get("enterpriseToEbitda"))
        pb = safe_num(info.get("priceToBook"))
        valuation_signal_count = 0
        for metric_value, benchmark_value in [
            (pe, bench["PE"]),
            (forward_pe, bench["PE"]),
            (ps_ratio, bench["PS"]),
            (pb, bench["PB"]),
            (ev_ebitda, bench["EV_EBITDA"]),
        ]:
            multiple_score = score_relative_multiple(metric_value, benchmark_value)
            v_score += multiple_score
            if has_numeric_value(metric_value):
                valuation_signal_count += 1
        if has_numeric_value(peg_ratio):
            valuation_signal_count += 1
            if peg_ratio <= 0:
                v_score -= 1
            elif peg_ratio <= settings["valuation_peg_threshold"] * 0.9:
                v_score += 1
            elif peg_ratio >= settings["valuation_peg_threshold"] * 1.35:
                v_score -= 1
        eps = safe_num(info.get("trailingEps"))
        bvps = safe_num(info.get("bookValue"))
        graham_num = None
        intrinsic_value = None
        if has_numeric_value(eps) and has_numeric_value(bvps) and eps > 0 and bvps > 0:
            graham_num = (22.5 * eps * bvps) ** 0.5
            intrinsic_value = graham_num
            valuation_signal_count += 1
            if price < graham_num * 0.85:
                v_score += 2
            elif price < graham_num:
                v_score += 1
            elif price > graham_num * settings["valuation_graham_overpriced_multiple"]:
                v_score -= 2
            elif price > graham_num * 1.15:
                v_score -= 1
        elif has_numeric_value(eps) and eps <= 0:
            v_score -= 1

        dcf_result = {}
        dcf_intrinsic_value = None
        dcf_upside = None
        dcf_last_updated = None
        if compute_dcf:
            dcf_result = build_sec_dcf_model(
                ticker,
                price,
                info,
                dcf_settings=dcf_settings,
                peer_benchmarks=bench,
            )
            if dcf_result.get("available") and has_numeric_value(dcf_result.get("intrinsic_value_per_share")):
                dcf_intrinsic_value = safe_num(dcf_result.get("intrinsic_value_per_share"))
                dcf_upside = safe_num(dcf_result.get("upside"))
                dcf_last_updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        if valuation_signal_count < 2 and v_score < settings["valuation_fair_score_threshold"]:
            v_val = "FAIR VALUE"
        elif v_score >= settings["valuation_under_score_threshold"]:
            v_val = "UNDERVALUED"
        elif v_score >= settings["valuation_fair_score_threshold"]:
            v_val = "FAIR VALUE"
        else:
            v_val = "OVERVALUED"
        valuation_confidence = calculate_valuation_confidence(valuation_signal_count)
        effective_v_score = v_score * (0.45 + 0.55 * valuation_confidence / 100)

        dividend_safety_score = calculate_dividend_safety_score(
            dividend_yield,
            payout_ratio,
            margins,
            current_ratio,
            debt_eq,
        )

        sentiment = self.analyze_sentiment(info, news, price, settings=settings)
        sentiment_conviction = calculate_sentiment_conviction(
            sentiment["score"],
            sentiment["analyst_opinions"],
            sentiment["recommendation_key"],
            sentiment["target_mean_price"],
            price,
            sentiment["headline_count"],
        )
        event_study = compute_event_study(news, hist, benchmark_ticker=DEFAULT_BENCHMARK_TICKER)
        stock_profile = classify_stock_profile(
            sector=sector,
            price=price,
            market_cap=market_cap,
            dividend_yield=dividend_yield,
            payout_ratio=payout_ratio,
            equity_beta=equity_beta,
            analyst_opinions=sentiment["analyst_opinions"],
            pe=pe,
            forward_pe=forward_pe,
            peg_ratio=peg_ratio,
            ps_ratio=ps_ratio,
            pb=pb,
            bench=bench,
            f_score=f_score,
            v_val=v_val,
            revenue_growth=revenue_growth,
            earnings_growth=earnings_growth,
            margins=margins,
            roe=roe,
            current_ratio=current_ratio,
            debt_eq=debt_eq,
            momentum_1y=momentum_1y,
        )
        risk_flags = build_risk_flags(
            eps=eps,
            debt_eq=debt_eq,
            current_ratio=current_ratio,
            overextended=overextended,
            distance_52w_high=distance_52w_high,
            range_position=range_position_52w,
            volatility_1y=volatility_1y,
            stock_profile=stock_profile,
        )
        engine_weights, engine_weight_profile = get_type_adjusted_engine_weights(stock_profile, settings)
        effective_sentiment_score = 0.0
        effective_tech_score = tech_score
        if has_numeric_value(trend_strength):
            effective_tech_score += np.clip(trend_strength / 50, -1.0, 1.0)
        relative_strength_1y = relative_strength_metrics.get("Relative_Strength_1Y")
        if has_numeric_value(relative_strength_1y):
            effective_tech_score += np.clip(relative_strength_1y * 4, -1.0, 1.0)
        effective_f_score = f_score
        if quality_score >= 3:
            effective_f_score += 0.5
        elif quality_score <= -1.5:
            effective_f_score -= 0.5
        if stock_profile["primary_type"] in {"Dividend / Income Stocks", "Defensive Stocks"}:
            effective_f_score += np.clip(dividend_safety_score / 4, -0.5, 1.0)
        if has_numeric_value(event_study.get("avg_abnormal_5d")):
            effective_f_score += np.clip(event_study["avg_abnormal_5d"] * 10, -0.75, 0.75)
        base_overall_score = (
            effective_tech_score * engine_weights["technical"]
            + effective_f_score * engine_weights["fundamental"]
            + effective_v_score * engine_weights["valuation"]
            + effective_sentiment_score * engine_weights["sentiment"]
        )
        base_overall_score -= min(len(risk_flags), 4) * 0.35
        overall_score, base_verdict, type_settings, type_logic_notes = apply_stock_type_framework(
            stock_profile=stock_profile,
            overall_score=base_overall_score,
            tech_score=tech_score,
            f_score=f_score,
            v_score=v_score,
            sentiment_score=sentiment["score"],
            v_fund=v_fund,
            v_val=v_val,
            regime=regime,
            bullish_trend=bullish_trend,
            bearish_trend=bearish_trend,
            data_quality="High",
            momentum_1y=momentum_1y,
            settings=settings,
        )

        assumption_profile = detect_matching_preset(settings)
        assumption_fingerprint = get_assumption_fingerprint(settings)
        assumption_snapshot = serialize_model_settings(settings)
        record = {
            "Ticker": ticker,
            "Price": price,
            "Verdict_Overall": base_verdict,
            "Verdict_Technical": v_tech,
            "Verdict_Fundamental": v_fund,
            "Verdict_Valuation": v_val,
            "Verdict_Sentiment": sentiment["verdict"],
            "Market_Regime": regime,
            "Score_Tech": tech_score,
            "Score_Fund": f_score,
            "Score_Val": v_score,
            "Score_Sentiment": sentiment["score"],
            "Sector": sector,
            "Industry": industry,
            "Stock_Type": stock_profile["primary_type"],
            "Cap_Bucket": stock_profile["cap_bucket"],
            "Style_Tags": stock_profile["style_tags"],
            "Type_Strategy": stock_profile["type_strategy"],
            "Type_Confidence": stock_profile["type_confidence"],
            "Engine_Weight_Profile": engine_weight_profile,
            "Peer_Count": peer_group.get("count"),
            "Peer_Group_Label": peer_group.get("group_label"),
            "Peer_Tickers": ", ".join(peer_group.get("tickers", [])),
            "Peer_Summary": peer_group.get("summary"),
            "Peer_Comparison": json.dumps(peer_group),
            "Market_Cap": market_cap,
            "Dividend_Yield": dividend_yield,
            "Payout_Ratio": payout_ratio,
            "Equity_Beta": equity_beta,
            "Relative_Strength_3M": relative_strength_metrics.get("Relative_Strength_3M"),
            "Relative_Strength_6M": relative_strength_metrics.get("Relative_Strength_6M"),
            "Relative_Strength_1Y": relative_strength_metrics.get("Relative_Strength_1Y"),
            "Trend_Strength": trend_strength,
            "Range_Position_52W": range_position_52w,
            "Distance_52W_High": distance_52w_high,
            "Distance_52W_Low": distance_52w_low,
            "Volatility_1M": volatility_1m,
            "Volatility_1Y": volatility_1y,
            "Momentum_1M_Risk_Adjusted": momentum_1m_risk_adjusted,
            "Quality_Score": quality_score,
            "Dividend_Safety_Score": dividend_safety_score,
            "Valuation_Signal_Count": valuation_signal_count,
            "Valuation_Confidence": valuation_confidence,
            "Sentiment_Conviction": sentiment_conviction,
            "Risk_Flags": " | ".join(risk_flags),
            "PE_Ratio": pe,
            "Forward_PE": forward_pe,
            "PEG_Ratio": peg_ratio,
            "PS_Ratio": ps_ratio,
            "PB_Ratio": pb,
            "EV_EBITDA": ev_ebitda,
            "Graham_Number": graham_num if graham_num else 0,
            "Intrinsic_Value": intrinsic_value if intrinsic_value else 0,
            "DCF_Intrinsic_Value": dcf_intrinsic_value if has_numeric_value(dcf_intrinsic_value) else None,
            "DCF_Upside": dcf_upside,
            "DCF_WACC": dcf_result.get("wacc"),
            "DCF_Risk_Free_Rate": dcf_result.get("risk_free_rate"),
            "DCF_Beta": dcf_result.get("beta"),
            "DCF_Cost_of_Equity": dcf_result.get("cost_of_equity"),
            "DCF_Cost_of_Debt": dcf_result.get("after_tax_cost_of_debt"),
            "DCF_Equity_Weight": dcf_result.get("equity_weight"),
            "DCF_Debt_Weight": dcf_result.get("debt_weight"),
            "DCF_Growth_Rate": dcf_result.get("selected_growth_rate"),
            "DCF_Terminal_Growth": dcf_result.get("terminal_growth_rate"),
            "DCF_Base_FCF": dcf_result.get("base_fcf"),
            "DCF_Enterprise_Value": dcf_result.get("enterprise_value"),
            "DCF_Equity_Value": dcf_result.get("equity_value"),
            "DCF_Historical_FCF_Growth": dcf_result.get("historical_fcf_growth"),
            "DCF_Historical_Revenue_Growth": dcf_result.get("historical_revenue_growth"),
            "DCF_Guidance_Growth": dcf_result.get("guidance_growth_rate"),
            "DCF_Source": dcf_result.get("growth_source", "Unavailable"),
            "DCF_Confidence": dcf_result.get("growth_confidence", "low"),
            "DCF_History": json.dumps(dcf_result.get("history", [])),
            "DCF_Projection": json.dumps(dcf_result.get("projection", [])),
            "DCF_Sensitivity": json.dumps(dcf_result.get("sensitivity", [])),
            "DCF_Guidance_Excerpts": json.dumps(dcf_result.get("guidance_excerpts", [])),
            "DCF_Guidance_Summary": dcf_result.get("guidance_summary") or dcf_result.get("error", ""),
            "DCF_Filing_Form": dcf_result.get("filing_form"),
            "DCF_Filing_Date": dcf_result.get("filing_date"),
            "DCF_Last_Updated": dcf_last_updated,
            "DCF_Assumptions": serialize_dcf_settings(dcf_settings),
            "Profit_Margins": margins,
            "ROE": roe,
            "Debt_to_Equity": debt_eq,
            "Revenue_Growth": revenue_growth,
            "Current_Ratio": current_ratio,
            "Target_Mean_Price": sentiment["target_mean_price"],
            "Recommendation_Key": sentiment["recommendation_key"],
            "Analyst_Opinions": sentiment["analyst_opinions"],
            "Sentiment_Headline_Count": sentiment["headline_count"],
            "Sentiment_Summary": sentiment["summary"],
            "Event_Study_Count": event_study.get("count"),
            "Event_Study_Avg_Abnormal_1D": event_study.get("avg_abnormal_1d"),
            "Event_Study_Avg_Abnormal_5D": event_study.get("avg_abnormal_5d"),
            "Event_Study_Summary": event_study.get("summary"),
            "Event_Study_Events": json.dumps(event_study.get("events", [])),
            "RSI": current_rsi,
            "MACD_Value": current_macd,
            "MACD_Signal": macd_signal,
            "SMA_Status": (
                "Bullish" if regime == "Bullish Trend"
                else "Bearish" if regime == "Bearish Trend"
                else "Neutral"
            ),
            "Momentum_1M": momentum_1m,
            "Momentum_1Y": momentum_1y,
            "Last_Data_Update": format_datetime_value(latest_data_timestamp),
            "Last_Updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Overall_Score": overall_score,
            "Assumption_Profile": assumption_profile,
            "Assumption_Fingerprint": assumption_fingerprint,
            "Assumption_Drift": calculate_assumption_drift(settings),
            "Assumption_Snapshot": assumption_snapshot,
        }
        completeness, missing_count, quality_label = assess_record_quality(record)
        bias_summary = summarize_engine_biases(
            tech_score,
            f_score,
            v_score,
            sentiment["score"],
            v_val,
            bullish_trend,
            bearish_trend,
        )
        decision_confidence = compute_decision_confidence(
            overall_score=overall_score,
            bias_summary=bias_summary,
            regime=regime,
            completeness=completeness,
        )
        decision_confidence = adjust_type_based_confidence(decision_confidence, stock_profile, quality_label)
        if has_numeric_value(trend_strength):
            decision_confidence += np.clip(trend_strength / 12, -4, 6)
        if has_numeric_value(quality_score):
            decision_confidence += np.clip(quality_score * 2.5, -5, 8)
        if has_numeric_value(valuation_confidence):
            decision_confidence += np.clip((valuation_confidence - 50) / 8, -4, 5)
        if has_numeric_value(sentiment_conviction):
            decision_confidence += np.clip((sentiment_conviction - 50) / 10, -3, 4)
        decision_confidence -= min(len(risk_flags), 5) * 2.0
        decision_confidence = float(np.clip(round(decision_confidence, 1), 5.0, 95.0))
        final_verdict = apply_confidence_guard(base_verdict, decision_confidence, quality_label, type_settings)
        record["Verdict_Overall"] = final_verdict
        record["Decision_Confidence"] = decision_confidence
        base_decision_notes = build_decision_notes(
            verdict=final_verdict,
            regime=regime,
            bias_summary=bias_summary,
            confidence=decision_confidence,
            data_quality=quality_label,
            current_rsi=current_rsi,
            v_val=v_val,
            v_fund=v_fund,
            bullish_trend=bullish_trend,
            bearish_trend=bearish_trend,
            overextended=overextended,
            pullback_recovery=pullback_recovery,
        )
        decision_note_parts = [
            f"Type: {stock_profile['primary_type']}",
            f"Cap: {stock_profile['cap_bucket']}",
        ]
        if stock_profile.get("classification_summary"):
            decision_note_parts.append(stock_profile["classification_summary"])
        decision_note_parts.extend(type_logic_notes[:2])
        if risk_flags:
            decision_note_parts.append("Risks: " + ", ".join(risk_flags[:3]))
        if base_decision_notes:
            decision_note_parts.extend(base_decision_notes.split(" | "))
        deduped_notes = []
        for note in decision_note_parts:
            cleaned_note = str(note).strip()
            if cleaned_note and cleaned_note not in deduped_notes:
                deduped_notes.append(cleaned_note)
        record["Decision_Notes"] = " | ".join(deduped_notes[:5])
        record["Data_Completeness"] = completeness
        record["Missing_Metric_Count"] = missing_count
        record["Data_Quality"] = quality_label
        return record

    def analyze(self, ticker, settings=None, persist=True, preloaded=None, compute_dcf=False, dcf_settings=None):
        active_settings = get_model_settings() if settings is None else settings
        ticker = ticker.strip().upper()
        self.last_error = None
        existing_row = None
        if persist:
            existing = self.db.get_analysis(ticker)
            if not existing.empty:
                existing_row = existing.iloc[0].to_dict()
        if preloaded is None:
            hist, info, news = self.get_data(ticker)
        else:
            hist, info, news = preloaded

        record = self.build_record_from_market_data(
            ticker,
            hist,
            info,
            news,
            settings=active_settings,
            compute_dcf=compute_dcf,
            dcf_settings=dcf_settings,
        )
        if record is None and self.last_error is None:
            self.last_error = (
                f"Unable to build an analysis for {ticker}. Yahoo returned incomplete or unusable market data."
            )
        if record is None and persist:
            if existing_row is not None:
                self.last_error = (
                    f"Live fetch failed for {ticker}; showing the most recent saved analysis instead."
                )
                return existing_row
        if record and persist:
            if existing_row and (not compute_dcf or not has_dcf_snapshot(record)):
                record.update(extract_dcf_fields(existing_row))
            self.db.save_analysis(record)
        return record


class PortfolioAnalyst:
    def __init__(self, db):
        self.db = db
        self.last_error = None

    def get_price_history(self, tickers, benchmark_ticker, period):
        download_list = list(dict.fromkeys(tickers + [benchmark_ticker]))
        raw, download_error = fetch_batch_history_with_retry(download_list, period=period)
        if raw is None or raw.empty:
            self.last_error = download_error or "Unable to download portfolio price history."
            return None, None

        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" not in raw.columns.get_level_values(0):
                self.last_error = "Downloaded portfolio data did not include close prices."
                return None, None
            close_prices = raw["Close"].copy()
        else:
            if isinstance(raw, pd.Series):
                close_prices = raw.to_frame(name=download_list[0])
            elif "Close" in raw.columns:
                close_prices = raw[["Close"]].copy()
                close_prices.columns = [download_list[0]]
            elif set(download_list).issubset(set(raw.columns)):
                close_prices = raw.copy()
            else:
                self.last_error = "Downloaded portfolio data did not include a Close column."
                return None, None

        if isinstance(close_prices, pd.Series):
            close_prices = close_prices.to_frame(name=download_list[0])
        close_prices = close_prices.sort_index().ffill(limit=3)

        available_assets = [
            ticker for ticker in tickers
            if ticker in close_prices.columns and close_prices[ticker].dropna().shape[0] >= 30
        ]
        missing_assets = [ticker for ticker in tickers if ticker not in available_assets]
        if benchmark_ticker not in close_prices.columns or close_prices[benchmark_ticker].dropna().shape[0] < 30:
            self.last_error = f"The benchmark {benchmark_ticker} does not have enough usable history for {period}."
            return None, None
        if len(available_assets) < 2:
            missing_text = ", ".join(missing_assets) if missing_assets else "the selected tickers"
            self.last_error = (
                f"Need at least two tickers with usable {period} history. Missing or too short: {missing_text}."
            )
            return None, None

        combined_columns = list(dict.fromkeys(available_assets + [benchmark_ticker]))
        aligned_prices = close_prices[combined_columns].dropna()
        if aligned_prices.empty or len(aligned_prices) < 30:
            self.last_error = (
                "The selected names do not share enough overlapping history for this lookback window. "
                "Try a shorter period or remove newer tickers."
            )
            return None, None

        return aligned_prices[available_assets], aligned_prices[benchmark_ticker]

    def get_asset_metadata(self, tickers):
        rows = []
        for ticker in tickers:
            name = ticker
            sector = "Unknown"
            cached = self.db.get_analysis(ticker)
            if not cached.empty:
                cached_row = cached.iloc[0]
                if not pd.isna(cached_row.get("Sector")):
                    sector = cached_row.get("Sector") or sector

            info, _ = fetch_ticker_info_with_retry(ticker)
            if info:
                name = info.get("shortName") or info.get("longName") or ticker
                sector = info.get("sector") or sector

            rows.append({"Ticker": ticker, "Name": name, "Sector": sector})

        return pd.DataFrame(rows)

    def calculate_asset_metrics(self, asset_returns, benchmark_returns, risk_free_rate, trading_days):
        risk_free_daily = risk_free_rate / trading_days
        annual_return = asset_returns.mean() * trading_days
        annual_volatility = asset_returns.std() * np.sqrt(trading_days)
        downside_diff = (asset_returns - risk_free_daily).clip(upper=0)
        downside_volatility = np.sqrt((downside_diff.pow(2)).mean()) * np.sqrt(trading_days)

        benchmark_var = benchmark_returns.var()
        betas = asset_returns.apply(lambda series: safe_divide(series.cov(benchmark_returns), benchmark_var))

        metrics = pd.DataFrame(
            {
                "Annual Return": annual_return,
                "Volatility": annual_volatility,
                "Downside Volatility": downside_volatility,
                "Beta": betas,
            }
        )
        metrics["Sharpe Ratio"] = (metrics["Annual Return"] - risk_free_rate) / metrics["Volatility"]
        metrics["Sortino Ratio"] = (metrics["Annual Return"] - risk_free_rate) / metrics["Downside Volatility"]
        metrics["Treynor Ratio"] = (metrics["Annual Return"] - risk_free_rate) / metrics["Beta"]
        metrics = metrics.replace([np.inf, -np.inf], np.nan)
        metrics.index.name = "Ticker"
        return metrics.reset_index()

    def calculate_portfolio_metrics(self, asset_returns, benchmark_returns, weights, risk_free_rate, trading_days):
        risk_free_daily = risk_free_rate / trading_days
        portfolio_returns = asset_returns @ weights
        annual_return = portfolio_returns.mean() * trading_days
        volatility = portfolio_returns.std() * np.sqrt(trading_days)
        downside_diff = (portfolio_returns - risk_free_daily).clip(upper=0)
        downside_volatility = np.sqrt((downside_diff.pow(2)).mean()) * np.sqrt(trading_days)
        beta = safe_divide(portfolio_returns.cov(benchmark_returns), benchmark_returns.var())
        sharpe = safe_divide(annual_return - risk_free_rate, volatility)
        sortino = safe_divide(annual_return - risk_free_rate, downside_volatility)
        treynor = safe_divide(annual_return - risk_free_rate, beta)

        return {
            "Return": annual_return,
            "Volatility": volatility,
            "Downside Volatility": downside_volatility,
            "Beta": beta,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Treynor": treynor,
        }

    def simulate_portfolios(self, asset_returns, benchmark_returns, risk_free_rate, max_weight, simulations, trading_days):
        rng = np.random.default_rng(42)
        tickers = list(asset_returns.columns)
        portfolios = []

        for _ in range(simulations):
            weights = cap_weights(rng.random(len(tickers)), max_weight)
            metrics = self.calculate_portfolio_metrics(
                asset_returns,
                benchmark_returns,
                weights,
                risk_free_rate,
                trading_days,
            )
            row = {**metrics}
            for ticker, weight in zip(tickers, weights):
                row[f"W_{ticker}"] = weight
            portfolios.append(row)

        portfolio_df = pd.DataFrame(portfolios).replace([np.inf, -np.inf], np.nan).dropna(subset=["Return", "Volatility", "Sharpe"])
        if portfolio_df.empty:
            return None, None, None, None, None

        frontier_rows = []
        best_return = -np.inf
        for _, row in portfolio_df.sort_values("Volatility").iterrows():
            if row["Return"] > best_return:
                frontier_rows.append(row)
                best_return = row["Return"]

        frontier = pd.DataFrame(frontier_rows)
        tangent = portfolio_df.loc[portfolio_df["Sharpe"].idxmax()]
        minimum_volatility = portfolio_df.loc[portfolio_df["Volatility"].idxmin()]

        cal_x = np.linspace(0, max(frontier["Volatility"].max(), tangent["Volatility"]) * 1.2, 60)
        cal_y = risk_free_rate + tangent["Sharpe"] * cal_x
        cal = pd.DataFrame({"Volatility": cal_x, "Return": cal_y})
        return portfolio_df, frontier, tangent, minimum_volatility, cal

    def build_recommendations(self, tickers, asset_metrics, metadata, tangent_portfolio):
        weight_map = {
            ticker: tangent_portfolio.get(f"W_{ticker}", 0.0)
            for ticker in tickers
        }
        equal_weight = 1 / len(tickers)
        recommendations = asset_metrics.merge(metadata, on="Ticker", how="left")
        recommendations["Recommended Weight"] = recommendations["Ticker"].map(weight_map).fillna(0.0)
        recommendations["Weight vs Equal"] = recommendations["Recommended Weight"] - equal_weight

        def classify_role(row):
            if row["Recommended Weight"] >= max(equal_weight * 1.35, 0.18):
                return "Core holding"
            if row["Recommended Weight"] >= equal_weight * 0.9:
                return "Supporting allocation"
            if row["Beta"] is not None and not pd.isna(row["Beta"]) and row["Beta"] < 0.9:
                return "Diversifier"
            return "Satellite position"

        def build_reason(row):
            reasons = []
            if not pd.isna(row["Sharpe Ratio"]) and row["Sharpe Ratio"] >= 1:
                reasons.append("strong Sharpe")
            if not pd.isna(row["Sortino Ratio"]) and row["Sortino Ratio"] >= 1:
                reasons.append("good downside efficiency")
            if not pd.isna(row["Treynor Ratio"]) and row["Treynor Ratio"] > 0:
                reasons.append("positive Treynor")
            if not pd.isna(row["Beta"]) and row["Beta"] < 0.9:
                reasons.append("helps diversify beta")
            return ", ".join(reasons) if reasons else "kept for balance and exposure"

        recommendations["Role"] = recommendations.apply(classify_role, axis=1)
        recommendations["Rationale"] = recommendations.apply(build_reason, axis=1)
        recommendations = recommendations.sort_values("Recommended Weight", ascending=False)
        return recommendations

    def build_portfolio_notes(self, recommendations, sector_exposure, tangent_portfolio, max_weight):
        notes = []
        effective_names = safe_divide(1, np.square(recommendations["Recommended Weight"]).sum())
        largest_position = recommendations.iloc[0]
        largest_sector = sector_exposure.iloc[0]

        if len(recommendations) < 5:
            notes.append("Fewer than five holdings means this portfolio is still fairly concentrated.")
        if largest_position["Recommended Weight"] >= max_weight * 0.95:
            notes.append(f"{largest_position['Ticker']} is pressing against the max position size, which signals strong conviction but higher single-name risk.")
        if largest_sector["Recommended Weight"] > 0.45:
            notes.append(f"{largest_sector['Sector']} is more than 45% of the allocation, so sector risk is elevated.")
        if effective_names is not None and effective_names < 4:
            notes.append("Effective diversification is low; the weights behave like fewer than four equally sized names.")
        if tangent_portfolio["Beta"] is not None and tangent_portfolio["Beta"] > 1.1:
            notes.append("The recommended portfolio is more aggressive than the benchmark on a beta basis.")
        if not notes:
            notes.append("The allocation is reasonably balanced across names and does not show a major concentration warning.")

        return notes, effective_names

    def analyze_portfolio(self, tickers, benchmark_ticker, period, risk_free_rate, max_weight, simulations):
        self.last_error = None
        settings = get_model_settings()
        trading_days = settings["trading_days_per_year"]
        if len(tickers) * max_weight < 1:
            self.last_error = "The max single-stock weight is too low for the number of requested names."
            return None

        try:
            asset_prices, benchmark_prices = self.get_price_history(tickers, benchmark_ticker, period)
            if asset_prices is None or benchmark_prices is None:
                return None

            asset_returns = asset_prices.pct_change().dropna()
            benchmark_returns = benchmark_prices.pct_change().dropna()
            common_index = asset_returns.index.intersection(benchmark_returns.index)
            asset_returns = asset_returns.loc[common_index]
            benchmark_returns = benchmark_returns.loc[common_index]
            if asset_returns.empty or len(asset_returns.columns) < 2:
                self.last_error = (
                    "The selected basket did not produce enough overlapping return history to build a portfolio recommendation."
                )
                return None

            asset_metrics = self.calculate_asset_metrics(asset_returns, benchmark_returns, risk_free_rate, trading_days)
            portfolio_df, frontier, tangent, minimum_volatility, cal = self.simulate_portfolios(
                asset_returns,
                benchmark_returns,
                risk_free_rate,
                max_weight,
                simulations,
                trading_days,
            )
            if portfolio_df is None:
                self.last_error = (
                    "Portfolio simulation could not find enough valid portfolios. Try fewer tickers, a shorter window, or a higher max weight."
                )
                return None

            valid_tickers = list(asset_returns.columns)
            metadata = self.get_asset_metadata(valid_tickers)
            recommendations = self.build_recommendations(valid_tickers, asset_metrics, metadata, tangent)
            sector_exposure = (
                recommendations.groupby("Sector", dropna=False)["Recommended Weight"]
                .sum()
                .reset_index()
                .sort_values("Recommended Weight", ascending=False)
            )
            notes, effective_names = self.build_portfolio_notes(recommendations, sector_exposure, tangent, max_weight)

            return {
                "asset_metrics": asset_metrics,
                "portfolio_cloud": portfolio_df,
                "frontier": frontier,
                "tangent": tangent,
                "minimum_volatility": minimum_volatility,
                "cal": cal,
                "recommendations": recommendations,
                "sector_exposure": sector_exposure,
                "notes": notes,
                "effective_names": effective_names,
                "benchmark": benchmark_ticker,
                "period": period,
            }
        except Exception as exc:
            self.last_error = f"Portfolio analysis hit an upstream or data-shape error: {summarize_fetch_error(exc)}"
            return None


def render_frontier_chart(portfolio_cloud, frontier, cal, tangent, minimum_volatility):
    cloud = portfolio_cloud[["Volatility", "Return", "Sharpe"]].copy()
    cloud["Series"] = "Simulated Portfolios"

    frontier_data = frontier[["Volatility", "Return"]].copy()
    frontier_data["Sharpe"] = np.nan
    frontier_data["Series"] = "Efficient Frontier"

    cal_data = cal.copy()
    cal_data["Sharpe"] = np.nan
    cal_data["Series"] = "CAL"

    marker_data = pd.DataFrame(
        [
            {
                "Volatility": tangent["Volatility"],
                "Return": tangent["Return"],
                "Sharpe": tangent["Sharpe"],
                "Series": "Max Sharpe",
            },
            {
                "Volatility": minimum_volatility["Volatility"],
                "Return": minimum_volatility["Return"],
                "Sharpe": minimum_volatility["Sharpe"],
                "Series": "Min Volatility",
            },
        ]
    )

    chart_data = pd.concat([cloud, frontier_data, cal_data, marker_data], ignore_index=True)
    spec = {
        "layer": [
            {
                "transform": [{"filter": "datum.Series == 'Simulated Portfolios'"}],
                "mark": {"type": "circle", "opacity": 0.35, "size": 45},
                "encoding": {
                    "x": {"field": "Volatility", "type": "quantitative", "title": "Annualized Volatility"},
                    "y": {"field": "Return", "type": "quantitative", "title": "Annualized Return"},
                    "color": {"field": "Sharpe", "type": "quantitative", "title": "Sharpe"},
                },
            },
            {
                "transform": [{"filter": "datum.Series == 'Efficient Frontier'"}],
                "mark": {"type": "line", "strokeWidth": 3, "color": "#0b7285"},
                "encoding": {
                    "x": {"field": "Volatility", "type": "quantitative"},
                    "y": {"field": "Return", "type": "quantitative"},
                },
            },
            {
                "transform": [{"filter": "datum.Series == 'CAL'"}],
                "mark": {"type": "line", "strokeWidth": 2, "strokeDash": [6, 4], "color": "#e03131"},
                "encoding": {
                    "x": {"field": "Volatility", "type": "quantitative"},
                    "y": {"field": "Return", "type": "quantitative"},
                },
            },
            {
                "transform": [{"filter": "datum.Series == 'Max Sharpe'"}],
                "mark": {"type": "point", "filled": True, "size": 170, "shape": "diamond", "color": "#2b8a3e"},
                "encoding": {
                    "x": {"field": "Volatility", "type": "quantitative"},
                    "y": {"field": "Return", "type": "quantitative"},
                },
            },
            {
                "transform": [{"filter": "datum.Series == 'Min Volatility'"}],
                "mark": {"type": "point", "filled": True, "size": 170, "shape": "square", "color": "#f08c00"},
                "encoding": {
                    "x": {"field": "Volatility", "type": "quantitative"},
                    "y": {"field": "Return", "type": "quantitative"},
                },
            },
        ]
    }
    st.vega_lite_chart(chart_data, spec, width="stretch")


st.set_page_config(page_title="ZB Compiler", layout="wide", page_icon="SE")

db = get_database_manager()
bot = StockAnalyst(db)
portfolio_bot = PortfolioAnalyst(db)
model_settings = get_model_settings()
active_preset_name = detect_matching_preset(model_settings)
active_assumption_fingerprint = get_assumption_fingerprint(model_settings)

st.title("ZB Compiler")
st.caption(f"Version: {APP_VERSION}")
if db.storage_notice:
    st.warning(db.storage_notice)

startup_refresh_summary = {
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
if RUN_STARTUP_REFRESH and os.environ.get("STOCK_ENGINE_SKIP_STARTUP_REFRESH") != "1":
    startup_badge = st.empty()
    startup_refresh_summary = refresh_saved_analyses_on_launch(db, model_settings, badge_placeholder=startup_badge)
    startup_badge.empty()

sensitivity_default_ticker = st.session_state.get("sensitivity_last_ticker") or st.session_state.get("single_ticker", "")
backtest_default_ticker = st.session_state.get("backtest_last_ticker") or st.session_state.get("single_ticker", "")

stock_tab, compare_tab, portfolio_tab, sensitivity_tab, backtest_tab, library_tab, readme_tab, changelog_tab, methodology_tab, options_tab = st.tabs(
    ["Stock Analysis", "Compare", "Portfolio", "Sensitivity", "Backtest", "Library", "ReadMe / Usage", "Changelog", "Methodology", "Options"]
)

with stock_tab:
    c1, c2 = st.columns([3, 1])
    with c1:
        txt_input = st.text_input("Enter Ticker Symbol (e.g., AAPL, NVDA, F)", "", key="single_ticker")
    with c2:
        st.write("")
        st.write("")
        if st.button("Run Full Analysis", type="primary", width="stretch"):
            if txt_input:
                with st.spinner(f"Running multiple engines on {txt_input}..."):
                    res = bot.analyze(txt_input)
                    if not res:
                        st.error(bot.last_error or "Unable to fetch enough market data for this ticker right now.")

    if txt_input:
        df = prepare_analysis_dataframe(db.get_analysis(txt_input.upper()))
        if not df.empty:
            row = df.iloc[0]
            peer_comparison = safe_json_loads(row.get("Peer_Comparison"), default={})
            peer_bench = peer_comparison.get("benchmarks", {}) if isinstance(peer_comparison, dict) else {}
            if not peer_bench:
                peer_bench = get_sector_benchmarks(row["Sector"], model_settings)
            peer_rows = pd.DataFrame(peer_comparison.get("rows", [])) if isinstance(peer_comparison, dict) else pd.DataFrame()
            event_study_events = pd.DataFrame(safe_json_loads(row.get("Event_Study_Events"), default=[]))
            dcf_assumptions = safe_json_loads(row.get("DCF_Assumptions"), default={})
            dcf_feedback = st.session_state.get("dcf_action_feedback")
            if isinstance(dcf_feedback, dict) and dcf_feedback.get("ticker") == str(row["Ticker"]):
                if dcf_feedback.get("kind") == "success":
                    st.success(str(dcf_feedback.get("message")))
                elif dcf_feedback.get("kind") == "warning":
                    st.warning(str(dcf_feedback.get("message")))
                st.session_state.pop("dcf_action_feedback", None)

            st.divider()
            col_main_1, col_main_2, col_main_3 = st.columns([1, 2, 1])
            with col_main_1:
                st.metric("Current Price", f"${row['Price']:,.2f}")
            with col_main_2:
                verdict_text = colorize_markdown_text(
                    f"VERDICT: {row['Verdict_Overall']}",
                    get_color(row["Verdict_Overall"]),
                )
                st.markdown(f"## {verdict_text}")
            with col_main_3:
                st.metric("Last Data Update", str(row.get("Last_Data_Update") or "Unknown"))
            st.caption(
                f"Sector: {row.get('Sector', 'Unknown')} | Industry: {row.get('Industry', 'Unknown')}"
            )

            render_analysis_signal_cards(
                [
                    {
                        "label": "Stock Type",
                        "value": str(row.get("Stock_Type", "Legacy")),
                        "note": "The stock category the model thinks fits best right now.",
                        "tone": "neutral",
                        "help": ANALYSIS_HELP_TEXT["Stock Type"],
                    },
                    {
                        "label": "Cap Bucket",
                        "value": str(row.get("Cap_Bucket", "Unknown")),
                        "note": "A quick size label based on the company's market value.",
                        "tone": "neutral",
                        "help": ANALYSIS_HELP_TEXT["Cap Bucket"],
                    },
                    {
                        "label": "Type Confidence",
                        "value": format_value(row.get("Type_Confidence"), "{:,.0f}", "/100"),
                        "note": "Higher numbers mean the model sees a cleaner fit.",
                        "tone": tone_from_metric_threshold(row.get("Type_Confidence"), good_min=70, bad_max=45),
                        "help": ANALYSIS_HELP_TEXT["Type Confidence"],
                    },
                    {
                        "label": "Market Cap",
                        "value": format_market_cap(row.get("Market_Cap")),
                        "note": "The market's current estimate of the company's total equity value.",
                        "tone": "neutral",
                        "help": ANALYSIS_HELP_TEXT["Market Cap"],
                    },
                ],
                columns=4,
            )
            if row.get("Style_Tags"):
                st.caption(f"This stock currently reads as: {row.get('Style_Tags')}")
            if row.get("Type_Strategy"):
                st.caption(f"The model's default playbook for this kind of stock: {row.get('Type_Strategy')}")

            st.subheader("Method Breakdown")
            st.info("This section shows how each part of the model is reading the stock right now, so you can see what is helping or hurting the final verdict.")
            st.caption("These results stay tied to the settings that were active when the analysis was run. If you changed something in Options, rerun the ticker to refresh this snapshot.")

            render_analysis_signal_cards(
                [
                    {
                        "label": "Technical",
                        "value": format_int(row["Score_Tech"]),
                        "note": "Price action, momentum, and trend signals.",
                        "tone": tone_from_metric_threshold(row["Score_Tech"], good_min=1, bad_max=-1),
                        "help": ANALYSIS_HELP_TEXT["Technical"],
                    },
                    {
                        "label": "Fundamental",
                        "value": format_int(row["Score_Fund"]),
                        "note": "Business strength, growth, and balance-sheet signals.",
                        "tone": tone_from_metric_threshold(row["Score_Fund"], good_min=1, bad_max=-1),
                        "help": ANALYSIS_HELP_TEXT["Fundamental"],
                    },
                    {
                        "label": "Valuation",
                        "value": format_int(row["Score_Val"]),
                        "note": "How cheap or expensive the stock looks versus its closest peer set.",
                        "tone": tone_from_metric_threshold(row["Score_Val"], good_min=1, bad_max=-1),
                        "help": ANALYSIS_HELP_TEXT["Valuation"],
                    },
                    {
                        "label": "Sentiment Context",
                        "value": format_int(row["Sentiment_Headline_Count"]),
                        "note": "Context only: recent headlines and analyst metadata with no directional score applied.",
                        "tone": "neutral",
                        "help": ANALYSIS_HELP_TEXT["Sentiment"],
                    },
                    {
                        "label": "Updated",
                        "value": str(row["Last_Updated"]),
                        "note": "When this saved analysis was last refreshed.",
                        "tone": "neutral",
                        "help": ANALYSIS_HELP_TEXT["Updated"],
                    },
                ],
                columns=5,
            )

            render_analysis_signal_cards(
                [
                    {
                        "label": "Overall Score",
                        "value": format_value(row.get("Overall_Score"), "{:,.1f}"),
                        "note": "The model's combined read after blending all engines.",
                        "tone": tone_from_metric_threshold(row.get("Overall_Score"), good_min=1, bad_max=-1),
                        "help": ANALYSIS_HELP_TEXT["Overall Score"],
                    },
                    {
                        "label": "Data Quality",
                        "value": str(row.get("Data_Quality", "Unknown")),
                        "note": "How complete and usable the source data was.",
                        "tone": tone_from_quality_label(row.get("Data_Quality", "Unknown")),
                        "help": ANALYSIS_HELP_TEXT["Data Quality"],
                    },
                    {
                        "label": "Assumption Profile",
                        "value": str(row.get("Assumption_Profile", "Legacy")),
                        "note": "The preset or custom settings used for this run.",
                        "tone": "neutral",
                        "help": ANALYSIS_HELP_TEXT["Assumption Profile"],
                    },
                    {
                        "label": "Missing Metrics",
                        "value": format_int(row.get("Missing_Metric_Count")),
                        "note": "How many important data points were unavailable.",
                        "tone": tone_from_metric_threshold(row.get("Missing_Metric_Count"), good_max=1, bad_min=5),
                        "help": ANALYSIS_HELP_TEXT["Missing Metrics"],
                    },
                    {
                        "label": "Consistency",
                        "value": format_value(row.get("Decision_Confidence"), "{:,.0f}", "/100"),
                        "note": "How consistently the model's signals lined up on this run.",
                        "tone": tone_from_metric_threshold(row.get("Decision_Confidence"), good_min=70, bad_max=45),
                        "help": ANALYSIS_HELP_TEXT["Consistency"],
                    },
                    {
                        "label": "Regime",
                        "value": str(row.get("Market_Regime", "Unknown")),
                        "note": "The market backdrop the model sees in the chart.",
                        "tone": tone_from_regime(row.get("Market_Regime", "Unknown")),
                        "help": ANALYSIS_HELP_TEXT["Regime"],
                    },
                ],
                columns=6,
            )
            st.caption(f"Fingerprint: {row.get('Assumption_Fingerprint', 'Legacy')}")
            render_analysis_signal_cards(
                [
                    {
                        "label": "Trend Strength",
                        "value": format_value(row.get("Trend_Strength"), "{:,.0f}"),
                        "note": "A quick read on how healthy the long-term trend looks.",
                        "tone": tone_from_metric_threshold(row.get("Trend_Strength"), good_min=20, bad_max=-20),
                        "help": ANALYSIS_HELP_TEXT["Trend Strength"],
                    },
                    {
                        "label": "Quality Score",
                        "value": format_value(row.get("Quality_Score"), "{:,.1f}"),
                        "note": "A shorthand measure of business durability.",
                        "tone": tone_from_metric_threshold(row.get("Quality_Score"), good_min=2, bad_max=0),
                        "help": ANALYSIS_HELP_TEXT["Quality Score"],
                    },
                    {
                        "label": "Dividend Safety",
                        "value": format_value(row.get("Dividend_Safety_Score"), "{:,.1f}"),
                        "note": "A rough check on how safe the dividend appears.",
                        "tone": tone_from_metric_threshold(row.get("Dividend_Safety_Score"), good_min=1.5, bad_max=0),
                        "help": ANALYSIS_HELP_TEXT["Dividend Safety"],
                    },
                    {
                        "label": "Valuation Confidence",
                        "value": format_value(row.get("Valuation_Confidence"), "{:,.0f}", "/100"),
                        "note": "Higher means the valuation read is backed by more usable inputs.",
                        "tone": tone_from_metric_threshold(row.get("Valuation_Confidence"), good_min=70, bad_max=40),
                        "help": ANALYSIS_HELP_TEXT["Valuation Confidence"],
                    },
                    {
                        "label": "Context Depth",
                        "value": format_value(row.get("Sentiment_Conviction"), "{:,.0f}", "/100"),
                        "note": "How much analyst and headline context was available for this company snapshot.",
                        "tone": "neutral",
                        "help": ANALYSIS_HELP_TEXT["Sentiment Conviction"],
                    },
                ],
                columns=5,
            )
            if row.get("Engine_Weight_Profile"):
                st.caption(f"Dynamic engine weights: {row.get('Engine_Weight_Profile')}")
            if row.get("Risk_Flags"):
                st.caption(f"Risk flags: {row.get('Risk_Flags')}")
            if row.get("Decision_Notes"):
                st.caption(str(row.get("Decision_Notes")))
            if str(row.get("Assumption_Fingerprint", "Legacy")) != active_assumption_fingerprint:
                st.warning(
                    "This saved analysis was generated under a different assumption set than the one currently active in Options."
                )

            tab_val, tab_fund, tab_tech, tab_sent, tab_dcf = st.tabs(
                ["Valuation Engine", "Fundamental Engine", "Technical Engine", "Sentiment Context", "DCF Lab"]
            )

            with tab_val:
                c_v1, c_v2 = st.columns([1, 2])
                with c_v1:
                    graham_discount = row.get("Graham Discount")
                    st.markdown(f"### Verdict: **{row['Verdict_Valuation']}**")
                    st.caption(
                        "This view compares the stock against a small peer set first, then falls back to sector benchmarks only when the peer data is too thin."
                    )
                    if row.get("Peer_Summary"):
                        st.caption(str(row.get("Peer_Summary")))
                    render_analysis_signal_cards(
                        [
                            {
                                "label": "Graham Fair Value",
                                "value": f"${row['Graham_Number']:,.2f}" if has_numeric_value(row["Graham_Number"]) and row["Graham_Number"] > 0 else "N/A",
                                "note": (
                                    f"Compared with today's price, the gap is ${row['Price'] - row['Graham_Number']:,.2f}."
                                    if has_numeric_value(row["Graham_Number"]) and row["Graham_Number"] > 0
                                    else "Only available when positive EPS and book value are both present."
                                ),
                                "tone": tone_from_metric_threshold(graham_discount, good_min=0.0, bad_max=-0.15),
                                "help": ANALYSIS_HELP_TEXT["Graham Fair Value"],
                            },
                            {
                                "label": "Graham Discount",
                                "value": format_percent(graham_discount),
                                "note": "Positive means the stock is trading below this fair-value estimate.",
                                "tone": tone_from_metric_threshold(graham_discount, good_min=0.0, bad_max=-0.15),
                                "help": ANALYSIS_HELP_TEXT["Graham Discount"],
                            },
                            {
                                "label": "Peer Group",
                                "value": str(row.get("Peer_Count") or 0),
                                "note": str(row.get("Peer_Group_Label") or "Closest comparable companies"),
                                "tone": "neutral",
                                "help": ANALYSIS_HELP_TEXT["Peer Group"],
                            },
                            {
                                "label": "Peer Summary",
                                "value": str(row.get("Peer_Group_Label") or "Fallback"),
                                "note": str(peer_comparison.get("benchmark_source") or "Closest peer group"),
                                "tone": "neutral",
                                "help": ANALYSIS_HELP_TEXT["Peer Summary"],
                            },
                            {
                                "label": "Relative Strength",
                                "value": format_percent(row.get("Relative_Strength_6M")),
                                "note": f"Six-month return relative to {DEFAULT_BENCHMARK_TICKER}.",
                                "tone": tone_from_metric_threshold(row.get("Relative_Strength_6M"), good_min=0.03, bad_max=-0.03),
                                "help": ANALYSIS_HELP_TEXT["Relative Strength"],
                            },
                            {
                                "label": "Valuation Consistency",
                                "value": format_value(row.get("Valuation_Confidence"), "{:,.0f}", "/100"),
                                "note": "Higher means the relative valuation read had more usable comparison points.",
                                "tone": "neutral",
                                "help": ANALYSIS_HELP_TEXT["Valuation Confidence"],
                            },
                        ],
                        columns=1,
                    )
                with c_v2:
                    render_analysis_signal_table(
                        [
                            {
                                "metric": "P/E Ratio",
                                "value": format_value(row["PE_Ratio"]),
                                "reference": format_value(peer_bench.get("PE")),
                                "status": "Cheap" if tone_from_relative_multiple(row["PE_Ratio"], peer_bench.get("PE")) == "good" else "Rich" if tone_from_relative_multiple(row["PE_Ratio"], peer_bench.get("PE")) == "bad" else "Fair",
                                "tone": tone_from_relative_multiple(row["PE_Ratio"], peer_bench.get("PE")),
                                "help": ANALYSIS_HELP_TEXT["P/E Ratio"],
                            },
                            {
                                "metric": "Forward P/E",
                                "value": format_value(row["Forward_PE"]),
                                "reference": format_value(peer_bench.get("PE")),
                                "status": "Cheap" if tone_from_relative_multiple(row["Forward_PE"], peer_bench.get("PE")) == "good" else "Rich" if tone_from_relative_multiple(row["Forward_PE"], peer_bench.get("PE")) == "bad" else "Fair",
                                "tone": tone_from_relative_multiple(row["Forward_PE"], peer_bench.get("PE")),
                                "help": ANALYSIS_HELP_TEXT["Forward P/E"],
                            },
                            {
                                "metric": "PEG Ratio",
                                "value": format_value(row["PEG_Ratio"]),
                                "reference": format_value(model_settings["valuation_peg_threshold"]),
                                "status": "Favorable" if tone_from_metric_threshold(row["PEG_Ratio"], good_max=model_settings["valuation_peg_threshold"] * 0.9, bad_min=model_settings["valuation_peg_threshold"] * 1.35) == "good" else "Stretched" if tone_from_metric_threshold(row["PEG_Ratio"], good_max=model_settings["valuation_peg_threshold"] * 0.9, bad_min=model_settings["valuation_peg_threshold"] * 1.35) == "bad" else "Mixed",
                                "tone": tone_from_metric_threshold(row["PEG_Ratio"], good_max=model_settings["valuation_peg_threshold"] * 0.9, bad_min=model_settings["valuation_peg_threshold"] * 1.35),
                                "help": ANALYSIS_HELP_TEXT["PEG Ratio"],
                            },
                            {
                                "metric": "P/S Ratio",
                                "value": format_value(row["PS_Ratio"]),
                                "reference": format_value(peer_bench.get("PS")),
                                "status": "Cheap" if tone_from_relative_multiple(row["PS_Ratio"], peer_bench.get("PS")) == "good" else "Rich" if tone_from_relative_multiple(row["PS_Ratio"], peer_bench.get("PS")) == "bad" else "Fair",
                                "tone": tone_from_relative_multiple(row["PS_Ratio"], peer_bench.get("PS")),
                                "help": ANALYSIS_HELP_TEXT["P/S Ratio"],
                            },
                            {
                                "metric": "EV/EBITDA",
                                "value": format_value(row["EV_EBITDA"]),
                                "reference": format_value(peer_bench.get("EV_EBITDA")),
                                "status": "Cheap" if tone_from_relative_multiple(row["EV_EBITDA"], peer_bench.get("EV_EBITDA")) == "good" else "Rich" if tone_from_relative_multiple(row["EV_EBITDA"], peer_bench.get("EV_EBITDA")) == "bad" else "Fair",
                                "tone": tone_from_relative_multiple(row["EV_EBITDA"], peer_bench.get("EV_EBITDA")),
                                "help": ANALYSIS_HELP_TEXT["EV/EBITDA"],
                            },
                            {
                                "metric": "P/B Ratio",
                                "value": format_value(row["PB_Ratio"]),
                                "reference": format_value(peer_bench.get("PB")),
                                "status": "Cheap" if tone_from_relative_multiple(row["PB_Ratio"], peer_bench.get("PB")) == "good" else "Rich" if tone_from_relative_multiple(row["PB_Ratio"], peer_bench.get("PB")) == "bad" else "Fair",
                                "tone": tone_from_relative_multiple(row["PB_Ratio"], peer_bench.get("PB")),
                                "help": ANALYSIS_HELP_TEXT["P/B Ratio"],
                            },
                        ],
                        reference_label="Peer Avg",
                    )
                    if row.get("Peer_Tickers"):
                        st.caption(f"Peer tickers: {row.get('Peer_Tickers')}")
                    if not peer_rows.empty:
                        peer_display = peer_rows.copy()
                        keep_columns = [
                            column
                            for column in [
                                "Ticker",
                                "Sector",
                                "Industry",
                                "Similarity",
                                "trailingPE",
                                "priceToSalesTrailing12Months",
                                "priceToBook",
                                "enterpriseToEbitda",
                                "revenueGrowth",
                                "beta",
                            ]
                            if column in peer_display.columns
                        ]
                        peer_display = peer_display[keep_columns]
                        rename_map = {
                            "trailingPE": "P/E",
                            "priceToSalesTrailing12Months": "P/S",
                            "priceToBook": "P/B",
                            "enterpriseToEbitda": "EV/EBITDA",
                            "revenueGrowth": "Revenue Growth",
                            "beta": "Beta",
                        }
                        peer_display = peer_display.rename(columns=rename_map)
                        if "Similarity" in peer_display.columns:
                            peer_display["Similarity"] = peer_display["Similarity"].map(lambda value: format_value(value, "{:,.2f}"))
                        for pct_column in ["Revenue Growth"]:
                            if pct_column in peer_display.columns:
                                peer_display[pct_column] = peer_display[pct_column].map(format_percent)
                        for numeric_column in ["P/E", "P/S", "P/B", "EV/EBITDA", "Beta"]:
                            if numeric_column in peer_display.columns:
                                peer_display[numeric_column] = peer_display[numeric_column].map(
                                    lambda value: format_value(value)
                                )
                        st.markdown("##### Peer Set")
                        st.dataframe(peer_display, width="stretch")

            with tab_fund:
                c_f1, c_f2 = st.columns([1, 2])
                with c_f1:
                    st.markdown(f"### Verdict: **{row['Verdict_Fundamental']}**")
                    st.caption("This view focuses on business strength, balance-sheet shape, and how the stock has reacted to recent company events.")
                    render_analysis_signal_cards(
                        [
                            {
                                "label": "Quality Score",
                                "value": format_value(row.get("Quality_Score"), "{:,.1f}"),
                                "note": "A compact read on profitability, balance-sheet quality, and growth stability.",
                                "tone": tone_from_metric_threshold(row.get("Quality_Score"), good_min=2, bad_max=0),
                                "help": ANALYSIS_HELP_TEXT["Quality Score"],
                            },
                            {
                                "label": "Dividend Safety",
                                "value": format_value(row.get("Dividend_Safety_Score"), "{:,.1f}"),
                                "note": "Useful mainly for income-oriented names.",
                                "tone": tone_from_metric_threshold(row.get("Dividend_Safety_Score"), good_min=1.5, bad_max=0),
                                "help": ANALYSIS_HELP_TEXT["Dividend Safety"],
                            },
                            {
                                "label": "Event Study",
                                "value": format_int(row.get("Event_Study_Count")),
                                "note": "Recent company events with usable price reactions.",
                                "tone": "neutral",
                                "help": ANALYSIS_HELP_TEXT["Event Study"],
                            },
                            {
                                "label": "5D Abnormal Move",
                                "value": format_percent(row.get("Event_Study_Avg_Abnormal_5D")),
                                "note": f"Average five-day move versus {DEFAULT_BENCHMARK_TICKER} after recent events.",
                                "tone": "neutral",
                                "help": ANALYSIS_HELP_TEXT["Event Study 5D"],
                            },
                        ],
                        columns=1,
                    )
                with c_f2:
                    render_analysis_signal_table(
                        [
                            {
                                "metric": "ROE",
                                "value": format_percent(row["ROE"]),
                                "reference": f">{model_settings['fund_roe_threshold'] * 100:.0f}%",
                                "status": "Strong" if tone_from_metric_threshold(row["ROE"], good_min=model_settings["fund_roe_threshold"], bad_max=max(0.0, model_settings["fund_roe_threshold"] * 0.5)) == "good" else "Weak" if tone_from_metric_threshold(row["ROE"], good_min=model_settings["fund_roe_threshold"], bad_max=max(0.0, model_settings["fund_roe_threshold"] * 0.5)) == "bad" else "Mixed",
                                "tone": tone_from_metric_threshold(row["ROE"], good_min=model_settings["fund_roe_threshold"], bad_max=max(0.0, model_settings["fund_roe_threshold"] * 0.5)),
                                "help": ANALYSIS_HELP_TEXT["ROE"],
                            },
                            {
                                "metric": "Profit Margin",
                                "value": format_percent(row["Profit_Margins"]),
                                "reference": f">{model_settings['fund_profit_margin_threshold'] * 100:.0f}%",
                                "status": "Strong" if tone_from_metric_threshold(row["Profit_Margins"], good_min=model_settings["fund_profit_margin_threshold"], bad_max=max(0.0, model_settings["fund_profit_margin_threshold"] * 0.5)) == "good" else "Weak" if tone_from_metric_threshold(row["Profit_Margins"], good_min=model_settings["fund_profit_margin_threshold"], bad_max=max(0.0, model_settings["fund_profit_margin_threshold"] * 0.5)) == "bad" else "Mixed",
                                "tone": tone_from_metric_threshold(row["Profit_Margins"], good_min=model_settings["fund_profit_margin_threshold"], bad_max=max(0.0, model_settings["fund_profit_margin_threshold"] * 0.5)),
                                "help": ANALYSIS_HELP_TEXT["Profit Margin"],
                            },
                            {
                                "metric": "Debt/Equity",
                                "value": format_value(row["Debt_to_Equity"], "{:,.0f}", "%"),
                                "reference": f"<{model_settings['fund_debt_good_threshold']:.0f}%",
                                "status": "Healthy" if tone_from_metric_threshold(row["Debt_to_Equity"], good_max=model_settings["fund_debt_good_threshold"], bad_min=model_settings["fund_debt_bad_threshold"]) == "good" else "Stretched" if tone_from_metric_threshold(row["Debt_to_Equity"], good_max=model_settings["fund_debt_good_threshold"], bad_min=model_settings["fund_debt_bad_threshold"]) == "bad" else "Watch",
                                "tone": tone_from_metric_threshold(row["Debt_to_Equity"], good_max=model_settings["fund_debt_good_threshold"], bad_min=model_settings["fund_debt_bad_threshold"]),
                                "help": ANALYSIS_HELP_TEXT["Debt/Equity"],
                            },
                            {
                                "metric": "Revenue Growth",
                                "value": format_percent(row["Revenue_Growth"]),
                                "reference": f">{model_settings['fund_revenue_growth_threshold'] * 100:.0f}%",
                                "status": "Strong" if tone_from_metric_threshold(row["Revenue_Growth"], good_min=model_settings["fund_revenue_growth_threshold"], bad_max=0.0) == "good" else "Weak" if tone_from_metric_threshold(row["Revenue_Growth"], good_min=model_settings["fund_revenue_growth_threshold"], bad_max=0.0) == "bad" else "Mixed",
                                "tone": tone_from_metric_threshold(row["Revenue_Growth"], good_min=model_settings["fund_revenue_growth_threshold"], bad_max=0.0),
                                "help": ANALYSIS_HELP_TEXT["Revenue Growth"],
                            },
                            {
                                "metric": "Current Ratio",
                                "value": format_value(row["Current_Ratio"]),
                                "reference": f">{model_settings['fund_current_ratio_good']:.1f}",
                                "status": "Healthy" if tone_from_metric_threshold(row["Current_Ratio"], good_min=model_settings["fund_current_ratio_good"], bad_max=model_settings["fund_current_ratio_bad"]) == "good" else "Weak" if tone_from_metric_threshold(row["Current_Ratio"], good_min=model_settings["fund_current_ratio_good"], bad_max=model_settings["fund_current_ratio_bad"]) == "bad" else "Mixed",
                                "tone": tone_from_metric_threshold(row["Current_Ratio"], good_min=model_settings["fund_current_ratio_good"], bad_max=model_settings["fund_current_ratio_bad"]),
                                "help": ANALYSIS_HELP_TEXT["Current Ratio"],
                            },
                            {
                                "metric": "Dividend Yield",
                                "value": format_percent(row.get("Dividend_Yield")),
                                "reference": "Income support",
                                "status": "Supportive" if tone_from_metric_threshold(row.get("Dividend_Yield"), good_min=0.02) == "good" else "Neutral",
                                "tone": tone_from_metric_threshold(row.get("Dividend_Yield"), good_min=0.02),
                                "help": ANALYSIS_HELP_TEXT["Dividend Yield"],
                            },
                            {
                                "metric": "Payout Ratio",
                                "value": format_percent(row.get("Payout_Ratio")),
                                "reference": "<75% preferred",
                                "status": "Safe" if tone_from_metric_threshold(row.get("Payout_Ratio"), good_max=0.75, bad_min=1.0) == "good" else "Stretched" if tone_from_metric_threshold(row.get("Payout_Ratio"), good_max=0.75, bad_min=1.0) == "bad" else "Mixed",
                                "tone": tone_from_metric_threshold(row.get("Payout_Ratio"), good_max=0.75, bad_min=1.0),
                                "help": ANALYSIS_HELP_TEXT["Payout Ratio"],
                            },
                            {
                                "metric": "Equity Beta",
                                "value": format_value(row.get("Equity_Beta")),
                                "reference": "<1.0 steadier",
                                "status": "Stable" if tone_from_metric_threshold(row.get("Equity_Beta"), good_max=1.0, bad_min=1.5) == "good" else "Volatile" if tone_from_metric_threshold(row.get("Equity_Beta"), good_max=1.0, bad_min=1.5) == "bad" else "Normal",
                                "tone": tone_from_metric_threshold(row.get("Equity_Beta"), good_max=1.0, bad_min=1.5),
                                "help": ANALYSIS_HELP_TEXT["Equity Beta"],
                            },
                        ],
                        reference_label="Target",
                    )
                if row.get("Event_Study_Summary"):
                    st.caption(str(row.get("Event_Study_Summary")))
                if not event_study_events.empty:
                    event_display = event_study_events.copy()
                    for column in ["Return_1D", "Return_5D", "Abnormal_1D", "Abnormal_5D"]:
                        if column in event_display.columns:
                            event_display[column] = event_display[column].map(format_percent)
                    st.markdown("##### Event Study")
                    st.dataframe(event_display, width="stretch")
                else:
                    st.caption("No recent dated company events were available for the event-study table.")

            with tab_tech:
                c_t1, c_t2 = st.columns([1, 2])
                with c_t1:
                    st.markdown(f"### Verdict: **{row['Verdict_Technical']}**")
                    st.caption("This view focuses on chart behavior: trend direction, momentum, and whether the stock looks stretched or healthy.")
                with c_t2:
                    render_analysis_signal_cards(
                        [
                            {
                                "label": "RSI (14)",
                                "value": format_value(row["RSI"], "{:,.1f}"),
                                "note": f"{int(model_settings['tech_rsi_oversold'])} oversold / {int(model_settings['tech_rsi_overbought'])} overbought",
                                "tone": tone_from_balanced_band(
                                    row["RSI"],
                                    healthy_min=model_settings["tech_rsi_oversold"] + 5,
                                    healthy_max=model_settings["tech_rsi_overbought"] - 5,
                                    caution_low=model_settings["tech_rsi_oversold"],
                                    caution_high=model_settings["tech_rsi_overbought"],
                                ),
                                "help": ANALYSIS_HELP_TEXT["RSI (14)"],
                            },
                            {
                                "label": "Trend",
                                "value": str(row["SMA_Status"]),
                                "note": "A quick read on the moving-average trend.",
                                "tone": tone_from_signal_text(row["SMA_Status"], positives={"BULLISH"}, negatives={"BEARISH"}),
                                "help": ANALYSIS_HELP_TEXT["Trend"],
                            },
                            {
                                "label": "MACD",
                                "value": format_value(row["MACD_Value"], "{:,.2f}"),
                                "note": f"Current signal: {row['MACD_Signal']}",
                                "tone": tone_from_signal_text(row["MACD_Signal"], positives={"BULLISH CROSSOVER"}, negatives={"BEARISH CROSSOVER"}),
                                "help": ANALYSIS_HELP_TEXT["MACD"],
                            },
                            {
                                "label": "1M Momentum",
                                "value": format_percent(row["Momentum_1M"]),
                                "note": "The stock's short-term move over roughly one month.",
                                "tone": tone_from_metric_threshold(row["Momentum_1M"], good_min=model_settings["tech_momentum_threshold"], bad_max=-model_settings["tech_momentum_threshold"]),
                                "help": ANALYSIS_HELP_TEXT["1M Momentum"],
                            },
                        ],
                        columns=4,
                    )
                    render_analysis_signal_cards(
                        [
                            {
                                "label": "3M Relative Strength",
                                "value": format_percent(row.get("Relative_Strength_3M")),
                                "note": f"Three-month return versus {DEFAULT_BENCHMARK_TICKER}.",
                                "tone": tone_from_metric_threshold(row.get("Relative_Strength_3M"), good_min=0.02, bad_max=-0.02),
                                "help": ANALYSIS_HELP_TEXT["Relative Strength"],
                            },
                            {
                                "label": "6M Relative Strength",
                                "value": format_percent(row.get("Relative_Strength_6M")),
                                "note": f"Six-month return versus {DEFAULT_BENCHMARK_TICKER}.",
                                "tone": tone_from_metric_threshold(row.get("Relative_Strength_6M"), good_min=0.03, bad_max=-0.03),
                                "help": ANALYSIS_HELP_TEXT["Relative Strength"],
                            },
                            {
                                "label": "1Y Relative Strength",
                                "value": format_percent(row.get("Relative_Strength_1Y")),
                                "note": f"One-year return versus {DEFAULT_BENCHMARK_TICKER}.",
                                "tone": tone_from_metric_threshold(row.get("Relative_Strength_1Y"), good_min=0.05, bad_max=-0.05),
                                "help": ANALYSIS_HELP_TEXT["Relative Strength"],
                            },
                        ],
                        columns=3,
                    )

                render_analysis_signal_table(
                    [
                        {
                            "metric": "RSI",
                            "value": format_value(row["RSI"], "{:,.1f}"),
                            "reference": "Balanced range",
                            "status": "Healthy" if tone_from_balanced_band(row["RSI"], healthy_min=model_settings["tech_rsi_oversold"] + 5, healthy_max=model_settings["tech_rsi_overbought"] - 5, caution_low=model_settings["tech_rsi_oversold"], caution_high=model_settings["tech_rsi_overbought"]) == "good" else "Extreme" if tone_from_balanced_band(row["RSI"], healthy_min=model_settings["tech_rsi_oversold"] + 5, healthy_max=model_settings["tech_rsi_overbought"] - 5, caution_low=model_settings["tech_rsi_oversold"], caution_high=model_settings["tech_rsi_overbought"]) == "bad" else "Mixed",
                            "tone": tone_from_balanced_band(row["RSI"], healthy_min=model_settings["tech_rsi_oversold"] + 5, healthy_max=model_settings["tech_rsi_overbought"] - 5, caution_low=model_settings["tech_rsi_oversold"], caution_high=model_settings["tech_rsi_overbought"]),
                            "help": ANALYSIS_HELP_TEXT["RSI"],
                        },
                        {
                            "metric": "200-Day Trend",
                            "value": str(row["SMA_Status"]),
                            "reference": "Trend direction",
                            "status": "Bullish" if tone_from_signal_text(row["SMA_Status"], positives={"BULLISH"}, negatives={"BEARISH"}) == "good" else "Bearish" if tone_from_signal_text(row["SMA_Status"], positives={"BULLISH"}, negatives={"BEARISH"}) == "bad" else "Neutral",
                            "tone": tone_from_signal_text(row["SMA_Status"], positives={"BULLISH"}, negatives={"BEARISH"}),
                            "help": ANALYSIS_HELP_TEXT["200-Day Trend"],
                        },
                        {
                            "metric": "MACD Signal",
                            "value": str(row["MACD_Signal"]),
                            "reference": "Momentum crossover",
                            "status": "Bullish" if tone_from_signal_text(row["MACD_Signal"], positives={"BULLISH CROSSOVER"}, negatives={"BEARISH CROSSOVER"}) == "good" else "Bearish" if tone_from_signal_text(row["MACD_Signal"], positives={"BULLISH CROSSOVER"}, negatives={"BEARISH CROSSOVER"}) == "bad" else "Neutral",
                            "tone": tone_from_signal_text(row["MACD_Signal"], positives={"BULLISH CROSSOVER"}, negatives={"BEARISH CROSSOVER"}),
                            "help": ANALYSIS_HELP_TEXT["MACD Signal"],
                        },
                        {
                            "metric": "1Y Momentum",
                            "value": format_percent(row["Momentum_1Y"]),
                            "reference": "Long-term move",
                            "status": "Strong" if tone_from_metric_threshold(row["Momentum_1Y"], good_min=max(model_settings["tech_momentum_threshold"] * 3, 0.10), bad_max=-max(model_settings["tech_momentum_threshold"] * 3, 0.10)) == "good" else "Weak" if tone_from_metric_threshold(row["Momentum_1Y"], good_min=max(model_settings["tech_momentum_threshold"] * 3, 0.10), bad_max=-max(model_settings["tech_momentum_threshold"] * 3, 0.10)) == "bad" else "Mixed",
                            "tone": tone_from_metric_threshold(row["Momentum_1Y"], good_min=max(model_settings["tech_momentum_threshold"] * 3, 0.10), bad_max=-max(model_settings["tech_momentum_threshold"] * 3, 0.10)),
                            "help": ANALYSIS_HELP_TEXT["1Y Momentum"],
                        },
                        {
                            "metric": "Trend Strength",
                            "value": format_value(row["Trend_Strength"], "{:,.0f}"),
                            "reference": ">20 constructive",
                            "status": "Strong" if tone_from_metric_threshold(row["Trend_Strength"], good_min=20, bad_max=-20) == "good" else "Weak" if tone_from_metric_threshold(row["Trend_Strength"], good_min=20, bad_max=-20) == "bad" else "Mixed",
                            "tone": tone_from_metric_threshold(row["Trend_Strength"], good_min=20, bad_max=-20),
                            "help": ANALYSIS_HELP_TEXT["Trend Strength"],
                        },
                    ],
                    reference_label="Read",
                )

            with tab_sent:
                c_s1, c_s2 = st.columns([1, 2])
                with c_s1:
                    st.markdown(f"### Verdict: **{row['Verdict_Sentiment']}**")
                    st.caption("This view is context only. It surfaces relevant analyst and headline information without classifying it as good or bad.")
                    target_price = row["Target_Mean_Price"]
                    render_analysis_signal_cards(
                        [
                            {
                                "label": "Headlines",
                                "value": format_int(row["Sentiment_Headline_Count"]),
                                "note": "Recent company-related headlines collected for context.",
                                "tone": "neutral",
                                "help": ANALYSIS_HELP_TEXT["Headlines"],
                            },
                            {
                                "label": "Analyst View",
                                "value": str(row["Recommendation_Key"]),
                                "note": "Raw recommendation label from the source feed, shown without interpretation.",
                                "tone": "neutral",
                                "help": ANALYSIS_HELP_TEXT["Analyst View"],
                            },
                            {
                                "label": "Target Mean",
                                "value": "N/A" if pd.isna(target_price) else f"${target_price:,.2f}",
                                "note": "Average analyst target price, shown as reference only.",
                                "tone": "neutral",
                                "help": ANALYSIS_HELP_TEXT["Target Mean"],
                            },
                            {
                                "label": "Context Depth",
                                "value": format_value(row.get("Sentiment_Conviction"), "{:,.0f}", "/100"),
                                "note": "How much analyst and headline context was available for this company.",
                                "tone": "neutral",
                                "help": ANALYSIS_HELP_TEXT["Sentiment Conviction"],
                            },
                        ],
                        columns=2,
                    )
                with c_s2:
                    st.write("Relevant company context")
                    context_lines = [
                        line.strip()
                        for line in str(row.get("Sentiment_Summary") or "").split("|")
                        if line.strip()
                    ]
                    if context_lines:
                        for line in context_lines:
                            st.write(f"- {line}")
                    else:
                        st.caption("No recent company-related context was available from the current source feed.")

            with tab_dcf:
                dcf_snapshot_exists = has_dcf_snapshot(row)
                dcf_download_bytes = build_dcf_download_bytes(row) if dcf_snapshot_exists else b""
                dcf_last_updated = str(row.get("DCF_Last_Updated") or row.get("Last_Updated") or "").strip()
                dcf_intrinsic_value = safe_num(row.get("DCF_Intrinsic_Value"))
                dcf_upside = safe_num(row.get("DCF Upside"))
                dcf_history = pd.DataFrame(safe_json_loads(row.get("DCF_History"), default=[]))
                dcf_projection = pd.DataFrame(safe_json_loads(row.get("DCF_Projection"), default=[]))
                dcf_sensitivity = pd.DataFrame(safe_json_loads(row.get("DCF_Sensitivity"), default=[]))
                dcf_excerpts = safe_json_loads(row.get("DCF_Guidance_Excerpts"), default=[])
                live_dcf_settings = get_dcf_settings()

                st.markdown("### Manual DCF Lab")
                st.caption("DCF is fully manual and only refreshes when you build it here, so library refreshes do not recompute cash-flow models for every saved ticker.")

                dcf_action_col_1, dcf_action_col_2 = st.columns([2, 1])
                with dcf_action_col_1:
                    with st.form(f"dcf_form_{row['Ticker']}"):
                        dcf_col_1, dcf_col_2, dcf_col_3 = st.columns(3)
                        projection_years = dcf_col_1.slider(
                            "Projection Years",
                            3,
                            10,
                            int(live_dcf_settings["projection_years"]),
                            1,
                            key=f"dcf_projection_years_{row['Ticker']}",
                        )
                        terminal_growth_percent = dcf_col_2.slider(
                            "Terminal Growth (%)",
                            0.0,
                            5.0,
                            float(live_dcf_settings["terminal_growth_rate"] * 100),
                            0.1,
                            key=f"dcf_terminal_growth_{row['Ticker']}",
                        )
                        growth_haircut = dcf_col_3.slider(
                            "Growth Haircut",
                            0.5,
                            1.2,
                            float(live_dcf_settings["growth_haircut"]),
                            0.05,
                            key=f"dcf_growth_haircut_{row['Ticker']}",
                        )

                        dcf_col_4, dcf_col_5, dcf_col_6 = st.columns(3)
                        market_risk_premium_percent = dcf_col_4.slider(
                            "Market Risk Premium (%)",
                            3.0,
                            9.0,
                            float(live_dcf_settings["market_risk_premium"] * 100),
                            0.1,
                            key=f"dcf_mrp_{row['Ticker']}",
                        )
                        cost_of_debt_percent = dcf_col_5.slider(
                            "Fallback Cost of Debt (%)",
                            1.0,
                            10.0,
                            float(live_dcf_settings["default_after_tax_cost_of_debt"] * 100),
                            0.1,
                            key=f"dcf_cost_of_debt_{row['Ticker']}",
                        )
                        manual_growth_rate_percent = dcf_col_6.number_input(
                            "Manual Starting Growth (%)",
                            min_value=-20.0,
                            max_value=50.0,
                            value=float((live_dcf_settings.get("manual_growth_rate") or 0.0) * 100),
                            step=0.5,
                            key=f"dcf_manual_growth_{row['Ticker']}",
                        )

                        dcf_col_7, dcf_col_8 = st.columns(2)
                        use_manual_growth = dcf_col_7.checkbox(
                            "Use manual starting growth rate",
                            value=live_dcf_settings.get("manual_growth_rate") is not None,
                            key=f"dcf_use_manual_growth_{row['Ticker']}",
                        )
                        risk_free_override_percent = dcf_col_8.number_input(
                            "Risk-Free Override (%)",
                            min_value=0.0,
                            max_value=10.0,
                            value=float((live_dcf_settings.get("risk_free_rate_override") or 0.0) * 100),
                            step=0.1,
                            key=f"dcf_risk_free_override_{row['Ticker']}",
                        )
                        use_risk_free_override = dcf_col_8.checkbox(
                            "Use risk-free override",
                            value=live_dcf_settings.get("risk_free_rate_override") is not None,
                            key=f"dcf_use_risk_free_override_{row['Ticker']}",
                        )

                        build_dcf_submit = st.form_submit_button(
                            "Build / Refresh DCF Snapshot",
                            type="primary",
                            width="stretch",
                        )

                    if build_dcf_submit:
                        updated_dcf_settings = normalize_dcf_settings(
                            {
                                "projection_years": projection_years,
                                "terminal_growth_rate": terminal_growth_percent / 100,
                                "growth_haircut": growth_haircut,
                                "market_risk_premium": market_risk_premium_percent / 100,
                                "default_after_tax_cost_of_debt": cost_of_debt_percent / 100,
                                "manual_growth_rate": manual_growth_rate_percent / 100 if use_manual_growth else None,
                                "risk_free_rate_override": risk_free_override_percent / 100 if use_risk_free_override else None,
                            }
                        )
                        st.session_state.dcf_settings = updated_dcf_settings
                        with st.spinner(f"Building DCF model for {row['Ticker']}..."):
                            dcf_record = bot.analyze(
                                str(row["Ticker"]),
                                settings=model_settings,
                                persist=False,
                                compute_dcf=True,
                                dcf_settings=updated_dcf_settings,
                            )
                        if not dcf_record:
                            st.error(bot.last_error or "Unable to build a DCF for this ticker right now.")
                        elif has_dcf_snapshot(dcf_record):
                            db.save_analysis(dcf_record)
                            st.session_state["dcf_action_feedback"] = {
                                "ticker": str(row["Ticker"]),
                                "kind": "success",
                                "message": f"Saved a fresh manual DCF snapshot for {row['Ticker']}.",
                            }
                            st.rerun()
                        else:
                            st.warning(
                                str(
                                    dcf_record.get("DCF_Guidance_Summary")
                                    or "A manual DCF could not be built from the available SEC filing data."
                                )
                            )
                with dcf_action_col_2:
                    st.download_button(
                        "Download DCF",
                        data=dcf_download_bytes,
                        file_name=f"{row['Ticker']}_dcf.json",
                        mime="application/json",
                        disabled=not dcf_snapshot_exists,
                        key=f"download_dcf_{row['Ticker']}",
                        width="stretch",
                    )
                    if dcf_snapshot_exists:
                        st.caption(f"Snapshot updated: {dcf_last_updated} ({format_age(dcf_last_updated)})")
                    else:
                        st.caption("No DCF snapshot is saved yet for this ticker.")

                snapshot_assumptions = normalize_dcf_settings(dcf_assumptions or live_dcf_settings)
                render_analysis_signal_cards(
                    [
                        {
                            "label": "DCF Fair Value",
                            "value": f"${dcf_intrinsic_value:,.2f}" if has_numeric_value(dcf_intrinsic_value) else "N/A",
                            "note": "Per-share estimate from the most recent saved DCF snapshot.",
                            "tone": "neutral",
                            "help": ANALYSIS_HELP_TEXT["DCF Fair Value"],
                        },
                        {
                            "label": "DCF Upside",
                            "value": format_percent(dcf_upside),
                            "note": "Gap between the snapshot value and the current stock price.",
                            "tone": "neutral",
                            "help": ANALYSIS_HELP_TEXT["DCF Upside"],
                        },
                        {
                            "label": "DCF WACC",
                            "value": format_percent(row.get("DCF_WACC")),
                            "note": "Discount rate used in the saved snapshot.",
                            "tone": "neutral",
                            "help": ANALYSIS_HELP_TEXT["DCF WACC"],
                        },
                        {
                            "label": "Growth Source",
                            "value": str(row.get("DCF_Source", "Unavailable")),
                            "note": str(row.get("DCF_Guidance_Summary") or "Source of the starting growth assumption."),
                            "tone": "neutral",
                            "help": ANALYSIS_HELP_TEXT["DCF Source"],
                        },
                        {
                            "label": "Projection Years",
                            "value": str(int(snapshot_assumptions.get("projection_years", DCF_PROJECTION_YEARS))),
                            "note": "Explicit forecast length used for the saved snapshot.",
                            "tone": "neutral",
                            "help": ANALYSIS_HELP_TEXT["DCF Fair Value"],
                        },
                        {
                            "label": "Terminal Growth",
                            "value": format_percent(snapshot_assumptions.get("terminal_growth_rate")),
                            "note": "Long-run rate used after the explicit projection window.",
                            "tone": "neutral",
                            "help": ANALYSIS_HELP_TEXT["Terminal Growth"],
                        },
                    ],
                    columns=3,
                )

                if dcf_snapshot_exists and not dcf_history.empty:
                    dcf_hist_col_1, dcf_hist_col_2 = st.columns(2)
                    with dcf_hist_col_1:
                        st.markdown("##### SEC History")
                        history_display = dcf_history.copy()
                        for money_column in ["Revenue", "OperatingCF", "CapEx", "FreeCashFlow"]:
                            if money_column in history_display.columns:
                                history_display[money_column] = history_display[money_column].map(format_market_cap)
                        st.dataframe(history_display, width="stretch")
                    with dcf_hist_col_2:
                        st.markdown("##### Projection")
                        projection_display = dcf_projection.copy()
                        if not projection_display.empty:
                            if "GrowthRate" in projection_display.columns:
                                projection_display["GrowthRate"] = projection_display["GrowthRate"].map(format_percent)
                            if "DiscountFactor" in projection_display.columns:
                                projection_display["DiscountFactor"] = projection_display["DiscountFactor"].map(
                                    lambda value: format_value(value, "{:,.3f}")
                                )
                            for money_column in ["FreeCashFlow", "PresentValue"]:
                                if money_column in projection_display.columns:
                                    projection_display[money_column] = projection_display[money_column].map(format_market_cap)
                            st.dataframe(projection_display, width="stretch")
                    if not dcf_sensitivity.empty:
                        st.markdown("##### Sensitivity Table")
                        sensitivity_display = dcf_sensitivity.copy()
                        if "WACC" in sensitivity_display.columns:
                            sensitivity_display["WACC"] = sensitivity_display["WACC"].map(format_percent)
                        for column in sensitivity_display.columns:
                            if column != "WACC":
                                sensitivity_display[column] = sensitivity_display[column].map(
                                    lambda value: f"${value:,.2f}" if has_numeric_value(value) else "N/A"
                                )
                        st.dataframe(sensitivity_display, width="stretch")
                else:
                    st.caption("Build a manual DCF snapshot to populate the valuation tables and sensitivity grid.")

                if dcf_excerpts:
                    st.markdown("##### Filing Takeaways")
                    for excerpt in dcf_excerpts[:5]:
                        st.write(f"- {excerpt}")
        else:
            st.info("Run the full analysis to save this ticker into the shared research library.")

with compare_tab:
    st.subheader("Compare Stocks")
    st.caption("Rank a watchlist with the same technical, fundamental, valuation, and context workflow before deciding what deserves portfolio weight.")

    with st.form("compare_form"):
        compare_col_1, compare_col_2 = st.columns([3, 1])
        with compare_col_1:
            compare_tickers_raw = st.text_area(
                "Tickers to Compare",
                value=DEFAULT_PORTFOLIO_TICKERS,
                help="Enter at least two ticker symbols separated by commas or spaces.",
            )
        with compare_col_2:
            compare_refresh = st.checkbox(
                "Refresh live data",
                value=False,
                help="If unchecked, the app reuses shared cached analyses when available.",
            )
            compare_submit = st.form_submit_button("Build Comparison", type="primary", width="stretch")

    if compare_submit:
        compare_tickers = parse_ticker_list(compare_tickers_raw)
        if len(compare_tickers) < 2:
            st.error("Enter at least two valid ticker symbols to compare.")
        else:
            with st.spinner("Pulling stock research and ranking the shortlist..."):
                comparison_df, failed_tickers, failure_reasons, refreshed_tickers, cached_tickers = collect_analysis_rows(
                    bot,
                    db,
                    compare_tickers,
                    refresh_live=compare_refresh,
                )

            if comparison_df.empty:
                st.session_state.pop("compare_result", None)
                st.session_state.pop("compare_meta", None)
                st.error("Comparison failed. Try again with valid tickers.")
                if failure_reasons:
                    st.caption(" | ".join(f"{ticker}: {reason}" for ticker, reason in failure_reasons.items()))
            else:
                st.session_state.compare_result = comparison_df
                st.session_state.compare_meta = {
                    "failed": failed_tickers,
                    "failure_reasons": failure_reasons,
                    "refreshed": refreshed_tickers,
                    "cached": cached_tickers,
                }

    if "compare_result" in st.session_state:
        comparison_df = prepare_analysis_dataframe(st.session_state.compare_result.copy())
        comparison_df = comparison_df.sort_values(
            ["Composite Score", "Target Upside", "Ticker"],
            ascending=[False, False, True],
            na_position="last",
        ).reset_index(drop=True)
        meta = st.session_state.get("compare_meta", {})

        top_pick = comparison_df.iloc[0]
        average_upside = comparison_df["Target Upside"].dropna().mean()
        average_dcf_upside = comparison_df["DCF Upside"].dropna().mean() if "DCF Upside" in comparison_df.columns else None
        render_analysis_signal_cards(
            [
                {
                    "label": "Highest Conviction",
                    "value": str(top_pick["Ticker"]),
                    "note": f"Current top-ranked name with a verdict of {top_pick['Verdict_Overall']}.",
                    "tone": tone_from_metric_threshold(top_pick.get("Decision_Confidence"), good_min=70, bad_max=45),
                    "help": ANALYSIS_HELP_TEXT["Highest Conviction"],
                },
                {
                    "label": "Average Composite Score",
                    "value": format_value(comparison_df["Composite Score"].mean(), "{:,.1f}"),
                    "note": "A higher average means the watchlist looks stronger overall.",
                    "tone": tone_from_metric_threshold(comparison_df["Composite Score"].mean(), good_min=1, bad_max=-1),
                    "help": ANALYSIS_HELP_TEXT["Average Composite Score"],
                },
                {
                    "label": "Average Target Upside",
                    "value": format_percent(average_upside),
                    "note": "The average analyst upside across the names in this shortlist.",
                    "tone": tone_from_metric_threshold(average_upside, good_min=0.10, bad_max=-0.05),
                    "help": ANALYSIS_HELP_TEXT["Average Target Upside"],
                },
                {
                    "label": "Average DCF Upside",
                    "value": format_percent(average_dcf_upside),
                    "note": "The average discount or premium implied by the cash-flow model across this shortlist.",
                    "tone": tone_from_metric_threshold(average_dcf_upside, good_min=0.10, bad_max=-0.05),
                    "help": ANALYSIS_HELP_TEXT["Average DCF Upside"],
                },
                {
                    "label": "Sectors Covered",
                    "value": str(comparison_df["Sector"].nunique()),
                    "note": "More sectors usually means the list is less concentrated in one theme.",
                    "tone": "neutral",
                    "help": ANALYSIS_HELP_TEXT["Sectors Covered"],
                },
            ],
            columns=5,
        )

        status_notes = []
        if meta.get("refreshed"):
            status_notes.append(f"Refreshed live: {', '.join(meta['refreshed'])}")
        if meta.get("cached"):
            status_notes.append(f"Loaded from cache: {', '.join(meta['cached'])}")
        if status_notes:
            st.caption(" | ".join(status_notes))
        if meta.get("failed"):
            st.warning(f"Could not analyze: {', '.join(meta['failed'])}")
            reason_lines = [
                f"{ticker}: {reason}"
                for ticker, reason in meta.get("failure_reasons", {}).items()
            ]
            if reason_lines:
                st.caption(" | ".join(reason_lines))
        if meta.get("cached") and calculate_assumption_drift(model_settings) > 0:
            st.caption("Cached rows keep their previous assumption set until you refresh them with live data.")
        if comparison_df["Assumption_Fingerprint"].nunique() > 1:
            st.caption("This comparison includes rows generated under different assumption fingerprints. Refresh live data for a cleaner apples-to-apples ranking.")

        st.subheader("Shortlist Ranking")
        render_help_legend(
            [
                ("Composite Score", ANALYSIS_HELP_TEXT["Composite Score"]),
                ("Consistency", ANALYSIS_HELP_TEXT["Consistency"]),
                ("Trend Strength", ANALYSIS_HELP_TEXT["Trend Strength"]),
                ("Quality Score", ANALYSIS_HELP_TEXT["Quality Score"]),
                ("Target Upside", ANALYSIS_HELP_TEXT["Target Mean"]),
                ("Graham Discount", ANALYSIS_HELP_TEXT["Graham Discount"]),
                ("DCF Upside", ANALYSIS_HELP_TEXT["DCF Upside"]),
                ("Freshness", ANALYSIS_HELP_TEXT["Freshness"]),
            ]
        )
        comparison_display = comparison_df[
            [
                "Ticker",
                "Sector",
                "Stock_Type",
                "Cap_Bucket",
                "Verdict_Overall",
                "Composite Score",
                "Decision_Confidence",
                "Trend_Strength",
                "Quality_Score",
                "Market_Regime",
                "Data_Quality",
                "Assumption_Profile",
                "Price",
                "Target Upside",
                "Graham Discount",
                "DCF Upside",
                "Score_Tech",
                "Score_Fund",
                "Score_Val",
                "Score_Sentiment",
                "Freshness",
            ]
        ].copy()
        comparison_display["Price"] = comparison_display["Price"].map(lambda value: f"${value:,.2f}" if pd.notna(value) else "N/A")
        comparison_display["Decision_Confidence"] = comparison_display["Decision_Confidence"].map(
            lambda value: format_value(value, "{:,.0f}", "/100")
        )
        comparison_display = comparison_display.rename(columns={"Decision_Confidence": "Consistency"})
        comparison_display["Trend_Strength"] = comparison_display["Trend_Strength"].map(
            lambda value: format_value(value, "{:,.0f}")
        )
        comparison_display["Quality_Score"] = comparison_display["Quality_Score"].map(
            lambda value: format_value(value, "{:,.1f}")
        )
        comparison_display["Target Upside"] = comparison_display["Target Upside"].map(format_percent)
        comparison_display["Graham Discount"] = comparison_display["Graham Discount"].map(format_percent)
        comparison_display["DCF Upside"] = comparison_display["DCF Upside"].map(format_percent)
        st.dataframe(comparison_display, width="stretch")

        engine_col, rationale_col = st.columns([2, 1])
        with engine_col:
            st.subheader("Engine Scorecard")
            render_help_legend(
                [
                    ("Technical", ANALYSIS_HELP_TEXT["Technical"]),
                    ("Fundamental", ANALYSIS_HELP_TEXT["Fundamental"]),
                    ("Valuation", ANALYSIS_HELP_TEXT["Valuation"]),
                    ("Sentiment", ANALYSIS_HELP_TEXT["Sentiment"]),
                    ("Composite Score", ANALYSIS_HELP_TEXT["Composite Score"]),
                ]
            )
            scorecard = comparison_df[
                ["Ticker", "Stock_Type", "Score_Tech", "Score_Fund", "Score_Val", "Score_Sentiment", "Composite Score"]
            ].copy()
            st.dataframe(scorecard, width="stretch")

        with rationale_col:
            st.subheader("What to Look For")
            st.write("- Higher composite scores usually deserve more diligence before moving into the portfolio tab.")
            st.write("- Stock type explains why the model may favor trend persistence for growth names but valuation discipline for value names.")
            st.write("- Large score disagreements across engines often mean the stock needs deeper judgment, not automatic sizing.")

with portfolio_tab:
    st.subheader("Portfolio Builder")
    st.caption("Use modern portfolio basics to recommend a risk-aware stock mix with Sharpe, Sortino, Treynor, the efficient frontier, and the Capital Allocation Line.")

    with st.form("portfolio_form"):
        p1, p2 = st.columns([3, 1])
        with p1:
            portfolio_tickers_raw = st.text_area(
                "Portfolio Tickers",
                value=DEFAULT_PORTFOLIO_TICKERS,
                help="Enter at least two tickers separated by commas or spaces.",
            )
        with p2:
            benchmark_ticker = st.text_input("Benchmark", value=DEFAULT_BENCHMARK_TICKER)
            lookback_period = st.selectbox("Lookback Period", ["1y", "3y", "5y"], index=1)
            risk_free_percent = st.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=15.0, value=4.0, step=0.25)

        p3, p4 = st.columns([3, 2])
        with p3:
            max_weight_percent = st.slider("Max Single-Stock Weight (%)", min_value=15, max_value=50, value=30, step=5)
        with p4:
            simulations = st.select_slider("Frontier Simulations", options=[1000, 2000, 3000, 4000, 5000], value=3000)

        portfolio_submit = st.form_submit_button("Build Portfolio Recommendation", type="primary", width="stretch")

    if portfolio_submit:
        parsed_tickers = parse_ticker_list(portfolio_tickers_raw)
        if len(parsed_tickers) < 2:
            st.error("Enter at least two valid ticker symbols for portfolio analysis.")
        elif len(parsed_tickers) * (max_weight_percent / 100) < 1:
            st.error("The max single-stock weight is too low for the number of tickers. Raise the cap or add more names.")
        else:
            with st.spinner("Building efficient frontier and CAL recommendation..."):
                portfolio_result = portfolio_bot.analyze_portfolio(
                    tickers=parsed_tickers,
                    benchmark_ticker=benchmark_ticker.strip().upper() or DEFAULT_BENCHMARK_TICKER,
                    period=lookback_period,
                    risk_free_rate=risk_free_percent / 100,
                    max_weight=max_weight_percent / 100,
                    simulations=simulations,
                )

            if not portfolio_result:
                st.session_state.pop("portfolio_result", None)
                st.session_state.pop("portfolio_config", None)
                st.error(
                    portfolio_bot.last_error
                    or "Portfolio analysis failed. Try different tickers or a longer lookback period."
                )
            else:
                st.session_state.portfolio_result = portfolio_result
                st.session_state.portfolio_config = {
                    "tickers": parsed_tickers,
                    "benchmark": benchmark_ticker.strip().upper() or DEFAULT_BENCHMARK_TICKER,
                    "period": lookback_period,
                    "risk_free_percent": risk_free_percent,
                    "max_weight_percent": max_weight_percent,
                    "simulations": simulations,
                }

    if "portfolio_result" in st.session_state:
        result = st.session_state.portfolio_result
        config = st.session_state.get("portfolio_config", {})
        tangent = result["tangent"]
        min_vol = result["minimum_volatility"]
        recommendations = result["recommendations"].copy()
        sector_exposure = result["sector_exposure"].copy()

        st.divider()
        st.caption(
            f"Benchmark: {config.get('benchmark', result['benchmark'])} | Lookback: {config.get('period', result['period'])} | "
            f"Risk-free rate: {config.get('risk_free_percent', 0):.2f}% | Max position: {config.get('max_weight_percent', 0)}%"
        )
        st.caption(f"Assumption profile: {active_preset_name} | Fingerprint: {active_assumption_fingerprint}")

        render_analysis_signal_cards(
            [
                {
                    "label": "Expected Return",
                    "value": format_percent(tangent["Return"]),
                    "note": "The annualized return estimate for the max-Sharpe portfolio.",
                    "tone": tone_from_metric_threshold(tangent["Return"], good_min=0.10, bad_max=0.03),
                    "help": ANALYSIS_HELP_TEXT["Expected Return"],
                },
                {
                    "label": "Volatility",
                    "value": format_percent(tangent["Volatility"]),
                    "note": "This is the expected bumpiness of returns over a full year.",
                    "tone": tone_from_metric_threshold(tangent["Volatility"], good_max=0.22, bad_min=0.35),
                    "help": ANALYSIS_HELP_TEXT["Volatility"],
                },
                {
                    "label": "Sharpe",
                    "value": format_value(tangent["Sharpe"]),
                    "note": "Higher Sharpe usually means a better return-to-risk tradeoff.",
                    "tone": tone_from_metric_threshold(tangent["Sharpe"], good_min=1.0, bad_max=0.3),
                    "help": ANALYSIS_HELP_TEXT["Sharpe"],
                },
                {
                    "label": "Sortino",
                    "value": format_value(tangent["Sortino"]),
                    "note": "This focuses on downside risk instead of all volatility.",
                    "tone": tone_from_metric_threshold(tangent["Sortino"], good_min=1.2, bad_max=0.4),
                    "help": ANALYSIS_HELP_TEXT["Sortino"],
                },
                {
                    "label": "Treynor",
                    "value": format_value(tangent["Treynor"]),
                    "note": "This compares excess return with market sensitivity, not total volatility.",
                    "tone": tone_from_metric_threshold(tangent["Treynor"], good_min=0.08, bad_max=0.0),
                    "help": ANALYSIS_HELP_TEXT["Treynor"],
                },
            ],
            columns=5,
        )

        render_analysis_signal_cards(
            [
                {
                    "label": "Portfolio Beta",
                    "value": format_value(tangent["Beta"]),
                    "note": "Around 1 means the portfolio has moved roughly in line with the benchmark.",
                    "tone": tone_from_balanced_band(tangent["Beta"], 0.8, 1.1, 0.6, 1.4),
                    "help": ANALYSIS_HELP_TEXT["Portfolio Beta"],
                },
                {
                    "label": "Downside Vol",
                    "value": format_percent(tangent["Downside Volatility"]),
                    "note": "This isolates the roughness coming from negative return swings.",
                    "tone": tone_from_metric_threshold(tangent["Downside Volatility"], good_max=0.15, bad_min=0.28),
                    "help": ANALYSIS_HELP_TEXT["Downside Vol"],
                },
                {
                    "label": "Min-Vol Return",
                    "value": format_percent(min_vol["Return"]),
                    "note": "The return estimate for the lowest-volatility portfolio the simulation found.",
                    "tone": tone_from_metric_threshold(min_vol["Return"], good_min=0.07, bad_max=0.02),
                    "help": ANALYSIS_HELP_TEXT["Min-Vol Return"],
                },
                {
                    "label": "Effective Names",
                    "value": format_value(result["effective_names"], "{:,.1f}"),
                    "note": "This shows how diversified the weights really are after concentration is considered.",
                    "tone": tone_from_metric_threshold(result["effective_names"], good_min=5, bad_max=3),
                    "help": ANALYSIS_HELP_TEXT["Effective Names"],
                },
            ],
            columns=4,
        )

        st.subheader("Efficient Frontier and CAL")
        st.caption("The green diamond is the tangent portfolio with the highest Sharpe ratio. The red dashed line is the Capital Allocation Line.")
        render_frontier_chart(result["portfolio_cloud"], result["frontier"], result["cal"], tangent, min_vol)

        st.subheader("Recommended Allocation")
        render_help_legend(
            [
                ("Sharpe", ANALYSIS_HELP_TEXT["Sharpe"]),
                ("Sortino", ANALYSIS_HELP_TEXT["Sortino"]),
                ("Treynor", ANALYSIS_HELP_TEXT["Treynor"]),
                ("Beta", ANALYSIS_HELP_TEXT["Portfolio Beta"]),
            ]
        )
        recommendations_display = recommendations[
            ["Ticker", "Name", "Sector", "Recommended Weight", "Role", "Sharpe Ratio", "Sortino Ratio", "Treynor Ratio", "Beta", "Rationale"]
        ].copy()
        recommendations_display["Recommended Weight"] = recommendations_display["Recommended Weight"].map(format_percent)
        recommendations_display["Sharpe Ratio"] = recommendations_display["Sharpe Ratio"].map(format_value)
        recommendations_display["Sortino Ratio"] = recommendations_display["Sortino Ratio"].map(format_value)
        recommendations_display["Treynor Ratio"] = recommendations_display["Treynor Ratio"].map(format_value)
        recommendations_display["Beta"] = recommendations_display["Beta"].map(format_value)
        st.dataframe(recommendations_display, width="stretch")

        exposure_col, metrics_col = st.columns([1, 2])
        with exposure_col:
            st.subheader("Sector Exposure")
            sector_display = sector_exposure.copy()
            sector_display["Recommended Weight"] = sector_display["Recommended Weight"].map(format_percent)
            st.dataframe(sector_display, width="stretch")

        with metrics_col:
            st.subheader("Per-Stock Metrics")
            render_help_legend(
                [
                    ("Annual Return", ANALYSIS_HELP_TEXT["Expected Return"]),
                    ("Volatility", ANALYSIS_HELP_TEXT["Volatility"]),
                    ("Downside Volatility", ANALYSIS_HELP_TEXT["Downside Vol"]),
                    ("Beta", ANALYSIS_HELP_TEXT["Portfolio Beta"]),
                    ("Sharpe", ANALYSIS_HELP_TEXT["Sharpe"]),
                    ("Sortino", ANALYSIS_HELP_TEXT["Sortino"]),
                    ("Treynor", ANALYSIS_HELP_TEXT["Treynor"]),
                ]
            )
            asset_display = result["asset_metrics"].copy()
            for column in ["Annual Return", "Volatility", "Downside Volatility"]:
                asset_display[column] = asset_display[column].map(format_percent)
            for column in ["Beta", "Sharpe Ratio", "Sortino Ratio", "Treynor Ratio"]:
                asset_display[column] = asset_display[column].map(format_value)
            st.dataframe(asset_display, width="stretch")

        st.subheader("Portfolio Building Notes")
        for note in result["notes"]:
            st.write(f"- {note}")

with sensitivity_tab:
    st.subheader("Sensitivity Check")
    st.caption("Run the same live market snapshot across guarded assumption scenarios to see whether the model is directionally stable or fragile.")
    st.caption("This does not overwrite the research library. It reuses one fresh market pull and reruns the scoring logic in memory.")

    with st.form("sensitivity_form"):
        sensitivity_col_1, sensitivity_col_2 = st.columns([3, 1])
        with sensitivity_col_1:
            sensitivity_ticker = st.text_input(
                "Ticker",
                value=sensitivity_default_ticker,
                help="Use this to check whether the verdict stays consistent across nearby assumption sets.",
            )
        with sensitivity_col_2:
            st.write("")
            st.write("")
            run_sensitivity = st.form_submit_button("Run Sensitivity Check", type="primary", width="stretch")

    if run_sensitivity:
        cleaned_ticker = sensitivity_ticker.strip().upper()
        if not cleaned_ticker:
            st.error("Enter a ticker to run a sensitivity check.")
        else:
            with st.spinner(f"Testing {cleaned_ticker} across assumption scenarios..."):
                sensitivity_df, sensitivity_summary = run_sensitivity_analysis(bot, cleaned_ticker, model_settings)

            if sensitivity_df is None or sensitivity_summary is None:
                st.session_state.pop("sensitivity_result", None)
                st.session_state.pop("sensitivity_summary", None)
                st.error(bot.last_error or "Sensitivity analysis failed. Try a different ticker.")
            else:
                st.session_state.sensitivity_result = sensitivity_df
                st.session_state.sensitivity_summary = sensitivity_summary
                st.session_state.sensitivity_last_ticker = cleaned_ticker

    if "sensitivity_result" in st.session_state:
        sensitivity_df = st.session_state.sensitivity_result.copy()
        sensitivity_summary = st.session_state.get("sensitivity_summary", {})
        checked_ticker = st.session_state.get("sensitivity_last_ticker", "")

        st.divider()
        st.caption(
            f"Ticker: {checked_ticker} | Active profile: {active_preset_name} | Active fingerprint: {active_assumption_fingerprint}"
        )

        render_analysis_signal_cards(
            [
                {
                    "label": "Robustness",
                    "value": sensitivity_summary.get("robustness_label", "N/A"),
                    "note": "This shows how stable the directional read stayed across guarded scenario changes.",
                    "tone": tone_from_signal_text(
                        sensitivity_summary.get("robustness_label"),
                        positives={"HIGH"},
                        negatives={"LOW"},
                    ),
                    "help": ANALYSIS_HELP_TEXT["Robustness"],
                },
                {
                    "label": "Dominant Bias",
                    "value": sensitivity_summary.get("dominant_bias", "N/A"),
                    "note": "The direction the model landed on most often across the scenarios.",
                    "tone": tone_from_signal_text(
                        sensitivity_summary.get("dominant_bias"),
                        positives={"BULLISH"},
                        negatives={"BEARISH"},
                    ),
                    "help": ANALYSIS_HELP_TEXT["Dominant Bias"],
                },
                {
                    "label": "Scenario Count",
                    "value": str(sensitivity_summary.get("scenario_count", 0)),
                    "note": "How many nearby assumption sets were tested against the same market snapshot.",
                    "tone": "neutral",
                    "help": ANALYSIS_HELP_TEXT["Scenario Count"],
                },
                {
                    "label": "Verdict Variety",
                    "value": str(sensitivity_summary.get("verdict_count", 0)),
                    "note": "More verdict variety usually means the name is more sensitive to model settings.",
                    "tone": tone_from_metric_threshold(sensitivity_summary.get("verdict_count", 0), good_max=2, bad_min=4),
                    "help": ANALYSIS_HELP_TEXT["Verdict Variety"],
                },
            ],
            columns=4,
        )

        robustness_ratio = sensitivity_summary.get("robustness_ratio")
        if robustness_ratio is not None:
            st.caption(f"Directional agreement across scenarios: {robustness_ratio * 100:.0f}%")
        if sensitivity_summary.get("robustness_label") == "Low":
            st.warning("This name is sensitive to the current assumption set. Treat the verdict as fragile and inspect the scenario table before acting on it.")
        elif sensitivity_summary.get("robustness_label") == "Medium":
            st.info("The direction is somewhat stable, but a few guarded assumption changes can still flip the conviction.")
        else:
            st.success("The direction stayed fairly consistent across the guarded scenario set.")

        render_help_legend(
            [
                ("Bias", ANALYSIS_HELP_TEXT["Bias"]),
                ("Consistency", ANALYSIS_HELP_TEXT["Consistency"]),
                ("Assumption Drift", ANALYSIS_HELP_TEXT["Assumption Drift"]),
                ("Fingerprint", ANALYSIS_HELP_TEXT["Fingerprint"]),
            ]
        )
        sensitivity_display = sensitivity_df.copy()
        sensitivity_display["Overall Score"] = sensitivity_display["Overall Score"].map(
            lambda value: format_value(value, "{:,.1f}")
        )
        sensitivity_display["Consistency"] = sensitivity_display["Consistency"].map(
            lambda value: format_value(value, "{:,.0f}", "/100")
        )
        sensitivity_display["Assumption Drift"] = sensitivity_display["Assumption Drift"].map(
            lambda value: format_value(value, "{:,.1f}", "%")
        )
        st.dataframe(sensitivity_display, width="stretch")

with backtest_tab:
    st.subheader("Signal Backtest")
    st.caption("Replay the current technical engine on historical prices to compare its trading path against a simple buy-and-hold baseline.")
    st.caption("This is a technical-rule backtest only. It does not recreate historical fundamentals, valuation, or news sentiment.")
    st.caption("The replay now uses stock-type-aware core sizing, trailing-stop behavior, and deeper-breakdown confirmation before fully exiting.")

    with st.form("backtest_form"):
        backtest_col_1, backtest_col_2, backtest_col_3 = st.columns([3, 1, 1])
        with backtest_col_1:
            backtest_ticker = st.text_input(
                "Ticker",
                value=backtest_default_ticker,
                help="The app replays the active technical thresholds over this price history.",
            )
        with backtest_col_2:
            backtest_period = st.selectbox("History Window", ["1y", "3y", "5y", "10y"], index=2)
        with backtest_col_3:
            st.write("")
            st.write("")
            run_backtest = st.form_submit_button("Run Backtest", type="primary", width="stretch")

    if run_backtest:
        cleaned_ticker = backtest_ticker.strip().upper()
        if not cleaned_ticker:
            st.error("Enter a ticker to run a backtest.")
        else:
            with st.spinner(f"Replaying technical signals on {cleaned_ticker}..."):
                hist, backtest_error = fetch_ticker_history_with_retry(cleaned_ticker, backtest_period)
                backtest_profile = {}
                saved_backtest_row = db.get_analysis(cleaned_ticker)
                if not saved_backtest_row.empty:
                    backtest_profile = extract_stock_profile_from_saved_row(saved_backtest_row.iloc[0])
                if not backtest_profile.get("primary_type") or backtest_profile.get("primary_type") == "Legacy":
                    backtest_info, _ = fetch_ticker_info_with_retry(cleaned_ticker)
                    if not backtest_info and not saved_backtest_row.empty:
                        backtest_info = build_info_fallback_from_saved_analysis(saved_backtest_row.iloc[0])
                    backtest_profile = infer_stock_profile_from_snapshot(
                        backtest_info,
                        hist,
                        model_settings,
                        db=db,
                        ticker=cleaned_ticker,
                    )
                backtest_result = compute_technical_backtest(hist, model_settings, stock_profile=backtest_profile)

            if hist is None or hist.empty:
                st.session_state.pop("backtest_result", None)
                st.session_state.pop("backtest_config", None)
                st.error(backtest_error or "Unable to load enough price history for this backtest.")
            elif backtest_result is None:
                st.session_state.pop("backtest_result", None)
                st.session_state.pop("backtest_config", None)
                st.error(
                    "Backtest needs roughly 250 trading days of usable price history. "
                    "Try a longer window or a ticker with more history."
                )
            else:
                st.session_state.backtest_result = backtest_result
                st.session_state.backtest_config = {
                    "ticker": cleaned_ticker,
                    "period": backtest_period,
                }
                st.session_state.backtest_last_ticker = cleaned_ticker

    if "backtest_result" in st.session_state:
        backtest_result = st.session_state.backtest_result
        backtest_config = st.session_state.get("backtest_config", {})
        backtest_metrics = backtest_result["metrics"]
        history_display = backtest_result["history"].copy()
        trade_log_display = backtest_result["trade_log"].copy()
        closed_trades_display = backtest_result.get("closed_trades", pd.DataFrame()).copy()
        backtest_profile = backtest_result.get("stock_profile", {})

        st.divider()
        st.caption(
            f"Ticker: {backtest_config.get('ticker', '')} | Window: {backtest_config.get('period', '')} | "
            f"Profile: {active_preset_name} | Fingerprint: {active_assumption_fingerprint}"
        )
        if backtest_profile:
            st.caption(
                f"Stock type: {backtest_profile.get('primary_type', 'Unknown')} | "
                f"Cap bucket: {backtest_profile.get('cap_bucket', 'Unknown')} | "
                f"Tags: {backtest_profile.get('style_tags', 'N/A')}"
            )
            if backtest_profile.get("type_strategy"):
                st.caption(backtest_profile["type_strategy"])

        render_analysis_signal_cards(
            [
                {
                    "label": "Strategy Return",
                    "value": format_percent(backtest_metrics["Strategy Total Return"]),
                    "note": "The total return generated by the trading rules in this replay.",
                    "tone": tone_from_metric_threshold(backtest_metrics["Strategy Total Return"], good_min=0.10, bad_max=-0.05),
                    "help": ANALYSIS_HELP_TEXT["Strategy Return"],
                },
                {
                    "label": "Benchmark Return",
                    "value": format_percent(backtest_metrics["Benchmark Total Return"]),
                    "note": "This is what simple buy-and-hold would have produced over the same period.",
                    "tone": tone_from_metric_threshold(backtest_metrics["Benchmark Total Return"], good_min=0.10, bad_max=-0.05),
                    "help": ANALYSIS_HELP_TEXT["Benchmark Return"],
                },
                {
                    "label": "Relative vs Benchmark",
                    "value": format_percent(backtest_metrics["Relative Return"]),
                    "note": "Positive means the strategy beat buy-and-hold. Negative means it lagged.",
                    "tone": tone_from_metric_threshold(backtest_metrics["Relative Return"], good_min=0.0, bad_max=-0.05),
                    "help": ANALYSIS_HELP_TEXT["Relative vs Benchmark"],
                },
                {
                    "label": "Strategy Sharpe",
                    "value": format_value(backtest_metrics["Strategy Sharpe"]),
                    "note": "This compares the strategy's return with the volatility it took to earn it.",
                    "tone": tone_from_metric_threshold(backtest_metrics["Strategy Sharpe"], good_min=0.8, bad_max=0.2),
                    "help": ANALYSIS_HELP_TEXT["Strategy Sharpe"],
                },
                {
                    "label": "Win Rate",
                    "value": format_percent(backtest_metrics["Win Rate"]),
                    "note": "This only counts closed trades, so it can stay unavailable when nothing closed.",
                    "tone": tone_from_metric_threshold(backtest_metrics["Win Rate"], good_min=0.55, bad_max=0.40),
                    "help": ANALYSIS_HELP_TEXT["Win Rate"],
                },
                {
                    "label": "Trading Costs",
                    "value": format_percent(backtest_metrics["Trading Costs"]),
                    "note": "Estimated transaction costs deducted when the replay changes exposure.",
                    "tone": "neutral",
                    "help": ANALYSIS_HELP_TEXT["Trading Costs"],
                },
                {
                    "label": "Max Drawdown",
                    "value": format_percent(backtest_metrics["Strategy Max Drawdown"]),
                    "note": "This shows the worst peak-to-trough drop during the replay.",
                    "tone": tone_from_metric_threshold(backtest_metrics["Strategy Max Drawdown"], good_min=-0.12, bad_max=-0.30),
                    "help": ANALYSIS_HELP_TEXT["Max Drawdown"],
                },
            ],
            columns=6,
        )

        render_analysis_signal_cards(
            [
                {
                    "label": "Average Exposure",
                    "value": format_percent(backtest_metrics["Average Exposure"]),
                    "note": "Higher means the replay stayed invested more consistently instead of sitting in cash.",
                    "tone": tone_from_metric_threshold(backtest_metrics["Average Exposure"], good_min=0.75, bad_max=0.45),
                    "help": ANALYSIS_HELP_TEXT["Average Exposure"],
                },
                {
                    "label": "Upside Capture",
                    "value": format_percent(backtest_metrics["Upside Capture"]),
                    "note": "This compares the strategy's gain with a positive buy-and-hold gain over the same window.",
                    "tone": tone_from_metric_threshold(backtest_metrics["Upside Capture"], good_min=0.90, bad_max=0.60),
                    "help": ANALYSIS_HELP_TEXT["Upside Capture"],
                },
                {
                    "label": "Position Changes",
                    "value": str(int(backtest_metrics["Position Changes"])),
                    "note": "Entries, exits, adds, and reductions all count toward this total.",
                    "tone": "neutral",
                    "help": ANALYSIS_HELP_TEXT["Position Changes"],
                },
                {
                    "label": "Closed Trades",
                    "value": str(int(backtest_metrics["Closed Trades"])),
                    "note": "Only completed trades count here, not open positions that are still running.",
                    "tone": "neutral",
                    "help": ANALYSIS_HELP_TEXT["Closed Trades"],
                },
                {
                    "label": "Avg Trade Return",
                    "value": format_percent(backtest_metrics["Average Trade Return"]),
                    "note": "This is the average result across the strategy's closed trades.",
                    "tone": tone_from_metric_threshold(backtest_metrics["Average Trade Return"], good_min=0.03, bad_max=-0.02),
                    "help": ANALYSIS_HELP_TEXT["Avg Trade Return"],
                },
            ],
            columns=5,
        )

        st.subheader("Equity Curve")
        chart_frame = history_display[["Date", "Strategy Equity", "Benchmark Equity"]].copy().set_index("Date")
        st.line_chart(chart_frame, width="stretch")

        st.subheader("Position Change Log")
        render_help_legend(
            [
                ("Signal", ANALYSIS_HELP_TEXT["Technical"]),
                ("Position", ANALYSIS_HELP_TEXT["Position Changes"]),
            ]
        )
        if trade_log_display.empty:
            st.info("No entries or exits were generated for this period under the active technical settings.")
        else:
            trade_log_display["Date"] = pd.to_datetime(trade_log_display["Date"]).dt.strftime("%Y-%m-%d")
            trade_log_display["Close"] = trade_log_display["Close"].map(lambda value: f"${value:,.2f}")
            trade_log_display["Position"] = trade_log_display["Position"].map(format_percent)
            if "Trading Cost" in trade_log_display.columns:
                trade_log_display["Trading Cost"] = trade_log_display["Trading Cost"].map(format_percent)
            st.dataframe(trade_log_display, width="stretch")

        st.subheader("Closed Trades")
        render_help_legend(
            [
                ("Return", ANALYSIS_HELP_TEXT["Avg Trade Return"]),
                ("Position Size", ANALYSIS_HELP_TEXT["Position Changes"]),
            ]
        )
        if closed_trades_display.empty:
            st.info("No closed trades were realized in this window, so win rate is not available yet.")
        else:
            closed_trades_display["Entry Date"] = pd.to_datetime(closed_trades_display["Entry Date"]).dt.strftime("%Y-%m-%d")
            closed_trades_display["Exit Date"] = pd.to_datetime(closed_trades_display["Exit Date"]).dt.strftime("%Y-%m-%d")
            closed_trades_display["Entry Price"] = closed_trades_display["Entry Price"].map(lambda value: f"${value:,.2f}")
            closed_trades_display["Exit Price"] = closed_trades_display["Exit Price"].map(lambda value: f"${value:,.2f}")
            closed_trades_display["Position Size"] = closed_trades_display["Position Size"].map(format_percent)
            closed_trades_display["Return"] = closed_trades_display["Return"].map(format_percent)
            st.dataframe(closed_trades_display, width="stretch")

with library_tab:
    st.subheader("Research Library")
    st.caption("Browse everything saved in the shared database so the research process stays visible across users and sessions.")
    if not db.supports_database_download:
        if db.storage_backend == "postgres":
            st.info("Database file export is unavailable when the app is connected to Postgres. Use the CSV export for library data.")
        else:
            st.info("Database export is unavailable in the current in-memory storage mode.")
    if startup_refresh_summary.get("error"):
        st.warning(f"Launch refresh hit an issue: {startup_refresh_summary['error']}")
    elif startup_refresh_summary.get("total", 0) > 0:
        st.caption(
            f"Launch refresh updated {startup_refresh_summary.get('updated', 0)} of "
            f"{startup_refresh_summary.get('total', 0)} stale saved analyses"
            + (
                f" and skipped {startup_refresh_summary.get('failed', 0)} tickers."
                if startup_refresh_summary.get("failed", 0)
                else "."
            )
        )

    library_df = prepare_analysis_dataframe(db.get_all_analyses())
    if library_df.empty:
        database_bytes = build_database_download_bytes(db.db_path if db.supports_database_download else None)
        export_col_1, export_col_2 = st.columns(2)
        with export_col_1:
            st.download_button(
                "Download Database",
                data=database_bytes,
                file_name=(db.db_path.name if db.supports_database_download else DB_FILENAME),
                mime="application/x-sqlite3",
                disabled=not bool(database_bytes),
                width="stretch",
            )
        with export_col_2:
            st.download_button(
                "Download Library CSV",
                data=b"",
                file_name="stock_engine_library.csv",
                mime="text/csv",
                disabled=True,
                width="stretch",
            )
        st.info("The library is empty right now. Run stock analyses or a comparison to populate the shared database.")
    else:
        sector_options = sorted(sector for sector in library_df["Sector"].dropna().unique())
        verdict_options = sorted(verdict for verdict in library_df["Verdict_Overall"].dropna().unique())
        stock_type_options = sorted(stock_type for stock_type in library_df["Stock_Type"].dropna().unique())
        filter_col_1, filter_col_2, filter_col_3, filter_col_4 = st.columns([2, 2, 2, 1])
        with filter_col_1:
            selected_sectors = st.multiselect("Sector Filter", sector_options, default=sector_options)
        with filter_col_2:
            selected_verdicts = st.multiselect("Verdict Filter", verdict_options, default=verdict_options)
        with filter_col_3:
            selected_stock_types = st.multiselect("Stock Type Filter", stock_type_options, default=stock_type_options)
        with filter_col_4:
            fresh_only = st.checkbox("Only show last 7 days", value=False)

        filtered_library = library_df.copy()
        if selected_sectors:
            filtered_library = filtered_library[filtered_library["Sector"].isin(selected_sectors)]
        else:
            filtered_library = filtered_library.iloc[0:0]
        if selected_verdicts:
            filtered_library = filtered_library[filtered_library["Verdict_Overall"].isin(selected_verdicts)]
        else:
            filtered_library = filtered_library.iloc[0:0]
        if selected_stock_types:
            filtered_library = filtered_library[filtered_library["Stock_Type"].isin(selected_stock_types)]
        else:
            filtered_library = filtered_library.iloc[0:0]
        if fresh_only:
            fresh_cutoff = datetime.datetime.now() - datetime.timedelta(days=7)
            filtered_library = filtered_library[
                filtered_library["Last_Updated_Parsed"].notna()
                & (filtered_library["Last_Updated_Parsed"] >= fresh_cutoff)
            ]

        export_frame = filtered_library if not filtered_library.empty else library_df
        database_bytes = build_database_download_bytes(db.db_path if db.supports_database_download else None)
        library_csv_bytes = build_library_csv_bytes(export_frame)
        export_col_1, export_col_2 = st.columns(2)
        with export_col_1:
            st.download_button(
                "Download Database",
                data=database_bytes,
                file_name=(db.db_path.name if db.supports_database_download else DB_FILENAME),
                mime="application/x-sqlite3",
                disabled=not bool(database_bytes),
                width="stretch",
            )
        with export_col_2:
            st.download_button(
                "Download Library CSV",
                data=library_csv_bytes,
                file_name="stock_engine_library.csv",
                mime="text/csv",
                disabled=export_frame.empty,
                width="stretch",
            )

        if filtered_library.empty:
            st.warning("No records match the current library filters.")
        else:
            if filtered_library["Assumption_Fingerprint"].nunique() > 1:
                st.caption("The current library view contains analyses generated under multiple assumption fingerprints.")
            fresh_24h = (
                filtered_library["Last_Updated_Parsed"].notna()
                & (filtered_library["Last_Updated_Parsed"] >= datetime.datetime.now() - datetime.timedelta(days=1))
            ).sum()
            render_analysis_signal_cards(
                [
                    {
                        "label": "Records",
                        "value": str(len(filtered_library)),
                        "note": "The number of saved analyses visible under the current filters.",
                        "tone": "neutral",
                        "help": ANALYSIS_HELP_TEXT["Records"],
                    },
                    {
                        "label": "Buy / Strong Buy",
                        "value": str(filtered_library["Verdict_Overall"].isin(["BUY", "STRONG BUY"]).sum()),
                        "note": "These are the names currently carrying a bullish final verdict.",
                        "tone": "good",
                        "help": ANALYSIS_HELP_TEXT["Buy / Strong Buy"],
                    },
                    {
                        "label": "Fresh in 24h",
                        "value": str(int(fresh_24h)),
                        "note": "Rows refreshed within the last day usually reflect the latest saved research pass.",
                        "tone": tone_from_metric_threshold(fresh_24h, good_min=5, bad_max=1),
                        "help": ANALYSIS_HELP_TEXT["Fresh in 24h"],
                    },
                    {
                        "label": "Tracked Sectors",
                        "value": str(filtered_library["Sector"].nunique()),
                        "note": "This shows how broad the current library slice is across industries.",
                        "tone": "neutral",
                        "help": ANALYSIS_HELP_TEXT["Tracked Sectors"],
                    },
                ],
                columns=4,
            )

            st.caption(f"Shared database: {db.storage_label}")

            render_help_legend(
                [
                    ("Composite Score", ANALYSIS_HELP_TEXT["Composite Score"]),
                    ("Consistency", ANALYSIS_HELP_TEXT["Consistency"]),
                    ("Trend Strength", ANALYSIS_HELP_TEXT["Trend Strength"]),
                    ("Quality Score", ANALYSIS_HELP_TEXT["Quality Score"]),
                    ("Target Upside", ANALYSIS_HELP_TEXT["Target Mean"]),
                    ("Graham Discount", ANALYSIS_HELP_TEXT["Graham Discount"]),
                    ("DCF Upside", ANALYSIS_HELP_TEXT["DCF Upside"]),
                    ("Freshness", ANALYSIS_HELP_TEXT["Freshness"]),
                ]
            )
            library_display = filtered_library[
                [
                    "Ticker",
                    "Sector",
                    "Stock_Type",
                "Cap_Bucket",
                "Verdict_Overall",
                "Composite Score",
                "Decision_Confidence",
                "Trend_Strength",
                "Quality_Score",
                "Market_Regime",
                "Data_Quality",
                "Assumption_Profile",
                "Price",
                    "Target Upside",
                    "Graham Discount",
                    "DCF Upside",
                    "Freshness",
                    "Last_Updated",
                ]
            ].copy()
            library_display["Price"] = library_display["Price"].map(lambda value: f"${value:,.2f}" if pd.notna(value) else "N/A")
            library_display["Decision_Confidence"] = library_display["Decision_Confidence"].map(
                lambda value: format_value(value, "{:,.0f}", "/100")
            )
            library_display = library_display.rename(columns={"Decision_Confidence": "Consistency"})
            library_display["Trend_Strength"] = library_display["Trend_Strength"].map(
                lambda value: format_value(value, "{:,.0f}")
            )
            library_display["Quality_Score"] = library_display["Quality_Score"].map(
                lambda value: format_value(value, "{:,.1f}")
            )
            library_display["Target Upside"] = library_display["Target Upside"].map(format_percent)
            library_display["Graham Discount"] = library_display["Graham Discount"].map(format_percent)
            library_display["DCF Upside"] = library_display["DCF Upside"].map(format_percent)
            st.dataframe(library_display, width="stretch")

            library_left, library_right = st.columns(2)
            with library_left:
                st.subheader("Sector Summary")
                render_help_legend(
                    [
                        ("Avg Composite Score", ANALYSIS_HELP_TEXT["Avg Composite Score"]),
                        ("Avg Target Upside", ANALYSIS_HELP_TEXT["Avg Target Upside"]),
                        ("Avg DCF Upside", ANALYSIS_HELP_TEXT["Avg DCF Upside"]),
                    ]
                )
                sector_summary = (
                    filtered_library.groupby("Sector", dropna=False)
                    .agg(
                        Records=("Ticker", "count"),
                        Avg_Composite_Score=("Composite Score", "mean"),
                        Avg_Target_Upside=("Target Upside", "mean"),
                        Avg_DCF_Upside=("DCF Upside", "mean"),
                    )
                    .reset_index()
                    .sort_values(["Records", "Avg_Composite_Score"], ascending=[False, False])
                )
                sector_summary["Avg_Composite_Score"] = sector_summary["Avg_Composite_Score"].map(
                    lambda value: format_value(value, "{:,.1f}")
                )
                sector_summary["Avg_Target_Upside"] = sector_summary["Avg_Target_Upside"].map(format_percent)
                sector_summary["Avg_DCF_Upside"] = sector_summary["Avg_DCF_Upside"].map(format_percent)
                st.dataframe(sector_summary, width="stretch")

            with library_right:
                st.subheader("Top Conviction Names")
                render_help_legend(
                    [
                        ("Composite Score", ANALYSIS_HELP_TEXT["Composite Score"]),
                        ("Target Upside", ANALYSIS_HELP_TEXT["Target Mean"]),
                        ("DCF Upside", ANALYSIS_HELP_TEXT["DCF Upside"]),
                        ("Freshness", ANALYSIS_HELP_TEXT["Freshness"]),
                    ]
                )
                conviction_table = filtered_library[
                    ["Ticker", "Verdict_Overall", "Composite Score", "Target Upside", "DCF Upside", "Freshness"]
                ].head(10).copy()
                conviction_table["Target Upside"] = conviction_table["Target Upside"].map(format_percent)
                conviction_table["DCF Upside"] = conviction_table["DCF Upside"].map(format_percent)
                st.dataframe(conviction_table, width="stretch")

with readme_tab:
    st.subheader("ReadMe / Usage")
    st.caption("Edit the README_USAGE_TEXT constant near the top of streamlit_app.py to customize this section.")
    if README_USAGE_TEXT.strip():
        st.markdown(README_USAGE_TEXT)
    else:
        st.text_area(
            "ReadMe / Usage Placeholder",
            value="",
            height=240,
            placeholder="Add your ReadMe / Usage copy in the README_USAGE_TEXT constant in streamlit_app.py.",
            disabled=True,
            label_visibility="collapsed",
        )

with changelog_tab:
    st.subheader("Changelog")
    st.caption("Recent updates to the stock model, portfolio engine, and research UI live here so the app stays inspectable over time.")

    changelog_metrics = st.columns(3)
    changelog_metrics[0].metric("Latest Logged Update", CHANGELOG_ENTRIES[0]["Date"])
    changelog_metrics[1].metric("Logged Changes", str(len(CHANGELOG_ENTRIES)))
    changelog_metrics[2].metric("App Version", APP_VERSION)

    st.dataframe(pd.DataFrame(CHANGELOG_ENTRIES), width="stretch")

    st.subheader("What Changed Most Recently")
    st.write("- The model now adds ten extra diagnostics such as trend strength, 52-week range context, volatility-adjusted momentum, quality score, dividend safety, valuation breadth, sentiment conviction, and explicit risk flags.")
    st.write("- The model now assigns each stock a primary type such as Growth, Value, Dividend, Cyclical, Defensive, Blue-Chip, size-based, or Speculative and uses that profile in verdict and backtest logic.")
    st.write("- The backtest now holds a core position during durable bullish regimes, exits later on deeper breakdowns, and reports win rate plus average closed-trade return.")
    st.write("- The Options tab now includes inline ? explanations for every slider and preset selector.")
    st.write("- Regime, consistency, and decision-note transparency remain visible across stock, compare, sensitivity, and library views.")

with methodology_tab:
    st.subheader("Methodology and Transparency")
    st.caption("This tab shows how the app forms verdicts, why the portfolio engine chooses certain weights, and what assumptions sit underneath the UI.")

    st.subheader("Model Flow")
    methodology_flow = pd.DataFrame(
        [
            {"Step": 1, "What Happens": "Download one year of price history plus company profile and news from Yahoo Finance."},
            {"Step": 2, "What Happens": "Score technical, fundamental, and peer-relative valuation layers. The DCF is optional and only runs when you create it manually."},
            {"Step": 3, "What Happens": "Classify market regime, measure engine agreement, and estimate decision consistency."},
            {"Step": 4, "What Happens": "Apply hold buffers and data-quality guardrails before publishing the final verdict."},
            {"Step": 5, "What Happens": "Store the full result with timestamp, assumption fingerprint, and quality stats in the shared library."},
        ]
    )
    st.dataframe(methodology_flow, width="stretch")

    methodology_col_1, methodology_col_2 = st.columns(2)
    with methodology_col_1:
        st.subheader("Single-Stock Engines")
        engine_framework = pd.DataFrame(
            [
                {
                    "Engine": "Technical",
                    "Uses": "RSI, MACD crossover and level, 50/200-day trend, trend tolerance bands, stretch limits, 1M and 1Y momentum",
                    "Strong Signals": "Healthy trend alignment, bullish MACD, supportive momentum, and controlled oversold reversals",
                },
                {
                    "Engine": "Fundamental",
                    "Uses": "ROE, profit margin, debt/equity, revenue growth, earnings growth, current ratio",
                    "Strong Signals": "High profitability, positive growth, sound liquidity, manageable leverage",
                },
                {
                    "Engine": "Valuation",
                    "Uses": "P/E, forward P/E, PEG, P/S, EV/EBITDA, P/B, five closest peers, Graham value, plus an optional manual SEC-based DCF snapshot",
                    "Strong Signals": "Positive earnings, cheaper-than-peer multiples, and discounts to fair-value anchors such as Graham value or a manual DCF snapshot",
                },
                {
                    "Engine": "Sentiment",
                    "Uses": "Company-related headlines, analyst recommendation labels, analyst depth, target mean price",
                    "Strong Signals": "The app now surfaces context only and leaves the interpretation to the user",
                },
            ]
        )
        st.dataframe(engine_framework, width="stretch")

    with methodology_col_2:
        st.subheader("Verdict Thresholds")
        verdict_table = pd.DataFrame(
            [
                {"Output": "Technical score", "Rule": ">= 4 strong buy, >= 2 buy, <= -2 sell, <= -4 strong sell"},
                {"Output": "Sentiment context", "Rule": "Displayed as reference only and not used as a directional score in the current build"},
                {
                    "Output": "Valuation verdict",
                    "Rule": (
                        f">= {model_settings['valuation_under_score_threshold']:.0f} undervalued, "
                        f">= {model_settings['valuation_fair_score_threshold']:.0f} fair value, otherwise overvalued"
                    ),
                },
                {
                    "Output": "Overall verdict",
                    "Rule": (
                        f"Base score thresholds are {model_settings['overall_strong_buy_threshold']:.0f} / "
                        f"{model_settings['overall_buy_threshold']:.0f} / "
                        f"{model_settings['overall_sell_threshold']:.0f} / "
                        f"{model_settings['overall_strong_sell_threshold']:.0f}; mixed regimes, low consistency, and low-quality data are pushed toward hold"
                    ),
                },
            ]
        )
        st.dataframe(verdict_table, width="stretch")

    st.subheader("Stock Type Framework")
    stock_type_framework = pd.DataFrame(
        [
            {"Type": "Growth Stocks", "How The Model Recognizes It": "Fast growth, premium multiples, low yield, and strong momentum", "Logic Tilt": "Trend persistence matters more than cheap valuation"},
            {"Type": "Value Stocks", "How The Model Recognizes It": "Undervaluation, discounted multiples, and at least stable fundamentals", "Logic Tilt": "Valuation and balance-sheet support matter more"},
            {"Type": "Dividend / Income Stocks", "How The Model Recognizes It": "Meaningful yield, payout support, income-heavy sectors, lower beta", "Logic Tilt": "Sustainability and steady compounding matter more"},
            {"Type": "Cyclical Stocks", "How The Model Recognizes It": "Economically sensitive sectors and bigger beta or cycle swings", "Logic Tilt": "Timing and regime confirmation matter more"},
            {"Type": "Defensive Stocks", "How The Model Recognizes It": "Resilient sectors, steadier beta, and stable fundamentals", "Logic Tilt": "The model tolerates slower upside and defends against over-trading"},
            {"Type": "Blue-Chip Stocks", "How The Model Recognizes It": "Large scale, quality metrics, broad coverage, and durable fundamentals", "Logic Tilt": "Quality durability gets extra room"},
            {"Type": "Small/Mid/Large-Cap", "How The Model Recognizes It": "Market capitalization bucket", "Logic Tilt": "Smaller caps require stronger confirmation; larger caps get more stability credit"},
            {"Type": "Speculative / Penny Stocks", "How The Model Recognizes It": "Tiny scale, low price, weak fundamentals, thin coverage, or extreme beta", "Logic Tilt": "Buy thresholds rise and conviction is capped"},
        ]
    )
    st.dataframe(stock_type_framework, width="stretch")

    st.subheader("Refinement Layer")
    refinement_df = pd.DataFrame(
        [
            {"Refinement": 1, "What Changed": "Trend Strength", "Purpose": "Uses SMA structure plus 1Y momentum as a continuous trend quality signal."},
            {"Refinement": 2, "What Changed": "52-Week Range Context", "Purpose": "Tracks whether price is breaking out, mid-range, or stuck near lows."},
            {"Refinement": 3, "What Changed": "Volatility-Adjusted Momentum", "Purpose": "Rewards momentum that is strong relative to realized volatility instead of raw price change alone."},
            {"Refinement": 4, "What Changed": "Quality Score", "Purpose": "Combines profitability, leverage, liquidity, and growth consistency into a cleaner business-quality signal."},
            {"Refinement": 5, "What Changed": "Dividend Safety Score", "Purpose": "Checks whether income stocks appear to have a more sustainable payout profile."},
            {"Refinement": 6, "What Changed": "Peer-Relative Valuation", "Purpose": "Anchors valuation to the five closest peers first and uses sector benchmarks only as a fallback."},
            {"Refinement": 7, "What Changed": "Context Depth", "Purpose": "Tracks how much analyst and headline context was available without turning that context into a directional sentiment score."},
            {"Refinement": 8, "What Changed": "Risk Flags", "Purpose": "Collects visible red flags like negative EPS, high debt, weak liquidity, high volatility, and speculation."},
            {"Refinement": 9, "What Changed": "Dynamic Engine Weights", "Purpose": "Lets Growth, Value, Income, Cyclical, and Speculative names use different engine mixes."},
            {"Refinement": 10, "What Changed": "Event Study and Trading Friction", "Purpose": "Adds recent event-reaction context to fundamentals and deducts transaction-cost estimates from the backtest."},
        ]
    )
    st.dataframe(refinement_df, width="stretch")

    st.subheader("Decision Guardrails")
    guardrail_df = pd.DataFrame(
        [
            {"Guardrail": "Trend Tolerance", "Purpose": "Avoids flipping trend signals on tiny moves around the moving averages."},
            {"Guardrail": "Stretch Limit", "Purpose": "Penalizes overextended rallies and recognizes washed-out rebounds before chasing price."},
            {"Guardrail": "Hold Buffer", "Purpose": "Makes mixed-engine or transition regimes require extra evidence before becoming directional."},
            {"Guardrail": "Consistency Floor", "Purpose": "Downgrades weakly aligned Buy or Sell calls back toward Hold."},
            {"Guardrail": "Data Quality Check", "Purpose": "Reduces conviction when too many important metrics are missing."},
        ]
    )
    st.dataframe(guardrail_df, width="stretch")

    st.subheader("Portfolio Workflow")
    portfolio_workflow = pd.DataFrame(
        [
            {"Step": 1, "What Happens": "Download adjusted close history for the chosen tickers and benchmark."},
            {"Step": 2, "What Happens": "Convert prices to returns and align all series on the same dates."},
            {"Step": 3, "What Happens": "Simulate capped portfolios and compute return, volatility, Sharpe, Sortino, Treynor, and beta."},
            {"Step": 4, "What Happens": "Trace the efficient frontier and identify the max-Sharpe tangent portfolio."},
            {"Step": 5, "What Happens": "Translate the tangent weights into practical roles, sector exposure, and concentration notes."},
        ]
    )
    st.dataframe(portfolio_workflow, width="stretch")

    methodology_col_3, methodology_col_4 = st.columns(2)
    with methodology_col_3:
        st.subheader("Peer Valuation Workflow")
        peer_workflow_df = pd.DataFrame(
            [
                {"Step": 1, "What Happens": "Start with the company being analyzed and pull its sector, industry, size, growth, profitability, leverage, and beta fields."},
                {"Step": 2, "What Happens": f"Search a cached universe for the {PEER_GROUP_SIZE} closest companies using those characteristics."},
                {"Step": 3, "What Happens": "Average the peer valuation multiples and use those averages as the main comparison set."},
                {"Step": 4, "What Happens": "If the peer set is too thin or missing too many usable metrics, fall back to scaled sector benchmarks."},
                {"Step": 5, "What Happens": "Keep Graham value separate and move DCF to a manual lab so cash-flow work stays optional and adjustable."},
            ]
        )
        st.dataframe(peer_workflow_df, width="stretch")

    with methodology_col_4:
        st.subheader("Current Model Assumptions")
        library_snapshot = prepare_analysis_dataframe(db.get_all_analyses())
        assumptions_df = pd.DataFrame(
            [
                {
                    "Setting": "Storage Mode",
                    "Value": (
                        "Postgres"
                        if db.storage_backend == "postgres"
                        else ("Persistent SQLite" if db.uses_persistent_storage else "In-memory")
                    ),
                },
                {"Setting": "Database Path", "Value": db.storage_label},
                {"Setting": "Active Profile", "Value": active_preset_name},
                {"Setting": "Assumption Fingerprint", "Value": active_assumption_fingerprint},
                {"Setting": "Trading Days per Year", "Value": int(model_settings["trading_days_per_year"])},
                {"Setting": "Default Benchmark", "Value": DEFAULT_BENCHMARK_TICKER},
                {"Setting": "Default Portfolio Universe", "Value": DEFAULT_PORTFOLIO_TICKERS},
                {
                    "Setting": "Engine Weights",
                    "Value": (
                        f"T {model_settings['weight_technical']:.1f} | "
                        f"F {model_settings['weight_fundamental']:.1f} | "
                        f"V {model_settings['weight_valuation']:.1f} | "
                        f"S {model_settings['weight_sentiment']:.1f}"
                    ),
                },
                {
                    "Setting": "RSI Band",
                    "Value": f"{int(model_settings['tech_rsi_oversold'])} / {int(model_settings['tech_rsi_overbought'])}",
                },
                {
                    "Setting": "Trend Tolerance",
                    "Value": f"{model_settings['tech_trend_tolerance'] * 100:.1f}%",
                },
                {
                    "Setting": "Extension Limit",
                    "Value": f"{model_settings['tech_extension_limit'] * 100:.1f}%",
                },
                {
                    "Setting": "Hold Buffer",
                    "Value": f"{model_settings['decision_hold_buffer']:.1f}",
                },
                {
                    "Setting": "Consistency Floor",
                    "Value": f"{model_settings['decision_min_confidence']:.0f}/100",
                },
                {
                    "Setting": "Backtest Cooldown",
                    "Value": f"{int(round(model_settings['backtest_cooldown_days']))} days",
                },
                {
                    "Setting": "Peer Group Size",
                    "Value": PEER_GROUP_SIZE,
                },
                {
                    "Setting": "Fallback Benchmark Scale",
                    "Value": f"{model_settings['valuation_benchmark_scale']:.2f}x",
                },
                {
                    "Setting": "Assumption Drift vs Defaults",
                    "Value": f"{calculate_assumption_drift(model_settings):.1f}%",
                },
                {"Setting": "Event Study Max Events", "Value": 5},
                {"Setting": "Backtest Transaction Cost", "Value": f"{model_settings['backtest_transaction_cost_bps']:.1f} bps"},
                {"Setting": "Cached Analyses in Library", "Value": len(library_snapshot)},
            ]
        )
        assumptions_df["Value"] = assumptions_df["Value"].map(str)
        st.dataframe(assumptions_df, width="stretch")

with options_tab:
    st.subheader("Model Options")
    st.caption("Tune the main assumptions with guardrails so the model stays interpretable and does not swing wildly from small changes.")
    st.caption("Changes apply to new stock analyses, refreshed comparisons, and new portfolio runs. Cached rows remain as previously analyzed until you rerun them.")
    st.caption(f"Active profile: {active_preset_name} | Fingerprint: {active_assumption_fingerprint}")

    preset_catalog = get_model_presets()
    preset_names = list(preset_catalog.keys())
    preset_index = preset_names.index(active_preset_name) if active_preset_name in preset_names else preset_names.index(get_default_preset_name())
    preset_col_1, preset_col_2 = st.columns([3, 1])
    with preset_col_1:
        preset_selection = st.selectbox(
            "Load Preset",
            preset_names,
            index=preset_index,
            help=OPTIONS_HELP_TEXT["load_preset"],
        )
        st.caption(PRESET_DESCRIPTIONS.get(preset_selection, ""))
    with preset_col_2:
        st.write("")
        st.write("")
        if st.button("Apply Preset", width="stretch"):
            st.session_state.model_settings = preset_catalog[preset_selection].copy()
            st.session_state.model_preset_name = preset_selection
            st.session_state.options_feedback = {
                "message": f"{preset_selection} preset loaded.",
                "notes": [
                    PRESET_DESCRIPTIONS.get(preset_selection, ""),
                    "You can still fine-tune any slider below and save the result as a custom assumption set.",
                ],
            }
            st.rerun()

    preset_snapshot = pd.DataFrame(
        [
            {
                "Preset": name,
                "Profile": PRESET_DESCRIPTIONS.get(name, ""),
                "Weights (T/F/V/S)": (
                    f"{values['weight_technical']:.1f} / {values['weight_fundamental']:.1f} / "
                    f"{values['weight_valuation']:.1f} / {values['weight_sentiment']:.1f}"
                ),
                "Trend Tol": f"{values['tech_trend_tolerance'] * 100:.0f}%",
                "Stretch": f"{values['tech_extension_limit'] * 100:.0f}%",
                "Hold Buffer": f"{values['decision_hold_buffer']:.1f}",
                "Consistency Floor": f"{values['decision_min_confidence']:.0f}/100",
                "Cooldown": f"{int(round(values['backtest_cooldown_days']))}d",
            }
            for name, values in preset_catalog.items()
        ]
    )
    st.subheader("Preset Snapshot")
    st.dataframe(preset_snapshot, width="stretch")

    feedback = st.session_state.pop("options_feedback", None)
    if feedback:
        st.success(feedback["message"])
        for note in feedback.get("notes", []):
            st.caption(note)

    assumption_drift = calculate_assumption_drift(model_settings)
    weight_values = [
        model_settings["weight_technical"],
        model_settings["weight_fundamental"],
        model_settings["weight_valuation"],
        model_settings["weight_sentiment"],
    ]
    options_metrics = st.columns(5)
    options_metrics[0].metric("Assumption Drift", format_value(assumption_drift, "{:,.1f}", "%"))
    options_metrics[1].metric("Trading Days", str(int(model_settings["trading_days_per_year"])))
    options_metrics[2].metric("Fallback Scale", format_value(model_settings["valuation_benchmark_scale"], "{:,.2f}", "x"))
    options_metrics[3].metric("Weight Spread", format_value(max(weight_values) - min(weight_values), "{:,.1f}"))
    options_metrics[4].metric("Consistency Floor", format_value(model_settings["decision_min_confidence"], "{:,.0f}", "/100"))

    if assumption_drift > 35:
        st.warning("Your active assumptions are materially different from the default model. Expect results to diverge more from the baseline.")
    else:
        st.info("The controls are intentionally range-limited so the model remains stable even when you tune it.")

    if st.button("Restore Default Assumptions", width="content"):
        st.session_state.model_settings = get_default_model_settings()
        st.session_state.model_preset_name = get_default_preset_name()
        st.session_state.options_feedback = {
            "message": "Default assumptions restored.",
            "notes": [],
        }
        st.rerun()

    with st.form("options_form"):
        st.subheader("Engine Weights")
        weight_col_1, weight_col_2, weight_col_3, weight_col_4 = st.columns(4)
        weight_technical = weight_col_1.slider("Technical", 0.5, 1.5, float(model_settings["weight_technical"]), 0.1, help=OPTIONS_HELP_TEXT["weight_technical"])
        weight_fundamental = weight_col_2.slider("Fundamental", 0.5, 1.5, float(model_settings["weight_fundamental"]), 0.1, help=OPTIONS_HELP_TEXT["weight_fundamental"])
        weight_valuation = weight_col_3.slider("Valuation", 0.5, 1.5, float(model_settings["weight_valuation"]), 0.1, help=OPTIONS_HELP_TEXT["weight_valuation"])
        weight_sentiment = weight_col_4.slider("Sentiment", 0.5, 1.5, float(model_settings["weight_sentiment"]), 0.1, help=OPTIONS_HELP_TEXT["weight_sentiment"], disabled=True)

        st.subheader("Technical and Fundamental Thresholds")
        tf_col_1, tf_col_2, tf_col_3, tf_col_4 = st.columns(4)
        tech_rsi_oversold = tf_col_1.slider("RSI Oversold", 20, 45, int(model_settings["tech_rsi_oversold"]), 1, help=OPTIONS_HELP_TEXT["tech_rsi_oversold"])
        tech_rsi_overbought = tf_col_2.slider("RSI Overbought", 55, 85, int(model_settings["tech_rsi_overbought"]), 1, help=OPTIONS_HELP_TEXT["tech_rsi_overbought"])
        tech_momentum_percent = tf_col_3.slider(
            "Momentum Trigger (%)",
            1,
            12,
            int(round(model_settings["tech_momentum_threshold"] * 100)),
            1,
            help=OPTIONS_HELP_TEXT["tech_momentum_threshold"],
        )
        fund_roe_percent = tf_col_4.slider(
            "ROE Threshold (%)",
            5,
            35,
            int(round(model_settings["fund_roe_threshold"] * 100)),
            1,
            help=OPTIONS_HELP_TEXT["fund_roe_threshold"],
        )

        tf_col_5, tf_col_6, tf_col_7, tf_col_8, tf_col_9 = st.columns(5)
        fund_margin_percent = tf_col_5.slider(
            "Profit Margin (%)",
            5,
            35,
            int(round(model_settings["fund_profit_margin_threshold"] * 100)),
            1,
            help=OPTIONS_HELP_TEXT["fund_profit_margin_threshold"],
        )
        fund_debt_good = tf_col_6.slider(
            "Healthy Debt/Equity",
            25,
            200,
            int(round(model_settings["fund_debt_good_threshold"])),
            5,
            help=OPTIONS_HELP_TEXT["fund_debt_good_threshold"],
        )
        fund_debt_bad = tf_col_7.slider(
            "High Debt/Equity",
            75,
            400,
            int(round(model_settings["fund_debt_bad_threshold"])),
            5,
            help=OPTIONS_HELP_TEXT["fund_debt_bad_threshold"],
        )
        fund_revenue_growth_percent = tf_col_8.slider(
            "Revenue Growth (%)",
            0,
            30,
            int(round(model_settings["fund_revenue_growth_threshold"] * 100)),
            1,
            help=OPTIONS_HELP_TEXT["fund_revenue_growth_threshold"],
        )
        fund_current_ratio_good = tf_col_9.slider(
            "Healthy Current Ratio",
            1.0,
            3.0,
            float(model_settings["fund_current_ratio_good"]),
            0.1,
            help=OPTIONS_HELP_TEXT["fund_current_ratio_good"],
        )

        st.subheader("Decision Stability")
        ds_col_1, ds_col_2, ds_col_3, ds_col_4 = st.columns(4)
        tech_trend_tolerance_percent = ds_col_1.slider(
            "Trend Tolerance (%)",
            0,
            5,
            int(round(model_settings["tech_trend_tolerance"] * 100)),
            1,
            help=OPTIONS_HELP_TEXT["tech_trend_tolerance"],
        )
        tech_extension_limit_percent = ds_col_2.slider(
            "Stretch Limit (%)",
            3,
            15,
            int(round(model_settings["tech_extension_limit"] * 100)),
            1,
            help=OPTIONS_HELP_TEXT["tech_extension_limit"],
        )
        decision_hold_buffer = ds_col_3.slider(
            "Hold Buffer",
            0.0,
            3.0,
            float(model_settings["decision_hold_buffer"]),
            0.5,
            help=OPTIONS_HELP_TEXT["decision_hold_buffer"],
        )
        decision_min_confidence = ds_col_4.slider(
            "Consistency Floor",
            35,
            80,
            int(round(model_settings["decision_min_confidence"])),
            1,
            help=OPTIONS_HELP_TEXT["decision_min_confidence"],
        )

        st.subheader("Valuation Fallbacks and Portfolio")
        vs_col_1, vs_col_2, vs_col_3, vs_col_4 = st.columns(4)
        valuation_benchmark_scale = vs_col_1.slider(
            "Fallback Benchmark Scale",
            0.8,
            1.2,
            float(model_settings["valuation_benchmark_scale"]),
            0.05,
            help=OPTIONS_HELP_TEXT["valuation_benchmark_scale"],
        )
        valuation_peg_threshold = vs_col_2.slider(
            "PEG Threshold",
            0.8,
            2.5,
            float(model_settings["valuation_peg_threshold"]),
            0.1,
            help=OPTIONS_HELP_TEXT["valuation_peg_threshold"],
        )
        valuation_graham_multiple = vs_col_3.slider(
            "Graham Overpriced Multiple",
            1.2,
            2.0,
            float(model_settings["valuation_graham_overpriced_multiple"]),
            0.05,
            help=OPTIONS_HELP_TEXT["valuation_graham_overpriced_multiple"],
        )
        trading_days_per_year = vs_col_4.slider(
            "Trading Days / Year",
            240,
            260,
            int(round(model_settings["trading_days_per_year"])),
            1,
            help=OPTIONS_HELP_TEXT["trading_days_per_year"],
        )

        vs_col_5, vs_col_6, vs_col_7, vs_col_8, vs_col_9 = st.columns(5)
        valuation_fair_score = vs_col_5.slider(
            "Fair Value Score Floor",
            1,
            4,
            int(round(model_settings["valuation_fair_score_threshold"])),
            1,
            help=OPTIONS_HELP_TEXT["valuation_fair_score_threshold"],
        )
        valuation_under_score = vs_col_6.slider(
            "Undervalued Score Floor",
            3,
            8,
            int(round(model_settings["valuation_under_score_threshold"])),
            1,
            help=OPTIONS_HELP_TEXT["valuation_under_score_threshold"],
        )
        sentiment_analyst_boost = vs_col_7.slider(
            "Analyst Sentiment Boost",
            0.0,
            4.0,
            float(model_settings["sentiment_analyst_boost"]),
            0.5,
            help=OPTIONS_HELP_TEXT["sentiment_analyst_boost"],
            disabled=True,
        )
        sentiment_upside_mid = vs_col_8.slider(
            "Moderate Upside (%)",
            2,
            15,
            int(round(model_settings["sentiment_upside_mid"] * 100)),
            1,
            help=OPTIONS_HELP_TEXT["sentiment_upside_mid"],
            disabled=True,
        )
        sentiment_upside_high = vs_col_9.slider(
            "Strong Upside (%)",
            8,
            30,
            int(round(model_settings["sentiment_upside_high"] * 100)),
            1,
            help=OPTIONS_HELP_TEXT["sentiment_upside_high"],
            disabled=True,
        )

        st.subheader("Overall Verdict Thresholds")
        ov_col_1, ov_col_2, ov_col_3, ov_col_4 = st.columns(4)
        overall_buy_threshold = ov_col_1.slider(
            "Buy Threshold",
            1,
            6,
            int(round(model_settings["overall_buy_threshold"])),
            1,
            help=OPTIONS_HELP_TEXT["overall_buy_threshold"],
        )
        overall_strong_buy_threshold = ov_col_2.slider(
            "Strong Buy Threshold",
            4,
            12,
            int(round(model_settings["overall_strong_buy_threshold"])),
            1,
            help=OPTIONS_HELP_TEXT["overall_strong_buy_threshold"],
        )
        overall_sell_magnitude = ov_col_3.slider(
            "Sell Threshold",
            1,
            6,
            int(round(abs(model_settings["overall_sell_threshold"]))),
            1,
            help=OPTIONS_HELP_TEXT["overall_sell_threshold"],
        )
        overall_strong_sell_magnitude = ov_col_4.slider(
            "Strong Sell Threshold",
            4,
            12,
            int(round(abs(model_settings["overall_strong_sell_threshold"]))),
            1,
            help=OPTIONS_HELP_TEXT["overall_strong_sell_threshold"],
        )

        downside_col_1, downside_col_2, current_ratio_bad_col = st.columns(3)
        sentiment_downside_mid = downside_col_1.slider(
            "Moderate Downside (%)",
            2,
            15,
            int(round(model_settings["sentiment_downside_mid"] * 100)),
            1,
            help=OPTIONS_HELP_TEXT["sentiment_downside_mid"],
            disabled=True,
        )
        sentiment_downside_high = downside_col_2.slider(
            "Deep Downside (%)",
            8,
            30,
            int(round(model_settings["sentiment_downside_high"] * 100)),
            1,
            help=OPTIONS_HELP_TEXT["sentiment_downside_high"],
            disabled=True,
        )
        fund_current_ratio_bad = current_ratio_bad_col.slider(
            "Weak Current Ratio",
            0.5,
            1.5,
            float(model_settings["fund_current_ratio_bad"]),
            0.1,
            help=OPTIONS_HELP_TEXT["fund_current_ratio_bad"],
        )

        backtest_cooldown_days = st.slider(
            "Backtest Re-entry Cooldown (days)",
            0,
            20,
            int(round(model_settings["backtest_cooldown_days"])),
            1,
            help=OPTIONS_HELP_TEXT["backtest_cooldown_days"],
        )
        backtest_transaction_cost_bps = st.slider(
            "Backtest Trading Cost (bps)",
            0.0,
            50.0,
            float(model_settings["backtest_transaction_cost_bps"]),
            1.0,
            help=OPTIONS_HELP_TEXT["backtest_transaction_cost_bps"],
        )

        save_options = st.form_submit_button("Save Assumptions", type="primary", width="stretch")

    if save_options:
        updated_settings = {
            "weight_technical": weight_technical,
            "weight_fundamental": weight_fundamental,
            "weight_valuation": weight_valuation,
            "weight_sentiment": weight_sentiment,
            "tech_rsi_oversold": tech_rsi_oversold,
            "tech_rsi_overbought": tech_rsi_overbought,
            "tech_momentum_threshold": tech_momentum_percent / 100,
            "tech_trend_tolerance": tech_trend_tolerance_percent / 100,
            "tech_extension_limit": tech_extension_limit_percent / 100,
            "fund_roe_threshold": fund_roe_percent / 100,
            "fund_profit_margin_threshold": fund_margin_percent / 100,
            "fund_debt_good_threshold": fund_debt_good,
            "fund_debt_bad_threshold": fund_debt_bad,
            "fund_revenue_growth_threshold": fund_revenue_growth_percent / 100,
            "fund_current_ratio_good": fund_current_ratio_good,
            "fund_current_ratio_bad": fund_current_ratio_bad,
            "valuation_benchmark_scale": valuation_benchmark_scale,
            "valuation_peg_threshold": valuation_peg_threshold,
            "valuation_graham_overpriced_multiple": valuation_graham_multiple,
            "valuation_fair_score_threshold": valuation_fair_score,
            "valuation_under_score_threshold": valuation_under_score,
            "sentiment_analyst_boost": sentiment_analyst_boost,
            "sentiment_upside_mid": sentiment_upside_mid / 100,
            "sentiment_upside_high": sentiment_upside_high / 100,
            "sentiment_downside_mid": sentiment_downside_mid / 100,
            "sentiment_downside_high": sentiment_downside_high / 100,
            "overall_buy_threshold": overall_buy_threshold,
            "overall_strong_buy_threshold": overall_strong_buy_threshold,
            "overall_sell_threshold": -overall_sell_magnitude,
            "overall_strong_sell_threshold": -overall_strong_sell_magnitude,
            "decision_hold_buffer": decision_hold_buffer,
            "decision_min_confidence": decision_min_confidence,
            "backtest_cooldown_days": backtest_cooldown_days,
            "backtest_transaction_cost_bps": backtest_transaction_cost_bps,
            "trading_days_per_year": trading_days_per_year,
        }
        normalized_settings, notes = normalize_model_settings(updated_settings)
        st.session_state.model_settings = normalized_settings
        st.session_state.model_preset_name = detect_matching_preset(normalized_settings)
        st.session_state.options_feedback = {
            "message": "Model assumptions updated for this session.",
            "notes": notes,
        }
        st.rerun()

