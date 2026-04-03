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

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

DB_FILENAME = "stocks_data.db"
APP_DIR = Path(__file__).resolve().parent
CONFIGURED_DB_PATH = Path(os.environ.get("STOCKS_DB_PATH", DB_FILENAME)).expanduser()
DB_PATH = CONFIGURED_DB_PATH if CONFIGURED_DB_PATH.is_absolute() else (APP_DIR / CONFIGURED_DB_PATH).resolve()
# Increase by 0.0.1 after each major update pass that includes 10+ meaningful changes.
APP_VERSION = "1.0.0"
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
}
FETCH_CACHE_LOCK = threading.RLock()
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
    "Stock_Type": "TEXT",
    "Cap_Bucket": "TEXT",
    "Style_Tags": "TEXT",
    "Type_Strategy": "TEXT",
    "Type_Confidence": "REAL",
    "Engine_Weight_Profile": "TEXT",
    "Market_Cap": "REAL",
    "Dividend_Yield": "REAL",
    "Payout_Ratio": "REAL",
    "Equity_Beta": "REAL",
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
    "RSI": "REAL",
    "MACD_Value": "REAL",
    "MACD_Signal": "TEXT",
    "SMA_Status": "TEXT",
    "Momentum_1M": "REAL",
    "Momentum_1Y": "REAL",
    "Last_Updated": "TEXT",
    "Overall_Score": "REAL",
    "Assumption_Profile": "TEXT",
    "Assumption_Fingerprint": "TEXT",
    "Assumption_Drift": "REAL",
    "Assumption_Snapshot": "TEXT",
    "Data_Completeness": "REAL",
    "Missing_Metric_Count": "INTEGER",
    "Data_Quality": "TEXT",
}
ANALYSIS_NUMERIC_COLUMNS = [
    name for name, definition in ANALYSIS_COLUMNS.items() if definition in {"REAL", "INTEGER"}
]
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
ANALYSIS_TONE_STYLES = {
    "good": {
        "accent": "#2b8a3e",
        "background": "rgba(43, 138, 62, 0.10)",
        "border": "rgba(43, 138, 62, 0.35)",
    },
    "bad": {
        "accent": "#c92a2a",
        "background": "rgba(201, 42, 42, 0.10)",
        "border": "rgba(201, 42, 42, 0.35)",
    },
    "neutral": {
        "accent": "#94a3b8",
        "background": "rgba(148, 163, 184, 0.08)",
        "border": "rgba(148, 163, 184, 0.24)",
    },
}
ANALYSIS_HELP_TEXT = {
    "Stock Type": "The model's best-fit stock category, such as Growth, Value, Dividend, Defensive, or Speculative.",
    "Cap Bucket": "A simple size label based on market value: Small-Cap, Mid-Cap, or Large-Cap.",
    "Type Confidence": "How confidently the model thinks this stock fits its assigned type.",
    "Market Cap": "The company's total market value based on share price times shares outstanding.",
    "Technical": "A score based on price trend, momentum, RSI, MACD, and moving averages. Higher is more constructive.",
    "Fundamental": "A score based on profitability, growth, leverage, and liquidity. Higher usually means a stronger business profile.",
    "Valuation": "A score showing whether the stock looks cheap, fair, or expensive compared with sector benchmarks and fair-value checks.",
    "Sentiment": "A score based on headline tone, analyst views, and target prices. Positive numbers mean the market tone is more supportive.",
    "Updated": "The last time this saved analysis was refreshed.",
    "Overall Score": "The model's combined score after blending technical, fundamental, valuation, and sentiment inputs.",
    "Data Quality": "A quick read on how complete and usable the underlying data was for this analysis.",
    "Assumption Profile": "The preset or custom settings used when this analysis was generated.",
    "Missing Metrics": "How many important data fields were unavailable during the analysis.",
    "Confidence": "How strongly the model trusts its final verdict given signal agreement, trend context, and data quality.",
    "Regime": "The current market backdrop the model sees in the stock: bullish trend, bearish trend, transition, or range-bound.",
    "Trend Strength": "A blended measure of long-term price trend quality using moving averages and one-year momentum.",
    "Quality Score": "A business-quality read based on returns, margins, balance-sheet strength, and growth consistency.",
    "Dividend Safety": "A rough check on whether the dividend looks sustainable based on payout ratio, profitability, liquidity, and debt.",
    "Valuation Confidence": "How much valuation evidence the model has available. More usable valuation inputs means higher confidence.",
    "Sentiment Conviction": "How strong and well-supported the sentiment signal is, not just whether it is positive or negative.",
    "Graham Fair Value": "A Graham-style fair value estimate based on earnings and book value when those inputs are available.",
    "Graham Discount": "Shows how far the current price sits below or above the Graham fair value estimate. Positive is cheaper.",
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
    "Headlines": "The number of recent news headlines the sentiment engine reviewed.",
    "Analyst View": "The current analyst recommendation signal the model could retrieve, such as Buy or Sell.",
    "Target Mean": "The average analyst target price. Higher than the current price can imply expected upside.",
    "Highest Conviction": "The top-ranked stock in the current comparison after blending the four engines with the active settings.",
    "Average Composite Score": "The average blended model score across the current comparison list. Higher means the group looks stronger overall.",
    "Average Target Upside": "The average gap between current price and analyst target price across the current list.",
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
}
CHANGELOG_ENTRIES = [
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
    "weight_sentiment": "Higher values make headlines, analyst views, and target prices matter more in the final score.",
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
    "decision_min_confidence": "Higher values make the final verdict stay on Hold unless engine agreement and data quality are stronger.",
    "valuation_benchmark_scale": "Lower values make sector valuation benchmarks stricter; higher values make them more forgiving.",
    "valuation_peg_threshold": "Lower values make the model harsher on expensive PEG ratios.",
    "valuation_graham_overpriced_multiple": "Lower values make the model call a stock overpriced sooner versus Graham value.",
    "trading_days_per_year": "Adjusts annualized portfolio and backtest metrics such as return, volatility, and Sharpe.",
    "valuation_fair_score_threshold": "Higher values require more valuation evidence before a name earns Fair Value instead of Overvalued.",
    "valuation_under_score_threshold": "Higher values require more valuation evidence before a name earns Undervalued.",
    "sentiment_analyst_boost": "Higher values make analyst recommendations contribute more to sentiment.",
    "sentiment_upside_mid": "Lower values let smaller analyst target upside count as a positive sentiment signal.",
    "sentiment_upside_high": "Lower values let strong upside bonuses trigger more easily.",
    "overall_buy_threshold": "Higher values require a larger combined score before the model upgrades from Hold to Buy.",
    "overall_strong_buy_threshold": "Higher values make Strong Buy rarer.",
    "overall_sell_threshold": "Higher values make the model wait for weaker negative evidence before switching from Hold to Sell.",
    "overall_strong_sell_threshold": "Higher values make Strong Sell rarer.",
    "sentiment_downside_mid": "Lower values let smaller analyst target downside count as a negative sentiment signal.",
    "sentiment_downside_high": "Lower values trigger a strong downside penalty sooner.",
    "backtest_cooldown_days": "Higher values force the replay to wait longer before re-entering after a position change.",
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


def escape_html_text(value):
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_help_badge(help_text):
    if not help_text:
        return ""
    safe_help = escape_html_text(help_text)
    return (
        '<span title="'
        + safe_help
        + '" style="display:inline-flex; align-items:center; justify-content:center; '
          'width:1.05rem; height:1.05rem; margin-left:0.38rem; border-radius:999px; '
          'border:1px solid rgba(148,163,184,0.35); color:#cbd5e1; font-size:0.72rem; '
          'font-weight:700; cursor:help; vertical-align:middle;">?</span>'
    )


def render_help_legend(items):
    fragments = []
    for label, help_text in items:
        if not help_text:
            continue
        safe_label = escape_html_text(label)
        fragments.append(
            f'<span style="display:inline-flex; align-items:center; margin-right:0.9rem;">'
            f'<span>{safe_label}</span>{build_help_badge(help_text)}</span>'
        )
    if fragments:
        st.markdown(
            f'<div style="font-size:0.83rem; color:#94a3b8; margin:0.15rem 0 0.45rem 0;">{"".join(fragments)}</div>',
            unsafe_allow_html=True,
        )


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
        style = ANALYSIS_TONE_STYLES.get(tone, ANALYSIS_TONE_STYLES["neutral"])
        label = escape_html_text(item.get("label", ""))
        value = escape_html_text(item.get("value", ""))
        note = escape_html_text(item.get("note", ""))
        help_badge = build_help_badge(item.get("help"))
        with cols[idx % columns]:
            st.markdown(
                f"""
                <div style="
                    border: 1px solid {style['border']};
                    background: {style['background']};
                    border-radius: 14px;
                    padding: 0.95rem 1rem;
                    min-height: 132px;
                    margin-bottom: 0.75rem;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    gap: 0.65rem;
                ">
                    <div style="min-height: 1.3rem; display: flex; align-items: center; flex-wrap: wrap; color: #cbd5e1; font-size: 0.82rem; font-weight: 600;">
                        <span>{label}</span>{help_badge}
                    </div>
                    <div style="font-size: 1.55rem; font-weight: 700; color: {style['accent']}; line-height: 1.15; min-height: 2.1rem; display: flex; align-items: center;">{value}</div>
                    <div style="font-size: 0.82rem; color: #94a3b8; min-height: 2.4rem; display: flex; align-items: flex-start;">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_analysis_signal_table(rows, reference_label="Reference"):
    if not rows:
        return

    body_rows = []
    for row in rows:
        tone = row.get("tone", "neutral")
        style = ANALYSIS_TONE_STYLES.get(tone, ANALYSIS_TONE_STYLES["neutral"])
        metric = escape_html_text(row.get("metric", ""))
        value = escape_html_text(row.get("value", ""))
        reference = escape_html_text(row.get("reference", ""))
        status = escape_html_text(row.get("status", tone.title()))
        help_badge = build_help_badge(row.get("help"))
        body_rows.append(
            f"""
            <tr>
                <td style="padding: 0.78rem 0.8rem; border-bottom: 1px solid rgba(148, 163, 184, 0.14); vertical-align: top; font-weight: 600; color: #e2e8f0;">
                    <span style="display: inline-flex; align-items: center; flex-wrap: wrap;">{metric}{help_badge}</span>
                </td>
                <td style="padding: 0.78rem 0.8rem; border-bottom: 1px solid rgba(148, 163, 184, 0.14); color: {style['accent']}; font-weight: 700; vertical-align: top;">{value}</td>
                <td style="padding: 0.78rem 0.8rem; border-bottom: 1px solid rgba(148, 163, 184, 0.14); color: #cbd5e1; vertical-align: top;">{reference}</td>
                <td style="padding: 0.78rem 0.8rem; border-bottom: 1px solid rgba(148, 163, 184, 0.14); vertical-align: top;">
                    <span style="
                        display: inline-block;
                        padding: 0.18rem 0.5rem;
                        border-radius: 999px;
                        color: {style['accent']};
                        background: {style['background']};
                        border: 1px solid {style['border']};
                        font-size: 0.78rem;
                        font-weight: 700;
                    ">{status}</span>
                </td>
            </tr>
            """
        )

    header_reference = escape_html_text(reference_label)
    st.markdown(
        f"""
        <div style="overflow-x: auto; margin-top: 0.4rem;">
            <table style="width: 100%; border-collapse: collapse; border-spacing: 0;">
                <thead>
                    <tr style="text-align: left; color: #94a3b8; border-bottom: 1px solid rgba(148, 163, 184, 0.20);">
                        <th style="padding: 0.55rem 0.8rem;">Metric</th>
                        <th style="padding: 0.55rem 0.8rem;">Value</th>
                        <th style="padding: 0.55rem 0.8rem;">{header_reference}</th>
                        <th style="padding: 0.55rem 0.8rem;">Read</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(body_rows)}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_library_csv_bytes(df):
    export_df = df.copy()
    export_df = export_df.drop(columns=["Last_Updated_Parsed"], errors="ignore")
    return export_df.to_csv(index=False).encode("utf-8")


def build_database_download_bytes(db_path):
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
                saved_rows = db.get_all_analyses()
                if saved_rows.empty or "Ticker" not in saved_rows.columns:
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

                refresh_candidates = saved_rows.copy()
                if "Last_Updated" in refresh_candidates.columns:
                    refresh_candidates["Last_Updated_Parsed"] = refresh_candidates["Last_Updated"].map(parse_last_updated)
                    stale_cutoff = datetime.datetime.now() - datetime.timedelta(hours=AUTO_REFRESH_STALE_AFTER_HOURS)
                    refresh_candidates = refresh_candidates[
                        refresh_candidates["Last_Updated_Parsed"].isna()
                        | (refresh_candidates["Last_Updated_Parsed"] < stale_cutoff)
                    ]
                    refresh_candidates = refresh_candidates.sort_values("Last_Updated_Parsed", ascending=True, na_position="first")

                tickers = (
                    refresh_candidates["Ticker"]
                    .dropna()
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .drop_duplicates()
                    .tolist()
                )
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
                            "processed": total,
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
                set_cached_fetch_payload("ticker_info", cache_key, info)
                return info, None
            alt_info = normalize_info_payload(getattr(ticker_handle, "get_info", lambda: {})() or {})
            if alt_info:
                set_cached_fetch_payload("ticker_info", cache_key, alt_info)
                return alt_info, None
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
    conviction = min(abs(float(sentiment_score)) * 10, 35)
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


def infer_stock_profile_from_snapshot(info, hist, settings=None):
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
    bench = get_sector_benchmarks(sector, active_settings)

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
        notes.append("Confidence is moderate, so the model stayed cautious")

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
    long_term_momentum_floor = max(active_settings["tech_momentum_threshold"] * 2, 0.08)
    trend_tolerance = active_settings["tech_trend_tolerance"]
    extension_limit = active_settings["tech_extension_limit"]
    cooldown_days = int(round(active_settings["backtest_cooldown_days"]))
    core_reentry_cooldown = max(1, cooldown_days // 2)
    core_target = 0.5
    full_target = 1.0
    entry_score_floor = 3
    core_score_floor = 1
    add_score_floor = 2
    hard_exit_score_floor = -5
    exit_break_multiplier = 1.5
    trailing_stop_threshold = -0.16
    allow_core_outside_bullish = False
    trim_to_core_on_non_bullish = False

    if primary_type in {"Growth Stocks", "Blue-Chip Stocks"}:
        core_reentry_cooldown = max(1, cooldown_days // 3)
        entry_score_floor = 2
        add_score_floor = 1
        hard_exit_score_floor = -6
        exit_break_multiplier = 1.8
        trailing_stop_threshold = -0.20
    elif primary_type == "Value Stocks":
        entry_score_floor = 2
        exit_break_multiplier = 1.7
        trailing_stop_threshold = -0.18
    elif primary_type in {"Dividend / Income Stocks", "Defensive Stocks", "Large-Cap Stocks"}:
        allow_core_outside_bullish = True
        entry_score_floor = 2
        core_score_floor = 0
        exit_break_multiplier = 1.8
        trailing_stop_threshold = -0.15
    elif primary_type == "Cyclical Stocks":
        core_target = 0.25
        entry_score_floor = 3
        add_score_floor = 2
        hard_exit_score_floor = -4
        exit_break_multiplier = 1.2
        trailing_stop_threshold = -0.12
        trim_to_core_on_non_bullish = True
    elif primary_type == "Mid-Cap Stocks":
        core_target = 0.5
        entry_score_floor = 3
        add_score_floor = 2
        exit_break_multiplier = 1.4
        trailing_stop_threshold = -0.14
    elif primary_type == "Small-Cap Stocks":
        core_target = 0.25
        full_target = 0.75
        entry_score_floor = 4
        core_score_floor = 2
        add_score_floor = 3
        hard_exit_score_floor = -4
        exit_break_multiplier = 1.2
        trailing_stop_threshold = -0.10
        trim_to_core_on_non_bullish = True
    elif primary_type == "Speculative / Penny Stocks":
        core_target = 0.0
        full_target = 0.5
        entry_score_floor = 4
        core_score_floor = 3
        add_score_floor = 4
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
    macd_bullish = analysis["MACD"].ge(analysis["MACD_Signal_Line"]).fillna(False)
    macd_bearish = analysis["MACD"].lt(analysis["MACD_Signal_Line"]).fillna(False)
    core_regime = bullish_regime | (~bearish_regime if allow_core_outside_bullish else False)
    trailing_stop_breach = analysis.get("Trailing_Drawdown_Quarter", pd.Series(index=analysis.index, dtype=float)).le(
        trailing_stop_threshold
    ).fillna(False)
    entry_signal = (
        bullish_regime
        & analysis["Tech Score"].ge(entry_score_floor)
        & macd_bullish
        & analysis["Momentum_1M"].gt(-active_settings["tech_momentum_threshold"]).fillna(False)
    )
    recovery_entry = (
        core_regime
        & analysis["RSI"].gt(active_settings["tech_rsi_oversold"]).fillna(False)
        & analysis["RSI"].shift(1).le(active_settings["tech_rsi_oversold"]).fillna(False)
        & macd_bullish
    )
    core_signal = (
        core_regime
        & analysis["Tech Score"].ge(core_score_floor)
        & macd_bullish
    )
    add_signal = (
        bullish_regime
        & analysis["Tech Score"].ge(add_score_floor)
        & analysis["Momentum_1M"].gt(-active_settings["tech_momentum_threshold"]).fillna(False)
    )
    overextended = (
        analysis["Close"].ge(analysis["SMA_50"] * (1 + extension_limit)).fillna(False)
        & analysis["RSI"].ge(active_settings["tech_rsi_overbought"] - 3).fillna(False)
    )
    tactical_reduce = (
        bullish_regime
        & (
            (overextended & analysis["Tech Score"].le(2))
            | (
                macd_bearish
                & analysis["Momentum_1M"].lt(0).fillna(False)
                & analysis["Tech Score"].le(0)
            )
        )
    )
    danger_reduce = (
        (bearish_regime | trailing_stop_breach)
        & (
            analysis["Tech Score"].le(-2)
            | (macd_bearish & analysis["Momentum_1M"].lt(0).fillna(False))
        )
    )
    hard_exit = (
        analysis["Tech Score"].le(hard_exit_score_floor)
        & macd_bearish
        & analysis["Momentum_1M"].lt(-active_settings["tech_momentum_threshold"]).fillna(False)
    )
    exit_signal = (
        hard_exit
        | (
            trailing_stop_breach
            & macd_bearish
            & analysis["Tech Score"].le(0)
        )
        | (
            bearish_regime
            & analysis["Close"].le(analysis["SMA_200"] * (1 - trend_tolerance * exit_break_multiplier)).fillna(False)
            & macd_bearish
            & analysis["Momentum_1Y"].lt(-long_term_momentum_floor).fillna(False)
        )
    )

    positions = []
    current_position = 0.0
    days_since_change = cooldown_days
    for is_bullish, is_bearish, enter_now, recover_now, core_now, add_now, reduce_now, danger_now, exit_now in zip(
        bullish_regime,
        bearish_regime,
        entry_signal,
        recovery_entry,
        core_signal,
        add_signal,
        tactical_reduce,
        danger_reduce,
        exit_signal,
    ):
        target_position = current_position
        if exit_now:
            target_position = 0.0
        elif is_bullish:
            if current_position < core_target and (core_now or enter_now or recover_now) and days_since_change >= core_reentry_cooldown:
                target_position = core_target
            if (enter_now or recover_now or add_now) and days_since_change >= cooldown_days:
                target_position = full_target
            elif reduce_now and current_position > core_target:
                target_position = core_target
        elif is_bearish:
            if danger_now and current_position > core_target:
                target_position = core_target
        else:
            if trim_to_core_on_non_bullish and current_position > core_target:
                target_position = core_target
            elif current_position < core_target and add_now and days_since_change >= core_reentry_cooldown:
                target_position = core_target
            elif reduce_now and current_position > core_target:
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

    if {"Price", "Target_Mean_Price"}.issubset(enriched.columns):
        enriched["Target Upside"] = np.where(
            enriched["Price"].notna() & (enriched["Price"] != 0),
            (enriched["Target_Mean_Price"] - enriched["Price"]) / enriched["Price"],
            np.nan,
        )

    if {"Price", "Graham_Number"}.issubset(enriched.columns):
        enriched["Graham Discount"] = np.where(
            enriched["Price"].notna() & (enriched["Price"] != 0) & enriched["Graham_Number"].notna(),
            (enriched["Graham_Number"] - enriched["Price"]) / enriched["Price"],
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
    if "Risk_Flags" not in enriched.columns:
        enriched["Risk_Flags"] = ""
    else:
        enriched["Risk_Flags"] = enriched["Risk_Flags"].fillna("")

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
                "Confidence": record.get("Decision_Confidence"),
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
    analysis["Strategy Return"] = analysis["Position"].shift(1).fillna(0.0) * analysis["Benchmark Return"]
    analysis["Benchmark Equity"] = (1 + analysis["Benchmark Return"]).cumprod()
    analysis["Strategy Equity"] = (1 + analysis["Strategy Return"]).cumprod()

    trading_days = active_settings["trading_days_per_year"]
    strategy_ann_return = analysis["Strategy Return"].mean() * trading_days
    strategy_vol = analysis["Strategy Return"].std() * np.sqrt(trading_days)
    benchmark_ann_return = analysis["Benchmark Return"].mean() * trading_days
    benchmark_vol = analysis["Benchmark Return"].std() * np.sqrt(trading_days)

    strategy_drawdown = analysis["Strategy Equity"] / analysis["Strategy Equity"].cummax() - 1
    benchmark_drawdown = analysis["Benchmark Equity"] / analysis["Benchmark Equity"].cummax() - 1

    trade_points = analysis["Position"].diff().fillna(analysis["Position"])
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
        }
    ).dropna(subset=["Action"])
    closed_trades_df, trade_summary = summarize_backtest_trades(analysis)

    metrics = {
        "Strategy Total Return": analysis["Strategy Equity"].iloc[-1] - 1,
        "Benchmark Total Return": analysis["Benchmark Equity"].iloc[-1] - 1,
        "Relative Return": analysis["Strategy Equity"].iloc[-1] - analysis["Benchmark Equity"].iloc[-1],
        "Strategy Annual Return": strategy_ann_return,
        "Benchmark Annual Return": benchmark_ann_return,
        "Strategy Volatility": strategy_vol,
        "Benchmark Volatility": benchmark_vol,
        "Strategy Sharpe": safe_divide(strategy_ann_return, strategy_vol),
        "Benchmark Sharpe": safe_divide(benchmark_ann_return, benchmark_vol),
        "Strategy Max Drawdown": strategy_drawdown.min(),
        "Benchmark Max Drawdown": benchmark_drawdown.min(),
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
        self.db_path = Path(db_name)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.RLock()
        self.create_tables()

    def _connect(self, allow_recover=True):
        # Open a fresh connection per operation so each session sees other users' commits.
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchall()
            return conn
        except sqlite3.DatabaseError as exc:
            if conn is not None:
                conn.close()
            if allow_recover and self._recover_database_file(exc):
                return self._connect(allow_recover=False)
            raise

    def _recover_database_file(self, exc):
        if not self.db_path.exists():
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

    def save_analysis(self, data):
        keys = list(data.keys())
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
            with self._connection() as conn:
                conn.execute(sql, list(data.values()))

    def get_analysis(self, ticker):
        try:
            with self._connection() as conn:
                return pd.read_sql_query(
                    "SELECT * FROM analysis WHERE Ticker=?",
                    conn,
                    params=(ticker,),
                )
        except (pd.errors.DatabaseError, sqlite3.DatabaseError):
            self.create_tables()
            with self._connection() as conn:
                return pd.read_sql_query(
                    "SELECT * FROM analysis WHERE Ticker=?",
                    conn,
                    params=(ticker,),
                )

    def get_all_analyses(self):
        try:
            with self._connection() as conn:
                return pd.read_sql_query("SELECT * FROM analysis", conn)
        except (pd.errors.DatabaseError, sqlite3.DatabaseError):
            self.create_tables()
            with self._connection() as conn:
                return pd.read_sql_query("SELECT * FROM analysis", conn)


@st.cache_resource
def get_database_manager():
    return DatabaseManager(DB_PATH)


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
        settings = get_model_settings() if settings is None else settings
        info = info or {}
        news = news or []
        score = 0
        headlines = []
        seen_titles = set()
        for idx, item in enumerate(news[:10]):
            title = (item.get("title") or "").strip()
            normalized_title = title.lower()
            if not title or normalized_title in seen_titles:
                continue
            seen_titles.add(normalized_title)
            headlines.append(title)
            tokens = extract_sentiment_tokens(title)
            positive_hits = len(tokens & POSITIVE_SENTIMENT_TERMS)
            negative_hits = len(tokens & NEGATIVE_SENTIMENT_TERMS)
            headline_score = min(positive_hits, 2) - min(negative_hits, 2)
            if headline_score != 0 and idx < 3:
                headline_score += 1 if headline_score > 0 else -1
            score += headline_score

        score = max(min(score, 6), -6)

        recommendation_key = (info.get("recommendationKey") or "").lower()
        analyst_opinions = safe_num(info.get("numberOfAnalystOpinions"))
        target_mean_price = safe_num(info.get("targetMeanPrice"))
        recommendation_mean = safe_num(info.get("recommendationMean"))

        if analyst_opinions is not None and analyst_opinions >= 15:
            analyst_scale = 1.0
        elif analyst_opinions is not None and analyst_opinions >= 5:
            analyst_scale = 0.75
        elif analyst_opinions is not None and analyst_opinions >= 1:
            analyst_scale = 0.5
        elif recommendation_key:
            analyst_scale = 0.5
        else:
            analyst_scale = 0.0

        analyst_points = 0
        if recommendation_key == "strong_buy":
            analyst_points = int(round((settings["sentiment_analyst_boost"] + 1) * analyst_scale))
        elif recommendation_key in {"buy", "outperform", "overweight"}:
            analyst_points = int(round(settings["sentiment_analyst_boost"] * analyst_scale))
        elif recommendation_key in {"underperform", "underweight", "sell", "strong_sell"}:
            base_penalty = settings["sentiment_analyst_boost"] + (1 if recommendation_key == "strong_sell" else 0)
            analyst_points = -int(round(base_penalty * analyst_scale))
        elif recommendation_mean is not None:
            if recommendation_mean <= 1.8:
                analyst_points = 1
            elif recommendation_mean >= 3.2:
                analyst_points = -1

        score += analyst_points

        if has_numeric_value(target_mean_price) and has_numeric_value(price) and price > 0:
            upside = (target_mean_price - price) / price
            strong_target_points = 2 if analyst_opinions is None or analyst_opinions >= 3 else 1
            moderate_target_points = 1 if analyst_opinions is None or analyst_opinions >= 3 else 0
            if upside > settings["sentiment_upside_high"]:
                score += strong_target_points
            elif upside > settings["sentiment_upside_mid"]:
                score += moderate_target_points
            elif upside < -settings["sentiment_downside_high"]:
                score -= strong_target_points
            elif upside < -settings["sentiment_downside_mid"]:
                score -= moderate_target_points

        score = int(max(min(round(score), 8), -8))

        return {
            "score": score,
            "verdict": score_to_sentiment(score),
            "recommendation_key": recommendation_key.upper() if recommendation_key else "N/A",
            "analyst_opinions": analyst_opinions,
            "target_mean_price": target_mean_price,
            "headline_count": len(headlines),
            "summary": " | ".join(headlines[:3]) if headlines else "No recent headlines available.",
        }

    def build_record_from_market_data(self, ticker, hist, info, news, settings=None):
        settings = get_model_settings() if settings is None else settings
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
        volatility_1m = calculate_realized_volatility(close, 22)
        volatility_1y = calculate_realized_volatility(close, 252)
        momentum_1m_risk_adjusted = safe_divide(momentum_1m, (volatility_1m / np.sqrt(12)) if has_numeric_value(volatility_1m) else None)
        range_position_52w, distance_52w_high, distance_52w_low = calculate_52w_context(close)
        trend_strength = calculate_trend_strength(price, sma50, sma200, momentum_1y)
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
        bench = get_sector_benchmarks(sector, settings)
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
        effective_sentiment_score = sentiment["score"] * (0.40 + 0.60 * sentiment_conviction / 100)
        effective_tech_score = tech_score
        if has_numeric_value(trend_strength):
            effective_tech_score += np.clip(trend_strength / 50, -1.0, 1.0)
        effective_f_score = f_score
        if quality_score >= 3:
            effective_f_score += 0.5
        elif quality_score <= -1.5:
            effective_f_score -= 0.5
        if stock_profile["primary_type"] in {"Dividend / Income Stocks", "Defensive Stocks"}:
            effective_f_score += np.clip(dividend_safety_score / 4, -0.5, 1.0)
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
            "Stock_Type": stock_profile["primary_type"],
            "Cap_Bucket": stock_profile["cap_bucket"],
            "Style_Tags": stock_profile["style_tags"],
            "Type_Strategy": stock_profile["type_strategy"],
            "Type_Confidence": stock_profile["type_confidence"],
            "Engine_Weight_Profile": engine_weight_profile,
            "Market_Cap": market_cap,
            "Dividend_Yield": dividend_yield,
            "Payout_Ratio": payout_ratio,
            "Equity_Beta": equity_beta,
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

    def analyze(self, ticker, settings=None, persist=True, preloaded=None):
        active_settings = get_model_settings() if settings is None else settings
        ticker = ticker.strip().upper()
        self.last_error = None
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
        )
        if record is None and self.last_error is None:
            self.last_error = (
                f"Unable to build an analysis for {ticker}. Yahoo returned incomplete or unusable market data."
            )
        if record is None and persist:
            existing = self.db.get_analysis(ticker)
            if not existing.empty:
                self.last_error = (
                    f"Live fetch failed for {ticker}; showing the most recent saved analysis instead."
                )
                return existing.iloc[0].to_dict()
        if record and persist:
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
    st.vega_lite_chart(chart_data, spec, use_container_width=True)


st.set_page_config(page_title="ZB Compiler", layout="wide", page_icon="SE")

db = get_database_manager()
bot = StockAnalyst(db)
portfolio_bot = PortfolioAnalyst(db)
model_settings = get_model_settings()
active_preset_name = detect_matching_preset(model_settings)
active_assumption_fingerprint = get_assumption_fingerprint(model_settings)

st.title("ZB Compiler")
st.caption(f"Version: {APP_VERSION}")

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
if os.environ.get("STOCK_ENGINE_SKIP_STARTUP_REFRESH") != "1":
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
        if st.button("Run Full Analysis", type="primary", use_container_width=True):
            if txt_input:
                with st.spinner(f"Running multiple engines on {txt_input}..."):
                    res = bot.analyze(txt_input)
                    if not res:
                        st.error(bot.last_error or "Unable to fetch enough market data for this ticker right now.")

    if txt_input:
        df = prepare_analysis_dataframe(db.get_analysis(txt_input.upper()))
        if not df.empty:
            row = df.iloc[0]
            sector_bench = get_sector_benchmarks(row["Sector"], model_settings)

            st.divider()
            col_main_1, col_main_2, col_main_3 = st.columns([1, 2, 1])
            with col_main_1:
                st.metric("Current Price", f"${row['Price']:,.2f}")
            with col_main_2:
                st.markdown(
                    f"<h2 style='text-align: center; color: {get_color(row['Verdict_Overall'])};'>VERDICT: {row['Verdict_Overall']}</h2>",
                    unsafe_allow_html=True,
                )
            with col_main_3:
                st.metric("Sector", str(row["Sector"]))

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
                        "note": "How cheap or expensive the stock looks.",
                        "tone": tone_from_metric_threshold(row["Score_Val"], good_min=1, bad_max=-1),
                        "help": ANALYSIS_HELP_TEXT["Valuation"],
                    },
                    {
                        "label": "Sentiment",
                        "value": format_int(row["Score_Sentiment"]),
                        "note": "News tone, analyst views, and target-price signals.",
                        "tone": tone_from_metric_threshold(row["Score_Sentiment"], good_min=1, bad_max=-1),
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
                        "label": "Confidence",
                        "value": format_value(row.get("Decision_Confidence"), "{:,.0f}", "/100"),
                        "note": "How strongly the model trusts its final call.",
                        "tone": tone_from_metric_threshold(row.get("Decision_Confidence"), good_min=70, bad_max=45),
                        "help": ANALYSIS_HELP_TEXT["Confidence"],
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
                        "label": "Sentiment Conviction",
                        "value": format_value(row.get("Sentiment_Conviction"), "{:,.0f}", "/100"),
                        "note": "How strong and well-supported the sentiment read looks.",
                        "tone": tone_from_metric_threshold(row.get("Sentiment_Conviction"), good_min=65, bad_max=35),
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

            tab_val, tab_fund, tab_tech, tab_sent = st.tabs(
                ["Valuation Engine", "Fundamental Engine", "Technical Engine", "Sentiment Engine"]
            )

            with tab_val:
                c_v1, c_v2 = st.columns([1, 2])
                with c_v1:
                    graham_discount = row.get("Graham Discount")
                    st.markdown(f"### Verdict: **{row['Verdict_Valuation']}**")
                    st.caption("This view asks a simple question: does the stock look cheap, expensive, or roughly fair compared with peers and fair-value estimates?")
                    render_analysis_signal_cards(
                        [
                            {
                                "label": "Graham Fair Value",
                                "value": f"${row['Graham_Number']:,.2f}",
                                "note": f"Compared with today's price, the gap is ${row['Price'] - row['Graham_Number']:,.2f}.",
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
                                "label": "Valuation Confidence",
                                "value": format_value(row.get("Valuation_Confidence"), "{:,.0f}", "/100"),
                                "note": "Higher means the valuation read is supported by more usable data.",
                                "tone": tone_from_metric_threshold(row.get("Valuation_Confidence"), good_min=70, bad_max=40),
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
                                "reference": format_value(sector_bench["PE"]),
                                "status": "Cheap" if tone_from_relative_multiple(row["PE_Ratio"], sector_bench["PE"]) == "good" else "Rich" if tone_from_relative_multiple(row["PE_Ratio"], sector_bench["PE"]) == "bad" else "Fair",
                                "tone": tone_from_relative_multiple(row["PE_Ratio"], sector_bench["PE"]),
                                "help": ANALYSIS_HELP_TEXT["P/E Ratio"],
                            },
                            {
                                "metric": "Forward P/E",
                                "value": format_value(row["Forward_PE"]),
                                "reference": format_value(sector_bench["PE"]),
                                "status": "Cheap" if tone_from_relative_multiple(row["Forward_PE"], sector_bench["PE"]) == "good" else "Rich" if tone_from_relative_multiple(row["Forward_PE"], sector_bench["PE"]) == "bad" else "Fair",
                                "tone": tone_from_relative_multiple(row["Forward_PE"], sector_bench["PE"]),
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
                                "reference": format_value(sector_bench["PS"]),
                                "status": "Cheap" if tone_from_relative_multiple(row["PS_Ratio"], sector_bench["PS"]) == "good" else "Rich" if tone_from_relative_multiple(row["PS_Ratio"], sector_bench["PS"]) == "bad" else "Fair",
                                "tone": tone_from_relative_multiple(row["PS_Ratio"], sector_bench["PS"]),
                                "help": ANALYSIS_HELP_TEXT["P/S Ratio"],
                            },
                            {
                                "metric": "EV/EBITDA",
                                "value": format_value(row["EV_EBITDA"]),
                                "reference": format_value(sector_bench["EV_EBITDA"]),
                                "status": "Cheap" if tone_from_relative_multiple(row["EV_EBITDA"], sector_bench["EV_EBITDA"]) == "good" else "Rich" if tone_from_relative_multiple(row["EV_EBITDA"], sector_bench["EV_EBITDA"]) == "bad" else "Fair",
                                "tone": tone_from_relative_multiple(row["EV_EBITDA"], sector_bench["EV_EBITDA"]),
                                "help": ANALYSIS_HELP_TEXT["EV/EBITDA"],
                            },
                            {
                                "metric": "P/B Ratio",
                                "value": format_value(row["PB_Ratio"]),
                                "reference": format_value(sector_bench["PB"]),
                                "status": "Cheap" if tone_from_relative_multiple(row["PB_Ratio"], sector_bench["PB"]) == "good" else "Rich" if tone_from_relative_multiple(row["PB_Ratio"], sector_bench["PB"]) == "bad" else "Fair",
                                "tone": tone_from_relative_multiple(row["PB_Ratio"], sector_bench["PB"]),
                                "help": ANALYSIS_HELP_TEXT["P/B Ratio"],
                            },
                        ],
                        reference_label="Benchmark",
                    )

            with tab_fund:
                c_f1, c_f2 = st.columns([1, 2])
                with c_f1:
                    st.markdown(f"### Verdict: **{row['Verdict_Fundamental']}**")
                    st.caption("This view focuses on business strength: profitability, growth, balance-sheet pressure, and short-term financial flexibility.")
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
                    st.caption("This view looks at what the recent news and analyst community are signaling, and how strong that signal really is.")
                    target_price = row["Target_Mean_Price"]
                    target_upside = safe_divide(target_price - row["Price"], row["Price"]) if has_numeric_value(target_price) and has_numeric_value(row["Price"]) else None
                    render_analysis_signal_cards(
                        [
                            {
                                "label": "Headlines",
                                "value": format_int(row["Sentiment_Headline_Count"]),
                                "note": "The number of recent headlines used in the sentiment read.",
                                "tone": tone_from_metric_threshold(row["Sentiment_Headline_Count"], good_min=1),
                                "help": ANALYSIS_HELP_TEXT["Headlines"],
                            },
                            {
                                "label": "Analyst View",
                                "value": str(row["Recommendation_Key"]),
                                "note": "The broad analyst recommendation signal the model found.",
                                "tone": tone_from_signal_text(
                                    row["Recommendation_Key"],
                                    positives={"BUY", "STRONG_BUY", "OUTPERFORM", "OVERWEIGHT"},
                                    negatives={"SELL", "STRONG_SELL", "UNDERPERFORM", "UNDERWEIGHT"},
                                ),
                                "help": ANALYSIS_HELP_TEXT["Analyst View"],
                            },
                            {
                                "label": "Target Mean",
                                "value": "N/A" if pd.isna(target_price) else f"${target_price:,.2f}",
                                "note": "The average analyst target price, when available.",
                                "tone": tone_from_metric_threshold(target_upside, good_min=model_settings["sentiment_upside_mid"], bad_max=-model_settings["sentiment_downside_mid"]),
                                "help": ANALYSIS_HELP_TEXT["Target Mean"],
                            },
                            {
                                "label": "Sentiment Conviction",
                                "value": format_value(row.get("Sentiment_Conviction"), "{:,.0f}", "/100"),
                                "note": "How strong and well-supported the sentiment signal looks.",
                                "tone": tone_from_metric_threshold(row.get("Sentiment_Conviction"), good_min=65, bad_max=35),
                                "help": ANALYSIS_HELP_TEXT["Sentiment Conviction"],
                            },
                        ],
                        columns=2,
                    )
                with c_s2:
                    st.write("What the model is seeing in the latest headlines")
                    st.caption(str(row["Sentiment_Summary"]))
        else:
            st.info("Run the full analysis to save this ticker into the shared research library.")

with compare_tab:
    st.subheader("Compare Stocks")
    st.caption("Rank a watchlist with the same four-engine model before deciding what deserves portfolio weight.")

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
            compare_submit = st.form_submit_button("Build Comparison", type="primary", use_container_width=True)

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
                    "label": "Sectors Covered",
                    "value": str(comparison_df["Sector"].nunique()),
                    "note": "More sectors usually means the list is less concentrated in one theme.",
                    "tone": "neutral",
                    "help": ANALYSIS_HELP_TEXT["Sectors Covered"],
                },
            ],
            columns=4,
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
                ("Confidence", ANALYSIS_HELP_TEXT["Confidence"]),
                ("Trend Strength", ANALYSIS_HELP_TEXT["Trend Strength"]),
                ("Quality Score", ANALYSIS_HELP_TEXT["Quality Score"]),
                ("Target Upside", ANALYSIS_HELP_TEXT["Target Mean"]),
                ("Graham Discount", ANALYSIS_HELP_TEXT["Graham Discount"]),
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
        comparison_display["Trend_Strength"] = comparison_display["Trend_Strength"].map(
            lambda value: format_value(value, "{:,.0f}")
        )
        comparison_display["Quality_Score"] = comparison_display["Quality_Score"].map(
            lambda value: format_value(value, "{:,.1f}")
        )
        comparison_display["Target Upside"] = comparison_display["Target Upside"].map(format_percent)
        comparison_display["Graham Discount"] = comparison_display["Graham Discount"].map(format_percent)
        st.dataframe(comparison_display, use_container_width=True)

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
            st.dataframe(scorecard, use_container_width=True)

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

        portfolio_submit = st.form_submit_button("Build Portfolio Recommendation", type="primary", use_container_width=True)

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
        st.dataframe(recommendations_display, use_container_width=True)

        exposure_col, metrics_col = st.columns([1, 2])
        with exposure_col:
            st.subheader("Sector Exposure")
            sector_display = sector_exposure.copy()
            sector_display["Recommended Weight"] = sector_display["Recommended Weight"].map(format_percent)
            st.dataframe(sector_display, use_container_width=True)

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
            st.dataframe(asset_display, use_container_width=True)

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
            run_sensitivity = st.form_submit_button("Run Sensitivity Check", type="primary", use_container_width=True)

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
                ("Confidence", ANALYSIS_HELP_TEXT["Confidence"]),
                ("Assumption Drift", ANALYSIS_HELP_TEXT["Assumption Drift"]),
                ("Fingerprint", ANALYSIS_HELP_TEXT["Fingerprint"]),
            ]
        )
        sensitivity_display = sensitivity_df.copy()
        sensitivity_display["Overall Score"] = sensitivity_display["Overall Score"].map(
            lambda value: format_value(value, "{:,.1f}")
        )
        sensitivity_display["Confidence"] = sensitivity_display["Confidence"].map(
            lambda value: format_value(value, "{:,.0f}", "/100")
        )
        sensitivity_display["Assumption Drift"] = sensitivity_display["Assumption Drift"].map(
            lambda value: format_value(value, "{:,.1f}", "%")
        )
        st.dataframe(sensitivity_display, use_container_width=True)

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
            run_backtest = st.form_submit_button("Run Backtest", type="primary", use_container_width=True)

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
                    backtest_profile = infer_stock_profile_from_snapshot(backtest_info, hist, model_settings)
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
            columns=3,
        )

        st.subheader("Equity Curve")
        chart_frame = history_display[["Date", "Strategy Equity", "Benchmark Equity"]].copy().set_index("Date")
        st.line_chart(chart_frame, use_container_width=True)

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
            st.dataframe(trade_log_display, use_container_width=True)

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
            st.dataframe(closed_trades_display, use_container_width=True)

with library_tab:
    st.subheader("Research Library")
    st.caption("Browse everything saved in the shared database so the research process stays visible across users and sessions.")
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
        database_bytes = build_database_download_bytes(DB_PATH)
        export_col_1, export_col_2 = st.columns(2)
        with export_col_1:
            st.download_button(
                "Download Database",
                data=database_bytes,
                file_name=DB_PATH.name,
                mime="application/x-sqlite3",
                disabled=not bool(database_bytes),
                use_container_width=True,
            )
        with export_col_2:
            st.download_button(
                "Download Library CSV",
                data=b"",
                file_name="stock_engine_library.csv",
                mime="text/csv",
                disabled=True,
                use_container_width=True,
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
        database_bytes = build_database_download_bytes(DB_PATH)
        library_csv_bytes = build_library_csv_bytes(export_frame)
        export_col_1, export_col_2 = st.columns(2)
        with export_col_1:
            st.download_button(
                "Download Database",
                data=database_bytes,
                file_name=DB_PATH.name,
                mime="application/x-sqlite3",
                disabled=not bool(database_bytes),
                use_container_width=True,
            )
        with export_col_2:
            st.download_button(
                "Download Library CSV",
                data=library_csv_bytes,
                file_name="stock_engine_library.csv",
                mime="text/csv",
                disabled=export_frame.empty,
                use_container_width=True,
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

            st.caption(f"Shared database path: {DB_PATH}")

            render_help_legend(
                [
                    ("Composite Score", ANALYSIS_HELP_TEXT["Composite Score"]),
                    ("Confidence", ANALYSIS_HELP_TEXT["Confidence"]),
                    ("Trend Strength", ANALYSIS_HELP_TEXT["Trend Strength"]),
                    ("Quality Score", ANALYSIS_HELP_TEXT["Quality Score"]),
                    ("Target Upside", ANALYSIS_HELP_TEXT["Target Mean"]),
                    ("Graham Discount", ANALYSIS_HELP_TEXT["Graham Discount"]),
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
                    "Freshness",
                    "Last_Updated",
                ]
            ].copy()
            library_display["Price"] = library_display["Price"].map(lambda value: f"${value:,.2f}" if pd.notna(value) else "N/A")
            library_display["Decision_Confidence"] = library_display["Decision_Confidence"].map(
                lambda value: format_value(value, "{:,.0f}", "/100")
            )
            library_display["Trend_Strength"] = library_display["Trend_Strength"].map(
                lambda value: format_value(value, "{:,.0f}")
            )
            library_display["Quality_Score"] = library_display["Quality_Score"].map(
                lambda value: format_value(value, "{:,.1f}")
            )
            library_display["Target Upside"] = library_display["Target Upside"].map(format_percent)
            library_display["Graham Discount"] = library_display["Graham Discount"].map(format_percent)
            st.dataframe(library_display, use_container_width=True)

            library_left, library_right = st.columns(2)
            with library_left:
                st.subheader("Sector Summary")
                render_help_legend(
                    [
                        ("Avg Composite Score", ANALYSIS_HELP_TEXT["Avg Composite Score"]),
                        ("Avg Target Upside", ANALYSIS_HELP_TEXT["Avg Target Upside"]),
                    ]
                )
                sector_summary = (
                    filtered_library.groupby("Sector", dropna=False)
                    .agg(
                        Records=("Ticker", "count"),
                        Avg_Composite_Score=("Composite Score", "mean"),
                        Avg_Target_Upside=("Target Upside", "mean"),
                    )
                    .reset_index()
                    .sort_values(["Records", "Avg_Composite_Score"], ascending=[False, False])
                )
                sector_summary["Avg_Composite_Score"] = sector_summary["Avg_Composite_Score"].map(
                    lambda value: format_value(value, "{:,.1f}")
                )
                sector_summary["Avg_Target_Upside"] = sector_summary["Avg_Target_Upside"].map(format_percent)
                st.dataframe(sector_summary, use_container_width=True)

            with library_right:
                st.subheader("Top Conviction Names")
                render_help_legend(
                    [
                        ("Composite Score", ANALYSIS_HELP_TEXT["Composite Score"]),
                        ("Target Upside", ANALYSIS_HELP_TEXT["Target Mean"]),
                        ("Freshness", ANALYSIS_HELP_TEXT["Freshness"]),
                    ]
                )
                conviction_table = filtered_library[
                    ["Ticker", "Verdict_Overall", "Composite Score", "Target Upside", "Freshness"]
                ].head(10).copy()
                conviction_table["Target Upside"] = conviction_table["Target Upside"].map(format_percent)
                st.dataframe(conviction_table, use_container_width=True)

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

    st.dataframe(pd.DataFrame(CHANGELOG_ENTRIES), use_container_width=True)

    st.subheader("What Changed Most Recently")
    st.write("- The model now adds ten extra diagnostics such as trend strength, 52-week range context, volatility-adjusted momentum, quality score, dividend safety, valuation breadth, sentiment conviction, and explicit risk flags.")
    st.write("- The model now assigns each stock a primary type such as Growth, Value, Dividend, Cyclical, Defensive, Blue-Chip, size-based, or Speculative and uses that profile in verdict and backtest logic.")
    st.write("- The backtest now holds a core position during durable bullish regimes, exits later on deeper breakdowns, and reports win rate plus average closed-trade return.")
    st.write("- The Options tab now includes inline ? explanations for every slider and preset selector.")
    st.write("- Regime, confidence, and decision-note transparency remain visible across stock, compare, sensitivity, and library views.")

with methodology_tab:
    st.subheader("Methodology and Transparency")
    st.caption("This tab shows how the app forms verdicts, why the portfolio engine chooses certain weights, and what assumptions sit underneath the UI.")

    st.subheader("Model Flow")
    methodology_flow = pd.DataFrame(
        [
            {"Step": 1, "What Happens": "Download one year of price history plus company profile and news from Yahoo Finance."},
            {"Step": 2, "What Happens": "Score four engines: technical, fundamental, valuation, and sentiment."},
            {"Step": 3, "What Happens": "Classify market regime, measure engine agreement, and estimate decision confidence."},
            {"Step": 4, "What Happens": "Apply hold buffers and data-quality guardrails before publishing the final verdict."},
            {"Step": 5, "What Happens": "Store the full result with timestamp, assumption fingerprint, and quality stats in the shared library."},
        ]
    )
    st.dataframe(methodology_flow, use_container_width=True)

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
                    "Uses": "P/E, forward P/E, PEG, P/S, EV/EBITDA, P/B, Graham value, premium/discount bands",
                    "Strong Signals": "Positive earnings, cheaper-than-sector multiples, discount to intrinsic value",
                },
                {
                    "Engine": "Sentiment",
                    "Uses": "Headline tone by word match, analyst recommendation, analyst depth, target mean price",
                    "Strong Signals": "Consistently constructive headlines and analyst targets with supporting coverage",
                },
            ]
        )
        st.dataframe(engine_framework, use_container_width=True)

    with methodology_col_2:
        st.subheader("Verdict Thresholds")
        verdict_table = pd.DataFrame(
            [
                {"Output": "Technical score", "Rule": ">= 4 strong buy, >= 2 buy, <= -2 sell, <= -4 strong sell"},
                {"Output": "Sentiment score", "Rule": ">= 3 positive, <= -3 negative, otherwise mixed"},
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
                        f"{model_settings['overall_strong_sell_threshold']:.0f}; mixed regimes, low confidence, and low-quality data are pushed toward hold"
                    ),
                },
            ]
        )
        st.dataframe(verdict_table, use_container_width=True)

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
    st.dataframe(stock_type_framework, use_container_width=True)

    st.subheader("Refinement Layer")
    refinement_df = pd.DataFrame(
        [
            {"Refinement": 1, "What Changed": "Trend Strength", "Purpose": "Uses SMA structure plus 1Y momentum as a continuous trend quality signal."},
            {"Refinement": 2, "What Changed": "52-Week Range Context", "Purpose": "Tracks whether price is breaking out, mid-range, or stuck near lows."},
            {"Refinement": 3, "What Changed": "Volatility-Adjusted Momentum", "Purpose": "Rewards momentum that is strong relative to realized volatility instead of raw price change alone."},
            {"Refinement": 4, "What Changed": "Quality Score", "Purpose": "Combines profitability, leverage, liquidity, and growth consistency into a cleaner business-quality signal."},
            {"Refinement": 5, "What Changed": "Dividend Safety Score", "Purpose": "Checks whether income stocks appear to have a more sustainable payout profile."},
            {"Refinement": 6, "What Changed": "Valuation Breadth", "Purpose": "Scales valuation influence based on how many usable valuation signals are actually available."},
            {"Refinement": 7, "What Changed": "Sentiment Conviction", "Purpose": "Separates noisy sentiment from higher-conviction sentiment backed by coverage and target alignment."},
            {"Refinement": 8, "What Changed": "Risk Flags", "Purpose": "Collects visible red flags like negative EPS, high debt, weak liquidity, high volatility, and speculation."},
            {"Refinement": 9, "What Changed": "Dynamic Engine Weights", "Purpose": "Lets Growth, Value, Income, Cyclical, and Speculative names use different engine mixes."},
            {"Refinement": 10, "What Changed": "Profile-Aware Trailing Stops", "Purpose": "Makes the backtest protect gains differently for growth, defensive, cyclical, and speculative stocks."},
        ]
    )
    st.dataframe(refinement_df, use_container_width=True)

    st.subheader("Decision Guardrails")
    guardrail_df = pd.DataFrame(
        [
            {"Guardrail": "Trend Tolerance", "Purpose": "Avoids flipping trend signals on tiny moves around the moving averages."},
            {"Guardrail": "Stretch Limit", "Purpose": "Penalizes overextended rallies and recognizes washed-out rebounds before chasing price."},
            {"Guardrail": "Hold Buffer", "Purpose": "Makes mixed-engine or transition regimes require extra evidence before becoming directional."},
            {"Guardrail": "Confidence Floor", "Purpose": "Downgrades weak-conviction Buy or Sell calls back toward Hold."},
            {"Guardrail": "Data Quality Check", "Purpose": "Reduces conviction when too many important metrics are missing."},
        ]
    )
    st.dataframe(guardrail_df, use_container_width=True)

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
    st.dataframe(portfolio_workflow, use_container_width=True)

    methodology_col_3, methodology_col_4 = st.columns(2)
    with methodology_col_3:
        st.subheader("Sector Valuation Benchmarks")
        scaled_benchmarks = {
            sector: {
                metric: value * model_settings["valuation_benchmark_scale"]
                for metric, value in benchmarks.items()
            }
            for sector, benchmarks in SECTOR_BENCHMARKS.items()
        }
        sector_benchmarks_df = (
            pd.DataFrame.from_dict(scaled_benchmarks, orient="index")
            .reset_index()
            .rename(columns={"index": "Sector"})
        )
        st.dataframe(sector_benchmarks_df, use_container_width=True)

    with methodology_col_4:
        st.subheader("Current Model Assumptions")
        library_snapshot = prepare_analysis_dataframe(db.get_all_analyses())
        assumptions_df = pd.DataFrame(
            [
                {"Setting": "Database Path", "Value": str(DB_PATH)},
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
                    "Setting": "Confidence Floor",
                    "Value": f"{model_settings['decision_min_confidence']:.0f}/100",
                },
                {
                    "Setting": "Backtest Cooldown",
                    "Value": f"{int(round(model_settings['backtest_cooldown_days']))} days",
                },
                {
                    "Setting": "Benchmark Scale",
                    "Value": f"{model_settings['valuation_benchmark_scale']:.2f}x",
                },
                {
                    "Setting": "Assumption Drift vs Defaults",
                    "Value": f"{calculate_assumption_drift(model_settings):.1f}%",
                },
                {"Setting": "Positive Sentiment Terms", "Value": len(POSITIVE_SENTIMENT_TERMS)},
                {"Setting": "Negative Sentiment Terms", "Value": len(NEGATIVE_SENTIMENT_TERMS)},
                {"Setting": "Cached Analyses in Library", "Value": len(library_snapshot)},
            ]
        )
        st.dataframe(assumptions_df, use_container_width=True)

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
        if st.button("Apply Preset", use_container_width=True):
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
                "Confidence Floor": f"{values['decision_min_confidence']:.0f}/100",
                "Cooldown": f"{int(round(values['backtest_cooldown_days']))}d",
            }
            for name, values in preset_catalog.items()
        ]
    )
    st.subheader("Preset Snapshot")
    st.dataframe(preset_snapshot, use_container_width=True)

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
    options_metrics[2].metric("Benchmark Scale", format_value(model_settings["valuation_benchmark_scale"], "{:,.2f}", "x"))
    options_metrics[3].metric("Weight Spread", format_value(max(weight_values) - min(weight_values), "{:,.1f}"))
    options_metrics[4].metric("Confidence Floor", format_value(model_settings["decision_min_confidence"], "{:,.0f}", "/100"))

    if assumption_drift > 35:
        st.warning("Your active assumptions are materially different from the default model. Expect results to diverge more from the baseline.")
    else:
        st.info("The controls are intentionally range-limited so the model remains stable even when you tune it.")

    if st.button("Restore Default Assumptions", use_container_width=False):
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
        weight_sentiment = weight_col_4.slider("Sentiment", 0.5, 1.5, float(model_settings["weight_sentiment"]), 0.1, help=OPTIONS_HELP_TEXT["weight_sentiment"])

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
            "Confidence Floor",
            35,
            80,
            int(round(model_settings["decision_min_confidence"])),
            1,
            help=OPTIONS_HELP_TEXT["decision_min_confidence"],
        )

        st.subheader("Valuation, Sentiment, and Portfolio")
        vs_col_1, vs_col_2, vs_col_3, vs_col_4 = st.columns(4)
        valuation_benchmark_scale = vs_col_1.slider(
            "Benchmark Scale",
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
        )
        sentiment_upside_mid = vs_col_8.slider(
            "Moderate Upside (%)",
            2,
            15,
            int(round(model_settings["sentiment_upside_mid"] * 100)),
            1,
            help=OPTIONS_HELP_TEXT["sentiment_upside_mid"],
        )
        sentiment_upside_high = vs_col_9.slider(
            "Strong Upside (%)",
            8,
            30,
            int(round(model_settings["sentiment_upside_high"] * 100)),
            1,
            help=OPTIONS_HELP_TEXT["sentiment_upside_high"],
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
        )
        sentiment_downside_high = downside_col_2.slider(
            "Deep Downside (%)",
            8,
            30,
            int(round(model_settings["sentiment_downside_high"] * 100)),
            1,
            help=OPTIONS_HELP_TEXT["sentiment_downside_high"],
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

        save_options = st.form_submit_button("Save Assumptions", type="primary", use_container_width=True)

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
