# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

import constants as const
import settings
import utils_fmt as fmt
import analysis_prep as prep


def render_methodology_view(db, model_settings, active_preset_name, active_assumption_fingerprint):
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
                {"Step": 2, "What Happens": f"Search a cached universe for the {const.PEER_GROUP_SIZE} closest companies using those characteristics."},
                {"Step": 3, "What Happens": "Average the peer valuation multiples and use those averages as the main comparison set."},
                {"Step": 4, "What Happens": "If the peer set is too thin or missing too many usable metrics, fall back to scaled sector benchmarks."},
                {"Step": 5, "What Happens": "Keep Graham value separate and move DCF to a manual lab so cash-flow work stays optional and adjustable."},
            ]
        )
        st.dataframe(peer_workflow_df, width="stretch")

    with methodology_col_4:
        st.subheader("Current Model Assumptions")
        library_snapshot = prep.prepare_analysis_dataframe(db.get_all_analyses())
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
                {"Setting": "Default Benchmark", "Value": const.DEFAULT_BENCHMARK_TICKER},
                {"Setting": "Default Portfolio Universe", "Value": const.DEFAULT_PORTFOLIO_TICKERS},
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
                    "Value": const.PEER_GROUP_SIZE,
                },
                {
                    "Setting": "Fallback Benchmark Scale",
                    "Value": f"{model_settings['valuation_benchmark_scale']:.2f}x",
                },
                {
                    "Setting": "Assumption Drift vs Defaults",
                    "Value": f"{settings.calculate_assumption_drift(model_settings):.1f}%",
                },
                {"Setting": "Event Study Max Events", "Value": 5},
                {"Setting": "Backtest Transaction Cost", "Value": f"{model_settings['backtest_transaction_cost_bps']:.1f} bps"},
                {"Setting": "Cached Analyses in Library", "Value": len(library_snapshot)},
            ]
        )
        assumptions_df["Value"] = assumptions_df["Value"].map(str)
        st.dataframe(assumptions_df, width="stretch")
