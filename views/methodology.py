# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

import constants as const
import analysis_prep as prep


def render_methodology_view(db, model_settings, active_preset_name, active_assumption_fingerprint):
    st.subheader("How This App Works")
    st.caption("A plain-English guide to how the app scores stocks and issues verdicts.")

    # ── Step 1: How It Works ──────────────────────────────────────────────────
    st.subheader("The Process")
    flow_df = pd.DataFrame(
        [
            {"Step": "1 – Gather", "What the App Does": "Downloads a year of price history, company financials, news headlines, and analyst ratings from Yahoo Finance."},
            {"Step": "2 – Score",  "What the App Does": "Runs three scoring engines — Technical, Fundamental, and Valuation — and adds up the results into a single score."},
            {"Step": "3 – Verdict","What the App Does": "Compares the total score to set thresholds and issues a verdict: Strong Buy, Buy, Hold, Sell, or Strong Sell."},
        ]
    )
    st.dataframe(flow_df, use_container_width=True)

    # ── Step 2: Three Engines ─────────────────────────────────────────────────
    st.subheader("The Three Scoring Engines")
    st.caption("Each engine contributes a score. The weighted sum of all three determines the final verdict.")
    engine_df = pd.DataFrame(
        [
            {
                "Engine": "Technical",
                "What It Measures": "Price trends, momentum, RSI, and MACD",
                "Think of It As…": "Is the stock moving in the right direction?",
            },
            {
                "Engine": "Fundamental",
                "What It Measures": "Profitability, revenue growth, debt levels, and liquidity",
                "Think of It As…": "Is the underlying business financially healthy?",
            },
            {
                "Engine": "Valuation",
                "What It Measures": "P/E, P/S, EV/EBITDA compared to peers, plus Graham value",
                "Think of It As…": "Is the stock cheap or expensive relative to similar companies?",
            },
        ]
    )
    st.dataframe(engine_df, use_container_width=True)
    st.caption("Sentiment (news headlines and analyst ratings) is shown as context only — it does not feed into the score.")

    # ── Step 3: Verdicts ──────────────────────────────────────────────────────
    st.subheader("What Each Verdict Means")
    verdict_df = pd.DataFrame(
        [
            {"Verdict": "Strong Buy",  "Meaning": f"All three engines are positive and the score clears {model_settings['overall_strong_buy_threshold']:.0f}. High conviction."},
            {"Verdict": "Buy",         "Meaning": f"Mostly positive signals; score is above {model_settings['overall_buy_threshold']:.0f}."},
            {"Verdict": "Hold",        "Meaning": "Mixed or insufficient signals. The app stays neutral when it lacks consistent evidence."},
            {"Verdict": "Sell",        "Meaning": f"Mostly negative signals; score is below {model_settings['overall_sell_threshold']:.0f}."},
            {"Verdict": "Strong Sell", "Meaning": f"All three engines are negative and the score falls below {model_settings['overall_strong_sell_threshold']:.0f}. High conviction."},
        ]
    )
    st.dataframe(verdict_df, use_container_width=True)
    st.info("The app is biased toward **Hold** when signals conflict — it needs consistent evidence before going directional.")

    # ── Step 4: Stock Type Adjustments ────────────────────────────────────────
    st.subheader("How the App Adapts by Stock Type")
    st.caption("Different kinds of stocks are judged on different criteria. The app detects the type automatically and shifts the engine weights accordingly.")
    type_df = pd.DataFrame(
        [
            {"Stock Type": "Growth",            "How the App Adapts": "Trend and momentum carry more weight than valuation multiples."},
            {"Stock Type": "Value",             "How the App Adapts": "Valuation and balance-sheet strength carry more weight."},
            {"Stock Type": "Income / Dividend", "How the App Adapts": "Payout sustainability matters most alongside stable fundamentals."},
            {"Stock Type": "Cyclical",          "How the App Adapts": "Timing relative to the economic cycle gets extra attention."},
            {"Stock Type": "Speculative / Penny","How the App Adapts": "A higher score is required to issue a Buy; maximum conviction is capped."},
        ]
    )
    st.dataframe(type_df, use_container_width=True)

    # ── Step 5: Current Settings ──────────────────────────────────────────────
    st.subheader("Current Settings")
    st.caption("The active configuration driving today's scores and verdicts.")
    library_snapshot = prep.prepare_analysis_dataframe(db.get_all_analyses())
    settings_df = pd.DataFrame(
        [
            {"Setting": "Active Profile",    "Value": active_preset_name},
            {"Setting": "Storage Mode",      "Value": "Postgres" if db.storage_backend == "postgres" else ("Persistent SQLite" if db.uses_persistent_storage else "In-memory")},
            {
                "Setting": "Engine Weights",
                "Value": (
                    f"Technical {model_settings['weight_technical']:.1f} | "
                    f"Fundamental {model_settings['weight_fundamental']:.1f} | "
                    f"Valuation {model_settings['weight_valuation']:.1f}"
                ),
            },
            {"Setting": "RSI Oversold / Overbought", "Value": f"{int(model_settings['tech_rsi_oversold'])} / {int(model_settings['tech_rsi_overbought'])}"},
            {
                "Setting": "Verdict Thresholds (Strong Buy / Buy / Sell / Strong Sell)",
                "Value": (
                    f"{model_settings['overall_strong_buy_threshold']:.0f} / "
                    f"{model_settings['overall_buy_threshold']:.0f} / "
                    f"{model_settings['overall_sell_threshold']:.0f} / "
                    f"{model_settings['overall_strong_sell_threshold']:.0f}"
                ),
            },
            {"Setting": "Hold Buffer",       "Value": f"{model_settings['decision_hold_buffer']:.1f}"},
            {"Setting": "Peer Group Size",   "Value": const.PEER_GROUP_SIZE},
            {"Setting": "Cached Analyses",   "Value": len(library_snapshot)},
        ]
    )
    settings_df["Value"] = settings_df["Value"].map(str)
    st.dataframe(settings_df, use_container_width=True)
