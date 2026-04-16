# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

import constants as const


def render_changelog_view():
    st.subheader("Changelog")
    st.caption("Recent updates to the stock model, portfolio engine, and research UI live here so the app stays inspectable over time.")

    changelog_metrics = st.columns(3)
    changelog_metrics[0].metric("Latest Logged Update", const.CHANGELOG_ENTRIES[0]["Date"])
    changelog_metrics[1].metric("Logged Changes", str(len(const.CHANGELOG_ENTRIES)))
    changelog_metrics[2].metric("App Version", const.APP_VERSION)

    st.dataframe(pd.DataFrame(const.CHANGELOG_ENTRIES), width="stretch")

    st.subheader("What Changed Most Recently")
    st.write("- The model now adds ten extra diagnostics such as trend strength, 52-week range context, volatility-adjusted momentum, quality score, dividend safety, valuation breadth, sentiment conviction, and explicit risk flags.")
    st.write("- The model now assigns each stock a primary type such as Growth, Value, Dividend, Cyclical, Defensive, Blue-Chip, size-based, or Speculative and uses that profile in verdict and backtest logic.")
    st.write("- The backtest now holds a core position during durable bullish regimes, exits later on deeper breakdowns, and reports win rate plus average closed-trade return.")
    st.write("- The Options tab now includes inline ? explanations for every slider and preset selector.")
    st.write("- Regime, consistency, and decision-note transparency remain visible across stock, compare, sensitivity, and library views.")
