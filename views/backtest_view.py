# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

import analytics_decision as decision
import backtest
import constants as const
import fetch
import utils_fmt as fmt
import utils_ui as ui


def render_backtest_view(db, model_settings, active_preset_name, active_assumption_fingerprint):
    st.subheader("Technical Signal Backtest")
    st.warning(
        "**Technical engine only.** This replay uses the Tech Score (price trend, SMA, RSI, MACD, momentum) "
        "to generate positions. It does **not** reconstruct historical fundamentals, valuation multiples, or "
        "news sentiment. The composite model you see on the Research tab has never been backtested. "
        "Do not use this chart to validate the full model's Buy/Sell verdicts."
    )
    st.caption(
        "Replay uses stock-type-aware core sizing, trailing-stop behaviour, and deeper-breakdown "
        "confirmation before fully exiting."
    )

    backtest_default_ticker = st.session_state.get("backtest_last_ticker") or st.session_state.get("single_ticker", "")

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
        cleaned_ticker = fmt.normalize_ticker(backtest_ticker)
        if not cleaned_ticker:
            st.error("Enter a ticker to run a backtest.")
        else:
            with st.spinner(f"Replaying technical signals on {cleaned_ticker}..."):
                hist, backtest_error = fetch.fetch_ticker_history_with_retry(cleaned_ticker, backtest_period)
                backtest_profile = {}
                saved_backtest_row = db.get_analysis(cleaned_ticker)
                if not saved_backtest_row.empty:
                    backtest_profile = decision.extract_stock_profile_from_saved_row(saved_backtest_row.iloc[0])
                if not backtest_profile.get("primary_type") or backtest_profile.get("primary_type") == "Legacy":
                    backtest_info, _ = fetch.fetch_ticker_info_with_retry(cleaned_ticker)
                    if not backtest_info and not saved_backtest_row.empty:
                        backtest_info = fetch.build_info_fallback_from_saved_analysis(saved_backtest_row.iloc[0])
                    backtest_profile = decision.infer_stock_profile_from_snapshot(
                        backtest_info,
                        hist,
                        model_settings,
                        db=db,
                        ticker=cleaned_ticker,
                    )
                backtest_result = backtest.compute_technical_backtest(hist, model_settings, stock_profile=backtest_profile)

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
        st.info(
            "Results below reflect the **technical rules only** — not the fundamental, valuation, or "
            "sentiment engines. Any outperformance vs buy-and-hold is attributable solely to the "
            "price/momentum rules, not to the composite verdict model."
        )
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

        ui.render_analysis_signal_cards(
            [
                {
                    "label": "Tech-Rule Return",
                    "value": fmt.format_percent(backtest_metrics["Strategy Total Return"]),
                    "note": "The total return generated by the technical trading rules in this replay.",
                    "tone": ui.tone_from_metric_threshold(backtest_metrics["Strategy Total Return"], good_min=0.10, bad_max=-0.05),
                    "help": const.ANALYSIS_HELP_TEXT["Strategy Return"],
                },
                {
                    "label": "Benchmark Return",
                    "value": fmt.format_percent(backtest_metrics["Benchmark Total Return"]),
                    "note": "This is what simple buy-and-hold would have produced over the same period.",
                    "tone": ui.tone_from_metric_threshold(backtest_metrics["Benchmark Total Return"], good_min=0.10, bad_max=-0.05),
                    "help": const.ANALYSIS_HELP_TEXT["Benchmark Return"],
                },
                {
                    "label": "Tech vs Buy-and-Hold",
                    "value": fmt.format_percent(backtest_metrics["Relative Return"]),
                    "note": "Positive means the tech rules beat buy-and-hold. Does not reflect the full composite model.",
                    "tone": ui.tone_from_metric_threshold(backtest_metrics["Relative Return"], good_min=0.0, bad_max=-0.05),
                    "help": const.ANALYSIS_HELP_TEXT["Relative vs Benchmark"],
                },
                {
                    "label": "Strategy Sharpe",
                    "value": fmt.format_value(backtest_metrics["Strategy Sharpe"]),
                    "note": "This compares the strategy's return with the volatility it took to earn it.",
                    "tone": ui.tone_from_metric_threshold(backtest_metrics["Strategy Sharpe"], good_min=0.8, bad_max=0.2),
                    "help": const.ANALYSIS_HELP_TEXT["Strategy Sharpe"],
                },
                {
                    "label": "Win Rate",
                    "value": fmt.format_percent(backtest_metrics["Win Rate"]),
                    "note": "This only counts closed trades, so it can stay unavailable when nothing closed.",
                    "tone": ui.tone_from_metric_threshold(backtest_metrics["Win Rate"], good_min=0.55, bad_max=0.40),
                    "help": const.ANALYSIS_HELP_TEXT["Win Rate"],
                },
                {
                    "label": "Trading Costs",
                    "value": fmt.format_percent(backtest_metrics["Trading Costs"]),
                    "note": "Estimated transaction costs deducted when the replay changes exposure.",
                    "tone": "neutral",
                    "help": const.ANALYSIS_HELP_TEXT["Trading Costs"],
                },
                {
                    "label": "Max Drawdown",
                    "value": fmt.format_percent(backtest_metrics["Strategy Max Drawdown"]),
                    "note": "This shows the worst peak-to-trough drop during the replay.",
                    "tone": ui.tone_from_metric_threshold(backtest_metrics["Strategy Max Drawdown"], good_min=-0.12, bad_max=-0.30),
                    "help": const.ANALYSIS_HELP_TEXT["Max Drawdown"],
                },
            ],
            columns=6,
        )

        ui.render_analysis_signal_cards(
            [
                {
                    "label": "Average Exposure",
                    "value": fmt.format_percent(backtest_metrics["Average Exposure"]),
                    "note": "Higher means the replay stayed invested more consistently instead of sitting in cash.",
                    "tone": ui.tone_from_metric_threshold(backtest_metrics["Average Exposure"], good_min=0.75, bad_max=0.45),
                    "help": const.ANALYSIS_HELP_TEXT["Average Exposure"],
                },
                {
                    "label": "Upside Capture",
                    "value": fmt.format_percent(backtest_metrics["Upside Capture"]),
                    "note": "This compares the strategy's gain with a positive buy-and-hold gain over the same window.",
                    "tone": ui.tone_from_metric_threshold(backtest_metrics["Upside Capture"], good_min=0.90, bad_max=0.60),
                    "help": const.ANALYSIS_HELP_TEXT["Upside Capture"],
                },
                {
                    "label": "Position Changes",
                    "value": str(int(backtest_metrics["Position Changes"])),
                    "note": "Entries, exits, adds, and reductions all count toward this total.",
                    "tone": "neutral",
                    "help": const.ANALYSIS_HELP_TEXT["Position Changes"],
                },
                {
                    "label": "Closed Trades",
                    "value": str(int(backtest_metrics["Closed Trades"])),
                    "note": "Only completed trades count here, not open positions that are still running.",
                    "tone": "neutral",
                    "help": const.ANALYSIS_HELP_TEXT["Closed Trades"],
                },
                {
                    "label": "Avg Trade Return",
                    "value": fmt.format_percent(backtest_metrics["Average Trade Return"]),
                    "note": "This is the average result across the strategy's closed trades.",
                    "tone": ui.tone_from_metric_threshold(backtest_metrics["Average Trade Return"], good_min=0.03, bad_max=-0.02),
                    "help": const.ANALYSIS_HELP_TEXT["Avg Trade Return"],
                },
            ],
            columns=5,
        )

        st.subheader("Equity Curve")
        chart_frame = history_display[["Date", "Strategy Equity", "Benchmark Equity"]].copy().set_index("Date")
        st.line_chart(chart_frame, width="stretch")

        st.subheader("Position Change Log")
        ui.render_help_legend(
            [
                ("Signal", const.ANALYSIS_HELP_TEXT["Technical"]),
                ("Position", const.ANALYSIS_HELP_TEXT["Position Changes"]),
            ]
        )
        if trade_log_display.empty:
            st.info("No entries or exits were generated for this period under the active technical settings.")
        else:
            trade_log_display["Date"] = pd.to_datetime(trade_log_display["Date"]).dt.strftime("%Y-%m-%d")
            trade_log_display["Close"] = trade_log_display["Close"].map(lambda value: f"${value:,.2f}")
            trade_log_display["Position"] = trade_log_display["Position"].map(fmt.format_percent)
            if "Trading Cost" in trade_log_display.columns:
                trade_log_display["Trading Cost"] = trade_log_display["Trading Cost"].map(fmt.format_percent)
            st.dataframe(trade_log_display, width="stretch")

        st.subheader("Closed Trades")
        ui.render_help_legend(
            [
                ("Return", const.ANALYSIS_HELP_TEXT["Avg Trade Return"]),
                ("Position Size", const.ANALYSIS_HELP_TEXT["Position Changes"]),
            ]
        )
        if closed_trades_display.empty:
            st.info("No closed trades were realized in this window, so win rate is not available yet.")
        else:
            closed_trades_display["Entry Date"] = pd.to_datetime(closed_trades_display["Entry Date"]).dt.strftime("%Y-%m-%d")
            closed_trades_display["Exit Date"] = pd.to_datetime(closed_trades_display["Exit Date"]).dt.strftime("%Y-%m-%d")
            closed_trades_display["Entry Price"] = closed_trades_display["Entry Price"].map(lambda value: f"${value:,.2f}")
            closed_trades_display["Exit Price"] = closed_trades_display["Exit Price"].map(lambda value: f"${value:,.2f}")
            closed_trades_display["Position Size"] = closed_trades_display["Position Size"].map(fmt.format_percent)
            closed_trades_display["Return"] = closed_trades_display["Return"].map(fmt.format_percent)
            st.dataframe(closed_trades_display, width="stretch")
