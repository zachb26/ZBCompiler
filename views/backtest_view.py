# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

import analytics_decision as decision
import backtest
import constants as const
import fetch
import utils_fmt as fmt
import utils_ui as ui


def _render_composite_validation(backtest_config: dict, model_settings: dict) -> None:
    """Expander with walk-forward composite model verdict hit-rate table."""
    ticker = backtest_config.get("ticker", "")
    if not ticker:
        return

    with st.expander("Composite Model Validation — Look-Ahead Warning (Experimental)", expanded=False):
        st.error(
            "**Look-ahead contamination — results are not true out-of-sample validation.** "
            "Three live inputs are mixed with historical prices: "
            "(1) shares outstanding is the *current* figure, not what it was at each quarter; "
            "(2) yfinance quarterly filings may include post-hoc restatements not visible at original filing dates; "
            "(3) sector valuation benchmarks are static constants, not contemporaneous peer multiples. "
            "For reliable forward validation, use the **Technical Signal Backtest** above (price-derived signals only)."
        )
        st.caption(
            "Reconstructs composite verdicts at each quarter using historical yfinance fundamentals "
            "(~8 quarters depth), then tracks 1M/3M/12M forward price returns by verdict bucket. "
            "EV/EBITDA and PEG excluded (not available historically). Filing lag approximated as 45 days. "
            "No guardrails, hold buffer, or confidence guard applied. Sentiment score = 0 (same as live model)."
        )

        run_col, _ = st.columns([1, 3])
        with run_col:
            run_composite = st.button(
                f"Run composite validation for {ticker}",
                key="run_composite_validation",
                type="primary",
            )

        if run_composite:
            st.session_state.pop("composite_validation_result", None)
            st.session_state.pop("composite_validation_ticker", None)
            with st.spinner(f"Rebuilding quarterly composite verdicts for {ticker}..."):
                hist, _ = fetch.fetch_ticker_history_with_retry(ticker, "5y")
                info, _ = fetch.fetch_ticker_info_with_retry(ticker)
                sector = (info or {}).get("sector", "Unknown")
                result = backtest.compute_composite_quarterly_backtest(
                    ticker=ticker,
                    hist=hist,
                    model_settings=model_settings,
                    sector=sector,
                )
            if result is None:
                st.warning(
                    "Not enough quarterly fundamental data to run composite validation. "
                    "This requires at least 4 quarters of aligned financials from yfinance."
                )
            else:
                st.session_state["composite_validation_result"] = result
                st.session_state["composite_validation_ticker"] = ticker

        cached_ticker = st.session_state.get("composite_validation_ticker")
        result = st.session_state.get("composite_validation_result")

        if result and cached_ticker == ticker:
            if result["warnings"]:
                for w in result["warnings"]:
                    st.caption(f"Data note: {w}")

            bucket_df = result["bucket_table"]
            obs_df = result["observations"]
            n_quarters = result["n_quarters"]

            st.caption(
                f"Observations: {n_quarters} quarters | "
                f"Verdicts with data: {', '.join(bucket_df['Verdict'].tolist()) if not bucket_df.empty else 'none'}"
            )

            if not bucket_df.empty:
                st.subheader("Verdict Hit-Rate Table")
                display_df = bucket_df.copy()
                for col in ["Avg 1M", "Avg 3M", "Avg 12M"]:
                    display_df[col] = display_df[col].map(
                        lambda v: fmt.format_percent(v) if v is not None else "—"
                    )
                for col in ["Hit% 3M", "Hit% 12M"]:
                    display_df[col] = display_df[col].map(
                        lambda v: f"{v:.0%}" if v is not None else "—"
                    )
                st.dataframe(display_df, width="stretch", hide_index=True)

                st.caption(
                    "Avg returns are simple price returns from the effective filing date. "
                    "Hit% = share of observations with a positive return over that window. "
                    "Small N values mean individual outcomes dominate — interpret cautiously."
                )

            st.subheader("Quarterly Observations")
            obs_display = obs_df.copy()
            for col in ["Fwd 1M", "Fwd 3M", "Fwd 12M"]:
                obs_display[col] = obs_display[col].map(
                    lambda v: fmt.format_percent(v) if v is not None else "—"
                )
            st.dataframe(obs_display, width="stretch", hide_index=True)
            st.warning(
                "Reminder: forward returns above reflect true historical price outcomes, but the "
                "verdicts that generated them were scored with live data (current shares outstanding, "
                "potentially-restated financials, static sector benchmarks). Do not use these hit-rates "
                "as evidence that the composite model would have performed this way in real time."
            )


def render_backtest_view(db, model_settings, active_preset_name, active_assumption_fingerprint):
    st.subheader("Technical Signal Backtest")
    st.warning(
        "**Technical engine only.** This replay uses the Tech Score (price trend, SMA, RSI, MACD, momentum) "
        "to generate positions. It does **not** reconstruct historical fundamentals, valuation multiples, or "
        "news sentiment. For a quarterly walk-forward validation of the full composite model, expand "
        "**Composite Model Validation** below after running the backtest."
    )
    st.caption(
        "Replay uses stock-type-aware core sizing, trailing-stop behaviour, and deeper-breakdown "
        "confirmation before fully exiting."
    )

    backtest_default_ticker = st.session_state.get("backtest_last_ticker") or st.session_state.get("single_ticker", "")

    with st.form("backtest_form"):
        backtest_col_1, backtest_col_2, backtest_col_3, backtest_col_4 = st.columns([3, 1, 1, 1])
        with backtest_col_1:
            backtest_ticker = st.text_input(
                "Ticker",
                value=backtest_default_ticker,
                help="The app replays the active technical thresholds over this price history.",
            )
        with backtest_col_2:
            backtest_period = st.selectbox("History Window", ["1y", "3y", "5y", "10y"], index=2)
        with backtest_col_3:
            backtest_cost_bps = st.number_input(
                "Cost (bps)",
                value=float(model_settings.get("backtest_transaction_cost_bps", 10.0)),
                min_value=0.0,
                max_value=50.0,
                step=5.0,
                help="Round-trip transaction cost per position change. 10–30 bps is realistic for retail including spread.",
            )
        with backtest_col_4:
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
                backtest_settings = {**model_settings, "backtest_transaction_cost_bps": backtest_cost_bps}
                backtest_result = backtest.compute_technical_backtest(hist, backtest_settings, stock_profile=backtest_profile)

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
                    "cost_bps": backtest_cost_bps,
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
            f"Cost assumption: {backtest_config.get('cost_bps', 10.0):.0f} bps per-trade | "
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
                    "label": "Annual Turnover",
                    "value": fmt.format_value(backtest_metrics["Annual Turnover"], "{:.2f}x"),
                    "note": "How many times full exposure is traded per year — multiply by cost per trade to estimate total drag.",
                    "tone": ui.tone_from_metric_threshold(backtest_metrics["Annual Turnover"], good_max=4, bad_min=10),
                    "help": const.ANALYSIS_HELP_TEXT["Annual Turnover"],
                },
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
                {
                    "label": "Annual Turnover",
                    "value": fmt.format_percent(backtest_metrics["Annual Turnover"]),
                    "note": "Fraction of the position turned over per year. High turnover amplifies cost drag.",
                    "tone": ui.tone_from_metric_threshold(backtest_metrics["Annual Turnover"], good_max=2.0, bad_min=5.0),
                    "help": const.ANALYSIS_HELP_TEXT["Annual Turnover"],
                },
            ],
            columns=6,
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

        st.divider()
        _render_composite_validation(backtest_config, model_settings)

        # --- Factor IC Diagnostics ---
        ic_diag = backtest_result.get("ic_diagnostics", {})
        if ic_diag and ic_diag.get("ic_summary"):
            st.divider()
            st.subheader("Factor IC Diagnostics")
            st.caption(
                "IC (Information Coefficient) is the Spearman rank correlation between the Tech Score on a "
                "given day and the stock's actual forward return. Values above 0.05 suggest meaningful "
                "predictive signal; near zero means the score is not predictive at that horizon. "
                "Hit Rate is the share of non-zero signals where the direction was correct."
            )

            ic_cards = []
            for horizon_label in ("1M", "3M", "12M"):
                row = ic_diag["ic_summary"].get(horizon_label, {})
                ic_val = row.get("ic")
                hit_val = row.get("hit_rate")
                ic_cards.append(
                    {
                        "label": f"IC ({horizon_label})",
                        "value": fmt.format_value(ic_val, fmt="{:.3f}") if ic_val is not None else "N/A",
                        "note": f"Rank correlation of Tech Score vs {horizon_label} forward return. n={row.get('n', 0)}",
                        "tone": ui.tone_from_metric_threshold(ic_val or 0, good_min=0.05, bad_max=-0.02),
                        "help": const.ANALYSIS_HELP_TEXT.get("Factor IC"),
                    }
                )
                ic_cards.append(
                    {
                        "label": f"Hit Rate ({horizon_label})",
                        "value": fmt.format_percent(hit_val) if hit_val is not None else "N/A",
                        "note": f"% of non-zero signals where direction matched {horizon_label} return",
                        "tone": ui.tone_from_metric_threshold(hit_val or 0, good_min=0.55, bad_max=0.45),
                        "help": const.ANALYSIS_HELP_TEXT.get("Hit Rate IC"),
                    }
                )
            ui.render_analysis_signal_cards(ic_cards, columns=6)

            ic_by_window = ic_diag.get("ic_by_window", {})
            if ic_by_window:
                st.subheader("IC by Lookback Window")
                st.caption("How predictive power varies depending on how much history you include.")
                win_rows = []
                for win_label, horizon_dict in ic_by_window.items():
                    win_rows.append(
                        {
                            "Window": win_label,
                            "IC 1M": f"{horizon_dict.get('1M'):.3f}" if horizon_dict.get("1M") is not None else "—",
                            "IC 3M": f"{horizon_dict.get('3M'):.3f}" if horizon_dict.get("3M") is not None else "—",
                            "IC 12M": f"{horizon_dict.get('12M'):.3f}" if horizon_dict.get("12M") is not None else "—",
                        }
                    )
                st.dataframe(pd.DataFrame(win_rows).set_index("Window"), width="stretch")

            rolling_ic_df = ic_diag.get("rolling_ic", pd.DataFrame())
            if not rolling_ic_df.empty:
                st.subheader("Rolling IC (1-Year Window, Monthly Sampled)")
                st.caption(
                    "Each point shows the IC computed over the trailing 252 trading days. "
                    "Declining IC signals that the Tech Score's predictive power is eroding."
                )
                st.line_chart(rolling_ic_df, width="stretch")

            sub_ic = ic_diag.get("sub_signal_ic", {})
            if sub_ic:
                st.subheader("Sub-Signal IC Breakdown")
                st.caption("Which components of the Tech Score are driving (or dragging) predictive power.")
                sub_rows = []
                for sig_name, horizon_dict in sub_ic.items():
                    sub_rows.append(
                        {
                            "Sub-Signal": sig_name,
                            "IC 1M": f"{horizon_dict.get('1M'):.3f}" if horizon_dict.get("1M") is not None else "—",
                            "IC 3M": f"{horizon_dict.get('3M'):.3f}" if horizon_dict.get("3M") is not None else "—",
                            "IC 12M": f"{horizon_dict.get('12M'):.3f}" if horizon_dict.get("12M") is not None else "—",
                        }
                    )
                st.dataframe(pd.DataFrame(sub_rows).set_index("Sub-Signal"), width="stretch")
