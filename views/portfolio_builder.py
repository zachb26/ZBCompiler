# -*- coding: utf-8 -*-
import streamlit as st

import constants as const
import utils_fmt as fmt
import utils_ui as ui
from ui.charts import render_frontier_chart


def render_portfolio_builder_view(portfolio_bot, active_preset_name, active_assumption_fingerprint):
    st.subheader("Portfolio Builder")
    st.caption("Use modern portfolio basics to recommend a risk-aware stock mix with Sharpe, Sortino, Treynor, the efficient frontier, and the Capital Allocation Line.")

    with st.form("portfolio_form"):
        p1, p2 = st.columns([3, 1])
        with p1:
            portfolio_tickers_raw = st.text_area(
                "Portfolio Tickers",
                value=const.DEFAULT_PORTFOLIO_TICKERS,
                help="Enter at least two tickers separated by commas or spaces.",
            )
        with p2:
            benchmark_ticker = st.text_input("Benchmark", value=const.DEFAULT_BENCHMARK_TICKER)
            lookback_period = st.selectbox("Lookback Period", ["1y", "3y", "5y"], index=1)
            risk_free_percent = st.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=15.0, value=4.0, step=0.25)

        p3, p4 = st.columns([3, 2])
        with p3:
            max_weight_percent = st.slider("Max Single-Stock Weight (%)", min_value=15, max_value=50, value=30, step=5)
        with p4:
            simulations = st.select_slider("Frontier Simulations", options=[1000, 2000, 3000, 4000, 5000], value=3000)

        portfolio_submit = st.form_submit_button("Build Portfolio Recommendation", type="primary", width="stretch")

    if portfolio_submit:
        parsed_tickers = fmt.parse_ticker_list(portfolio_tickers_raw)
        if len(parsed_tickers) < 2:
            st.error("Enter at least two valid ticker symbols for portfolio analysis.")
        elif len(parsed_tickers) * (max_weight_percent / 100) < 1:
            st.error("The max single-stock weight is too low for the number of tickers. Raise the cap or add more names.")
        else:
            with st.spinner("Building efficient frontier and CAL recommendation..."):
                verdicts = {}
                try:
                    all_analyses = portfolio_bot.db.get_all_analyses()
                    if not all_analyses.empty and "Verdict_Overall" in all_analyses.columns:
                        for _, row in all_analyses.iterrows():
                            t = str(row.get("Ticker", "")).strip().upper()
                            v = row.get("Verdict_Overall")
                            if t and v:
                                verdicts[t] = v
                except Exception:
                    pass

                portfolio_result = portfolio_bot.analyze_portfolio(
                    tickers=parsed_tickers,
                    benchmark_ticker=fmt.normalize_ticker(benchmark_ticker) or const.DEFAULT_BENCHMARK_TICKER,
                    period=lookback_period,
                    risk_free_rate=risk_free_percent / 100,
                    max_weight=max_weight_percent / 100,
                    simulations=simulations,
                    verdicts=verdicts or None,
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
                    "benchmark": fmt.normalize_ticker(benchmark_ticker) or const.DEFAULT_BENCHMARK_TICKER,
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
        filtered = result.get("filtered_strong_sell", [])
        if filtered:
            st.warning(
                f"**Excluded from optimizer (STRONG SELL verdict):** {', '.join(filtered)}. "
                "These tickers were removed before simulation. Add more tickers if you need them back."
            )
        if result.get("verdict_blend_applied"):
            st.info(
                "Expected returns are blended: 75% historical mean + 25% verdict-implied signal "
                "(STRONG BUY +6 pp / BUY +3 pp / HOLD neutral / SELL \u22123 pp over the risk-free rate). "
                "Volatility, beta, and drawdown metrics remain anchored to historical data."
            )
        st.caption(
            f"Benchmark: {config.get('benchmark', result['benchmark'])} | Lookback: {config.get('period', result['period'])} | "
            f"Risk-free rate: {config.get('risk_free_percent', 0):.2f}% | Max position: {config.get('max_weight_percent', 0)}%"
        )
        st.caption(f"Assumption profile: {active_preset_name} | Fingerprint: {active_assumption_fingerprint}")

        ui.render_analysis_signal_cards(
            [
                {
                    "label": "Expected Return",
                    "value": fmt.format_percent(tangent["Return"]),
                    "note": "The annualized return estimate for the max-Sharpe portfolio.",
                    "tone": ui.tone_from_metric_threshold(tangent["Return"], good_min=0.10, bad_max=0.03),
                    "help": const.ANALYSIS_HELP_TEXT["Expected Return"],
                },
                {
                    "label": "Volatility",
                    "value": fmt.format_percent(tangent["Volatility"]),
                    "note": "This is the expected bumpiness of returns over a full year.",
                    "tone": ui.tone_from_metric_threshold(tangent["Volatility"], good_max=0.22, bad_min=0.35),
                    "help": const.ANALYSIS_HELP_TEXT["Volatility"],
                },
                {
                    "label": "Sharpe",
                    "value": fmt.format_value(tangent["Sharpe"]),
                    "note": "Higher Sharpe usually means a better return-to-risk tradeoff.",
                    "tone": ui.tone_from_metric_threshold(tangent["Sharpe"], good_min=1.0, bad_max=0.3),
                    "help": const.ANALYSIS_HELP_TEXT["Sharpe"],
                },
                {
                    "label": "Sortino",
                    "value": fmt.format_value(tangent["Sortino"]),
                    "note": "This focuses on downside risk instead of all volatility.",
                    "tone": ui.tone_from_metric_threshold(tangent["Sortino"], good_min=1.2, bad_max=0.4),
                    "help": const.ANALYSIS_HELP_TEXT["Sortino"],
                },
                {
                    "label": "Treynor",
                    "value": fmt.format_value(tangent["Treynor"]),
                    "note": "This compares excess return with market sensitivity, not total volatility.",
                    "tone": ui.tone_from_metric_threshold(tangent["Treynor"], good_min=0.08, bad_max=0.0),
                    "help": const.ANALYSIS_HELP_TEXT["Treynor"],
                },
            ],
            columns=5,
        )

        ui.render_analysis_signal_cards(
            [
                {
                    "label": "Portfolio Beta",
                    "value": fmt.format_value(tangent["Beta"]),
                    "note": "Around 1 means the portfolio has moved roughly in line with the benchmark.",
                    "tone": ui.tone_from_balanced_band(tangent["Beta"], 0.8, 1.1, 0.6, 1.4),
                    "help": const.ANALYSIS_HELP_TEXT["Portfolio Beta"],
                },
                {
                    "label": "Downside Vol",
                    "value": fmt.format_percent(tangent["Downside Volatility"]),
                    "note": "This isolates the roughness coming from negative return swings.",
                    "tone": ui.tone_from_metric_threshold(tangent["Downside Volatility"], good_max=0.15, bad_min=0.28),
                    "help": const.ANALYSIS_HELP_TEXT["Downside Vol"],
                },
                {
                    "label": "Min-Vol Return",
                    "value": fmt.format_percent(min_vol["Return"]),
                    "note": "The return estimate for the lowest-volatility portfolio the simulation found.",
                    "tone": ui.tone_from_metric_threshold(min_vol["Return"], good_min=0.07, bad_max=0.02),
                    "help": const.ANALYSIS_HELP_TEXT["Min-Vol Return"],
                },
                {
                    "label": "Effective Names",
                    "value": fmt.format_value(result["effective_names"], "{:,.1f}"),
                    "note": "This shows how diversified the weights really are after concentration is considered.",
                    "tone": ui.tone_from_metric_threshold(result["effective_names"], good_min=5, bad_max=3),
                    "help": const.ANALYSIS_HELP_TEXT["Effective Names"],
                },
            ],
            columns=4,
        )

        st.subheader("Efficient Frontier and CAL")
        st.caption("The green diamond is the tangent portfolio with the highest Sharpe ratio. The red dashed line is the Capital Allocation Line.")
        render_frontier_chart(result["portfolio_cloud"], result["frontier"], result["cal"], tangent, min_vol)

        st.subheader("Recommended Allocation")
        ui.render_help_legend(
            [
                ("Sharpe", const.ANALYSIS_HELP_TEXT["Sharpe"]),
                ("Sortino", const.ANALYSIS_HELP_TEXT["Sortino"]),
                ("Treynor", const.ANALYSIS_HELP_TEXT["Treynor"]),
                ("Beta", const.ANALYSIS_HELP_TEXT["Portfolio Beta"]),
            ]
        )
        recommendations_display = recommendations[
            ["Ticker", "Name", "Sector", "Recommended Weight", "Role", "Sharpe Ratio", "Sortino Ratio", "Treynor Ratio", "Beta", "Rationale"]
        ].copy()
        recommendations_display["Recommended Weight"] = recommendations_display["Recommended Weight"].map(fmt.format_percent)
        recommendations_display["Sharpe Ratio"] = recommendations_display["Sharpe Ratio"].map(fmt.format_value)
        recommendations_display["Sortino Ratio"] = recommendations_display["Sortino Ratio"].map(fmt.format_value)
        recommendations_display["Treynor Ratio"] = recommendations_display["Treynor Ratio"].map(fmt.format_value)
        recommendations_display["Beta"] = recommendations_display["Beta"].map(fmt.format_value)
        st.dataframe(recommendations_display, width="stretch")

        exposure_col, metrics_col = st.columns([1, 2])
        with exposure_col:
            st.subheader("Sector Exposure")
            sector_display = sector_exposure.copy()
            sector_display["Recommended Weight"] = sector_display["Recommended Weight"].map(fmt.format_percent)
            st.dataframe(sector_display, width="stretch")

        with metrics_col:
            st.subheader("Per-Stock Metrics")
            ui.render_help_legend(
                [
                    ("Annual Return", const.ANALYSIS_HELP_TEXT["Expected Return"]),
                    ("Volatility", const.ANALYSIS_HELP_TEXT["Volatility"]),
                    ("Downside Volatility", const.ANALYSIS_HELP_TEXT["Downside Vol"]),
                    ("Beta", const.ANALYSIS_HELP_TEXT["Portfolio Beta"]),
                    ("Sharpe", const.ANALYSIS_HELP_TEXT["Sharpe"]),
                    ("Sortino", const.ANALYSIS_HELP_TEXT["Sortino"]),
                    ("Treynor", const.ANALYSIS_HELP_TEXT["Treynor"]),
                ]
            )
            asset_display = result["asset_metrics"].copy()
            for column in ["Annual Return", "Volatility", "Downside Volatility"]:
                asset_display[column] = asset_display[column].map(fmt.format_percent)
            for column in ["Beta", "Sharpe Ratio", "Sortino Ratio", "Treynor Ratio"]:
                asset_display[column] = asset_display[column].map(fmt.format_value)
            st.dataframe(asset_display, width="stretch")

        st.subheader("Portfolio Building Notes")
        for note in result["notes"]:
            st.write(f"- {note}")
