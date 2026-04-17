# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

import json

import constants as const
import dcf as dcf_engine
import exports
import fetch
import cache
import settings
import utils_fmt as fmt
import utils_time as tutil
import utils_ui as ui
import analysis_prep as prep


def render_single_stock_view(db, bot, model_settings, active_assumption_fingerprint):
    if "single_ticker" not in st.session_state:
        _prefill = st.session_state.get("new_analyst_ticker", "")
        if _prefill:
            st.session_state["single_ticker"] = _prefill
    c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment="bottom")
    with c1:
        txt_input = st.text_input("Enter Ticker Symbol (e.g., AAPL, NVDA, F)", key="single_ticker")
    with c2:
        if st.button("Run Full Analysis", type="primary", width="stretch"):
            if txt_input:
                with st.spinner(f"Running multiple engines on {txt_input}..."):
                    res = bot.analyze(txt_input)
                    if not res:
                        st.error(bot.last_error or "Unable to fetch enough market data for this ticker right now.")
    with c3:
        if st.button("Refresh Data", width="stretch", help="Evict cached market data for this ticker and re-run the full analysis with fresh data."):
            if txt_input:
                ticker_clean = fmt.normalize_ticker(txt_input)
                cache.evict_ticker_from_cache(ticker_clean)
                with st.spinner(f"Refreshing {ticker_clean} with fresh data..."):
                    res = bot.analyze(ticker_clean)
                    if res:
                        st.success(f"Data refreshed for {ticker_clean}.")
                    else:
                        st.error(bot.last_error or "Unable to fetch fresh data for this ticker right now.")

    if txt_input:
        df = prep.prepare_analysis_dataframe(db.get_analysis(txt_input.upper()))
        if not df.empty:
            row = df.iloc[0]
            peer_comparison = fetch.safe_json_loads(row.get("Peer_Comparison"), default={})
            peer_bench = peer_comparison.get("benchmarks", {}) if isinstance(peer_comparison, dict) else {}
            if not peer_bench:
                peer_bench = const.get_sector_benchmarks(row["Sector"], model_settings)
            peer_rows = pd.DataFrame(peer_comparison.get("rows", [])) if isinstance(peer_comparison, dict) else pd.DataFrame()
            event_study_events = pd.DataFrame(fetch.safe_json_loads(row.get("Event_Study_Events"), default=[]))
            dcf_assumptions = fetch.safe_json_loads(row.get("DCF_Assumptions"), default={})
            company_download_bytes = exports.build_company_analysis_download_bytes(row)
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
                verdict_text = fmt.colorize_markdown_text(
                    f"VERDICT: {row['Verdict_Overall']}",
                    fmt.get_color(row["Verdict_Overall"]),
                )
                st.markdown(f"## {verdict_text}")
            with col_main_3:
                st.metric("Last Data Update", str(row.get("Last_Data_Update") or "Unknown"))
            st.caption(
                f"Sector: {row.get('Sector', 'Unknown')} | Industry: {row.get('Industry', 'Unknown')}"
            )

            ui.render_analysis_signal_cards(
                [
                    {
                        "label": "Stock Type",
                        "value": str(row.get("Stock_Type", "Legacy")),
                        "note": "The stock category the model thinks fits best right now.",
                        "tone": "neutral",
                        "help": const.ANALYSIS_HELP_TEXT["Stock Type"],
                    },
                    {
                        "label": "Cap Bucket",
                        "value": str(row.get("Cap_Bucket", "Unknown")),
                        "note": "A quick size label based on the company's market value.",
                        "tone": "neutral",
                        "help": const.ANALYSIS_HELP_TEXT["Cap Bucket"],
                    },
                    {
                        "label": "Type Confidence",
                        "value": fmt.format_value(row.get("Type_Confidence"), "{:,.0f}", "/100"),
                        "note": "Higher numbers mean the model sees a cleaner fit.",
                        "tone": ui.tone_from_metric_threshold(row.get("Type_Confidence"), good_min=70, bad_max=45),
                        "help": const.ANALYSIS_HELP_TEXT["Type Confidence"],
                    },
                    {
                        "label": "Market Cap",
                        "value": fmt.format_market_cap(row.get("Market_Cap")),
                        "note": "The market's current estimate of the company's total equity value.",
                        "tone": "neutral",
                        "help": const.ANALYSIS_HELP_TEXT["Market Cap"],
                    },
                ],
                columns=4,
            )
            if row.get("Style_Tags"):
                st.caption(f"This stock currently reads as: {row.get('Style_Tags')}")
            if row.get("Type_Strategy"):
                st.caption(f"The model's default playbook for this kind of stock: {row.get('Type_Strategy')}")

            with st.expander("Method Breakdown", expanded=False):
                st.info("This section shows how each part of the model is reading the stock right now, so you can see what is helping or hurting the final verdict.")
                st.caption("These results stay tied to the settings that were active when the analysis was run. If you changed something in Options, rerun the ticker to refresh this snapshot.")

                ui.render_analysis_signal_cards(
                    [
                        {
                            "label": "Technical",
                            "value": fmt.format_int(row["Score_Tech"]),
                            "note": "Price action, momentum, and trend signals.",
                            "tone": ui.tone_from_metric_threshold(row["Score_Tech"], good_min=1, bad_max=-1),
                            "help": const.ANALYSIS_HELP_TEXT["Technical"],
                        },
                        {
                            "label": "Fundamental",
                            "value": fmt.format_int(row["Score_Fund"]),
                            "note": "Business strength, growth, and balance-sheet signals.",
                            "tone": ui.tone_from_metric_threshold(row["Score_Fund"], good_min=1, bad_max=-1),
                            "help": const.ANALYSIS_HELP_TEXT["Fundamental"],
                        },
                        {
                            "label": "Valuation",
                            "value": fmt.format_int(row["Score_Val"]),
                            "note": "How cheap or expensive the stock looks versus its closest peer set.",
                            "tone": ui.tone_from_metric_threshold(row["Score_Val"], good_min=1, bad_max=-1),
                            "help": const.ANALYSIS_HELP_TEXT["Valuation"],
                        },
                        {
                            "label": "Sentiment Context",
                            "value": fmt.format_int(row["Sentiment_Headline_Count"]),
                            "note": "Context only: recent headlines and analyst metadata with no directional score applied.",
                            "tone": "neutral",
                            "help": const.ANALYSIS_HELP_TEXT["Sentiment"],
                        },
                        {
                            "label": "Updated",
                            "value": str(row["Last_Updated"]),
                            "note": "When this saved analysis was last refreshed.",
                            "tone": "neutral",
                            "help": const.ANALYSIS_HELP_TEXT["Updated"],
                        },
                    ],
                    columns=5,
                )

                ui.render_analysis_signal_cards(
                    [
                        {
                            "label": "Overall Score",
                            "value": fmt.format_value(row.get("Overall_Score"), "{:,.1f}"),
                            "note": "The model's combined read after blending all engines.",
                            "tone": ui.tone_from_metric_threshold(row.get("Overall_Score"), good_min=1, bad_max=-1),
                            "help": const.ANALYSIS_HELP_TEXT["Overall Score"],
                        },
                        {
                            "label": "Data Quality",
                            "value": str(row.get("Data_Quality", "Unknown")),
                            "note": "How complete and usable the source data was.",
                            "tone": ui.tone_from_quality_label(row.get("Data_Quality", "Unknown")),
                            "help": const.ANALYSIS_HELP_TEXT["Data Quality"],
                        },
                        {
                            "label": "Assumption Profile",
                            "value": str(row.get("Assumption_Profile", "Legacy")),
                            "note": "The preset or custom settings used for this run.",
                            "tone": "neutral",
                            "help": const.ANALYSIS_HELP_TEXT["Assumption Profile"],
                        },
                        {
                            "label": "Missing Metrics",
                            "value": fmt.format_int(row.get("Missing_Metric_Count")),
                            "note": "How many important data points were unavailable.",
                            "tone": ui.tone_from_metric_threshold(row.get("Missing_Metric_Count"), good_max=1, bad_min=5),
                            "help": const.ANALYSIS_HELP_TEXT["Missing Metrics"],
                        },
                        {
                            "label": "Consistency",
                            "value": fmt.format_value(row.get("Decision_Confidence"), "{:,.0f}", "/100"),
                            "note": "How consistently the model's signals lined up on this run.",
                            "tone": ui.tone_from_metric_threshold(row.get("Decision_Confidence"), good_min=70, bad_max=45),
                            "help": const.ANALYSIS_HELP_TEXT["Consistency"],
                        },
                        {
                            "label": "Regime",
                            "value": str(row.get("Market_Regime", "Unknown")),
                            "note": "The market backdrop the model sees in the chart.",
                            "tone": ui.tone_from_regime(row.get("Market_Regime", "Unknown")),
                            "help": const.ANALYSIS_HELP_TEXT["Regime"],
                        },
                    ],
                    columns=6,
                )
                st.caption(f"Fingerprint: {row.get('Assumption_Fingerprint', 'Legacy')}")
                ui.render_analysis_signal_cards(
                    [
                        {
                            "label": "Trend Strength",
                            "value": fmt.format_value(row.get("Trend_Strength"), "{:,.0f}"),
                            "note": "A quick read on how healthy the long-term trend looks.",
                            "tone": ui.tone_from_metric_threshold(row.get("Trend_Strength"), good_min=20, bad_max=-20),
                            "help": const.ANALYSIS_HELP_TEXT["Trend Strength"],
                        },
                        {
                            "label": "Quality Score",
                            "value": fmt.format_value(row.get("Quality_Score"), "{:,.1f}"),
                            "note": "A shorthand measure of business durability.",
                            "tone": ui.tone_from_metric_threshold(row.get("Quality_Score"), good_min=2, bad_max=0),
                            "help": const.ANALYSIS_HELP_TEXT["Quality Score"],
                        },
                        {
                            "label": "Dividend Safety",
                            "value": fmt.format_value(row.get("Dividend_Safety_Score"), "{:,.1f}"),
                            "note": "A rough check on how safe the dividend appears.",
                            "tone": ui.tone_from_metric_threshold(row.get("Dividend_Safety_Score"), good_min=1.5, bad_max=0),
                            "help": const.ANALYSIS_HELP_TEXT["Dividend Safety"],
                        },
                        {
                            "label": "Valuation Confidence",
                            "value": fmt.format_value(row.get("Valuation_Confidence"), "{:,.0f}", "/100"),
                            "note": "Higher means the valuation read is backed by more usable inputs.",
                            "tone": ui.tone_from_metric_threshold(row.get("Valuation_Confidence"), good_min=70, bad_max=40),
                            "help": const.ANALYSIS_HELP_TEXT["Valuation Confidence"],
                        },
                        {
                            "label": "Context Depth",
                            "value": fmt.format_value(row.get("Sentiment_Conviction"), "{:,.0f}", "/100"),
                            "note": "How much analyst and headline context was available for this company snapshot.",
                            "tone": "neutral",
                            "help": const.ANALYSIS_HELP_TEXT["Sentiment Conviction"],
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
                    ui.render_analysis_signal_cards(
                        [
                            {
                                "label": "Graham Fair Value",
                                "value": f"${row['Graham_Number']:,.2f}" if fetch.has_numeric_value(row["Graham_Number"]) and row["Graham_Number"] > 0 else "N/A",
                                "note": (
                                    f"Compared with today's price, the gap is ${row['Price'] - row['Graham_Number']:,.2f}."
                                    if fetch.has_numeric_value(row["Graham_Number"]) and row["Graham_Number"] > 0
                                    else "Only available when positive EPS and book value are both present."
                                ),
                                "tone": ui.tone_from_metric_threshold(graham_discount, good_min=0.0, bad_max=-0.15),
                                "help": const.ANALYSIS_HELP_TEXT["Graham Fair Value"],
                            },
                            {
                                "label": "Graham Discount",
                                "value": fmt.format_percent(graham_discount),
                                "note": "Positive means the stock is trading below this fair-value estimate.",
                                "tone": ui.tone_from_metric_threshold(graham_discount, good_min=0.0, bad_max=-0.15),
                                "help": const.ANALYSIS_HELP_TEXT["Graham Discount"],
                            },
                            {
                                "label": "Peer Group",
                                "value": str(row.get("Peer_Count") or 0),
                                "note": str(row.get("Peer_Group_Label") or "Closest comparable companies"),
                                "tone": "neutral",
                                "help": const.ANALYSIS_HELP_TEXT["Peer Group"],
                            },
                            {
                                "label": "Peer Summary",
                                "value": str(row.get("Peer_Group_Label") or "Fallback"),
                                "note": str(peer_comparison.get("benchmark_source") or "Closest peer group"),
                                "tone": "neutral",
                                "help": const.ANALYSIS_HELP_TEXT["Peer Summary"],
                            },
                            {
                                "label": "Relative Strength",
                                "value": fmt.format_percent(row.get("Relative_Strength_6M")),
                                "note": f"Six-month return relative to {const.DEFAULT_BENCHMARK_TICKER}.",
                                "tone": ui.tone_from_metric_threshold(row.get("Relative_Strength_6M"), good_min=0.03, bad_max=-0.03),
                                "help": const.ANALYSIS_HELP_TEXT["Relative Strength"],
                            },
                            {
                                "label": "Valuation Consistency",
                                "value": fmt.format_value(row.get("Valuation_Confidence"), "{:,.0f}", "/100"),
                                "note": "Higher means the relative valuation read had more usable comparison points.",
                                "tone": "neutral",
                                "help": const.ANALYSIS_HELP_TEXT["Valuation Confidence"],
                            },
                        ],
                        columns=1,
                    )
                with c_v2:
                    ui.render_analysis_signal_table(
                        [
                            {
                                "metric": "P/E Ratio",
                                "value": fmt.format_value(row["PE_Ratio"]),
                                "reference": fmt.format_value(peer_bench.get("PE")),
                                "status": "Cheap" if ui.tone_from_relative_multiple(row["PE_Ratio"], peer_bench.get("PE")) == "good" else "Rich" if ui.tone_from_relative_multiple(row["PE_Ratio"], peer_bench.get("PE")) == "bad" else "Fair",
                                "tone": ui.tone_from_relative_multiple(row["PE_Ratio"], peer_bench.get("PE")),
                                "help": const.ANALYSIS_HELP_TEXT["P/E Ratio"],
                            },
                            {
                                "metric": "Forward P/E",
                                "value": fmt.format_value(row["Forward_PE"]),
                                "reference": fmt.format_value(peer_bench.get("PE")),
                                "status": "Cheap" if ui.tone_from_relative_multiple(row["Forward_PE"], peer_bench.get("PE")) == "good" else "Rich" if ui.tone_from_relative_multiple(row["Forward_PE"], peer_bench.get("PE")) == "bad" else "Fair",
                                "tone": ui.tone_from_relative_multiple(row["Forward_PE"], peer_bench.get("PE")),
                                "help": const.ANALYSIS_HELP_TEXT["Forward P/E"],
                            },
                            {
                                "metric": "PEG Ratio",
                                "value": fmt.format_value(row["PEG_Ratio"]),
                                "reference": fmt.format_value(model_settings["valuation_peg_threshold"]),
                                "status": "Favorable" if ui.tone_from_metric_threshold(row["PEG_Ratio"], good_max=model_settings["valuation_peg_threshold"] * 0.9, bad_min=model_settings["valuation_peg_threshold"] * 1.35) == "good" else "Stretched" if ui.tone_from_metric_threshold(row["PEG_Ratio"], good_max=model_settings["valuation_peg_threshold"] * 0.9, bad_min=model_settings["valuation_peg_threshold"] * 1.35) == "bad" else "Mixed",
                                "tone": ui.tone_from_metric_threshold(row["PEG_Ratio"], good_max=model_settings["valuation_peg_threshold"] * 0.9, bad_min=model_settings["valuation_peg_threshold"] * 1.35),
                                "help": const.ANALYSIS_HELP_TEXT["PEG Ratio"],
                            },
                            {
                                "metric": "P/S Ratio",
                                "value": fmt.format_value(row["PS_Ratio"]),
                                "reference": fmt.format_value(peer_bench.get("PS")),
                                "status": "Cheap" if ui.tone_from_relative_multiple(row["PS_Ratio"], peer_bench.get("PS")) == "good" else "Rich" if ui.tone_from_relative_multiple(row["PS_Ratio"], peer_bench.get("PS")) == "bad" else "Fair",
                                "tone": ui.tone_from_relative_multiple(row["PS_Ratio"], peer_bench.get("PS")),
                                "help": const.ANALYSIS_HELP_TEXT["P/S Ratio"],
                            },
                            {
                                "metric": "EV/EBITDA",
                                "value": fmt.format_value(row["EV_EBITDA"]),
                                "reference": fmt.format_value(peer_bench.get("EV_EBITDA")),
                                "status": "Cheap" if ui.tone_from_relative_multiple(row["EV_EBITDA"], peer_bench.get("EV_EBITDA")) == "good" else "Rich" if ui.tone_from_relative_multiple(row["EV_EBITDA"], peer_bench.get("EV_EBITDA")) == "bad" else "Fair",
                                "tone": ui.tone_from_relative_multiple(row["EV_EBITDA"], peer_bench.get("EV_EBITDA")),
                                "help": const.ANALYSIS_HELP_TEXT["EV/EBITDA"],
                            },
                            {
                                "metric": "P/B Ratio",
                                "value": fmt.format_value(row["PB_Ratio"]),
                                "reference": fmt.format_value(peer_bench.get("PB")),
                                "status": "Cheap" if ui.tone_from_relative_multiple(row["PB_Ratio"], peer_bench.get("PB")) == "good" else "Rich" if ui.tone_from_relative_multiple(row["PB_Ratio"], peer_bench.get("PB")) == "bad" else "Fair",
                                "tone": ui.tone_from_relative_multiple(row["PB_Ratio"], peer_bench.get("PB")),
                                "help": const.ANALYSIS_HELP_TEXT["P/B Ratio"],
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
                            peer_display["Similarity"] = peer_display["Similarity"].map(lambda value: fmt.format_value(value, "{:,.2f}"))
                        for pct_column in ["Revenue Growth"]:
                            if pct_column in peer_display.columns:
                                peer_display[pct_column] = peer_display[pct_column].map(fmt.format_percent)
                        for numeric_column in ["P/E", "P/S", "P/B", "EV/EBITDA", "Beta"]:
                            if numeric_column in peer_display.columns:
                                peer_display[numeric_column] = peer_display[numeric_column].map(
                                    lambda value: fmt.format_value(value)
                                )
                        st.markdown("##### Peer Set")
                        st.dataframe(peer_display, width="stretch")

            with tab_fund:
                c_f1, c_f2 = st.columns([1, 2])
                with c_f1:
                    st.markdown(f"### Verdict: **{row['Verdict_Fundamental']}**")
                    st.caption("This view focuses on business strength, balance-sheet shape, and how the stock has reacted to recent company events.")
                    ui.render_analysis_signal_cards(
                        [
                            {
                                "label": "Quality Score",
                                "value": fmt.format_value(row.get("Quality_Score"), "{:,.1f}"),
                                "note": "A compact read on profitability, balance-sheet quality, and growth stability.",
                                "tone": ui.tone_from_metric_threshold(row.get("Quality_Score"), good_min=2, bad_max=0),
                                "help": const.ANALYSIS_HELP_TEXT["Quality Score"],
                            },
                            {
                                "label": "Dividend Safety",
                                "value": fmt.format_value(row.get("Dividend_Safety_Score"), "{:,.1f}"),
                                "note": "Useful mainly for income-oriented names.",
                                "tone": ui.tone_from_metric_threshold(row.get("Dividend_Safety_Score"), good_min=1.5, bad_max=0),
                                "help": const.ANALYSIS_HELP_TEXT["Dividend Safety"],
                            },
                            {
                                "label": "Event Study",
                                "value": fmt.format_int(row.get("Event_Study_Count")),
                                "note": "Recent company events with usable price reactions.",
                                "tone": "neutral",
                                "help": const.ANALYSIS_HELP_TEXT["Event Study"],
                            },
                            {
                                "label": "5D Abnormal Move",
                                "value": fmt.format_percent(row.get("Event_Study_Avg_Abnormal_5D")),
                                "note": f"Average five-day move versus {const.DEFAULT_BENCHMARK_TICKER} after recent events.",
                                "tone": "neutral",
                                "help": const.ANALYSIS_HELP_TEXT["Event Study 5D"],
                            },
                        ],
                        columns=1,
                    )
                with c_f2:
                    ui.render_analysis_signal_table(
                        [
                            {
                                "metric": "ROE",
                                "value": fmt.format_percent(row["ROE"]),
                                "reference": f">{model_settings['fund_roe_threshold'] * 100:.0f}%",
                                "status": "Strong" if ui.tone_from_metric_threshold(row["ROE"], good_min=model_settings["fund_roe_threshold"], bad_max=max(0.0, model_settings["fund_roe_threshold"] * 0.5)) == "good" else "Weak" if ui.tone_from_metric_threshold(row["ROE"], good_min=model_settings["fund_roe_threshold"], bad_max=max(0.0, model_settings["fund_roe_threshold"] * 0.5)) == "bad" else "Mixed",
                                "tone": ui.tone_from_metric_threshold(row["ROE"], good_min=model_settings["fund_roe_threshold"], bad_max=max(0.0, model_settings["fund_roe_threshold"] * 0.5)),
                                "help": const.ANALYSIS_HELP_TEXT["ROE"],
                            },
                            {
                                "metric": "Profit Margin",
                                "value": fmt.format_percent(row["Profit_Margins"]),
                                "reference": f">{model_settings['fund_profit_margin_threshold'] * 100:.0f}%",
                                "status": "Strong" if ui.tone_from_metric_threshold(row["Profit_Margins"], good_min=model_settings["fund_profit_margin_threshold"], bad_max=max(0.0, model_settings["fund_profit_margin_threshold"] * 0.5)) == "good" else "Weak" if ui.tone_from_metric_threshold(row["Profit_Margins"], good_min=model_settings["fund_profit_margin_threshold"], bad_max=max(0.0, model_settings["fund_profit_margin_threshold"] * 0.5)) == "bad" else "Mixed",
                                "tone": ui.tone_from_metric_threshold(row["Profit_Margins"], good_min=model_settings["fund_profit_margin_threshold"], bad_max=max(0.0, model_settings["fund_profit_margin_threshold"] * 0.5)),
                                "help": const.ANALYSIS_HELP_TEXT["Profit Margin"],
                            },
                            {
                                "metric": "Debt/Equity",
                                "value": fmt.format_value(row["Debt_to_Equity"], "{:,.0f}", "%"),
                                "reference": f"<{model_settings['fund_debt_good_threshold']:.0f}%",
                                "status": "Healthy" if ui.tone_from_metric_threshold(row["Debt_to_Equity"], good_max=model_settings["fund_debt_good_threshold"], bad_min=model_settings["fund_debt_bad_threshold"]) == "good" else "Stretched" if ui.tone_from_metric_threshold(row["Debt_to_Equity"], good_max=model_settings["fund_debt_good_threshold"], bad_min=model_settings["fund_debt_bad_threshold"]) == "bad" else "Watch",
                                "tone": ui.tone_from_metric_threshold(row["Debt_to_Equity"], good_max=model_settings["fund_debt_good_threshold"], bad_min=model_settings["fund_debt_bad_threshold"]),
                                "help": const.ANALYSIS_HELP_TEXT["Debt/Equity"],
                            },
                            {
                                "metric": "Revenue Growth",
                                "value": fmt.format_percent(row["Revenue_Growth"]),
                                "reference": f">{model_settings['fund_revenue_growth_threshold'] * 100:.0f}%",
                                "status": "Strong" if ui.tone_from_metric_threshold(row["Revenue_Growth"], good_min=model_settings["fund_revenue_growth_threshold"], bad_max=0.0) == "good" else "Weak" if ui.tone_from_metric_threshold(row["Revenue_Growth"], good_min=model_settings["fund_revenue_growth_threshold"], bad_max=0.0) == "bad" else "Mixed",
                                "tone": ui.tone_from_metric_threshold(row["Revenue_Growth"], good_min=model_settings["fund_revenue_growth_threshold"], bad_max=0.0),
                                "help": const.ANALYSIS_HELP_TEXT["Revenue Growth"],
                            },
                            {
                                "metric": "Current Ratio",
                                "value": fmt.format_value(row["Current_Ratio"]),
                                "reference": f">{model_settings['fund_current_ratio_good']:.1f}",
                                "status": "Healthy" if ui.tone_from_metric_threshold(row["Current_Ratio"], good_min=model_settings["fund_current_ratio_good"], bad_max=model_settings["fund_current_ratio_bad"]) == "good" else "Weak" if ui.tone_from_metric_threshold(row["Current_Ratio"], good_min=model_settings["fund_current_ratio_good"], bad_max=model_settings["fund_current_ratio_bad"]) == "bad" else "Mixed",
                                "tone": ui.tone_from_metric_threshold(row["Current_Ratio"], good_min=model_settings["fund_current_ratio_good"], bad_max=model_settings["fund_current_ratio_bad"]),
                                "help": const.ANALYSIS_HELP_TEXT["Current Ratio"],
                            },
                            {
                                "metric": "Dividend Yield",
                                "value": fmt.format_percent(row.get("Dividend_Yield")),
                                "reference": "Income support",
                                "status": "Supportive" if ui.tone_from_metric_threshold(row.get("Dividend_Yield"), good_min=0.02) == "good" else "Neutral",
                                "tone": ui.tone_from_metric_threshold(row.get("Dividend_Yield"), good_min=0.02),
                                "help": const.ANALYSIS_HELP_TEXT["Dividend Yield"],
                            },
                            {
                                "metric": "Payout Ratio",
                                "value": fmt.format_percent(row.get("Payout_Ratio")),
                                "reference": "<75% preferred",
                                "status": "Safe" if ui.tone_from_metric_threshold(row.get("Payout_Ratio"), good_max=0.75, bad_min=1.0) == "good" else "Stretched" if ui.tone_from_metric_threshold(row.get("Payout_Ratio"), good_max=0.75, bad_min=1.0) == "bad" else "Mixed",
                                "tone": ui.tone_from_metric_threshold(row.get("Payout_Ratio"), good_max=0.75, bad_min=1.0),
                                "help": const.ANALYSIS_HELP_TEXT["Payout Ratio"],
                            },
                            {
                                "metric": "Equity Beta",
                                "value": fmt.format_value(row.get("Equity_Beta")),
                                "reference": "<1.0 steadier",
                                "status": "Stable" if ui.tone_from_metric_threshold(row.get("Equity_Beta"), good_max=1.0, bad_min=1.5) == "good" else "Volatile" if ui.tone_from_metric_threshold(row.get("Equity_Beta"), good_max=1.0, bad_min=1.5) == "bad" else "Normal",
                                "tone": ui.tone_from_metric_threshold(row.get("Equity_Beta"), good_max=1.0, bad_min=1.5),
                                "help": const.ANALYSIS_HELP_TEXT["Equity Beta"],
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
                            event_display[column] = event_display[column].map(fmt.format_percent)
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
                    ui.render_analysis_signal_cards(
                        [
                            {
                                "label": "RSI (14)",
                                "value": fmt.format_value(row["RSI"], "{:,.1f}"),
                                "note": f"{int(model_settings['tech_rsi_oversold'])} oversold / {int(model_settings['tech_rsi_overbought'])} overbought",
                                "tone": ui.tone_from_balanced_band(
                                    row["RSI"],
                                    healthy_min=model_settings["tech_rsi_oversold"] + 5,
                                    healthy_max=model_settings["tech_rsi_overbought"] - 5,
                                    caution_low=model_settings["tech_rsi_oversold"],
                                    caution_high=model_settings["tech_rsi_overbought"],
                                ),
                                "help": const.ANALYSIS_HELP_TEXT["RSI (14)"],
                            },
                            {
                                "label": "Trend",
                                "value": str(row["SMA_Status"]),
                                "note": "A quick read on the moving-average trend.",
                                "tone": ui.tone_from_signal_text(row["SMA_Status"], positives={"BULLISH"}, negatives={"BEARISH"}),
                                "help": const.ANALYSIS_HELP_TEXT["Trend"],
                            },
                            {
                                "label": "MACD",
                                "value": fmt.format_value(row["MACD_Value"], "{:,.2f}"),
                                "note": f"Current signal: {row['MACD_Signal']}",
                                "tone": ui.tone_from_signal_text(row["MACD_Signal"], positives={"BULLISH CROSSOVER"}, negatives={"BEARISH CROSSOVER"}),
                                "help": const.ANALYSIS_HELP_TEXT["MACD"],
                            },
                            {
                                "label": "1M Momentum",
                                "value": fmt.format_percent(row["Momentum_1M"]),
                                "note": "The stock's short-term move over roughly one month.",
                                "tone": ui.tone_from_metric_threshold(row["Momentum_1M"], good_min=model_settings["tech_momentum_threshold"], bad_max=-model_settings["tech_momentum_threshold"]),
                                "help": const.ANALYSIS_HELP_TEXT["1M Momentum"],
                            },
                        ],
                        columns=4,
                    )
                    ui.render_analysis_signal_cards(
                        [
                            {
                                "label": "3M Relative Strength",
                                "value": fmt.format_percent(row.get("Relative_Strength_3M")),
                                "note": f"Three-month return versus {const.DEFAULT_BENCHMARK_TICKER}.",
                                "tone": ui.tone_from_metric_threshold(row.get("Relative_Strength_3M"), good_min=0.02, bad_max=-0.02),
                                "help": const.ANALYSIS_HELP_TEXT["Relative Strength"],
                            },
                            {
                                "label": "6M Relative Strength",
                                "value": fmt.format_percent(row.get("Relative_Strength_6M")),
                                "note": f"Six-month return versus {const.DEFAULT_BENCHMARK_TICKER}.",
                                "tone": ui.tone_from_metric_threshold(row.get("Relative_Strength_6M"), good_min=0.03, bad_max=-0.03),
                                "help": const.ANALYSIS_HELP_TEXT["Relative Strength"],
                            },
                            {
                                "label": "1Y Relative Strength",
                                "value": fmt.format_percent(row.get("Relative_Strength_1Y")),
                                "note": f"One-year return versus {const.DEFAULT_BENCHMARK_TICKER}.",
                                "tone": ui.tone_from_metric_threshold(row.get("Relative_Strength_1Y"), good_min=0.05, bad_max=-0.05),
                                "help": const.ANALYSIS_HELP_TEXT["Relative Strength"],
                            },
                        ],
                        columns=3,
                    )

                ui.render_analysis_signal_table(
                    [
                        {
                            "metric": "RSI",
                            "value": fmt.format_value(row["RSI"], "{:,.1f}"),
                            "reference": "Balanced range",
                            "status": "Healthy" if ui.tone_from_balanced_band(row["RSI"], healthy_min=model_settings["tech_rsi_oversold"] + 5, healthy_max=model_settings["tech_rsi_overbought"] - 5, caution_low=model_settings["tech_rsi_oversold"], caution_high=model_settings["tech_rsi_overbought"]) == "good" else "Extreme" if ui.tone_from_balanced_band(row["RSI"], healthy_min=model_settings["tech_rsi_oversold"] + 5, healthy_max=model_settings["tech_rsi_overbought"] - 5, caution_low=model_settings["tech_rsi_oversold"], caution_high=model_settings["tech_rsi_overbought"]) == "bad" else "Mixed",
                            "tone": ui.tone_from_balanced_band(row["RSI"], healthy_min=model_settings["tech_rsi_oversold"] + 5, healthy_max=model_settings["tech_rsi_overbought"] - 5, caution_low=model_settings["tech_rsi_oversold"], caution_high=model_settings["tech_rsi_overbought"]),
                            "help": const.ANALYSIS_HELP_TEXT["RSI"],
                        },
                        {
                            "metric": "200-Day Trend",
                            "value": str(row["SMA_Status"]),
                            "reference": "Trend direction",
                            "status": "Bullish" if ui.tone_from_signal_text(row["SMA_Status"], positives={"BULLISH"}, negatives={"BEARISH"}) == "good" else "Bearish" if ui.tone_from_signal_text(row["SMA_Status"], positives={"BULLISH"}, negatives={"BEARISH"}) == "bad" else "Neutral",
                            "tone": ui.tone_from_signal_text(row["SMA_Status"], positives={"BULLISH"}, negatives={"BEARISH"}),
                            "help": const.ANALYSIS_HELP_TEXT["200-Day Trend"],
                        },
                        {
                            "metric": "MACD Signal",
                            "value": str(row["MACD_Signal"]),
                            "reference": "Momentum crossover",
                            "status": "Bullish" if ui.tone_from_signal_text(row["MACD_Signal"], positives={"BULLISH CROSSOVER"}, negatives={"BEARISH CROSSOVER"}) == "good" else "Bearish" if ui.tone_from_signal_text(row["MACD_Signal"], positives={"BULLISH CROSSOVER"}, negatives={"BEARISH CROSSOVER"}) == "bad" else "Neutral",
                            "tone": ui.tone_from_signal_text(row["MACD_Signal"], positives={"BULLISH CROSSOVER"}, negatives={"BEARISH CROSSOVER"}),
                            "help": const.ANALYSIS_HELP_TEXT["MACD Signal"],
                        },
                        {
                            "metric": "1Y Momentum",
                            "value": fmt.format_percent(row["Momentum_1Y"]),
                            "reference": "Long-term move",
                            "status": "Strong" if ui.tone_from_metric_threshold(row["Momentum_1Y"], good_min=max(model_settings["tech_momentum_threshold"] * 3, 0.10), bad_max=-max(model_settings["tech_momentum_threshold"] * 3, 0.10)) == "good" else "Weak" if ui.tone_from_metric_threshold(row["Momentum_1Y"], good_min=max(model_settings["tech_momentum_threshold"] * 3, 0.10), bad_max=-max(model_settings["tech_momentum_threshold"] * 3, 0.10)) == "bad" else "Mixed",
                            "tone": ui.tone_from_metric_threshold(row["Momentum_1Y"], good_min=max(model_settings["tech_momentum_threshold"] * 3, 0.10), bad_max=-max(model_settings["tech_momentum_threshold"] * 3, 0.10)),
                            "help": const.ANALYSIS_HELP_TEXT["1Y Momentum"],
                        },
                        {
                            "metric": "Trend Strength",
                            "value": fmt.format_value(row["Trend_Strength"], "{:,.0f}"),
                            "reference": ">20 constructive",
                            "status": "Strong" if ui.tone_from_metric_threshold(row["Trend_Strength"], good_min=20, bad_max=-20) == "good" else "Weak" if ui.tone_from_metric_threshold(row["Trend_Strength"], good_min=20, bad_max=-20) == "bad" else "Mixed",
                            "tone": ui.tone_from_metric_threshold(row["Trend_Strength"], good_min=20, bad_max=-20),
                            "help": const.ANALYSIS_HELP_TEXT["Trend Strength"],
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
                    ui.render_analysis_signal_cards(
                        [
                            {
                                "label": "Headlines",
                                "value": fmt.format_int(row["Sentiment_Headline_Count"]),
                                "note": "Recent company-related headlines collected for context.",
                                "tone": "neutral",
                                "help": const.ANALYSIS_HELP_TEXT["Headlines"],
                            },
                            {
                                "label": "Analyst View",
                                "value": str(row["Recommendation_Key"]),
                                "note": "Raw recommendation label from the source feed, shown without interpretation.",
                                "tone": "neutral",
                                "help": const.ANALYSIS_HELP_TEXT["Analyst View"],
                            },
                            {
                                "label": "Target Mean",
                                "value": "N/A" if pd.isna(target_price) else f"${target_price:,.2f}",
                                "note": "Average analyst target price, shown as reference only.",
                                "tone": "neutral",
                                "help": const.ANALYSIS_HELP_TEXT["Target Mean"],
                            },
                            {
                                "label": "Context Depth",
                                "value": fmt.format_value(row.get("Sentiment_Conviction"), "{:,.0f}", "/100"),
                                "note": "How much analyst and headline context was available for this company.",
                                "tone": "neutral",
                                "help": const.ANALYSIS_HELP_TEXT["Sentiment Conviction"],
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
                dcf_snapshot_exists = exports.has_dcf_snapshot(row)
                dcf_download_bytes = exports.build_dcf_download_bytes(row) if dcf_snapshot_exists else b""
                dcf_last_updated = str(row.get("DCF_Last_Updated") or row.get("Last_Updated") or "").strip()
                dcf_intrinsic_value = fmt.safe_num(row.get("DCF_Intrinsic_Value"))
                dcf_upside = fmt.safe_num(row.get("DCF Upside"))
                dcf_history = pd.DataFrame(fetch.safe_json_loads(row.get("DCF_History"), default=[]))
                dcf_projection = pd.DataFrame(fetch.safe_json_loads(row.get("DCF_Projection"), default=[]))
                dcf_sensitivity = pd.DataFrame(fetch.safe_json_loads(row.get("DCF_Sensitivity"), default=[]))
                dcf_excerpts = fetch.safe_json_loads(row.get("DCF_Guidance_Excerpts"), default=[])
                live_dcf_settings = settings.get_dcf_settings()

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
                        updated_dcf_settings = settings.normalize_dcf_settings(
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
                        elif exports.has_dcf_snapshot(dcf_record):
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
                        st.caption(f"Snapshot updated: {dcf_last_updated} ({tutil.format_age(dcf_last_updated)})")
                    else:
                        st.caption("No DCF snapshot is saved yet for this ticker.")

                snapshot_assumptions = settings.normalize_dcf_settings(dcf_assumptions or live_dcf_settings)
                ui.render_analysis_signal_cards(
                    [
                        {
                            "label": "DCF Fair Value",
                            "value": f"${dcf_intrinsic_value:,.2f}" if fetch.has_numeric_value(dcf_intrinsic_value) else "N/A",
                            "note": "Per-share estimate from the most recent saved DCF snapshot.",
                            "tone": "neutral",
                            "help": const.ANALYSIS_HELP_TEXT["DCF Fair Value"],
                        },
                        {
                            "label": "DCF Upside",
                            "value": fmt.format_percent(dcf_upside),
                            "note": "Gap between the snapshot value and the current stock price.",
                            "tone": "neutral",
                            "help": const.ANALYSIS_HELP_TEXT["DCF Upside"],
                        },
                        {
                            "label": "DCF WACC",
                            "value": fmt.format_percent(row.get("DCF_WACC")),
                            "note": "Discount rate used in the saved snapshot.",
                            "tone": "neutral",
                            "help": const.ANALYSIS_HELP_TEXT["DCF WACC"],
                        },
                        {
                            "label": "Growth Source",
                            "value": str(row.get("DCF_Source", "Unavailable")),
                            "note": str(row.get("DCF_Guidance_Summary") or "Source of the starting growth assumption."),
                            "tone": "neutral",
                            "help": const.ANALYSIS_HELP_TEXT["DCF Source"],
                        },
                        {
                            "label": "Projection Years",
                            "value": str(int(snapshot_assumptions.get("projection_years", const.DCF_PROJECTION_YEARS))),
                            "note": "Explicit forecast length used for the saved snapshot.",
                            "tone": "neutral",
                            "help": const.ANALYSIS_HELP_TEXT["DCF Fair Value"],
                        },
                        {
                            "label": "Terminal Growth",
                            "value": fmt.format_percent(snapshot_assumptions.get("terminal_growth_rate")),
                            "note": "Long-run rate used after the explicit projection window.",
                            "tone": "neutral",
                            "help": const.ANALYSIS_HELP_TEXT["Terminal Growth"],
                        },
                    ],
                    columns=3,
                )

                # ── Bull / Base / Bear Scenario Builder ──────────────────────
                dcf_bull_fv     = fmt.safe_num(row.get("DCF_Bull_Fair_Value"))
                dcf_bear_fv     = fmt.safe_num(row.get("DCF_Bear_Fair_Value"))
                dcf_blended_fv  = fmt.safe_num(row.get("DCF_Blended_Fair_Value"))
                scenario_probs  = fetch.safe_json_loads(row.get("DCF_Scenario_Probs"), default={})
                bull_saved_asmp = fetch.safe_json_loads(row.get("DCF_Bull_Assumptions"), default={})
                bear_saved_asmp = fetch.safe_json_loads(row.get("DCF_Bear_Assumptions"), default={})
                base_saved_asmp = fetch.safe_json_loads(row.get("DCF_Assumptions"), default={})
                current_price   = fmt.safe_num(row.get("Price"))

                with st.expander("Bull / Base / Bear Scenarios", expanded=bool(fetch.has_numeric_value(dcf_bull_fv))):
                    st.caption("Build three DCF scenarios with independent assumptions and probability weights. The blended target is the probability-weighted fair value.")

                    with st.form(f"scenario_form_{row['Ticker']}"):
                        st.markdown("**Scenario Assumptions**")
                        scen_bull_col, scen_base_col, scen_bear_col = st.columns(3)

                        def _live_dcf_default(key, fallback):
                            return float(live_dcf_settings.get(key) or fallback)

                        with scen_bull_col:
                            st.markdown("🟢 **Bull**")
                            bull_growth = st.number_input("Starting Growth (%)", min_value=-20.0, max_value=50.0,
                                value=float((bull_saved_asmp.get("manual_growth_rate") or _live_dcf_default("manual_growth_rate", 0.08)) * 100),
                                step=0.5, key=f"bull_growth_{row['Ticker']}")
                            bull_tgr = st.number_input("Terminal Growth (%)", min_value=0.0, max_value=5.0,
                                value=float((bull_saved_asmp.get("terminal_growth_rate") or _live_dcf_default("terminal_growth_rate", const.DCF_TERMINAL_GROWTH_RATE)) * 100),
                                step=0.1, key=f"bull_tgr_{row['Ticker']}")
                            bull_haircut = st.number_input("Growth Haircut", min_value=0.5, max_value=1.2,
                                value=float(bull_saved_asmp.get("growth_haircut") or _live_dcf_default("growth_haircut", const.DCF_GROWTH_HAIRCUT)),
                                step=0.05, key=f"bull_haircut_{row['Ticker']}")
                            bull_mrp = st.number_input("Market Risk Prem (%)", min_value=3.0, max_value=9.0,
                                value=float((bull_saved_asmp.get("market_risk_premium") or _live_dcf_default("market_risk_premium", const.DCF_DEFAULT_MARKET_RISK_PREMIUM)) * 100),
                                step=0.1, key=f"bull_mrp_{row['Ticker']}")

                        with scen_base_col:
                            st.markdown("⚪ **Base**")
                            base_growth = st.number_input("Starting Growth (%)", min_value=-20.0, max_value=50.0,
                                value=float((base_saved_asmp.get("manual_growth_rate") or _live_dcf_default("manual_growth_rate", 0.05)) * 100),
                                step=0.5, key=f"base_growth_{row['Ticker']}")
                            base_tgr = st.number_input("Terminal Growth (%)", min_value=0.0, max_value=5.0,
                                value=float((base_saved_asmp.get("terminal_growth_rate") or _live_dcf_default("terminal_growth_rate", const.DCF_TERMINAL_GROWTH_RATE)) * 100),
                                step=0.1, key=f"base_tgr_{row['Ticker']}")
                            base_haircut = st.number_input("Growth Haircut", min_value=0.5, max_value=1.2,
                                value=float(base_saved_asmp.get("growth_haircut") or _live_dcf_default("growth_haircut", const.DCF_GROWTH_HAIRCUT)),
                                step=0.05, key=f"base_haircut_{row['Ticker']}")
                            base_mrp = st.number_input("Market Risk Prem (%)", min_value=3.0, max_value=9.0,
                                value=float((base_saved_asmp.get("market_risk_premium") or _live_dcf_default("market_risk_premium", const.DCF_DEFAULT_MARKET_RISK_PREMIUM)) * 100),
                                step=0.1, key=f"base_mrp_{row['Ticker']}")

                        with scen_bear_col:
                            st.markdown("🔴 **Bear**")
                            bear_growth = st.number_input("Starting Growth (%)", min_value=-20.0, max_value=50.0,
                                value=float((bear_saved_asmp.get("manual_growth_rate") or _live_dcf_default("manual_growth_rate", 0.02)) * 100),
                                step=0.5, key=f"bear_growth_{row['Ticker']}")
                            bear_tgr = st.number_input("Terminal Growth (%)", min_value=0.0, max_value=5.0,
                                value=float((bear_saved_asmp.get("terminal_growth_rate") or _live_dcf_default("terminal_growth_rate", const.DCF_TERMINAL_GROWTH_RATE)) * 100),
                                step=0.1, key=f"bear_tgr_{row['Ticker']}")
                            bear_haircut = st.number_input("Growth Haircut", min_value=0.5, max_value=1.2,
                                value=float(bear_saved_asmp.get("growth_haircut") or _live_dcf_default("growth_haircut", const.DCF_GROWTH_HAIRCUT)),
                                step=0.05, key=f"bear_haircut_{row['Ticker']}")
                            bear_mrp = st.number_input("Market Risk Prem (%)", min_value=3.0, max_value=9.0,
                                value=float((bear_saved_asmp.get("market_risk_premium") or _live_dcf_default("market_risk_premium", const.DCF_DEFAULT_MARKET_RISK_PREMIUM)) * 100),
                                step=0.1, key=f"bear_mrp_{row['Ticker']}")

                        st.markdown("**Probability Weights** (must sum to 100)")
                        prob_col_bull, prob_col_base, prob_col_bear = st.columns(3)
                        p_bull = prob_col_bull.number_input("Bull %", min_value=0, max_value=100,
                            value=int(scenario_probs.get("bull", 25)), step=5, key=f"p_bull_{row['Ticker']}")
                        p_base = prob_col_base.number_input("Base %", min_value=0, max_value=100,
                            value=int(scenario_probs.get("base", 55)), step=5, key=f"p_base_{row['Ticker']}")
                        p_bear = prob_col_bear.number_input("Bear %", min_value=0, max_value=100,
                            value=int(scenario_probs.get("bear", 20)), step=5, key=f"p_bear_{row['Ticker']}")

                        build_scenarios_submit = st.form_submit_button("Build Scenarios", type="primary", width="stretch")

                    if build_scenarios_submit:
                        prob_total = p_bull + p_base + p_bear
                        if prob_total != 100:
                            st.error(f"Probability weights must sum to 100 (currently {prob_total}).")
                        else:
                            def _make_scenario_settings(growth_pct, tgr_pct, haircut, mrp_pct):
                                return settings.normalize_dcf_settings({
                                    **live_dcf_settings,
                                    "manual_growth_rate": growth_pct / 100,
                                    "terminal_growth_rate": tgr_pct / 100,
                                    "growth_haircut": haircut,
                                    "market_risk_premium": mrp_pct / 100,
                                })

                            bull_s = _make_scenario_settings(bull_growth, bull_tgr, bull_haircut, bull_mrp)
                            base_s = _make_scenario_settings(base_growth, base_tgr, base_haircut, base_mrp)
                            bear_s = _make_scenario_settings(bear_growth, bear_tgr, bear_haircut, bear_mrp)

                            ticker_sym = str(row["Ticker"])
                            live_price = fmt.safe_num(row.get("Price")) or 0.0

                            with st.spinner(f"Building three-case DCF for {ticker_sym}..."):
                                live_info, _ = fetch.fetch_ticker_info_with_retry(ticker_sym)
                                three_case = dcf_engine.build_three_case_dcf(
                                    ticker_sym, live_price, live_info or {},
                                    bull_settings=bull_s,
                                    base_settings=base_s,
                                    bear_settings=bear_s,
                                )

                            bull_r = three_case.get("bull", {})
                            base_r = three_case.get("base", {})
                            bear_r = three_case.get("bear", {})

                            failed = [
                                c for c, r in [("bull", bull_r), ("base", base_r), ("bear", bear_r)]
                                if not r.get("available")
                            ]
                            if failed:
                                st.error(f"Could not build DCF for {', '.join(failed)} case(s). Check that SEC data is available for this ticker.")
                            else:
                                bfv  = float(bull_r["intrinsic_value_per_share"])
                                bsfv = float(base_r["intrinsic_value_per_share"])
                                brfv = float(bear_r["intrinsic_value_per_share"])
                                blended = bfv * (p_bull / 100) + bsfv * (p_base / 100) + brfv * (p_bear / 100)

                                def _upside(fv):
                                    if live_price and live_price > 0:
                                        return (fv - live_price) / live_price
                                    return None

                                scenario_record = {
                                    "Ticker": ticker_sym,
                                    "DCF_Bull_Fair_Value": bfv,
                                    "DCF_Bull_Upside": _upside(bfv),
                                    "DCF_Bull_Assumptions": json.dumps({
                                        "manual_growth_rate": bull_growth / 100,
                                        "terminal_growth_rate": bull_tgr / 100,
                                        "growth_haircut": bull_haircut,
                                        "market_risk_premium": bull_mrp / 100,
                                    }),
                                    "DCF_Bear_Fair_Value": brfv,
                                    "DCF_Bear_Upside": _upside(brfv),
                                    "DCF_Bear_Assumptions": json.dumps({
                                        "manual_growth_rate": bear_growth / 100,
                                        "terminal_growth_rate": bear_tgr / 100,
                                        "growth_haircut": bear_haircut,
                                        "market_risk_premium": bear_mrp / 100,
                                    }),
                                    "DCF_Scenario_Probs": json.dumps({"bull": p_bull, "base": p_base, "bear": p_bear}),
                                    "DCF_Blended_Fair_Value": blended,
                                }
                                db.save_analysis(scenario_record)
                                st.session_state["dcf_action_feedback"] = {
                                    "ticker": ticker_sym,
                                    "kind": "success",
                                    "message": f"Bull/Base/Bear scenarios saved for {ticker_sym}.",
                                }
                                st.rerun()

                    # Display saved scenario results
                    has_scenarios = fetch.has_numeric_value(dcf_bull_fv) and fetch.has_numeric_value(dcf_bear_fv)
                    if has_scenarios:
                        st.markdown("**Scenario Results**")
                        p_b  = scenario_probs.get("bull", 25)
                        p_bs = scenario_probs.get("base", 55)
                        p_br = scenario_probs.get("bear", 20)
                        bull_upside = fmt.safe_num(row.get("DCF_Bull_Upside"))
                        bear_upside = fmt.safe_num(row.get("DCF_Bear_Upside"))
                        base_fv = fmt.safe_num(row.get("DCF_Intrinsic_Value"))
                        base_upside = fmt.safe_num(row.get("DCF_Upside"))
                        blended_upside = (
                            (dcf_blended_fv - current_price) / current_price
                            if fetch.has_numeric_value(dcf_blended_fv) and fetch.has_numeric_value(current_price) and current_price > 0
                            else None
                        )
                        ui.render_analysis_signal_cards(
                            [
                                {
                                    "label": f"Bull Case ({p_b}%)",
                                    "value": f"${dcf_bull_fv:,.2f}" if fetch.has_numeric_value(dcf_bull_fv) else "N/A",
                                    "note": fmt.format_percent(bull_upside) + " upside" if fetch.has_numeric_value(bull_upside) else "",
                                    "tone": "good",
                                },
                                {
                                    "label": f"Base Case ({p_bs}%)",
                                    "value": f"${base_fv:,.2f}" if fetch.has_numeric_value(base_fv) else "N/A",
                                    "note": fmt.format_percent(base_upside) + " upside" if fetch.has_numeric_value(base_upside) else "",
                                    "tone": "neutral",
                                },
                                {
                                    "label": f"Bear Case ({p_br}%)",
                                    "value": f"${dcf_bear_fv:,.2f}" if fetch.has_numeric_value(dcf_bear_fv) else "N/A",
                                    "note": fmt.format_percent(bear_upside) + " upside" if fetch.has_numeric_value(bear_upside) else "",
                                    "tone": "bad",
                                },
                                {
                                    "label": "Blended Target",
                                    "value": f"${dcf_blended_fv:,.2f}" if fetch.has_numeric_value(dcf_blended_fv) else "N/A",
                                    "note": fmt.format_percent(blended_upside) + " upside" if fetch.has_numeric_value(blended_upside) else "Probability-weighted fair value",
                                    "tone": "neutral",
                                },
                            ],
                            columns=4,
                        )

                # ── End Scenario Builder ──────────────────────────────────────
                if dcf_snapshot_exists and not dcf_history.empty:
                    dcf_hist_col_1, dcf_hist_col_2 = st.columns(2)
                    with dcf_hist_col_1:
                        st.markdown("##### SEC History")
                        history_display = dcf_history.copy()
                        for money_column in ["Revenue", "OperatingCF", "CapEx", "FreeCashFlow"]:
                            if money_column in history_display.columns:
                                history_display[money_column] = history_display[money_column].map(fmt.format_market_cap)
                        st.dataframe(history_display, width="stretch")
                    with dcf_hist_col_2:
                        st.markdown("##### Projection")
                        projection_display = dcf_projection.copy()
                        if not projection_display.empty:
                            if "GrowthRate" in projection_display.columns:
                                projection_display["GrowthRate"] = projection_display["GrowthRate"].map(fmt.format_percent)
                            if "DiscountFactor" in projection_display.columns:
                                projection_display["DiscountFactor"] = projection_display["DiscountFactor"].map(
                                    lambda value: fmt.format_value(value, "{:,.3f}")
                                )
                            for money_column in ["FreeCashFlow", "PresentValue"]:
                                if money_column in projection_display.columns:
                                    projection_display[money_column] = projection_display[money_column].map(fmt.format_market_cap)
                            st.dataframe(projection_display, width="stretch")
                    if not dcf_sensitivity.empty:
                        st.markdown("##### Sensitivity Table")
                        sensitivity_display = dcf_sensitivity.copy()
                        if "WACC" in sensitivity_display.columns:
                            sensitivity_display["WACC"] = sensitivity_display["WACC"].map(fmt.format_percent)
                        for column in sensitivity_display.columns:
                            if column != "WACC":
                                sensitivity_display[column] = sensitivity_display[column].map(
                                    lambda value: f"${value:,.2f}" if fetch.has_numeric_value(value) else "N/A"
                                )
                        st.dataframe(sensitivity_display, width="stretch")
                else:
                    st.caption("Build a manual DCF snapshot to populate the valuation tables and sensitivity grid.")

                if dcf_excerpts:
                    st.markdown("##### Filing Takeaways")
                    for excerpt in dcf_excerpts[:5]:
                        st.write(f"- {excerpt}")

            st.divider()
            st.download_button(
                "Download Company Data",
                data=company_download_bytes,
                file_name=f"{row['Ticker']}_analysis_snapshot.json",
                mime="application/json",
                key=f"download_company_data_{row['Ticker']}",
                width="stretch",
            )
        else:
            st.info("Run the full analysis to save this ticker into the shared research library.")
