# -*- coding: utf-8 -*-
import streamlit as st

import constants as const
import exports
import utils_fmt as fmt
import utils_ui as ui
import skill_briefs as briefs
import analysis_prep as prep


def render_new_analyst_view(db, analyst):
    st.subheader("New Analyst")
    st.caption("This starter view focuses on the three beginner-friendly lenses: fundamentals, technicals, and sentiment context.")
    st.caption("Advanced valuation labs, peer comparisons, SEC filing automation, backtests, and model controls are reserved for Senior Analysts.")

    with st.form("new_analyst_form"):
        input_col, action_col = st.columns([3, 1])
        with input_col:
            new_analyst_ticker = st.text_input(
                "Ticker",
                value=st.session_state.get("new_analyst_ticker", ""),
                help="Enter one stock symbol to pull the latest saved or refreshed research snapshot.",
            )
        with action_col:
            st.write("")
            st.write("")
            run_new_analyst = st.form_submit_button("Analyze Stock", type="primary", width="stretch")

    if run_new_analyst:
        cleaned_ticker = fmt.normalize_ticker(new_analyst_ticker)
        st.session_state.new_analyst_ticker = cleaned_ticker
        if not cleaned_ticker:
            st.error("Enter a ticker to analyze.")
        else:
            with st.spinner(f"Running the starter workflow on {cleaned_ticker}..."):
                record = analyst.analyze(cleaned_ticker)
            if not record:
                st.error(analyst.last_error or "Unable to build a starter analysis for this ticker right now.")

    starter_ticker = fmt.normalize_ticker(st.session_state.get("new_analyst_ticker", ""))
    if not starter_ticker:
        st.info("Enter a ticker above to open the beginner-friendly analyst view.")
        return

    starter_df = prep.prepare_analysis_dataframe(db.get_analysis(starter_ticker))
    if starter_df.empty:
        st.warning("No saved analysis is available yet for that ticker. Run the analysis first.")
        return

    row = starter_df.iloc[0]
    company_download_bytes = exports.build_company_analysis_download_bytes(row)
    ui.render_analysis_signal_cards(
        [
            {
                "label": "Current Price",
                "value": fmt.format_value(row.get("Price"), "${:,.2f}"),
                "note": "The latest saved price in the shared research store.",
                "tone": "neutral",
                "help": const.ANALYSIS_HELP_TEXT["Market Cap"],
            },
            {
                "label": "Overall Verdict",
                "value": str(row.get("Verdict_Overall") or "Unknown"),
                "note": "A combined read built from the app's research layers.",
                "tone": ui.tone_from_signal_text(
                    row.get("Verdict_Overall"),
                    positives={"BUY", "STRONG BUY"},
                    negatives={"SELL", "STRONG SELL"},
                ),
                "help": const.ANALYSIS_HELP_TEXT["Overall Score"],
            },
            {
                "label": "Sector",
                "value": str(row.get("Sector") or "Unknown"),
                "note": "Useful for framing how the company fits into the broader market.",
                "tone": "neutral",
                "help": const.ANALYSIS_HELP_TEXT["Tracked Sectors"],
            },
            {
                "label": "Updated",
                "value": str(row.get("Freshness") or "Unknown"),
                "note": "How recently this saved analysis was refreshed.",
                "tone": "neutral",
                "help": const.ANALYSIS_HELP_TEXT["Freshness"],
            },
        ],
        columns=4,
    )
    st.download_button(
        "Download Company Data",
        data=company_download_bytes,
        file_name=f"{row.get('Ticker', starter_ticker)}_analysis_snapshot.json",
        mime="application/json",
        key=f"download_starter_company_data_{starter_ticker}",
        width="stretch",
    )

    with st.expander("Export for Claude Code Skills"):
        st.caption(
            "Download pre-formatted input briefs for the installed financial skills. "
            "In Claude Code, run the skill (e.g. `/equity-research:earnings`) and attach the brief when prompted."
        )
        _skill_row = row.to_dict() if hasattr(row, "to_dict") else row
        _brief_cols = st.columns(2)
        with _brief_cols[0]:
            st.download_button(
                "Earnings Brief (.md)",
                data=briefs.build_earnings_skill_brief(_skill_row).encode("utf-8"),
                file_name=f"{starter_ticker}_earnings_brief.md",
                mime="text/markdown",
                key=f"skill_earnings_{starter_ticker}",
                help="Input for /equity-research:earnings",
                width="stretch",
            )
            st.download_button(
                "DCF Brief (.md)",
                data=briefs.build_dcf_skill_brief(_skill_row).encode("utf-8"),
                file_name=f"{starter_ticker}_dcf_brief.md",
                mime="text/markdown",
                key=f"skill_dcf_{starter_ticker}",
                help="Input for /financial-analysis:dcf-model",
                disabled=not row.get("DCF_Intrinsic_Value"),
                width="stretch",
            )
        with _brief_cols[1]:
            st.download_button(
                "Comps Brief (.md)",
                data=briefs.build_comps_skill_brief(_skill_row).encode("utf-8"),
                file_name=f"{starter_ticker}_comps_brief.md",
                mime="text/markdown",
                key=f"skill_comps_{starter_ticker}",
                help="Input for /financial-analysis:comps-analysis",
                width="stretch",
            )
            st.download_button(
                "IC Memo Brief (.md)",
                data=briefs.build_ic_memo_skill_brief(_skill_row).encode("utf-8"),
                file_name=f"{starter_ticker}_ic_memo_brief.md",
                mime="text/markdown",
                key=f"skill_ic_{starter_ticker}",
                help="Input for /private-equity:ic-memo or /investment-banking:one-pager",
                width="stretch",
            )

    fund_tab, tech_tab, sent_tab = st.tabs(["Fundamental Analysis", "Technical Analysis", "Sentiment Analysis"])

    with fund_tab:
        st.caption("Fundamentals describe business quality: profitability, growth, leverage, and liquidity.")
        ui.render_analysis_signal_cards(
            [
                {
                    "label": "Fundamental Verdict",
                    "value": str(row.get("Verdict_Fundamental") or "Unknown"),
                    "note": "A quick summary of the business-quality read.",
                    "tone": ui.tone_from_signal_text(row.get("Verdict_Fundamental"), positives={"STRONG"}, negatives={"WEAK"}),
                    "help": const.ANALYSIS_HELP_TEXT["Fundamental"],
                },
                {
                    "label": "Quality Score",
                    "value": fmt.format_value(row.get("Quality_Score"), "{:,.1f}"),
                    "note": "Higher scores usually mean the business looks healthier and more durable.",
                    "tone": ui.tone_from_metric_threshold(row.get("Quality_Score"), good_min=2, bad_max=0),
                    "help": const.ANALYSIS_HELP_TEXT["Quality Score"],
                },
                {
                    "label": "ROE",
                    "value": fmt.format_percent(row.get("ROE")),
                    "note": "Return on equity shows how efficiently profit is generated from shareholder capital.",
                    "tone": ui.tone_from_metric_threshold(row.get("ROE"), good_min=0.15, bad_max=0.05),
                    "help": const.ANALYSIS_HELP_TEXT["ROE"],
                },
                {
                    "label": "Profit Margin",
                    "value": fmt.format_percent(row.get("Profit_Margins")),
                    "note": "Positive and improving margins can point to a sturdier business model.",
                    "tone": ui.tone_from_metric_threshold(row.get("Profit_Margins"), good_min=0.12, bad_max=0.0),
                    "help": const.ANALYSIS_HELP_TEXT["Profit Margin"],
                },
            ],
            columns=4,
        )
        ui.render_analysis_signal_table(
            [
                {
                    "metric": "Revenue Growth",
                    "value": fmt.format_percent(row.get("Revenue_Growth")),
                    "reference": "Higher is usually better",
                    "status": "Strong" if ui.tone_from_metric_threshold(row.get("Revenue_Growth"), good_min=0.08, bad_max=-0.02) == "good" else "Weak" if ui.tone_from_metric_threshold(row.get("Revenue_Growth"), good_min=0.08, bad_max=-0.02) == "bad" else "Mixed",
                    "tone": ui.tone_from_metric_threshold(row.get("Revenue_Growth"), good_min=0.08, bad_max=-0.02),
                    "help": const.ANALYSIS_HELP_TEXT["Revenue Growth"],
                },
                {
                    "metric": "Current Ratio",
                    "value": fmt.format_value(row.get("Current_Ratio")),
                    "reference": "Above 1 is usually healthier",
                    "status": "Healthy" if ui.tone_from_metric_threshold(row.get("Current_Ratio"), good_min=1.2, bad_max=1.0) == "good" else "Tight" if ui.tone_from_metric_threshold(row.get("Current_Ratio"), good_min=1.2, bad_max=1.0) == "bad" else "Mixed",
                    "tone": ui.tone_from_metric_threshold(row.get("Current_Ratio"), good_min=1.2, bad_max=1.0),
                    "help": const.ANALYSIS_HELP_TEXT["Current Ratio"],
                },
                {
                    "metric": "Debt / Equity",
                    "value": fmt.format_value(row.get("Debt_to_Equity")),
                    "reference": "Lower is usually safer",
                    "status": "Low" if ui.tone_from_metric_threshold(row.get("Debt_to_Equity"), good_max=100, bad_min=200) == "good" else "High" if ui.tone_from_metric_threshold(row.get("Debt_to_Equity"), good_max=100, bad_min=200) == "bad" else "Moderate",
                    "tone": ui.tone_from_metric_threshold(row.get("Debt_to_Equity"), good_max=100, bad_min=200),
                    "help": const.ANALYSIS_HELP_TEXT["Debt/Equity"],
                },
                {
                    "metric": "Dividend Yield",
                    "value": fmt.format_percent(row.get("Dividend_Yield")),
                    "reference": "Income context",
                    "status": "Income",
                    "tone": "neutral",
                    "help": const.ANALYSIS_HELP_TEXT["Dividend Yield"],
                },
            ],
            reference_label="Guide",
        )

    with tech_tab:
        st.caption("Technicals focus on the chart: momentum, trend direction, and whether price looks stretched or healthy.")
        ui.render_analysis_signal_cards(
            [
                {
                    "label": "Technical Verdict",
                    "value": str(row.get("Verdict_Technical") or "Unknown"),
                    "note": "A shorthand read on the chart backdrop.",
                    "tone": ui.tone_from_signal_text(row.get("Verdict_Technical"), positives={"BUY", "STRONG BUY"}, negatives={"SELL", "STRONG SELL"}),
                    "help": const.ANALYSIS_HELP_TEXT["Technical"],
                },
                {
                    "label": "RSI",
                    "value": fmt.format_value(row.get("RSI"), "{:,.1f}"),
                    "note": "Very high or very low values can signal stretched price action.",
                    "tone": ui.tone_from_balanced_band(row.get("RSI"), 35, 65, 30, 70),
                    "help": const.ANALYSIS_HELP_TEXT["RSI"],
                },
                {
                    "label": "Trend",
                    "value": str(row.get("SMA_Status") or "Unknown"),
                    "note": "This compares price and moving averages to judge the broader trend.",
                    "tone": ui.tone_from_signal_text(row.get("SMA_Status"), positives={"BULLISH"}, negatives={"BEARISH"}),
                    "help": const.ANALYSIS_HELP_TEXT["200-Day Trend"],
                },
                {
                    "label": "MACD Signal",
                    "value": str(row.get("MACD_Signal") or "Unknown"),
                    "note": "This helps show whether momentum is improving or fading.",
                    "tone": ui.tone_from_signal_text(row.get("MACD_Signal"), positives={"BULLISH CROSSOVER"}, negatives={"BEARISH CROSSOVER"}),
                    "help": const.ANALYSIS_HELP_TEXT["MACD Signal"],
                },
            ],
            columns=4,
        )
        ui.render_analysis_signal_table(
            [
                {
                    "metric": "1M Momentum",
                    "value": fmt.format_percent(row.get("Momentum_1M")),
                    "reference": "Recent move",
                    "status": "Strong" if ui.tone_from_metric_threshold(row.get("Momentum_1M"), good_min=0.04, bad_max=-0.04) == "good" else "Weak" if ui.tone_from_metric_threshold(row.get("Momentum_1M"), good_min=0.04, bad_max=-0.04) == "bad" else "Mixed",
                    "tone": ui.tone_from_metric_threshold(row.get("Momentum_1M"), good_min=0.04, bad_max=-0.04),
                    "help": const.ANALYSIS_HELP_TEXT["1M Momentum"],
                },
                {
                    "metric": "1Y Momentum",
                    "value": fmt.format_percent(row.get("Momentum_1Y")),
                    "reference": "Longer trend",
                    "status": "Strong" if ui.tone_from_metric_threshold(row.get("Momentum_1Y"), good_min=0.10, bad_max=-0.10) == "good" else "Weak" if ui.tone_from_metric_threshold(row.get("Momentum_1Y"), good_min=0.10, bad_max=-0.10) == "bad" else "Mixed",
                    "tone": ui.tone_from_metric_threshold(row.get("Momentum_1Y"), good_min=0.10, bad_max=-0.10),
                    "help": const.ANALYSIS_HELP_TEXT["1Y Momentum"],
                },
                {
                    "metric": "Trend Strength",
                    "value": fmt.format_value(row.get("Trend_Strength"), "{:,.0f}"),
                    "reference": "Above 20 is constructive",
                    "status": "Strong" if ui.tone_from_metric_threshold(row.get("Trend_Strength"), good_min=20, bad_max=-20) == "good" else "Weak" if ui.tone_from_metric_threshold(row.get("Trend_Strength"), good_min=20, bad_max=-20) == "bad" else "Mixed",
                    "tone": ui.tone_from_metric_threshold(row.get("Trend_Strength"), good_min=20, bad_max=-20),
                    "help": const.ANALYSIS_HELP_TEXT["Trend Strength"],
                },
                {
                    "metric": "6M Relative Strength",
                    "value": fmt.format_percent(row.get("Relative_Strength_6M")),
                    "reference": f"Versus {const.DEFAULT_BENCHMARK_TICKER}",
                    "status": "Leader" if ui.tone_from_metric_threshold(row.get("Relative_Strength_6M"), good_min=0.03, bad_max=-0.03) == "good" else "Laggard" if ui.tone_from_metric_threshold(row.get("Relative_Strength_6M"), good_min=0.03, bad_max=-0.03) == "bad" else "Mixed",
                    "tone": ui.tone_from_metric_threshold(row.get("Relative_Strength_6M"), good_min=0.03, bad_max=-0.03),
                    "help": const.ANALYSIS_HELP_TEXT["Relative Strength"],
                },
            ],
            reference_label="Guide",
        )

    with sent_tab:
        st.caption("Sentiment here is context only: headlines, analyst labels, and market chatter that can help frame the story.")
        ui.render_analysis_signal_cards(
            [
                {
                    "label": "Headline Count",
                    "value": fmt.format_int(row.get("Sentiment_Headline_Count")),
                    "note": "How many recent company-related headlines were available.",
                    "tone": "neutral",
                    "help": const.ANALYSIS_HELP_TEXT["Headlines"],
                },
                {
                    "label": "Analyst View",
                    "value": str(row.get("Recommendation_Key") or "N/A"),
                    "note": "A raw analyst label from the feed, shown without interpretation.",
                    "tone": "neutral",
                    "help": const.ANALYSIS_HELP_TEXT["Analyst View"],
                },
                {
                    "label": "Target Mean",
                    "value": fmt.format_value(row.get("Target_Mean_Price"), "${:,.2f}"),
                    "note": "The average analyst target price, shown as context rather than a prediction.",
                    "tone": "neutral",
                    "help": const.ANALYSIS_HELP_TEXT["Target Mean"],
                },
                {
                    "label": "Context Depth",
                    "value": fmt.format_value(row.get("Sentiment_Conviction"), "{:,.0f}", "/100"),
                    "note": "Higher means the app had more analyst and headline context to work with.",
                    "tone": "neutral",
                    "help": const.ANALYSIS_HELP_TEXT["Sentiment Conviction"],
                },
            ],
            columns=4,
        )
        context_lines = [
            line.strip()
            for line in str(row.get("Sentiment_Summary") or "").split("|")
            if line.strip()
        ]
        if context_lines:
            st.markdown("##### Recent Context")
            for line in context_lines:
                st.write(f"- {line}")
        else:
            st.info("No recent company-context lines were available in the current saved snapshot.")
