# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

import constants as const
import settings
import utils_fmt as fmt
import utils_ui as ui
import analysis_prep as prep


def render_comparison_view(db, bot, model_settings, active_preset_name, active_assumption_fingerprint):
    st.subheader("Compare Stocks")
    st.caption("Rank a watchlist with the same technical, fundamental, valuation, and context workflow before deciding what deserves portfolio weight.")

    with st.form("compare_form"):
        compare_col_1, compare_col_2 = st.columns([3, 1])
        with compare_col_1:
            compare_tickers_raw = st.text_area(
                "Tickers to Compare",
                value=const.DEFAULT_PORTFOLIO_TICKERS,
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
        compare_tickers = fmt.parse_ticker_list(compare_tickers_raw)
        if len(compare_tickers) < 2:
            st.error("Enter at least two valid ticker symbols to compare.")
        else:
            with st.spinner("Pulling stock research and ranking the shortlist..."):
                comparison_df, failed_tickers, failure_reasons, refreshed_tickers, cached_tickers = prep.collect_analysis_rows(
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
        comparison_df = prep.prepare_analysis_dataframe(st.session_state.compare_result.copy())
        comparison_df = comparison_df.sort_values(
            ["Composite Score", "Target Upside", "Ticker"],
            ascending=[False, False, True],
            na_position="last",
        ).reset_index(drop=True)
        meta = st.session_state.get("compare_meta", {})

        top_pick = comparison_df.iloc[0]
        average_upside = comparison_df["Target Upside"].dropna().mean()
        average_dcf_upside = comparison_df["DCF Upside"].dropna().mean() if "DCF Upside" in comparison_df.columns else None
        ui.render_analysis_signal_cards(
            [
                {
                    "label": "Highest Conviction",
                    "value": str(top_pick["Ticker"]),
                    "note": f"Current top-ranked name with a verdict of {top_pick['Verdict_Overall']}.",
                    "tone": ui.tone_from_metric_threshold(top_pick.get("Decision_Confidence"), good_min=70, bad_max=45),
                    "help": const.ANALYSIS_HELP_TEXT["Highest Conviction"],
                },
                {
                    "label": "Average Composite Score",
                    "value": fmt.format_value(comparison_df["Composite Score"].mean(), "{:,.1f}"),
                    "note": "A higher average means the watchlist looks stronger overall.",
                    "tone": ui.tone_from_metric_threshold(comparison_df["Composite Score"].mean(), good_min=1, bad_max=-1),
                    "help": const.ANALYSIS_HELP_TEXT["Average Composite Score"],
                },
                {
                    "label": "Average Target Upside",
                    "value": fmt.format_percent(average_upside),
                    "note": "The average analyst upside across the names in this shortlist.",
                    "tone": ui.tone_from_metric_threshold(average_upside, good_min=0.10, bad_max=-0.05),
                    "help": const.ANALYSIS_HELP_TEXT["Average Target Upside"],
                },
                {
                    "label": "Average DCF Upside",
                    "value": fmt.format_percent(average_dcf_upside),
                    "note": "The average discount or premium implied by the cash-flow model across this shortlist.",
                    "tone": ui.tone_from_metric_threshold(average_dcf_upside, good_min=0.10, bad_max=-0.05),
                    "help": const.ANALYSIS_HELP_TEXT["Average DCF Upside"],
                },
                {
                    "label": "Sectors Covered",
                    "value": str(comparison_df["Sector"].nunique()),
                    "note": "More sectors usually means the list is less concentrated in one theme.",
                    "tone": "neutral",
                    "help": const.ANALYSIS_HELP_TEXT["Sectors Covered"],
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
        if meta.get("cached") and settings.calculate_assumption_drift(model_settings) > 0:
            st.caption("Cached rows keep their previous assumption set until you refresh them with live data.")
        if comparison_df["Assumption_Fingerprint"].nunique() > 1:
            st.caption("This comparison includes rows generated under different assumption fingerprints. Refresh live data for a cleaner apples-to-apples ranking.")

        st.subheader("Shortlist Ranking")
        ui.render_help_legend(
            [
                ("Composite Score", const.ANALYSIS_HELP_TEXT["Composite Score"]),
                ("Consistency", const.ANALYSIS_HELP_TEXT["Consistency"]),
                ("Trend Strength", const.ANALYSIS_HELP_TEXT["Trend Strength"]),
                ("Quality Score", const.ANALYSIS_HELP_TEXT["Quality Score"]),
                ("Target Upside", const.ANALYSIS_HELP_TEXT["Target Mean"]),
                ("Graham Discount", const.ANALYSIS_HELP_TEXT["Graham Discount"]),
                ("DCF Upside", const.ANALYSIS_HELP_TEXT["DCF Upside"]),
                ("Freshness", const.ANALYSIS_HELP_TEXT["Freshness"]),
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
            lambda value: fmt.format_value(value, "{:,.0f}", "/100")
        )
        comparison_display = comparison_display.rename(columns={"Decision_Confidence": "Consistency"})
        comparison_display["Trend_Strength"] = comparison_display["Trend_Strength"].map(
            lambda value: fmt.format_value(value, "{:,.0f}")
        )
        comparison_display["Quality_Score"] = comparison_display["Quality_Score"].map(
            lambda value: fmt.format_value(value, "{:,.1f}")
        )
        comparison_display["Target Upside"] = comparison_display["Target Upside"].map(fmt.format_percent)
        comparison_display["Graham Discount"] = comparison_display["Graham Discount"].map(fmt.format_percent)
        comparison_display["DCF Upside"] = comparison_display["DCF Upside"].map(fmt.format_percent)
        st.dataframe(comparison_display, width="stretch")

        engine_col, rationale_col = st.columns([2, 1])
        with engine_col:
            st.subheader("Engine Scorecard")
            ui.render_help_legend(
                [
                    ("Technical", const.ANALYSIS_HELP_TEXT["Technical"]),
                    ("Fundamental", const.ANALYSIS_HELP_TEXT["Fundamental"]),
                    ("Valuation", const.ANALYSIS_HELP_TEXT["Valuation"]),
                    ("Sentiment", const.ANALYSIS_HELP_TEXT["Sentiment"]),
                    ("Composite Score", const.ANALYSIS_HELP_TEXT["Composite Score"]),
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
