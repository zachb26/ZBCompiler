# -*- coding: utf-8 -*-
import streamlit as st

import constants as const
import utils_fmt as fmt
import utils_ui as ui
import analysis_prep as prep


def render_sensitivity_view(bot, model_settings, active_preset_name, active_assumption_fingerprint):
    st.subheader("Sensitivity Check")
    st.caption("Run the same live market snapshot across guarded assumption scenarios to see whether the model is directionally stable or fragile.")
    st.caption("This does not overwrite the research library. It reuses one fresh market pull and reruns the scoring logic in memory.")

    sensitivity_default_ticker = st.session_state.get("sensitivity_last_ticker") or st.session_state.get("single_ticker", "")

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
            run_sensitivity = st.form_submit_button("Run Sensitivity Check", type="primary", width="stretch")

    if run_sensitivity:
        cleaned_ticker = fmt.normalize_ticker(sensitivity_ticker)
        if not cleaned_ticker:
            st.error("Enter a ticker to run a sensitivity check.")
        else:
            with st.spinner(f"Testing {cleaned_ticker} across assumption scenarios..."):
                sensitivity_df, sensitivity_summary = prep.run_sensitivity_analysis(bot, cleaned_ticker, model_settings)

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

        ui.render_analysis_signal_cards(
            [
                {
                    "label": "Robustness",
                    "value": sensitivity_summary.get("robustness_label", "N/A"),
                    "note": "This shows how stable the directional read stayed across guarded scenario changes.",
                    "tone": ui.tone_from_signal_text(
                        sensitivity_summary.get("robustness_label"),
                        positives={"HIGH"},
                        negatives={"LOW"},
                    ),
                    "help": const.ANALYSIS_HELP_TEXT["Robustness"],
                },
                {
                    "label": "Dominant Bias",
                    "value": sensitivity_summary.get("dominant_bias", "N/A"),
                    "note": "The direction the model landed on most often across the scenarios.",
                    "tone": ui.tone_from_signal_text(
                        sensitivity_summary.get("dominant_bias"),
                        positives={"BULLISH"},
                        negatives={"BEARISH"},
                    ),
                    "help": const.ANALYSIS_HELP_TEXT["Dominant Bias"],
                },
                {
                    "label": "Scenario Count",
                    "value": str(sensitivity_summary.get("scenario_count", 0)),
                    "note": "How many nearby assumption sets were tested against the same market snapshot.",
                    "tone": "neutral",
                    "help": const.ANALYSIS_HELP_TEXT["Scenario Count"],
                },
                {
                    "label": "Verdict Variety",
                    "value": str(sensitivity_summary.get("verdict_count", 0)),
                    "note": "More verdict variety usually means the name is more sensitive to model settings.",
                    "tone": ui.tone_from_metric_threshold(sensitivity_summary.get("verdict_count", 0), good_max=2, bad_min=4),
                    "help": const.ANALYSIS_HELP_TEXT["Verdict Variety"],
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

        ui.render_help_legend(
            [
                ("Bias", const.ANALYSIS_HELP_TEXT["Bias"]),
                ("Consistency", const.ANALYSIS_HELP_TEXT["Consistency"]),
                ("Assumption Drift", const.ANALYSIS_HELP_TEXT["Assumption Drift"]),
                ("Fingerprint", const.ANALYSIS_HELP_TEXT["Fingerprint"]),
            ]
        )
        sensitivity_display = sensitivity_df.copy()
        sensitivity_display["Overall Score"] = sensitivity_display["Overall Score"].map(
            lambda value: fmt.format_value(value, "{:,.1f}")
        )
        sensitivity_display["Consistency"] = sensitivity_display["Consistency"].map(
            lambda value: fmt.format_value(value, "{:,.0f}", "/100")
        )
        sensitivity_display["Assumption Drift"] = sensitivity_display["Assumption Drift"].map(
            lambda value: fmt.format_value(value, "{:,.1f}", "%")
        )
        st.dataframe(sensitivity_display, width="stretch")
