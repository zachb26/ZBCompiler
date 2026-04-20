# -*- coding: utf-8 -*-
import pandas as pd
import streamlit as st

import constants as const
import settings
import utils_fmt as fmt


def render_options_view(model_settings, active_preset_name, active_assumption_fingerprint):
    st.subheader("Model Options")
    st.caption("Tune the main assumptions with guardrails so the model stays interpretable and does not swing wildly from small changes.")
    st.caption("Changes apply to new stock analyses, refreshed comparisons, and new portfolio runs. Cached rows remain as previously analyzed until you rerun them.")
    st.caption(f"Active profile: {active_preset_name} | Fingerprint: {active_assumption_fingerprint}")

    preset_catalog = settings.get_model_presets()
    st.caption("Switch presets using the sidebar selector.")

    feedback = st.session_state.pop("options_feedback", None)
    if feedback:
        st.success(feedback["message"])
        for note in feedback.get("notes", []):
            st.caption(note)

    preset_snapshot = pd.DataFrame(
        [
            {
                "Preset": name,
                "Profile": const.PRESET_DESCRIPTIONS.get(name, ""),
                "Weights (T/F/V/S)": (
                    f"{values['weight_technical']:.1f} / {values['weight_fundamental']:.1f} / "
                    f"{values['weight_valuation']:.1f} / {values['weight_sentiment']:.1f}"
                ),
                "Trend Tol": f"{values['tech_trend_tolerance'] * 100:.0f}%",
                "Stretch": f"{values['tech_extension_limit'] * 100:.0f}%",
                "Hold Buffer": f"{values['decision_hold_buffer']:.1f}",
                "Consistency Floor": f"{values['decision_min_confidence']:.0f}/100",
                "Cooldown": f"{int(round(values['backtest_cooldown_days']))}d",
            }
            for name, values in preset_catalog.items()
        ]
    )
    st.subheader("Preset Snapshot")
    st.dataframe(preset_snapshot, width="stretch")

    assumption_drift = settings.calculate_assumption_drift(model_settings)
    weight_values = [
        model_settings["weight_technical"],
        model_settings["weight_fundamental"],
        model_settings["weight_valuation"],
        model_settings["weight_sentiment"],
    ]
    options_metrics = st.columns(5)
    options_metrics[0].metric("Assumption Drift", fmt.format_value(assumption_drift, "{:,.1f}", "%"))
    options_metrics[1].metric("Trading Days", str(int(model_settings["trading_days_per_year"])))
    options_metrics[2].metric("Fallback Scale", fmt.format_value(model_settings["valuation_benchmark_scale"], "{:,.2f}", "x"))
    options_metrics[3].metric("Weight Spread", fmt.format_value(max(weight_values) - min(weight_values), "{:,.1f}"))
    options_metrics[4].metric("Consistency Floor", fmt.format_value(model_settings["decision_min_confidence"], "{:,.0f}", "/100"))

    if assumption_drift > 35:
        st.warning("Your active assumptions are materially different from the default model. Expect results to diverge more from the baseline.")
    else:
        st.info("The controls are intentionally range-limited so the model remains stable even when you tune it.")

    if st.button("Restore Default Assumptions", width="content"):
        st.session_state.model_settings = settings.get_default_model_settings()
        st.session_state.model_preset_name = settings.get_default_preset_name()
        st.session_state.options_feedback = {
            "message": "Default assumptions restored.",
            "notes": [],
        }
        st.rerun()

    with st.form("options_form"):
        st.subheader("Engine Weights")
        weight_col_1, weight_col_2, weight_col_3, weight_col_4 = st.columns(4)
        weight_technical = weight_col_1.slider("Technical", 0.5, 1.5, float(model_settings["weight_technical"]), 0.1, help=const.OPTIONS_HELP_TEXT["weight_technical"])
        weight_fundamental = weight_col_2.slider("Fundamental", 0.5, 1.5, float(model_settings["weight_fundamental"]), 0.1, help=const.OPTIONS_HELP_TEXT["weight_fundamental"])
        weight_valuation = weight_col_3.slider("Valuation", 0.5, 1.5, float(model_settings["weight_valuation"]), 0.1, help=const.OPTIONS_HELP_TEXT["weight_valuation"])
        weight_sentiment = weight_col_4.slider("Sentiment", 0.5, 1.5, float(model_settings["weight_sentiment"]), 0.1, help=const.OPTIONS_HELP_TEXT["weight_sentiment"], disabled=True)

        st.subheader("Technical and Fundamental Thresholds")
        tf_col_1, tf_col_2, tf_col_3, tf_col_4 = st.columns(4)
        tech_rsi_oversold = tf_col_1.slider("RSI Oversold", 20, 45, int(model_settings["tech_rsi_oversold"]), 1, help=const.OPTIONS_HELP_TEXT["tech_rsi_oversold"])
        tech_rsi_overbought = tf_col_2.slider("RSI Overbought", 55, 85, int(model_settings["tech_rsi_overbought"]), 1, help=const.OPTIONS_HELP_TEXT["tech_rsi_overbought"])
        tech_momentum_percent = tf_col_3.slider(
            "Momentum Trigger (%)",
            1,
            12,
            int(round(model_settings["tech_momentum_threshold"] * 100)),
            1,
            help=const.OPTIONS_HELP_TEXT["tech_momentum_threshold"],
        )
        fund_roe_percent = tf_col_4.slider(
            "ROE Threshold (%)",
            5,
            35,
            int(round(model_settings["fund_roe_threshold"] * 100)),
            1,
            help=const.OPTIONS_HELP_TEXT["fund_roe_threshold"],
        )

        tf_col_5, tf_col_6, tf_col_7, tf_col_8, tf_col_9 = st.columns(5)
        fund_margin_percent = tf_col_5.slider(
            "Profit Margin (%)",
            5,
            35,
            int(round(model_settings["fund_profit_margin_threshold"] * 100)),
            1,
            help=const.OPTIONS_HELP_TEXT["fund_profit_margin_threshold"],
        )
        fund_debt_good = tf_col_6.slider(
            "Healthy Debt/Equity",
            25,
            200,
            int(round(model_settings["fund_debt_good_threshold"])),
            5,
            help=const.OPTIONS_HELP_TEXT["fund_debt_good_threshold"],
        )
        fund_debt_bad = tf_col_7.slider(
            "High Debt/Equity",
            75,
            400,
            int(round(model_settings["fund_debt_bad_threshold"])),
            5,
            help=const.OPTIONS_HELP_TEXT["fund_debt_bad_threshold"],
        )
        fund_revenue_growth_percent = tf_col_8.slider(
            "Revenue Growth (%)",
            0,
            30,
            int(round(model_settings["fund_revenue_growth_threshold"] * 100)),
            1,
            help=const.OPTIONS_HELP_TEXT["fund_revenue_growth_threshold"],
        )
        fund_current_ratio_good = tf_col_9.slider(
            "Healthy Current Ratio",
            1.0,
            3.0,
            float(model_settings["fund_current_ratio_good"]),
            0.1,
            help=const.OPTIONS_HELP_TEXT["fund_current_ratio_good"],
        )

        st.subheader("Decision Stability")
        ds_col_1, ds_col_2, ds_col_3, ds_col_4 = st.columns(4)
        tech_trend_tolerance_percent = ds_col_1.slider(
            "Trend Tolerance (%)",
            0,
            5,
            int(round(model_settings["tech_trend_tolerance"] * 100)),
            1,
            help=const.OPTIONS_HELP_TEXT["tech_trend_tolerance"],
        )
        tech_extension_limit_percent = ds_col_2.slider(
            "Stretch Limit (%)",
            3,
            15,
            int(round(model_settings["tech_extension_limit"] * 100)),
            1,
            help=const.OPTIONS_HELP_TEXT["tech_extension_limit"],
        )
        decision_hold_buffer = ds_col_3.slider(
            "Hold Buffer",
            0.0,
            3.0,
            float(model_settings["decision_hold_buffer"]),
            0.5,
            help=const.OPTIONS_HELP_TEXT["decision_hold_buffer"],
        )
        decision_min_confidence = ds_col_4.slider(
            "Consistency Floor",
            35,
            80,
            int(round(model_settings["decision_min_confidence"])),
            1,
            help=const.OPTIONS_HELP_TEXT["decision_min_confidence"],
        )

        st.subheader("Valuation Fallbacks and Portfolio")
        vs_col_1, vs_col_2, vs_col_3, vs_col_4 = st.columns(4)
        valuation_benchmark_scale = vs_col_1.slider(
            "Fallback Benchmark Scale",
            0.8,
            1.2,
            float(model_settings["valuation_benchmark_scale"]),
            0.05,
            help=const.OPTIONS_HELP_TEXT["valuation_benchmark_scale"],
        )
        valuation_peg_threshold = vs_col_2.slider(
            "PEG Threshold",
            0.8,
            2.5,
            float(model_settings["valuation_peg_threshold"]),
            0.1,
            help=const.OPTIONS_HELP_TEXT["valuation_peg_threshold"],
        )
        valuation_graham_multiple = vs_col_3.slider(
            "Graham Overpriced Multiple",
            1.2,
            2.0,
            float(model_settings["valuation_graham_overpriced_multiple"]),
            0.05,
            help=const.OPTIONS_HELP_TEXT["valuation_graham_overpriced_multiple"],
        )
        trading_days_per_year = vs_col_4.slider(
            "Trading Days / Year",
            240,
            260,
            int(round(model_settings["trading_days_per_year"])),
            1,
            help=const.OPTIONS_HELP_TEXT["trading_days_per_year"],
        )

        vs_col_5, vs_col_6, vs_col_7, vs_col_8, vs_col_9 = st.columns(5)
        valuation_fair_score = vs_col_5.slider(
            "Fair Value Score Floor",
            1,
            4,
            int(round(model_settings["valuation_fair_score_threshold"])),
            1,
            help=const.OPTIONS_HELP_TEXT["valuation_fair_score_threshold"],
        )
        valuation_under_score = vs_col_6.slider(
            "Undervalued Score Floor",
            3,
            8,
            int(round(model_settings["valuation_under_score_threshold"])),
            1,
            help=const.OPTIONS_HELP_TEXT["valuation_under_score_threshold"],
        )
        sentiment_analyst_boost = vs_col_7.slider(
            "Analyst Sentiment Boost",
            0.0,
            4.0,
            float(model_settings["sentiment_analyst_boost"]),
            0.5,
            help=const.OPTIONS_HELP_TEXT["sentiment_analyst_boost"],
            disabled=True,
        )
        sentiment_upside_mid = vs_col_8.slider(
            "Moderate Upside (%)",
            2,
            15,
            int(round(model_settings["sentiment_upside_mid"] * 100)),
            1,
            help=const.OPTIONS_HELP_TEXT["sentiment_upside_mid"],
            disabled=True,
        )
        sentiment_upside_high = vs_col_9.slider(
            "Strong Upside (%)",
            8,
            30,
            int(round(model_settings["sentiment_upside_high"] * 100)),
            1,
            help=const.OPTIONS_HELP_TEXT["sentiment_upside_high"],
            disabled=True,
        )

        st.subheader("Overall Verdict Thresholds")
        ov_col_1, ov_col_2, ov_col_3, ov_col_4 = st.columns(4)
        overall_buy_threshold = ov_col_1.slider(
            "Buy Threshold",
            1,
            6,
            int(round(model_settings["overall_buy_threshold"])),
            1,
            help=const.OPTIONS_HELP_TEXT["overall_buy_threshold"],
        )
        overall_strong_buy_threshold = ov_col_2.slider(
            "Strong Buy Threshold",
            4,
            12,
            int(round(model_settings["overall_strong_buy_threshold"])),
            1,
            help=const.OPTIONS_HELP_TEXT["overall_strong_buy_threshold"],
        )
        overall_sell_magnitude = ov_col_3.slider(
            "Sell Threshold",
            1,
            6,
            int(round(abs(model_settings["overall_sell_threshold"]))),
            1,
            help=const.OPTIONS_HELP_TEXT["overall_sell_threshold"],
        )
        overall_strong_sell_magnitude = ov_col_4.slider(
            "Strong Sell Threshold",
            4,
            12,
            int(round(abs(model_settings["overall_strong_sell_threshold"]))),
            1,
            help=const.OPTIONS_HELP_TEXT["overall_strong_sell_threshold"],
        )

        downside_col_1, downside_col_2, current_ratio_bad_col = st.columns(3)
        sentiment_downside_mid = downside_col_1.slider(
            "Moderate Downside (%)",
            2,
            15,
            int(round(model_settings["sentiment_downside_mid"] * 100)),
            1,
            help=const.OPTIONS_HELP_TEXT["sentiment_downside_mid"],
            disabled=True,
        )
        sentiment_downside_high = downside_col_2.slider(
            "Deep Downside (%)",
            8,
            30,
            int(round(model_settings["sentiment_downside_high"] * 100)),
            1,
            help=const.OPTIONS_HELP_TEXT["sentiment_downside_high"],
            disabled=True,
        )
        fund_current_ratio_bad = current_ratio_bad_col.slider(
            "Weak Current Ratio",
            0.5,
            1.5,
            float(model_settings["fund_current_ratio_bad"]),
            0.1,
            help=const.OPTIONS_HELP_TEXT["fund_current_ratio_bad"],
        )

        backtest_cooldown_days = st.slider(
            "Backtest Re-entry Cooldown (days)",
            0,
            20,
            int(round(model_settings["backtest_cooldown_days"])),
            1,
            help=const.OPTIONS_HELP_TEXT["backtest_cooldown_days"],
        )
        backtest_transaction_cost_bps = st.slider(
            "Backtest Trading Cost (bps)",
            0.0,
            50.0,
            float(model_settings["backtest_transaction_cost_bps"]),
            1.0,
            help=const.OPTIONS_HELP_TEXT["backtest_transaction_cost_bps"],
        )

        save_options = st.form_submit_button("Save Assumptions", type="primary", width="stretch")

    if save_options:
        updated_settings = {
            "weight_technical": weight_technical,
            "weight_fundamental": weight_fundamental,
            "weight_valuation": weight_valuation,
            "weight_sentiment": weight_sentiment,
            "tech_rsi_oversold": tech_rsi_oversold,
            "tech_rsi_overbought": tech_rsi_overbought,
            "tech_momentum_threshold": tech_momentum_percent / 100,
            "tech_trend_tolerance": tech_trend_tolerance_percent / 100,
            "tech_extension_limit": tech_extension_limit_percent / 100,
            "fund_roe_threshold": fund_roe_percent / 100,
            "fund_profit_margin_threshold": fund_margin_percent / 100,
            "fund_debt_good_threshold": fund_debt_good,
            "fund_debt_bad_threshold": fund_debt_bad,
            "fund_revenue_growth_threshold": fund_revenue_growth_percent / 100,
            "fund_current_ratio_good": fund_current_ratio_good,
            "fund_current_ratio_bad": fund_current_ratio_bad,
            "valuation_benchmark_scale": valuation_benchmark_scale,
            "valuation_peg_threshold": valuation_peg_threshold,
            "valuation_graham_overpriced_multiple": valuation_graham_multiple,
            "valuation_fair_score_threshold": valuation_fair_score,
            "valuation_under_score_threshold": valuation_under_score,
            "sentiment_analyst_boost": sentiment_analyst_boost,
            "sentiment_upside_mid": sentiment_upside_mid / 100,
            "sentiment_upside_high": sentiment_upside_high / 100,
            "sentiment_downside_mid": sentiment_downside_mid / 100,
            "sentiment_downside_high": sentiment_downside_high / 100,
            "overall_buy_threshold": overall_buy_threshold,
            "overall_strong_buy_threshold": overall_strong_buy_threshold,
            "overall_sell_threshold": -overall_sell_magnitude,
            "overall_strong_sell_threshold": -overall_strong_sell_magnitude,
            "decision_hold_buffer": decision_hold_buffer,
            "decision_min_confidence": decision_min_confidence,
            "backtest_cooldown_days": backtest_cooldown_days,
            "backtest_transaction_cost_bps": backtest_transaction_cost_bps,
            "trading_days_per_year": trading_days_per_year,
        }
        normalized_settings, notes = settings.normalize_model_settings(updated_settings)
        st.session_state.model_settings = normalized_settings
        st.session_state.model_preset_name = settings.detect_matching_preset(normalized_settings)
        st.session_state.options_feedback = {
            "message": "Model assumptions updated for this session.",
            "notes": notes,
        }
        st.rerun()
