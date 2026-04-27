# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import streamlit as st

import constants as const
import utils_fmt as fmt
import utils_ui as ui


def render_frontier_chart(portfolio_cloud, frontier, cal, tangent, minimum_volatility):
    cloud = portfolio_cloud[["Volatility", "Return", "Sharpe"]].copy()
    cloud["Series"] = "Simulated Portfolios"

    frontier_data = frontier[["Volatility", "Return"]].copy()
    frontier_data["Sharpe"] = np.nan
    frontier_data["Series"] = "Efficient Frontier"

    cal_data = cal.copy()
    cal_data["Sharpe"] = np.nan
    cal_data["Series"] = "CAL"

    marker_data = pd.DataFrame(
        [
            {
                "Volatility": tangent["Volatility"],
                "Return": tangent["Return"],
                "Sharpe": tangent["Sharpe"],
                "Series": "Max Sharpe",
            },
            {
                "Volatility": minimum_volatility["Volatility"],
                "Return": minimum_volatility["Return"],
                "Sharpe": minimum_volatility["Sharpe"],
                "Series": "Min Volatility",
            },
        ]
    )

    chart_data = pd.concat([cloud, frontier_data, cal_data, marker_data], ignore_index=True)
    spec = {
        "layer": [
            {
                "transform": [{"filter": "datum.Series == 'Simulated Portfolios'"}],
                "mark": {"type": "circle", "opacity": 0.35, "size": 45},
                "encoding": {
                    "x": {"field": "Volatility", "type": "quantitative", "title": "Annualized Volatility"},
                    "y": {"field": "Return", "type": "quantitative", "title": "Annualized Return"},
                    "color": {"field": "Sharpe", "type": "quantitative", "title": "Sharpe"},
                },
            },
            {
                "transform": [{"filter": "datum.Series == 'Efficient Frontier'"}],
                "mark": {"type": "line", "strokeWidth": 3, "color": "#0b7285"},
                "encoding": {
                    "x": {"field": "Volatility", "type": "quantitative"},
                    "y": {"field": "Return", "type": "quantitative"},
                },
            },
            {
                "transform": [{"filter": "datum.Series == 'CAL'"}],
                "mark": {"type": "line", "strokeWidth": 2, "strokeDash": [6, 4], "color": "#e03131"},
                "encoding": {
                    "x": {"field": "Volatility", "type": "quantitative"},
                    "y": {"field": "Return", "type": "quantitative"},
                },
            },
            {
                "transform": [{"filter": "datum.Series == 'Max Sharpe'"}],
                "mark": {"type": "point", "filled": True, "size": 170, "shape": "diamond", "color": "#2b8a3e"},
                "encoding": {
                    "x": {"field": "Volatility", "type": "quantitative"},
                    "y": {"field": "Return", "type": "quantitative"},
                },
            },
            {
                "transform": [{"filter": "datum.Series == 'Min Volatility'"}],
                "mark": {"type": "point", "filled": True, "size": 170, "shape": "square", "color": "#f08c00"},
                "encoding": {
                    "x": {"field": "Volatility", "type": "quantitative"},
                    "y": {"field": "Return", "type": "quantitative"},
                },
            },
        ]
    }
    st.vega_lite_chart(chart_data, spec, width="stretch")


def render_portfolio_result(result, config, active_preset_name, active_assumption_fingerprint):
    tangent = result["tangent"]
    minimum_volatility = result["minimum_volatility"]
    recommendations = result["recommendations"].copy()
    sector_exposure = result["sector_exposure"].copy()

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
                "label": "CVaR-95",
                "value": fmt.format_percent(tangent["CVaR-95"]),
                "note": "Expected annualized loss in the worst 5% of return days. Lower is better.",
                "tone": ui.tone_from_metric_threshold(tangent["CVaR-95"], good_max=0.20, bad_min=0.40),
                "help": const.ANALYSIS_HELP_TEXT["CVaR-95"],
            },
            {
                "label": "Ulcer Index",
                "value": fmt.format_percent(tangent["Ulcer Index"]),
                "note": "Combines drawdown depth and duration. Lower means the portfolio recovers faster.",
                "tone": ui.tone_from_metric_threshold(tangent["Ulcer Index"], good_max=0.08, bad_min=0.20),
                "help": const.ANALYSIS_HELP_TEXT["Ulcer Index"],
            },
            {
                "label": "Min-Vol Return",
                "value": fmt.format_percent(minimum_volatility["Return"]),
                "note": "The return estimate for the lowest-volatility portfolio the simulation found.",
                "tone": ui.tone_from_metric_threshold(minimum_volatility["Return"], good_min=0.07, bad_max=0.02),
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
        columns=6,
    )

    ui.render_analysis_signal_cards(
        [
            {
                "label": "CVaR-95",
                "value": fmt.format_percent(tangent["CVaR95"]),
                "note": "Expected annualized loss in the worst 5% of trading days.",
                "tone": ui.tone_from_metric_threshold(tangent["CVaR95"], good_min=-0.15, bad_max=-0.35),
                "help": const.ANALYSIS_HELP_TEXT["CVaR95"],
            },
            {
                "label": "Ulcer Index",
                "value": fmt.format_value(tangent["UlcerIndex"], "{:.2f}"),
                "note": "Lower is better — combines drawdown depth and duration into one number.",
                "tone": ui.tone_from_metric_threshold(tangent["UlcerIndex"], good_max=5, bad_min=15),
                "help": const.ANALYSIS_HELP_TEXT["UlcerIndex"],
            },
            {
                "label": "Max Drawdown",
                "value": fmt.format_percent(tangent["MaxDrawdown"]),
                "note": "The worst peak-to-trough decline the tangent portfolio experienced.",
                "tone": ui.tone_from_metric_threshold(tangent["MaxDrawdown"], good_min=-0.15, bad_max=-0.35),
                "help": const.ANALYSIS_HELP_TEXT["MaxDrawdown"],
            },
        ],
        columns=3,
    )

    st.markdown("##### Efficient Frontier and CAL")
    st.caption("The green diamond is the tangent portfolio with the highest Sharpe ratio. The red dashed line is the Capital Allocation Line.")
    render_frontier_chart(result["portfolio_cloud"], result["frontier"], result["cal"], tangent, minimum_volatility)

    st.markdown("##### Recommended Allocation")
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
        st.markdown("##### Sector Exposure")
        sector_display = sector_exposure.copy()
        sector_display["Recommended Weight"] = sector_display["Recommended Weight"].map(fmt.format_percent)
        st.dataframe(sector_display, width="stretch")

    with metrics_col:
        st.markdown("##### Per-Stock Metrics")
        asset_display = result["asset_metrics"].copy()
        for column in ["Annual Return", "Volatility", "Downside Volatility"]:
            asset_display[column] = asset_display[column].map(fmt.format_percent)
        for column in ["Beta", "Sharpe Ratio", "Sortino Ratio", "Treynor Ratio"]:
            asset_display[column] = asset_display[column].map(fmt.format_value)
        st.dataframe(asset_display, width="stretch")

    st.markdown("##### Portfolio Building Notes")
    for note in result["notes"]:
        st.write(f"- {note}")
