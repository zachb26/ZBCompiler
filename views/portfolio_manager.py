# -*- coding: utf-8 -*-
import datetime

import numpy as np
import pandas as pd
import streamlit as st

import constants as const
import fetch
import utils_fmt as fmt
import utils_ui as ui
import analysis_prep as prep
import skill_briefs as briefs
from ui.charts import render_portfolio_result


def normalize_recommendation_label(value):
    normalized = fmt.normalize_ticker(value)
    if "BUY" in normalized:
        return "Buy"
    if "SELL" in normalized:
        return "Sell"
    return "Hold"


def build_portfolio_composition_snapshot(library_df, portfolio_tickers):
    normalized_tickers = [fmt.normalize_ticker(ticker) for ticker in portfolio_tickers or [] if str(ticker).strip()]
    unique_tickers = list(dict.fromkeys(normalized_tickers))
    if not unique_tickers:
        return {
            "holdings_count": 0,
            "analyzed_count": 0,
            "weighted_pe": None,
            "weighted_beta": None,
            "weighted_dividend_yield": None,
            "sector_breakdown": pd.DataFrame(columns=["Sector", "Holdings", "Weight"]),
            "holdings_frame": pd.DataFrame(),
        }

    holdings_frame = pd.DataFrame()
    if library_df is not None and not library_df.empty:
        holdings_frame = library_df[library_df["Ticker"].isin(unique_tickers)].copy()
        holdings_frame = holdings_frame.drop_duplicates(subset=["Ticker"], keep="first")

    analyzed_count = holdings_frame["Ticker"].nunique() if not holdings_frame.empty else 0
    if holdings_frame.empty:
        return {
            "holdings_count": len(unique_tickers),
            "analyzed_count": 0,
            "weighted_pe": None,
            "weighted_beta": None,
            "weighted_dividend_yield": None,
            "sector_breakdown": pd.DataFrame(columns=["Sector", "Holdings", "Weight"]),
            "holdings_frame": holdings_frame,
        }

    equal_weights = np.repeat(1 / len(holdings_frame), len(holdings_frame))

    def weighted_average(column_name):
        values = pd.to_numeric(holdings_frame.get(column_name), errors="coerce")
        mask = values.notna()
        if not mask.any():
            return None
        local_weights = equal_weights[mask.to_numpy()]
        return float(np.average(values[mask], weights=local_weights))

    sector_breakdown = (
        holdings_frame.groupby("Sector", dropna=False)["Ticker"]
        .count()
        .reset_index(name="Holdings")
        .sort_values(["Holdings", "Sector"], ascending=[False, True])
    )
    sector_breakdown["Weight"] = sector_breakdown["Holdings"] / sector_breakdown["Holdings"].sum()

    return {
        "holdings_count": len(unique_tickers),
        "analyzed_count": int(analyzed_count),
        "weighted_pe": weighted_average("PE_Ratio"),
        "weighted_beta": weighted_average("Equity_Beta"),
        "weighted_dividend_yield": weighted_average("Dividend_Yield"),
        "sector_breakdown": sector_breakdown,
        "holdings_frame": holdings_frame,
    }


def build_trade_flags_dataframe(db, library_df, selected_portfolio, view_all_portfolios=False):
    memberships = db.get_portfolio_memberships()
    if memberships.empty:
        return pd.DataFrame()

    if not view_all_portfolios:
        memberships = memberships[memberships["portfolio"] == selected_portfolio]
    if memberships.empty:
        return pd.DataFrame()

    decision_log = db.get_decision_log()
    if decision_log.empty:
        return pd.DataFrame()

    latest_decisions = (
        decision_log.sort_values(["timestamp", "id"], ascending=[False, False])
        .drop_duplicates(subset=["portfolio", "ticker"], keep="first")
    )

    current_rows = pd.DataFrame()
    if library_df is not None and not library_df.empty:
        current_rows = (
            library_df.sort_values("Last_Updated_Parsed", ascending=False, na_position="last")
            .drop_duplicates(subset=["Ticker"], keep="first")
            .set_index("Ticker", drop=False)
        )

    flag_rows = []
    for membership in memberships.itertuples(index=False):
        portfolio_name = str(membership.portfolio or "").strip()
        ticker = fmt.normalize_ticker(membership.ticker)
        if not portfolio_name or not ticker:
            continue
        matching_decision = latest_decisions[
            (latest_decisions["portfolio"] == portfolio_name) & (latest_decisions["ticker"] == ticker)
        ]
        if matching_decision.empty or ticker not in current_rows.index:
            continue
        decision_row = matching_decision.iloc[0]
        current_row = current_rows.loc[ticker]
        current_recommendation = normalize_recommendation_label(current_row.get("Verdict_Overall"))
        last_recommendation = normalize_recommendation_label(decision_row.get("recommendation"))
        if current_recommendation == last_recommendation:
            continue
        flag_rows.append(
            {
                "Portfolio": portfolio_name,
                "Ticker": ticker,
                "Last Logged Decision": last_recommendation,
                "Current Model Verdict": str(current_row.get("Verdict_Overall") or "Unknown"),
                "Current Recommendation": current_recommendation,
                "Freshness": str(current_row.get("Freshness") or "Unknown"),
                "Risk Flags": str(current_row.get("Risk_Flags") or ""),
                "Latest Decision Timestamp": str(decision_row.get("timestamp") or ""),
            }
        )

    if not flag_rows:
        return pd.DataFrame()
    return pd.DataFrame(flag_rows).sort_values(["Portfolio", "Ticker"]).reset_index(drop=True)


def build_catalyst_calendar(tickers, n_days=30):
    today = datetime.date.today()
    cutoff = today + datetime.timedelta(days=n_days)
    rows = []

    for ticker in tickers or []:
        cal, _ = fetch.fetch_ticker_calendar_with_retry(fmt.normalize_ticker(ticker))
        if not cal:
            continue
        for event_key, event_label in [("earnings_date", "Earnings"), ("ex_div_date", "Ex-Dividend")]:
            dt = cal.get(event_key)
            if dt and today <= dt <= cutoff:
                rows.append({
                    "Date": dt,
                    "Days Away": (dt - today).days,
                    "Ticker": fmt.normalize_ticker(ticker),
                    "Event": event_label,
                })

    for fomc_date in const.FOMC_MEETING_DATES:
        if today <= fomc_date <= cutoff:
            rows.append({
                "Date": fomc_date,
                "Days Away": (fomc_date - today).days,
                "Ticker": "FOMC",
                "Event": "Fed Meeting",
            })

    if not rows:
        return pd.DataFrame(columns=["Date", "Days Away", "Ticker", "Event"])
    df = pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
    df["Date"] = df["Date"].astype(str)
    return df


def _render_catalyst_calendar(tickers: list[str]) -> None:
    """Display upcoming earnings and ex-dividend dates for *tickers*."""
    st.markdown("##### Catalyst Calendar")
    lookforward = st.number_input(
        "Look-forward window (days)",
        min_value=7,
        max_value=90,
        value=const.CATALYST_LOOKFORWARD_DAYS,
        step=1,
        key="pm_catalyst_lookforward",
        help="Scan for earnings and ex-dividend events within this many calendar days.",
    )
    today = datetime.date.today()
    cutoff = today + datetime.timedelta(days=int(lookforward))

    event_rows = []
    with st.spinner("Fetching catalyst dates..."):
        for ticker in tickers:
            cal = fetch.fetch_calendar_events(ticker)
            for event_label, date_val in (
                ("Earnings", cal.get("earnings_date")),
                ("Ex-Dividend", cal.get("ex_div_date")),
            ):
                if date_val is None:
                    continue
                if isinstance(date_val, datetime.datetime):
                    date_val = date_val.date()
                if today <= date_val <= cutoff:
                    days_away = (date_val - today).days
                    note = ""
                    if event_label == "Earnings" and days_away <= const.EARNINGS_CAUTION_DAYS:
                        note = "Size-Down Zone"
                    event_rows.append({
                        "Ticker": ticker,
                        "Event": event_label,
                        "Date": date_val.strftime("%Y-%m-%d"),
                        "Days Away": days_away,
                        "Note": note,
                    })

    catalyst_count = len({r["Ticker"] for r in event_rows})
    st.metric("Positions with Near-Term Catalysts", catalyst_count)

    if not event_rows:
        st.info(f"No catalysts found in the next {int(lookforward)} days.")
        return

    catalyst_df = (
        pd.DataFrame(event_rows)
        .sort_values("Days Away")
        .reset_index(drop=True)
    )
    st.dataframe(catalyst_df, width="stretch")


def render_portfolio_manager_view(db, portfolio_bot, active_preset_name, active_assumption_fingerprint):
    from ui.auth import render_password_gate

    st.subheader("Portfolio Manager")
    st.caption("View portfolio dashboards without a password. Authentication is only required for portfolio changes and decision logging.")

    portfolio_memberships_available = db.supports_portfolio_memberships
    decision_log_available = db.supports_decision_log
    if not portfolio_memberships_available or not decision_log_available:
        unavailable_features = []
        if not portfolio_memberships_available:
            unavailable_features.append("portfolio memberships")
        if not decision_log_available:
            unavailable_features.append("decision logging")
        st.warning(
            "The connected Postgres database is available for research rows, but "
            + " and ".join(unavailable_features)
            + " could not be initialized. The affected Portfolio Manager tools are running in read-only mode."
        )

    selected_portfolio = st.selectbox("Portfolio", const.PORTFOLIO_OPTIONS, key="pm_selected_portfolio")
    memberships_df = db.get_portfolio_memberships()
    selected_tickers = db.get_portfolio_tickers(selected_portfolio)
    library_df = prep.prepare_analysis_dataframe(db.get_all_analyses())
    composition = build_portfolio_composition_snapshot(library_df, selected_tickers)

    ui.render_analysis_signal_cards(
        [
            {
                "label": "Holdings",
                "value": str(composition["holdings_count"]),
                "note": "Total tickers currently assigned to the selected portfolio.",
                "tone": "neutral",
                "help": const.ANALYSIS_HELP_TEXT["Records"],
            },
            {
                "label": "Analyzed Holdings",
                "value": str(composition["analyzed_count"]),
                "note": "How many assigned names already have a saved research row in the library.",
                "tone": ui.tone_from_metric_threshold(composition["analyzed_count"], good_min=max(1, composition["holdings_count"] - 1), bad_max=0),
                "help": const.ANALYSIS_HELP_TEXT["Freshness"],
            },
            {
                "label": "Weighted Avg P/E",
                "value": fmt.format_value(composition["weighted_pe"]),
                "note": "An equal-weight snapshot of the selected portfolio's earnings multiple.",
                "tone": "neutral",
                "help": const.ANALYSIS_HELP_TEXT["P/E Ratio"],
            },
            {
                "label": "Weighted Avg Beta",
                "value": fmt.format_value(composition["weighted_beta"]),
                "note": "An equal-weight snapshot of overall market sensitivity.",
                "tone": "neutral",
                "help": const.ANALYSIS_HELP_TEXT["Portfolio Beta"],
            },
            {
                "label": "Weighted Avg Dividend Yield",
                "value": fmt.format_percent(composition["weighted_dividend_yield"]),
                "note": "An equal-weight snapshot of income exposure across the selected portfolio.",
                "tone": "neutral",
                "help": const.ANALYSIS_HELP_TEXT["Dividend Yield"],
            },
        ],
        columns=5,
    )

    sector_breakdown = composition["sector_breakdown"].copy()
    if not sector_breakdown.empty:
        sector_breakdown["Weight"] = sector_breakdown["Weight"].map(fmt.format_percent)
    sector_col, holdings_col = st.columns([1, 2])
    with sector_col:
        st.markdown("##### Sector Breakdown")
        if sector_breakdown.empty:
            st.info("No analyzed holdings are available yet for a sector breakdown.")
        else:
            st.dataframe(sector_breakdown, width="stretch")
    with holdings_col:
        st.markdown("##### Portfolio Memberships")
        if not portfolio_memberships_available:
            st.info("Portfolio membership data is unavailable in the connected Postgres database.")
        elif memberships_df.empty:
            st.info("No portfolio memberships have been saved yet.")
        else:
            membership_view = memberships_df.copy()
            membership_view.columns = ["Ticker", "Portfolio"]
            st.dataframe(membership_view, width="stretch")

    st.markdown("##### Upcoming Catalysts")
    catalyst_window = st.selectbox(
        "Look-ahead Window",
        [7, 14, 30, 60],
        index=2,
        key="pm_catalyst_window",
        help="Show earnings dates, ex-dividend dates, and FOMC meetings within this many calendar days.",
    )
    catalyst_df = build_catalyst_calendar(selected_tickers, n_days=catalyst_window)
    if catalyst_df.empty:
        st.info(f"No earnings dates, ex-dividend dates, or FOMC meetings found within the next {catalyst_window} days for the assigned tickers.")
    else:
        st.dataframe(catalyst_df, width="stretch")

    st.markdown("##### Change Controls")
    manager_authenticated = render_password_gate(
        "portfolio_manager_authenticated",
        const.PORTFOLIO_MANAGER_PASSWORD_SECRET,
        "Portfolio Manager Access",
        "Unlock this section to manage portfolio memberships and write new entries to the decision log.",
        "Unlock Portfolio Manager",
    )
    if manager_authenticated:
        manage_col, decision_col = st.columns(2)
        with manage_col:
            st.markdown("###### Manage Tickers")
            if not portfolio_memberships_available:
                st.info("Portfolio membership edits are unavailable in the connected Postgres database.")
            else:
                manage_ticker = st.text_input(
                    "Ticker to Assign",
                    value=st.session_state.get("pm_manage_ticker", ""),
                    key="pm_manage_ticker",
                )
                selected_assignments = st.multiselect(
                    "Assign to Portfolios",
                    const.PORTFOLIO_OPTIONS,
                    key="pm_manage_portfolios",
                    help="Select one or more portfolios. Leaving this blank removes the ticker from all portfolios.",
                )
                if st.button("Save Memberships", key="pm_save_memberships", width="stretch"):
                    cleaned_ticker = fmt.normalize_ticker(manage_ticker)
                    if not cleaned_ticker:
                        st.error("Enter a ticker before saving portfolio memberships.")
                    else:
                        db.save_portfolio_memberships(cleaned_ticker, selected_assignments)
                        st.session_state.pm_membership_feedback = f"Updated portfolio assignments for {cleaned_ticker}."
                        st.rerun()
                if st.session_state.get("pm_membership_feedback"):
                    st.success(st.session_state.pop("pm_membership_feedback"))

        with decision_col:
            st.markdown("###### Decision Log Entry")
            if not decision_log_available:
                st.info("Decision log writes are unavailable in the connected Postgres database.")
            else:
                default_portfolio_index = const.PORTFOLIO_OPTIONS.index(selected_portfolio) if selected_portfolio in const.PORTFOLIO_OPTIONS else 0
                with st.form("pm_decision_log_form"):
                    log_portfolio = st.selectbox("Portfolio", const.PORTFOLIO_OPTIONS, index=default_portfolio_index)
                    log_ticker = st.text_input("Ticker")
                    log_recommendation = st.selectbox("Recommendation", ["Buy", "Sell", "Hold"])
                    log_rationale = st.text_area("Rationale", height=140)
                    save_log_entry = st.form_submit_button("Save Decision", type="primary", width="stretch")
                if save_log_entry:
                    cleaned_ticker = fmt.normalize_ticker(log_ticker)
                    cleaned_rationale = str(log_rationale or "").strip()
                    if not cleaned_ticker or not cleaned_rationale:
                        st.error("Portfolio, ticker, recommendation, and rationale are all required.")
                    else:
                        db.add_decision_log_entry(log_portfolio, cleaned_ticker, log_recommendation, cleaned_rationale)
                        st.session_state.pm_log_feedback = f"Saved {log_recommendation} decision for {cleaned_ticker}."
                        st.rerun()
                if st.session_state.get("pm_log_feedback"):
                    st.success(st.session_state.pop("pm_log_feedback"))

    st.markdown("##### Performance Dashboard")
    with st.form("pm_dashboard_form"):
        pm_col_1, pm_col_2, pm_col_3 = st.columns([2, 1, 1])
        with pm_col_1:
            benchmark_ticker = st.text_input("Benchmark", value=const.DEFAULT_BENCHMARK_TICKER, key="pm_benchmark")
        with pm_col_2:
            lookback_period = st.selectbox("Lookback Period", ["1y", "3y", "5y"], index=1, key="pm_lookback_period")
        with pm_col_3:
            risk_free_percent = st.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=15.0, value=4.0, step=0.25, key="pm_risk_free")
        pm_col_4, pm_col_5 = st.columns([2, 1])
        with pm_col_4:
            max_weight_percent = st.slider("Max Single-Stock Weight (%)", min_value=15, max_value=50, value=30, step=5, key="pm_max_weight")
        with pm_col_5:
            simulations = st.select_slider("Frontier Simulations", options=[1000, 2000, 3000, 4000, 5000], value=3000, key="pm_simulations")
        refresh_dashboard = st.form_submit_button("Build / Refresh Selected Portfolio Dashboard", type="primary", width="stretch")

    if refresh_dashboard:
        if len(selected_tickers) < 2:
            st.error("Assign at least two tickers to the selected portfolio before building its dashboard.")
        elif len(selected_tickers) * (max_weight_percent / 100) < 1:
            st.error("The max single-stock weight is too low for the number of assigned holdings.")
        else:
            with st.spinner(f"Building the {selected_portfolio} dashboard..."):
                portfolio_result = portfolio_bot.analyze_portfolio(
                    tickers=selected_tickers,
                    benchmark_ticker=fmt.normalize_ticker(benchmark_ticker) or const.DEFAULT_BENCHMARK_TICKER,
                    period=lookback_period,
                    risk_free_rate=risk_free_percent / 100,
                    max_weight=max_weight_percent / 100,
                    simulations=simulations,
                )
            if not portfolio_result:
                st.session_state.pop("pm_portfolio_result", None)
                st.session_state.pop("pm_portfolio_config", None)
                st.error(portfolio_bot.last_error or "Unable to build the selected portfolio dashboard right now.")
            else:
                portfolio_result["selected_portfolio"] = selected_portfolio
                st.session_state.pm_portfolio_result = portfolio_result
                st.session_state.pm_portfolio_config = {
                    "selected_portfolio": selected_portfolio,
                    "benchmark": fmt.normalize_ticker(benchmark_ticker) or const.DEFAULT_BENCHMARK_TICKER,
                    "period": lookback_period,
                    "risk_free_percent": risk_free_percent,
                    "max_weight_percent": max_weight_percent,
                    "simulations": simulations,
                }

    pm_result = st.session_state.get("pm_portfolio_result")
    pm_config = st.session_state.get("pm_portfolio_config", {})
    if pm_result and pm_config.get("selected_portfolio") == selected_portfolio:
        render_portfolio_result(pm_result, pm_config, active_preset_name, active_assumption_fingerprint)
        asset_prices = pm_result.get("asset_prices")
        recommendations = pm_result.get("recommendations", pd.DataFrame()).copy()
        if isinstance(asset_prices, pd.DataFrame) and not asset_prices.empty and not recommendations.empty:
            total_returns = (asset_prices.iloc[-1] / asset_prices.iloc[0] - 1).rename("Period Return")
            pnl_table = recommendations.merge(
                total_returns.reset_index().rename(columns={"index": "Ticker"}),
                on="Ticker",
                how="left",
            )
            pnl_table = pnl_table[["Ticker", "Name", "Sector", "Recommended Weight", "Period Return", "Role"]].copy()
            pnl_table["Recommended Weight"] = pnl_table["Recommended Weight"].map(fmt.format_percent)
            pnl_table["Period Return"] = pnl_table["Period Return"].map(fmt.format_percent)
            st.markdown("##### Position-Level P&L Snapshot")
            st.caption("This uses the selected lookback window's total return as a clean position-level performance proxy.")
            st.dataframe(pnl_table, width="stretch")

        _render_catalyst_calendar(recommendations["Ticker"].tolist() if not recommendations.empty else selected_tickers)

        with st.expander("Export for Claude Code Skills"):
            st.caption(
                "Download a pre-formatted brief for wealth management skills. "
                "In Claude Code, run `/wealth-management:rebalance` and attach the file when prompted."
            )
            _pm_brief = briefs.build_rebalance_skill_brief(pm_result, selected_portfolio)
            st.download_button(
                "Rebalance Brief (.md)",
                data=_pm_brief.encode("utf-8"),
                file_name="portfolio_rebalance_brief.md",
                mime="text/markdown",
                key="skill_pm_rebalance",
                help="Input for /wealth-management:rebalance",
                width="stretch",
            )

    st.markdown("##### Decision Log")
    if not decision_log_available:
        st.info("Decision log history is unavailable in the connected Postgres database.")
    else:
        decision_filter_col_1, decision_filter_col_2 = st.columns([1, 1])
        with decision_filter_col_1:
            log_portfolio_filter = st.selectbox(
                "Portfolio Filter",
                ["All Portfolios"] + const.PORTFOLIO_OPTIONS,
                index=(["All Portfolios"] + const.PORTFOLIO_OPTIONS).index(selected_portfolio),
                key="pm_log_portfolio_filter",
            )
        with decision_filter_col_2:
            log_ticker_filter = st.text_input("Ticker Filter", key="pm_log_ticker_filter")

        filtered_log = db.get_decision_log(
            portfolio=None if log_portfolio_filter == "All Portfolios" else log_portfolio_filter,
            ticker=log_ticker_filter,
        )
        if filtered_log.empty:
            st.info("No decision log entries match the current filters.")
        else:
            log_display = filtered_log.copy()
            log_display.columns = ["ID", "Timestamp", "Portfolio", "Ticker", "Recommendation", "Rationale"]
            st.dataframe(log_display, width="stretch")

    st.markdown("##### Trade Flags")
    if not decision_log_available:
        st.info("Trade flags are unavailable until decision logging is available in the connected Postgres database.")
    else:
        view_all_portfolios = st.toggle(
            "View All Portfolios",
            value=False,
            help="Turn this on to scan every portfolio for decision flips instead of only the selected one.",
        )
        flag_mode = "All Portfolios" if view_all_portfolios else selected_portfolio
        st.caption(f"Current scope: {flag_mode}")
        trade_flags = build_trade_flags_dataframe(
            db,
            library_df,
            selected_portfolio=selected_portfolio,
            view_all_portfolios=view_all_portfolios,
        )
        if trade_flags.empty:
            st.success("No trade flags were found between the latest logged decisions and the current saved model verdicts.")
        else:
            st.dataframe(trade_flags, width="stretch")
