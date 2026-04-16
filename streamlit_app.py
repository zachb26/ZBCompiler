# -*- coding: utf-8 -*-
import datetime
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import copy
import tempfile
import shutil
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import psycopg
import streamlit as st
import yfinance as yf

import constants as const
import utils_fmt as fmt
import utils_time as tutil
import utils_ui as ui
import utils_news as news
import exports
import skill_briefs as briefs
import cache
import fetch
import sec_ai as sec
import dcf
import analytics_tech as tech
import analytics_scoring as scoring
import analytics_decision as decision
import settings
import analysis_prep as prep
import backtest


def compute_relative_strength(close, benchmark_close, window):
    if close is None or benchmark_close is None or window <= 0:
        return None
    close = close.dropna()
    benchmark_close = benchmark_close.dropna()
    aligned = pd.concat([close.rename("stock"), benchmark_close.rename("benchmark")], axis=1, join="inner").dropna()
    if len(aligned) <= window:
        return None
    stock_return = fetch.safe_divide(aligned["stock"].iloc[-1] - aligned["stock"].iloc[-window - 1], aligned["stock"].iloc[-window - 1])
    benchmark_return = fetch.safe_divide(
        aligned["benchmark"].iloc[-1] - aligned["benchmark"].iloc[-window - 1],
        aligned["benchmark"].iloc[-window - 1],
    )
    if stock_return is None or benchmark_return is None:
        return None
    return float(stock_return - benchmark_return)



def get_startup_refresh_snapshot():
    with const.STARTUP_REFRESH_LOCK:
        return const.STARTUP_REFRESH_STATE.copy()


def render_compiling_badge(placeholder, message):
    if placeholder is None:
        return
    safe_message = (
        str(message)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
    placeholder.markdown(
        f"""
        <style>
            .compile-badge {{
                position: fixed;
                right: 1rem;
                bottom: 1rem;
                z-index: 99999;
                padding: 0.8rem 1rem;
                border-radius: 14px;
                background: rgba(15, 23, 42, 0.94);
                color: #f8fafc;
                border: 1px solid rgba(148, 163, 184, 0.35);
                box-shadow: 0 16px 36px rgba(15, 23, 42, 0.24);
                font-size: 0.92rem;
                line-height: 1.35;
                max-width: 320px;
            }}
        </style>
        <div class="compile-badge">{safe_message}</div>
        """,
        unsafe_allow_html=True,
    )


def format_startup_refresh_message(state):
    message = "Compiling... (May take a while.)"
    if state.get("running"):
        total = int(state.get("total", 0))
        processed = int(state.get("processed", 0))
        if total > 0:
            message += f"\nRefreshing stale analyses {processed}/{total}"
        else:
            message += "\nRefreshing stale analyses"
    return message


def collect_stale_analysis_tickers(db, stale_after_hours=const.AUTO_REFRESH_STALE_AFTER_HOURS):
    saved_rows = db.get_all_analyses()
    if saved_rows.empty or "Ticker" not in saved_rows.columns:
        return []

    refresh_candidates = saved_rows.copy()
    if "Last_Updated" in refresh_candidates.columns:
        refresh_candidates["Last_Updated_Parsed"] = refresh_candidates["Last_Updated"].map(tutil.parse_last_updated)
        stale_cutoff = datetime.datetime.now() - datetime.timedelta(hours=stale_after_hours)
        refresh_candidates = refresh_candidates[
            refresh_candidates["Last_Updated_Parsed"].isna()
            | (refresh_candidates["Last_Updated_Parsed"] < stale_cutoff)
        ]
        refresh_candidates = refresh_candidates.sort_values("Last_Updated_Parsed", ascending=True, na_position="first")

    return (
        refresh_candidates["Ticker"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .drop_duplicates()
        .tolist()
    )


def refresh_saved_analyses_on_launch(db, settings, badge_placeholder=None):
    while True:
        with const.STARTUP_REFRESH_LOCK:
            if const.STARTUP_REFRESH_STATE["complete"]:
                return const.STARTUP_REFRESH_STATE.copy()
            if not const.STARTUP_REFRESH_STATE["started"]:
                const.STARTUP_REFRESH_STATE.update(
                    {
                        "started": True,
                        "running": True,
                        "complete": False,
                        "total": 0,
                        "processed": 0,
                        "updated": 0,
                        "failed": 0,
                        "error": None,
                        "started_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "finished_at": None,
                    }
                )
                is_leader = True
            else:
                is_leader = False

        if is_leader:
            try:
                tickers = collect_stale_analysis_tickers(db, stale_after_hours=const.AUTO_REFRESH_STALE_AFTER_HOURS)
                if not tickers:
                    with const.STARTUP_REFRESH_LOCK:
                        const.STARTUP_REFRESH_STATE.update(
                            {
                                "running": False,
                                "complete": True,
                                "total": 0,
                                "processed": 0,
                                "updated": 0,
                                "failed": 0,
                                "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            }
                        )
                    return get_startup_refresh_snapshot()
                analyst = StockAnalyst(db)
                total = len(tickers)

                with const.STARTUP_REFRESH_LOCK:
                    const.STARTUP_REFRESH_STATE["total"] = total

                render_compiling_badge(badge_placeholder, format_startup_refresh_message(get_startup_refresh_snapshot()))

                updated_count = 0
                failed_count = 0
                failure_streak = 0
                for idx, ticker in enumerate(tickers, start=1):
                    try:
                        record = analyst.analyze(ticker, settings=settings, persist=True)
                    except Exception as exc:
                        record = None
                        analyst.last_error = fetch.summarize_fetch_error(exc)

                    if record is None:
                        failed_count += 1
                        failure_streak += 1
                    else:
                        updated_count += 1
                        failure_streak = 0

                    if idx == 1 or idx % const.AUTO_REFRESH_STATUS_UPDATE_INTERVAL == 0 or idx == total:
                        with const.STARTUP_REFRESH_LOCK:
                            const.STARTUP_REFRESH_STATE["processed"] = idx
                            const.STARTUP_REFRESH_STATE["updated"] = updated_count
                            const.STARTUP_REFRESH_STATE["failed"] = failed_count
                        render_compiling_badge(
                            badge_placeholder,
                            format_startup_refresh_message(get_startup_refresh_snapshot()),
                        )
                    if failure_streak >= const.AUTO_REFRESH_FAILURE_STREAK_LIMIT:
                        with const.STARTUP_REFRESH_LOCK:
                            const.STARTUP_REFRESH_STATE["error"] = (
                                "Launch refresh paused after repeated upstream fetch failures. "
                                "Manual analysis still works and saved rows remain available."
                            )
                            const.STARTUP_REFRESH_STATE["processed"] = idx
                            const.STARTUP_REFRESH_STATE["updated"] = updated_count
                            const.STARTUP_REFRESH_STATE["failed"] = failed_count
                        break
                    time.sleep(const.AUTO_REFRESH_REQUEST_DELAY_SECONDS)

                with const.STARTUP_REFRESH_LOCK:
                    const.STARTUP_REFRESH_STATE.update(
                        {
                            "running": False,
                            "complete": True,
                            "processed": updated_count + failed_count,
                            "updated": updated_count,
                            "failed": failed_count,
                            "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )
                return get_startup_refresh_snapshot()
            except Exception as exc:
                with const.STARTUP_REFRESH_LOCK:
                    const.STARTUP_REFRESH_STATE.update(
                        {
                            "running": False,
                            "complete": True,
                            "error": fetch.summarize_fetch_error(exc),
                            "finished_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                    )
                return get_startup_refresh_snapshot()

        snapshot = get_startup_refresh_snapshot()
        if snapshot["running"]:
            render_compiling_badge(badge_placeholder, format_startup_refresh_message(snapshot))
            time.sleep(0.25)
            continue
        return snapshot



from database import DatabaseManager, get_database_manager
from analyst import StockAnalyst, PortfolioAnalyst

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


def get_secret_value(secret_name):
    try:
        value = st.secrets[secret_name]
    except Exception:
        value = os.environ.get(secret_name)
    text = str(value or "").strip()
    return text or None


def render_password_gate(session_key, secret_name, heading, description, button_label):
    if st.session_state.get(session_key):
        return True

    st.markdown(f"#### {heading}")
    st.caption(description)
    expected_password = get_secret_value(secret_name)
    if not expected_password:
        st.error(
            f"Configuration missing. Add `{secret_name}` in Streamlit Secrets before using this section."
        )
        return False

    password_key = f"{session_key}_password"
    error_key = f"{session_key}_error"
    entered_password = st.text_input("Password", type="password", key=password_key)
    if st.button(button_label, key=f"{session_key}_submit", width="stretch"):
        if entered_password == expected_password:
            st.session_state[session_key] = True
            st.session_state.pop(error_key, None)
            st.session_state[password_key] = ""
            st.rerun()
        else:
            st.session_state[error_key] = "Incorrect password."

    if st.session_state.get(error_key):
        st.error(st.session_state[error_key])
    return False


def normalize_recommendation_label(value):
    normalized = fmt.normalize_ticker(value)
    if "BUY" in normalized:
        return "Buy"
    if "SELL" in normalized:
        return "Sell"
    return "Hold"


def build_sector_news_dataframe(tickers, max_tickers=12, max_items=18):
    news_rows = []
    seen_titles = set()
    for ticker in list(tickers or [])[:max_tickers]:
        ticker_news, _ = fetch.fetch_ticker_news_with_retry(fmt.normalize_ticker(ticker), attempts=1)
        for item in ticker_news or []:
            title = news.extract_news_title(item)
            if not title:
                continue
            lowered_title = title.lower()
            if not any(keyword in lowered_title for keyword in const.FUNDAMENTAL_EVENT_KEYWORDS):
                continue
            dedupe_key = lowered_title.strip()
            if dedupe_key in seen_titles:
                continue
            seen_titles.add(dedupe_key)
            published = news.extract_news_publish_time(item)
            publisher = ""
            if isinstance(item, dict):
                publisher = str(item.get("publisher") or item.get("source") or "").strip()
                if not publisher and isinstance(item.get("content"), dict):
                    publisher = str(item["content"].get("publisher") or "").strip()
            news_rows.append(
                {
                    "Published": tutil.format_datetime_value(published, fallback="Unknown"),
                    "Ticker": fmt.normalize_ticker(ticker),
                    "Publisher": publisher or "Unknown",
                    "Headline": title,
                    "_published_sort": published or datetime.datetime.min,
                }
            )

    if not news_rows:
        return pd.DataFrame(columns=["Published", "Ticker", "Publisher", "Headline"])

    news_df = pd.DataFrame(news_rows)
    news_df = news_df.sort_values(["_published_sort", "Ticker"], ascending=[False, True]).head(max_items)
    return news_df.drop(columns=["_published_sort"]).reset_index(drop=True)


def build_sector_weekly_briefing(sector_name, sector_df, sector_news_df):
    if sector_df is None or sector_df.empty:
        return f"Weekly Briefing: {sector_name}\n\nNo tracked names are currently saved for this sector."

    sector_name = str(sector_name or "Selected Sector")
    bullish_count = int(sector_df["Verdict_Overall"].isin(["BUY", "STRONG BUY"]).sum())
    bearish_count = int(sector_df["Verdict_Overall"].isin(["SELL", "STRONG SELL"]).sum())
    avg_rs_3m = sector_df["Relative_Strength_3M"].dropna().mean() if "Relative_Strength_3M" in sector_df.columns else None
    avg_rs_6m = sector_df["Relative_Strength_6M"].dropna().mean() if "Relative_Strength_6M" in sector_df.columns else None

    movers_df = sector_df.sort_values(
        ["Relative_Strength_3M", "Composite Score", "Ticker"],
        ascending=[False, False, True],
        na_position="last",
    )
    top_movers = movers_df.head(3)["Ticker"].tolist()
    laggards = movers_df.tail(3)["Ticker"].tolist()

    risk_rows = sector_df[
        sector_df["Risk_Flags"].fillna("").astype(str).str.strip() != ""
    ][["Ticker", "Risk_Flags"]].head(3)

    lines = [
        f"Weekly Briefing: {sector_name}",
        "",
        f"Tracked names: {len(sector_df)}",
        f"Bullish verdicts: {bullish_count}",
        f"Bearish verdicts: {bearish_count}",
        f"Average 3M relative strength: {fmt.format_percent(avg_rs_3m)}",
        f"Average 6M relative strength: {fmt.format_percent(avg_rs_6m)}",
        "",
        "Top movers:",
    ]
    if top_movers:
        for ticker in top_movers:
            lines.append(f"- {ticker}")
    else:
        lines.append("- No standout movers were available.")

    lines.append("")
    lines.append("Watch list / laggards:")
    if laggards:
        for ticker in laggards:
            lines.append(f"- {ticker}")
    else:
        lines.append("- No laggards were available.")

    lines.append("")
    lines.append("Risk flags:")
    if risk_rows.empty:
        lines.append("- No major saved risk flags were surfaced in the current sector slice.")
    else:
        for risk_row in risk_rows.itertuples(index=False):
            lines.append(f"- {risk_row.Ticker}: {risk_row.Risk_Flags}")

    lines.append("")
    lines.append("Relevant headlines:")
    if sector_news_df is None or sector_news_df.empty:
        lines.append("- No recent fundamental-event headlines were available from the current feed.")
    else:
        for news_row in sector_news_df.head(5).itertuples(index=False):
            lines.append(f"- {news_row.Ticker}: {news_row.Headline}")

    return "\n".join(lines)


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
        columns=4,
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


def render_sector_leader_view(db):
    st.subheader("Sector Leader")
    st.caption("Compare all tracked names inside one sector, review relative strength and valuation side-by-side, and prepare a meeting-ready weekly briefing.")

    library_df = prep.prepare_analysis_dataframe(db.get_all_analyses())
    if library_df.empty:
        st.info("The research library is empty right now. Save some analyses first to unlock the sector dashboard.")
        return

    sector_options = sorted(sector for sector in library_df["Sector"].dropna().unique() if str(sector).strip())
    if not sector_options:
        st.info("No sectors are available in the current saved library.")
        return

    default_sector = st.session_state.get("sector_leader_sector")
    default_index = sector_options.index(default_sector) if default_sector in sector_options else 0
    selected_sector = st.selectbox("Sector", sector_options, index=default_index, key="sector_leader_sector")
    sector_df = library_df[library_df["Sector"] == selected_sector].copy().reset_index(drop=True)
    sector_tickers = sector_df["Ticker"].dropna().astype(str).str.upper().tolist()

    ui.render_analysis_signal_cards(
        [
            {
                "label": "Tracked Names",
                "value": str(len(sector_df)),
                "note": "Saved research rows currently available for this sector.",
                "tone": "neutral",
                "help": const.ANALYSIS_HELP_TEXT["Records"],
            },
            {
                "label": "Bullish Verdicts",
                "value": str(int(sector_df["Verdict_Overall"].isin(["BUY", "STRONG BUY"]).sum())),
                "note": "Names with a current Buy or Strong Buy verdict.",
                "tone": "good",
                "help": const.ANALYSIS_HELP_TEXT["Buy / Strong Buy"],
            },
            {
                "label": "Average 6M Relative Strength",
                "value": fmt.format_percent(sector_df["Relative_Strength_6M"].dropna().mean()),
                "note": f"Average six-month return relative to {const.DEFAULT_BENCHMARK_TICKER}.",
                "tone": ui.tone_from_metric_threshold(sector_df["Relative_Strength_6M"].dropna().mean(), good_min=0.03, bad_max=-0.03),
                "help": const.ANALYSIS_HELP_TEXT["Relative Strength"],
            },
            {
                "label": "Average Composite Score",
                "value": fmt.format_value(sector_df["Composite Score"].dropna().mean(), "{:,.1f}"),
                "note": "A quick read on whether the sector list looks broadly constructive or mixed.",
                "tone": ui.tone_from_metric_threshold(sector_df["Composite Score"].dropna().mean(), good_min=1, bad_max=-1),
                "help": const.ANALYSIS_HELP_TEXT["Average Composite Score"],
            },
        ],
        columns=4,
    )

    st.markdown("##### Sector Scoreboard")
    sector_table = sector_df[
        [
            "Ticker",
            "Industry",
            "Verdict_Overall",
            "PE_Ratio",
            "EV_EBITDA",
            "PS_Ratio",
            "Relative_Strength_3M",
            "Relative_Strength_6M",
            "Relative_Strength_1Y",
            "Risk_Flags",
            "Freshness",
        ]
    ].copy()
    for column_name in ["PE_Ratio", "EV_EBITDA", "PS_Ratio"]:
        sector_table[column_name] = sector_table[column_name].map(fmt.format_value)
    for column_name in ["Relative_Strength_3M", "Relative_Strength_6M", "Relative_Strength_1Y"]:
        sector_table[column_name] = sector_table[column_name].map(fmt.format_percent)
    st.dataframe(sector_table, width="stretch")

    st.markdown("##### Sector News")
    sector_news_df = build_sector_news_dataframe(sector_tickers)
    if sector_news_df.empty:
        st.info("No recent fundamental-event-tagged headlines were available for the saved names in this sector.")
    else:
        st.dataframe(sector_news_df, width="stretch")

    briefing_text = build_sector_weekly_briefing(selected_sector, sector_df, sector_news_df)
    st.markdown("##### Weekly Briefing")
    st.download_button(
        "Download Weekly Briefing",
        data=briefing_text.encode("utf-8"),
        file_name=f"{selected_sector.lower().replace(' ', '_')}_weekly_briefing.txt",
        mime="text/plain",
        width="stretch",
    )
    st.text_area("Briefing Preview", value=briefing_text, height=260)


def render_portfolio_manager_view(db, portfolio_bot, active_preset_name, active_assumption_fingerprint):
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


st.set_page_config(page_title="OSIG Research Tool", layout="wide", page_icon="SE")

if not const.DATABASE_URL:
    st.error(
        "A PostgreSQL connection is required. "
        "Set the `STOCKS_DATABASE_URL` (or `DATABASE_URL`) environment variable to a valid PostgreSQL DSN and restart the app."
    )
    st.stop()
db = get_database_manager()
bot = StockAnalyst(db)
portfolio_bot = PortfolioAnalyst(db)
model_settings = settings.get_model_settings()
active_preset_name = settings.detect_matching_preset(model_settings)
active_assumption_fingerprint = settings.get_assumption_fingerprint(model_settings)

st.title("OSIG Research Tool")
storage_status = "Connected to Postgres" if db.storage_backend == "postgres" else "Using SQLite"
st.caption(f"Version: {const.APP_VERSION} | {storage_status}")
if db.storage_notice:
    st.warning(db.storage_notice)

startup_refresh_summary = {
    "started": False,
    "running": False,
    "complete": False,
    "total": 0,
    "processed": 0,
    "updated": 0,
    "failed": 0,
    "error": None,
    "started_at": None,
    "finished_at": None,
}
if const.RUN_STARTUP_REFRESH and os.environ.get("STOCK_ENGINE_SKIP_STARTUP_REFRESH") != "1":
    startup_badge = st.empty()
    startup_refresh_summary = refresh_saved_analyses_on_launch(db, model_settings, badge_placeholder=startup_badge)
    startup_badge.empty()

sensitivity_default_ticker = st.session_state.get("sensitivity_last_ticker") or st.session_state.get("single_ticker", "")
backtest_default_ticker = st.session_state.get("backtest_last_ticker") or st.session_state.get("single_ticker", "")

senior_analyst_tools_enabled = False
hidden_senior_placeholders = []
stock_tab = None
compare_tab = None
portfolio_tab = None
sensitivity_tab = None
backtest_tab = None
library_tab = None
readme_tab = None
changelog_tab = None
methodology_tab = None
options_tab = None

new_analyst_tab, analyst_senior_tab, sector_leader_tab, portfolio_manager_tab = st.tabs(
    ["New Analyst", "Senior Analyst", "Sector Leader", "Portfolio Manager"]
)

with new_analyst_tab:
    analyst_new_tab, compare_tab, methodology_tab, readme_tab, options_tab = st.tabs(
        ["Analysis", "Comparison", "Methodology", "ReadMe", "Controls"]
    )
    with analyst_new_tab:
        render_new_analyst_view(db, bot)

with analyst_senior_tab:
    senior_analyst_tools_enabled = render_password_gate(
        "senior_analyst_authenticated",
        const.SENIOR_ANALYST_PASSWORD_SECRET,
        "Senior Analyst Access",
        "Unlock the full analyst toolkit, including valuation labs, peer analysis, SEC filing automation, backtesting, and model controls.",
        "Unlock Senior Analyst",
    )
    if senior_analyst_tools_enabled:
        stock_tab, ai_reports_tab, sensitivity_tab, backtest_tab, library_tab, senior_reference_tab = st.tabs(
            ["Single Stock", "AI Reports", "Sensitivity", "Backtest", "Library", "Changelog"]
        )
        changelog_tab = senior_reference_tab.container()
        with ai_reports_tab:
            sec.render_ai_reports_tab(db)
    else:
        stock_tab = st.empty()
        sensitivity_tab = st.empty()
        backtest_tab = st.empty()
        library_tab = st.empty()
        changelog_tab = st.empty()
        hidden_senior_placeholders = [
            stock_tab,
            sensitivity_tab,
            backtest_tab,
            library_tab,
            changelog_tab,
        ]

with sector_leader_tab:
    render_sector_leader_view(db)

with portfolio_manager_tab:
    render_portfolio_manager_view(db, portfolio_bot, active_preset_name, active_assumption_fingerprint)

portfolio_tab = st.empty()

with stock_tab:
    c1, c2 = st.columns([3, 1])
    with c1:
        txt_input = st.text_input("Enter Ticker Symbol (e.g., AAPL, NVDA, F)", "", key="single_ticker")
    with c2:
        st.write("")
        st.write("")
        if st.button("Run Full Analysis", type="primary", width="stretch"):
            if txt_input:
                with st.spinner(f"Running multiple engines on {txt_input}..."):
                    res = bot.analyze(txt_input)
                    if not res:
                        st.error(bot.last_error or "Unable to fetch enough market data for this ticker right now.")

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
            st.download_button(
                "Download Company Data",
                data=company_download_bytes,
                file_name=f"{row['Ticker']}_analysis_snapshot.json",
                mime="application/json",
                key=f"download_company_data_{row['Ticker']}",
                width="stretch",
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

            st.subheader("Method Breakdown")
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
        else:
            st.info("Run the full analysis to save this ticker into the shared research library.")

with compare_tab:
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

with portfolio_tab:
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
                portfolio_result = portfolio_bot.analyze_portfolio(
                    tickers=parsed_tickers,
                    benchmark_ticker=fmt.normalize_ticker(benchmark_ticker) or const.DEFAULT_BENCHMARK_TICKER,
                    period=lookback_period,
                    risk_free_rate=risk_free_percent / 100,
                    max_weight=max_weight_percent / 100,
                    simulations=simulations,
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

with sensitivity_tab:
    st.subheader("Sensitivity Check")
    st.caption("Run the same live market snapshot across guarded assumption scenarios to see whether the model is directionally stable or fragile.")
    st.caption("This does not overwrite the research library. It reuses one fresh market pull and reruns the scoring logic in memory.")

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

with backtest_tab:
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

with library_tab:
    st.subheader("Research Library")
    st.caption("Browse everything saved in the shared database so the research process stays visible across users and sessions.")
    if not db.supports_database_download:
        if db.storage_backend == "postgres":
            st.info("Database file export is unavailable when the app is connected to Postgres. Use the CSV export for library data.")
        else:
            st.info("Database export is unavailable in the current in-memory storage mode.")
    if startup_refresh_summary.get("error"):
        st.warning(f"Launch refresh hit an issue: {startup_refresh_summary['error']}")
    elif startup_refresh_summary.get("total", 0) > 0:
        st.caption(
            f"Launch refresh updated {startup_refresh_summary.get('updated', 0)} of "
            f"{startup_refresh_summary.get('total', 0)} stale saved analyses"
            + (
                f" and skipped {startup_refresh_summary.get('failed', 0)} tickers."
                if startup_refresh_summary.get("failed", 0)
                else "."
            )
        )

    library_df = prep.prepare_analysis_dataframe(db.get_all_analyses())
    if library_df.empty:
        database_bytes = exports.build_database_download_bytes(db.db_path if db.supports_database_download else None)
        export_col_1, export_col_2 = st.columns(2)
        with export_col_1:
            st.download_button(
                "Download Database",
                data=database_bytes,
                file_name=(db.db_path.name if db.supports_database_download else const.DB_FILENAME),
                mime="application/x-sqlite3",
                disabled=not bool(database_bytes),
                width="stretch",
            )
        with export_col_2:
            st.download_button(
                "Download Library CSV",
                data=b"",
                file_name="stock_engine_library.csv",
                mime="text/csv",
                disabled=True,
                width="stretch",
            )
        st.info("The library is empty right now. Run stock analyses or a comparison to populate the shared database.")
    else:
        sector_options = sorted(sector for sector in library_df["Sector"].dropna().unique())
        verdict_options = sorted(verdict for verdict in library_df["Verdict_Overall"].dropna().unique())
        stock_type_options = sorted(stock_type for stock_type in library_df["Stock_Type"].dropna().unique())
        filter_col_1, filter_col_2, filter_col_3, filter_col_4 = st.columns([2, 2, 2, 1])
        with filter_col_1:
            selected_sectors = st.multiselect("Sector Filter", sector_options, default=sector_options)
        with filter_col_2:
            selected_verdicts = st.multiselect("Verdict Filter", verdict_options, default=verdict_options)
        with filter_col_3:
            selected_stock_types = st.multiselect("Stock Type Filter", stock_type_options, default=stock_type_options)
        with filter_col_4:
            fresh_only = st.checkbox("Only show last 7 days", value=False)

        filtered_library = library_df.copy()
        if selected_sectors:
            filtered_library = filtered_library[filtered_library["Sector"].isin(selected_sectors)]
        else:
            filtered_library = filtered_library.iloc[0:0]
        if selected_verdicts:
            filtered_library = filtered_library[filtered_library["Verdict_Overall"].isin(selected_verdicts)]
        else:
            filtered_library = filtered_library.iloc[0:0]
        if selected_stock_types:
            filtered_library = filtered_library[filtered_library["Stock_Type"].isin(selected_stock_types)]
        else:
            filtered_library = filtered_library.iloc[0:0]
        if fresh_only:
            fresh_cutoff = datetime.datetime.now() - datetime.timedelta(days=7)
            filtered_library = filtered_library[
                filtered_library["Last_Updated_Parsed"].notna()
                & (filtered_library["Last_Updated_Parsed"] >= fresh_cutoff)
            ]

        export_frame = filtered_library if not filtered_library.empty else library_df
        database_bytes = exports.build_database_download_bytes(db.db_path if db.supports_database_download else None)
        library_csv_bytes = exports.build_library_csv_bytes(export_frame)
        export_col_1, export_col_2 = st.columns(2)
        with export_col_1:
            st.download_button(
                "Download Database",
                data=database_bytes,
                file_name=(db.db_path.name if db.supports_database_download else const.DB_FILENAME),
                mime="application/x-sqlite3",
                disabled=not bool(database_bytes),
                width="stretch",
            )
        with export_col_2:
            st.download_button(
                "Download Library CSV",
                data=library_csv_bytes,
                file_name="stock_engine_library.csv",
                mime="text/csv",
                disabled=export_frame.empty,
                width="stretch",
            )

        if filtered_library.empty:
            st.warning("No records match the current library filters.")
        else:
            if filtered_library["Assumption_Fingerprint"].nunique() > 1:
                st.caption("The current library view contains analyses generated under multiple assumption fingerprints.")
            fresh_24h = (
                filtered_library["Last_Updated_Parsed"].notna()
                & (filtered_library["Last_Updated_Parsed"] >= datetime.datetime.now() - datetime.timedelta(days=1))
            ).sum()
            ui.render_analysis_signal_cards(
                [
                    {
                        "label": "Records",
                        "value": str(len(filtered_library)),
                        "note": "The number of saved analyses visible under the current filters.",
                        "tone": "neutral",
                        "help": const.ANALYSIS_HELP_TEXT["Records"],
                    },
                    {
                        "label": "Buy / Strong Buy",
                        "value": str(filtered_library["Verdict_Overall"].isin(["BUY", "STRONG BUY"]).sum()),
                        "note": "These are the names currently carrying a bullish final verdict.",
                        "tone": "good",
                        "help": const.ANALYSIS_HELP_TEXT["Buy / Strong Buy"],
                    },
                    {
                        "label": "Fresh in 24h",
                        "value": str(int(fresh_24h)),
                        "note": "Rows refreshed within the last day usually reflect the latest saved research pass.",
                        "tone": ui.tone_from_metric_threshold(fresh_24h, good_min=5, bad_max=1),
                        "help": const.ANALYSIS_HELP_TEXT["Fresh in 24h"],
                    },
                    {
                        "label": "Tracked Sectors",
                        "value": str(filtered_library["Sector"].nunique()),
                        "note": "This shows how broad the current library slice is across industries.",
                        "tone": "neutral",
                        "help": const.ANALYSIS_HELP_TEXT["Tracked Sectors"],
                    },
                ],
                columns=4,
            )

            st.caption(f"Shared database: {db.storage_label}")

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
            library_display = filtered_library[
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
                    "Freshness",
                    "Last_Updated",
                ]
            ].copy()
            library_display["Price"] = library_display["Price"].map(lambda value: f"${value:,.2f}" if pd.notna(value) else "N/A")
            library_display["Decision_Confidence"] = library_display["Decision_Confidence"].map(
                lambda value: fmt.format_value(value, "{:,.0f}", "/100")
            )
            library_display = library_display.rename(columns={"Decision_Confidence": "Consistency"})
            library_display["Trend_Strength"] = library_display["Trend_Strength"].map(
                lambda value: fmt.format_value(value, "{:,.0f}")
            )
            library_display["Quality_Score"] = library_display["Quality_Score"].map(
                lambda value: fmt.format_value(value, "{:,.1f}")
            )
            library_display["Target Upside"] = library_display["Target Upside"].map(fmt.format_percent)
            library_display["Graham Discount"] = library_display["Graham Discount"].map(fmt.format_percent)
            library_display["DCF Upside"] = library_display["DCF Upside"].map(fmt.format_percent)
            st.dataframe(library_display, width="stretch")

            library_left, library_right = st.columns(2)
            with library_left:
                st.subheader("Sector Summary")
                ui.render_help_legend(
                    [
                        ("Avg Composite Score", const.ANALYSIS_HELP_TEXT["Avg Composite Score"]),
                        ("Avg Target Upside", const.ANALYSIS_HELP_TEXT["Avg Target Upside"]),
                        ("Avg DCF Upside", const.ANALYSIS_HELP_TEXT["Avg DCF Upside"]),
                    ]
                )
                sector_summary = (
                    filtered_library.groupby("Sector", dropna=False)
                    .agg(
                        Records=("Ticker", "count"),
                        Avg_Composite_Score=("Composite Score", "mean"),
                        Avg_Target_Upside=("Target Upside", "mean"),
                        Avg_DCF_Upside=("DCF Upside", "mean"),
                    )
                    .reset_index()
                    .sort_values(["Records", "Avg_Composite_Score"], ascending=[False, False])
                )
                sector_summary["Avg_Composite_Score"] = sector_summary["Avg_Composite_Score"].map(
                    lambda value: fmt.format_value(value, "{:,.1f}")
                )
                sector_summary["Avg_Target_Upside"] = sector_summary["Avg_Target_Upside"].map(fmt.format_percent)
                sector_summary["Avg_DCF_Upside"] = sector_summary["Avg_DCF_Upside"].map(fmt.format_percent)
                st.dataframe(sector_summary, width="stretch")

            with library_right:
                st.subheader("Top Conviction Names")
                ui.render_help_legend(
                    [
                        ("Composite Score", const.ANALYSIS_HELP_TEXT["Composite Score"]),
                        ("Target Upside", const.ANALYSIS_HELP_TEXT["Target Mean"]),
                        ("DCF Upside", const.ANALYSIS_HELP_TEXT["DCF Upside"]),
                        ("Freshness", const.ANALYSIS_HELP_TEXT["Freshness"]),
                    ]
                )
                conviction_table = filtered_library[
                    ["Ticker", "Verdict_Overall", "Composite Score", "Target Upside", "DCF Upside", "Freshness"]
                ].head(10).copy()
                conviction_table["Target Upside"] = conviction_table["Target Upside"].map(fmt.format_percent)
                conviction_table["DCF Upside"] = conviction_table["DCF Upside"].map(fmt.format_percent)
                st.dataframe(conviction_table, width="stretch")

with readme_tab:
    st.subheader("ReadMe / Usage")
    st.caption("Edit the README_USAGE_TEXT constant near the top of streamlit_app.py to customize this section.")
    if const.README_USAGE_TEXT.strip():
        st.markdown(const.README_USAGE_TEXT)
    else:
        st.text_area(
            "ReadMe / Usage Placeholder",
            value="",
            height=240,
            placeholder="Add your ReadMe / Usage copy in the README_USAGE_TEXT constant in streamlit_app.py.",
            disabled=True,
            label_visibility="collapsed",
        )

with changelog_tab:
    st.subheader("Changelog")
    st.caption("Recent updates to the stock model, portfolio engine, and research UI live here so the app stays inspectable over time.")

    changelog_metrics = st.columns(3)
    changelog_metrics[0].metric("Latest Logged Update", const.CHANGELOG_ENTRIES[0]["Date"])
    changelog_metrics[1].metric("Logged Changes", str(len(const.CHANGELOG_ENTRIES)))
    changelog_metrics[2].metric("App Version", const.APP_VERSION)

    st.dataframe(pd.DataFrame(const.CHANGELOG_ENTRIES), width="stretch")

    st.subheader("What Changed Most Recently")
    st.write("- The model now adds ten extra diagnostics such as trend strength, 52-week range context, volatility-adjusted momentum, quality score, dividend safety, valuation breadth, sentiment conviction, and explicit risk flags.")
    st.write("- The model now assigns each stock a primary type such as Growth, Value, Dividend, Cyclical, Defensive, Blue-Chip, size-based, or Speculative and uses that profile in verdict and backtest logic.")
    st.write("- The backtest now holds a core position during durable bullish regimes, exits later on deeper breakdowns, and reports win rate plus average closed-trade return.")
    st.write("- The Options tab now includes inline ? explanations for every slider and preset selector.")
    st.write("- Regime, consistency, and decision-note transparency remain visible across stock, compare, sensitivity, and library views.")

with methodology_tab:
    st.subheader("Methodology and Transparency")
    st.caption("This tab shows how the app forms verdicts, why the portfolio engine chooses certain weights, and what assumptions sit underneath the UI.")

    st.subheader("Model Flow")
    methodology_flow = pd.DataFrame(
        [
            {"Step": 1, "What Happens": "Download one year of price history plus company profile and news from Yahoo Finance."},
            {"Step": 2, "What Happens": "Score technical, fundamental, and peer-relative valuation layers. The DCF is optional and only runs when you create it manually."},
            {"Step": 3, "What Happens": "Classify market regime, measure engine agreement, and estimate decision consistency."},
            {"Step": 4, "What Happens": "Apply hold buffers and data-quality guardrails before publishing the final verdict."},
            {"Step": 5, "What Happens": "Store the full result with timestamp, assumption fingerprint, and quality stats in the shared library."},
        ]
    )
    st.dataframe(methodology_flow, width="stretch")

    methodology_col_1, methodology_col_2 = st.columns(2)
    with methodology_col_1:
        st.subheader("Single-Stock Engines")
        engine_framework = pd.DataFrame(
            [
                {
                    "Engine": "Technical",
                    "Uses": "RSI, MACD crossover and level, 50/200-day trend, trend tolerance bands, stretch limits, 1M and 1Y momentum",
                    "Strong Signals": "Healthy trend alignment, bullish MACD, supportive momentum, and controlled oversold reversals",
                },
                {
                    "Engine": "Fundamental",
                    "Uses": "ROE, profit margin, debt/equity, revenue growth, earnings growth, current ratio",
                    "Strong Signals": "High profitability, positive growth, sound liquidity, manageable leverage",
                },
                {
                    "Engine": "Valuation",
                    "Uses": "P/E, forward P/E, PEG, P/S, EV/EBITDA, P/B, five closest peers, Graham value, plus an optional manual SEC-based DCF snapshot",
                    "Strong Signals": "Positive earnings, cheaper-than-peer multiples, and discounts to fair-value anchors such as Graham value or a manual DCF snapshot",
                },
                {
                    "Engine": "Sentiment",
                    "Uses": "Company-related headlines, analyst recommendation labels, analyst depth, target mean price",
                    "Strong Signals": "The app now surfaces context only and leaves the interpretation to the user",
                },
            ]
        )
        st.dataframe(engine_framework, width="stretch")

    with methodology_col_2:
        st.subheader("Verdict Thresholds")
        verdict_table = pd.DataFrame(
            [
                {"Output": "Technical score", "Rule": ">= 4 strong buy, >= 2 buy, <= -2 sell, <= -4 strong sell"},
                {"Output": "Sentiment context", "Rule": "Displayed as reference only and not used as a directional score in the current build"},
                {
                    "Output": "Valuation verdict",
                    "Rule": (
                        f">= {model_settings['valuation_under_score_threshold']:.0f} undervalued, "
                        f">= {model_settings['valuation_fair_score_threshold']:.0f} fair value, otherwise overvalued"
                    ),
                },
                {
                    "Output": "Overall verdict",
                    "Rule": (
                        f"Base score thresholds are {model_settings['overall_strong_buy_threshold']:.0f} / "
                        f"{model_settings['overall_buy_threshold']:.0f} / "
                        f"{model_settings['overall_sell_threshold']:.0f} / "
                        f"{model_settings['overall_strong_sell_threshold']:.0f}; mixed regimes, low consistency, and low-quality data are pushed toward hold"
                    ),
                },
            ]
        )
        st.dataframe(verdict_table, width="stretch")

    st.subheader("Stock Type Framework")
    stock_type_framework = pd.DataFrame(
        [
            {"Type": "Growth Stocks", "How The Model Recognizes It": "Fast growth, premium multiples, low yield, and strong momentum", "Logic Tilt": "Trend persistence matters more than cheap valuation"},
            {"Type": "Value Stocks", "How The Model Recognizes It": "Undervaluation, discounted multiples, and at least stable fundamentals", "Logic Tilt": "Valuation and balance-sheet support matter more"},
            {"Type": "Dividend / Income Stocks", "How The Model Recognizes It": "Meaningful yield, payout support, income-heavy sectors, lower beta", "Logic Tilt": "Sustainability and steady compounding matter more"},
            {"Type": "Cyclical Stocks", "How The Model Recognizes It": "Economically sensitive sectors and bigger beta or cycle swings", "Logic Tilt": "Timing and regime confirmation matter more"},
            {"Type": "Defensive Stocks", "How The Model Recognizes It": "Resilient sectors, steadier beta, and stable fundamentals", "Logic Tilt": "The model tolerates slower upside and defends against over-trading"},
            {"Type": "Blue-Chip Stocks", "How The Model Recognizes It": "Large scale, quality metrics, broad coverage, and durable fundamentals", "Logic Tilt": "Quality durability gets extra room"},
            {"Type": "Small/Mid/Large-Cap", "How The Model Recognizes It": "Market capitalization bucket", "Logic Tilt": "Smaller caps require stronger confirmation; larger caps get more stability credit"},
            {"Type": "Speculative / Penny Stocks", "How The Model Recognizes It": "Tiny scale, low price, weak fundamentals, thin coverage, or extreme beta", "Logic Tilt": "Buy thresholds rise and conviction is capped"},
        ]
    )
    st.dataframe(stock_type_framework, width="stretch")

    st.subheader("Refinement Layer")
    refinement_df = pd.DataFrame(
        [
            {"Refinement": 1, "What Changed": "Trend Strength", "Purpose": "Uses SMA structure plus 1Y momentum as a continuous trend quality signal."},
            {"Refinement": 2, "What Changed": "52-Week Range Context", "Purpose": "Tracks whether price is breaking out, mid-range, or stuck near lows."},
            {"Refinement": 3, "What Changed": "Volatility-Adjusted Momentum", "Purpose": "Rewards momentum that is strong relative to realized volatility instead of raw price change alone."},
            {"Refinement": 4, "What Changed": "Quality Score", "Purpose": "Combines profitability, leverage, liquidity, and growth consistency into a cleaner business-quality signal."},
            {"Refinement": 5, "What Changed": "Dividend Safety Score", "Purpose": "Checks whether income stocks appear to have a more sustainable payout profile."},
            {"Refinement": 6, "What Changed": "Peer-Relative Valuation", "Purpose": "Anchors valuation to the five closest peers first and uses sector benchmarks only as a fallback."},
            {"Refinement": 7, "What Changed": "Context Depth", "Purpose": "Tracks how much analyst and headline context was available without turning that context into a directional sentiment score."},
            {"Refinement": 8, "What Changed": "Risk Flags", "Purpose": "Collects visible red flags like negative EPS, high debt, weak liquidity, high volatility, and speculation."},
            {"Refinement": 9, "What Changed": "Dynamic Engine Weights", "Purpose": "Lets Growth, Value, Income, Cyclical, and Speculative names use different engine mixes."},
            {"Refinement": 10, "What Changed": "Event Study and Trading Friction", "Purpose": "Adds recent event-reaction context to fundamentals and deducts transaction-cost estimates from the backtest."},
        ]
    )
    st.dataframe(refinement_df, width="stretch")

    st.subheader("Decision Guardrails")
    guardrail_df = pd.DataFrame(
        [
            {"Guardrail": "Trend Tolerance", "Purpose": "Avoids flipping trend signals on tiny moves around the moving averages."},
            {"Guardrail": "Stretch Limit", "Purpose": "Penalizes overextended rallies and recognizes washed-out rebounds before chasing price."},
            {"Guardrail": "Hold Buffer", "Purpose": "Makes mixed-engine or transition regimes require extra evidence before becoming directional."},
            {"Guardrail": "Consistency Floor", "Purpose": "Downgrades weakly aligned Buy or Sell calls back toward Hold."},
            {"Guardrail": "Data Quality Check", "Purpose": "Reduces conviction when too many important metrics are missing."},
        ]
    )
    st.dataframe(guardrail_df, width="stretch")

    st.subheader("Portfolio Workflow")
    portfolio_workflow = pd.DataFrame(
        [
            {"Step": 1, "What Happens": "Download adjusted close history for the chosen tickers and benchmark."},
            {"Step": 2, "What Happens": "Convert prices to returns and align all series on the same dates."},
            {"Step": 3, "What Happens": "Simulate capped portfolios and compute return, volatility, Sharpe, Sortino, Treynor, and beta."},
            {"Step": 4, "What Happens": "Trace the efficient frontier and identify the max-Sharpe tangent portfolio."},
            {"Step": 5, "What Happens": "Translate the tangent weights into practical roles, sector exposure, and concentration notes."},
        ]
    )
    st.dataframe(portfolio_workflow, width="stretch")

    methodology_col_3, methodology_col_4 = st.columns(2)
    with methodology_col_3:
        st.subheader("Peer Valuation Workflow")
        peer_workflow_df = pd.DataFrame(
            [
                {"Step": 1, "What Happens": "Start with the company being analyzed and pull its sector, industry, size, growth, profitability, leverage, and beta fields."},
                {"Step": 2, "What Happens": f"Search a cached universe for the {const.PEER_GROUP_SIZE} closest companies using those characteristics."},
                {"Step": 3, "What Happens": "Average the peer valuation multiples and use those averages as the main comparison set."},
                {"Step": 4, "What Happens": "If the peer set is too thin or missing too many usable metrics, fall back to scaled sector benchmarks."},
                {"Step": 5, "What Happens": "Keep Graham value separate and move DCF to a manual lab so cash-flow work stays optional and adjustable."},
            ]
        )
        st.dataframe(peer_workflow_df, width="stretch")

    with methodology_col_4:
        st.subheader("Current Model Assumptions")
        library_snapshot = prep.prepare_analysis_dataframe(db.get_all_analyses())
        assumptions_df = pd.DataFrame(
            [
                {
                    "Setting": "Storage Mode",
                    "Value": (
                        "Postgres"
                        if db.storage_backend == "postgres"
                        else ("Persistent SQLite" if db.uses_persistent_storage else "In-memory")
                    ),
                },
                {"Setting": "Database Path", "Value": db.storage_label},
                {"Setting": "Active Profile", "Value": active_preset_name},
                {"Setting": "Assumption Fingerprint", "Value": active_assumption_fingerprint},
                {"Setting": "Trading Days per Year", "Value": int(model_settings["trading_days_per_year"])},
                {"Setting": "Default Benchmark", "Value": const.DEFAULT_BENCHMARK_TICKER},
                {"Setting": "Default Portfolio Universe", "Value": const.DEFAULT_PORTFOLIO_TICKERS},
                {
                    "Setting": "Engine Weights",
                    "Value": (
                        f"T {model_settings['weight_technical']:.1f} | "
                        f"F {model_settings['weight_fundamental']:.1f} | "
                        f"V {model_settings['weight_valuation']:.1f} | "
                        f"S {model_settings['weight_sentiment']:.1f}"
                    ),
                },
                {
                    "Setting": "RSI Band",
                    "Value": f"{int(model_settings['tech_rsi_oversold'])} / {int(model_settings['tech_rsi_overbought'])}",
                },
                {
                    "Setting": "Trend Tolerance",
                    "Value": f"{model_settings['tech_trend_tolerance'] * 100:.1f}%",
                },
                {
                    "Setting": "Extension Limit",
                    "Value": f"{model_settings['tech_extension_limit'] * 100:.1f}%",
                },
                {
                    "Setting": "Hold Buffer",
                    "Value": f"{model_settings['decision_hold_buffer']:.1f}",
                },
                {
                    "Setting": "Consistency Floor",
                    "Value": f"{model_settings['decision_min_confidence']:.0f}/100",
                },
                {
                    "Setting": "Backtest Cooldown",
                    "Value": f"{int(round(model_settings['backtest_cooldown_days']))} days",
                },
                {
                    "Setting": "Peer Group Size",
                    "Value": const.PEER_GROUP_SIZE,
                },
                {
                    "Setting": "Fallback Benchmark Scale",
                    "Value": f"{model_settings['valuation_benchmark_scale']:.2f}x",
                },
                {
                    "Setting": "Assumption Drift vs Defaults",
                    "Value": f"{settings.calculate_assumption_drift(model_settings):.1f}%",
                },
                {"Setting": "Event Study Max Events", "Value": 5},
                {"Setting": "Backtest Transaction Cost", "Value": f"{model_settings['backtest_transaction_cost_bps']:.1f} bps"},
                {"Setting": "Cached Analyses in Library", "Value": len(library_snapshot)},
            ]
        )
        assumptions_df["Value"] = assumptions_df["Value"].map(str)
        st.dataframe(assumptions_df, width="stretch")

with options_tab:
    st.subheader("Model Options")
    st.caption("Tune the main assumptions with guardrails so the model stays interpretable and does not swing wildly from small changes.")
    st.caption("Changes apply to new stock analyses, refreshed comparisons, and new portfolio runs. Cached rows remain as previously analyzed until you rerun them.")
    st.caption(f"Active profile: {active_preset_name} | Fingerprint: {active_assumption_fingerprint}")

    preset_catalog = settings.get_model_presets()
    preset_names = list(preset_catalog.keys())
    preset_index = preset_names.index(active_preset_name) if active_preset_name in preset_names else preset_names.index(settings.get_default_preset_name())
    preset_col_1, preset_col_2 = st.columns([3, 1])
    with preset_col_1:
        preset_selection = st.selectbox(
            "Load Preset",
            preset_names,
            index=preset_index,
            help=const.OPTIONS_HELP_TEXT["load_preset"],
        )
        st.caption(const.PRESET_DESCRIPTIONS.get(preset_selection, ""))
    with preset_col_2:
        st.write("")
        st.write("")
        if st.button("Apply Preset", width="stretch"):
            st.session_state.model_settings = preset_catalog[preset_selection].copy()
            st.session_state.model_preset_name = preset_selection
            st.session_state.options_feedback = {
                "message": f"{preset_selection} preset loaded.",
                "notes": [
                    const.PRESET_DESCRIPTIONS.get(preset_selection, ""),
                    "You can still fine-tune any slider below and save the result as a custom assumption set.",
                ],
            }
            st.rerun()

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

    feedback = st.session_state.pop("options_feedback", None)
    if feedback:
        st.success(feedback["message"])
        for note in feedback.get("notes", []):
            st.caption(note)

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

for hidden_placeholder in hidden_senior_placeholders:
    try:
        hidden_placeholder.empty()
    except Exception:
        pass

try:
    portfolio_tab.empty()
except Exception:
    pass

