# -*- coding: utf-8 -*-
import datetime
import logging
import time

import streamlit as st

logger = logging.getLogger(__name__)

import constants as const
import fetch
import utils_time as tutil


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
    from analyst import StockAnalyst

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
                        logger.warning("Startup refresh failed for %s: %s", ticker, exc)

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
                logger.error("Startup refresh run crashed: %s", exc)
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
