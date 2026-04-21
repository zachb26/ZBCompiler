# -*- coding: utf-8 -*-
import logging
import os

import streamlit as st

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

import constants as const
import settings
import sec_ai as sec

from database import get_database_manager
from analyst import StockAnalyst, PortfolioAnalyst
from services.startup_refresh import refresh_saved_analyses_on_launch
from ui.auth import render_password_gate
from views.new_analyst import render_new_analyst_view
from views.comparison import render_comparison_view
from views.senior_analyst import render_single_stock_view
from views.sensitivity import render_sensitivity_view
from views.backtest_view import render_backtest_view
from views.library import render_library_view
from views.readme import render_readme_view
from views.changelog import render_changelog_view
from views.methodology import render_methodology_view
from views.options import render_options_view
from views.sector_leader import render_sector_leader_view
from views.portfolio_manager import render_portfolio_manager_view
from views.portfolio_builder import render_portfolio_builder_view


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

st.image("osig_logo.png", width=300)
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

with st.sidebar:
    st.header("Model Preset")
    preset_catalog = settings.get_model_presets()
    preset_names = list(preset_catalog.keys())
    preset_index = preset_names.index(active_preset_name) if active_preset_name in preset_names else preset_names.index(settings.get_default_preset_name())
    sidebar_preset = st.selectbox("Preset", preset_names, index=preset_index, label_visibility="collapsed")
    st.caption(const.PRESET_DESCRIPTIONS.get(sidebar_preset, ""))
    if st.button("Apply Preset", type="primary", use_container_width=True):
        st.session_state.model_settings = preset_catalog[sidebar_preset].copy()
        st.session_state.model_preset_name = sidebar_preset
        st.session_state.options_feedback = {
            "message": f"{sidebar_preset} preset loaded.",
            "notes": [],
        }
        st.rerun()
    st.caption(f"Active: **{active_preset_name}**")
    st.divider()
    st.caption("Full model controls and sliders are in Senior Analyst → Controls.")

quick_links_tab, new_analyst_tab, analyst_senior_tab, sector_leader_tab, portfolio_manager_tab, methodology_tab, readme_tab = st.tabs(
    ["Quick Links", "New Analyst", "Senior Analyst", "Sector Leader", "Portfolio Manager", "Methodology", "ReadMe"]
)

_sa_locked = not st.session_state.get("senior_analyst_authenticated", False)
_pm_locked = not st.session_state.get("portfolio_manager_authenticated", False)
_locked_selectors = []
if _sa_locked:
    _locked_selectors.append("div[data-baseweb='tab-list'] button[role='tab']:nth-child(3)")
if _pm_locked:
    _locked_selectors.append("div[data-baseweb='tab-list'] button[role='tab']:nth-child(5)")
if _locked_selectors:
    _locked_css = ", ".join(_locked_selectors)
    st.markdown(
        f"<style>{_locked_css} {{ opacity: 0.4 !important; color: #888 !important; cursor: not-allowed !important; }}</style>",
        unsafe_allow_html=True,
    )

# ── VOTING LINK — change this URL to update the active vote ──────────────────
VOTING_LINK = "https://docs.google.com/forms/d/e/1FAIpQLScXEFap4fq51ycL54ugLasHjt0U2dsJuJpSEmGfDsP8xSVkMg/viewform"
# ─────────────────────────────────────────────────────────────────────────────

with quick_links_tab:
    st.header("Quick Links")
    st.markdown("Essential resources for OSIG members.")
    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Box")
        st.markdown("Club files, research, and documents.")
        st.link_button("Open Box", "https://oregonstate.box.com/s/1gred7wfd4zrigao6e23dht8l7y4z50j", use_container_width=True)

        st.markdown("")
        st.subheader("Yahoo Finance")
        st.markdown("Quotes, news, and market data.")
        st.link_button("Open Yahoo Finance", "https://finance.yahoo.com", use_container_width=True)

    with col2:
        st.subheader("Slack")
        st.markdown("Team communication and channels.")
        st.link_button("Open Slack", "https://osu-osig.slack.com", use_container_width=True)

        st.markdown("")
        st.subheader("S&P 500 Heat Map")
        st.markdown("Live sector and market cap heat map.")
        st.link_button("Open Heat Map", "https://finviz.com/map.ashx", use_container_width=True)

    with col3:
        st.subheader("Current Vote")
        st.markdown("Active club voting thread on Slack.")
        st.link_button("Go to Vote", VOTING_LINK, use_container_width=True, type="primary")

with new_analyst_tab:
    analyst_new_tab, compare_tab = st.tabs(["Analysis", "Comparison"])
    with analyst_new_tab:
        render_new_analyst_view(db, bot)
    with compare_tab:
        render_comparison_view(db, bot, model_settings, active_preset_name, active_assumption_fingerprint)

with analyst_senior_tab:
    senior_analyst_tools_enabled = render_password_gate(
        "senior_analyst_authenticated",
        const.SENIOR_ANALYST_PASSWORD_SECRET,
        "Senior Analyst Access",
        "Unlock the full analyst toolkit, including valuation labs, peer analysis, SEC filing automation, backtesting, and model controls.",
        "Unlock Senior Analyst",
    )
    if senior_analyst_tools_enabled:
        stock_tab, ai_reports_tab, sensitivity_tab, backtest_tab, library_tab, options_tab, senior_reference_tab = st.tabs(
            ["Single Stock", "AI Reports", "Sensitivity", "Backtest", "Library", "Controls", "Changelog"]
        )
        with stock_tab:
            render_single_stock_view(db, bot, model_settings, active_assumption_fingerprint)
        with ai_reports_tab:
            sec.render_ai_reports_tab(db)
        with sensitivity_tab:
            render_sensitivity_view(bot, model_settings, active_preset_name, active_assumption_fingerprint)
        with backtest_tab:
            render_backtest_view(db, model_settings, active_preset_name, active_assumption_fingerprint)
        with library_tab:
            render_library_view(db, startup_refresh_summary)
        with options_tab:
            render_options_view(model_settings, active_preset_name, active_assumption_fingerprint)
        with senior_reference_tab:
            render_changelog_view()

with sector_leader_tab:
    render_sector_leader_view(db)

with portfolio_manager_tab:
    portfolio_main_tab, portfolio_builder_tab = st.tabs(["Portfolio Manager", "Portfolio Builder"])
    with portfolio_main_tab:
        render_portfolio_manager_view(db, portfolio_bot, active_preset_name, active_assumption_fingerprint)
    with portfolio_builder_tab:
        render_portfolio_builder_view(portfolio_bot, active_preset_name, active_assumption_fingerprint)

with methodology_tab:
    render_methodology_view(db, model_settings, active_preset_name, active_assumption_fingerprint)

with readme_tab:
    render_readme_view()
