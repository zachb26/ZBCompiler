# -*- coding: utf-8 -*-
import datetime

import pandas as pd
import streamlit as st

import constants as const
import fetch
import utils_fmt as fmt
import utils_news as news
import utils_time as tutil
import utils_ui as ui
import analysis_prep as prep


def render_macro_regime_panel(macro_data):
    """Render a 4-card macro backdrop panel above the sector view."""

    def _card(label, entry, low_is_good, good_thresh, bad_thresh, fmt_fn, note, help_text):
        val = (entry or {}).get("value")
        pct = (entry or {}).get("pct_rank")
        error = (entry or {}).get("error")
        if val is None:
            return {"label": label, "value": "N/A", "note": error or "Unavailable.", "tone": "neutral", "help": help_text}
        display = fmt_fn(val)
        if pct is not None:
            display += f" ({pct:.0f}p)"
        if low_is_good:
            tone = "good" if val <= good_thresh else ("bad" if val >= bad_thresh else "neutral")
        else:
            tone = "good" if val >= good_thresh else ("bad" if val <= bad_thresh else "neutral")
        return {"label": label, "value": display, "note": note, "tone": tone, "help": help_text}

    cards = [
        _card(
            "VIX", macro_data.get("VIX"),
            low_is_good=True, good_thresh=18, bad_thresh=25,
            fmt_fn=lambda v: f"{v:.1f}",
            note="Implied equity volatility. >25 = elevated stress.",
            help_text="CBOE Volatility Index. Above 25 historically signals risk-off. Percentile rank over the prior 52 weeks.",
        ),
        _card(
            "2s10s Spread", macro_data.get("2s10s"),
            low_is_good=False, good_thresh=0.10, bad_thresh=-0.25,
            fmt_fn=lambda v: f"{v:+.2f}%",
            note="10Y minus 2Y. Negative = inverted curve.",
            help_text="Treasury 10Y minus 2Y spread. Sustained inversion below -0.25% is a historically reliable recession leading indicator.",
        ),
        _card(
            "HY Spread (bp)", macro_data.get("HY_OAS"),
            low_is_good=True, good_thresh=300, bad_thresh=450,
            fmt_fn=lambda v: f"{v:.0f}bp",
            note="High-yield OAS. >450bp = risk-off credit.",
            help_text="ICE BofA US High Yield Option-Adjusted Spread. Widening above 450bp signals deteriorating risk appetite.",
        ),
        _card(
            "DXY", macro_data.get("DXY"),
            low_is_good=False, good_thresh=float("inf"), bad_thresh=float("-inf"),
            fmt_fn=lambda v: f"{v:.1f}",
            note="Trade-weighted dollar index.",
            help_text="Trade Weighted U.S. Dollar Index (Broad). A rising DXY tends to tighten global financial conditions and pressure EM and commodity sectors.",
        ),
    ]
    ui.render_analysis_signal_cards(cards, columns=4)

    vix_val = (macro_data.get("VIX") or {}).get("value")
    spread_val = (macro_data.get("2s10s") or {}).get("value")
    hy_val = (macro_data.get("HY_OAS") or {}).get("value")

    risk_off = sum([
        vix_val is not None and vix_val > 25,
        spread_val is not None and spread_val < -0.25,
        hy_val is not None and hy_val > 450,
    ])
    warn = sum([
        vix_val is not None and 20 < vix_val <= 25,
        spread_val is not None and -0.25 <= spread_val < 0,
        hy_val is not None and 380 < hy_val <= 450,
    ])

    if risk_off >= 2:
        st.warning(
            "Macro: Risk-off — multiple elevated stress signals detected. "
            "Defensives and quality have historically led in this backdrop; consider underweighting cyclicals and high-beta names."
        )
    elif risk_off == 1 or warn >= 2:
        st.info(
            "Macro: Mixed signals — at least one stress indicator is elevated. "
            "Monitor for follow-through before making aggressive sector rotation bets."
        )
    else:
        st.success("Macro: No elevated stress signals. Current readings are consistent with a constructive risk backdrop.")


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


def _macro_signal(value, thresholds: dict, invert: bool = False) -> str:
    """Return 'red', 'yellow', or 'green' based on caution/ok thresholds."""
    if value is None:
        return "neutral"
    caution = thresholds["caution"]
    ok = thresholds["ok"]
    if not invert:
        if value >= caution:
            return "bad"
        if value >= ok:
            return "neutral"
        return "good"
    else:
        if value <= caution:
            return "bad"
        if value <= ok:
            return "neutral"
        return "good"


def _render_macro_regime_panel() -> None:
    """Fetch and display the macro regime overlay at the top of the Sector tab."""
    with st.expander("Macro Regime", expanded=True):
        with st.spinner("Loading macro indicators..."):
            try:
                m = fetch.fetch_macro_indicators()
            except Exception:
                st.warning("Macro indicators are unavailable right now.")
                return

        thresholds = const.MACRO_THRESHOLDS

        def fmt_val(v, decimals=2, suffix=""):
            return f"{v:.{decimals}f}{suffix}" if v is not None else "N/A"

        two_ten = m.get("two_ten_spread")
        hy_oas = m.get("hy_oas_bps")
        vix = m.get("vix")
        vix_ratio = m.get("vix_ratio")
        dxy = m.get("dxy")

        tone_map = {"good": "🟢", "neutral": "🟡", "bad": "🔴"}

        two_ten_tone = _macro_signal(two_ten, thresholds["two_ten_spread"], invert=True)
        hy_tone = _macro_signal(hy_oas, thresholds["hy_oas_bps"], invert=False)
        vix_tone = _macro_signal(vix, thresholds["vix"], invert=False)
        ratio_tone = _macro_signal(vix_ratio, thresholds["vix_ratio"], invert=False)
        dxy_tone = _macro_signal(dxy, thresholds["dxy_level"], invert=False)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(
                f"{tone_map[two_ten_tone]} 2s10s Spread",
                fmt_val(two_ten, 2, "%"),
                help="10Y minus 2Y Treasury yield. Negative = inverted curve (recession signal).",
            )
        with col2:
            st.metric(
                f"{tone_map[hy_tone]} HY OAS (bps)",
                fmt_val(hy_oas, 0),
                help="ICE BofA HY option-adjusted spread. Rising = credit stress / risk-off.",
            )
        with col3:
            st.metric(
                f"{tone_map[vix_tone]} VIX",
                fmt_val(vix, 1),
                help="CBOE Volatility Index. >30 = elevated fear, >20 = caution.",
            )
        with col4:
            st.metric(
                f"{tone_map[ratio_tone]} VIX / VIX3M",
                fmt_val(vix_ratio, 2),
                help="VIX divided by 3-month VIX. >1.0 = near-term vol backwardation (acute stress).",
            )
        with col5:
            st.metric(
                f"{tone_map[dxy_tone]} DXY",
                fmt_val(dxy, 1),
                help="US dollar index. High dollar can pressure EM and commodity sectors.",
            )

        red_signals = [
            label for label, tone in (
                ("yield curve inversion", two_ten_tone),
                ("HY credit stress", hy_tone),
                ("elevated VIX", vix_tone),
                ("vol backwardation", ratio_tone),
                ("strong dollar headwind", dxy_tone),
            )
            if tone == "bad"
        ]
        yellow_signals = [
            label for label, tone in (
                ("flat curve", two_ten_tone),
                ("elevated spreads", hy_tone),
                ("VIX caution zone", vix_tone),
                ("vol term structure flat", ratio_tone),
                ("USD elevated", dxy_tone),
            )
            if tone == "neutral"
        ]

        if red_signals:
            st.caption(f"Macro backdrop: Stress signals — {', '.join(red_signals)}.")
        elif yellow_signals:
            st.caption(f"Macro backdrop: Mixed — watch {', '.join(yellow_signals)}.")
        else:
            st.caption("Macro backdrop: Risk-On — yield curve normal, credit benign, vol calm.")


def render_sector_leader_view(db):
    st.subheader("Sector Leader")
    st.caption("Compare all tracked names inside one sector, review relative strength and valuation side-by-side, and prepare a meeting-ready weekly briefing.")

    _render_macro_regime_panel()
    st.divider()

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
