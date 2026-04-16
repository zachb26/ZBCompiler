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
