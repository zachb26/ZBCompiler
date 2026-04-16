# -*- coding: utf-8 -*-
import datetime

import pandas as pd
import streamlit as st

import constants as const
import exports
import utils_fmt as fmt
import utils_ui as ui
import analysis_prep as prep


def render_library_view(db, startup_refresh_summary):
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
