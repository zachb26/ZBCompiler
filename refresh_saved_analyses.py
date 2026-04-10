import argparse
import csv
import datetime
import os
import time
from pathlib import Path

from seed_universe import load_engine_module


def write_report(report_rows, report_path):
    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["Ticker", "Status", "Detail", "Verdict", "Last_Updated"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)


def main():
    parser = argparse.ArgumentParser(description="Refresh stale saved analyses without doing the work on Streamlit page load.")
    parser.add_argument("--db-path", default="stocks_data.db", help="SQLite database path when not using Postgres.")
    parser.add_argument(
        "--database-url",
        default=os.environ.get("STOCKS_DATABASE_URL", os.environ.get("DATABASE_URL", "")).strip(),
        help="Postgres URL. When provided, it overrides --db-path.",
    )
    parser.add_argument("--preset", default="Balanced", help="Model preset: Balanced, Conservative, or Aggressive.")
    parser.add_argument("--stale-after-hours", type=float, default=12.0, help="Refresh rows older than this many hours.")
    parser.add_argument("--sleep-seconds", type=float, default=0.2, help="Delay between ticker requests.")
    parser.add_argument("--failure-streak-limit", type=int, default=6, help="Pause after this many consecutive failures.")
    parser.add_argument("--max-tickers", type=int, default=0, help="Optional cap on how many stale tickers to process.")
    parser.add_argument("--report-path", default="refresh_saved_analyses_report.csv", help="CSV report output path.")
    args = parser.parse_args()

    app_path = Path(__file__).resolve().parent / "streamlit_app.py"
    database_url = args.database_url.strip()
    sqlite_path = Path(args.db_path).expanduser().resolve()

    module = load_engine_module(
        app_path,
        db_path=None if database_url else sqlite_path,
        database_url=database_url or None,
        module_name="stock_engine_refresh_app",
    )

    preset_catalog = module.get_model_presets()
    if args.preset not in preset_catalog:
        valid = ", ".join(sorted(preset_catalog))
        raise SystemExit(f"Unknown preset '{args.preset}'. Valid options: {valid}")

    storage_target = database_url or sqlite_path
    db = module.DatabaseManager(storage_target)
    analyst = module.StockAnalyst(db)
    settings = preset_catalog[args.preset]
    tickers = module.collect_stale_analysis_tickers(db, stale_after_hours=args.stale_after_hours)

    if args.max_tickers > 0:
        tickers = tickers[: args.max_tickers]

    report_rows = []
    updated_count = 0
    failed_count = 0
    skipped_count = 0
    failure_streak = 0
    started_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"Refresh started at {started_at}")
    print(f"Refreshing {len(tickers)} stale saved analysis row(s) in {db.storage_label} using the {args.preset} preset.")

    if not tickers:
        write_report(report_rows, args.report_path)
        print("No stale saved analyses found.")
        print(f"Report written to {Path(args.report_path).resolve()}")
        return

    for idx, ticker in enumerate(tickers, start=1):
        status = "updated"
        detail = ""
        verdict = ""
        last_updated = ""

        try:
            record = analyst.analyze(ticker, settings=settings, persist=True)
        except Exception as exc:
            record = None
            analyst.last_error = module.summarize_fetch_error(exc)

        if record is None:
            failed_count += 1
            failure_streak += 1
            status = "failed"
            detail = analyst.last_error or "Unknown fetch failure."
            print(f"[{idx}/{len(tickers)}] {ticker}: failed - {detail}")
        else:
            failure_streak = 0
            updated_count += 1
            detail = analyst.last_error or ""
            verdict = record.get("Verdict_Overall", "")
            last_updated = record.get("Last_Updated", "")
            if detail and "showing the most recent saved analysis instead" in detail.lower():
                status = "reused_saved"
                skipped_count += 1
            elif detail:
                status = "updated_with_fallback"
            print(f"[{idx}/{len(tickers)}] {ticker}: {status} -> {verdict}")

        report_rows.append(
            {
                "Ticker": ticker,
                "Status": status,
                "Detail": detail,
                "Verdict": verdict,
                "Last_Updated": last_updated,
            }
        )

        if args.failure_streak_limit > 0 and failure_streak >= args.failure_streak_limit:
            print("")
            print("Refresh paused after repeated upstream fetch failures. Saved rows remain available.")
            break

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    write_report(report_rows, args.report_path)
    print("")
    print(
        f"Completed. Updated: {updated_count} | Reused saved: {skipped_count} | Failed: {failed_count} | "
        f"Report: {Path(args.report_path).resolve()}"
    )


if __name__ == "__main__":
    main()
