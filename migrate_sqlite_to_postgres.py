import argparse
import os
import sqlite3
from pathlib import Path

import pandas as pd

from seed_universe import load_engine_module


def normalize_scalar(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except Exception:
            pass
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def build_record(row, allowed_columns):
    record = {}
    for column_name in allowed_columns:
        if column_name not in row.index:
            continue
        record[column_name] = normalize_scalar(row[column_name])

    ticker = str(record.get("Ticker") or "").strip().upper()
    if not ticker:
        return None
    record["Ticker"] = ticker
    return record


def main():
    parser = argparse.ArgumentParser(description="Copy saved analyses from SQLite into Postgres.")
    parser.add_argument("--sqlite-path", default="stocks_data.db", help="Path to the source SQLite database.")
    parser.add_argument(
        "--postgres-url",
        default="",
        help="Target Postgres URL. Falls back to STOCKS_DATABASE_URL or DATABASE_URL when omitted.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on how many rows to migrate.")
    args = parser.parse_args()

    postgres_url = (
        args.postgres_url
        or ""
    ).strip()
    if not postgres_url:
        postgres_url = os.environ.get("STOCKS_DATABASE_URL", os.environ.get("DATABASE_URL", "")).strip()
    if not postgres_url:
        raise SystemExit("Provide --postgres-url or set STOCKS_DATABASE_URL / DATABASE_URL.")

    sqlite_path = Path(args.sqlite_path).expanduser().resolve()
    if not sqlite_path.exists():
        raise SystemExit(f"SQLite source not found: {sqlite_path}")

    app_path = Path(__file__).resolve().parent / "streamlit_app.py"
    module = load_engine_module(
        app_path,
        database_url=postgres_url,
        module_name="stock_engine_migrate_app",
    )

    source_conn = sqlite3.connect(str(sqlite_path), timeout=30, check_same_thread=False)
    try:
        source_frame = pd.read_sql_query("SELECT * FROM analysis", source_conn)
    finally:
        source_conn.close()

    if source_frame.empty:
        print(f"No rows found in {sqlite_path}. Nothing to migrate.")
        return

    if args.limit > 0:
        source_frame = source_frame.head(args.limit)

    target_db = module.DatabaseManager(postgres_url)
    allowed_columns = list(module.ANALYSIS_COLUMNS.keys())

    total_rows = len(source_frame)
    migrated_count = 0
    skipped_count = 0
    failed_count = 0

    print(f"Migrating {total_rows} row(s) from {sqlite_path} to {target_db.storage_label}.")

    for idx, (_, row) in enumerate(source_frame.iterrows(), start=1):
        record = build_record(row, allowed_columns)
        if not record:
            skipped_count += 1
            print(f"[{idx}/{total_rows}] skipped row with no ticker")
            continue

        try:
            target_db.save_analysis(record)
            migrated_count += 1
            if idx == 1 or idx % 25 == 0 or idx == total_rows:
                print(f"[{idx}/{total_rows}] migrated {record['Ticker']}")
        except Exception as exc:
            failed_count += 1
            print(f"[{idx}/{total_rows}] failed {record['Ticker']}: {exc}")

    target_total = len(target_db.get_all_analyses())
    print("")
    print(f"Completed. Migrated: {migrated_count} | Skipped: {skipped_count} | Failed: {failed_count}")
    print(f"Target database now contains {target_total} saved analysis row(s).")


if __name__ == "__main__":
    main()
