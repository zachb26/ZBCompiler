import argparse
import csv
import datetime
import importlib.util
import os
import sys
import time
import types
from pathlib import Path

import pandas as pd


class SessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


class DummyBlock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def markdown(self, *args, **kwargs):
        return None

    def empty(self):
        return DummyBlock()

    def container(self, *args, **kwargs):
        return DummyBlock()

    def form(self, *args, **kwargs):
        return DummyBlock()

    def spinner(self, *args, **kwargs):
        return DummyBlock()

    def expander(self, *args, **kwargs):
        return DummyBlock()

    def popover(self, *args, **kwargs):
        return DummyBlock()

    def columns(self, spec):
        count = spec if isinstance(spec, int) else len(spec)
        return [DummyBlock() for _ in range(count)]

    def tabs(self, labels):
        return [DummyBlock() for _ in labels]

    def title(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def subheader(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def success(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None

    def divider(self, *args, **kwargs):
        return None

    def dataframe(self, *args, **kwargs):
        return None

    def metric(self, *args, **kwargs):
        return None

    def vega_lite_chart(self, *args, **kwargs):
        return None

    def line_chart(self, *args, **kwargs):
        return None

    def area_chart(self, *args, **kwargs):
        return None

    def bar_chart(self, *args, **kwargs):
        return None

    def table(self, *args, **kwargs):
        return None

    def json(self, *args, **kwargs):
        return None

    def code(self, *args, **kwargs):
        return None

    def badge(self, *args, **kwargs):
        return None

    def download_button(self, *args, **kwargs):
        return False

    def rerun(self):
        return None

    def text_input(self, label, value="", *args, **kwargs):
        return kwargs.get("value", value)

    def text_area(self, label, value="", *args, **kwargs):
        return kwargs.get("value", value)

    def checkbox(self, label, value=False, *args, **kwargs):
        return kwargs.get("value", value)

    def toggle(self, label, value=False, *args, **kwargs):
        return kwargs.get("value", value)

    def button(self, *args, **kwargs):
        return False

    def form_submit_button(self, *args, **kwargs):
        return False

    def selectbox(self, label, options, index=0, *args, **kwargs):
        if not options:
            return None
        chosen_index = kwargs.get("index", index)
        chosen_index = min(max(int(chosen_index), 0), len(options) - 1)
        return options[chosen_index]

    def radio(self, label, options, index=0, *args, **kwargs):
        if not options:
            return None
        chosen_index = kwargs.get("index", index)
        chosen_index = min(max(int(chosen_index), 0), len(options) - 1)
        return options[chosen_index]

    def segmented_control(self, label, options, selection_mode="single", default=None, *args, **kwargs):
        if default is not None:
            return default
        if selection_mode == "multi":
            return []
        if not options:
            return None
        return options[0]

    def select_slider(self, label, options=None, value=None, *args, **kwargs):
        if value is not None:
            return value
        if options:
            return options[0]
        return None

    def number_input(self, label, value=0, *args, **kwargs):
        return kwargs.get("value", value)

    def slider(self, label, min_value=None, max_value=None, value=None, *args, **kwargs):
        return kwargs.get("value", value)

    def multiselect(self, label, options, default=None, *args, **kwargs):
        return kwargs.get("default", default if default is not None else [])

    def date_input(self, label, value=None, *args, **kwargs):
        return kwargs.get("value", value)

    def file_uploader(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        def _dummy_method(*args, **kwargs):
            return None

        return _dummy_method


class DummyStreamlit(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = SessionState()

    def cache_resource(self, func=None, **kwargs):
        if func is None:
            return lambda wrapped: wrapped
        return func

    def set_page_config(self, *args, **kwargs):
        return None

    def empty(self):
        return DummyBlock()

    def container(self, *args, **kwargs):
        return DummyBlock()

    def columns(self, spec):
        count = spec if isinstance(spec, int) else len(spec)
        return [DummyBlock() for _ in range(count)]

    def tabs(self, labels):
        return [DummyBlock() for _ in labels]

    def form(self, *args, **kwargs):
        return DummyBlock()

    def spinner(self, *args, **kwargs):
        return DummyBlock()

    def expander(self, *args, **kwargs):
        return DummyBlock()

    def popover(self, *args, **kwargs):
        return DummyBlock()

    def title(self, *args, **kwargs):
        return None

    def markdown(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def subheader(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def success(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None

    def divider(self, *args, **kwargs):
        return None

    def dataframe(self, *args, **kwargs):
        return None

    def metric(self, *args, **kwargs):
        return None

    def vega_lite_chart(self, *args, **kwargs):
        return None

    def line_chart(self, *args, **kwargs):
        return None

    def area_chart(self, *args, **kwargs):
        return None

    def bar_chart(self, *args, **kwargs):
        return None

    def table(self, *args, **kwargs):
        return None

    def json(self, *args, **kwargs):
        return None

    def code(self, *args, **kwargs):
        return None

    def badge(self, *args, **kwargs):
        return None

    def download_button(self, *args, **kwargs):
        return False

    def rerun(self):
        return None

    def text_input(self, label, value="", *args, **kwargs):
        return kwargs.get("value", value)

    def text_area(self, label, value="", *args, **kwargs):
        return kwargs.get("value", value)

    def checkbox(self, label, value=False, *args, **kwargs):
        return kwargs.get("value", value)

    def toggle(self, label, value=False, *args, **kwargs):
        return kwargs.get("value", value)

    def button(self, *args, **kwargs):
        return False

    def form_submit_button(self, *args, **kwargs):
        return False

    def selectbox(self, label, options, index=0, *args, **kwargs):
        if not options:
            return None
        chosen_index = kwargs.get("index", index)
        chosen_index = min(max(int(chosen_index), 0), len(options) - 1)
        return options[chosen_index]

    def radio(self, label, options, index=0, *args, **kwargs):
        if not options:
            return None
        chosen_index = kwargs.get("index", index)
        chosen_index = min(max(int(chosen_index), 0), len(options) - 1)
        return options[chosen_index]

    def segmented_control(self, label, options, selection_mode="single", default=None, *args, **kwargs):
        if default is not None:
            return default
        if selection_mode == "multi":
            return []
        if not options:
            return None
        return options[0]

    def select_slider(self, label, options=None, value=None, *args, **kwargs):
        if value is not None:
            return value
        if options:
            return options[0]
        return None

    def number_input(self, label, value=0, *args, **kwargs):
        return kwargs.get("value", value)

    def slider(self, label, min_value=None, max_value=None, value=None, *args, **kwargs):
        return kwargs.get("value", value)

    def multiselect(self, label, options, default=None, *args, **kwargs):
        return kwargs.get("default", default if default is not None else [])

    def date_input(self, label, value=None, *args, **kwargs):
        return kwargs.get("value", value)

    def file_uploader(self, *args, **kwargs):
        return None


def configure_storage_env(db_path=None, database_url=None):
    if database_url:
        os.environ["STOCKS_DATABASE_URL"] = str(database_url)
        os.environ.pop("STOCKS_DB_PATH", None)
        return

    os.environ["STOCKS_DATABASE_URL"] = ""
    if db_path is not None:
        os.environ["STOCKS_DB_PATH"] = str(db_path)


def load_engine_module(app_path, db_path=None, database_url=None, module_name="stock_engine_seed_app"):
    os.environ["STOCK_ENGINE_SKIP_STARTUP_REFRESH"] = "1"
    configure_storage_env(db_path=db_path, database_url=database_url)
    sys.modules["streamlit"] = DummyStreamlit()

    spec = importlib.util.spec_from_file_location(module_name, app_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_tickers(module, tickers_arg, tickers_file):
    if tickers_arg:
        return module.parse_ticker_list(tickers_arg)

    source_path = Path(tickers_file)
    suffix = source_path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        try:
            frame = pd.read_csv(source_path, sep=sep)
            for column in ["Ticker", "ticker", "Symbol", "symbol"]:
                if column in frame.columns:
                    values = frame[column].dropna().astype(str).tolist()
                    return module.parse_ticker_list(",".join(values))
        except Exception:
            pass

    return module.parse_ticker_list(source_path.read_text(encoding="utf-8"))


def write_report(report_rows, report_path):
    if not report_rows:
        return

    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(report_rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)


def main():
    parser = argparse.ArgumentParser(description="Seed the Stock Engine SQLite database with a universe of tickers.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--tickers", help="Comma- or space-separated ticker list.")
    source_group.add_argument("--tickers-file", help="Path to a TXT/CSV/TSV file containing tickers.")
    parser.add_argument("--db-path", default="stocks_data.db", help="SQLite database path to seed.")
    parser.add_argument("--preset", default="Balanced", help="Model preset: Balanced, Conservative, or Aggressive.")
    parser.add_argument("--sleep-seconds", type=float, default=0.25, help="Delay between ticker requests.")
    parser.add_argument("--skip-fresh-hours", type=float, default=24.0, help="Skip rows updated within this many hours. Use 0 to disable.")
    parser.add_argument("--max-tickers", type=int, default=0, help="Optional cap on how many tickers to process.")
    parser.add_argument("--report-path", default="seed_universe_report.csv", help="CSV report output path.")
    args = parser.parse_args()

    app_path = Path(__file__).resolve().parent / "streamlit_app.py"
    db_path = Path(args.db_path).expanduser().resolve()
    module = load_engine_module(app_path, db_path)

    preset_catalog = module.get_model_presets()
    if args.preset not in preset_catalog:
        valid = ", ".join(sorted(preset_catalog))
        raise SystemExit(f"Unknown preset '{args.preset}'. Valid options: {valid}")

    tickers = load_tickers(module, args.tickers, args.tickers_file)
    if not tickers:
        raise SystemExit("No valid ticker symbols were found.")

    if args.max_tickers and args.max_tickers > 0:
        tickers = tickers[: args.max_tickers]

    db = module.DatabaseManager(db_path)
    analyst = module.StockAnalyst(db)
    settings = preset_catalog[args.preset]
    fresh_cutoff = None
    if args.skip_fresh_hours > 0:
        fresh_cutoff = datetime.datetime.now() - datetime.timedelta(hours=args.skip_fresh_hours)

    report_rows = []
    success_count = 0
    skipped_count = 0
    failed_count = 0

    print(f"Seeding {len(tickers)} tickers into {db_path} using the {args.preset} preset.")

    for idx, ticker in enumerate(tickers, start=1):
        ticker = ticker.strip().upper()
        status = "updated"
        detail = ""

        if fresh_cutoff is not None:
            existing = db.get_analysis(ticker)
            if not existing.empty:
                existing_stamp = module.parse_last_updated(existing.iloc[0].get("Last_Updated"))
                if existing_stamp is not None and existing_stamp >= fresh_cutoff:
                    skipped_count += 1
                    status = "skipped_fresh"
                    detail = f"Last updated {existing.iloc[0].get('Last_Updated')}"
                    report_rows.append(
                        {
                            "Ticker": ticker,
                            "Status": status,
                            "Detail": detail,
                            "Verdict": existing.iloc[0].get("Verdict_Overall"),
                            "Last_Updated": existing.iloc[0].get("Last_Updated"),
                        }
                    )
                    print(f"[{idx}/{len(tickers)}] {ticker}: skipped (fresh)")
                    continue

        record = analyst.analyze(ticker, settings=settings, persist=True)
        if record is None:
            failed_count += 1
            status = "failed"
            detail = analyst.last_error or "Unknown fetch failure."
            last_updated = ""
            verdict = ""
            print(f"[{idx}/{len(tickers)}] {ticker}: failed - {detail}")
        else:
            last_updated = record.get("Last_Updated", "")
            verdict = record.get("Verdict_Overall", "")
            detail = analyst.last_error or ""
            if detail and "showing the most recent saved analysis instead" in detail.lower():
                status = "reused_saved"
            elif detail:
                status = "updated_with_fallback"
            success_count += 1
            suffix = f" ({detail})" if detail else ""
            print(f"[{idx}/{len(tickers)}] {ticker}: {status} -> {verdict}{suffix}")

        report_rows.append(
            {
                "Ticker": ticker,
                "Status": status,
                "Detail": detail,
                "Verdict": verdict,
                "Last_Updated": last_updated,
            }
        )

        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    write_report(report_rows, args.report_path)
    print("")
    print(f"Completed. Updated: {success_count} | Skipped: {skipped_count} | Failed: {failed_count}")
    print(f"Report written to {Path(args.report_path).resolve()}")


if __name__ == "__main__":
    main()
