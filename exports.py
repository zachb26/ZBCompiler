# -*- coding: utf-8 -*-
"""
exports.py — Download / export byte builders.

Builds the raw bytes for every file the app offers as a download:
  * Library CSV
  * SQLite database backup
  * Company analysis snapshot JSON
  * DCF model JSON

No streamlit imports. Depends on constants and utils_fmt / utils_time.
"""

import datetime
import json
import os
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from constants import APP_VERSION, DCF_ANALYSIS_COLUMNS
from utils_fmt import has_numeric_value, normalize_ticker, safe_json_loads, safe_num
from utils_time import format_datetime_value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def summarize_fetch_error(exc):
    """Return a short human-readable summary of an exception message."""
    text = str(exc)
    low = text.lower()
    if "404" in text or "not found" in low:
        return "ticker not found (404)"
    if "429" in text or "too many" in low or "rate limit" in low:
        return "rate-limited by data provider"
    if "timed out" in low or "timeout" in low:
        return "request timed out"
    if "connection" in low or "network" in low:
        return "network/connection error"
    snippet = text[:120].replace("\n", " ")
    return f"fetch error: {snippet}"


def is_postgres_database_url(value):
    """Return True when *value* looks like a PostgreSQL connection DSN."""
    text = str(value or "").strip().lower()
    return text.startswith("postgresql://") or text.startswith("postgres://")


def build_postgres_connection_error_message(dsn, exc):
    """Build a helpful error message for a PostgreSQL connection failure.

    Appends Supabase-specific hints when the DSN contains a Supabase pooler host.
    """
    message = summarize_fetch_error(exc)
    lowered_message = message.lower()
    dsn_text = str(dsn or "").strip()
    dsn_lower = dsn_text.lower()
    hints = []

    if "pooler.supabase.com" in dsn_lower:
        hints.append(
            "Supabase pooler connections should use the exact connection string from Connect in the Supabase dashboard."
        )
        if "password authentication failed for user" in lowered_message:
            if "for user \"postgres\"" in message or "for user 'postgres'" in message:
                hints.append(
                    "For Supabase session pooler on port 5432, the username is usually 'postgres.<project-ref>' rather than plain 'postgres'."
                )
            hints.append(
                "Make sure you are using the database password from Project Settings -> Database, not your Supabase login password or an API key."
            )
            hints.append("If needed, reset the database password in Supabase and update STOCKS_DATABASE_URL.")

    if not hints:
        return message
    return f"{message} {' '.join(hints)}"


# ---------------------------------------------------------------------------
# Library CSV
# ---------------------------------------------------------------------------

def build_library_csv_bytes(df):
    """Return UTF-8 CSV bytes for the library DataFrame, dropping helper columns."""
    export_df = df.copy()
    export_df = export_df.drop(columns=["Last_Updated_Parsed"], errors="ignore")
    return export_df.to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------------
# Database backup
# ---------------------------------------------------------------------------

def build_database_download_bytes(db_path):
    """Return raw bytes for a live SQLite backup of *db_path*.

    Uses ``sqlite3.Connection.backup`` so the snapshot is consistent even
    under concurrent writes.  Returns ``b""`` when *db_path* is None or
    the file does not exist.
    """
    if not db_path:
        return b""

    source_path = Path(db_path)
    if not source_path.exists():
        return b""

    source_conn = sqlite3.connect(source_path, timeout=30, check_same_thread=False)
    temp_path = None
    try:
        fd, temp_name = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        temp_path = Path(temp_name)
        backup_conn = sqlite3.connect(temp_path, timeout=30, check_same_thread=False)
        try:
            source_conn.backup(backup_conn)
        finally:
            backup_conn.close()
        return temp_path.read_bytes()
    finally:
        source_conn.close()
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


# ---------------------------------------------------------------------------
# DCF model download
# ---------------------------------------------------------------------------

def extract_dcf_fields(record):
    """Return a dict of all DCF_* column values from a saved analysis record.

    *record* can be a plain dict or any object with a ``.get()`` method.
    """
    if record is None:
        return {}

    extracted = {}
    for field_name in DCF_ANALYSIS_COLUMNS:
        value = record.get(field_name) if hasattr(record, "get") else None
        if isinstance(value, float) and pd.isna(value):
            value = None
        extracted[field_name] = value
    return extracted


def has_dcf_snapshot(record):
    """Return True when a saved analysis record contains usable DCF data."""
    if record is None or not hasattr(record, "get"):
        return False

    if has_numeric_value(record.get("DCF_Intrinsic_Value")):
        return True

    last_updated = str(record.get("DCF_Last_Updated") or "").strip()
    if last_updated:
        return True

    for field_name in ["DCF_History", "DCF_Projection", "DCF_Sensitivity"]:
        payload = str(record.get(field_name) or "").strip()
        if payload and payload not in {"[]", "{}"}:
            return True
    return False


def build_dcf_download_bytes(record):
    """Serialise a DCF model snapshot from a saved analysis record to JSON bytes.

    Returns ``b""`` when no DCF snapshot exists in *record*.
    """
    if not has_dcf_snapshot(record):
        return b""

    payload = {
        "ticker": normalize_ticker(record.get("Ticker", "")),
        "price": safe_num(record.get("Price")),
        "assumption_profile": record.get("Assumption_Profile"),
        "analysis_last_updated": record.get("Last_Updated"),
        "dcf_last_updated": record.get("DCF_Last_Updated") or record.get("Last_Updated"),
        "dcf_assumptions": safe_json_loads(record.get("DCF_Assumptions"), default={}),
        "dcf_summary": {
            "intrinsic_value_per_share": safe_num(record.get("DCF_Intrinsic_Value")),
            "upside": safe_num(record.get("DCF_Upside")),
            "wacc": safe_num(record.get("DCF_WACC")),
            "risk_free_rate": safe_num(record.get("DCF_Risk_Free_Rate")),
            "beta": safe_num(record.get("DCF_Beta")),
            "cost_of_equity": safe_num(record.get("DCF_Cost_of_Equity")),
            "cost_of_debt": safe_num(record.get("DCF_Cost_of_Debt")),
            "equity_weight": safe_num(record.get("DCF_Equity_Weight")),
            "debt_weight": safe_num(record.get("DCF_Debt_Weight")),
            "growth_rate": safe_num(record.get("DCF_Growth_Rate")),
            "terminal_growth": safe_num(record.get("DCF_Terminal_Growth")),
            "base_fcf": safe_num(record.get("DCF_Base_FCF")),
            "enterprise_value": safe_num(record.get("DCF_Enterprise_Value")),
            "equity_value": safe_num(record.get("DCF_Equity_Value")),
            "historical_fcf_growth": safe_num(record.get("DCF_Historical_FCF_Growth")),
            "historical_revenue_growth": safe_num(record.get("DCF_Historical_Revenue_Growth")),
            "guidance_growth": safe_num(record.get("DCF_Guidance_Growth")),
            "source": record.get("DCF_Source"),
            "confidence": record.get("DCF_Confidence"),
            "filing_form": record.get("DCF_Filing_Form"),
            "filing_date": record.get("DCF_Filing_Date"),
            "guidance_summary": record.get("DCF_Guidance_Summary"),
        },
        "history": safe_json_loads(record.get("DCF_History"), default=[]),
        "projection": safe_json_loads(record.get("DCF_Projection"), default=[]),
        "sensitivity": safe_json_loads(record.get("DCF_Sensitivity"), default=[]),
        "guidance_excerpts": safe_json_loads(record.get("DCF_Guidance_Excerpts"), default=[]),
    }
    return json.dumps(payload, indent=2).encode("utf-8")


# ---------------------------------------------------------------------------
# Company analysis snapshot
# ---------------------------------------------------------------------------

def normalize_download_payload(value):
    """Recursively convert a nested payload to JSON-serialisable types.

    * pandas Timestamp / datetime → formatted string
    * numpy scalars → Python scalars
    * NaN → None
    * dicts and lists are traversed recursively
    """
    if isinstance(value, dict):
        return {str(key): normalize_download_payload(sub_value) for key, sub_value in value.items()}
    if isinstance(value, list):
        return [normalize_download_payload(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_download_payload(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return format_datetime_value(value)
    if isinstance(value, datetime.datetime):
        return format_datetime_value(value)
    if isinstance(value, datetime.date):
        return value.isoformat()
    if isinstance(value, np.generic):
        return normalize_download_payload(value.item())
    if isinstance(value, float) and pd.isna(value):
        return None
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def build_company_analysis_download_bytes(record):
    """Serialise a full company analysis row as a JSON snapshot.

    *record* must be a pandas Series or any object with a ``.to_dict()`` method.
    Returns ``b""`` when *record* is None or lacks ``.to_dict()``.
    """
    if record is None or not hasattr(record, "to_dict"):
        return b""

    raw_row = record.to_dict()
    parsed_sections = {
        "peer_comparison": safe_json_loads(raw_row.get("Peer_Comparison"), default={}),
        "event_study_events": safe_json_loads(raw_row.get("Event_Study_Events"), default=[]),
        "dcf_assumptions": safe_json_loads(raw_row.get("DCF_Assumptions"), default={}),
        "dcf_history": safe_json_loads(raw_row.get("DCF_History"), default=[]),
        "dcf_projection": safe_json_loads(raw_row.get("DCF_Projection"), default=[]),
        "dcf_sensitivity": safe_json_loads(raw_row.get("DCF_Sensitivity"), default=[]),
        "dcf_guidance_excerpts": safe_json_loads(raw_row.get("DCF_Guidance_Excerpts"), default=[]),
    }
    payload = {
        "app_version": APP_VERSION,
        "exported_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": normalize_ticker(raw_row.get("Ticker")),
        "analysis": normalize_download_payload(raw_row),
        "parsed_sections": normalize_download_payload(parsed_sections),
    }
    return json.dumps(payload, indent=2).encode("utf-8")
