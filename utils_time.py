# -*- coding: utf-8 -*-
"""
utils_time.py — Date/time parsing and formatting helpers.

No streamlit imports. No imports from other application modules.
"""

import datetime

import pandas as pd


def parse_last_updated(value):
    """Parse a ``YYYY-MM-DD HH:MM`` or ``YYYY-MM-DD HH:MM:SS`` string to a datetime.

    Returns None when *value* is None, NaN, or unparseable.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def format_age(value):
    """Return a human-readable age string such as ``"3h ago"`` for a timestamp.

    *value* is first passed through :func:`parse_last_updated`.
    Returns ``"Unknown"`` when the timestamp cannot be parsed.
    """
    stamp = parse_last_updated(value)
    if stamp is None:
        return "Unknown"

    elapsed = datetime.datetime.now() - stamp
    total_minutes = max(int(elapsed.total_seconds() // 60), 0)
    if total_minutes == 0:
        return "Just now"
    if total_minutes < 60:
        return f"{total_minutes}m ago"

    total_hours = total_minutes // 60
    if total_hours < 24:
        return f"{total_hours}h ago"

    total_days = total_hours // 24
    return f"{total_days}d ago"


def parse_any_datetime(value):
    """Parse *value* into a timezone-naive ``datetime.datetime``.

    Handles: pandas Timestamp, Python datetime/date, Unix epoch integers
    (seconds or milliseconds), and common ISO-8601 strings.

    Returns None when parsing fails.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime.datetime):
        if value.tzinfo is not None:
            return value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return value
    if isinstance(value, datetime.date):
        return datetime.datetime.combine(value, datetime.time.min)
    if isinstance(value, (int, float)):
        raw_value = float(value)
        if raw_value <= 0:
            return None
        if raw_value > 1_000_000_000_000:
            raw_value = raw_value / 1000
        try:
            return datetime.datetime.fromtimestamp(raw_value)
        except (OverflowError, OSError, ValueError):
            return None

    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        parsed = pd.to_datetime(text, utc=False)
        if pd.isna(parsed):
            return None
        if isinstance(parsed, pd.Timestamp):
            return parse_any_datetime(parsed.to_pydatetime())
    except Exception:
        return None
    return None


def format_datetime_value(value, fallback="Unknown"):
    """Format *value* as ``YYYY-MM-DD`` or ``YYYY-MM-DD HH:MM`` depending on precision.

    Returns *fallback* when *value* cannot be parsed.
    """
    stamp = parse_any_datetime(value)
    if stamp is None:
        return fallback
    if stamp.hour == 0 and stamp.minute == 0 and stamp.second == 0:
        return stamp.strftime("%Y-%m-%d")
    return stamp.strftime("%Y-%m-%d %H:%M")


def approximate_trading_days_for_period(period):
    """Return the approximate number of trading days for a yfinance period string.

    Returns None when *period* is not recognised.
    """
    return {
        "3mo": 63,
        "6mo": 126,
        "1y": 252,
        "18mo": 378,
        "2y": 504,
        "3y": 756,
        "5y": 1260,
        "10y": 2520,
    }.get(str(period).lower())


def trim_history_to_period(hist, period):
    """Trim a price-history DataFrame to the last *period* worth of trading days.

    Returns an empty DataFrame when *hist* is None or empty.
    Returns a copy of *hist* unchanged when the period is not recognised or
    the frame is shorter than the target window.
    """
    if hist is None or hist.empty:
        return pd.DataFrame()
    trading_days = approximate_trading_days_for_period(period)
    if trading_days is None or len(hist) <= trading_days:
        return hist.copy()
    return hist.tail(trading_days).copy()
