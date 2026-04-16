# -*- coding: utf-8 -*-
"""
cache.py — In-memory TTL fetch-cache management.

Manages the global FETCH_CACHE dict using the thread-safe FETCH_CACHE_LOCK.
No streamlit imports. Depends on constants and utils_fmt.
"""

import copy
import time

import pandas as pd

from constants import (
    FETCH_CACHE,
    FETCH_CACHE_LOCK,
    FETCH_CACHE_MAX_ENTRIES,
    FETCH_CACHE_TTL_SECONDS,
    FETCH_STALE_FALLBACK_TTL_SECONDS,
)
from utils_fmt import safe_num


# ---------------------------------------------------------------------------
# Error summarization
# ---------------------------------------------------------------------------

def summarize_fetch_error(exc):
    """Return a short human-readable description of a fetch exception."""
    if exc is None:
        return "Unknown upstream fetch error."
    message = str(exc).strip() or exc.__class__.__name__
    message = " ".join(message.split())
    return message[:220]


# ---------------------------------------------------------------------------
# Cache payload cloning
# ---------------------------------------------------------------------------

def clone_cached_payload(payload):
    """Return a deep-copy of *payload* safe for mutation by the caller.

    DataFrames and Series are copied with ``.copy(deep=True)``; everything
    else uses ``copy.deepcopy``.
    """
    if isinstance(payload, pd.DataFrame):
        return payload.copy(deep=True)
    if isinstance(payload, pd.Series):
        return payload.copy(deep=True)
    return copy.deepcopy(payload)


# ---------------------------------------------------------------------------
# Cache access primitives
# ---------------------------------------------------------------------------

def get_cached_fetch_payload(bucket, key, max_age_seconds=FETCH_CACHE_TTL_SECONDS):
    """Retrieve a payload from the in-memory cache.

    Parameters
    ----------
    bucket:
        Top-level namespace key inside FETCH_CACHE (e.g. ``"ticker_history"``).
    key:
        Secondary key (usually a ticker string or tuple).
    max_age_seconds:
        Maximum age (in seconds) of a cached entry before it is treated as
        expired.  Entries older than ``FETCH_STALE_FALLBACK_TTL_SECONDS`` are
        evicted from the cache entirely.

    Returns
    -------
    payload | None
        A deep copy of the cached payload, or None on cache miss / expiry.
    """
    with FETCH_CACHE_LOCK:
        cache_entry = FETCH_CACHE[bucket].get(key)
        if not cache_entry:
            return None
        age_seconds = time.time() - cache_entry["timestamp"]
        if age_seconds > max_age_seconds:
            if age_seconds > FETCH_STALE_FALLBACK_TTL_SECONDS:
                FETCH_CACHE[bucket].pop(key, None)
            return None
        return clone_cached_payload(cache_entry["payload"])


def set_cached_fetch_payload(bucket, key, payload):
    """Write *payload* into the cache, evicting the oldest entry when the bucket is full."""
    with FETCH_CACHE_LOCK:
        bucket_store = FETCH_CACHE[bucket]
        max_entries = FETCH_CACHE_MAX_ENTRIES.get(bucket, 200)
        if len(bucket_store) >= max_entries:
            evict_count = max(1, len(bucket_store) - max_entries + 1)
            for _ in range(evict_count):
                bucket_store.pop(next(iter(bucket_store)), None)
        bucket_store[key] = {
            "timestamp": time.time(),
            "payload": clone_cached_payload(payload),
        }


# ---------------------------------------------------------------------------
# Payload normalization helpers
# ---------------------------------------------------------------------------

def normalize_history_frame(raw_history):
    """Ensure a yfinance history DataFrame has a flat DatetimeIndex and a ``Close`` column.

    Handles MultiIndex columns (produced by ``yf.download`` for a single ticker),
    renames columns to Title-Case, falls back to ``Adj Close`` when ``Close`` is
    absent, and strips timezone information from the index.

    Returns an empty DataFrame when *raw_history* is None or lacks a usable
    Close series.
    """
    if raw_history is None:
        return pd.DataFrame()

    hist = raw_history.copy()
    if isinstance(hist, pd.Series):
        hist = hist.to_frame(name="Close")

    if isinstance(hist.columns, pd.MultiIndex):
        if "Close" in hist.columns.get_level_values(0):
            hist = hist["Close"].copy()
            if isinstance(hist, pd.Series):
                hist = hist.to_frame(name="Close")
        else:
            hist.columns = [
                " ".join(str(part) for part in col if str(part) != "").strip()
                for col in hist.columns.to_flat_index()
            ]

    renamed_columns = {column: str(column).strip().title() for column in hist.columns}
    hist = hist.rename(columns=renamed_columns)

    if "Close" not in hist.columns and "Adj Close" in hist.columns:
        hist["Close"] = hist["Adj Close"]

    if "Close" not in hist.columns:
        return pd.DataFrame()

    hist = hist.sort_index()
    hist = hist[~hist.index.duplicated(keep="last")]
    if getattr(hist.index, "tz", None) is not None:
        hist.index = hist.index.tz_localize(None)
    hist["Close"] = pd.to_numeric(hist["Close"], errors="coerce")
    hist = hist.dropna(subset=["Close"])
    return hist


def normalize_info_payload(info):
    """Return a cleaned copy of a yfinance info dict, stripping None and NaN values."""
    if not isinstance(info, dict):
        return {}
    cleaned = {}
    for key, value in info.items():
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        cleaned[key] = value
    return cleaned


def normalize_fast_info_payload(fast_info):
    """Return a minimal dict from a yfinance fast_info object.

    Maps known yfinance fast_info attributes to canonical keys.  Returns an
    empty dict when *fast_info* is None or has no ``.items()`` method.
    """
    if fast_info is None:
        return {}
    if hasattr(fast_info, "items"):
        payload = dict(fast_info.items())
    else:
        return {}
    field_map = {
        "marketCap": ["marketCap", "market_cap"],
        "beta": ["beta"],
        "sharesOutstanding": ["shares", "sharesOutstanding"],
        "lastPrice": ["lastPrice", "last_price"],
    }
    normalized = {}
    for output_key, candidate_keys in field_map.items():
        for candidate_key in candidate_keys:
            value = payload.get(candidate_key)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            normalized[output_key] = value
            break
    return normalized


def normalize_news_payload(news):
    """Return a list of news item dicts, or an empty list on failure."""
    if not isinstance(news, list):
        return []
    return [item for item in news if isinstance(item, dict)]


def build_info_fallback_from_saved_analysis(saved_row):
    """Construct a minimal yfinance-style info dict from a saved analysis row.

    *saved_row* can be a pandas DataFrame (first row is used) or a pandas Series /
    dict-like.  Returns an empty dict when the input is None or empty.
    """
    if saved_row is None or (isinstance(saved_row, pd.DataFrame) and saved_row.empty):
        return {}

    row = saved_row.iloc[0] if isinstance(saved_row, pd.DataFrame) else saved_row

    fallback_map = {
        "sector": row.get("Sector"),
        "industry": row.get("Industry"),
        "marketCap": row.get("Market_Cap"),
        "dividendYield": row.get("Dividend_Yield"),
        "payoutRatio": row.get("Payout_Ratio"),
        "beta": row.get("Equity_Beta"),
        "trailingPE": row.get("PE_Ratio"),
        "forwardPE": row.get("Forward_PE"),
        "pegRatio": row.get("PEG_Ratio"),
        "priceToSalesTrailing12Months": row.get("PS_Ratio"),
        "priceToBook": row.get("PB_Ratio"),
        "enterpriseToEbitda": row.get("EV_EBITDA"),
        "returnOnEquity": row.get("ROE"),
        "profitMargins": row.get("Profit_Margins"),
        "debtToEquity": row.get("Debt_to_Equity"),
        "revenueGrowth": row.get("Revenue_Growth"),
        "currentRatio": row.get("Current_Ratio"),
        "targetMeanPrice": row.get("Target_Mean_Price"),
        "recommendationKey": row.get("Recommendation_Key"),
        "numberOfAnalystOpinions": row.get("Analyst_Opinions"),
    }
    cleaned = {}
    for key, value in fallback_map.items():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        cleaned[key] = value
    return cleaned
