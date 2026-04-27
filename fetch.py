# -*- coding: utf-8 -*-
"""
fetch.py — Data-fetching layer.

Covers:
- Cache helpers: summarize_fetch_error, clone_cached_payload,
  get_cached_fetch_payload, set_cached_fetch_payload
- Payload normalisers: normalize_history_frame, normalize_info_payload,
  normalize_fast_info_payload, normalize_news_payload,
  build_info_fallback_from_saved_analysis
- Peer-group discovery: get_peer_universe_tickers, score_peer_similarity,
  build_peer_candidate_info, find_closest_peer_group,
  build_relative_peer_benchmarks
- Event study: classify_event_category, compute_event_study
- Filing helpers: extract_filing_takeaways_from_text
- yfinance wrappers with retry and stale-fallback:
  fetch_batch_history_via_individual_tickers,
  fetch_ticker_history_with_retry, fetch_batch_history_with_retry,
  fetch_ticker_info_with_retry, fetch_ticker_news_with_retry
- Utility shims that live here (also re-exported from utils_fmt):
  has_numeric_value, calculate_rsi, score_relative_multiple,
  extract_sentiment_tokens, safe_divide, safe_json_loads
"""

import copy
import datetime
import io
import json
import logging
import re
import time
import urllib.request

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import yfinance as yf

from constants import (
    APP_DIR,
    AUTO_REFRESH_REQUEST_DELAY_SECONDS,
    CATALYST_CACHE_TTL,
    DEFAULT_BENCHMARK_TICKER,
    DEFAULT_PORTFOLIO_TICKERS,
    FETCH_CACHE,
    FETCH_CACHE_LOCK,
    FETCH_CACHE_MAX_ENTRIES,
    FETCH_CACHE_TTL_SECONDS,
    FETCH_STALE_FALLBACK_TTL_SECONDS,
    FILING_TAKEAWAY_PATTERNS,
    FUNDAMENTAL_EVENT_KEYWORDS,
    MACRO_CACHE_TTL,
    PEER_GROUP_SIZE,
    PEER_METRIC_MAP,
    PEER_MIN_REQUIRED,
    PEER_SEARCH_CANDIDATE_LIMIT,
    PEER_UNIVERSE_FILENAME,
    normalize_sector,
)
from utils_fmt import normalize_ticker, parse_ticker_list, safe_num
from utils_time import format_datetime_value, trim_history_to_period
from utils_news import extract_news_publish_time, extract_news_title


# ---------------------------------------------------------------------------
# Utility functions (also present in utils_fmt; duplicated here for modules
# that only import from fetch to avoid a circular dep)
# ---------------------------------------------------------------------------

def has_numeric_value(value):
    """Return True when *value* is a non-None, non-NaN number."""
    return value is not None and not pd.isna(value)


def calculate_rsi(close, period=14):
    """Compute Wilder-smoothed RSI for the given close price series."""
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100)
    rsi = rsi.where(avg_gain != 0, 0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)
    return rsi


def score_relative_multiple(value, benchmark, cheap_buffer=0.9, expensive_buffer=1.25):
    """Return +1 cheap, -1 expensive, 0 neutral vs a peer/sector benchmark."""
    if not has_numeric_value(value):
        return 0
    if value <= 0:
        return -1
    if not has_numeric_value(benchmark) or benchmark <= 0:
        return 0
    if value <= benchmark * cheap_buffer:
        return 1
    if value >= benchmark * expensive_buffer:
        return -1
    return 0


def extract_sentiment_tokens(text):
    """Return a set of lowercase word tokens from *text*."""
    return set(re.findall(r"[a-z]+", (text or "").lower()))


def safe_divide(numerator, denominator):
    """Divide and return None when denominator is zero or missing."""
    if denominator is None or pd.isna(denominator) or abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def safe_json_loads(value, default=None):
    """Parse JSON string *value*, returning *default* dict/list on failure."""
    fallback = {} if default is None else default
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return copy.deepcopy(fallback)
    if isinstance(value, (dict, list)):
        return copy.deepcopy(value)
    try:
        return json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return copy.deepcopy(fallback)


# ---------------------------------------------------------------------------
# Fetch-error helpers
# ---------------------------------------------------------------------------

def summarize_fetch_error(exc):
    """Return a concise, single-line description of *exc* (≤220 chars)."""
    if exc is None:
        return "Unknown upstream fetch error."
    message = str(exc).strip() or exc.__class__.__name__
    message = " ".join(message.split())
    return message[:220]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def clone_cached_payload(payload):
    """Return a deep copy of *payload* (handles DataFrame/Series correctly)."""
    if isinstance(payload, pd.DataFrame):
        return payload.copy(deep=True)
    if isinstance(payload, pd.Series):
        return payload.copy(deep=True)
    return copy.deepcopy(payload)


def get_cached_fetch_payload(bucket, key, max_age_seconds=FETCH_CACHE_TTL_SECONDS):
    """Fetch from the in-memory cache; return None on miss or expiry."""
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
    """Write *payload* to the in-memory cache, evicting oldest entry if full."""
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
# Payload normalisers
# ---------------------------------------------------------------------------

def normalize_history_frame(raw_history):
    """Coerce a raw yfinance history download into a clean Close-indexed DataFrame."""
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
    """Strip None/NaN entries from a raw yfinance info dict."""
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
    """Extract common fields from a yfinance fast_info object."""
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
    """Return a list of dicts from a raw yfinance news list."""
    if not isinstance(news, list):
        return []
    normalized = []
    for item in news:
        if isinstance(item, dict):
            normalized.append(item)
    return normalized


def build_info_fallback_from_saved_analysis(saved_row):
    """Build a minimal info dict from a previously saved database row."""
    if saved_row is None or isinstance(saved_row, pd.DataFrame) and saved_row.empty:
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


# ---------------------------------------------------------------------------
# Peer-group discovery
# ---------------------------------------------------------------------------

def get_peer_universe_tickers(db=None):
    """Return an ordered list of known tickers to use as peer-search candidates."""
    tickers = []
    seen = set()

    def add_many(values):
        for raw_value in values or []:
            ticker = normalize_ticker(raw_value)
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            tickers.append(ticker)

    if db is not None:
        try:
            saved_rows = db.get_all_analyses()
        except Exception:
            saved_rows = pd.DataFrame()
        if not saved_rows.empty and "Ticker" in saved_rows.columns:
            add_many(saved_rows["Ticker"].dropna().tolist())

    peer_file = APP_DIR / PEER_UNIVERSE_FILENAME
    if peer_file.exists():
        try:
            add_many(peer_file.read_text(encoding="utf-8").splitlines())
        except OSError:
            pass

    add_many(parse_ticker_list(DEFAULT_PORTFOLIO_TICKERS))
    return tickers


def score_peer_similarity(target_info, candidate_info):
    """
    Return (priority, similarity_score) for a candidate vs a target company.

    priority 0 = same industry, 1 = same sector, 2 = different.
    Higher similarity_score is better.
    """
    if not isinstance(candidate_info, dict) or not candidate_info:
        return None, None

    target_sector = str(target_info.get("sector") or "").strip().lower()
    candidate_sector = str(candidate_info.get("sector") or "").strip().lower()
    target_industry = str(target_info.get("industry") or "").strip().lower()
    candidate_industry = str(candidate_info.get("industry") or "").strip().lower()

    if target_industry and candidate_industry and target_industry == candidate_industry:
        priority = 0
    elif target_sector and candidate_sector and target_sector == candidate_sector:
        priority = 1
    else:
        priority = 2

    score = 0.0
    if priority == 0:
        score += 6.0
    elif priority == 1:
        score += 3.5
    else:
        score -= 4.0

    target_market_cap = safe_num(target_info.get("marketCap"))
    candidate_market_cap = safe_num(candidate_info.get("marketCap"))
    if has_numeric_value(target_market_cap) and has_numeric_value(candidate_market_cap) and target_market_cap > 0 and candidate_market_cap > 0:
        score -= abs(np.log(candidate_market_cap / target_market_cap))

    for key, weight in [
        ("beta", 0.8),
        ("revenueGrowth", 1.0),
        ("profitMargins", 0.7),
        ("returnOnEquity", 0.6),
    ]:
        target_value = safe_num(target_info.get(key))
        candidate_value = safe_num(candidate_info.get(key))
        if has_numeric_value(target_value) and has_numeric_value(candidate_value):
            score -= min(abs(candidate_value - target_value) * weight * 4, 2.5)

    return priority, float(score)


def build_peer_candidate_info(candidate_ticker, db=None):
    """Return an info dict for *candidate_ticker* using cache → DB → live fetch."""
    cached_info = get_cached_fetch_payload("ticker_info", normalize_ticker(candidate_ticker))
    if cached_info:
        return cached_info

    fallback_info = {}
    if db is not None:
        saved_row = db.get_analysis(candidate_ticker)
        if not saved_row.empty:
            fallback_info = build_info_fallback_from_saved_analysis(saved_row.iloc[0])
            if fallback_info:
                return fallback_info

    info, _ = fetch_ticker_info_with_retry(candidate_ticker, attempts=1)
    if info:
        return info
    return fallback_info


def find_closest_peer_group(ticker, info, db=None, peer_count=PEER_GROUP_SIZE):
    """
    Scan the peer universe and return a peer-group dict for *ticker*.

    The returned dict has keys: count, group_label, tickers, summary,
    averages (keyed by PEER_METRIC_MAP output names), rows.
    Results are cached in the 'peer_group' bucket.
    """
    # Import here to avoid circular dependency (constants.get_sector_benchmarks
    # is needed by build_relative_peer_benchmarks below)
    from constants import get_sector_benchmarks  # noqa: F401

    cache_key = (normalize_ticker(ticker), str(info.get("sector") or "").strip(), str(info.get("industry") or "").strip())
    cached_group = get_cached_fetch_payload("peer_group", cache_key, max_age_seconds=FETCH_STALE_FALLBACK_TTL_SECONDS)
    if cached_group:
        return cached_group

    target_ticker = normalize_ticker(ticker)
    target_info = info or {}
    universe = get_peer_universe_tickers(db)
    if db is not None:
        try:
            db_tickers_set = set(
                db.get_all_analyses()["Ticker"].dropna().str.strip().str.upper()
            )
        except Exception:
            db_tickers_set = set()
        db_first = [t for t in universe if normalize_ticker(t) in db_tickers_set]
        file_only = [t for t in universe if normalize_ticker(t) not in db_tickers_set]
        universe = db_first + file_only

    target_sector = str(target_info.get("sector") or "").strip()
    target_industry = str(target_info.get("industry") or "").strip()

    candidates = []
    scanned = 0
    close_match_count = 0
    for candidate_ticker in universe:
        candidate_ticker = normalize_ticker(candidate_ticker)
        if not candidate_ticker or candidate_ticker == target_ticker:
            continue

        candidate_info = build_peer_candidate_info(candidate_ticker, db=db)
        if not candidate_info:
            continue

        priority, similarity_score = score_peer_similarity(target_info, candidate_info)
        if similarity_score is None:
            continue

        if priority <= 1:
            close_match_count += 1

        candidates.append(
            {
                "Ticker": candidate_ticker,
                "Name": str(candidate_info.get("shortName") or candidate_info.get("longName") or candidate_ticker),
                "Sector": normalize_sector(str(candidate_info.get("sector") or "")),
                "Industry": str(candidate_info.get("industry") or ""),
                "Similarity": similarity_score,
                "Priority": priority,
                "marketCap": safe_num(candidate_info.get("marketCap")),
                "trailingPE": safe_num(candidate_info.get("trailingPE")),
                "forwardPE": safe_num(candidate_info.get("forwardPE")),
                "pegRatio": safe_num(candidate_info.get("pegRatio")),
                "priceToSalesTrailing12Months": safe_num(candidate_info.get("priceToSalesTrailing12Months")),
                "priceToBook": safe_num(candidate_info.get("priceToBook")),
                "enterpriseToEbitda": safe_num(candidate_info.get("enterpriseToEbitda")),
                "returnOnEquity": safe_num(candidate_info.get("returnOnEquity")),
                "profitMargins": safe_num(candidate_info.get("profitMargins")),
                "debtToEquity": safe_num(candidate_info.get("debtToEquity")),
                "revenueGrowth": safe_num(candidate_info.get("revenueGrowth")),
                "currentRatio": safe_num(candidate_info.get("currentRatio")),
                "beta": safe_num(candidate_info.get("beta")),
            }
        )
        scanned += 1
        if scanned >= PEER_SEARCH_CANDIDATE_LIMIT and len(candidates) >= peer_count:
            if close_match_count >= peer_count:
                break

    candidates = sorted(
        candidates,
        key=lambda row: (row["Priority"], -row["Similarity"], row["Ticker"]),
    )
    selected = candidates[:peer_count]

    averages = {}
    for output_key, input_key in PEER_METRIC_MAP.items():
        values = [row[input_key] for row in selected if has_numeric_value(row.get(input_key))]
        averages[output_key] = float(np.median(values)) if values else None

    group_label = target_industry or target_sector or "Closest peers"
    peer_names = ", ".join(row["Ticker"] for row in selected[:peer_count]) if selected else "None found"
    summary = (
        f"{len(selected)} closest peers from {group_label}: {peer_names}."
        if selected
        else "Not enough comparable peers were available in the current universe."
    )
    payload = {
        "count": len(selected),
        "group_label": group_label,
        "tickers": [row["Ticker"] for row in selected],
        "summary": summary,
        "averages": averages,
        "rows": selected,
    }
    set_cached_fetch_payload("peer_group", cache_key, payload)
    return payload


def build_relative_peer_benchmarks(ticker, info, db=None, settings=None):
    """
    Return (benchmarks_dict, peer_group_dict) anchored to closest peers,
    with sector fallback when the peer group is thin.
    """
    from constants import get_sector_benchmarks

    peer_group = find_closest_peer_group(ticker, info, db=db, peer_count=PEER_GROUP_SIZE)
    sector_fallback = get_sector_benchmarks(normalize_sector(info.get("sector", "")), settings=settings)
    averages = peer_group.get("averages", {})
    benchmarks = {
        "PE": averages.get("PE"),
        "PS": averages.get("PS"),
        "PB": averages.get("PB"),
        "EV_EBITDA": averages.get("EV_EBITDA"),
    }
    usable_peer_metrics = sum(1 for value in benchmarks.values() if has_numeric_value(value))
    if peer_group.get("count", 0) < PEER_MIN_REQUIRED or usable_peer_metrics < 2:
        for metric_name, fallback_value in sector_fallback.items():
            if not has_numeric_value(benchmarks.get(metric_name)):
                benchmarks[metric_name] = fallback_value
        source = "Peer group with sector fallback"
    else:
        source = "Closest peer group"
    peer_group["benchmark_source"] = source
    peer_group["benchmarks"] = benchmarks
    return benchmarks, peer_group


# ---------------------------------------------------------------------------
# Event study
# ---------------------------------------------------------------------------

def classify_event_category(title):
    """Return a short category label for a news headline."""
    lowered = str(title or "").lower()
    if any(token in lowered for token in ["earnings", "eps", "quarter", "guidance"]):
        return "Earnings / Guidance"
    if any(token in lowered for token in ["dividend", "buyback", "repurchase"]):
        return "Capital Return"
    if any(token in lowered for token in ["acquisition", "merger", "deal"]):
        return "M&A"
    if any(token in lowered for token in ["filing", "10-k", "10-q", "sec"]):
        return "Filing"
    if any(token in lowered for token in ["debt", "liquidity", "cash flow", "margin"]):
        return "Balance Sheet / Cash Flow"
    return "Company Event"


def compute_event_study(news, hist, benchmark_ticker=DEFAULT_BENCHMARK_TICKER):
    """
    Compute 1-day and 5-day abnormal returns around fundamental news events.

    Returns a dict with keys: count, avg_abnormal_1d, avg_abnormal_5d,
    summary, events.
    """
    close = pd.Series(dtype=float)
    if hist is not None and not hist.empty and "Close" in hist.columns:
        close = hist["Close"].dropna().astype(float)
    if close.empty:
        return {
            "count": 0,
            "avg_abnormal_1d": None,
            "avg_abnormal_5d": None,
            "summary": "No usable price history was available for an event study.",
            "events": [],
        }

    benchmark_hist, _ = fetch_ticker_history_with_retry(benchmark_ticker, period="1y", attempts=2)
    benchmark_close = (
        trim_history_to_period(benchmark_hist, "1y")["Close"].dropna().astype(float)
        if benchmark_hist is not None and not benchmark_hist.empty and "Close" in benchmark_hist.columns
        else pd.Series(dtype=float)
    )
    aligned = pd.concat(
        [close.rename("stock"), benchmark_close.rename("benchmark")],
        axis=1,
        join="outer",
    ).sort_index()
    aligned = aligned.reindex(close.index)
    if "benchmark" in aligned.columns:
        aligned["benchmark"] = aligned["benchmark"].ffill(limit=3)

    fundamental_events = []
    seen_titles = set()
    for item in news or []:
        title = extract_news_title(item)
        lowered = title.lower()
        if not title or title.lower() in seen_titles:
            continue
        if not any(keyword in lowered for keyword in FUNDAMENTAL_EVENT_KEYWORDS):
            continue
        seen_titles.add(title.lower())
        published = extract_news_publish_time(item)
        if published is None:
            continue
        normalized_published = datetime.datetime.combine(published.date(), datetime.time.min)
        event_loc = aligned.index.searchsorted(normalized_published)
        if event_loc >= len(aligned):
            continue
        stock_1d = safe_divide(aligned["stock"].iloc[min(event_loc + 1, len(aligned) - 1)] - aligned["stock"].iloc[event_loc], aligned["stock"].iloc[event_loc])
        stock_5d = safe_divide(aligned["stock"].iloc[min(event_loc + 5, len(aligned) - 1)] - aligned["stock"].iloc[event_loc], aligned["stock"].iloc[event_loc])
        bench_1d = None
        bench_5d = None
        if "benchmark" in aligned.columns and has_numeric_value(aligned["benchmark"].iloc[event_loc]):
            bench_1d = safe_divide(
                aligned["benchmark"].iloc[min(event_loc + 1, len(aligned) - 1)] - aligned["benchmark"].iloc[event_loc],
                aligned["benchmark"].iloc[event_loc],
            )
            bench_5d = safe_divide(
                aligned["benchmark"].iloc[min(event_loc + 5, len(aligned) - 1)] - aligned["benchmark"].iloc[event_loc],
                aligned["benchmark"].iloc[event_loc],
            )
        fundamental_events.append(
            {
                "Date": format_datetime_value(published),
                "Category": classify_event_category(title),
                "Headline": title,
                "Return_1D": stock_1d,
                "Return_5D": stock_5d,
                "Abnormal_1D": None if bench_1d is None or stock_1d is None else stock_1d - bench_1d,
                "Abnormal_5D": None if bench_5d is None or stock_5d is None else stock_5d - bench_5d,
            }
        )
        if len(fundamental_events) >= 5:
            break

    avg_abnormal_1d = None
    avg_abnormal_5d = None
    if fundamental_events:
        abnormal_1d_values = [event["Abnormal_1D"] for event in fundamental_events if has_numeric_value(event.get("Abnormal_1D"))]
        abnormal_5d_values = [event["Abnormal_5D"] for event in fundamental_events if has_numeric_value(event.get("Abnormal_5D"))]
        if abnormal_1d_values:
            avg_abnormal_1d = float(np.mean(abnormal_1d_values))
        if abnormal_5d_values:
            avg_abnormal_5d = float(np.mean(abnormal_5d_values))

    if fundamental_events:
        summary = f"{len(fundamental_events)} recent company events captured for reaction study."
        if has_numeric_value(avg_abnormal_5d):
            summary += f" Average 5D move versus {benchmark_ticker}: {avg_abnormal_5d * 100:.1f}%."
    else:
        summary = "No recent company events with usable dates were available for an event study."

    return {
        "count": len(fundamental_events),
        "avg_abnormal_1d": avg_abnormal_1d,
        "avg_abnormal_5d": avg_abnormal_5d,
        "summary": summary,
        "events": fundamental_events,
    }


# ---------------------------------------------------------------------------
# Filing helpers
# ---------------------------------------------------------------------------

def extract_filing_takeaways_from_text(filing_text, max_takeaways=5):
    """Extract up to *max_takeaways* key sentences from raw filing text."""
    from sec_ai import strip_html_to_text  # local import to avoid circular dep
    clean_text = strip_html_to_text(filing_text)
    if not clean_text:
        return []

    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", clean_text) if sentence.strip()]
    takeaways = []
    seen = set()
    for sentence in sentences:
        lowered = sentence.lower()
        if not any(pattern in lowered for pattern in FILING_TAKEAWAY_PATTERNS):
            continue
        cleaned_sentence = " ".join(sentence.split())
        if len(cleaned_sentence) < 60:
            continue
        dedupe_key = cleaned_sentence.lower()[:260]
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        takeaways.append(cleaned_sentence)
        if len(takeaways) >= max_takeaways:
            break
    return takeaways


# ---------------------------------------------------------------------------
# yfinance wrappers
# ---------------------------------------------------------------------------

def fetch_batch_history_via_individual_tickers(tickers, period):
    """
    Fetch price history one ticker at a time and combine into a Close DataFrame.

    Used as a fallback when the batch yf.download fails.
    """
    close_frames = []
    failure_messages = []

    for ticker in tickers:
        hist, error = fetch_ticker_history_with_retry(ticker, period, attempts=2)
        if hist is None or hist.empty or "Close" not in hist.columns:
            if error:
                failure_messages.append(f"{ticker}: {error}")
            continue
        close_frames.append(hist["Close"].rename(ticker))
        time.sleep(AUTO_REFRESH_REQUEST_DELAY_SECONDS)

    if not close_frames:
        return pd.DataFrame(), " | ".join(failure_messages[:4]) if failure_messages else None

    combined = pd.concat(close_frames, axis=1, join="outer").sort_index()
    return combined, None


def fetch_ticker_history_with_retry(ticker, period, attempts=3):
    """
    Fetch OHLCV history for *ticker* with retry, stale-cache fallback,
    and alternate-period widening for the 1y case.

    Returns (DataFrame, error_string_or_None).
    """
    cache_key = (ticker.upper(), period)
    cached_history = get_cached_fetch_payload("ticker_history", cache_key)
    if cached_history is not None and not cached_history.empty:
        return cached_history, None

    last_error = None
    for attempt in range(attempts):
        try:
            hist = normalize_history_frame(yf.Ticker(ticker).history(period=period, auto_adjust=True))
            if not hist.empty:
                set_cached_fetch_payload("ticker_history", cache_key, hist)
                return hist, None
            download_fallback = yf.download(
                ticker.upper(),
                period=period,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            hist = normalize_history_frame(download_fallback)
            if not hist.empty:
                set_cached_fetch_payload("ticker_history", cache_key, hist)
                return hist, None
            last_error = f"Yahoo returned no usable {period} price history for {ticker}."
        except Exception as exc:
            logger.warning("fetch_ticker_history attempt %d failed (%s, %s): %s", attempt + 1, ticker, period, exc)
            last_error = summarize_fetch_error(exc)
        if attempt < attempts - 1:
            time.sleep(0.5 * (attempt + 1))

    alternate_periods = []
    if str(period).lower() == "1y":
        alternate_periods = ["18mo", "2y"]
    for alternate_period in alternate_periods:
        try:
            hist = normalize_history_frame(yf.download(
                ticker.upper(),
                period=alternate_period,
                auto_adjust=True,
                progress=False,
                threads=False,
            ))
            hist = trim_history_to_period(hist, period)
            if not hist.empty:
                set_cached_fetch_payload("ticker_history", cache_key, hist)
                return hist, None
        except Exception as exc:
            logger.warning("fetch_batch_history individual-ticker fallback failed (%s, %s): %s", ticker, period, exc)
            last_error = summarize_fetch_error(exc)

    stale_history = get_cached_fetch_payload(
        "ticker_history",
        cache_key,
        max_age_seconds=FETCH_STALE_FALLBACK_TTL_SECONDS,
    )
    if stale_history is not None and not stale_history.empty:
        return stale_history, None
    return pd.DataFrame(), last_error


def fetch_batch_history_with_retry(tickers, period, attempts=3):
    """
    Fetch price history for multiple *tickers* with batch download, retry,
    individual-ticker fallback, and stale-cache fallback.

    Returns (DataFrame, error_string_or_None).
    """
    normalized_tickers = tuple(dict.fromkeys(ticker.upper() for ticker in tickers))
    cache_key = (normalized_tickers, period)
    cached_history = get_cached_fetch_payload("batch_history", cache_key)
    if cached_history is not None and not cached_history.empty:
        return cached_history, None

    last_error = None
    for attempt in range(attempts):
        try:
            raw = yf.download(
                list(normalized_tickers),
                period=period,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if raw is not None and not raw.empty:
                set_cached_fetch_payload("batch_history", cache_key, raw)
                return raw.copy(), None
            last_error = "Yahoo returned no portfolio price history for the selected basket."
        except Exception as exc:
            logger.warning("fetch_batch_history attempt %d failed: %s", attempt + 1, exc)
            last_error = summarize_fetch_error(exc)
        if attempt < attempts - 1:
            time.sleep(0.5 * (attempt + 1))

    individual_history, fallback_error = fetch_batch_history_via_individual_tickers(normalized_tickers, period)
    if individual_history is not None and not individual_history.empty:
        set_cached_fetch_payload("batch_history", cache_key, individual_history)
        return individual_history.copy(), None
    if fallback_error:
        last_error = fallback_error

    stale_history = get_cached_fetch_payload(
        "batch_history",
        cache_key,
        max_age_seconds=FETCH_STALE_FALLBACK_TTL_SECONDS,
    )
    if stale_history is not None and not stale_history.empty:
        return stale_history, None
    return pd.DataFrame(), last_error


def fetch_ticker_info_with_retry(ticker, attempts=3):
    """
    Fetch company info for *ticker* with retry and stale-cache fallback.

    Returns (info_dict, error_string_or_None).
    """
    cache_key = ticker.upper()
    cached_info = get_cached_fetch_payload("ticker_info", cache_key)
    if cached_info:
        return cached_info, None

    last_error = None
    ticker_handle = yf.Ticker(ticker)
    for attempt in range(attempts):
        try:
            info = normalize_info_payload(ticker_handle.info or {})
            if info:
                fast_info = normalize_fast_info_payload(getattr(ticker_handle, "fast_info", None))
                info = {**fast_info, **info}
                set_cached_fetch_payload("ticker_info", cache_key, info)
                return info, None
            alt_info = normalize_info_payload(getattr(ticker_handle, "get_info", lambda: {})() or {})
            if alt_info:
                fast_info = normalize_fast_info_payload(getattr(ticker_handle, "fast_info", None))
                alt_info = {**fast_info, **alt_info}
                set_cached_fetch_payload("ticker_info", cache_key, alt_info)
                return alt_info, None
            fast_info = normalize_fast_info_payload(getattr(ticker_handle, "fast_info", None))
            if fast_info:
                set_cached_fetch_payload("ticker_info", cache_key, fast_info)
                return fast_info, None
            last_error = f"Yahoo returned no company profile data for {ticker}."
        except Exception as exc:
            logger.warning("fetch_ticker_info attempt %d failed (%s): %s", attempt + 1, ticker, exc)
            last_error = summarize_fetch_error(exc)
        if attempt < attempts - 1:
            time.sleep(0.35 * (attempt + 1))

    stale_info = get_cached_fetch_payload(
        "ticker_info",
        cache_key,
        max_age_seconds=FETCH_STALE_FALLBACK_TTL_SECONDS,
    )
    if stale_info:
        return stale_info, None
    return {}, last_error


def fetch_ticker_news_with_retry(ticker, attempts=2):
    """
    Fetch recent news items for *ticker* with retry and stale-cache fallback.

    Returns (news_list, error_string_or_None).
    """
    cache_key = ticker.upper()
    cached_news = get_cached_fetch_payload("ticker_news", cache_key)
    if cached_news:
        return cached_news, None

    last_error = None
    for attempt in range(attempts):
        try:
            news = normalize_news_payload(getattr(yf.Ticker(ticker), "news", []) or [])
            if news:
                set_cached_fetch_payload("ticker_news", cache_key, news)
                return news, None
            info_fallback = normalize_info_payload(fetch_ticker_info_with_retry(ticker, attempts=1)[0] or {})
            company_name = str(info_fallback.get("shortName") or info_fallback.get("longName") or ticker).strip()
            if company_name:
                last_error = f"Yahoo returned no recent news items for {ticker} ({company_name})."
            else:
                last_error = f"Yahoo returned no recent news items for {ticker}."
        except Exception as exc:
            logger.warning("fetch_ticker_news attempt %d failed (%s): %s", attempt + 1, ticker, exc)
            last_error = summarize_fetch_error(exc)
        if attempt < attempts - 1:
            time.sleep(0.25 * (attempt + 1))

    stale_news = get_cached_fetch_payload(
        "ticker_news",
        cache_key,
        max_age_seconds=FETCH_STALE_FALLBACK_TTL_SECONDS,
    )
    if stale_news:
        return stale_news, None
    return [], last_error


# ---------------------------------------------------------------------------
# Catalyst calendar
# ---------------------------------------------------------------------------

def fetch_calendar_events(ticker: str) -> dict:
    """
    Return upcoming earnings and dividend dates for *ticker* from yfinance.

    Dict keys: earnings_date, ex_div_date, dividend_date (each a
    datetime.date or None).  Never raises — returns all-None on any failure.
    """
    cache_key = ticker.upper()
    cached = get_cached_fetch_payload("calendar", cache_key, max_age_seconds=CATALYST_CACHE_TTL)
    if cached is not None:
        return cached

    result = {"earnings_date": None, "ex_div_date": None, "dividend_date": None}
    try:
        cal = yf.Ticker(ticker).calendar or {}
        if not isinstance(cal, dict):
            cal = {}

        earnings_raw = cal.get("Earnings Date") or cal.get("earningsDate")
        if earnings_raw:
            if not isinstance(earnings_raw, (list, tuple)):
                earnings_raw = [earnings_raw]
            for entry in earnings_raw:
                try:
                    if hasattr(entry, "date"):
                        result["earnings_date"] = entry.date()
                    elif isinstance(entry, datetime.date):
                        result["earnings_date"] = entry
                    break
                except Exception:
                    pass

        for key_name, result_key in (("Ex-Dividend Date", "ex_div_date"), ("Dividend Date", "dividend_date")):
            raw = cal.get(key_name)
            if raw is not None:
                try:
                    if hasattr(raw, "date"):
                        result[result_key] = raw.date()
                    elif isinstance(raw, datetime.date):
                        result[result_key] = raw
                except Exception:
                    pass
    except Exception as exc:
        logger.debug("fetch_calendar_events failed for %s: %s", ticker, exc)

    set_cached_fetch_payload("calendar", cache_key, result)
    return result


# ---------------------------------------------------------------------------
# Macro indicators
# ---------------------------------------------------------------------------

def _fetch_fred_series_last_value(series_id: str) -> float | None:
    """Download the last non-NaN value from a FRED public CSV series."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw = resp.read().decode("utf-8")
        df = pd.read_csv(io.StringIO(raw))
        col = df.columns[-1]
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            return None
        return float(series.iloc[-1])
    except Exception as exc:
        logger.debug("FRED fetch for %s failed: %s", series_id, exc)
        return None


def _fetch_yf_last_close(symbol: str) -> float | None:
    """Return the most recent close price for a yfinance symbol."""
    try:
        hist = yf.Ticker(symbol).history(period="5d")
        if hist.empty:
            return None
        return float(hist["Close"].dropna().iloc[-1])
    except Exception as exc:
        logger.debug("yfinance close fetch failed for %s: %s", symbol, exc)
        return None


def fetch_macro_indicators() -> dict:
    """
    Return a snapshot of key macro indicators used for regime assessment.

    Dict keys: two_ten_spread, hy_oas_bps, vix, vix3m, vix_ratio, dxy.
    All values are float or None.  Never raises.
    """
    cache_key = "latest"
    cached = get_cached_fetch_payload("macro_indicators", cache_key, max_age_seconds=MACRO_CACHE_TTL)
    if cached is not None:
        return cached

    two_ten = _fetch_fred_series_last_value("T10Y2Y")
    hy_oas = _fetch_fred_series_last_value("BAMLH0A0HYM2")
    vix = _fetch_yf_last_close("^VIX")
    vix3m = _fetch_yf_last_close("^VIX3M")
    dxy = _fetch_yf_last_close("DX-Y.NYB")

    vix_ratio = None
    if vix is not None and vix3m is not None and vix3m > 0:
        vix_ratio = round(vix / vix3m, 3)

    result = {
        "two_ten_spread": two_ten,
        "hy_oas_bps": hy_oas,
        "vix": vix,
        "vix3m": vix3m,
        "vix_ratio": vix_ratio,
        "dxy": dxy,
    }
    set_cached_fetch_payload("macro_indicators", cache_key, result)
    return result


def fetch_earnings_trend_with_retry(ticker, attempts=2):
    """
    Fetch the EPS estimate trend DataFrame for *ticker* with retry and
    stale-cache fallback.

    Returns (DataFrame_or_None, error_string_or_None).
    The DataFrame has period labels as its index (e.g. '0q', '+1q', '0y',
    '+1y') and columns that include epsTrend.current / epsTrend.30daysAgo /
    epsTrend.90daysAgo and epsRevisions.upLast30days /
    epsRevisions.downLast30days, depending on the yfinance version.
    """
    cache_key = ticker.upper()
    cached = get_cached_fetch_payload("earnings_trend", cache_key)
    if cached is not None and not cached.empty:
        return cached, None

    last_error = None
    for attempt in range(attempts):
        try:
            trend = getattr(yf.Ticker(ticker), "earnings_trend", None)
            if trend is not None and isinstance(trend, pd.DataFrame) and not trend.empty:
                set_cached_fetch_payload("earnings_trend", cache_key, trend)
                return trend, None
            last_error = f"Yahoo returned no earnings trend data for {ticker}."
        except Exception as exc:
            logger.warning("fetch_earnings_trend attempt %d failed (%s): %s", attempt + 1, ticker, exc)
            last_error = summarize_fetch_error(exc)
        if attempt < attempts - 1:
            time.sleep(0.35 * (attempt + 1))

    stale = get_cached_fetch_payload(
        "earnings_trend",
        cache_key,
        max_age_seconds=FETCH_STALE_FALLBACK_TTL_SECONDS,
    )
    if stale is not None and not stale.empty:
        return stale, None
    return None, last_error


def fetch_annual_financials_with_retry(ticker, attempts=2):
    """
    Fetch annual financial statements for *ticker* with retry and stale-cache
    fallback.

    Returns (balance_sheet_df, income_df, cashflow_df, error_string_or_None).
    Each DataFrame has financial line items as the index and fiscal-year dates
    as columns (most recent year first).  Any DataFrame may be None or empty
    when data is unavailable.
    """
    cache_key = ticker.upper()

    cached_bs  = get_cached_fetch_payload("annual_balance_sheet",  cache_key)
    cached_inc = get_cached_fetch_payload("annual_income_stmt",    cache_key)
    cached_cf  = get_cached_fetch_payload("annual_cashflow",       cache_key)
    if cached_bs is not None and cached_inc is not None and cached_cf is not None:
        return cached_bs, cached_inc, cached_cf, None

    last_error = None
    for attempt in range(attempts):
        try:
            handle = yf.Ticker(ticker)
            bs  = getattr(handle, "balance_sheet",  None)
            inc = getattr(handle, "income_stmt",    None)
            cf  = getattr(handle, "cash_flow",      None)
            # Normalise: keep only DataFrames with at least 2 columns (2 fiscal years)
            def _valid(df):
                return isinstance(df, pd.DataFrame) and not df.empty and len(df.columns) >= 1
            bs  = bs  if _valid(bs)  else pd.DataFrame()
            inc = inc if _valid(inc) else pd.DataFrame()
            cf  = cf  if _valid(cf)  else pd.DataFrame()
            set_cached_fetch_payload("annual_balance_sheet",  cache_key, bs)
            set_cached_fetch_payload("annual_income_stmt",    cache_key, inc)
            set_cached_fetch_payload("annual_cashflow",       cache_key, cf)
            return bs, inc, cf, None
        except Exception as exc:
            logger.warning("fetch_annual_financials attempt %d failed (%s): %s", attempt + 1, ticker, exc)
            last_error = summarize_fetch_error(exc)
        if attempt < attempts - 1:
            time.sleep(0.35 * (attempt + 1))

    # Stale fallback
    stale_bs  = get_cached_fetch_payload("annual_balance_sheet",  cache_key, max_age_seconds=FETCH_STALE_FALLBACK_TTL_SECONDS) or pd.DataFrame()
    stale_inc = get_cached_fetch_payload("annual_income_stmt",    cache_key, max_age_seconds=FETCH_STALE_FALLBACK_TTL_SECONDS) or pd.DataFrame()
    stale_cf  = get_cached_fetch_payload("annual_cashflow",       cache_key, max_age_seconds=FETCH_STALE_FALLBACK_TTL_SECONDS) or pd.DataFrame()
    return stale_bs, stale_inc, stale_cf, last_error


def fetch_options_data_with_retry(ticker, attempts=2):
    """
    Fetch options chain data for the nearest 2 expirations (skipping any that
    expire within 5 days) with retry and stale-cache fallback.

    Returns (payload_or_None, error_string_or_None).

    Payload shape:
        {
            "expirations": ["YYYY-MM-DD", ...],
            "chains": {
                "YYYY-MM-DD": {"calls": pd.DataFrame, "puts": pd.DataFrame},
                ...
            }
        }

    Returns (None, error) for tickers with no listed options or on failure.
    """
    import datetime as _dt

    cache_key = ticker.upper()
    cached = get_cached_fetch_payload("options_data", cache_key)
    if cached is not None:
        return cached, None

    last_error = None
    for attempt in range(attempts):
        try:
            handle = yf.Ticker(ticker)
            raw_exps = getattr(handle, "options", None) or []
            if not raw_exps:
                last_error = f"No listed options for {ticker}."
                break

            today = _dt.date.today()
            min_date = today + _dt.timedelta(days=5)
            valid_exps = [
                e for e in raw_exps
                if _dt.date.fromisoformat(e) >= min_date
            ]
            if not valid_exps:
                last_error = f"All expirations for {ticker} are within 5 days."
                break

            selected = valid_exps[:2]
            chains = {}
            for exp in selected:
                chain = handle.option_chain(exp)
                calls = chain.calls if hasattr(chain, "calls") and isinstance(chain.calls, pd.DataFrame) else pd.DataFrame()
                puts  = chain.puts  if hasattr(chain, "puts")  and isinstance(chain.puts,  pd.DataFrame) else pd.DataFrame()
                chains[exp] = {"calls": calls, "puts": puts}

            payload = {"expirations": selected, "chains": chains}
            set_cached_fetch_payload("options_data", cache_key, payload)
            return payload, None

        except Exception as exc:
            logger.warning("fetch_options_data attempt %d failed (%s): %s", attempt + 1, ticker, exc)
            last_error = summarize_fetch_error(exc)
        if attempt < attempts - 1:
            time.sleep(0.3 * (attempt + 1))

    stale = get_cached_fetch_payload(
        "options_data", cache_key, max_age_seconds=FETCH_STALE_FALLBACK_TTL_SECONDS
    )
    if stale is not None:
        return stale, None
    return None, last_error


def _pick_column_value(df_row, *name_variants):
    """Return the first non-None numeric value found among *name_variants* in *df_row*."""
    for name in name_variants:
        value = df_row.get(name)
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def compute_eps_revision_signal(trend_df):
    """
    Derive 4-week and 12-week EPS revision magnitude plus analyst breadth from
    a yfinance earnings_trend DataFrame.

    Returns (eps_rev_4w, eps_rev_12w, breadth_4w, revision_signal) where:
      - eps_rev_4w / eps_rev_12w are fractional changes in the consensus EPS
        estimate over ~4 and ~12 weeks respectively (positive = upward revision)
      - breadth_4w is the fraction of analysts revising upward over 4 weeks
        (0–1), or None if not available
      - revision_signal is a float in [-1.0, 1.0] ready to add to
        effective_f_score

    Gracefully returns (None, None, None, 0.0) when data is unavailable.
    """
    if trend_df is None or not isinstance(trend_df, pd.DataFrame) or trend_df.empty:
        return None, None, None, 0.0

    # Prefer the current fiscal year row ("0y"); fall back to next year ("+1y")
    row = None
    for period_label in ("0y", "+1y"):
        if period_label in trend_df.index:
            row = trend_df.loc[period_label]
            break
    if row is None:
        # Last resort: use whatever row is available
        row = trend_df.iloc[0]

    # Convert Series to dict for uniform access
    row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)

    # --- Magnitude: consensus EPS estimate shift ---
    current = _pick_column_value(row_dict, "epsTrend.current", "current")
    ago_30  = _pick_column_value(row_dict, "epsTrend.30daysAgo", "30daysAgo")
    ago_90  = _pick_column_value(row_dict, "epsTrend.90daysAgo", "90daysAgo")

    eps_rev_4w = None
    eps_rev_12w = None
    if current is not None and ago_30 is not None and abs(ago_30) > 1e-6:
        eps_rev_4w = (current - ago_30) / abs(ago_30)
    if current is not None and ago_90 is not None and abs(ago_90) > 1e-6:
        eps_rev_12w = (current - ago_90) / abs(ago_90)

    # --- Breadth: fraction of analysts revising up over ~4 weeks ---
    breadth_4w = None
    up30   = _pick_column_value(row_dict, "epsRevisions.upLast30days",   "upLast30days")
    down30 = _pick_column_value(row_dict, "epsRevisions.downLast30days", "downLast30days")
    if up30 is not None and down30 is not None:
        total = up30 + down30
        if total > 0:
            breadth_4w = up30 / total

    # --- Aggregate into a single ±1.0 signal ---
    signal = 0.0
    if has_numeric_value(eps_rev_4w):
        if eps_rev_4w >= 0.03:
            signal += 0.35
        elif eps_rev_4w <= -0.03:
            signal -= 0.35
    if has_numeric_value(eps_rev_12w):
        if eps_rev_12w >= 0.05:
            signal += 0.40
        elif eps_rev_12w <= -0.05:
            signal -= 0.40
    if has_numeric_value(breadth_4w):
        if breadth_4w >= 0.60:
            signal += 0.25
        elif breadth_4w <= 0.30:
            signal -= 0.25

    return eps_rev_4w, eps_rev_12w, breadth_4w, float(np.clip(signal, -1.0, 1.0))


def fetch_ticker_calendar_with_retry(ticker, attempts=2):
    """
    Fetch the earnings and ex-dividend calendar for *ticker* via yfinance.

    Returns (dict_or_None, error_string_or_None).
    Returned dict keys (all values may be None):
        earnings_date — next expected earnings date as datetime.date
        ex_div_date   — next ex-dividend date as datetime.date
    Cached for 24 hours (earnings dates rarely change intraday).
    """
    import datetime as _dt

    cache_key = ticker.upper()
    cached = get_cached_fetch_payload("ticker_calendar", cache_key, max_age_seconds=86400)
    if cached is not None:
        return cached, None

    last_error = None
    for attempt in range(attempts):
        try:
            cal = getattr(yf.Ticker(ticker), "calendar", None)
            if cal is None:
                last_error = f"No calendar data available for {ticker}."
                break

            # yfinance may return a dict or a single-row DataFrame
            if isinstance(cal, pd.DataFrame):
                cal_dict = {}
                for col in cal.columns:
                    vals = cal[col].dropna().tolist()
                    cal_dict[col] = vals[0] if vals else None
                cal = cal_dict

            def _to_date(val):
                if val is None:
                    return None
                if isinstance(val, (list, tuple)):
                    val = val[0] if val else None
                if val is None:
                    return None
                try:
                    ts = pd.Timestamp(val)
                    return None if pd.isna(ts) else ts.date()
                except Exception:
                    return None

            result = {
                "earnings_date": _to_date(cal.get("Earnings Date")),
                "ex_div_date":   _to_date(cal.get("Ex-Dividend Date")),
            }
            set_cached_fetch_payload("ticker_calendar", cache_key, result)
            return result, None
        except Exception as exc:
            logger.warning("fetch_ticker_calendar attempt %d failed (%s): %s", attempt + 1, ticker, exc)
            last_error = summarize_fetch_error(exc)
        if attempt < attempts - 1:
            time.sleep(0.35 * (attempt + 1))

    stale = get_cached_fetch_payload("ticker_calendar", cache_key, max_age_seconds=FETCH_STALE_FALLBACK_TTL_SECONDS)
    if stale is not None:
        return stale, None
    return None, last_error


def fetch_macro_indicators(lookback_days=400):
    """
    Fetch macro regime indicators from FRED (VIX, 2s10s yield spread, HY OAS,
    DXY) using the free FRED CSV endpoint.  No API key required.

    Returns a dict keyed by short name:
        {
            "VIX":    {"value": float, "pct_rank": float, "prev": float, "error": str_or_None},
            "2s10s":  {...},
            "HY_OAS": {...},
            "DXY":    {...},
        }
    Cached for 6 hours (FRED series update once per day).
    """
    import datetime as _dt
    import io
    from constants import MACRO_FRED_SERIES

    cache_key = "daily"
    cached = get_cached_fetch_payload("macro_indicators", cache_key, max_age_seconds=21600)
    if cached is not None:
        return cached

    try:
        import requests as _req
    except ImportError:
        err = "requests library not installed; macro data unavailable."
        return {name: {"value": None, "pct_rank": None, "prev": None, "error": err} for name in MACRO_FRED_SERIES}

    start_date = (_dt.date.today() - _dt.timedelta(days=lookback_days)).isoformat()
    result = {}
    for short_name, series_id in MACRO_FRED_SERIES.items():
        try:
            url = (
                f"https://fred.stlouisfed.org/graph/fredgraph.csv"
                f"?id={series_id}&observation_start={start_date}"
            )
            response = _req.get(url, timeout=15)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.text), na_values=".")
            df.columns = ["DATE", "VALUE"]
            df = df.dropna(subset=["VALUE"]).reset_index(drop=True)
            if df.empty:
                result[short_name] = {"value": None, "pct_rank": None, "prev": None,
                                      "error": f"No data returned for {series_id}."}
                continue
            latest = float(df["VALUE"].iloc[-1])
            prev = float(df["VALUE"].iloc[-2]) if len(df) >= 2 else None
            year_slice = df["VALUE"].tail(252)
            pct_rank = float((year_slice < latest).mean() * 100) if len(year_slice) >= 10 else None
            result[short_name] = {"value": latest, "pct_rank": pct_rank, "prev": prev, "error": None}
        except Exception as exc:
            result[short_name] = {"value": None, "pct_rank": None, "prev": None,
                                  "error": summarize_fetch_error(exc)}

    if result:
        set_cached_fetch_payload("macro_indicators", cache_key, result)
    return result
