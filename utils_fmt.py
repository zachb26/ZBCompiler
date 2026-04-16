# -*- coding: utf-8 -*-
"""
utils_fmt.py — Stateless formatting, string, and number helper functions.

All functions here are pure (no side effects, no I/O, no streamlit).
They are imported widely across the codebase.

Functions defined here (per module spec):
  normalize_ticker, safe_num, get_color, escape_markdown_text,
  colorize_markdown_text, tone_to_color, format_value, format_percent,
  format_int, format_market_cap, escape_html_text,
  parse_ticker_list, has_numeric_value, safe_divide, safe_json_loads,
  calculate_rsi, score_relative_multiple, extract_sentiment_tokens
"""

import copy
import json
import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Ticker / text normalization
# ---------------------------------------------------------------------------

def normalize_ticker(value):
    """Strip whitespace and upper-case a raw ticker string."""
    return str(value or "").strip().upper()


def parse_ticker_list(raw_text):
    """Split a free-form ticker string into a deduplicated list of normalized tickers.

    Accepts comma, space, or newline separators.
    """
    tickers = []
    seen = set()
    normalized = raw_text.replace("\n", ",").replace(" ", ",")
    for token in normalized.split(","):
        ticker = normalize_ticker(token)
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        tickers.append(ticker)
    return tickers


# ---------------------------------------------------------------------------
# Safe numeric helpers
# ---------------------------------------------------------------------------

def safe_num(value):
    """Coerce *value* to a Python float, returning None on failure."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        if value.lower() in ["n/a", "none", "nan", "inf"]:
            return None
        try:
            return float(value.replace("%", "").replace(",", "").strip())
        except ValueError:
            return None
    return None


def has_numeric_value(value):
    """Return True when *value* is a real (non-null, non-NaN) number."""
    return value is not None and not pd.isna(value)


def safe_divide(numerator, denominator):
    """Divide *numerator* by *denominator*, returning None on zero or missing denominator."""
    if denominator is None or pd.isna(denominator) or abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def safe_json_loads(value, default=None):
    """Parse a JSON string, returning a deep copy of *default* on failure.

    If *value* is already a dict or list it is deep-copied and returned directly.
    """
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
# Formatting helpers
# ---------------------------------------------------------------------------

def format_value(value, fmt="{:,.2f}", suffix=""):
    """Format a numeric *value* with *fmt* and an optional *suffix*.

    Returns the string ``"N/A"`` when *value* is None or NaN.
    """
    if value is None or pd.isna(value):
        return "N/A"
    return f"{fmt.format(value)}{suffix}"


def format_percent(value):
    """Format a decimal fraction as a percentage string (e.g. 0.12 → "12.0%").

    Returns ``"N/A"`` when *value* is None or NaN.
    """
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.1f}%"


def format_int(value):
    """Format *value* as an integer string.

    Returns ``"N/A"`` when *value* is None or NaN.
    """
    if value is None or pd.isna(value):
        return "N/A"
    return str(int(value))


def format_market_cap(value):
    """Format a raw market-cap number with T / B / M suffix.

    Returns ``"N/A"`` when *value* is None or NaN.
    """
    if value is None or pd.isna(value):
        return "N/A"
    value = float(value)
    if abs(value) >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"${value:,.0f}"


# ---------------------------------------------------------------------------
# Colour / tone helpers
# ---------------------------------------------------------------------------

def get_color(verdict):
    """Map a verdict string to a basic colour name (green / red / gray)."""
    if "STRONG BUY" in verdict or verdict in {"BUY", "STRONG", "UNDERVALUED", "POSITIVE"}:
        return "green"
    if "STRONG SELL" in verdict or verdict in {"SELL", "WEAK", "OVERVALUED", "NEGATIVE"}:
        return "red"
    return "gray"


def tone_to_color(tone):
    """Convert a tone string (good / bad / neutral) to a colour name."""
    return {
        "good": "green",
        "bad": "red",
        "neutral": "gray",
    }.get(str(tone or "neutral"), "gray")


# ---------------------------------------------------------------------------
# Markdown / HTML helpers
# ---------------------------------------------------------------------------

def escape_markdown_text(value):
    """Escape the minimum Markdown characters needed to display *value* safely."""
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def colorize_markdown_text(value, color):
    """Wrap *value* in a Streamlit colour shortcode (e.g. ``:green[...]``)."""
    safe_text = escape_markdown_text(value)
    if color == "green":
        return f":green[{safe_text}]"
    if color == "red":
        return f":red[{safe_text}]"
    if color in {"gray", "grey"}:
        return f":gray[{safe_text}]"
    return safe_text


def escape_html_text(value):
    """Escape HTML special characters in *value*."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# ---------------------------------------------------------------------------
# Technical analysis helpers
# ---------------------------------------------------------------------------

def calculate_rsi(close, period=14):
    """Compute RSI for a price series using exponential smoothing.

    Parameters
    ----------
    close:
        Pandas Series of closing prices.
    period:
        Look-back window (default 14).

    Returns
    -------
    pandas.Series
        RSI values on the same index as *close*.
    """
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
    """Score a valuation multiple relative to its benchmark.

    Returns +1 (cheap), 0 (neutral), or -1 (expensive / missing).
    """
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
    """Return the set of lower-case word tokens found in *text*."""
    return set(re.findall(r"[a-z]+", (text or "").lower()))
