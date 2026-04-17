# -*- coding: utf-8 -*-
"""
analytics_tech.py — Pure technical-indicator computations.

Functions:
    calculate_realized_volatility
    calculate_trend_strength
    calculate_52w_context
    calculate_quality_score
    calculate_dividend_safety_score
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fetch import has_numeric_value, safe_divide


def compute_relative_strength(
    close: pd.Series,
    benchmark_close: pd.Series,
    window: int,
) -> float | None:
    if close is None or benchmark_close is None or window <= 0:
        return None
    close = close.dropna()
    benchmark_close = benchmark_close.dropna()
    aligned = pd.concat([close.rename("stock"), benchmark_close.rename("benchmark")], axis=1, join="inner").dropna()
    if len(aligned) <= window:
        return None
    stock_return = safe_divide(aligned["stock"].iloc[-1] - aligned["stock"].iloc[-window - 1], aligned["stock"].iloc[-window - 1])
    benchmark_return = safe_divide(
        aligned["benchmark"].iloc[-1] - aligned["benchmark"].iloc[-window - 1],
        aligned["benchmark"].iloc[-window - 1],
    )
    if stock_return is None or benchmark_return is None:
        return None
    return float(stock_return - benchmark_return)


def calculate_realized_volatility(close: pd.Series, window: int) -> float | None:
    """Return annualised realised volatility over the last *window* periods."""
    if close is None or len(close) < max(window, 2):
        return None
    returns = close.pct_change().dropna()
    if len(returns) < window:
        return None
    return float(returns.tail(window).std() * np.sqrt(252))


def calculate_trend_strength(
    price: float | None,
    sma50: float | None,
    sma200: float | None,
    momentum_1y: float | None = None,
) -> float | None:
    """
    Return a composite trend-strength score in the range [-100, 100].

    Combines price-vs-200SMA distance, SMA50-vs-SMA200 distance, and
    1-year momentum into a single directional quality signal.
    """
    components = []
    if has_numeric_value(price) and has_numeric_value(sma200) and sma200 > 0:
        components.append(np.clip((price / sma200 - 1) * 100, -25, 25))
    if has_numeric_value(sma50) and has_numeric_value(sma200) and sma200 > 0:
        components.append(np.clip((sma50 / sma200 - 1) * 120, -25, 25))
    if has_numeric_value(momentum_1y):
        components.append(np.clip(momentum_1y * 80, -25, 25))
    if not components:
        return None
    return float(np.clip(sum(components), -100, 100))


def calculate_52w_context(
    close: pd.Series,
) -> tuple[float | None, float | None, float | None]:
    """
    Return (range_position, distance_from_high, distance_from_low) over
    the trailing 252-bar window.

    range_position  — 0.0 at 52w low, 1.0 at 52w high
    distance_high   — negative pct from the 52w high (e.g. -0.15 = 15% below)
    distance_low    — positive pct above the 52w low
    """
    if close is None or close.empty:
        return None, None, None
    window = close.tail(min(len(close), 252))
    rolling_high = close._get_numeric_data().max() if hasattr(close, "_get_numeric_data") else float(window.max())
    rolling_low = float(window.min())
    rolling_high = float(window.max())
    price = float(window.iloc[-1])
    if not has_numeric_value(price) or not has_numeric_value(rolling_high) or not has_numeric_value(rolling_low):
        return None, None, None
    range_position = safe_divide(price - rolling_low, rolling_high - rolling_low)
    distance_high = safe_divide(price - rolling_high, rolling_high)
    distance_low = safe_divide(price - rolling_low, rolling_low)
    return range_position, distance_high, distance_low


def calculate_quality_score(
    roe: float | None,
    margins: float | None,
    debt_eq: float | None,
    revenue_growth: float | None,
    earnings_growth: float | None,
    current_ratio: float | None,
    settings: dict[str, Any],
) -> float:
    """
    Return a quality score in the range [-4, 5].

    Combines ROE, profit margin, leverage, growth consistency, and
    current ratio using the active *settings* thresholds.
    """
    score = 0.0
    if has_numeric_value(roe):
        score += 1.25 if roe >= settings["fund_roe_threshold"] else (-1 if roe < 0 else 0)
    if has_numeric_value(margins):
        score += 1.25 if margins >= settings["fund_profit_margin_threshold"] else (-1 if margins < 0 else 0)
    if has_numeric_value(debt_eq):
        if 0 <= debt_eq < settings["fund_debt_good_threshold"]:
            score += 1.0
        elif debt_eq > settings["fund_debt_bad_threshold"]:
            score -= 1.0
    if has_numeric_value(revenue_growth) and has_numeric_value(earnings_growth):
        if revenue_growth > 0 and earnings_growth > 0:
            score += 1.0
        elif revenue_growth < 0 and earnings_growth < 0:
            score -= 1.0
    elif has_numeric_value(revenue_growth):
        score += 0.5 if revenue_growth > 0 else -0.5
    elif has_numeric_value(earnings_growth):
        score += 0.5 if earnings_growth > 0 else -0.5
    if has_numeric_value(current_ratio):
        if current_ratio >= settings["fund_current_ratio_good"]:
            score += 0.5
        elif current_ratio < settings["fund_current_ratio_bad"]:
            score -= 0.5
    return float(np.clip(score, -4, 5))


def calculate_dividend_safety_score(
    dividend_yield: float | None,
    payout_ratio: float | None,
    margins: float | None,
    current_ratio: float | None,
    debt_eq: float | None,
) -> float:
    """
    Return a dividend-safety score in the range [-3, 4].

    Rewards sustainable yield, covered payout, positive margins, adequate
    liquidity, and manageable leverage.
    """
    score = 0.0
    if has_numeric_value(dividend_yield):
        if dividend_yield >= 0.06:
            score += 0.5
        elif dividend_yield >= 0.025:
            score += 1.0
    if has_numeric_value(payout_ratio):
        if 0 < payout_ratio <= 0.65:
            score += 2.0
        elif payout_ratio <= 0.85:
            score += 1.0
        elif payout_ratio > 1.0:
            score -= 2.0
        elif payout_ratio > 0.85:
            score -= 1.0
    if has_numeric_value(margins) and margins > 0:
        score += 0.5
    if has_numeric_value(current_ratio):
        score += 0.5 if current_ratio >= 1.0 else -0.5
    if has_numeric_value(debt_eq):
        if debt_eq < 120:
            score += 0.5
        elif debt_eq > 220:
            score -= 0.5
    return float(np.clip(score, -3, 4))
