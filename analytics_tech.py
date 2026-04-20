# -*- coding: utf-8 -*-
"""
analytics_tech.py — Pure technical-indicator computations.

Functions:
    calculate_realized_volatility
    calculate_trend_strength
    calculate_52w_context
    calculate_quality_score
    calculate_dividend_safety_score
    compute_piotroski_fscore
    compute_altman_zscore
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


# ---------------------------------------------------------------------------
# Piotroski F-Score
# ---------------------------------------------------------------------------

def _stmt_value(df: pd.DataFrame, row_labels: list[str], col_idx: int = 0):
    """
    Return the first available numeric value from *df* matching any of
    *row_labels*, at column position *col_idx* (0 = most recent year).
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    cols = df.columns
    if col_idx >= len(cols):
        return None
    col = cols[col_idx]
    for label in row_labels:
        if label in df.index:
            val = df.loc[label, col]
            if val is not None and not (isinstance(val, float) and pd.isna(val)):
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
    return None


def compute_piotroski_fscore(
    info: dict,
    balance_sheet: pd.DataFrame | None = None,
    income_stmt: pd.DataFrame | None = None,
    cashflow_stmt: pd.DataFrame | None = None,
) -> tuple[int | None, dict]:
    """
    Compute the Piotroski F-Score (0–9) from nine binary tests.

    Each test returns 1 (pass), 0 (fail), or None (not evaluable due to missing
    data).  The composite score is the sum of all non-None results.  Returns
    (None, breakdown) when fewer than 4 tests can be evaluated.

    Profitability (F1–F4):
        F1  ROA > 0
        F2  Operating cash flow > 0
        F3  ΔROA > 0 (YoY improvement)
        F4  Accrual quality: OCF / Total Assets > ROA

    Leverage / Liquidity (F5–F7):
        F5  Long-term debt ratio decreased YoY
        F6  Current ratio improved YoY
        F7  No new shares issued YoY

    Operating Efficiency (F8–F9):
        F8  Gross margin improved YoY
        F9  Asset turnover improved YoY
    """
    from utils_fmt import safe_num  # local import to avoid circular deps

    breakdown: dict[str, int | None] = {}

    # -- F1: ROA > 0 ----------------------------------------------------------
    roa = safe_num(info.get("returnOnAssets"))
    breakdown["F1_ROA_Positive"] = (1 if roa > 0 else 0) if has_numeric_value(roa) else None

    # -- F2: Operating Cash Flow > 0 ------------------------------------------
    ocf = safe_num(info.get("operatingCashflow"))
    breakdown["F2_OCF_Positive"] = (1 if ocf > 0 else 0) if has_numeric_value(ocf) else None

    # -- F3: ΔROA > 0 (YoY) ---------------------------------------------------
    f3: int | None = None
    net_inc_curr = _stmt_value(income_stmt, ["Net Income", "Net Income Common Stockholders"], 0)
    net_inc_prev = _stmt_value(income_stmt, ["Net Income", "Net Income Common Stockholders"], 1)
    ta_curr = _stmt_value(balance_sheet, ["Total Assets"], 0)
    ta_prev = _stmt_value(balance_sheet, ["Total Assets"], 1)
    if (has_numeric_value(net_inc_curr) and has_numeric_value(ta_curr) and ta_curr != 0
            and has_numeric_value(net_inc_prev) and has_numeric_value(ta_prev) and ta_prev != 0):
        roa_curr = net_inc_curr / ta_curr
        roa_prev = net_inc_prev / ta_prev
        f3 = 1 if roa_curr > roa_prev else 0
    elif has_numeric_value(info.get("earningsGrowth")):
        eg = safe_num(info.get("earningsGrowth"))
        f3 = 1 if (has_numeric_value(eg) and eg > 0) else 0
    breakdown["F3_Delta_ROA"] = f3

    # -- F4: Accrual quality: OCF/Assets > ROA --------------------------------
    f4: int | None = None
    total_assets = safe_num(info.get("totalAssets")) if not has_numeric_value(ta_curr) else ta_curr
    if has_numeric_value(ocf) and has_numeric_value(total_assets) and total_assets != 0 and has_numeric_value(roa):
        f4 = 1 if (ocf / total_assets > roa) else 0
    breakdown["F4_Accrual_Quality"] = f4

    # -- F5: Long-term debt ratio decreased YoY --------------------------------
    f5: int | None = None
    ltd_curr = _stmt_value(balance_sheet, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], 0)
    ltd_prev = _stmt_value(balance_sheet, ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"], 1)
    if (has_numeric_value(ltd_curr) and has_numeric_value(ta_curr) and ta_curr != 0
            and has_numeric_value(ltd_prev) and has_numeric_value(ta_prev) and ta_prev != 0):
        lev_curr = ltd_curr / ta_curr
        lev_prev = ltd_prev / ta_prev
        f5 = 1 if lev_curr <= lev_prev else 0
    breakdown["F5_Leverage_Decreased"] = f5

    # -- F6: Current ratio improved YoY --------------------------------------
    f6: int | None = None
    ca_curr = _stmt_value(balance_sheet, ["Current Assets", "Total Current Assets"], 0)
    cl_curr = _stmt_value(balance_sheet, ["Current Liabilities", "Total Current Liabilities", "Current Liabilities Net Minority Interest"], 0)
    ca_prev = _stmt_value(balance_sheet, ["Current Assets", "Total Current Assets"], 1)
    cl_prev = _stmt_value(balance_sheet, ["Current Liabilities", "Total Current Liabilities", "Current Liabilities Net Minority Interest"], 1)
    if (has_numeric_value(ca_curr) and has_numeric_value(cl_curr) and cl_curr != 0
            and has_numeric_value(ca_prev) and has_numeric_value(cl_prev) and cl_prev != 0):
        cr_curr = ca_curr / cl_curr
        cr_prev = ca_prev / cl_prev
        f6 = 1 if cr_curr > cr_prev else 0
    breakdown["F6_Current_Ratio_Improved"] = f6

    # -- F7: No new share dilution --------------------------------------------
    f7: int | None = None
    shares_curr = _stmt_value(balance_sheet, ["Share Issued", "Ordinary Shares Number", "Common Stock"], 0)
    shares_prev = _stmt_value(balance_sheet, ["Share Issued", "Ordinary Shares Number", "Common Stock"], 1)
    if has_numeric_value(shares_curr) and has_numeric_value(shares_prev) and shares_prev > 0:
        f7 = 1 if shares_curr <= shares_prev * 1.01 else 0
    breakdown["F7_No_Dilution"] = f7

    # -- F8: Gross margin improved YoY ----------------------------------------
    f8: int | None = None
    gp_curr = _stmt_value(income_stmt, ["Gross Profit"], 0)
    rev_curr = _stmt_value(income_stmt, ["Total Revenue"], 0)
    gp_prev = _stmt_value(income_stmt, ["Gross Profit"], 1)
    rev_prev = _stmt_value(income_stmt, ["Total Revenue"], 1)
    if (has_numeric_value(gp_curr) and has_numeric_value(rev_curr) and rev_curr != 0
            and has_numeric_value(gp_prev) and has_numeric_value(rev_prev) and rev_prev != 0):
        gm_curr = gp_curr / rev_curr
        gm_prev = gp_prev / rev_prev
        f8 = 1 if gm_curr > gm_prev else 0
    breakdown["F8_Gross_Margin_Improved"] = f8

    # -- F9: Asset turnover improved YoY -------------------------------------
    f9: int | None = None
    if (has_numeric_value(rev_curr) and has_numeric_value(ta_curr) and ta_curr != 0
            and has_numeric_value(rev_prev) and has_numeric_value(ta_prev) and ta_prev != 0):
        at_curr = rev_curr / ta_curr
        at_prev = rev_prev / ta_prev
        f9 = 1 if at_curr > at_prev else 0
    breakdown["F9_Asset_Turnover_Improved"] = f9

    tests = [breakdown[k] for k in breakdown]
    evaluable = [t for t in tests if t is not None]
    if len(evaluable) < 4:
        return None, breakdown

    score = int(sum(evaluable))
    return score, breakdown


# ---------------------------------------------------------------------------
# Altman Z-Score
# ---------------------------------------------------------------------------

def compute_altman_zscore(info: dict) -> tuple[float | None, str | None]:
    """
    Compute the Altman Z-Score for public companies.

    Formula: Z = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5

        X1 = Working Capital / Total Assets
        X2 = Retained Earnings / Total Assets
        X3 = EBIT / Total Assets  (operating income; fallback to EBITDA)
        X4 = Market Cap / Total Liabilities
        X5 = Revenue / Total Assets

    Zones:
        > 2.99  → "Safe"
        1.81–2.99 → "Grey Zone"
        < 1.81  → "Distress"

    Returns (z_score, zone) or (None, None) when fewer than 3 of 5
    components are available.

    Note: the classic Z-Score was calibrated on manufacturing firms.  For
    financial companies and highly asset-light businesses the absolute
    thresholds are less reliable; treat as directional only.
    """
    from utils_fmt import safe_num  # local import to avoid circular deps

    total_assets = safe_num(info.get("totalAssets"))
    if not has_numeric_value(total_assets) or total_assets <= 0:
        return None, None

    # X1: Working Capital / Total Assets
    x1: float | None = None
    curr_assets = safe_num(info.get("totalCurrentAssets"))
    curr_liabs  = safe_num(info.get("totalCurrentLiabilities"))
    if has_numeric_value(curr_assets) and has_numeric_value(curr_liabs):
        x1 = (curr_assets - curr_liabs) / total_assets

    # X2: Retained Earnings / Total Assets
    x2: float | None = None
    retained = safe_num(info.get("retainedEarnings"))
    if has_numeric_value(retained):
        x2 = retained / total_assets

    # X3: EBIT / Total Assets  (prefer operatingIncome; fall back to ebitda)
    x3: float | None = None
    op_income = safe_num(info.get("operatingIncome") or info.get("ebit"))
    if not has_numeric_value(op_income):
        op_income = safe_num(info.get("ebitda"))
    if has_numeric_value(op_income):
        x3 = op_income / total_assets

    # X4: Market Cap / Total Liabilities
    x4: float | None = None
    mkt_cap   = safe_num(info.get("marketCap"))
    total_liab = safe_num(info.get("totalLiab") or info.get("total_liab"))
    if has_numeric_value(mkt_cap) and has_numeric_value(total_liab) and total_liab > 0:
        x4 = mkt_cap / total_liab

    # X5: Revenue / Total Assets
    x5: float | None = None
    revenue = safe_num(info.get("totalRevenue"))
    if has_numeric_value(revenue):
        x5 = revenue / total_assets

    components = [x1, x2, x3, x4, x5]
    available = [c for c in components if c is not None]
    if len(available) < 3:
        return None, None

    coefficients = [1.2, 1.4, 3.3, 0.6, 1.0]
    z = sum(coef * val for coef, val in zip(coefficients, components) if val is not None)
    z = round(z, 2)

    if z > 2.99:
        zone = "Safe"
    elif z > 1.81:
        zone = "Grey Zone"
    else:
        zone = "Distress"

    return z, zone
