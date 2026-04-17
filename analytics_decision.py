# -*- coding: utf-8 -*-
"""
analytics_decision.py — Stock-type classification, engine-weight adjustments,
and risk-flag construction.

Functions:
    calculate_valuation_confidence
    calculate_sentiment_conviction
    get_type_adjusted_engine_weights
    build_risk_flags
    classify_cap_bucket
    build_stock_type_strategy
    classify_stock_profile
    extract_stock_profile_from_saved_row
    infer_stock_profile_from_snapshot
    apply_stock_type_framework
    adjust_type_based_confidence
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from constants import CYCLICAL_SECTORS, DEFENSIVE_SECTORS, INCOME_SECTORS, QUALITY_SECTORS, STOCK_TYPE_STRATEGIES
from fetch import has_numeric_value, safe_num, score_relative_multiple
from analytics_scoring import resolve_overall_verdict


# ---------------------------------------------------------------------------
# Valuation and sentiment confidence helpers
# ---------------------------------------------------------------------------

def calculate_valuation_confidence(signal_count: float | None) -> float:
    """Return a 20–95 valuation confidence score based on available signal count."""
    if not has_numeric_value(signal_count):
        return 25.0
    signal_count = float(signal_count)
    return float(np.clip(20 + signal_count * 12, 20, 95))


def calculate_sentiment_conviction(
    sentiment_score: float,
    analyst_opinions: float | None,
    recommendation_key: str | None,
    target_mean_price: float | None,
    price: float | None,
    headline_count: int | None,
) -> float:
    """
    Return a 10–95 sentiment conviction score that combines analyst
    opinion depth, target-price upside, and headline volume.
    """
    conviction = min(abs(float(sentiment_score)) * 2, 10)
    if has_numeric_value(analyst_opinions):
        conviction += min(float(analyst_opinions) * 2.5, 25)
    if recommendation_key and recommendation_key != "N/A":
        conviction += 10
    if has_numeric_value(target_mean_price) and has_numeric_value(price) and price > 0:
        conviction += min(abs((target_mean_price - price) / price) * 100, 20)
    conviction += min((headline_count or 0) * 3, 10)
    return float(np.clip(conviction, 10, 95))


# ---------------------------------------------------------------------------
# Engine-weight adjustment
# ---------------------------------------------------------------------------

def get_type_adjusted_engine_weights(stock_profile: dict[str, Any], settings: dict[str, Any]) -> tuple[dict[str, float], str]:
    """
    Return (weights_dict, weight_profile_string) with per-engine multipliers
    applied for the detected stock primary type.
    """
    primary_type = stock_profile.get("primary_type", "")
    modifiers = {
        "technical": 1.0,
        "fundamental": 1.0,
        "valuation": 1.0,
        "sentiment": 1.0,
    }
    if primary_type == "Growth Stocks":
        modifiers.update({"technical": 1.2, "fundamental": 1.05, "valuation": 0.75, "sentiment": 1.15})
    elif primary_type == "Value Stocks":
        modifiers.update({"technical": 0.85, "fundamental": 1.2, "valuation": 1.35, "sentiment": 0.8})
    elif primary_type == "Dividend / Income Stocks":
        modifiers.update({"technical": 0.75, "fundamental": 1.25, "valuation": 1.1, "sentiment": 0.7})
    elif primary_type == "Cyclical Stocks":
        modifiers.update({"technical": 1.25, "fundamental": 0.95, "valuation": 0.9, "sentiment": 1.0})
    elif primary_type == "Defensive Stocks":
        modifiers.update({"technical": 0.8, "fundamental": 1.25, "valuation": 1.0, "sentiment": 0.7})
    elif primary_type == "Blue-Chip Stocks":
        modifiers.update({"technical": 0.9, "fundamental": 1.2, "valuation": 0.95, "sentiment": 0.8})
    elif primary_type == "Small-Cap Stocks":
        modifiers.update({"technical": 1.15, "fundamental": 1.0, "valuation": 0.8, "sentiment": 1.05})
    elif primary_type == "Mid-Cap Stocks":
        modifiers.update({"technical": 1.05, "fundamental": 1.05, "valuation": 0.95, "sentiment": 0.95})
    elif primary_type == "Large-Cap Stocks":
        modifiers.update({"technical": 0.95, "fundamental": 1.1, "valuation": 1.0, "sentiment": 0.85})
    elif primary_type == "Speculative / Penny Stocks":
        modifiers.update({"technical": 1.35, "fundamental": 0.7, "valuation": 0.55, "sentiment": 1.25})

    weights = {
        "technical": settings["weight_technical"] * modifiers["technical"],
        "fundamental": settings["weight_fundamental"] * modifiers["fundamental"],
        "valuation": settings["weight_valuation"] * modifiers["valuation"],
        "sentiment": settings["weight_sentiment"] * modifiers["sentiment"],
    }
    weight_profile = (
        f"T {weights['technical']:.2f} | F {weights['fundamental']:.2f} | "
        f"V {weights['valuation']:.2f} | S {weights['sentiment']:.2f}"
    )
    return weights, weight_profile


# ---------------------------------------------------------------------------
# Risk flags
# ---------------------------------------------------------------------------

def build_risk_flags(
    *,
    eps: float | None,
    debt_eq: float | None,
    current_ratio: float | None,
    overextended: bool,
    distance_52w_high: float | None,
    range_position: float | None,
    volatility_1y: float | None,
    stock_profile: dict[str, Any],
) -> list[str]:
    """
    Return a list of short risk-flag labels for visible red flags in the
    current snapshot.
    """
    flags = []
    if has_numeric_value(eps) and eps <= 0:
        flags.append("Negative EPS")
    if has_numeric_value(debt_eq) and debt_eq > 220:
        flags.append("High Debt")
    if has_numeric_value(current_ratio) and current_ratio < 1.0:
        flags.append("Weak Liquidity")
    if overextended:
        flags.append("Overextended")
    if has_numeric_value(distance_52w_high) and distance_52w_high <= -0.20:
        flags.append("Far Below 52W High")
    if has_numeric_value(range_position) and range_position <= 0.15:
        flags.append("Near 52W Low")
    if has_numeric_value(volatility_1y) and volatility_1y >= 0.55:
        flags.append("High Volatility")
    if stock_profile.get("primary_type") == "Speculative / Penny Stocks":
        flags.append("Speculative")
    return flags


# ---------------------------------------------------------------------------
# Cap-bucket and stock-type classification
# ---------------------------------------------------------------------------

def classify_cap_bucket(market_cap: float | None) -> str:
    """Return 'Small-Cap', 'Mid-Cap', 'Large-Cap', or 'Unknown'."""
    if not has_numeric_value(market_cap) or market_cap <= 0:
        return "Unknown"
    if market_cap < 2_000_000_000:
        return "Small-Cap"
    if market_cap < 10_000_000_000:
        return "Mid-Cap"
    return "Large-Cap"


def build_stock_type_strategy(primary_type: str) -> str:
    """Return the strategy string for *primary_type* from the constant map."""
    return STOCK_TYPE_STRATEGIES.get(
        primary_type,
        "Use the stock's blend of quality, valuation, and regime signals instead of forcing it into a one-size-fits-all strategy.",
    )


def classify_stock_profile(
    *,
    sector,
    price,
    market_cap,
    dividend_yield,
    payout_ratio,
    equity_beta,
    analyst_opinions,
    pe,
    forward_pe,
    peg_ratio,
    ps_ratio,
    pb,
    bench,
    f_score,
    v_val,
    revenue_growth,
    earnings_growth,
    margins,
    roe,
    current_ratio,
    debt_eq,
    momentum_1y,
):
    """
    Score each candidate stock type and return the winning primary_type
    together with cap_bucket, style_tags, type_strategy, and classification
    metadata.
    """
    cap_bucket = classify_cap_bucket(market_cap)
    scores = {
        "Growth Stocks": 0.0,
        "Value Stocks": 0.0,
        "Dividend / Income Stocks": 0.0,
        "Cyclical Stocks": 0.0,
        "Defensive Stocks": 0.0,
        "Blue-Chip Stocks": 0.0,
        "Small-Cap Stocks": 0.0,
        "Mid-Cap Stocks": 0.0,
        "Large-Cap Stocks": 0.0,
        "Speculative / Penny Stocks": 0.0,
    }
    reasons = {name: [] for name in scores}

    if sector in {"Technology", "Communication Services"}:
        scores["Growth Stocks"] += 1
        reasons["Growth Stocks"].append("innovation-heavy sector")
    if has_numeric_value(revenue_growth) and revenue_growth >= 0.12:
        scores["Growth Stocks"] += 2
        reasons["Growth Stocks"].append("double-digit revenue growth")
    if has_numeric_value(earnings_growth) and earnings_growth >= 0.12:
        scores["Growth Stocks"] += 2
        reasons["Growth Stocks"].append("double-digit earnings growth")
    if has_numeric_value(momentum_1y) and momentum_1y >= 0.20:
        scores["Growth Stocks"] += 1
        reasons["Growth Stocks"].append("strong long-term momentum")
    if (
        (has_numeric_value(pe) and has_numeric_value(bench.get("PE")) and pe >= bench["PE"] * 1.10)
        or (has_numeric_value(ps_ratio) and has_numeric_value(bench.get("PS")) and ps_ratio >= bench["PS"] * 1.10)
        or (has_numeric_value(peg_ratio) and peg_ratio >= 1.2)
    ):
        scores["Growth Stocks"] += 1
        reasons["Growth Stocks"].append("premium valuation profile")
    if not has_numeric_value(dividend_yield) or dividend_yield < 0.015:
        scores["Growth Stocks"] += 1
        reasons["Growth Stocks"].append("low payout orientation")

    if v_val == "UNDERVALUED":
        scores["Value Stocks"] += 2
        reasons["Value Stocks"].append("undervalued by the current valuation engine")
    if has_numeric_value(pe) and has_numeric_value(bench.get("PE")) and pe <= bench["PE"] * 0.85:
        scores["Value Stocks"] += 1
        reasons["Value Stocks"].append("discounted earnings multiple")
    if has_numeric_value(pb) and has_numeric_value(bench.get("PB")) and pb <= bench["PB"] * 0.85:
        scores["Value Stocks"] += 1
        reasons["Value Stocks"].append("discounted price-to-book")
    if has_numeric_value(ps_ratio) and has_numeric_value(bench.get("PS")) and ps_ratio <= bench["PS"] * 0.85:
        scores["Value Stocks"] += 1
        reasons["Value Stocks"].append("discounted price-to-sales")
    if f_score >= 1:
        scores["Value Stocks"] += 1
        reasons["Value Stocks"].append("fundamentals are at least stable")

    if has_numeric_value(dividend_yield) and dividend_yield >= 0.03:
        scores["Dividend / Income Stocks"] += 2
        reasons["Dividend / Income Stocks"].append("meaningful dividend yield")
    if has_numeric_value(dividend_yield) and dividend_yield >= 0.05:
        scores["Dividend / Income Stocks"] += 1
        reasons["Dividend / Income Stocks"].append("high income orientation")
    if has_numeric_value(payout_ratio) and 0 < payout_ratio <= 0.80:
        scores["Dividend / Income Stocks"] += 1
        reasons["Dividend / Income Stocks"].append("payout looks more sustainable")
    if sector in INCOME_SECTORS:
        scores["Dividend / Income Stocks"] += 1
        reasons["Dividend / Income Stocks"].append("income-heavy sector")
    if has_numeric_value(equity_beta) and equity_beta < 1.0:
        scores["Dividend / Income Stocks"] += 1
        reasons["Dividend / Income Stocks"].append("below-market beta")

    if sector in CYCLICAL_SECTORS:
        scores["Cyclical Stocks"] += 2
        reasons["Cyclical Stocks"].append("economically sensitive sector")
    if has_numeric_value(equity_beta) and equity_beta > 1.10:
        scores["Cyclical Stocks"] += 1
        reasons["Cyclical Stocks"].append("higher beta profile")
    if has_numeric_value(momentum_1y) and abs(momentum_1y) >= 0.20:
        scores["Cyclical Stocks"] += 1
        reasons["Cyclical Stocks"].append("large cycle-like price swings")

    if sector in DEFENSIVE_SECTORS:
        scores["Defensive Stocks"] += 2
        reasons["Defensive Stocks"].append("defensive sector exposure")
    if has_numeric_value(equity_beta) and equity_beta < 1.0:
        scores["Defensive Stocks"] += 1
        reasons["Defensive Stocks"].append("lower beta profile")
    if has_numeric_value(margins) and margins > 0:
        scores["Defensive Stocks"] += 1
        reasons["Defensive Stocks"].append("positive margins")
    if f_score >= 1:
        scores["Defensive Stocks"] += 1
        reasons["Defensive Stocks"].append("stable operating fundamentals")

    if cap_bucket == "Large-Cap":
        scores["Large-Cap Stocks"] += 3
        reasons["Large-Cap Stocks"].append("market cap above $10B")
    elif cap_bucket == "Mid-Cap":
        scores["Mid-Cap Stocks"] += 3
        reasons["Mid-Cap Stocks"].append("market cap between $2B and $10B")
    elif cap_bucket == "Small-Cap":
        scores["Small-Cap Stocks"] += 3
        reasons["Small-Cap Stocks"].append("market cap below $2B")

    if has_numeric_value(market_cap) and market_cap >= 100_000_000_000:
        scores["Blue-Chip Stocks"] += 2
        reasons["Blue-Chip Stocks"].append("very large market capitalization")
    if has_numeric_value(market_cap) and market_cap >= 300_000_000_000:
        scores["Blue-Chip Stocks"] += 1
        reasons["Blue-Chip Stocks"].append("mega-cap scale")
    if sector in QUALITY_SECTORS:
        scores["Blue-Chip Stocks"] += 1
        reasons["Blue-Chip Stocks"].append("quality-heavy sector")
    if f_score >= 2:
        scores["Blue-Chip Stocks"] += 1
        reasons["Blue-Chip Stocks"].append("solid fundamental footing")
    if has_numeric_value(analyst_opinions) and analyst_opinions >= 10:
        scores["Blue-Chip Stocks"] += 1
        reasons["Blue-Chip Stocks"].append("broad analyst coverage")
    if has_numeric_value(roe) and roe > 0 and has_numeric_value(current_ratio) and current_ratio >= 1.0:
        scores["Blue-Chip Stocks"] += 1
        reasons["Blue-Chip Stocks"].append("quality balance-sheet profile")

    if has_numeric_value(price) and price < 5:
        scores["Speculative / Penny Stocks"] += 2
        reasons["Speculative / Penny Stocks"].append("low share price")
    if has_numeric_value(market_cap) and market_cap < 500_000_000:
        scores["Speculative / Penny Stocks"] += 2
        reasons["Speculative / Penny Stocks"].append("micro-cap scale")
    elif cap_bucket == "Small-Cap":
        scores["Speculative / Penny Stocks"] += 1
        reasons["Speculative / Penny Stocks"].append("small-cap risk profile")
    if f_score <= 0:
        scores["Speculative / Penny Stocks"] += 1
        reasons["Speculative / Penny Stocks"].append("weak or unproven fundamentals")
    if has_numeric_value(equity_beta) and equity_beta >= 1.8:
        scores["Speculative / Penny Stocks"] += 1
        reasons["Speculative / Penny Stocks"].append("very high beta")
    if has_numeric_value(analyst_opinions) and analyst_opinions < 3:
        scores["Speculative / Penny Stocks"] += 1
        reasons["Speculative / Penny Stocks"].append("thin coverage")
    if has_numeric_value(debt_eq) and debt_eq > 250:
        scores["Speculative / Penny Stocks"] += 1
        reasons["Speculative / Penny Stocks"].append("high leverage")

    style_tags = []
    for tag_name in [
        "Growth Stocks",
        "Value Stocks",
        "Dividend / Income Stocks",
        "Cyclical Stocks",
        "Defensive Stocks",
        "Blue-Chip Stocks",
    ]:
        if scores[tag_name] >= 3:
            style_tags.append(tag_name.replace(" Stocks", ""))
    if cap_bucket != "Unknown":
        style_tags.append(cap_bucket)

    top_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_type, top_score = top_scores[0]
    second_score = top_scores[1][1] if len(top_scores) > 1 else 0

    if scores["Speculative / Penny Stocks"] >= 4:
        primary_type = "Speculative / Penny Stocks"
    elif scores["Growth Stocks"] >= 5 and scores["Growth Stocks"] >= scores["Value Stocks"] + 1:
        primary_type = "Growth Stocks"
    elif scores["Value Stocks"] >= 5 and scores["Value Stocks"] >= scores["Growth Stocks"]:
        primary_type = "Value Stocks"
    elif scores["Dividend / Income Stocks"] >= 4 and scores["Dividend / Income Stocks"] >= scores["Defensive Stocks"]:
        primary_type = "Dividend / Income Stocks"
    elif scores["Defensive Stocks"] >= 4 and scores["Defensive Stocks"] >= scores["Cyclical Stocks"]:
        primary_type = "Defensive Stocks"
    elif scores["Cyclical Stocks"] >= 4:
        primary_type = "Cyclical Stocks"
    elif scores["Blue-Chip Stocks"] >= 5 and cap_bucket == "Large-Cap":
        primary_type = "Blue-Chip Stocks"
    elif top_score >= 4:
        primary_type = top_type
    elif cap_bucket == "Small-Cap":
        primary_type = "Small-Cap Stocks"
    elif cap_bucket == "Mid-Cap":
        primary_type = "Mid-Cap Stocks"
    else:
        primary_type = "Large-Cap Stocks" if cap_bucket == "Large-Cap" else "Blue-Chip Stocks"

    confidence = float(np.clip(45 + top_score * 8 + (top_score - second_score) * 6, 35, 95))
    top_reasons = reasons.get(primary_type, [])
    summary = ", ".join(top_reasons[:3]) if top_reasons else "mixed factor profile"
    return {
        "primary_type": primary_type,
        "cap_bucket": cap_bucket,
        "style_tags": " | ".join(style_tags) if style_tags else primary_type.replace(" Stocks", ""),
        "type_strategy": build_stock_type_strategy(primary_type),
        "type_confidence": confidence,
        "classification_summary": summary,
        "market_cap": market_cap,
        "dividend_yield": dividend_yield,
        "payout_ratio": payout_ratio,
        "equity_beta": equity_beta,
    }


def extract_stock_profile_from_saved_row(saved_row: dict[str, Any] | None) -> dict[str, Any]:
    """Build a minimal stock-profile dict from a previously saved database row."""
    if saved_row is None:
        return {}
    return {
        "primary_type": saved_row.get("Stock_Type", "Unknown"),
        "cap_bucket": saved_row.get("Cap_Bucket", "Unknown"),
        "style_tags": saved_row.get("Style_Tags", ""),
        "type_strategy": saved_row.get("Type_Strategy", ""),
        "type_confidence": safe_num(saved_row.get("Type_Confidence")),
        "classification_summary": "",
        "market_cap": safe_num(saved_row.get("Market_Cap")),
        "dividend_yield": safe_num(saved_row.get("Dividend_Yield")),
        "payout_ratio": safe_num(saved_row.get("Payout_Ratio")),
        "equity_beta": safe_num(saved_row.get("Equity_Beta")),
    }


def infer_stock_profile_from_snapshot(
    info: dict[str, Any],
    hist: pd.DataFrame,
    settings: dict[str, Any] | None = None,
    db: Any = None,
    ticker: str | None = None,
) -> dict[str, Any]:
    """
    Classify the stock type using live info and price history.

    Falls back to default model settings when *settings* is None.
    """
    from settings import get_model_settings
    from fetch import build_relative_peer_benchmarks

    active_settings = get_model_settings() if settings is None else settings
    info = info or {}
    if hist is None or hist.empty or "Close" not in hist.columns:
        return {}

    close = hist["Close"].dropna().astype(float)
    if close.empty:
        return {}
    price = float(close.iloc[-1])
    momentum_1y = (price / close.iloc[0] - 1) if len(close) > 1 else None
    sector = info.get("sector", "Unknown")
    bench, _ = build_relative_peer_benchmarks(
        ticker or info.get("symbol") or "",
        info,
        db=db,
        settings=active_settings,
    )

    roe = safe_num(info.get("returnOnEquity"))
    margins = safe_num(info.get("profitMargins"))
    debt_eq = safe_num(info.get("debtToEquity"))
    revenue_growth = safe_num(info.get("revenueGrowth"))
    earnings_growth = safe_num(info.get("earningsGrowth"))
    current_ratio = safe_num(info.get("currentRatio"))

    f_score = 0
    if has_numeric_value(roe):
        f_score += 1 if roe >= active_settings["fund_roe_threshold"] else (-1 if roe < 0 else 0)
    if has_numeric_value(margins):
        f_score += 1 if margins >= active_settings["fund_profit_margin_threshold"] else (-1 if margins < 0 else 0)
    if has_numeric_value(debt_eq):
        f_score += 1 if 0 <= debt_eq < active_settings["fund_debt_good_threshold"] else (-1 if debt_eq > active_settings["fund_debt_bad_threshold"] else 0)
    if has_numeric_value(revenue_growth):
        f_score += 1 if revenue_growth >= active_settings["fund_revenue_growth_threshold"] else (-1 if revenue_growth < 0 else 0)
    if has_numeric_value(earnings_growth):
        f_score += 1 if earnings_growth >= active_settings["fund_revenue_growth_threshold"] else (-1 if earnings_growth < 0 else 0)
    if has_numeric_value(current_ratio):
        f_score += 1 if current_ratio >= active_settings["fund_current_ratio_good"] else (-1 if current_ratio < active_settings["fund_current_ratio_bad"] else 0)

    pe = safe_num(info.get("trailingPE"))
    forward_pe = safe_num(info.get("forwardPE"))
    peg_ratio = safe_num(info.get("pegRatio"))
    ps_ratio = safe_num(info.get("priceToSalesTrailing12Months"))
    pb = safe_num(info.get("priceToBook"))
    ev_ebitda = safe_num(info.get("enterpriseToEbitda"))
    v_score = 0
    valuation_signal_count = 0
    for metric_value, benchmark_value in [
        (pe, bench["PE"]),
        (forward_pe, bench["PE"]),
        (ps_ratio, bench["PS"]),
        (pb, bench["PB"]),
        (ev_ebitda, bench["EV_EBITDA"]),
    ]:
        multiple_score = score_relative_multiple(metric_value, benchmark_value)
        v_score += multiple_score
        if has_numeric_value(metric_value):
            valuation_signal_count += 1
    if has_numeric_value(peg_ratio):
        valuation_signal_count += 1
        if peg_ratio <= active_settings["valuation_peg_threshold"] * 0.9:
            v_score += 1
        elif peg_ratio >= active_settings["valuation_peg_threshold"] * 1.35:
            v_score -= 1

    if valuation_signal_count < 2 and v_score < active_settings["valuation_fair_score_threshold"]:
        v_val = "FAIR VALUE"
    elif v_score >= active_settings["valuation_under_score_threshold"]:
        v_val = "UNDERVALUED"
    elif v_score >= active_settings["valuation_fair_score_threshold"]:
        v_val = "FAIR VALUE"
    else:
        v_val = "OVERVALUED"

    return classify_stock_profile(
        sector=sector,
        price=price,
        market_cap=safe_num(info.get("marketCap")),
        dividend_yield=safe_num(info.get("dividendYield")),
        payout_ratio=safe_num(info.get("payoutRatio")),
        equity_beta=safe_num(info.get("beta")),
        analyst_opinions=safe_num(info.get("numberOfAnalystOpinions")),
        pe=pe,
        forward_pe=forward_pe,
        peg_ratio=peg_ratio,
        ps_ratio=ps_ratio,
        pb=pb,
        bench=bench,
        f_score=f_score,
        v_val=v_val,
        revenue_growth=revenue_growth,
        earnings_growth=earnings_growth,
        margins=margins,
        roe=roe,
        current_ratio=current_ratio,
        debt_eq=debt_eq,
        momentum_1y=momentum_1y,
    )


def apply_stock_type_framework(
    *,
    stock_profile: dict[str, Any],
    overall_score: float,
    tech_score: float,
    f_score: float,
    v_score: float,
    sentiment_score: float,
    v_fund: str,
    v_val: str,
    regime: str,
    bullish_trend: bool,
    bearish_trend: bool,
    data_quality: str,
    momentum_1y: float | None,
    settings: dict[str, Any],
) -> tuple[float, str, dict[str, Any], list[str]]:
    """
    Apply stock-type-specific score adjustments and threshold overrides,
    then call resolve_overall_verdict to compute the final verdict.

    Returns (adjusted_score, verdict, local_settings, notes).
    """
    primary_type = stock_profile.get("primary_type", "")
    cap_bucket = stock_profile.get("cap_bucket", "Unknown")
    dividend_yield = stock_profile.get("dividend_yield")
    payout_ratio = stock_profile.get("payout_ratio")
    adjusted_score = overall_score
    local_settings = dict(settings)
    notes = []
    cap_verdict_to = None

    if primary_type == "Growth Stocks":
        local_settings["decision_hold_buffer"] = max(0.0, settings["decision_hold_buffer"] - 0.5)
        local_settings["overall_buy_threshold"] = max(1.0, settings["overall_buy_threshold"] - 1.0)
        if bullish_trend and (f_score >= 1 or sentiment_score >= 1):
            adjusted_score += 1.0
            notes.append("Growth profile rewards durable trend strength")
        if v_val == "OVERVALUED" and bullish_trend and has_numeric_value(momentum_1y) and momentum_1y > 0.25:
            adjusted_score += 0.75
            notes.append("Premium valuation is tolerated more for proven compounders")
        if bearish_trend and sentiment_score <= 0:
            adjusted_score -= 1.0
            notes.append("Growth setups lose conviction faster in downtrends")

    elif primary_type == "Value Stocks":
        local_settings["decision_hold_buffer"] = settings["decision_hold_buffer"] + 0.5
        if v_val == "UNDERVALUED" and f_score >= 1:
            adjusted_score += 1.25
            notes.append("Value profile leans harder on valuation plus stable fundamentals")
        if bearish_trend and tech_score <= -2:
            adjusted_score -= 0.75
            notes.append("Deep value still waits for price damage to stabilize")
        if sentiment_score <= -2:
            adjusted_score -= 0.5

    elif primary_type == "Dividend / Income Stocks":
        local_settings["overall_sell_threshold"] = settings["overall_sell_threshold"] - 1.0
        local_settings["overall_strong_sell_threshold"] = settings["overall_strong_sell_threshold"] - 1.0
        if has_numeric_value(dividend_yield) and dividend_yield >= 0.03:
            adjusted_score += 0.75
            notes.append("Income profile rewards durable shareholder yield")
        if has_numeric_value(payout_ratio) and payout_ratio > 0.95:
            adjusted_score -= 1.0
            notes.append("An overstretched payout ratio weakens the income case")
        if regime == "Range-bound" and f_score >= 1 and v_val != "OVERVALUED":
            adjusted_score += 0.5

    elif primary_type == "Cyclical Stocks":
        local_settings["decision_hold_buffer"] = settings["decision_hold_buffer"] + 1.0
        if regime == "Bullish Trend" and tech_score >= 2 and sentiment_score >= 0:
            adjusted_score += 1.25
            notes.append("Cyclicals want momentum and regime confirmation")
        if regime in {"Transition", "Range-bound"}:
            adjusted_score -= 0.75
            notes.append("Cycle timing risk pushes mixed setups toward hold")
        if bearish_trend or (has_numeric_value(momentum_1y) and momentum_1y < 0):
            adjusted_score -= 1.0

    elif primary_type == "Defensive Stocks":
        local_settings["overall_sell_threshold"] = settings["overall_sell_threshold"] - 1.0
        local_settings["overall_strong_sell_threshold"] = settings["overall_strong_sell_threshold"] - 1.0
        if f_score >= 1 and v_val != "OVERVALUED":
            adjusted_score += 0.75
            notes.append("Defensives get more credit for steady fundamentals")
        if regime in {"Range-bound", "Transition"} and f_score >= 1:
            adjusted_score += 0.5
        if bearish_trend and f_score <= 0:
            adjusted_score -= 0.5

    elif primary_type == "Blue-Chip Stocks":
        local_settings["decision_hold_buffer"] = max(0.0, settings["decision_hold_buffer"] - 0.5)
        if f_score >= 1 and sentiment_score >= 0:
            adjusted_score += 0.75
            notes.append("Blue-chip profile gives more room to durable quality")
        if v_val == "OVERVALUED" and bullish_trend and f_score >= 2:
            adjusted_score += 0.5
        if bearish_trend and f_score <= 0:
            adjusted_score -= 0.75

    elif primary_type == "Small-Cap Stocks":
        local_settings["decision_hold_buffer"] = settings["decision_hold_buffer"] + 0.5
        local_settings["overall_buy_threshold"] = settings["overall_buy_threshold"] + 1.0
        if tech_score >= 3 and sentiment_score >= 1 and bullish_trend:
            adjusted_score += 0.75
            notes.append("Small caps need stronger confirmation before leaning in")
        if data_quality == "Low" or bearish_trend:
            adjusted_score -= 1.25

    elif primary_type == "Mid-Cap Stocks":
        if f_score >= 1 and bullish_trend:
            adjusted_score += 0.5
            notes.append("Mid-cap profile likes balanced trend plus fundamentals")
        if bearish_trend and f_score <= 0:
            adjusted_score -= 0.75

    elif primary_type == "Large-Cap Stocks":
        if f_score >= 1:
            adjusted_score += 0.5
            notes.append("Large caps get a modest durability benefit")
        if bearish_trend and f_score <= 0:
            adjusted_score -= 0.5

    elif primary_type == "Speculative / Penny Stocks":
        local_settings["decision_hold_buffer"] = settings["decision_hold_buffer"] + 1.0
        local_settings["overall_buy_threshold"] = settings["overall_buy_threshold"] + 1.0
        local_settings["overall_strong_buy_threshold"] = settings["overall_strong_buy_threshold"] + 2.0
        cap_verdict_to = "BUY"
        if tech_score >= 4 and sentiment_score >= 2 and bullish_trend:
            adjusted_score += 0.5
            notes.append("Speculative profile only rewards strong confirmation")
        else:
            adjusted_score -= 1.0
        if data_quality == "Low" or f_score <= 0:
            adjusted_score -= 1.0
            notes.append("Weak data or fundamentals are penalized harder")

    if cap_bucket == "Small-Cap" and primary_type != "Speculative / Penny Stocks":
        adjusted_score -= 0.25
    elif cap_bucket == "Large-Cap" and primary_type not in {"Blue-Chip Stocks", "Large-Cap Stocks"} and f_score >= 1:
        adjusted_score += 0.25

    verdict = resolve_overall_verdict(
        overall_score=adjusted_score,
        tech_score=tech_score,
        f_score=f_score,
        v_score=v_score,
        sentiment_score=sentiment_score,
        v_fund=v_fund,
        v_val=v_val,
        regime=regime,
        bullish_trend=bullish_trend,
        bearish_trend=bearish_trend,
        settings=local_settings,
    )

    if cap_verdict_to == "BUY" and verdict == "STRONG BUY":
        verdict = "BUY"

    return adjusted_score, verdict, local_settings, notes


def adjust_type_based_confidence(confidence: float, stock_profile: dict[str, Any], data_quality: str) -> float:
    """
    Nudge the confidence score up for durable types (Blue-Chip, Defensive)
    and down for riskier types (Small-Cap, Speculative).
    """
    primary_type = stock_profile.get("primary_type", "")
    adjusted_confidence = float(confidence)
    if primary_type in {"Blue-Chip Stocks", "Defensive Stocks"} and data_quality != "Low":
        adjusted_confidence += 4
    elif primary_type in {"Small-Cap Stocks", "Speculative / Penny Stocks"}:
        adjusted_confidence -= 6 if primary_type == "Speculative / Penny Stocks" else 3
    return float(np.clip(round(adjusted_confidence, 1), 5.0, 95.0))
