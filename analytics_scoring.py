# -*- coding: utf-8 -*-
"""
analytics_scoring.py — Scoring converters, market-regime classifiers, and
overall-verdict resolution.

Functions:
    cap_weights
    score_to_signal
    score_to_sentiment
    score_trend_distance
    step_signal_toward_neutral
    has_bullish_trend
    has_bearish_trend
    classify_market_regime
    summarize_engine_biases
    compute_decision_confidence
    apply_confidence_guard
    build_decision_notes
    resolve_overall_verdict
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fetch import has_numeric_value


# ---------------------------------------------------------------------------
# Weight helpers
# ---------------------------------------------------------------------------

def cap_weights(weights: pd.Series, max_weight: float) -> pd.Series:
    """
    Iteratively cap individual portfolio weights at *max_weight* and
    redistribute the excess to un-capped positions.

    Returns a Series that sums to 1.0.
    """
    capped = weights / weights.sum()
    if max_weight >= 1:
        return capped

    for _ in range(10):
        over_limit = capped > max_weight
        if not over_limit.any():
            break

        excess = (capped[over_limit] - max_weight).sum()
        capped[over_limit] = max_weight
        under_limit = capped < max_weight
        if not under_limit.any():
            break
        capped[under_limit] += excess * (capped[under_limit] / capped[under_limit].sum())
        capped = capped / capped.sum()

    return capped / capped.sum()


# ---------------------------------------------------------------------------
# Score → label converters
# ---------------------------------------------------------------------------

def score_to_signal(score: float, strong_buy: float = 4, buy: float = 2, sell: float = -2, strong_sell: float = -4) -> str:
    """Map a numeric composite score to a 5-level directional label."""
    if score >= strong_buy:
        return "STRONG BUY"
    if score >= buy:
        return "BUY"
    if score <= strong_sell:
        return "STRONG SELL"
    if score <= sell:
        return "SELL"
    return "HOLD"


def score_to_sentiment(score: float) -> str:
    """Map a sentiment score to POSITIVE / NEGATIVE / MIXED."""
    if score >= 3:
        return "POSITIVE"
    if score <= -3:
        return "NEGATIVE"
    return "MIXED"


def score_trend_distance(value: float | None, baseline: float | None, tolerance: float = 0.02) -> int:
    """Return +1 above, -1 below, 0 inside the *tolerance* band around *baseline*."""
    if not has_numeric_value(value) or not has_numeric_value(baseline) or baseline <= 0:
        return 0
    if value >= baseline * (1 + tolerance):
        return 1
    if value <= baseline * (1 - tolerance):
        return -1
    return 0


def step_signal_toward_neutral(signal: str) -> str:
    """Demote a signal by one step toward HOLD."""
    transitions = {
        "STRONG BUY": "BUY",
        "BUY": "HOLD",
        "HOLD": "HOLD",
        "SELL": "HOLD",
        "STRONG SELL": "SELL",
    }
    return transitions.get(signal, "HOLD")


# ---------------------------------------------------------------------------
# Trend helpers
# ---------------------------------------------------------------------------

def has_bullish_trend(price: float | None, sma50: float | None, sma200: float | None, momentum_1y: float | None = None) -> bool:
    """Return True when price and SMA structure are aligned bullishly."""
    if not has_numeric_value(price) or not has_numeric_value(sma200):
        return False
    long_term_ok = (
        has_numeric_value(sma50) and sma50 >= sma200
    ) or (
        has_numeric_value(momentum_1y) and momentum_1y >= 0
    )
    return price >= sma200 and long_term_ok


def has_bearish_trend(price: float | None, sma50: float | None, sma200: float | None, momentum_1y: float | None = None) -> bool:
    """Return True when price and SMA structure are aligned bearishly."""
    if not has_numeric_value(price) or not has_numeric_value(sma200):
        return False
    long_term_weak = (
        has_numeric_value(sma50) and sma50 <= sma200
    ) or (
        has_numeric_value(momentum_1y) and momentum_1y <= 0
    )
    return price <= sma200 and long_term_weak


def classify_market_regime(
    price: float | None,
    sma50: float | None,
    sma200: float | None,
    momentum_1y: float | None = None,
    tolerance: float = 0.02,
) -> str:
    """
    Return one of: 'Bullish Trend', 'Bearish Trend', 'Range-bound',
    'Transition', 'Unclear'.
    """
    if not has_numeric_value(price) or not has_numeric_value(sma200):
        return "Unclear"
    if has_bullish_trend(price, sma50, sma200, momentum_1y) and price >= sma200 * (1 + tolerance):
        return "Bullish Trend"
    if has_bearish_trend(price, sma50, sma200, momentum_1y) and price <= sma200 * (1 - tolerance):
        return "Bearish Trend"
    if abs((price - sma200) / sma200) <= tolerance and (momentum_1y is None or abs(momentum_1y) <= 0.10):
        return "Range-bound"
    return "Transition"


# ---------------------------------------------------------------------------
# Decision confidence
# ---------------------------------------------------------------------------

def summarize_engine_biases(
    tech_score: float,
    f_score: float,
    v_score: float,
    sentiment_score: float,
    v_val: str,
    bullish_trend: bool,
    bearish_trend: bool,
) -> dict[str, Any]:
    """
    Return a dict summarising per-engine directional biases and aggregate
    bullish / bearish counts.
    """
    tech_bullish = tech_score >= 3 or (tech_score >= 2 and bullish_trend)
    tech_bearish = tech_score <= -4 or (tech_score <= -3 and bearish_trend)
    fund_bullish = f_score >= 2
    fund_bearish = f_score <= -2
    valuation_bullish = v_val == "UNDERVALUED"
    valuation_bearish = v_val == "OVERVALUED" and v_score <= -2
    sentiment_bullish = sentiment_score >= 2
    sentiment_bearish = sentiment_score <= -2
    bullish_count = sum([tech_bullish, fund_bullish, valuation_bullish, sentiment_bullish])
    bearish_count = sum([tech_bearish, fund_bearish, valuation_bearish, sentiment_bearish])
    return {
        "tech_bullish": tech_bullish,
        "tech_bearish": tech_bearish,
        "fund_bullish": fund_bullish,
        "fund_bearish": fund_bearish,
        "valuation_bullish": valuation_bullish,
        "valuation_bearish": valuation_bearish,
        "sentiment_bullish": sentiment_bullish,
        "sentiment_bearish": sentiment_bearish,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "mixed": bullish_count >= 2 and bearish_count >= 2,
    }


def compute_decision_confidence(
    overall_score: float,
    bias_summary: dict[str, Any],
    regime: str,
    completeness: float | None,
) -> float:
    """
    Compute a 0–100 decision consistency score.

    Penalises opposing engine signals, mixed regimes, and low data
    completeness.
    """
    dominant_count = max(bias_summary["bullish_count"], bias_summary["bearish_count"])
    opposing_count = min(bias_summary["bullish_count"], bias_summary["bearish_count"])
    confidence = 28 + dominant_count * 14 - opposing_count * 10 + min(abs(overall_score) * 2.0, 18)

    if regime in {"Bullish Trend", "Bearish Trend"}:
        confidence += 8
    elif regime == "Transition":
        confidence -= 6
    elif regime == "Range-bound":
        confidence -= 10

    if completeness is not None:
        confidence += (float(completeness) - 0.5) * 35

    return float(np.clip(round(confidence, 1), 5.0, 95.0))


def apply_confidence_guard(verdict: str, confidence: float, data_quality: str, settings: dict[str, Any]) -> str:
    """
    Demote a verdict toward HOLD when consistency is below the active
    floor thresholds or data quality is Low.
    """
    guarded = verdict
    strong_floor = settings["decision_min_confidence"] + 10
    base_floor = settings["decision_min_confidence"]

    if guarded in {"STRONG BUY", "STRONG SELL"} and confidence < strong_floor:
        guarded = step_signal_toward_neutral(guarded)
    if guarded in {"BUY", "SELL"} and confidence < base_floor:
        guarded = "HOLD"
    if data_quality == "Low" and guarded in {"BUY", "SELL", "STRONG BUY", "STRONG SELL"}:
        guarded = "HOLD"
    return guarded


def build_decision_notes(
    verdict: str,
    regime: str,
    bias_summary: dict[str, Any],
    confidence: float,
    data_quality: str,
    current_rsi: float | None,
    v_val: str,
    v_fund: str,
    bullish_trend: bool,
    bearish_trend: bool,
    overextended: bool,
    pullback_recovery: bool,
) -> str:
    """
    Return a pipe-separated string of ≤4 human-readable decision notes
    explaining the current verdict.
    """
    notes = [f"Regime: {regime}"]
    if bias_summary["mixed"] and verdict == "HOLD":
        notes.append("Conflicting engine signals pushed the model toward hold")
    elif verdict in {"BUY", "STRONG BUY"} and bullish_trend:
        notes.append("Trend and engine alignment lean constructive")
    elif verdict in {"SELL", "STRONG SELL"} and bearish_trend:
        notes.append("Trend and engine alignment lean defensive")

    if overextended and verdict in {"BUY", "HOLD"}:
        notes.append("Price looks stretched above the short-term trend")
    if pullback_recovery:
        notes.append("RSI has recovered from oversold conditions")
    if v_val == "UNDERVALUED":
        notes.append("Valuation provides some downside support")
    elif v_val == "OVERVALUED":
        notes.append("Valuation still looks stretched")

    if v_fund == "STRONG":
        notes.append("Fundamentals remain supportive")
    elif v_fund == "WEAK":
        notes.append("Fundamentals are a headwind")

    if has_numeric_value(current_rsi) and current_rsi >= 70 and verdict in {"BUY", "STRONG BUY", "HOLD"}:
        notes.append("Momentum is hot enough to warrant patience on entries")
    if data_quality == "Low":
        notes.append("Low data completeness reduced conviction")
    elif confidence < 60:
        notes.append("Consistency is moderate, so the model stayed cautious")

    deduped = []
    for note in notes:
        if note not in deduped:
            deduped.append(note)
    return " | ".join(deduped[:4])


def resolve_overall_verdict(
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
    settings: dict[str, Any],
) -> str:
    """
    Apply all qualitative guardrails to translate the composite score into
    a final 5-level verdict label.

    Mixed-signal handling: the hold-buffer penalty is applied once (at the
    threshold level); a 2v2 engine split that still clears the raised
    threshold is demoted one step rather than killed outright.
    """
    bias_summary = summarize_engine_biases(
        tech_score,
        f_score,
        v_score,
        sentiment_score,
        v_val,
        bullish_trend,
        bearish_trend,
    )
    buy_threshold = settings["overall_buy_threshold"]
    strong_buy_threshold = settings["overall_strong_buy_threshold"]
    sell_threshold = settings["overall_sell_threshold"]
    strong_sell_threshold = settings["overall_strong_sell_threshold"]
    if bias_summary["mixed"] or regime in {"Range-bound", "Transition"}:
        hold_buffer = settings["decision_hold_buffer"]
        buy_threshold += hold_buffer
        strong_buy_threshold += hold_buffer
        sell_threshold -= hold_buffer
        strong_sell_threshold -= hold_buffer

    if v_val == "UNDERVALUED" and v_fund == "STRONG" and sentiment_score >= 2:
        verdict = "STRONG BUY" if tech_score >= 2 and not bearish_trend else "BUY"
    elif v_val == "OVERVALUED" and sentiment_score <= -2 and bearish_trend and f_score <= 0:
        verdict = "STRONG SELL" if tech_score <= -3 else "SELL"
    else:
        verdict = score_to_signal(
            overall_score,
            strong_buy=strong_buy_threshold,
            buy=buy_threshold,
            sell=sell_threshold,
            strong_sell=strong_sell_threshold,
        )

    if verdict in {"BUY", "STRONG BUY"}:
        if bias_summary["bearish_count"] >= 2:
            if bias_summary["mixed"]:
                verdict = "BUY"
            else:
                verdict = "HOLD"
        elif bearish_trend and f_score <= 0 and sentiment_score <= 0:
            verdict = "HOLD"
        elif v_val == "OVERVALUED" and f_score < 2:
            verdict = step_signal_toward_neutral(verdict)

    if verdict in {"SELL", "STRONG SELL"}:
        if bias_summary["bullish_count"] >= 2:
            if bias_summary["mixed"]:
                verdict = "SELL"
            else:
                verdict = "HOLD"
        elif bullish_trend and f_score >= 1:
            verdict = "HOLD"
        elif v_val == "UNDERVALUED" and f_score >= 0 and sentiment_score > -3:
            verdict = step_signal_toward_neutral(verdict)

    if verdict == "STRONG BUY" and (bias_summary["bearish_count"] > 0 or bearish_trend):
        verdict = "BUY"
    if verdict == "STRONG SELL" and (bias_summary["bullish_count"] > 0 or bullish_trend):
        verdict = "SELL"

    if verdict == "HOLD":
        if (
            bias_summary["bullish_count"] >= 3
            and bias_summary["bearish_count"] == 0
            and regime == "Bullish Trend"
            and overall_score >= buy_threshold - 1
        ):
            verdict = "BUY"
        elif (
            bias_summary["bearish_count"] >= 3
            and bias_summary["bullish_count"] == 0
            and regime == "Bearish Trend"
            and overall_score <= sell_threshold + 1
        ):
            verdict = "SELL"

    return verdict
