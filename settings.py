# -*- coding: utf-8 -*-
"""
settings.py — Model settings normalization, session-state accessors, and
assumption fingerprinting.

Functions that read/write ``st.session_state`` import streamlit.
Pure normalization helpers (normalize_model_settings, normalize_dcf_settings,
etc.) can be called without a Streamlit context.

Depends on: constants, utils_fmt (safe_num).
"""

import copy
import hashlib
import json

import numpy as np
import streamlit as st

from constants import (
    DEFAULT_DCF_SETTINGS,
    DEFAULT_MODEL_SETTINGS,
    MODEL_PRESETS,
)
from utils_fmt import safe_num


# ---------------------------------------------------------------------------
# Defaults / presets
# ---------------------------------------------------------------------------

def get_default_model_settings():
    """Return a fresh copy of the default model settings dict."""
    return DEFAULT_MODEL_SETTINGS.copy()


def get_default_preset_name():
    """Return the name of the default model preset (``"Balanced"``)."""
    return "Balanced"


def get_default_dcf_settings():
    """Return a fresh deep copy of the default DCF settings dict."""
    return copy.deepcopy(DEFAULT_DCF_SETTINGS)


# ---------------------------------------------------------------------------
# DCF settings normalization
# ---------------------------------------------------------------------------

def normalize_dcf_settings(settings):
    """Validate and clamp every field in a DCF settings dict.

    Parameters
    ----------
    settings:
        Raw settings dict (may be incomplete).

    Returns
    -------
    dict
        Fully populated and clamped settings dict.
    """
    normalized = get_default_dcf_settings()
    normalized.update(settings or {})

    normalized["projection_years"] = int(min(max(int(round(float(normalized["projection_years"]))), 3), 10))
    normalized["terminal_growth_rate"] = float(min(max(float(normalized["terminal_growth_rate"]), 0.0), 0.05))
    normalized["growth_haircut"] = float(min(max(float(normalized["growth_haircut"]), 0.5), 1.2))
    normalized["max_growth_rate"] = float(min(max(float(normalized["max_growth_rate"]), 0.05), 0.50))
    normalized["min_growth_rate"] = float(min(max(float(normalized["min_growth_rate"]), -0.20), 0.10))
    if normalized["min_growth_rate"] >= normalized["max_growth_rate"]:
        normalized["min_growth_rate"] = min(normalized["max_growth_rate"] - 0.01, -0.01)

    normalized["market_risk_premium"] = float(min(max(float(normalized["market_risk_premium"]), 0.03), 0.09))
    normalized["default_after_tax_cost_of_debt"] = float(
        min(max(float(normalized["default_after_tax_cost_of_debt"]), 0.01), 0.10)
    )

    for optional_key in ["risk_free_rate_override", "manual_growth_rate"]:
        optional_value = safe_num(normalized.get(optional_key))
        if optional_value is None:
            normalized[optional_key] = None
        elif optional_key == "risk_free_rate_override":
            normalized[optional_key] = float(min(max(optional_value, 0.0), 0.10))
        else:
            normalized[optional_key] = float(min(max(optional_value, -0.20), 0.50))

    return normalized


def get_dcf_settings():
    """Return the active DCF settings from ``st.session_state``, normalizing first."""
    if "dcf_settings" not in st.session_state:
        st.session_state.dcf_settings = normalize_dcf_settings(get_default_dcf_settings())
        return st.session_state.dcf_settings

    normalized = normalize_dcf_settings(st.session_state.dcf_settings)
    st.session_state.dcf_settings = normalized
    return normalized


# ---------------------------------------------------------------------------
# Model settings normalization
# ---------------------------------------------------------------------------

def normalize_model_settings(settings):
    """Validate, clamp, and cross-validate every field in a model settings dict.

    Parameters
    ----------
    settings:
        Raw settings dict (may be incomplete).

    Returns
    -------
    tuple[dict, list[str]]
        ``(normalized_settings, notes)`` where *notes* is a list of strings
        describing any automatic adjustments that were applied.
    """
    normalized = get_default_model_settings()
    normalized.update(settings or {})
    notes = []

    normalized["tech_rsi_oversold"] = min(max(float(normalized["tech_rsi_oversold"]), 20.0), 45.0)
    normalized["tech_rsi_overbought"] = min(max(float(normalized["tech_rsi_overbought"]), 55.0), 85.0)
    if normalized["tech_rsi_oversold"] >= normalized["tech_rsi_overbought"]:
        normalized["tech_rsi_overbought"] = min(85.0, normalized["tech_rsi_oversold"] + 5.0)
        notes.append("The RSI overbought threshold was nudged above the oversold threshold.")

    normalized["tech_momentum_threshold"] = min(max(float(normalized["tech_momentum_threshold"]), 0.01), 0.12)
    normalized["tech_trend_tolerance"] = min(max(float(normalized["tech_trend_tolerance"]), 0.0), 0.05)
    normalized["tech_extension_limit"] = min(max(float(normalized["tech_extension_limit"]), 0.03), 0.15)
    normalized["fund_roe_threshold"] = min(max(float(normalized["fund_roe_threshold"]), 0.05), 0.35)
    normalized["fund_profit_margin_threshold"] = min(max(float(normalized["fund_profit_margin_threshold"]), 0.05), 0.35)
    normalized["fund_debt_good_threshold"] = min(max(float(normalized["fund_debt_good_threshold"]), 25.0), 200.0)
    normalized["fund_debt_bad_threshold"] = min(max(float(normalized["fund_debt_bad_threshold"]), 75.0), 400.0)
    if normalized["fund_debt_bad_threshold"] <= normalized["fund_debt_good_threshold"]:
        normalized["fund_debt_bad_threshold"] = min(400.0, normalized["fund_debt_good_threshold"] + 50.0)
        notes.append("The high-debt threshold was lifted above the low-debt threshold.")

    normalized["fund_revenue_growth_threshold"] = min(max(float(normalized["fund_revenue_growth_threshold"]), 0.0), 0.30)
    normalized["fund_current_ratio_good"] = min(max(float(normalized["fund_current_ratio_good"]), 1.0), 3.0)
    normalized["fund_current_ratio_bad"] = min(max(float(normalized["fund_current_ratio_bad"]), 0.5), 1.5)
    if normalized["fund_current_ratio_bad"] >= normalized["fund_current_ratio_good"]:
        normalized["fund_current_ratio_bad"] = max(0.5, normalized["fund_current_ratio_good"] - 0.2)
        notes.append("The weak current-ratio threshold was moved below the healthy threshold.")

    normalized["valuation_benchmark_scale"] = min(max(float(normalized["valuation_benchmark_scale"]), 0.8), 1.2)
    normalized["valuation_peg_threshold"] = min(max(float(normalized["valuation_peg_threshold"]), 0.8), 2.5)
    normalized["valuation_graham_overpriced_multiple"] = min(
        max(float(normalized["valuation_graham_overpriced_multiple"]), 1.2), 2.0
    )
    normalized["valuation_under_score_threshold"] = min(
        max(float(normalized["valuation_under_score_threshold"]), 3.0), 8.0
    )
    normalized["valuation_fair_score_threshold"] = min(
        max(float(normalized["valuation_fair_score_threshold"]), 1.0), 4.0
    )
    if normalized["valuation_under_score_threshold"] <= normalized["valuation_fair_score_threshold"]:
        normalized["valuation_under_score_threshold"] = min(
            8.0, normalized["valuation_fair_score_threshold"] + 1.0
        )
        notes.append("The undervalued score floor was kept above the fair-value score floor.")

    normalized["sentiment_analyst_boost"] = min(max(float(normalized["sentiment_analyst_boost"]), 0.0), 4.0)
    normalized["sentiment_upside_mid"] = min(max(float(normalized["sentiment_upside_mid"]), 0.02), 0.15)
    normalized["sentiment_upside_high"] = min(max(float(normalized["sentiment_upside_high"]), 0.08), 0.30)
    if normalized["sentiment_upside_high"] <= normalized["sentiment_upside_mid"]:
        normalized["sentiment_upside_high"] = min(0.30, normalized["sentiment_upside_mid"] + 0.05)
        notes.append("The high-upside threshold was kept above the moderate-upside threshold.")

    normalized["sentiment_downside_mid"] = min(max(float(normalized["sentiment_downside_mid"]), 0.02), 0.15)
    normalized["sentiment_downside_high"] = min(max(float(normalized["sentiment_downside_high"]), 0.08), 0.30)
    if normalized["sentiment_downside_high"] <= normalized["sentiment_downside_mid"]:
        normalized["sentiment_downside_high"] = min(0.30, normalized["sentiment_downside_mid"] + 0.05)
        notes.append("The deep-downside threshold was kept above the moderate-downside threshold.")

    normalized["overall_buy_threshold"] = min(max(float(normalized["overall_buy_threshold"]), 1.0), 6.0)
    normalized["overall_strong_buy_threshold"] = min(
        max(float(normalized["overall_strong_buy_threshold"]), 4.0), 12.0
    )
    if normalized["overall_strong_buy_threshold"] <= normalized["overall_buy_threshold"]:
        normalized["overall_strong_buy_threshold"] = min(12.0, normalized["overall_buy_threshold"] + 2.0)
        notes.append("The strong-buy threshold was kept above the buy threshold.")

    normalized["overall_sell_threshold"] = -min(max(abs(float(normalized["overall_sell_threshold"])), 1.0), 6.0)
    normalized["overall_strong_sell_threshold"] = -min(
        max(abs(float(normalized["overall_strong_sell_threshold"])), 4.0), 12.0
    )
    if abs(normalized["overall_strong_sell_threshold"]) <= abs(normalized["overall_sell_threshold"]):
        normalized["overall_strong_sell_threshold"] = -min(12.0, abs(normalized["overall_sell_threshold"]) + 2.0)
        notes.append("The strong-sell threshold was kept below the sell threshold.")

    normalized["decision_hold_buffer"] = min(max(float(normalized["decision_hold_buffer"]), 0.0), 3.0)
    normalized["decision_min_confidence"] = min(max(float(normalized["decision_min_confidence"]), 35.0), 80.0)
    normalized["backtest_cooldown_days"] = float(min(max(float(normalized["backtest_cooldown_days"]), 0.0), 20.0))
    normalized["backtest_transaction_cost_bps"] = float(
        min(max(float(normalized["backtest_transaction_cost_bps"]), 0.0), 50.0)
    )
    normalized["backtest_min_position_change"] = float(
        min(max(float(normalized["backtest_min_position_change"]), 0.0), 0.5)
    )

    for key in ["weight_technical", "weight_fundamental", "weight_valuation", "weight_sentiment"]:
        normalized[key] = min(max(float(normalized[key]), 0.5), 1.5)

    normalized["trading_days_per_year"] = float(min(max(float(normalized["trading_days_per_year"]), 240.0), 260.0))
    return normalized, notes


def serialize_model_settings(settings):
    """Return a deterministic JSON string from a model settings dict."""
    normalized, _ = normalize_model_settings(settings)
    rounded = {}
    for key, value in normalized.items():
        rounded[key] = round(float(value), 6) if isinstance(value, float) else value
    return json.dumps(rounded, sort_keys=True)


def serialize_dcf_settings(settings):
    """Return a deterministic JSON string from a DCF settings dict."""
    normalized = normalize_dcf_settings(settings)
    rounded = {}
    for key, value in normalized.items():
        if isinstance(value, float):
            rounded[key] = round(value, 6)
        else:
            rounded[key] = value
    return json.dumps(rounded, sort_keys=True)


def get_model_presets():
    """Return all model presets normalised through :func:`normalize_model_settings`."""
    return {name: normalize_model_settings(values)[0] for name, values in MODEL_PRESETS.items()}


def get_assumption_fingerprint(settings=None):
    """Return a 10-character SHA-1 fingerprint of the active (or provided) settings."""
    active_settings = get_model_settings() if settings is None else settings
    payload = serialize_model_settings(active_settings)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def detect_matching_preset(settings=None):
    """Return the preset name that exactly matches *settings*, or ``"Custom"``."""
    active_settings = get_model_settings() if settings is None else settings
    for name, preset in get_model_presets().items():
        matches = all(np.isclose(active_settings[key], preset[key]) for key in preset)
        if matches:
            return name
    return "Custom"


def get_model_settings():
    """Return the active model settings from ``st.session_state``, normalizing first."""
    if "model_settings" not in st.session_state:
        normalized, _ = normalize_model_settings(get_default_model_settings())
        st.session_state.model_settings = normalized
        st.session_state.model_preset_name = get_default_preset_name()
        return normalized

    normalized, _ = normalize_model_settings(st.session_state.model_settings)
    st.session_state.model_settings = normalized
    st.session_state.model_preset_name = detect_matching_preset(normalized)
    return normalized


def calculate_assumption_drift(settings=None):
    """Return the mean percentage drift of *settings* from the default baseline.

    A value of 0 means the active settings are identical to the defaults.
    Returns a float between 0 and ~100 (no hard upper bound).
    """
    baseline = get_default_model_settings()
    active_settings = get_model_settings() if settings is None else settings
    deviations = []

    for key, default_value in baseline.items():
        current_value = float(active_settings.get(key, default_value))
        scale = max(abs(float(default_value)), 0.05)
        deviations.append(abs(current_value - float(default_value)) / scale)

    return np.mean(deviations) * 100 if deviations else 0.0
