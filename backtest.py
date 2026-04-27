# -*- coding: utf-8 -*-
"""
backtest.py — Position-generation, trade summarisation, and technical backtest
engine.

Functions:
    derive_backtest_positions
    summarize_backtest_trades
    compute_technical_backtest
    compute_composite_quarterly_backtest
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

import constants as const
from fetch import calculate_rsi, has_numeric_value, safe_divide, score_relative_multiple
from analytics_scoring import score_to_signal
from settings import get_model_settings
from utils_fmt import safe_num


# ---------------------------------------------------------------------------
# Position generation
# ---------------------------------------------------------------------------

def derive_backtest_positions(
    analysis: pd.DataFrame,
    settings: dict[str, Any] | None = None,
    stock_profile: dict[str, Any] | None = None,
) -> pd.Series:
    """
    Generate a daily position-size Series (0.0 – 1.45) from the technical
    signal columns already present in *analysis*.

    Position targets are modulated by *stock_profile* primary type and the
    active settings thresholds.
    """
    active_settings = get_model_settings() if settings is None else settings
    if analysis is None or analysis.empty:
        return pd.Series(dtype=float)

    primary_type = (stock_profile or {}).get("primary_type", "")
    trend_tolerance = active_settings["tech_trend_tolerance"]
    cooldown_days = int(round(active_settings["backtest_cooldown_days"]))
    full_target = 1.0
    core_target = 0.5
    danger_floor = 0.25
    core_reentry_cooldown = max(1, cooldown_days // 2)
    full_reentry_cooldown = cooldown_days
    entry_score_floor = 3
    hard_exit_score_floor = -5
    exit_break_multiplier = 1.5
    trailing_stop_threshold = -0.16
    allow_core_outside_bullish = False
    trim_to_core_on_non_bullish = False
    long_term_momentum_floor = max(active_settings["tech_momentum_threshold"] * 2, 0.08)
    initial_floor_if_not_bearish = 0.0
    growth_compounder_types = {"Growth Stocks"}
    overweight_target = None
    disable_bearish_reduction = False
    disable_exit = False
    ignore_initial_bearish_gate = False
    danger_score_floor = -2
    danger_momentum_multiplier = 1.0
    danger_requires_price_below_sma50 = False
    exit_requires_sma50_cross = False
    exit_momentum_floor = -long_term_momentum_floor

    if primary_type in growth_compounder_types:
        # Let leading compounders stay fully involved and briefly overweight the
        # strongest trend/momentum combinations so the replay can keep pace with
        # benchmark winners instead of constantly lagging them.
        core_target = 1.0
        full_target = 1.35
        danger_floor = 0.75
        allow_core_outside_bullish = True
        entry_score_floor = 1
        hard_exit_score_floor = -7
        exit_break_multiplier = 2.5
        trailing_stop_threshold = -0.30
        initial_floor_if_not_bearish = 1.0
    elif primary_type in {"Blue-Chip Stocks", "Large-Cap Stocks"}:
        # Treat established leaders more like core holdings: keep them invested,
        # allow a modest tactical add, and stop trading them out on ordinary
        # weakness where benchmark lag tends to come from.
        core_target = 1.0
        full_target = 1.15
        danger_floor = 1.15
        allow_core_outside_bullish = True
        entry_score_floor = 0
        initial_floor_if_not_bearish = 1.0
        disable_bearish_reduction = True
        disable_exit = True
        ignore_initial_bearish_gate = True
    elif primary_type == "Value Stocks":
        core_target = 1.0
        full_target = 1.0
        danger_floor = 1.0
        allow_core_outside_bullish = True
        entry_score_floor = 0
        initial_floor_if_not_bearish = 1.0
        disable_bearish_reduction = True
        disable_exit = True
        ignore_initial_bearish_gate = True
    elif primary_type in {"Dividend / Income Stocks", "Defensive Stocks"}:
        core_target = 1.0
        full_target = 1.0
        danger_floor = 1.0
        allow_core_outside_bullish = True
        entry_score_floor = 0
        initial_floor_if_not_bearish = 1.0
        disable_bearish_reduction = True
        disable_exit = True
        ignore_initial_bearish_gate = True
    elif primary_type == "Cyclical Stocks":
        core_target = 0.5
        danger_floor = 0.0
        hard_exit_score_floor = -4
        exit_break_multiplier = 1.2
        trailing_stop_threshold = -0.12
        trim_to_core_on_non_bullish = True
    elif primary_type == "Mid-Cap Stocks":
        core_target = 0.5
        danger_floor = 0.25
        exit_break_multiplier = 1.4
        trailing_stop_threshold = -0.14
    elif primary_type == "Small-Cap Stocks":
        core_target = 0.25
        full_target = 0.75
        danger_floor = 0.0
        entry_score_floor = 4
        hard_exit_score_floor = -4
        exit_break_multiplier = 1.2
        trailing_stop_threshold = -0.10
        trim_to_core_on_non_bullish = True
    elif primary_type == "Speculative / Penny Stocks":
        core_target = 0.0
        full_target = 0.5
        danger_floor = 0.0
        entry_score_floor = 4
        hard_exit_score_floor = -4
        exit_break_multiplier = 1.0
        trailing_stop_threshold = -0.08
        trim_to_core_on_non_bullish = True

    bullish_regime = (
        analysis["Close"].ge(analysis["SMA_200"] * (1 + trend_tolerance))
        & (
            analysis["SMA_50"].ge(analysis["SMA_200"] * (1 + trend_tolerance / 2))
            | analysis["Momentum_1Y"].gt(0)
        )
    )
    bearish_regime = (
        analysis["Close"].le(analysis["SMA_200"] * (1 - trend_tolerance))
        & (
            analysis["SMA_50"].le(analysis["SMA_200"] * (1 - trend_tolerance / 2))
            | analysis["Momentum_1Y"].lt(0)
        )
    )
    bullish_regime = bullish_regime.fillna(False)
    bearish_regime = bearish_regime.fillna(False)
    macd_bearish = analysis["MACD"].lt(analysis["MACD_Signal_Line"]).fillna(False)
    macd_bullish = analysis["MACD"].ge(analysis["MACD_Signal_Line"]).fillna(False)
    core_regime = bullish_regime | (~bearish_regime if allow_core_outside_bullish else False)
    trailing_stop_breach = analysis.get("Trailing_Drawdown_Quarter", pd.Series(index=analysis.index, dtype=float)).le(
        trailing_stop_threshold
    ).fillna(False)
    strong_bullish = (
        bullish_regime
        & macd_bullish
        & analysis["Tech Score"].ge(4)
        & analysis["Momentum_1Y"].gt(0.15).fillna(False)
        & analysis["Close"].ge(analysis["SMA_50"]).fillna(False)
    )
    entry_signal = core_regime & analysis["Tech Score"].ge(entry_score_floor)
    add_signal = bullish_regime & analysis["Tech Score"].ge(max(entry_score_floor, 2))
    danger_reduce = pd.Series(False, index=analysis.index, dtype=bool)
    if not disable_bearish_reduction:
        danger_reduce = (
            (bearish_regime | trailing_stop_breach)
            & (
                analysis["Tech Score"].le(danger_score_floor)
                | (
                    macd_bearish
                    & analysis["Momentum_1M"].lt(
                        -active_settings["tech_momentum_threshold"] * danger_momentum_multiplier
                    ).fillna(False)
                    & (
                        analysis["Close"].lt(analysis["SMA_50"]).fillna(False)
                        if danger_requires_price_below_sma50
                        else True
                    )
                )
            )
        )
    exit_signal = pd.Series(False, index=analysis.index, dtype=bool)
    if not disable_exit:
        exit_signal = (
            (analysis["Tech Score"].le(hard_exit_score_floor) & macd_bearish)
            | (
                bearish_regime
                & analysis["Close"].le(analysis["SMA_200"] * (1 - trend_tolerance * exit_break_multiplier)).fillna(False)
                & macd_bearish
                & (
                    analysis["SMA_50"].le(analysis["SMA_200"]).fillna(False)
                    if exit_requires_sma50_cross
                    else True
                )
                & analysis["Momentum_1Y"].lt(exit_momentum_floor).fillna(False)
            )
        )

    if primary_type in growth_compounder_types:
        overweight_target = 1.45
        danger_score_floor = -4
        danger_momentum_multiplier = 1.5
        danger_requires_price_below_sma50 = True
        exit_requires_sma50_cross = True
        exit_momentum_floor = -0.22
        danger_reduce = (
            (bearish_regime | trailing_stop_breach)
            & (
                analysis["Tech Score"].le(danger_score_floor)
                | (
                    macd_bearish
                    & analysis["Close"].lt(analysis["SMA_50"]).fillna(False)
                    & analysis["Momentum_1M"].lt(
                        -active_settings["tech_momentum_threshold"] * danger_momentum_multiplier
                    ).fillna(False)
                )
            )
        )
        exit_signal = (
            (analysis["Tech Score"].le(hard_exit_score_floor) & macd_bearish)
            | (
                bearish_regime
                & analysis["SMA_50"].le(analysis["SMA_200"]).fillna(False)
                & analysis["Close"].le(analysis["SMA_200"] * (1 - trend_tolerance * exit_break_multiplier)).fillna(False)
                & analysis["Momentum_1Y"].lt(exit_momentum_floor).fillna(False)
            )
        )

    positions = []
    first_bearish = bool(bearish_regime.iloc[0]) if len(bearish_regime) else False
    current_position = 0.0 if first_bearish and not ignore_initial_bearish_gate else initial_floor_if_not_bearish
    days_since_change = full_reentry_cooldown
    for is_bullish, is_bearish, enter_now, add_now, danger_now, exit_now, strong_now in zip(
        bullish_regime,
        bearish_regime,
        entry_signal,
        add_signal,
        danger_reduce,
        exit_signal,
        strong_bullish,
    ):
        target_position = current_position
        if exit_now:
            target_position = 0.0
        elif is_bullish:
            if current_position < core_target and enter_now and days_since_change >= core_reentry_cooldown:
                target_position = core_target
            if add_now and days_since_change >= full_reentry_cooldown:
                target_position = full_target
            if overweight_target is not None and strong_now and days_since_change >= core_reentry_cooldown:
                target_position = max(target_position, overweight_target)
        elif is_bearish:
            if danger_now and current_position > danger_floor:
                target_position = danger_floor
        else:
            if trim_to_core_on_non_bullish and current_position > core_target:
                target_position = core_target
            elif allow_core_outside_bullish and enter_now and current_position < core_target and days_since_change >= core_reentry_cooldown:
                target_position = core_target

        if target_position != current_position:
            current_position = target_position
            days_since_change = 0
        else:
            days_since_change += 1
        positions.append(current_position)

    return pd.Series(positions, index=analysis.index, dtype=float)


# ---------------------------------------------------------------------------
# Trade summarisation
# ---------------------------------------------------------------------------

def summarize_backtest_trades(analysis: pd.DataFrame) -> dict[str, Any]:
    """
    Walk the Position column with FIFO lot accounting and return
    (closed_trades_df, summary_dict).
    """
    if analysis is None or analysis.empty or "Close" not in analysis.columns or "Position" not in analysis.columns:
        return pd.DataFrame(), {
            "Closed Trades": 0,
            "Win Rate": None,
            "Average Trade Return": None,
        }

    open_lots = []
    closed_trades = []
    previous_position = 0.0

    for row in analysis.itertuples():
        date = row.Index
        close = safe_num(row.Close)
        current_position = safe_num(row.Position) or 0.0
        if close is None:
            previous_position = current_position
            continue

        position_change = round(current_position - previous_position, 6)
        if position_change > 0:
            open_lots.append(
                {
                    "entry_date": date,
                    "entry_price": close,
                    "size": position_change,
                }
            )
        elif position_change < 0:
            remaining_to_close = abs(position_change)
            while remaining_to_close > 1e-9 and open_lots:
                lot = open_lots[0]
                closed_size = min(lot["size"], remaining_to_close)
                closed_trades.append(
                    {
                        "Entry Date": lot["entry_date"],
                        "Exit Date": date,
                        "Entry Price": lot["entry_price"],
                        "Exit Price": close,
                        "Position Size": closed_size,
                        "Return": safe_divide(close - lot["entry_price"], lot["entry_price"]),
                        "Holding Days": (date - lot["entry_date"]).days if hasattr(date, "__sub__") else None,
                    }
                )
                lot["size"] -= closed_size
                remaining_to_close -= closed_size
                if lot["size"] <= 1e-9:
                    open_lots.pop(0)

        previous_position = current_position

    if not closed_trades:
        return pd.DataFrame(), {
            "Closed Trades": 0,
            "Win Rate": None,
            "Average Trade Return": None,
        }

    closed_trades_df = pd.DataFrame(closed_trades)
    return closed_trades_df, {
        "Closed Trades": len(closed_trades_df),
        "Win Rate": (closed_trades_df["Return"] > 0).mean(),
        "Average Trade Return": closed_trades_df["Return"].mean(),
    }


# ---------------------------------------------------------------------------
# Factor IC diagnostics
# ---------------------------------------------------------------------------

def compute_factor_ic(analysis: pd.DataFrame, rolling_window: int = 252) -> dict[str, Any]:
    """
    Compute Information Coefficient (IC) diagnostics for the Tech Score.

    IC is the Spearman rank correlation between the Tech Score on each date and
    the stock's actual forward return over 1M / 3M / 12M horizons.  Values
    above ~0.05 indicate meaningful predictive signal.

    Returns a dict with:
        ic_summary    – overall IC and hit-rate per forward horizon
        ic_by_window  – IC matrix sliced over full / last-3Y / last-1Y history
        rolling_ic    – monthly-sampled rolling IC DataFrame (1M and 3M horizons)
        sub_signal_ic – per-sub-signal IC for each horizon
    """
    from scipy.stats import spearmanr  # noqa: PLC0415

    close = analysis["Close"]
    horizons = {"1M": 22, "3M": 63, "12M": 252}
    fwd = {label: close.shift(-days) / close - 1 for label, days in horizons.items()}

    def _ic_hit(factor: pd.Series, ret: pd.Series):
        valid = factor.notna() & ret.notna()
        n = int(valid.sum())
        if n < 30:
            return None, None, n
        ic_val, _ = spearmanr(factor[valid], ret[valid])
        nonzero = valid & (factor != 0)
        hit = (
            float((np.sign(factor[nonzero]) == np.sign(ret[nonzero])).mean())
            if nonzero.sum() > 10
            else None
        )
        return float(ic_val), hit, n

    score = analysis["Tech Score"]
    n_total = len(analysis)

    # Overall IC and hit rate across all available data
    ic_summary: dict[str, Any] = {}
    for label, ret in fwd.items():
        ic_val, hit, n = _ic_hit(score, ret)
        ic_summary[label] = {"ic": ic_val, "hit_rate": hit, "n": n}

    # IC sliced over fixed lookback windows
    ic_by_window: dict[str, dict[str, Any]] = {}
    for win_label, win_days in [("Full", n_total), ("3Y", 756), ("1Y", 252)]:
        if n_total < win_days:
            continue
        slice_score = score.iloc[-win_days:]
        ic_by_window[win_label] = {}
        for label, ret in fwd.items():
            ic_val, _, _ = _ic_hit(slice_score, ret.iloc[-win_days:])
            ic_by_window[win_label][label] = ic_val

    # Rolling IC — 252-day window, monthly steps
    rolling_records: list[dict[str, Any]] = []
    step = 22
    for i in range(rolling_window, n_total, step):
        sl = slice(i - rolling_window, i)
        row: dict[str, Any] = {"Date": analysis.index[i - 1]}
        for label in ("1M", "3M"):
            ic_val, _, _ = _ic_hit(score.iloc[sl], fwd[label].iloc[sl])
            row[f"IC_{label}"] = ic_val
        rolling_records.append(row)
    rolling_ic_df = (
        pd.DataFrame(rolling_records).set_index("Date") if rolling_records else pd.DataFrame()
    )

    # Sub-signal IC
    sub_cols = [c for c in analysis.columns if c.startswith("sig_")]
    sub_signal_ic: dict[str, dict[str, Any]] = {}
    for col in sub_cols:
        sig_name = col.replace("sig_", "")
        sub_signal_ic[sig_name] = {}
        for label, ret in fwd.items():
            ic_val, _, _ = _ic_hit(analysis[col], ret)
            sub_signal_ic[sig_name][label] = ic_val

    return {
        "ic_summary": ic_summary,
        "ic_by_window": ic_by_window,
        "rolling_ic": rolling_ic_df,
        "sub_signal_ic": sub_signal_ic,
    }


# ---------------------------------------------------------------------------
# Full backtest engine
# ---------------------------------------------------------------------------

def compute_technical_backtest(
    hist: pd.DataFrame,
    settings: dict[str, Any] | None = None,
    stock_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute a full technical backtest on *hist* (a DataFrame with a Close
    column) and return a result dict with keys: history, trade_log,
    closed_trades, metrics, stock_profile.

    Returns None when there is insufficient data.
    """
    active_settings = get_model_settings() if settings is None else settings
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None

    close = hist["Close"].dropna().copy()
    if len(close) < 250:
        return None

    analysis = pd.DataFrame(index=close.index)
    analysis["Close"] = close
    analysis["RSI"] = calculate_rsi(close)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    analysis["MACD"] = ema12 - ema26
    analysis["MACD_Signal_Line"] = analysis["MACD"].ewm(span=9, adjust=False).mean()
    analysis["SMA_50"] = close.rolling(50).mean()
    analysis["SMA_200"] = close.rolling(200).mean()
    analysis["Momentum_1M"] = close / close.shift(22) - 1
    analysis["Momentum_1Y"] = close / close.shift(252) - 1
    analysis["Volatility_1M"] = close.pct_change().rolling(22).std() * np.sqrt(252)
    analysis["Volatility_1Y"] = close.pct_change().rolling(252).std() * np.sqrt(252)
    analysis["Momentum_1M_Risk_Adjusted"] = analysis["Momentum_1M"] / (analysis["Volatility_1M"] / np.sqrt(12))
    analysis["Rolling_High_252"] = close.rolling(252, min_periods=20).max()
    analysis["Rolling_Low_252"] = close.rolling(252, min_periods=20).min()
    analysis["Range_Position_252"] = (close - analysis["Rolling_Low_252"]) / (
        analysis["Rolling_High_252"] - analysis["Rolling_Low_252"]
    )
    analysis["Distance_52W_High"] = (close - analysis["Rolling_High_252"]) / analysis["Rolling_High_252"]
    analysis["Trend_Strength"] = (
        ((close / analysis["SMA_200"]) - 1).clip(-0.25, 0.25).fillna(0) * 100
        + ((analysis["SMA_50"] / analysis["SMA_200"]) - 1).clip(-0.25, 0.25).fillna(0) * 120
        + analysis["Momentum_1Y"].clip(-0.3125, 0.3125).fillna(0) * 80
    )
    analysis["Quarter_High"] = close.rolling(63, min_periods=20).max()
    analysis["Trailing_Drawdown_Quarter"] = (close / analysis["Quarter_High"]) - 1

    trend_tolerance = active_settings["tech_trend_tolerance"]
    extension_limit = active_settings["tech_extension_limit"]
    tech_score = pd.Series(0, index=analysis.index, dtype=float)
    tech_score += np.where(
        analysis["SMA_200"].notna(),
        np.where(
            analysis["Close"] >= analysis["SMA_200"] * (1 + trend_tolerance),
            1,
            np.where(analysis["Close"] <= analysis["SMA_200"] * (1 - trend_tolerance), -1, 0),
        ),
        0,
    )
    tech_score += np.where(
        analysis["SMA_50"].notna() & analysis["SMA_200"].notna(),
        np.where(
            analysis["SMA_50"] >= analysis["SMA_200"] * (1 + trend_tolerance / 2),
            1,
            np.where(analysis["SMA_50"] <= analysis["SMA_200"] * (1 - trend_tolerance / 2), -1, 0),
        ),
        0,
    )
    tech_score += np.where(
        analysis["SMA_50"].notna(),
        np.where(
            analysis["Close"] >= analysis["SMA_50"] * (1 + trend_tolerance / 2),
            1,
            np.where(analysis["Close"] <= analysis["SMA_50"] * (1 - trend_tolerance / 2), -1, 0),
        ),
        0,
    )
    tech_score += np.where(
        analysis["RSI"] < active_settings["tech_rsi_oversold"],
        np.where(analysis["Close"] >= analysis["SMA_200"] * 0.95, 2, 1),
        0,
    )
    tech_score += np.where(
        analysis["RSI"] > active_settings["tech_rsi_overbought"],
        np.where(analysis["Close"] <= analysis["SMA_50"] * 1.02, -2, -1),
        0,
    )
    tech_score += np.where(
        analysis["MACD"].notna() & analysis["MACD_Signal_Line"].notna(),
        np.where(analysis["MACD"] > analysis["MACD_Signal_Line"], 1, -1),
        0,
    )
    tech_score += np.where(analysis["MACD"] > 0, 1, np.where(analysis["MACD"] < 0, -1, 0))
    tech_score += np.where(analysis["Momentum_1M"] > active_settings["tech_momentum_threshold"], 1, 0)
    tech_score += np.where(analysis["Momentum_1M"] < -active_settings["tech_momentum_threshold"], -1, 0)
    tech_score += np.where(analysis["Momentum_1M_Risk_Adjusted"] > 0.75, 1, 0)
    tech_score += np.where(analysis["Momentum_1M_Risk_Adjusted"] < -0.75, -1, 0)
    long_term_momentum_threshold = max(active_settings["tech_momentum_threshold"] * 3, 0.10)
    tech_score += np.where(analysis["Momentum_1Y"] > long_term_momentum_threshold, 1, 0)
    tech_score += np.where(analysis["Momentum_1Y"] < -long_term_momentum_threshold, -1, 0)
    tech_score += np.where(analysis["Trend_Strength"] > 30, 1, 0)
    tech_score += np.where(analysis["Trend_Strength"] < -30, -1, 0)
    tech_score += np.where(
        analysis["Range_Position_252"].ge(0.80) & analysis["Close"].ge(analysis["SMA_200"]),
        1,
        0,
    )
    tech_score += np.where(
        analysis["Range_Position_252"].le(0.20) & analysis["Close"].le(analysis["SMA_200"]),
        -1,
        0,
    )
    tech_score += np.where(
        analysis["RSI"].shift(1).le(active_settings["tech_rsi_oversold"])
        & analysis["RSI"].gt(active_settings["tech_rsi_oversold"])
        & analysis["MACD"].ge(analysis["MACD_Signal_Line"]),
        1,
        0,
    )
    tech_score += np.where(
        analysis["Close"].ge(analysis["SMA_50"] * (1 + extension_limit))
        & analysis["RSI"].ge(active_settings["tech_rsi_overbought"] - 5),
        -1,
        0,
    )
    tech_score += np.where(
        analysis["Close"].le(analysis["SMA_50"] * (1 - extension_limit))
        & analysis["RSI"].le(active_settings["tech_rsi_oversold"] + 5),
        1,
        0,
    )

    # --- Sub-signal components for IC breakdown ---
    analysis["sig_Trend"] = (
        np.where(
            analysis["SMA_200"].notna(),
            np.where(
                analysis["Close"] >= analysis["SMA_200"] * (1 + trend_tolerance), 1,
                np.where(analysis["Close"] <= analysis["SMA_200"] * (1 - trend_tolerance), -1, 0),
            ),
            0,
        )
        + np.where(
            analysis["SMA_50"].notna() & analysis["SMA_200"].notna(),
            np.where(
                analysis["SMA_50"] >= analysis["SMA_200"] * (1 + trend_tolerance / 2), 1,
                np.where(analysis["SMA_50"] <= analysis["SMA_200"] * (1 - trend_tolerance / 2), -1, 0),
            ),
            0,
        )
        + np.where(
            analysis["SMA_50"].notna(),
            np.where(
                analysis["Close"] >= analysis["SMA_50"] * (1 + trend_tolerance / 2), 1,
                np.where(analysis["Close"] <= analysis["SMA_50"] * (1 - trend_tolerance / 2), -1, 0),
            ),
            0,
        )
    )
    analysis["sig_Momentum"] = (
        np.where(analysis["Momentum_1M"] > active_settings["tech_momentum_threshold"], 1, 0)
        + np.where(analysis["Momentum_1M"] < -active_settings["tech_momentum_threshold"], -1, 0)
        + np.where(analysis["Momentum_1M_Risk_Adjusted"] > 0.75, 1, 0)
        + np.where(analysis["Momentum_1M_Risk_Adjusted"] < -0.75, -1, 0)
        + np.where(analysis["Momentum_1Y"] > long_term_momentum_threshold, 1, 0)
        + np.where(analysis["Momentum_1Y"] < -long_term_momentum_threshold, -1, 0)
        + np.where(analysis["Trend_Strength"] > 30, 1, 0)
        + np.where(analysis["Trend_Strength"] < -30, -1, 0)
    )
    analysis["sig_Oscillator"] = (
        np.where(
            analysis["RSI"] < active_settings["tech_rsi_oversold"],
            np.where(analysis["Close"] >= analysis["SMA_200"] * 0.95, 2, 1),
            0,
        )
        + np.where(
            analysis["RSI"] > active_settings["tech_rsi_overbought"],
            np.where(analysis["Close"] <= analysis["SMA_50"] * 1.02, -2, -1),
            0,
        )
        + np.where(
            analysis["MACD"].notna() & analysis["MACD_Signal_Line"].notna(),
            np.where(analysis["MACD"] > analysis["MACD_Signal_Line"], 1, -1),
            0,
        )
        + np.where(analysis["MACD"] > 0, 1, np.where(analysis["MACD"] < 0, -1, 0))
    )
    analysis["sig_Range"] = (
        np.where(
            analysis["Range_Position_252"].ge(0.80) & analysis["Close"].ge(analysis["SMA_200"]), 1, 0
        )
        + np.where(
            analysis["Range_Position_252"].le(0.20) & analysis["Close"].le(analysis["SMA_200"]), -1, 0
        )
        + np.where(
            analysis["RSI"].shift(1).le(active_settings["tech_rsi_oversold"])
            & analysis["RSI"].gt(active_settings["tech_rsi_oversold"])
            & analysis["MACD"].ge(analysis["MACD_Signal_Line"]),
            1,
            0,
        )
    )
    analysis["sig_Extension"] = (
        np.where(
            analysis["Close"].ge(analysis["SMA_50"] * (1 + extension_limit))
            & analysis["RSI"].ge(active_settings["tech_rsi_overbought"] - 5),
            -1,
            0,
        )
        + np.where(
            analysis["Close"].le(analysis["SMA_50"] * (1 - extension_limit))
            & analysis["RSI"].le(active_settings["tech_rsi_oversold"] + 5),
            1,
            0,
        )
    )

    analysis["Tech Score"] = tech_score
    analysis["Signal"] = analysis["Tech Score"].apply(score_to_signal)
    analysis["Position"] = derive_backtest_positions(analysis, active_settings, stock_profile=stock_profile)
    min_position_change = active_settings.get("backtest_min_position_change", 0.0)
    if min_position_change > 0:
        positions = analysis["Position"].copy()
        prev = positions.iloc[0]
        for i in range(1, len(positions)):
            if abs(positions.iloc[i] - prev) < min_position_change:
                positions.iloc[i] = prev
            else:
                prev = positions.iloc[i]
        analysis["Position"] = positions
    analysis["Benchmark Return"] = analysis["Close"].pct_change().fillna(0.0)
    trade_points = analysis["Position"].diff().fillna(analysis["Position"])
    trading_cost_rate = active_settings.get("backtest_transaction_cost_bps", 0.0) / 10000
    analysis["Trading Cost"] = trade_points.abs() * trading_cost_rate
    _trading_days_yr = active_settings.get("trading_days_per_year", 252.0)
    years_in_period = len(analysis) / _trading_days_yr
    annual_turnover = trade_points.abs().sum() / years_in_period if years_in_period > 0 else 0.0
    analysis["Strategy Return"] = (
        analysis["Position"].shift(1).fillna(0.0) * analysis["Benchmark Return"] - analysis["Trading Cost"]
    )
    analysis["Benchmark Equity"] = (1 + analysis["Benchmark Return"]).cumprod()
    analysis["Strategy Equity"] = (1 + analysis["Strategy Return"]).cumprod()
    strategy_total_return = analysis["Strategy Equity"].iloc[-1] - 1
    benchmark_total_return = analysis["Benchmark Equity"].iloc[-1] - 1
    average_exposure = analysis["Position"].mean()
    upside_capture = (
        safe_divide(analysis["Strategy Equity"].iloc[-1], analysis["Benchmark Equity"].iloc[-1])
        if benchmark_total_return > 0
        else None
    )

    trading_days = active_settings["trading_days_per_year"]
    strategy_ann_return = analysis["Strategy Return"].mean() * trading_days
    strategy_vol = analysis["Strategy Return"].std() * np.sqrt(trading_days)
    benchmark_ann_return = analysis["Benchmark Return"].mean() * trading_days
    benchmark_vol = analysis["Benchmark Return"].std() * np.sqrt(trading_days)

    strategy_drawdown = analysis["Strategy Equity"] / analysis["Strategy Equity"].cummax() - 1
    benchmark_drawdown = analysis["Benchmark Equity"] / analysis["Benchmark Equity"].cummax() - 1

    trade_log = pd.DataFrame(
        {
            "Date": analysis.index,
            "Action": np.select(
                [
                    trade_points >= 0.75,
                    trade_points > 0,
                    trade_points <= -0.75,
                    trade_points < 0,
                ],
                ["Enter", "Add", "Exit", "Reduce"],
                default=None,
            ),
            "Close": analysis["Close"],
            "Signal": analysis["Signal"],
            "Tech Score": analysis["Tech Score"],
            "Position": analysis["Position"],
            "Trading Cost": analysis["Trading Cost"],
        }
    ).dropna(subset=["Action"])
    closed_trades_df, trade_summary = summarize_backtest_trades(analysis)
    ic_diagnostics = compute_factor_ic(analysis)

    metrics = {
        "Strategy Total Return": strategy_total_return,
        "Benchmark Total Return": benchmark_total_return,
        "Relative Return": analysis["Strategy Equity"].iloc[-1] - analysis["Benchmark Equity"].iloc[-1],
        "Strategy Annual Return": strategy_ann_return,
        "Benchmark Annual Return": benchmark_ann_return,
        "Strategy Volatility": strategy_vol,
        "Benchmark Volatility": benchmark_vol,
        "Strategy Sharpe": safe_divide(strategy_ann_return, strategy_vol),
        "Benchmark Sharpe": safe_divide(benchmark_ann_return, benchmark_vol),
        "Strategy Max Drawdown": strategy_drawdown.min(),
        "Benchmark Max Drawdown": benchmark_drawdown.min(),
        "Trading Costs": analysis["Trading Cost"].sum(),
        "Average Exposure": average_exposure,
        "Upside Capture": upside_capture,
        "Annual Turnover": trade_points.abs().sum() / max(len(analysis) / trading_days, 1),
        "Position Changes": len(trade_log),
        "Closed Trades": trade_summary["Closed Trades"],
        "Win Rate": trade_summary["Win Rate"],
        "Average Trade Return": trade_summary["Average Trade Return"],
        "Annual Turnover": annual_turnover,
    }

    return {
        "history": analysis.reset_index().rename(columns={"index": "Date"}),
        "trade_log": trade_log.reset_index(drop=True),
        "closed_trades": closed_trades_df,
        "metrics": metrics,
        "stock_profile": stock_profile or {},
        "ic_diagnostics": ic_diagnostics,
    }


# ---------------------------------------------------------------------------
# Composite model walk-forward validation
# ---------------------------------------------------------------------------

def compute_composite_quarterly_backtest(
    ticker: str,
    hist: pd.DataFrame,
    model_settings: dict[str, Any] | None = None,
    sector: str = "Unknown",
) -> dict | None:
    """Walk-forward composite model validation using quarterly fundamentals.

    For each calendar quarter where yfinance provides fundamental data, rebuilds
    F-score, V-score, and Tech Score, assembles the composite verdict, then tracks
    forward 1M/3M/12M price returns. Returns a hit-rate table grouped by verdict
    bucket, so you can see whether STRONG BUY != HOLD in actual outcomes.

    Returns None when fewer than 4 quarters of aligned fundamental data are
    available, or when price history is too short.

    Caveats (shown in UI):
    - Uses yfinance quarterly filings (~8 quarters depth typical)
    - Static sector benchmarks for valuation (no live peer reconstruction)
    - EV/EBITDA and PEG excluded from historical V-score
    - Shares outstanding: current figure used (not historically adjusted)
    - 45-day filing lag approximated
    - No guardrails, hold buffer, or confidence guard applied
    - Sentiment engine excluded (score=0, same as live model)
    """
    active_settings = get_model_settings() if model_settings is None else model_settings

    if hist is None or hist.empty or "Close" not in hist.columns:
        return None

    close = hist["Close"].dropna().copy()
    if len(close) < 250:
        return None

    # Ensure timezone-naive DatetimeIndex for consistent slicing
    if close.index.tzinfo is not None:
        close.index = close.index.tz_localize(None)

    # --- Vectorized tech score (same computation as compute_technical_backtest) ---
    analysis = pd.DataFrame(index=close.index)
    analysis["Close"] = close
    analysis["RSI"] = calculate_rsi(close)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    analysis["MACD"] = ema12 - ema26
    analysis["MACD_Signal_Line"] = analysis["MACD"].ewm(span=9, adjust=False).mean()
    analysis["SMA_50"] = close.rolling(50).mean()
    analysis["SMA_200"] = close.rolling(200).mean()
    analysis["Momentum_1M"] = close / close.shift(22) - 1
    analysis["Momentum_1Y"] = close / close.shift(252) - 1
    analysis["Volatility_1M"] = close.pct_change().rolling(22).std() * np.sqrt(252)
    analysis["Momentum_1M_Risk_Adjusted"] = analysis["Momentum_1M"] / (
        analysis["Volatility_1M"] / np.sqrt(12)
    )
    analysis["Rolling_High_252"] = close.rolling(252, min_periods=20).max()
    analysis["Rolling_Low_252"] = close.rolling(252, min_periods=20).min()
    analysis["Range_Position_252"] = (close - analysis["Rolling_Low_252"]) / (
        analysis["Rolling_High_252"] - analysis["Rolling_Low_252"]
    )
    analysis["Trend_Strength"] = (
        ((close / analysis["SMA_200"]) - 1).clip(-0.25, 0.25).fillna(0) * 100
        + ((analysis["SMA_50"] / analysis["SMA_200"]) - 1).clip(-0.25, 0.25).fillna(0) * 120
        + analysis["Momentum_1Y"].clip(-0.3125, 0.3125).fillna(0) * 80
    )
    trend_tolerance = active_settings["tech_trend_tolerance"]
    extension_limit = active_settings["tech_extension_limit"]
    tech_score_series = pd.Series(0, index=analysis.index, dtype=float)
    tech_score_series += np.where(
        analysis["SMA_200"].notna(),
        np.where(analysis["Close"] >= analysis["SMA_200"] * (1 + trend_tolerance), 1,
                 np.where(analysis["Close"] <= analysis["SMA_200"] * (1 - trend_tolerance), -1, 0)),
        0,
    )
    tech_score_series += np.where(
        analysis["SMA_50"].notna() & analysis["SMA_200"].notna(),
        np.where(analysis["SMA_50"] >= analysis["SMA_200"] * (1 + trend_tolerance / 2), 1,
                 np.where(analysis["SMA_50"] <= analysis["SMA_200"] * (1 - trend_tolerance / 2), -1, 0)),
        0,
    )
    tech_score_series += np.where(
        analysis["SMA_50"].notna(),
        np.where(analysis["Close"] >= analysis["SMA_50"] * (1 + trend_tolerance / 2), 1,
                 np.where(analysis["Close"] <= analysis["SMA_50"] * (1 - trend_tolerance / 2), -1, 0)),
        0,
    )
    tech_score_series += np.where(
        analysis["RSI"] < active_settings["tech_rsi_oversold"],
        np.where(analysis["Close"] >= analysis["SMA_200"] * 0.95, 2, 1), 0,
    )
    tech_score_series += np.where(
        analysis["RSI"] > active_settings["tech_rsi_overbought"],
        np.where(analysis["Close"] <= analysis["SMA_50"] * 1.02, -2, -1), 0,
    )
    tech_score_series += np.where(
        analysis["MACD"].notna() & analysis["MACD_Signal_Line"].notna(),
        np.where(analysis["MACD"] > analysis["MACD_Signal_Line"], 1, -1), 0,
    )
    tech_score_series += np.where(analysis["MACD"] > 0, 1, np.where(analysis["MACD"] < 0, -1, 0))
    tech_score_series += np.where(analysis["Momentum_1M"] > active_settings["tech_momentum_threshold"], 1, 0)
    tech_score_series += np.where(analysis["Momentum_1M"] < -active_settings["tech_momentum_threshold"], -1, 0)
    tech_score_series += np.where(analysis["Momentum_1M_Risk_Adjusted"] > 0.75, 1, 0)
    tech_score_series += np.where(analysis["Momentum_1M_Risk_Adjusted"] < -0.75, -1, 0)
    lt_mom_thresh = max(active_settings["tech_momentum_threshold"] * 3, 0.10)
    tech_score_series += np.where(analysis["Momentum_1Y"] > lt_mom_thresh, 1, 0)
    tech_score_series += np.where(analysis["Momentum_1Y"] < -lt_mom_thresh, -1, 0)
    tech_score_series += np.where(analysis["Trend_Strength"] > 30, 1, 0)
    tech_score_series += np.where(analysis["Trend_Strength"] < -30, -1, 0)
    tech_score_series += np.where(
        analysis["Range_Position_252"].ge(0.80) & analysis["Close"].ge(analysis["SMA_200"]), 1, 0,
    )
    tech_score_series += np.where(
        analysis["Range_Position_252"].le(0.20) & analysis["Close"].le(analysis["SMA_200"]), -1, 0,
    )
    tech_score_series += np.where(
        analysis["RSI"].shift(1).le(active_settings["tech_rsi_oversold"])
        & analysis["RSI"].gt(active_settings["tech_rsi_oversold"])
        & analysis["MACD"].ge(analysis["MACD_Signal_Line"]),
        1, 0,
    )
    tech_score_series += np.where(
        analysis["Close"].ge(analysis["SMA_50"] * (1 + extension_limit))
        & analysis["RSI"].ge(active_settings["tech_rsi_overbought"] - 5),
        -1, 0,
    )
    tech_score_series += np.where(
        analysis["Close"].le(analysis["SMA_50"] * (1 - extension_limit))
        & analysis["RSI"].le(active_settings["tech_rsi_oversold"] + 5),
        1, 0,
    )

    # --- Fetch quarterly fundamentals ---
    try:
        t = yf.Ticker(ticker)
        qf = t.quarterly_financials
        qbs = t.quarterly_balance_sheet
        shares_outstanding = safe_num(t.info.get("sharesOutstanding"))
    except Exception:
        return None

    if qf is None or qf.empty or qbs is None or qbs.empty:
        return None

    def _normalize_ts(ts):
        ts = pd.Timestamp(ts)
        return ts.tz_localize(None) if ts.tzinfo else ts

    qf_dates = {_normalize_ts(d) for d in qf.columns}
    qbs_dates = {_normalize_ts(d) for d in qbs.columns}
    common_dates = sorted(qf_dates & qbs_dates)

    if len(common_dates) < 4:
        return None

    # Rebuild normalized column maps so we can look up by normalised key
    qf_norm = {_normalize_ts(c): c for c in qf.columns}
    qbs_norm = {_normalize_ts(c): c for c in qbs.columns}

    bench = const.get_sector_benchmarks(sector, active_settings)
    strong_buy_thresh = active_settings["overall_strong_buy_threshold"]
    buy_thresh = active_settings["overall_buy_threshold"]
    sell_thresh = active_settings["overall_sell_threshold"]
    strong_sell_thresh = active_settings["overall_strong_sell_threshold"]
    FILING_LAG_DAYS = 45

    def _get_row(df, *names):
        for name in names:
            if name in df.index:
                return df.loc[name]
        return None

    def _q_val(row, norm_date, norm_col_map):
        if row is None:
            return None
        orig_col = norm_col_map.get(norm_date)
        if orig_col is None or orig_col not in row.index:
            return None
        v = row[orig_col]
        return None if pd.isna(v) else float(v)

    def _ttm_sum(row, norm_col_map, dates_up_to, n=4):
        if row is None:
            return None
        window = [d for d in dates_up_to if norm_col_map.get(d) in row.index][-n:]
        vals = [float(row[norm_col_map[d]]) for d in window
                if not pd.isna(row[norm_col_map[d]])]
        return sum(vals) if vals else None

    rev_row = _get_row(qf, "Total Revenue")
    ni_row = _get_row(qf, "Net Income")
    ca_row = _get_row(qbs, "Current Assets")
    cl_row = _get_row(qbs, "Current Liabilities", "Total Current Liabilities")
    debt_row = _get_row(qbs, "Total Debt", "Long Term Debt")
    eq_row = _get_row(qbs, "Stockholders Equity", "Total Equity Gross Minority Interest",
                      "Common Stock Equity")

    warnings_list: list[str] = []
    if not has_numeric_value(shares_outstanding) or shares_outstanding <= 0:
        warnings_list.append("Shares outstanding unavailable — P/E and P/S excluded from V-score.")
    if rev_row is None:
        warnings_list.append("Revenue data unavailable — growth and P/S excluded.")
    if ni_row is None:
        warnings_list.append("Net income data unavailable — profitability metrics excluded.")

    all_qf_dates_sorted = sorted(qf_dates)
    observations = []

    for q_date in common_dates:
        effective_date = q_date + pd.Timedelta(days=FILING_LAG_DAYS)

        # Price at or just before effective_date
        prior_prices = close.loc[:effective_date]
        if prior_prices.empty:
            continue
        price_at_date = float(prior_prices.iloc[-1])

        # Tech score at effective_date (last valid value on or before that date)
        prior_tech = tech_score_series.loc[:effective_date]
        if prior_tech.empty or pd.isna(prior_tech.iloc[-1]):
            continue
        tech_score_at = float(prior_tech.iloc[-1])

        # --- F-score ---
        dates_up_to_q = [d for d in all_qf_dates_sorted if d <= q_date]
        dates_up_to_q_minus4 = dates_up_to_q[:-4] if len(dates_up_to_q) >= 8 else []

        rev_ttm = _ttm_sum(rev_row, qf_norm, dates_up_to_q)
        ni_ttm = _ttm_sum(ni_row, qf_norm, dates_up_to_q)
        rev_ttm_prior = _ttm_sum(rev_row, qf_norm, dates_up_to_q_minus4) if dates_up_to_q_minus4 else None
        ni_ttm_prior = _ttm_sum(ni_row, qf_norm, dates_up_to_q_minus4) if dates_up_to_q_minus4 else None

        ca = _q_val(ca_row, q_date, qbs_norm)
        cl = _q_val(cl_row, q_date, qbs_norm)
        debt = _q_val(debt_row, q_date, qbs_norm)
        equity = _q_val(eq_row, q_date, qbs_norm)

        margins = safe_divide(ni_ttm, rev_ttm) if (ni_ttm is not None and rev_ttm) else None
        roe = safe_divide(ni_ttm, equity) if (ni_ttm is not None and equity) else None
        debt_eq = safe_divide(debt, equity) if (debt is not None and equity) else None
        current_ratio = safe_divide(ca, cl) if (ca is not None and cl) else None
        revenue_growth = (
            safe_divide(rev_ttm - rev_ttm_prior, abs(rev_ttm_prior))
            if (rev_ttm is not None and rev_ttm_prior) else None
        )
        earnings_growth = (
            safe_divide(ni_ttm - ni_ttm_prior, abs(ni_ttm_prior))
            if (ni_ttm is not None and ni_ttm_prior) else None
        )

        f_score = 0
        if has_numeric_value(roe):
            if roe >= active_settings["fund_roe_threshold"]:
                f_score += 1
            elif roe < 0 or roe < active_settings["fund_roe_threshold"] * 0.5:
                f_score -= 1
        if has_numeric_value(margins):
            if margins >= active_settings["fund_profit_margin_threshold"]:
                f_score += 1
            elif margins < 0 or margins < active_settings["fund_profit_margin_threshold"] * 0.5:
                f_score -= 1
        if has_numeric_value(debt_eq):
            if 0 <= debt_eq < active_settings["fund_debt_good_threshold"]:
                f_score += 1
            elif debt_eq > active_settings["fund_debt_bad_threshold"]:
                f_score -= 1
        if has_numeric_value(revenue_growth):
            if revenue_growth >= active_settings["fund_revenue_growth_threshold"]:
                f_score += 1
            elif revenue_growth < 0:
                f_score -= 1
        if has_numeric_value(earnings_growth):
            if earnings_growth >= active_settings["fund_revenue_growth_threshold"]:
                f_score += 1
            elif earnings_growth < 0:
                f_score -= 1
        if has_numeric_value(current_ratio):
            if current_ratio >= active_settings["fund_current_ratio_good"]:
                f_score += 1
            elif current_ratio < active_settings["fund_current_ratio_bad"]:
                f_score -= 1

        # --- V-score (P/E, P/S, P/B vs static sector benchmarks) ---
        v_score = 0
        if has_numeric_value(shares_outstanding) and shares_outstanding > 0:
            market_cap_hist = price_at_date * shares_outstanding
            pe_hist = safe_divide(market_cap_hist, ni_ttm) if has_numeric_value(ni_ttm) and ni_ttm > 0 else None
            ps_hist = safe_divide(market_cap_hist, rev_ttm) if has_numeric_value(rev_ttm) and rev_ttm > 0 else None
            bvps = safe_divide(equity, shares_outstanding) if has_numeric_value(equity) and equity > 0 else None
            pb_hist = safe_divide(price_at_date, bvps) if has_numeric_value(bvps) and bvps > 0 else None
            for metric_val, bench_val in [
                (pe_hist, bench["PE"]),
                (ps_hist, bench["PS"]),
                (pb_hist, bench["PB"]),
            ]:
                v_score += score_relative_multiple(metric_val, bench_val)

        # --- Composite score (matches analyst.py normalization) ---
        norm_tech = tech_score_at / const.TECH_SCORE_MAX * const.FUND_SCORE_MAX
        norm_val = v_score / const.VAL_SCORE_MAX * const.FUND_SCORE_MAX
        composite = (
            norm_tech * active_settings["weight_technical"]
            + f_score * active_settings["weight_fundamental"]
            + norm_val * active_settings["weight_valuation"]
        )

        # --- Verdict (raw threshold, no guardrails) ---
        if composite >= strong_buy_thresh:
            verdict = "STRONG BUY"
        elif composite >= buy_thresh:
            verdict = "BUY"
        elif composite <= strong_sell_thresh:
            verdict = "STRONG SELL"
        elif composite <= sell_thresh:
            verdict = "SELL"
        else:
            verdict = "HOLD"

        # --- Forward returns from effective_date ---
        fwd_1m = fwd_3m = fwd_12m = None
        future_prices = close.loc[effective_date:]
        if not future_prices.empty:
            if len(future_prices) >= 23:
                fwd_1m = float(future_prices.iloc[22] / price_at_date - 1)
            if len(future_prices) >= 64:
                fwd_3m = float(future_prices.iloc[63] / price_at_date - 1)
            if len(future_prices) >= 253:
                fwd_12m = float(future_prices.iloc[252] / price_at_date - 1)

        observations.append({
            "Date": effective_date.date(),
            "Verdict": verdict,
            "Tech Score": round(tech_score_at, 1),
            "F-Score": f_score,
            "V-Score": v_score,
            "Composite": round(composite, 2),
            "Fwd 1M": fwd_1m,
            "Fwd 3M": fwd_3m,
            "Fwd 12M": fwd_12m,
        })

    if len(observations) < 2:
        return None

    obs_df = pd.DataFrame(observations)

    # --- Verdict bucket table ---
    verdict_order = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]
    rows = []
    for v in verdict_order:
        subset = obs_df[obs_df["Verdict"] == v]
        if subset.empty:
            continue
        fwd1 = subset["Fwd 1M"].dropna()
        fwd3 = subset["Fwd 3M"].dropna()
        fwd12 = subset["Fwd 12M"].dropna()
        rows.append({
            "Verdict": v,
            "N": len(subset),
            "Avg 1M": fwd1.mean() if not fwd1.empty else None,
            "Avg 3M": fwd3.mean() if not fwd3.empty else None,
            "Avg 12M": fwd12.mean() if not fwd12.empty else None,
            "Hit% 3M": (fwd3 > 0).mean() if not fwd3.empty else None,
            "Hit% 12M": (fwd12 > 0).mean() if not fwd12.empty else None,
        })

    bucket_df = pd.DataFrame(rows) if rows else pd.DataFrame()

    warnings_list.insert(
        0,
        "Look-ahead bias: shares outstanding is the current figure; quarterly filings may contain "
        "restatements; sector benchmarks are static, not period-specific.",
    )

    return {
        "observations": obs_df,
        "bucket_table": bucket_df,
        "n_quarters": len(obs_df),
        "warnings": warnings_list,
    }
