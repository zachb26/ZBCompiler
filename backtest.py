# -*- coding: utf-8 -*-
"""
backtest.py — Position-generation, trade summarisation, and technical backtest
engine.

Functions:
    derive_backtest_positions
    summarize_backtest_trades
    compute_technical_backtest
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fetch import calculate_rsi, has_numeric_value, safe_divide
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

    analysis["Tech Score"] = tech_score
    analysis["Signal"] = analysis["Tech Score"].apply(score_to_signal)
    analysis["Position"] = derive_backtest_positions(analysis, active_settings, stock_profile=stock_profile)
    analysis["Benchmark Return"] = analysis["Close"].pct_change().fillna(0.0)
    trade_points = analysis["Position"].diff().fillna(analysis["Position"])
    trading_cost_rate = active_settings.get("backtest_transaction_cost_bps", 0.0) / 10000
    analysis["Trading Cost"] = trade_points.abs() * trading_cost_rate
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
        "Position Changes": len(trade_log),
        "Closed Trades": trade_summary["Closed Trades"],
        "Win Rate": trade_summary["Win Rate"],
        "Average Trade Return": trade_summary["Average Trade Return"],
    }

    return {
        "history": analysis.reset_index().rename(columns={"index": "Date"}),
        "trade_log": trade_log.reset_index(drop=True),
        "closed_trades": closed_trades_df,
        "metrics": metrics,
        "stock_profile": stock_profile or {},
    }
