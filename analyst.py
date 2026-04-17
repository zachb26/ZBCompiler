# -*- coding: utf-8 -*-
"""
analyst.py — StockAnalyst and PortfolioAnalyst classes.

StockAnalyst: fetches data, runs the full scoring pipeline, saves to DB.
PortfolioAnalyst: builds efficient-frontier, Monte Carlo, and rebalance output.
"""
import logging

import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

from constants import *
from utils_fmt import *
from utils_time import *
from utils_news import *
from exports import *
from skill_briefs import *
from cache import *
from fetch import *
from sec_ai import *
from dcf import *
from analytics_tech import *
from analytics_scoring import *
from analytics_decision import *
from settings import *
from analysis_prep import *
from backtest import *
from database import DatabaseManager


class StockAnalyst:
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.last_error = None

    def get_data(self, ticker):
        self.last_error = None
        hist, hist_error = fetch_ticker_history_with_retry(ticker, period="1y")
        if hist is None or hist.empty:
            self.last_error = hist_error or f"Unable to load price history for {ticker}."
            return None, {}, []

        info, info_error = fetch_ticker_info_with_retry(ticker)
        news, news_error = fetch_ticker_news_with_retry(ticker)

        if not info:
            saved = self.db.get_analysis(ticker)
            if not saved.empty:
                info = build_info_fallback_from_saved_analysis(saved.iloc[0])
                if info:
                    if info_error:
                        self.last_error = f"Live profile data was unavailable for {ticker}; reused saved fundamentals."
                elif info_error:
                    self.last_error = info_error
            elif info_error:
                self.last_error = info_error

        if not news and news_error and self.last_error is None:
            self.last_error = news_error

        return hist, info, news

    def analyze_sentiment(self, info, news, price, settings=None):
        info = info or {}
        news = news or []
        recommendation_key = (info.get("recommendationKey") or "").lower()
        analyst_opinions = safe_num(info.get("numberOfAnalystOpinions"))
        target_mean_price = safe_num(info.get("targetMeanPrice"))
        headlines = build_news_context_lines(news, max_items=6)
        context_parts = []
        if recommendation_key:
            context_parts.append(f"Analyst view: {recommendation_key.upper()}")
        if has_numeric_value(analyst_opinions):
            context_parts.append(f"Analyst count: {int(round(analyst_opinions))}")
        if has_numeric_value(target_mean_price):
            context_parts.append(f"Target mean: ${target_mean_price:,.2f}")
        context_parts.extend(headlines)
        summary = " | ".join(context_parts[:6]) if context_parts else "No recent news or analyst context was available."

        return {
            "score": 0,
            "verdict": "CONTEXT ONLY",
            "recommendation_key": recommendation_key.upper() if recommendation_key else "N/A",
            "analyst_opinions": analyst_opinions,
            "target_mean_price": target_mean_price,
            "headline_count": len(headlines),
            "summary": summary,
        }

    def build_record_from_market_data(self, ticker, hist, info, news, settings=None, compute_dcf=False, dcf_settings=None):
        settings = get_model_settings() if settings is None else settings
        dcf_settings = get_dcf_settings() if dcf_settings is None else normalize_dcf_settings(dcf_settings)
        ticker = normalize_ticker(ticker)
        info = info or {}
        news = news or []
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None

        close = hist["Close"].dropna().astype(float)
        if close.empty:
            return None
        price = float(close.iloc[-1])

        rsi_series = calculate_rsi(close)
        current_rsi = safe_num(rsi_series.iloc[-1])
        previous_rsi = safe_num(rsi_series.iloc[-2]) if len(rsi_series) > 1 else None

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_signal_line = macd_line.ewm(span=9, adjust=False).mean()
        current_macd = safe_num(macd_line.iloc[-1])
        current_macd_signal = safe_num(macd_signal_line.iloc[-1])

        sma50 = safe_num(close.rolling(50, min_periods=50).mean().iloc[-1])
        sma200 = safe_num(close.rolling(200, min_periods=200).mean().iloc[-1])
        momentum_1m = (price / close.iloc[-22] - 1) if len(close) > 22 else None
        momentum_1y = (price / close.iloc[0] - 1) if len(close) > 1 else None
        benchmark_hist, _ = fetch_ticker_history_with_retry(DEFAULT_BENCHMARK_TICKER, period="1y", attempts=2)
        benchmark_close = (
            benchmark_hist["Close"].dropna().astype(float)
            if benchmark_hist is not None and not benchmark_hist.empty and "Close" in benchmark_hist.columns
            else pd.Series(dtype=float)
        )
        relative_strength_metrics = {
            label: compute_relative_strength(close, benchmark_close, window)
            for label, window in BENCHMARK_RELATIVE_STRENGTH_WINDOWS.items()
        }
        volatility_1m = calculate_realized_volatility(close, 22)
        volatility_1y = calculate_realized_volatility(close, 252)
        momentum_1m_risk_adjusted = safe_divide(momentum_1m, (volatility_1m / np.sqrt(12)) if has_numeric_value(volatility_1m) else None)
        range_position_52w, distance_52w_high, distance_52w_low = calculate_52w_context(close)
        trend_strength = calculate_trend_strength(price, sma50, sma200, momentum_1y)
        latest_price_timestamp = close.index[-1] if len(close.index) else None
        latest_news_timestamp = max(
            [extract_news_publish_time(item) for item in news if extract_news_publish_time(item) is not None],
            default=None,
        )
        latest_data_timestamp = max(
            [stamp for stamp in [latest_price_timestamp, latest_news_timestamp] if stamp is not None],
            default=latest_price_timestamp,
        )
        trend_tolerance = settings["tech_trend_tolerance"]
        extension_limit = settings["tech_extension_limit"]

        tech_score = 0
        if has_numeric_value(sma200):
            tech_score += score_trend_distance(price, sma200, trend_tolerance)
        if has_numeric_value(sma50) and has_numeric_value(sma200):
            tech_score += score_trend_distance(sma50, sma200, trend_tolerance / 2)
        if has_numeric_value(sma50):
            tech_score += score_trend_distance(price, sma50, trend_tolerance / 2)
        if has_numeric_value(current_rsi):
            if current_rsi < settings["tech_rsi_oversold"]:
                if has_numeric_value(sma200) and price >= sma200 * 0.95:
                    tech_score += 2
                else:
                    tech_score += 1
            elif current_rsi > settings["tech_rsi_overbought"]:
                if has_numeric_value(sma50) and price <= sma50 * 1.02:
                    tech_score -= 2
                else:
                    tech_score -= 1
        macd_signal = "Neutral"
        if has_numeric_value(current_macd) and has_numeric_value(current_macd_signal):
            if current_macd > current_macd_signal:
                tech_score += 1
                macd_signal = "Bullish Crossover"
            else:
                tech_score -= 1
                macd_signal = "Bearish Crossover"
            if current_macd > 0:
                tech_score += 1
            elif current_macd < 0:
                tech_score -= 1
        if has_numeric_value(momentum_1m):
            if momentum_1m > settings["tech_momentum_threshold"]:
                tech_score += 1
            elif momentum_1m < -settings["tech_momentum_threshold"]:
                tech_score -= 1
        if has_numeric_value(momentum_1m_risk_adjusted):
            if momentum_1m_risk_adjusted >= 0.75:
                tech_score += 1
            elif momentum_1m_risk_adjusted <= -0.75:
                tech_score -= 1
        long_term_momentum_threshold = max(settings["tech_momentum_threshold"] * 3, 0.10)
        if has_numeric_value(momentum_1y):
            if momentum_1y > long_term_momentum_threshold:
                tech_score += 1
            elif momentum_1y < -long_term_momentum_threshold:
                tech_score -= 1
        relative_strength_6m = relative_strength_metrics.get("Relative_Strength_6M")
        if has_numeric_value(relative_strength_6m):
            if relative_strength_6m >= 0.05:
                tech_score += 1
            elif relative_strength_6m <= -0.05:
                tech_score -= 1
        bullish_trend = has_bullish_trend(price, sma50, sma200, momentum_1y)
        bearish_trend = has_bearish_trend(price, sma50, sma200, momentum_1y)
        regime = classify_market_regime(price, sma50, sma200, momentum_1y, tolerance=trend_tolerance)
        overextended = has_numeric_value(sma50) and price >= sma50 * (1 + extension_limit)
        washed_out = has_numeric_value(sma50) and price <= sma50 * (1 - extension_limit)
        pullback_recovery = (
            bullish_trend
            and has_numeric_value(previous_rsi)
            and has_numeric_value(current_rsi)
            and previous_rsi <= settings["tech_rsi_oversold"]
            and current_rsi > settings["tech_rsi_oversold"]
            and has_numeric_value(current_macd)
            and has_numeric_value(current_macd_signal)
            and current_macd >= current_macd_signal
        )
        if pullback_recovery:
            tech_score += 1
        if has_numeric_value(trend_strength):
            if trend_strength >= 30:
                tech_score += 1
            elif trend_strength <= -30:
                tech_score -= 1
        if has_numeric_value(range_position_52w):
            if range_position_52w >= 0.80 and bullish_trend:
                tech_score += 1
            elif range_position_52w <= 0.20 and bearish_trend:
                tech_score -= 1
        if overextended and bullish_trend and has_numeric_value(current_rsi) and current_rsi >= settings["tech_rsi_overbought"] - 5:
            tech_score -= 1
        if washed_out and bearish_trend and has_numeric_value(current_rsi) and current_rsi <= settings["tech_rsi_oversold"] + 5:
            tech_score += 1
        v_tech = score_to_signal(tech_score)

        f_score = 0
        roe = safe_num(info.get("returnOnEquity"))
        margins = safe_num(info.get("profitMargins"))
        debt_eq = safe_num(info.get("debtToEquity"))
        revenue_growth = safe_num(info.get("revenueGrowth"))
        earnings_growth = safe_num(info.get("earningsGrowth"))
        current_ratio = safe_num(info.get("currentRatio"))
        market_cap = safe_num(info.get("marketCap"))
        dividend_yield = safe_num(info.get("dividendYield"))
        payout_ratio = safe_num(info.get("payoutRatio"))
        equity_beta = safe_num(info.get("beta"))
        if has_numeric_value(roe):
            if roe >= settings["fund_roe_threshold"]:
                f_score += 1
            elif roe < 0 or roe < settings["fund_roe_threshold"] * 0.5:
                f_score -= 1
        if has_numeric_value(margins):
            if margins >= settings["fund_profit_margin_threshold"]:
                f_score += 1
            elif margins < 0 or margins < settings["fund_profit_margin_threshold"] * 0.5:
                f_score -= 1
        if has_numeric_value(debt_eq):
            if 0 <= debt_eq < settings["fund_debt_good_threshold"]:
                f_score += 1
            elif debt_eq > settings["fund_debt_bad_threshold"]:
                f_score -= 1
        if has_numeric_value(revenue_growth):
            if revenue_growth >= settings["fund_revenue_growth_threshold"]:
                f_score += 1
            elif revenue_growth < 0:
                f_score -= 1
        if has_numeric_value(earnings_growth):
            if earnings_growth >= settings["fund_revenue_growth_threshold"]:
                f_score += 1
            elif earnings_growth < 0:
                f_score -= 1
        if has_numeric_value(current_ratio):
            if current_ratio >= settings["fund_current_ratio_good"]:
                f_score += 1
            elif current_ratio < settings["fund_current_ratio_bad"]:
                f_score -= 1
        quality_score = calculate_quality_score(
            roe,
            margins,
            debt_eq,
            revenue_growth,
            earnings_growth,
            current_ratio,
            settings,
        )
        if quality_score >= 3:
            f_score += 1
        elif quality_score <= -1.5:
            f_score -= 1
        if f_score >= 4:
            v_fund = "STRONG"
        elif f_score >= 1:
            v_fund = "STABLE"
        else:
            v_fund = "WEAK"

        v_score = 0
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")
        bench, peer_group = build_relative_peer_benchmarks(ticker, info, db=self.db, settings=settings)
        pe = safe_num(info.get("trailingPE"))
        forward_pe = safe_num(info.get("forwardPE"))
        peg_ratio = safe_num(info.get("pegRatio"))
        ps_ratio = safe_num(info.get("priceToSalesTrailing12Months"))
        ev_ebitda = safe_num(info.get("enterpriseToEbitda"))
        pb = safe_num(info.get("priceToBook"))
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
            if peg_ratio <= 0:
                v_score -= 1
            elif peg_ratio <= settings["valuation_peg_threshold"] * 0.9:
                v_score += 1
            elif peg_ratio >= settings["valuation_peg_threshold"] * 1.35:
                v_score -= 1
        eps = safe_num(info.get("trailingEps"))
        bvps = safe_num(info.get("bookValue"))
        graham_num = None
        intrinsic_value = None
        graham_adj = 0
        if has_numeric_value(eps) and has_numeric_value(bvps) and eps > 0 and bvps > 0:
            graham_num = (22.5 * eps * bvps) ** 0.5
            intrinsic_value = graham_num
            valuation_signal_count += 1
            if price < graham_num * 0.85:
                graham_adj = 2
            elif price < graham_num:
                graham_adj = 1
            elif price > graham_num * settings["valuation_graham_overpriced_multiple"]:
                graham_adj = -2
            elif price > graham_num * 1.15:
                graham_adj = -1
        elif has_numeric_value(eps) and eps <= 0:
            v_score -= 1

        dcf_result = {}
        dcf_intrinsic_value = None
        dcf_upside = None
        dcf_last_updated = None
        if compute_dcf:
            dcf_result = build_sec_dcf_model(
                ticker,
                price,
                info,
                dcf_settings=dcf_settings,
                peer_benchmarks=bench,
            )
            if dcf_result.get("available") and has_numeric_value(dcf_result.get("intrinsic_value_per_share")):
                dcf_intrinsic_value = safe_num(dcf_result.get("intrinsic_value_per_share"))
                dcf_upside = safe_num(dcf_result.get("upside"))
                dcf_last_updated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

        if valuation_signal_count < 2 and v_score < settings["valuation_fair_score_threshold"]:
            v_val = "FAIR VALUE"
        elif v_score >= settings["valuation_under_score_threshold"]:
            v_val = "UNDERVALUED"
        elif v_score >= settings["valuation_fair_score_threshold"]:
            v_val = "FAIR VALUE"
        else:
            v_val = "OVERVALUED"
        valuation_confidence = calculate_valuation_confidence(valuation_signal_count)

        dividend_safety_score = calculate_dividend_safety_score(
            dividend_yield,
            payout_ratio,
            margins,
            current_ratio,
            debt_eq,
        )

        sentiment = self.analyze_sentiment(info, news, price, settings=settings)
        sentiment_conviction = calculate_sentiment_conviction(
            sentiment["score"],
            sentiment["analyst_opinions"],
            sentiment["recommendation_key"],
            sentiment["target_mean_price"],
            price,
            sentiment["headline_count"],
        )
        event_study = compute_event_study(news, hist, benchmark_ticker=DEFAULT_BENCHMARK_TICKER)
        stock_profile = classify_stock_profile(
            sector=sector,
            price=price,
            market_cap=market_cap,
            dividend_yield=dividend_yield,
            payout_ratio=payout_ratio,
            equity_beta=equity_beta,
            analyst_opinions=sentiment["analyst_opinions"],
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
        _GRAHAM_FULL = {"Value Stocks", "Dividend / Income Stocks", "Blue-Chip Stocks"}
        _GRAHAM_LIGHT = {"Growth Stocks", "Speculative / Penny Stocks"}
        if graham_adj != 0:
            _pt = stock_profile["primary_type"]
            if _pt in _GRAHAM_FULL:
                v_score += graham_adj
            elif _pt in _GRAHAM_LIGHT:
                v_score += graham_adj * 0.25
            else:
                v_score += graham_adj * 0.5
        effective_v_score = v_score * (0.45 + 0.55 * valuation_confidence / 100)
        risk_flags = build_risk_flags(
            eps=eps,
            debt_eq=debt_eq,
            current_ratio=current_ratio,
            overextended=overextended,
            distance_52w_high=distance_52w_high,
            range_position=range_position_52w,
            volatility_1y=volatility_1y,
            stock_profile=stock_profile,
        )
        engine_weights, engine_weight_profile = get_type_adjusted_engine_weights(stock_profile, settings)
        effective_sentiment_score = 0.0
        effective_tech_score = tech_score
        if has_numeric_value(trend_strength):
            effective_tech_score += np.clip(trend_strength / 50, -1.0, 1.0)
        relative_strength_1y = relative_strength_metrics.get("Relative_Strength_1Y")
        if has_numeric_value(relative_strength_1y):
            effective_tech_score += np.clip(relative_strength_1y * 4, -1.0, 1.0)
        effective_f_score = f_score
        if quality_score >= 3:
            effective_f_score += 0.5
        elif quality_score <= -1.5:
            effective_f_score -= 0.5
        if stock_profile["primary_type"] in {"Dividend / Income Stocks", "Defensive Stocks"}:
            effective_f_score += np.clip(dividend_safety_score / 4, -0.5, 1.0)
        if has_numeric_value(event_study.get("avg_abnormal_5d")):
            effective_f_score += np.clip(event_study["avg_abnormal_5d"] * 10, -0.75, 0.75)
        # Normalise each engine score to the FUND_SCORE_MAX reference scale so
        # that a weight of 1.0 means the same maximum contribution per engine.
        norm_tech = effective_tech_score / TECH_SCORE_MAX * FUND_SCORE_MAX
        norm_val = effective_v_score / VAL_SCORE_MAX * FUND_SCORE_MAX
        base_overall_score = (
            norm_tech * engine_weights["technical"]
            + effective_f_score * engine_weights["fundamental"]
            + norm_val * engine_weights["valuation"]
            + effective_sentiment_score * engine_weights["sentiment"]
        )
        base_overall_score -= min(len(risk_flags), 4) * 0.35
        overall_score, base_verdict, type_settings, type_logic_notes = apply_stock_type_framework(
            stock_profile=stock_profile,
            overall_score=base_overall_score,
            tech_score=tech_score,
            f_score=f_score,
            v_score=v_score,
            sentiment_score=0,
            v_fund=v_fund,
            v_val=v_val,
            regime=regime,
            bullish_trend=bullish_trend,
            bearish_trend=bearish_trend,
            data_quality="High",
            momentum_1y=momentum_1y,
            settings=settings,
        )

        assumption_profile = detect_matching_preset(settings)
        assumption_fingerprint = get_assumption_fingerprint(settings)
        assumption_snapshot = serialize_model_settings(settings)
        record = {
            "Ticker": ticker,
            "Price": price,
            "Verdict_Overall": base_verdict,
            "Verdict_Technical": v_tech,
            "Verdict_Fundamental": v_fund,
            "Verdict_Valuation": v_val,
            "Verdict_Sentiment": sentiment["verdict"],
            "Market_Regime": regime,
            "Score_Tech": tech_score,
            "Score_Fund": f_score,
            "Score_Val": v_score,
            "Score_Sentiment": sentiment["score"],
            "Sector": sector,
            "Industry": industry,
            "Stock_Type": stock_profile["primary_type"],
            "Cap_Bucket": stock_profile["cap_bucket"],
            "Style_Tags": stock_profile["style_tags"],
            "Type_Strategy": stock_profile["type_strategy"],
            "Type_Confidence": stock_profile["type_confidence"],
            "Engine_Weight_Profile": engine_weight_profile,
            "Peer_Count": peer_group.get("count"),
            "Peer_Group_Label": peer_group.get("group_label"),
            "Peer_Tickers": ", ".join(peer_group.get("tickers", [])),
            "Peer_Summary": peer_group.get("summary"),
            "Peer_Comparison": json.dumps(peer_group),
            "Market_Cap": market_cap,
            "Dividend_Yield": dividend_yield,
            "Payout_Ratio": payout_ratio,
            "Equity_Beta": equity_beta,
            "Relative_Strength_3M": relative_strength_metrics.get("Relative_Strength_3M"),
            "Relative_Strength_6M": relative_strength_metrics.get("Relative_Strength_6M"),
            "Relative_Strength_1Y": relative_strength_metrics.get("Relative_Strength_1Y"),
            "Trend_Strength": trend_strength,
            "Range_Position_52W": range_position_52w,
            "Distance_52W_High": distance_52w_high,
            "Distance_52W_Low": distance_52w_low,
            "Volatility_1M": volatility_1m,
            "Volatility_1Y": volatility_1y,
            "Momentum_1M_Risk_Adjusted": momentum_1m_risk_adjusted,
            "Quality_Score": quality_score,
            "Dividend_Safety_Score": dividend_safety_score,
            "Valuation_Signal_Count": valuation_signal_count,
            "Valuation_Confidence": valuation_confidence,
            "Sentiment_Conviction": sentiment_conviction,
            "Risk_Flags": " | ".join(risk_flags),
            "PE_Ratio": pe,
            "Forward_PE": forward_pe,
            "PEG_Ratio": peg_ratio,
            "PS_Ratio": ps_ratio,
            "PB_Ratio": pb,
            "EV_EBITDA": ev_ebitda,
            "Graham_Number": graham_num if graham_num else 0,
            "Intrinsic_Value": intrinsic_value if intrinsic_value else 0,
            "DCF_Intrinsic_Value": dcf_intrinsic_value if has_numeric_value(dcf_intrinsic_value) else None,
            "DCF_Upside": dcf_upside,
            "DCF_WACC": dcf_result.get("wacc"),
            "DCF_Risk_Free_Rate": dcf_result.get("risk_free_rate"),
            "DCF_Beta": dcf_result.get("beta"),
            "DCF_Cost_of_Equity": dcf_result.get("cost_of_equity"),
            "DCF_Cost_of_Debt": dcf_result.get("after_tax_cost_of_debt"),
            "DCF_Equity_Weight": dcf_result.get("equity_weight"),
            "DCF_Debt_Weight": dcf_result.get("debt_weight"),
            "DCF_Growth_Rate": dcf_result.get("selected_growth_rate"),
            "DCF_Terminal_Growth": dcf_result.get("terminal_growth_rate"),
            "DCF_Base_FCF": dcf_result.get("base_fcf"),
            "DCF_Enterprise_Value": dcf_result.get("enterprise_value"),
            "DCF_Equity_Value": dcf_result.get("equity_value"),
            "DCF_Historical_FCF_Growth": dcf_result.get("historical_fcf_growth"),
            "DCF_Historical_Revenue_Growth": dcf_result.get("historical_revenue_growth"),
            "DCF_Guidance_Growth": dcf_result.get("guidance_growth_rate"),
            "DCF_Source": dcf_result.get("growth_source", "Unavailable"),
            "DCF_Confidence": dcf_result.get("growth_confidence", "low"),
            "DCF_History": json.dumps(dcf_result.get("history", [])),
            "DCF_Projection": json.dumps(dcf_result.get("projection", [])),
            "DCF_Sensitivity": json.dumps(dcf_result.get("sensitivity", [])),
            "DCF_Guidance_Excerpts": json.dumps(dcf_result.get("guidance_excerpts", [])),
            "DCF_Guidance_Summary": dcf_result.get("guidance_summary") or dcf_result.get("error", ""),
            "DCF_Filing_Form": dcf_result.get("filing_form"),
            "DCF_Filing_Date": dcf_result.get("filing_date"),
            "DCF_Last_Updated": dcf_last_updated,
            "DCF_Assumptions": serialize_dcf_settings(dcf_settings),
            "Profit_Margins": margins,
            "ROE": roe,
            "Debt_to_Equity": debt_eq,
            "Revenue_Growth": revenue_growth,
            "Current_Ratio": current_ratio,
            "Target_Mean_Price": sentiment["target_mean_price"],
            "Recommendation_Key": sentiment["recommendation_key"],
            "Analyst_Opinions": sentiment["analyst_opinions"],
            "Sentiment_Headline_Count": sentiment["headline_count"],
            "Sentiment_Summary": sentiment["summary"],
            "Event_Study_Count": event_study.get("count"),
            "Event_Study_Avg_Abnormal_1D": event_study.get("avg_abnormal_1d"),
            "Event_Study_Avg_Abnormal_5D": event_study.get("avg_abnormal_5d"),
            "Event_Study_Summary": event_study.get("summary"),
            "Event_Study_Events": json.dumps(event_study.get("events", [])),
            "RSI": current_rsi,
            "MACD_Value": current_macd,
            "MACD_Signal": macd_signal,
            "SMA_Status": (
                "Bullish" if regime == "Bullish Trend"
                else "Bearish" if regime == "Bearish Trend"
                else "Neutral"
            ),
            "Momentum_1M": momentum_1m,
            "Momentum_1Y": momentum_1y,
            "Last_Data_Update": format_datetime_value(latest_data_timestamp),
            "Last_Updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Overall_Score": overall_score,
            "Assumption_Profile": assumption_profile,
            "Assumption_Fingerprint": assumption_fingerprint,
            "Assumption_Drift": calculate_assumption_drift(settings),
            "Assumption_Snapshot": assumption_snapshot,
        }
        completeness, missing_count, quality_label = assess_record_quality(record)
        bias_summary = summarize_engine_biases(
            tech_score,
            f_score,
            v_score,
            0,
            v_val,
            bullish_trend,
            bearish_trend,
        )
        decision_confidence = compute_decision_confidence(
            overall_score=overall_score,
            bias_summary=bias_summary,
            regime=regime,
            completeness=completeness,
        )
        decision_confidence = adjust_type_based_confidence(decision_confidence, stock_profile, quality_label)
        if has_numeric_value(trend_strength):
            decision_confidence += np.clip(trend_strength / 12, -4, 6)
        if has_numeric_value(quality_score):
            decision_confidence += np.clip(quality_score * 2.5, -5, 8)
        if has_numeric_value(valuation_confidence):
            decision_confidence += np.clip((valuation_confidence - 50) / 8, -4, 5)
        if has_numeric_value(sentiment_conviction):
            decision_confidence += np.clip((sentiment_conviction - 50) / 10, -3, 4)
        decision_confidence -= min(len(risk_flags), 5) * 2.0
        decision_confidence = float(np.clip(round(decision_confidence, 1), 5.0, 95.0))
        final_verdict = apply_confidence_guard(base_verdict, decision_confidence, quality_label, type_settings)
        record["Verdict_Overall"] = final_verdict
        record["Decision_Confidence"] = decision_confidence
        base_decision_notes = build_decision_notes(
            verdict=final_verdict,
            regime=regime,
            bias_summary=bias_summary,
            confidence=decision_confidence,
            data_quality=quality_label,
            current_rsi=current_rsi,
            v_val=v_val,
            v_fund=v_fund,
            bullish_trend=bullish_trend,
            bearish_trend=bearish_trend,
            overextended=overextended,
            pullback_recovery=pullback_recovery,
        )
        decision_note_parts = [
            f"Type: {stock_profile['primary_type']}",
            f"Cap: {stock_profile['cap_bucket']}",
        ]
        if stock_profile.get("classification_summary"):
            decision_note_parts.append(stock_profile["classification_summary"])
        decision_note_parts.extend(type_logic_notes[:2])
        if risk_flags:
            decision_note_parts.append("Risks: " + ", ".join(risk_flags[:3]))
        if base_decision_notes:
            decision_note_parts.extend(base_decision_notes.split(" | "))
        deduped_notes = []
        for note in decision_note_parts:
            cleaned_note = str(note).strip()
            if cleaned_note and cleaned_note not in deduped_notes:
                deduped_notes.append(cleaned_note)
        record["Decision_Notes"] = " | ".join(deduped_notes[:5])
        record["Data_Completeness"] = completeness
        record["Missing_Metric_Count"] = missing_count
        record["Data_Quality"] = quality_label
        return record

    def analyze(self, ticker, settings=None, persist=True, preloaded=None, compute_dcf=False, dcf_settings=None):
        active_settings = get_model_settings() if settings is None else settings
        ticker = normalize_ticker(ticker)
        self.last_error = None
        existing_row = None
        if persist:
            existing = self.db.get_analysis(ticker)
            if not existing.empty:
                existing_row = existing.iloc[0].to_dict()
        if preloaded is None:
            hist, info, news = self.get_data(ticker)
        else:
            hist, info, news = preloaded

        record = self.build_record_from_market_data(
            ticker,
            hist,
            info,
            news,
            settings=active_settings,
            compute_dcf=compute_dcf,
            dcf_settings=dcf_settings,
        )
        if record is None and self.last_error is None:
            self.last_error = (
                f"Unable to build an analysis for {ticker}. Yahoo returned incomplete or unusable market data."
            )
        if record is None and persist:
            if existing_row is not None:
                self.last_error = (
                    f"Live fetch failed for {ticker}; showing the most recent saved analysis instead."
                )
                return existing_row
        if record and persist:
            if existing_row and (not compute_dcf or not has_dcf_snapshot(record)):
                record.update(extract_dcf_fields(existing_row))
            self.db.save_analysis(record)
        return record


class PortfolioAnalyst:
    def __init__(self, db):
        self.db = db
        self.last_error = None

    def get_price_history(self, tickers, benchmark_ticker, period):
        download_list = list(dict.fromkeys(tickers + [benchmark_ticker]))
        raw, download_error = fetch_batch_history_with_retry(download_list, period=period)
        if raw is None or raw.empty:
            self.last_error = download_error or "Unable to download portfolio price history."
            return None, None

        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" not in raw.columns.get_level_values(0):
                self.last_error = "Downloaded portfolio data did not include close prices."
                return None, None
            close_prices = raw["Close"].copy()
        else:
            if isinstance(raw, pd.Series):
                close_prices = raw.to_frame(name=download_list[0])
            elif "Close" in raw.columns:
                close_prices = raw[["Close"]].copy()
                close_prices.columns = [download_list[0]]
            elif set(download_list).issubset(set(raw.columns)):
                close_prices = raw.copy()
            else:
                self.last_error = "Downloaded portfolio data did not include a Close column."
                return None, None

        if isinstance(close_prices, pd.Series):
            close_prices = close_prices.to_frame(name=download_list[0])
        close_prices = close_prices.sort_index().ffill(limit=3)

        available_assets = [
            ticker for ticker in tickers
            if ticker in close_prices.columns and close_prices[ticker].dropna().shape[0] >= 30
        ]
        missing_assets = [ticker for ticker in tickers if ticker not in available_assets]
        if benchmark_ticker not in close_prices.columns or close_prices[benchmark_ticker].dropna().shape[0] < 30:
            self.last_error = f"The benchmark {benchmark_ticker} does not have enough usable history for {period}."
            return None, None
        if len(available_assets) < 2:
            missing_text = ", ".join(missing_assets) if missing_assets else "the selected tickers"
            self.last_error = (
                f"Need at least two tickers with usable {period} history. Missing or too short: {missing_text}."
            )
            return None, None

        combined_columns = list(dict.fromkeys(available_assets + [benchmark_ticker]))
        aligned_prices = close_prices[combined_columns].dropna()
        if aligned_prices.empty or len(aligned_prices) < 30:
            self.last_error = (
                "The selected names do not share enough overlapping history for this lookback window. "
                "Try a shorter period or remove newer tickers."
            )
            return None, None

        return aligned_prices[available_assets], aligned_prices[benchmark_ticker]

    def get_asset_metadata(self, tickers):
        rows = []
        for ticker in tickers:
            name = ticker
            sector = "Unknown"
            cached = self.db.get_analysis(ticker)
            if not cached.empty:
                cached_row = cached.iloc[0]
                if not pd.isna(cached_row.get("Sector")):
                    sector = cached_row.get("Sector") or sector

            info, _ = fetch_ticker_info_with_retry(ticker)
            if info:
                name = info.get("shortName") or info.get("longName") or ticker
                sector = info.get("sector") or sector

            rows.append({"Ticker": ticker, "Name": name, "Sector": sector})

        return pd.DataFrame(rows)

    def calculate_asset_metrics(self, asset_returns, benchmark_returns, risk_free_rate, trading_days):
        risk_free_daily = risk_free_rate / trading_days
        annual_return = asset_returns.mean() * trading_days
        annual_volatility = asset_returns.std() * np.sqrt(trading_days)
        downside_diff = (asset_returns - risk_free_daily).clip(upper=0)
        downside_volatility = np.sqrt((downside_diff.pow(2)).mean()) * np.sqrt(trading_days)

        benchmark_var = benchmark_returns.var()
        betas = asset_returns.apply(lambda series: safe_divide(series.cov(benchmark_returns), benchmark_var))

        metrics = pd.DataFrame(
            {
                "Annual Return": annual_return,
                "Volatility": annual_volatility,
                "Downside Volatility": downside_volatility,
                "Beta": betas,
            }
        )
        metrics["Sharpe Ratio"] = (metrics["Annual Return"] - risk_free_rate) / metrics["Volatility"]
        metrics["Sortino Ratio"] = (metrics["Annual Return"] - risk_free_rate) / metrics["Downside Volatility"]
        metrics["Treynor Ratio"] = (metrics["Annual Return"] - risk_free_rate) / metrics["Beta"]
        metrics = metrics.replace([np.inf, -np.inf], np.nan)
        metrics.index.name = "Ticker"
        return metrics.reset_index()

    def calculate_portfolio_metrics(self, asset_returns, benchmark_returns, weights, risk_free_rate, trading_days):
        risk_free_daily = risk_free_rate / trading_days
        portfolio_returns = asset_returns @ weights
        annual_return = portfolio_returns.mean() * trading_days
        volatility = portfolio_returns.std() * np.sqrt(trading_days)
        downside_diff = (portfolio_returns - risk_free_daily).clip(upper=0)
        downside_volatility = np.sqrt((downside_diff.pow(2)).mean()) * np.sqrt(trading_days)
        beta = safe_divide(portfolio_returns.cov(benchmark_returns), benchmark_returns.var())
        sharpe = safe_divide(annual_return - risk_free_rate, volatility)
        sortino = safe_divide(annual_return - risk_free_rate, downside_volatility)
        treynor = safe_divide(annual_return - risk_free_rate, beta)

        return {
            "Return": annual_return,
            "Volatility": volatility,
            "Downside Volatility": downside_volatility,
            "Beta": beta,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Treynor": treynor,
        }

    def simulate_portfolios(self, asset_returns, benchmark_returns, risk_free_rate, max_weight, simulations, trading_days):
        rng = np.random.default_rng(42)
        tickers = list(asset_returns.columns)
        portfolios = []

        for _ in range(simulations):
            weights = cap_weights(rng.random(len(tickers)), max_weight)
            metrics = self.calculate_portfolio_metrics(
                asset_returns,
                benchmark_returns,
                weights,
                risk_free_rate,
                trading_days,
            )
            row = {**metrics}
            for ticker, weight in zip(tickers, weights):
                row[f"W_{ticker}"] = weight
            portfolios.append(row)

        portfolio_df = pd.DataFrame(portfolios).replace([np.inf, -np.inf], np.nan).dropna(subset=["Return", "Volatility", "Sharpe"])
        if portfolio_df.empty:
            return None, None, None, None, None

        sorted_df = portfolio_df.sort_values("Volatility").reset_index(drop=True)
        sorted_df["_cummax"] = sorted_df["Return"].cummax()
        frontier = sorted_df[sorted_df["Return"] == sorted_df["_cummax"]].drop(columns="_cummax")
        tangent = portfolio_df.loc[portfolio_df["Sharpe"].idxmax()]
        minimum_volatility = portfolio_df.loc[portfolio_df["Volatility"].idxmin()]

        cal_x = np.linspace(0, max(frontier["Volatility"].max(), tangent["Volatility"]) * 1.2, 60)
        cal_y = risk_free_rate + tangent["Sharpe"] * cal_x
        cal = pd.DataFrame({"Volatility": cal_x, "Return": cal_y})
        return portfolio_df, frontier, tangent, minimum_volatility, cal

    def build_recommendations(self, tickers, asset_metrics, metadata, tangent_portfolio):
        weight_map = {
            ticker: tangent_portfolio.get(f"W_{ticker}", 0.0)
            for ticker in tickers
        }
        equal_weight = 1 / len(tickers)
        recommendations = asset_metrics.merge(metadata, on="Ticker", how="left")
        recommendations["Recommended Weight"] = recommendations["Ticker"].map(weight_map).fillna(0.0)
        recommendations["Weight vs Equal"] = recommendations["Recommended Weight"] - equal_weight

        def classify_role(row):
            if row["Recommended Weight"] >= max(equal_weight * 1.35, 0.18):
                return "Core holding"
            if row["Recommended Weight"] >= equal_weight * 0.9:
                return "Supporting allocation"
            if row["Beta"] is not None and not pd.isna(row["Beta"]) and row["Beta"] < 0.9:
                return "Diversifier"
            return "Satellite position"

        def build_reason(row):
            reasons = []
            if not pd.isna(row["Sharpe Ratio"]) and row["Sharpe Ratio"] >= 1:
                reasons.append("strong Sharpe")
            if not pd.isna(row["Sortino Ratio"]) and row["Sortino Ratio"] >= 1:
                reasons.append("good downside efficiency")
            if not pd.isna(row["Treynor Ratio"]) and row["Treynor Ratio"] > 0:
                reasons.append("positive Treynor")
            if not pd.isna(row["Beta"]) and row["Beta"] < 0.9:
                reasons.append("helps diversify beta")
            return ", ".join(reasons) if reasons else "kept for balance and exposure"

        recommendations["Role"] = recommendations.apply(classify_role, axis=1)
        recommendations["Rationale"] = recommendations.apply(build_reason, axis=1)
        recommendations = recommendations.sort_values("Recommended Weight", ascending=False)
        return recommendations

    def build_portfolio_notes(self, recommendations, sector_exposure, tangent_portfolio, max_weight):
        notes = []
        effective_names = safe_divide(1, np.square(recommendations["Recommended Weight"]).sum())
        largest_position = recommendations.iloc[0]
        largest_sector = sector_exposure.iloc[0]

        if len(recommendations) < 5:
            notes.append("Fewer than five holdings means this portfolio is still fairly concentrated.")
        if largest_position["Recommended Weight"] >= max_weight * 0.95:
            notes.append(f"{largest_position['Ticker']} is pressing against the max position size, which signals strong conviction but higher single-name risk.")
        if largest_sector["Recommended Weight"] > 0.45:
            notes.append(f"{largest_sector['Sector']} is more than 45% of the allocation, so sector risk is elevated.")
        if effective_names is not None and effective_names < 4:
            notes.append("Effective diversification is low; the weights behave like fewer than four equally sized names.")
        if tangent_portfolio["Beta"] is not None and tangent_portfolio["Beta"] > 1.1:
            notes.append("The recommended portfolio is more aggressive than the benchmark on a beta basis.")
        if not notes:
            notes.append("The allocation is reasonably balanced across names and does not show a major concentration warning.")

        return notes, effective_names

    def analyze_portfolio(self, tickers, benchmark_ticker, period, risk_free_rate, max_weight, simulations):
        self.last_error = None
        settings = get_model_settings()
        trading_days = settings["trading_days_per_year"]
        if len(tickers) * max_weight < 1:
            self.last_error = "The max single-stock weight is too low for the number of requested names."
            return None

        try:
            asset_prices, benchmark_prices = self.get_price_history(tickers, benchmark_ticker, period)
            if asset_prices is None or benchmark_prices is None:
                return None

            asset_returns = asset_prices.pct_change().dropna()
            benchmark_returns = benchmark_prices.pct_change().dropna()
            common_index = asset_returns.index.intersection(benchmark_returns.index)
            asset_returns = asset_returns.loc[common_index]
            benchmark_returns = benchmark_returns.loc[common_index]
            if asset_returns.empty or len(asset_returns.columns) < 2:
                self.last_error = (
                    "The selected basket did not produce enough overlapping return history to build a portfolio recommendation."
                )
                return None

            asset_metrics = self.calculate_asset_metrics(asset_returns, benchmark_returns, risk_free_rate, trading_days)
            portfolio_df, frontier, tangent, minimum_volatility, cal = self.simulate_portfolios(
                asset_returns,
                benchmark_returns,
                risk_free_rate,
                max_weight,
                simulations,
                trading_days,
            )
            if portfolio_df is None:
                self.last_error = (
                    "Portfolio simulation could not find enough valid portfolios. Try fewer tickers, a shorter window, or a higher max weight."
                )
                return None

            valid_tickers = list(asset_returns.columns)
            metadata = self.get_asset_metadata(valid_tickers)
            recommendations = self.build_recommendations(valid_tickers, asset_metrics, metadata, tangent)
            sector_exposure = (
                recommendations.groupby("Sector", dropna=False)["Recommended Weight"]
                .sum()
                .reset_index()
                .sort_values("Recommended Weight", ascending=False)
            )
            notes, effective_names = self.build_portfolio_notes(recommendations, sector_exposure, tangent, max_weight)

            return {
                "asset_prices": asset_prices,
                "benchmark_prices": benchmark_prices,
                "asset_metrics": asset_metrics,
                "portfolio_cloud": portfolio_df,
                "frontier": frontier,
                "tangent": tangent,
                "minimum_volatility": minimum_volatility,
                "cal": cal,
                "recommendations": recommendations,
                "sector_exposure": sector_exposure,
                "notes": notes,
                "effective_names": effective_names,
                "benchmark": benchmark_ticker,
                "period": period,
            }
        except Exception as exc:
            logger.error("PortfolioAnalyst.analyze failed: %s", exc)
            self.last_error = f"Portfolio analysis hit an upstream or data-shape error: {summarize_fetch_error(exc)}"
            return None


