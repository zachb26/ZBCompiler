# -*- coding: utf-8 -*-
import datetime
import sqlite3

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

DB_FILENAME = "stocks_data.db"
TRADING_DAYS = 252
DEFAULT_BENCHMARK_TICKER = "SPY"
DEFAULT_PORTFOLIO_TICKERS = "AAPL, MSFT, NVDA, JNJ, XOM"

SECTOR_BENCHMARKS = {
    "Technology": {"PE": 30, "PS": 6.0, "PB": 8.0, "EV_EBITDA": 20},
    "Healthcare": {"PE": 25, "PS": 4.0, "PB": 4.0, "EV_EBITDA": 15},
    "Financial Services": {"PE": 14, "PS": 3.0, "PB": 1.5, "EV_EBITDA": 10},
    "Energy": {"PE": 10, "PS": 1.5, "PB": 1.8, "EV_EBITDA": 6},
    "Consumer Cyclical": {"PE": 20, "PS": 2.5, "PB": 4.0, "EV_EBITDA": 14},
    "Industrials": {"PE": 20, "PS": 2.0, "PB": 3.5, "EV_EBITDA": 12},
    "Utilities": {"PE": 18, "PS": 2.5, "PB": 2.0, "EV_EBITDA": 10},
    "Consumer Defensive": {"PE": 22, "PS": 2.0, "PB": 4.0, "EV_EBITDA": 15},
    "Real Estate": {"PE": 35, "PS": 6.0, "PB": 3.0, "EV_EBITDA": 18},
    "Communication Services": {"PE": 20, "PS": 4.0, "PB": 3.0, "EV_EBITDA": 12},
    "Basic Materials": {"PE": 15, "PS": 1.5, "PB": 2.0, "EV_EBITDA": 8},
}
DEFAULT_BENCHMARKS = {"PE": 20, "PS": 3.0, "PB": 3.0, "EV_EBITDA": 12}
ANALYSIS_COLUMNS = {
    "Ticker": "TEXT PRIMARY KEY",
    "Price": "REAL",
    "Verdict_Overall": "TEXT",
    "Verdict_Technical": "TEXT",
    "Verdict_Fundamental": "TEXT",
    "Verdict_Valuation": "TEXT",
    "Verdict_Sentiment": "TEXT",
    "Score_Tech": "INTEGER",
    "Score_Fund": "INTEGER",
    "Score_Val": "INTEGER",
    "Score_Sentiment": "INTEGER",
    "Sector": "TEXT",
    "PE_Ratio": "REAL",
    "Forward_PE": "REAL",
    "PEG_Ratio": "REAL",
    "PS_Ratio": "REAL",
    "PB_Ratio": "REAL",
    "EV_EBITDA": "REAL",
    "Graham_Number": "REAL",
    "Intrinsic_Value": "REAL",
    "Profit_Margins": "REAL",
    "ROE": "REAL",
    "Debt_to_Equity": "REAL",
    "Revenue_Growth": "REAL",
    "Current_Ratio": "REAL",
    "Target_Mean_Price": "REAL",
    "Recommendation_Key": "TEXT",
    "Analyst_Opinions": "REAL",
    "Sentiment_Headline_Count": "INTEGER",
    "Sentiment_Summary": "TEXT",
    "RSI": "REAL",
    "MACD_Value": "REAL",
    "MACD_Signal": "TEXT",
    "SMA_Status": "TEXT",
    "Momentum_1M": "REAL",
    "Momentum_1Y": "REAL",
    "Last_Updated": "TEXT",
}
POSITIVE_SENTIMENT_TERMS = {
    "beat", "beats", "growth", "surge", "surges", "strong", "bullish", "buy",
    "outperform", "upgrade", "record", "expands", "expansion", "profit",
    "profits", "optimistic", "momentum", "gain", "gains",
}
NEGATIVE_SENTIMENT_TERMS = {
    "miss", "misses", "lawsuit", "drop", "drops", "fall", "falls", "weak",
    "bearish", "sell", "downgrade", "cut", "cuts", "risk", "risks", "probe",
    "investigation", "decline", "declines", "loss", "losses", "warning",
}


def safe_num(value):
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


def get_color(verdict):
    if "STRONG BUY" in verdict or verdict in {"BUY", "STRONG", "UNDERVALUED", "POSITIVE"}:
        return "green"
    if "STRONG SELL" in verdict or verdict in {"SELL", "WEAK", "OVERVALUED", "NEGATIVE"}:
        return "red"
    return "gray"


def format_value(value, fmt="{:,.2f}", suffix=""):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{fmt.format(value)}{suffix}"


def format_percent(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.1f}%"


def format_int(value):
    if value is None or pd.isna(value):
        return "N/A"
    return str(int(value))


def safe_divide(numerator, denominator):
    if denominator is None or pd.isna(denominator) or abs(denominator) < 1e-12:
        return None
    return numerator / denominator


def parse_ticker_list(raw_text):
    tickers = []
    seen = set()
    normalized = raw_text.replace("\n", ",").replace(" ", ",")
    for token in normalized.split(","):
        ticker = token.strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        tickers.append(ticker)
    return tickers


def cap_weights(weights, max_weight):
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


def score_to_signal(score, strong_buy=4, buy=2, sell=-2, strong_sell=-4):
    if score >= strong_buy:
        return "STRONG BUY"
    if score >= buy:
        return "BUY"
    if score <= strong_sell:
        return "STRONG SELL"
    if score <= sell:
        return "SELL"
    return "HOLD"


def score_to_sentiment(score):
    if score >= 3:
        return "POSITIVE"
    if score <= -3:
        return "NEGATIVE"
    return "MIXED"


class DatabaseManager:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        column_sql = ",\n                ".join(
            f"{name} {definition}" for name, definition in ANALYSIS_COLUMNS.items()
        )
        self.conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS analysis (
                {column_sql}
            )
            """
        )
        existing_columns = {
            row[1] for row in self.conn.execute("PRAGMA table_info(analysis)").fetchall()
        }
        for name, definition in ANALYSIS_COLUMNS.items():
            if name not in existing_columns:
                self.conn.execute(f"ALTER TABLE analysis ADD COLUMN {name} {definition}")
        self.conn.commit()

    def save_analysis(self, data):
        keys = list(data.keys())
        placeholders = ", ".join(["?"] * len(keys))
        columns = ", ".join(keys)
        sql = f"INSERT OR REPLACE INTO analysis ({columns}) VALUES ({placeholders})"
        self.conn.execute(sql, list(data.values()))
        self.conn.commit()

    def get_analysis(self, ticker):
        return pd.read_sql_query(
            "SELECT * FROM analysis WHERE Ticker=?",
            self.conn,
            params=(ticker,),
        )


class StockAnalyst:
    def __init__(self, db):
        self.db = db

    def get_data(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            info = stock.info
            news = getattr(stock, "news", []) or []
            return hist, info, news
        except Exception:
            return None, None, []

    def analyze_sentiment(self, info, news, price):
        score = 0
        headlines = []
        for item in news[:8]:
            title = (item.get("title") or "").strip()
            if not title:
                continue
            headlines.append(title)
            lowered = title.lower()
            score += sum(term in lowered for term in POSITIVE_SENTIMENT_TERMS)
            score -= sum(term in lowered for term in NEGATIVE_SENTIMENT_TERMS)

        recommendation_key = (info.get("recommendationKey") or "").lower()
        analyst_opinions = safe_num(info.get("numberOfAnalystOpinions"))
        target_mean_price = safe_num(info.get("targetMeanPrice"))

        if recommendation_key in {"strong_buy", "buy"}:
            score += 2
        elif recommendation_key in {"underperform", "sell"}:
            score -= 2

        if target_mean_price and price:
            upside = (target_mean_price - price) / price
            if upside > 0.15:
                score += 2
            elif upside > 0.05:
                score += 1
            elif upside < -0.15:
                score -= 2
            elif upside < -0.05:
                score -= 1

        return {
            "score": score,
            "verdict": score_to_sentiment(score),
            "recommendation_key": recommendation_key.upper() if recommendation_key else "N/A",
            "analyst_opinions": analyst_opinions,
            "target_mean_price": target_mean_price,
            "headline_count": len(headlines),
            "summary": " | ".join(headlines[:3]) if headlines else "No recent headlines available.",
        }

    def analyze(self, ticker):
        ticker = ticker.strip().upper()
        hist, info, news = self.get_data(ticker)
        if hist is None or hist.empty or not info:
            return None

        delta = hist["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        current_rsi = (100 - (100 / (1 + rs))).iloc[-1]

        ema12 = hist["Close"].ewm(span=12, adjust=False).mean()
        ema26 = hist["Close"].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        macd_signal_line = macd_line.ewm(span=9, adjust=False).mean()
        current_macd = macd_line.iloc[-1]
        current_macd_signal = macd_signal_line.iloc[-1]

        sma50 = hist["Close"].rolling(50).mean().iloc[-1]
        sma200 = hist["Close"].rolling(200).mean().iloc[-1]
        price = hist["Close"].iloc[-1]
        momentum_1m = (price / hist["Close"].iloc[-22] - 1) if len(hist) > 22 else None
        momentum_1y = (price / hist["Close"].iloc[0] - 1) if len(hist) > 1 else None

        tech_score = 0
        tech_score += 1 if price > sma200 else -1
        tech_score += 1 if sma50 > sma200 else -1
        if current_rsi < 30:
            tech_score += 2
        elif current_rsi > 70:
            tech_score -= 2
        if current_macd > current_macd_signal:
            tech_score += 1
            macd_signal = "Bullish Crossover"
        else:
            tech_score -= 1
            macd_signal = "Bearish Crossover"
        if momentum_1m is not None:
            if momentum_1m > 0.05:
                tech_score += 1
            elif momentum_1m < -0.05:
                tech_score -= 1
        v_tech = score_to_signal(tech_score)

        f_score = 0
        roe = safe_num(info.get("returnOnEquity"))
        margins = safe_num(info.get("profitMargins"))
        debt_eq = safe_num(info.get("debtToEquity"))
        revenue_growth = safe_num(info.get("revenueGrowth"))
        current_ratio = safe_num(info.get("currentRatio"))
        if roe and roe > 0.15:
            f_score += 1
        if margins and margins > 0.20:
            f_score += 1
        if debt_eq and debt_eq < 100:
            f_score += 1
        elif debt_eq and debt_eq > 200:
            f_score -= 1
        if revenue_growth and revenue_growth > 0.10:
            f_score += 1
        elif revenue_growth and revenue_growth < 0:
            f_score -= 1
        if current_ratio and current_ratio > 1.2:
            f_score += 1
        elif current_ratio and current_ratio < 1.0:
            f_score -= 1
        if f_score >= 3:
            v_fund = "STRONG"
        elif f_score >= 1:
            v_fund = "STABLE"
        else:
            v_fund = "WEAK"

        v_score = 0
        sector = info.get("sector", "Unknown")
        bench = SECTOR_BENCHMARKS.get(sector, DEFAULT_BENCHMARKS)
        pe = safe_num(info.get("trailingPE"))
        forward_pe = safe_num(info.get("forwardPE"))
        peg_ratio = safe_num(info.get("pegRatio"))
        ps_ratio = safe_num(info.get("priceToSalesTrailing12Months"))
        ev_ebitda = safe_num(info.get("enterpriseToEbitda"))
        pb = safe_num(info.get("priceToBook"))
        if pe and pe < bench["PE"]:
            v_score += 1
        if forward_pe and forward_pe < bench["PE"]:
            v_score += 1
        if peg_ratio and peg_ratio < 1.5:
            v_score += 1
        if ps_ratio and ps_ratio < bench["PS"]:
            v_score += 1
        if ev_ebitda and ev_ebitda < bench["EV_EBITDA"]:
            v_score += 1
        if pb and pb < bench["PB"]:
            v_score += 1
        eps = safe_num(info.get("trailingEps"))
        bvps = safe_num(info.get("bookValue"))
        graham_num = None
        intrinsic_value = None
        if eps and bvps and eps > 0 and bvps > 0:
            graham_num = (22.5 * eps * bvps) ** 0.5
            intrinsic_value = graham_num
            if price < graham_num:
                v_score += 2
            elif price > graham_num * 1.5:
                v_score -= 1
        if v_score >= 5:
            v_val = "UNDERVALUED"
        elif v_score >= 2:
            v_val = "FAIR VALUE"
        else:
            v_val = "OVERVALUED"

        sentiment = self.analyze_sentiment(info, news, price)
        overall_score = tech_score + f_score + v_score + sentiment["score"]
        if v_val == "UNDERVALUED" and v_fund == "STRONG" and sentiment["score"] >= 2:
            final_verdict = "STRONG BUY" if tech_score >= 2 else "BUY"
        elif v_val == "OVERVALUED" and sentiment["score"] <= -2:
            final_verdict = "STRONG SELL" if tech_score <= -2 else "SELL"
        else:
            final_verdict = score_to_signal(overall_score, strong_buy=8, buy=3, sell=-3, strong_sell=-8)

        record = {
            "Ticker": ticker,
            "Price": price,
            "Verdict_Overall": final_verdict,
            "Verdict_Technical": v_tech,
            "Verdict_Fundamental": v_fund,
            "Verdict_Valuation": v_val,
            "Verdict_Sentiment": sentiment["verdict"],
            "Score_Tech": tech_score,
            "Score_Fund": f_score,
            "Score_Val": v_score,
            "Score_Sentiment": sentiment["score"],
            "Sector": sector,
            "PE_Ratio": pe,
            "Forward_PE": forward_pe,
            "PEG_Ratio": peg_ratio,
            "PS_Ratio": ps_ratio,
            "PB_Ratio": pb,
            "EV_EBITDA": ev_ebitda,
            "Graham_Number": graham_num if graham_num else 0,
            "Intrinsic_Value": intrinsic_value if intrinsic_value else 0,
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
            "RSI": current_rsi,
            "MACD_Value": current_macd,
            "MACD_Signal": macd_signal,
            "SMA_Status": "Bullish" if price > sma200 else "Bearish",
            "Momentum_1M": momentum_1m,
            "Momentum_1Y": momentum_1y,
            "Last_Updated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        self.db.save_analysis(record)
        return record


class PortfolioAnalyst:
    def __init__(self, db):
        self.db = db

    def get_price_history(self, tickers, benchmark_ticker, period):
        download_list = list(dict.fromkeys(tickers + [benchmark_ticker]))
        raw = yf.download(download_list, period=period, auto_adjust=True, progress=False)
        if raw.empty:
            return None, None

        if isinstance(raw.columns, pd.MultiIndex):
            close_prices = raw["Close"].copy()
        else:
            close_prices = raw.to_frame(name=download_list[0]) if isinstance(raw, pd.Series) else raw[["Close"]].copy()
            if "Close" in close_prices.columns:
                close_prices.columns = [download_list[0]]

        available_assets = [ticker for ticker in tickers if ticker in close_prices.columns]
        if benchmark_ticker not in close_prices.columns or len(available_assets) < 2:
            return None, None

        combined_columns = list(dict.fromkeys(available_assets + [benchmark_ticker]))
        aligned_prices = close_prices[combined_columns].dropna()
        if aligned_prices.empty or len(aligned_prices) < 30:
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

            try:
                info = yf.Ticker(ticker).info
                name = info.get("shortName") or info.get("longName") or ticker
                sector = info.get("sector") or sector
            except Exception:
                pass

            rows.append({"Ticker": ticker, "Name": name, "Sector": sector})

        return pd.DataFrame(rows)

    def calculate_asset_metrics(self, asset_returns, benchmark_returns, risk_free_rate):
        risk_free_daily = risk_free_rate / TRADING_DAYS
        annual_return = asset_returns.mean() * TRADING_DAYS
        annual_volatility = asset_returns.std() * np.sqrt(TRADING_DAYS)
        downside_diff = (asset_returns - risk_free_daily).clip(upper=0)
        downside_volatility = np.sqrt((downside_diff.pow(2)).mean()) * np.sqrt(TRADING_DAYS)

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

    def calculate_portfolio_metrics(self, asset_returns, benchmark_returns, weights, risk_free_rate):
        risk_free_daily = risk_free_rate / TRADING_DAYS
        portfolio_returns = asset_returns @ weights
        annual_return = portfolio_returns.mean() * TRADING_DAYS
        volatility = portfolio_returns.std() * np.sqrt(TRADING_DAYS)
        downside_diff = (portfolio_returns - risk_free_daily).clip(upper=0)
        downside_volatility = np.sqrt((downside_diff.pow(2)).mean()) * np.sqrt(TRADING_DAYS)
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

    def simulate_portfolios(self, asset_returns, benchmark_returns, risk_free_rate, max_weight, simulations):
        rng = np.random.default_rng(42)
        tickers = list(asset_returns.columns)
        portfolios = []

        for _ in range(simulations):
            weights = cap_weights(rng.random(len(tickers)), max_weight)
            metrics = self.calculate_portfolio_metrics(asset_returns, benchmark_returns, weights, risk_free_rate)
            row = {**metrics}
            for ticker, weight in zip(tickers, weights):
                row[f"W_{ticker}"] = weight
            portfolios.append(row)

        portfolio_df = pd.DataFrame(portfolios).replace([np.inf, -np.inf], np.nan).dropna(subset=["Return", "Volatility", "Sharpe"])
        if portfolio_df.empty:
            return None, None, None, None, None

        frontier_rows = []
        best_return = -np.inf
        for _, row in portfolio_df.sort_values("Volatility").iterrows():
            if row["Return"] > best_return:
                frontier_rows.append(row)
                best_return = row["Return"]

        frontier = pd.DataFrame(frontier_rows)
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
        if len(tickers) * max_weight < 1:
            return None

        asset_prices, benchmark_prices = self.get_price_history(tickers, benchmark_ticker, period)
        if asset_prices is None or benchmark_prices is None:
            return None

        asset_returns = asset_prices.pct_change().dropna()
        benchmark_returns = benchmark_prices.pct_change().dropna()
        common_index = asset_returns.index.intersection(benchmark_returns.index)
        asset_returns = asset_returns.loc[common_index]
        benchmark_returns = benchmark_returns.loc[common_index]
        if asset_returns.empty or len(asset_returns.columns) < 2:
            return None

        asset_metrics = self.calculate_asset_metrics(asset_returns, benchmark_returns, risk_free_rate)
        portfolio_df, frontier, tangent, minimum_volatility, cal = self.simulate_portfolios(
            asset_returns,
            benchmark_returns,
            risk_free_rate,
            max_weight,
            simulations,
        )
        if portfolio_df is None:
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


def render_frontier_chart(portfolio_cloud, frontier, cal, tangent, minimum_volatility):
    cloud = portfolio_cloud[["Volatility", "Return", "Sharpe"]].copy()
    cloud["Series"] = "Simulated Portfolios"

    frontier_data = frontier[["Volatility", "Return"]].copy()
    frontier_data["Sharpe"] = np.nan
    frontier_data["Series"] = "Efficient Frontier"

    cal_data = cal.copy()
    cal_data["Sharpe"] = np.nan
    cal_data["Series"] = "CAL"

    marker_data = pd.DataFrame(
        [
            {
                "Volatility": tangent["Volatility"],
                "Return": tangent["Return"],
                "Sharpe": tangent["Sharpe"],
                "Series": "Max Sharpe",
            },
            {
                "Volatility": minimum_volatility["Volatility"],
                "Return": minimum_volatility["Return"],
                "Sharpe": minimum_volatility["Sharpe"],
                "Series": "Min Volatility",
            },
        ]
    )

    chart_data = pd.concat([cloud, frontier_data, cal_data, marker_data], ignore_index=True)
    spec = {
        "layer": [
            {
                "transform": [{"filter": "datum.Series == 'Simulated Portfolios'"}],
                "mark": {"type": "circle", "opacity": 0.35, "size": 45},
                "encoding": {
                    "x": {"field": "Volatility", "type": "quantitative", "title": "Annualized Volatility"},
                    "y": {"field": "Return", "type": "quantitative", "title": "Annualized Return"},
                    "color": {"field": "Sharpe", "type": "quantitative", "title": "Sharpe"},
                },
            },
            {
                "transform": [{"filter": "datum.Series == 'Efficient Frontier'"}],
                "mark": {"type": "line", "strokeWidth": 3, "color": "#0b7285"},
                "encoding": {
                    "x": {"field": "Volatility", "type": "quantitative"},
                    "y": {"field": "Return", "type": "quantitative"},
                },
            },
            {
                "transform": [{"filter": "datum.Series == 'CAL'"}],
                "mark": {"type": "line", "strokeWidth": 2, "strokeDash": [6, 4], "color": "#e03131"},
                "encoding": {
                    "x": {"field": "Volatility", "type": "quantitative"},
                    "y": {"field": "Return", "type": "quantitative"},
                },
            },
            {
                "transform": [{"filter": "datum.Series == 'Max Sharpe'"}],
                "mark": {"type": "point", "filled": True, "size": 170, "shape": "diamond", "color": "#2b8a3e"},
                "encoding": {
                    "x": {"field": "Volatility", "type": "quantitative"},
                    "y": {"field": "Return", "type": "quantitative"},
                },
            },
            {
                "transform": [{"filter": "datum.Series == 'Min Volatility'"}],
                "mark": {"type": "point", "filled": True, "size": 170, "shape": "square", "color": "#f08c00"},
                "encoding": {
                    "x": {"field": "Volatility", "type": "quantitative"},
                    "y": {"field": "Return", "type": "quantitative"},
                },
            },
        ]
    }
    st.vega_lite_chart(chart_data, spec, use_container_width=True)


st.set_page_config(page_title="Stock Engine Pro", layout="wide", page_icon="SE")

if "db" not in st.session_state:
    st.session_state.db = DatabaseManager(DB_FILENAME)
    st.session_state.analyst = StockAnalyst(st.session_state.db)
    st.session_state.portfolio_analyst = PortfolioAnalyst(st.session_state.db)
elif "analyst" not in st.session_state:
    st.session_state.analyst = StockAnalyst(st.session_state.db)
elif "portfolio_analyst" not in st.session_state:
    st.session_state.portfolio_analyst = PortfolioAnalyst(st.session_state.db)

db = st.session_state.db
bot = st.session_state.analyst
portfolio_bot = st.session_state.portfolio_analyst

st.title("Stock Engine Pro")
st.markdown("### Single-Stock Analysis and Portfolio Construction")

stock_tab, portfolio_tab = st.tabs(["Stock Analysis", "Portfolio"])

with stock_tab:
    c1, c2 = st.columns([3, 1])
    with c1:
        txt_input = st.text_input("Enter Ticker Symbol (e.g., AAPL, NVDA, F)", "", key="single_ticker")
    with c2:
        st.write("")
        st.write("")
        if st.button("Run Full Analysis", type="primary", use_container_width=True):
            if txt_input:
                with st.spinner(f"Running multiple engines on {txt_input}..."):
                    res = bot.analyze(txt_input)
                    if not res:
                        st.error("Data fetch failed.")

    if txt_input:
        df = db.get_analysis(txt_input.upper())
        if not df.empty:
            row = df.iloc[0]
            sector_bench = SECTOR_BENCHMARKS.get(row["Sector"], DEFAULT_BENCHMARKS)

            st.divider()
            col_main_1, col_main_2, col_main_3 = st.columns([1, 2, 1])
            with col_main_1:
                st.metric("Current Price", f"${row['Price']:,.2f}")
            with col_main_2:
                st.markdown(
                    f"<h2 style='text-align: center; color: {get_color(row['Verdict_Overall'])};'>VERDICT: {row['Verdict_Overall']}</h2>",
                    unsafe_allow_html=True,
                )
            with col_main_3:
                st.metric("Sector", str(row["Sector"]))

            st.subheader("Method Breakdown")
            st.info("Observe how the verdict changes across technical, fundamental, valuation, and sentiment engines.")

            summary_cols = st.columns(5)
            summary_cols[0].metric("Technical", format_int(row["Score_Tech"]))
            summary_cols[1].metric("Fundamental", format_int(row["Score_Fund"]))
            summary_cols[2].metric("Valuation", format_int(row["Score_Val"]))
            summary_cols[3].metric("Sentiment", format_int(row["Score_Sentiment"]))
            summary_cols[4].metric("Updated", str(row["Last_Updated"]))

            tab_val, tab_fund, tab_tech, tab_sent = st.tabs(
                ["Valuation Engine", "Fundamental Engine", "Technical Engine", "Sentiment Engine"]
            )

            with tab_val:
                c_v1, c_v2 = st.columns([1, 2])
                with c_v1:
                    st.markdown(f"### Verdict: **{row['Verdict_Valuation']}**")
                    st.caption("Based on relative multiples and Graham-style fair value.")
                    st.metric(
                        "Graham Fair Value",
                        f"${row['Graham_Number']:,.2f}",
                        delta=f"{row['Price'] - row['Graham_Number']:,.2f} diff",
                        delta_color="inverse",
                    )
                with c_v2:
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Metric": ["P/E Ratio", "Forward P/E", "PEG Ratio", "P/S Ratio", "EV/EBITDA", "P/B Ratio"],
                                "Stock Value": [
                                    format_value(row["PE_Ratio"]),
                                    format_value(row["Forward_PE"]),
                                    format_value(row["PEG_Ratio"]),
                                    format_value(row["PS_Ratio"]),
                                    format_value(row["EV_EBITDA"]),
                                    format_value(row["PB_Ratio"]),
                                ],
                                "Benchmark": [
                                    format_value(sector_bench["PE"]),
                                    format_value(sector_bench["PE"]),
                                    "1.50",
                                    format_value(sector_bench["PS"]),
                                    format_value(sector_bench["EV_EBITDA"]),
                                    format_value(sector_bench["PB"]),
                                ],
                            }
                        ),
                        use_container_width=True,
                    )

            with tab_fund:
                c_f1, c_f2 = st.columns([1, 2])
                with c_f1:
                    st.markdown(f"### Verdict: **{row['Verdict_Fundamental']}**")
                    st.caption("Based on profitability, growth, leverage, and liquidity.")
                with c_f2:
                    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
                    col_m1.metric("ROE", format_percent(row["ROE"]), "Target: >15%")
                    col_m2.metric("Profit Margin", format_percent(row["Profit_Margins"]), "Target: >20%")
                    col_m3.metric("Debt/Equity", format_value(row["Debt_to_Equity"], "{:,.0f}", "%"), "Target: <100%", delta_color="inverse")
                    col_m4.metric("Revenue Growth", format_percent(row["Revenue_Growth"]), "Target: >10%")
                    col_m5.metric("Current Ratio", format_value(row["Current_Ratio"]), "Target: >1.2")

            with tab_tech:
                c_t1, c_t2 = st.columns([1, 2])
                with c_t1:
                    st.markdown(f"### Verdict: **{row['Verdict_Technical']}**")
                    st.caption("Based on RSI, MACD, moving averages, and momentum.")
                with c_t2:
                    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
                    col_t1.metric("RSI (14)", format_value(row["RSI"], "{:,.1f}"), "30=Oversold, 70=Overbought")
                    col_t2.metric("Trend", str(row["SMA_Status"]))
                    col_t3.metric("MACD", format_value(row["MACD_Value"], "{:,.2f}"), str(row["MACD_Signal"]))
                    col_t4.metric("1M Momentum", format_percent(row["Momentum_1M"]), "Short-term move")

                st.dataframe(
                    pd.DataFrame(
                        {
                            "Signal": ["RSI", "200-Day Trend", "MACD Signal", "1Y Momentum"],
                            "Value": [
                                format_value(row["RSI"], "{:,.1f}"),
                                str(row["SMA_Status"]),
                                str(row["MACD_Signal"]),
                                format_percent(row["Momentum_1Y"]),
                            ],
                        }
                    ),
                    use_container_width=True,
                )

            with tab_sent:
                c_s1, c_s2 = st.columns([1, 2])
                with c_s1:
                    st.markdown(f"### Verdict: **{row['Verdict_Sentiment']}**")
                    st.caption("Based on recent headline tone and analyst expectations.")
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Headlines", format_int(row["Sentiment_Headline_Count"]))
                    s2.metric("Analyst View", str(row["Recommendation_Key"]))
                    target_price = row["Target_Mean_Price"]
                    s3.metric("Target Mean", "N/A" if pd.isna(target_price) else f"${target_price:,.2f}")
                with c_s2:
                    st.write("Recent sentiment summary")
                    st.caption(str(row["Sentiment_Summary"]))

with portfolio_tab:
    st.subheader("Portfolio Builder")
    st.caption("Use modern portfolio basics to recommend a risk-aware stock mix with Sharpe, Sortino, Treynor, the efficient frontier, and the Capital Allocation Line.")

    with st.form("portfolio_form"):
        p1, p2 = st.columns([3, 1])
        with p1:
            portfolio_tickers_raw = st.text_area(
                "Portfolio Tickers",
                value=DEFAULT_PORTFOLIO_TICKERS,
                help="Enter at least two tickers separated by commas or spaces.",
            )
        with p2:
            benchmark_ticker = st.text_input("Benchmark", value=DEFAULT_BENCHMARK_TICKER)
            lookback_period = st.selectbox("Lookback Period", ["1y", "3y", "5y"], index=1)
            risk_free_percent = st.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=15.0, value=4.0, step=0.25)

        p3, p4 = st.columns([3, 2])
        with p3:
            max_weight_percent = st.slider("Max Single-Stock Weight (%)", min_value=15, max_value=50, value=30, step=5)
        with p4:
            simulations = st.select_slider("Frontier Simulations", options=[1000, 2000, 3000, 4000, 5000], value=3000)

        portfolio_submit = st.form_submit_button("Build Portfolio Recommendation", type="primary", use_container_width=True)

    if portfolio_submit:
        parsed_tickers = parse_ticker_list(portfolio_tickers_raw)
        if len(parsed_tickers) < 2:
            st.error("Enter at least two valid ticker symbols for portfolio analysis.")
        elif len(parsed_tickers) * (max_weight_percent / 100) < 1:
            st.error("The max single-stock weight is too low for the number of tickers. Raise the cap or add more names.")
        else:
            with st.spinner("Building efficient frontier and CAL recommendation..."):
                portfolio_result = portfolio_bot.analyze_portfolio(
                    tickers=parsed_tickers,
                    benchmark_ticker=benchmark_ticker.strip().upper() or DEFAULT_BENCHMARK_TICKER,
                    period=lookback_period,
                    risk_free_rate=risk_free_percent / 100,
                    max_weight=max_weight_percent / 100,
                    simulations=simulations,
                )

            if not portfolio_result:
                st.error("Portfolio analysis failed. Try different tickers or a longer lookback period.")
            else:
                st.session_state.portfolio_result = portfolio_result
                st.session_state.portfolio_config = {
                    "tickers": parsed_tickers,
                    "benchmark": benchmark_ticker.strip().upper() or DEFAULT_BENCHMARK_TICKER,
                    "period": lookback_period,
                    "risk_free_percent": risk_free_percent,
                    "max_weight_percent": max_weight_percent,
                    "simulations": simulations,
                }

    if "portfolio_result" in st.session_state:
        result = st.session_state.portfolio_result
        config = st.session_state.get("portfolio_config", {})
        tangent = result["tangent"]
        min_vol = result["minimum_volatility"]
        recommendations = result["recommendations"].copy()
        sector_exposure = result["sector_exposure"].copy()

        st.divider()
        st.caption(
            f"Benchmark: {config.get('benchmark', result['benchmark'])} | Lookback: {config.get('period', result['period'])} | "
            f"Risk-free rate: {config.get('risk_free_percent', 0):.2f}% | Max position: {config.get('max_weight_percent', 0)}%"
        )

        metric_cols = st.columns(5)
        metric_cols[0].metric("Expected Return", format_percent(tangent["Return"]))
        metric_cols[1].metric("Volatility", format_percent(tangent["Volatility"]))
        metric_cols[2].metric("Sharpe", format_value(tangent["Sharpe"]))
        metric_cols[3].metric("Sortino", format_value(tangent["Sortino"]))
        metric_cols[4].metric("Treynor", format_value(tangent["Treynor"]))

        metric_cols_2 = st.columns(4)
        metric_cols_2[0].metric("Portfolio Beta", format_value(tangent["Beta"]))
        metric_cols_2[1].metric("Downside Vol", format_percent(tangent["Downside Volatility"]))
        metric_cols_2[2].metric("Min-Vol Return", format_percent(min_vol["Return"]))
        metric_cols_2[3].metric("Effective Names", format_value(result["effective_names"], "{:,.1f}"))

        st.subheader("Efficient Frontier and CAL")
        st.caption("The green diamond is the tangent portfolio with the highest Sharpe ratio. The red dashed line is the Capital Allocation Line.")
        render_frontier_chart(result["portfolio_cloud"], result["frontier"], result["cal"], tangent, min_vol)

        st.subheader("Recommended Allocation")
        recommendations_display = recommendations[
            ["Ticker", "Name", "Sector", "Recommended Weight", "Role", "Sharpe Ratio", "Sortino Ratio", "Treynor Ratio", "Beta", "Rationale"]
        ].copy()
        recommendations_display["Recommended Weight"] = recommendations_display["Recommended Weight"].map(format_percent)
        recommendations_display["Sharpe Ratio"] = recommendations_display["Sharpe Ratio"].map(format_value)
        recommendations_display["Sortino Ratio"] = recommendations_display["Sortino Ratio"].map(format_value)
        recommendations_display["Treynor Ratio"] = recommendations_display["Treynor Ratio"].map(format_value)
        recommendations_display["Beta"] = recommendations_display["Beta"].map(format_value)
        st.dataframe(recommendations_display, use_container_width=True)

        exposure_col, metrics_col = st.columns([1, 2])
        with exposure_col:
            st.subheader("Sector Exposure")
            sector_display = sector_exposure.copy()
            sector_display["Recommended Weight"] = sector_display["Recommended Weight"].map(format_percent)
            st.dataframe(sector_display, use_container_width=True)

        with metrics_col:
            st.subheader("Per-Stock Metrics")
            asset_display = result["asset_metrics"].copy()
            for column in ["Annual Return", "Volatility", "Downside Volatility"]:
                asset_display[column] = asset_display[column].map(format_percent)
            for column in ["Beta", "Sharpe Ratio", "Sortino Ratio", "Treynor Ratio"]:
                asset_display[column] = asset_display[column].map(format_value)
            st.dataframe(asset_display, use_container_width=True)

        st.subheader("Portfolio Building Notes")
        for note in result["notes"]:
            st.write(f"- {note}")
