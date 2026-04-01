# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import os
import time
import json
import sqlite3
import argparse
from colorama import init, Fore, Style

# Initialize colorama for colored console output
init(autoreset=True)

# --- 1. CONFIGURATION MANAGEMENT ---
DEFAULT_CONFIG = {
    "RSI_OVERSOLD": 30,
    "RSI_OVERBOUGHT": 70,
    "SMA_SHORT": 50,
    "SMA_LONG": 200,
    "PE_THRESHOLD_LOW": 15,
    "ROE_MIN": 0.15,
    "VOLUME_THRESHOLD": 500000,
    "ATR_MULTIPLIER": 2.5,
    "REFRESH_HOURS": 24,
    "DB_FILENAME": "stocks_data.db",
    "EXCEL_FILENAME": "Stock_Analysis_Report.xlsx",
    "SUMMARY_OUTPUT_COLS": ["Ticker", "Price", "Verdict", "Fund_Score", "Risk_Level", "Stop_Loss", "Target_Price", "Rel_Strength_SP500"]
}

def load_config():
    """Loads config from JSON or creates default if missing."""
    config_file = 'config.json'
    if not os.path.exists(config_file):
        print(Fore.YELLOW + "Creating default 'config.json'...")
        with open(config_file, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG
    else:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                # Ensure new keys are in config if user has old file
                for key, default in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = default
                return config
        except Exception as e:
            print(Fore.RED + f"Error loading config: {e}. Using defaults.")
            return DEFAULT_CONFIG

# --- 2. DATABASE MANAGEMENT ---
class DatabaseManager:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        # Expanded Schema for new metrics
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis (
                Ticker TEXT PRIMARY KEY,
                Price REAL,
                Verdict TEXT,
                Fund_Score INTEGER,
                Risk_Level TEXT,
                Stop_Loss REAL,
                Target_Price REAL,
                Rel_Strength_SP500 TEXT,
                News_Sentiment TEXT,
                RSI REAL,
                Trend TEXT,
                MACD_Hist REAL,
                PE_Ratio REAL,
                PS_Ratio REAL,
                PB_Ratio REAL,
                Beta REAL,
                Volatility REAL,
                Debt_to_Equity REAL,
                ROE REAL,
                Div_Yield REAL,
                Avg_Volume REAL,
                Liquidity_Status TEXT,
                Notes TEXT,
                Last_Updated TEXT
            )
        ''')
        self.conn.commit()

    def save_record(self, data):
        keys = list(data.keys())
        values = list(data.values())
        placeholders = ', '.join(['?'] * len(keys))
        columns = ', '.join(keys)
        
        sql = f'''
            INSERT INTO analysis ({columns}) VALUES ({placeholders})
            ON CONFLICT(Ticker) DO UPDATE SET
            {', '.join([f"{k}=excluded.{k}" for k in keys])}
        '''
        
        self.cursor.execute(sql, values)
        self.conn.commit()

    def fetch_all(self):
        return pd.read_sql_query("SELECT * FROM analysis", self.conn)

    def fetch_tickers(self):
        self.cursor.execute("SELECT Ticker FROM analysis")
        return [row[0] for row in self.cursor.fetchall()]

    def get_last_updated(self, ticker):
        self.cursor.execute("SELECT Last_Updated FROM analysis WHERE Ticker=?", (ticker,))
        result = self.cursor.fetchone()
        if result:
            return datetime.datetime.strptime(result[0], "%Y-%m-%d %H:%M")
        return None

    def close(self):
        self.conn.close()

# --- 3. CORE ANALYST ---
class StockAnalyst:
    def __init__(self, config, db_manager):
        self.cfg = config
        self.db = db_manager
        print(Fore.CYAN + "Fetching S&P 500 baseline data...")
        self.sp500_hist = self.fetch_sp500()

    def fetch_sp500(self):
        try:
            sp = yf.Ticker("^GSPC")
            hist = sp.history(period="1y")
            return hist
        except:
            return pd.DataFrame()

    def fetch_data(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            # Use '1y' as the default analysis period for indicator calculation
            hist = stock.history(period="1y") 
            info = stock.info
            if hist.empty: return None, None
            return hist, info
        except Exception as e:
            print(Fore.RED + f"Error fetching {ticker}: {e}")
            return None, None

    def calculate_atr(self, df):
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window=14).mean().iloc[-1]

    def calculate_technicals(self, df):
        # Technicals (unchanged)
        df['SMA_50'] = df['Close'].rolling(window=self.cfg['SMA_SHORT']).mean()
        df['SMA_200'] = df['Close'].rolling(window=self.cfg['SMA_LONG']).mean()
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_Line'] = ema12 - ema26
        df['Signal_Line'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD_Line'] - df['Signal_Line']
        df['Returns'] = df['Close'].pct_change()
        volatility = df['Returns'].std() * np.sqrt(252)
        return df, volatility

    def get_fundamental_score(self, info):
        """Piotroski F-Score Proxy (0-9)."""
        score = 0
        
        # Profitability (4 points)
        if info.get('returnOnAssets', -1) > 0: score += 1
        if info.get('returnOnEquity', -1) > 0: score += 1
        if info.get('grossMargins', -1) > 0.3: score += 1 # Good Gross Margin
        if info.get('profitMargins', -1) > 0.05: score += 1 # Good Profit Margin
        
        # Leverage, Liquidity (2 points)
        current_ratio = info.get('currentRatio')
        if current_ratio and current_ratio > 1.5: score += 1 # Stronger liquidity
        debt_equity = info.get('debtToEquity')
        if debt_equity is not None and debt_equity < 50: score += 1 # Low Debt/Equity (conservative)
        
        # Efficiency, Growth (3 points)
        if info.get('revenueGrowth', -1) > 0.10: score += 1 # Revenue Growth > 10%
        if info.get('earningsGrowth', -1) > 0.10: score += 1 # Earnings Growth > 10%
        if info.get('totalAssets', 0) > info.get('totalAssets', 0) * 1.05: score += 1 # Asset Turnover Improvement (Proxied)

        return score
    
    def get_news_sentiment(self, ticker):
        """Placeholder for news API integration."""
        # In a real application, you would call a News API here.
        # Example: NewsAPI, Alpha Vantage News, etc.
        
        # For this script, we'll return a static result
        headlines = [
            f"**{ticker}** sees 10% price target boost after earnings.",
            "Market anxiety over interest rates.",
            f"Major institutional sale of **{ticker}** shares reported."
        ]
        
        # Simple keyword-based sentiment proxy
        positive_keywords = ['boost', 'strong', 'growth', 'buy', 'upgrade']
        negative_keywords = ['sell', 'cut', 'downgrade', 'anxiety', 'risk']
        
        sentiment = 0
        for h in headlines:
            if any(k in h.lower() for k in positive_keywords): sentiment += 1
            if any(k in h.lower() for k in negative_keywords): sentiment -= 1

        if sentiment > 0:
            return "Positive", "\n".join(headlines)
        elif sentiment < 0:
            return "Negative", "\n".join(headlines)
        else:
            return "Neutral", "\n".join(headlines)


    def analyze_ticker(self, ticker, force_refresh=False):
        """Main analysis logic for a single ticker with refresh check."""
        ticker = ticker.strip().upper()
        
        # Check refresh requirement
        last_update = self.db.get_last_updated(ticker)
        if last_update and not force_refresh:
            age = (datetime.datetime.now() - last_update).total_seconds() / 3600
            if age < self.cfg['REFRESH_HOURS']:
                print(f"  > Skipping {ticker}. Data is fresh (Age: {age:.1f} hours).")
                return None

        print(Fore.GREEN + f"Processing {ticker}..." + Style.RESET_ALL)
        hist, info = self.fetch_data(ticker)
        
        if hist is None: 
            print(Fore.RED + f"  > Failed {ticker}" + Style.RESET_ALL)
            return None

        # 1. Technicals & Risk
        df, volatility = self.calculate_technicals(hist)
        last_row = df.iloc[-1]
        atr = self.calculate_atr(df)
        
        stop_loss_price = last_row['Close'] - (atr * self.cfg['ATR_MULTIPLIER'])
        avg_volume = info.get('averageVolume', 0)
        is_liquid = "Liquid" if avg_volume > self.cfg['VOLUME_THRESHOLD'] else "Illiquid"
        beta = info.get('beta', 1.0)
        
        risk_level = "Medium"
        if beta > 1.5 or volatility > 0.40 or is_liquid == "Illiquid": risk_level = "High"
        if beta < 0.8 and volatility < 0.20 and is_liquid == "Liquid": risk_level = "Low"

        # 2. Relative Strength
        stock_return = (last_row['Close'] - df.iloc[0]['Close']) / df.iloc[0]['Close']
        rel_strength_str = "N/A"
        if not self.sp500_hist.empty and not self.sp500_hist.empty:
            sp_start = self.sp500_hist.iloc[0]['Close']
            sp_end = self.sp500_hist.iloc[-1]['Close']
            sp500_return = (sp_end - sp_start) / sp_start
            rel_diff = stock_return - sp500_return
            rel_strength_str = f"{rel_diff*100:+.1f}% vs SPY"

        # 3. Fundamentals & News
        fund_score = self.get_fundamental_score(info)
        news_sentiment, news_notes = self.get_news_sentiment(ticker)

        # 4. Generate Verdict
        verdict_score = 0
        reasons = []

        if last_row['RSI'] < self.cfg['RSI_OVERSOLD']: verdict_score += 2; reasons.append("Oversold")
        elif last_row['RSI'] > self.cfg['RSI_OVERBOUGHT']: verdict_score -= 2; reasons.append("Overbought")
        if last_row['MACD_Line'] > last_row['Signal_Line']: verdict_score += 1; reasons.append("MACD Bullish")
        
        if fund_score >= 7: verdict_score += 1; reasons.append("Strong Funds")
        if fund_score <= 3: verdict_score -= 1; reasons.append("Weak Funds")

        pe = info.get('trailingPE')
        if pe and 0 < pe < self.cfg['PE_THRESHOLD_LOW']: verdict_score += 1; reasons.append("Low P/E")
        
        if verdict_score >= 3: verdict = "STRONG BUY"
        elif verdict_score >= 1: verdict = "BUY"
        elif verdict_score <= -3: verdict = "STRONG SELL"
        elif verdict_score <= -1: verdict = "SELL"
        else: verdict = "HOLD"

        # 5. Construct Data Payload
        data = {
            'Ticker': ticker,
            'Price': last_row['Close'],
            'Verdict': verdict,
            'Fund_Score': fund_score,
            'Risk_Level': risk_level,
            'Stop_Loss': stop_loss_price,
            'Target_Price': info.get('targetMeanPrice', 0),
            'Rel_Strength_SP500': rel_strength_str,
            'News_Sentiment': news_sentiment,
            'RSI': last_row['RSI'],
            'Trend': "Bullish" if last_row['SMA_50'] > last_row['SMA_200'] else "Bearish",
            'MACD_Hist': last_row['MACD_Hist'],
            'PE_Ratio': info.get('trailingPE', 0),
            'PS_Ratio': info.get('priceToSalesTrailing12Months', 0),
            'PB_Ratio': info.get('priceToBook', 0),
            'Beta': beta,
            'Volatility': volatility,
            'Debt_to_Equity': info.get('debtToEquity', 0),
            'ROE': info.get('returnOnEquity', 0),
            'Div_Yield': info.get('dividendYield', 0),
            'Avg_Volume': avg_volume,
            'Liquidity_Status': is_liquid,
            'Notes': f"{', '.join(reasons)}. News: {news_notes.replace('\n', ' | ')}",
            'Last_Updated': datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        
        self.db.save_record(data)
        return data

    def refresh_all(self):
        """Refreshes all tickers currently in the database."""
        tickers = self.db.fetch_tickers()
        print(Fore.MAGENTA + f"\n--- Starting Full Refresh for {len(tickers)} stocks ---" + Style.RESET_ALL)
        for t in tickers:
            self.analyze_ticker(t, force_refresh=True)
        print(Fore.MAGENTA + "--- Refresh Complete ---" + Style.RESET_ALL)

    def export_to_excel(self, tickers_analyzed=None):
        filename = self.cfg['EXCEL_FILENAME']
        df = self.db.fetch_all()
        
        if df.empty:
            print("No data to export.")
            return

        # 1. Prepare DataFrames
        
        # Dashboard/Summary Sheet
        summary_cols = self.cfg['SUMMARY_OUTPUT_COLS']
        available_cols = [c for c in summary_cols if c in df.columns]
        df_summary = df[available_cols].copy()
        
        # Performance Sheet
        df_performance = df[['Ticker', 'Rel_Strength_SP500', 'Volatility', 'Beta']].copy()
        
        # Deep Dive Sheet (Filtering for only the LATEST batch)
        if tickers_analyzed:
            df_detail = df[df['Ticker'].isin(tickers_analyzed)].copy()
        else:
            # If no tickers analyzed, export everything
            df_detail = df.copy()

        try:
            writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        except PermissionError:
            print(Fore.RED + f"\n[!] PERMISSION ERROR: Close '{filename}' and press Enter.")
            input("Press Enter to retry...")
            try:
                writer = pd.ExcelWriter(filename, engine='xlsxwriter')
            except Exception:
                print(Fore.RED + "    Still failed. Data is safe in DB.")
                return

        # 2. Write Sheets
        df_summary.to_excel(writer, sheet_name='Dashboard', index=False)
        df_performance.to_excel(writer, sheet_name='Performance', index=False)
        df_detail.to_excel(writer, sheet_name='Deep Dive', index=False)

        # 3. Formatting
        workbook = writer.book
        green_fmt = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
        red_fmt = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        money_fmt = workbook.add_format({'num_format': '$#,##0.00'})
        pct_fmt   = workbook.add_format({'num_format': '0.0%'})
        num_fmt   = workbook.add_format({'num_format': '0.00'})
        
        # Format Dashboard (Summary)
        sheet_sum = writer.sheets['Dashboard']
        last_row = len(df_summary) + 1
        
        # Conditional Format: Verdict (find column index of 'Verdict')
        verdict_col_idx = df_summary.columns.get_loc('Verdict')
        verdict_col_letter = chr(ord('A') + verdict_col_idx)
        sheet_sum.conditional_format(f'{verdict_col_letter}2:{verdict_col_letter}{last_row}', {'type': 'text', 'criteria': 'containing', 'value': 'BUY', 'format': green_fmt})
        sheet_sum.conditional_format(f'{verdict_col_letter}2:{verdict_col_letter}{last_row}', {'type': 'text', 'criteria': 'containing', 'value': 'SELL', 'format': red_fmt})

        # Set column widths/formats for common cols
        for col in df_summary.columns:
            col_idx = df_summary.columns.get_loc(col)
            col_letter = chr(ord('A') + col_idx)
            if col in ['Price', 'Stop_Loss', 'Target_Price']:
                sheet_sum.set_column(f'{col_letter}:{col_letter}', 10, money_fmt)
            elif col == 'Risk_Level':
                sheet_sum.conditional_format(f'{col_letter}2:{col_letter}{last_row}', {'type': 'text', 'criteria': 'containing', 'value': 'High', 'format': red_fmt})
                sheet_sum.set_column(f'{col_letter}:{col_letter}', 10)
            else:
                 sheet_sum.set_column(f'{col_letter}:{col_letter}', 12)
        
        # Format Performance Sheet
        sheet_perf = writer.sheets['Performance']
        sheet_perf.set_column('B:B', 15, pct_fmt)
        sheet_perf.set_column('C:D', 8, num_fmt)

        writer.close()
        print(Fore.GREEN + f"\nReport successfully updated: '{filename}'" + Style.RESET_ALL)

# --- 4. MAIN EXECUTION & ARGPARSE ---

def parse_args():
    parser = argparse.ArgumentParser(description="Stock Valuation and Analysis Tool")
    parser.add_argument(
        '--analyze', 
        nargs='+', 
        help="Analyze and export a list of space-separated tickers (e.g., --analyze AAPL MSFT)."
    )
    parser.add_argument(
        '--refresh', 
        action='store_true', 
        help="Refresh all existing tickers in the database and export."
    )
    parser.add_argument(
        '--config', 
        action='store_true', 
        help="Display the current configuration settings."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config()
    db = DatabaseManager(config['DB_FILENAME'])
    bot = StockAnalyst(config, db)

    # --- Command Line Argument Handling ---
    if args.config:
        print("\n--- Current Configuration (config.json) ---")
        print(json.dumps(config, indent=4))
        return

    if args.refresh:
        bot.refresh_all()
        bot.export_to_excel()
        return

    if args.analyze:
        print(f"\n--- Analyzing Tickers from Command Line ---")
        analyzed_tickers = []
        for t in args.analyze:
            result = bot.analyze_ticker(t)
            if result:
                analyzed_tickers.append(t.strip().upper())
        if analyzed_tickers:
            bot.export_to_excel(analyzed_tickers)
        return

    # --- Interactive Mode ---
    print("\n--- Interactive Stock Valuation Tool v3 ---")
    print(Fore.CYAN + "Commands: Enter tickers (comma-separated), 'refresh', 'config', or 'exit'." + Style.RESET_ALL)
    
    while True:
        user_input = input(Fore.YELLOW + "\n$ " + Style.RESET_ALL)
        command = user_input.lower().strip()
        
        if command in ['exit', 'quit', 'q']:
            print("Exiting application. Database connection closed.")
            break
        
        if command == 'refresh':
            bot.refresh_all()
            bot.export_to_excel()
            continue
        
        if command == 'config':
            print("\n--- Current Configuration (config.json) ---")
            print(json.dumps(config, indent=4))
            continue
            
        if not command:
            continue
            
        # Treat input as comma-separated tickers
        ticker_list = [t.strip().upper() for t in user_input.split(',') if t.strip()]
        
        analyzed_tickers = []
        for t in ticker_list:
            result = bot.analyze_ticker(t)
            if result:
                analyzed_tickers.append(t)
        
        if analyzed_tickers:
            bot.export_to_excel(analyzed_tickers)
        else:
            print(Fore.YELLOW + "No new data to export." + Style.RESET_ALL)
            
    db.close()

if __name__ == "__main__":
    main()