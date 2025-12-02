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
import pickle
from colorama import init, Fore, Style
from scipy.stats import norm

# Initialize colorama
init(autoreset=True)

# --- UI CONSTANTS ---

# Fixed: Using a raw string (r"""") to prevent backslash escape sequence warnings.
COMPILER_ART = r"""
 /$$$$$$$  /$$$$$$$   /$$$$$$  /$$$$$$$$  /$$$$$$  /$$  /$$  /$$$$$$  /$$$$$$$$  /$$$$$$  /$$  /$$
| $$__  $$| $$__  $$ /$$__  $$|_____ $$  |_  $$_/| $$$ | $$ /$$__  $$|__  $$__/ /$$__  $$| $$$ | $$
| $$  \ $$| $$  \ $$| $$  \ $$     /$$/    | $$  | $$$$| $$| $$  \__/    | $$  | $$  \ $$| $$$$| $$
| $$$$$$$ | $$$$$$$/| $$$$$$$$    /$$/     | $$  | $$ $$ $$| $$ /$$$$    | $$  | $$  | $$| $$ $$ $$
| $$__  $$| $$__  $$| $$__  $$   /$$/      | $$  | $$  $$$$| $$|_  $$    | $$  | $$  | $$| $$  $$$$
| $$  \ $$| $$  \ $$| $$  | $$  /$$/       | $$  | $$\  $$$| $$  \ $$    | $$  | $$  | $$| $$\  $$$
| $$$$$$$/| $$  | $$| $$  | $$ /$$$$$$$$  /$$$$$$| $$ \  $$|  $$$$$$/    | $$  |  $$$$$$/| $$ \  $$
|_______/ |__/  |__/|__/  |__/|________/ |______/|__/  \__/ \______/     |__/   \______/ |__/  \__/
"""

COMMANDS_HELP = f"""
{Fore.CYAN}--- COMMAND GUIDE ---{Style.RESET_ALL}
{Fore.GREEN}/analyze TICKER1,TICKER2{Style.RESET_ALL} : Analyze new stocks or update existing records.
{Fore.GREEN}/refresh{Style.RESET_ALL}            : Refresh all existing tickers in the database (uses smart caching).
{Fore.GREEN}/backtest TICKER{Style.RESET_ALL}      : Run a 5-year simulation of the model's technical strategy.
{Fore.GREEN}/profile NAME{Style.RESET_ALL}         : Switch the analysis configuration profile (e.g., conservative, aggressive).
{Fore.GREEN}/exit{Style.RESET_ALL}               : Close the application and database connection.
"""

# --- CONSTANTS & BENCHMARKS ---
SECTOR_PE_BENCHMARKS = {
    "Technology": 35, "Healthcare": 25, "Financial Services": 15,
    "Energy": 12, "Consumer Cyclical": 20, "Industrials": 20,
    "Utilities": 18, "Consumer Defensive": 22, "Real Estate": 35,
    "Communication Services": 20, "Basic Materials": 15
}

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
    "VAR_CONFIDENCE": 0.95,
    "DB_FILENAME": "stocks_data.db",
    "EXCEL_FILENAME": "Stock_Analysis_Report.xlsx",
    "SUMMARY_OUTPUT_COLS": ["Ticker", "Price", "Verdict", "Fund_Score", "Risk_Level", "Sector", "VaR_95", "Rel_Strength_SP500"]
}

# --- 1. UTILITIES & CONFIG ---

def ensure_directories():
    if not os.path.exists('configs'): os.makedirs('configs')
    if not os.path.exists('cache'): os.makedirs('cache')

def load_config(profile_name="default"):
    """Loads a specific config profile."""
    ensure_directories()
    config_path = os.path.join('configs', f'{profile_name}.json')
    
    if not os.path.exists(config_path):
        if profile_name == 'default':
            with open(config_path, 'w') as f:
                json.dump(DEFAULT_CONFIG, f, indent=4)
            return DEFAULT_CONFIG
        else:
            print(Fore.RED + f"Profile '{profile_name}' not found. Loading default." + Style.RESET_ALL)
            return load_config("default")
    
    try:
        with open(config_path, 'r') as f:
            user_cfg = json.load(f)
            merged = DEFAULT_CONFIG.copy()
            merged.update(user_cfg)
            return merged
    except Exception as e:
        print(Fore.RED + f"Error loading config: {e}" + Style.RESET_ALL)
        return DEFAULT_CONFIG

def safe_num(value):
    """Returns None if value is None, 0, or 'N/A'. Used for strict data validation."""
    if value is None: return None
    if isinstance(value, str) and value.lower() in ['n/a', 'none', 'nan']: return None
    try:
        f_val = float(value)
        if f_val == 0: return None
        return f_val
    except:
        return None

# --- 2. DATA & CACHE MANAGEMENT ---
class DataManager:
    def __init__(self):
        ensure_directories()

    def get_price_history(self, ticker, period="5y"):
        """Smart fetching: loads from pickle cache, then updates only new data."""
        cache_file = os.path.join('cache', f"{ticker}.pkl")
        
        # 1. Try Load Cache
        if os.path.exists(cache_file):
            try:
                cached_df = pd.read_pickle(cache_file)
                # Ensure the index is timezone naive for consistent comparison
                if cached_df.index.tz is not None:
                     cached_df.index = cached_df.index.tz_localize(None)
                     
                last_date = cached_df.index[-1]
                today = datetime.datetime.now().date()
                
                # If cache is fresh (less than 1 day old), return it
                if (today - last_date.date()).days < 1:
                    return cached_df
                
                # Fetch only missing data
                start_date = (last_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                new_data = yf.download(ticker, start=start_date, progress=False)
                
                if not new_data.empty:
                    if isinstance(new_data.columns, pd.MultiIndex):
                        new_data.columns = new_data.columns.get_level_values(0)
                        
                    combined_df = pd.concat([cached_df, new_data])
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    combined_df.to_pickle(cache_file)
                    return combined_df
                return cached_df
            except Exception as e:
                print(Fore.YELLOW + f"Cache corrupted for {ticker}, redownloading. ({e})" + Style.RESET_ALL)

        # 2. Fresh Download
        try:
            df = yf.Ticker(ticker).history(period=period)
            if df.empty: return None
            df.to_pickle(cache_file)
            return df
        except Exception as e:
            print(Fore.RED + f"Download failed for {ticker}: {e}" + Style.RESET_ALL)
            return None

    def get_info(self, ticker):
        """Fetches fundamental info (no caching for now to keep fundamentals fresh)."""
        try:
            return yf.Ticker(ticker).info
        except:
            return {}

# --- 3. DATABASE ---
class DatabaseManager:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis (
                Ticker TEXT PRIMARY KEY,
                Price REAL,
                Verdict TEXT,
                Fund_Score INTEGER,
                Risk_Level TEXT,
                Sector TEXT,
                VaR_95 REAL,
                Stop_Loss REAL,
                Target_Price REAL,
                Rel_Strength_SP500 TEXT,
                RSI REAL,
                Trend TEXT,
                PE_Ratio REAL,
                Sector_PE REAL,
                ROE REAL,
                Debt_to_Equity REAL,
                Notes TEXT,
                Last_Updated TEXT
            )
        ''')
        self.conn.commit()

    def save_record(self, data):
        keys = list(data.keys())
        placeholders = ', '.join(['?'] * len(keys))
        columns = ', '.join(keys)
        sql = f'''INSERT INTO analysis ({columns}) VALUES ({placeholders})
                  ON CONFLICT(Ticker) DO UPDATE SET {', '.join([f"{k}=excluded.{k}" for k in keys])}'''
        self.cursor.execute(sql, list(data.values()))
        self.conn.commit()

    def fetch_all(self):
        return pd.read_sql_query("SELECT * FROM analysis", self.conn)

    def fetch_tickers(self):
        self.cursor.execute("SELECT Ticker FROM analysis")
        return [row[0] for row in self.cursor.fetchall()]

    def get_last_updated(self, ticker):
        self.cursor.execute("SELECT Last_Updated FROM analysis WHERE Ticker=?", (ticker,))
        res = self.cursor.fetchone()
        return datetime.datetime.strptime(res[0], "%Y-%m-%d %H:%M") if res else None
    
    def close(self):
        self.conn.close()

# --- 4. BACKTESTING ENGINE ---
class Backtester:
    def __init__(self, data_manager, config):
        self.dm = data_manager
        self.cfg = config

    def run(self, ticker):
        print(Fore.CYAN + f"Running 5-Year Backtest for {ticker}..." + Style.RESET_ALL)
        df = self.dm.get_price_history(ticker, period="5y")
        if df is None or len(df) < 250:
            print(Fore.RED + "Not enough history for backtest." + Style.RESET_ALL)
            return

        # Pre-calculate Indicators
        df['SMA_50'] = df['Close'].rolling(window=self.cfg['SMA_SHORT']).mean()
        df['SMA_200'] = df['Close'].rolling(window=self.cfg['SMA_LONG']).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # ATR for trailing stop
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()

        # Simulation Variables
        in_position = False
        entry_price = 0
        stop_loss = 0
        balance = 10000 # Starting Cash
        shares = 0
        trades = []

        # Iterate (Start after 200 days for SMA)
        for i in range(200, len(df)):
            row = df.iloc[i]
            date = df.index[i]
            
            # Logic: Simplified Technical Strategy for Backtest
            buy_signal = (row['SMA_50'] > row['SMA_200']) and (row['RSI'] < 45) # Pullback in uptrend
            sell_signal = (row['RSI'] > 75) or (row['SMA_50'] < row['SMA_200'])

            if not in_position and buy_signal:
                in_position = True
                entry_price = row['Close']
                shares = balance / entry_price
                balance = 0
                stop_loss = entry_price - (row['ATR'] * self.cfg['ATR_MULTIPLIER'])
                trades.append({'Type': 'Buy', 'Date': date, 'Price': entry_price})

            elif in_position:
                # Update Trailing Stop if price moved up
                new_stop = row['Close'] - (row['ATR'] * self.cfg['ATR_MULTIPLIER'])
                if new_stop > stop_loss:
                    stop_loss = new_stop

                # Check Exits
                if row['Low'] < stop_loss: # Stopped Out
                    final_price = stop_loss
                    balance = shares * final_price
                    in_position = False
                    trades.append({'Type': 'Sell (Stop)', 'Date': date, 'Price': final_price, 'Return': (final_price - entry_price)/entry_price})
                elif sell_signal: # Take Profit / Signal Exit
                    final_price = row['Close']
                    balance = shares * final_price
                    in_position = False
                    trades.append({'Type': 'Sell (Signal)', 'Date': date, 'Price': final_price, 'Return': (final_price - entry_price)/entry_price})

        # Final Outcome
        final_val = balance if not in_position else shares * df.iloc[-1]['Close']
        
        # Calculate Total Return (CAGR is more appropriate for multi-year)
        years = (df.index[-1] - df.index[200]).days / 365.25
        cagr = ((final_val / 10000) ** (1/years)) - 1 if years > 0 else 0
        
        total_trades = len([t for t in trades if t['Type'].startswith('Sell')])
        wins = len([t for t in trades if t['Type'].startswith('Sell') and t['Return'] > 0])
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        print(f"\n{Fore.GREEN}--- BACKTEST RESULTS: {ticker} ---{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}Trades Executed:{Style.RESET_ALL} {total_trades}")
        print(f"{Fore.MAGENTA}Starting Capital:{Style.RESET_ALL} $10,000.00")
        print(f"{Fore.MAGENTA}Final Value:{Style.RESET_ALL}    ${final_val:,.2f}")
        print(f"{Fore.MAGENTA}CAGR (Annualized):{Style.RESET_ALL} {cagr:.2%}")
        print(f"{Fore.MAGENTA}Win Rate:{Style.RESET_ALL}         {win_rate:.1%}")

# --- 5. ANALYST ENGINE ---
class StockAnalyst:
    def __init__(self, config, db_manager, data_manager):
        self.cfg = config
        self.db = db_manager
        self.dm = data_manager
        self.sp500_hist = self.dm.get_price_history("^GSPC", period="1y")

    def calculate_metrics(self, df):
        # 1. Technicals (SMA, RSI, MACD) - Calculation logic remains the same
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

        # 2. Advanced Risk (VaR & Volatility)
        df['Returns'] = df['Close'].pct_change()
        returns = df['Returns'].dropna()
        
        mean = np.mean(returns)
        std_dev = np.std(returns)
        # VaR (Value at Risk) is calculated as the expected worst loss
        var_95 = norm.ppf(1 - self.cfg['VAR_CONFIDENCE'], mean, std_dev) * np.sqrt(252) * 100
        
        volatility = std_dev * np.sqrt(252)

        return df, var_95, volatility

    def analyze_ticker(self, ticker, force_refresh=False):
        ticker = ticker.strip().upper()
        
        if not force_refresh:
            last_upd = self.db.get_last_updated(ticker)
            if last_upd and (datetime.datetime.now() - last_upd).total_seconds() < self.cfg['REFRESH_HOURS']*3600:
                print(f"  > Skipping {ticker} (Data is fresh).")
                return

        print(Fore.GREEN + f"Processing {ticker}..." + Style.RESET_ALL)
        
        hist = self.dm.get_price_history(ticker, period="2y")
        info = self.dm.get_info(ticker)
        
        if hist is None or info == {}:
            print(Fore.RED + f"  > Data Error for {ticker}" + Style.RESET_ALL)
            return

        df, var_95, volatility = self.calculate_metrics(hist)
        last_row = df.iloc[-1]
        
        # ATR for Stop Loss
        high_low = df['High'] - df['Low']
        ranges = pd.concat([high_low, np.abs(df['High']-df['Close'].shift())], axis=1).max(axis=1)
        atr = ranges.rolling(14).mean().iloc[-1]
        stop_loss = last_row['Close'] - (atr * self.cfg['ATR_MULTIPLIER'])

        # --- SCORING & CONTEXT ---
        score = 0
        reasons = []
        
        # 1. Technicals (Max 5 pts)
        if last_row['RSI'] < self.cfg['RSI_OVERSOLD']: score += 2; reasons.append("Oversold")
        elif last_row['RSI'] > self.cfg['RSI_OVERBOUGHT']: score -= 2; reasons.append("Overbought")
        
        if last_row['MACD_Line'] > last_row['Signal_Line']: score += 1; reasons.append("MACD Bullish")
        if last_row['SMA_50'] > last_row['SMA_200']: score += 1
        else: score -= 1

        # 2. Fundamentals with Sector Context (Max 5 pts)
        sector = info.get('sector', 'Unknown')
        pe = safe_num(info.get('trailingPE'))
        roe = safe_num(info.get('returnOnEquity'))
        debt_eq = safe_num(info.get('debtToEquity'))

        sector_pe_benchmark = SECTOR_PE_BENCHMARKS.get(sector, 20)
        
        if pe:
            if pe < sector_pe_benchmark * 0.8:
                score += 2
                reasons.append(f"Cheap vs Sector (PE {pe:.1f})")
            elif pe < self.cfg['PE_THRESHOLD_LOW']:
                score += 1
                reasons.append("Low Absolute P/E")
        
        if roe and roe > self.cfg['ROE_MIN']: score += 1
        if debt_eq and debt_eq < 50: score += 1

        # 3. Risk Assessment (VaR)
        var_abs = abs(var_95) 
        if var_abs > 35:
            risk_level = "High"
            score -= 1 
        elif var_abs < 15:
            risk_level = "Low"
        else: 
            risk_level = "Medium"

        # Final Verdict based on composite score
        if score >= 4: verdict = "STRONG BUY"
        elif score >= 1: verdict = "BUY"
        elif score <= -3: verdict = "STRONG SELL"
        elif score <= -1: verdict = "SELL"
        else: verdict = "HOLD"

        # Relative Strength
        stock_ret = (last_row['Close'] - df.iloc[0]['Close']) / df.iloc[0]['Close']
        rel_str = "N/A"
        if not self.sp500_hist.empty:
            spy_ret = (self.sp500_hist.iloc[-1]['Close'] - self.sp500_hist.iloc[0]['Close']) / self.sp500_hist.iloc[0]['Close']
            rel_str = f"{(stock_ret - spy_ret)*100:+.1f}%"

        # Save
        data_point = {
            'Ticker': ticker,
            'Price': last_row['Close'],
            'Verdict': verdict,
            'Fund_Score': score,
            'Risk_Level': risk_level,
            'Sector': sector,
            'VaR_95': round(var_95, 2),
            'Stop_Loss': stop_loss,
            'Target_Price': safe_num(info.get('targetMeanPrice')),
            'Rel_Strength_SP500': rel_str,
            'RSI': last_row['RSI'],
            'Trend': "Bullish" if last_row['SMA_50'] > last_row['SMA_200'] else "Bearish",
            'PE_Ratio': pe,
            'Sector_PE': sector_pe_benchmark,
            'ROE': roe,
            'Debt_to_Equity': debt_eq,
            'Notes': ", ".join(reasons),
            'Last_Updated': datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        self.db.save_record(data_point)
        return ticker

    def export_to_excel(self):
        filename = self.cfg['EXCEL_FILENAME']
        df = self.db.fetch_all()
        if df.empty: return

        # Split Data
        cols_summary = self.cfg['SUMMARY_OUTPUT_COLS']
        present_cols = [c for c in cols_summary if c in df.columns]
        
        df_dash = df[present_cols].copy()
        df_risk = df[['Ticker', 'Risk_Level', 'VaR_95', 'Stop_Loss', 'Debt_to_Equity', 'Sector']].copy()
        df_fund = df[['Ticker', 'PE_Ratio', 'Sector_PE', 'ROE', 'Target_Price', 'Sector']].copy()
        df_tech = df[['Ticker', 'RSI', 'Trend', 'Notes']].copy()

        try:
            writer = pd.ExcelWriter(filename, engine='xlsxwriter')
            df_dash.to_excel(writer, sheet_name='Dashboard', index=False)
            df_risk.to_excel(writer, sheet_name='Risk Analysis', index=False)
            df_fund.to_excel(writer, sheet_name='Fundamentals', index=False)
            df_tech.to_excel(writer, sheet_name='Technicals', index=False)
            
            # Formatter Helper
            wb = writer.book
            fmt_money = wb.add_format({'num_format': '$#,##0.00'})
            fmt_num = wb.add_format({'num_format': '0.00'})
            fmt_pct = wb.add_format({'num_format': '0.0%'})
            fmt_green = wb.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
            fmt_red = wb.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})

            def auto_format(df, sheet_name):
                ws = writer.sheets[sheet_name]
                last_row = df.shape[0] + 1 # Calculate the last row number for dynamic range
                
                for i, col in enumerate(df.columns):
                    width = max(df[col].astype(str).str.len().max(), len(col)) + 2
                    if col == 'Notes':
                        width = 60
                    
                    ws.set_column(i, i, max(width, 10))
                    
                    # Apply specific formats
                    if col in ['Price', 'Stop_Loss', 'Target_Price']: ws.set_column(i, i, width, fmt_money)
                    if col in ['VaR_95']: ws.set_column(i, i, width, fmt_num) 
                    if col in ['PE_Ratio', 'Sector_PE', 'ROE', 'Debt_to_Equity', 'RSI', 'Fund_Score']: ws.set_column(i, i, width, fmt_num)
                    
                    # FIX: Use the calculated last_row variable for dynamic range
                    if sheet_name == 'Dashboard' and col == 'Verdict':
                        range_str = f'{chr(65+i)}2:{chr(65+i)}{last_row}'
                        ws.conditional_format(range_str, {'type':'text', 'criteria':'containing', 'value':'BUY', 'format':fmt_green})
                        ws.conditional_format(range_str, {'type':'text', 'criteria':'containing', 'value':'SELL', 'format':fmt_red})

            auto_format(df_dash, 'Dashboard')
            auto_format(df_risk, 'Risk Analysis')
            auto_format(df_fund, 'Fundamentals')
            auto_format(df_tech, 'Technicals')

            writer.close()
            print(Fore.GREEN + f"Report Updated: {filename}" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Export Failed: {e}" + Style.RESET_ALL)

# --- 6. MAIN ---
def main():
    profile = "default"
    
    # Initial Load
    config = load_config(profile)
    dm = DataManager()
    db = DatabaseManager(config['DB_FILENAME'])
    bot = StockAnalyst(config, db, dm)
    backtester = Backtester(dm, config)

    # --- UI Display ---
    print(Fore.MAGENTA + COMPILER_ART + Style.RESET_ALL)
    print(Fore.BLUE + "  Stock Engine Pro | Advanced Quantitative Analysis Tool" + Style.RESET_ALL)
    print("  --------------------------------------------------")
    print(Fore.YELLOW + f"  Current Profile: {profile} | Database: {config['DB_FILENAME']} | Cache: Enabled" + Style.RESET_ALL)
    print(COMMANDS_HELP)

    while True:
        try:
            user_input = input(Fore.YELLOW + "\n>> " + Style.RESET_ALL).strip()
            if not user_input: continue

            if user_input.startswith('/'):
                parts = user_input.split()
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else None

                if cmd in ['/quit', '/exit']: break
                
                elif cmd == '/refresh':
                    tickers = db.fetch_tickers()
                    print(f"Refreshing {len(tickers)} tickers...")
                    for t in tickers: bot.analyze_ticker(t, force_refresh=True)
                    bot.export_to_excel()

                elif cmd == '/backtest':
                    if arg: backtester.run(arg)
                    else: print(Fore.RED + "Usage: /backtest TICKER" + Style.RESET_ALL)

                elif cmd == '/profile':
                    if arg:
                        profile = arg
                        config = load_config(profile)
                        bot = StockAnalyst(config, db, dm)
                        print(Fore.GREEN + f"Switched to profile: {profile}" + Style.RESET_ALL)
                    else:
                        print(Fore.RED + "Usage: /profile NAME" + Style.RESET_ALL)
                else:
                    print(Fore.RED + "Unknown command. Type /analyze or check command guide." + Style.RESET_ALL)
            else:
                # Ticker Analysis
                tickers = [t.strip().upper() for t in user_input.split(',')]
                for t in tickers:
                    bot.analyze_ticker(t)
                bot.export_to_excel()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(Fore.RED + f"An unexpected error occurred: {e}" + Style.RESET_ALL)
            time.sleep(1)

    db.close()
    print(Fore.MAGENTA + "Database connection closed. Exiting." + Style.RESET_ALL)

if __name__ == "__main__":
    main()