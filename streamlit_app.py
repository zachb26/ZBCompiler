# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import os
import io
import json
import sqlite3
import pickle
import streamlit as st
from scipy.stats import norm
from plotly import graph_objects as go # Import Plotly for charting

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
    "ROE_MIN": 0.15, #gorp
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

def load_config(profile_name):
    """Loads configuration from a profile-specific JSON file."""
    try:
        # Construct the file name based on the profile
        config_file = f"config_{profile_name.lower()}.json"
        
        # Fallback to default if a custom file is not found
        if not os.path.exists(config_file):
             config_file = "config.json"
             
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Could not load configuration for profile '{profile_name}'. Loading default config.")
        # Load a safe default or raise an error
        return {
             "DB_NAME": "stock_analysis.db", 
             "EXCEL_FILENAME": "Default_Stock_Analysis.xlsx", 
             "PE_THRESHOLD_LOW": 18.0, 
             "ROE_MIN": 0.10,
             "VAR_THRESHOLD_HIGH": 0.35, 
             "RSI_OVERBOUGHT": 70, 
             "RSI_OVERSOLD": 30
        }
    
    try:
        with open(config_path, 'r') as f:
            user_cfg = json.load(f)
            merged = DEFAULT_CONFIG.copy()
            merged.update(user_cfg)
            return merged
    except Exception as e:
        st.error(f"Error loading config: {e}")
        return DEFAULT_CONFIG

def safe_num(value):
    """Returns None if value is None, 0, or 'N/A'."""
    if value is None: return None
    if isinstance(value, str) and value.lower() in ['n/a', 'none', 'nan']: return None
    try:
        f_val = float(value)
        if f_val == 0: return None
        return f_val
    except:
        return None

# --- UI Formatting Helpers ---

def get_color_code(text):
    """Returns a hex color based on the text content (for Styler/HTML background)."""
    if 'BUY' in text or text == 'Low' or text == 'Strong':
        return '#C6EFCE'  # Light Green
    elif 'SELL' in text or text == 'High' or text == 'Weak':
        return '#FFC7CE'  # Light Red
    return '#FFEB9C' # Hold or Medium/Moderate (Light Yellow)

def get_icon(text):
    """Returns an emoji icon based on the text."""
    icon_map = {
        'STRONG BUY': '🚀', 'BUY': '⬆️', 'HOLD': '↔️', 'SELL': '⬇️', 'STRONG SELL': '💥',
        'Low': '✅', 'Medium': '⚠️', 'High': '🚨', 'Strong': '💪', 'Moderate': '🤝', 'Weak': '👎'
    }
    return icon_map.get(text, '❓')

def format_text_html(text, type='verdict'):
    """Formats text for display (only for the single-ticker view)."""
    color = get_color_code(text)
    icon = get_icon(text)
    
    # Use HTML span tag with inline CSS for custom color
    content = f"{icon} {text}"
    return f'<span style="color: {color};">**{content}**</span>'

# New function for Styler logic
def style_verdict_column(s):
    """Returns CSS style for background coloring in st.dataframe."""
    is_buy = s.str.contains('BUY')
    is_sell = s.str.contains('SELL')
    is_hold = s.str.contains('HOLD')
    
    # Create a Series of CSS strings
    styles = pd.Series('', index=s.index)
    styles[is_buy] = 'background-color: #C6EFCE; color: #006100;' # Light Green BG, Dark Green Text
    styles[is_sell] = 'background-color: #FFC7CE; color: #9C0006;' # Light Red BG, Dark Red Text
    styles[is_hold] = 'background-color: #FFEB9C; color: #9C5700;' # Light Yellow BG, Dark Yellow Text
    
    return styles

# --- 2. DATA & CACHE MANAGEMENT ---
class DataManager:
    def __init__(self):
        ensure_directories()

    def get_price_history(self, ticker, period="5y"):
        cache_file = os.path.join('cache', f"{ticker}.pkl")
        
        # Omitted detailed cache logic for brevity in this response, 
        # assumes existing robust implementation.
        try:
            df = yf.Ticker(ticker).history(period=period)
            if df.empty: return None
            if df.index.tz is not None:
                 df.index = df.index.tz_localize(None)
            df.to_pickle(cache_file)
            return df
        except Exception as e:
            st.error(f"Download failed for {ticker}: {e}")
            return None

    def get_info(self, ticker):
        try:
            return yf.Ticker(ticker).info
        except:
            return {}

# --- 3. DATABASE ---
class DatabaseManager:
    def __init__(self, db_name):
        # Use check_same_thread=False for Streamlit's multi-threading environment
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_table()

    # In the DatabaseManager class

    def create_table(self):
        # 1. FORCE DROPS THE OLD, INCORRECTLY STRUCTURED TABLE
        self.cursor.execute("DROP TABLE IF EXISTS analysis") 

        # 2. CREATES THE NEW TABLE WITH THE REQUIRED 20 COLUMNS
        self.cursor.execute('''
            CREATE TABLE analysis (
                Ticker TEXT PRIMARY KEY, Price REAL, Verdict TEXT, Fund_Score INTEGER,
                Risk_Level TEXT, Sector TEXT, VaR_95 REAL, Stop_Loss REAL,
                Target_Price REAL, Rel_Strength_SP500 TEXT, RSI REAL, Trend TEXT,
                PE_Ratio REAL, Sector_PE REAL, ROE REAL, Debt_to_Equity REAL,
                Notes TEXT, Last_Updated TEXT, Fund_Verdict TEXT, Tech_Verdict TEXT
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

    def fetch_record(self, ticker):
        return pd.read_sql_query(f"SELECT * FROM analysis WHERE Ticker='{ticker}'", self.conn)

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
        df = self.dm.get_price_history(ticker, period="5y")
        if df is None or len(df) < 250:
            return None

        # --- Indicator Calculations ---
        df['SMA_50'] = df['Close'].rolling(window=self.cfg['SMA_SHORT']).mean()
        df['SMA_200'] = df['Close'].rolling(window=self.cfg['SMA_LONG']).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Calculate ATR
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()

        # --- Simulation Variables ---
        in_position = False
        entry_price = 0
        stop_loss = 0
        balance = 10000 
        shares = 0
        trades = []

        # Start loop after enough data for SMA200
        for i in range(200, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1] # Access previous day for "Hook" logic
            date = df.index[i]
            
            # --- STRATEGY LOGIC ---
            
            # 1. Trend Filter: Long-term trend must be UP
            uptrend = row['SMA_50'] > row['SMA_200']
            
            # 2. Entry Signal: "Pullback with Hook"
            # RSI is in 'dip' territory (e.g., < 50) BUT has ticked up since yesterday
            rsi_dip = row['RSI'] < 55 # Slightly higher threshold to catch mild pullbacks
            rsi_hook = row['RSI'] > prev_row['RSI'] # Momentum turning up
            buy_signal = uptrend and rsi_dip and rsi_hook

            # 3. Exit Signal: Trend Reversal only (Let winners run)
            # We removed 'row['RSI'] > 75' to avoid selling early in strong trends
            sell_signal = row['SMA_50'] < row['SMA_200']

            if not in_position and buy_signal:
                in_position = True
                entry_price = row['Close']
                shares = balance / entry_price
                balance = 0
                # Set initial Trailing Stop
                stop_loss = entry_price - (row['ATR'] * self.cfg['ATR_MULTIPLIER'])
                trades.append({'Type': 'Buy', 'Date': date, 'Price': entry_price, 'Action': 'Entry'})

            elif in_position:
                # Update Trailing Stop (Only move UP)
                new_stop = row['Close'] - (row['ATR'] * self.cfg['ATR_MULTIPLIER'])
                if new_stop > stop_loss:
                    stop_loss = new_stop

                # CHECK EXITS
                # Priority 1: Stop Loss Hit
                if row['Low'] < stop_loss: 
                    final_price = stop_loss
                    balance = shares * final_price
                    in_position = False
                    ret = (final_price - entry_price)/entry_price
                    trades.append({'Type': 'Sell (Stop)', 'Date': date, 'Price': final_price, 'Return': ret, 'Action': 'Exit'})
                
                # Priority 2: Technical Exit Signal
                elif sell_signal:
                    final_price = row['Close']
                    balance = shares * final_price
                    in_position = False
                    ret = (final_price - entry_price)/entry_price
                    trades.append({'Type': 'Sell (Signal)', 'Date': date, 'Price': final_price, 'Return': ret, 'Action': 'Exit'})
        
        # --- Compilation (Same as before) ---
        final_val = balance if not in_position else shares * df.iloc[-1]['Close']
        years = (df.index[-1] - df.index[200]).days / 365.25
        cagr = ((final_val / 10000) ** (1/years)) - 1 if years > 0 else 0
        total_trades = len([t for t in trades if t['Type'].startswith('Sell')])
        wins = len([t for t in trades if t['Type'].startswith('Sell') and t['Return'] > 0])
        win_rate = wins / total_trades if total_trades > 0 else 0
        
        trade_df = pd.DataFrame(trades)
        
        return {
            "ticker": ticker,
            "trades": total_trades,
            "start_capital": 10000,
            "final_value": final_val,
            "cagr": cagr,
            "win_rate": win_rate,
            "trade_log": trade_df,
            "price_history": df.loc[df.index >= df.index[200], ['Close']]
        }

# --- 5. ANALYST ENGINE ---
class StockAnalyst:
    def __init__(self, config, db_manager, data_manager):
        self.cfg = config
        self.db = db_manager
        self.dm = data_manager
        self.sp500_hist = self.dm.get_price_history("^GSPC", period="1y")

    def calculate_metrics(self, df):
        # [Metric calculations remain the same...]
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
        returns = df['Returns'].dropna()
        mean = np.mean(returns)
        std_dev = np.std(returns)
        var_95 = norm.ppf(1 - self.cfg['VAR_CONFIDENCE'], mean, std_dev) * np.sqrt(252) * 100
        return df, var_95, std_dev # Volatility not strictly needed for UI

    def analyze_ticker(self, ticker, force_refresh=False):
        ticker = ticker.strip().upper()
        
        if not force_refresh:
            last_upd = self.db.get_last_updated(ticker)
            if last_upd and (datetime.datetime.now() - last_upd).total_seconds() < self.cfg['REFRESH_HOURS']*3600:
                st.toast(f"Skipping {ticker} (Data is fresh).")
                return

        st.toast(f"Processing {ticker}...", icon="⏳")
        
        hist = self.dm.get_price_history(ticker, period="2y")
        info = self.dm.get_info(ticker)
        
        if hist is None or info == {}:
            st.error(f"Data Error for {ticker}")
            return

        df, var_95, _ = self.calculate_metrics(hist)
        last_row = df.iloc[-1]
        
        high_low = df['High'] - df['Low']
        ranges = pd.concat([high_low, np.abs(df['High']-df['Close'].shift())], axis=1).max(axis=1)
        atr = ranges.rolling(14).mean().iloc[-1]
        stop_loss = last_row['Close'] - (atr * self.cfg['ATR_MULTIPLIER'])

        # --- SCORING & CONTEXT --- (Logic remains the same)
        tech_score = 0; reasons = []
        if last_row['RSI'] < self.cfg['RSI_OVERSOLD']: tech_score += 2; reasons.append("Oversold")
        elif last_row['RSI'] > self.cfg['RSI_OVERBOUGHT']: tech_score -= 2; reasons.append("Overbought")
        if last_row['MACD_Line'] > last_row['Signal_Line']: tech_score += 1; reasons.append("MACD Bullish")
        if last_row['SMA_50'] > last_row['SMA_200']: tech_score += 1
        else: tech_score -= 1
        
        if tech_score >= 2: tech_verdict = "Bullish"
        elif tech_score <= 0: tech_verdict = "Bearish"
        else: tech_verdict = "Neutral"

        fund_score = 0
        sector = info.get('sector', 'Unknown')
        pe = safe_num(info.get('trailingPE'))
        roe = safe_num(info.get('returnOnEquity'))
        debt_eq = safe_num(info.get('debtToEquity'))
        sector_pe_benchmark = SECTOR_PE_BENCHMARKS.get(sector, 20)
        
        if pe:
            if pe < sector_pe_benchmark * 0.8: fund_score += 2; reasons.append(f"Cheap vs Sector (PE {pe:.1f})")
            elif pe < self.cfg['PE_THRESHOLD_LOW']: fund_score += 1; reasons.append("Low Absolute P/E")
        if roe and roe > self.cfg['ROE_MIN']: fund_score += 1
        if debt_eq and debt_eq < 50: fund_score += 1

        if fund_score >= 2: fund_verdict = "Strong"
        elif fund_score == 1: fund_verdict = "Moderate"
        else: fund_verdict = "Weak"

        risk_level = "Medium"
        var_abs = abs(var_95) 
        if var_abs > 35: risk_level = "High"; risk_score = -1 
        elif var_abs < 15: risk_level = "Low"; risk_score = 0
        else: risk_level = "Medium"; risk_score = 0

        score = tech_score + fund_score + risk_score
        if score >= 4: verdict = "STRONG BUY"
        elif score >= 1: verdict = "BUY"
        elif score <= -3: verdict = "STRONG SELL"
        elif score <= -1: verdict = "SELL"
        else: verdict = "HOLD"

        stock_ret = (last_row['Close'] - df.iloc[0]['Close']) / df.iloc[0]['Close']
        rel_str = "N/A"
        if not self.sp500_hist.empty:
            spy_ret = (self.sp500_hist.iloc[-1]['Close'] - self.sp500_hist.iloc[0]['Close']) / self.sp500_hist.iloc[0]['Close']
            rel_str = f"{(stock_ret - spy_ret)*100:+.1f}%"

        data_point = {
            'Ticker': ticker, 
            'Price': last_row['Close'], 
            'Verdict': verdict, 
            'Fund_Score': fund_score,
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
            'Last_Updated': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            'Fund_Verdict': fund_verdict, 
            'Tech_Verdict': tech_verdict
        }
        
        # Save to database
        self.db.save_record(data_point)

    def export_to_excel(self):
        # [Excel export logic remains the same: writing to in-memory buffer]
        df = self.db.fetch_all()
        if df.empty: return io.BytesIO()

        output_buffer = io.BytesIO()
        writer = pd.ExcelWriter(output_buffer, engine='xlsxwriter')

    def export_to_excel(self):
        df = self.db.fetch_all()
        if df.empty: return io.BytesIO()

        output_buffer = io.BytesIO()
        writer = pd.ExcelWriter(output_buffer, engine='xlsxwriter')

        # --- Define Column Lists for Safe Subsetting ---
        
        # Summary Columns
        cols_summary = ["Ticker", "Price", "Verdict", "Fund_Score", "Tech_Verdict", "Risk_Level", "VaR_95", "Rel_Strength_SP500", "Last_Updated"]
        # Risk Columns
        cols_risk = ['Ticker', 'Risk_Level', 'VaR_95', 'Stop_Loss', 'Debt_to_Equity', 'Sector']
        # Fundamental Columns (where your error is occurring)
        cols_fund = ['Ticker', 'Fund_Verdict', 'PE_Ratio', 'Sector_PE', 'ROE', 'Target_Price', 'Sector']
        # Technical Columns
        cols_tech = ['Ticker', 'Tech_Verdict', 'RSI', 'Trend', 'Notes']
        
        # --- Safe Subsetting using List Comprehension ---
        
        # This checks if the column 'c' exists in the fetched DataFrame (df.columns) before selecting it.
        df_dash = df[[c for c in cols_summary if c in df.columns]].copy()
        df_risk = df[[c for c in cols_risk if c in df.columns]].copy()
        df_fund = df[[c for c in cols_fund if c in df.columns]].copy()
        df_tech = df[[c for c in cols_tech if c in df.columns]].copy()
        
        # ... (rest of the excel writing code continues below)

        # Methodology Data (same as before)
        methodology_data = {
            'Category': ["Overall Verdict", "Overall Verdict", "Overall Verdict", "Overall Verdict", "Overall Verdict",
                         "Fundamental Verdict", "Fundamental Verdict", "Fundamental Verdict", "Technical Verdict", 
                         "Technical Verdict", "Technical Verdict", "Risk Verdict", "Risk Verdict", "Risk Verdict"],
            'Metric / Verdict': ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL", "Strong", "Moderate", "Weak",
                                 "Bullish", "Neutral", "Bearish", "High", "Medium", "Low"],
            'Basis': ["Composite Score"]*5 + ["Fundamental Score"]*3 + ["Technical Score"]*3 + ["Value-at-Risk (VaR) 95%"]*3,
            'Logic / Details': [
                "Composite Score >= 4", "Composite Score >= 1", "Composite Score is 0", "Composite Score <= -1", "Composite Score <= -3",
                f"Fund Score >= 2. (P/E < Sector, P/E < {self.cfg['PE_THRESHOLD_LOW']}, ROE > {self.cfg['ROE_MIN']*100}%, D/E < 50)",
                "Fundamental Score == 1.", "Fundamental Score == 0.",
                f"Tech Score >= 2. (RSI < {self.cfg['RSI_OVERSOLD']}, MACD Bullish, SMA 50 > 200)", "Technical Score == 1.",
                f"Tech Score <= 0. (RSI > {self.cfg['RSI_OVERBOUGHT']}, SMA 50 < 200)",
                "Annualized 95% VaR > 35%.", "Annualized 95% VaR between 15% and 35%.", "Annualized 95% VaR < 15%."
            ]
        }
        df_method = pd.DataFrame(methodology_data)

        df_dash.to_excel(writer, sheet_name='Dashboard', index=False)
        df_risk.to_excel(writer, sheet_name='Risk Analysis', index=False)
        df_fund.to_excel(writer, sheet_name='Fundamentals', index=False)
        df_tech.to_excel(writer, sheet_name='Technicals', index=False)
        df_method.to_excel(writer, sheet_name='Methodology', index=False)
        
        # [Excel formatting code remains the same]
        wb = writer.book
        fmt_money = wb.add_format({'num_format': '$#,##0.00'})
        fmt_num = wb.add_format({'num_format': '0.00'})
        fmt_wrap = wb.add_format({'text_wrap': True, 'valign': 'top'})
        fmt_green = wb.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
        fmt_red = wb.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})

        def auto_format(df, sheet_name):
            ws = writer.sheets[sheet_name]
            last_row = df.shape[0] + 1
            for i, col in enumerate(df.columns):
                width = max(df[col].astype(str).str.len().max(), len(col)) + 2
                if col in ['Notes', 'Logic / Details']: width = 80
                elif col in ['Basis', 'Metric / Verdict']: width = 25
                ws.set_column(i, i, max(width, 10))
                if col in ['Price', 'Stop_Loss', 'Target_Price']: ws.set_column(i, i, width, fmt_money)
                if col in ['VaR_95', 'PE_Ratio', 'Sector_PE', 'ROE', 'Debt_to_Equity', 'RSI', 'Fund_Score']: ws.set_column(i, i, width, fmt_num)
                if col in ['Notes', 'Logic / Details', 'Basis']: ws.set_column(i, i, width, fmt_wrap)
                if sheet_name == 'Dashboard' and col == 'Verdict':
                    range_str = f'{chr(65+i)}2:{chr(65+i)}{last_row}'
                    ws.conditional_format(range_str, {'type':'text', 'criteria':'containing', 'value':'BUY', 'format':fmt_green})
                    ws.conditional_format(range_str, {'type':'text', 'criteria':'containing', 'value':'SELL', 'format':fmt_red})

        auto_format(df_dash, 'Dashboard')
        auto_format(df_risk, 'Risk Analysis')
        auto_format(df_fund, 'Fundamentals')
        auto_format(df_tech, 'Technicals')
        auto_format(df_method, 'Methodology')

        writer.close()
        return output_buffer.getvalue()

# --- 6. STREAMLIT APP ---

# --- App State & Caching ---
@st.cache_resource
def get_data_manager():
    return DataManager()

@st.cache_resource
def get_db_manager(db_name):
    return DatabaseManager(db_name)

# --- Main App UI ---
st.set_page_config(
    page_title="Stock Engine Pro", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Core Components ---
ensure_directories()
profile_files = [f.split('.')[0] for f in os.listdir('configs') if f.endswith('.json')]
if 'default' not in profile_files: profile_files.insert(0, 'default')

st.sidebar.title("⚙️ Configuration")
profile = st.sidebar.selectbox("Select Analysis Profile", profile_files)

config = load_config(profile)
dm = get_data_manager()
db = get_db_manager(config['DB_FILENAME'])
bot = StockAnalyst(config, db, dm)
backtester = Backtester(dm, config)

st.sidebar.markdown(f"**Profile:** `{profile}` | **DB:** `{config['DB_FILENAME']}`")
st.sidebar.divider()

# --- HEADER ---
st.title("💰 Quant Stock Analyst")
st.markdown("---")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs([
    "🚀 Quick Analysis & Dashboard", 
    "📈 Strategy Backtest", 
    "🗃️ Data Explorer & Methodology"
])

with tab1:
    st.header("Stock Analysis Center")
    col_input, col_run = st.columns([3, 1])
    
    with col_input:
        ticker_input = st.text_input("Enter Tickers (e.g., MSFT, AAPL, SPY)", value="MSFT,AAPL", key="ticker_input")
    
    with col_run:
        st.write(" ") # Spacer for alignment
        if st.button("Run Analysis", type="primary", use_container_width=True):
            tickers = [t.strip().upper() for t in ticker_input.split(',')]
            if not any(tickers):
                st.warning("Please enter at least one ticker.")
            else:
                with st.spinner(f"Analyzing {', '.join(tickers)}..."):
                    for t in tickers:
                        bot.analyze_ticker(t)
                st.toast("Analysis complete!", icon="✅")

    st.subheader("Single Ticker Detailed View")
    
    # --- Single Ticker Analysis Card ---
    current_df = db.fetch_all()
    tickers_list = current_df['Ticker'].tolist()
    
    if tickers_list:
        ticker_select = st.selectbox("Select a Ticker for Detailed Analysis", tickers_list)
        # Fetch the record safely.
        record = db.fetch_record(ticker_select).iloc[0]

        # 1. Main Verdict and Price
        col_price, col_verdict, col_risk = st.columns(3)
        
        with col_price:
            st.metric(
                label="Current Price",
                value=f"${record.get('Price', 0):,.2f}",
                delta=f"VaR 95%: {record.get('VaR_95', 'N/A')}%"
            )

        with col_verdict:
            verdict_text = record.get('Verdict', 'N/A')
            verdict_html = format_text_html(verdict_text, type='verdict')
            st.markdown(f"**Overall Verdict**")
            # CRITICAL FIX: Use unsafe_allow_html=True
            st.markdown(f"## {verdict_html}", unsafe_allow_html=True)
            
        with col_risk:
            risk_level = record.get('Risk_Level', 'N/A')
            risk_html = format_text_html(risk_level, type='risk')
            st.markdown(f"**Risk Profile**")
            # CRITICAL FIX: Use unsafe_allow_html=True
            st.markdown(f"## {risk_html}", unsafe_allow_html=True)
        
        st.divider()

        # 2. Sub-Verdicts and Fundamentals
        st.subheader("Component Analysis")
        col_fund, col_tech, col_stats, col_targets = st.columns(4)
        
        # Fundamentals
        with col_fund:
            # FIX: Used .get() to access potentially missing column Fund_Verdict
            st.metric("Fundamental Verdict", record.get('Fund_Verdict', 'N/A - Run Analysis'), 
                      help=f"Score: {record.get('Fund_Score', 'N/A')}")
            st.info(f"P/E Ratio: {record.get('PE_Ratio', 'N/A'):.2f} (Sector: {record.get('Sector_PE', 'N/A'):.0f})")

        # Technicals
        with col_tech:
            st.metric("Technical Verdict", record.get('Tech_Verdict', 'N/A - Run Analysis'))
            st.info(f"RSI: {record.get('RSI', 'N/A'):.2f} | Trend: {record.get('Trend', 'N/A')}")

        # Valuation/Risk Details
        with col_stats:
            st.metric("Relative Strength vs S&P500 (1Y)", record.get('Rel_Strength_SP500', 'N/A'))
            st.warning(f"Stop Loss: ${record.get('Stop_Loss', 0):.2f}")

        # Target/ROE
        with col_targets:
            target_price = record.get('Target_Price')
            st.metric("Target Price (Analyst Consensus)", f"${target_price:,.2f}" if target_price else "N/A")
            roe = record.get('ROE')
            st.success(f"ROE: {roe*100:.1f}%" if roe else "N/A")

        st.markdown(f"**Last Updated:** {record.get('Last_Updated', 'N/A')} | **Sector:** {record.get('Sector', 'N/A')}")
        if record.get('Notes'):
            st.caption(f"**Key Indicators:** {record.get('Notes')}")
    else:
        st.info("No data available yet. Run an analysis above to populate the dashboard.")
    
    st.divider()

    # --- Full Data Table and Export ---
    st.subheader("Full Analysis Data")
    
    if not current_df.empty:
        # 1. Prepare data for Styler (must be a copy)
        # Select base columns, ensuring only existing ones are used
        base_cols = ['Ticker', 'Price', 'Verdict', 'Fund_Score', 'Tech_Verdict', 'Risk_Level', 'VaR_95', 'Last_Updated']
        df_style = current_df[[c for c in base_cols if c in current_df.columns]].copy()
        
        # 2. Add icon to the plain-text 'Verdict' column
        df_style['Verdict'] = df_style['Verdict'].apply(lambda x: f"{get_icon(x)} {x}")
        
        # 3. Create the Styler object and apply the color function
        # The apply function is applied column-wise (axis=0) or row-wise (axis=1).
        # We need to apply the color based on the verdict, which means checking the whole 'Verdict' column.
        styled_df = df_style.style.apply(style_verdict_column, subset=['Verdict'], axis=0)
        
        # 4. Display the Styled DataFrame
        st.dataframe(
            styled_df, # Pass the styled object, not the raw dataframe
            use_container_width=True,
            hide_index=True,
            column_order=['Ticker', 'Price', 'Verdict', 'Fund_Score', 'Tech_Verdict', 'Risk_Level', 'VaR_95', 'Last_Updated'],
            column_config={
                'Price': st.column_config.NumberColumn(format="$%.2f"),
                'VaR_95': st.column_config.NumberColumn(format="%.2f%%"),
                'Verdict': st.column_config.TextColumn(label='Overall Verdict')
            }
        )

    # Download Button
    excel_data = bot.export_to_excel()
    if st.download_button(
        label="⬇️ Download Full Report (Excel)",
        data=excel_data,
        file_name=config['EXCEL_FILENAME'],
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        disabled=current_df.empty,
        help="Downloads a multi-sheet report including Dashboard, Fundamentals, Technicals, Risk, and Methodology."
    ):
        st.toast("Download started.")


with tab2:
    st.header("Strategy Backtesting")
    st.info("Runs a 5-year simulation of the model's technical strategy using moving averages and RSI.")
    
    col_bt_ticker, col_bt_run = st.columns([3, 1])
    with col_bt_ticker:
        backtest_ticker = st.text_input("Enter a Single Ticker for Backtest", "SPY", key="backtest_ticker_input")
    
    backtest_button = col_bt_run.button("Run Backtest", type="primary", use_container_width=True)

    if backtest_button:
        if not backtest_ticker:
            st.warning("Please enter a ticker for backtesting.")
        else:
            with st.spinner(f"Running 5-year simulation for {backtest_ticker}..."):
                results = backtester.run(backtest_ticker.strip().upper())
            
            if results:
                st.success(f"Backtest for {results['ticker']} complete!")
                
                # --- Backtest Results Metrics ---
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Starting Capital", "$10,000.00")
                col2.metric(
                    "Final Value", 
                    f"${results['final_value']:,.2f}", 
                    delta=f"{((results['final_value']/results['start_capital'])-1):.2%}"
                )
                col3.metric("CAGR (Annualized)", f"{results['cagr']:.2%}")
                col4.metric("Win Rate", f"{results['win_rate']:.1%}", f"{results['trades']} Trades")

                st.divider()

                # --- Price History Chart with Trades ---
                st.subheader("Price History and Trade Markers")
                
                fig = go.Figure(data=[go.Scatter(x=results['price_history'].index, y=results['price_history']['Close'], name='Close Price', line=dict(color='blue'))])
                
                if not results['trade_log'].empty:
                    buys = results['trade_log'][results['trade_log']['Action'] == 'Entry']
                    sells = results['trade_log'][results['trade_log']['Action'] == 'Exit']
                    
                    fig.add_trace(go.Scatter(x=buys['Date'], y=buys['Price'], mode='markers', name='Buy', 
                                             marker=dict(symbol='triangle-up', size=10, color='green')))
                    fig.add_trace(go.Scatter(x=sells['Date'], y=sells['Price'], mode='markers', name='Sell', 
                                             marker=dict(symbol='triangle-down', size=10, color='red')))

                fig.update_layout(title=f"{backtest_ticker} Price History with Strategy Trades", 
                                  xaxis_title="Date", 
                                  yaxis_title="Price ($)",
                                  height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("View Detailed Trade Log"):
                    st.dataframe(results['trade_log'], use_container_width=True)


with tab3:
    st.header("Database & Methodology")
    
    # --- Database Maintenance ---
    st.subheader("Data Management")
    col_db_info, col_db_button = st.columns([3, 1])
    
    with col_db_info:
        st.markdown(f"Database: `{config['DB_FILENAME']}` | Records: **{len(db.fetch_all())}**")
    
    with col_db_button:
        if st.button("🔄 Refresh All Tickers", help="Force updates all existing records in the database.", use_container_width=True):
            with st.spinner("Refreshing all tickers... This may take a while."):
                tickers = db.fetch_tickers()
                for t in tickers:
                    bot.analyze_ticker(t, force_refresh=True)
            st.success("Database refresh complete!")

    st.divider()
    
    # --- Methodology ---
    st.subheader("Analysis Methodology")
    st.markdown("The system uses a **Composite Score** derived from Fundamental, Technical, and Risk factors to determine the final verdict.")
    
    # Re-generate methodology data for direct display (since it's not stored in the DB)
    methodology_data = {
        'Category': ["Overall Verdict", "Overall Verdict", "Overall Verdict", "Overall Verdict", "Overall Verdict",
                     "Fundamental Verdict", "Fundamental Verdict", "Fundamental Verdict", "Technical Verdict", 
                     "Technical Verdict", "Technical Verdict", "Risk Verdict", "Risk Verdict", "Risk Verdict"],
        'Metric / Verdict': ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL", "Strong", "Moderate", "Weak",
                             "Bullish", "Neutral", "Bearish", "High", "Medium", "Low"],
        'Basis': ["Composite Score"]*5 + ["Fundamental Score"]*3 + ["Technical Score"]*3 + ["Value-at-Risk (VaR) 95%"]*3,
        'Logic / Details': [
            "Composite Score >= 4", "Composite Score >= 1", "Composite Score is 0", "Composite Score <= -1", "Composite Score <= -3",
            f"Fundamental Score >= 2. (Gains from: P/E < Sector, P/E < {config['PE_THRESHOLD_LOW']}, ROE > {config['ROE_MIN']*100}%, D/E < 50)",
            "Fundamental Score == 1. (Meets one criterion)", "Fundamental Score == 0. (Meets no criteria)",
            f"Technical Score >= 2. (Gains from: RSI < {config['RSI_OVERSOLD']}, MACD Bullish, SMA 50 > SMA 200)", "Technical Score == 1.",
            f"Technical Score <= 0. (Loses from: RSI > {config['RSI_OVERBOUGHT']}, SMA 50 < SMA 200)",
            "Annualized 95% VaR > 35%.", "Annualized 95% VaR between 15% and 35%.", "Annualized 95% VaR < 15%."
        ]
    }
    df_method = pd.DataFrame(methodology_data)
    
    st.dataframe(
        df_method,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Logic / Details': st.column_config.TextColumn(width='large')
        }
    )













