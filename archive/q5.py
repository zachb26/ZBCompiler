# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import os
import time
import json
import sqlite3

# --- 1. CONFIGURATION MANAGEMENT ---
DEFAULT_CONFIG = {
    "RSI_OVERSOLD": 30,
    "RSI_OVERBOUGHT": 70,
    "SMA_SHORT": 50,
    "SMA_LONG": 200,
    "PE_THRESHOLD_LOW": 15,
    "PE_THRESHOLD_HIGH": 50,
    "PROFIT_MARGIN_MIN": 0.10,  # 10%
    "ROE_MIN": 0.15,            # 15%
    "DB_FILENAME": "stocks_data.db",
    "EXCEL_FILENAME": "Stock_Analysis_Report.xlsx"
}

def load_config():
    """Loads config from JSON or creates default if missing."""
    config_file = 'config.json'
    if not os.path.exists(config_file):
        print("Creating default 'config.json'...")
        with open(config_file, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        return DEFAULT_CONFIG
    else:
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}. Using defaults.")
            return DEFAULT_CONFIG

# --- 2. DATABASE MANAGEMENT ---
class DatabaseManager:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        # We store everything as TEXT or REAL to keep it simple for this script
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS analysis (
                Ticker TEXT PRIMARY KEY,
                Price REAL,
                Verdict TEXT,
                RSI REAL,
                Trend TEXT,
                MACD_Hist REAL,
                PE_Ratio REAL,
                Beta REAL,
                Volatility REAL,
                Debt_to_Equity REAL,
                ROE REAL,
                Div_Yield REAL,
                High_52W REAL,
                Notes TEXT,
                Last_Updated TEXT
            )
        ''')
        self.conn.commit()

    def save_record(self, data):
        """Upsert (Update or Insert) a record."""
        # Convert dictionary values to a list in the correct order
        params = (
            data['Ticker'], data['Price'], data['Verdict'], data['RSI'], 
            data['Trend (50/200)'], data['MACD Hist'], data['P/E Ratio'], 
            data['Beta'], data['Volatility'], data['Debt/Equity'], 
            data['ROE'], data['Div Yield'], data['52W High'], 
            data['Notes'], data['Last Updated']
        )
        
        sql = '''
            INSERT INTO analysis 
            (Ticker, Price, Verdict, RSI, Trend, MACD_Hist, PE_Ratio, Beta, Volatility, 
             Debt_to_Equity, ROE, Div_Yield, High_52W, Notes, Last_Updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(Ticker) DO UPDATE SET
                Price=excluded.Price,
                Verdict=excluded.Verdict,
                RSI=excluded.RSI,
                Trend=excluded.Trend,
                MACD_Hist=excluded.MACD_Hist,
                PE_Ratio=excluded.PE_Ratio,
                Beta=excluded.Beta,
                Volatility=excluded.Volatility,
                Debt_to_Equity=excluded.Debt_to_Equity,
                ROE=excluded.ROE,
                Div_Yield=excluded.Div_Yield,
                High_52W=excluded.High_52W,
                Notes=excluded.Notes,
                Last_Updated=excluded.Last_Updated
        '''
        self.cursor.execute(sql, params)
        self.conn.commit()

    def fetch_all(self):
        query = "SELECT * FROM analysis"
        return pd.read_sql_query(query, self.conn)

    def close(self):
        self.conn.close()

# --- 3. CORE ANALYST ---
class StockAnalyst:
    def __init__(self, config, db_manager):
        self.cfg = config
        self.db = db_manager

    def fetch_data(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            info = stock.info
            if hist.empty: return None, None
            return hist, info
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return None, None

    def calculate_indicators(self, df):
        # Moving Averages
        df['SMA_50'] = df['Close'].rolling(window=self.cfg['SMA_SHORT']).mean()
        df['SMA_200'] = df['Close'].rolling(window=self.cfg['SMA_LONG']).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_Line'] = ema12 - ema26
        df['Signal_Line'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD_Line'] - df['Signal_Line']

        # Volatility (Annualized Standard Deviation of Daily Returns)
        df['Returns'] = df['Close'].pct_change()
        # 252 trading days in a year
        volatility = df['Returns'].std() * np.sqrt(252) 
        
        return df, volatility

    def generate_signal(self, row, info, volatility):
        score = 0
        reasons = []

        # Technicals
        if row['RSI'] < self.cfg['RSI_OVERSOLD']:
            score += 2
            reasons.append("Oversold")
        elif row['RSI'] > self.cfg['RSI_OVERBOUGHT']:
            score -= 2
            reasons.append("Overbought")

        if row['Close'] > row['SMA_200']: score += 1
        else: score -= 1

        if row['MACD_Line'] > row['Signal_Line']:
            score += 1
            reasons.append("MACD Bullish")

        # Fundamentals
        pe = info.get('trailingPE')
        if pe and pe < self.cfg['PE_THRESHOLD_LOW']:
            score += 1
            reasons.append("Low P/E")
        
        roe = info.get('returnOnEquity', 0)
        if roe and roe > self.cfg['ROE_MIN']:
            score += 1
            reasons.append("High ROE")

        # Risk check (Penalize high volatility)
        if volatility > 0.50: # >50% annualized volatility
            score -= 1
            reasons.append("High Volatility")

        # Final Verdict
        if score >= 3: verdict = "STRONG BUY"
        elif score >= 1: verdict = "BUY"
        elif score <= -3: verdict = "STRONG SELL"
        elif score <= -1: verdict = "SELL"
        else: verdict = "HOLD"

        return verdict, ", ".join(reasons)

    def analyze_tickers(self, new_tickers):
        clean_tickers = [t.strip().upper() for t in new_tickers if t.strip()]
        print(f"Analyzing {len(clean_tickers)} stocks...")

        for ticker in clean_tickers:
            print(f"Processing {ticker}...")
            hist, info = self.fetch_data(ticker)
            if hist is None: 
                print(f"  > Failed to fetch data for {ticker}")
                continue

            # Run Math
            df, volatility = self.calculate_indicators(hist)
            last_row = df.iloc[-1]

            # Generate Verdict
            verdict, notes = self.generate_signal(last_row, info, volatility)

            # Extract New Fundamentals
            debt_equity = info.get('debtToEquity', 'N/A') # Usually returned as percent (e.g. 150.0) or None
            roe = info.get('returnOnEquity', 'N/A')       # Usually decimal (e.g. 0.15)
            div_yield = info.get('dividendYield', 'N/A')  # Usually decimal (e.g. 0.05)
            beta = info.get('beta', 'N/A')

            # Prepare Data Payload
            data_point = {
                'Ticker': ticker,
                'Price': last_row['Close'],
                'Verdict': verdict,
                'RSI': last_row['RSI'],
                'Trend (50/200)': "Bullish" if last_row['SMA_50'] > last_row['SMA_200'] else "Bearish",
                'MACD Hist': last_row['MACD_Hist'],
                'P/E Ratio': info.get('trailingPE', 'N/A'),
                'Beta': beta,
                'Volatility': volatility,
                'Debt/Equity': debt_equity,
                'ROE': roe,
                'Div Yield': div_yield,
                '52W High': info.get('fiftyTwoWeekHigh', 0),
                'Notes': notes,
                'Last Updated': datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            
            # Save to DB immediately
            self.db.save_record(data_point)

    def export_to_excel(self):
        filename = self.cfg['EXCEL_FILENAME']
        
        # Fetch ALL data from DB
        df = self.db.fetch_all()
        
        if df.empty:
            print("Database is empty. Nothing to export.")
            return

        # Attempt to save
        try:
            writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        except PermissionError:
            print(f"\n[!] PERMISSION ERROR: Could not save to '{filename}'.")
            print("    Please CLOSE the Excel file and press Enter.")
            input("    Press Enter to retry...")
            try:
                writer = pd.ExcelWriter(filename, engine='xlsxwriter')
            except Exception as e:
                print(f"    Still failed ({e}). Data is safe in DB.")
                return

        # Write to sheet
        df.to_excel(writer, sheet_name='Analysis', index=False)
        
        workbook  = writer.book
        worksheet = writer.sheets['Analysis']

        # Formats
        green_fmt = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
        red_fmt = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        money_fmt = workbook.add_format({'num_format': '$#,##0.00'})
        pct_fmt   = workbook.add_format({'num_format': '0.00%'})
        num_fmt   = workbook.add_format({'num_format': '0.00'})

        # Conditional Formatting (Verdict)
        last_row = len(df) + 1
        worksheet.conditional_format(f'C2:C{last_row}', {'type': 'text', 'criteria': 'containing', 'value': 'BUY', 'format': green_fmt})
        worksheet.conditional_format(f'C2:C{last_row}', {'type': 'text', 'criteria': 'containing', 'value': 'SELL', 'format': red_fmt})

        # Column Formatting
        # B: Price, M: 52W High -> Money
        worksheet.set_column('B:B', 10, money_fmt) 
        worksheet.set_column('M:M', 10, money_fmt)
        
        # H: Beta, I: Volatility -> Number/Percent
        worksheet.set_column('H:H', 8, num_fmt)  # Beta
        worksheet.set_column('I:I', 10, pct_fmt) # Volatility
        
        # K: ROE, L: Div Yield -> Percent
        worksheet.set_column('K:L', 10, pct_fmt)

        # Width Adjustments
        worksheet.set_column('A:A', 8)   # Ticker
        worksheet.set_column('C:C', 12)  # Verdict
        worksheet.set_column('N:N', 40)  # Notes

        writer.close()
        print(f"\nReport updated: '{filename}' (Total Stocks: {len(df)})")

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    # Load settings
    config = load_config()
    
    # Initialize DB
    db = DatabaseManager(config['DB_FILENAME'])
    
    # Initialize Analyst
    bot = StockAnalyst(config, db)
    
    print("--- Pro Stock Valuation Tool ---")
    print(f"Database: {config['DB_FILENAME']}")
    print(f"Config:   Loaded from config.json")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nEnter tickers (comma-separated): ")
        
        if user_input.lower() in ['exit', 'quit', 'q', 'done']:
            break
        
        if not user_input.strip():
            continue
            
        ticker_list = user_input.split(',')
        
        # Analyze & Save to DB
        bot.analyze_tickers(ticker_list)
        
        # Export from DB to Excel
        bot.export_to_excel()
    
    db.close()