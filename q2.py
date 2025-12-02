# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import os

class StockAnalyst:
    def __init__(self):
        # We no longer require tickers on init. 
        # We initialize an empty list to store cumulative results.
        self.results = []

    def fetch_data(self, ticker):
        """Fetches 1 year of historical data and current info."""
        try:
            stock = yf.Ticker(ticker)
            # mute warnings from yfinance if needed
            hist = stock.history(period="1y")
            info = stock.info

            if hist.empty:
                return None, None
            return hist, info
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
            return None, None

    def calculate_indicators(self, df):
        """Calculates RSI, MACD, Bollinger Bands, and Moving Averages."""
        # 1. Moving Averages
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()

        # 2. RSI (Relative Strength Index) - 14 Days
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 3. MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD_Line'] = ema12 - ema26
        df['Signal_Line'] = df['MACD_Line'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD_Line'] - df['Signal_Line']

        # 4. Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
        df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])

        return df

    def generate_signal(self, row, info):
        """Generates a Buy/Sell/Hold verdict based on logic."""
        score = 0
        reasons = []

        # Technical Signals
        if row['RSI'] < 30:
            score += 2
            reasons.append("Oversold (RSI < 30)")
        elif row['RSI'] > 70:
            score -= 2
            reasons.append("Overbought (RSI > 70)")

        if row['Close'] > row['SMA_200']:
            score += 1
        else:
            score -= 1

        if row['MACD_Line'] > row['Signal_Line']:
            score += 1
            reasons.append("MACD Bullish Crossover")

        # Fundamental Signals
        pe = info.get('trailingPE')
        if pe and pe < 15:
            score += 1
            reasons.append("Low P/E")

        # Final Verdict
        if score >= 3: verdict = "STRONG BUY"
        elif score >= 1: verdict = "BUY"
        elif score <= -3: verdict = "STRONG SELL"
        elif score <= -1: verdict = "SELL"
        else: verdict = "HOLD"

        return verdict, ", ".join(reasons)

    def analyze_tickers(self, new_tickers):
        """Analyzes a list of new tickers and appends to results."""
        # Clean inputs
        clean_tickers = [t.strip().upper() for t in new_tickers if t.strip()]
        
        print(f"Analyzing {len(clean_tickers)} stocks...")

        for ticker in clean_tickers:
            # Check if we already analyzed this ticker in this session to avoid duplicates
            if any(d['Ticker'] == ticker for d in self.results):
                print(f"Skipping {ticker} (already in list)...")
                continue

            print(f"Processing {ticker}...")
            hist, info = self.fetch_data(ticker)
            if hist is None: 
                print(f"Could not fetch data for {ticker}")
                continue

            # Run Math
            df = self.calculate_indicators(hist)
            last_row = df.iloc[-1]

            # Generate Verdict
            verdict, notes = self.generate_signal(last_row, info)

            # Store Data
            data_point = {
                'Ticker': ticker,
                'Price': last_row['Close'],
                'Verdict': verdict,
                'RSI': last_row['RSI'],
                'Trend (50/200)': "Bullish" if last_row['SMA_50'] > last_row['SMA_200'] else "Bearish",
                'MACD Hist': last_row['MACD_Hist'],
                'P/E Ratio': info.get('trailingPE', 'N/A'),
                '52W High': info.get('fiftyTwoWeekHigh', 0),
                'Notes': notes,
                'Last Updated': datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            self.results.append(data_point)

    def export_to_excel(self, filename="Stock_Analysis_Report.xlsx"):
        if not self.results:
            print("No data to save.")
            return

        df = pd.DataFrame(self.results)

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(filename, engine='xlsxwriter')
        df.to_excel(writer, sheet_name='Analysis', index=False)

        workbook  = writer.book
        worksheet = writer.sheets['Analysis']

        # Add some formats.
        green_fmt = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
        red_fmt = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})
        money_fmt = workbook.add_format({'num_format': '$#,##0.00'})

        # --- BUG FIX: Dynamic Range ---
        # Calculate the last row number (DataFrame length + 1 header row)
        last_row = len(df) + 1
        # Create the dynamic range string, e.g., "C2:C105"
        verdict_range = f'C2:C{last_row}'

        # Apply conditional formatting to the Verdict column (C)
        worksheet.conditional_format(verdict_range, {'type': 'text',
                                                 'criteria': 'containing',
                                                 'value': 'BUY',
                                                 'format': green_fmt})
        worksheet.conditional_format(verdict_range, {'type': 'text',
                                                 'criteria': 'containing',
                                                 'value': 'SELL',
                                                 'format': red_fmt})

        # Format Price columns
        worksheet.set_column('B:B', 10, money_fmt) # Price
        worksheet.set_column('H:H', 10, money_fmt) # 52W High

        # Auto-adjust column widths (approximation)
        worksheet.set_column('A:A', 8)  # Ticker
        worksheet.set_column('C:C', 15) # Verdict
        worksheet.set_column('I:I', 40) # Notes

        try:
            writer.close()
            print(f"\nSuccess! Report updated: '{filename}' (Total Stocks: {len(self.results)})")
        except PermissionError:
            print(f"\nERROR: Close '{filename}' before running the script so I can save the file!")

if __name__ == "__main__":
    # Initialize the analyst ONCE (outside the loop)
    bot = StockAnalyst()
    
    print("--- Stock Valuation Tool ---")
    print("Type 'exit' or 'q' to quit.")

    while True:
        user_input = input("\nEnter tickers (comma-separated): ")
        
        # Check for exit condition
        if user_input.lower() in ['exit', 'quit', 'q', 'done']:
            print("Exiting...")
            break
        
        # Parse tickers
        if not user_input.strip():
            print("Please enter a valid ticker.")
            continue
            
        ticker_list = user_input.split(',')
        
        # Analyze new tickers (this appends to the existing bot.results list)
        bot.analyze_tickers(ticker_list)
        
        # Export the FULL list (old + new) to the Excel file
        bot.export_to_excel()