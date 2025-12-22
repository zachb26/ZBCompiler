# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import os
import io
import sqlite3
import hashlib
import time
import streamlit as st
from scipy.stats import norm
import plotly.graph_objects as go

# --- CONFIGURATION & BENCHMARKS ---
DB_FILENAME = "stock_pro_v5.db" # Versioned to ensure clean slate

# Sector Benchmarks for "Relative Valuation"
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
    "Basic Materials": {"PE": 15, "PS": 1.5, "PB": 2.0, "EV_EBITDA": 8}
}
DEFAULT_BENCHMARKS = {"PE": 20, "PS": 3.0, "PB": 3.0, "EV_EBITDA": 12}

# --- 1. UTILITIES ---
def safe_num(value):
    """Safely converts value to float, handling None/Strings."""
    if value is None: return None
    if isinstance(value, (int, float)): return float(value)
    if isinstance(value, str):
        if value.lower() in ['n/a', 'none', 'nan', 'inf']: return None
        try: return float(value.replace('%', '').replace(',', '').strip())
        except: return None
    return None

def get_color(verdict):
    """Returns color code for UI badges."""
    if 'STRONG BUY' in verdict: return 'green'
    if 'BUY' in verdict: return 'green'
    if 'STRONG SELL' in verdict: return 'red'
    if 'SELL' in verdict: return 'red'
    return 'gray' # Hold/Neutral

# --- 2. DATABASE MANAGER (Auth & Persistence) ---
class DatabaseManager:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        # Users Table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT
            )
        ''')
        # Watchlist Table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS watchlist (
                username TEXT,
                ticker TEXT,
                added_date TEXT,
                PRIMARY KEY (username, ticker)
            )
        ''')
        # Analysis Cache Table (Expanded for Method Comparison)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS analysis (
                Ticker TEXT PRIMARY KEY, 
                Price REAL, 
                
                -- The 4 Separate Verdicts
                Verdict_Overall TEXT,
                Verdict_Technical TEXT,
                Verdict_Fundamental TEXT,
                Verdict_Valuation TEXT,
                
                -- Scores
                Score_Tech INTEGER,
                Score_Fund INTEGER,
                Score_Val INTEGER,
                
                -- Key Metrics
                Sector TEXT,
                PE_Ratio REAL, Forward_PE REAL, PEG_Ratio REAL,
                PS_Ratio REAL, PB_Ratio REAL, EV_EBITDA REAL,
                Graham_Number REAL, Intrinsic_Value REAL,
                Profit_Margins REAL, ROE REAL, Debt_to_Equity REAL,
                RSI REAL, MACD_Signal TEXT, SMA_Status TEXT,
                
                Last_Updated TEXT
            )
        ''')
        self.conn.commit()

    # --- User Management ---
    def add_user(self, username, password):
        pwd_hash = hashlib.sha256(password.encode()).hexdigest()
        try:
            self.conn.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, pwd_hash))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError: return False

    def verify_user(self, username, password):
        pwd_hash = hashlib.sha256(password.encode()).hexdigest()
        res = self.conn.execute("SELECT * FROM users WHERE username=? AND password_hash=?", (username, pwd_hash)).fetchone()
        return res is not None

    # --- Watchlist Management ---
    def toggle_watchlist(self, username, ticker):
        exists = self.conn.execute("SELECT 1 FROM watchlist WHERE username=? AND ticker=?", (username, ticker)).fetchone()
        if exists:
            self.conn.execute("DELETE FROM watchlist WHERE username=? AND ticker=?", (username, ticker))
            status = "removed"
        else:
            date = datetime.datetime.now().strftime("%Y-%m-%d")
            self.conn.execute("INSERT INTO watchlist VALUES (?, ?, ?)", (username, ticker, date))
            status = "added"
        self.conn.commit()
        return status

    def get_user_watchlist(self, username):
        query = """
            SELECT w.ticker, a.Price, a.Verdict_Overall, a.Verdict_Valuation, a.Verdict_Technical 
            FROM watchlist w
            LEFT JOIN analysis a ON w.ticker = a.Ticker
            WHERE w.username = ?
        """
        return pd.read_sql_query(query, self.conn, params=(username,))

    # --- Analysis Data ---
    def save_analysis(self, data):
        keys = list(data.keys())
        placeholders = ', '.join(['?'] * len(keys))
        columns = ', '.join(keys)
        sql = f"INSERT OR REPLACE INTO analysis ({columns}) VALUES ({placeholders})"
        self.conn.execute(sql, list(data.values()))
        self.conn.commit()

    def get_analysis(self, ticker):
        return pd.read_sql_query("SELECT * FROM analysis WHERE Ticker=?", self.conn, params=(ticker,))

# --- 3. MULTI-METHOD ANALYST ENGINE ---
class StockAnalyst:
    def __init__(self, db):
        self.db = db

    def get_data(self, ticker):
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            info = stock.info
            return hist, info
        except:
            return None, None

    def analyze(self, ticker):
        ticker = ticker.strip().upper()
        hist, info = self.get_data(ticker)
        
        if hist is None or hist.empty or not info:
            return None

        # --- A. TECHNICAL METHOD (Timing) ---
        # 1. RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # 2. Trends (SMA)
        sma50 = hist['Close'].rolling(50).mean().iloc[-1]
        sma200 = hist['Close'].rolling(200).mean().iloc[-1]
        price = hist['Close'].iloc[-1]
        
        # Tech Scoring
        tech_score = 0
        tech_reasons = []
        
        if price > sma200: tech_score += 1; tech_reasons.append("Price > 200 SMA (Bullish)")
        else: tech_score -= 1; tech_reasons.append("Price < 200 SMA (Bearish)")
        
        if sma50 > sma200: tech_score += 1 # Golden Cross territory
        
        if current_rsi < 30: tech_score += 2; tech_reasons.append("RSI Oversold (Buy Dip)")
        elif current_rsi > 70: tech_score -= 2; tech_reasons.append("RSI Overbought (Sell Rip)")
        
        if tech_score >= 2: v_tech = "BUY"
        elif tech_score <= -1: v_tech = "SELL"
        else: v_tech = "HOLD"

        # --- B. FUNDAMENTAL METHOD (Health/Quality) ---
        f_score = 0
        roe = safe_num(info.get('returnOnEquity'))
        margins = safe_num(info.get('profitMargins'))
        debt_eq = safe_num(info.get('debtToEquity'))
        
        if roe and roe > 0.15: f_score += 1
        if margins and margins > 0.20: f_score += 1
        if debt_eq and debt_eq < 100: f_score += 1
        elif debt_eq and debt_eq > 200: f_score -= 1 # Penalty for high debt

        if f_score >= 2: v_fund = "STRONG"
        elif f_score >= 1: v_fund = "STABLE"
        else: v_fund = "WEAK"

        # --- C. VALUATION METHOD (Price vs Value) ---
        v_score = 0
        sector = info.get('sector', 'Unknown')
        bench = SECTOR_BENCHMARKS.get(sector, DEFAULT_BENCHMARKS)
        
        # 1. Relative Valuation (Comps)
        pe = safe_num(info.get('trailingPE'))
        ev_ebitda = safe_num(info.get('enterpriseToEbitda'))
        pb = safe_num(info.get('priceToBook'))
        
        if pe and pe < bench['PE']: v_score += 1
        if ev_ebitda and ev_ebitda < bench['EV_EBITDA']: v_score += 1
        if pb and pb < bench['PB']: v_score += 1

        # 2. Intrinsic Value (Graham Number)
        # Graham Number = Sqrt(22.5 * EPS * BookValuePerShare)
        eps = safe_num(info.get('trailingEps'))
        bvps = safe_num(info.get('bookValue'))
        graham_num = None
        
        if eps and bvps and eps > 0 and bvps > 0:
            graham_num = (22.5 * eps * bvps) ** 0.5
            if price < graham_num: v_score += 2 # Strong value signal
            elif price > graham_num * 1.5: v_score -= 1 # Overvalued

        if v_score >= 3: v_val = "UNDERVALUED"
        elif v_score >= 1: v_val = "FAIR VALUE"
        else: v_val = "OVERVALUED"

        # --- D. COMPOSITE VERDICT ---
        # Weighted decision: Valuation (40%) + Fundamentals (30%) + Technicals (30%)
        # Simple logic gate for final output:
        
        final_verdict = "HOLD"
        if v_val == "UNDERVALUED" and v_fund != "WEAK":
            if v_tech == "BUY": final_verdict = "STRONG BUY" # The "Perfect Storm"
            else: final_verdict = "BUY" # Good company, cheap price, waiting for tech signal
        elif v_val == "OVERVALUED":
            if v_tech == "SELL": final_verdict = "STRONG SELL"
            else: final_verdict = "SELL"
        elif v_val == "FAIR VALUE":
            if v_tech == "BUY": final_verdict = "BUY" # Swing trade opportunity
            else: final_verdict = "HOLD"

        # Save to DB
        record = {
            'Ticker': ticker, 'Price': price,
            'Verdict_Overall': final_verdict,
            'Verdict_Technical': v_tech,
            'Verdict_Fundamental': v_fund,
            'Verdict_Valuation': v_val,
            'Score_Tech': tech_score, 'Score_Fund': f_score, 'Score_Val': v_score,
            'Sector': sector,
            'PE_Ratio': pe, 'EV_EBITDA': ev_ebitda, 'PB_Ratio': pb,
            'Graham_Number': graham_num if graham_num else 0,
            'Profit_Margins': margins, 'ROE': roe, 'Debt_to_Equity': debt_eq,
            'RSI': current_rsi, 'SMA_Status': "Bullish" if price > sma200 else "Bearish",
            'Last_Updated': datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        self.db.save_analysis(record)
        return record

# --- 4. STREAMLIT FRONTEND ---
st.set_page_config(page_title="Stock Engine Pro", layout="wide", page_icon="📈")

# Initialize Logic
if 'db' not in st.session_state:
    st.session_state.db = DatabaseManager(DB_FILENAME)
    st.session_state.analyst = StockAnalyst(st.session_state.db)

db = st.session_state.db
bot = st.session_state.analyst

# --- AUTHENTICATION SIDEBAR ---
with st.sidebar:
    st.title("🔐 Access")
    if 'user' not in st.session_state: st.session_state.user = None

    if not st.session_state.user:
        tab_l, tab_r = st.tabs(["Login", "Register"])
        with tab_l:
            u = st.text_input("User", key="l_u")
            p = st.text_input("Pass", type="password", key="l_p")
            if st.button("Log In"):
                if db.verify_user(u, p):
                    st.session_state.user = u
                    st.rerun()
                else: st.error("Fail")
        with tab_r:
            u2 = st.text_input("User", key="r_u")
            p2 = st.text_input("Pass", type="password", key="r_p")
            if st.button("Register"):
                if db.add_user(u2, p2): st.success("Created! Login now.")
                else: st.error("Taken")
    else:
        st.success(f"User: {st.session_state.user}")
        if st.button("Logout"):
            st.session_state.user = None
            st.rerun()
        
        st.divider()
        st.subheader("⭐ Watchlist")
        wl = db.get_user_watchlist(st.session_state.user)
        if not wl.empty:
            for i, row in wl.iterrows():
                col1, col2 = st.columns([4,1])
                col1.markdown(f"**{row['ticker']}**: {row['Verdict_Overall']}")
                if col2.button("✖", key=f"rm_{row['ticker']}"):
                    db.toggle_watchlist(st.session_state.user, row['ticker'])
                    st.rerun()
        else:
            st.caption("Empty.")

# --- MAIN APP ---
st.title("📈 Stock Engine Pro")
st.markdown("### Multi-Method Analysis: Technicals, Fundamentals, & Valuation")

if st.session_state.user:
    # INPUT SECTION
    c1, c2 = st.columns([3, 1])
    with c1:
        txt_input = st.text_input("Enter Ticker Symbol (e.g., AAPL, NVDA, F)", "")
    with c2:
        st.write("")
        st.write("")
        if st.button("🔍 Run Full Analysis", type="primary", use_container_width=True):
            if txt_input:
                with st.spinner(f"Running multiple engines on {txt_input}..."):
                    res = bot.analyze(txt_input)
                    if not res: st.error("Data fetch failed.")

    # FETCH CURRENT DATA
    if txt_input:
        df = db.get_analysis(txt_input.upper())
        
        if not df.empty:
            row = df.iloc[0]
            
            # --- TOP HEADER: PRICE & OVERALL VERDICT ---
            st.divider()
            col_main_1, col_main_2, col_main_3 = st.columns([1, 2, 1])
            
            with col_main_1:
                st.metric("Current Price", f"${row['Price']:,.2f}")
            
            with col_main_2:
                # The "Verdict Matrix" Summary
                st.markdown(f"<h2 style='text-align: center; color: {get_color(row['Verdict_Overall'])};'>VERDICT: {row['Verdict_Overall']}</h2>", unsafe_allow_html=True)
            
            with col_main_3:
                # Watchlist Toggle
                wl_label = "Remove from Watchlist" if not db.get_user_watchlist(st.session_state.user).empty and txt_input.upper() in db.get_user_watchlist(st.session_state.user)['ticker'].values else "Add to Watchlist"
                if st.button(f"⭐ {wl_label}", use_container_width=True):
                    db.toggle_watchlist(st.session_state.user, txt_input.upper())
                    st.rerun()

            # --- THE "METHOD VISIBILITY" SECTION ---
            st.subheader("📊 Method Breakdown")
            st.info("Observe how the verdict changes based on the analysis method used.")
            
            tab_val, tab_fund, tab_tech = st.tabs(["💰 Valuation Engine", "🏢 Fundamental Engine", "📉 Technical Engine"])
            
            # 1. VALUATION TAB
            with tab_val:
                c_v1, c_v2 = st.columns([1, 2])
                with c_v1:
                    st.markdown(f"### Verdict: **{row['Verdict_Valuation']}**")
                    st.caption("Based on Multiples & Graham Number")
                    st.metric("Graham Fair Value", f"${row['Graham_Number']:,.2f}", delta=f"{row['Price'] - row['Graham_Number']:,.2f} diff", delta_color="inverse")
                with c_v2:
                    # Comparison Chart
                    st.dataframe(pd.DataFrame({
                        "Metric": ["P/E Ratio", "EV/EBITDA", "P/B Ratio"],
                        "Stock Value": [row['PE_Ratio'], row['EV_EBITDA'], row['PB_Ratio']],
                        "Sector Benchmark": ["Analysis vs Sector", "Lower is Better", "Lower is Better"] 
                    }), use_container_width=True)

            # 2. FUNDAMENTAL TAB
            with tab_fund:
                c_f1, c_f2 = st.columns([1, 2])
                with c_f1:
                    st.markdown(f"### Verdict: **{row['Verdict_Fundamental']}**")
                    st.caption("Based on Financial Health & Solvency")
                with c_f2:
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("ROE", f"{row['ROE']*100:.1f}%", "Target: >15%")
                    col_m2.metric("Profit Margin", f"{row['Profit_Margins']*100:.1f}%", "Target: >20%")
                    col_m3.metric("Debt/Equity", f"{row['Debt_to_Equity']:.0f}%", "Target: <100%", delta_color="inverse")

            # 3. TECHNICAL TAB
            with tab_tech:
                c_t1, c_t2 = st.columns([1, 2])
                with c_t1:
                    st.markdown(f"### Verdict: **{row['Verdict_Technical']}**")
                    st.caption("Based on RSI & Moving Averages")
                with c_t2:
                    col_t1, col_t2 = st.columns(2)
                    col_t1.metric("RSI (14)", f"{row['RSI']:.1f}", "30=Oversold, 70=Overbought")
                    col_t2.metric("Trend (200 SMA)", row['SMA_Status'])

else:
    st.warning("Please Login or Register (Sidebar) to access the Stock Engine.")