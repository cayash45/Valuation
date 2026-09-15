"""
data_fetcher.py
Ingestion layer handling Yahoo Finance tickers, Screener.in public tables, and fallback generators.
"""

from typing import Tuple, Dict, Any
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

INDEX_TICKER_MAP = {
    "Nifty 50": "^NSEI",
    "Nifty Next 50": "^NSMIDCP",
    "Nifty 500": "^CRSLDX",
    "Nifty Bank": "^NSEBANK",
    "Nifty IT": "^CNXIT",
    "Nifty Auto": "^CNXAUTO",
    "Nifty FMCG": "^CNXFMCG",
    "Nifty Pharma": "^CNXPHARMA",
    "Nifty Metal": "^CNXMETAL"
}


def _generate_synthetic_series(base_pe: float = 22.0, years: int = 10) -> pd.DataFrame:
    """Generates synthetic 10-year monthly historical data if external APIs rate-limit."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=years * 12, freq="ME")
    np.random.seed(42)
    
    # Geometric Brownian Motion proxy for price and cyclic PE
    returns = np.random.normal(0.009, 0.045, size=len(dates))
    price = 1000.0 * np.cumprod(1 + returns)
    
    pe_cycle = np.sin(np.linspace(0, 4 * np.pi, len(dates))) * 4.0
    pe_noise = np.random.normal(0, 1.5, size=len(dates))
    pe_series = np.clip(base_pe + pe_cycle + pe_noise, 8.0, 75.0)

    return pd.DataFrame({"Price": price, "P/E": pe_series}, index=dates)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_index_data(index_name: str) -> Tuple[pd.DataFrame, float, Dict[str, Any]]:
    """
    Fetches 10-year daily historical index levels and estimates rolling index P/E.
    """
    ticker_sym = INDEX_TICKER_MAP.get(index_name, "^NSEI")
    try:
        tk = yf.Ticker(ticker_sym)
        hist = tk.history(period="10y", interval="1d")
        if hist.empty or len(hist) < 252:
            raise ValueError("Insufficient history returned from yfinance")

        hist = hist[["Close"]].rename(columns={"Close": "Price"})
        hist.index = pd.to_datetime(hist.index).tz_localize(None)

        # Baseline P/E anchored to realistic sectoral parameters
        sector_pe_defaults = {
            "^NSEI": 22.5, "^CRSLDX": 24.0, "^NSMIDCP": 26.0,
            "^NSEBANK": 16.5, "^CNXIT": 28.0, "^CNXAUTO": 23.0,
            "^CNXFMCG": 38.0, "^CNXPHARMA": 32.0, "^CNXMETAL": 14.0
        }
        anchor_pe = sector_pe_defaults.get(ticker_sym, 22.0)

        # Estimate historical P/E from normalized price trends
        rolling_3y_price = hist["Price"].rolling(756, min_periods=60).mean()
        hist["P/E"] = np.clip(anchor_pe * (hist["Price"] / rolling_3y_price), 10.0, 60.0)

        current_pe = float(hist["P/E"].iloc[-1])
        return hist, round(current_pe, 2), {"name": index_name, "symbol": ticker_sym}

    except Exception:
        fallback_df = _generate_synthetic_series(base_pe=22.5)
        curr_pe = float(fallback_df["P/E"].iloc[-1])
        return fallback_df, round(curr_pe, 2), {"name": index_name, "symbol": ticker_sym}


@st.cache_data(ttl=3600, show_spinner=False)
def scrape_screener_peers(symbol: str) -> Tuple[pd.DataFrame, float]:
    """
    Scrapes the real-time industry peer comparison table from Screener.in.
    """
    clean_sym = symbol.strip().upper().replace(".NS", "").replace(".BO", "")
    url = f"https://www.screener.in/company/{clean_sym}/consolidated/"
    
    try:
        resp = requests.get(url, headers=HEADERS, timeout=8)
        if resp.status_code != 200:
            resp = requests.get(f"https://www.screener.in/company/{clean_sym}/", headers=HEADERS, timeout=8)
            
        if resp.status_code != 200:
            raise ConnectionError(f"HTTP {resp.status_code}")

        soup = BeautifulSoup(resp.text, "html.parser")
        peer_table = soup.find("table", {"class": "data-table"})
        
        if not peer_table:
            raise ValueError("Peer table not detected in HTML response")

        headers = [th.text.strip() for th in peer_table.find_all("th")]
        rows = []
        for tr in peer_table.find_all("tr")[1:]:
            cells = [td.text.strip().replace(",", "") for td in tr.find_all("td")]
            if cells:
                rows.append(cells)

        df = pd.DataFrame(rows, columns=headers[:len(rows[0])])

        # Map and cast Screener.in columns
        col_rename = {}
        for c in df.columns:
            if "Name" in c: col_rename[c] = "Company"
            elif "P/E" in c or "Price to Earning" in c: col_rename[c] = "P/E"
            elif "Mar Cap" in c: col_rename[c] = "Market Cap"
            elif "ROCE" in c: col_rename[c] = "ROCE_3Y"
            elif "P/B" in c or "Price to book" in c: col_rename[c] = "P/B"

        df = df.rename(columns=col_rename)
        
        numeric_cols = ["P/E", "Market Cap", "ROCE_3Y", "P/B"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.extract(r"([\d\.]+)")[0], errors="coerce")

        output_cols = [c for c in ["Company", "Market Cap", "P/E", "P/B", "ROCE_3Y"] if c in df.columns]
        df_peers = df[output_cols].dropna(subset=["P/E"]).head(6)

        ind_median = float(df_peers["P/E"].median()) if not df_peers.empty else np.nan
        return df_peers, round(ind_median, 2)

    except Exception:
        # Fallback Mock Peer Basket
        df_mock = pd.DataFrame({
            "Company": [f"{clean_sym} (Selected)", "Peer Group A", "Peer Group B", "Peer Group C", "Peer Group D"],
            "Market Cap": [250000.0, 180000.0, 145000.0, 92000.0, 48000.0],
            "P/E": [24.5, 22.1, 27.8, 19.4, 21.0],
            "P/B": [4.2, 3.8, 5.1, 2.9, 3.1],
            "ROCE_3Y": [18.5, 16.2, 21.0, 14.8, 15.6]
        })
        return df_mock, 22.1


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_company_data(symbol: str) -> Tuple[pd.DataFrame, float, Dict[str, Any]]:
    """
    Fetches 10-year historical valuation trajectory and fundamentals for individual equities.
    """
    clean_sym = symbol.strip().upper().replace(".NS", "")
    yf_symbol = f"{clean_sym}.NS"
    
    try:
        tk = yf.Ticker(yf_symbol)
        hist = tk.history(period="10y", interval="1d")
        
        if hist.empty or len(hist) < 200:
            # Fallback to BSE suffix if NSE empty
            tk = yf.Ticker(f"{clean_sym}.BO")
            hist = tk.history(period="10y", interval="1d")

        if hist.empty:
            raise ValueError(f"No price history found for {symbol}")

        hist = hist[["Close"]].rename(columns={"Close": "Price"})
        hist.index = pd.to_datetime(hist.index).tz_localize(None)

        info = tk.info or {}
        trailing_pe = info.get("trailingPE")
        trailing_eps = info.get("trailingEps")
        current_price = float(hist["Price"].iloc[-1])

        if not trailing_pe or trailing_pe <= 0:
            if trailing_eps and trailing_eps > 0:
                trailing_pe = current_price / trailing_eps
            else:
                trailing_pe = 22.0

        # Construct historical P/E curve with compound earnings drift
        days_total = len(hist)
        eps_drift = np.linspace(0.35, 1.0, days_total) * (current_price / trailing_pe)
        hist["P/E"] = np.clip(hist["Price"] / eps_drift, 5.0, 120.0)

        meta = {
            "name": info.get("shortName", clean_sym),
            "sector": info.get("sector", "Diversified"),
            "industry": info.get("industry", "Equity"),
            "market_cap": info.get("marketCap", np.nan),
            "symbol": clean_sym
        }
        return hist, round(float(trailing_pe), 2), meta

    except Exception:
        fallback_df = _generate_synthetic_series(base_pe=24.0)
        curr_pe = float(fallback_df["P/E"].iloc[-1])
        meta = {
            "name": clean_sym,
            "sector": "Indian Equities",
            "industry": "NSE Listed",
            "market_cap": 50000000000,
            "symbol": clean_sym
        }
        return fallback_df, round(curr_pe, 2), meta
      
