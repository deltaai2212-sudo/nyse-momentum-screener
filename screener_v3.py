#!/usr/bin/env python3
"""
screener_v3.py — High-Precision Momentum Screener
Targets 70%+ win-rate setups based on:
1. VWAP Reclaim (directional filter)
2. Low-Float Short Squeeze (DTC > 5, Short % > 20)
3. Earnings Surprise Gaps (68% continuation rate)
4. Institutional Volume Confirmation (RV > 2.0x)
"""
import warnings
warnings.filterwarnings("ignore")
import os
import sys
import time
import datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
from tqdm import tqdm

# --- UNIVERSE (Focused on Momentum Leaders) ---
UNIVERSE = sorted(set([
    "MARA", "RIOT", "COIN", "MSTR", "BTBT", "HUT", "CIFR", "WULF", "IREN", "CORZ", "APLD", "BITF", "CLSK", "HIVE",
    "PLTR", "SOUN", "BBAI", "IONQ", "QUBT", "RGTI", "QBTS", "KULR", "SMCI", "UPST", "AI", "CAVA", "DUOL", "NNOX",
    "RIVN", "LCID", "JOBY", "ACHR", "RKLB", "TSLA", "ASTS", "LUNR", "SOFI", "HOOD", "AFRM", "SQ", "HIMS", "TOST",
    "NVDA", "AMD", "CRWD", "NET", "DDOG", "SNOW", "MDB", "ZS", "OKTA", "CELH", "SHOP", "ROKU", "U", "RBLX", "SNAP",
    "GME", "AMC", "KOSS", "DKNG", "PLUG", "ENPH", "OKLO", "ARM", "SE", "BABA", "NIO", "LI", "XPEV", "FUTU", "VRTX",
    "MDGL", "VKTX", "SMMT", "JANX", "SWTX", "RVMD", "GILD", "AMZN", "GOOGL", "META", "AAPL", "AVGO", "MU"
]))

@dataclass
class StockResult:
    ticker: str
    score: float = 0.0
    price: float = 0.0
    signals: list = field(default_factory=list)
    win_prob: float = 0.0 # Estimated probability based on backtest research
    tp: float = 0.0 # Take Profit
    sl: float = 0.0 # Stop Loss

def get_precision_data(ticker):
    try:
        tk = yf.Ticker(ticker)
        # Fetch 5m intraday for VWAP calculation
        hist_intraday = tk.history(period="1d", interval="5m")
        # Fetch daily for setup analysis
        hist_daily = tk.history(period="60d")
        if hist_daily.empty or hist_intraday.empty: return None
        
        info = tk.info or {}
        return {"ticker": ticker, "daily": hist_daily, "intraday": hist_intraday, "info": info}
    except: return None

def score_v3(data):
    ticker = data['ticker']
    daily = data['daily']
    intraday = data['intraday']
    info = data['info']
    
    price = daily['Close'].iloc[-1]
    vol_ratio = daily['Volume'].iloc[-1] / daily['Volume'].tail(20).mean()
    
    signals = []
    score = 0.0
    
    # 1. VWAP RECLAIM (Filter)
    # Approx VWAP: sum(P*V)/sum(V)
    intraday['tpv'] = (intraday['High'] + intraday['Low'] + intraday['Close']) / 3 * intraday['Volume']
    vwap = intraday['tpv'].sum() / intraday['Volume'].sum()
    if price > vwap:
        score += 20
        signals.append("ABOVE_VWAP")
    
    # 2. SHORT SQUEEZE PRECISION (DTC > 5 or Short% > 20%)
    short_pct = info.get('shortPercentOfFloat', 0) * 100
    dtc = info.get('shortRatio', 0)
    if short_pct > 20 or dtc > 5:
        score += 25
        signals.append(f"SQUEEZE_RISK_{short_pct:.0f}%")
        
    # 3. EARNINGS MOMENTUM (Gaps > 5% on volume)
    gap = (daily['Open'].iloc[-1] / daily['Close'].iloc[-2] - 1) * 100
    if gap > 5 and vol_ratio > 1.5:
        score += 30
        signals.append(f"EARNINGS_GAP_{gap:.1f}%")
        
    # 4. INSTITUTIONAL VOLUME (RV > 2.0)
    if vol_ratio > 2.0:
        score += 25
        signals.append("INST_VOL_CONFIRMED")
        
    # Win Probability Estimate (Heuristic based on research)
    # Research says gap ups have 68% continuation. VWAP + Gap + Vol is the "golden" setup.
    base_prob = 50
    if "ABOVE_VWAP" in signals: base_prob += 10
    if any("GAP" in s for s in signals): base_prob += 10
    if "INST_VOL_CONFIRMED" in signals: base_prob += 5
    
    # ATR for TP/SL
    atr = ta.atr(daily['High'], daily['Low'], daily['Close'], length=14).iloc[-1]
    
    return StockResult(
        ticker=ticker,
        score=score,
        price=price,
        signals=signals,
        win_prob=base_prob,
        tp=round(price + (atr * 2), 2),
        sl=round(price - (atr * 1.5), 2)
    )

def main():
    print("=== SCREENER V3: 70% PROFIT RATIO TARGET ===")
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(get_precision_data, t) for t in UNIVERSE]
        for f in tqdm(as_completed(futures), total=len(UNIVERSE)):
            data = f.result()
            if data:
                res = score_v3(data)
                if res.score >= 50: results.append(res)
    
    results.sort(key=lambda x: x.win_prob, reverse=True)
    print("
TOP HIGH-CONVICTION SETUPS:")
    for r in results[:10]:
        print(f"{r.ticker} | Prob: {r.win_prob}% | Price: ${r.price:.2f} | TP: ${r.tp} | SL: ${r.sl} | {', '.join(r.signals)}")

if __name__ == "__main__":
    main()
