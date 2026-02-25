#!/usr/bin/env python3
"""
Full-Universe US Market Multi-Factor Stock Screener
====================================================
Scans NYSE + NASDAQ + AMEX + ARCA + BATS — targeting 10,000-15,000+
tradeable US securities — through a multi-phase funnel:

  Phase 1  – Bulk yfinance download for fast momentum / volume filter
  Phase 2  – Deep-dive analysis on top candidates (technicals,
             fundamentals, options flow, catalysts, news sentiment)

Architecture
------------
1. Fetch full US ticker universe from NASDAQ FTP (primary),
   Finviz screener (fallback), or hardcoded list (emergency).
2. Phase-1 bulk filter (fast_info + 50-day history).
3. Deep-dive scoring on survivors (technical, fundamental,
   options, catalyst, news).
4. Print ranked report to stdout + optional CSV.

Exchanges covered: NYSE (N) · NASDAQ · AMEX (A) · ARCA (P) ·
                   BATS (Z) · other OTC/regional (V)

Requirements: Python 3.9+, yfinance, pandas, numpy, requests,
              beautifulsoup4.  No other packages.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import logging
import os
import random
import re
import signal
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

# ── Logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("screener")

# ── Timeout protection ──────────────────────────────────────────────
class TickerTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TickerTimeout("Ticker analysis timed out")


# ── Configuration ───────────────────────────────────────────────────
PHASE1_TOP_N = 200          # survivors from bulk filter
DEEP_DIVE_COUNT = 200       # how many get full scoring
QUICK_UNIVERSE_CAP = 500    # --quick mode cap
TICKER_TIMEOUT_SEC = 30     # per-ticker wall-clock limit

US_SOURCE = "NASDAQ_FTP"    # primary source tag for logging

# yfinance download params
YF_PERIOD = "3mo"
YF_INTERVAL = "1d"
YF_GROUP_SIZE = 500         # tickers per yf.download batch

# Scoring caps (unchanged)
TECH_CAP = 30
FUND_CAP = 25
OPT_CAP = 20
CAT_CAP = 25
NEWS_CAP = 15

# ── Keyword sets for news scoring ───────────────────────────────────
_POS_WORDS = {
    "upgrade", "beat", "raise", "buy", "outperform", "bullish",
    "record", "surge", "high", "growth", "exceed", "positive",
    "strong", "upside", "breakout", "momentum", "rally", "soar",
    "spike", "explode", "moon", "short", "squeeze",
}
_NEG_WORDS = {
    "downgrade", "miss", "cut", "sell", "underperform", "bearish",
    "low", "decline", "negative", "weak", "downside", "warning",
    "risk", "loss", "dilution", "offering", "reverse", "split",
    "delist", "bankruptcy", "halt",
}

# ── TickerResult dataclass ──────────────────────────────────────────
@dataclass
class TickerResult:
    symbol: str
    price: float = 0.0
    change_pct: float = 0.0
    volume: int = 0
    avg_volume: int = 0
    market_cap: float = 0.0
    sector: str = ""
    technical_score: float = 0.0
    fundamental_score: float = 0.0
    options_score: float = 0.0
    catalyst_score: float = 0.0
    news_score: float = 0.0
    total_score: float = 0.0
    notes: List[str] = field(default_factory=list)
    error: str = ""


# =====================================================================
#  TICKER UNIVERSE FETCH
# =====================================================================

# ---------- helpers --------------------------------------------------

_INVALID_CHARS = re.compile(r"[\$\./\-\+\^\s]")


def _clean_tickers(raw: List[str]) -> List[str]:
    """Remove junk symbols, deduplicate, shuffle."""
    seen: set = set()
    clean: List[str] = []
    for t in raw:
        t = t.strip().upper()
        if not t or len(t) < 1 or len(t) > 5:
            continue
        if _INVALID_CHARS.search(t):
            continue
        if t not in seen:
            seen.add(t)
            clean.append(t)
    random.shuffle(clean)
    log.info("Fetched %d raw, %d valid after cleaning", len(raw), len(clean))
    return clean


def _get(url: str, timeout: int = 15) -> Optional[str]:
    """Simple GET with timeout; returns text or None."""
    try:
        r = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (stock-screener)"
        })
        r.raise_for_status()
        return r.text
    except Exception as exc:
        log.warning("GET %s failed: %s", url, exc)
        return None


# ---------- Source 1: NASDAQ FTP (primary) ---------------------------

def _fetch_nasdaq_ftp() -> List[str]:
    """
    Fetch from both NASDAQ FTP pipe-delimited files:
      nasdaqlisted.txt  -> all NASDAQ symbols
      otherlisted.txt   -> NYSE (N), AMEX (A), ARCA (P), BATS (Z), other (V)
    Returns combined raw ticker list (may contain dupes).
    """
    tickers: List[str] = []

    # -- nasdaqlisted.txt ------------------------------------------------
    url1 = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
    body = _get(url1, timeout=20)
    if body:
        for line in body.splitlines()[1:]:          # skip header
            if line.startswith("File Creation"):    # footer
                continue
            parts = line.split("|")
            if len(parts) < 4:
                continue
            sym = parts[0].strip()
            test_issue = parts[3].strip() if len(parts) > 3 else ""
            if test_issue == "Y":
                continue
            tickers.append(sym)
        log.info("nasdaqlisted.txt → %d symbols", len(tickers))
    else:
        log.warning("nasdaqlisted.txt download failed")

    # -- otherlisted.txt -------------------------------------------------
    url2 = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
    body2 = _get(url2, timeout=20)
    before = len(tickers)
    if body2:
        for line in body2.splitlines()[1:]:
            if line.startswith("File Creation"):
                continue
            parts = line.split("|")
            if len(parts) < 7:
                continue
            sym = parts[0].strip()          # ACT Symbol
            # Exchange: N=NYSE, A=AMEX, P=ARCA, Z=BATS, V=other
            # Take ALL codes — no filter on exchange
            test_issue = parts[6].strip() if len(parts) > 6 else ""
            if test_issue == "Y":
                continue
            tickers.append(sym)
        log.info("otherlisted.txt  → %d symbols (total so far %d)",
                 len(tickers) - before, len(tickers))
    else:
        log.warning("otherlisted.txt download failed")

    return tickers


# ---------- Source 2: Finviz fallback --------------------------------

def _parse_finviz_page(html: str) -> List[str]:
    """Extract ticker symbols from one Finviz screener results page."""
    soup = BeautifulSoup(html, "html.parser")
    tickers: List[str] = []
    # Finviz puts tickers in <a> tags inside the screener table
    for a_tag in soup.find_all("a", class_="screener-link-primary"):
        t = a_tag.get_text(strip=True)
        if t and t.isalpha():
            tickers.append(t)
    return tickers


def scrape_finviz_screen() -> List[str]:
    """
    Paginate Finviz screener for all US exchanges.
    No cap / volume / momentum filters — just exchange selection.
    """
    base = ("https://finviz.com/screener.ashx?v=111"
            "&f=exch_nasdaqsm,exch_nasdaq,exch_nyse,exch_amex&ft=4")
    tickers: List[str] = []
    r_offset = 1
    max_pages = 800  # safety: ~16,000 stocks at 20/page
    while r_offset < max_pages * 20:
        url = f"{base}&r={r_offset}"
        html = _get(url, timeout=15)
        if not html:
            break
        page_tickers = _parse_finviz_page(html)
        if not page_tickers:
            break
        tickers.extend(page_tickers)
        r_offset += 20
        time.sleep(0.35)  # polite rate-limit
    log.info("Finviz fallback → %d symbols", len(tickers))
    return tickers


# ---------- Source 3: Hardcoded 600+ fallback ------------------------

_HARDCODED_TICKERS = [
    # ── MEGA / LARGE CAP (NYSE + NASDAQ) ──
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA",
    "BRK", "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "AVGO", "HD",
    "CVX", "MRK", "ABBV", "LLY", "COST", "PEP", "KO", "ADBE", "WMT",
    "BAC", "CRM", "MCD", "CSCO", "TMO", "ACN", "ABT", "NFLX", "AMD",
    "LIN", "DHR", "TXN", "WFC", "PM", "NEE", "UPS", "ORCL", "INTC",
    "RTX", "QCOM", "HON", "AMGN", "UNP", "INTU", "CAT", "LOW", "SPGI",
    "BA", "GE", "ISRG", "BLK", "DE", "GILD", "MDLZ", "ADP", "SYK",
    "BKNG", "VRTX", "SBUX", "MMC", "PLD", "TMUS", "ADI", "LRCX",
    "TJX", "CB", "PANW", "AXP", "CI", "SCHW", "REGN", "ZTS", "MO",
    "SO", "FISV", "CME", "DUK", "EQIX", "BSX", "BDX", "SNPS", "CDNS",
    "ICE", "CL", "CSX", "EOG", "AON", "NOC", "SHW", "MPC", "PH",
    "MCK", "APD", "EMR", "ORLY", "TT", "PSX", "GD", "PYPL",
    "USB", "ITW", "WELL", "PNC", "HUM", "MRVL", "FCX", "ROP",
    # ── MID-CAP / GROWTH ──
    "CRWD", "DDOG", "SNOW", "NET", "ZS", "MDB", "TEAM", "HUBS",
    "FTNT", "TTD", "WDAY", "OKTA", "BILL", "DKNG", "RBLX", "PINS",
    "SNAP", "ROKU", "LYFT", "UBER", "DASH", "ABNB", "COIN", "HOOD",
    "SQ", "SOFI", "AFRM", "UPST", "MELI", "SE", "GRAB", "NU",
    "SHOP", "SPOT", "U", "PATH", "CFLT", "FROG", "ESTC", "GTLB",
    "MNDY", "DOCN", "TOST", "CAVA", "CART", "BIRK", "DUOL",
    "APP", "CELH", "ELF", "ONON", "DECK", "LULU", "MNST", "WING",
    "TXRH", "CMG", "CTAS", "ODFL", "SAIA", "XPO", "KNSL", "ARES",
    "OWL", "APO", "KKR", "BX", "CG", "HLNE", "TPG", "BAM",
    # ── BIOTECH / PHARMA ──
    "MRNA", "BNTX", "BMRN", "ALNY", "SRPT", "EXAS", "IONS", "PCVX",
    "LEGN", "ARGX", "KRYS", "XENE", "RYTM", "CRNX", "ACAD", "PTCT",
    "INSM", "RARE", "HALO", "NBIX", "AXSM", "CORT", "TGTX", "FOLD",
    "IRON", "RVMD", "KRTX", "PRTA", "ANNX", "DAWN", "SAVA", "APLS",
    "RCKT", "BEAM", "CRSP", "NTLA", "EDIT", "VERV", "PRME",
    "VERA", "IMVT", "RLAY", "CYTK", "CPRX", "GERN",
    # ── SMALL-CAP / SPECULATIVE ──
    "PLTR", "IONQ", "RGTI", "QUBT", "SMCI", "SOUN", "BBAI", "JOBY",
    "LILM", "ACHR", "LUNR", "RKLB", "ASTS", "SATS", "SPCE",
    "DNA", "GEVO", "PLUG", "FCEL", "BE", "BLDP", "CHPT", "BLNK",
    "EVGO", "QS", "MVST", "PTRA", "GOEV", "LCID", "RIVN", "FSR",
    "NKLA", "LAZR", "LIDR", "OUST", "AEVA", "CPNG", "BABA",
    "JD", "PDD", "BIDU", "NIO", "XPEV", "LI", "BILI", "TME",
    "IQ", "FUTU", "TIGR", "WB", "ZH", "MNSO", "YMM", "DADA",
    "FFIE", "MULN", "NRDS", "CLOV", "WISH",
    # ── FINANCIALS / REITS ──
    "GS", "MS", "C", "HBAN", "KEY", "RF", "CFG", "FITB",
    "MTB", "SIVB", "ZION", "CMA", "TFC", "STT", "NTRS", "BK",
    "ALLY", "DFS", "COF", "AIG", "MET", "PRU", "AFL", "TRV",
    "ALL", "PGR", "HIG", "L", "GL", "WRB", "RE", "RNR",
    "AMT", "CCI", "SBAC", "DLR", "SPG", "O", "VICI", "GLPI",
    "NNN", "WPC", "ADC", "STAG", "FR", "REXR", "ARE", "BXP",
    "VTR", "PEAK", "OHI", "MPW", "DOC", "HR",
    # ── ENERGY / MATERIALS ──
    "SLB", "HAL", "BKR", "OXY", "COP", "DVN", "PXD", "FANG",
    "APA", "CTRA", "EQT", "AR", "RRC", "SWN", "MUR", "OVV",
    "SM", "CHRD", "MTDR", "PR", "VNOM", "MGY", "CLR",
    "KMI", "WMB", "OKE", "ET", "EPD", "MPLX", "PAA", "TRGP",
    "NUE", "STLD", "CLF", "X", "RS", "ATI", "CMC", "CENX",
    "AA", "ARNC", "KALU", "HAYN", "TMST", "IOSP", "TROX",
    "APD", "LIN", "ECL", "SHW", "PPG", "RPM", "VMC", "MLM",
    # ── INDUSTRIALS / DEFENCE ──
    "LMT", "RTX", "GD", "NOC", "BA", "TXT", "HII", "LHX",
    "LDOS", "BAH", "SAIC", "CACI", "BWXT", "HEI", "TDG", "MOG",
    "CW", "GR", "SPR", "ERJ", "ALGT", "HA", "ALK", "JBLU",
    "DAL", "UAL", "LUV", "AAL", "SAVE", "SKYW", "MESA", "ATSG",
    "FDX", "UPS", "XPO", "SAIA", "ODFL", "JBHT", "LSTR", "WERN",
    "KNX", "SNDR", "CHRW", "EXPD", "FWRD", "HUBG", "MATX",
    # ── CONSUMER / RETAIL ──
    "AMZN", "WMT", "TGT", "COST", "DG", "DLTR", "BJ", "OLLI",
    "FIVE", "ROST", "TJX", "BURL", "GPS", "ANF", "AEO", "URBN",
    "LULU", "NKE", "CROX", "SKX", "DECK", "ONON", "BIRD", "HOKA",
    "DIS", "CMCSA", "WBD", "PARA", "NFLX", "ROKU", "LGF",
    "RBLX", "EA", "TTWO", "ATVI", "ZNGA", "PLTK", "SKLZ",
    "MCD", "SBUX", "CMG", "YUM", "QSR", "WING", "TXRH", "DRI",
    "CAKE", "EAT", "DIN", "JACK", "SHAK", "BROS", "LOCO",
    # ── TECH HARDWARE / SEMIS ──
    "TSM", "ASML", "KLAC", "AMAT", "LRCX", "TER", "ONTO",
    "COHR", "MKSI", "ENTG", "AEHR", "ACLS", "WOLF", "SLAB",
    "MPWR", "SWKS", "QRVO", "MCHP", "NXPI", "ON", "DIOD",
    "STM", "UMC", "GFS", "RMBS", "CRUS", "SITM", "POWI",
    "ALGM", "AMBA", "LSCC", "INDI", "SMTC", "ACMR",
    # ── ETFS (broad coverage) ──
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "VXF", "VTWO",
    "MDY", "IJR", "IVV", "SPLG", "SCHB", "ITOT", "VT", "VXUS",
    "EFA", "EEM", "VWO", "IEMG", "GLD", "SLV", "GDX", "GDXJ",
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
    "XLB", "XLRE", "XLC", "XBI", "IBB", "ARKG", "ARKK", "ARKF",
    "ARKQ", "ARKW", "SOXL", "SOXS", "TQQQ", "SQQQ", "SPXL",
    "SPXS", "UVXY", "VXX", "TLT", "TBT", "HYG", "LQD", "JNK",
    "SCHD", "VYM", "DVY", "SDY", "DGRO", "NOBL", "VIG", "DGRW",
    "JEPI", "JEPQ", "XYLD", "QYLD", "RYLD", "NUSI", "DIVO",
    "SMH", "SOXX", "PSI", "QTEC", "SKYY", "HACK", "BUG", "CIBR",
    "KWEB", "FXI", "MCHI", "ASHR", "YINN", "YANG", "EWJ", "EWZ",
    "EWY", "EWT", "INDA", "SMIN", "VNM", "FM", "GREK",
]


# ---------- main fetch function --------------------------------------

def fetch_all_us_tickers() -> List[str]:
    """
    Three-tier ticker fetch covering the full US tradeable universe:
      1. NASDAQ FTP files (primary)    → 10k-15k symbols
      2. Finviz screener  (fallback)   → 5k-10k symbols
      3. Hardcoded list   (emergency)  → ~600 symbols
    Returns cleaned, deduplicated, shuffled list.
    """
    # --- Source 1: NASDAQ FTP ---
    log.info("[%s] Attempting NASDAQ FTP fetch …", US_SOURCE)
    raw = _fetch_nasdaq_ftp()
    if len(raw) >= 5000:
        return _clean_tickers(raw)

    # --- Source 2: Finviz ---
    log.info("[%s] FTP yielded only %d; trying Finviz …", US_SOURCE, len(raw))
    try:
        raw2 = scrape_finviz_screen()
        if len(raw2) >= 3000:
            return _clean_tickers(raw2)
        # merge whatever we got
        raw.extend(raw2)
        if len(raw) >= 3000:
            return _clean_tickers(raw)
    except Exception as exc:
        log.warning("Finviz fallback failed: %s", exc)

    # --- Source 3: Hardcoded ---
    log.info("[%s] Using hardcoded %d-ticker emergency list",
             US_SOURCE, len(_HARDCODED_TICKERS))
    raw.extend(_HARDCODED_TICKERS)
    return _clean_tickers(raw)


# =====================================================================
#  MARKET REGIME CHECK
# =====================================================================

def check_market_regime() -> Dict[str, Any]:
    """
    Quick read on broad market conditions via SPY.
    Returns dict with trend, breadth estimate, and vix-proxy.
    """
    regime: Dict[str, Any] = {
        "trend": "neutral",
        "spy_above_200d": None,
        "spy_above_50d": None,
        "vix_proxy": None,
    }
    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period="1y", interval="1d")
        if hist.empty:
            return regime
        close = hist["Close"]
        last = close.iloc[-1]
        if len(close) >= 200:
            regime["spy_above_200d"] = bool(last > close.rolling(200).mean().iloc[-1])
        if len(close) >= 50:
            regime["spy_above_50d"] = bool(last > close.rolling(50).mean().iloc[-1])
        # simple vix proxy: 20-day realized vol annualised
        if len(close) >= 21:
            ret = close.pct_change().dropna().tail(20)
            regime["vix_proxy"] = round(float(ret.std() * np.sqrt(252) * 100), 1)
        if regime["spy_above_200d"] and regime["spy_above_50d"]:
            regime["trend"] = "bullish"
        elif not regime["spy_above_200d"] and not regime["spy_above_50d"]:
            regime["trend"] = "bearish"
    except Exception as exc:
        log.warning("Market regime check failed: %s", exc)
    return regime


# =====================================================================
#  PHASE 1 – BULK FILTER
# =====================================================================

def phase1_bulk_filter(tickers: List[str], top_n: int = PHASE1_TOP_N,
                       ) -> List[Tuple[str, float, float, int]]:
    """
    Fast screen: download daily bars for all tickers in batches,
    rank by a simple momentum × relative-volume composite.
    Returns list of (symbol, last_close, change_pct, volume) sorted desc.
    """
    log.info("Phase-1: bulk-filtering %d tickers (top %d)", len(tickers), top_n)
    results: List[Tuple[str, float, float, float, int]] = []

    for i in range(0, len(tickers), YF_GROUP_SIZE):
        batch = tickers[i : i + YF_GROUP_SIZE]
        try:
            df = yf.download(
                batch,
                period=YF_PERIOD,
                interval=YF_INTERVAL,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as exc:
            log.warning("yf.download batch %d failed: %s", i // YF_GROUP_SIZE, exc)
            continue

        for sym in batch:
            try:
                if len(batch) == 1:
                    sub = df
                else:
                    sub = df[sym] if sym in df.columns.get_level_values(0) else None
                if sub is None or sub.empty:
                    continue
                sub = sub.dropna(subset=["Close"])
                if len(sub) < 10:
                    continue
                close = sub["Close"]
                vol = sub["Volume"]
                last_close = float(close.iloc[-1])
                if last_close < 1.0:
                    continue  # penny stocks
                pct_5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) >= 6 else 0.0
                avg_vol = float(vol.tail(20).mean()) if len(vol) >= 20 else float(vol.mean())
                last_vol = float(vol.iloc[-1])
                if avg_vol < 50_000:
                    continue  # illiquid
                rel_vol = last_vol / avg_vol if avg_vol > 0 else 1.0
                composite = pct_5d * 0.6 + (rel_vol - 1) * 40 * 0.4
                results.append((sym, last_close, pct_5d, composite, int(last_vol)))
            except Exception:
                continue

    results.sort(key=lambda x: x[3], reverse=True)
    top = results[:top_n]
    log.info("Phase-1 done: %d passed liquidity filter, returning top %d",
             len(results), len(top))
    return [(s, p, c, v) for s, p, c, _, v in top]


# =====================================================================
#  SCORING FUNCTIONS
# =====================================================================

def score_technical(hist: pd.DataFrame, info: dict) -> Tuple[float, List[str]]:
    """
    Technical scoring: trend, RSI, MACD, Bollinger position,
    plus gap-up and volume-surge detection.  Cap = 30.
    """
    pts = 0.0
    notes: List[str] = []
    if hist.empty or len(hist) < 20:
        return 0, ["Insufficient history"]

    close = hist["Close"]
    high = hist["High"]
    low = hist["Low"]
    volume = hist["Volume"]
    last = float(close.iloc[-1])

    # ── Moving-average trend ──
    if len(close) >= 50:
        ma50 = float(close.rolling(50).mean().iloc[-1])
        if last > ma50:
            pts += 4
            notes.append(f"Above MA50 ({ma50:.2f})")
        else:
            pts -= 2
    if len(close) >= 20:
        ma20 = float(close.rolling(20).mean().iloc[-1])
        if last > ma20:
            pts += 3

    # ── RSI(14) ──
    if len(close) >= 15:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 100
        rsi = 100 - 100 / (1 + rs)
        if 50 < rsi < 70:
            pts += 4
            notes.append(f"RSI {rsi:.0f} bullish")
        elif 30 < rsi <= 50:
            pts += 1
        elif rsi >= 70:
            pts += 1
            notes.append(f"RSI {rsi:.0f} overbought")
        else:
            pts -= 1
            notes.append(f"RSI {rsi:.0f} oversold")

    # ── MACD ──
    if len(close) >= 26:
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        sig = macd.ewm(span=9).mean()
        if float(macd.iloc[-1]) > float(sig.iloc[-1]):
            pts += 4
            notes.append("MACD bullish cross")
        else:
            pts -= 1

    # ── Bollinger Band position ──
    if len(close) >= 20:
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = float(sma20.iloc[-1] + 2 * std20.iloc[-1])
        lower = float(sma20.iloc[-1] - 2 * std20.iloc[-1])
        band_range = upper - lower
        if band_range > 0:
            position = (last - lower) / band_range
            if 0.5 < position < 0.85:
                pts += 3
                notes.append(f"BB position {position:.0%}")
            elif position >= 0.85:
                pts += 1
                notes.append(f"BB near upper {position:.0%}")

    # ── NEW: Gap-up detection ──
    if len(hist) >= 2:
        yesterday_close = float(close.iloc[-2])
        today_open = float(hist["Open"].iloc[-1])
        if yesterday_close > 0 and today_open > yesterday_close * 1.005:
            gap_pct = (today_open / yesterday_close - 1) * 100
            pts += 3
            notes.append(f"Gap up {gap_pct:.1f}%")

    # ── NEW: Volume-surge detection ──
    if len(volume) >= 21:
        avg_vol = float(volume.tail(20).iloc[:-1].mean())  # 20-day avg excl today
        today_vol = float(volume.iloc[-1])
        if avg_vol > 0 and today_vol > 2 * avg_vol:
            price_up = float(close.iloc[-1]) > float(close.iloc[-2]) if len(close) >= 2 else False
            if price_up:
                ratio = today_vol / avg_vol
                pts += 2
                notes.append(f"Vol surge {ratio:.1f}x")

    return min(pts, TECH_CAP), notes


def score_fundamentals(info: dict) -> Tuple[float, List[str]]:
    """Fundamental quality + value scoring.  Cap = 25."""
    pts = 0.0
    notes: List[str] = []

    # PE ratio
    pe = info.get("forwardPE") or info.get("trailingPE")
    if pe and 0 < pe < 25:
        pts += 4
        notes.append(f"PE {pe:.1f}")
    elif pe and 25 <= pe < 50:
        pts += 2

    # Revenue growth
    rev_growth = info.get("revenueGrowth")
    if rev_growth and rev_growth > 0.15:
        pts += 5
        notes.append(f"Rev growth {rev_growth:.0%}")
    elif rev_growth and rev_growth > 0.05:
        pts += 3

    # Profit margin
    margin = info.get("profitMargins")
    if margin and margin > 0.20:
        pts += 4
        notes.append(f"Margin {margin:.0%}")
    elif margin and margin > 0.10:
        pts += 2

    # ROE
    roe = info.get("returnOnEquity")
    if roe and roe > 0.20:
        pts += 4
        notes.append(f"ROE {roe:.0%}")
    elif roe and roe > 0.10:
        pts += 2

    # Debt / equity
    de = info.get("debtToEquity")
    if de is not None:
        if de < 50:
            pts += 3
            notes.append(f"Low D/E {de:.0f}%")
        elif de > 200:
            pts -= 2
            notes.append(f"High D/E {de:.0f}%")

    # Free cash flow yield
    mcap = info.get("marketCap") or 0
    fcf = info.get("freeCashflow") or 0
    if mcap > 0 and fcf > 0:
        fcf_yield = fcf / mcap
        if fcf_yield > 0.05:
            pts += 3
            notes.append(f"FCF yield {fcf_yield:.1%}")

    return min(pts, FUND_CAP), notes


def score_options(info: dict) -> Tuple[float, List[str]]:
    """Options-implied signals.  Cap = 20."""
    pts = 0.0
    notes: List[str] = []

    iv = info.get("impliedVolatility")
    if iv:
        if iv > 0.6:
            pts += 3
            notes.append(f"High IV {iv:.0%}")
        elif iv > 0.3:
            pts += 1

    # Put/call OI ratio from info (if available)
    pcr = info.get("putCallRatio")
    if pcr:
        if pcr < 0.7:
            pts += 4
            notes.append(f"Bullish P/C {pcr:.2f}")
        elif pcr > 1.3:
            pts -= 2
            notes.append(f"Bearish P/C {pcr:.2f}")

    # Try options chain for richer data
    try:
        ticker_obj = None
        sym = info.get("symbol")
        if sym:
            ticker_obj = yf.Ticker(sym)
            exps = ticker_obj.options
            if exps:
                chain = ticker_obj.option_chain(exps[0])
                call_oi = int(chain.calls["openInterest"].sum())
                put_oi = int(chain.puts["openInterest"].sum())
                total_oi = call_oi + put_oi
                if total_oi > 10_000:
                    pts += 3
                    notes.append(f"Active options OI {total_oi:,}")
                if call_oi > 0 and put_oi > 0:
                    r = put_oi / call_oi
                    if r < 0.7:
                        pts += 3
                        notes.append(f"Call-heavy OI ratio {r:.2f}")
                    elif r > 1.5:
                        pts -= 1
                # unusual volume
                call_vol = chain.calls["volume"].sum()
                put_vol = chain.puts["volume"].sum()
                opt_vol = call_vol + put_vol
                if opt_vol > 5000:
                    pts += 2
                    notes.append(f"Opt vol {int(opt_vol):,}")
    except Exception:
        pass

    return min(pts, OPT_CAP), notes


def score_catalyst(info: dict) -> Tuple[float, List[str]]:
    """Catalyst / event scoring.  Cap = 25."""
    pts = 0.0
    notes: List[str] = []

    # Earnings proximity
    earn_date = info.get("earningsDate")
    if earn_date:
        if isinstance(earn_date, list):
            earn_date = earn_date[0] if earn_date else None
        if earn_date:
            try:
                if hasattr(earn_date, "timestamp"):
                    ed = earn_date
                else:
                    ed = pd.Timestamp(earn_date)
                days_to = (ed - pd.Timestamp.now()).days
                if 0 < days_to <= 14:
                    pts += 5
                    notes.append(f"Earnings in {days_to}d")
                elif 0 < days_to <= 30:
                    pts += 3
                    notes.append(f"Earnings in {days_to}d")
            except Exception:
                pass

    # Analyst recommendation
    rec = info.get("recommendationMean")
    if rec:
        if rec <= 2.0:
            pts += 4
            notes.append(f"Strong analyst buy ({rec:.1f})")
        elif rec <= 2.5:
            pts += 2
            notes.append(f"Analyst buy ({rec:.1f})")

    # Target price upside
    target = info.get("targetMeanPrice")
    price = info.get("currentPrice") or info.get("regularMarketPrice") or 0
    if target and price and price > 0:
        upside = (target - price) / price
        if upside > 0.20:
            pts += 4
            notes.append(f"Target upside {upside:.0%}")
        elif upside > 0.10:
            pts += 2

    # Insider buying
    insider = info.get("heldPercentInsiders")
    if insider and insider > 0.10:
        pts += 2
        notes.append(f"Insider own {insider:.1%}")

    # ── NEW: Short-squeeze signal ──
    short_pct = info.get("shortPercentOfFloat") or 0
    if short_pct > 0.20:
        pts += 5
        notes.append(f"High short float {short_pct:.0%} – squeeze risk")
    elif short_pct > 0.10:
        pts += 2
        notes.append(f"Elevated short float {short_pct:.0%}")

    return min(pts, CAT_CAP), notes


def score_news(symbol: str) -> Tuple[float, List[str]]:
    """Simple headline-sentiment scoring via yfinance news.  Cap = 15."""
    pts = 0.0
    notes: List[str] = []
    try:
        t = yf.Ticker(symbol)
        news = t.news or []
        if not news:
            return 0, ["No recent news"]
        pos = neg = 0
        for item in news[:10]:
            title = (item.get("title") or "").lower()
            for w in _POS_WORDS:
                if w in title:
                    pos += 1
            for w in _NEG_WORDS:
                if w in title:
                    neg += 1
        net = pos - neg
        if net >= 3:
            pts += 6
            notes.append(f"News very positive ({pos}+/{neg}-)")
        elif net >= 1:
            pts += 3
            notes.append(f"News positive ({pos}+/{neg}-)")
        elif net <= -3:
            pts -= 4
            notes.append(f"News very negative ({pos}+/{neg}-)")
        elif net <= -1:
            pts -= 2
            notes.append(f"News negative ({pos}+/{neg}-)")
        else:
            notes.append(f"News neutral ({pos}+/{neg}-)")
    except Exception:
        notes.append("News fetch failed")
    return min(max(pts, -NEWS_CAP), NEWS_CAP), notes


# =====================================================================
#  DEEP-DIVE ANALYSIS
# =====================================================================

def analyse_ticker(symbol: str) -> TickerResult:
    """Full multi-factor analysis on a single ticker."""
    res = TickerResult(symbol=symbol)
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
        hist = t.history(period="6mo", interval="1d")
        if hist.empty:
            res.error = "No history"
            return res

        res.price = float(info.get("currentPrice") or info.get("regularMarketPrice")
                          or hist["Close"].iloc[-1])
        prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else res.price
        res.change_pct = round((res.price / prev - 1) * 100, 2) if prev else 0
        res.volume = int(info.get("volume") or info.get("regularMarketVolume") or 0)
        res.avg_volume = int(info.get("averageVolume") or 0)
        res.market_cap = float(info.get("marketCap") or 0)
        res.sector = info.get("sector") or ""

        # scores
        res.technical_score, tn = score_technical(hist, info)
        res.notes.extend(tn)
        res.fundamental_score, fn = score_fundamentals(info)
        res.notes.extend(fn)
        res.options_score, on = score_options(info)
        res.notes.extend(on)
        res.catalyst_score, cn = score_catalyst(info)
        res.notes.extend(cn)
        res.news_score, nn = score_news(symbol)
        res.notes.extend(nn)

        res.total_score = (res.technical_score + res.fundamental_score
                           + res.options_score + res.catalyst_score
                           + res.news_score)
    except TickerTimeout:
        res.error = "Timed out"
    except Exception as exc:
        res.error = str(exc)[:120]
    return res


def deep_dive(candidates: List[Tuple[str, float, float, int]],
              count: int = DEEP_DIVE_COUNT) -> List[TickerResult]:
    """Run full analysis on the top Phase-1 survivors."""
    log.info("Phase-2: deep-diving %d candidates", min(len(candidates), count))
    results: List[TickerResult] = []
    for i, (sym, price, chg, vol) in enumerate(candidates[:count]):
        log.info("  [%d/%d] %s ($%.2f, %+.1f%%)",
                 i + 1, min(len(candidates), count), sym, price, chg)
        # timeout protection
        try:
            if hasattr(signal, "SIGALRM"):
                old = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(TICKER_TIMEOUT_SEC)
            r = analyse_ticker(sym)
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old)
        except TickerTimeout:
            r = TickerResult(symbol=sym, error="Timed out")
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
        except Exception as exc:
            r = TickerResult(symbol=sym, error=str(exc)[:120])
        results.append(r)
    results.sort(key=lambda r: r.total_score, reverse=True)
    return results


# =====================================================================
#  REPORTING
# =====================================================================

def print_report(results: List[TickerResult], regime: Dict[str, Any]):
    """Pretty-print ranked results to stdout."""
    print("\n" + "=" * 80)
    print("  US MULTI-EXCHANGE STOCK SCREENER  ".center(80, "="))
    print("=" * 80)
    print(f"  Run: {dt.datetime.now():%Y-%m-%d %H:%M}   "
          f"Regime: {regime.get('trend', '?').upper()}   "
          f"VIX-proxy: {regime.get('vix_proxy', '?')}%")
    print(f"  SPY>200d: {regime.get('spy_above_200d')}   "
          f"SPY>50d: {regime.get('spy_above_50d')}")
    print("=" * 80)

    ranked = [r for r in results if not r.error]
    errors = [r for r in results if r.error]

    if not ranked:
        print("\n  No valid results.\n")
        return

    print(f"\n  TOP {min(30, len(ranked))} PICKS")
    print("-" * 80)
    fmt = "  {rank:>3}. {sym:<6} ${price:>8.2f} ({chg:>+6.1f}%)  "
    fmt += "T:{t:>4.0f} F:{f:>4.0f} O:{o:>4.0f} C:{c:>4.0f} N:{n:>4.0f} "
    fmt += "= {total:>5.0f}  {sector}"

    for i, r in enumerate(ranked[:30], 1):
        print(fmt.format(
            rank=i, sym=r.symbol, price=r.price, chg=r.change_pct,
            t=r.technical_score, f=r.fundamental_score,
            o=r.options_score, c=r.catalyst_score, n=r.news_score,
            total=r.total_score, sector=r.sector[:18],
        ))
        if r.notes:
            # print top 5 notes
            top_notes = [n for n in r.notes if not n.startswith("News")][:5]
            if top_notes:
                print(f"       └ {' | '.join(top_notes)}")

    if errors:
        print(f"\n  ({len(errors)} tickers had errors)")

    print("\n" + "=" * 80)


def write_csv(results: List[TickerResult], path: str):
    """Write results to CSV."""
    fieldnames = [
        "rank", "symbol", "price", "change_pct", "volume", "avg_volume",
        "market_cap", "sector", "tech", "fund", "opts", "catalyst", "news",
        "total", "notes",
    ]
    ranked = [r for r in results if not r.error]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(ranked, 1):
            w.writerow({
                "rank": i,
                "symbol": r.symbol,
                "price": r.price,
                "change_pct": r.change_pct,
                "volume": r.volume,
                "avg_volume": r.avg_volume,
                "market_cap": r.market_cap,
                "sector": r.sector,
                "tech": r.technical_score,
                "fund": r.fundamental_score,
                "opts": r.options_score,
                "catalyst": r.catalyst_score,
                "news": r.news_score,
                "total": r.total_score,
                "notes": " | ".join(r.notes[:8]),
            })
    log.info("CSV written to %s (%d rows)", path, len(ranked))


# =====================================================================
#  MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full-Universe US Market Multi-Factor Stock Screener")
    parser.add_argument("--quick", action="store_true",
                        help=f"Fast mode: cap universe at {QUICK_UNIVERSE_CAP}")
    parser.add_argument("--csv", type=str, default="",
                        help="Output CSV path")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("US MULTI-EXCHANGE STOCK SCREENER starting")
    log.info("Exchanges: NYSE · NASDAQ · AMEX · ARCA · BATS")
    log.info("=" * 60)

    # 1. Market regime
    regime = check_market_regime()
    log.info("Market regime: %s  (VIX-proxy %.1f%%)",
             regime["trend"], regime.get("vix_proxy") or 0)

    # 2. Fetch universe
    tickers = fetch_all_us_tickers()
    log.info("Universe: %d tickers", len(tickers))

    if args.quick:
        tickers = tickers[:QUICK_UNIVERSE_CAP]
        log.info("--quick mode: capped to %d tickers", len(tickers))

    # 3. Phase 1 — bulk filter
    survivors = phase1_bulk_filter(tickers, top_n=PHASE1_TOP_N)

    # 4. Phase 2 — deep dive
    results = deep_dive(survivors, count=DEEP_DIVE_COUNT)

    # 5. Report
    print_report(results, regime)
    if args.csv:
        write_csv(results, args.csv)

    log.info("Done.")


if __name__ == "__main__":
    main()
