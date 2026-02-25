#!/usr/bin/env python3
"""
nyse_screener.py — Multi-Factor NYSE Stock Screener
=====================================================
A quantitative research framework for educational purposes.
Screens NYSE / S&P 500 stocks through a 5-layer scoring model
(Catalyst · Options/UOA · Technical · News · Fundamentals)
and surfaces the top 3 candidates over a projected 3-day horizon.

DISCLAIMER: This tool is for **educational and research purposes only**.
It does not constitute financial advice. Always do your own due diligence.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import statistics
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FINVIZ_SCREEN_URL = (
    "https://finviz.com/screener.ashx?"
    "v=111&f=cap_midover,idx_sp500|idx_nyse,sh_avgvol_over500,"
    "sh_relvol_over1,ta_perf_curr_up&ft=4"
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
REQUEST_DELAY = 1.5          # seconds between per-ticker HTTP calls
DEEP_DIVE_COUNT = 30         # analyse top N from the initial screen
FINAL_TOP_N = 3              # surface top N in the report
VIX_WARN_THRESHOLD = 30.0
SP500_TICKER = "^GSPC"
VIX_TICKER = "^VIX"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("screener")


# ====================================================================== #
#  Data-class that carries per-ticker results through the pipeline        #
# ====================================================================== #
@dataclass
class TickerResult:
    symbol: str
    company: str = ""
    price: float = 0.0
    market_cap: str = "N/A"
    volume: int = 0
    avg_volume: int = 0
    sector: str = "N/A"

    # layer scores (filled in progressively)
    catalyst_score: float = 0.0
    catalyst_notes: str = "N/A"

    options_score: float = 0.0
    options_notes: str = "N/A"

    technical_score: float = 0.0
    technical_notes: str = "N/A"

    news_score: float = 0.0
    news_notes: str = "N/A"

    fundamentals_score: float = 0.0
    fundamentals_notes: str = "N/A"

    @property
    def total_score(self) -> float:
        return (
            self.catalyst_score
            + self.options_score
            + self.technical_score
            + self.news_score
            + self.fundamentals_score
        )


# ====================================================================== #
#  Helper – polite requests with retry                                    #
# ====================================================================== #
def _get(url: str, params: dict | None = None, retries: int = 2,
         timeout: int = 15) -> Optional[requests.Response]:
    """GET with retry & back-off.  Returns None on total failure."""
    for attempt in range(retries + 1):
        try:
            resp = requests.get(
                url, headers=HEADERS, params=params, timeout=timeout
            )
            if resp.status_code == 200:
                return resp
            log.warning("HTTP %s for %s (attempt %d)", resp.status_code, url, attempt + 1)
        except requests.RequestException as exc:
            log.warning("Request error for %s: %s (attempt %d)", url, exc, attempt + 1)
        time.sleep(2 * (attempt + 1))
    return None


# ====================================================================== #
#  0. Market Regime Gate                                                  #
# ====================================================================== #
def check_market_regime() -> Tuple[bool, str]:
    """
    Checks the S&P 500 trend (price vs 200-SMA) and the VIX level.
    Returns (ok: bool, message: str).
    If VIX > 30 or S&P below 200 SMA  →  ok=False with warning.
    """
    log.info("Checking market regime (S&P 500 trend + VIX) ...")
    regime_ok = True
    notes: list[str] = []

    try:
        sp = yf.Ticker(SP500_TICKER)
        hist = sp.history(period="1y")
        if hist.empty:
            return True, "⚠  Could not fetch S&P 500 data – proceeding with caution."
        sma200 = hist["Close"].rolling(200).mean().iloc[-1]
        last_close = hist["Close"].iloc[-1]
        if last_close < sma200:
            regime_ok = False
            notes.append(
                f"S&P 500 ({last_close:,.0f}) is BELOW its 200-SMA ({sma200:,.0f})."
            )
        else:
            notes.append(
                f"S&P 500 ({last_close:,.0f}) is above its 200-SMA ({sma200:,.0f})."
            )
    except Exception as exc:
        notes.append(f"S&P 500 check failed: {exc}")

    try:
        vix = yf.Ticker(VIX_TICKER)
        vhist = vix.history(period="5d")
        if not vhist.empty:
            vix_last = vhist["Close"].iloc[-1]
            if vix_last > VIX_WARN_THRESHOLD:
                regime_ok = False
                notes.append(f"VIX is elevated at {vix_last:.2f} (threshold {VIX_WARN_THRESHOLD}).")
            else:
                notes.append(f"VIX at {vix_last:.2f} — within normal range.")
        else:
            notes.append("VIX data unavailable.")
    except Exception as exc:
        notes.append(f"VIX check failed: {exc}")

    msg = " | ".join(notes)
    if not regime_ok:
        msg = "⚠  CAUTION — Adverse market regime detected. " + msg
    else:
        msg = "✅  Market regime appears constructive. " + msg
    return regime_ok, msg


# ====================================================================== #
#  1. Initial Screen — Finviz scraper                                     #
# ====================================================================== #
def _parse_finviz_page(url: str) -> List[Dict[str, str]]:
    """Scrape one page of Finviz screener results."""
    resp = _get(url)
    if resp is None:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    rows: list[dict[str, str]] = []

    # Finviz stores results in a table with id 'screener-views-table'
    # or inside <table … class="screener_table"> — layout varies.
    # We look for all <tr> inside the main results table.
    table = soup.find("table", class_="screener_table") or soup.find(
        "table", {"id": "screener-views-table"}
    )
    if table is None:
        # fallback: grab the largest table on the page
        tables = soup.find_all("table")
        if not tables:
            return rows
        table = max(tables, key=lambda t: len(t.find_all("tr")))

    header_cells = table.find_all("td", class_="table-top") or []
    if not header_cells:
        # try <th>
        header_cells = table.find_all("th")
    headers = [c.get_text(strip=True) for c in header_cells]

    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        # skip header row or rows not matching column count
        if len(tds) != len(headers) or len(tds) < 2:
            continue
        vals = [td.get_text(strip=True) for td in tds]
        row = dict(zip(headers, vals))
        if row.get("Ticker") or row.get("No."):
            rows.append(row)
    return rows


def scrape_finviz_screen(max_tickers: int = 200) -> pd.DataFrame:
    """
    Paginate through Finviz screener and return a DataFrame of results.
    Finviz pages results in batches of 20 (r=1, r=21, r=41 …).
    """
    log.info("Scraping Finviz screener for initial candidates …")
    all_rows: list[dict] = []
    page = 1
    while len(all_rows) < max_tickers:
        url = FINVIZ_SCREEN_URL + f"&r={page}"
        rows = _parse_finviz_page(url)
        if not rows:
            break
        all_rows.extend(rows)
        page += 20
        time.sleep(REQUEST_DELAY)

    if not all_rows:
        log.warning("Finviz returned 0 rows — will fall back to a default universe.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # normalise column names
    df.columns = [c.strip() for c in df.columns]
    log.info("Finviz returned %d candidates.", len(df))
    return df


def build_initial_universe(finviz_df: pd.DataFrame) -> List[str]:
    """
    Return a list of ticker symbols from the Finviz screen.
    Falls back to a well-known set if scraping fails.
    """
    FALLBACK = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "JPM",
        "V", "UNH", "JNJ", "WMT", "PG", "MA", "HD", "XOM", "LLY", "ABBV",
        "MRK", "PEP", "KO", "COST", "AVGO", "TMO", "MCD", "CRM", "ACN",
        "NFLX", "LIN", "AMD",
    ]
    if finviz_df.empty:
        log.info("Using fallback universe of %d tickers.", len(FALLBACK))
        return FALLBACK

    ticker_col = None
    for candidate in ("Ticker", "ticker", "Symbol", "symbol"):
        if candidate in finviz_df.columns:
            ticker_col = candidate
            break
    if ticker_col is None:
        log.warning("Could not locate ticker column in Finviz data — using fallback.")
        return FALLBACK

    tickers = finviz_df[ticker_col].dropna().unique().tolist()
    return tickers[:max(DEEP_DIVE_COUNT * 3, 60)]  # generous initial pool


# ====================================================================== #
#  2A. Catalyst Layer  (25 pts)                                           #
# ====================================================================== #
def _upcoming_earnings_check(ticker_obj: yf.Ticker) -> Tuple[bool, str]:
    """Check if earnings date falls within the next 14 days."""
    try:
        cal = ticker_obj.calendar
        if cal is None:
            return False, ""
        # yfinance returns a dict or DataFrame depending on version
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
        elif isinstance(cal, pd.DataFrame):
            dates = cal.loc["Earnings Date"].tolist() if "Earnings Date" in cal.index else []
        else:
            dates = []
        now = dt.datetime.now(tz=dt.timezone.utc)
        for d in dates:
            if isinstance(d, str):
                d = pd.Timestamp(d)
            if hasattr(d, "date"):
                diff = (d - now).days if hasattr(d, "days") else (d.date() - now.date()).days
                if 0 <= diff <= 14:
                    return True, f"Earnings in ~{diff}d"
    except Exception:
        pass
    return False, ""


def _insider_buy_signal(symbol: str) -> Tuple[bool, str]:
    """Quick check for recent insider purchases via OpenInsider."""
    url = f"http://openinsider.com/screener?s={symbol}&o=&pl=&ph=&ll=&lh=&fd=30&fdr=&td=0&tdr=&feession=&fq=&fquarter=&sid=&ta=1&tl=&tc=&tr=&tdiv=&sig=&sod=68&sfl=&sio=&sit=&rp=1"
    resp = _get(url, timeout=10)
    if resp is None:
        return False, ""
    soup = BeautifulSoup(resp.text, "html.parser")
    buy_table = soup.find("table", class_="tinytable")
    if buy_table:
        rows = buy_table.find_all("tr")
        buy_count = max(0, len(rows) - 1)  # minus header
        if buy_count >= 2:
            return True, f"{buy_count} insider buys (30d)"
    return False, ""


def score_catalyst(result: TickerResult, ticker_obj: yf.Ticker) -> None:
    """Catalyst layer — up to 25 points."""
    points = 0.0
    notes_parts: list[str] = []

    # 1) Earnings catalyst (0-10 pts)
    has_earnings, e_note = _upcoming_earnings_check(ticker_obj)
    if has_earnings:
        points += 10
        notes_parts.append(e_note)

    # 2) Insider buying (0-8 pts)
    has_insider, i_note = _insider_buy_signal(result.symbol)
    if has_insider:
        points += 8
        notes_parts.append(i_note)
    time.sleep(REQUEST_DELAY)

    # 3) Recent major news / M&A / FDA — keyword scan via Yahoo Finance news
    try:
        news_items = ticker_obj.news or []
        catalyst_keywords = [
            "fda", "approval", "acquisition", "merger", "buyout",
            "upgrade", "initiate", "partnership", "contract", "award",
        ]
        catalyst_hits = 0
        for item in news_items[:10]:
            title = (item.get("title") or item.get("content", {}).get("title", "")).lower()
            if any(kw in title for kw in catalyst_keywords):
                catalyst_hits += 1
        cat_pts = min(catalyst_hits * 3.5, 7)
        points += cat_pts
        if catalyst_hits:
            notes_parts.append(f"{catalyst_hits} catalyst headline(s)")
    except Exception:
        pass

    result.catalyst_score = min(points, 25)
    result.catalyst_notes = "; ".join(notes_parts) if notes_parts else "No major catalysts detected"


# ====================================================================== #
#  2B. Options / Unusual Options Activity  (20 pts)                       #
# ====================================================================== #
def score_options(result: TickerResult, ticker_obj: yf.Ticker) -> None:
    """
    Options / UOA layer — up to 20 points.
    Uses yfinance options chain data as a proxy for unusual activity.
    """
    points = 0.0
    notes_parts: list[str] = []

    try:
        expirations = ticker_obj.options
        if not expirations:
            result.options_score = 0
            result.options_notes = "No options data available"
            return

        # Analyse the nearest expiration
        nearest_exp = expirations[0]
        chain = ticker_obj.option_chain(nearest_exp)
        calls = chain.calls
        puts = chain.puts

        if calls.empty and puts.empty:
            result.options_score = 0
            result.options_notes = "Options chain empty"
            return

        # --- Put/Call volume ratio ---
        total_call_vol = calls["volume"].sum() if "volume" in calls.columns else 0
        total_put_vol = puts["volume"].sum() if "volume" in puts.columns else 0
        total_call_vol = 0 if pd.isna(total_call_vol) else total_call_vol
        total_put_vol = 0 if pd.isna(total_put_vol) else total_put_vol

        if total_put_vol > 0:
            pc_ratio = total_call_vol / total_put_vol
        else:
            pc_ratio = float("inf") if total_call_vol > 0 else 1.0

        # High call-to-put ratio is potentially constructive
        if pc_ratio > 3.0:
            points += 8
            notes_parts.append(f"Call/Put vol ratio {pc_ratio:.1f}")
        elif pc_ratio > 1.5:
            points += 5
            notes_parts.append(f"Call/Put vol ratio {pc_ratio:.1f}")

        # --- Unusual volume on OTM calls ---
        if not calls.empty and "volume" in calls.columns and "openInterest" in calls.columns:
            calls_clean = calls.dropna(subset=["volume", "openInterest"])
            if not calls_clean.empty:
                calls_clean = calls_clean[calls_clean["openInterest"] > 0]
                if not calls_clean.empty:
                    calls_clean = calls_clean.copy()
                    calls_clean["vol_oi"] = calls_clean["volume"] / calls_clean["openInterest"]
                    hot = calls_clean[calls_clean["vol_oi"] > 2.0]
                    if len(hot) >= 3:
                        points += 7
                        notes_parts.append(f"{len(hot)} strikes with Vol/OI > 2")
                    elif len(hot) >= 1:
                        points += 4
                        notes_parts.append(f"{len(hot)} strike(s) with Vol/OI > 2")

        # --- Implied-volatility skew (simple heuristic) ---
        if "impliedVolatility" in calls.columns:
            avg_call_iv = calls["impliedVolatility"].mean()
            avg_put_iv = puts["impliedVolatility"].mean() if "impliedVolatility" in puts.columns else avg_call_iv
            if not (pd.isna(avg_call_iv) or pd.isna(avg_put_iv)):
                skew = avg_put_iv - avg_call_iv
                if skew > 0.10:
                    points += 5
                    notes_parts.append(f"IV skew (puts premium): {skew:.2f}")
                elif skew > 0.03:
                    points += 2
                    notes_parts.append(f"Mild IV skew: {skew:.2f}")
    except Exception as exc:
        notes_parts.append(f"Options analysis error: {exc}")

    result.options_score = min(points, 20)
    result.options_notes = "; ".join(notes_parts) if notes_parts else "N/A"


# ====================================================================== #
#  2C. Technical Layer  (30 pts)                                          #
# ====================================================================== #
def _compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    last = rsi.iloc[-1]
    return float(last) if not pd.isna(last) else 50.0


def _compute_macd(close: pd.Series) -> Tuple[float, float]:
    """Return (MACD line, signal line) last values."""
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return float(macd_line.iloc[-1]), float(signal.iloc[-1])


def score_technical(result: TickerResult, ticker_obj: yf.Ticker) -> None:
    """Technical layer — up to 30 points (RSI, MAs, MACD, momentum)."""
    points = 0.0
    notes_parts: list[str] = []

    try:
        hist = ticker_obj.history(period="6mo")
        if hist.empty or len(hist) < 50:
            result.technical_score = 0
            result.technical_notes = "Insufficient price history"
            return

        close = hist["Close"]
        last_price = close.iloc[-1]

        # --- RSI (0-8 pts) ---
        rsi = _compute_rsi(close)
        if 40 <= rsi <= 65:
            points += 8
            notes_parts.append(f"RSI {rsi:.1f} (constructive zone)")
        elif 30 <= rsi < 40:
            points += 6
            notes_parts.append(f"RSI {rsi:.1f} (nearing oversold)")
        elif 65 < rsi <= 75:
            points += 4
            notes_parts.append(f"RSI {rsi:.1f} (strong momentum)")
        else:
            points += 1
            notes_parts.append(f"RSI {rsi:.1f}")

        # --- Moving Averages (0-10 pts) ---
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        ema9 = close.ewm(span=9, adjust=False).mean().iloc[-1]

        above_count = sum([
            last_price > sma20,
            last_price > sma50,
            last_price > ema9,
            sma20 > sma50,   # golden alignment
        ])
        ma_pts = min(above_count * 2.5, 10)
        points += ma_pts
        notes_parts.append(f"Above {above_count}/3 MAs, SMA20{'>' if sma20 > sma50 else '<'}SMA50")

        # --- MACD (0-6 pts) ---
        macd_val, signal_val = _compute_macd(close)
        if macd_val > signal_val and macd_val > 0:
            points += 6
            notes_parts.append("MACD bullish crossover (above zero)")
        elif macd_val > signal_val:
            points += 4
            notes_parts.append("MACD bullish crossover (below zero)")
        elif macd_val > 0:
            points += 2
            notes_parts.append("MACD positive but weakening")

        # --- 5-day momentum (0-6 pts) ---
        if len(close) >= 6:
            mom_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100
            if 1 <= mom_5d <= 8:
                points += 6
                notes_parts.append(f"5d momentum +{mom_5d:.1f}%")
            elif 0 < mom_5d < 1:
                points += 3
                notes_parts.append(f"5d momentum +{mom_5d:.1f}%")
            elif mom_5d > 8:
                points += 2  # might be overextended
                notes_parts.append(f"5d momentum +{mom_5d:.1f}% (extended)")
            else:
                notes_parts.append(f"5d momentum {mom_5d:.1f}%")

    except Exception as exc:
        notes_parts.append(f"Technical analysis error: {exc}")

    result.technical_score = min(points, 30)
    result.technical_notes = "; ".join(notes_parts) if notes_parts else "N/A"


# ====================================================================== #
#  2D. News / Sentiment Layer  (15 pts)                                   #
# ====================================================================== #
# Simple keyword-based sentiment — no external NLP API required.
_POS_WORDS = {
    "beat", "beats", "surpass", "upgrade", "bullish", "outperform",
    "record", "growth", "strong", "positive", "boost", "surge",
    "raises", "raise", "expand", "innovative", "win", "award",
    "partnership", "launch", "approval", "profit", "upside",
}
_NEG_WORDS = {
    "miss", "downgrade", "bearish", "underperform", "decline",
    "loss", "weak", "negative", "cut", "recall", "lawsuit",
    "layoff", "investigation", "fraud", "debt", "warning",
    "concern", "risk", "downturn", "falling", "crash",
}


def _simple_sentiment(text: str) -> float:
    """Return a score in [-1, 1] based on keyword frequency."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    pos = len(words & _POS_WORDS)
    neg = len(words & _NEG_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def score_news(result: TickerResult, ticker_obj: yf.Ticker) -> None:
    """News sentiment layer — up to 15 points."""
    points = 0.0
    notes_parts: list[str] = []

    try:
        news_items = ticker_obj.news or []
        if not news_items:
            result.news_score = 0
            result.news_notes = "No recent news found"
            return

        sentiments: list[float] = []
        for item in news_items[:15]:
            title = item.get("title") or item.get("content", {}).get("title", "")
            snippet = item.get("summary") or item.get("content", {}).get("summary", "")
            text = f"{title} {snippet}"
            sentiments.append(_simple_sentiment(text))

        if sentiments:
            avg_sent = statistics.mean(sentiments)
            positive_ratio = sum(1 for s in sentiments if s > 0) / len(sentiments)

            if avg_sent > 0.3:
                points += 10
                notes_parts.append(f"Strongly positive news ({avg_sent:+.2f})")
            elif avg_sent > 0.1:
                points += 7
                notes_parts.append(f"Positive news ({avg_sent:+.2f})")
            elif avg_sent > -0.1:
                points += 4
                notes_parts.append(f"Neutral news ({avg_sent:+.2f})")
            else:
                points += 0
                notes_parts.append(f"Negative news ({avg_sent:+.2f})")

            # Bonus for consistency
            if positive_ratio >= 0.7:
                points += 5
                notes_parts.append(f"{positive_ratio:.0%} headlines positive")
        else:
            notes_parts.append("Could not parse news sentiment")

    except Exception as exc:
        notes_parts.append(f"News analysis error: {exc}")

    result.news_score = min(points, 15)
    result.news_notes = "; ".join(notes_parts) if notes_parts else "N/A"


# ====================================================================== #
#  2E. Fundamentals Layer  (10 pts)                                       #
# ====================================================================== #
def _parse_market_cap_value(mc_str: str) -> float:
    """Convert '1.2T', '500B', '80M' → float."""
    mc_str = mc_str.strip().upper()
    multipliers = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3}
    for suffix, mult in multipliers.items():
        if mc_str.endswith(suffix):
            try:
                return float(mc_str[:-1]) * mult
            except ValueError:
                return 0.0
    try:
        return float(mc_str.replace(",", ""))
    except ValueError:
        return 0.0


def score_fundamentals(result: TickerResult, ticker_obj: yf.Ticker) -> None:
    """Fundamentals layer — up to 10 points (market cap, volume, D/E)."""
    points = 0.0
    notes_parts: list[str] = []

    try:
        info = ticker_obj.info or {}

        # Market cap (0-3 pts) — prefer large/mega caps for stability
        mcap = info.get("marketCap", 0) or 0
        if mcap >= 100e9:
            points += 3
            notes_parts.append("Mega-cap")
        elif mcap >= 10e9:
            points += 2
            notes_parts.append("Large-cap")
        elif mcap >= 2e9:
            points += 1
            notes_parts.append("Mid-cap")
        else:
            notes_parts.append("Small-cap")

        result.market_cap = (
            f"${mcap / 1e9:.1f}B" if mcap >= 1e9 else f"${mcap / 1e6:.0f}M"
        ) if mcap else "N/A"

        # Volume vs average (0-4 pts)
        vol = info.get("volume", 0) or 0
        avg_vol = info.get("averageVolume", 1) or 1
        result.volume = vol
        result.avg_volume = avg_vol
        rel_vol = vol / avg_vol if avg_vol else 1
        if rel_vol >= 2.0:
            points += 4
            notes_parts.append(f"Rel vol {rel_vol:.1f}x")
        elif rel_vol >= 1.3:
            points += 2
            notes_parts.append(f"Rel vol {rel_vol:.1f}x")
        else:
            notes_parts.append(f"Rel vol {rel_vol:.1f}x")

        # Debt / Equity (0-3 pts)
        de = info.get("debtToEquity")
        if de is not None:
            de_val = float(de)
            if de_val < 50:
                points += 3
                notes_parts.append(f"D/E {de_val:.0f}%")
            elif de_val < 100:
                points += 2
                notes_parts.append(f"D/E {de_val:.0f}%")
            elif de_val < 200:
                points += 1
                notes_parts.append(f"D/E {de_val:.0f}%")
            else:
                notes_parts.append(f"D/E {de_val:.0f}% (high)")
        else:
            notes_parts.append("D/E N/A")

        result.company = info.get("shortName") or info.get("longName") or result.symbol
        result.sector = info.get("sector", "N/A")
        result.price = info.get("currentPrice") or info.get("regularMarketPrice") or 0

    except Exception as exc:
        notes_parts.append(f"Fundamentals error: {exc}")

    result.fundamentals_score = min(points, 10)
    result.fundamentals_notes = "; ".join(notes_parts) if notes_parts else "N/A"


# ====================================================================== #
#  3. Deep-Dive Pipeline                                                  #
# ====================================================================== #
def analyse_ticker(symbol: str) -> TickerResult:
    """Run full 5-layer analysis on a single ticker."""
    result = TickerResult(symbol=symbol)
    try:
        ticker_obj = yf.Ticker(symbol)

        # Order matters for rate-limiting — fundamentals first (cached by yfinance)
        score_fundamentals(result, ticker_obj)
        score_technical(result, ticker_obj)
        score_catalyst(result, ticker_obj)
        score_options(result, ticker_obj)
        score_news(result, ticker_obj)
    except Exception as exc:
        log.error("Fatal error analysing %s: %s", symbol, exc)

    return result


def deep_dive(tickers: List[str], top_n: int = DEEP_DIVE_COUNT) -> List[TickerResult]:
    """Analyse *top_n* tickers and return sorted results."""
    subset = tickers[:top_n]
    log.info("Starting deep-dive analysis on %d tickers …", len(subset))

    results: list[TickerResult] = []
    for i, sym in enumerate(subset, 1):
        log.info("[%d/%d] Analysing %s …", i, len(subset), sym)
        res = analyse_ticker(sym)
        results.append(res)
        time.sleep(REQUEST_DELAY)

    results.sort(key=lambda r: r.total_score, reverse=True)
    return results


# ====================================================================== #
#  4. Reporting / Pretty Output                                           #
# ====================================================================== #
def print_header(text: str) -> None:
    width = 72
    print("\n" + "=" * width)
    print(f" {text}".center(width))
    print("=" * width)


def print_report(results: List[TickerResult], regime_msg: str,
                 final_n: int = FINAL_TOP_N) -> None:
    """Print the final scored report to stdout."""
    print_header("NYSE MULTI-FACTOR STOCK SCREENER")
    print(f"  Run date : {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Horizon  : 3-day analytical window")
    print(f"  Universe : {len(results)} tickers analysed")
    print(f"\n  Market Regime: {regime_msg}")

    print_header(f"TOP {final_n} CANDIDATES")

    top = results[:final_n]
    for rank, r in enumerate(top, 1):
        print(f"\n  ── #{rank}  {r.symbol}  ({r.company}) {'─' * 30}")
        print(f"  Price: ${r.price:,.2f}   Mkt Cap: {r.market_cap}   Sector: {r.sector}")
        print(f"  ┌────────────────────────┬────────┐")
        print(f"  │  Layer                 │ Score  │")
        print(f"  ├────────────────────────┼────────┤")
        print(f"  │  Catalyst   (max 25)   │ {r.catalyst_score:5.1f}  │  {r.catalyst_notes}")
        print(f"  │  Options    (max 20)   │ {r.options_score:5.1f}  │  {r.options_notes}")
        print(f"  │  Technical  (max 30)   │ {r.technical_score:5.1f}  │  {r.technical_notes}")
        print(f"  │  News       (max 15)   │ {r.news_score:5.1f}  │  {r.news_notes}")
        print(f"  │  Fundamental(max 10)   │ {r.fundamentals_score:5.1f}  │  {r.fundamentals_notes}")
        print(f"  ├────────────────────────┼────────┤")
        print(f"  │  TOTAL      (max 100)  │ {r.total_score:5.1f}  │")
        print(f"  └────────────────────────┴────────┘")

    # Summary table of all analysed tickers
    print_header("FULL RANKINGS (all analysed tickers)")
    print(f"  {'Rank':<5} {'Ticker':<8} {'Total':>6}  {'Cat':>4} {'Opt':>4} {'Tech':>4} {'News':>4} {'Fund':>4}  Company")
    print(f"  {'─'*5} {'─'*8} {'─'*6}  {'─'*4} {'─'*4} {'─'*4} {'─'*4} {'─'*4}  {'─'*20}")
    for i, r in enumerate(results, 1):
        print(
            f"  {i:<5} {r.symbol:<8} {r.total_score:6.1f}  "
            f"{r.catalyst_score:4.1f} {r.options_score:4.1f} "
            f"{r.technical_score:4.1f} {r.news_score:4.1f} "
            f"{r.fundamentals_score:4.1f}  {r.company[:28]}"
        )

    print("\n" + "=" * 72)
    print("  DISCLAIMER: This output is for EDUCATIONAL and RESEARCH purposes")
    print("  only. It is NOT financial advice. Always perform your own analysis")
    print("  and consult a licensed professional before making decisions.")
    print("=" * 72 + "\n")


# ====================================================================== #
#  Main entry-point                                                       #
# ====================================================================== #
def main() -> None:
    start = time.time()

    # 0. Market regime check
    regime_ok, regime_msg = check_market_regime()
    print(f"\n{'─'*72}")
    print(f"  {regime_msg}")
    print(f"{'─'*72}\n")
    if not regime_ok:
        log.warning("Adverse regime detected — results should be interpreted with extra caution.")

    # 1. Build initial universe from Finviz screen
    finviz_df = scrape_finviz_screen()
    universe = build_initial_universe(finviz_df)
    log.info("Initial universe: %d tickers", len(universe))

    # 2. Deep-dive on top candidates
    results = deep_dive(universe, top_n=DEEP_DIVE_COUNT)

    # 3. Report
    print_report(results, regime_msg, final_n=FINAL_TOP_N)

    elapsed = time.time() - start
    log.info("Screening completed in %.1f seconds.", elapsed)


if __name__ == "__main__":
    main()
