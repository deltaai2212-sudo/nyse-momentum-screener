#!/usr/bin/env python3
"""
nyse_screener.py — Full-Universe NYSE Multi-Factor Stock Screener
==================================================================
A quantitative research framework for educational purposes.

Architecture
────────────
  Phase 0 : Market regime gate  (S&P 500 trend + VIX)
  Phase 1 : Fetch ALL NYSE tickers (2,400+) from multiple sources
  Phase 2 : Fast bulk filter — RSI, MA, momentum, rel-vol → top 150
  Phase 3 : Full 5-layer deep dive on the 150 survivors
            Catalyst (25) · Options (20) · Technical (30)
            News (15) · Fundamentals (10)
  Phase 4 : Report — top 3 scorecards + full ranking table

Designed to run inside a GitHub Actions job (≤ 6 h wall-clock).

DISCLAIMER: This tool is for **educational and research purposes only**.
It does not constitute financial advice.  Always do your own due diligence.
"""

from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import logging
import math
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

# ======================================================================
# Configuration
# ======================================================================
# -- Sources -----------------------------------------------------------
NYSE_SOURCE = "ftp"  # "ftp" | "finviz" | "fallback"

FINVIZ_SCREEN_URL = (
    "https://finviz.com/screener.ashx?"
    "v=111&f=cap_midover,idx_sp500|idx_nyse,sh_avgvol_over500,"
    "sh_relvol_over1,ta_perf_curr_up&ft=4"
)
FINVIZ_NYSE_ALL_URL = (
    "https://finviz.com/screener.ashx?v=111&f=exch_nyse&ft=4"
)
NASDAQ_FTP_OTHER = (
    "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# -- Phase 1 (bulk filter) --------------------------------------------
PHASE1_BATCH_SIZE = 50        # tickers per yfinance bulk download
PHASE1_PERIOD = "3mo"         # OHLCV lookback for the fast screen
PHASE1_TOP_N = 150            # survivors forwarded to Phase 2
PHASE1_MIN_ROWS = 30          # minimum trading days required

# -- Phase 2 (deep dive) ----------------------------------------------
DEEP_DIVE_COUNT = 150         # same as PHASE1_TOP_N
FINAL_TOP_N = 3               # surface top N in the report
REQUEST_DELAY = 1.0           # seconds between per-ticker API calls

# -- Market regime -----------------------------------------------------
VIX_WARN_THRESHOLD = 30.0
SP500_TICKER = "^GSPC"
VIX_TICKER = "^VIX"

# -- Timeout protection (GitHub Actions) --------------------------------
MAX_PHASE1_MINUTES = 20
MAX_TOTAL_SECONDS = 5 * 3600  # 5 hours hard ceiling

# -- Quick mode (--quick flag) -----------------------------------------
QUICK_UNIVERSE_CAP = 200

# ======================================================================
# Logging
# ======================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("screener")

# Global start time — used for timeout checks
_GLOBAL_START: float = 0.0


def _elapsed() -> float:
    """Seconds since the script started."""
    return time.time() - _GLOBAL_START


def _timeout_reached() -> bool:
    return _elapsed() > MAX_TOTAL_SECONDS


# ======================================================================
# TickerResult dataclass
# ======================================================================
@dataclass
class TickerResult:
    symbol: str
    company: str = ""
    price: float = 0.0
    market_cap: str = "N/A"
    volume: int = 0
    avg_volume: int = 0
    sector: str = "N/A"

    # Phase 1 fast score
    fast_score: float = 0.0

    # Phase 2 layer scores
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


# ======================================================================
# Helper — polite requests with retry
# ======================================================================
def _get(
    url: str,
    params: dict | None = None,
    retries: int = 2,
    timeout: int = 15,
) -> Optional[requests.Response]:
    """GET with retry and back-off.  Returns None on total failure."""
    for attempt in range(retries + 1):
        try:
            resp = requests.get(
                url, headers=HEADERS, params=params, timeout=timeout
            )
            if resp.status_code == 200:
                return resp
            log.warning(
                "HTTP %s for %s (attempt %d)",
                resp.status_code, url[:80], attempt + 1,
            )
        except requests.RequestException as exc:
            log.warning(
                "Request error for %s: %s (attempt %d)",
                url[:80], exc, attempt + 1,
            )
        time.sleep(2 * (attempt + 1))
    return None


# ======================================================================
# Ticker validation helper
# ======================================================================
_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")


def _is_valid_ticker(sym: str) -> bool:
    """Return True for plausible equity tickers (1-5 uppercase letters)."""
    return bool(_TICKER_RE.match(sym))


# ======================================================================
# 0. Market Regime Gate
# ======================================================================
def check_market_regime() -> Tuple[bool, str]:
    """
    Checks the S&P 500 trend (price vs 200-SMA) and the VIX level.
    Returns (ok, message).
    """
    log.info("Phase 0 — Checking market regime (S&P 500 trend + VIX) ...")
    regime_ok = True
    notes: list[str] = []

    try:
        sp = yf.Ticker(SP500_TICKER)
        hist = sp.history(period="1y")
        if hist.empty:
            return True, "⚠  Could not fetch S&P 500 data — proceeding with caution."
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
                notes.append(
                    f"VIX is elevated at {vix_last:.2f} (threshold {VIX_WARN_THRESHOLD})."
                )
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


# ======================================================================
# 1. Fetch ALL NYSE Tickers (multiple sources)
# ======================================================================

# --- 1a. NASDAQ FTP (otherlisted.txt) ---------------------------------
def _fetch_nyse_from_ftp() -> List[str]:
    """
    Download the NASDAQ 'otherlisted.txt' file which lists non-NASDAQ
    securities.  Rows where Exchange == 'N' are NYSE-listed equities.
    """
    log.info("  [FTP] Fetching NYSE tickers from nasdaqtrader.com ...")
    resp = _get(NASDAQ_FTP_OTHER, timeout=20)
    if resp is None:
        log.warning("  [FTP] Download failed.")
        return []

    try:
        # The file is pipe-delimited with a header row and a trailing
        # 'File Creation Time' footer line.
        lines = resp.text.strip().split("\n")
        # Remove footer
        lines = [l for l in lines if not l.startswith("File Creation Time")]
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|")

        # Columns: ACT Symbol | Security Name | Exchange | ...
        # Exchange == 'N' -> NYSE,  'A' -> AMEX,  'P' -> ARCA  etc.
        col_exchange = None
        for c in df.columns:
            if "exchange" in c.lower():
                col_exchange = c
                break
        col_symbol = None
        for c in df.columns:
            if "symbol" in c.lower():
                col_symbol = c
                break
            if c.strip() == "ACT Symbol":
                col_symbol = c
                break

        if col_exchange is None or col_symbol is None:
            log.warning("  [FTP] Unexpected columns: %s", list(df.columns))
            # Try to get all symbols anyway
            for c in df.columns:
                if "symbol" in c.lower() or "act" in c.lower():
                    tickers = df[c].dropna().astype(str).tolist()
                    tickers = [t.strip() for t in tickers if _is_valid_ticker(t.strip())]
                    log.info("  [FTP] Retrieved %d tickers (all exchanges).", len(tickers))
                    return tickers
            return []

        nyse_df = df[df[col_exchange].str.strip() == "N"]
        tickers = nyse_df[col_symbol].dropna().astype(str).tolist()
        tickers = [t.strip() for t in tickers if _is_valid_ticker(t.strip())]
        log.info("  [FTP] Retrieved %d NYSE tickers.", len(tickers))
        return tickers
    except Exception as exc:
        log.warning("  [FTP] Parse error: %s", exc)
        return []


# --- 1b. Finviz full NYSE screen (all pages) -------------------------
def _parse_finviz_page(url: str) -> List[Dict[str, str]]:
    """Scrape one page of Finviz screener results."""
    resp = _get(url)
    if resp is None:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    rows: list[dict[str, str]] = []

    table = soup.find("table", class_="screener_table") or soup.find(
        "table", {"id": "screener-views-table"}
    )
    if table is None:
        tables = soup.find_all("table")
        if not tables:
            return rows
        table = max(tables, key=lambda t: len(t.find_all("tr")))

    header_cells = table.find_all("td", class_="table-top") or []
    if not header_cells:
        header_cells = table.find_all("th")
    headers = [c.get_text(strip=True) for c in header_cells]

    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) != len(headers) or len(tds) < 2:
            continue
        vals = [td.get_text(strip=True) for td in tds]
        row = dict(zip(headers, vals))
        if row.get("Ticker") or row.get("No."):
            rows.append(row)
    return rows


def scrape_finviz_screen(
    base_url: str = FINVIZ_NYSE_ALL_URL,
    max_tickers: int = 5000,
    label: str = "NYSE-all",
) -> pd.DataFrame:
    """
    Paginate through Finviz screener and return a DataFrame.
    Finviz pages results in batches of 20 (r=1, r=21, r=41 ...).
    """
    log.info("  [Finviz] Scraping %s screener ...", label)
    all_rows: list[dict] = []
    page = 1
    empty_streak = 0
    while len(all_rows) < max_tickers:
        url = base_url + f"&r={page}"
        rows = _parse_finviz_page(url)
        if not rows:
            empty_streak += 1
            if empty_streak >= 3:
                break
        else:
            empty_streak = 0
            all_rows.extend(rows)
        page += 20
        time.sleep(REQUEST_DELAY)
        if _timeout_reached():
            log.warning("  [Finviz] Timeout reached during scraping — stopping.")
            break

    if not all_rows:
        log.warning("  [Finviz] Returned 0 rows.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df.columns = [c.strip() for c in df.columns]
    log.info("  [Finviz] Scraped %d rows from %s.", len(df), label)
    return df


def _fetch_nyse_from_finviz() -> List[str]:
    """Scrape all pages of Finviz with idx_nyse (no cap/vol filters)."""
    df = scrape_finviz_screen(
        base_url=FINVIZ_NYSE_ALL_URL, max_tickers=5000, label="NYSE-all"
    )
    if df.empty:
        return []
    for col in ("Ticker", "ticker", "Symbol", "symbol"):
        if col in df.columns:
            tickers = df[col].dropna().unique().tolist()
            tickers = [t.strip() for t in tickers if _is_valid_ticker(t.strip())]
            return tickers
    return []


# --- 1c. Hardcoded fallback ------------------------------------------
_FALLBACK_NYSE_500: List[str] = [
    # Mega / large caps (top ~250 by market cap)
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "BRK-B",
    "JPM", "V", "UNH", "JNJ", "WMT", "PG", "MA", "HD", "XOM", "LLY",
    "ABBV", "MRK", "PEP", "KO", "COST", "AVGO", "TMO", "MCD", "CRM",
    "NFLX", "LIN", "AMD", "ACN", "ABT", "DHR", "TXN", "PM", "NEE",
    "UNP", "RTX", "LOW", "HON", "AMGN", "IBM", "GE", "CAT", "SPGI",
    "INTU", "BA", "ELV", "DE", "ADP", "BLK", "MDLZ", "GILD", "ADI",
    "SYK", "VRTX", "MMC", "CB", "SCHW", "PLD", "CI", "TJX", "BMY",
    "AMT", "SO", "DUK", "MO", "ZTS", "PGR", "CVS", "BSX", "BDX",
    "CME", "ICE", "CL", "SHW", "MCK", "AON", "EQIX", "WM", "NOC",
    "GD", "EMR", "APD", "ITW", "FDX", "ECL", "PH", "ORLY", "AJG",
    "NSC", "TRV", "MSI", "SLB", "AEP", "D", "WELL", "RCL", "PSA",
    "SPG", "ALL", "KMB", "AFL", "AIG", "MET", "PRU", "TFC", "USB",
    "PNC", "COF", "BK", "FITB", "KEY", "CFG", "RF", "HBAN", "MTB",
    "NTRS", "STT", "CMA", "ZION",
    # Industrials / materials
    "MMM", "DOW", "DD", "NEM", "FCX", "CTVA", "WY", "IP", "PKG",
    "AVY", "SEE", "SON", "OLN", "CE", "HUN", "RPM", "EMN",
    # Healthcare
    "HCA", "HUM", "CNC", "DXCM", "IQV", "BAX", "EW", "RMD", "HOLX",
    "MTD", "STE", "WAT", "A", "BIO", "PKI", "TFX", "XRAY",
    # Tech
    "HPQ", "HPE", "DELL", "WDC", "STX", "NTAP", "GLW", "TER", "KEYS",
    "ZBRA", "JNPR", "FFIV",
    # Consumer
    "TGT", "DG", "DLTR", "ROST", "BBY", "GPS", "KSS", "M", "JWN",
    "NKE", "RL", "VFC", "PVH", "TPR", "HBI", "LEVI",
    "SJM", "K", "GIS", "CPB", "CAG", "HSY", "MKC", "HRL", "TSN",
    "BG", "ADM",
    # Energy
    "CVX", "COP", "EOG", "PXD", "PSX", "VLO", "MPC", "OXY", "HAL",
    "BKR", "DVN", "FANG", "HES", "APA", "OVV", "MRO",
    # Utilities
    "SRE", "EXC", "XEL", "ED", "WEC", "ES", "AEE", "CMS", "DTE",
    "FE", "ETR", "PPL", "CNP", "NI", "AES", "PNW", "NRG", "OGE",
    "ATO", "LNT",
    # REITs
    "CCI", "O", "DLR", "VICI", "AVB", "EQR", "MAA", "UDR", "CPT",
    "ESS", "SUI", "ELS", "REG", "FRT", "KIM", "BXP", "SLG", "HIW",
    "VTR", "OHI", "MPW", "PEAK", "HST", "RHP",
    # Financials
    "GS", "MS", "C", "WFC", "BAC", "AXP", "DFS", "SYF", "ALLY",
    "NYCB", "FNB", "SNV", "HWC", "OZK", "BOKF", "FHN", "WAL",
    "WTFC", "PNFP", "IBOC", "UMBF", "CBSH", "GBCI", "ABCB", "SFNC",
    "FULT", "TRMK", "UBSI", "VLY", "FBP",
    # Telecom / Media
    "T", "VZ", "CMCSA", "DIS", "FOX", "FOXA", "PARA", "WBD", "IPG",
    "OMC",
    # Transport
    "CSX", "DAL", "UAL", "LUV", "AAL", "JBHT", "XPO", "CHRW",
    "EXPD", "UPS",
    # Energy infrastructure
    "WMB", "KMI", "ET", "OKE", "TRGP", "EPD", "PAA", "AM", "CTRA",
    "AR", "RRC", "EQT", "SWN", "CNX",
    # Autos
    "GM", "F", "TM", "HMC",
    # Pharma (international ADRs on NYSE)
    "PFE", "AZN", "NVO", "GSK", "SNY", "NVS", "TAK",
    # Consumer staples
    "CLX", "CHD", "EL", "SWK", "LEG", "MAS", "OC", "ALLE", "IR",
    "XYL", "AWK", "WTRG",
    # Industrials
    "AME", "ROK", "ROP", "NDSN", "TRMB", "FTV", "LDOS", "BAH",
    "SAIC", "J", "LHX", "HII", "TXT", "AOS", "SNA", "WSO", "GGG",
    "GNRC",
    # Agriculture / chemicals
    "CF", "MOS", "NTR", "FMC", "ALB",
    # Metals / mining
    "GOLD", "AEM", "KGC", "AGI", "AU", "NUE", "STLD", "RS", "CMC",
    "ATI", "X", "CLF",
    # Leisure / hospitality
    "LVS", "MGM", "WYNN", "CZR", "PENN", "MAR", "HLT", "H", "CHH",
    "CCL", "NCLH",
    # Restaurants
    "YUM", "DPZ", "CMG", "QSR", "WEN", "DRI", "TXRH", "EAT",
    # Mid-caps (additional breadth)
    "GLOB", "FND", "LSCC", "AXON", "TDG", "IRM", "DECK", "GWW",
    "URI", "FAST", "ODFL", "SAIA", "WDAY", "SNOW", "DDOG", "NET",
    "ZS", "CRWD", "PANW", "FTNT", "NOW", "TEAM", "VEEV", "ANSS",
    "CPRT", "CSGP", "VRSK", "CDW", "BR", "TTEK", "PAYC", "WEX",
    "GWRE", "PCOR", "PEGA", "MANH", "BILL", "SHOP", "SQ", "PYPL",
    "FIS", "FISV", "GPN", "WU", "EEFT", "FLYW",
    "WST", "TECH", "ICLR", "MEDP", "CRL",
    "PODD", "ALGN", "ISRG", "IDXX", "RVTY",
    "POOL", "WSM", "RH", "BURL", "FIVE", "ULTA", "LULU",
    "CMI", "DOV", "IEX", "NDSN", "RBC", "WTS", "BMI", "FELE",
    "SITE", "BCPC", "IOSP", "KWR", "ESE",
]


def _get_fallback_tickers() -> List[str]:
    """Return hardcoded list of ~500 major NYSE stocks."""
    seen: set[str] = set()
    out: list[str] = []
    for t in _FALLBACK_NYSE_500:
        s = t.replace(".", "-")  # BRK.B -> BRK-B for yfinance
        if s not in seen and _is_valid_ticker(s):
            seen.add(s)
            out.append(s)
    log.info("  [Fallback] Loaded %d hardcoded NYSE tickers.", len(out))
    return out


def fetch_all_nyse_tickers() -> List[str]:
    """
    Attempt multiple sources in order:
      1. NASDAQ FTP file (otherlisted.txt) — filter Exchange == 'N'
      2. Finviz full NYSE screen (all pages, no cap/vol filters)
      3. Hardcoded fallback of 500+ major NYSE stocks
    Returns a deduplicated list of valid ticker symbols (1,000+ target).
    """
    log.info("Phase 1 — Fetching full NYSE ticker universe ...")
    all_tickers: list[str] = []
    sources_used: list[str] = []

    # Source 1: FTP
    ftp_tickers = _fetch_nyse_from_ftp()
    if ftp_tickers:
        all_tickers.extend(ftp_tickers)
        sources_used.append(f"FTP({len(ftp_tickers)})")

    # Source 2: Finviz (only if FTP gave < 1000)
    if len(set(all_tickers)) < 1000:
        fv_tickers = _fetch_nyse_from_finviz()
        if fv_tickers:
            all_tickers.extend(fv_tickers)
            sources_used.append(f"Finviz({len(fv_tickers)})")

    # Deduplicate
    seen: set[str] = set()
    unique: list[str] = []
    for t in all_tickers:
        t = t.strip().upper().replace(".", "-")
        if t not in seen and _is_valid_ticker(t):
            seen.add(t)
            unique.append(t)

    # Source 3: fallback if still below 1000
    if len(unique) < 1000:
        fb = _get_fallback_tickers()
        for t in fb:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        sources_used.append(f"Fallback({len(fb)})")

    # Also merge in the original momentum-filtered Finviz screen for bonus picks
    try:
        fv_filtered = scrape_finviz_screen(
            base_url=FINVIZ_SCREEN_URL,
            max_tickers=300,
            label="momentum-filter",
        )
        if not fv_filtered.empty:
            for col in ("Ticker", "ticker", "Symbol", "symbol"):
                if col in fv_filtered.columns:
                    for t in fv_filtered[col].dropna().unique():
                        t = t.strip().upper().replace(".", "-")
                        if t not in seen and _is_valid_ticker(t):
                            seen.add(t)
                            unique.append(t)
                    break
    except Exception:
        pass

    log.info(
        "Phase 1 complete — %d unique NYSE tickers collected  [sources: %s]",
        len(unique),
        ", ".join(sources_used) if sources_used else "none",
    )
    return unique


# ======================================================================
# 2. Phase 1 — Fast Bulk Filter
# ======================================================================
def _compute_rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """Vectorised RSI over a whole Series — returns a Series."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _fast_score_dataframe(df: pd.DataFrame) -> Dict[str, float]:
    """
    Given a multi-ticker OHLCV DataFrame (from yf.download with multiple
    tickers), compute a fast composite score (0-100) per ticker.

    Metrics (equal-weight for speed):
      1. RSI-14           -> 0-25 pts  (sweet spot 40-65)
      2. Price vs SMA-20  -> 0-25 pts
      3. Price vs SMA-50  -> 0-25 pts
      4. 5-day momentum   -> 0-25 pts
    """
    scores: dict[str, float] = {}

    # df has MultiIndex columns: (Price, Ticker)
    try:
        close_df = df["Close"] if "Close" in df.columns.get_level_values(0) else df
    except Exception:
        close_df = df

    if isinstance(close_df, pd.Series):
        # Only one ticker — cannot iterate columns
        return scores

    for ticker in close_df.columns:
        try:
            close = close_df[ticker].dropna()
            if len(close) < PHASE1_MIN_ROWS:
                continue

            last = close.iloc[-1]
            if pd.isna(last) or last <= 0:
                continue

            pts = 0.0

            # 1. RSI-14  (0-25)
            rsi_series = _compute_rsi_series(close, 14)
            rsi = rsi_series.iloc[-1]
            if pd.isna(rsi):
                rsi = 50.0
            if 40 <= rsi <= 65:
                pts += 25
            elif 30 <= rsi < 40 or 65 < rsi <= 75:
                pts += 15
            elif rsi > 75:
                pts += 5   # overbought caution
            else:
                pts += 5

            # 2. Price vs SMA-20  (0-25)
            sma20 = close.rolling(20).mean().iloc[-1]
            if not pd.isna(sma20) and sma20 > 0:
                pct_above_20 = (last - sma20) / sma20
                if pct_above_20 > 0.05:
                    pts += 25
                elif pct_above_20 > 0:
                    pts += 18
                elif pct_above_20 > -0.03:
                    pts += 10
                else:
                    pts += 0

            # 3. Price vs SMA-50  (0-25)
            if len(close) >= 50:
                sma50 = close.rolling(50).mean().iloc[-1]
                if not pd.isna(sma50) and sma50 > 0:
                    pct_above_50 = (last - sma50) / sma50
                    if pct_above_50 > 0.05:
                        pts += 25
                    elif pct_above_50 > 0:
                        pts += 18
                    elif pct_above_50 > -0.03:
                        pts += 10
                    else:
                        pts += 0
            else:
                pts += 10  # neutral if not enough data

            # 4. 5-day momentum  (0-25)
            if len(close) >= 6:
                mom = (close.iloc[-1] / close.iloc[-6] - 1) * 100
                if 1 <= mom <= 8:
                    pts += 25
                elif 0 < mom < 1:
                    pts += 15
                elif mom > 8:
                    pts += 10  # extended
                elif mom > -2:
                    pts += 5
                else:
                    pts += 0

            scores[ticker] = pts
        except Exception:
            continue

    return scores


def phase1_bulk_filter(
    tickers: List[str], top_n: int = PHASE1_TOP_N
) -> List[str]:
    """
    Downloads OHLCV data in batches for ALL tickers.
    Computes fast metrics: RSI, MA alignment, momentum.
    Returns top_n tickers sorted by composite fast-score.
    Logs progress every batch.
    """
    log.info(
        "Phase 2 — Fast bulk filter on %d tickers (batch size %d) ...",
        len(tickers), PHASE1_BATCH_SIZE,
    )
    phase1_start = time.time()

    all_scores: dict[str, float] = {}
    n_batches = math.ceil(len(tickers) / PHASE1_BATCH_SIZE)

    for batch_idx in range(n_batches):
        lo = batch_idx * PHASE1_BATCH_SIZE
        hi = min(lo + PHASE1_BATCH_SIZE, len(tickers))
        batch = tickers[lo:hi]

        if batch_idx % 5 == 0 or batch_idx == n_batches - 1:
            log.info(
                "  Phase 1 batch %d/%d  (tickers %d-%d: %s ... %s)  [%d scored so far]",
                batch_idx + 1, n_batches, lo + 1, hi,
                batch[0], batch[-1], len(all_scores),
            )

        try:
            data = yf.download(
                batch,
                period=PHASE1_PERIOD,
                progress=False,
                threads=True,
                group_by="column",
            )
            if data is not None and not data.empty:
                batch_scores = _fast_score_dataframe(data)
                all_scores.update(batch_scores)
        except Exception as exc:
            log.warning("  Batch %d failed: %s", batch_idx + 1, exc)

        # Timeout guard
        phase1_elapsed = time.time() - phase1_start
        if phase1_elapsed > MAX_PHASE1_MINUTES * 60:
            log.warning(
                "  Phase 1 exceeded %d min — stopping early (%d tickers scored).",
                MAX_PHASE1_MINUTES, len(all_scores),
            )
            break
        if _timeout_reached():
            log.warning("  Global timeout reached in Phase 1 — stopping.")
            break

    # Sort and select top N
    ranked = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)

    # If too few passed, take whatever we have
    effective_top_n = top_n
    if len(ranked) < 50 and len(ranked) > 0:
        effective_top_n = len(ranked)
        log.warning(
            "  Only %d tickers scored — taking all of them.", len(ranked)
        )
    elif len(ranked) == 0:
        log.error("  Phase 1 scored 0 tickers.")
        return []

    survivors = [sym for sym, _ in ranked[:effective_top_n]]

    phase1_time = time.time() - phase1_start
    log.info(
        "Phase 1 complete: %d tickers scored -> top %d candidates  (%.1f min)",
        len(all_scores), len(survivors), phase1_time / 60,
    )

    # Log top-10 fast scores for transparency
    for i, (sym, sc) in enumerate(ranked[:10], 1):
        log.info("    fast-rank #%d  %s  %.0f/100", i, sym, sc)

    # Adjust Phase 2 size if Phase 1 was slow
    if phase1_time > MAX_PHASE1_MINUTES * 60 and effective_top_n > 100:
        log.info(
            "  Reducing Phase 2 pool from %d to 100 (time pressure).",
            effective_top_n,
        )
        survivors = survivors[:100]

    return survivors


# ======================================================================
# 3. Phase 2 — Full 5-Layer Deep Dive
# ======================================================================

# --- 3A. Catalyst (25 pts) -------------------------------------------
def _upcoming_earnings_check(ticker_obj: yf.Ticker) -> Tuple[bool, str]:
    """Check if earnings date falls within the next 14 days."""
    try:
        cal = ticker_obj.calendar
        if cal is None:
            return False, ""
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
        elif isinstance(cal, pd.DataFrame):
            dates = (
                cal.loc["Earnings Date"].tolist()
                if "Earnings Date" in cal.index
                else []
            )
        else:
            dates = []
        now = dt.datetime.now(tz=dt.timezone.utc)
        for d in dates:
            if isinstance(d, str):
                d = pd.Timestamp(d)
            if hasattr(d, "date"):
                try:
                    diff = (d.date() - now.date()).days
                except Exception:
                    diff = -1
                if 0 <= diff <= 14:
                    return True, f"Earnings in ~{diff}d"
    except Exception:
        pass
    return False, ""


def _insider_buy_signal(symbol: str) -> Tuple[bool, str]:
    """Quick check for recent insider purchases via OpenInsider."""
    url = (
        f"http://openinsider.com/screener?s={symbol}&o=&pl=&ph=&ll=&lh="
        f"&fd=30&fdr=&td=0&tdr=&feession=&fq=&fquarter=&sid=&ta=1&tl="
        f"&tc=&tr=&tdiv=&sig=&sod=68&sfl=&sio=&sit=&rp=1"
    )
    resp = _get(url, timeout=10)
    if resp is None:
        return False, ""
    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        buy_table = soup.find("table", class_="tinytable")
        if buy_table:
            rows = buy_table.find_all("tr")
            buy_count = max(0, len(rows) - 1)
            if buy_count >= 2:
                return True, f"{buy_count} insider buys (30d)"
    except Exception:
        pass
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

    # 3) Recent major news / M&A / FDA — keyword scan
    try:
        news_items = ticker_obj.news or []
        catalyst_keywords = [
            "fda", "approval", "acquisition", "merger", "buyout",
            "upgrade", "initiate", "partnership", "contract", "award",
        ]
        catalyst_hits = 0
        for item in news_items[:10]:
            title = (
                item.get("title")
                or item.get("content", {}).get("title", "")
            ).lower()
            if any(kw in title for kw in catalyst_keywords):
                catalyst_hits += 1
        cat_pts = min(catalyst_hits * 3.5, 7)
        points += cat_pts
        if catalyst_hits:
            notes_parts.append(f"{catalyst_hits} catalyst headline(s)")
    except Exception:
        pass

    result.catalyst_score = min(points, 25)
    result.catalyst_notes = (
        "; ".join(notes_parts) if notes_parts else "No major catalysts detected"
    )


# --- 3B. Options / UOA (20 pts) --------------------------------------
def score_options(result: TickerResult, ticker_obj: yf.Ticker) -> None:
    """Options / UOA layer — up to 20 points."""
    points = 0.0
    notes_parts: list[str] = []

    try:
        expirations = ticker_obj.options
        if not expirations:
            result.options_score = 0
            result.options_notes = "No options data available"
            return

        nearest_exp = expirations[0]
        chain = ticker_obj.option_chain(nearest_exp)
        calls = chain.calls
        puts = chain.puts

        if calls.empty and puts.empty:
            result.options_score = 0
            result.options_notes = "Options chain empty"
            return

        # Put / Call volume ratio
        total_call_vol = calls["volume"].sum() if "volume" in calls.columns else 0
        total_put_vol = puts["volume"].sum() if "volume" in puts.columns else 0
        total_call_vol = 0 if pd.isna(total_call_vol) else total_call_vol
        total_put_vol = 0 if pd.isna(total_put_vol) else total_put_vol

        if total_put_vol > 0:
            pc_ratio = total_call_vol / total_put_vol
        else:
            pc_ratio = float("inf") if total_call_vol > 0 else 1.0

        if pc_ratio > 3.0:
            points += 8
            notes_parts.append(f"Call/Put vol ratio {pc_ratio:.1f}")
        elif pc_ratio > 1.5:
            points += 5
            notes_parts.append(f"Call/Put vol ratio {pc_ratio:.1f}")

        # Unusual volume on OTM calls
        if (
            not calls.empty
            and "volume" in calls.columns
            and "openInterest" in calls.columns
        ):
            calls_clean = calls.dropna(subset=["volume", "openInterest"])
            if not calls_clean.empty:
                calls_clean = calls_clean[calls_clean["openInterest"] > 0]
                if not calls_clean.empty:
                    calls_clean = calls_clean.copy()
                    calls_clean["vol_oi"] = (
                        calls_clean["volume"] / calls_clean["openInterest"]
                    )
                    hot = calls_clean[calls_clean["vol_oi"] > 2.0]
                    if len(hot) >= 3:
                        points += 7
                        notes_parts.append(f"{len(hot)} strikes with Vol/OI > 2")
                    elif len(hot) >= 1:
                        points += 4
                        notes_parts.append(f"{len(hot)} strike(s) with Vol/OI > 2")

        # Implied-volatility skew
        if "impliedVolatility" in calls.columns:
            avg_call_iv = calls["impliedVolatility"].mean()
            avg_put_iv = (
                puts["impliedVolatility"].mean()
                if "impliedVolatility" in puts.columns
                else avg_call_iv
            )
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


# --- 3C. Technical (30 pts) ------------------------------------------
def _compute_rsi(series: pd.Series, period: int = 14) -> float:
    """Single-value RSI for a close-price Series."""
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
    """Technical layer — up to 30 points."""
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

        # RSI (0-8 pts)
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

        # Moving Averages (0-10 pts)
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        ema9 = close.ewm(span=9, adjust=False).mean().iloc[-1]

        above_count = sum([
            last_price > sma20,
            last_price > sma50,
            last_price > ema9,
            sma20 > sma50,
        ])
        ma_pts = min(above_count * 2.5, 10)
        points += ma_pts
        notes_parts.append(
            f"Above {above_count}/3 MAs, SMA20{'>' if sma20 > sma50 else '<'}SMA50"
        )

        # MACD (0-6 pts)
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

        # 5-day momentum (0-6 pts)
        if len(close) >= 6:
            mom_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100
            if 1 <= mom_5d <= 8:
                points += 6
                notes_parts.append(f"5d momentum +{mom_5d:.1f}%")
            elif 0 < mom_5d < 1:
                points += 3
                notes_parts.append(f"5d momentum +{mom_5d:.1f}%")
            elif mom_5d > 8:
                points += 2
                notes_parts.append(f"5d momentum +{mom_5d:.1f}% (extended)")
            else:
                notes_parts.append(f"5d momentum {mom_5d:.1f}%")

    except Exception as exc:
        notes_parts.append(f"Technical analysis error: {exc}")

    result.technical_score = min(points, 30)
    result.technical_notes = "; ".join(notes_parts) if notes_parts else "N/A"


# --- 3D. News / Sentiment (15 pts) -----------------------------------
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
            title = (
                item.get("title")
                or item.get("content", {}).get("title", "")
            )
            snippet = (
                item.get("summary")
                or item.get("content", {}).get("summary", "")
            )
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

            if positive_ratio >= 0.7:
                points += 5
                notes_parts.append(f"{positive_ratio:.0%} headlines positive")
        else:
            notes_parts.append("Could not parse news sentiment")

    except Exception as exc:
        notes_parts.append(f"News analysis error: {exc}")

    result.news_score = min(points, 15)
    result.news_notes = "; ".join(notes_parts) if notes_parts else "N/A"


# --- 3E. Fundamentals (10 pts) ---------------------------------------
def score_fundamentals(result: TickerResult, ticker_obj: yf.Ticker) -> None:
    """Fundamentals layer — up to 10 points."""
    points = 0.0
    notes_parts: list[str] = []

    try:
        info = ticker_obj.info or {}

        # Market cap (0-3 pts)
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

        result.company = (
            info.get("shortName") or info.get("longName") or result.symbol
        )
        result.sector = info.get("sector", "N/A")
        result.price = (
            info.get("currentPrice") or info.get("regularMarketPrice") or 0
        )

    except Exception as exc:
        notes_parts.append(f"Fundamentals error: {exc}")

    result.fundamentals_score = min(points, 10)
    result.fundamentals_notes = (
        "; ".join(notes_parts) if notes_parts else "N/A"
    )


# ======================================================================
# Analyse single ticker (all 5 layers)
# ======================================================================
def analyse_ticker(symbol: str) -> TickerResult:
    """Run full 5-layer analysis on a single ticker."""
    result = TickerResult(symbol=symbol)
    try:
        ticker_obj = yf.Ticker(symbol)
        score_fundamentals(result, ticker_obj)
        score_technical(result, ticker_obj)
        score_catalyst(result, ticker_obj)
        score_options(result, ticker_obj)
        score_news(result, ticker_obj)
    except Exception as exc:
        log.error("Fatal error analysing %s: %s", symbol, exc)
    return result


def deep_dive(
    tickers: List[str], top_n: int = DEEP_DIVE_COUNT
) -> List[TickerResult]:
    """Analyse *top_n* tickers through the full 5-layer pipeline."""
    subset = tickers[:top_n]
    log.info(
        "Phase 3 — Full 5-layer deep dive on %d tickers ...", len(subset)
    )

    results: list[TickerResult] = []
    for i, sym in enumerate(subset, 1):
        if _timeout_reached():
            log.warning(
                "  Global timeout reached at ticker %d/%d — stopping deep dive.",
                i, len(subset),
            )
            break
        log.info("  [%d/%d] Analysing %s ...", i, len(subset), sym)
        res = analyse_ticker(sym)
        results.append(res)
        time.sleep(REQUEST_DELAY)

    results.sort(key=lambda r: r.total_score, reverse=True)
    return results


# ======================================================================
# 4. Reporting
# ======================================================================
def print_header(text: str) -> None:
    width = 76
    print("\n" + "=" * width)
    print(f" {text}".center(width))
    print("=" * width)


def print_report(
    results: List[TickerResult],
    regime_msg: str,
    total_universe: int,
    phase1_survivors: int,
    final_n: int = FINAL_TOP_N,
) -> None:
    """Print the final scored report to stdout."""
    print_header("NYSE FULL-UNIVERSE MULTI-FACTOR STOCK SCREENER")
    print(f"  Run date    : {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Horizon     : 3-day analytical window")
    print(f"  Universe    : {total_universe:,} NYSE tickers fetched")
    print(f"  Phase 1     : -> {phase1_survivors:,} survivors (fast filter)")
    print(f"  Phase 2     : -> {len(results):,} fully analysed")
    print(f"  Runtime     : {_elapsed():.0f}s ({_elapsed()/60:.1f} min)")
    print(f"\n  Market Regime: {regime_msg}")

    print_header(f"TOP {final_n} CANDIDATES")

    top = results[:final_n]
    for rank, r in enumerate(top, 1):
        divider = "-" * 30
        print(f"\n  -- #{rank}  {r.symbol}  ({r.company}) {divider}")
        print(
            f"  Price: ${r.price:,.2f}   Mkt Cap: {r.market_cap}   Sector: {r.sector}"
        )
        print(f"  +------------------------+--------+")
        print(f"  |  Layer                 | Score  |")
        print(f"  +------------------------+--------+")
        print(
            f"  |  Catalyst   (max 25)   | {r.catalyst_score:5.1f}  |  {r.catalyst_notes}"
        )
        print(
            f"  |  Options    (max 20)   | {r.options_score:5.1f}  |  {r.options_notes}"
        )
        print(
            f"  |  Technical  (max 30)   | {r.technical_score:5.1f}  |  {r.technical_notes}"
        )
        print(
            f"  |  News       (max 15)   | {r.news_score:5.1f}  |  {r.news_notes}"
        )
        print(
            f"  |  Fundamental(max 10)   | {r.fundamentals_score:5.1f}  |  {r.fundamentals_notes}"
        )
        print(f"  +------------------------+--------+")
        print(
            f"  |  TOTAL      (max 100)  | {r.total_score:5.1f}  |"
        )
        print(f"  +------------------------+--------+")

    # Full rankings table
    print_header("FULL RANKINGS (all analysed tickers)")
    print(
        f"  {'Rank':<5} {'Ticker':<8} {'Total':>6}  "
        f"{'Cat':>4} {'Opt':>4} {'Tech':>4} {'News':>4} {'Fund':>4}  Company"
    )
    print(
        f"  {'-'*5} {'-'*8} {'-'*6}  "
        f"{'-'*4} {'-'*4} {'-'*4} {'-'*4} {'-'*4}  {'-'*20}"
    )
    for i, r in enumerate(results, 1):
        print(
            f"  {i:<5} {r.symbol:<8} {r.total_score:6.1f}  "
            f"{r.catalyst_score:4.1f} {r.options_score:4.1f} "
            f"{r.technical_score:4.1f} {r.news_score:4.1f} "
            f"{r.fundamentals_score:4.1f}  {r.company[:28]}"
        )

    print("\n" + "=" * 76)
    print("  DISCLAIMER: This output is for EDUCATIONAL and RESEARCH purposes")
    print("  only. It is NOT financial advice. Always perform your own analysis")
    print("  and consult a licensed professional before making decisions.")
    print("=" * 76 + "\n")


# ======================================================================
# 5. CLI Argument Parsing
# ======================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NYSE Full-Universe Multi-Factor Stock Screener (educational)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            f"Quick mode: cap universe at {QUICK_UNIVERSE_CAP} tickers "
            "and reduce Phase 2 pool (useful for testing)."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=FINAL_TOP_N,
        help=f"Number of top candidates to display (default {FINAL_TOP_N}).",
    )
    parser.add_argument(
        "--phase1-top",
        type=int,
        default=PHASE1_TOP_N,
        help=f"Phase 1 survivors forwarded to deep dive (default {PHASE1_TOP_N}).",
    )
    return parser.parse_args()


# ======================================================================
# Main
# ======================================================================
def main() -> None:
    global _GLOBAL_START
    _GLOBAL_START = time.time()

    args = parse_args()
    final_n = args.top
    phase1_top = args.phase1_top

    if args.quick:
        log.info(
            "Quick mode enabled — capping universe at %d tickers.",
            QUICK_UNIVERSE_CAP,
        )
        phase1_top = min(phase1_top, 50)

    # -- Phase 0: Market Regime Gate -----------------------------------
    regime_ok, regime_msg = check_market_regime()
    print(f"\n{'-' * 76}")
    print(f"  {regime_msg}")
    print(f"{'-' * 76}\n")
    if not regime_ok:
        log.warning(
            "Adverse regime — results should be interpreted with extra caution."
        )

    # -- Phase 1: Fetch ALL NYSE tickers -------------------------------
    universe = fetch_all_nyse_tickers()
    total_universe = len(universe)

    if args.quick:
        universe = universe[:QUICK_UNIVERSE_CAP]
        log.info("Quick mode: trimmed universe to %d tickers.", len(universe))

    if not universe:
        log.error("No tickers collected — cannot proceed.")
        sys.exit(1)

    # -- Phase 2: Fast bulk filter -> top N survivors -------------------
    survivors = phase1_bulk_filter(universe, top_n=phase1_top)
    phase1_survivors = len(survivors)

    if not survivors:
        log.error("Phase 1 produced 0 survivors — cannot proceed.")
        sys.exit(1)

    # -- Phase 3: Full 5-layer deep dive --------------------------------
    results = deep_dive(survivors, top_n=len(survivors))

    if not results:
        log.error("Deep dive produced 0 results — cannot proceed.")
        sys.exit(1)

    # -- Phase 4: Report ------------------------------------------------
    print_report(
        results,
        regime_msg,
        total_universe=total_universe,
        phase1_survivors=phase1_survivors,
        final_n=final_n,
    )

    log.info(
        "Screening completed in %.1f seconds (%.1f min).",
        _elapsed(), _elapsed() / 60,
    )


if __name__ == "__main__":
    main()
