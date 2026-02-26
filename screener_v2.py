#!/usr/bin/env python3
"""
screener_v2.py — Explosive Mover Screener
Screens ~500 high-volatility US small/mid caps for setups likely to gain 10%+ in 1-3 days.
Scoring: Catalyst (40) + Technical Breakout (30) + Momentum (20) + Short Squeeze (20) + Bonus (20)
Uses yfinance only (no paid APIs). Runs in parallel with ThreadPoolExecutor.
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


# ──────────────────────────────────────────────────────────────────────────────
# 1. UNIVERSE — ~500 high-vol small/mid cap tickers
# ──────────────────────────────────────────────────────────────────────────────

UNIVERSE = sorted(set([
    # ── Crypto / Bitcoin miners ──
    "MARA", "RIOT", "COIN", "MSTR", "BTBT", "HUT", "CIFR", "WULF", "IREN",
    "CORZ", "APLD", "BITF", "CLSK", "ARBK", "HIVE", "SATO", "GREE",

    # ── AI / Quantum computing ──
    "PLTR", "SOUN", "BBAI", "GFAI", "IONQ", "QUBT", "RGTI", "QBTS",
    "KULR", "SMCI", "UPST", "PATH", "AI", "BIGB", "PRCT", "CAVA",
    "DUOL", "NNOX", "SDGR", "RXRX",

    # ── EV / Space / Mobility ──
    "RIVN", "LCID", "JOBY", "ACHR", "RKLB", "TSLA", "GOEV", "PTRA",
    "WKHS", "RIDE", "NKLA", "HYLN", "FSR", "SPCE", "MNTS", "RDW",
    "ASTR", "ASTS", "BKSY", "LUNR", "SATL", "SPIR",

    # ── Fintech ──
    "SOFI", "HOOD", "AFRM", "OPEN", "SQ", "DAVE", "DLO", "BILL",
    "HIMS", "CLOV", "SMAR", "MAPS", "MQ", "PSFE", "NUVEI", "PAYO",
    "TOST", "RSKD", "FLYW", "RELY", "ADYN",

    # ── High-beta tech ──
    "NVDA", "AMD", "CRWD", "NET", "DDOG", "SNOW", "MDB", "ZS", "OKTA",
    "CELH", "SHOP", "ROKU", "U", "RBLX", "SNAP", "PINS", "SPOT",
    "FUBO", "GENI", "CURI", "IRNT", "MNDY", "GLBE", "TMDX",

    # ── Biotech (large set – biggest movers on catalyst) ──
    "SKIN", "ACMR", "ARDX", "KROS", "IMVT", "MDGL", "RVNC", "TGTX",
    "VERA", "CPRX", "ACVA", "BRLT", "RAPT", "PRAX", "REPL", "VERV",
    "BEAM", "EDIT", "CRSP", "NTLA", "FATE", "BLUE", "PGEN", "AGEN",
    "NKTR", "SRPT", "RARE", "FOLD", "MNKD", "CLDX", "CTIC", "ACAD",
    "AVXL", "SAVA", "ATNF", "OTIC", "IMNM", "PRTA", "ALDX", "ANAB",
    "ARCT", "AXSM", "BHVN", "CCXI", "CHRS", "CMRX", "DVAX", "ELAN",
    "ENTA", "ETON", "FGEN", "FULC", "GOSS", "HALO", "HRTX",
    "IMAB", "INVA", "KALA", "KDMN", "KMPH", "MGNX", "MGTX",
    "MNLO", "NEOS", "NRIX", "NUVB", "OBSV", "OCGN", "OMER",
    "ONCT", "ORMP", "OTGN", "OVID", "PCVX", "PLRX", "PRVB",
    "PTCT", "PTGX", "QURE", "RCEL", "RCKT", "REGN", "RLAY",
    "RNAC", "RYTM", "SAGE", "SCLX", "SGMO", "SNDX", "SPPI",
    "SRNE", "STRO", "TARS", "TBPH", "TCMD", "TCRR", "THRX",
    "TRVN", "URGN", "UTHR", "VBIV", "VCEL", "VCNX", "VKTX",
    "VNDA", "VRDN", "VRTX", "VTGN", "VXRT", "XOMA", "YMAB",
    "ZIOP", "APLS", "BMRN", "EXAS", "MIRM", "MRSN", "NUVL",
    "ROIV", "SRRK", "TVTX", "XNCR", "ZNTL",

    # ── Cannabis ──
    "TLRY", "CGC", "ACB", "CRON", "SNDL", "GRNH", "MAPS",

    # ── Meme / high-retail-interest ──
    "GME", "AMC", "BBBY", "KOSS", "EXPR", "WKHS", "IRNT",
    "ATER", "CLOV", "WISH", "SDC", "RKT", "SKLZ", "DKNG",

    # ── Energy / miners (volatile subset) ──
    "FCEL", "PLUG", "BLDP", "BE", "CHPT", "EVGO", "QS", "MVST",
    "STEM", "ARRY", "ENPH", "SEDG", "RUN", "NOVA", "SHLS",
    "MP", "LAC", "LTHM", "ALB", "PLL", "SLI", "UUUU", "CCJ",
    "UEC", "DNN", "NXE", "LEU", "OKLO",

    # ── Additional small-cap momentum / recent IPOs ──
    "CAVA", "CART", "ARM", "BIRK", "ONON", "VFS", "GRAB",
    "GTLB", "BRZE", "CFLT", "DOCN", "IOT", "ASAN",
    "ESTC", "PCOR", "FRSH", "ALKT", "AUR", "LAZR",
    "OUST", "LIDR", "CPNG", "SE", "GRAB", "BABA", "JD",
    "NIO", "XPEV", "LI", "ZK", "YMM", "TUYA", "WB",
    "BEKE", "DADA", "IQ", "FUTU", "TIGR",
    "BTDR", "CELU", "BMBL", "MTCH", "ABNB", "DASH",

    # ── More biotech / pharma small caps ──
    "ABCL", "AGIO", "ALNY", "ARWR", "BNTX", "CRNX", "DNLI",
    "DRTS", "EXAI", "GILD", "HRMY", "INSM", "IONS", "IRTC",
    "JANX", "KRYS", "LGND", "LQDA", "NBIX", "PEPG", "PSNL",
    "RVMD", "SGEN", "SMMT", "SWTX", "TECH", "TXG", "VRTX",
    "XERS",

    # ── More tech / software / SaaS ──
    "APPF", "AVDX", "BRSP", "CDNA", "CWAN", "DV", "EVBG",
    "FIVN", "FOUR", "HOOD", "HCP", "KD", "KNBE",
    "LMND", "MGNI", "NTNX", "OLO", "PUBM", "S", "SEMR",
    "TENB", "TOST", "WEAV", "YOU", "ZI", "ZUO",

    # ── Sector ETFs (for sector rotation calc only — not screened) ──
]))

# ETFs for sector rotation scoring
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLB", "XLI", "XLV", "XLY", "XLRE",
               "XLP", "XLU", "XLC", "ARKK", "IBB", "XBI"]

# Rough sector mapping (ETF → keywords in industry/sector from yfinance)
ETF_SECTOR_MAP = {
    "XLK": ["technology", "software", "semiconductor", "electronic"],
    "XLF": ["financial", "bank", "insurance", "capital markets"],
    "XLE": ["energy", "oil", "gas", "petroleum"],
    "XLB": ["materials", "chemical", "mining"],
    "XLI": ["industrial", "aerospace", "defense", "machinery"],
    "XLV": ["health", "biotech", "pharma", "medical", "drug"],
    "XLY": ["consumer", "retail", "restaurant", "apparel", "auto"],
    "XLRE": ["real estate", "reit"],
    "XLP": ["staple", "food", "beverage", "household"],
    "XLU": ["utility", "electric power"],
    "XLC": ["communication", "media", "entertainment", "telecom"],
    "ARKK": ["technology", "genomic", "fintech", "autonomous"],
    "IBB": ["biotech", "pharma", "drug"],
    "XBI": ["biotech", "pharma", "drug", "genomic"],
}


# ──────────────────────────────────────────────────────────────────────────────
# 2. DATA CLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class StockResult:
    ticker: str = ""
    score: float = 0.0
    price: float = 0.0
    market_cap: float = 0.0
    sector: str = ""
    industry: str = ""
    signals: list = field(default_factory=list)
    short_pct: float = 0.0
    days_to_cover: float = 0.0
    dist_from_52wk_high: float = 0.0
    earnings_date: str = ""
    ret_3d: float = 0.0
    ret_5d: float = 0.0
    ret_10d: float = 0.0
    atr_pct: float = 0.0
    volume_ratio: float = 0.0
    price_target: float = 0.0
    # sub-scores
    catalyst_score: float = 0.0
    breakout_score: float = 0.0
    momentum_score: float = 0.0
    squeeze_score: float = 0.0
    volatility_score: float = 0.0
    sector_score: float = 0.0
    fundamental_score: float = 0.0
    passed_phase1: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# 3. SECTOR ROTATION — precompute leading ETFs
# ──────────────────────────────────────────────────────────────────────────────

def get_sector_rotation() -> dict:
    """Return dict: etf -> 5d return, plus 'top_etfs' list of leaders."""
    result = {}
    try:
        data = yf.download(SECTOR_ETFS, period="10d", progress=False, threads=True)
        if data.empty:
            return {"top_etfs": []}
        close = data["Close"] if "Close" in data.columns else data.get("Adj Close", pd.DataFrame())
        if close.empty:
            return {"top_etfs": []}
        for etf in SECTOR_ETFS:
            if etf in close.columns:
                prices = close[etf].dropna()
                if len(prices) >= 5:
                    ret5 = (prices.iloc[-1] / prices.iloc[-5] - 1) * 100
                    result[etf] = ret5
        # top 3 ETFs
        sorted_etfs = sorted(result.items(), key=lambda x: x[1], reverse=True)
        result["top_etfs"] = [e[0] for e in sorted_etfs[:3]]
    except Exception:
        result["top_etfs"] = []
    return result


def stock_in_leading_sector(sector: str, industry: str, top_etfs: list) -> bool:
    """Check if stock's sector/industry matches any leading ETF's keywords."""
    combined = (sector + " " + industry).lower()
    for etf in top_etfs:
        keywords = ETF_SECTOR_MAP.get(etf, [])
        for kw in keywords:
            if kw in combined:
                return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# 4. PHASE 1 — Quick filter (price, volume, basic momentum)
# ──────────────────────────────────────────────────────────────────────────────

def phase1_screen(ticker: str) -> Optional[dict]:
    """Quick screen: price >= $1, avg vol >= 500k, ATR% >= 2%.
    Returns basic data dict or None if filtered out."""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="60d")
        if hist.empty or len(hist) < 20:
            return None

        price = hist["Close"].iloc[-1]
        if price < 1.0:
            return None

        avg_vol_30 = hist["Volume"].tail(30).mean()
        if avg_vol_30 < 500_000:
            return None

        # ATR% check
        hi = hist["High"].tail(20)
        lo = hist["Low"].tail(20)
        cl = hist["Close"].tail(20)
        tr = pd.concat([
            hi - lo,
            (hi - cl.shift(1)).abs(),
            (lo - cl.shift(1)).abs()
        ], axis=1).max(axis=1)
        atr14 = tr.tail(14).mean()
        atr_pct = (atr14 / price) * 100 if price > 0 else 0
        if atr_pct < 2.0:
            return None

        # Volume spike today
        vol_today = hist["Volume"].iloc[-1]
        vol_ratio = vol_today / avg_vol_30 if avg_vol_30 > 0 else 0

        # Quick momentum: 3-day return
        if len(hist) >= 4:
            ret_3d = (price / hist["Close"].iloc[-4] - 1) * 100
        else:
            ret_3d = 0

        return {
            "ticker": ticker,
            "price": price,
            "avg_vol_30": avg_vol_30,
            "vol_today": vol_today,
            "vol_ratio": vol_ratio,
            "atr_pct": atr_pct,
            "ret_3d": ret_3d,
            "hist": hist,
        }
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 5. PHASE 2 — Deep scoring
# ──────────────────────────────────────────────────────────────────────────────

def deep_score(p1: dict, sector_data: dict) -> Optional[StockResult]:
    """Full 8-category scoring on a Phase-1 survivor."""
    ticker = p1["ticker"]
    hist = p1["hist"]
    price = p1["price"]
    atr_pct = p1["atr_pct"]
    vol_ratio = p1["vol_ratio"]

    result = StockResult(ticker=ticker, price=price, atr_pct=atr_pct, volume_ratio=vol_ratio)
    signals = []

    try:
        tk = yf.Ticker(ticker)
        info = {}
        try:
            info = tk.info or {}
        except Exception:
            pass

        # ── Basic info ──
        mkt_cap = info.get("marketCap", 0) or 0
        result.market_cap = mkt_cap
        result.sector = info.get("sector", "")
        result.industry = info.get("industry", "")
        beta = info.get("beta", 1.0) or 1.0

        # Filter: skip micro-caps < $50M (but allow if info is missing)
        if mkt_cap > 0 and mkt_cap < 50_000_000:
            return None

        close = hist["Close"]
        volume = hist["Volume"]
        avg_vol_30 = p1["avg_vol_30"]

        # ── Returns ──
        if len(close) >= 4:
            result.ret_3d = (close.iloc[-1] / close.iloc[-4] - 1) * 100
        if len(close) >= 6:
            result.ret_5d = (close.iloc[-1] / close.iloc[-6] - 1) * 100
        if len(close) >= 11:
            result.ret_10d = (close.iloc[-1] / close.iloc[-11] - 1) * 100

        # ── 52-week high distance ──
        try:
            wk52_hi = info.get("fiftyTwoWeekHigh", close.max())
            result.dist_from_52wk_high = (price / wk52_hi - 1) * 100 if wk52_hi else 0
        except Exception:
            result.dist_from_52wk_high = 0

        # ══════════════════════════════════════════════════════════════════
        # CATEGORY 1: CATALYST SCORING (max 40)
        # ══════════════════════════════════════════════════════════════════
        cat_score = 0.0
        try:
            cal = tk.calendar
            if cal is not None:
                # yfinance returns either a DataFrame or dict
                earnings_date = None
                if isinstance(cal, pd.DataFrame):
                    if "Earnings Date" in cal.index:
                        earnings_date = cal.loc["Earnings Date"].iloc[0]
                elif isinstance(cal, dict):
                    ed = cal.get("Earnings Date", [])
                    if ed:
                        earnings_date = ed[0] if isinstance(ed, list) else ed

                if earnings_date is not None:
                    if isinstance(earnings_date, str):
                        earnings_date = pd.Timestamp(earnings_date)
                    elif not isinstance(earnings_date, pd.Timestamp):
                        earnings_date = pd.Timestamp(earnings_date)

                    result.earnings_date = earnings_date.strftime("%Y-%m-%d")
                    today = pd.Timestamp.now().normalize()
                    days_until = (earnings_date - today).days

                    if 0 <= days_until <= 5:
                        cat_score += 20
                        signals.append("EARNINGS_SOON")

                    # Recent beat: gap-up after earnings in last 2 days
                    if -2 <= days_until <= 0 and result.ret_3d > 3:
                        cat_score += 25
                        signals.append("EARNINGS_BEAT_GAPUP")
        except Exception:
            pass

        # Unusual volume as proxy for options/catalyst activity
        if vol_ratio > 3:
            cat_score += 15
            signals.append("UNUSUAL_VOLUME_3X")
        elif vol_ratio > 2:
            cat_score += 8

        result.catalyst_score = min(cat_score, 40)

        # ══════════════════════════════════════════════════════════════════
        # CATEGORY 2: TECHNICAL BREAKOUT (max 30)
        # ══════════════════════════════════════════════════════════════════
        brk_score = 0.0

        # 52-week high breakout
        if len(close) >= 2:
            wk52_hi = close.max()  # from available history (up to 60d; check longer)
            try:
                hist_long = tk.history(period="1y")
                if not hist_long.empty:
                    wk52_hi = hist_long["Close"].max()
            except Exception:
                pass

            prev_close = close.iloc[-2]
            if price >= wk52_hi * 0.99 and vol_ratio > 3:
                brk_score += 30
                signals.append("52WK_HIGH_BREAKOUT")
            elif price >= wk52_hi * 0.99 and vol_ratio > 2:
                brk_score += 20
                signals.append("NEAR_52WK_HIGH")

        # 20-day high breakout
        if len(close) >= 20:
            high_20d = close.tail(20).max()
            if price >= high_20d * 0.99 and vol_ratio > 2:
                brk_score += 20
                signals.append("20D_HIGH_BREAKOUT")

        # Bollinger squeeze + breakout
        if len(close) >= 20:
            try:
                bb = ta.bbands(close, length=20, std=2)
                if bb is not None and not bb.empty:
                    upper_col = [c for c in bb.columns if "BBU" in c]
                    lower_col = [c for c in bb.columns if "BBL" in c]
                    mid_col = [c for c in bb.columns if "BBM" in c]
                    if upper_col and lower_col and mid_col:
                        bbu = bb[upper_col[0]].iloc[-1]
                        bbl = bb[lower_col[0]].iloc[-1]
                        bbm = bb[mid_col[0]].iloc[-1]
                        bb_width = (bbu - bbl) / bbm * 100 if bbm > 0 else 100
                        if bb_width < 5 and price > bbu:
                            brk_score += 15
                            signals.append("BB_SQUEEZE_BREAKOUT")
            except Exception:
                pass

        # Gap up > 3%
        if len(close) >= 2:
            gap = (hist["Open"].iloc[-1] / close.iloc[-2] - 1) * 100
            if gap > 3:
                brk_score += 20
                signals.append(f"GAP_UP_{gap:.1f}%")

        # 200-day MA reclaim
        if len(close) >= 30:
            try:
                hist_200 = tk.history(period="1y")
                if len(hist_200) >= 200:
                    ma200 = hist_200["Close"].tail(200).mean()
                    # Was below 200MA for >30 days and now above
                    recent_below = (hist_200["Close"].tail(60) < ma200).sum()
                    if price > ma200 and recent_below > 30:
                        brk_score += 15
                        signals.append("200MA_RECLAIM")
            except Exception:
                pass

        result.breakout_score = min(brk_score, 30)

        # ══════════════════════════════════════════════════════════════════
        # CATEGORY 3: MOMENTUM (max 20)
        # ══════════════════════════════════════════════════════════════════
        mom_score = 0.0

        # RSI
        if len(close) >= 14:
            try:
                rsi_series = ta.rsi(close, length=14)
                if rsi_series is not None and len(rsi_series) >= 2:
                    rsi_now = rsi_series.iloc[-1]
                    rsi_prev = rsi_series.iloc[-2]
                    if 55 <= rsi_now <= 75:
                        mom_score += 10
                        signals.append(f"RSI_{rsi_now:.0f}")
                    if rsi_now > 50 and rsi_prev < 50:
                        mom_score += 15
                        signals.append("RSI_CROSS_50")
            except Exception:
                pass

        # Returns
        if result.ret_3d > 5:
            mom_score += 10
            signals.append(f"3D_RET_{result.ret_3d:.1f}%")
        if result.ret_5d > 8:
            mom_score += 10
            signals.append(f"5D_RET_{result.ret_5d:.1f}%")

        # Volume
        if vol_ratio > 5:
            mom_score += 20
            signals.append("VOL_5X")
        elif vol_ratio > 2:
            mom_score += 10
            signals.append("VOL_2X")

        # Price above all key MAs
        if len(close) >= 50:
            try:
                ma5 = close.tail(5).mean()
                ma10 = close.tail(10).mean()
                ma20 = close.tail(20).mean()
                ma50 = close.tail(50).mean()
                if price > ma5 and price > ma10 and price > ma20 and price > ma50:
                    mom_score += 10
                    signals.append("ABOVE_ALL_MAs")
            except Exception:
                pass

        result.momentum_score = min(mom_score, 20)

        # ══════════════════════════════════════════════════════════════════
        # CATEGORY 4: SHORT SQUEEZE (max 20)
        # ══════════════════════════════════════════════════════════════════
        sq_score = 0.0
        short_pct = info.get("shortPercentOfFloat", 0) or 0
        short_ratio = info.get("shortRatio", 0) or 0
        result.short_pct = short_pct * 100 if short_pct < 1 else short_pct  # normalize
        result.days_to_cover = short_ratio

        if result.short_pct > 30:
            sq_score += 20
            signals.append(f"SHORT_{result.short_pct:.0f}%")
        elif result.short_pct > 20:
            sq_score += 10
            signals.append(f"SHORT_{result.short_pct:.0f}%")

        if short_ratio > 5:
            sq_score += 10
            signals.append(f"DTC_{short_ratio:.1f}")

        result.squeeze_score = min(sq_score, 20)

        # ══════════════════════════════════════════════════════════════════
        # CATEGORY 5: VOLATILITY / BETA (bonus up to 15)
        # ══════════════════════════════════════════════════════════════════
        vol_score = 0.0

        # Historical volatility
        if len(close) >= 30:
            try:
                log_ret = np.log(close / close.shift(1)).dropna()
                hvol = log_ret.tail(30).std() * np.sqrt(252) * 100
                if hvol > 50:
                    vol_score += 10
                    signals.append(f"HVOL_{hvol:.0f}%")
            except Exception:
                pass

        if beta > 1.5:
            vol_score += 5
            signals.append(f"BETA_{beta:.1f}")

        result.volatility_score = vol_score

        # ══════════════════════════════════════════════════════════════════
        # CATEGORY 6: SECTOR ROTATION (bonus up to 10)
        # ══════════════════════════════════════════════════════════════════
        sec_score = 0.0
        top_etfs = sector_data.get("top_etfs", [])
        if stock_in_leading_sector(result.sector, result.industry, top_etfs):
            sec_score += 10
            signals.append("HOT_SECTOR")
        result.sector_score = sec_score

        # ══════════════════════════════════════════════════════════════════
        # CATEGORY 7: FUNDAMENTAL (bonus up to 10)
        # ══════════════════════════════════════════════════════════════════
        fund_score = 0.0
        rev_growth = info.get("revenueGrowth", 0) or 0
        if rev_growth > 0.20 and result.ret_5d > 0:
            fund_score += 10
            signals.append(f"REV_GROWTH_{rev_growth*100:.0f}%")
        result.fundamental_score = fund_score

        # ══════════════════════════════════════════════════════════════════
        # TOTAL SCORE
        # ══════════════════════════════════════════════════════════════════
        result.score = (
            result.catalyst_score +
            result.breakout_score +
            result.momentum_score +
            result.squeeze_score +
            result.volatility_score +
            result.sector_score +
            result.fundamental_score
        )

        result.signals = signals

        # Price target: at least +10%, scale up with score
        multiplier = max(1.10, 1.10 + (result.score - 40) * 0.002)
        result.price_target = round(price * multiplier, 2)

        return result

    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 6. MAIN ORCHESTRATOR
# ──────────────────────────────────────────────────────────────────────────────

def main():
    start = time.time()
    SCORE_THRESHOLD = 40
    MAX_WORKERS = 20
    PHASE1_KEEP = 150
    TOP_N = 10

    print("=" * 80)
    print("  SCREENER V2 — Explosive 10%+ Mover Scanner")
    print(f"  Universe: {len(UNIVERSE)} tickers  |  Date: {dt.date.today()}")
    print("=" * 80)

    # ── Sector rotation (pre-compute) ──
    print("\n[1/4] Computing sector rotation...")
    sector_data = get_sector_rotation()
    top_etfs = sector_data.get("top_etfs", [])
    print(f"  Leading sectors: {', '.join(top_etfs) if top_etfs else 'N/A'}")
    for etf in top_etfs:
        ret = sector_data.get(etf, 0)
        print(f"    {etf}: {ret:+.1f}% (5d)")

    # ── Phase 1: Quick screen ──
    print(f"\n[2/4] Phase 1: Quick screening {len(UNIVERSE)} tickers...")
    phase1_results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(phase1_screen, t): t for t in UNIVERSE}
        for future in tqdm(as_completed(futures), total=len(futures), desc="  Phase 1",
                          ncols=80, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
            res = future.result()
            if res is not None:
                phase1_results.append(res)

    # Sort by composite of vol_ratio + abs(ret_3d) + atr_pct to pick most active
    phase1_results.sort(
        key=lambda x: x["vol_ratio"] * 2 + abs(x["ret_3d"]) + x["atr_pct"],
        reverse=True
    )
    phase1_results = phase1_results[:PHASE1_KEEP]

    print(f"  Phase 1 survivors: {len(phase1_results)} (kept top {PHASE1_KEEP})")

    # ── Phase 2: Deep scoring ──
    print(f"\n[3/4] Phase 2: Deep scoring {len(phase1_results)} tickers...")
    scored_results: list[StockResult] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(deep_score, p, sector_data): p["ticker"] for p in phase1_results}
        for future in tqdm(as_completed(futures), total=len(futures), desc="  Phase 2",
                          ncols=80, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
            res = future.result()
            if res is not None and res.score >= SCORE_THRESHOLD:
                scored_results.append(res)

    # Sort by total score descending
    scored_results.sort(key=lambda x: x.score, reverse=True)
    top_picks = scored_results[:TOP_N]

    # ── Output ──
    elapsed = time.time() - start
    print(f"\n[4/4] Results — {len(scored_results)} stocks scored >= {SCORE_THRESHOLD}")
    print(f"  Total runtime: {elapsed:.1f}s")

    if not top_picks:
        print("\n  ⚠  No stocks met the threshold today. Market may be too quiet.")
        print("     Try lowering SCORE_THRESHOLD or broadening the universe.")
        return

    print("\n" + "=" * 80)
    print(f"  🔥 TOP {len(top_picks)} HIGH-CONVICTION EXPLOSIVE MOVER CANDIDATES")
    print("=" * 80)

    for i, r in enumerate(top_picks, 1):
        cap_str = f"${r.market_cap/1e9:.1f}B" if r.market_cap >= 1e9 else f"${r.market_cap/1e6:.0f}M"
        sig_str = " + ".join(r.signals[:6]) if r.signals else "MOMENTUM"

        print(f"\n  #{i}  {r.ticker}  —  SCORE: {r.score:.0f}")
        print(f"  {'─' * 60}")
        print(f"  Price: ${r.price:.2f}   |  Mkt Cap: {cap_str}   |  Sector: {r.sector}")
        print(f"  Signals: {sig_str}")
        print(f"  Short Interest: {r.short_pct:.1f}%   |  Days to Cover: {r.days_to_cover:.1f}")
        print(f"  Dist from 52wk High: {r.dist_from_52wk_high:.1f}%")
        print(f"  Returns:  3d={r.ret_3d:+.1f}%  |  5d={r.ret_5d:+.1f}%  |  10d={r.ret_10d:+.1f}%")
        print(f"  ATR%: {r.atr_pct:.1f}%   |  Vol Ratio: {r.volume_ratio:.1f}x")
        print(f"  Earnings: {r.earnings_date or 'N/A'}")
        print(f"  📈 Price Target: ${r.price_target:.2f}  (+{(r.price_target/r.price - 1)*100:.0f}%)")
        print(f"  Scores: CAT={r.catalyst_score:.0f} BRK={r.breakout_score:.0f} "
              f"MOM={r.momentum_score:.0f} SQ={r.squeeze_score:.0f} "
              f"VOL={r.volatility_score:.0f} SEC={r.sector_score:.0f} FUN={r.fundamental_score:.0f}")

    # ── Save CSV ──
    output_path = "screener_v2_results.csv"
    rows = []
    for r in scored_results:
        rows.append({
            "Ticker": r.ticker,
            "Score": r.score,
            "Price": round(r.price, 2),
            "MarketCap": r.market_cap,
            "Sector": r.sector,
            "Industry": r.industry,
            "Signals": " | ".join(r.signals),
            "ShortInterest%": round(r.short_pct, 1),
            "DaysToCover": round(r.days_to_cover, 1),
            "DistFrom52wkHigh%": round(r.dist_from_52wk_high, 1),
            "EarningsDate": r.earnings_date,
            "Return3d%": round(r.ret_3d, 1),
            "Return5d%": round(r.ret_5d, 1),
            "Return10d%": round(r.ret_10d, 1),
            "ATR%": round(r.atr_pct, 1),
            "VolumeRatio": round(r.volume_ratio, 1),
            "PriceTarget": r.price_target,
            "CatalystScore": r.catalyst_score,
            "BreakoutScore": r.breakout_score,
            "MomentumScore": r.momentum_score,
            "SqueezeScore": r.squeeze_score,
            "VolatilityScore": r.volatility_score,
            "SectorScore": r.sector_score,
            "FundamentalScore": r.fundamental_score,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n  💾 Full results saved to: {output_path}")
    print(f"     ({len(rows)} stocks above threshold)")

    print("\n" + "=" * 80)
    print("  ⚠  DISCLAIMER: This is a screening tool, NOT financial advice.")
    print("  All scores are heuristic. Verify setups manually before trading.")
    print("  Past momentum does not guarantee future results.")
    print("=" * 80)


if __name__ == "__main__":
    main()
