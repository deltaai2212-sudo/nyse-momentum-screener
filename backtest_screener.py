#!/usr/bin/env python3
"""
backtest_screener.py — Historical backtest for a predictive stock screening algorithm.

Tests technical scoring signals on 20 random trading days (Jan 2024–Dec 2025),
picks the top-10 scored stocks each day, then measures 1-day and 2-day forward returns.
Compares against a random-10 benchmark on the same dates.

Requirements: yfinance, pandas, numpy
"""

import random
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 1. UNIVERSE — 150 well-known liquid US stocks across sectors
# ──────────────────────────────────────────────────────────────────────────────
UNIVERSE = [
    # Technology (25)
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSM", "AVGO", "ORCL", "CRM",
    "AMD", "ADBE", "INTC", "QCOM", "TXN", "NOW", "SHOP", "SNOW", "PLTR", "NET",
    "MU", "MRVL", "AMAT", "LRCX", "KLAC",
    # Finance (20)
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB",
    "PNC", "TFC", "COF", "FITB", "KEY", "COIN", "HOOD", "ICE", "CME", "SPGI",
    # Healthcare (20)
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "ISRG", "VRTX", "REGN", "ZTS", "SYK", "BSX", "MDT", "HCA",
    # Energy (12)
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "DVN",
    "HAL", "FANG",
    # Consumer Discretionary (18)
    "TSLA", "HD", "NKE", "MCD", "SBUX", "LOW", "TJX", "BKNG", "MAR", "CMG",
    "ROST", "DHI", "LEN", "GM", "F", "ABNB", "DASH", "LULU",
    # Consumer Staples (10)
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "KHC", "GIS",
    # Industrials (15)
    "CAT", "DE", "UNP", "UPS", "HON", "BA", "RTX", "LMT", "GE", "MMM",
    "FDX", "WM", "ETN", "ITW", "EMR",
    # Communication / Media (8)
    "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS", "RBLX", "TTWO",
    # Materials & Utilities (7)
    "LIN", "APD", "FCX", "NEM", "NEE", "SO", "DUK",
    # Small / Mid-cap growth (15)
    "SOFI", "DKNG", "CRWD", "ZS", "SMCI", "CELH", "CAVA", "DUOL",
    "ARM", "IONQ", "RIVN", "MARA", "RIOT", "AFRM", "U",
]

assert len(UNIVERSE) == 150, f"Universe has {len(UNIVERSE)} tickers, expected 150"

# ──────────────────────────────────────────────────────────────────────────────
# 2. GENERATE 20 RANDOM TRADING DAYS (skip weekends)
# ──────────────────────────────────────────────────────────────────────────────
def generate_test_dates(n: int = 20, seed: int = 42) -> list[datetime]:
    """Return *n* random weekday dates between 2024-01-15 and 2025-11-28."""
    rng = random.Random(seed)
    start = datetime(2024, 1, 15)
    end = datetime(2025, 11, 28)
    all_weekdays = []
    d = start
    while d <= end:
        if d.weekday() < 5:  # Mon–Fri
            all_weekdays.append(d)
        d += timedelta(days=1)
    chosen = sorted(rng.sample(all_weekdays, n))
    return chosen


# ──────────────────────────────────────────────────────────────────────────────
# 3. DOWNLOAD DATA IN BULK
# ──────────────────────────────────────────────────────────────────────────────
def download_bulk_data() -> dict[str, pd.DataFrame]:
    """Download OHLCV for every ticker in UNIVERSE. Returns dict[ticker -> df]."""
    print("Downloading historical data for 150 tickers (this may take a minute)...")
    # We need data from mid-2023 onward to have enough lookback for indicators
    raw = yf.download(
        UNIVERSE,
        start="2023-06-01",
        end="2025-12-15",
        group_by="ticker",
        threads=True,
        timeout=15,
        progress=True,
    )

    data: dict[str, pd.DataFrame] = {}
    for ticker in UNIVERSE:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw[ticker].copy()
            else:
                df = raw.copy()
            df = df.dropna(subset=["Close"])
            if len(df) >= 60:  # need at least 60 rows for indicators
                data[ticker] = df
        except Exception:
            continue

    print(f"  → Usable tickers: {len(data)} / {len(UNIVERSE)}\n")
    return data


# ──────────────────────────────────────────────────────────────────────────────
# 4. SCORING FUNCTION — only uses data UP TO signal_date (no lookahead)
# ──────────────────────────────────────────────────────────────────────────────
def score_stock(df: pd.DataFrame, signal_date: datetime) -> int | None:
    """
    Score a single stock using data available on *signal_date*.
    Returns integer score or None if insufficient data.
    """
    # Slice to only data available up to signal_date (inclusive)
    ts = pd.Timestamp(signal_date)
    hist = df.loc[df.index <= ts].copy()
    if len(hist) < 55:
        return None

    close = hist["Close"].values.astype(float)
    high = hist["High"].values.astype(float)
    low = hist["Low"].values.astype(float)
    volume = hist["Volume"].values.astype(float)

    score = 0

    # --- Moving averages ---
    ma20 = np.mean(close[-20:])
    ma50 = np.mean(close[-50:])
    last_price = close[-1]

    # --- Bollinger Band squeeze ---
    std20 = np.std(close[-20:], ddof=1)
    bb_width = (2 * std20) / ma20 if ma20 > 0 else 0
    # Compute BB width over last 50 days
    bb_widths = []
    for i in range(50):
        idx = len(close) - 50 + i
        if idx >= 20:
            window = close[idx - 20 : idx]
            s = np.std(window, ddof=1)
            m = np.mean(window)
            bb_widths.append((2 * s) / m if m > 0 else 0)
    if bb_widths:
        pct_rank = np.sum(np.array(bb_widths) <= bb_width) / len(bb_widths)
        if pct_rank <= 0.20:
            score += 5  # BB squeeze

    # --- Price near MA20 ---
    pct_from_ma20 = abs(last_price - ma20) / ma20 if ma20 > 0 else 1
    if pct_from_ma20 <= 0.02:
        score += 3

    # --- Price just above MA20 (within 1%) ---
    pct_above_ma20 = (last_price - ma20) / ma20 if ma20 > 0 else -1
    if 0 <= pct_above_ma20 <= 0.01:
        score += 2

    # --- Price just above MA50 (within 1%) ---
    pct_above_ma50 = (last_price - ma50) / ma50 if ma50 > 0 else -1
    if 0 <= pct_above_ma50 <= 0.01:
        score += 3

    # --- RSI (14-day) ---
    deltas = np.diff(close[-15:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0.0001
    rs = avg_gain / max(avg_loss, 1e-10)
    rsi = 100 - (100 / (1 + rs))

    if 45 <= rsi <= 60:
        score += 6
    elif 40 <= rsi < 45:
        score += 3
    if rsi > 70:
        score -= 5
    if rsi < 35:
        score -= 3

    # --- Volume drying up (last 3 days < 60% of 20-day avg) ---
    vol_20avg = np.mean(volume[-20:])
    vol_last3 = volume[-3:]
    if vol_20avg > 0 and all(v < 0.60 * vol_20avg for v in vol_last3):
        score += 4

    # --- Volume declining 3 consecutive days while price flat/up ---
    if len(volume) >= 4 and len(close) >= 4:
        vol_declining = (volume[-1] < volume[-2] < volume[-3])
        price_flat_up = (close[-1] >= close[-3] * 0.99)  # within 1% or up
        if vol_declining and price_flat_up:
            score += 3

    # --- MACD histogram turning less negative for 3 days ---
    if len(close) >= 35:
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
        macd_line = ema12 - ema26
        signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
        histogram = macd_line - signal_line
        if len(histogram) >= 4:
            h = histogram[-4:]
            # "Less negative" means still negative but increasing
            if h[-3] < 0 and h[-2] < 0 and h[-1] < 0:
                if h[-1] > h[-2] > h[-3]:
                    score += 3

    return score


# ──────────────────────────────────────────────────────────────────────────────
# 5. FORWARD RETURN CALCULATION
# ──────────────────────────────────────────────────────────────────────────────
def get_forward_returns(df: pd.DataFrame, signal_date: datetime) -> tuple[float | None, float | None]:
    """Return (1-day %, 2-day %) forward returns AFTER signal_date."""
    ts = pd.Timestamp(signal_date)
    future = df.loc[df.index > ts]
    if len(future) < 1:
        return None, None

    # The close ON the signal date
    on_date = df.loc[df.index <= ts]
    if len(on_date) == 0:
        return None, None
    entry_price = float(on_date["Close"].iloc[-1])
    if entry_price <= 0:
        return None, None

    ret_1d = (float(future["Close"].iloc[0]) / entry_price - 1) * 100

    ret_2d = None
    if len(future) >= 2:
        ret_2d = (float(future["Close"].iloc[1]) / entry_price - 1) * 100

    return ret_1d, ret_2d


# ──────────────────────────────────────────────────────────────────────────────
# 6. MAIN BACKTEST
# ──────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 75)
    print("  BACKTEST: Predictive Stock Screener Algorithm")
    print("  Period: Jan 2024 – Dec 2025  |  20 test dates  |  150-stock universe")
    print("=" * 75, "\n")

    # Download data
    data = download_bulk_data()
    if len(data) < 50:
        print("ERROR: Too few tickers with valid data. Check network / yfinance.")
        sys.exit(1)

    test_dates = generate_test_dates(20, seed=42)
    rng_bench = random.Random(99)  # separate RNG for benchmark random picks

    all_picks = []        # list of dicts for algo picks
    all_bench_picks = []  # list of dicts for benchmark (random) picks

    header = (
        f"{'Date':<12} | {'Top Picks (Score)':<48} | {'Avg 1d':>7} | {'Avg 2d':>7} | {'Win 1d':>6}"
    )
    sep = "-" * len(header)

    print(header)
    print(sep)

    for test_date in test_dates:
        date_str = test_date.strftime("%Y-%m-%d")

        # Score every stock
        scores: dict[str, int] = {}
        for ticker, df in data.items():
            s = score_stock(df, test_date)
            if s is not None:
                scores[ticker] = s

        if len(scores) < 10:
            print(f"{date_str:<12} | SKIPPED — only {len(scores)} scoreable stocks")
            continue

        # Top 10 by score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:10]

        # Forward returns for algo picks
        returns_1d = []
        returns_2d = []
        pick_labels = []
        for ticker, sc in ranked:
            r1, r2 = get_forward_returns(data[ticker], test_date)
            pick_labels.append(f"{ticker}({sc})")
            rec = {
                "date": date_str,
                "ticker": ticker,
                "score": sc,
                "ret_1d": r1,
                "ret_2d": r2,
                "type": "algo",
            }
            all_picks.append(rec)
            if r1 is not None:
                returns_1d.append(r1)
            if r2 is not None:
                returns_2d.append(r2)

        # Random benchmark: pick 10 random from same available tickers
        available = list(scores.keys())
        bench_tickers = rng_bench.sample(available, min(10, len(available)))
        for ticker in bench_tickers:
            r1, r2 = get_forward_returns(data[ticker], test_date)
            all_bench_picks.append({
                "date": date_str,
                "ticker": ticker,
                "score": scores.get(ticker, 0),
                "ret_1d": r1,
                "ret_2d": r2,
                "type": "benchmark",
            })

        avg_1d = np.mean(returns_1d) if returns_1d else float("nan")
        avg_2d = np.mean(returns_2d) if returns_2d else float("nan")
        wins_1d = sum(1 for r in returns_1d if r > 0)
        total_1d = len(returns_1d)

        picks_str = " ".join(pick_labels[:5])  # show top 5 for readability
        if len(pick_labels) > 5:
            picks_str += " ..."

        win_tag = f"{wins_1d}/{total_1d}"
        print(
            f"{date_str:<12} | {picks_str:<48} | {avg_1d:>+6.2f}% | {avg_2d:>+6.2f}% | {win_tag:>6}"
        )

    # ── Summary statistics ─────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  SUMMARY")
    print("=" * 75)

    df_algo = pd.DataFrame(all_picks)
    df_bench = pd.DataFrame(all_bench_picks)

    if df_algo.empty:
        print("No algo picks generated. Something went wrong.")
        sys.exit(1)

    total_algo = len(df_algo)
    valid_1d_algo = df_algo.dropna(subset=["ret_1d"])
    valid_2d_algo = df_algo.dropna(subset=["ret_2d"])

    algo_wr1 = (valid_1d_algo["ret_1d"] > 0).mean() * 100 if len(valid_1d_algo) else 0
    algo_wr2 = (valid_2d_algo["ret_2d"] > 0).mean() * 100 if len(valid_2d_algo) else 0
    algo_avg1 = valid_1d_algo["ret_1d"].mean() if len(valid_1d_algo) else 0
    algo_avg2 = valid_2d_algo["ret_2d"].mean() if len(valid_2d_algo) else 0

    valid_1d_bench = df_bench.dropna(subset=["ret_1d"])
    valid_2d_bench = df_bench.dropna(subset=["ret_2d"])
    bench_wr1 = (valid_1d_bench["ret_1d"] > 0).mean() * 100 if len(valid_1d_bench) else 0
    bench_wr2 = (valid_2d_bench["ret_2d"] > 0).mean() * 100 if len(valid_2d_bench) else 0
    bench_avg1 = valid_1d_bench["ret_1d"].mean() if len(valid_1d_bench) else 0
    bench_avg2 = valid_2d_bench["ret_2d"].mean() if len(valid_2d_bench) else 0

    # Best and worst trades
    best_row = valid_1d_algo.loc[valid_1d_algo["ret_1d"].idxmax()] if len(valid_1d_algo) else None
    # Also check 2d
    if len(valid_2d_algo):
        best_2d = valid_2d_algo.loc[valid_2d_algo["ret_2d"].idxmax()]
        if best_row is not None and best_2d["ret_2d"] > best_row["ret_1d"]:
            best_row = best_2d
            best_col = "ret_2d"
        else:
            best_col = "ret_1d"
    else:
        best_col = "ret_1d"

    worst_row = valid_1d_algo.loc[valid_1d_algo["ret_1d"].idxmin()] if len(valid_1d_algo) else None
    if len(valid_2d_algo):
        worst_2d = valid_2d_algo.loc[valid_2d_algo["ret_2d"].idxmin()]
        if worst_row is not None and worst_2d["ret_2d"] < worst_row["ret_1d"]:
            worst_row = worst_2d
            worst_col = "ret_2d"
        else:
            worst_col = "ret_1d"
    else:
        worst_col = "ret_1d"

    print(f"\n  Total picks tested:   {total_algo}")
    print(f"  Test dates completed: {df_algo['date'].nunique()}")
    print()
    print(f"  Win rate (1-day):     {algo_wr1:5.1f}%   vs Benchmark: {bench_wr1:5.1f}%   → Alpha: {algo_wr1 - bench_wr1:+.1f}%")
    print(f"  Win rate (2-day):     {algo_wr2:5.1f}%   vs Benchmark: {bench_wr2:5.1f}%   → Alpha: {algo_wr2 - bench_wr2:+.1f}%")
    print()
    print(f"  Avg return (1-day):   {algo_avg1:+.3f}%  vs Benchmark: {bench_avg1:+.3f}%")
    print(f"  Avg return (2-day):   {algo_avg2:+.3f}%  vs Benchmark: {bench_avg2:+.3f}%")
    print()

    if best_row is not None:
        print(f"  Best single trade:    {best_row['ticker']} {best_row[best_col]:+.2f}% ({best_row['date']})")
    if worst_row is not None:
        print(f"  Worst single trade:   {worst_row['ticker']} {worst_row[worst_col]:+.2f}% ({worst_row['date']})")

    # ── Save CSV ───────────────────────────────────────────────────────────
    combined = pd.concat([df_algo, df_bench], ignore_index=True)
    csv_path = "backtest_results.csv"
    combined.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"\n  Results saved to: {csv_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
