"""Loads the universe and daily bars. Real data comes from Yahoo Finance (yfinance);
`synthetic()` builds fake data so the engine can be tested with no network."""
import csv
import os
import time

import numpy as np
import pandas as pd

from .indicators import INDEX_TICKERS, SECTOR_ETF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP500_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"


def load_universe(refresh=False):
    """Returns {ticker: {"name":..., "sector":...}}. S&P 500 + universe/extra.csv."""
    path = os.path.join(ROOT, "universe", "sp500.csv")
    if refresh:
        try:
            import urllib.request

            raw = urllib.request.urlopen(SP500_URL, timeout=30).read().decode("utf-8")
            if raw.count("\n") > 400:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(raw)
        except Exception as e:  # keep the committed copy
            print("universe refresh failed:", e)
    uni = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = row["Symbol"].strip().replace(".", "-")
            uni[t] = {"name": row["Security"].strip(), "sector": row["GICS Sector"].strip()}
    extra = os.path.join(ROOT, "universe", "extra.csv")
    if os.path.exists(extra):
        with open(extra, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                t = parts[0].upper().replace(".", "-")
                if t in uni:
                    continue
                sector = parts[1] if len(parts) > 1 and parts[1] in SECTOR_ETF else None
                name = parts[2] if len(parts) > 2 else t
                uni[t] = {"name": name, "sector": sector}
    return uni


def _clean(df):
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].astype("float64")
    df = df[df["close"].notna() & (df["close"] > 0)]
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    return df


def download(tickers, period="5y", chunk=80, pause=1.5):
    """Split-adjusted daily OHLCV from Yahoo. Returns ({ticker: DataFrame}, [failed])."""
    import yfinance as yf

    out, failed = {}, []
    tickers = list(dict.fromkeys(tickers))
    for i in range(0, len(tickers), chunk):
        part = tickers[i : i + chunk]
        for attempt in range(3):
            try:
                raw = yf.download(part, period=period, interval="1d", auto_adjust=False, group_by="ticker",
                                  threads=True, progress=False)
                break
            except Exception as e:
                print(f"chunk {i} attempt {attempt} failed: {e}")
                raw = None
                time.sleep(5 * (attempt + 1))
        for t in part:
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = _clean(df.dropna(how="all"))
                if len(df) >= 30:
                    out[t] = df
            except Exception:
                pass
        time.sleep(pause)
    missing = [t for t in tickers if t not in out]
    for t in missing:  # one slow retry each
        try:
            df = _clean(yf.Ticker(t).history(period=period, interval="1d", auto_adjust=False))
            if len(df) >= 30:
                out[t] = df
        except Exception:
            pass
        time.sleep(0.5)
    failed = [t for t in tickers if t not in out]
    return out, failed


def externals(tickers, asof):
    """Best-effort next earnings date and market cap for a short list of tickers."""
    import yfinance as yf

    res = {}
    for t in tickers:
        info = {"earn_days": None, "earn_date": None, "mcap": None}
        try:
            tk = yf.Ticker(t)
            try:
                info["mcap"] = float(tk.fast_info["market_cap"])
            except Exception:
                pass
            cal = tk.calendar or {}
            dates = cal.get("Earnings Date") or []
            if not isinstance(dates, (list, tuple)):
                dates = [dates]
            fut = sorted(pd.Timestamp(x) for x in dates if pd.Timestamp(x) >= asof)
            if fut:
                info["earn_date"] = fut[0].strftime("%Y-%m-%d")
                info["earn_days"] = int(np.busday_count(asof.date(), fut[0].date()))
        except Exception as e:
            info["error"] = str(e)[:80]
        res[t] = info
        time.sleep(0.25)
    return res


def align(px):
    """Reindex everything to SPY's trading calendar."""
    idx = px["SPY"].index
    return {t: df.reindex(idx) for t, df in px.items()}, idx


def synthetic(n_tickers=60, n_days=900, seed=7):
    """Random-walk market with a common factor, sector factors and a few engineered patterns."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2026-10-09", periods=n_days)
    mkt = rng.normal(0.0004, 0.009, n_days)
    sectors = list(SECTOR_ETF)
    sec_f = {s: rng.normal(0.0, 0.005, n_days) for s in sectors}

    def bars(ret, start, vol_base, jumps=None):
        c = start * np.exp(np.cumsum(ret))
        o = np.empty(n_days)
        o[0] = c[0]
        overnight = rng.normal(0, 0.004, n_days)
        if jumps is not None:
            overnight = overnight + 0.85 * jumps  # most of an earnings move happens at the open
        o[1:] = c[:-1] * np.exp(overnight[1:])
        spread = np.abs(rng.normal(0.008, 0.004, n_days)) + 0.002
        h = np.maximum(o, c) * (1 + spread)
        l = np.minimum(o, c) * (1 - spread)
        v = vol_base * np.exp(rng.normal(0, 0.35, n_days)) * (1 + 8 * np.abs(ret))
        return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": np.round(v)}, index=idx)

    px = {"SPY": bars(mkt, 400, 6e7), "QQQ": bars(mkt * 1.2 + rng.normal(0, 0.003, n_days), 350, 4e7),
          "IWM": bars(mkt * 1.1 + rng.normal(0, 0.004, n_days), 200, 3e7)}
    vix = 18 * np.exp(np.cumsum(-4 * mkt + rng.normal(0, 0.03, n_days)) * 0.5)
    vix = np.clip(vix, 10, 60)
    px["^VIX"] = pd.DataFrame({"open": vix, "high": vix * 1.03, "low": vix * 0.97, "close": vix, "volume": 0.0}, index=idx)
    for s, etf in SECTOR_ETF.items():
        px[etf] = bars(mkt + sec_f[s], 100, 1e7)
    uni = {}
    for i in range(n_tickers):
        t = f"SYN{i:03d}"
        s = sectors[i % len(sectors)]
        drift = rng.normal(0.0003, 0.0006)
        ret = mkt * rng.uniform(0.6, 1.6) + sec_f[s] * rng.uniform(0.5, 1.2) + rng.normal(drift, 0.016, n_days)
        jumps = np.zeros(n_days)
        if i % 4 == 0:  # an earnings-style gap now and then
            for k in rng.choice(np.arange(300, n_days - 5), 8, replace=False):
                jumps[k] = rng.choice([0.08, -0.08, 0.1])
            ret = ret + jumps
        df = bars(ret, rng.uniform(15, 300), rng.uniform(8e5, 9e6), jumps)
        if i % 11 == 0:  # late listing
            df.iloc[: rng.integers(100, 500)] = np.nan
            df = df.dropna()
        px[t] = df
        uni[t] = {"name": f"Synthetic {i}", "sector": s}
    return px, uni
