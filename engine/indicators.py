"""Indicators for one ticker, aligned to the master trading calendar.

Every field on the returned namespace is a float64 numpy array with one value per
trading day. NaN means "unknown" (not enough history, or the stock did not trade yet).
Boolean facts are stored as 1.0 / 0.0.
"""
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

SECTOR_ETF = {
    "Information Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}
INDEX_TICKERS = ["SPY", "QQQ", "IWM", "^VIX"]


def _rsi(c, n):
    dlt = c.diff()
    up = dlt.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-dlt.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


def _streak(b):
    b = b.fillna(False).astype(bool)
    return b.astype(int).groupby((~b).cumsum()).cumsum().astype("float64")


def _since_max(s, n):
    """Days since the highest value of the last n days (0 = today)."""
    return (n - 1) - s.rolling(n).apply(np.argmax, raw=True)


def _f(b):
    return b.astype("float64")


def _any(b, n):
    return _f(b).rolling(n).max()


def _cnt(b, n):
    return _f(b).rolling(n).sum()


def _ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _tr(h, l, pc):
    return pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


# --------------------------------------------------------------------------- market


def build_market(px, closes, sectors_of):
    """px: dict ticker -> DataFrame(open, high, low, close, volume) on the master index.
    closes: DataFrame (dates x universe tickers) of closes.
    sectors_of: dict ticker -> sector name.
    Returns (M, S, X): market namespace, sector dict, cross-sectional rank dict."""
    idx = closes.index
    P = {}

    def idx_block(name, key):
        df = px[key]
        c = df["close"]
        P[f"{name}_c"] = c
        for n in (20, 50, 200):
            P[f"{name}_sma{n}"] = c.rolling(n).mean()
        for n in (1, 5, 21, 63, 126):
            P[f"{name}_ret{n}"] = c / c.shift(n) - 1
        return df

    spy = idx_block("spy", "SPY")
    idx_block("qqq", "QQQ")
    idx_block("iwm", "IWM")
    sc, sh_, sl, sv = spy["close"], spy["high"], spy["low"], spy["volume"]
    spc = sc.shift(1)
    P["spy_sma50_p10"] = P["spy_sma50"].shift(10)
    P["spy_sma200_p21"] = P["spy_sma200"].shift(21)
    P["spy_dist_high"] = sc / sh_.rolling(252).max() - 1
    P["spy_dist25"] = _cnt((sc < spc * 0.998) & (sv > sv.shift(1)), 25)
    P["spy_rsi14"] = _rsi(sc, 14)
    P["spy_worst10"] = P["spy_ret1"].rolling(10).min()
    P["spy_up"] = _f(sc > spc)
    P["spy_atrp"] = _tr(sh_, sl, spc).rolling(14).mean() / sc
    vix = px["^VIX"]["close"]
    P["vix"] = vix
    P["vix_sma20"] = vix.rolling(20).mean()
    P["vix_chg5"] = vix / vix.shift(5) - 1

    # breadth across the whole universe
    C = closes
    valid = C.notna()
    nvalid = valid.sum(axis=1).replace(0, np.nan)
    above = {}
    for n in (20, 50, 200):
        above[n] = (C > C.rolling(n).mean()) & valid
        P[f"br_above{n}"] = above[n].sum(axis=1) / nvalid * 100
    P["br_above50_p5"] = P["br_above50"].shift(5)
    nh = ((C >= C.rolling(55).max()) & valid).sum(axis=1)
    nl = ((C <= C.rolling(55).min()) & valid).sum(axis=1)
    P["br_nhnl"] = (nh - nl).astype("float64")
    P["br_nhnl10"] = P["br_nhnl"].rolling(10).mean()
    adv = ((C > C.shift(1)) & valid).sum(axis=1).astype("float64")
    dec = ((C < C.shift(1)) & valid).sum(axis=1).astype("float64")
    P["br_advdec"] = adv - dec
    P["br_ad10"] = adv.rolling(10).sum() / dec.rolling(10).sum()
    for k in ("br_above20", "br_above50", "br_above200", "br_nhnl", "br_advdec"):
        P[k] = P[k].where(sc.notna())

    M = NS(**{k: s.to_numpy(dtype="float64") for k, s in P.items()})
    M.index = idx
    M.spy_close_series = sc

    # regime: 2 = green, 1 = yellow, 0 = red
    with np.errstate(invalid="ignore"):
        green = (M.spy_c > M.spy_sma200) & (M.spy_c > M.spy_sma50) & (M.br_above50 >= 50) & (M.vix < 25)
        red = (M.spy_c < M.spy_sma200) | (M.vix > 30) | ((M.spy_c < M.spy_sma50) & (M.br_above50 < 35))
    M.regime = np.where(red, 0.0, np.where(green, 2.0, 1.0))
    M.regime[np.isnan(M.spy_sma200) | np.isnan(M.vix)] = np.nan

    # sectors
    S = {}
    etf_ret21, etf_ret63 = {}, {}
    for sec, etf in SECTOR_ETF.items():
        if etf in px:
            etf_ret21[sec] = px[etf]["close"] / px[etf]["close"].shift(21) - 1
            etf_ret63[sec] = px[etf]["close"] / px[etf]["close"].shift(63) - 1
    rank21 = pd.DataFrame(etf_ret21).rank(axis=1, ascending=False)
    rank63 = pd.DataFrame(etf_ret63).rank(axis=1, ascending=False)
    for sec, etf in SECTOR_ETF.items():
        if etf not in px:
            continue
        e = px[etf]
        ec = e["close"]
        members = [t for t in C.columns if sectors_of.get(t) == sec]
        Q = {
            "sec_c": ec,
            "sec_sma50": ec.rolling(50).mean(),
            "sec_sma200": ec.rolling(200).mean(),
            "sec_ret21": etf_ret21[sec],
            "sec_ret63": etf_ret63[sec],
            "sec_rank21": rank21[sec],
            "sec_rank63": rank63[sec],
            "sec_dist_high": ec / e["high"].rolling(252).max() - 1,
            "sec_up": _f(ec > ec.shift(1)).where(ec.notna()),
        }
        Q["sec_sma50_p10"] = Q["sec_sma50"].shift(10)
        if members:
            mv = valid[members].sum(axis=1).replace(0, np.nan)
            Q["sec_breadth50"] = above[50][members].sum(axis=1) / mv * 100
        else:
            Q["sec_breadth50"] = pd.Series(np.nan, index=idx)
        S[sec] = NS(**{k: s.to_numpy(dtype="float64") for k, s in Q.items()})
    nan = np.full(len(idx), np.nan)
    S[None] = NS(**{k: nan for k in vars(next(iter(S.values())))}) if S else NS()

    # cross-sectional relative-strength percentile (0-100) inside the universe
    X = {}
    for n in (21, 63, 126, 252):
        r = C / C.shift(n) - 1
        X[n] = r.rank(axis=1, pct=True) * 100
    return M, S, X


# --------------------------------------------------------------------------- ticker


def build(df, M, sec, xr, cfg, ext=None):
    """df: OHLCV DataFrame on the master index. M: market namespace. sec: sector namespace.
    xr: dict n -> Series of universe percentile ranks for this ticker.
    ext: optional {"earn_days": int|None, "mcap": float|None} known only for today."""
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    pc = c.shift(1)
    P = {"o": o, "h": h, "l": l, "c": c, "v": v, "pc": pc}
    P["nbars"] = c.notna().cumsum().astype("float64")

    # moving averages
    for n in (5, 10, 20, 50, 100, 150, 200):
        P[f"sma{n}"] = c.rolling(n).mean()
    P["ema9"], P["ema21"] = _ema(c, 9), _ema(c, 21)
    P["sma20_p5"] = P["sma20"].shift(5)
    P["sma50_p10"] = P["sma50"].shift(10)
    P["sma150_p21"] = P["sma150"].shift(21)
    P["sma200_p21"] = P["sma200"].shift(21)
    P["sma200_p63"] = P["sma200"].shift(63)
    P["pct_above50_50d"] = _cnt(c > P["sma50"], 50) / 50 * 100
    P["all_above20_10d"] = _f(_cnt(c > P["sma20"], 10) >= 10)

    # ranges, highs, lows
    tr = _tr(h, l, pc)
    for n in (5, 14, 20, 50):
        P[f"atr{n}"] = tr.rolling(n).mean()
    atr = P["atr14"]
    P["atrp"] = atr / c
    for n in (3, 5, 10, 14, 20, 55, 126, 252):
        P[f"hh{n}"] = h.rolling(n).max()
        P[f"ll{n}"] = l.rolling(n).min()
        P[f"hp{n}"] = P[f"hh{n}"].shift(1)  # prior n days, excluding today
        P[f"lp{n}"] = P[f"ll{n}"].shift(1)
    for n in (20, 55, 252):
        P[f"hc{n}"] = c.rolling(n).max()
        P[f"hcp{n}"] = P[f"hc{n}"].shift(1)
    P["hh20_p20"] = P["hh20"].shift(20)
    P["ll20_p20"] = P["ll20"].shift(20)
    P["ll20_p10"] = P["ll20"].shift(10)
    P["hh5_p5"] = P["hh5"].shift(5)
    P["ll5_p5"] = P["ll5"].shift(5)
    P["hh252_p63"] = P["hh252"].shift(63)
    P["mid252"] = (P["hh252"] + P["ll252"]) / 2
    P["mid55"] = (P["hh55"] + P["ll55"]) / 2
    P["pos20"] = (c - P["ll20"]) / (P["hh20"] - P["ll20"])
    P["pos10"] = (c - P["ll10"]) / (P["hh10"] - P["ll10"])
    P["pos5"] = (c - P["ll5"]) / (P["hh5"] - P["ll5"])
    P["dist52h"] = c / P["hh252"] - 1
    P["dist52l"] = c / P["ll252"] - 1
    P["dd20"] = c / P["hh20"] - 1
    P["dsh20"] = _since_max(h, 20)
    P["base_age"] = _since_max(h.shift(1), 55) + 1
    P["base_depth"] = (P["hh55"] - P["ll20"]) / P["hh55"]
    for n in (5, 10, 20):
        P[f"range{n}"] = (P[f"hh{n}"] - P[f"ll{n}"]) / c
    P["range10_p10"] = P["range10"].shift(10)
    P["range10_p20"] = P["range10"].shift(20)

    # returns and momentum
    for n in (1, 2, 3, 5, 10, 21, 63, 126, 252):
        P[f"ret{n}"] = c / c.shift(n) - 1
    P["ret21_p21"] = P["ret21"].shift(21)
    P["ret21_p5"] = P["ret21"].shift(5)
    P["mom12_1"] = c.shift(21) / c.shift(252) - 1
    P["rsi14"], P["rsi2"] = _rsi(c, 14), _rsi(c, 2)
    P["rsi14_p5"] = P["rsi14"].shift(5)
    P["rsi14_min20"] = P["rsi14"].rolling(20).min()
    P["rsi14_max20"] = P["rsi14"].rolling(20).max()
    macd = _ema(c, 12) - _ema(c, 26)
    P["macd"] = macd
    P["macds"] = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    P["macdh"] = P["macd"] - P["macds"]
    P["macdh_p1"] = P["macdh"].shift(1)
    P["stoch"] = (c - P["ll14"]) / (P["hh14"] - P["ll14"]) * 100
    up, dn = c > pc, c < pc
    P["up"], P["dn"] = _f(up), _f(dn)
    P["updays20"] = _cnt(up, 20)
    P["up_streak"], P["dn_streak"] = _streak(up), _streak(dn)
    P["newhc20_5"] = _any(c >= P["hc20"], 5)
    P["newhc55_10"] = _any(c >= P["hc55"], 10)
    P["newhh252_21"] = _any(h >= P["hh252"], 21)
    P["newll55_20"] = _any(l <= P["ll55"], 20)

    # ADX
    um, dm = h.diff(), -l.diff()
    pdm = pd.Series(np.where((um > dm) & (um > 0), um, 0.0), index=c.index).where(pc.notna())
    mdm = pd.Series(np.where((dm > um) & (dm > 0), dm, 0.0), index=c.index).where(pc.notna())
    atrw = tr.where(pc.notna()).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atrw
    mdi = 100 * mdm.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean() / atrw
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    P["pdi"], P["mdi"] = pdi, mdi
    P["adx"] = dx.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    P["adx_p5"] = P["adx"].shift(5)

    # volume
    for n in (5, 10, 20, 50):
        P[f"v{n}"] = v.rolling(n).mean()
    P["rvol"] = v / P["v20"].shift(1)
    dv = c * v
    P["dv20"], P["dv50"] = dv.rolling(20).mean(), dv.rolling(50).mean()
    P["dv20_p63"] = P["dv20"].shift(63)
    P["dvmin20"] = dv.rolling(20).min()
    P["zero_vol60"] = _cnt(v <= 0, 60)
    upv, dnv = v.where(up, 0.0).where(pc.notna()), v.where(dn, 0.0).where(pc.notna())
    P["udr20"] = upv.rolling(20).sum() / dnv.rolling(20).sum()
    P["udr50"] = upv.rolling(50).sum() / dnv.rolling(50).sum()
    obv = (np.sign(c - pc).fillna(0) * v.fillna(0)).cumsum().where(c.notna())
    P["obv"] = obv
    P["obv_sma20"] = obv.rolling(20).mean()
    P["obv_p20"] = obv.shift(20)
    P["obv_newhigh10"] = _any(obv >= obv.rolling(55).max(), 10)
    P["acc20"] = _cnt(up & (v > P["v50"].shift(1)), 20)
    distday = (c < pc * 0.998) & (v > v.shift(1))
    P["dist25"], P["dist5"] = _cnt(distday, 25), _cnt(distday, 5)
    pp = up & (v > dnv.shift(1).rolling(10).max())
    P["pp"], P["pp10"] = _f(pp), _any(pp, 10)
    P["maxvol20_up"] = _f(upv.rolling(20).max() >= v.rolling(20).max())
    P["maxvol50_up"] = _f(upv.rolling(50).max() >= v.rolling(50).max())
    P["last_dn_rvol"] = P["rvol"].where(dn).ffill()
    P["last_up_rvol"] = P["rvol"].where(up).ffill()
    P["vsum5"] = v.rolling(5).sum()
    P["vsum5_p5"] = P["vsum5"].shift(5)
    rng = h - l
    mfm = (((c - l) - (h - c)) / rng).where(rng > 0, 0.0).where(c.notna())
    P["cmf20"] = (mfm * v).rolling(20).sum() / v.rolling(20).sum()
    adl = (mfm * v).fillna(0).cumsum().where(c.notna())
    P["adl"], P["adl_sma20"] = adl, adl.rolling(20).mean()
    tp = (h + l + c) / 3
    rmf = tp * v
    pos = rmf.where(tp > tp.shift(1), 0.0).rolling(14).sum()
    neg = rmf.where(tp < tp.shift(1), 0.0).rolling(14).sum()
    P["mfi"] = 100 - 100 / (1 + pos / neg)
    P["vwap20"] = (tp * v).rolling(20).sum() / v.rolling(20).sum()
    P["vwap50"] = (tp * v).rolling(50).sum() / v.rolling(50).sum()
    P["heavy_dn10"] = _any(dn & (P["rvol"] > 2), 10)
    P["climax_dn10"] = _any(dn & (P["rvol"] > 4), 10)
    P["vmax3_p"] = v.shift(1).rolling(3).max()
    hv_close = c.where(v >= v.rolling(50).max()).ffill()
    P["hv50_close"] = hv_close

    # candles
    P["clv"] = ((c - l) / rng).where(rng > 0, 0.5).where(c.notna())
    P["body"] = ((c - o).abs() / rng).where(rng > 0, 0.0).where(c.notna())
    P["uw"] = ((h - np.maximum(o, c)) / rng).where(rng > 0, 0.0).where(c.notna())
    P["lw"] = ((np.minimum(o, c) - l) / rng).where(rng > 0, 0.0).where(c.notna())
    P["rng"] = rng
    P["gap"] = o / pc - 1
    P["h_p1"], P["l_p1"] = h.shift(1), l.shift(1)
    P["mid_p1"] = (h.shift(1) + l.shift(1)) / 2
    P["c_p3"] = c.shift(3)
    inside = (h <= h.shift(1)) & (l >= l.shift(1))
    nr7 = rng <= rng.rolling(7).min()
    P["inside"], P["nr7"] = _f(inside), _f(nr7)
    P["quiet3"] = _any(inside | nr7, 3)
    P["inside_break"] = _f(inside.shift(1, fill_value=False) & (c > h.shift(1)))
    P["engulf_bull"] = _f((c > o) & (c.shift(1) < o.shift(1)) & (c >= o.shift(1)) & (o <= c.shift(1)))
    bear_eng = (c < o) & (c.shift(1) > o.shift(1)) & (o >= c.shift(1)) & (c <= o.shift(1))
    P["engulf_bear_hv3"] = _any(bear_eng & (P["rvol"] > 1.5), 3)
    P["reversal_up"] = _f((l < P["lp5"]) & (c > pc))
    P["wide_dn5"] = _any(dn & (rng > 1.5 * atr), 5)
    P["wide_any10"] = _any(tr > 3 * atr.shift(1), 10)
    P["upper_half5"] = _cnt(P["clv"] >= 0.5, 5)
    P["bear_rev5"] = _any((h >= P["hh20"]) & (P["clv"] <= 0.25) & (P["rvol"] > 1.5), 5)
    P["close_below50_hv10"] = _any((c < P["sma50"]) & dn & (P["rvol"] > 1.5), 10)
    P["closed_below200_20"] = _any(c < P["sma200"], 20)
    P["death_cross40"] = _any((P["sma50"] < P["sma200"]) & (P["sma50"].shift(1) >= P["sma200"].shift(1)), 40)
    P["tight3"] = (c.rolling(3).max() - c.rolling(3).min()) / c

    # gaps
    P["worst_gap60"] = P["gap"].rolling(60).min()
    P["worst_gap20"] = P["gap"].rolling(20).min()
    P["worst_day60"] = P["ret1"].rolling(60).min()
    P["absgap20"] = P["gap"].abs().rolling(20).mean()
    P["biggaps60"] = _cnt(P["gap"].abs() > 0.04, 60)
    P["med_range20"] = (rng / c).rolling(20).median()

    def open_gap(th, win):
        out = pd.Series(0.0, index=c.index)
        for k in range(win):
            was_gap = (P["gap"].shift(k) >= th)
            floor = pc.shift(k)  # close before the gap day
            low_since = l.rolling(k + 1).min() if k else l
            out = np.maximum(out, _f(was_gap & (low_since > floor)))
        return out

    P["open_gap4_5"] = open_gap(0.04, 5)
    P["open_gap2_20"] = open_gap(0.02, 20)

    # volatility / contraction
    sd20 = c.rolling(20).std()
    bbu, bbl = P["sma20"] + 2 * sd20, P["sma20"] - 2 * sd20
    P["bbu"], P["bbl"] = bbu, bbl
    bbw = (bbu - bbl) / P["sma20"]
    P["bbw_rank"] = bbw.rolling(120).rank(pct=True)
    ema20 = _ema(c, 20)
    squeeze = (bbu < ema20 + 1.5 * P["atr20"]) & (bbl > ema20 - 1.5 * P["atr20"])
    P["squeeze"] = _f(squeeze)
    P["squeeze_fired5"] = _f((_any(squeeze, 6) > 0) & ~squeeze)
    r1 = c.pct_change(fill_method=None)
    for n in (10, 21, 50, 63):
        P[f"sd{n}"] = r1.rolling(n).std()

    # relative strength
    spy_c = pd.Series(M.spy_c, index=c.index)
    rs = c / spy_c
    P["rs"] = rs
    P["rs_sma20"], P["rs_sma50"] = rs.rolling(20).mean(), rs.rolling(50).mean()
    P["rs_sma50_p10"] = P["rs_sma50"].shift(10)
    P["rs_newhigh55_5"] = _any(rs >= rs.rolling(55).max(), 5)
    P["rs_newhigh126_10"] = _any(rs >= rs.rolling(126).max(), 10)
    P["rs_lead"] = _f((rs >= rs.rolling(55).max()) & (c < P["hc55"]))
    P["rs_ll55"] = rs.rolling(55).min()
    for n in (5, 21, 63, 126):
        P[f"ex{n}"] = P[f"ret{n}"] - pd.Series(getattr(M, f"spy_ret{n}"), index=c.index)
    P["ex63_qqq"] = P["ret63"] - pd.Series(M.qqq_ret63, index=c.index)
    spy_r1 = pd.Series(M.spy_ret1, index=c.index)
    spy_dn = spy_r1 < 0
    P["downcap_stock"] = P["ret1"].where(spy_dn, 0.0).rolling(21).sum()
    P["downcap_spy"] = spy_r1.where(spy_dn, 0.0).rolling(21).sum()
    for n in (21, 63, 126, 252):
        P[f"rk{n}"] = xr[n]
    P["rk63_p21"] = xr[63].shift(21)
    P["rkc"] = 0.4 * xr[63] + 0.2 * xr[126] + 0.2 * xr[252] + 0.2 * xr[21]

    # generic trade geometry (the setup-specific stop is in setups.py)
    stop = np.minimum(c - atr, np.maximum(P["ll10"] - 0.1 * atr, c - 2.5 * atr))
    P["stop_g"] = stop
    risk = (c - stop) / c
    P["risk_g"] = risk
    P["risk_atr"] = (c - stop) / atr
    P["target_g"] = c + 2 * (c - stop)
    P["best15_63"] = (c / c.shift(15) - 1).rolling(63).max()
    equity, capf = cfg["equity_usd"], cfg["max_position_pct"] / 100
    risk_usd = equity * cfg["risk_per_trade_pct"] / 100
    shares = np.floor(np.minimum(risk_usd / (c - stop), equity * capf / c))
    P["shares_g"] = shares
    P["fee_share"] = (2 * cfg["commission_per_trade_usd"]) / (shares * (c - stop)).where(shares > 0)
    P["room55_R"] = ((P["hh55"] - c) / (c - stop)).where(c < P["hh55"] * 0.99, 99.0)

    # external facts, known only for the last day
    n = len(c)
    earn = np.full(n, np.nan)
    mcap = np.full(n, np.nan)
    if ext:
        if ext.get("earn_days") is not None:
            earn[-1] = ext["earn_days"]
        if ext.get("mcap") is not None:
            mcap[-1] = ext["mcap"]

    d = NS()
    cna = c.isna().to_numpy()
    for k, s in P.items():
        a = np.asarray(s, dtype="float64").copy()
        a[cna] = np.nan
        a[~np.isfinite(a)] = np.nan
        setattr(d, k, a)
    d.earn_days, d.mcap = earn, mcap
    d.cfg = cfg
    d.M, d.S = M, sec
    d.index = c.index
    return d
