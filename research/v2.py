"""Version 2 research. See RESEARCH.md for the plan that was written before this ran.

    python -m research.v2 prepare  DATA.npz CACHE_DIR        # indicators, 300 checks, signals, exits
    python -m research.v2 select   CACHE_DIR                 # train + validation only
    python -m research.v2 holdout  CACHE_DIR                 # the single look at the test period

Nothing in `select` reads a signal or a day after the validation period.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from engine import backtest as bt
from engine import checks, pipeline, setups
from engine.indicators import SECTOR_ETF, build, build_market

TRAIN = ("2022-10-25", "2024-06-28")
VALID = ("2024-07-01", "2025-06-30")
TEST = ("2025-07-01", "2099-12-31")
KS = list(setups.PRIORITY)                     # BRK GAP TGT PB MR
EXITS = ["V1", "E0", "E1", "E2", "E3"]
EXIT_DEF = {                                   # rr, tmax, trail (ATR), breakeven
    "E0": (2.0, 15, None, False), "E1": (3.0, 30, None, False),
    "E2": (None, 40, 3.0, False), "E3": (3.0, 30, None, True),
}
SIZINGS = {"base": (1.0, 20), "big": (1.5, 30)}   # risk % of equity, position cap %
EQUITY = 10000.0


FEE_MIN = 7.5      # IBI agreement: 1 cent per share, minimum $7.5 per trade
FEE_SHARE = 0.01


def fee(shares):
    return np.maximum(FEE_MIN, FEE_SHARE * shares)


# ---------------------------------------------------------------- prepare


def load(path):
    z = np.load(path, allow_pickle=False)
    idx = pd.to_datetime(z["dates"])
    px, uni = {}, {}
    for j, t in enumerate(z["tickers"]):
        t = str(t)
        px[t] = pd.DataFrame({"open": z["o"][:, j], "high": z["h"][:, j], "low": z["l"][:, j], "close": z["c"][:, j],
                              "volume": z["v"][:, j]}, index=idx).astype("float64")
        if z["in_universe"][j]:
            uni[t] = {"name": str(z["names"][j]), "sector": str(z["sectors"][j]) or None}
    return px, uni, [str(d) for d in z["dates"]]


def sim_exit(o, h, l, c, i, stop, atr_i, rr, tmax, trail, be):
    n = len(c)
    e = i + 1
    if e >= n:
        return None
    entry = o[e]
    if not np.isfinite(entry) or not np.isfinite(stop) or entry <= stop or entry > c[i] + atr_i:
        return None
    risk = entry - stop
    target = entry + rr * risk if rr else np.inf
    cur, hc = stop, entry
    last = min(e + tmax, n) - 1
    for j in range(e, last + 1):
        if not np.isfinite(c[j]):
            continue
        if j > e and o[j] <= cur:
            return e, entry, j, o[j]
        if l[j] <= cur:
            return e, entry, j, cur
        if j > e and o[j] >= target:
            return e, entry, j, o[j]
        if h[j] >= target:
            return e, entry, j, target
        hc = max(hc, c[j])                       # only today's close; used from tomorrow on
        if trail:
            cur = max(cur, hc - trail * atr_i)
        if be and c[j] >= entry + risk:
            cur = max(cur, entry)
    if last < e + tmax - 1:
        return None                              # still open at the end of the data
    return e, entry, last, c[last]


def prepare(data_path, out):
    os.makedirs(out, exist_ok=True)
    cfg = pipeline.load_config()
    px, uni, dates = load(data_path)
    idx = px["SPY"].index
    tickers = [t for t in uni if t in px]
    sectors_of = {t: uni[t]["sector"] for t in tickers}
    closes = pd.DataFrame({t: px[t]["close"] for t in tickers}, index=idx)
    M, S, X = build_market(px, closes, sectors_of)
    T, N, K = len(idx), len(tickers), len(checks.C)
    bits = np.full((T, N, K), -1, dtype=np.int8)
    f32 = lambda: np.full((T, N), np.nan, dtype="float32")
    O, H, L, C, ATR = f32(), f32(), f32(), f32(), f32()
    gate = np.zeros((T, N), dtype=bool)
    sig = []                                     # (j, i, k, stop, atr, close, v1 score)
    t0 = time.time()
    for j, t in enumerate(tickers):
        d = build(px[t], M, S.get(sectors_of[t], S[None]), {n: X[n][t] for n in X}, cfg)
        mat = checks.run_all(d)
        b = np.where(np.isnan(mat), -1, mat).astype(np.int8)
        bits[:, j, :] = b
        O[:, j], H[:, j], L[:, j], C[:, j], ATR[:, j] = d.o, d.h, d.l, d.c, d.atr14
        gate[:, j] = setups.gates(d)
        sc = checks.score(mat)
        su, _ = setups.compute(d)
        for k, name in enumerate(KS):
            s, stp = su[name]
            prev = -99
            for i in np.flatnonzero(s[260:]) + 260:
                if i - prev <= 5:
                    continue
                prev = i
                sig.append((j, i, k, stp[i], d.atr14[i], d.c[i], sc[i]))
        if (j + 1) % 100 == 0:
            print(f"  {j + 1}/{N}  {time.time() - t0:.0f}s", flush=True)
    sig = np.array(sig, dtype="float64")
    n = len(sig)
    sj, si, sk = sig[:, 0].astype(int), sig[:, 1].astype(int), sig[:, 2].astype(int)
    # exits: entry price, exit price, exit index (NaN when skipped or still open)
    ent = np.full((n, len(EXITS)), np.nan)
    ext = np.full((n, len(EXITS)), np.nan)
    xi = np.full((n, len(EXITS)), -1, dtype=int)
    for a in range(n):
        j, i, k = sj[a], si[a], sk[a]
        o, h, l, c = O[:, j].astype("float64"), H[:, j].astype("float64"), L[:, j].astype("float64"), C[:, j].astype("float64")
        for e, name in enumerate(EXITS):
            if name == "V1":
                rr, tmax, trail, be = setups.SETUPS[KS[k]][1], setups.SETUPS[KS[k]][2], None, False
            else:
                rr, tmax, trail, be = EXIT_DEF[name]
            r = sim_exit(o, h, l, c, i, sig[a, 3], sig[a, 4], rr, tmax, trail, be)
            if r:
                ent[a, e], ext[a, e], xi[a, e] = r[1], r[3], r[2]
    sbits = bits[si, sj, :]
    np.save(os.path.join(out, "bits.npy"), bits)
    np.savez(os.path.join(out, "core.npz"), O=O, H=H, L=L, C=C, ATR=ATR, gate=gate, regime=M.regime, spy=M.spy_c,
             sig=sig, ent=ent, ext=ext, xi=xi, sbits=sbits, dates=np.array(dates), tickers=np.array(tickers),
             sectors=np.array([sectors_of[t] or "" for t in tickers]))
    print(f"prepared: {T} days, {N} tickers, {n} signals, {time.time() - t0:.0f}s")


# ---------------------------------------------------------------- shared helpers


class Cache:
    def __init__(self, d):
        z = np.load(os.path.join(d, "core.npz"))
        for k in z.files:
            setattr(self, k, z[k])
        self.dates = [str(x) for x in self.dates]
        self.dir = d
        self.sj, self.si, self.sk = self.sig[:, 0].astype(int), self.sig[:, 1].astype(int), self.sig[:, 2].astype(int)
        self._bits = None

    @property
    def bits(self):
        if self._bits is None:
            self._bits = np.load(os.path.join(self.dir, "bits.npy"), mmap_mode="r")
        return self._bits

    def span(self, period):
        a = np.searchsorted(self.dates, period[0], side="left")
        b = np.searchsorted(self.dates, period[1], side="right") - 1
        return int(a), int(min(b, len(self.dates) - 1))


def net_r(Z, e, sizing):
    """Net result of every signal under exit e, in R (planned risk), plus shares."""
    risk_pct, cap_pct = SIZINGS[sizing]
    risk_usd = EQUITY * risk_pct / 100
    entry, exit_, stop = Z.ent[:, e], Z.ext[:, e], Z.sig[:, 3]
    with np.errstate(invalid="ignore", divide="ignore"):
        sh = np.floor(np.minimum(risk_usd / (entry - stop), EQUITY * cap_pct / 100 / entry))
        r = (sh * (exit_ - entry) - 2 * fee(sh)) / risk_usd
    r[~(sh >= 1)] = np.nan
    return r


def stats(r):
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return {"n": int(len(r)), "avg": None, "t": None, "win": None}
    sd = r.std(ddof=1)
    return {"n": int(len(r)), "avg": round(float(r.mean()), 3), "t": round(float(r.mean() / (sd / np.sqrt(len(r)))), 2) if sd > 0 else None,
            "win": round(float((r > 0).mean() * 100), 1)}


def learn(bits, r, tmin, min_n=30):
    """Weight per check: Welch t of (R when passed) minus (R when failed), on the given signals."""
    P, F = (bits == 1).astype("float64"), (bits == 0).astype("float64")
    npass, nfail = P.sum(0), F.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mp, mf = P.T @ r / npass, F.T @ r / nfail
        vp, vf = P.T @ (r * r) / npass - mp**2, F.T @ (r * r) / nfail - mf**2
        t = (mp - mf) / np.sqrt(vp / npass + vf / nfail)
    t[(npass < min_n) | (nfail < min_n) | ~np.isfinite(t)] = 0.0
    return np.where(np.abs(t) >= tmin, np.clip(t, -4, 4), 0.0)


def score(bits, w):
    X = (bits == 1).astype("float64")
    X[bits == -1] = 0.5
    return X @ w


def masks(Z, e, period):
    """Signals whose signal day is in the period and whose exit completed inside it."""
    a, b = Z.span(period)
    return (Z.si >= a) & (Z.si <= b) & (Z.xi[:, e] >= 0) & (Z.xi[:, e] <= b)


def s_portfolio(Z, e, sizing, keep_mask, rank, period, max_pos, max_new=2, max_sector=2):
    """Portfolio simulation over precomputed trades (entry at next open, exits as simulated)."""
    a, b = Z.span(period)
    risk_pct, cap_pct = SIZINGS[sizing]
    cand = np.flatnonzero(keep_mask & (Z.si >= a) & (Z.si <= b) & (Z.xi[:, e] >= 0) & (Z.xi[:, e] <= b))
    by_day = {}
    for s in cand:
        by_day.setdefault(Z.si[s], []).append(s)
    cash, pos, pending, curve, trades = EQUITY, [], [], [], []
    equity = EQUITY
    for t in range(a, b + 1):
        for s in pending:
            entry, stop = Z.ent[s, e], Z.sig[s, 3]
            mult = {2: 1.0, 1: 0.5}.get(int(Z.regime[t - 1]) if np.isfinite(Z.regime[t - 1]) else 1, 0.0)
            sh = int(np.floor(min(equity * risk_pct / 100 * mult / (entry - stop), equity * cap_pct / 100 / entry,
                                  max(cash - FEE_MIN, 0) / entry)))
            if sh >= 1:
                cash -= sh * entry + fee(sh)
                pos.append((s, sh))
        pending = []
        keep = []
        for s, sh in pos:
            if Z.xi[s, e] == t:
                cash += sh * Z.ext[s, e] - fee(sh)
                pnl = sh * (Z.ext[s, e] - Z.ent[s, e]) - 2 * fee(sh)
                trades.append(pnl)
            else:
                keep.append((s, sh))
        pos = keep
        equity = cash + sum(sh * (Z.C[t, Z.sj[s]] if np.isfinite(Z.C[t, Z.sj[s]]) else Z.ent[s, e]) for s, sh in pos)
        curve.append(equity)
        reg = Z.regime[t]
        if not np.isfinite(reg) or reg < 1:
            continue
        slots = min(max_pos - len(pos), max_new)
        if slots <= 0 or t not in by_day:
            continue
        held = {Z.sj[s] for s, _ in pos}
        secs = {}
        for s, _ in pos:
            secs[Z.sectors[Z.sj[s]]] = secs.get(Z.sectors[Z.sj[s]], 0) + 1
        for s in sorted(by_day[t], key=lambda x: -rank[x]):
            if slots <= 0:
                break
            j = Z.sj[s]
            sec = Z.sectors[j]
            if j in held or (sec and secs.get(sec, 0) >= max_sector):
                continue
            pending.append(s)
            held.add(j)
            secs[sec] = secs.get(sec, 0) + 1
            slots -= 1
    return port_stats(Z, np.array(curve), a, b, trades)


def port_stats(Z, curve, a, b, trades, start=EQUITY):
    spy = Z.spy[a:b + 1]
    with np.errstate(invalid="ignore"):
        ew = np.nancumprod(np.r_[1.0, 1 + np.nan_to_num(np.nanmean(Z.C[a + 1:b + 1] / Z.C[a:b] - 1, axis=1))])
    peak = np.maximum.accumulate(np.r_[start, curve])[1:]
    tr = np.array(trades, dtype="float64")
    return {"ret": round(float(curve[-1] / start - 1) * 100, 1), "maxdd": round(float((curve / peak - 1).min()) * 100, 1),
            "spy": round(float(spy[-1] / spy[0] - 1) * 100, 1),
            "spy_maxdd": round(float((spy / np.maximum.accumulate(spy) - 1).min()) * 100, 1),
            "ew": round(float(ew[-1] - 1) * 100, 1), "trades": int(len(tr)),
            "win": round(float((tr > 0).mean() * 100), 0) if len(tr) else None,
            "days": int(b - a + 1), "from": Z.dates[a], "to": Z.dates[b]}


# ---------------------------------------------------------------- family S


def s_configs():
    for g in KS + ["ALL"]:
        for e in range(len(EXITS)):
            yield g, e, None, 1.0
            for tmin in (1.5, 2.0):
                for keepf in (0.25, 0.5):
                    yield g, e, tmin, keepf


def s_eval(Z, g, e, tmin, keepf, fit, test, sizing="base"):
    """Learn on the `fit` periods, evaluate on `test`. Returns stats and the selection mask."""
    r = net_r(Z, e, sizing)
    grp = np.ones(len(r), bool) if g == "ALL" else (Z.sk == KS.index(g))
    ok = grp & np.isfinite(r)
    fm = np.zeros(len(r), bool)
    for p in fit:
        fm |= masks(Z, e, p)
    fm &= ok
    tm = masks(Z, e, test) & ok
    if tmin is None:
        return {"fit": stats(r[fm]), "test": stats(r[tm])}, tm, np.zeros(len(r)), None
    w = learn(Z.sbits[fm], r[fm], tmin)
    sc = score(Z.sbits, w)
    if not fm.any() or not np.any(w):
        return {"fit": stats(r[fm][:0]), "test": stats(r[tm][:0])}, tm & False, sc, w
    thr = np.quantile(sc[fm], 1 - keepf)
    sel = sc >= thr
    return {"fit": stats(r[fm & sel]), "test": stats(r[tm & sel]), "thr": float(thr), "n_weights": int(np.count_nonzero(w))}, tm & sel, sc, w


def select_s(Z):
    rows = []
    for g, e, tmin, keepf in s_configs():
        st, _, _, _ = s_eval(Z, g, e, tmin, keepf, [TRAIN], VALID)
        rows.append({"group": g, "exit": EXITS[e], "tmin": tmin, "keep": keepf, "train": st["fit"], "valid": st["test"]})
    ok = [x for x in rows if x["valid"]["n"] >= 60 and (x["train"]["avg"] or -1) > 0 and x["valid"]["t"] is not None]
    ok.sort(key=lambda x: -x["valid"]["t"])
    return rows, ok


# ---------------------------------------------------------------- family R


def r_weights(Z, period, Hh, tmin):
    a, b = Z.span(period)
    rows = np.arange(a, b - Hh + 1, Hh)
    K = Z.bits.shape[2]
    diffs = np.full((len(rows), K), np.nan)
    for n_, t in enumerate(rows):
        with np.errstate(invalid="ignore", divide="ignore"):
            y = (Z.C[t + Hh] / Z.C[t] - 1) - (Z.spy[t + Hh] / Z.spy[t] - 1)
        ok = Z.gate[t] & np.isfinite(y)
        B = np.asarray(Z.bits[t][ok])
        yy = y[ok].astype("float64") * 100
        P, F = (B == 1).astype("float64"), (B == 0).astype("float64")
        npass, nfail = P.sum(0), F.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = P.T @ yy / npass - F.T @ yy / nfail
        d[(npass < 5) | (nfail < 5)] = np.nan
        diffs[n_] = d
    cnt = np.isfinite(diffs).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.nanmean(diffs, 0) / (np.nanstd(diffs, 0, ddof=1) / np.sqrt(cnt))
    t[(cnt < 20) | ~np.isfinite(t)] = 0.0
    for j, cat in enumerate(checks.CAT_OF):
        if cat == "MK":
            t[j] = 0.0
    return np.where(np.abs(t) >= tmin, np.clip(t, -4, 4), 0.0)


def r_sim(Z, w, period, Hh, Nn, max_sector=2):
    a, b = Z.span(period)
    cash, equity = EQUITY, EQUITY
    pos = {}                                     # j -> [shares, entry, stop, day bought]
    buys, sells, curve, trades = [], [], [], []
    for t in range(a, b + 1):
        for j in sells:
            if j in pos:
                sh, entry = pos[j][0], pos[j][1]
                px_ = Z.O[t, j] if np.isfinite(Z.O[t, j]) else Z.C[t - 1, j]
                cash += sh * px_ - fee(sh)
                trades.append(sh * (px_ - entry) - 2 * fee(sh))
                del pos[j]
        for j in buys:
            px_ = Z.O[t, j]
            if j in pos or not np.isfinite(px_) or not np.isfinite(Z.ATR[t - 1, j]):
                continue
            sh = int(np.floor(min(equity / Nn, max(cash - FEE_MIN, 0)) / px_))
            if sh >= 1:
                cash -= sh * px_ + fee(sh)
                pos[j] = [sh, px_, px_ - 3 * Z.ATR[t - 1, j], t]
        buys, sells = [], []
        for j in list(pos):
            sh, entry, stop, t_in = pos[j]
            ex = None
            if t > t_in and np.isfinite(Z.O[t, j]) and Z.O[t, j] <= stop:
                ex = Z.O[t, j]
            elif np.isfinite(Z.L[t, j]) and Z.L[t, j] <= stop:
                ex = stop
            if ex is not None:
                cash += sh * ex - fee(sh)
                trades.append(sh * (ex - entry) - 2 * fee(sh))
                del pos[j]
        equity = cash + sum(p[0] * (Z.C[t, j] if np.isfinite(Z.C[t, j]) else p[1]) for j, p in pos.items())
        curve.append(equity)
        if (t - a) % Hh != 0 or t == b:
            continue
        sc = score(np.asarray(Z.bits[t]), w)
        elig = Z.gate[t] & np.isfinite(Z.C[t])
        order = [j for j in np.argsort(-sc) if elig[j]]
        rank = {j: r for r, j in enumerate(order)}
        sells = [j for j in pos if rank.get(j, 10**9) >= 3 * Nn]
        if np.isfinite(Z.regime[t]) and Z.regime[t] >= 1:
            keepers = [j for j in pos if j not in sells]
            secs = {}
            for j in keepers:
                secs[Z.sectors[j]] = secs.get(Z.sectors[j], 0) + 1
            need = Nn - len(keepers)
            for j in order:
                if need <= 0:
                    break
                if j in pos:
                    continue
                s = Z.sectors[j]
                if s and secs.get(s, 0) >= max_sector:
                    continue
                buys.append(j)
                secs[s] = secs.get(s, 0) + 1
                need -= 1
    return port_stats(Z, np.array(curve), a, b, trades)


def select_r(Z):
    rows = []
    for Hh in (10, 20):
        for tmin in (1.5, 2.0):
            w = r_weights(Z, TRAIN, Hh, tmin)
            for Nn in (3, 5):
                rows.append({"H": Hh, "tmin": tmin, "N": Nn, "n_weights": int(np.count_nonzero(w)),
                             "train": r_sim(Z, w, TRAIN, Hh, Nn), "valid": r_sim(Z, w, VALID, Hh, Nn)})
    ok = [x for x in rows if x["valid"]["maxdd"] >= -20 and x["train"]["ret"] > x["train"]["spy"]]
    ok.sort(key=lambda x: -(x["valid"]["ret"] - x["valid"]["spy"]))
    return rows, ok


# ---------------------------------------------------------------- commands


def cmd_select(d):
    Z = Cache(d)
    rows_s, ok_s = select_s(Z)
    rows_r, ok_r = select_r(Z)
    out = {"S": {"all": rows_s, "eligible": ok_s[:15]}, "R": {"all": rows_r, "eligible": ok_r}}
    json.dump(out, open(os.path.join(d, "select.json"), "w"), ensure_ascii=False, indent=1)
    print("FAMILY S: unfiltered baselines on train / validation")
    for x in rows_s:
        if x["tmin"] is None and x["exit"] in ("V1", "E1", "E2"):
            print(f"  {x['group']:4} {x['exit']:3} train {x['train']}  valid {x['valid']}")
    print("FAMILY S: best eligible by validation t")
    for x in ok_s[:12]:
        print(f"  {x['group']:4} {x['exit']:3} tmin {x['tmin']} keep {x['keep']}  train {x['train']}  valid {x['valid']}")
    print("FAMILY R")
    for x in rows_r:
        print(f"  H{x['H']} tmin {x['tmin']} N{x['N']} w={x['n_weights']}  train {x['train']['ret']} vs spy {x['train']['spy']} ew {x['train']['ew']} dd {x['train']['maxdd']}"
              f" | valid {x['valid']['ret']} vs spy {x['valid']['spy']} ew {x['valid']['ew']} dd {x['valid']['maxdd']} trades {x['valid']['trades']}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prepare":
        prepare(sys.argv[2], sys.argv[3])
    elif cmd == "select":
        cmd_select(sys.argv[2])
