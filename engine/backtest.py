"""Honest-as-possible evidence from history.

Three things are measured, all with entry at the next day's open after a signal:
1. per-setup statistics (every signal on its own, in R after commissions)
2. a portfolio simulation that follows the written risk rules
3. for each of the 300 checks: what happened in the 10 days after it passed vs failed

Known limits, repeated in the output: the universe is today's index members
(survivorship bias), dividends are ignored, fills are assumed at the open and at the
stop/target price, and nothing here is a promise about the future.
"""
import numpy as np

from .setups import PRIORITY, SETUPS


def sim_trade(o, h, l, c, i, stop, rr, tmax, atr_i):
    """Signal at close of day i, entry at open of i+1.
    Returns (entry_idx, entry, exit_idx, exit, reason) or (None, reason)."""
    n = len(c)
    e = i + 1
    if e >= n:
        return None, "no_next_day"
    entry = o[e]
    if not np.isfinite(entry) or not np.isfinite(stop):
        return None, "no_data"
    if entry <= stop:
        return None, "opened_below_stop"
    if entry > c[i] + atr_i:
        return None, "gapped_away"
    target = entry + rr * (entry - stop)
    last = min(e + tmax, n) - 1
    for j in range(e, last + 1):
        if not np.isfinite(c[j]):
            continue
        if j > e and o[j] <= stop:
            return (e, entry, j, o[j], "stop_gap"), None
        if l[j] <= stop:
            return (e, entry, j, stop, "stop"), None
        if j > e and o[j] >= target:
            return (e, entry, j, o[j], "target_gap"), None
        if h[j] >= target:
            return (e, entry, j, target, "target"), None
    if last < e + tmax - 1:
        return None, "still_open"
    return (e, entry, last, c[last], "time"), None


def size(entry, stop, equity, cfg, mult=1.0, cash=None):
    risk_usd = equity * cfg["risk_per_trade_pct"] / 100 * mult
    cap = equity * cfg["max_position_pct"] / 100
    lim = [risk_usd / (entry - stop), cap / entry]
    if cash is not None:
        lim.append(max(cash - cfg["commission_per_trade_usd"], 0) / entry)
    return int(np.floor(min(lim)))


def _stats(rs, pnl=None):
    rs = np.asarray(rs, dtype="float64")
    if len(rs) == 0:
        return {"n": 0}
    wins, losses = rs[rs > 0], rs[rs <= 0]
    pf = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else None
    return {
        "n": int(len(rs)),
        "win_rate": round(float((rs > 0).mean() * 100), 1),
        "avg_r": round(float(rs.mean()), 3),
        "median_r": round(float(np.median(rs)), 3),
        "avg_win_r": round(float(wins.mean()), 2) if len(wins) else None,
        "avg_loss_r": round(float(losses.mean()), 2) if len(losses) else None,
        "profit_factor": round(pf, 2) if pf is not None else None,
        "t_stat": round(float(rs.mean() / (rs.std(ddof=1) / np.sqrt(len(rs)))), 2) if len(rs) > 2 and rs.std() > 0 else None,
    }


def per_setup(B, cfg, start):
    """Every signal on its own. B holds the T x N matrices."""
    O, H, L, C, ATR, REG = B["O"], B["H"], B["L"], B["C"], B["ATR"], B["regime"]
    T, N = C.shape
    years = B["years"]
    out = {}
    eq = cfg["equity_usd"]
    fee = cfg["commission_per_trade_usd"]
    for k in PRIORITY:
        _, rr, tmax = SETUPS[k]
        sig, stp = B["SIG"][k], B["STP"][k]
        rows = []  # (year, regime, gross R, net R, days held, reason)
        skipped = {}
        for j in range(N):
            days = np.flatnonzero(sig[start:, j]) + start
            prev = -99
            for i in days:
                if i - prev <= 5:  # same idea, same week: count once
                    continue
                prev = i
                res, why = sim_trade(O[:, j], H[:, j], L[:, j], C[:, j], i, stp[i, j], rr, tmax, ATR[i, j])
                if res is None:
                    skipped[why] = skipped.get(why, 0) + 1
                    continue
                e, entry, x, exit_, reason = res
                sh = size(entry, stp[i, j], eq, cfg)
                if sh < 1:
                    skipped["too_small"] = skipped.get("too_small", 0) + 1
                    continue
                gross = (exit_ - entry) / (entry - stp[i, j])
                net_usd = sh * (exit_ - entry) - 2 * fee
                planned = eq * cfg["risk_per_trade_pct"] / 100
                rows.append((years[i], REG[i], gross, net_usd / planned, x - e + 1, reason, B["SC"][i, j]))
        if not rows:
            out[k] = {"all": {"n": 0}, "skipped": skipped}
            continue
        yr = np.array([r[0] for r in rows])
        rg = np.array([r[1] for r in rows])
        g = np.array([r[2] for r in rows])
        nr = np.array([r[3] for r in rows])
        res = {"all": _stats(nr), "gross_avg_r": round(float(g.mean()), 3),
               "avg_days": round(float(np.mean([r[4] for r in rows])), 1),
               "exits": {x: int(sum(1 for r in rows if r[5].startswith(x))) for x in ("stop", "target", "time")},
               "by_regime": {name: _stats(nr[rg == v]) for name, v in (("green", 2), ("yellow", 1), ("red", 0))},
               "by_year": {str(int(y)): _stats(nr[yr == y]) for y in sorted(set(yr))},
               "skipped": skipped}
        sc = np.array([r[6] for r in rows], dtype="float64")
        if np.isfinite(sc).sum() >= 40:
            med = float(np.nanmedian(sc))
            res["by_score"] = {"median_score": round(med, 1), "above": _stats(nr[sc >= med]), "below": _stats(nr[sc < med])}
        out[k] = res
    return out


def portfolio(B, cfg, start, tickers, sectors):
    """Follows the rulebook day by day. Returns summary, weekly equity curve and trades."""
    O, H, L, C, ATR, REG, SC = B["O"], B["H"], B["L"], B["C"], B["ATR"], B["regime"], B["SC"]
    T, N = C.shape
    fee = cfg["commission_per_trade_usd"]
    cash = float(cfg["equity_usd"])
    equity = cash
    peak = equity
    pos = []       # dicts
    pending = []   # orders decided at yesterday's close
    trades = []
    curve = []
    pause_until = -1
    exposure_days = 0
    mult_of = {2: 1.0, 1: cfg["yellow_size_mult"], 0: 0.0}
    for t in range(start, T):
        # 1. fill yesterday's orders at today's open
        for od in pending:
            j = od["j"]
            entry = O[t, j]
            if not np.isfinite(entry) or entry <= od["stop"] or entry > od["ref"] + od["atr"]:
                continue
            sh = size(entry, od["stop"], equity, cfg, od["mult"], cash)
            if sh < 1:
                continue
            cash -= sh * entry + fee
            _, rr, tmax = SETUPS[od["setup"]]
            pos.append({"j": j, "sh": sh, "entry": entry, "stop": od["stop"], "target": entry + rr * (entry - od["stop"]),
                        "t0": t, "tmax": tmax, "setup": od["setup"], "risk_usd": sh * (entry - od["stop"])})
        pending = []
        # 2. exits on today's bar
        keep = []
        for p in pos:
            j = p["j"]
            o_, h_, l_, c_ = O[t, j], H[t, j], L[t, j], C[t, j]
            exit_, why = None, None
            if np.isfinite(c_):
                if t > p["t0"] and o_ <= p["stop"]:
                    exit_, why = o_, "stop"
                elif l_ <= p["stop"]:
                    exit_, why = p["stop"], "stop"
                elif t > p["t0"] and o_ >= p["target"]:
                    exit_, why = o_, "target"
                elif h_ >= p["target"]:
                    exit_, why = p["target"], "target"
                elif t - p["t0"] + 1 >= p["tmax"]:
                    exit_, why = c_, "time"
            if exit_ is None:
                keep.append(p)
                continue
            cash += p["sh"] * exit_ - fee
            pnl = p["sh"] * (exit_ - p["entry"]) - 2 * fee
            trades.append({"t": tickers[j], "setup": p["setup"], "in": int(p["t0"]), "out": int(t), "entry": float(p["entry"]),
                           "exit": float(exit_), "why": why, "pnl": float(pnl), "r": float(pnl / max(p["risk_usd"], 1e-9))})
        pos = keep
        # 3. mark to market
        held = 0.0
        for p in pos:
            px_ = C[t, p["j"]]
            if not np.isfinite(px_):
                px_ = p["entry"]
            held += p["sh"] * px_
        equity = cash + held
        peak = max(peak, equity)
        dd = equity / peak - 1
        if pos:
            exposure_days += 1
        curve.append(equity)
        # 4. new orders from today's close
        if dd <= -cfg["dd_pause_pct"] / 100 and t > pause_until:
            pause_until = t + cfg["dd_pause_days"]
        reg = REG[t]
        if not np.isfinite(reg) or reg < 1 or t <= pause_until:
            continue
        slots = min(cfg["max_open_positions"] - len(pos), cfg["max_new_per_day"])
        if slots <= 0:
            continue
        cands = []
        for k in PRIORITY:
            for j in np.flatnonzero(B["SIG"][k][t]):
                cands.append((SC[t, j], j, k))
        cands.sort(key=lambda x: -x[0] if np.isfinite(x[0]) else 0)
        heldj = {p["j"] for p in pos}
        sec_count = {}
        for p in pos:
            sec_count[sectors[p["j"]]] = sec_count.get(sectors[p["j"]], 0) + 1
        mult = mult_of[int(reg)] * (0.5 if dd <= -cfg["dd_half_size_pct"] / 100 else 1.0)
        open_risk = sum(max(p["sh"] * (C[t, p["j"]] - p["stop"]), 0) for p in pos if np.isfinite(C[t, p["j"]]))
        for sc, j, k in cands:
            if slots <= 0:
                break
            if j in heldj or any(od["j"] == j for od in pending):
                continue
            if not np.isfinite(sc) or sc < cfg["min_score"]:
                continue
            s = sectors[j]
            if s is not None and sec_count.get(s, 0) >= cfg["max_per_sector"]:
                continue
            new_risk = equity * cfg["risk_per_trade_pct"] / 100 * mult
            if open_risk + new_risk > equity * cfg["max_open_risk_pct"] / 100:
                break
            pending.append({"j": j, "setup": k, "stop": B["STP"][k][t, j], "ref": C[t, j], "atr": ATR[t, j], "mult": mult})
            sec_count[s] = sec_count.get(s, 0) + 1
            open_risk += new_risk
            slots -= 1
    curve = np.array(curve)
    rs = [x["r"] for x in trades]
    n_days = len(curve)
    spy = B["spy"][start:]
    ew = B["ew"][start:]
    run_peak = np.maximum.accumulate(curve)
    summ = {
        "start_equity": cfg["equity_usd"],
        "end_equity": round(float(curve[-1]), 0),
        "return_pct": round(float(curve[-1] / cfg["equity_usd"] - 1) * 100, 1),
        "cagr_pct": round(float((curve[-1] / cfg["equity_usd"]) ** (252 / n_days) - 1) * 100, 1),
        "max_dd_pct": round(float((curve / run_peak - 1).min()) * 100, 1),
        "spy_return_pct": round(float(spy[-1] / spy[0] - 1) * 100, 1),
        "spy_max_dd_pct": round(float((spy / np.maximum.accumulate(spy) - 1).min()) * 100, 1),
        "universe_ew_return_pct": round(float(ew[-1] / ew[0] - 1) * 100, 1),
        "days_in_market_pct": round(exposure_days / n_days * 100, 0),
        "trades": _stats(rs),
        "by_setup": {k: _stats([x["r"] for x in trades if x["setup"] == k]) for k in PRIORITY},
        "commission_total": round(len(trades) * 2 * fee, 0),
        "years": round(n_days / 252, 1),
    }
    step = 5
    pts = [{"d": B["dates"][start + i], "eq": round(float(curve[i]), 0),
            "spy": round(float(spy[i] / spy[0] * cfg["equity_usd"]), 0),
            "ew": round(float(ew[i] / ew[0] * cfg["equity_usd"]), 0)} for i in list(range(0, n_days, step)) + [n_days - 1]]
    for x in trades:
        x["in"], x["out"] = B["dates"][x["in"]], B["dates"][x["out"]]
        for f in ("entry", "exit", "pnl", "r"):
            x[f] = round(x[f], 2)
    return summ, pts, trades


class Lift:
    """For each check: what happened in the 10 trading days after it passed vs failed.

    Stocks on the same day move together, so the comparison is made inside each day
    (passers minus failers, return in excess of SPY) and then averaged over days that do
    not overlap. Checks that are the same for every stock on a day (market regime,
    breadth) are compared across days instead, on the universe's plain forward return.
    """

    def __init__(self, n_checks, T, start, horizon=10):
        self.h = horizon
        self.rows = np.arange(start, T - horizon, horizon)
        shape = (len(self.rows), n_checks)
        self.n_p, self.s_p, self.a_p = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        self.n_f, self.s_f, self.a_f = np.zeros(shape), np.zeros(shape), np.zeros(shape)

    def add(self, mat, c, spy, nbars):
        rows = self.rows
        if len(rows) == 0:
            return
        with np.errstate(invalid="ignore", divide="ignore"):
            y_abs = (c[rows + self.h] / c[rows] - 1) * 100
            y = y_abs - (spy[rows + self.h] / spy[rows] - 1) * 100
        ok = np.isfinite(y) & (nbars[rows] >= 252)
        m = mat[rows].astype("float64")
        m[~ok] = np.nan
        p, f = np.nan_to_num(m, nan=0.0), np.nan_to_num(1 - m, nan=0.0)
        y, y_abs = np.where(ok, y, 0.0)[:, None], np.where(ok, y_abs, 0.0)[:, None]
        self.n_p += p
        self.s_p += p * y
        self.a_p += p * y_abs
        self.n_f += f
        self.s_f += f * y
        self.a_f += f * y_abs

    def result(self, ids):
        out = []
        for j, cid in enumerate(ids):
            n_p, n_f = self.n_p[:, j], self.n_f[:, j]
            tot_p, tot_f = n_p.sum(), n_f.sum()
            rec = {"id": cid, "pass_rate": round(float(tot_p / (tot_p + tot_f) * 100), 0) if tot_p + tot_f else None,
                   "edge": None, "t": None, "days": 0, "mode": None}
            both = (n_p >= 5) & (n_f >= 5)
            if both.sum() >= 30:
                diff = self.s_p[both, j] / n_p[both] - self.s_f[both, j] / n_f[both]
                sd = diff.std(ddof=1)
                rec.update(mode="stocks", days=int(both.sum()), edge=round(float(diff.mean()), 2),
                           t=round(float(diff.mean() / (sd / np.sqrt(len(diff)))), 1) if sd > 0 else None)
            else:
                n = n_p + n_f
                have = n > 0
                avg = np.where(have, (self.a_p[:, j] + self.a_f[:, j]) / np.where(have, n, 1), np.nan)
                dp, df = avg[have & (n_p > n_f)], avg[have & (n_p <= n_f)]
                if len(dp) >= 12 and len(df) >= 12:
                    se = np.sqrt(dp.var(ddof=1) / len(dp) + df.var(ddof=1) / len(df))
                    rec.update(mode="market", days=int(len(dp) + len(df)), edge=round(float(dp.mean() - df.mean()), 2),
                               t=round(float((dp.mean() - df.mean()) / se), 1) if se > 0 else None)
            out.append(rec)
        return out
