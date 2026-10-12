"""One pass over the universe: indicators -> 300 checks -> setups -> today's table and history evidence."""
import datetime as dt
import json
import os

import numpy as np
import pandas as pd

from . import backtest as bt
from . import checks, setups
from .data import ROOT, align
from .indicators import SECTOR_ETF, build, build_market

ENGINE_VERSION = "1.0"
REGIME_HE = {2: ("green", "ירוק"), 1: ("yellow", "צהוב"), 0: ("red", "אדום")}


def load_config():
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as f:
        return json.load(f)


def _r(x, n=2):
    if x is None:
        return None
    x = float(x)
    return round(x, n) if np.isfinite(x) else None


def _pct(x, n=1):
    return _r(x * 100, n) if x is not None and np.isfinite(x) else None


def run(px, uni, cfg, do_backtest=True, ext_fn=None, log=print):
    px, idx = align(px)
    for need in ["SPY", "QQQ", "IWM", "^VIX"]:
        if need not in px:
            raise RuntimeError(f"missing index data: {need}")
    tickers = [t for t in uni if t in px and t not in SECTOR_ETF.values()]
    sectors_of = {t: uni[t]["sector"] for t in tickers}
    closes = pd.DataFrame({t: px[t]["close"] for t in tickers}, index=idx)
    M, S, X = build_market(px, closes, sectors_of)
    T, N = len(idx), len(tickers)
    dates = [d.strftime("%Y-%m-%d") for d in idx]
    asof = idx[-1]
    start = min(260, T - 30)

    f32 = lambda: np.full((T, N), np.nan, dtype="float32")
    B = {"O": f32(), "H": f32(), "L": f32(), "C": f32(), "ATR": f32(), "SC": f32(),
         "SIG": {k: np.zeros((T, N), dtype=bool) for k in setups.SETUPS},
         "STP": {k: f32() for k in setups.SETUPS},
         "regime": M.regime, "dates": dates, "years": idx.year.to_numpy(), "spy": M.spy_c}
    lift = bt.Lift(len(checks.C), T, start)
    cat_idx = {k: [i for i, c in enumerate(checks.CAT_OF) if c == k] for k, _ in checks.CATEGORIES}
    rows = {}
    for j, t in enumerate(tickers):
        df = px[t]
        d = build(df, M, S.get(sectors_of[t], S[None]), {n: X[n][t] for n in X}, cfg)
        mat = checks.run_all(d)
        sc = checks.score(mat)
        su, watch = setups.compute(d)
        B["O"][:, j], B["H"][:, j], B["L"][:, j], B["C"][:, j] = d.o, d.h, d.l, d.c
        B["ATR"][:, j], B["SC"][:, j] = d.atr14, sc
        for k, (sig, stp) in su.items():
            B["SIG"][k][:, j] = sig
            B["STP"][k][:, j] = stp
        if do_backtest:
            lift.add(mat, d.c, M.spy_c, d.nbars)
        if not np.isfinite(d.c[-1]):
            continue
        g = bool(setups.gates(d)[-1])
        today = [k for k in setups.PRIORITY if su[k][0][-1]]
        rows[t] = {
            "j": j, "gate": g, "bits": mat[-1].copy(), "score": float(sc[-1]) if np.isfinite(sc[-1]) else None,
            "setups": today, "watch": [k for k, v in watch.items() if v[-1]],
            "stops": {k: float(su[k][1][-1]) for k in today},
            "c": float(d.c[-1]), "pc": float(d.pc[-1]) if np.isfinite(d.pc[-1]) else None, "atr": float(d.atr14[-1]),
            "stop_g": float(d.stop_g[-1]) if np.isfinite(d.stop_g[-1]) else None,
            "m": {"rvol": _r(d.rvol[-1]), "atrp": _pct(d.atrp[-1]), "rsi14": _r(d.rsi14[-1], 0), "rk63": _r(d.rk63[-1], 0),
                  "rkc": _r(d.rkc[-1], 0), "dist52h": _pct(d.dist52h[-1]), "dv20m": _r(d.dv20[-1] / 1e6, 0),
                  "ret5": _pct(d.ret5[-1]), "ret21": _pct(d.ret21[-1]), "ret63": _pct(d.ret63[-1]), "ex21": _pct(d.ex21[-1]),
                  "vs50": _pct(d.c[-1] / d.sma50[-1] - 1), "vs200": _pct(d.c[-1] / d.sma200[-1] - 1),
                  "hi55": _r(d.hh55[-1]), "lo10": _r(d.ll10[-1]), "sma20": _r(d.sma20[-1]), "sma50": _r(d.sma50[-1])},
        }
        if (j + 1) % 100 == 0:
            log(f"  {j + 1}/{N} tickers")
    with np.errstate(invalid="ignore"):
        B["ew"] = np.nancumprod(1 + np.nan_to_num(np.nanmean(B["C"][1:] / B["C"][:-1] - 1, axis=1), nan=0.0))
        B["ew"] = np.concatenate([[1.0], B["ew"]])

    # ---- history first: the table needs to know which setups their own history suspends
    backtest, suspended = None, {}
    if do_backtest:
        log("backtest: per-setup")
        per = bt.per_setup(B, cfg, start)
        log("backtest: portfolio")
        secs = [sectors_of[t] for t in tickers]
        summ, curve, trades = bt.portfolio(B, cfg, start, tickers, secs)
        scen = {}
        for f in cfg.get("commission_scenarios_usd", []):
            if f == cfg["commission_per_trade_usd"]:
                sm = summ
            else:
                sm, _, _ = bt.portfolio(B, dict(cfg, commission_per_trade_usd=f), start, tickers, secs)
            scen[str(f)] = {k: sm[k] for k in ("return_pct", "max_dd_pct", "halted_on")} | {"trades": sm["trades"]}
        backtest = {
            "asof": dates[-1], "from": dates[start], "to": dates[-1], "engine_version": ENGINE_VERSION,
            "commission_usd": cfg["commission_per_trade_usd"], "fee_scenarios": cfg.get("commission_scenarios_usd", []),
            "per_setup": per, "portfolio": summ, "portfolio_by_fee": scen, "curve": curve, "last_trades": trades[-40:],
            "n_trades": len(trades), "check_lift": lift.result(checks.IDS),
            "caveats": [
                "היקום הוא חברי המדדים של היום, כולל חברות שנכנסו למדד כי עלו. זה מנפח את התוצאות (הטיית שרידות).",
                "דיבידנדים לא נספרים, לא במניות ולא ב-SPY.",
                "ההנחה היא ביצוע במחיר הפתיחה ובמחיר הסטופ או היעד בדיוק. בפועל יש החלקה.",
                "ביום שבו נגעו גם בסטופ וגם ביעד, נספר סטופ.",
                "הכללים לא כוילו על הנתונים האלה, אבל גם לא נבדקו על נתונים שלא ראינו. עבר אינו מבטיח עתיד.",
            ],
        }
    else:
        try:
            with open(os.path.join(ROOT, "out", "backtest.json"), encoding="utf-8") as f:
                per = json.load(f).get("per_setup", {})
        except (OSError, ValueError):
            per = {}
    for k, v in per.items():
        a = v.get("all", {})
        if a.get("n", 0) >= 50 and a.get("avg_r") is not None and a["avg_r"] < 0 and (a.get("t_stat") or 0) <= -2:
            suspended[k] = f"תוחלת היסטורית {a['avg_r']}R על {a['n']} איתותים (t={a['t_stat']})"

    # ---- today's table
    def tier(setups_, watch_):
        if any(k not in suspended for k in setups_):
            return 0          # a setup that may be traded
        if setups_:
            return 1          # a setup, but its own history suspends it
        return 2 if watch_ else 3

    for r in rows.values():   # a tradable setup comes before a suspended one
        r["setups"] = [k for k in r["setups"] if k not in suspended] + [k for k in r["setups"] if k in suspended]
    order = sorted((t for t in rows if rows[t]["gate"] and rows[t]["score"] is not None),
                   key=lambda t: (tier(rows[t]["setups"], rows[t]["watch"]), -rows[t]["score"]))
    table_t = order[: cfg["table_size"]]
    ext = ext_fn(table_t, asof) if ext_fn else {}
    reg = int(M.regime[-1]) if np.isfinite(M.regime[-1]) else 1
    mult = {2: 1.0, 1: cfg["yellow_size_mult"], 0: 0.0}[reg]
    i_earn, i_mcap = checks.IDS.index("RF21"), checks.IDS.index("RF22")
    table = []
    for rank, t in enumerate(table_t, 1):
        r = rows[t]
        bits = r["bits"]
        e = ext.get(t, {})
        if e.get("earn_days") is not None:
            bits[i_earn] = 1.0 if e["earn_days"] > 5 else 0.0
        if e.get("mcap") is not None:
            bits[i_mcap] = 1.0 if e["mcap"] > 2e9 else 0.0
        known = ~np.isnan(bits)
        score = float(np.nansum(bits) / known.sum() * 100)
        cats = {k: int(round(np.nansum(bits[ix]) / max((~np.isnan(bits[ix])).sum(), 1) * 100)) for k, ix in cat_idx.items()}
        plan = None
        setup = r["setups"][0] if r["setups"] else None
        stop = r["stops"][setup] if setup else r["stop_g"]
        if stop is not None and stop < r["c"]:
            rr, tmax = (setups.SETUPS[setup][1], setups.SETUPS[setup][2]) if setup else (2.0, 15)
            sh = bt.size(r["c"], stop, cfg["equity_usd"], cfg, mult if mult > 0 else 1.0)
            plan = {"setup": setup, "ref_close": _r(r["c"]), "stop": _r(stop), "target": _r(r["c"] + rr * (r["c"] - stop)),
                    "rr": rr, "time_stop_days": tmax, "risk_pct": _pct((r["c"] - stop) / r["c"]),
                    "max_entry": _r(r["c"] + r["atr"]), "shares": sh, "position_usd": _r(sh * r["c"], 0),
                    "risk_usd": _r(sh * (r["c"] - stop), 0), "provisional": setup is None}
        m = dict(r["m"])
        m["earn_date"], m["earn_days"] = e.get("earn_date"), e.get("earn_days")
        m["mcap_b"] = _r(e["mcap"] / 1e9, 1) if e.get("mcap") else None
        table.append({
            "rank": rank, "t": t, "name": uni[t]["name"], "sector": uni[t]["sector"], "price": _r(r["c"]),
            "chg": _pct(r["c"] / r["pc"] - 1, 2) if r["pc"] else None, "score": round(score, 1), "cats": cats,
            "setups": r["setups"], "watch": r["watch"], "plan": plan, "m": m,
            "suspended": [k for k in r["setups"] if k in suspended],
            "passed": int(np.nansum(bits)), "failed": int((bits == 0).sum()), "unknown": int((~known).sum()),
            "red_flags": [checks.IDS[i] for i in cat_idx["RF"] if bits[i] == 0],
            "bits": "".join("-" if np.isnan(b) else ("1" if b else "0") for b in bits),
        })
    table.sort(key=lambda x: (tier(x["setups"], x["watch"]), -x["score"]))
    for i, x in enumerate(table, 1):
        x["rank"] = i

    gated = [t for t in rows if rows[t]["gate"] and rows[t]["score"] is not None]
    scores = np.array([rows[t]["score"] for t in gated]) if gated else np.array([np.nan])
    mk_bits = None
    if rows:
        any_bits = next(iter(rows.values()))["bits"]
        mk_bits = [any_bits[i] for i in cat_idx["MK"]]
    sector_rows = []
    for sec, etf in SECTOR_ETF.items():
        s = S.get(sec)
        if s is None:
            continue
        sector_rows.append({"sector": sec, "etf": etf, "ret21": _pct(s.sec_ret21[-1]), "ret63": _pct(s.sec_ret63[-1]),
                            "rank21": _r(s.sec_rank21[-1], 0), "above50": bool(s.sec_c[-1] > s.sec_sma50[-1]),
                            "breadth50": _r(s.sec_breadth50[-1], 0)})
    sector_rows.sort(key=lambda x: x["rank21"] if x["rank21"] is not None else 99)
    latest = {
        "asof": dates[-1], "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "engine_version": ENGINE_VERSION,
        "regime": {
            "label": REGIME_HE[reg][0], "label_he": REGIME_HE[reg][1], "size_mult": mult,
            "spy": _r(M.spy_c[-1]), "spy_chg": _pct(M.spy_ret1[-1], 2), "spy_vs_sma50": _pct(M.spy_c[-1] / M.spy_sma50[-1] - 1),
            "spy_vs_sma200": _pct(M.spy_c[-1] / M.spy_sma200[-1] - 1), "spy_from_high": _pct(M.spy_dist_high[-1]),
            "vix": _r(M.vix[-1]), "breadth50": _r(M.br_above50[-1], 0), "breadth200": _r(M.br_above200[-1], 0),
            "nh_nl": _r(M.br_nhnl[-1], 0),
            "market_checks_passed": int(np.nansum(mk_bits)) if mk_bits is not None else None, "market_checks_total": 25,
            "rule": "ירוק: SPY מעל ממוצע 50 ו-200, רוחב 50%+, VIX מתחת ל-25. אדום: SPY מתחת לממוצע 200, או VIX מעל 30, או SPY מתחת לממוצע 50 עם רוחב מתחת ל-35%. אחרת צהוב.",
        },
        "universe": {"requested": len(uni), "loaded": N, "with_today_bar": len(rows), "passed_gates": len(gated),
                     "score_median": _r(np.nanmedian(scores), 1), "score_p90": _r(np.nanpercentile(scores, 90), 1),
                     "score_max": _r(np.nanmax(scores), 1)},
        "suspended_setups": suspended,
        "signals_today": {k: int(sum(1 for t in gated if k in rows[t]["setups"])) for k in setups.PRIORITY},
        "watch_today": {k: int(sum(1 for t in gated if k in rows[t]["watch"])) for k in setups.WATCH},
        "sectors": sector_rows,
        "table": table,
    }
    # last 12 bars for every ticker: the paper portfolio fills and exits from these
    keep = 12
    bars = {}
    for t in list(rows) + ["SPY"]:
        df = px[t].iloc[-keep:]
        bars[t] = [[_r(a), _r(b), _r(c_), _r(e_)] if np.isfinite(e_) else None
                   for a, b, c_, e_ in zip(df["open"], df["high"], df["low"], df["close"])]
    prices = {"asof": dates[-1], "dates": dates[-keep:], "bars": bars,
              "atr": {t: _r(rows[t]["atr"]) for t in rows}, "sector": {t: uni[t]["sector"] for t in rows}}

    out = {"latest": latest, "prices": prices, "catalog": {**checks.catalog(), "setups": [
        {"id": k, "name": v[0], "target_r": v[1], "time_stop_days": v[2], "rules": setups.RULES_HE[k]}
        for k, v in setups.SETUPS.items()], "watch": setups.WATCH, "engine_version": ENGINE_VERSION, "config": cfg}}

    if backtest:
        out["backtest"] = backtest
    return out


def write(out, outdir=None):
    outdir = outdir or os.path.join(ROOT, "out")
    os.makedirs(os.path.join(outdir, "history"), exist_ok=True)

    def dump(name, obj):
        with open(os.path.join(outdir, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))

    dump("latest.json", out["latest"])
    dump("prices.json", out["prices"])
    dump("catalog.json", out["catalog"])
    dump(os.path.join("history", out["latest"]["asof"] + ".json"), out["latest"])
    if "backtest" in out:
        dump("backtest.json", out["backtest"])
