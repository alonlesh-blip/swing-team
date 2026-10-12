"""Paper portfolio bookkeeping. Pure standard library, so it runs anywhere.

    python engine/paper.py --state state.json --prices out/prices.json --config config.json \
        [--decisions decisions.json] --out state.json

The boss decides, this file does the arithmetic and enforces the hard limits of the
rulebook. Orders decided after the close of day D are filled at the open of the next
trading day. Exits: stop first, then target, then time stop at the close.
"""
import argparse
import json
import math
import sys

ALLOWED = {"BRK": (2.0, 15), "PB": (2.0, 15), "TGT": (2.0, 15), "GAP": (2.0, 15), "MR": (1.5, 7)}


def new_state(cfg):
    return {"version": 1, "start_date": None, "start_equity": cfg["equity_usd"], "cash": float(cfg["equity_usd"]),
            "last_date": None, "spy_start": None, "positions": [], "pending": [], "closed": [], "cancelled": [],
            "rejected": [], "curve": [], "paused": False, "log": []}


def _size(entry, stop, equity, cash, cfg, mult):
    risk_usd = equity * cfg["risk_per_trade_pct"] / 100 * mult
    lim = [risk_usd / (entry - stop), equity * cfg["max_position_pct"] / 100 / entry,
           max(cash - cfg["commission_per_trade_usd"], 0) / entry]
    return int(math.floor(min(lim)))


def _equity(st, closes):
    return st["cash"] + sum(p["shares"] * closes.get(p["t"], p["last"]) for p in st["positions"])


def _drawdown(st):
    if not st["curve"]:
        return 0.0
    peak = max(max(x["eq"] for x in st["curve"]), st["start_equity"])
    return st["curve"][-1]["eq"] / peak - 1


def step_day(st, d, i, prices, cfg):
    """Process one trading day's bars."""
    fee = cfg["commission_per_trade_usd"]
    bars = prices["bars"]
    prev_eq = st["curve"][-1]["eq"] if st["curve"] else st["start_equity"]

    def bar(t):
        b = bars.get(t)
        return b[i] if b and i < len(b) else None

    # 1. fill orders decided before today
    for od in st["pending"]:
        b = bar(od["t"])
        why = None
        if b is None:
            why = "אין נתוני מסחר"
        elif b[0] <= od["stop"]:
            why = "נפתחה מתחת לסטופ"
        elif b[0] > od["max_entry"]:
            why = "נפתחה רחוק מדי מעל מחיר הייחוס"
        if why is None:
            sh = _size(b[0], od["stop"], prev_eq, st["cash"], cfg, od.get("size_mult", 1.0))
            if sh < 1:
                why = "הגודל המחושב קטן ממניה אחת"
        if why:
            st["cancelled"].append({"t": od["t"], "setup": od["setup"], "decided": od["decided"], "date": d, "why": why})
            continue
        entry = b[0]
        st["cash"] -= sh * entry + fee
        st["positions"].append({
            "t": od["t"], "setup": od["setup"], "shares": sh, "entry": entry, "entry_date": d, "stop": od["stop"],
            "target": round(entry + od["rr"] * (entry - od["stop"]), 2), "rr": od["rr"], "tmax": od["tmax"], "days": 0,
            "risk_usd": round(sh * (entry - od["stop"]), 2), "last": entry, "sector": prices.get("sector", {}).get(od["t"]),
            "note": od.get("note", ""), "close_next_open": False})
    st["pending"] = []

    # 2. exits
    keep = []
    for p in st["positions"]:
        b = bar(p["t"])
        if b is None:
            keep.append(p)
            continue
        o, h, l, c = b
        p["days"] += 1
        first = p["entry_date"] == d
        exit_, why = None, None
        if p.get("close_next_open") and not first:
            exit_, why = o, "boss"
        elif not first and o <= p["stop"]:
            exit_, why = o, "stop"
        elif l <= p["stop"]:
            exit_, why = p["stop"], "stop"
        elif not first and o >= p["target"]:
            exit_, why = o, "target"
        elif h >= p["target"]:
            exit_, why = p["target"], "target"
        elif p["days"] >= p["tmax"]:
            exit_, why = c, "time"
        if exit_ is None:
            p["last"] = c
            keep.append(p)
            continue
        st["cash"] += p["shares"] * exit_ - fee
        pnl = p["shares"] * (exit_ - p["entry"]) - 2 * fee
        st["closed"].append({"t": p["t"], "setup": p["setup"], "shares": p["shares"], "entry": p["entry"],
                             "entry_date": p["entry_date"], "exit": round(exit_, 2), "exit_date": d, "why": why,
                             "days": p["days"], "pnl": round(pnl, 2), "r": round(pnl / max(p["risk_usd"], 0.01), 2),
                             "note": p.get("note", "")})
    st["positions"] = keep

    # 3. mark to market
    closes = {p["t"]: p["last"] for p in st["positions"]}
    eq = _equity(st, closes)
    spy = bar("SPY")
    spy_c = spy[3] if spy else None
    if st["spy_start"] is None and spy_c:
        st["spy_start"] = spy_c
    st["curve"].append({"d": d, "eq": round(eq, 2),
                        "spy": round(st["start_equity"] * spy_c / st["spy_start"], 2) if spy_c and st["spy_start"] else None})
    st["last_date"] = d
    if _drawdown(st) <= -cfg["dd_pause_pct"] / 100 and not st["paused"]:
        st["paused"] = True
        st["log"].append({"d": d, "msg": f"ירידה של {cfg['dd_pause_pct']}% מהשיא: אין כניסות חדשות עד בדיקה ידנית"})


def apply_decisions(st, dec, prices, cfg):
    """Queue the boss's orders for the next open, after enforcing the hard limits."""
    if dec.get("unpause"):
        st["paused"] = False
        st["log"].append({"d": prices["asof"], "msg": "ההשהיה בוטלה ידנית"})
    for cl in dec.get("close", []):
        for p in st["positions"]:
            if p["t"] == cl["t"]:
                p["close_next_open"] = True
                p["note"] = (p.get("note", "") + " | סגירה יזומה: " + cl.get("reason", "")).strip(" |")
    dd = _drawdown(st)
    eq = st["curve"][-1]["eq"] if st["curve"] else st["start_equity"]
    for n, od in enumerate(dec.get("orders", [])):
        t = od.get("t")
        why = None
        rr, tmax = ALLOWED.get(od.get("setup"), (None, None))
        held = {p["t"] for p in st["positions"]} | {p["t"] for p in st["pending"]}
        slots = cfg["max_open_positions"] - len(st["positions"]) - len(st["pending"])
        sector = prices.get("sector", {}).get(t)
        same_sector = sum(1 for p in st["positions"] + st["pending"] if sector and p.get("sector") == sector)
        open_risk = sum(max(p["shares"] * (p["last"] - p["stop"]), 0) for p in st["positions"])
        mult = float(od.get("size_mult", 1.0)) * (0.5 if dd <= -cfg["dd_half_size_pct"] / 100 else 1.0)
        if dec.get("date") != prices["asof"]:
            why = "ההחלטה מבוססת על טבלה מתאריך אחר מהמחירים"
        elif dec.get("regime") == "red":
            why = "משטר אדום: אין כניסות חדשות"
        elif st["paused"]:
            why = "התיק בהשהיה אחרי ירידה מהשיא"
        elif rr is None:
            why = "סטאפ לא מוכר"
        elif t not in prices["bars"] or prices["bars"][t][-1] is None:
            why = "אין מחירים למניה"
        elif t in held:
            why = "כבר בתיק או בהזמנה"
        elif slots <= 0:
            why = "התיק מלא"
        elif n >= cfg["max_new_per_day"]:
            why = "מעבר למספר הכניסות היומי המותר"
        elif same_sector >= cfg["max_per_sector"]:
            why = "מעבר למגבלת הסקטור"
        elif not (0 < od["stop"] < od["ref_close"] <= od["max_entry"]):
            why = "מחירי התוכנית לא עקביים"
        elif (od["ref_close"] - od["stop"]) / od["ref_close"] > 0.10:
            why = "הסטופ רחוק מ-10% מהמחיר"
        elif mult <= 0 or mult > 1:
            why = "מכפיל גודל לא חוקי"
        elif open_risk + eq * cfg["risk_per_trade_pct"] / 100 * mult > eq * cfg["max_open_risk_pct"] / 100 + 0.01:
            why = "מעבר לתקרת הסיכון הפתוח"
        if why:
            st["rejected"].append({"t": t, "date": prices["asof"], "why": why})
            continue
        st["pending"].append({"t": t, "setup": od["setup"], "stop": float(od["stop"]), "rr": rr, "tmax": tmax,
                              "ref_close": float(od["ref_close"]), "max_entry": float(od["max_entry"]),
                              "size_mult": mult, "decided": prices["asof"], "sector": sector, "note": od.get("note", "")})


def stats(st, cfg):
    rs = [x["r"] for x in st["closed"]]
    pnl = [x["pnl"] for x in st["closed"]]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    curve = st["curve"]
    eq = curve[-1]["eq"] if curve else st["start_equity"]
    peak, mdd = st["start_equity"], 0.0
    for x in curve:
        peak = max(peak, x["eq"])
        mdd = min(mdd, x["eq"] / peak - 1)
    spy_eq = curve[-1]["spy"] if curve and curve[-1]["spy"] else None
    n_days = max(len(curve) - 1, 0)
    weeks = n_days / 5
    g = cfg["go_live"]
    exp = sum(rs) / len(rs) if rs else None
    ret = eq / st["start_equity"] - 1
    spy_ret = spy_eq / st["start_equity"] - 1 if spy_eq else None
    crit = [
        {"name": f"לפחות {g['min_weeks']} שבועות על הנייר", "value": round(weeks, 1), "ok": weeks >= g["min_weeks"]},
        {"name": f"לפחות {g['min_closed_trades']} עסקאות סגורות", "value": len(rs), "ok": len(rs) >= g["min_closed_trades"]},
        {"name": f"תוחלת של {g['min_expectancy_r']}R לעסקה אחרי עמלות", "value": round(exp, 2) if exp is not None else None,
         "ok": exp is not None and exp >= g["min_expectancy_r"]},
        {"name": f"ירידה מקסימלית עד {g['max_drawdown_pct']}%", "value": round(mdd * 100, 1), "ok": mdd * 100 >= -g["max_drawdown_pct"]},
        {"name": "תשואה חיובית וגבוהה מ-SPY באותה תקופה", "value": round(ret * 100, 1),
         "ok": spy_ret is not None and ret > 0 and ret > spy_ret},
    ]
    st["stats"] = {
        "equity": round(eq, 2), "cash": round(st["cash"], 2), "return_pct": round(ret * 100, 2),
        "spy_return_pct": round(spy_ret * 100, 2) if spy_ret is not None else None, "max_dd_pct": round(mdd * 100, 2),
        "trading_days": n_days, "weeks": round(weeks, 1), "closed": len(rs), "open": len(st["positions"]),
        "pending": len(st["pending"]), "win_rate": round(len(wins) / len(rs) * 100, 1) if rs else None,
        "expectancy_r": round(exp, 3) if exp is not None else None,
        "avg_win_r": round(sum(wins) / len(wins), 2) if wins else None,
        "avg_loss_r": round(sum(losses) / len(losses), 2) if losses else None,
        "profit_factor": round(sum(wins) / -sum(losses), 2) if losses and sum(losses) < 0 else None,
        "pnl_total": round(sum(pnl), 2), "commissions": round((2 * len(rs) + len(st["positions"])) * cfg["commission_per_trade_usd"], 2),
        "open_risk_usd": round(sum(max(p["shares"] * (p["last"] - p["stop"]), 0) for p in st["positions"]), 2),
        "drawdown_now_pct": round(_drawdown(st) * 100, 2), "paused": st["paused"],
    }
    st["go_live"] = {"ready": all(c["ok"] for c in crit), "criteria": crit,
                     "next": f"אם כל התנאים מתקיימים: מתחילים ב-{g['first_stage_usd']} דולר ל-{g['first_stage_weeks']} שבועות, ורק אחר כך הסכום המלא. ההחלטה של אלון."}
    for k in ("cancelled", "rejected", "log"):
        st[k] = st[k][-60:]
    st["curve"] = st["curve"][-520:]
    return st


def run(st, prices, cfg, dec=None):
    dates = prices["dates"]
    if st["last_date"] is None:
        # first run: nothing to fill yet, just anchor the start
        i = len(dates) - 1
        st["start_date"] = dates[i]
        spy = prices["bars"]["SPY"][i]
        st["spy_start"] = spy[3]
        st["curve"].append({"d": dates[i], "eq": float(st["start_equity"]), "spy": float(st["start_equity"])})
        st["last_date"] = dates[i]
    else:
        new = [(i, d) for i, d in enumerate(dates) if d > st["last_date"]]
        if new and dates[0] > st["last_date"] and (st["positions"] or st["pending"]):
            st["log"].append({"d": dates[-1], "msg": "פער בנתונים: חסרים ימי מסחר בין הריצות, ייתכן שיציאות הוחמצו"})
        for i, d in new:
            step_day(st, d, i, prices, cfg)
    if dec:
        apply_decisions(st, dec, prices, cfg)
    return stats(st, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state")
    ap.add_argument("--init", action="store_true", help="start a brand new portfolio (ignores --state)")
    ap.add_argument("--prices", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--decisions")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8"))
    prices = json.load(open(a.prices, encoding="utf-8"))
    if a.init:
        st = new_state(cfg)
    else:
        st = json.load(open(a.state, encoding="utf-8"))
        if "cash" not in st and isinstance(st.get("data"), dict):
            st = st["data"]  # unwrap a database export
        if "cash" not in st or "positions" not in st:
            sys.exit("state file is not a portfolio. Refusing to continue; use --init only to start a new portfolio.")
    dec = json.load(open(a.decisions, encoding="utf-8")) if a.decisions else None
    st = run(st, prices, cfg, dec)
    json.dump(st, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    s = st["stats"]
    print(json.dumps({"last_date": st["last_date"], "stats": s, "positions": st["positions"], "pending": st["pending"],
                      "closed_last": st["closed"][-5:], "cancelled_last": st["cancelled"][-3:],
                      "rejected_last": st["rejected"][-3:], "go_live_ready": st["go_live"]["ready"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.exit(main())
