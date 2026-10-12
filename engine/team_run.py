"""Glue for the daily team session. Pure standard library.

    python team_run.py prep   --dir WORK      # before the agents think
    python team_run.py finish --dir WORK      # after the boss decided

WORK must hold latest.json, prices.json, catalog.json, backtest.json, config.json and
(if the paper portfolio already exists) state_in.json.

prep   -> advances the paper portfolio over new trading days (state_a.json) and prints
          the briefing the agents work from. Also writes detail_<TICKER>.txt per row.
finish -> reads agents.json, takes every order's numbers from the engine's table (the
          agents never type prices), applies the hard limits, and writes state.json,
          team_latest.json, run_slim.json.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper  # noqa: E402

VERDICTS = {"news": {"ok", "caution", "block"}, "money": {"support", "neutral", "against"},
            "skeptic": {"veto", "doubt", "pass"}}
STATIONS = ["scanner", "tagger", "news", "money", "checkers", "skeptic", "boss"]


def load(d, name, default=None):
    p = os.path.join(d, name)
    if not os.path.exists(p):
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def dump(d, name, obj):
    with open(os.path.join(d, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))


def suspended_setups(bt):
    """A setup whose own history is clearly negative may not be traded."""
    out = {}
    for k, v in (bt or {}).get("per_setup", {}).items():
        a = v.get("all", {})
        if a.get("n", 0) >= 50 and a.get("avg_r") is not None and a["avg_r"] < 0 and (a.get("t_stat") or 0) <= -2:
            out[k] = f"תוחלת היסטורית {a['avg_r']}R על {a['n']} איתותים (t={a['t_stat']})"
    return out


def prep(d):
    L, P, C, B, cfg = (load(d, n) for n in ("latest.json", "prices.json", "catalog.json", "backtest.json", "config.json"))
    if not L or not P or not C or not cfg:
        sys.exit("missing input files")
    if L["asof"] != P["asof"]:
        sys.exit(f"latest.json ({L['asof']}) and prices.json ({P['asof']}) are from different days")
    st = load(d, "state_in.json")
    if st is None:
        st = paper.new_state(cfg)
        fresh = True
    else:
        if "cash" not in st or "positions" not in st:
            sys.exit("state_in.json is not a portfolio; refusing to start a new one over it")
        fresh = False
    before = st.get("last_date")
    st = paper.run(st, P, cfg)
    dump(d, "state_a.json", st)

    names = {c["id"]: c["name"] for c in C["checks"]}
    ids = [c["id"] for c in C["checks"]]
    cat_of = {c["id"]: c["cat"] for c in C["checks"]}
    catn = {c["id"]: c["name"] for c in C["categories"]}
    lift = {x["id"]: x for x in (B or {}).get("check_lift", [])}
    sus = L.get("suspended_setups", suspended_setups(B))
    reg = L["regime"]
    s = st["stats"]
    print("=== BRIEFING ===")
    print(json.dumps({
        "asof": L["asof"], "regime": {k: reg.get(k) for k in ("label", "label_he", "size_mult", "spy", "spy_chg", "spy_vs_sma50",
                                                                "spy_vs_sma200", "spy_from_high", "vix", "breadth50", "breadth200",
                                                                "market_checks_passed")},
        "universe": L["universe"], "signals_today": L["signals_today"], "watch_today": L["watch_today"],
        "sectors_top3": [x["sector"] for x in L["sectors"][:3]], "sectors_bottom3": [x["sector"] for x in L["sectors"][-3:]],
        "rules": {k: cfg[k] for k in ("risk_per_trade_pct", "max_position_pct", "max_open_positions", "max_new_per_day",
                                      "max_per_sector", "max_open_risk_pct", "min_score", "commission_per_trade_usd")},
        "suspended_setups": sus,
    }, ensure_ascii=False))
    print("=== PAPER PORTFOLIO (after processing new trading days) ===")
    print(json.dumps({"new_portfolio": fresh, "processed_from": before, "last_date": st["last_date"],
                      "stats": {k: s[k] for k in ("equity", "return_pct", "spy_return_pct", "closed", "open", "expectancy_r",
                                                  "open_risk_usd", "drawdown_now_pct", "paused")},
                      "positions": [{k: p[k] for k in ("t", "setup", "entry", "last", "stop", "target", "days", "tmax", "sector")}
                                    for p in st["positions"]],
                      "closed_today": [c for c in st["closed"] if c["exit_date"] == st["last_date"] and before != st["last_date"]],
                      "cancelled_today": [c for c in st["cancelled"] if c["date"] == st["last_date"] and before != st["last_date"]],
                      "free_slots": cfg["max_open_positions"] - len(st["positions"])}, ensure_ascii=False))
    if B:
        print("=== HISTORY PER SETUP (net R after commissions; t is significance) ===")
        for k, v in B.get("per_setup", {}).items():
            a = v.get("all", {})
            cur = v.get("by_regime", {}).get(reg["label"], {})
            print(json.dumps({"setup": k, "n": a.get("n"), "win_rate": a.get("win_rate"), "avg_r": a.get("avg_r"),
                              "t": a.get("t_stat"), "in_current_regime": {"n": cur.get("n"), "avg_r": cur.get("avg_r")},
                              "score_above_median_avg_r": (v.get("by_score") or {}).get("above", {}).get("avg_r"),
                              "score_below_median_avg_r": (v.get("by_score") or {}).get("below", {}).get("avg_r")}, ensure_ascii=False))
    print("=== SCANNER TABLE (rows with a setup first, then watch tags, then by score) ===")
    for r in L["table"]:
        fails = [ids[i] for i, b in enumerate(r["bits"]) if b == "0"]
        by = {}
        for f in fails:
            by.setdefault(cat_of[f], []).append(f)
        strong = [f"{names[f]} (t={lift[f]['t']})" for f in fails if f in lift and lift[f].get("t") is not None and lift[f]["t"] >= 2]
        row = {"rank": r["rank"], "t": r["t"], "name": r["name"], "sector": r["sector"], "setups": r["setups"],
               "suspended": r.get("suspended", []),
               "watch": r["watch"], "score": r["score"], "cats": r["cats"], "price": r["price"], "chg": r["chg"],
               "plan": r["plan"], "m": r["m"], "red_flags": [names[x] for x in r["red_flags"]],
               "failed_checks_with_proven_edge": strong[:8]}
        if r["setups"] or r["watch"] or r["rank"] <= 12:
            print(json.dumps(row, ensure_ascii=False))
        with open(os.path.join(d, f"detail_{r['t']}.txt"), "w", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, indent=1) + "\n\nFAILED CHECKS BY CATEGORY\n")
            for c, fs in by.items():
                f.write(f"\n{catn[c]} ({len(fs)}):\n" + "\n".join(f"  - {names[x]}" for x in fs) + "\n")
    print("=== SETUP RULES ===")
    for sdef in C["setups"]:
        print(sdef["id"], sdef["name"], "|", " ; ".join(sdef["rules"]))
    print(f"\nPer-ticker detail (all failed checks) is in {d}/detail_<TICKER>.txt")


def _need(cond, msg, errs):
    if not cond:
        errs.append(msg)


def finish(d):
    L, P, C, B, cfg = (load(d, n) for n in ("latest.json", "prices.json", "catalog.json", "backtest.json", "config.json"))
    st = load(d, "state_a.json")
    A = load(d, "agents.json")
    if not (L and P and cfg and st and A):
        sys.exit("missing input files (run prep first, and write agents.json)")
    rows = {r["t"]: r for r in L["table"]}
    errs = []
    for k in STATIONS:
        _need(isinstance(A.get(k), dict), f"agents.{k} missing", errs)
        if isinstance(A.get(k), dict):
            _need(isinstance(A[k].get("stat"), str) and len(A[k]["stat"]) <= 24, f"agents.{k}.stat must be a short string", errs)
            _need(isinstance(A[k].get("summary"), str) and A[k]["summary"], f"agents.{k}.summary missing", errs)
            for it in A[k].get("items", []):
                _need(it.get("t") in rows, f"agents.{k}: {it.get('t')} is not in today's table", errs)
                if k in VERDICTS:
                    _need(it.get("verdict") in VERDICTS[k], f"agents.{k}: bad verdict for {it.get('t')}", errs)
    if errs:
        sys.exit("agents.json problems:\n- " + "\n- ".join(errs))
    boss = A["boss"]
    _need(isinstance(boss.get("headline"), str) and boss["headline"], "boss.headline missing", errs)
    sus = L.get("suspended_setups", suspended_setups(B))
    blocked = {i["t"] for i in A["news"].get("items", []) if i["verdict"] == "block"}
    vetoed = {i["t"] for i in A["skeptic"].get("items", []) if i["verdict"] == "veto"}
    reviewed = {i["t"] for i in A["skeptic"].get("items", [])}
    orders, refused = [], []
    for o in boss.get("orders", []):
        t = o.get("t")
        r = rows.get(t)
        why = None
        if r is None:
            why = "לא בטבלה של היום"
        elif not r["setups"]:
            why = "אין סטאפ היום, רק מעקב"
        elif r["setups"][0] in sus:
            why = "הסטאפ מושעה: " + sus[r["setups"][0]]
        elif r["score"] < cfg["min_score"]:
            why = f"ציון {r['score']} מתחת לסף {cfg['min_score']}"
        elif t in blocked:
            why = "קורא החדשות פסל"
        elif t in vetoed:
            why = "וטו של המפקפק"
        elif t not in reviewed:
            why = "המפקפק לא עבר על העסקה"
        elif r["plan"] is None or r["plan"].get("provisional"):
            why = "אין תוכנית מסחר מהמנוע"
        elif r["m"].get("earn_days") is not None and r["m"]["earn_days"] <= 5:
            why = "דוח רבעוני בתוך 5 ימי מסחר"
        if why:
            refused.append({"t": t, "why": why})
            continue
        p = r["plan"]
        orders.append({"t": t, "setup": p["setup"], "stop": p["stop"], "ref_close": p["ref_close"], "max_entry": p["max_entry"],
                       "target": p["target"], "shares": p["shares"], "risk_usd": p["risk_usd"],
                       "size_mult": L["regime"]["size_mult"], "note": str(o.get("note", ""))[:300]})
    n_rej = len(st["rejected"])
    dec = {"date": L["asof"], "regime": L["regime"]["label"], "orders": orders,
           "close": [c for c in boss.get("close", []) if c.get("t")], "unpause": bool(boss.get("unpause"))}
    st = paper.run(st, P, cfg, dec)
    for rj in st["rejected"][n_rej:] if len(st["rejected"]) >= n_rej else []:
        refused.append({"t": rj["t"], "why": rj["why"]})
    queued = {p["t"] for p in st["pending"]}
    accepted = [o for o in orders if o["t"] in queued]
    boss_out = dict(boss)
    boss_out["orders"] = accepted
    boss_out["decision"] = "trade" if accepted else "no_trade"
    boss_out["passed_on"] = (boss.get("passed_on") or []) + [x for x in refused]
    if refused and not accepted and boss.get("orders"):
        boss_out["headline"] = "אין עסקה היום: החוקים עצרו את ההזמנה"
    boss_out["stat"] = (f"{len(accepted)} אושרו" if len(accepted) != 1 else "1 אושרה") if accepted else "אין עסקה"
    A = dict(A, boss=boss_out)
    doc = {k: L.get(k) for k in ("regime", "universe", "signals_today", "watch_today", "suspended_setups", "sectors", "table",
                               "engine_version")}
    doc.update({"date": L["asof"], "agents": A})
    dump(d, "team_latest.json", doc)
    dump(d, "run_slim.json", {"date": L["asof"], "regime": L["regime"]["label"], "headline": boss_out["headline"],
                              "decision": boss_out["decision"],
                              "orders": [{"t": o["t"], "setup": o["setup"], "stop": o["stop"], "target": o["target"]} for o in accepted],
                              "vetoes": ", ".join(sorted(vetoed)), "shortlist": A["scanner"].get("shortlist", [])[:12],
                              "equity": st["stats"]["equity"]})
    dump(d, "state.json", st)
    print(json.dumps({"accepted_orders": accepted, "refused": refused, "headline": boss_out["headline"],
                      "portfolio": st["stats"], "go_live_ready": st["go_live"]["ready"]}, ensure_ascii=False, indent=1))
    print(f"\nwrote {d}/team_latest.json, run_slim.json, state.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "finish"])
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    (prep if a.cmd == "prep" else finish)(a.dir)
