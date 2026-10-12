"""Run with:  python -m tests.test_engine   (no network needed)"""
import copy
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import backtest as bt, checks, data, paper, pipeline, setups  # noqa: E402
from engine.data import align  # noqa: E402
from engine.indicators import build, build_market  # noqa: E402

CFG = pipeline.load_config()
ID = {c[0]: i for i, c in enumerate(checks.C)}
NAME = {c[0]: c[2] for c in checks.C}


def ctx_for(px, uni, t):
    px2, idx = align(px)
    tick = [x for x in uni if x in px2]
    closes = pd.DataFrame({x: px2[x]["close"] for x in tick}, index=idx)
    M, S, X = build_market(px2, closes, {x: uni[x]["sector"] for x in tick})
    return build(px2[t], M, S.get(uni[t]["sector"], S[None]), {n: X[n][t] for n in X}, CFG)


def test_registry():
    assert len(checks.C) == 300
    assert len({c[2] for c in checks.C}) == 300, "duplicate check names"
    cat = checks.catalog()
    assert sum(c["count"] for c in cat["categories"]) == 300


def test_no_lookahead():
    """A check's answer for day t must not change when later days are added."""
    px, uni = data.synthetic(n_tickers=24, n_days=700, seed=3)
    for t in ("SYN000", "SYN004", "SYN013"):
        full = ctx_for(px, uni, t)
        mf = checks.run_all(full)
        sf, _ = setups.compute(full)
        for cut in (450, 610):
            pxc = {k: v[v.index <= full.index[cut]] for k, v in px.items()}
            part = ctx_for(pxc, uni, t)
            mp = checks.run_all(part)
            a, b = mf[cut], mp[-1]
            same = (a == b) | (np.isnan(a) & np.isnan(b))
            bad = [checks.IDS[i] for i in np.flatnonzero(~same)]
            assert not bad, f"look-ahead in {t} at {cut}: {bad}"
            sp, _ = setups.compute(part)
            for k in setups.SETUPS:
                assert bool(sf[k][0][cut]) == bool(sp[k][0][-1]), (t, k, cut)
                if sf[k][0][cut]:
                    assert abs(sf[k][1][cut] - sp[k][1][-1]) < 1e-6


def crafted(n=420, seed=1):
    """A calm uptrend, a tight base, then a breakout on volume, then an open gap."""
    rng = np.random.default_rng(seed)
    px, uni = data.synthetic(n_tickers=12, n_days=n, seed=seed)
    idx = px["SPY"].index
    c = np.empty(n)
    c[:360] = 50 * np.exp(np.linspace(0, 0.55, 360)) * (1 + rng.normal(0, 0.004, 360))
    c[360:400] = c[359] * (1 + rng.normal(0, 0.003, 40))            # 40-day tight base
    c[360:400] = np.minimum(c[360:400], c[359] * 1.004)
    c[400] = c[359] * 1.045                                          # breakout close
    c[401:] = c[400] * (1 + np.linspace(0.0, 0.03, n - 401))
    o = np.r_[c[0], c[:-1]]
    o[410] = c[409] * 1.06                                           # gap up
    c[410:] = o[410] * (1 + np.linspace(0.005, 0.10, n - 410))
    o[411:] = c[410:-1]
    h, l = np.maximum(o, c) * 1.004, np.minimum(o, c) * 0.996
    h[400] = c[400] * 1.002
    v = np.full(n, 2e6)
    v[380:400] = 1.2e6
    v[400] = 6e6
    v[410] = 8e6
    px["TEST"] = pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": v}, index=idx)
    uni["TEST"] = {"name": "Test", "sector": "Industrials"}
    return px, uni


def test_crafted_pattern():
    px, uni = crafted()
    d = ctx_for(px, uni, "TEST")
    m = checks.run_all(d)
    at = lambda cid, i: m[i, ID[cid]]
    # trend checks hold in the uptrend
    for cid in ("TR03", "TR06", "TR11", "TR12", "TR16", "MO04", "MO05"):
        assert at(cid, 399) == 1, (cid, NAME[cid])
    # base is tight before the breakout
    for cid in ("VL09", "VL11", "ST09", "ST10"):
        assert at(cid, 399) == 1, (cid, NAME[cid])
    # breakout day
    for cid in ("ST01", "ST02", "ST04", "ST07", "VO01", "VO02", "VO03", "CA01", "CA02", "CA05", "CA06"):
        assert at(cid, 400) == 1, (cid, NAME[cid])
    assert at("ST02", 399) == 0 and at("VO02", 399) == 0
    su, watch = setups.compute(d)
    assert su["BRK"][0][400], "breakout setup should fire on the breakout day"
    assert not su["BRK"][0][399] and not su["BRK"][0][401]
    stop = su["BRK"][1][400]
    assert d.c[400] - 2.01 * d.atr14[400] <= stop <= d.c[400] - 0.99 * d.atr14[400]
    assert watch["W_BRK"][399], "the day before should be tagged as close to a breakout"
    # open gap
    assert at("CA13", 410) == 1 and at("ST22", 410) == 1 and at("ST22", 414) == 1
    assert at("ST22", 409) == 0 and at("ST22", 416) == 0  # older than 5 days
    assert at("ST23", 420 - 1) == 1
    assert su["GAP"][0][411] or su["GAP"][0][412] or su["GAP"][0][413], "gap follow-through should fire"
    # red flags are clean in a healthy trend
    assert at("RF08", 400) == 1 and at("RF20", 400) == 1


def test_falling_stock_fails_trend():
    px, uni = data.synthetic(n_tickers=12, n_days=420, seed=2)
    idx = px["SPY"].index
    c = 100 * np.exp(np.linspace(0, -0.7, 420))
    o = np.r_[c[0], c[:-1]]
    px["DOWN"] = pd.DataFrame({"open": o, "high": o * 1.003, "low": c * 0.997, "close": c, "volume": 3e6}, index=idx)
    uni["DOWN"] = {"name": "Down", "sector": "Energy"}
    d = ctx_for(px, uni, "DOWN")
    m = checks.run_all(d)
    for cid in ("TR02", "TR03", "TR06", "TR11", "TR16", "MO03", "MO04", "MO10", "RF20", "RS06"):
        assert m[-1, ID[cid]] == 0, (cid, NAME[cid])
    su, _ = setups.compute(d)
    assert not any(su[k][0].any() for k in ("BRK", "PB", "TGT"))


def test_sim_trade():
    n = 30
    o = np.full(n, 100.0); h = np.full(n, 101.0); l = np.full(n, 99.5); c = np.full(n, 100.0)
    res, why = bt.sim_trade(o, h, l, c, 5, 95.0, 2.0, 15, 2.0)
    assert res[4] == "time" and res[2] == 6 + 15 - 1 and res[3] == 100.0
    h2 = h.copy(); h2[9] = 111.0
    assert bt.sim_trade(o, h2, l, c, 5, 95.0, 2.0, 15, 2.0)[0][3:] == (110.0, "target")
    l2 = l.copy(); l2[9] = 94.0
    assert bt.sim_trade(o, h2, l2, c, 5, 95.0, 2.0, 15, 2.0)[0][3:] == (95.0, "stop"), "stop wins a same-day tie"
    o3 = o.copy(); o3[8] = 92.0; l3 = l.copy(); l3[8] = 91.0
    assert bt.sim_trade(o3, h, l3, c, 5, 95.0, 2.0, 15, 2.0)[0][3:] == (92.0, "stop_gap")
    o4 = o.copy(); o4[6] = 94.0
    assert bt.sim_trade(o4, h, l, c, 5, 95.0, 2.0, 15, 2.0) == (None, "opened_below_stop")
    o5 = o.copy(); o5[6] = 103.0
    assert bt.sim_trade(o5, h, l, c, 5, 95.0, 2.0, 15, 2.0) == (None, "gapped_away")
    assert bt.sim_trade(o, h, l, c, 20, 95.0, 2.0, 15, 2.0) == (None, "still_open")
    assert bt.size(100, 95, 10000, CFG) == 20          # $100 risk / $5 per share
    assert bt.size(100, 99, 10000, CFG) == 20          # capped at 20% of equity = $2,000
    assert bt.size(100, 95, 10000, CFG, 0.5) == 10


def test_paper():
    cfg = copy.deepcopy(CFG)
    fee = cfg["commission_per_trade_usd"]
    dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    def P(k, aaa, bbb):
        return {"asof": dates[k - 1], "dates": dates[:k], "sector": {"AAA": "Energy", "BBB": "Energy", "CCC": "Energy"},
                "bars": {"SPY": [[400, 401, 399, 400]] * k, "AAA": aaa[:k], "BBB": bbb[:k], "CCC": bbb[:k]}}
    aaa = [[100, 101, 99, 100], [101, 103, 100.5, 102], [102, 112, 101, 111], [111, 112, 110, 111]]
    bbb = [[50, 51, 49, 50], [50, 50.5, 47, 48], [48, 49, 47, 48], [48, 49, 47, 48]]
    st = paper.run(paper.new_state(cfg), P(1, aaa, bbb), cfg, {"date": dates[0], "regime": "green", "orders": [
        {"t": "AAA", "setup": "BRK", "stop": 96.0, "ref_close": 100.0, "max_entry": 102.0},
        {"t": "BBB", "setup": "PB", "stop": 48.0, "ref_close": 50.0, "max_entry": 51.0},
        {"t": "CCC", "setup": "PB", "stop": 48.0, "ref_close": 50.0, "max_entry": 51.0}]})
    assert len(st["pending"]) == 2 and st["rejected"][-1]["t"] == "CCC", "third order breaks the daily limit"
    st = paper.run(st, P(2, aaa, bbb), cfg)
    # AAA filled at 101: risk/share 5 -> 20 shares capped by 20% of equity (2000/101 = 19)
    a = [p for p in st["positions"] if p["t"] == "AAA"][0]
    assert a["shares"] == 19 and a["entry"] == 101 and a["target"] == 111.0
    # BBB filled at 50 (40 shares by cap), stopped the same day at 48
    b = st["closed"][0]
    assert b["t"] == "BBB" and b["why"] == "stop" and b["shares"] == 40
    assert abs(b["pnl"] - (40 * -2 - 2 * fee)) < 1e-6
    st = paper.run(st, P(3, aaa, bbb), cfg)
    x = st["closed"][-1]
    assert x["t"] == "AAA" and x["why"] == "target" and abs(x["pnl"] - (19 * 10 - 2 * fee)) < 1e-6
    expect = cfg["equity_usd"] + (40 * -2 - 2 * fee) + (19 * 10 - 2 * fee)
    assert abs(st["stats"]["equity"] - expect) < 0.01 and abs(st["cash"] - expect) < 0.01
    assert st["stats"]["closed"] == 2 and not st["go_live"]["ready"]
    # stale decision is refused
    st = paper.run(st, P(4, aaa, bbb), cfg, {"date": dates[0], "regime": "green", "orders": [
        {"t": "AAA", "setup": "BRK", "stop": 105.0, "ref_close": 111.0, "max_entry": 113.0}]})
    assert not st["pending"] and "תאריך" in st["rejected"][-1]["why"]
    # red regime is refused
    st = paper.run(st, P(4, aaa, bbb), cfg, {"date": dates[3], "regime": "red", "orders": [
        {"t": "AAA", "setup": "BRK", "stop": 105.0, "ref_close": 111.0, "max_entry": 113.0}]})
    assert not st["pending"]


def test_pipeline_runs():
    px, uni = data.synthetic(n_tickers=40, n_days=800, seed=5)
    out = pipeline.run(px, uni, CFG, do_backtest=True, log=lambda *a: None)
    L = out["latest"]
    assert 25 <= len(L["table"]) <= 40 and all(len(r["bits"]) == 300 for r in L["table"])
    assert all(r["passed"] + r["failed"] + r["unknown"] == 300 for r in L["table"])
    json.dumps(out, ensure_ascii=False)  # everything serialises
    lift = out["backtest"]["check_lift"]
    assert len(lift) == 300
    # on a random walk nothing should look like a money machine
    port = out["backtest"]["portfolio"]
    assert -60 < port["return_pct"] < 150


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
    print("all tests passed")
