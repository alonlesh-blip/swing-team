"""The five setups the tagger is allowed to use, written as exact rules.

Every setup returns, for each trading day, whether the rule fired at that day's close
and where the stop would sit. Entry is always the next day's open.
"""
import numpy as np

# id -> (Hebrew name, target in R, time stop in trading days)
SETUPS = {
    "BRK": ("פריצה מבסיס", 2.0, 15),
    "PB": ("פולבק במגמה עולה", 2.0, 15),
    "TGT": ("יציאה מכיווץ", 2.0, 15),
    "GAP": ("המשך אחרי גאפ", 2.0, 15),
    "MR": ("היפוך ממכירת יתר במגמה", 1.5, 7),
}
PRIORITY = ["BRK", "GAP", "TGT", "PB", "MR"]
WATCH = {
    "W_BRK": "קרובה לפריצה",
    "W_PB": "בפולבק, מחכה לנר היפוך",
    "W_TGT": "בכיווץ, מחכה ליציאה",
}

RULES_HE = {
    "BRK": [
        "סגירה ראשונה מעל שיא 55 הימים הקודמים",
        "נפח יחסי 1.5 ומעלה",
        "סגירה ב-40% העליונים של טווח היום",
        "מחיר מעל ממוצע 50, וממוצע 50 מעל ממוצע 200",
        "טווח 20 הימים שלפני הפריצה עד 20%, והשיא הקודם נקבע לפני 10 ימים לפחות",
        "לא יותר מ-20% מעל ממוצע 50",
        "סטופ: מתחת לשפל יום הפריצה, לא יותר מ-2 ATR ולא פחות מ-1 ATR",
    ],
    "PB": [
        "ממוצע 50 מעל ממוצע 200 ועולה, מחיר מעל ממוצע 50",
        "דירוג כוח יחסי רבעוני 60 ומעלה",
        "פולבק של 4% עד 15% משיא 20 יום, שהתחיל לפני 3 ימים לפחות",
        "השפל נגע באזור EMA21 ב-3 הימים האחרונים",
        "נר היפוך: סגירה מעל שיא אתמול ומעל הפתיחה",
        "סטופ: מתחת לשפל 5 ימים, בין 1 ל-2.5 ATR",
    ],
    "TGT": [
        "אתמול: רוחב בולינגר ברבע התחתון של 120 יום וטווח 10 ימים עד 10%",
        "היום: סגירה מעל שיא 10 הימים הקודמים בנפח יחסי 1.2 ומעלה",
        "מחיר מעל ממוצע 50, ממוצע 50 מעל ממוצע 200",
        "עד 15% מתחת לשיא השנתי, סגירה בחצי העליון של היום",
        "סטופ: מתחת לשפל 10 ימים, בין 1 ל-2 ATR",
    ],
    "GAP": [
        "גאפ עלייה של 5% ומעלה ב-3 הימים האחרונים, בנפח יחסי 2.5 ומעלה",
        "יום הגאפ נסגר בחצי העליון של הטווח שלו",
        "מאז הגאפ השפלים נשארו מעל שפל יום הגאפ",
        "היום: סגירה מעל כל השיאים מאז הגאפ",
        "סטופ: מתחת לשפל יום הגאפ, בין 1 ל-2.5 ATR",
    ],
    "MR": [
        "מחיר מעל ממוצע 200, ממוצע 50 מעל ממוצע 200",
        "RSI2 מתחת ל-10",
        "3 ימי ירידה רצופים, או ירידה של 6% ב-3 ימים",
        "לא יותר מ-8% מתחת לממוצע 50, דירוג כוח יחסי חצי-שנתי 50 ומעלה",
        "סטופ: 2 ATR מתחת לסגירה. יעד 1.5R, יציאה אחרי 7 ימים",
    ],
}


def sh(a, k):
    out = np.full_like(a, np.nan)
    if k < len(a):
        out[k:] = a[:-k]
    return out


def _t(x):
    """NaN-safe boolean."""
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(x, nan=0.0) > 0


def gates(d):
    with np.errstate(invalid="ignore"):
        g = (d.c >= 10) & (d.dv20 >= 20e6) & (d.nbars >= 252) & (d.atrp >= 0.01) & (d.atrp <= 0.08)
    return g


def compute(d):
    """Returns {setup_id: (signal bool array, stop array)} and {watch_id: bool array}."""
    c, o, h, l, atr = d.c, d.o, d.h, d.l, d.atr14
    out = {}
    with np.errstate(invalid="ignore"):
        trend = (c > d.sma50) & (d.sma50 > d.sma200)

        # BRK
        sig = (
            (c > d.hp55) & (d.pc <= sh(d.hp55, 1)) & (d.rvol >= 1.5) & (d.clv >= 0.6) & trend
            & ((d.hp20 - d.lp20) / c <= 0.20) & (d.base_age >= 10) & (c <= d.sma50 * 1.20)
        )
        stop = np.minimum(np.maximum(l - 0.1 * atr, c - 2 * atr), c - atr)
        out["BRK"] = (sig, stop)

        # PB
        depth = (d.hh20 - d.ll5) / d.hh20
        sig = (
            trend & (d.sma50 > d.sma50_p10) & (d.rk63 >= 60) & (depth >= 0.04) & (depth <= 0.15)
            & (d.dsh20 >= 3) & (d.ll3 <= d.ema21 * 1.01) & (c > d.h_p1) & (c > o)
        )
        stop = np.maximum(np.minimum(d.ll5 - 0.1 * atr, c - atr), c - 2.5 * atr)
        out["PB"] = (sig, stop)

        # TGT
        sig = (
            (sh(d.bbw_rank, 1) <= 0.25) & (sh(d.range10, 1) <= 0.10) & (c > d.hp10) & (d.rvol >= 1.2)
            & trend & (d.dist52h >= -0.15) & (d.clv >= 0.5)
        )
        stop = np.minimum(np.maximum(d.ll10 - 0.1 * atr, c - 2 * atr), c - atr)
        out["TGT"] = (sig, stop)

        # GAP
        sig = np.zeros(len(c), dtype=bool)
        gap_low = np.full(len(c), np.nan)
        for k in (1, 2, 3):
            gday = (sh(d.gap, k) >= 0.05) & (sh(d.rvol, k) >= 2.5) & (sh(d.clv, k) >= 0.5)
            low_since = l.copy()
            for j in range(1, k):
                low_since = np.fmin(low_since, sh(l, j))
            high_before = sh(h, 1)
            for j in range(2, k + 1):
                high_before = np.fmax(high_before, sh(h, j))
            s = gday & (low_since > sh(l, k)) & (c > high_before)
            new = s & ~sig
            gap_low = np.where(new, sh(l, k), gap_low)
            sig |= s
        stop = np.maximum(np.minimum(gap_low - 0.1 * atr, c - atr), c - 2.5 * atr)
        out["GAP"] = (sig, stop)

        # MR
        sig = (
            (c > d.sma200) & (d.sma50 > d.sma200) & (d.rsi2 < 10) & ((d.dn_streak >= 3) | (d.ret3 <= -0.06))
            & (c >= d.sma50 * 0.92) & (d.rk126 >= 50)
        )
        out["MR"] = (sig, c - 2 * atr)

        g = gates(d)
        for k, (s, st) in out.items():
            out[k] = (_t(s) & g & np.isfinite(st), st)

        watch = {
            "W_BRK": g & trend & (c <= d.hh55) & (c >= d.hh55 * 0.97) & (d.range10 <= 0.10) & (d.base_age >= 10),
            "W_PB": g & trend & (d.sma50 > d.sma50_p10) & (d.rk63 >= 60) & (depth >= 0.04) & (depth <= 0.15)
            & (d.ll3 <= d.ema21 * 1.01) & ~((c > d.h_p1) & (c > o)),
            "W_TGT": g & trend & (d.bbw_rank <= 0.25) & (d.range10 <= 0.10) & (d.dist52h >= -0.15) & (c <= d.hp10),
        }
        watch = {k: _t(v) for k, v in watch.items()}
    return out, watch
