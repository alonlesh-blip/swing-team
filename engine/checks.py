"""The 300 checkers. Each one looks at exactly one thing and answers yes / no / unknown.

A check is (id, category, Hebrew name, function). The function receives the indicator
namespace of one ticker and returns a float array over all trading days:
1.0 = pass, 0.0 = fail, NaN = unknown. "Pass" is always the good direction for a long
swing trade, so red-flag checks are phrased as "no <bad thing>".
"""
import numpy as np

CATEGORIES = [
    ("TR", "מגמה"),
    ("MO", "מומנטום"),
    ("RS", "כוח יחסי"),
    ("VO", "נפח"),
    ("VL", "תנודתיות וכיווץ"),
    ("ST", "מבנה מחיר"),
    ("CA", "נרות"),
    ("LQ", "נזילות"),
    ("MK", "משטר שוק"),
    ("SB", "סקטור ורוחב"),
    ("TG", "גיאומטריית עסקה"),
    ("RF", "דגלים אדומים"),
]


def _mask(r, *xs):
    for x in xs:
        if isinstance(x, np.ndarray):
            r[np.isnan(x)] = np.nan
    return r


def gt(a, b):
    with np.errstate(invalid="ignore"):
        return _mask((a > b).astype("float64"), a, b)


def ge(a, b):
    with np.errstate(invalid="ignore"):
        return _mask((a >= b).astype("float64"), a, b)


def lt(a, b):
    return gt(b, a)


def le(a, b):
    return ge(b, a)


def btw(a, lo, hi):
    with np.errstate(invalid="ignore"):
        return _mask(((a >= lo) & (a <= hi)).astype("float64"), a, lo, hi)


def yes(a):
    """a is already 1/0/NaN."""
    return a.copy()


def no(a):
    return 1.0 - a


def AND(*xs):
    r = np.ones_like(xs[0])
    for x in xs:
        r = r * x
    return r


def OR(*xs):
    r = np.zeros_like(xs[0])
    for x in xs:
        r = np.maximum(r, x)  # NaN propagates
    return r


C = []  # the registry


def add(cat, name, fn):
    n = sum(1 for c in C if c[1] == cat) + 1
    C.append((f"{cat}{n:02d}", cat, name, fn))


# ------------------------------------------------------------------ TR · מגמה (30)
add("TR", "סגירה מעל ממוצע 10", lambda d: gt(d.c, d.sma10))
add("TR", "סגירה מעל ממוצע 20", lambda d: gt(d.c, d.sma20))
add("TR", "סגירה מעל ממוצע 50", lambda d: gt(d.c, d.sma50))
add("TR", "סגירה מעל ממוצע 100", lambda d: gt(d.c, d.sma100))
add("TR", "סגירה מעל ממוצע 150", lambda d: gt(d.c, d.sma150))
add("TR", "סגירה מעל ממוצע 200", lambda d: gt(d.c, d.sma200))
add("TR", "ממוצע 10 מעל ממוצע 20", lambda d: gt(d.sma10, d.sma20))
add("TR", "ממוצע 20 מעל ממוצע 50", lambda d: gt(d.sma20, d.sma50))
add("TR", "ממוצע 50 מעל ממוצע 100", lambda d: gt(d.sma50, d.sma100))
add("TR", "ממוצע 50 מעל ממוצע 150", lambda d: gt(d.sma50, d.sma150))
add("TR", "ממוצע 50 מעל ממוצע 200", lambda d: gt(d.sma50, d.sma200))
add("TR", "ממוצע 150 מעל ממוצע 200", lambda d: gt(d.sma150, d.sma200))
add("TR", "ממוצע 20 עולה (מול לפני 5 ימים)", lambda d: gt(d.sma20, d.sma20_p5))
add("TR", "ממוצע 50 עולה (מול לפני 10 ימים)", lambda d: gt(d.sma50, d.sma50_p10))
add("TR", "ממוצע 150 עולה (מול לפני חודש)", lambda d: gt(d.sma150, d.sma150_p21))
add("TR", "ממוצע 200 עולה (מול לפני חודש)", lambda d: gt(d.sma200, d.sma200_p21))
add("TR", "ממוצע 200 עולה גם מול לפני רבעון", lambda d: gt(d.sma200, d.sma200_p63))
add("TR", "EMA9 מעל EMA21", lambda d: gt(d.ema9, d.ema21))
add("TR", "סגירה מעל EMA21", lambda d: gt(d.c, d.ema21))
add("TR", "ADX מעל 20 (יש מגמה)", lambda d: gt(d.adx, 20))
add("TR", "ADX מעל 30 (מגמה חזקה)", lambda d: gt(d.adx, 30))
add("TR", "‎+DI מעל ‎-DI", lambda d: gt(d.pdi, d.mdi))
add("TR", "ADX עולה (מול לפני 5 ימים)", lambda d: gt(d.adx, d.adx_p5))
add("TR", "לפחות 25% מעל שפל 52 שבועות", lambda d: ge(d.dist52l, 0.25))
add("TR", "בטווח 25% משיא 52 שבועות", lambda d: ge(d.dist52h, -0.25))
add("TR", "בטווח 10% משיא 52 שבועות", lambda d: ge(d.dist52h, -0.10))
add("TR", "שיא 20 יום גבוה משיא 20 הימים שלפניו", lambda d: gt(d.hh20, d.hh20_p20))
add("TR", "שפל 20 יום גבוה משפל 20 הימים שלפניו", lambda d: gt(d.ll20, d.ll20_p20))
add("TR", "80% מ-50 הימים האחרונים נסגרו מעל ממוצע 50", lambda d: ge(d.pct_above50_50d, 80))
add("TR", "סגירה בחצי העליון של טווח 52 שבועות", lambda d: gt(d.c, d.mid252))

# ------------------------------------------------------------------ MO · מומנטום (30)
add("MO", "תשואת 5 ימים חיובית", lambda d: gt(d.ret5, 0))
add("MO", "תשואת 10 ימים חיובית", lambda d: gt(d.ret10, 0))
add("MO", "תשואת חודש (21) חיובית", lambda d: gt(d.ret21, 0))
add("MO", "תשואת רבעון (63) חיובית", lambda d: gt(d.ret63, 0))
add("MO", "תשואת חצי שנה (126) חיובית", lambda d: gt(d.ret126, 0))
add("MO", "תשואת שנה (252) חיובית", lambda d: gt(d.ret252, 0))
add("MO", "תשואת רבעון מעל 10%", lambda d: gt(d.ret63, 0.10))
add("MO", "תשואת חצי שנה מעל 20%", lambda d: gt(d.ret126, 0.20))
add("MO", "מומנטום 12-1 חיובי (שנה בלי החודש האחרון)", lambda d: gt(d.mom12_1, 0))
add("MO", "RSI14 מעל 50", lambda d: gt(d.rsi14, 50))
add("MO", "RSI14 מעל 60", lambda d: gt(d.rsi14, 60))
add("MO", "RSI14 בין 50 ל-70", lambda d: btw(d.rsi14, 50, 70))
add("MO", "RSI14 עולה (מול לפני 5 ימים)", lambda d: gt(d.rsi14, d.rsi14_p5))
add("MO", "MACD מעל קו האות", lambda d: gt(d.macd, d.macds))
add("MO", "MACD מעל אפס", lambda d: gt(d.macd, 0))
add("MO", "היסטוגרמת MACD עולה מול אתמול", lambda d: gt(d.macdh, d.macdh_p1))
add("MO", "היסטוגרמת MACD חיובית", lambda d: gt(d.macdh, 0))
add("MO", "סטוכסטי 14 מעל 50", lambda d: gt(d.stoch, 50))
add("MO", "סטוכסטי 14 מעל 80", lambda d: gt(d.stoch, 80))
add("MO", "מומנטום מאיץ: תשואת חודש גבוהה מלפני 5 ימים", lambda d: gt(d.ret21, d.ret21_p5))
add("MO", "שיא סגירה של 20 יום ב-5 הימים האחרונים", lambda d: yes(d.newhc20_5))
add("MO", "שיא סגירה של 55 יום ב-10 הימים האחרונים", lambda d: yes(d.newhc55_10))
add("MO", "שיא 52 שבועות בחודש האחרון", lambda d: yes(d.newhh252_21))
add("MO", "לפחות 11 ימי עלייה מתוך 20", lambda d: ge(d.updays20, 11))
add("MO", "החודש האחרון חזק מהחודש שלפניו", lambda d: gt(d.ret21, d.ret21_p21))
add("MO", "סגירה מעל הסגירה של לפני 3 ימים", lambda d: gt(d.c, d.c_p3))
add("MO", "RSI14 לא ירד מתחת ל-40 ב-20 יום", lambda d: ge(d.rsi14_min20, 40))
add("MO", "תשואת רבעון בין 0 ל-60% (חזק, לא פרבולי)", lambda d: btw(d.ret63, 0, 0.60))
add("MO", "RSI2 מעל 50 (חוזק קצר טווח)", lambda d: gt(d.rsi2, 50))
add("MO", "סגירה בחצי העליון של טווח 20 יום", lambda d: ge(d.pos20, 0.5))

# ------------------------------------------------------------------ RS · כוח יחסי (25)
add("RS", "מכה את SPY ב-5 ימים", lambda d: gt(d.ex5, 0))
add("RS", "מכה את SPY בחודש", lambda d: gt(d.ex21, 0))
add("RS", "מכה את SPY ברבעון", lambda d: gt(d.ex63, 0))
add("RS", "מכה את SPY בחצי שנה", lambda d: gt(d.ex126, 0))
add("RS", "קו RS מעל ממוצע 20 שלו", lambda d: gt(d.rs, d.rs_sma20))
add("RS", "קו RS מעל ממוצע 50 שלו", lambda d: gt(d.rs, d.rs_sma50))
add("RS", "ממוצע 50 של קו RS עולה", lambda d: gt(d.rs_sma50, d.rs_sma50_p10))
add("RS", "קו RS בשיא 55 יום ב-5 הימים האחרונים", lambda d: yes(d.rs_newhigh55_5))
add("RS", "קו RS בשיא חצי שנה ב-10 הימים האחרונים", lambda d: yes(d.rs_newhigh126_10))
add("RS", "קו RS בשיא לפני המחיר", lambda d: yes(d.rs_lead))
add("RS", "דירוג חודש ביקום מעל 50", lambda d: gt(d.rk21, 50))
add("RS", "דירוג חודש ביקום מעל 70", lambda d: gt(d.rk21, 70))
add("RS", "דירוג רבעון ביקום מעל 50", lambda d: gt(d.rk63, 50))
add("RS", "דירוג רבעון ביקום מעל 70", lambda d: gt(d.rk63, 70))
add("RS", "דירוג רבעון ביקום מעל 80", lambda d: gt(d.rk63, 80))
add("RS", "דירוג רבעון ביקום מעל 90", lambda d: gt(d.rk63, 90))
add("RS", "דירוג חצי שנה ביקום מעל 70", lambda d: gt(d.rk126, 70))
add("RS", "דירוג שנה ביקום מעל 70", lambda d: gt(d.rk252, 70))
add("RS", "דירוג משוקלל ביקום מעל 80", lambda d: gt(d.rkc, 80))
add("RS", "דירוג הרבעון השתפר בחודש האחרון", lambda d: gt(d.rk63, d.rk63_p21))
add("RS", "מכה את ETF הסקטור ברבעון", lambda d: gt(d.ret63, d.S.sec_ret63))
add("RS", "מכה את ETF הסקטור בחודש", lambda d: gt(d.ret21, d.S.sec_ret21))
add("RS", "יורדת פחות מ-SPY בימי ירידה (חודש)", lambda d: gt(d.downcap_stock, d.downcap_spy))
add("RS", "מכה את QQQ ברבעון", lambda d: gt(d.ex63_qqq, 0))
add("RS", "קו RS לפחות 3% מעל שפל 55 יום שלו", lambda d: ge(d.rs, d.rs_ll55 * 1.03))

# ------------------------------------------------------------------ VO · נפח (30)
add("VO", "נפח היום מעל ממוצע 20", lambda d: gt(d.rvol, 1.0))
add("VO", "נפח יחסי מעל 1.5", lambda d: gt(d.rvol, 1.5))
add("VO", "נפח יחסי מעל 2", lambda d: gt(d.rvol, 2.0))
add("VO", "ממוצע נפח 20 מעל ממוצע 50 (עניין גובר)", lambda d: gt(d.v20, d.v50))
add("VO", "נפח עליות גדול מנפח ירידות (20 יום)", lambda d: gt(d.udr20, 1.0))
add("VO", "נפח עליות פי 1.5 מנפח ירידות (20 יום)", lambda d: gt(d.udr20, 1.5))
add("VO", "נפח עליות גדול מנפח ירידות (50 יום)", lambda d: gt(d.udr50, 1.0))
add("VO", "נפח עליות פי 1.25 מנפח ירידות (50 יום)", lambda d: gt(d.udr50, 1.25))
add("VO", "OBV מעל ממוצע 20 שלו", lambda d: gt(d.obv, d.obv_sma20))
add("VO", "OBV בשיא 55 יום ב-10 הימים האחרונים", lambda d: yes(d.obv_newhigh10))
add("VO", "OBV עלה ב-20 הימים האחרונים", lambda d: gt(d.obv, d.obv_p20))
add("VO", "לפחות 5 ימי צבירה ב-20 יום", lambda d: ge(d.acc20, 5))
add("VO", "ימי צבירה יותר מימי פיזור", lambda d: gt(d.acc20, d.dist25))
add("VO", "לכל היותר 3 ימי פיזור ב-25 יום", lambda d: le(d.dist25, 3))
add("VO", "Pocket pivot ב-10 הימים האחרונים", lambda d: yes(d.pp10))
add("VO", "Pocket pivot היום", lambda d: yes(d.pp))
add("VO", "התייבשות נפח: ממוצע 5 מתחת ל-70% מממוצע 50", lambda d: lt(d.v5, d.v50 * 0.7))
add("VO", "ממוצע נפח 10 מתחת לממוצע 50 (שקט)", lambda d: lt(d.v10, d.v50))
add("VO", "יום הנפח הגבוה ב-20 יום היה יום עלייה", lambda d: yes(d.maxvol20_up))
add("VO", "יום הנפח הגבוה ב-50 יום היה יום עלייה", lambda d: yes(d.maxvol50_up))
add("VO", "ביום הירידה האחרון הנפח היה מתחת לממוצע", lambda d: lt(d.last_dn_rvol, 1.0))
add("VO", "ביום העלייה האחרון הנפח היה מעל הממוצע", lambda d: gt(d.last_up_rvol, 1.0))
add("VO", "המחזור הדולרי גדל מול לפני רבעון", lambda d: gt(d.dv20, d.dv20_p63))
add("VO", "נפח 5 הימים האחרונים גבוה מ-5 הקודמים", lambda d: gt(d.vsum5, d.vsum5_p5))
add("VO", "זרימת כסף CMF20 חיובית", lambda d: gt(d.cmf20, 0))
add("VO", "זרימת כסף CMF20 מעל 0.1", lambda d: gt(d.cmf20, 0.1))
add("VO", "MFI14 מעל 50", lambda d: gt(d.mfi, 50))
add("VO", "קו צבירה/פיזור מעל ממוצע 20 שלו", lambda d: gt(d.adl, d.adl_sma20))
add("VO", "אין יום ירידה בנפח כפול ב-10 ימים", lambda d: no(d.heavy_dn10))
add("VO", "נפח היום גבוה מכל אחד מ-3 הימים הקודמים", lambda d: gt(d.v, d.vmax3_p))

# ------------------------------------------------------------------ VL · תנודתיות וכיווץ (25)
add("VL", "ATR בין 1.5% ל-6% מהמחיר", lambda d: btw(d.atrp, 0.015, 0.06))
add("VL", "ATR מתחת ל-4% מהמחיר", lambda d: lt(d.atrp, 0.04))
add("VL", "ATR5 קטן מ-ATR20 (כיווץ)", lambda d: lt(d.atr5, d.atr20))
add("VL", "ATR14 קטן מ-ATR50", lambda d: lt(d.atr14, d.atr50))
add("VL", "רוחב בולינגר ברבע התחתון של 120 יום", lambda d: le(d.bbw_rank, 0.25))
add("VL", "רוחב בולינגר בעשירון התחתון של 120 יום", lambda d: le(d.bbw_rank, 0.10))
add("VL", "סקוויז: בולינגר בתוך קלטנר", lambda d: yes(d.squeeze))
add("VL", "סקוויז השתחרר ב-5 הימים האחרונים", lambda d: yes(d.squeeze_fired5))
add("VL", "טווח 10 ימים צר מ-10%", lambda d: lt(d.range10, 0.10))
add("VL", "טווח 10 ימים צר מ-6%", lambda d: lt(d.range10, 0.06))
add("VL", "טווח 20 יום צר מ-15%", lambda d: lt(d.range20, 0.15))
add("VL", "טווח 5 ימים צר מ-5%", lambda d: lt(d.range5, 0.05))
add("VL", "NR7 היום (הטווח הצר ב-7 ימים)", lambda d: yes(d.nr7))
add("VL", "יום פנימי היום", lambda d: yes(d.inside))
add("VL", "יום שקט (NR7 או פנימי) ב-3 הימים האחרונים", lambda d: yes(d.quiet3))
add("VL", "תנודתיות 10 ימים נמוכה מ-50 יום", lambda d: lt(d.sd10, d.sd50))
add("VL", "טווח 10 הימים צר מ-10 הימים שלפניהם", lambda d: lt(d.range10, d.range10_p10))
add("VL", "גם הכיווץ הקודם היה צר מקודמו", lambda d: lt(d.range10_p10, d.range10_p20))
add("VL", "סגירה מתחת לרצועת בולינגר העליונה", lambda d: lt(d.c, d.bbu))
add("VL", "תנודתיות חודש נמוכה מתנודתיות רבעון", lambda d: lt(d.sd21, d.sd63))
add("VL", "טווח היום קטן מ-ATR", lambda d: lt(d.rng, d.atr14))
add("VL", "אין יום עם טווח מעל 3 ATR ב-10 ימים", lambda d: no(d.wide_any10))
add("VL", "גאפ פתיחה ממוצע מתחת ל-1% (20 יום)", lambda d: lt(d.absgap20, 0.01))
add("VL", "ATR% לא יותר מפי 3 מזה של SPY", lambda d: le(d.atrp, d.M.spy_atrp * 3))
add("VL", "3 סגירות אחרונות בטווח של 1.5%", lambda d: le(d.tight3, 0.015))

# ------------------------------------------------------------------ ST · מבנה מחיר (30)
add("ST", "סגירה מעל שיא 20 הימים הקודמים", lambda d: gt(d.c, d.hp20))
add("ST", "סגירה מעל שיא 55 הימים הקודמים", lambda d: gt(d.c, d.hp55))
add("ST", "סגירה מעל שיא חצי השנה הקודמת", lambda d: gt(d.c, d.hp126))
add("ST", "שיא 52 שבועות היום", lambda d: ge(d.h, d.hh252))
add("ST", "בטווח 3% מתחת לשיא 20 יום", lambda d: ge(d.c, d.hh20 * 0.97))
add("ST", "בטווח 5% מתחת לשיא 55 יום", lambda d: ge(d.c, d.hh55 * 0.95))
add("ST", "סגירה מעל שיא הסגירות של 55 הימים הקודמים", lambda d: gt(d.c, d.hcp55))
add("ST", "הבסיס נמשך 15 ימים לפחות", lambda d: ge(d.base_age, 15))
add("ST", "עומק הבסיס עד 20%", lambda d: le(d.base_depth, 0.20))
add("ST", "עומק הבסיס עד 12%", lambda d: le(d.base_depth, 0.12))
add("ST", "המחיר בטווח 2% מממוצע 20", lambda d: btw(d.c, d.sma20 * 0.98, d.sma20 * 1.02))
add("ST", "השפל נגע באזור EMA21 ב-3 הימים האחרונים", lambda d: le(d.ll3, d.ema21 * 1.01))
add("ST", "השפל נגע באזור ממוצע 50 ב-5 ימים והסגירה מעליו", lambda d: AND(le(d.ll5, d.sma50 * 1.02), gt(d.c, d.sma50)))
add("ST", "עד 1.5 ATR מעל ממוצע 20", lambda d: le(d.c, d.sma20 + 1.5 * d.atr14))
add("ST", "פולבק בריא: 3% עד 12% מתחת לשיא 20 יום", lambda d: btw(d.dd20, -0.12, -0.03))
add("ST", "3 עד 8 ימים מאז שיא 20 יום", lambda d: btw(d.dsh20, 3, 8))
add("ST", "שפל 5 הימים מעל ממוצע 50", lambda d: gt(d.ll5, d.sma50))
add("ST", "שפל 5 הימים גבוה משפל 5 הימים שלפניהם", lambda d: gt(d.ll5, d.ll5_p5))
add("ST", "שיא 5 הימים גבוה משיא 5 הימים שלפניהם", lambda d: gt(d.hh5, d.hh5_p5))
add("ST", "סגירה מעל VWAP של 20 יום", lambda d: gt(d.c, d.vwap20))
add("ST", "סגירה מעל VWAP של 50 יום", lambda d: gt(d.c, d.vwap50))
add("ST", "גאפ עלייה מעל 4% ב-5 ימים שלא נסגר", lambda d: yes(d.open_gap4_5))
add("ST", "גאפ עלייה מעל 2% ב-20 יום שלא נסגר", lambda d: yes(d.open_gap2_20))
add("ST", "מעל הסגירה של יום הנפח הגבוה ב-50 יום", lambda d: ge(d.c, d.hv50_close))
add("ST", "אין תקרה קרובה: השיא השנתי רחוק 8%+ או שכבר שם", lambda d: OR(le(d.dist52h, -0.08), ge(d.dist52h, -0.02)))
add("ST", "מעל השיא השנתי כפי שהיה לפני רבעון", lambda d: gt(d.c, d.hh252_p63))
add("ST", "שפל 20 יום לא נשבר ב-10 הימים האחרונים", lambda d: gt(d.ll10, d.ll20_p10))
add("ST", "סגירה בחצי העליון של טווח 55 יום", lambda d: gt(d.c, d.mid55))
add("ST", "שפל 5 הימים מעל ממוצע 20", lambda d: gt(d.ll5, d.sma20))
add("ST", "10 סגירות רצופות מעל ממוצע 20", lambda d: yes(d.all_above20_10d))

# ------------------------------------------------------------------ CA · נרות (25)
add("CA", "יום עלייה היום", lambda d: yes(d.up))
add("CA", "סגירה ב-40% העליונים של טווח היום", lambda d: ge(d.clv, 0.6))
add("CA", "סגירה ברבע העליון של טווח היום", lambda d: ge(d.clv, 0.75))
add("CA", "סגירה מעל הפתיחה", lambda d: gt(d.c, d.o))
add("CA", "סגירה מעל שיא אתמול", lambda d: gt(d.c, d.h_p1))
add("CA", "סגירה מעל שיא 3 הימים הקודמים", lambda d: gt(d.c, d.hp3))
add("CA", "שפל היום מעל שפל אתמול", lambda d: gt(d.l, d.l_p1))
add("CA", "גוף הנר מעל חצי מהטווח", lambda d: gt(d.body, 0.5))
add("CA", "פתיל עליון קטן מ-30% מהטווח", lambda d: lt(d.uw, 0.3))
add("CA", "פתיל תחתון ארוך (מעל 40%): נדחו מחירים נמוכים", lambda d: gt(d.lw, 0.4))
add("CA", "בולעני שורי היום", lambda d: yes(d.engulf_bull))
add("CA", "יום היפוך: שפל 5 ימים חדש וסגירה חיובית", lambda d: yes(d.reversal_up))
add("CA", "גאפ עלייה היום (מעל 0.5%)", lambda d: gt(d.gap, 0.005))
add("CA", "השפל היום נשאר מעל סגירת אתמול", lambda d: gt(d.l, d.pc))
add("CA", "לפחות 2 ימי עלייה רצופים", lambda d: ge(d.up_streak, 2))
add("CA", "לפחות 3 ימי עלייה רצופים", lambda d: ge(d.up_streak, 3))
add("CA", "נר עלייה רחב: טווח מעל ATR ביום עלייה", lambda d: AND(gt(d.rng, d.atr14), yes(d.up)))
add("CA", "סגירה ברבע העליון של טווח 5 ימים", lambda d: ge(d.pos5, 0.75))
add("CA", "סגירה בשליש העליון של טווח 10 ימים", lambda d: ge(d.pos10, 0.667))
add("CA", "פריצה מעל יום פנימי של אתמול", lambda d: yes(d.inside_break))
add("CA", "אין נר ירידה רחב (מעל 1.5 ATR) ב-5 ימים", lambda d: no(d.wide_dn5))
add("CA", "3 מ-5 הימים האחרונים נסגרו בחצי העליון", lambda d: ge(d.upper_half5, 3))
add("CA", "השפל היום לא שבר את שפל 3 הימים הקודמים", lambda d: ge(d.l, d.lp3))
add("CA", "שינוי יומי בין 0 ל-5%", lambda d: btw(d.ret1, 0, 0.05))
add("CA", "סגירה מעל אמצע הנר של אתמול", lambda d: gt(d.c, d.mid_p1))

# ------------------------------------------------------------------ LQ · נזילות (15)
add("LQ", "מחיר מעל 10 דולר", lambda d: gt(d.c, 10))
add("LQ", "מחיר מעל 20 דולר", lambda d: gt(d.c, 20))
add("LQ", "מחזור יומי ממוצע מעל 20 מיליון דולר", lambda d: gt(d.dv20, 20e6))
add("LQ", "מחזור יומי ממוצע מעל 50 מיליון דולר", lambda d: gt(d.dv20, 50e6))
add("LQ", "מחזור יומי ממוצע מעל 100 מיליון דולר", lambda d: gt(d.dv20, 100e6))
add("LQ", "נפח ממוצע מעל 500 אלף מניות", lambda d: gt(d.v20, 5e5))
add("LQ", "נפח ממוצע מעל מיליון מניות", lambda d: gt(d.v20, 1e6))
add("LQ", "שנה לפחות של היסטוריית מסחר", lambda d: ge(d.nbars, 252))
add("LQ", "שנתיים לפחות של היסטוריית מסחר", lambda d: ge(d.nbars, 500))
add("LQ", "אין יום בלי מסחר ב-60 יום", lambda d: le(d.zero_vol60, 0))
add("LQ", "גם ביום החלש ב-20 יום המחזור מעל 5 מיליון דולר", lambda d: gt(d.dvmin20, 5e6))
add("LQ", "טווח יומי חציוני מתחת ל-5%", lambda d: lt(d.med_range20, 0.05))
add("LQ", "המחיר מאפשר לפחות 5 מניות בפוזיציה מלאה", lambda d: le(d.c, d.cfg["equity_usd"] * d.cfg["max_position_pct"] / 100 / 5))
add("LQ", "המחזור לא נחלש: 20 יום לפחות 60% מ-50 יום", lambda d: ge(d.dv20, d.dv50 * 0.6))
add("LQ", "הגאפ הגרוע למטה ב-60 יום קטן מ-6%", lambda d: gt(d.worst_gap60, -0.06))

# ------------------------------------------------------------------ MK · משטר שוק (25)
add("MK", "SPY מעל ממוצע 20", lambda d: gt(d.M.spy_c, d.M.spy_sma20))
add("MK", "SPY מעל ממוצע 50", lambda d: gt(d.M.spy_c, d.M.spy_sma50))
add("MK", "SPY מעל ממוצע 200", lambda d: gt(d.M.spy_c, d.M.spy_sma200))
add("MK", "SPY: ממוצע 50 מעל ממוצע 200", lambda d: gt(d.M.spy_sma50, d.M.spy_sma200))
add("MK", "SPY: ממוצע 50 עולה", lambda d: gt(d.M.spy_sma50, d.M.spy_sma50_p10))
add("MK", "SPY: ממוצע 200 עולה", lambda d: gt(d.M.spy_sma200, d.M.spy_sma200_p21))
add("MK", "SPY בטווח 5% מהשיא השנתי", lambda d: ge(d.M.spy_dist_high, -0.05))
add("MK", "SPY בטווח 10% מהשיא השנתי", lambda d: ge(d.M.spy_dist_high, -0.10))
add("MK", "QQQ מעל ממוצע 50", lambda d: gt(d.M.qqq_c, d.M.qqq_sma50))
add("MK", "QQQ מעל ממוצע 200", lambda d: gt(d.M.qqq_c, d.M.qqq_sma200))
add("MK", "IWM מעל ממוצע 50", lambda d: gt(d.M.iwm_c, d.M.iwm_sma50))
add("MK", "IWM מעל ממוצע 200", lambda d: gt(d.M.iwm_c, d.M.iwm_sma200))
add("MK", "VIX מתחת ל-20", lambda d: lt(d.M.vix, 20))
add("MK", "VIX מתחת ל-25", lambda d: lt(d.M.vix, 25))
add("MK", "VIX מתחת ל-30", lambda d: lt(d.M.vix, 30))
add("MK", "VIX מתחת לממוצע 20 שלו", lambda d: lt(d.M.vix, d.M.vix_sma20))
add("MK", "VIX לא קפץ מעל 20% ב-5 ימים", lambda d: le(d.M.vix_chg5, 0.20))
add("MK", "SPY לא ירד יותר מ-3% ב-5 ימים", lambda d: gt(d.M.spy_ret5, -0.03))
add("MK", "SPY חיובי בחודש האחרון", lambda d: gt(d.M.spy_ret21, 0))
add("MK", "SPY: לכל היותר 4 ימי פיזור ב-25 יום", lambda d: le(d.M.spy_dist25, 4))
add("MK", "SPY: RSI14 מעל 45", lambda d: gt(d.M.spy_rsi14, 45))
add("MK", "SPY לא ירד יותר מ-2% ביום אחד ב-10 ימים", lambda d: gt(d.M.spy_worst10, -0.02))
add("MK", "QQQ מוביל את SPY בחודש (תיאבון לסיכון)", lambda d: gt(d.M.qqq_ret21, d.M.spy_ret21))
add("MK", "IWM מוביל את SPY בחודש (רוחב)", lambda d: gt(d.M.iwm_ret21, d.M.spy_ret21))
add("MK", "SPY ביום עלייה היום", lambda d: yes(d.M.spy_up))

# ------------------------------------------------------------------ SB · סקטור ורוחב (20)
add("SB", "ETF הסקטור מעל ממוצע 50", lambda d: gt(d.S.sec_c, d.S.sec_sma50))
add("SB", "ETF הסקטור מעל ממוצע 200", lambda d: gt(d.S.sec_c, d.S.sec_sma200))
add("SB", "ETF הסקטור: ממוצע 50 עולה", lambda d: gt(d.S.sec_sma50, d.S.sec_sma50_p10))
add("SB", "הסקטור מכה את SPY בחודש", lambda d: gt(d.S.sec_ret21, d.M.spy_ret21))
add("SB", "הסקטור מכה את SPY ברבעון", lambda d: gt(d.S.sec_ret63, d.M.spy_ret63))
add("SB", "הסקטור בין 4 המובילים בחודש", lambda d: le(d.S.sec_rank21, 4))
add("SB", "הסקטור בין 4 המובילים ברבעון", lambda d: le(d.S.sec_rank63, 4))
add("SB", "הסקטור לא בין 3 החלשים בחודש", lambda d: le(d.S.sec_rank21, 8))
add("SB", "הסקטור בטווח 5% מהשיא השנתי", lambda d: ge(d.S.sec_dist_high, -0.05))
add("SB", "ETF הסקטור ביום עלייה היום", lambda d: yes(d.S.sec_up))
add("SB", "יותר מ-50% מהמניות מעל ממוצע 50", lambda d: gt(d.M.br_above50, 50))
add("SB", "יותר מ-60% מהמניות מעל ממוצע 50", lambda d: gt(d.M.br_above50, 60))
add("SB", "יותר מ-50% מהמניות מעל ממוצע 200", lambda d: gt(d.M.br_above200, 50))
add("SB", "הרוחב משתפר: יותר מניות מעל ממוצע 50 מלפני 5 ימים", lambda d: gt(d.M.br_above50, d.M.br_above50_p5))
add("SB", "יותר מ-50% מהמניות מעל ממוצע 20", lambda d: gt(d.M.br_above20, 50))
add("SB", "יותר שיאים חדשים משפלים חדשים היום", lambda d: gt(d.M.br_nhnl, 0))
add("SB", "יותר שיאים משפלים בממוצע 10 ימים", lambda d: gt(d.M.br_nhnl10, 0))
add("SB", "יותר עולות מיורדות היום", lambda d: gt(d.M.br_advdec, 0))
add("SB", "יותר עולות מיורדות ב-10 ימים", lambda d: gt(d.M.br_ad10, 1.0))
add("SB", "יותר מ-50% ממניות הסקטור מעל ממוצע 50", lambda d: gt(d.S.sec_breadth50, 50))

# ------------------------------------------------------------------ TG · גיאומטריית עסקה (20)
add("TG", "הסיכון לסטופ עד 8% מהמחיר", lambda d: le(d.risk_g, 0.08))
add("TG", "הסיכון לסטופ עד 6% מהמחיר", lambda d: le(d.risk_g, 0.06))
add("TG", "הסטופ לא צמוד מדי: לפחות 1.5% מהמחיר", lambda d: ge(d.risk_g, 0.015))
add("TG", "הפוזיציה מנצלת לפחות חצי מתקציב הסיכון", lambda d: ge(d.risk_g, d.cfg["risk_per_trade_pct"] / d.cfg["max_position_pct"] / 2))
add("TG", "הסטופ מתחת לשפל 5 ימים", lambda d: lt(d.stop_g, d.ll5))
add("TG", "הסטופ מתחת לשפל 10 ימים", lambda d: lt(d.stop_g, d.ll10))
add("TG", "הסטופ מתחת לממוצע 20", lambda d: lt(d.stop_g, d.sma20))
add("TG", "הסטופ מתחת ל-EMA21", lambda d: lt(d.stop_g, d.ema21))
add("TG", "הסטופ מעל ממוצע 50 (יציאה לפני שבירת מגמה)", lambda d: gt(d.stop_g, d.sma50))
add("TG", "הסטופ מעל ממוצע 200", lambda d: gt(d.stop_g, d.sma200))
add("TG", "יש ממוצע נע (20 או 50) בין המחיר לסטופ", lambda d: OR(AND(lt(d.sma20, d.c), gt(d.sma20, d.stop_g)), AND(lt(d.sma50, d.c), gt(d.sma50, d.stop_g))))
add("TG", "הסיכון לא עולה על 2 ATR", lambda d: le(d.risk_atr, 2.0))
add("TG", "הסיכון לפחות 1.5 ATR (מרווח לרעש)", lambda d: ge(d.risk_atr, 1.5))
add("TG", "יעד 2R מתחת לשיא השנתי, או שהמחיר כבר בשיא", lambda d: OR(le(d.target_g, d.hh252), ge(d.dist52h, -0.02)))
add("TG", "מרחק של 1.5R לפחות עד שיא 55 יום, או שכבר מעליו", lambda d: ge(d.room55_R, 1.5))
add("TG", "המניה כבר עשתה תנועה בגודל 2R ב-15 יום ברבעון האחרון", lambda d: ge(d.best15_63, 2 * d.risk_g))
add("TG", "הגאפ הגרוע ב-60 יום קטן מהמרחק לסטופ", lambda d: gt(d.worst_gap60, -d.risk_g))
add("TG", "העמלות קטנות מרבע מהסיכון", lambda d: le(d.fee_share, 0.25))
add("TG", "אפשר לקנות לפחות 3 מניות", lambda d: ge(d.shares_g, 3))
add("TG", "היום הגרוע ב-60 יום קטן מפי 1.5 מהסיכון", lambda d: gt(d.worst_day60, -1.5 * d.risk_g))

# ------------------------------------------------------------------ RF · דגלים אדומים (25)
add("RF", "אין גאפ ירידה מעל 8% ב-20 יום", lambda d: gt(d.worst_gap20, -0.08))
add("RF", "אין יום ירידה מעל 10% ב-60 יום", lambda d: gt(d.worst_day60, -0.10))
add("RF", "לא מתוח: עד 20% מעל ממוצע 50", lambda d: le(d.c, d.sma50 * 1.20))
add("RF", "לא מתוח: עד 3 ATR מעל ממוצע 20", lambda d: le(d.c, d.sma20 + 3 * d.atr14))
add("RF", "RSI14 מתחת ל-80", lambda d: lt(d.rsi14, 80))
add("RF", "לא פרבולי: עלה פחות מ-30% ב-10 ימים", lambda d: lt(d.ret10, 0.30))
add("RF", "לא יותר מ-6 ימי עלייה רצופים", lambda d: le(d.up_streak, 6))
add("RF", "אין צלב מוות ב-40 הימים האחרונים", lambda d: no(d.death_cross40))
add("RF", "אין שפל 55 יום חדש ב-20 הימים האחרונים", lambda d: no(d.newll55_20))
add("RF", "אין סגירה מתחת לממוצע 50 בנפח כבד ב-10 ימים", lambda d: no(d.close_below50_hv10))
add("RF", "אין בולעני דובי בנפח גבוה ב-3 ימים", lambda d: no(d.engulf_bear_hv3))
add("RF", "אין פתיל עליון ארוך (מעל 60%) היום", lambda d: le(d.uw, 0.6))
add("RF", "ATR מתחת ל-8% מהמחיר", lambda d: lt(d.atrp, 0.08))
add("RF", "לא ירד יותר מ-15% משיא 20 יום", lambda d: gt(d.dd20, -0.15))
add("RF", "אין קלימקס מכירות (ירידה בנפח פי 4) ב-10 ימים", lambda d: no(d.climax_dn10))
add("RF", "לא מפגר אחרי SPY ביותר מ-10% בחודש", lambda d: gt(d.ex21, -0.10))
add("RF", "קו RS לא בשפל 55 יום", lambda d: gt(d.rs, d.rs_ll55))
add("RF", "אין סטייה דובית: שיא מחיר עם RSI חלש ב-10 נקודות", lambda d: no(AND(ge(d.c, d.hc20), lt(d.rsi14, d.rsi14_max20 - 10))))
add("RF", "פחות מ-3 ימי פיזור ב-5 הימים האחרונים", lambda d: lt(d.dist5, 3))
add("RF", "לא נסגר מתחת לממוצע 200 ב-20 יום", lambda d: no(d.closed_below200_20))
add("RF", "אין דוח רבעוני ב-5 ימי המסחר הקרובים", lambda d: gt(d.earn_days, 5))
add("RF", "שווי שוק מעל 2 מיליארד דולר", lambda d: gt(d.mcap, 2e9))
add("RF", "אין היפוך דובי בשיא בנפח גבוה ב-5 ימים", lambda d: no(d.bear_rev5))
add("RF", "לא מתוח לטווח ארוך: עד 70% מעל ממוצע 200", lambda d: le(d.c, d.sma200 * 1.70))
add("RF", "פחות מ-3 גאפים של 4%+ ב-60 יום", lambda d: lt(d.biggaps60, 3))

assert len(C) == 300, len(C)
assert len({c[0] for c in C}) == 300
IDS = [c[0] for c in C]
CAT_OF = [c[1] for c in C]


def catalog():
    return {
        "categories": [{"id": k, "name": n, "count": CAT_OF.count(k)} for k, n in CATEGORIES],
        "checks": [{"id": i, "cat": cat, "name": name} for i, cat, name, _ in C],
    }


def run_all(d, min_bars=60):
    """Returns a (days x 300) float32 matrix: 1 pass, 0 fail, NaN unknown."""
    out = np.empty((len(d.c), len(C)), dtype="float32")
    young = ~(d.nbars >= min_bars)
    for j, (_, _, _, fn) in enumerate(C):
        r = np.asarray(fn(d), dtype="float64")
        r[young] = np.nan
        out[:, j] = r
    return out


def score(mat):
    """Share of known checks that pass, 0-100, per day."""
    known = ~np.isnan(mat)
    k = known.sum(axis=1)
    p = np.nansum(mat, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(k > 0, p / k * 100, np.nan)
