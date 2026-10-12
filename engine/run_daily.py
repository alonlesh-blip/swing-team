"""Daily entry point.  python -m engine.run_daily [--synthetic] [--no-backtest] [--out DIR]"""
import argparse
import sys
import time

from . import data, pipeline
from .indicators import INDEX_TICKERS, SECTOR_ETF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true", help="fake data, no network (for tests)")
    ap.add_argument("--no-backtest", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = pipeline.load_config()
    t0 = time.time()
    if a.synthetic:
        px, uni = data.synthetic()
        ext_fn, failed = None, []
    else:
        uni = data.load_universe(refresh=True)
        want = list(uni) + INDEX_TICKERS + list(SECTOR_ETF.values())
        print(f"downloading {len(want)} tickers")
        px, failed = data.download(want, period=cfg["history_period"])
        print(f"loaded {len(px)}, failed {len(failed)}: {failed[:30]}")
        if len(px) < 0.8 * len(want):
            print("too many download failures, not publishing")
            sys.exit(2)
        ext_fn = data.externals
    out = pipeline.run(px, uni, cfg, do_backtest=not a.no_backtest, ext_fn=ext_fn)
    out["latest"]["universe"]["failed_downloads"] = failed[:50]
    pipeline.write(out, a.out)
    L = out["latest"]
    print(f"asof {L['asof']} regime {L['regime']['label']} signals {L['signals_today']} "
          f"table {len(L['table'])} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
