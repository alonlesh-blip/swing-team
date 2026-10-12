"""One-off: save the raw daily bars so research can run offline.

    python -m engine.dump_data bars.npz
"""
import sys

import numpy as np

from . import data, pipeline
from .indicators import INDEX_TICKERS, SECTOR_ETF


def main(path):
    cfg = pipeline.load_config()
    uni = data.load_universe(refresh=True)
    want = list(uni) + INDEX_TICKERS + list(SECTOR_ETF.values())
    px, failed = data.download(want, period=cfg["history_period"])
    print(f"loaded {len(px)}, failed {failed}")
    px, idx = data.align(px)
    tickers = [t for t in want if t in px]
    stack = lambda col: np.column_stack([px[t][col].to_numpy(dtype="float32") for t in tickers])
    np.savez_compressed(
        path, dates=np.array([d.strftime("%Y-%m-%d") for d in idx]), tickers=np.array(tickers),
        names=np.array([uni.get(t, {}).get("name", t) for t in tickers]),
        sectors=np.array([uni.get(t, {}).get("sector") or "" for t in tickers]),
        in_universe=np.array([t in uni for t in tickers]),
        o=stack("open"), h=stack("high"), l=stack("low"), c=stack("close"), v=stack("volume"))
    print("saved", path, len(idx), "days", len(tickers), "tickers")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bars.npz")
