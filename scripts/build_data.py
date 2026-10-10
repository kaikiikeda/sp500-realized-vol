"""Download the raw data and build data/processed/panel.csv.

Sources (all free, no API key):
  * S&P 500 close 1999-01 .. 2018-12 : bundled with the `arch` Python package
  * S&P 500 close last ~10 years     : FRED series SP500
  * VIX close 1990 .. present        : Cboe history, mirrored at github.com/datasets/finance-vix

The raw S&P 500 and VIX files are not committed to this repo (index data is
licensed by S&P Dow Jones Indices and Cboe). Run this script to fetch them.

Usage:  python scripts/build_data.py [--offline]
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from volforecast.data import PROCESSED, RAW, SPLICE_DATE, build_panel, load_raw  # noqa: E402

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500"
VIX_URL = "https://raw.githubusercontent.com/datasets/finance-vix/main/data/vix-daily.csv"


def fetch(url: str, dest: Path) -> None:
    print(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())


def main(offline: bool) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    old_path = RAW / "sp500_arch_1999_2018.csv"
    if not old_path.exists():
        import arch.data.sp500

        s = arch.data.sp500.load()["Close"].rename("close")
        s.index.name = "date"
        s.to_csv(old_path)

    new_path = RAW / "sp500_fred_2016_2026.csv"
    vix_path = RAW / "vix_cboe.csv"
    if not offline:
        tmp = RAW / "_fred_tmp.csv"
        fetch(FRED_URL, tmp)
        f = pd.read_csv(tmp, index_col=0, parse_dates=True, na_values=".").iloc[:, 0].dropna()
        f.rename("close").rename_axis("date").to_csv(new_path)
        tmp.unlink()
        fetch(VIX_URL, vix_path)

    for p in (old_path, new_path, vix_path):
        if not p.exists():
            raise SystemExit(f"missing {p}; run without --offline")

    # Splice check: the two S&P sources must agree on their overlap.
    old = pd.read_csv(old_path, index_col=0, parse_dates=True)["close"]
    new = pd.read_csv(new_path, index_col=0, parse_dates=True)["close"]
    ov = pd.concat([old, new], axis=1, keys=["arch", "fred"]).dropna()
    max_diff = (ov["arch"] - ov["fred"]).abs().max()
    print(f"splice overlap: {len(ov)} days, max |diff| = {max_diff:.3f} index points")
    assert max_diff < 0.5, "S&P sources disagree on the overlap"

    spx, vix = load_raw()
    panel = build_panel(spx, vix)
    panel.to_csv(PROCESSED / "panel.csv")
    print(
        f"panel: {panel.index[0].date()} .. {panel.index[-1].date()}, {len(panel)} days "
        f"(splice at {SPLICE_DATE.date()}); last date with a full 21-day target: "
        f"{panel['rv21'].last_valid_index().date()}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="use files already in data/raw")
    main(ap.parse_args().offline)
