"""Data loading, splicing and target construction.

Conventions used everywhere in this project
-------------------------------------------
* ``r``       daily log return of the S&P 500 in percent: 100 * ln(P_t / P_{t-1}).
* ``H = 21``  forecast horizon in trading days (about one calendar month, matching
              the 30-calendar-day window of the VIX).
* ``rv21``    realized volatility over the NEXT 21 trading days, annualized, in
              percent: sqrt(252/21 * sum_{k=1..21} r_{t+k}^2). It is indexed at the
              forecast origin t, so ``rv21[t]`` is the quantity a forecast made at
              the close of day t is trying to hit. It is unknown at time t.
* ``rv21_past`` the same statistic over the PAST 21 days (t-20..t). Known at t.

All forecasts are expressed in the same units as ``rv21`` (annualized vol, %).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

H = 21
ANN = 252

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

# The 1999-2018 history ships with the `arch` package; FRED only serves the most
# recent ten years of SP500. The two overlap for 2016-10 .. 2018-12 and agree to
# within rounding (max abs diff 0.04 index points), which build_data.py checks.
SPLICE_DATE = pd.Timestamp("2016-10-10")


def load_raw() -> tuple[pd.Series, pd.Series]:
    """Return (spx_close, vix_close) as date-indexed Series from data/raw."""
    old = pd.read_csv(RAW / "sp500_arch_1999_2018.csv", index_col=0, parse_dates=True)["close"]
    new = pd.read_csv(RAW / "sp500_fred_2016_2026.csv", index_col=0, parse_dates=True)["close"]
    spx = pd.concat([old[old.index < SPLICE_DATE], new[new.index >= SPLICE_DATE]]).sort_index()
    spx = spx[~spx.index.duplicated()].rename("spx")

    vix = pd.read_csv(RAW / "vix_cboe.csv", index_col=0, parse_dates=True)["CLOSE"].rename("vix")
    return spx, vix


def build_panel(spx: pd.Series, vix: pd.Series) -> pd.DataFrame:
    """Daily panel on S&P 500 trading days with returns, VIX and RV targets."""
    # Returns are computed on the full S&P calendar first, so a missing VIX print
    # can never silently turn into a two-day return.
    r = (100.0 * np.log(spx).diff()).rename("r")
    df = pd.concat([spx, r, vix], axis=1, join="inner").dropna()
    r2 = df["r"] ** 2

    # Past-window statistics (known at t)
    df["rv21_past"] = np.sqrt(ANN / H * r2.rolling(H).sum())

    # Forward target (unknown at t): sum of r^2 over t+1 .. t+H
    fwd_sum = r2[::-1].rolling(H).sum()[::-1].shift(-1)
    df["rv21"] = np.sqrt(ANN / H * fwd_sum)
    df["var21"] = ANN / H * fwd_sum  # annualized variance, %^2
    return df


def load_panel() -> pd.DataFrame:
    path = PROCESSED / "panel.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    spx, vix = load_raw()
    return build_panel(spx, vix)


def to_vol(daily_var_sum: np.ndarray | pd.Series) -> np.ndarray | pd.Series:
    """Convert a forecast of sum_{k=1..H} E[r_{t+k}^2] (in %^2) to annualized vol %."""
    return np.sqrt(ANN / H * daily_var_sum)
