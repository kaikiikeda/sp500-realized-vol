"""A stylized implied-vs-realized trade driven by the forecast-VIX gap.

There is no free history of S&P 500 option prices, so the trade is a *synthetic
21-day variance swap struck at the VIX*. Per unit of vega notional, the payoff of a
short position opened at t is

    pnl_t = (K_t^2 - RV_{t,t+21}^2) / (2 K_t),      K_t = VIX_t,

which is positive when realized vol comes in below implied. It is quoted in vol
points. This is a simplification and the write-up says so plainly:

* The VIX is close to, but not exactly, the square root of the 30-day variance
  swap rate, and it covers 30 calendar days rather than exactly 21 trading days.
* There are no bid-ask spreads, margin, or convexity adjustments; a flat cost per
  trade is applied as a sensitivity.
* Positions opened every day overlap. Summary statistics that need independence
  (Sharpe, drawdown) use non-overlapping monthly samples, averaged over all 21
  possible starting offsets so the result does not depend on an arbitrary start day.

The rule was fixed before looking at results and uses the project's own model
(sv_kf) as the signal. Results for other models are shown for comparison only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import H


def swap_pnl(vix: pd.Series, rv: pd.Series) -> pd.Series:
    """Short-variance-swap payoff per unit vega notional, in vol points."""
    return (vix**2 - rv**2) / (2 * vix)


def positions(forecast: pd.Series, vix: pd.Series, rule: str, theta: float) -> pd.Series:
    """+1 = short vol, -1 = long vol, 0 = flat."""
    gap = forecast - vix  # negative => implied looks expensive
    if rule == "always_short":
        return pd.Series(1.0, index=vix.index)
    if rule == "short_when_rich":
        return (gap < -theta).astype(float)
    if rule == "long_short":
        return (gap < -theta).astype(float) - (gap > theta).astype(float)
    raise ValueError(rule)


def summarize(pos: pd.Series, pnl_short: pd.Series, cost: float = 0.0) -> dict:
    trade_pnl = pos * pnl_short - cost * pos.abs()
    active = pos != 0
    # Non-overlapping monthly series for each of the 21 offsets
    sharpes, dds, means = [], [], []
    for off in range(H):
        x = trade_pnl.iloc[off::H]
        means.append(x.mean())
        sharpes.append(x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else np.nan)
        eq = x.cumsum()
        dds.append((eq - eq.cummax()).min())
    return {
        "pct_months_active": active.mean() * 100,
        "mean_pnl_per_month": np.mean(means),
        "mean_pnl_when_active": trade_pnl[active].mean() if active.any() else np.nan,
        "hit_rate_when_active": (trade_pnl[active] > 0).mean() * 100 if active.any() else np.nan,
        "worst_trade": trade_pnl.min(),
        "sharpe_ann": np.nanmean(sharpes),
        "max_drawdown": np.mean(dds),
    }


def run_backtest(fc: pd.DataFrame, rv: pd.Series, models=("sv_kf", "sv_vix", "gjr", "har_vix"),
                 thetas=(0.0, 2.0, 4.0), costs=(0.0, 0.5)) -> pd.DataFrame:
    vix = fc["vix"]
    pnl = swap_pnl(vix, rv)
    rows = []
    for cost in costs:
        rows.append({"signal": "-", "rule": "always_short", "theta": np.nan, "cost": cost,
                     **summarize(positions(vix, vix, "always_short", 0), pnl, cost)})
        for m in models:
            for rule in ("short_when_rich", "long_short"):
                for th in thetas:
                    rows.append({"signal": m, "rule": rule, "theta": th, "cost": cost,
                                 **summarize(positions(fc[m], vix, rule, th), pnl, cost)})
    return pd.DataFrame(rows)


def equity_curves(fc: pd.DataFrame, rv: pd.Series, signal: str = "sv_kf", theta: float = 2.0) -> pd.DataFrame:
    """Cumulative P&L of laddered positions (1/21 notional opened each day), by close date."""
    vix = fc["vix"]
    pnl = swap_pnl(vix, rv)
    out = {}
    for label, rule in [("always short", "always_short"),
                        (f"short only when {signal} < VIX - {theta:g}", "short_when_rich"),
                        (f"long/short on {signal} gap (±{theta:g})", "long_short")]:
        pos = positions(fc[signal], vix, rule, theta)
        # booked when the swap settles, 21 trading days after it was opened
        out[label] = (pos * pnl / H).shift(H).fillna(0.0).cumsum()
    return pd.DataFrame(out)
