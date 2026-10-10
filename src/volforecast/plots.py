"""Static figures for the README (matplotlib, light theme)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.labelcolor": INK2, "font.size": 10, "lines.linewidth": 1.4,
})

LABELS = {"hist": "Past 21d RV", "vix": "VIX", "garch": "GARCH(1,1)", "gjr": "GJR-GARCH",
          "har": "HAR", "har_vix": "HAR + VIX", "sv_kf": "SV (Kalman)", "sv_vix": "SV + VIX (Kalman)"}


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_latent_state(panel: pd.DataFrame, filt_vol: np.ndarray, smooth_vol: np.ndarray, path: Path):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    k = 21  # skip the filter's burn-in (diffuse start) so the first days don't dominate
    ix = panel.index[k:]
    ax.plot(ix, panel["rv21_past"].iloc[k:], color=MUTED, lw=1.0, label="Past 21d realized vol")
    ax.plot(ix, filt_vol[k:], color=BLUE, lw=1.0, label="SV filtered (uses data up to t)")
    ax.plot(ix, smooth_vol[k:], color=ORANGE, lw=1.0, label="SV smoothed (uses the whole sample)")
    ax.set_yscale("log")
    ax.set_yticks([5, 10, 20, 40, 80])
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylabel("annualized vol, % (log scale)")
    ax.set_ylim(3, 130)
    ax.set_title("Volatility as a hidden state: S&P 500, 1999-2026")
    ax.legend(loc="upper left", ncols=3)
    _save(fig, path)


def fig_forecasts(fc: pd.DataFrame, rv: pd.Series, path: Path, start="2019-06-01", end="2023-06-30"):
    m = (fc.index >= start) & (fc.index <= end)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rv.index[m], rv[m], color=INK, lw=1.6, label="Realized vol, next 21 days (target)")
    for c, col in [("vix", ORANGE), ("gjr", VIOLET), ("sv_kf", BLUE), ("sv_vix", AQUA)]:
        ax.plot(fc.index[m], fc.loc[m, c], color=col, lw=1.2, label=LABELS[c] + " forecast")
    ax.set_ylabel("annualized vol, %")
    ax.set_title("Forecasts vs what happened, 2019-2023 (each point is a forecast made that day)")
    ax.legend(loc="upper right")
    _save(fig, path)


def fig_vrp(fc: pd.DataFrame, rv: pd.Series, path: Path):
    gap = (fc["vix"] - rv).rolling(63).mean()
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.fill_between(gap.index, 0, gap, where=gap >= 0, color=BLUE, alpha=0.35, lw=0)
    ax.fill_between(gap.index, 0, gap, where=gap < 0, color=RED, alpha=0.35, lw=0)
    ax.plot(gap.index, gap, color=INK2, lw=0.8)
    ax.set_ylabel("vol points")
    ax.set_title("VIX minus the realized vol that followed (3-month average)")
    ax.text(0.01, 0.92, "above 0: implied was too high (option sellers won)", transform=ax.transAxes, color=INK2)
    ax.text(0.01, 0.06, "below 0: realized beat implied (option buyers won)", transform=ax.transAxes, color=INK2)
    _save(fig, path)


def fig_cum_loss(fc: pd.DataFrame, rv: pd.Series, qlike, path: Path, bench="har"):
    fig, ax = plt.subplots(figsize=(10, 4))
    base = qlike(rv.to_numpy(), fc[bench].to_numpy())
    for c, col in [("vix", ORANGE), ("gjr", VIOLET), ("sv_kf", BLUE), ("sv_vix", AQUA)]:
        d = np.cumsum(qlike(rv.to_numpy(), fc[c].to_numpy()) - base)
        ax.plot(fc.index, d, color=col, label=LABELS[c])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel(f"cumulative QLIKE minus {LABELS[bench]}")
    ax.set_title(f"Who beats HAR, and when? Below zero = better than {LABELS[bench]} so far")
    ax.legend(loc="upper left")
    _save(fig, path)


def fig_equity(curves: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    for c, col in zip(curves.columns, [MUTED, BLUE, ORANGE]):
        ax.plot(curves.index, curves[c], color=col, label=c)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("cumulative P&L, vol points\n(1/21 vega notional opened daily)")
    ax.set_title("Synthetic 21-day variance swap struck at the VIX, no costs")
    ax.legend(loc="upper left")
    _save(fig, path)


def fig_lookahead(tab: pd.DataFrame, path: Path):
    order = ["sv_kf", "sv_fullparams", "sv_smoothed"]
    names = ["Walk-forward\n(honest)", "Full-sample\nparameters", "Full-sample params\n+ smoothed states"]
    vals = tab.loc[order, "qlike"].to_numpy()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(names, vals, color=[BLUE, YELLOW, RED], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", color=INK)
    ax.set_ylabel("out-of-sample QLIKE (lower = better)")
    ax.set_title("Same model, three backtests: what look-ahead is worth")
    ax.grid(axis="x", visible=False)
    _save(fig, path)
