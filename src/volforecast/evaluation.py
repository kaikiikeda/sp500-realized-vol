"""Forecast evaluation: loss functions, Diebold-Mariano tests, Mincer-Zarnowitz.

Two losses are reported because they answer different questions:

* RMSE on volatility (vol points): easy to read, but dominated by crisis months.
* QLIKE on variance, L = RV^2/F^2 - log(RV^2/F^2) - 1. Patton (2011) shows QLIKE and
  MSE are the losses that rank forecasts correctly when the target (realized vol
  from daily returns) is itself a noisy proxy for true volatility. QLIKE punishes
  under-prediction more than over-prediction, which is the right asymmetry for
  someone who is short volatility.

Forecasts are made every day for a 21-day window, so consecutive errors overlap
and are autocorrelated by construction. The Diebold-Mariano statistic therefore
uses a Newey-West (Bartlett) long-run variance with H-1 = 20 lags.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .data import H


def qlike(rv: np.ndarray, f: np.ndarray) -> np.ndarray:
    ratio = (rv / f) ** 2
    return ratio - np.log(ratio) - 1.0


def sq_err(rv: np.ndarray, f: np.ndarray) -> np.ndarray:
    return (rv - f) ** 2


def newey_west_var(x: np.ndarray, lags: int) -> float:
    x = x - x.mean()
    n = len(x)
    v = x @ x / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        v += 2 * w * (x[k:] @ x[:-k]) / n
    return v


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, lags: int = H - 1) -> tuple[float, float]:
    """DM statistic and two-sided p-value for H0: E[loss_a - loss_b] = 0.

    Negative statistic => model a has lower loss than model b.
    """
    d = loss_a - loss_b
    stat = d.mean() / np.sqrt(newey_west_var(d, lags) / len(d))
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def mincer_zarnowitz(rv: np.ndarray, f: np.ndarray) -> dict:
    """OLS of realized on forecast. Unbiased & efficient => a=0, b=1."""
    X = np.column_stack([np.ones_like(f), f])
    beta, *_ = np.linalg.lstsq(X, rv, rcond=None)
    resid = rv - X @ beta
    r2 = 1 - resid.var() / rv.var()
    return {"mz_a": beta[0], "mz_b": beta[1], "mz_r2": r2}


def loss_table(fc: pd.DataFrame, rv: pd.Series, bench: str = "har") -> pd.DataFrame:
    """One row per model: RMSE, MAE, QLIKE, bias, MZ stats and DM tests vs ``bench``."""
    rows = []
    rvv = rv.to_numpy()
    q_b, s_b = qlike(rvv, fc[bench].to_numpy()), sq_err(rvv, fc[bench].to_numpy())
    for m in fc.columns:
        f = fc[m].to_numpy()
        q, s = qlike(rvv, f), sq_err(rvv, f)
        row = {
            "model": m,
            "rmse": np.sqrt(s.mean()),
            "mae": np.abs(rvv - f).mean(),
            "qlike": q.mean(),
            "bias": (f - rvv).mean(),
            **mincer_zarnowitz(rvv, f),
        }
        if m != bench:
            row["dm_qlike_vs_" + bench], row["p_qlike"] = dm_test(q, q_b)
            row["dm_mse_vs_" + bench], row["p_mse"] = dm_test(s, s_b)
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


PERIODS = {
    "2005-07 pre-crisis": ("2005-01-01", "2007-06-30"),
    "2007-09 GFC": ("2007-07-01", "2009-06-30"),
    "2009-19 post-GFC": ("2009-07-01", "2019-12-31"),
    "2020 COVID": ("2020-01-01", "2020-12-31"),
    "2021-26 recent": ("2021-01-01", "2026-12-31"),
}


def period_table(fc: pd.DataFrame, rv: pd.Series, loss: str = "qlike") -> pd.DataFrame:
    fn = qlike if loss == "qlike" else sq_err
    out = {}
    for name, (a, b) in PERIODS.items():
        m = (fc.index >= a) & (fc.index <= b)
        if m.sum() == 0:
            continue
        L = {c: fn(rv[m].to_numpy(), fc.loc[m, c].to_numpy()).mean() for c in fc.columns}
        if loss != "qlike":
            L = {k: np.sqrt(v) for k, v in L.items()}
        out[f"{name} (n={m.sum()})"] = L
    return pd.DataFrame(out).T
