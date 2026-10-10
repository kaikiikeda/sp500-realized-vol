"""Walk-forward forecasters for 21-day-ahead S&P 500 realized volatility.

Every model follows the same protocol, which is what makes the comparison fair:

1. Parameters are re-estimated on an expanding window at each refit date T
   (every 21 trading days). Only returns up to and including T are used. For the
   HAR regressions the training targets must also be fully observed by T, so the
   last usable training row is T - 21.
2. Between refits the parameters are frozen and each model is *filtered* forward
   one day at a time, so the forecast made at origin t uses data through t only.
3. Every forecast is the annualized vol (%) expected over t+1 .. t+21, the same
   units as the target ``rv21``.

Models
------
hist      past 21-day realized vol (random walk; the naive benchmark)
vix       VIX close at t, used as-is (the market's risk-neutral forecast)
garch     GARCH(1,1), Gaussian QML
gjr       GJR-GARCH(1,1,1): adds the leverage effect (bigger response to down days)
har       HAR on daily squared returns, direct 21-day-ahead regression (Corsi 2009)
har_vix   HAR plus the VIX-implied daily variance as an extra regressor
sv_kf     stochastic volatility, QML + Kalman filter (this project's main model)
sv_vix    bivariate state-space model: returns AND the VIX as two noisy measurements
          of the same hidden volatility (added after sv_kf's results; see README)
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model

from . import sv_kalman as sv
from .data import ANN, H, to_vol

REFIT_EVERY = 21
MIN_DAILY_VAR = 0.01  # floor for regression forecasts, %^2/day (about 1.6% annualized)


@dataclass
class Schedule:
    """Refit dates as integer positions into the panel."""

    refits: np.ndarray  # positions T at which models are re-estimated
    end: int  # last origin position (inclusive)

    def blocks(self):
        for i, T in enumerate(self.refits):
            stop = self.refits[i + 1] if i + 1 < len(self.refits) else self.end + 1
            yield T, stop  # origins T .. stop-1 use parameters estimated at T


def make_schedule(index: pd.DatetimeIndex, oos_start: str, end_pos: int) -> Schedule:
    first = int(index.searchsorted(pd.Timestamp(oos_start)))
    return Schedule(np.arange(first, end_pos + 1, REFIT_EVERY), end_pos)


# --------------------------------------------------------------------------- HAR
def har_features(r: pd.Series, vix: pd.Series | None = None) -> pd.DataFrame:
    r2 = r**2
    X = pd.DataFrame(
        {"d": r2, "w": r2.rolling(5).mean(), "m": r2.rolling(22).mean()}, index=r.index
    )
    if vix is not None:
        X["iv"] = vix**2 / ANN  # VIX-implied daily variance, %^2
    return X


def har_target(r: pd.Series) -> pd.Series:
    """Average daily variance over t+1 .. t+H (the HAR regression target)."""
    r2 = r**2
    return (r2[::-1].rolling(H).sum()[::-1].shift(-1) / H).rename("y")


def run_har(panel: pd.DataFrame, sched: Schedule, with_vix: bool) -> pd.Series:
    X = har_features(panel["r"], panel["vix"] if with_vix else None)
    X.insert(0, "const", 1.0)
    y = har_target(panel["r"])
    Xv, yv = X.to_numpy(), y.to_numpy()
    out = np.full(len(panel), np.nan)
    for T, stop in sched.blocks():
        rows = np.arange(0, T - H + 1)  # targets fully observed by T
        ok = rows[np.isfinite(Xv[rows]).all(1) & np.isfinite(yv[rows])]
        beta, *_ = np.linalg.lstsq(Xv[ok], yv[ok], rcond=None)
        pred = np.maximum(Xv[T:stop] @ beta, MIN_DAILY_VAR)
        out[T:stop] = to_vol(H * pred)
    return pd.Series(out, index=panel.index, name="har_vix" if with_vix else "har")


# ------------------------------------------------------------------------- GARCH
def run_garch(panel: pd.DataFrame, sched: Schedule, leverage: bool) -> pd.Series:
    r = panel["r"].to_numpy()
    out = np.full(len(panel), np.nan)
    for T, stop in sched.blocks():
        am = arch_model(r[:stop], mean="Constant", vol="GARCH", p=1, o=int(leverage), q=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = am.fit(last_obs=T + 1, disp="off")
        fc = res.forecast(start=T, horizon=H, reindex=False)
        out[T:stop] = to_vol(fc.variance.to_numpy().sum(axis=1))
    return pd.Series(out, index=panel.index, name="gjr" if leverage else "garch")


# ---------------------------------------------------------------- SV + Kalman
def run_sv(panel: pd.DataFrame, sched: Schedule, return_params: bool = False):
    r = panel["r"].to_numpy()
    out = np.full(len(panel), np.nan)
    rows = []
    for T, stop in sched.blocks():
        # Always start from the same point: the QML surface is flat in phi near 1 and
        # warm starts were found to stall slightly short of the optimum.
        res = sv.fit(r[: T + 1])
        p = sv.params_of(res)
        filt = sv.filter_with(r[:stop], res)  # frozen params, filtered to stop-1
        st, cv = sv.filtered_arrays(filt)
        out[T:stop] = to_vol(sv.forecast_cumvar(st[T:stop], cv[T:stop], p["phi"], p["sigma2_eta"], H))
        rows.append({"date": panel.index[T], **p, "mu": st[T, 0]})
    s = pd.Series(out, index=panel.index, name="sv_kf")
    return (s, pd.DataFrame(rows).set_index("date")) if return_params else s


def run_sv_vix(panel: pd.DataFrame, sched: Schedule, verbose: bool = False) -> pd.Series:
    r, v = panel["r"].to_numpy(), panel["vix"].to_numpy()
    out = np.full(len(panel), np.nan)
    for i, (T, stop) in enumerate(sched.blocks()):
        res = sv.fit_sv_vix(r[: T + 1], v[: T + 1])
        mu, _, _, phi, s2e, _, _ = res.params
        filt = sv.filter_sv_vix(r[:stop], v[:stop], res)
        g = filt.filtered_state[0, T:stop]
        Pgg = filt.filtered_state_cov[0, 0, T:stop]
        out[T:stop] = to_vol(sv.forecast_cumvar_sv_vix(g, Pgg, mu, phi, s2e, H))
        if verbose and i % 50 == 0:
            print(f"    sv_vix refit {i}/{len(sched.refits)}", flush=True)
    return pd.Series(out, index=panel.index, name="sv_vix")


def sv_lookahead_variants(panel: pd.DataFrame) -> pd.DataFrame:
    """Two deliberately *wrong* SV forecasts, to measure what look-ahead is worth.

    sv_fullparams : parameters estimated on the whole sample, states filtered (causal)
    sv_smoothed   : whole-sample parameters AND RTS-smoothed states (non-causal)

    The second one is the same mistake as comparing forecasts against
    full-sample smoothed components in the car-sales project.
    """
    r = panel["r"].to_numpy()
    res = sv.fit(r)
    p = sv.params_of(res)
    out = {}
    for name, smoothed in [("sv_fullparams", False), ("sv_smoothed", True)]:
        st, cv = sv.filtered_arrays(res, smoothed=smoothed)
        out[name] = to_vol(sv.forecast_cumvar(st, cv, p["phi"], p["sigma2_eta"], H))
    return pd.DataFrame(out, index=panel.index)


# ---------------------------------------------------------------------- driver
def run_all(panel: pd.DataFrame, sched: Schedule, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    sl = slice(sched.refits[0], sched.end + 1)
    fc = pd.DataFrame(index=panel.index)
    fc["hist"] = panel["rv21_past"]
    fc["vix"] = panel["vix"]
    for name, fn in [
        ("garch", lambda: run_garch(panel, sched, leverage=False)),
        ("gjr", lambda: run_garch(panel, sched, leverage=True)),
        ("har", lambda: run_har(panel, sched, with_vix=False)),
        ("har_vix", lambda: run_har(panel, sched, with_vix=True)),
    ]:
        if verbose:
            print(f"  fitting {name} ...", flush=True)
        fc[name] = fn()
    if verbose:
        print("  fitting sv_kf ...", flush=True)
    fc["sv_kf"], sv_params = run_sv(panel, sched, return_params=True)
    if verbose:
        print("  fitting sv_vix (slowest; ~15 min) ...", flush=True)
    fc["sv_vix"] = run_sv_vix(panel, sched, verbose)
    return fc.iloc[sl], sv_params
