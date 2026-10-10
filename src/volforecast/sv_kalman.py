"""Stochastic volatility as a linear state-space model, estimated by QML + Kalman filter.

Model (Harvey, Ruiz & Shephard 1994)
------------------------------------
    r_t   = exp(h_t / 2) * eps_t,            eps_t ~ N(0, 1)
    h_t   = mu + phi * (h_{t-1} - mu) + sigma_eta * eta_t

Squaring and taking logs makes the observation equation linear in the hidden state:

    y_t := log(r_t^2) + 1.2704 = h_t + xi_t,
    xi_t = log(eps_t^2) + 1.2704,   E[xi_t] = 0,  Var[xi_t] = pi^2 / 2.

xi_t is log-chi-squared, not Gaussian, so the Kalman filter is the best *linear*
filter and the Gaussian likelihood is a quasi-likelihood (QML). Days with an exactly
zero return (3 in 27 years) would give log(0); they are passed to the filter as
missing observations, which the Kalman filter handles exactly (predict, no update).

Why not the usual small offset log(r^2 + kappa)? Because I tried it and it is not
harmless. The offset compresses the long left tail of log(eps^2), so the noise
variance is no longer pi^2/2, while the model still assumes it is. On simulated data
with true sigma_eta^2 = 0.0225, kappa = 0.005 gave an estimate of 0.008, a 3x
understatement of how much volatility moves (tests/test_no_lookahead.py checks
parameter recovery so this cannot creep back in).

Implementation
--------------
The model is exactly a statsmodels ``UnobservedComponents`` model with a fixed
intercept (mu), an AR(1) component (h_t - mu) and an irregular term (xi_t) whose
variance is *fixed* at pi^2/2 rather than estimated. That is the same class I used
for the car-sales structural time series work, which is the point: the machinery
transfers unchanged; only the observation equation is new.

Forecasting the next-H-day variance
-----------------------------------
Given the filtered state at t (mean a, covariance P over [mu, ar]), the k-step-ahead
log-variance is Gaussian with

    m_k = mu_t + phi^k * ar_t
    v_k = [1, phi^k] P [1, phi^k]' + sigma_eta^2 (1 - phi^{2k}) / (1 - phi^2)

and E[r_{t+k}^2] = E[exp(h_{t+k})] = exp(m_k + v_k / 2). Summing k = 1..H gives the
forecast of cumulative variance, which ``data.to_vol`` annualizes.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents

LOG_CHI2_MEAN = -1.2704  # E[log eps^2] for eps ~ N(0,1)
LOG_CHI2_VAR = np.pi**2 / 2  # Var[log eps^2]


def transform(r: pd.Series | np.ndarray) -> np.ndarray:
    """y_t = log(r_t^2) + 1.2704, with exact zeros mapped to NaN (missing)."""
    r2 = np.asarray(r, dtype=float) ** 2
    with np.errstate(divide="ignore"):
        y = np.log(r2) - LOG_CHI2_MEAN
    y[r2 == 0] = np.nan
    return y


def _model(y: np.ndarray) -> UnobservedComponents:
    return UnobservedComponents(
        y, level=True, stochastic_level=False, irregular=True, autoregressive=1
    )


def fit(r_train: np.ndarray, start_params: np.ndarray | None = None):
    """QML fit on training returns. Returns statsmodels results.

    start_params are the FREE parameters [sigma2.ar, ar.L1]. statsmodels' default
    start values can land in a degenerate optimum (sigma2.ar -> 0, a constant-vol
    model), so we always start from the same persistent, plausible point.
    """
    if start_params is None:
        start_params = np.array([0.02, 0.97])
    mod = _model(transform(r_train))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with mod.fix_params({"sigma2.irregular": LOG_CHI2_VAR}):
            res = mod.fit(start_params=start_params, disp=False, maxiter=500)
    return res


def params_of(res) -> dict:
    p = dict(zip(res.model.param_names, res.params))
    return {"phi": p["ar.L1"], "sigma2_eta": p["sigma2.ar"]}


def filter_with(r_all: np.ndarray, res):
    """Run the filter over r_all with parameters frozen at ``res`` (no re-estimation)."""
    mod = _model(transform(r_all))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mod.filter(res.params)


def smooth_with(r_all: np.ndarray, res):
    """Filter AND run the backward RTS smoother with frozen parameters (non-causal)."""
    mod = _model(transform(r_all))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mod.smooth(res.params)


def forecast_cumvar(states: np.ndarray, covs: np.ndarray, phi: float, s2: float, H: int) -> np.ndarray:
    """Vectorized H-step cumulative variance forecasts.

    states: (n, 2) filtered [mu, ar] means;  covs: (n, 2, 2) filtered covariances.
    Returns (n,) array of sum_{k=1..H} E[r_{t+k}^2] in %^2.
    """
    k = np.arange(1, H + 1)
    pk = phi**k  # (H,)
    m = states[:, [0]] + states[:, [1]] * pk  # (n, H)
    p00, p01, p11 = covs[:, 0, 0][:, None], covs[:, 0, 1][:, None], covs[:, 1, 1][:, None]
    v_state = p00 + 2 * pk * p01 + pk**2 * p11
    v_shock = s2 * (1 - pk**2) / (1 - phi**2)
    return np.exp(m + 0.5 * (v_state + v_shock)).sum(axis=1)


def filtered_arrays(res, smoothed: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(n,2) state means and (n,2,2) covariances; filtered by default.

    smoothed=True returns the RTS-smoothed states, which use the WHOLE sample.
    They are only used to demonstrate how much look-ahead flatters a backtest.
    """
    if smoothed:
        return res.smoothed_state.T, np.moveaxis(res.smoothed_state_cov, -1, 0)
    return res.filtered_state.T, np.moveaxis(res.filtered_state_cov, -1, 0)


def latent_vol(res, smoothed: bool = False) -> np.ndarray:
    """Annualized spot vol implied by the latent state, exp(h/2) * sqrt(252), in %."""
    st, cv = filtered_arrays(res, smoothed)
    h = st[:, 0] + st[:, 1]
    v = cv[:, 0, 0] + 2 * cv[:, 0, 1] + cv[:, 1, 1]
    return np.sqrt(252 * np.exp(h + 0.5 * v))


# =============================================================================
# Extension: two noisy measurements of one hidden volatility (returns + VIX)
# =============================================================================
from statsmodels.tsa.statespace.mlemodel import MLEModel  # noqa: E402

AR_CAP = 0.9995


class SVWithVIX(MLEModel):
    """Bivariate state-space model: returns and the VIX both measure h_t.

    States  x_t = [g_t, u_t]
        g_t = h_t - mu       = phi * g_{t-1} + eta_t,   Var eta = s2_eta
        u_t (VIX premium)    = rho * u_{t-1} + zeta_t,  Var zeta = s2_u
    Observations
        y1_t = log(r_t^2) + 1.2704         = mu + g_t + xi_t,       Var xi = pi^2/2 (fixed)
        y2_t = log(VIX_t^2 / 252)          = d + b * g_t + u_t

    The VIX is a forecast of variance under the risk-neutral measure, so it carries
    a premium u_t over the physical variance. That premium is persistent (it moves
    with fear, not day to day), so it is modeled as its own AR(1) state instead of
    white noise. If u_t were treated as white noise the filter would trust the VIX
    almost completely; with a persistent u_t it has to learn how much of a VIX move
    is volatility and how much is a change in the premium.
    """

    param_names_ = ["mu", "d", "b", "phi", "s2_eta", "rho", "s2_u"]

    def __init__(self, y1: np.ndarray, y2: np.ndarray):
        endog = np.column_stack([y1, y2])
        super().__init__(endog, k_states=2, k_posdef=2, initialization="stationary")
        self["selection"] = np.eye(2)
        self["transition"] = np.eye(2)
        self["obs_cov"] = np.diag([LOG_CHI2_VAR, 1e-6])

    @property
    def param_names(self):
        return self.param_names_

    @property
    def start_params(self):
        y1, y2 = self.endog[:, 0], self.endog[:, 1]
        return np.array([np.nanmean(y1), np.nanmean(y2), 1.0, 0.98, 0.02, 0.98, 0.005])

    def transform_params(self, u):
        p = np.array(u, dtype=float)
        # cap persistence just below 1 so the stationary initialization always exists
        p[3], p[5] = AR_CAP * np.tanh(u[3]), AR_CAP * np.tanh(u[5])
        p[4], p[6] = np.exp(u[4]), np.exp(u[6])
        return p

    def untransform_params(self, p):
        u = np.array(p, dtype=float)
        u[3], u[5] = np.arctanh(p[3] / AR_CAP), np.arctanh(p[5] / AR_CAP)
        u[4], u[6] = np.log(p[4]), np.log(p[6])
        return u

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        mu, d, b, phi, s2e, rho, s2u = params
        self["obs_intercept"] = np.array([[mu], [d]])
        self["design"] = np.array([[1.0, 0.0], [b, 1.0]])
        self["transition"] = np.array([[phi, 0.0], [0.0, rho]])
        self["state_cov"] = np.diag([s2e, s2u])


def vix_obs(vix) -> np.ndarray:
    return np.log(np.asarray(vix, dtype=float) ** 2 / 252)


def fit_sv_vix(r_train, vix_train):
    mod = SVWithVIX(transform(r_train), vix_obs(vix_train))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mod.fit(disp=False, maxiter=1000)


def filter_sv_vix(r_all, vix_all, res):
    mod = SVWithVIX(transform(r_all), vix_obs(vix_all))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return mod.filter(res.params)


def forecast_cumvar_sv_vix(g: np.ndarray, Pgg: np.ndarray, mu: float, phi: float, s2: float, H: int) -> np.ndarray:
    """Same formula as forecast_cumvar, with mu as a fixed parameter."""
    k = np.arange(1, H + 1)
    pk = phi**k
    m = mu + g[:, None] * pk
    v = Pgg[:, None] * pk**2 + s2 * (1 - pk**2) / (1 - phi**2)
    return np.exp(m + 0.5 * v).sum(axis=1)
