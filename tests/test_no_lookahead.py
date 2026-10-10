"""Look-ahead tests.

The defining property of an honest forecast: changing the future must not change
the forecast. Each test corrupts every return (and VIX print) after a cutoff date
and checks that forecasts made on or before that date are bit-for-bit unchanged.

This is the test I wish I had written before the car-sales bug where forecasts
were being compared against full-sample smoothed components.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from volforecast import models as M  # noqa: E402
from volforecast import sv_kalman as sv  # noqa: E402
from volforecast.data import build_panel  # noqa: E402


def synthetic_panel(n=1600, seed=0) -> pd.DataFrame:
    """Simulated SV returns and a VIX-like series, so tests need no downloads."""
    rng = np.random.default_rng(seed)
    h = np.empty(n)
    h[0] = 0.0
    for t in range(1, n):
        h[t] = 0.98 * h[t - 1] + 0.15 * rng.standard_normal()
    r = np.exp(h / 2) * rng.standard_normal(n)
    idx = pd.bdate_range("2000-01-03", periods=n)
    spx = pd.Series(1000 * np.exp(np.cumsum(r) / 100), index=idx, name="spx")
    vix = pd.Series(np.sqrt(252 * np.exp(h)) * 1.15 + rng.normal(0, 0.5, n), index=idx, name="vix")
    return build_panel(spx, vix.clip(lower=5))


def corrupt_after(panel: pd.DataFrame, pos: int) -> pd.DataFrame:
    p = panel.copy()
    rng = np.random.default_rng(99)
    p.iloc[pos + 1:, p.columns.get_loc("r")] = rng.standard_normal(len(p) - pos - 1) * 5
    p.iloc[pos + 1:, p.columns.get_loc("vix")] = 80.0
    return p


@pytest.fixture(scope="module")
def setup():
    panel = synthetic_panel()
    sched = M.make_schedule(panel.index, panel.index[1000].strftime("%Y-%m-%d"), 1300)
    cutoff = 1150  # inside the evaluation window, mid-block
    return panel, sched, cutoff


@pytest.mark.parametrize("name", ["garch", "gjr", "har", "har_vix", "sv_kf", "sv_vix"])
def test_forecasts_ignore_the_future(setup, name):
    panel, sched, cutoff = setup
    bad = corrupt_after(panel, cutoff)
    run = {
        "garch": lambda p: M.run_garch(p, sched, leverage=False),
        "gjr": lambda p: M.run_garch(p, sched, leverage=True),
        "har": lambda p: M.run_har(p, sched, with_vix=False),
        "har_vix": lambda p: M.run_har(p, sched, with_vix=True),
        "sv_kf": lambda p: M.run_sv(p, sched),
        "sv_vix": lambda p: M.run_sv_vix(p, sched),
    }[name]
    a, b = run(panel), run(bad)
    lo = sched.refits[0]
    np.testing.assert_allclose(a.iloc[lo:cutoff + 1], b.iloc[lo:cutoff + 1], rtol=0, atol=1e-10)
    # and the corruption really does change later forecasts (the test has teeth)
    assert not np.allclose(a.iloc[cutoff + 2:1301], b.iloc[cutoff + 2:1301])


def test_smoothed_states_DO_use_the_future():
    """Sanity check of the test design: the RTS smoother is supposed to fail it."""
    panel = synthetic_panel()
    cutoff = 1150
    bad = corrupt_after(panel, cutoff)
    fit = sv.fit(panel["r"].to_numpy()[:1000])
    res_a = sv.smooth_with(panel["r"].to_numpy(), fit)
    res_b = sv.smooth_with(bad["r"].to_numpy(), fit)
    sm_a, _ = sv.filtered_arrays(res_a, smoothed=True)
    sm_b, _ = sv.filtered_arrays(res_b, smoothed=True)
    f_a, _ = sv.filtered_arrays(res_a)
    f_b, _ = sv.filtered_arrays(res_b)
    assert np.allclose(f_a[: cutoff + 1], f_b[: cutoff + 1])  # filter: causal
    assert not np.allclose(sm_a[cutoff - 50: cutoff + 1], sm_b[cutoff - 50: cutoff + 1])  # smoother: not


def test_target_is_forward_looking():
    panel = synthetic_panel(200)
    t = 50
    expected = np.sqrt(252 / 21 * (panel["r"].iloc[t + 1: t + 22] ** 2).sum())
    assert panel["rv21"].iloc[t] == pytest.approx(expected)
    past = np.sqrt(252 / 21 * (panel["r"].iloc[t - 20: t + 1] ** 2).sum())
    assert panel["rv21_past"].iloc[t] == pytest.approx(past)


def test_sv_recovers_simulated_parameters():
    panel = synthetic_panel(6000, seed=1)
    p = sv.params_of(sv.fit(panel["r"].to_numpy()))
    assert 0.95 < p["phi"] < 0.995  # true 0.98
    assert 0.01 < p["sigma2_eta"] < 0.05  # true 0.0225
