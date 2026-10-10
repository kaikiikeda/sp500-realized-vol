"""Run the full study: walk-forward forecasts, evaluation, backtest, figures.

Usage:  python scripts/run_all.py            (about a minute on a laptop)
Outputs: results/forecasts.csv, results/tables/*.csv, results/figures/*.png,
         results/summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volforecast import backtest as bt  # noqa: E402
from volforecast import evaluation as ev  # noqa: E402
from volforecast import models as M  # noqa: E402
from volforecast import plots as P  # noqa: E402
from volforecast import sv_kalman as sv  # noqa: E402
from volforecast.data import load_panel  # noqa: E402

OOS_START = "2005-01-03"  # ~6 years of training data before the first forecast
RES = ROOT / "results"
TAB = RES / "tables"
FIG = RES / "figures"


def main():
    for d in (TAB, FIG):
        d.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    end = panel.index.get_loc(panel["rv21"].last_valid_index())
    sched = M.make_schedule(panel.index, OOS_START, end)
    print(f"OOS {panel.index[sched.refits[0]].date()} .. {panel.index[end].date()}: "
          f"{end - sched.refits[0] + 1} daily forecasts, {len(sched.refits)} refits")

    fc, sv_params = M.run_all(panel, sched)
    rv = panel["rv21"].loc[fc.index]
    assert fc.notna().all().all(), "missing forecasts"
    out = fc.copy()
    out["rv21_target"] = rv
    out.round(4).to_csv(RES / "forecasts.csv")
    sv_params.round(6).to_csv(TAB / "sv_parameters_by_refit.csv")

    # 1. Main loss table (benchmark = HAR, the hardest simple model to beat)
    lt = ev.loss_table(fc, rv, bench="har")
    lt.round(4).to_csv(TAB / "losses_full_oos.csv")
    lt_vix = ev.loss_table(fc, rv, bench="vix")
    lt_vix.round(4).to_csv(TAB / "losses_vs_vix.csv")
    print(lt.round(3).to_string())

    # 2. Sub-periods
    pq = ev.period_table(fc, rv, "qlike")
    pr = ev.period_table(fc, rv, "mse")
    pq.round(4).to_csv(TAB / "qlike_by_period.csv")
    pr.round(3).to_csv(TAB / "rmse_by_period.csv")
    print(pq.round(3).to_string())

    # 3. Look-ahead experiment
    la = M.sv_lookahead_variants(panel).loc[fc.index]
    la["sv_kf"] = fc["sv_kf"]
    la_tab = ev.loss_table(la, rv, bench="sv_kf")[["rmse", "qlike", "mz_r2"]]
    la_tab.round(4).to_csv(TAB / "lookahead_experiment.csv")
    print(la_tab.round(3).to_string())

    # 4. Backtest
    btab = bt.run_backtest(fc, rv)
    btab.round(3).to_csv(TAB / "backtest.csv", index=False)
    curves = bt.equity_curves(fc, rv, "sv_kf", 2.0)
    print(btab[btab.cost == 0].round(2).to_string())

    # 5. Figures
    full_res = sv.fit(panel["r"].to_numpy())
    sm = sv.smooth_with(panel["r"].to_numpy(), full_res)
    P.fig_latent_state(panel, sv.latent_vol(sm), sv.latent_vol(sm, smoothed=True),
                       FIG / "01_latent_vol.png")
    P.fig_forecasts(fc, rv, FIG / "02_forecasts_2019_2023.png")
    P.fig_cum_loss(fc, rv, ev.qlike, FIG / "03_cumulative_qlike_vs_har.png")
    P.fig_vrp(fc, rv, FIG / "04_variance_risk_premium.png")
    P.fig_equity(curves, FIG / "05_backtest_equity.png")
    P.fig_lookahead(la_tab, FIG / "06_lookahead.png")

    # 6. Machine-readable summary (the README numbers are copied from here)
    vrp = fc["vix"] - rv
    summary = {
        "oos_start": str(fc.index[0].date()), "oos_end": str(fc.index[-1].date()),
        "n_forecasts": int(len(fc)), "n_refits": int(len(sched.refits)),
        "rank_by_qlike": lt["qlike"].sort_values().index.tolist(),
        "rank_by_rmse": lt["rmse"].sort_values().index.tolist(),
        "losses": lt.round(4).to_dict(orient="index"),
        "vix_minus_rv_mean": round(float(vrp.mean()), 3),
        "vix_above_rv_pct": round(float((vrp > 0).mean() * 100), 1),
        "sv_params_full_sample": {k: round(float(v), 5) for k, v in sv.params_of(full_res).items()},
        "lookahead": la_tab.round(4).to_dict(orient="index"),
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    import pairwise_dm  # same folder; reads results/forecasts.csv written above
    pairwise_dm.main()
    print("done")


if __name__ == "__main__":
    main()
