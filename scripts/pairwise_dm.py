"""Pairwise Diebold-Mariano tests between all models, from results/forecasts.csv.

Writes results/tables/dm_pairwise_{qlike,mse,mae}.csv. Cell (row a, column b) holds
the DM statistic for loss(a) - loss(b): negative means a beat b. A second table per
loss holds the p-values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from volforecast.evaluation import dm_test, qlike, sq_err  # noqa: E402


def main():
    f = pd.read_csv(ROOT / "results" / "forecasts.csv", index_col=0, parse_dates=True)
    y = f.pop("rv21_target").to_numpy()
    losses = {"qlike": qlike, "mse": sq_err, "mae": lambda a, b: np.abs(a - b)}
    for name, fn in losses.items():
        L = {m: fn(y, f[m].to_numpy()) for m in f.columns}
        stat = pd.DataFrame(index=f.columns, columns=f.columns, dtype=float)
        pval = stat.copy()
        for a in f.columns:
            for b in f.columns:
                if a != b:
                    stat.loc[a, b], pval.loc[a, b] = dm_test(L[a], L[b])
        stat.round(3).to_csv(ROOT / "results" / "tables" / f"dm_pairwise_{name}.csv")
        pval.round(4).to_csv(ROOT / "results" / "tables" / f"dm_pairwise_{name}_pvalues.csv")
    print("pairwise DM tables written")


if __name__ == "__main__":
    main()
