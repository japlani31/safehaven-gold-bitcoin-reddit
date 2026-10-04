"""Uji H3: apakah hedge share naik saat ketidakpastian (ln EPU) tinggi?
OLS  HShare_t = a + b lnEPU_t + e_t  dengan HAC (Newey-West, lag 20).
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
p = pd.read_csv(BASE / "data_processed" / "panel_tvpvar.csv",
                parse_dates=["date"]).set_index("date")


def ols_nw(y, x, L=20):
    x = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    e = y - x @ b
    n, k = x.shape
    xtx_inv = np.linalg.inv(x.T @ x)
    S = (x * e[:, None]).T @ (x * e[:, None])
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        G = (x[l:] * e[l:, None]).T @ (x[:-l] * e[:-l, None])
        S += w * (G + G.T)
    V = xtx_inv @ S @ xtx_inv
    t = b / np.sqrt(np.diag(V))
    return b[1], t[1]


for dep in ["HSHARE_C", "HSHARE_G"]:
    d = p[[dep, "EPU"]].dropna()
    r = d[dep].corr(d["EPU"])
    b, t = ols_nw(d[dep].to_numpy(), d["EPU"].to_numpy())
    print(f"{dep}: corr(lnEPU)={r:+.3f} | OLS b={b:+.4f} (NW t={t:+.2f}) | N={len(d)}")
