"""
13_determinants.py — R7-lite: Determinants of dynamic connectedness (Tabel §4.5).
OLS dgn Newey-West (lag 20):
  y_t = a + b1 lnEPU_t + b2 D_crisis_t + b3 D_mania2021_t + e_t
  y  in {TCI, NET_SENT_C, HSHARE_C, HSHARE_G}
D_crisis   = 1 pada jendela COVID (2020-02-20..04-30), LUNA (2022-05), FTX (2022-11)
D_mania21  = 1 pada 2021-01-01..2021-12-31
Output: print tabel siap salin ke manuskrip.
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
TAB = BASE / "output" / "tables"

panel = pd.read_csv(BASE / "data_processed" / "panel_tvpvar.csv",
                    parse_dates=["date"]).set_index("date")
tci = pd.read_csv(TAB / "dyn_tci.csv", parse_dates=["date"]).set_index("date")
net = pd.read_csv(TAB / "dyn_net.csv", parse_dates=["date"]).set_index("date")

df = panel[["EPU", "HSHARE_C", "HSHARE_G"]].copy()
df["TCI"] = tci["TCI"]
df["NET_SENTC"] = net["SENT_C"]
df = df.dropna()

d = df.index
crisis = (((d >= "2020-02-20") & (d <= "2020-04-30")) |
          ((d >= "2022-05-01") & (d <= "2022-05-31")) |
          ((d >= "2022-11-01") & (d <= "2022-11-30"))).astype(float)
mania = ((d >= "2021-01-01") & (d <= "2021-12-31")).astype(float)


def ols_nw(y, X, L=20):
    X = np.column_stack([np.ones(len(y))] + X)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    xtx_inv = np.linalg.inv(X.T @ X)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for l in range(1, L + 1):
        w = 1 - l / (L + 1)
        G = (X[l:] * e[l:, None]).T @ (X[:-l] * e[:-l, None])
        S += w * (G + G.T)
    V = xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.diag(V))
    r2 = 1 - e.var() / y.var()
    return b, b / se, r2


X = [df["EPU"].to_numpy(), crisis, mania]
labels = ["lnEPU", "D_crisis", "D_mania2021"]
print(f"{'dep var':12s} " + " ".join(f"{l:>22s}" for l in labels) + f" {'R2':>7s}")
rows = []
for dep in ["TCI", "NET_SENTC", "HSHARE_C", "HSHARE_G"]:
    b, t, r2 = ols_nw(df[dep].to_numpy(), X)
    cells = []
    for k in range(1, 4):
        star = "***" if abs(t[k]) > 2.58 else "**" if abs(t[k]) > 1.96 else "*" if abs(t[k]) > 1.65 else ""
        cells.append(f"{b[k]:+.3f}{star} ({t[k]:+.2f})")
    print(f"{dep:12s} " + " ".join(f"{c:>22s}" for c in cells) + f" {r2:7.3f}")
    rows.append({"dep": dep, **{l: c for l, c in zip(labels, cells)}, "R2": round(r2, 3)})

pd.DataFrame(rows).to_csv(TAB / "determinants_table.csv", index=False)
print(f"\nN = {len(df)} | Newey-West lag 20 | disimpan -> determinants_table.csv")
