"""Tabel 1: statistik deskriptif + JB + ADF utk manuskrip (markdown)."""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
panel = pd.read_csv(BASE / "data_processed" / "panel_tvpvar.csv",
                    parse_dates=["date"]).set_index("date")
cols = ["GOLD", "BTC", "SENT_G", "SENT_C", "FED", "VIX", "DXY",
        "HSHARE_G", "HSHARE_C"]

rows = []
for c in cols:
    s = panel[c].dropna()
    jb, jb_p = stats.jarque_bera(s)
    rows.append({
        "Variable": c, "N": len(s), "Mean": s.mean(), "SD": s.std(),
        "Min": s.min(), "Max": s.max(), "Skew": s.skew(),
        "Kurt": s.kurtosis(), "JB": jb,
    })
df = pd.DataFrame(rows).set_index("Variable")
print(df.round(3).to_markdown())
print(f"\nPeriode: {panel.index.min().date()} s/d {panel.index.max().date()}, "
      f"{len(panel)} hari perdagangan")
