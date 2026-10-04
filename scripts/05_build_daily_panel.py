"""
05_build_daily_panel.py
Gabungkan semua sumber menjadi satu panel harian 5-hari-kerja siap TVP-VAR.

Aturan penyelarasan (standar literatur):
  - BTC & sentimen Reddit hidup 7 hari -> nilai Sab+Min+Sen dirata-rata ke Senin
  - Return = 100 * dln(P); EPU = ln(level)
  - Missing hari libur bursa: forward-fill maks 3 hari lalu drop sisa NA

Output: data_processed/panel_tvpvar.csv (kolom: GOLD, BTC, SENT_G, SENT_C,
        FED, VIX, DXY [+ BEI, EPU robustness; HSHARE_G, HSHARE_C model alternatif])
"""
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
MARKET = BASE / "data_raw" / "market_daily.csv"
FED = BASE / "data_raw" / "fed_dff.csv"
BEI = BASE / "data_raw" / "breakeven_t10yie.csv"
EPU = BASE / "data_raw" / "epu_daily.csv"
SENT = BASE / "data_processed" / "sentiment_daily.csv"
OUT = BASE / "data_processed" / "panel_tvpvar.csv"


def weekend_to_monday(s: pd.Series) -> pd.Series:
    """Rata-rata nilai Sabtu+Minggu+Senin -> ditempatkan di Senin."""
    df = s.to_frame("v")
    df["wd"] = df.index.dayofweek
    # geser Sabtu(5)->+2 hari, Minggu(6)->+1 hari agar jatuh di Senin
    df.index = df.index + pd.to_timedelta(
        np.select([df["wd"] == 5, df["wd"] == 6], [2, 1], default=0), unit="D")
    return df.groupby(df.index)["v"].mean()


def main():
    mkt = pd.read_csv(MARKET, index_col="date", parse_dates=True)
    fed = pd.read_csv(FED, index_col="date", parse_dates=True)
    bei = pd.read_csv(BEI, index_col="date", parse_dates=True)
    epu = pd.read_csv(EPU, index_col="date", parse_dates=True)
    sent = pd.read_csv(SENT, index_col="date", parse_dates=True)

    # 7-hari -> 5-hari
    btc = weekend_to_monday(mkt["BTC"].dropna())
    sent5 = sent.apply(lambda col: weekend_to_monday(col.dropna()))
    fed5 = weekend_to_monday(fed["FED"].dropna())   # DFF terbit 7 hari/minggu
    epu5 = weekend_to_monday(epu["EPU"].dropna())

    # kalender = hari perdagangan emas (bursa tradisional)
    cal = mkt["GOLD"].dropna().index

    panel = pd.DataFrame(index=cal)
    panel["GOLD"] = 100 * np.log(mkt["GOLD"]).diff()
    panel["BTC"] = 100 * np.log(btc).diff().reindex(cal)
    panel["VIX"] = mkt["VIX"].reindex(cal)              # level (robust: dln)
    panel["DXY"] = 100 * np.log(mkt["DXY"]).diff().reindex(cal)
    panel["FED"] = fed5.reindex(cal).diff()             # X3 utama: d(Fed Funds)
    panel["BEI"] = bei["BEI"].reindex(cal).diff()       # X3 alternatif: d(breakeven)
    panel["EPU"] = np.log(epu5).reindex(cal)            # robustness
    for c in ["SENT_G", "SENT_C", "HSHARE_G", "HSHARE_C"]:
        if c in sent5.columns:
            panel[c] = sent5[c].reindex(cal)

    panel = panel.ffill(limit=3).dropna()
    # batas sampel baseline: narasi Reddit tersedia s/d 2025-12-31
    panel = panel.loc["2017-01-01":"2025-12-31"]
    panel.index.name = "date"
    panel.to_csv(OUT)
    print(f"Saved {len(panel)} rows x {panel.shape[1]} cols -> {OUT}")
    print(panel.describe().T[["count", "mean", "std", "min", "max"]].round(3))


if __name__ == "__main__":
    main()
