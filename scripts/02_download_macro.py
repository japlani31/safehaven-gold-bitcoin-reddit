"""
02_download_macro.py  (menggantikan 02_download_epu.py)
Unduh variabel makro harian — semua gratis, tanpa API key.

X3 UTAMA : Fed Funds Effective Rate (FRED: DFF) — harian, 7 hari/minggu
X3 ALT   : 10Y Breakeven Inflation Rate (FRED: T10YIE) — ekspektasi inflasi harian
ROBUSTNESS: Daily US EPU (policyuncertainty.com), Daily GPR (matteoiacoviello.com)

Output: data_raw/fed_dff.csv, data_raw/breakeven_t10yie.csv,
        data_raw/epu_daily.csv, data_raw/gpr_daily.csv
"""
import _sslfix  # noqa: F401  — WAJIB paling atas (patch SSL, lihat _sslfix.py)

import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data_raw"

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
EPU_URL = "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv"
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"


def get_fred(series_id: str, colname: str) -> pd.DataFrame:
    df = pd.read_csv(FRED_CSV.format(sid=series_id))
    df.columns = ["date", colname]
    df["date"] = pd.to_datetime(df["date"])
    df[colname] = pd.to_numeric(df[colname], errors="coerce")  # "." = missing
    return df.set_index("date").dropna()


def main():
    OUT.mkdir(exist_ok=True)

    print("FRED DFF (Fed Funds Effective Rate, daily) ...")
    dff = get_fred("DFF", "FED")
    dff.to_csv(OUT / "fed_dff.csv")
    print(f"  {len(dff)} rows ({dff.index.min().date()} .. {dff.index.max().date()})")

    print("FRED T10YIE (10Y Breakeven Inflation, daily) ...")
    bei = get_fred("T10YIE", "BEI")
    bei.to_csv(OUT / "breakeven_t10yie.csv")
    print(f"  {len(bei)} rows")

    print("Daily US EPU (robustness) ...")
    if (OUT / "epu_daily.csv").exists():
        print("  sudah ada (konversi dari All_Daily_Policy_Data.csv user) — skip")
    else:
        try:
            epu = pd.read_csv(EPU_URL)
            epu["date"] = pd.to_datetime(epu[["year", "month", "day"]])
            epu = epu.set_index("date")[["daily_policy_index"]].rename(
                columns={"daily_policy_index": "EPU"})
            epu.to_csv(OUT / "epu_daily.csv")
            print(f"  {len(epu)} rows")
        except Exception as e:
            print(f"  EPU gagal ({e}) — unduh manual dari policyuncertainty.com/us_daily.html")

    print("Daily GPR (robustness) ...")
    try:
        gpr = pd.read_excel(GPR_URL)
        gpr.to_csv(OUT / "gpr_daily.csv", index=False)
        print(f"  {len(gpr)} rows")
    except Exception as e:
        print(f"  GPR gagal ({e}) — unduh manual dari matteoiacoviello.com/gpr.htm")


if __name__ == "__main__":
    main()
