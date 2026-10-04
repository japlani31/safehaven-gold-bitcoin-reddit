"""Cek volume komentar harian per aset — QC sebelum tahap NLP."""
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data_processed" / "reddit_parsed"

parts = [pd.read_parquet(p, columns=["date", "asset"]) for p in OUT.glob("*.parquet")]
df = pd.concat(parts)
daily = df.groupby(["asset", "date"]).size().rename("n").reset_index()
daily["year"] = pd.to_datetime(daily["date"]).dt.year

print("=== Median komentar per HARI per tahun ===")
piv = daily.pivot_table(index="year", columns="asset", values="n", aggfunc="median")
print(piv.round(0).to_string())

print("\n=== Cakupan hari (2015-01-01 .. 2025-12-31 = 4017 hari) ===")
all_days = pd.date_range("2015-01-01", "2025-12-31", freq="D")
for a in sorted(daily["asset"].unique()):
    d = daily[daily["asset"] == a]
    missing = len(all_days) - d["date"].nunique()
    print(f"{a:8s}: {missing:4d} hari kosong | min={d['n'].min():>4} "
          f"median={d['n'].median():>7.0f} max={d['n'].max():>9,}")
