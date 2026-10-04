"""
03b_sample_comments.py
Stratified daily sampling: maksimal ~CAP komentar per (aset, hari).

Alasan (untuk bagian Data manuskrip):
  - 43,7 juta komentar tidak feasible untuk FinBERT/BERTopic penuh.
  - Indeks bullishness harian adalah estimator proporsi; dengan n=500/hari
    standard error <= 1/sqrt(500) ~ 0.045 — presisi lebih dari cukup.
  - Sampling Bernoulli per hari dengan seed tetap (reproducible).
  - Hari dengan komentar <= CAP diambil SEMUA (tidak ada distorsi di
    hari-hari sepi, penting untuk emas 2017-2018).

Input : data_processed/reddit_parsed/<Subreddit>.parquet
Output: data_processed/comments_sampled.parquet
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
IN = BASE / "data_processed" / "reddit_parsed"
OUT = BASE / "data_processed" / "comments_sampled.parquet"

CAP = 500                      # target komentar per (aset, hari)
START = pd.Timestamp("2017-01-01").date()   # baseline sample (keputusan 2026-07-03)
SEED = 42


def main():
    rng = np.random.default_rng(SEED)

    # --- Tahap 1: hitung jumlah komentar per (aset, hari) — hanya kolom kecil
    counts = {}
    for p in sorted(IN.glob("*.parquet")):
        d = pd.read_parquet(p, columns=["date", "asset"])
        d = d[d["date"] >= START]
        c = d.groupby(["asset", "date"]).size()
        for k, v in c.items():
            counts[k] = counts.get(k, 0) + v
    counts = pd.Series(counts)
    print(f"Hari-aset unik: {len(counts):,} | total komentar >= {START}: {counts.sum():,}")

    # probabilitas keep per (aset, hari): min(1, CAP/n)
    keep_p = (CAP / counts).clip(upper=1.0)

    # --- Tahap 2: baca per row-group, thinning Bernoulli
    kept_parts = []
    for p in sorted(IN.glob("*.parquet")):
        pf = pq.ParquetFile(p)
        n_kept = 0
        for rg in range(pf.num_row_groups):
            df = pf.read_row_group(rg).to_pandas()
            df = df[df["date"] >= START]
            if df.empty:
                continue
            probs = keep_p.reindex(
                pd.MultiIndex.from_frame(df[["asset", "date"]])).to_numpy()
            mask = rng.random(len(df)) < probs
            kept = df[mask]
            kept_parts.append(kept)
            n_kept += len(kept)
        print(f"{p.stem:16s} -> {n_kept:,} komentar tersampel")

    sample = pd.concat(kept_parts, ignore_index=True)
    sample.to_parquet(OUT, index=False)
    print(f"\nTOTAL sampel: {len(sample):,} -> {OUT}")
    daily_n = sample.groupby(["asset", "date"]).size()
    print("Komentar/hari dalam sampel: median per aset:")
    print(daily_n.groupby("asset").median().to_string())


if __name__ == "__main__":
    main()
