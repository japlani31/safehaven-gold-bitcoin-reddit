"""
04b_refine_labels.py
Adjudikasi manual label topik (validasi co-author, 2026-07-04) + re-agregasi.

Prosedur standar text-as-data: klasifikasi algoritmik (04) -> review manusia
atas topik terbesar -> tabel override didokumentasikan di appendix manuskrip.

Perubahan definisi:
  - Kategori baru GEN (general): relevan aset & bersentimen, tapi bukan
    narasi hedge/spec. Masuk indeks sentimen, keluar dari hedge share.
  - HSHARE = n_HEDGE / (n_HEDGE + n_SPEC)   <- komposisi narasi murni
  - DROP   = topik sampah/meta-komunitas, dibuang total.

Output: data_processed/sentiment_daily.csv (menimpa versi 04)
        output/tables/topic_overrides.csv  (dokumentasi utk appendix)
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PROC = BASE / "data_processed"
TAB = BASE / "output" / "tables"

# (asset, topic) -> label baru. Alasan didokumentasikan utk appendix.
OVERRIDES = {
    # --- CRYPTO ---
    ("crypto", 12): ("SPEC", "prediksi harga (btc 100k) = spekulasi, bukan hedge"),
    ("crypto", 13): ("SPEC", "obrolan market cap/harga koin"),
    ("crypto", 52): ("DROP", "infrastruktur exchange/wallet, bukan narasi"),
    ("crypto", 1):  ("DROP", "r/CC Moons token = meta-komunitas, bukan pasar"),
    ("crypto", 3):  ("GEN",  "diskusi nilai bitcoin umum, bukan hedge spesifik"),
    ("crypto", 2):  ("GEN",  "obrolan crypto vs stocks umum"),
    ("crypto", 29): ("GEN",  "gosip tokoh (CEO/SBF/CZ), sentimen relevan"),
    # --- GOLD ---
    ("gold", 0):  ("GEN",  "catch-all obrolan emas; sentimen relevan, bukan narasi"),
    ("gold", 13): ("DROP", "sampah semantik (gloves/chicken)"),
    ("gold", 60): ("DROP", "cerita warisan keluarga, bukan pasar"),
    ("gold", 14): ("DROP", "perhiasan/konsumsi, bukan investasi"),
    ("gold", 11): ("HEDGE", "budaya stacking = akumulasi store-of-value"),
    ("gold", 12): ("GEN",  "diskusi produk koin (eagle/buffalo)"),
}


def main():
    df = pd.read_parquet(PROC / "comments_scored.parquet")
    print(f"{len(df):,} komentar ter-skor dimuat")

    # dokumentasi override
    pd.DataFrame([
        {"asset": a, "topic": t, "new_label": lab, "reason": why}
        for (a, t), (lab, why) in OVERRIDES.items()
    ]).to_csv(TAB / "topic_overrides.csv", index=False)

    for (a, t), (lab, _) in OVERRIDES.items():
        df.loc[(df["asset"] == a) & (df["topic"] == t), "meta"] = lab
    df = df[df["meta"] != "DROP"].copy()
    print("distribusi meta final:")
    print(df.groupby(["asset", "meta"]).size().unstack(fill_value=0).to_string())

    grp = df.groupby(["asset", "date"])
    daily = pd.DataFrame({
        "pos": grp.apply(lambda x: (x["sent"] == 1).sum(), include_groups=False),
        "neg": grp.apply(lambda x: (x["sent"] == -1).sum(), include_groups=False),
        "n": grp.size(),
        "n_hedge": grp.apply(lambda x: (x["meta"] == "HEDGE").sum(),
                             include_groups=False),
        "n_spec": grp.apply(lambda x: (x["meta"] == "SPEC").sum(),
                            include_groups=False),
    }).reset_index()
    daily["sent_index"] = (daily["pos"] - daily["neg"]) / daily["n"]
    denom = (daily["n_hedge"] + daily["n_spec"]).clip(lower=1)
    daily["hedge_share"] = daily["n_hedge"] / denom

    wide = daily.pivot(index="date", columns="asset",
                       values=["sent_index", "hedge_share", "n"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.rename(columns={
        "sent_index_gold": "SENT_G", "sent_index_crypto": "SENT_C",
        "hedge_share_gold": "HSHARE_G", "hedge_share_crypto": "HSHARE_C",
        "n_gold": "N_G", "n_crypto": "N_C"})
    wide.to_csv(PROC / "sentiment_daily.csv")
    print(f"\nSELESAI -> {PROC / 'sentiment_daily.csv'}")
    print(wide.describe().T[["count", "mean", "std", "min", "max"]].round(3))


if __name__ == "__main__":
    main()
