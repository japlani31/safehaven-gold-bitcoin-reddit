"""
build_supplementary.py — Online Supplementary Material utk Quality & Quantity.
Isi: S1 pipeline data (tahap & jumlah), S2 seed vocabularies, S3 inventori
topik top-30 per aset (label final), S4 definisi episode, S5 figur volume
harian komentar ter-skor, S6 versi perangkat lunak.
Output: manuscript/supplementary_material.md + .docx, output/figures/figS1_volume.png
"""
import re
import subprocess
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
MAN = BASE / "manuscript"
TAB = BASE / "output" / "tables"
FIG = BASE / "output" / "figures"
PANDOC = (Path(r"C:\Users\ardia\AppData\Local\Temp\claude"
          r"\C--Users-ardia-Documents-Submit-Artikel-Scopus-DATA-IMF-BANK-DUNIA-IMF-world-bank"
          r"\df340b6b-ddb3-431b-8a9f-20c04d46a03f\scratchpad\tools\pandoc-3.10\pandoc.exe"))

OVERRIDES = {  # identik dgn 04b
    ("crypto", 12): "SPEC", ("crypto", 13): "SPEC", ("crypto", 52): "DROP",
    ("crypto", 1): "DROP", ("crypto", 3): "GEN", ("crypto", 2): "GEN",
    ("crypto", 29): "GEN",
    ("gold", 0): "GEN", ("gold", 13): "DROP", ("gold", 60): "DROP",
    ("gold", 14): "DROP", ("gold", 11): "HEDGE", ("gold", 12): "GEN",
}
HEDGE_SEEDS = ["inflation hedge", "store of value", "safe haven asset",
               "protect savings from inflation", "currency debasement",
               "preserve purchasing power", "hedge against uncertainty"]
SPEC_SEEDS = ["to the moon", "pump and dump", "buy the dip", "lambo",
              "leverage trading", "gambling", "all time high",
              "get rich quick", "100x gains", "short squeeze"]


def topic_tables():
    scored = pd.read_parquet(BASE / "data_processed" / "comments_scored.parquet",
                             columns=["asset", "topic", "meta", "date"])
    for (a, t), lab in OVERRIDES.items():
        scored.loc[(scored["asset"] == a) & (scored["topic"] == t), "meta"] = lab
    scored = scored[scored["meta"] != "DROP"]

    out = {}
    for asset in ["crypto", "gold"]:
        info = pd.read_csv(TAB / f"topic_info_{asset}.csv")
        m = (scored[scored["asset"] == asset]
             .groupby("topic").agg(meta=("meta", "first"), n=("meta", "size")))
        info = info.merge(m, left_on="Topic", right_index=True)
        top = info.sort_values("n", ascending=False).head(30)
        rows = ["| Topic | Label | Comments | Representative terms |",
                "|---|---|---|---|"]
        for _, r in top.iterrows():
            words = " ".join(str(r.top_words).split()[:8]).replace("|", "/")
            rows.append(f"| {r.Topic} | {r.meta} | {r.n:,} | {words} |")
        out[asset] = "\n".join(rows)
    return out, scored


def volume_figure(scored):
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    daily = (scored.groupby(["asset", "date"]).size().rename("n")
             .reset_index())
    fig, ax = plt.subplots(figsize=(9, 3))
    for asset, style in [("crypto", "-"), ("gold", "--")]:
        d = daily[daily["asset"] == asset].set_index("date")["n"]
        d.index = pd.to_datetime(d.index)
        ax.plot(d.index, d.rolling(30, min_periods=5).mean(), style,
                color="black", lw=1.0,
                label=f"{asset.capitalize()} (30-day MA)")
    ax.set_ylabel("Scored comments per day")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "figS1_volume.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / "figS1_volume.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    tt, scored = topic_tables()
    volume_figure(scored)
    vers = {}
    for pkg in ["torch", "bertopic", "sentence-transformers", "transformers",
                "pandas", "pyarrow", "zstandard", "numpy", "scipy"]:
        try:
            vers[pkg] = version(pkg)
        except Exception:
            vers[pkg] = "n/a"
    ver_rows = "\n".join(f"| {k} | {v} |" for k, v in vers.items())

    md = f"""# Online Supplementary Material

**Manuscript:** Safe Haven or Speculative Bubble? Decoding Gold and Digital Asset Dynamics through BERTopic Narratives and Global Macro Indicators

*(Prepared for double-blind review; no author-identifying information.)*

## S1. Data pipeline: stages and counts

**Table S1. Corpus construction from raw archive to final scored sample.**

| Stage | Filter applied | Comments remaining |
|---|---|---|
| 1. Raw archive extraction | r/CryptoCurrency, r/Bitcoin, r/Gold, r/investing (keyword-filtered), 2015 to 2025; removal of deleted and moderated content, comments under 20 characters, and bot messages | 43,755,202 |
| 2. Sample window | Comments dated 1 January 2017 onward | 42,122,137 |
| 3. Stratified daily sampling | Cap of 500 comments per asset class per day; Bernoulli thinning with keep probability cap/volume, fixed seed 42 | 2,461,436 |
| 4. Topic classification | BERTopic assignment; NOISE topics and comments below cosine similarity 0.30 removed | 1,215,564 |
| 5. Human adjudication | Thirteen label overrides; four incoherent or off-market topics dropped (Table B2 of the manuscript) | 1,136,025 |

The final scored corpus comprises 771,353 cryptocurrency comments (167,956 general, 143,307 hedging, 460,090 speculative) and 364,672 gold comments (124,369 general, 166,927 hedging, 73,376 speculative).

## S2. Seed vocabularies for narrative meta-labeling

Topic-level meta-labels were assigned by cosine similarity between each topic's ten most representative terms and the mean sentence embedding of the following seed phrases, with a minimum similarity of 0.25; labels were then reviewed manually as described in Section 3.3 of the manuscript.

**Hedging seeds:** {", ".join(HEDGE_SEEDS)}.

**Speculation seeds:** {", ".join(SPEC_SEEDS)}.

## S3. Topic inventories (30 highest-volume topics per asset, final labels)

**Table S2. Cryptocurrency topics.**

{tt['crypto']}

**Table S3. Gold topics.**

{tt['gold']}

The complete inventories (109 cryptocurrency topics, 74 gold topics) are provided as comma-separated files in the reproduction package.

## S4. Episode definitions

**Table S4. Event windows used in the episode analysis (Table 3 of the manuscript).**

| Episode | Window |
|---|---|
| Crypto mania 2017 | 2017-11-01 to 2018-01-31 |
| COVID-19 crash | 2020-02-20 to 2020-04-30 |
| Retail mania 2021 | 2021-01-01 to 2021-05-31 |
| Market peak | 2021-10-01 to 2021-12-31 |
| Fed tightening | 2022-03-01 to 2022-12-31 |
| Terra/LUNA collapse | 2022-05-01 to 2022-05-31 |
| FTX collapse | 2022-11-01 to 2022-11-30 |
| Spot ETF approval | 2024-01-01 to 2024-02-29 |

## S5. Daily volume of scored comments

**Figure S1.** Thirty-day moving average of the number of scored comments per day by asset class, 2017 to 2025 (after sampling, topic classification, and noise removal). Scored cryptocurrency volume fluctuates between roughly 150 and 300 comments per day, below the 500 sampling cap because approximately half of sampled comments are classified as noise. Scored gold volume rises from roughly 10 comments per day in 2017 and 2018 to about 200 by 2025, which motivates the 2019 subsample robustness check and the thin-early-sample discussion in the manuscript. (File: figS1_volume, 300 dpi.)

## S6. Software environment

**Table S5. Software and package versions.**

| Package | Version |
|---|---|
| Python | 3.14.2 |
| R | 4.6.0 |
| ConnectednessApproach (R) | 1.0.0 |
| rmgarch (R) | latest CRAN at estimation date |
{ver_rows}

Estimations were run on a consumer laptop (NVIDIA RTX 4050 Laptop GPU, 6 GB VRAM). All random seeds are fixed (seed 42 for sampling and topic model fitting).
"""
    out_md = MAN / "supplementary_material.md"
    out_md.write_text(md, encoding="utf-8")
    r = subprocess.run([str(PANDOC), str(out_md), "-o",
                        str(MAN / "supplementary_material.docx")],
                       capture_output=True, text=True)
    print("supplementary_material.docx:",
          "OK" if r.returncode == 0 else r.stderr[:300])
    print("figS1 tersimpan.")


if __name__ == "__main__":
    main()
