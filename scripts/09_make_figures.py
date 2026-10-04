"""
09_make_figures.py — Figur publikasi (Fig 1-5) untuk manuskrip.
Output: output/figures/fig<N>_*.pdf + .png (300 dpi)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
TAB = BASE / "output" / "tables"
FIG = BASE / "output" / "figures"
PROC = BASE / "data_processed"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 9.5,
    "axes.linewidth": 0.6, "lines.linewidth": 0.8, "figure.dpi": 110,
})

# Episode krisis untuk shading abu-abu (konsisten di semua figur)
SHADE = [
    ("2020-02-20", "2020-04-30"),   # COVID crash
    ("2022-05-01", "2022-05-31"),   # LUNA
    ("2022-11-01", "2022-11-30"),   # FTX
]
EVENTS = {"2024-01-10": "ETF spot BTC"}   # garis vertikal

VARS = ["GOLD", "BTC", "SENT_G", "SENT_C", "FED", "VIX", "DXY"]
LABEL = {"GOLD": "Gold return", "BTC": "Bitcoin return",
         "SENT_G": "Gold sentiment", "SENT_C": "Crypto sentiment",
         "FED": "Δ Fed Funds", "VIX": "VIX", "DXY": "DXY return"}


def shade(ax):
    for a, b in SHADE:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="0.85", zorder=0)
    for d in EVENTS:
        ax.axvline(pd.Timestamp(d), color="0.4", ls=":", lw=0.7, zorder=0)


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name} tersimpan")


def main():
    panel = pd.read_csv(PROC / "panel_tvpvar.csv", parse_dates=["date"]).set_index("date")
    tci = pd.read_csv(TAB / "dyn_tci.csv", parse_dates=["date"]).set_index("date")
    net = pd.read_csv(TAB / "dyn_net.csv", parse_dates=["date"]).set_index("date")
    npdc = pd.read_csv(TAB / "dyn_npdc_pairs.csv", parse_dates=["date"]).set_index("date")

    # ---- Fig 1: deret waktu 7 variabel ------------------------------------
    fig, axes = plt.subplots(4, 2, figsize=(9, 8), sharex=True)
    axes = axes.ravel()
    for k, v in enumerate(VARS):
        ax = axes[k]
        ax.plot(panel.index, panel[v], color="black", lw=0.5)
        ax.set_title(LABEL[v], loc="left")
        shade(ax)
    axes[7].axis("off")
    fig.tight_layout()
    save(fig, "fig1_series")

    # ---- Fig 2: Total Connectedness Index ---------------------------------
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.fill_between(tci.index, tci["TCI"], color="0.3", alpha=0.9, lw=0)
    shade(ax)
    ax.set_ylabel("Total Connectedness Index (%)")
    ax.set_ylim(0, 50)
    for d, lbl in {"2020-03-16": "COVID", "2022-05-05": "LUNA",
                   "2022-11-10": "FTX", "2024-08-05": "Carry unwind"}.items():
        ax.annotate(lbl, xy=(pd.Timestamp(d), tci.loc[:d, "TCI"].iloc[-1]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig2_TCI_pub")

    # ---- Fig 3: NET directional spillovers (7 panel) -----------------------
    fig, axes = plt.subplots(4, 2, figsize=(9, 8), sharex=True)
    axes = axes.ravel()
    for k, v in enumerate(VARS):
        ax = axes[k]
        s = net[v]
        ax.fill_between(s.index, s, 0, where=s >= 0, color="0.25", lw=0)
        ax.fill_between(s.index, s, 0, where=s < 0, color="0.65", lw=0)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(f"NET {LABEL[v]}", loc="left")
        shade(ax)
    axes[7].axis("off")
    fig.tight_layout()
    save(fig, "fig3_NET_pub")

    # ---- Fig 4: NPDC pasangan kunci (4 panel) ------------------------------
    pairs = {
        "NPDC_BTC_SENTC": "BTC ↔ Crypto sentiment",
        "NPDC_GOLD_SENTG": "Gold ↔ Gold sentiment",
        "NPDC_BTC_VIX": "BTC ↔ VIX",
        "NPDC_GOLD_VIX": "Gold ↔ VIX",
    }
    fig, axes = plt.subplots(2, 2, figsize=(9, 5), sharex=True)
    for ax, (c, t) in zip(axes.ravel(), pairs.items()):
        s = npdc[c].rolling(20, min_periods=1).mean()
        ax.fill_between(s.index, s, 0, where=s >= 0, color="0.25", lw=0)
        ax.fill_between(s.index, s, 0, where=s < 0, color="0.65", lw=0)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_title(t + "  (MA-20)", loc="left")
        shade(ax)
    fig.tight_layout()
    save(fig, "fig4_NPDC_pub")

    # ---- Fig 5: Hedge Share vs EPU (evolusi narasi) -------------------------
    fig, ax = plt.subplots(figsize=(9, 3.4))
    hs_g = panel["HSHARE_G"].rolling(30, min_periods=5).mean()
    hs_c = panel["HSHARE_C"].rolling(30, min_periods=5).mean()
    ax.plot(hs_g.index, hs_g, color="black", lw=1.1, label="Gold hedge share")
    ax.plot(hs_c.index, hs_c, color="black", lw=1.1, ls="--",
            label="Crypto hedge share")
    ax.set_ylabel("Hedge share (MA-30)")
    ax.set_ylim(0, 1)
    ax2 = ax.twinx()
    epu = panel["EPU"].rolling(30, min_periods=5).mean()
    ax2.plot(epu.index, epu, color="0.55", lw=0.8, label="ln EPU (right axis)")
    ax2.set_ylabel("ln Daily US EPU", color="0.45")
    shade(ax)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=7.5, frameon=False)
    fig.tight_layout()
    save(fig, "fig5_hedgeshare_epu")

    print("Semua figur selesai ->", FIG)


if __name__ == "__main__":
    main()
