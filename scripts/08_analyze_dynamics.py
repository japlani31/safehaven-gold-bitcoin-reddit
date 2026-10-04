"""
08_analyze_dynamics.py
Analisis episode dari deret dinamis TVP-VAR (output 07):
  - TCI per episode krisis vs rata-rata sampel
  - Kapan sentimen berbalik menjadi NET TRANSMITTER (inti H1)
  - NPDC BTC<->SENT_C dan GOLD<->SENT_G per episode
Output: output/tables/episode_analysis.csv + print ringkasan
"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
TAB = BASE / "output" / "tables"

EPISODES = {
    "Full sample":       ("2017-01-01", "2025-12-31"),
    "Mania kripto 2017": ("2017-11-01", "2018-01-31"),
    "COVID crash":       ("2020-02-20", "2020-04-30"),
    "Mania ritel 2021":  ("2021-01-01", "2021-05-31"),
    "ATH Nov 2021":      ("2021-10-01", "2021-12-31"),
    "Pengetatan Fed":    ("2022-03-01", "2022-12-31"),
    "LUNA kolaps":       ("2022-05-01", "2022-05-31"),
    "FTX kolaps":        ("2022-11-01", "2022-11-30"),
    "ETF spot BTC":      ("2024-01-01", "2024-02-29"),
    "2025":              ("2025-01-01", "2025-12-31"),
}


def main():
    tci = pd.read_csv(TAB / "dyn_tci.csv", parse_dates=["date"]).set_index("date")
    net = pd.read_csv(TAB / "dyn_net.csv", parse_dates=["date"]).set_index("date")
    try:
        npdc = pd.read_csv(TAB / "dyn_npdc_pairs.csv",
                           parse_dates=["date"]).set_index("date")
    except FileNotFoundError:
        npdc = None

    rows = []
    for name, (a, b) in EPISODES.items():
        t = tci.loc[a:b, "TCI"]
        nn = net.loc[a:b]
        row = {
            "episode": name,
            "hari": len(t),
            "TCI_mean": t.mean(),
            "TCI_max": t.max(),
            "NET_BTC": nn["BTC"].mean(),
            "NET_GOLD": nn["GOLD"].mean(),
            "NET_SENTC": nn["SENT_C"].mean(),
            "NET_SENTG": nn["SENT_G"].mean(),
            "SENTC_transmit_%": (nn["SENT_C"] > 0).mean() * 100,
            "SENTG_transmit_%": (nn["SENT_G"] > 0).mean() * 100,
        }
        if npdc is not None:
            for c in npdc.columns:
                row[c] = npdc.loc[a:b, c].mean()
        rows.append(row)

    df = pd.DataFrame(rows).set_index("episode")
    df.to_csv(TAB / "episode_analysis.csv")
    pd.set_option("display.width", 200)
    print("=== TCI & NET per episode ===")
    print(df[["hari", "TCI_mean", "TCI_max", "NET_BTC", "NET_GOLD",
              "NET_SENTC", "NET_SENTG"]].round(2).to_string())
    print("\n=== Persen hari sentimen jadi NET TRANSMITTER ===")
    print(df[["SENTC_transmit_%", "SENTG_transmit_%"]].round(1).to_string())
    if npdc is not None:
        print("\n=== NPDC pasangan kunci (rata-rata per episode) ===")
        print(df[[c for c in df.columns if c.startswith("NPDC")]]
              .round(2).to_string())

    # puncak-puncak TCI sepanjang sampel
    print("\n=== 10 hari TCI tertinggi ===")
    print(tci["TCI"].nlargest(10).round(1).to_string())


if __name__ == "__main__":
    main()
