# Replication package: Safe-haven talk or speculative trade? Reddit narratives and shock transmission between gold and Bitcoin

Authors: Ardiansyah Japlani (Universitas Lampung; Universitas Muhammadiyah Metro), Ernie Hendrawaty (Universitas Lampung), Igo Febrianto (Universitas Lampung).

## Contents

| Folder | Content |
|---|---|
| `scripts/` | Data download (01, 02), Reddit parsing and sampling (03, 03b), BERTopic and FinBERT (04, 04b), daily panel (05), TVP-VAR connectedness and dynamics (06 to 08), figures (09), robustness and QVAR (10, 11, 12, 12a), DCC-GARCH portfolio (11c), second-stage regressions (13) |
| `data_derived/` | `panel_tvpvar.csv` (2,263 trading days) and `sentiment_daily.csv` (daily narrative indices) |
| `tables/` | All output tables, topic inventories, label overrides, and robustness logs |
| `validation/` | Coding protocol, sampled comment IDs, FinBERT/BERTopic labels, blind annotator labels, and summary (Section 3.4) |
| `figures/` | Figures of the manuscript and supplement |

## Data

Market data: Yahoo Finance (GC=F, BTC-USD, ^VIX, DX-Y.NYB). Macro data: FRED (DFF, T10YIE) and the daily US Economic Policy Uncertainty index. Reddit comments: public subreddit archives compiled from Pushshift and Arctic Shift (r/CryptoCurrency, r/Bitcoin, r/Gold, r/investing). Comment texts are not redistributed; comment IDs are provided so that the sample can be rebuilt from the archives.

## Run order

Python requirements in `requirements.txt`, R packages in `r_packages.txt`. Run the scripts in numeric order from `scripts/`.

## License

Code: MIT License. Derived data, tables, and figures: CC BY 4.0.
