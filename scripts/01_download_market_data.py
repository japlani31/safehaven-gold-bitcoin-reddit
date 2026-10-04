"""
01_download_market_data.py
Unduh data harian pasar via Yahoo Finance chart API (gratis).
Memakai urllib langsung (bukan yfinance) karena lebih transparan
terhadap masalah SSL di mesin ini — lihat _sslfix.py.

Output: data_raw/market_daily.csv
"""
import _sslfix  # noqa: F401  — WAJIB paling atas (patch SSL)

import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "data_raw"

START = datetime(2015, 1, 1, tzinfo=timezone.utc)
END = datetime(2026, 7, 1, tzinfo=timezone.utc)

TICKERS = {
    "GC=F": "GOLD",       # Gold futures (COMEX) — proksi XAU/USD
    "BTC-USD": "BTC",     # Bitcoin
    "^VIX": "VIX",        # CBOE Volatility Index
    "DX-Y.NYB": "DXY",    # US Dollar Index
    "CL=F": "WTI",        # WTI crude — robustness
    "ETH-USD": "ETH",     # Ethereum — robustness
}

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
API = ("https://query1.finance.yahoo.com/v8/finance/chart/{t}"
       "?period1={p1}&period2={p2}&interval=1d")


def fetch(ticker: str) -> pd.Series:
    url = API.format(t=quote(ticker), p1=int(START.timestamp()),
                     p2=int(END.timestamp()))
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as r:
        j = json.loads(r.read())
    res = j["chart"]["result"][0]
    ts = res["timestamp"]
    ind = res["indicators"]
    close = (ind.get("adjclose", [{}])[0].get("adjclose")
             or ind["quote"][0]["close"])
    idx = pd.to_datetime(ts, unit="s", utc=True).tz_convert(None).normalize()
    s = pd.Series(close, index=idx, dtype="float64").dropna()
    return s[~s.index.duplicated(keep="last")]


def main():
    frames = {}
    for ticker, name in TICKERS.items():
        print(f"Downloading {name} ({ticker}) ...", end=" ")
        try:
            s = fetch(ticker)
            frames[name] = s
            print(f"{len(s)} rows ({s.index.min().date()} .. {s.index.max().date()})")
        except Exception as e:
            print(f"FAILED: {e}")
        time.sleep(1)  # sopan terhadap rate limit Yahoo

    panel = pd.DataFrame(frames)
    panel.index.name = "date"
    OUT.mkdir(exist_ok=True)
    panel.to_csv(OUT / "market_daily.csv")
    print(f"\nSaved {len(panel)} rows x {panel.shape[1]} cols -> {OUT / 'market_daily.csv'}")
    print(panel.describe().T[["count", "mean", "min", "max"]].round(2))


if __name__ == "__main__":
    main()
