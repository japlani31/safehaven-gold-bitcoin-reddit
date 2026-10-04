"""
03_reddit_ingest.py
Parse dump Reddit (.zst ATAU .jsonl) dari Arctic Shift / Academic Torrents.
Streaming penuh: aman untuk file besar (r/CryptoCurrency 4 GB zst) tanpa
menghabiskan RAM — hasil ditulis bertahap ke parquet per subreddit.

Letakkan file di data_raw/reddit/ dengan nama <Subreddit>_comments.<ext>.
Subreddit yang tidak terdaftar di ASSET_MAP otomatis dilewati.

Output: data_processed/reddit_parsed/<Subreddit>.parquet
Kebutuhan: pip install zstandard pandas pyarrow
"""
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data_raw" / "reddit"
OUT = BASE / "data_processed" / "reddit_parsed"

# Subreddit -> aset. r/investing dipilah lewat keyword (gold vs crypto).
ASSET_MAP = {
    "CryptoCurrency": "crypto",
    "Bitcoin": "crypto",
    "Gold": "gold",
    "investing": "keyword",
}

GOLD_KW = re.compile(r"\b(gold|xau|bullion|precious metal)\b", re.I)
CRYPTO_KW = re.compile(r"\b(bitcoin|btc|crypto|ethereum|eth|altcoin)\b", re.I)
BOT_KW = re.compile(r"\bI am a bot\b|\bbot action\b", re.I)

MIN_LEN = 20        # buang komentar terlalu pendek
MAX_CHARS = 600     # FinBERT/MiniLM hanya membaca ~256 token pertama
START_TS = pd.Timestamp("2015-01-01").timestamp()
CHUNK = 200_000     # baris per flush ke parquet

SCHEMA = pa.schema([
    ("id", pa.string()),
    ("date", pa.date32()),
    ("subreddit", pa.string()),
    ("asset", pa.string()),
    ("score", pa.int32()),
    ("text", pa.string()),
])


def stream_lines(path: Path):
    """Baca file .zst atau .jsonl baris-per-baris tanpa load penuh ke memori."""
    if path.suffix in (".jsonl", ".ndjson", ".json"):
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            yield from fh
        return
    with open(path, "rb") as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(fh) as reader:
            buffer = ""
            while True:
                chunk = reader.read(2**24)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="ignore")
                lines = buffer.split("\n")
                buffer = lines[-1]
                yield from lines[:-1]
        if buffer.strip():
            yield buffer


def classify_asset(tag: str, text: str):
    if tag == "keyword":
        if GOLD_KW.search(text):
            return "gold"
        if CRYPTO_KW.search(text):
            return "crypto"
        return None
    return tag


def process_file(dump_file: Path, sub: str, tag: str) -> int:
    out_path = OUT / f"{sub}.parquet"
    writer = pq.ParquetWriter(out_path, SCHEMA, compression="zstd")
    rows, n_kept, n_seen = [], 0, 0

    def flush():
        nonlocal rows
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
            rows = []

    for line in stream_lines(dump_file):
        n_seen += 1
        if n_seen % 2_000_000 == 0:
            print(f"    ... {n_seen/1e6:.0f}M baris dibaca, {n_kept:,} disimpan")
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = obj.get("created_utc")
        if ts is None or float(ts) < START_TS:
            continue
        text = obj.get("body") or obj.get("selftext") or ""
        if len(text) < MIN_LEN or text in ("[deleted]", "[removed]"):
            continue
        if BOT_KW.search(text):
            continue
        asset = classify_asset(tag, text)
        if asset is None:
            continue
        rows.append({
            "id": obj.get("id"),
            "date": pd.to_datetime(float(ts), unit="s").date(),
            "subreddit": sub,
            "asset": asset,
            "score": int(obj.get("score") or 0),
            "text": text[:MAX_CHARS],
        })
        n_kept += 1
        if len(rows) >= CHUNK:
            flush()

    flush()
    writer.close()
    return n_kept


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(list(RAW.glob("*.zst")) + list(RAW.glob("*.jsonl"))
                   + list(RAW.glob("*.ndjson")))
    if not files:
        print(f"TIDAK ADA FILE di {RAW} — unduh dump dulu (lihat docstring)")
        return

    total = 0
    for dump_file in files:
        sub = dump_file.stem.split("_")[0]
        tag = ASSET_MAP.get(sub)
        if tag is None:
            print(f"Skip {dump_file.name} (subreddit '{sub}' tidak terdaftar)")
            continue
        print(f"Parsing {dump_file.name} ({dump_file.stat().st_size/1e9:.2f} GB) ...")
        n = process_file(dump_file, sub, tag)
        print(f"  -> {n:,} komentar disimpan ke {sub}.parquet")
        total += n

    print(f"\nTOTAL: {total:,} komentar. Ringkasan per subreddit:")
    for p in sorted(OUT.glob("*.parquet")):
        df = pd.read_parquet(p, columns=["date", "asset"])
        print(f"  {p.stem:16s} {len(df):>10,} | {df['date'].min()} .. {df['date'].max()}")


if __name__ == "__main__":
    main()
