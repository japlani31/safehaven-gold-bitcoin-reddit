"""
04_bertopic_finbert.py  —  Tahap NLP inti (GPU: RTX 4050)

Pipeline per aset (gold / crypto):
  1. Embedding semua komentar (MiniLM, GPU, normalized)
  2. BERTopic DI-FIT pada subsampel acak 300k (UMAP+HDBSCAN tidak feasible
     untuk 2,46 jt dokumen) -> topik + centroid embedding per topik
  3. SEMUA komentar di-assign ke topik terdekat via cosine-to-centroid
     (threshold 0.30; di bawah itu = NOISE)
  4. Meta-label topik: HEDGE / SPEC / NOISE via cosine ke seed keywords
  5. FinBERT (GPU, fp16) memberi skor sentimen komentar non-NOISE
  6. Agregasi harian -> SENT_G, SENT_C, HSHARE_G, HSHARE_C

Output:
  data_processed/sentiment_daily.csv      (deret waktu untuk TVP-VAR)
  data_processed/comments_scored.parquet  (level-komentar, utk robustness)
  output/tables/topic_info_<asset>.csv    (utk validasi manual + lampiran)

Checkpoint: embeddings & skor disimpan bertahap — crash tidak mengulang dari nol.
"""
import _sslfix  # noqa: F401  — WAJIB: patch SSL utk unduhan model HuggingFace

import gc
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def gpu_guard():
    """Jeda antar-chunk agar GPU laptop tidak overheat ('GPU is lost' 2x).
    Duty cycle diturunkan: istirahat 3 dtk normal, lebih lama jika panas."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        t = int(out)
        if t >= 83:
            print(f"  [gpu] {t} C — pendinginan 60 dtk", flush=True)
            time.sleep(60)
        elif t >= 75:
            time.sleep(15)
        else:
            time.sleep(3)
    except Exception:
        time.sleep(3)

BASE = Path(__file__).resolve().parents[1]
IN = BASE / "data_processed" / "comments_sampled.parquet"
PROC = BASE / "data_processed"
TAB = BASE / "output" / "tables"
CKPT = PROC / "nlp_checkpoints"

EMB_MODEL = "all-MiniLM-L6-v2"
FIT_SAMPLE = 300_000       # dokumen untuk fit BERTopic per aset
MIN_CLUSTER = 400          # ukuran topik minimum pada sampel fit
SIM_THRESHOLD = 0.30       # cosine minimum agar dokumen masuk topik
SEED = 42

HEDGE_SEEDS = ["inflation hedge", "store of value", "safe haven asset",
               "protect savings from inflation", "currency debasement",
               "preserve purchasing power", "hedge against uncertainty"]
SPEC_SEEDS = ["to the moon", "pump and dump", "buy the dip", "lambo",
              "leverage trading", "gambling", "all time high",
              "get rich quick", "100x gains", "short squeeze"]
META_MARGIN = 0.25         # cosine minimum topik->seed agar bukan NOISE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------- embeddings
EMB_DIM = 384
EMB_CHUNK = 20_000   # simpan ke disk tiap chunk ini (tahan-crash)


def embed_texts(texts, st_model, tag):
    """Embed INKREMENTAL ke memmap float16; resume otomatis jika crash.

    File: emb_<tag>.dat (memmap N x 384) + emb_<tag>.done (int: baris selesai).
    GPU laptop bisa 'lost' di beban lama; checkpoint tiap 20k baris membuat
    restart hanya kehilangan <1 menit."""
    n = len(texts)
    dat = CKPT / f"emb_{tag}.dat"
    done_f = CKPT / f"emb_{tag}.done"

    emb = np.memmap(dat, dtype=np.float16, mode=("r+" if dat.exists() else "w+"),
                    shape=(n, EMB_DIM))
    start = int(done_f.read_text()) if done_f.exists() else 0
    if start >= n:
        print(f"  [emb {tag}] lengkap dari checkpoint ({n:,})")
        return np.asarray(emb)
    if start:
        print(f"  [emb {tag}] resume dari baris {start:,}/{n:,}")

    t0 = time.time()
    for i in range(start, n, EMB_CHUNK):
        j = min(i + EMB_CHUNK, n)
        vec = st_model.encode(texts[i:j], batch_size=64,
                              show_progress_bar=False, convert_to_numpy=True,
                              normalize_embeddings=True)
        emb[i:j] = vec.astype(np.float16)
        emb.flush()
        done_f.write_text(str(j))
        rate = (j - start) / (time.time() - t0)
        eta = (n - j) / rate / 60 if rate else 0
        print(f"  [emb {tag}] {j:,}/{n:,} ({rate:.0f} dok/s, sisa ~{eta:.0f} mnt)",
              flush=True)
        torch.cuda.empty_cache()
        gpu_guard()
    print(f"  [emb {tag}] selesai dalam {(time.time()-t0)/60:.1f} menit")
    return np.asarray(emb)


# ------------------------------------------------------------------ bertopic
def fit_topics(texts, emb, asset):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    rng = np.random.default_rng(SEED)
    idx = (np.arange(len(texts)) if len(texts) <= FIT_SAMPLE
           else rng.choice(len(texts), FIT_SAMPLE, replace=False))
    fit_docs = [texts[i] for i in idx]
    fit_emb = emb[idx].astype(np.float32)

    topic_model = BERTopic(
        umap_model=UMAP(n_neighbors=15, n_components=5, metric="cosine",
                        random_state=SEED, low_memory=True),
        hdbscan_model=HDBSCAN(min_cluster_size=MIN_CLUSTER, min_samples=20,
                              core_dist_n_jobs=-1),
        calculate_probabilities=False, verbose=True)
    topics, _ = topic_model.fit_transform(fit_docs, fit_emb)
    topics = np.asarray(topics)

    # centroid embedding per topik (tanpa outlier -1)
    tids = sorted(t for t in set(topics) if t != -1)
    cents = np.vstack([fit_emb[topics == t].mean(axis=0) for t in tids])
    cents /= np.linalg.norm(cents, axis=1, keepdims=True)

    info = topic_model.get_topic_info()
    top_words = {t: " ".join(w for w, _ in topic_model.get_topic(t)[:10])
                 for t in tids}
    info["top_words"] = info["Topic"].map(top_words)
    info.to_csv(TAB / f"topic_info_{asset}.csv", index=False)
    print(f"  [topic {asset}] {len(tids)} topik "
          f"(outlier fit-sample: {(topics == -1).mean():.1%})")
    del topic_model
    gc.collect()
    return tids, cents, top_words


def assign_all(emb, cents, batch=200_000):
    """Cosine-to-centroid untuk semua dokumen (GPU, chunked)."""
    cent_t = torch.tensor(cents, dtype=torch.float16, device=DEVICE)
    out = np.empty(len(emb), dtype=np.int32)
    sims = np.empty(len(emb), dtype=np.float32)
    for i in range(0, len(emb), batch):
        e = torch.tensor(emb[i:i + batch], dtype=torch.float16, device=DEVICE)
        s = e @ cent_t.T
        v, a = s.max(dim=1)
        out[i:i + batch] = a.cpu().numpy()
        sims[i:i + batch] = v.float().cpu().numpy()
        del e, s
    return out, sims


def label_topics(tids, top_words, st_model):
    """Meta-kategori topik via cosine ke seed embeddings."""
    h = st_model.encode(HEDGE_SEEDS, normalize_embeddings=True).mean(axis=0)
    s = st_model.encode(SPEC_SEEDS, normalize_embeddings=True).mean(axis=0)
    h /= np.linalg.norm(h); s /= np.linalg.norm(s)
    w_emb = st_model.encode([top_words[t] for t in tids],
                            normalize_embeddings=True)
    sim_h, sim_s = w_emb @ h, w_emb @ s
    meta = {}
    for k, t in enumerate(tids):
        if max(sim_h[k], sim_s[k]) < META_MARGIN:
            meta[t] = "NOISE"
        else:
            meta[t] = "HEDGE" if sim_h[k] >= sim_s[k] else "SPEC"
    return meta


# ------------------------------------------------------------------- finbert
FB_CHUNK = 20_000   # checkpoint FinBERT tiap chunk ini


def run_finbert(texts):
    """FinBERT INKREMENTAL ke memmap int8; resume otomatis jika crash."""
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer)
    n = len(texts)
    dat = CKPT / "finbert_scores.dat"
    done_f = CKPT / "finbert_scores.done"
    scores = np.memmap(dat, dtype=np.int8, mode=("r+" if dat.exists() else "w+"),
                       shape=(n,))
    start = int(done_f.read_text()) if done_f.exists() else 0
    if start >= n:
        print(f"  [finbert] lengkap dari checkpoint ({n:,})")
        return np.asarray(scores)
    if start:
        print(f"  [finbert] resume dari {start:,}/{n:,}")

    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = (AutoModelForSequenceClassification
             .from_pretrained("ProsusAI/finbert", dtype=torch.float16)
             .to(DEVICE).eval())
    lab2score = np.array([1, -1, 0], dtype=np.int8)  # positive, negative, neutral

    t0, B = time.time(), 64
    with torch.no_grad():
        for i in range(start, n, FB_CHUNK):
            j = min(i + FB_CHUNK, n)
            for k in range(i, j, B):
                m = min(k + B, j)
                enc = tok(texts[k:m], truncation=True, max_length=128,
                          padding=True, return_tensors="pt").to(DEVICE)
                pred = model(**enc).logits.argmax(dim=-1).cpu().numpy()
                scores[k:m] = lab2score[pred]
            scores.flush()
            done_f.write_text(str(j))
            rate = (j - start) / (time.time() - t0)
            eta = (n - j) / rate / 60 if rate else 0
            print(f"  [finbert] {j:,}/{n:,} ({rate:.0f} dok/s, sisa ~{eta:.0f} mnt)",
                  flush=True)
            torch.cuda.empty_cache()
            gpu_guard()
    return np.asarray(scores)


# ---------------------------------------------------------------------- main
def main():
    from sentence_transformers import SentenceTransformer

    CKPT.mkdir(exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE} | "
          f"{torch.cuda.get_device_name(0) if DEVICE == 'cuda' else ''}")

    df = pd.read_parquet(IN)
    df = df.sort_values(["asset", "date"]).reset_index(drop=True)
    print(f"{len(df):,} komentar dimuat")

    st_model = SentenceTransformer(EMB_MODEL, device=DEVICE)

    parts = []
    for asset, g in df.groupby("asset"):
        print(f"\n=== ASET: {asset} ({len(g):,} komentar) ===")
        g = g.reset_index(drop=True)
        texts = g["text"].tolist()

        emb = embed_texts(texts, st_model, asset)
        tids, cents, top_words = fit_topics(texts, emb, asset)
        assign, sims = assign_all(emb, cents)
        meta_map = label_topics(tids, top_words, st_model)

        g["topic"] = [tids[a] for a in assign]
        g["meta"] = g["topic"].map(meta_map)
        g.loc[sims < SIM_THRESHOLD, "meta"] = "NOISE"
        print("  meta distribusi:",
              g["meta"].value_counts(normalize=True).round(3).to_dict())
        parts.append(g)
        del emb
        gc.collect()
        torch.cuda.empty_cache()

    df = pd.concat(parts, ignore_index=True)
    keep = df[df["meta"] != "NOISE"].reset_index(drop=True)
    print(f"\nNon-NOISE: {len(keep):,} / {len(df):,} "
          f"({len(keep)/len(df):.1%}) -> FinBERT")

    keep["sent"] = run_finbert(keep["text"].tolist())

    keep.drop(columns=["text"]).to_parquet(
        PROC / "comments_scored.parquet", index=False)

    # ---- agregasi harian --------------------------------------------------
    grp = keep.groupby(["asset", "date"])
    daily = pd.DataFrame({
        "pos": grp.apply(lambda x: (x["sent"] == 1).sum(), include_groups=False),
        "neg": grp.apply(lambda x: (x["sent"] == -1).sum(), include_groups=False),
        "n": grp.size(),
        "hedge_share": grp.apply(lambda x: (x["meta"] == "HEDGE").mean(),
                                 include_groups=False),
    }).reset_index()
    daily["sent_index"] = (daily["pos"] - daily["neg"]) / daily["n"]

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
