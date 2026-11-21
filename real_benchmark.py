"""Recall@10 + latency of HNSW vs exact search on real sentence embeddings (AG News).

python real_benchmark.py                                   # sizes 10k / 50k / 120k, this engine
python real_benchmark.py 10000 --ref ../vector-search-engine  # + the reference engine, same eval
python real_benchmark.py 120000 --filters --deletes        # filtered search + delete/repair tables
Embeddings are cached in data/ so re-runs skip the model.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one core each: both HNSWs are single-threaded

import argparse
import time

import numpy as np

from hnsw import HNSWIndex
from recall_benchmark import brute_ms, evaluate, exact_topk, header, load_ref, ours, theirs
from search import FlatIndex

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
K, QUERIES, EFS = 10, 500, (16, 32, 64, 128)


def embeddings():
    os.makedirs("data", exist_ok=True)
    path = "data/agnews_minilm.npz"
    if not os.path.exists(path):
        from datasets import load_dataset
        from sentence_transformers import SentenceTransformer

        ds = load_dataset("fancyzhx/ag_news")
        docs, queries = ds["train"]["text"], ds["test"]["text"][:QUERIES]
        model = SentenceTransformer(MODEL)
        enc = lambda t: model.encode(t, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
        np.savez(path, docs=enc(docs).astype(np.float32), queries=enc(queries).astype(np.float32))
    z = np.load(path)
    return z["docs"], z["queries"]


def recall_ms(search, queries, truth):
    t = time.perf_counter()
    found = [{i for i, _ in search(q)} for q in queries]
    ms = 1000 * (time.perf_counter() - t) / len(queries)
    return sum(len(a & b) for a, b in zip(truth, found)) / (K * len(queries)), ms


def filter_table(data, queries, ef=64):
    """Each doc gets a random tag in [0, 1), uncorrelated with its text; the filter
    keeps tag < s. Graph walk vs exact scan of the allowed rows, and what the planner picks."""
    idx, flat = HNSWIndex(data.shape[1]), FlatIndex(data.shape[1])
    idx.add(data)
    flat.add(data)
    tag = np.random.default_rng(0).random(len(data))
    print(f"\nfiltered search, ef={ef}: recall@10 vs exact filtered top-10 · ms/query")
    print(f"{'allowed':>8} {'rows':>7} {'graph walk':>16} {'exact scan':>16} {'planner':>8}")
    for s in (0.5, 0.2, 0.1, 0.05, 0.01, 0.001):
        mask = tag < s
        rows = int(mask.sum())
        truth = [{i for i, _ in flat.search(q, K, filter=mask)} for q in queries]
        cells = []
        for cost in (0, float("inf")):  # force the graph walk, then force the scan
            idx.exact_cost = cost
            cells.append("%.3f · %.2f ms" % recall_ms(lambda q: idx.search(q, K, ef, filter=mask), queries, truth))
        idx.exact_cost = HNSWIndex.exact_cost
        pick = "scan" if rows * rows < idx.exact_cost * ef * len(idx) else "graph"
        print(f"{s:>8.1%} {rows:>7,} {cells[0]:>16} {cells[1]:>16} {pick:>8}", flush=True)


def delete_table(data, queries, ef=64):
    """Delete a random fraction, repair in place, compare with rebuilding on what is left."""
    print(f"\ndeletes, ef={ef}: recall@10 vs exact top-10 over the live docs")
    print(f"{'deleted':>8} {'repair':>8} {'recall':>7} {'rebuild':>8} {'recall':>7}")
    for frac in (0.1, 0.5):
        idx, flat = HNSWIndex(data.shape[1]), FlatIndex(data.shape[1])
        idx.add(data)
        flat.add(data)
        gone = np.random.default_rng(1).choice(len(data), int(frac * len(data)), replace=False)
        t = time.perf_counter()
        idx.delete(gone)
        repair_s = time.perf_counter() - t
        flat.delete(gone)
        truth = [{i for i, _ in flat.search(q, K)} for q in queries]
        live = np.flatnonzero(~idx.dead[: idx.n])
        t = time.perf_counter()
        fresh = HNSWIndex(data.shape[1])
        fresh.add(data[live])
        rebuild_s = time.perf_counter() - t
        r1, _ = recall_ms(lambda q: idx.search(q, K, ef), queries, truth)
        r2, _ = recall_ms(lambda q: [(int(live[i]), s) for i, s in fresh.search(q, K, ef)], queries, truth)
        print(f"{frac:>8.0%} {repair_s:>7.2f}s {r1:>7.3f} {rebuild_s:>7.1f}s {r2:>7.3f}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sizes", nargs="*", type=int)
    p.add_argument("--ref", help="path to a clone of roridemonslayer/vector-search-engine")
    p.add_argument("--filters", action="store_true", help="filtered-search table (random tag, selectivity sweep)")
    p.add_argument("--deletes", action="store_true", help="delete 10%% / 50%%: repair vs rebuild")
    a = p.parse_args()
    ref = load_ref(a.ref) if a.ref else None
    docs, queries = embeddings()

    for n in a.sizes or [10_000, 50_000, len(docs)]:
        data = docs[:n]
        truth = exact_topk(data.astype(np.float64), queries.astype(np.float64), K)
        flat = FlatIndex(data.shape[1])
        flat.add(data)
        print(f"\n{n:,} AG News docs x {data.shape[1]} dims (all-MiniLM-L6-v2), {len(queries)} queries")
        print(f"exact, NumPy (this repo):        {brute_ms(lambda q: flat.search(q, K), queries):.2f} ms/query")
        if ref:
            catalog = [(i, v) for i, v in enumerate(data.tolist())]
            ms = brute_ms(lambda q: ref[0].search(q.tolist(), catalog, k=K), queries[:20])
            print(f"exact, pure Python (reference):  {ms:.2f} ms/query  (20 queries)")
        header(K)
        evaluate("ours M=16 efC=100 (default)", ours(), data, queries, truth, K, EFS)
        if ref:
            evaluate("ours M=8 efC=40 (matched)", ours(M=8, ef_construction=40), data, queries, truth, K, EFS)
            evaluate("reference M=8 efC=40", theirs(ref[1]), data, queries, truth, K, EFS)
        if a.filters:
            filter_table(data, queries)
        if a.deletes:
            delete_table(data, queries)


if __name__ == "__main__":
    main()
