"""Recall@10 + latency of HNSW vs exact search on real sentence embeddings (AG News).

python real_benchmark.py                                   # sizes 10k / 50k / 120k, this engine
python real_benchmark.py 10000 --ref ../vector-search-engine  # + the reference engine, same eval
Embeddings are cached in data/ so re-runs skip the model.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one core each: both HNSWs are single-threaded

import argparse

import numpy as np

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sizes", nargs="*", type=int)
    p.add_argument("--ref", help="path to a clone of roridemonslayer/vector-search-engine")
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


if __name__ == "__main__":
    main()
