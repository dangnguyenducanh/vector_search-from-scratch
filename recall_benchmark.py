"""HNSW recall@5 + speed vs exact search, using the reference repo's eval protocol.

python recall_benchmark.py                                # this engine only
python recall_benchmark.py --ref ../vector-search-engine  # + the reference engine, same data

Protocol copied from roridemonslayer/vector-search-engine recall_benchmark.py:
1000 and 5000 random 32-d vectors (random.Random(7), values in [-0.5, 0.5)),
50 queries (random.Random(99)), recall@5 against exact cosine top-5, ef in 10/25/50/100.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one core each: both HNSWs are single-threaded

import argparse
import importlib.util
import random
import sys
import time

import numpy as np

from hnsw import HNSWIndex
from search import FlatIndex

DIMS, QUERIES, K = 32, 50, 5


def load_ref(path):
    """Import the reference repo's search.py + hnsw.py under other names (ours share them)."""
    def load(name, file):
        spec = importlib.util.spec_from_file_location(name, os.path.join(path, file))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    ours = sys.modules.get("search")
    ref_search = load("ref_search", "search.py")
    sys.modules["search"] = ref_search  # their hnsw.py does `from search import ...`
    try:
        ref_hnsw = load("ref_hnsw", "hnsw.py")
    finally:
        sys.modules["search"] = ours
    return ref_search, ref_hnsw


def ours(M=16, ef_construction=100):
    def build(data):
        idx = HNSWIndex(data.shape[1], M=M, ef_construction=ef_construction)
        idx.add(data)
        return lambda q, k, ef: [i for i, _ in idx.search(q, k, ef)]
    return build


def theirs(ref_hnsw):
    def build(data):
        random.seed(0)  # their layer coin-flips use the global RNG
        graph, layers = {}, {}
        for i, v in enumerate(data.tolist()):
            ref_hnsw.insert(graph, layers, i, v)
        entry = max(layers, key=layers.get)  # how their benchmark picks the start node
        return lambda q, k, ef: [n for n, _ in ref_hnsw.layered_search(
            graph, q.tolist(), layers, start=entry, k=k, ef=ef)]
    return build


def exact_topk(data, queries, k):
    """Ground truth in float64 (same math as either brute force, no float32 ties)."""
    d = data / np.linalg.norm(data, axis=1, keepdims=True)
    q = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    return [set(row[:k]) for row in np.argsort(-(q @ d.T), axis=1)]


def brute_ms(fn, queries):
    t = time.perf_counter()
    for q in queries:
        fn(q)
    return 1000 * (time.perf_counter() - t) / len(queries)


def evaluate(name, build, data, queries, truth, k, efs):
    t = time.perf_counter()
    search = build(data)
    build_s = time.perf_counter() - t
    for ef in efs:
        t = time.perf_counter()
        found = [set(search(q, k, ef)) for q in queries]
        ms = 1000 * (time.perf_counter() - t) / len(queries)
        recall = sum(len(a & b) for a, b in zip(truth, found)) / (k * len(queries))
        print(f"{name:<28} {build_s:>8.1f} {ef:>5} {recall:>9.3f} {ms:>9.2f}", flush=True)


def header(k):
    print(f"{'engine':<28} {'build s':>8} {'ef':>5} {'recall@' + str(k):>9} {'ms/query':>9}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", help="path to a clone of roridemonslayer/vector-search-engine")
    a = p.parse_args()
    ref = load_ref(a.ref) if a.ref else None

    for n in (1000, 5000):
        rng = random.Random(7)
        data = np.array([[rng.random() - 0.5 for _ in range(DIMS)] for _ in range(n)])
        qrng = random.Random(99)
        queries = np.array([[qrng.random() - 0.5 for _ in range(DIMS)] for _ in range(QUERIES)])
        truth = exact_topk(data, queries, K)

        flat = FlatIndex(DIMS)
        flat.add(data)
        print(f"\n{n} vectors, {DIMS} dims, {QUERIES} queries")
        print(f"exact, NumPy (this repo):        {brute_ms(lambda q: flat.search(q, K), queries):.2f} ms/query")
        if ref:
            catalog = [(i, v) for i, v in enumerate(data.tolist())]
            ms = brute_ms(lambda q: ref[0].search(q.tolist(), catalog, k=K), queries)
            print(f"exact, pure Python (reference):  {ms:.2f} ms/query")
        header(K)
        efs = (10, 25, 50, 100)
        evaluate("ours M=16 efC=100 (default)", ours(), data, queries, truth, K, efs)
        evaluate("ours M=8 efC=40 (matched)", ours(M=8, ef_construction=40), data, queries, truth, K, efs)
        if ref:
            evaluate("reference M=8 efC=40", theirs(ref[1]), data, queries, truth, K, efs)


if __name__ == "__main__":
    main()
