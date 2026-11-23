"""Numba kernels vs the C++ port (cpp/), same index, same data, same process.

make -C cpp && python cpp_benchmark.py            # AG News 120k (embeddings cached by real_benchmark.py)
python cpp_benchmark.py 10000 50000 --reps 7

Both backends run on one pinned core and are timed interleaved (numba, cpp, numba, ...);
each cell is the median over --reps passes of all 500 queries, so a busy host hits both alike.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import statistics
import time

import numpy as np

from hnsw import HNSWIndex
from real_benchmark import embeddings
from recall_benchmark import exact_topk
from search import normalize

K, EFS, BACKENDS = 10, (16, 32, 64, 128), ("numba", "cpp")


def timed(fn):
    t = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t


def edge_overlap(a, b):
    """Share of layer-0 links the two graphs have in common (Jaccard over all edges)."""
    same = total = 0
    for u in range(a.n):
        x, y = set(a.l0[u, : a.c0[u]].tolist()), set(b.l0[u, : b.c0[u]].tolist())
        same += len(x & y)
        total += len(x | y)
    return same / total


def run(data, queries, reps):
    n = len(data)
    truth = exact_topk(data.astype(np.float64), queries.astype(np.float64), K)
    HNSWIndex(data.shape[1], backend="numba").add(data[:200])  # compile / load the Numba cache first
    idx, build = {}, {}
    for b in BACKENDS:
        idx[b] = HNSWIndex(data.shape[1], backend=b)
        _, build[b] = timed(lambda: idx[b].add(data))
    assert np.array_equal(idx["numba"].levels, idx["cpp"].levels)  # same seed -> same layers
    print(f"\n{n:,} AG News docs x {data.shape[1]} (M=16, efC=100), {len(queries)} queries, 1 pinned core")
    print(f"build: numba {build['numba']:.1f} s · cpp {build['cpp']:.1f} s · "
          f"{build['numba'] / build['cpp']:.2f}x   layer-0 edge overlap {edge_overlap(idx['numba'], idx['cpp']):.3f}")
    print(f"{'ef':>4} {'recall numba':>13} {'recall cpp':>11} {'numba µs':>9} {'cpp µs':>8} {'speedup':>8}")
    for ef in EFS:
        us, rec = {b: [] for b in BACKENDS}, {}
        for _ in range(reps):
            for b in BACKENDS:
                found, s = timed(lambda: [idx[b].search(q, K, ef) for q in queries])
                us[b].append(1e6 * s / len(queries))
                rec[b] = sum(len(t & {i for i, _ in f}) for t, f in zip(truth, found)) / (K * len(queries))
        med = {b: statistics.median(us[b]) for b in BACKENDS}
        print(f"{ef:>4} {rec['numba']:>13.3f} {rec['cpp']:>11.3f} {med['numba']:>9.1f} {med['cpp']:>8.1f} "
              f"{med['numba'] / med['cpp']:>7.2f}x", flush=True)
    gone = np.random.default_rng(1).choice(n, n // 10, replace=False)
    live = np.setdiff1d(np.arange(n), gone)
    rows = exact_topk(data[live].astype(np.float64), queries.astype(np.float64), K)  # rows of data[live]
    truth = [set(live[list(t)].tolist()) for t in rows]
    cells = []
    for b in BACKENDS:
        _, s = timed(lambda: idx[b].delete(gone))
        found = [{i for i, _ in idx[b].search(q, K, 64)} for q in queries]
        cells.append(f"{b} {s:.2f} s · recall {sum(len(t & f) for t, f in zip(truth, found)) / (K * len(queries)):.3f}")
    print("delete 10% + repair, ef=64: " + "   ".join(cells))


def export(docs, queries):
    """Raw files for the pure C++ benchmark (cpp/bench.cpp), each = int32 rows, int32 cols, data.
    levels are the ones HNSWIndex(seed=0) draws, so hnsw_bench builds the same graph."""
    idx = HNSWIndex(docs.shape[1])
    lv = (-np.log(1.0 - idx.rng.random(len(docs))) * idx.mL).astype(np.int32)
    truth = np.array([sorted(t) for t in exact_topk(docs.astype(np.float64), queries.astype(np.float64), K)], np.int32)
    for name, arr in (("docs", normalize(docs)), ("queries", normalize(queries)), ("levels", lv[:, None]),
                      ("truth", truth)):
        with open(f"data/{name}.bin", "wb") as f:
            np.array(arr.shape, np.int32).tofile(f)
            np.ascontiguousarray(arr).tofile(f)
    print(f"wrote data/{{docs,queries,levels,truth}}.bin for {len(docs):,} docs: run make -C cpp bench && ./cpp/hnsw_bench data")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sizes", nargs="*", type=int)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--core", type=int, default=None, help="pin to this CPU (default: the first allowed one)")
    p.add_argument("--export", action="store_true", help="write data/*.bin for cpp/hnsw_bench (all 120k docs) and exit")
    a = p.parse_args()
    os.sched_setaffinity(0, {a.core if a.core is not None else min(os.sched_getaffinity(0))})
    docs, queries = embeddings()
    if a.export:
        return export(docs, queries)
    for n in a.sizes or [len(docs)]:
        run(docs[:n], queries, a.reps)


if __name__ == "__main__":
    main()
