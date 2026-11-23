# C++ backend experiment: plan, eval, results

## Goal

Lower per-query latency (and build time) by moving the graph loops from Numba to C++,
with no change to the algorithm, the graph layout, the Python API or the file format.
Then measure it against the Numba version on the same data, same machine, same graph seed.

## Where the time goes (before)

AG News 120k × 384, one thread, `search(q, k=10, ef)`:

| ef | `search()` (Python API) | Numba kernel call alone |
|---:|---:|---:|
| 16 | ~80 µs | ~75 µs |
| 64 | ~210 µs | ~210 µs |

(Busy shared host, so absolute numbers run ~1.5× above the README's.) The Python wrapper
(normalize, planner, result list) costs ~5–10 µs. Numba dispatch is 0.5 µs. **The kernel is the
latency**, so a C++ port only helps if the kernel itself gets faster, not by removing Python.

## Design

- `cpp/hnsw.cpp`: one file, `extern "C"`, line-for-line port of the `@njit` kernels:
  `_search_layer`, `_descend`, `_select`, `_connect`, `_insert`, `_query`, `_repair` (+ `_component`, `_relink`).
- **Same memory.** The kernels take raw pointers into the NumPy arrays `HNSWIndex` already owns
  (`vecs, l0, c0, up, cu, slot, levels, visited, dead`). Save/load, the filter planner, growth,
  input checks stay in Python and are shared by both backends.
- **Binding: `ctypes` (stdlib), inside `hnsw.py`.** No pybind11/nanobind dependency. Pointers are packed into one
  struct, rebuilt only after arrays reallocate (`add`, `load`), so a query call passes ~8 scalars.
- `HNSWIndex(..., backend="numba" | "cpp")`. Default stays `numba` until the eval says otherwise.
- Build: `make -C cpp` → `cpp/libhnsw.so`, loaded by `hnsw.py` (`g++ -O3 -march=native`, reassoc/contract only, like Numba's flags).

What C++ changes inside the kernel (the only places a speedup can come from):

1. Heaps on reused `std::vector`s with `push_heap/pop_heap` instead of Numba typed lists of tuples
   (allocation per search).
2. Software prefetch of neighbor vectors before scoring them (as hnswlib does).
3. AVX2/FMA dot product with 4 accumulators (384-d = 48 FMAs, latency-bound with 1 accumulator).
4. No per-call `np.empty` for results / candidate buffers.

Ties break exactly as Numba's `heapq` on `(score, id)` tuples, so the two backends build the same graph
except where float summation order flips a comparison.

## Eval

`python cpp_benchmark.py` (AG News, cached embeddings), both backends **interleaved** in the same
process, pinned to one core, median of repeats (the host is shared and noisy):

1. **Parity.** Same seed → same levels. Report layer-0 edge overlap between the two graphs and
   recall@10 at ef ∈ {16, 32, 64, 128} against exact top-10 (should match within noise).
2. **Latency.** `search()` µs/query through the Python API, per ef, both backends.
3. **Build.** `add()` of all vectors, both backends.
4. **Deletes.** 10% delete + repair time and post-repair recall.
5. **Correctness.** `test_engine.py` runs every HNSW test against both backends.

## Not in scope

Multi-threaded build, batched multi-threaded queries, int8/PQ. C++ makes all three straightforward;
they change throughput, not single-query latency, so they are listed rather than built.

Also `cpp/bench.cpp` (`make -C cpp bench` → `cpp/hnsw_bench`): a standalone C++ program (no Python at all) that loads raw vectors exported by
`python cpp_benchmark.py --export`, builds with the same layer draws and times 500 single queries.

## Results (2026-11-23)

Host caveat: a shared 24-core box at load average 40–49 during every run (README build of 22 s took
47–63 s here), so absolute µs are ~2× the README's and run-to-run noise is ±15%. Ratios from interleaved
runs on one pinned core are the trustworthy part.

**Correctness / parity: identical.**

- `test_engine.py`: 11/11 on both backends; also 11/11 under ASan + UBSan.
- Random add/delete/filter fuzz, 300 sequences (2–8 dims, M down to 2, duplicate and zero vectors), under ASan:
  0 cut-off nodes after `delete()` on both backends; the same 153 after a plain insert on both (known insert
  behavior, see README "Not built").
- Same seed → same layers; layer-0 edge overlap between the two graphs 0.998 (differences = float summation order).
- Recall@10, 120k: 0.917 / 0.960 / 0.982 / 0.994 at ef 16/32/64/128 for Numba, C++ via Python, and pure C++ alike.

**Latency: no win.** AG News 120k, M=16, efC=100, 1 core.

| ef | Numba (Python API) | C++ (Python API) | C++ standalone binary | C++ / Numba |
|---:|---:|---:|---:|---:|
| 16 | 119 µs | 175 µs | 170 µs | 0.68× |
| 32 | 192 µs | 273 µs | 159 µs | 0.70× |
| 64 | 340 µs | 461 µs | 252 µs | 0.74× |
| 128 | 675 µs | 774 µs | 545 µs | 0.87× |
| build | 47.5 s | 63.5 s | 58.2 s | 0.75× |
| delete 10% + repair | 2.13 s | 2.78 s | | 0.77× |

(The standalone column ran at a different minute, so compare it only loosely.)
Bare kernels, best of 20 passes at ef=64, alternating processes: Numba 233–254 µs, C++ 248–301 µs,
C++ without prefetch 312–317 µs. So the prefetch helps C++, but C++ at best ties Numba.

**Why.** Numba already compiles these loops with LLVM to the same kind of machine code C++ gets
(the README had it tied with hnswlib's C++). The query is memory-bound: each hop reads a random
1.5 KB vector out of a 184 MB array, and the dot product and heap are a small share of that.
Python overhead is ~5–10 µs of a 120–675 µs query. A language change can't remove the cache misses.

**Decision.** Numba stays the engine on `main`. This C++ backend lives only on the
`experiment/cpp-backend` branch, as an opt-in (`HNSWIndex(dim, backend="cpp")` or `HNSW_BACKEND=cpp`)
with the tests running on both.

## What would actually cut latency

Fewer bytes per hop, which C++ makes easy (explicit int8/fp16 SIMD):

1. **int8 (or fp16) vectors for the graph walk**, re-rank the final ef candidates in float32:
   4× (2×) less memory traffic per distance. This is the expected big win on this workload.
2. **Reorder node ids by graph BFS** so neighbors sit close in memory (better cache/TLB locality).
3. Multi-threaded build and batched queries: throughput, not single-query latency.
