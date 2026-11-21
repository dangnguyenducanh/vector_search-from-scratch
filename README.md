# vector_search-from-scratch

Semantic search engine with an exact index and an HNSW index written from scratch in Python. NumPy stores the vectors and runs exact search, and Numba compiles the graph loops.
It supports deletes and metadata filters. There is no FAISS, no hnswlib and no vector DB; the only model is the sentence encoder that turns text into vectors.

Search `"something warm and cozy for winter"` and it returns `"chunky knit sweater for cold days"`: zero shared words, matched on meaning.

```
text ──> all-MiniLM-L6-v2 ──> 384-d vector ──> L2-normalize once
                                                  │
                 FlatIndex (exact, one matmul) ◄──┴──► HNSWIndex (approximate, graph walk)
```

```bash
pip install -r requirements.txt
python cli.py build examples/products.txt products.npz
python cli.py search products.npz "something warm and cozy for winter" -k 3
python test_engine.py                                          # unit tests (~10s; the first run compiles)
python real_benchmark.py                                       # AG News 10k/50k/120k (embeds once, caches in data/)
python real_benchmark.py 120000 --filters --deletes            # filtered search + delete/repair tables
python recall_benchmark.py --ref ../vector-search-engine       # reference repo's eval, both engines
python real_benchmark.py 10000 --ref ../vector-search-engine   # AG News 10k, both engines
```

```python
from hnsw import HNSWIndex

idx = HNSWIndex(dim=384)                                    # M=16, ef_construction=100
idx.add(vectors, payloads=[{"text": t, "topic": c} for t, c in zip(texts, topics)])
idx.search(query, k=5, ef=64)                               # [(id, cosine), ...] best first
idx.search(query, k=5, filter=lambda p: p["topic"] == "Sports")  # or a bool mask over ids
idx.delete([3, 17])                                         # tombstone + repair the graph
idx.save("index.npz"); idx = HNSWIndex.load("index.npz")
```

## How it works

| Decision | Why |
|---|---|
| Normalize every vector once on insert | cosine(a, b) becomes a·b. No magnitudes recomputed per comparison. |
| Graph in flat int32 arrays. Layer 0 is `n × 2M`; upper layers keep a row only for the ~1/M nodes that reach them | No Python object per link. Each layer is a real graph, and upper layers are sparse long-range "express lanes". |
| Hot loops compiled with Numba (`@njit`): beam search, neighbor selection, insert, delete repair | The pure-NumPy version of the same algorithm paid Python overhead on every hop: 98 s to build 120k vectors. Compiled, it takes 22 s. |
| Level ~ `floor(-ln U / ln M)` | The paper's m_L = 1/ln M: P(level ≥ l) = M^-l, so ~log_M N layers. |
| Neighbor selection = diversity heuristic (paper Alg. 4) | Drop a candidate that is closer to a neighbor already kept than to the new node. Links spread out in different directions, so clusters stay connected. Ties are kept, as in hnswlib; otherwise exact duplicates leave a node with one link. |
| Back-links shrunk with the same heuristic, cap M (2M on layer 0) | Degree stays bounded without cutting the only bridge out of a cluster. |
| Beam search (`ef`) per layer on insert, greedy above layer 0 on query | The paper's insert/search (Alg. 1, 2, 5). |
| Persistence: `np.savez` + JSON meta, `allow_pickle=False`; `load()` checks every id, count and shape | Loading a pickle can execute code. The compiled loops don't bounds-check, so a corrupt or crafted file must fail on load, not read past an array. Saves write a temp file and swap it in, so a failed save keeps the old file. |

## Deletes

`delete(ids)` tombstones the ids and repairs the graph in the same call:

1. **Local repair** (as in FreshDiskANN). Every live node that linked to a deleted node takes that node's live neighbors as candidates instead, re-pruned with the same heuristic. Deleted nodes lose all their links, so a search never walks through one.
2. **Connectivity check.** One pass finds the entry's component on layer 0: the nodes the entry reaches that can also walk back to it. Every other live node (an orphan, a dead end, an island that the local repair left behind) is re-inserted with links into the component. If no member has room for the link back, the node is spliced into its nearest member's link `a → b` as `a → u → b`, so no existing path breaks.
3. If the entry point was deleted, the highest remaining node takes over. An entry left without links on a layer is re-linked there.

After every delete, every live node is reachable from the entry and can walk back to it. The tests cover the cases that broke a first version: everything around the entry deleted, everything around an upper-layer node deleted, 95% of the index deleted, and duplicate vectors. A fuzzer ran 3,000 random add/delete/filter/save/load sequences on the final code (2–8 dims, M down to 2, duplicate and zero vectors) without a violation. Ids are never reused, and deleted vectors stay in memory until you rebuild.

**AG News 120k, ef=64**, recall@10 against exact top-10 over the live documents (`real_benchmark.py 120000 --deletes`):

| deleted | repair in place | recall | rebuild from scratch | recall |
|---:|---:|---:|---:|---:|
| 10% | 1.2 s | 0.980 | 23.0 s | 0.983 |
| 50% | 1.3 s | 0.973 | 9.9 s | 0.986 |

Repair is 7–20× faster than a rebuild and loses 0.003–0.013 recall. The gap grows with smaller M or lower ef. Each `delete()` call scans the graph once, so batch your deletes.

## Metadata filtering

`search(q, filter=...)` takes a bool mask over ids, or a predicate on the payload. A small cost model picks one of two paths per query:

- **Graph walk.** The normal beam search, except only allowed nodes enter the results. Disallowed nodes are still walked through, so a filter never cuts the graph into pieces. Cost grows like ef / s for selectivity s.
- **Exact scan.** One matmul over the allowed rows. Cost grows like n · s.

The planner scans when `n_allowed × s < 10 × ef`. The constant 10 is where the two costs crossed on AG News 120k.

**AG News 120k, ef=64.** Each document gets a random tag, uncorrelated with its text, and the filter keeps a fraction of them. Cells show recall@10 against exact filtered top-10 · ms/query (`real_benchmark.py 120000 --filters`):

| allowed | rows | graph walk | exact scan | planner picks |
|---:|---:|---|---|---|
| 50% | 60,151 | 0.990 · 0.31 ms | 1.000 · 33.0 ms | graph |
| 20% | 24,129 | 0.994 · 0.59 ms | 1.000 · 10.6 ms | graph |
| 10% | 12,192 | 0.996 · 1.00 ms | 1.000 · 2.77 ms | graph |
| 5% | 6,022 | 0.996 · 1.75 ms | 1.000 · 0.86 ms | scan |
| 1% | 1,215 | 0.999 · 6.63 ms | 1.000 · 0.13 ms | scan |
| 0.1% | 111 | 1.000 · 37.7 ms | 1.000 · 0.03 ms | scan |

The planner picks the faster path in every row. The worst query it serves is 1.0 ms, and recall stays ≥ 0.99. A graph walk alone would take 38 ms at 0.1%, and a scan alone would take 33 ms at 50%.

## Checked against hnswlib (C++)

**Recall.** On 3k clustered 32-d vectors with `M=12, ef_construction=80`, recall@10 against exact search, mean ± std over 10 build seeds (hnswlib 0.8.0, 1 thread):

| ef | this repo | hnswlib |
|---:|---:|---:|
| 16 | 0.728 ± 0.021 | 0.724 ± 0.023 |
| 32 | 0.851 ± 0.021 | 0.847 ± 0.010 |
| 64 | 0.927 ± 0.016 | 0.927 ± 0.008 |
| 128 | 0.980 ± 0.007 | 0.980 ± 0.001 |

**Speed.** AG News 120k × 384, `M=16, ef_construction=100`, both on 1 thread, one query per call, hnswlib 0.8.0 built by pip from source:

| | this repo (Numba) | hnswlib (C++) |
|---|---:|---:|
| build | 22 s | 27 s |
| ef=16 | 0.917 · 0.057 ms | 0.914 · 0.066 ms |
| ef=32 | 0.960 · 0.090 ms | 0.957 · 0.101 ms |
| ef=64 | 0.982 · 0.145 ms | 0.982 · 0.172 ms |
| ef=128 | 0.994 · 0.264 ms | 0.995 · 0.298 ms |

The graph is the same quality as hnswlib's. On this machine the compiled loops also run as fast, slightly faster here: a single-thread run of one build, not a general claim.

## Benchmark: 120k AG News articles, real sentence embeddings

AG News train set (120k articles) embedded with all-MiniLM-L6-v2 (384-d). Queries are the first 500 test-set articles.
Metric is recall@10 against exact cosine top-10, with `M=16, ef_construction=100`. Both sides run on **one CPU thread**,
one query at a time, and the baseline is a vectorized NumPy matmul + `argpartition`, not a Python loop.

| vectors | build | exact ms/query | ef=16 | ef=32 | ef=64 | ef=128 |
|---:|---:|---:|---|---|---|---|
| 10k | 0.9s | 0.37 | 0.955 · 0.03 ms · 12× | 0.985 · 0.04 ms · 9× | 0.994 · 0.06 ms · 6× | 0.998 · 0.11 ms · 3× |
| 50k | 7.4s | 3.79 | 0.925 · 0.05 ms · 76× | 0.973 · 0.09 ms · 42× | 0.989 · 0.14 ms · 27× | 0.996 · 0.24 ms · 16× |
| 120k | 22s | 9.05 | 0.917 · 0.06 ms · 150× | 0.960 · 0.10 ms · 90× | **0.982 · 0.15 ms · 60×** | 0.994 · 0.28 ms · 32× |

Each cell is recall@10 · HNSW latency · speedup over exact search. Latencies are rounded to 0.01 ms, so the speedups below 0.1 ms are approximate.

- **Exact search is linear, HNSW is close to logarithmic.** From 10k to 120k vectors, exact search gets 24× slower; HNSW at ef=64 gets 2.5× slower.
- **Compiling the loops changed where HNSW wins.** The pure-NumPy version only tied a good brute force at 10k, because it paid Python overhead on every hop. Compiled, it is 6× faster at 10k, and at 120k ef=64 it went from 0.52 ms to 0.15 ms.
- **ef is the recall/latency setting.** It is chosen per query, with no rebuild.
- **Fine print.** If you batch queries, exact search drops to ~1.15 ms/query on one thread (one matrix-matrix multiply), so HNSW's edge over batched exact search is ~8× at ef=64, not 60×. About 3% of the test articles are near-duplicate wire re-posts of training articles.

## Head-to-head with the reference repo

[roridemonslayer/vector-search-engine](https://github.com/roridemonslayer/vector-search-engine) is a pure-Python HNSW that inspired this one.
Both engines run on the same machine and the same data, and they are scored against the same exact top-k. Both use the reference's
build settings (8 links per insert, cap 16, ef_construction=40). Here that is `HNSWIndex(M=8, ef_construction=40)`, since M0 = 2M = 16.
`--ref PATH` loads the reference's own `search.py`/`hnsw.py` unmodified.

**Real embeddings: 10k AG News articles, 500 queries, recall@10** (`real_benchmark.py 10000 --ref`)

| ef | this repo | reference | query speedup |
|---:|---|---|---:|
| 16 | **0.905** · 0.02 ms | 0.623 · 4.77 ms | ~240× |
| 32 | **0.954** · 0.03 ms | 0.793 · 6.71 ms | ~220× |
| 64 | **0.974** · 0.04 ms | 0.903 · 9.76 ms | ~240× |
| 128 | **0.989** · 0.07 ms | 0.959 · 15.2 ms | ~220× |

Build time: 0.3 s vs 77.8 s. Exact search: 0.34 ms (NumPy) vs 203 ms (the reference's pure-Python brute force).

**The reference's own eval: random 32-d vectors, 50 queries, recall@5** (`recall_benchmark.py --ref`)

| n | ef | this repo | reference |
|---:|---:|---|---|
| 1000 | 10 / 25 / 50 / 100 | 0.664 / 0.892 / 0.964 / 0.996 · 0.01–0.03 ms | 0.644 / 0.880 / 0.976 / 0.996 · 0.36–1.42 ms |
| 5000 | 10 / 25 / 50 / 100 | 0.468 / 0.732 / 0.876 / 0.960 · 0.01–0.04 ms | 0.392 / 0.704 / 0.864 / 0.944 · 0.47–2.27 ms |

What the comparison shows:
- **On uniform random vectors the two graphs are about equal.** At 1k the recall differences fall inside the noise of a 50-query eval (bootstrap 95% CI ±0.04–0.07), and the reference wins at ef=50. At 5k this repo leads by 0.01–0.08 in this run. That data has no clusters, so neighbor diversity has little to fix. The query speedup there (~35–55×) comes from compiled code.
- **On real, clustered embeddings the graph quality differs.** At the same settings this repo gets 0.91 recall at ef=16, against 0.62 for the reference. The likely cause is that keeping only the 16 nearest links walls clusters off from each other, while picking diverse links keeps them connected. Per-layer link lists also differ from the reference, and no ablation separates the two effects.
- **The reference's "HNSW beats brute force" depends on its baseline.** At 5k × 32-d, NumPy exact search takes 0.02 ms: about as fast as this repo's HNSW, and 20× faster than the reference's. The reference's HNSW only wins against its own pure-Python brute force (10 ms).

## Project structure

```
vector_search-from-scratch/
├── search.py            # vector store (tombstones, filter masks) + exact FlatIndex (one matmul + argpartition)
├── hnsw.py              # HNSWIndex: array graph, Numba loops, diversity heuristic, deletes + repair, filters, .npz save/load
├── real_benchmark.py    # AG News real embeddings: recall@10 + latency vs exact; --filters, --deletes, --ref
├── recall_benchmark.py  # the reference repo's eval protocol on both engines (--ref)
├── test_engine.py       # unit tests (python -m pytest -q, or python test_engine.py)
├── cli.py               # build an index from a text file, search it
├── examples/products.txt
├── requirements.txt
└── README.md
```

## Not built (yet)

- Reusing deleted slots. Deleted vectors stay in memory until a rebuild.
- Multi-threaded build. The compiled loops run on one core.
- Vector compression (PQ / int8).
- Connectivity checks on insert. Like the paper and hnswlib, insert can leave a rare unreachable node; `delete()` fixes those too.

## Stack

Python, NumPy, [Numba](https://numba.pydata.org/), [`sentence-transformers`](https://www.sbert.net/) (`all-MiniLM-L6-v2`) for embeddings only, Hugging Face `datasets` for AG News.
