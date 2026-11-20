# vector_search-from-scratch

Semantic search engine with an exact index and an HNSW index written from scratch in Python + NumPy.
No FAISS, no hnswlib, no vector DB. The only model is the sentence encoder that turns text into vectors.

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
python test_engine.py                                          # unit tests, ~2s
python real_benchmark.py                                       # AG News 10k/50k/120k (embeds once, caches in data/)
python recall_benchmark.py --ref ../vector-search-engine       # reference repo's eval, both engines
python real_benchmark.py 10000 --ref ../vector-search-engine   # AG News 10k, both engines
```

## How it works

| Decision | Why |
|---|---|
| Normalize every vector once on insert | cosine(a, b) becomes a·b. No magnitudes recomputed per comparison. |
| Score a whole neighbor list in one call (`vecs[nbrs] @ q`) | Python pays per *hop*, NumPy pays per *dimension*. |
| Per-layer adjacency `links[node][level]` | Each layer is a real graph. Upper layers are sparse long-range "express lanes". |
| Level ~ `floor(-ln U / ln M)` | The paper's m_L = 1/ln M: P(level ≥ l) = M^-l, so ~log_M N layers. |
| Neighbor selection = diversity heuristic (paper Alg. 4) | Keep a candidate only if it is closer to the new node than to every neighbor already kept. Links spread out in different directions, so clusters stay connected. |
| Back-links shrunk with the same heuristic, cap M (2M on layer 0) | Degree stays bounded without cutting the only bridge out of a cluster. |
| Beam search (`ef`) per layer on insert, greedy ef=1 above layer 0 on query | The paper's insert/search (Alg. 1, 2, 5). |
| Persistence: `np.savez` (CSR-flattened links) + JSON meta, `allow_pickle=False` | Loading a pickle can execute code; an `.npz` of int/float arrays cannot. |

**Checked against hnswlib (C++).** On 3k clustered 32-d vectors with `M=12, ef_construction=80`, recall@10 vs exact,
mean ± std over 10 build seeds (hnswlib 0.8.0, 1 thread):

| ef | this repo | hnswlib |
|---:|---:|---:|
| 16 | 0.728 ± 0.021 | 0.728 ± 0.029 |
| 32 | 0.851 ± 0.021 | 0.854 ± 0.017 |
| 64 | 0.927 ± 0.016 | 0.931 ± 0.013 |
| 128 | 0.980 ± 0.007 | 0.978 ± 0.007 |

Recall matches within 0.004 at every ef; the only gap is Python vs C++ speed.

## Benchmark: 120k AG News articles, real sentence embeddings

AG News train set (120k articles) embedded with all-MiniLM-L6-v2 (384-d). Queries are the first 500 test-set articles.
Metric is recall@10 against exact cosine top-10, with `M=16, ef_construction=100`. Both sides run on **one CPU thread**,
one query at a time, and the baseline is a vectorized NumPy matmul + `argpartition`, not a Python loop.

| vectors | build | exact ms/query | ef=16 | ef=32 | ef=64 | ef=128 |
|---:|---:|---:|---|---|---|---|
| 10k | 6s | 0.36 | 0.957 · 0.13 ms · 2.8× | 0.985 · 0.20 ms · 1.7× | 0.994 · 0.35 ms · 1.0× | 0.998 · 0.62 ms · 0.6× |
| 50k | 37s | 3.52 | 0.924 · 0.18 ms · 19.6× | 0.973 · 0.27 ms · 13.1× | 0.989 · 0.46 ms · 7.7× | 0.996 · 0.82 ms · 4.3× |
| 120k | 98s | 8.62 | 0.917 · 0.20 ms · 43.1× | 0.960 · 0.31 ms · 28.0× | **0.982 · 0.52 ms · 16.6×** | 0.994 · 0.94 ms · 9.1× |

Each cell is recall@10 · HNSW latency · speedup over exact search.

- **Exact search is linear, HNSW is close to logarithmic.** From 10k to 120k vectors, exact search gets 24× slower; HNSW at ef=64 gets only 1.5× slower.
- **At 10k, a good brute force ties HNSW.** The whole matrix fits in cache. HNSW pays Python overhead on every hop, so it only wins once N is large or ef is small.
- **ef is the recall/latency setting.** It is chosen per query, with no rebuild.
- **Fine print:** if you batch queries, exact search drops to ~1.15 ms/query on one thread (one matrix-matrix multiply), so HNSW's edge over batched exact is ~2×, not 16×. About 3% of the test articles are near-duplicate wire re-posts of training articles.

## Head-to-head with the reference repo

[roridemonslayer/vector-search-engine](https://github.com/roridemonslayer/vector-search-engine) is a pure-Python HNSW that inspired this one.
Both engines run on the same machine and the same data, and they are scored against the same exact top-k. Both use the reference's
build settings (8 links per insert, cap 16, ef_construction=40): here that is `HNSWIndex(M=8, ef_construction=40)`, since M0 = 2M = 16.
`--ref PATH` loads the reference's own `search.py`/`hnsw.py` unmodified.

**Real embeddings: 10k AG News articles, 500 queries, recall@10** (`real_benchmark.py 10000 --ref`)

| ef | this repo | reference | query speedup |
|---:|---|---|---:|
| 16 | **0.906** · 0.12 ms | 0.623 · 4.75 ms | 40× |
| 32 | **0.954** · 0.18 ms | 0.793 · 6.75 ms | 38× |
| 64 | **0.973** · 0.29 ms | 0.903 · 9.78 ms | 34× |
| 128 | **0.988** · 0.50 ms | 0.959 · 15.0 ms | 30× |

Build time: 2.3 s vs 77.9 s (34×). Exact search: 0.34 ms (NumPy) vs 200 ms (the reference's pure-Python brute force).

**The reference's own eval: random 32-d vectors, 50 queries, recall@5** (`recall_benchmark.py --ref`)

| n | ef | this repo | reference |
|---:|---:|---|---|
| 1000 | 10 / 25 / 50 / 100 | 0.664 / 0.892 / 0.964 / 0.996 · 0.06–0.30 ms | 0.644 / 0.880 / 0.976 / 0.996 · 0.37–1.42 ms |
| 5000 | 10 / 25 / 50 / 100 | 0.468 / 0.732 / 0.876 / 0.960 · 0.07–0.35 ms | 0.392 / 0.704 / 0.864 / 0.944 · 0.51–2.52 ms |

What the comparison shows:
- **On uniform random vectors the two graphs are about equal.** At 1k the recall differences fall inside the noise of a 50-query eval (bootstrap 95% CI ±0.04–0.07), and the reference wins ef=50 in all 5 build seeds tried. At 5k this repo leads by 0.01–0.08 in this run. That data has no clusters, so neighbor diversity has little to fix. The 5–7× query speedup comes from NumPy math.
- **On real, clustered embeddings the graph quality differs.** At the same settings this repo gets 0.91 vs 0.62 recall at ef=16. The likely cause: keeping only the 16 nearest links walls clusters off from each other, while picking diverse links keeps them connected. Per-layer link lists also differ from the reference, and no ablation separates the two effects.
- **The reference's "HNSW beats brute force" depends on its baseline.** At 5k × 32-d, NumPy exact search takes 0.02 ms, faster than either HNSW. HNSW only wins against the reference's pure-Python brute force (10 ms).

## Project structure

```
vector_search-from-scratch/
├── search.py            # exact search: normalize, FlatIndex (one matmul + argpartition)
├── hnsw.py              # HNSWIndex: per-layer graph, diversity heuristic, beam search, .npz save/load
├── real_benchmark.py    # AG News real embeddings: recall@10 + latency vs exact (--ref: + reference engine)
├── recall_benchmark.py  # the reference repo's eval protocol on both engines (--ref)
├── test_engine.py       # unit tests (python -m pytest -q, or python test_engine.py)
├── cli.py               # build an index from a text file, search it
├── examples/products.txt
├── requirements.txt
└── README.md
```

## Not built (yet)

- Deletes (HNSW needs tombstones + repair).
- Metadata filtering.
- Compiled inner loop (Numba/C) for faster builds.

## Stack

Python, NumPy, [`sentence-transformers`](https://www.sbert.net/) (`all-MiniLM-L6-v2`) for embeddings only, Hugging Face `datasets` for AG News.
