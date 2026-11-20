# tests for the engine: python -m pytest -q   (or python test_engine.py)
import os
import tempfile

import numpy as np

from hnsw import HNSWIndex
from search import FlatIndex, normalize


def test_cosine_is_dot_after_normalize():
    a, b = normalize([3, 4]), normalize([6, 8])
    assert abs(float(a @ b) - 1.0) < 1e-6  # same direction, different length


def test_flat_is_exact():
    idx = FlatIndex(3)
    idx.add([[0.9, 0.8, 0.1], [0.8, 0.9, 0.1], [0.1, 0.8, 0.9]], ["tee A", "tee B", "gown"])
    top = idx.search([0.85, 0.9, 0.05], k=3)
    assert [idx.payloads[i] for i, _ in top] == ["tee B", "tee A", "gown"]


def _clustered(n, dim, rng):
    # clustered data is where "keep the closest M" pruning disconnects the graph
    centers = rng.standard_normal((20, dim))
    return centers[rng.integers(0, 20, n)] + 0.3 * rng.standard_normal((n, dim))


def test_hnsw_recall_vs_flat():
    rng = np.random.default_rng(0)
    data, queries = _clustered(3000, 32, rng), _clustered(100, 32, rng)
    flat, hnsw = FlatIndex(32), HNSWIndex(32, M=12, ef_construction=80)
    flat.add(data)
    hnsw.add(data)
    hits = sum(len({i for i, _ in flat.search(q, 10)} & {i for i, _ in hnsw.search(q, 10, ef=128)})
               for q in queries)
    recall = hits / (10 * len(queries))
    assert recall > 0.97, recall  # hnswlib (C++), same params, 10 build seeds: 0.978 +- 0.007


def test_hnsw_finds_exact_match_and_round_trips():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((500, 16))
    idx = HNSWIndex(16)
    idx.add(data, [f"doc{i}" for i in range(500)])
    assert idx.search(data[123], k=1)[0][0] == 123
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "idx")  # no extension: save/load must use the exact path
        idx.save(path)
        back = HNSWIndex.load(path)
    assert back.links == idx.links and back.entry == idx.entry and back.payloads == idx.payloads
    assert back.search(data[7], k=5) == idx.search(data[7], k=5)
    back.add(rng.standard_normal((10, 16)))  # loaded index stays writable
    assert len(back) == 510


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
