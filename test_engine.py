# tests for the engine: python -m pytest -q   (or python test_engine.py)
import os
import tempfile

import numpy as np

from hnsw import HNSWIndex, _links
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


def _recall(a, b, queries, k=10, ef=128, **kw):
    hits = sum(len({i for i, _ in a.search(q, k, **kw)} & {i for i, _ in b.search(q, k, ef=ef, **kw)})
               for q in queries)
    return hits / (k * len(queries))


def test_hnsw_recall_vs_flat():
    rng = np.random.default_rng(0)
    data, queries = _clustered(3000, 32, rng), _clustered(100, 32, rng)
    flat, hnsw = FlatIndex(32), HNSWIndex(32, M=12, ef_construction=80)
    flat.add(data)
    hnsw.add(data)
    recall = _recall(flat, hnsw, queries)
    assert recall > 0.97, recall  # hnswlib (C++), same params, 10 build seeds: 0.978 +- 0.007


def test_hnsw_finds_exact_match_and_round_trips():
    rng = np.random.default_rng(1)
    data = rng.standard_normal((1500, 16))
    idx = HNSWIndex(16)
    idx.add(data[:200], [f"doc{i}" for i in range(200)])
    idx.add(data[200:], [f"doc{i}" for i in range(200, 1500)])  # past the first 1024 rows: arrays grow
    assert idx.search(data[123], k=1)[0][0] == 123 and idx.search(data[1400], k=1)[0][0] == 1400
    idx.delete([3, 4])
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "idx")  # no extension: save/load must use the exact path
        idx.save(path)
        back = HNSWIndex.load(path)
    for a in HNSWIndex.ARRAYS:
        rows = idx.n_up if a in ("up", "cu") else idx.n
        assert np.array_equal(getattr(back, a), getattr(idx, a)[:rows]), a
    assert back.entry == idx.entry and back.payloads == idx.payloads and len(back) == 1498
    assert back.search(data[7], k=5) == idx.search(data[7], k=5)
    back.add(rng.standard_normal((10, 16)))  # loaded index stays writable
    assert len(back) == 1508


def test_failed_save_keeps_the_old_file_and_bad_files_fail_to_load():
    idx = HNSWIndex(4)
    idx.add(np.eye(4))
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "idx.npz")
        idx.save(path)
        with open(path, "rb") as f:
            before = f.read()
        for bad in (np.int64(7), ("a", 1)):  # not JSON / changed by JSON (tuple -> list)
            other = HNSWIndex(4)
            other.add([[1, 0, 0, 0]], [bad])
            try:
                other.save(path)
                raise AssertionError("expected TypeError")
            except TypeError:
                pass
            with open(path, "rb") as f:
                assert f.read() == before
        z = dict(np.load(path))
        z["l0"] = z["l0"] + 10**6  # neighbor ids past the end: the compiled loops would read garbage
        with open(path, "wb") as f:
            np.savez(f, **z)
        try:
            HNSWIndex.load(path)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_delete_repairs_graph():
    rng = np.random.default_rng(2)
    data, queries = _clustered(3000, 32, rng), _clustered(100, 32, rng)
    flat, hnsw = FlatIndex(32), HNSWIndex(32, M=12, ef_construction=80)
    flat.add(data)
    hnsw.add(data)
    gone = np.append(rng.choice(3000, 1500, replace=False), hnsw.entry)  # half, plus the entry point
    flat.delete(gone)
    hnsw.delete(gone)
    assert len(hnsw) == len(flat) == 3000 - len(set(gone.tolist()))
    assert not hnsw.dead[hnsw.entry]
    g = hnsw._g()
    for u in range(hnsw.n):  # dead nodes are fully unlinked: search can't walk into them
        for lv in range(hnsw.levels[u] + 1):
            nbrs = _links(g, u, lv)
            assert (len(nbrs) == 0) if hnsw.dead[u] else not hnsw.dead[nbrs].any()
    fresh = HNSWIndex(32, M=12, ef_construction=80)
    fresh.add(data[~hnsw.dead[: hnsw.n]])
    live = np.flatnonzero(~hnsw.dead[: hnsw.n])
    hits = sum(len({i for i, _ in flat.search(q, 10)} & {int(live[i]) for i, _ in fresh.search(q, 10, ef=128)})
               for q in queries)
    repaired = _recall(flat, hnsw, queries)
    assert repaired > hits / 1000 - 0.02, (repaired, hits / 1000)  # as good as rebuilding from scratch
    hnsw.add(data[:5])  # inserting after deletes still works
    assert hnsw.search(data[0], k=1)[0][0] == 3000


def _neighbors(idx, u):
    return [y for lv in range(idx.levels[u] + 1) for y in _links(idx._g(), u, lv).tolist()]


def _cut_off(idx):
    """Live nodes on layer 0 that the entry can't reach, or that can't walk back to it."""
    edges = {u: _links(idx._g(), u, 0).tolist() for u in range(idx.n)}
    back = {u: [] for u in range(idx.n)}
    for u, vs in edges.items():
        for v in vs:
            back[v].append(u)
    reached = []
    for adj in (edges, back):
        seen, todo = {idx.entry}, [idx.entry]
        while todo:
            for y in adj[todo.pop()]:
                if y not in seen:
                    seen.add(y)
                    todo.append(y)
        reached.append(seen)
    return [u for u in np.flatnonzero(~idx.dead[: idx.n]) if u not in reached[0] or u not in reached[1]]


def test_delete_never_cuts_off_live_nodes():
    # what broke a first version: deleting everything around the entry, or around an
    # upper-layer node, left it a dead end; deleting 95% left islands nothing links into
    rng = np.random.default_rng(4)
    for case in ("entry", "upper", "heavy"):
        data = rng.standard_normal((600, 8))
        idx = HNSWIndex(8, M=4, ef_construction=32, seed=3)
        idx.add(data)
        u = idx.entry if case == "entry" else int(np.flatnonzero(idx.levels[:600] == 1)[0])
        gone = {x for v in _neighbors(idx, u) for x in [v, *_neighbors(idx, v)]} - {u}
        if case == "heavy":
            gone = set(rng.choice(600, 570, replace=False).tolist())
        idx.delete(sorted(gone))
        assert not _cut_off(idx), case
        live = np.flatnonzero(~idx.dead[:600])
        assert all(idx.search(data[i], k=1, ef=len(live))[0][0] == i for i in live), case
        if case != "heavy":
            assert len(idx.search(data[u] + 0.01, k=10)) == 10, case
        for lv in range(1, idx.levels[idx.entry] + 1):  # the hierarchy still works from the entry
            others = [v for v in live if v != idx.entry and idx.levels[v] >= lv]
            assert not others or len(_links(idx._g(), idx.entry, lv)) > 0, (case, lv)


def test_duplicate_vectors_keep_their_links():
    data = np.random.default_rng(5).standard_normal((300, 16))
    idx = HNSWIndex(16, M=8, ef_construction=64)
    idx.add(np.vstack([data, data]))  # every vector twice: ties everywhere in the heuristic
    assert idx.c0[:600].min() > 1
    assert all(len(idx.search(q, 10)) == 10 for q in data[:50])


def test_delete_edge_cases():
    idx = HNSWIndex(4)
    idx.add(np.eye(4))
    idx.delete(2)
    idx.delete([2, 2])  # already gone: no-op
    assert len(idx) == 3 and 2 not in [i for i, _ in idx.search([0, 0, 1, 0], k=4)]
    try:
        idx.delete(9)
        raise AssertionError("expected IndexError")
    except IndexError:
        pass
    try:
        idx.delete(np.array([False, False, False, True]))  # a mask, not ids: would delete id 0
        raise AssertionError("expected TypeError")
    except TypeError:
        pass
    idx.delete([0, 1, 3])  # empty index again
    assert len(idx) == 0 and idx.search([1, 0, 0, 0]) == []
    idx.add([[0, 1, 0, 0]])
    assert idx.search([0, 1, 0, 0], k=1)[0][0] == 4
    assert FlatIndex(4).search([1, 0, 0, 0], filter=lambda p: True) == []  # empty index, callable filter


def test_filtered_search():
    rng = np.random.default_rng(3)
    data, queries = _clustered(3000, 32, rng), _clustered(100, 32, rng)
    tags = rng.integers(0, 10, 3000)
    payloads = [{"tag": int(t)} for t in tags]
    flat, hnsw = FlatIndex(32), HNSWIndex(32, M=12, ef_construction=80)
    flat.add(data, payloads)
    hnsw.add(data, payloads)
    mask = tags == 3  # 10% of the index
    hnsw.exact_cost = 0  # force the graph walk
    assert _recall(flat, hnsw, queries, filter=mask) > 0.95
    assert all(mask[i] for q in queries for i, _ in hnsw.search(q, 10, filter=mask))
    by_payload = hnsw.search(queries[0], 10, filter=lambda p: p["tag"] == 3)
    assert by_payload == hnsw.search(queries[0], 10, filter=mask)
    hnsw.exact_cost = HNSWIndex.exact_cost
    tiny = mask & (np.arange(3000) < 300)  # ~30 rows: the planner scans them exactly
    assert all(hnsw.search(q, 10, filter=tiny) == flat.search(q, 10, filter=tiny) for q in queries[:10])
    hnsw.delete(np.flatnonzero(tiny)[:5])
    flat.delete(np.flatnonzero(tiny)[:5])
    assert hnsw.search(queries[0], 10, filter=tiny) == flat.search(queries[0], 10, filter=tiny)
    assert hnsw.search(queries[0], 10, filter=np.zeros(3000, bool)) == []


def test_bad_input_is_rejected():
    idx = HNSWIndex(4)
    idx.add(np.eye(4))
    for bad in (lambda: idx.search([1, 0, 0]),                      # wrong dim: would read past the row
                lambda: idx.search([1, 0, 0, 0], filter=[0, 1, 2, 3]),   # ids, not a mask
                lambda: idx.search([1, 0, 0, 0], filter=np.ones(3, bool)),
                lambda: idx.add([[1, 2, 3]])):
        try:
            bad()
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
