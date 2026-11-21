"""HNSW approximate nearest-neighbor index (Malkov & Yashunin, 2016).

The graph lives in flat int32 arrays and the hot loops (beam search, neighbor
selection, insert, delete repair) are compiled with Numba. No ANN library.
"""
import heapq
import json
import math
import os

import numpy as np
from numba import njit

from search import _Store, topk

# reassoc lets LLVM vectorize the dot-product loop; no inf/nan assumptions
JIT = dict(cache=True, nogil=True, fastmath={"reassoc", "contract"})


# --- graph access ----------------------------------------------------------
# g = (l0, c0, up, cu, slot). Layer 0: l0[node, :c0[node]]. Upper layers only
# store the ~1/M nodes that reach them: up[slot[node], level - 1, :cu[...]].
@njit(**JIT)
def _links(g, node, level):
    l0, c0, up, cu, slot = g
    if level == 0:
        return l0[node, : c0[node]]
    s = slot[node]
    return up[s, level - 1, : cu[s, level - 1]]


@njit(**JIT)
def _set_links(g, node, level, ids):
    l0, c0, up, cu, slot = g
    if level == 0:
        l0[node, : len(ids)] = ids
        c0[node] = len(ids)
    else:
        s = slot[node]
        up[s, level - 1, : len(ids)] = ids
        cu[s, level - 1] = len(ids)


@njit(**JIT)
def _has(ids, x):
    for y in ids:
        if y == x:
            return True
    return False


@njit(**JIT)
def _dot(a, b):
    s = np.float32(0.0)
    for i in range(a.shape[0]):
        s += a[i] * b[i]
    return s


# --- search primitives -----------------------------------------------------
@njit(**JIT)
def _search_layer(vecs, g, q, eps, ef, level, visited, tag, allow):
    """Beam search on one layer; returns up to ef (ids, scores), best first.

    allow: bool mask of nodes that may be returned (empty = all). Other nodes are
    still walked through, so a filter never cuts the graph into pieces.
    """
    filtered = allow.shape[0] > 0
    cand = [(np.float32(0.0), np.int32(0))]  # max-heap on score (stored negated)
    found = [(np.float32(0.0), np.int32(0))]  # min-heap, worst kept result on top
    cand.pop()
    found.pop()
    for e in eps:
        visited[e] = tag
        s = _dot(vecs[e], q)
        heapq.heappush(cand, (-s, e))
        if not filtered or allow[e]:
            heapq.heappush(found, (s, e))
            if len(found) > ef:
                heapq.heappop(found)
    while len(cand) > 0:
        neg, node = heapq.heappop(cand)
        if len(found) >= ef and -neg < found[0][0]:
            break  # best unexplored is worse than our worst kept -> done
        for nb in _links(g, node, level):
            if visited[nb] == tag:
                continue
            visited[nb] = tag
            s = _dot(vecs[nb], q)
            if len(found) < ef or s > found[0][0]:
                heapq.heappush(cand, (-s, nb))
                if not filtered or allow[nb]:
                    heapq.heappush(found, (s, nb))
                    if len(found) > ef:
                        heapq.heappop(found)
    ids = np.empty(len(found), np.int32)
    scores = np.empty(len(found), np.float32)
    for i in range(len(found) - 1, -1, -1):  # min-heap pops worst first
        s, e = heapq.heappop(found)
        scores[i] = s
        ids[i] = e
    return ids, scores


@njit(**JIT)
def _descend(vecs, g, levels, q, entry, to_level):
    """Greedy walk from the top layer down to (but not into) to_level."""
    ep, best = entry, _dot(vecs[entry], q)
    for level in range(levels[entry], to_level, -1):
        moved = True
        while moved:
            moved = False
            for nb in _links(g, ep, level):
                s = _dot(vecs[nb], q)
                if s > best:
                    ep, best, moved = nb, s, True
    return ep


@njit(**JIT)
def _select(vecs, ids, scores, m):
    """Diversity heuristic (paper Alg. 4): drop a candidate that is closer to a
    neighbor already kept than to the base. Keeps links pointing in different
    directions, so clusters stay connected to each other. Ties are kept, as in
    hnswlib: with `>=`, exact duplicates would leave a node with one link.
    ids/scores are sorted best-first; scores are similarities to the base."""
    if len(ids) <= m:
        return ids.copy()
    kept = np.empty(m, np.int32)
    k = 0
    for i in range(len(ids)):
        ok = True
        for j in range(k):
            if _dot(vecs[ids[i]], vecs[kept[j]]) > scores[i]:
                ok = False
                break
        if ok:
            kept[k] = ids[i]
            k += 1
            if k == m:
                break
    return kept[:k]


@njit(**JIT)
def _connect(vecs, g, node, level, ids, scores, M, cap, shrink):
    """Link node to the selected candidates and link them back. A neighbor list
    that goes over cap is shrunk with the same heuristic, or, with shrink=False,
    left as it is (the back-link is skipped, nobody else loses a link)."""
    nbrs = _select(vecs, ids, scores, M)
    _set_links(g, node, level, nbrs)
    for nb in nbrs:
        cur = _links(g, nb, level)
        if _has(cur, node):
            continue
        m = len(cur)
        cand = np.empty(m + 1, np.int32)
        cand[:m] = cur
        cand[m] = node
        if m < cap:
            _set_links(g, nb, level, cand)
            continue
        if not shrink:
            continue
        sc = np.empty(m + 1, np.float32)
        for i in range(m + 1):
            sc[i] = _dot(vecs[cand[i]], vecs[nb])
        order = np.argsort(-sc)
        _set_links(g, nb, level, _select(vecs, cand[order], sc[order], cap))


@njit(**JIT)
def _insert(vecs, g, levels, visited, tag, entry, start, stop, M, M0, efc):
    """Insert rows start..stop-1 (paper Alg. 1). Returns (entry, tag)."""
    everyone = np.zeros(0, np.bool_)
    eps = np.empty(1, np.int32)
    for node in range(start, stop):
        if entry < 0:
            entry = node
            continue
        q, level, top = vecs[node], levels[node], levels[entry]
        eps[0] = _descend(vecs, g, levels, q, entry, level) if level < top else entry
        cur = eps
        for lc in range(min(level, top), -1, -1):
            tag += 1
            ids, scores = _search_layer(vecs, g, q, cur, efc, lc, visited, tag, everyone)
            _connect(vecs, g, node, lc, ids, scores, M, M0 if lc == 0 else M, True)
            cur = ids
        if level > top:
            entry = node
    return entry, tag


@njit(**JIT)
def _query(vecs, g, levels, visited, tag, entry, q, k, ef, allow):
    eps = np.empty(1, np.int32)
    eps[0] = _descend(vecs, g, levels, q, entry, 0)
    ids, scores = _search_layer(vecs, g, q, eps, max(ef, k), 0, visited, tag, allow)
    return ids[:k], scores[:k]


@njit(**JIT)
def _component(g, dead, entry, n):
    """(reach, back): layer-0 nodes the entry can reach, and nodes that can walk back to it."""
    l0, c0, up, cu, slot = g
    stack = np.empty(n, np.int32)
    fwd = np.zeros(n, np.bool_)
    fwd[entry] = True
    stack[0] = entry
    top = 1
    while top > 0:
        top -= 1
        x = stack[top]
        for y in l0[x, : c0[x]]:
            if not fwd[y]:
                fwd[y] = True
                stack[top] = y
                top += 1
    start = np.zeros(n + 1, np.int64)  # reverse edges, CSR
    for x in range(n):
        for y in l0[x, : c0[x]]:
            start[y + 1] += 1
    for i in range(n):
        start[i + 1] += start[i]
    rev = np.empty(start[n], np.int32)
    fill = start[:n].copy()
    for x in range(n):
        for y in l0[x, : c0[x]]:
            rev[fill[y]] = x
            fill[y] += 1
    bwd = np.zeros(n, np.bool_)
    bwd[entry] = True
    stack[0] = entry
    top = 1
    while top > 0:
        top -= 1
        x = stack[top]
        for y in rev[start[x]: start[x + 1]]:
            if not bwd[y]:
                bwd[y] = True
                stack[top] = y
                top += 1
    return fwd, bwd


@njit(**JIT)
def _relink(vecs, g, visited, tag, u, seed, level, efc, M, M0, allow, shrink):
    """Re-insert u on one layer, searching from seed. u is pre-marked visited, so
    the walk never enters it or returns it. Returns tag."""
    tag += 1
    visited[u] = tag
    eps = np.empty(1, np.int32)
    eps[0] = seed
    ids, scores = _search_layer(vecs, g, vecs[u], eps, efc, level, visited, tag, allow)
    _connect(vecs, g, u, level, ids, scores, M, M0 if level == 0 else M, shrink)
    return tag


@njit(**JIT)
def _repair(vecs, g, levels, dead, visited, tag, entry, n, M, M0, efc):
    """Unlink tombstoned nodes (FreshDiskANN-style local repair). A live node
    that pointed at a dead one takes the dead node's live neighbors as
    candidates instead, re-pruned with the same heuristic. Then every live node
    that is cut off on layer 0 (an orphan, a dead end, or an island the entry
    can't reach or that can't walk back) is re-inserted next to the entry's
    component. Returns tag."""
    for u in range(n):
        if dead[u]:
            continue
        for lv in range(levels[u] + 1):
            cur = _links(g, u, lv)
            hit = False
            for x in cur:
                if dead[x]:
                    hit = True
                    break
            if not hit:
                continue
            cap = M0 if lv == 0 else M
            tag += 1
            visited[u] = tag  # never link u to itself
            buf = np.empty(len(cur) * (cap + 1), np.int32)
            m = 0
            for x in cur:
                if dead[x]:
                    for y in _links(g, x, lv):
                        if not dead[y] and visited[y] != tag:
                            visited[y] = tag
                            buf[m] = y
                            m += 1
                elif visited[x] != tag:
                    visited[x] = tag
                    buf[m] = x
                    m += 1
            cand = buf[:m]
            sc = np.empty(m, np.float32)
            for i in range(m):
                sc[i] = _dot(vecs[cand[i]], vecs[u])
            order = np.argsort(-sc)
            _set_links(g, u, lv, _select(vecs, cand[order], sc[order], cap))
    nothing = np.empty(0, np.int32)
    for d in range(n):
        if dead[d]:
            for lv in range(levels[d] + 1):
                _set_links(g, d, lv, nothing)
    for lv in range(levels[entry], -1, -1):  # an entry with no links on a layer is a dead end there
        if len(_links(g, entry, lv)) == 0:  # link it from any other live node on that layer
            for s in range(n):
                if not dead[s] and s != entry and levels[s] >= lv:
                    tag = _relink(vecs, g, visited, tag, entry, s, lv, efc, M, M0, np.zeros(0, np.bool_), True)
                    break
    # The entry's component: nodes it reaches that can also walk back to it. A path
    # between two members never passes through a non-member, so editing links from or
    # to non-members can't cut a member off; the component only grows from here.
    reach, back = _component(g, dead, entry, n)
    member = reach & back
    for u in range(n):
        if dead[u] or member[u]:
            continue
        # out-links only into the component (u can walk back); back-links only where a
        # member has room (nobody loses a link)
        tag = _relink(vecs, g, visited, tag, u, entry, 0, efc, M, M0, member, False)
        mine = _links(g, u, 0)
        joined = False
        for nb in mine:
            if _has(_links(g, nb, 0), u):
                joined = True
                break
        if not joined:  # every list full (u is a duplicate, or farther than all their links):
            # splice u into its nearest member's link nb -> b, making it nb -> u -> b,
            # so every path that used the link still exists
            cur = _links(g, mine[0], 0)  # a view: writing cur[j] edits the graph
            j = 0
            for i in range(1, len(cur)):
                if _dot(vecs[cur[i]], vecs[u]) > _dot(vecs[cur[j]], vecs[u]):
                    j = i
            b = cur[j]
            cur[j] = u
            if not _has(mine, b):
                ext = np.empty(len(mine) + 1, np.int32)  # len(mine) <= M < M0: room
                ext[:-1] = mine
                ext[-1] = b
                _set_links(g, u, 0, ext)
        member[u] = True
    return tag


class HNSWIndex(_Store):
    """Hierarchical Navigable Small World graph over the vector store.

    exact_cost is the query planner for filters. A graph walk restricted to a
    fraction s of the index visits roughly ef / s nodes; an exact scan of the
    allowed rows costs n * s dot products. Scan when n_allowed * s < exact_cost * ef.
    10 puts the switch where the two costs crossed on AG News 120k (s ~ 7%).
    """

    exact_cost = 10

    def __init__(self, dim, M=16, ef_construction=100, seed=0):
        dim, M, ef_construction = int(dim), int(M), int(ef_construction)  # numpy ints break save()
        if M < 2 or ef_construction < 1:
            raise ValueError("need M >= 2 and ef_construction >= 1")
        super().__init__(dim)
        self.M, self.M0 = M, 2 * M  # layer 0 is denser, as in the paper
        self.ef_construction = ef_construction
        self.mL = 1 / math.log(M)  # level multiplier: P(level >= l) = M^-l
        self.rng = np.random.default_rng(seed)
        self.entry = -1
        self.levels = np.zeros(0, np.int32)
        self.l0 = np.zeros((0, self.M0), np.int32)
        self.c0 = np.zeros(0, np.int32)
        self.slot = np.zeros(0, np.int32)  # row in `up`, -1 for layer-0-only nodes
        self.up = np.zeros((0, 1, M), np.int32)
        self.cu = np.zeros((0, 1), np.int32)
        self.n_up = 0
        self._visited = np.zeros(0, np.int64)  # visited[x] == tag  <=>  seen in this search
        self._tag = 0

    def _g(self):
        return self.l0, self.c0, self.up, self.cu, self.slot

    def _grow(self, extra):
        super()._grow(extra)
        self.levels = np.pad(self.levels, (0, extra))
        self.l0 = np.pad(self.l0, ((0, extra), (0, 0)))
        self.c0 = np.pad(self.c0, (0, extra))
        self.slot = np.pad(self.slot, (0, extra), constant_values=-1)
        self._visited = np.pad(self._visited, (0, extra))

    # --- public API --------------------------------------------------------
    def add(self, vecs, payloads=None):
        rows = self._append(vecs, payloads)
        if not rows:
            return
        lv = (-np.log(1.0 - self.rng.random(len(rows))) * self.mL).astype(np.int32)
        self.levels[rows.start: rows.stop] = lv
        hi = rows.start + np.flatnonzero(lv)
        rows_up, depth = self.up.shape[:2]
        more_rows = max(0, self.n_up + len(hi) - rows_up)
        if more_rows:
            more_rows = max(more_rows, rows_up, 64)  # amortized doubling
        more_depth = max(0, int(lv.max()) - depth)
        if more_rows or more_depth:
            self.up = np.pad(self.up, ((0, more_rows), (0, more_depth), (0, 0)))
            self.cu = np.pad(self.cu, ((0, more_rows), (0, more_depth)))
        self.slot[hi] = np.arange(self.n_up, self.n_up + len(hi))
        self.n_up += len(hi)
        self.entry, self._tag = _insert(self.vecs, self._g(), self.levels, self._visited, self._tag,
                                        self.entry, rows.start, rows.stop, self.M, self.M0,
                                        self.ef_construction)

    def search(self, query, k=10, ef=50, filter=None):
        """Top-k (id, score), best first. filter: bool mask over ids, or a predicate
        on the payload. Deleted ids never come back."""
        q = self._query(query)
        if self.entry < 0 or k <= 0:
            return []
        allow = np.zeros(0, bool)
        if filter is not None:
            allow = self._allowed(filter)
            ids = np.flatnonzero(allow)
            if len(ids) * len(ids) < self.exact_cost * max(ef, k) * len(self):
                return [(int(ids[i]), s) for i, s in topk(self.vecs[ids] @ q, k)]
        self._tag += 1
        ids, scores = _query(self.vecs, self._g(), self.levels, self._visited, self._tag,
                             self.entry, q, k, ef, allow)
        return list(zip(ids.tolist(), scores.tolist()))

    def delete(self, ids):
        """Tombstone ids and repair the graph around them, so search never walks
        through a deleted node. Ids are not reused; the vectors stay in memory."""
        gone = super().delete(ids)
        if not len(gone):
            return
        if not len(self):
            self.entry = -1
            self.c0[:] = 0
            self.cu[:] = 0
            return
        if self.dead[self.entry]:  # new entry: the highest live node
            live = np.flatnonzero(~self.dead[: self.n])
            self.entry = int(live[np.argmax(self.levels[live])])
        self._tag = _repair(self.vecs, self._g(), self.levels, self.dead, self._visited, self._tag,
                            self.entry, self.n, self.M, self.M0, self.ef_construction)

    # --- persistence: npz + json, no pickle (loading pickle runs code) -----
    ARRAYS = dict(vecs=np.float32, dead=np.bool_, levels=np.int32, l0=np.int32, c0=np.int32,
                  slot=np.int32, up=np.int32, cu=np.int32)

    def save(self, path):
        meta = json.dumps(dict(dim=self.dim, M=self.M, ef_construction=self.ef_construction,
                               entry=self.entry, payloads=self.payloads))
        if json.loads(meta)["payloads"] != self.payloads:  # tuples, int dict keys...: filters would change
            raise TypeError("payloads must be JSON types to save: str/int/float/bool/None, lists, str-keyed dicts")
        cut = {a: getattr(self, a)[: self.n_up if a in ("up", "cu") else self.n] for a in self.ARRAYS}
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:  # file handle: np.savez(path) would append ".npz" to bare names
            np.savez(f, meta=np.array(meta), **cut)
        os.replace(tmp, path)  # atomic: a failed save never clobbers the previous file

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        idx = cls(meta["dim"], M=meta["M"], ef_construction=meta["ef_construction"])
        for a, dtype in cls.ARRAYS.items():
            setattr(idx, a, np.ascontiguousarray(z[a], dtype=dtype))
        idx.n, idx.n_up, idx.n_dead = len(idx.vecs), len(idx.up), int(idx.dead.sum())
        idx._visited = np.zeros(idx.n, np.int64)
        idx.payloads, idx.entry = meta["payloads"], int(meta["entry"])
        idx._check()
        return idx

    def _check(self):
        """The compiled loops don't bounds-check, so a corrupt or crafted file must
        fail here with an error instead of reading or writing past an array."""
        n, lv, up, slot = self.n, self.levels, self.up, self.slot
        ok = (self.vecs.shape == (n, self.dim) and self.l0.shape == (n, self.M0)
              and up.ndim == 3 and up.shape[1] >= 1 and up.shape[2] == self.M and self.cu.shape == up.shape[:2]
              and all(a.shape == (n,) for a in (self.dead, lv, self.c0, slot))
              and isinstance(self.payloads, list) and len(self.payloads) == n
              and (0 <= self.entry < n and not self.dead[self.entry] if len(self) else self.entry == -1)
              and np.all((lv >= 0) & (lv <= up.shape[1]))
              and np.all((self.c0 >= 0) & (self.c0 <= self.M0)) and np.all((self.cu >= 0) & (self.cu <= self.M))
              and np.all((self.l0 >= 0) & (self.l0 < n)) and np.all((up >= 0) & (up < n))
              and np.all((slot >= -1) & (slot < len(up))) and np.all(slot[lv > 0] >= 0))
        if not ok:
            raise ValueError("corrupt index file")
