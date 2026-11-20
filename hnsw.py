"""HNSW approximate nearest-neighbor index (Malkov & Yashunin, 2016), NumPy only."""
import heapq
import json
import math

import numpy as np

from search import _Store, normalize


class HNSWIndex(_Store):
    """Hierarchical Navigable Small World graph.

    links[node][level] is that node's neighbor list on that level, so every
    layer is its own graph (upper layers = sparse long-range "express lanes").
    """

    def __init__(self, dim, M=16, ef_construction=100, seed=0):
        super().__init__(dim)
        self.M, self.M0 = M, 2 * M  # layer 0 is denser, as in the paper
        self.ef_construction = ef_construction
        self.mL = 1 / math.log(M)  # level multiplier: P(level >= l) = M^-l
        self.rng = np.random.default_rng(seed)
        self.levels = []
        self.links = []
        self.entry = -1

    # --- search primitives -------------------------------------------------
    def _search_layer(self, q, entry_points, ef, level):
        """Beam search on one layer; returns up to ef (score, node), best first."""
        vecs, links = self.vecs, self.links
        visited = set(entry_points)
        scores = (vecs[entry_points] @ q).tolist()
        candidates = [(-s, e) for s, e in zip(scores, entry_points)]  # max-heap
        found = [(s, e) for s, e in zip(scores, entry_points)]  # min-heap, worst on top
        heapq.heapify(candidates)
        heapq.heapify(found)
        while len(found) > ef:
            heapq.heappop(found)
        while candidates:
            neg, node = heapq.heappop(candidates)
            if -neg < found[0][0] and len(found) >= ef:
                break  # best unexplored is worse than our worst kept -> done
            fresh = [n for n in links[node][level] if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            for s, n in zip((vecs[fresh] @ q).tolist(), fresh):
                if len(found) < ef or s > found[0][0]:
                    heapq.heappush(candidates, (-s, n))
                    heapq.heappush(found, (s, n))
                    if len(found) > ef:
                        heapq.heappop(found)
        return sorted(found, reverse=True)

    def _descend(self, q, to_level):
        """Greedy ef=1 walk from the top layer down to (but not into) to_level."""
        ep = self.entry
        for level in range(self.levels[ep], to_level, -1):
            ep = self._search_layer(q, [ep], 1, level)[0][1]
        return ep

    def _select(self, candidates, m):
        """Diversity heuristic (paper Alg. 4): keep a candidate only if it is closer
        to the base than to every neighbor already kept. Keeps links pointing in
        different directions, so clusters stay connected to each other."""
        if len(candidates) <= m:
            return [n for _, n in candidates]
        ids = [n for _, n in candidates]  # already sorted best-first
        pair = self.vecs[ids] @ self.vecs[ids].T
        kept = []
        for i, (s, _) in enumerate(candidates):
            if all(pair[i, j] < s for j in kept):
                kept.append(i)
                if len(kept) == m:
                    break
        return [ids[i] for i in kept]

    # --- public API --------------------------------------------------------
    def add(self, vecs, payloads=None):
        for node in self._append(vecs, payloads):
            self._insert(node)

    def _insert(self, node):
        q = self.vecs[node]
        level = int(-math.log(1.0 - self.rng.random()) * self.mL)
        self.levels.append(level)
        self.links.append([[] for _ in range(level + 1)])
        if self.entry < 0:
            self.entry = node
            return
        top = self.levels[self.entry]
        eps = [self._descend(q, level)] if level < top else [self.entry]
        for lc in range(min(level, top), -1, -1):
            found = self._search_layer(q, eps, self.ef_construction, lc)
            cap = self.M0 if lc == 0 else self.M
            neighbors = self._select(found, self.M)
            self.links[node][lc] = neighbors
            for nb in neighbors:  # connect back, shrink if over capacity
                nl = self.links[nb][lc]
                nl.append(node)
                if len(nl) > cap:
                    s = (self.vecs[nl] @ self.vecs[nb]).tolist()
                    self.links[nb][lc] = self._select(sorted(zip(s, nl), reverse=True), cap)
            eps = [n for _, n in found]
        if level > top:
            self.entry = node

    def search(self, query, k=10, ef=50):
        if self.n == 0:
            return []
        q = normalize(query)
        ep = self._descend(q, 0)
        found = self._search_layer(q, [ep], max(ef, k), 0)[:k]
        return [(n, float(s)) for s, n in found]

    # --- persistence: npz + json, no pickle (loading pickle runs code) -----
    def save(self, path):
        flat, counts = [], []
        for node_links in self.links:
            for nl in node_links:
                counts.append(len(nl))
                flat.extend(nl)
        meta = dict(dim=self.dim, M=self.M, ef_construction=self.ef_construction,
                    entry=self.entry, payloads=self.payloads)
        with open(path, "wb") as f:  # file handle: np.savez(path) would append ".npz" to bare names
            np.savez(f, vecs=self.vecs[: self.n], levels=np.array(self.levels, dtype=np.int32),
                     counts=np.array(counts, dtype=np.int32), flat=np.array(flat, dtype=np.int32),
                     meta=np.array(json.dumps(meta)))

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        idx = cls(meta["dim"], M=meta["M"], ef_construction=meta["ef_construction"])
        idx.vecs = z["vecs"].copy()
        idx.n = len(idx.vecs)
        idx.payloads = meta["payloads"]
        idx.entry = meta["entry"]
        idx.levels = z["levels"].tolist()
        counts, flat, pos, c = z["counts"].tolist(), z["flat"].tolist(), 0, 0
        for lvl in idx.levels:
            node_links = []
            for _ in range(lvl + 1):
                node_links.append(flat[pos: pos + counts[c]])
                pos += counts[c]
                c += 1
            idx.links.append(node_links)
        return idx
