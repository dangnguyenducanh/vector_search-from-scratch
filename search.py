"""Exact (brute-force) vector search + the shared vector store. NumPy only.

Every vector is L2-normalized once on insert, so cosine similarity == dot product
and scoring many vectors is one matmul. Higher score = closer.
"""
import numpy as np


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def topk(scores, k):
    """(index, score) of the k highest scores, best first. O(n) select, then sort k."""
    k = min(k, len(scores))
    if k <= 0:
        return []
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top], kind="stable")]
    return [(int(i), float(scores[i])) for i in top]


class _Store:
    """Growable float32 matrix of normalized vectors, one payload per row, and a
    tombstone flag per row. Ids are row numbers: stable, never reused."""

    def __init__(self, dim):
        self.dim = dim
        self.vecs = np.empty((0, dim), dtype=np.float32)
        self.dead = np.zeros(0, dtype=bool)
        self.n = self.n_dead = 0
        self.payloads = []

    def _grow(self, extra):
        self.vecs = np.pad(self.vecs, ((0, extra), (0, 0)))
        self.dead = np.pad(self.dead, (0, extra))

    def _append(self, vecs, payloads):
        vecs = normalize(np.atleast_2d(vecs))
        if vecs.ndim != 2 or vecs.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got shape {vecs.shape}")
        if payloads is not None and (isinstance(payloads, str) or len(payloads) != len(vecs)):
            raise ValueError("payloads must be a list with one entry per vector")
        need = self.n + len(vecs)
        if need > len(self.vecs):  # amortized doubling, not a copy per insert
            self._grow(max(need, 2 * len(self.vecs), 1024) - len(self.vecs))
        start = self.n
        self.vecs[start:need] = vecs
        self.n = need
        self.payloads.extend(payloads if payloads is not None else range(start, need))
        return range(start, need)

    def _query(self, query):
        q = normalize(query)
        if q.shape != (self.dim,):
            raise ValueError(f"expected one query of dim {self.dim}, got shape {q.shape}")
        return q

    def _allowed(self, filter):
        """Live rows that pass `filter`: a bool mask over ids, or a predicate on the payload."""
        if filter is None:
            return ~self.dead[: self.n]
        if callable(filter):
            filter = np.array([bool(filter(p)) for p in self.payloads], dtype=bool)
        allow = np.asarray(filter)
        if allow.size == 0:  # [] is a valid mask for an empty index
            allow = allow.astype(bool)
        if allow.dtype != bool or allow.shape != (self.n,):
            raise ValueError(f"filter must be a bool mask of length {self.n} or a callable on the payload")
        return allow & ~self.dead[: self.n]

    def delete(self, ids):
        """Tombstone ids: they never appear in results again. Returns the newly deleted ids."""
        ids = np.asarray(ids).reshape(-1)
        if ids.size and not np.issubdtype(ids.dtype, np.integer):  # a bool mask would delete ids 0 and 1
            raise TypeError("ids must be integers; for a mask pass np.flatnonzero(mask)")
        ids = np.unique(ids.astype(np.int64))
        if ids.size and (ids[0] < 0 or ids[-1] >= self.n):
            raise IndexError(f"ids must be in [0, {self.n})")
        ids = ids[~self.dead[ids]]
        self.dead[ids] = True
        self.n_dead += len(ids)
        return ids

    def __len__(self):
        return self.n - self.n_dead


class FlatIndex(_Store):
    """Exact search: score every vector. Ground truth for HNSW recall."""

    def add(self, vecs, payloads=None):
        self._append(vecs, payloads)

    def search(self, query, k=10, filter=None):
        scores = self.vecs[: self.n] @ self._query(query)
        if filter is not None or self.n_dead:
            allow = self._allowed(filter)
            scores = np.where(allow, scores, -np.inf)
            k = min(k, int(allow.sum()))
        return topk(scores, k)
