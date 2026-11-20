"""Exact (brute-force) vector search + the shared vector store. NumPy only.

Every vector is L2-normalized once on insert, so cosine similarity == dot product
and scoring many vectors is one matmul. Higher score = closer.
"""
import numpy as np


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


class _Store:
    """Growable float32 matrix of normalized vectors + one payload per row."""

    def __init__(self, dim):
        self.dim = dim
        self.vecs = np.empty((0, dim), dtype=np.float32)
        self.n = 0
        self.payloads = []

    def _append(self, vecs, payloads):
        vecs = normalize(np.atleast_2d(vecs))
        if vecs.shape[1] != self.dim:
            raise ValueError(f"expected dim {self.dim}, got {vecs.shape[1]}")
        if payloads is not None and (isinstance(payloads, str) or len(payloads) != len(vecs)):
            raise ValueError("payloads must be a list with one entry per vector")
        need = self.n + len(vecs)
        if need > len(self.vecs):  # amortized doubling, not a copy per insert
            grown = np.empty((max(need, 2 * len(self.vecs), 1024), self.dim), dtype=np.float32)
            grown[: self.n] = self.vecs[: self.n]
            self.vecs = grown
        start = self.n
        self.vecs[start:need] = vecs
        self.n = need
        self.payloads.extend(payloads if payloads is not None else range(start, need))
        return range(start, need)

    def __len__(self):
        return self.n


class FlatIndex(_Store):
    """Exact search: score every vector. Ground truth for HNSW recall."""

    def add(self, vecs, payloads=None):
        self._append(vecs, payloads)

    def search(self, query, k=10):
        scores = self.vecs[: self.n] @ normalize(query)
        k = min(k, self.n)
        top = np.argpartition(-scores, k - 1)[:k]  # O(n), then sort only k
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]
