"""Semantic search over a text file, one document per line.

python cli.py build examples/products.txt products.npz
python cli.py search products.npz "something warm and cozy for winter" -k 3
"""
import argparse

from hnsw import HNSWIndex

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def embed(texts):
    from sentence_transformers import SentenceTransformer  # slow import, only when needed

    return SentenceTransformer(MODEL).encode(texts, batch_size=256, normalize_embeddings=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("docs")
    b.add_argument("index")
    s = sub.add_parser("search")
    s.add_argument("index")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--ef", type=int, default=64)
    a = p.parse_args()

    if a.cmd == "build":
        with open(a.docs, encoding="utf-8") as f:
            docs = [line.strip() for line in f if line.strip()]
        vecs = embed(docs)
        idx = HNSWIndex(vecs.shape[1])
        idx.add(vecs, docs)
        idx.save(a.index)
        print(f"indexed {len(idx)} docs -> {a.index}")
    else:
        idx = HNSWIndex.load(a.index)
        for node, score in idx.search(embed([a.query])[0], k=a.k, ef=a.ef):
            print(f"{score:.3f}  {idx.payloads[node]}")


if __name__ == "__main__":
    main()
