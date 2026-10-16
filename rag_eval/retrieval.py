"""Retrievers. BM25 works fully offline; dense options are pluggable."""
from __future__ import annotations
import math
from collections import Counter
from dataclasses import dataclass
import numpy as np
from .chunking import Chunk
from .text import tokenize


@dataclass
class Hit:
    chunk: Chunk
    score: float  # normalised to 0..1


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(tokenize(c.label + " " + c.text)) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 1
        df = Counter(t for d in self.docs for t in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def _raw(self, q: list[str], i: int) -> float:
        d, s = self.docs[i], 0.0
        for t in q:
            if t in d:
                tf = d[t]
                s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
        return s

    def search(self, query: str, k: int = 4) -> list[Hit]:
        q = tokenize(query)
        if not q or not self.chunks:
            return []
        # normalise by the best possible score for this query => comparable 0..1 confidence
        ideal = sum(self.idf.get(t, 0) * (self.k1 + 1) for t in set(q)) or 1.0
        scored = sorted(((self._raw(q, i), i) for i in range(len(self.chunks))), reverse=True)[:k]
        return [Hit(self.chunks[i], min(1.0, s / ideal)) for s, i in scored if s > 0]


class DenseRetriever:
    """Cosine similarity over embeddings from sentence-transformers or OpenAI."""

    def __init__(self, chunks: list[Chunk], backend: str):
        self.chunks, self.name = chunks, backend
        self._embed = self._make_embedder(backend)
        self.matrix = self._embed([c.label + "\n" + c.text for c in chunks])

    @staticmethod
    def _make_embedder(backend: str):
        if backend == "sentence-transformers":
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            return lambda xs: np.asarray(model.encode(xs, normalize_embeddings=True))
        if backend == "openai":
            from openai import OpenAI
            client = OpenAI()

            def emb(xs):
                r = client.embeddings.create(model="text-embedding-3-small", input=xs)
                m = np.asarray([d.embedding for d in r.data])
                return m / np.linalg.norm(m, axis=1, keepdims=True)
            return emb
        raise ValueError(f"Unknown embeddings backend: {backend}")

    def search(self, query: str, k: int = 4) -> list[Hit]:
        q = self._embed([query])[0]
        sims = self.matrix @ q
        idx = np.argsort(-sims)[:k]
        return [Hit(self.chunks[i], float(max(0.0, sims[i]))) for i in idx]


def make_retriever(chunks: list[Chunk], backend: str = "bm25"):
    return BM25Retriever(chunks) if backend in ("bm25", "", None) else DenseRetriever(chunks, backend)
