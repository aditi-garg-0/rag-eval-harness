"""
Retrieval backends: sparse (BM25, dependency-light, always available),
dense (sentence-transformers, requires model download), and hybrid.

Designed so the ablation runner can swap retrievers without touching
any other part of the pipeline.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from rag.chunking import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BaseRetriever(ABC):
    name: str = "base"

    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None:
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        ...


class BM25Retriever(BaseRetriever):
    """Sparse lexical retrieval. No downloads required -- this is the
    reliable baseline every other retriever is compared against."""

    name = "bm25"

    def __init__(self):
        self.chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def index(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        if self._bm25 is None:
            raise RuntimeError("Call index() before retrieve().")
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self.chunks, scores), key=lambda x: -x[1])[:top_k]
        return [ScoredChunk(chunk=c, score=float(s)) for c, s in ranked]


class DenseRetriever(BaseRetriever):
    """Embedding-based retrieval via sentence-transformers.

    Requires internet access to download the model weights the first time
    (blocked in sandboxed/offline environments). Instantiation is lazy --
    import + model load only happens on first `.index()` call, so the rest
    of the codebase can reference this class without requiring the
    dependency at import time.
    """

    name = "dense"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self.chunks: list[Chunk] = []
        self._embeddings = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers not installed. Run: "
                    "pip install sentence-transformers"
                ) from e
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def index(self, chunks: list[Chunk]) -> None:
        model = self._load_model()
        self.chunks = chunks
        self._embeddings = model.encode(
            [c.text for c in chunks], normalize_embeddings=True,
            show_progress_bar=False,
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        import numpy as np
        if self._embeddings is None:
            raise RuntimeError("Call index() before retrieve().")
        model = self._load_model()
        q_emb = model.encode([query], normalize_embeddings=True)[0]
        sims = self._embeddings @ q_emb
        ranked_idx = np.argsort(-sims)[:top_k]
        return [ScoredChunk(chunk=self.chunks[i], score=float(sims[i]))
                for i in ranked_idx]


class HybridRetriever(BaseRetriever):
    """Combines BM25 and dense scores via weighted, min-max normalized
    reciprocal rank fusion. Falls back gracefully -- if dense retrieval
    is unavailable (no model access), this degrades to pure BM25 rather
    than crashing, which matters for the "fully local, no downloads"
    ablation runs."""

    name = "hybrid"

    def __init__(self, alpha: float = 0.5,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.alpha = alpha  # weight on dense score; (1-alpha) on BM25
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(model_name=model_name)
        self._dense_available = True

    def index(self, chunks: list[Chunk]) -> None:
        self.bm25.index(chunks)
        try:
            self.dense.index(chunks)
        except ImportError:
            self._dense_available = False

    @staticmethod
    def _normalize(scored: list[ScoredChunk]) -> dict[str, float]:
        if not scored:
            return {}
        scores = [s.score for s in scored]
        lo, hi = min(scores), max(scores)
        span = hi - lo or 1.0
        return {s.chunk.chunk_id: (s.score - lo) / span for s in scored}

    def retrieve(self, query: str, top_k: int = 5) -> list[ScoredChunk]:
        n = max(top_k * 4, 20)
        bm25_hits = self.bm25.retrieve(query, top_k=n)
        bm25_norm = self._normalize(bm25_hits)

        if self._dense_available:
            dense_hits = self.dense.retrieve(query, top_k=n)
            dense_norm = self._normalize(dense_hits)
        else:
            dense_hits, dense_norm = [], {}

        by_id = {c.chunk.chunk_id: c.chunk for c in bm25_hits + dense_hits}
        combined = {}
        for cid in by_id:
            b = bm25_norm.get(cid, 0.0)
            d = dense_norm.get(cid, 0.0)
            weight = self.alpha if self._dense_available else 0.0
            combined[cid] = weight * d + (1 - weight) * b

        ranked = sorted(combined.items(), key=lambda x: -x[1])[:top_k]
        return [ScoredChunk(chunk=by_id[cid], score=score) for cid, score in ranked]


RETRIEVERS = {
    "bm25": BM25Retriever,
    "dense": DenseRetriever,
    "hybrid": HybridRetriever,
}
