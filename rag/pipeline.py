"""
End-to-end RAG pipeline. Wires together chunking -> query transform ->
retrieval -> rerank -> generation behind one configurable object, so a
single (chunker, retriever, query_transform, reranker, generator) tuple
is one "condition" the ablation runner can sweep over.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from rag.chunking import Document, Chunk, chunk_document
from rag.generation import BaseGenerator
from rag.query_transform import BaseQueryTransform, IdentityTransform
from rag.rerank import BaseReranker, NoOpReranker
from rag.retrieval import BaseRetriever, ScoredChunk

# How many first-stage candidates to pull before handing them to a
# reranker. Only matters when the reranker isn't a no-op -- reranking
# needs a larger candidate pool than top_k to have anything to sort
# through, otherwise it's just relabeling the retriever's own order.
RERANK_CANDIDATE_MULTIPLIER = 4


@dataclass
class RAGResult:
    question: str
    answer: str
    retrieved_chunks: list[ScoredChunk]
    retrieval_latency_sec: float = 0.0
    generation_latency_sec: float = 0.0
    config: dict = field(default_factory=dict)


class RAGPipeline:
    def __init__(
        self,
        retriever: BaseRetriever,
        generator: BaseGenerator,
        chunk_strategy: str = "recursive",
        chunk_kwargs: dict | None = None,
        top_k: int = 5,
        query_transform: BaseQueryTransform | None = None,
        reranker: BaseReranker | None = None,
    ):
        self.retriever = retriever
        self.generator = generator
        self.chunk_strategy = chunk_strategy
        self.chunk_kwargs = chunk_kwargs or {}
        self.top_k = top_k
        self.query_transform = query_transform or IdentityTransform()
        self.reranker = reranker or NoOpReranker()
        self._indexed = False
        self.chunks: list[Chunk] = []

    def ingest(self, documents: list[Document]) -> None:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(
                chunk_document(doc, strategy=self.chunk_strategy, **self.chunk_kwargs)
            )
        self.chunks = all_chunks
        self.retriever.index(all_chunks)
        self._indexed = True

    def query(self, question: str) -> RAGResult:
        if not self._indexed:
            raise RuntimeError("Call ingest() before query().")

        needs_candidates = self.reranker.name != "none"
        fetch_k = self.top_k * RERANK_CANDIDATE_MULTIPLIER if needs_candidates else self.top_k

        t0 = time.time()
        candidates = self.query_transform.get_chunks(question, self.retriever, fetch_k)
        retrieved = self.reranker.rerank(question, candidates, self.top_k)
        retrieval_latency = time.time() - t0

        t0 = time.time()
        answer = self.generator.answer(question, [r.chunk for r in retrieved])
        generation_latency = time.time() - t0

        return RAGResult(
            question=question,
            answer=answer,
            retrieved_chunks=retrieved,
            retrieval_latency_sec=retrieval_latency,
            generation_latency_sec=generation_latency,
            config={
                "chunk_strategy": self.chunk_strategy,
                "chunk_kwargs": self.chunk_kwargs,
                "retriever": self.retriever.name,
                "query_transform": self.query_transform.name,
                "reranker": self.reranker.name,
                "generator": self.generator.name,
                "top_k": self.top_k,
            },
        )
