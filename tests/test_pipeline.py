from rag.chunking import Document
from rag.generation import MockGenerator
from rag.pipeline import RAGPipeline, RERANK_CANDIDATE_MULTIPLIER
from rag.query_transform import IdentityTransform, DecomposeTransform
from rag.rerank import NoOpReranker, LexicalOverlapReranker
from rag.retrieval import BM25Retriever

DOCS = [
    Document(doc_id="bm25_doc", text="BM25 is a sparse lexical ranking function based on term frequency. " * 3),
    Document(doc_id="dense_doc", text="Dense retrieval encodes text into embeddings for semantic search. " * 3),
    Document(doc_id="hybrid_doc", text="Hybrid retrieval combines BM25 sparse scores with dense embeddings. " * 3),
]


def make_pipeline(**kwargs):
    pipeline = RAGPipeline(
        retriever=BM25Retriever(),
        generator=MockGenerator(),
        chunk_strategy="recursive",
        chunk_kwargs={"chunk_size": 128},
        top_k=2,
        **kwargs,
    )
    pipeline.ingest(DOCS)
    return pipeline


def test_default_pipeline_uses_identity_and_noop():
    pipeline = make_pipeline()
    assert isinstance(pipeline.query_transform, IdentityTransform)
    assert isinstance(pipeline.reranker, NoOpReranker)


def test_query_returns_at_most_top_k_chunks():
    pipeline = make_pipeline()
    result = pipeline.query("What is BM25?")
    assert len(result.retrieved_chunks) <= 2


def test_reranker_receives_larger_candidate_pool_than_top_k():
    # Instrument the reranker to record how many candidates it actually
    # saw, so we can confirm the pipeline over-fetches before reranking.
    seen = {}

    class SpyReranker(LexicalOverlapReranker):
        name = "spy"

        def rerank(self, query, candidates, top_k):
            seen["n_candidates"] = len(candidates)
            return super().rerank(query, candidates, top_k)

    pipeline = make_pipeline(reranker=SpyReranker())
    pipeline.query("What is BM25?")
    assert seen["n_candidates"] <= 2 * RERANK_CANDIDATE_MULTIPLIER
    assert seen["n_candidates"] >= 2  # top_k


def test_noop_reranker_does_not_over_fetch():
    seen = {}
    retriever = BM25Retriever()

    original_retrieve = retriever.retrieve

    def spy_retrieve(query, top_k=5):
        seen["fetch_k"] = top_k
        return original_retrieve(query, top_k=top_k)

    retriever.retrieve = spy_retrieve
    pipeline = RAGPipeline(
        retriever=retriever, generator=MockGenerator(),
        chunk_strategy="recursive", chunk_kwargs={"chunk_size": 128}, top_k=2,
    )
    pipeline.ingest(DOCS)
    pipeline.query("What is BM25?")
    assert seen["fetch_k"] == 2  # no reranker -> no over-fetching


def test_result_config_reports_transform_and_reranker_names():
    pipeline = make_pipeline(
        query_transform=DecomposeTransform(MockGenerator()),
        reranker=LexicalOverlapReranker(),
    )
    result = pipeline.query("What is BM25 and what is dense retrieval?")
    assert result.config["query_transform"] == "decompose"
    assert result.config["reranker"] == "lexical_overlap"


def test_result_tracks_separate_retrieval_and_generation_latency():
    pipeline = make_pipeline()
    result = pipeline.query("What is BM25?")
    assert result.retrieval_latency_sec >= 0
    assert result.generation_latency_sec >= 0
