from rag.chunking import Document, chunk_document
from rag.generation import MockGenerator
from rag.retrieval import BM25Retriever
from rag.query_transform import (
    IdentityTransform, HyDETransform, MultiQueryFusionTransform, DecomposeTransform,
    QUERY_TRANSFORMS,
)

DOCS = [
    Document(doc_id="bm25_doc", text="BM25 is a sparse lexical ranking function based on term frequency."),
    Document(doc_id="dense_doc", text="Dense retrieval encodes text into embeddings for semantic search."),
    Document(doc_id="hybrid_doc", text="Hybrid retrieval combines BM25 sparse scores with dense embeddings."),
]


def make_indexed_retriever():
    chunks = []
    for d in DOCS:
        chunks.extend(chunk_document(d, strategy="recursive", chunk_size=512))
    retriever = BM25Retriever()
    retriever.index(chunks)
    return retriever


def test_identity_transform_matches_plain_retrieve():
    retriever = make_indexed_retriever()
    out = IdentityTransform().get_chunks("What is BM25?", retriever, top_k=2)
    direct = retriever.retrieve("What is BM25?", top_k=2)
    assert [c.chunk.chunk_id for c in out] == [c.chunk.chunk_id for c in direct]


def test_hyde_falls_back_to_identity_with_mock_generator():
    retriever = make_indexed_retriever()
    transform = HyDETransform(MockGenerator())
    out = transform.get_chunks("What is BM25?", retriever, top_k=2)
    direct = retriever.retrieve("What is BM25?", top_k=2)
    assert [c.chunk.chunk_id for c in out] == [c.chunk.chunk_id for c in direct]


def test_multi_query_fusion_returns_top_k_with_mock_generator():
    retriever = make_indexed_retriever()
    transform = MultiQueryFusionTransform(MockGenerator(), n_queries=3)
    out = transform.get_chunks("What is BM25 and how does it compare to dense retrieval?", retriever, top_k=2)
    assert len(out) <= 2
    assert all(sc.score > 0 for sc in out)


def test_decompose_splits_on_conjunction_and_covers_both_topics():
    retriever = make_indexed_retriever()
    transform = DecomposeTransform(MockGenerator())
    out = transform.get_chunks("What is BM25 and what is dense retrieval?", retriever, top_k=3)
    doc_ids = {sc.chunk.doc_id for sc in out}
    # Splitting on "and" should retrieve for both sub-questions, so both
    # topics should be represented rather than one dominating.
    assert "bm25_doc" in doc_ids
    assert "dense_doc" in doc_ids


def test_decompose_no_conjunction_behaves_like_identity():
    retriever = make_indexed_retriever()
    transform = DecomposeTransform(MockGenerator())
    out = transform.get_chunks("What is BM25?", retriever, top_k=2)
    direct = retriever.retrieve("What is BM25?", top_k=2)
    assert {c.chunk.chunk_id for c in out} == {c.chunk.chunk_id for c in direct}


def test_registry_contains_expected_keys():
    assert set(QUERY_TRANSFORMS) == {"identity", "hyde", "multi_query", "decompose"}
