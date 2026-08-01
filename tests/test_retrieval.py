from rag.chunking import Document, chunk_document
from rag.retrieval import BM25Retriever


def build_chunks():
    docs = [
        Document(doc_id="d1", text="The cat sat on the warm mat in the sun."),
        Document(doc_id="d2", text="Retrieval augmented generation improves factual grounding."),
        Document(doc_id="d3", text="Dogs are loyal animals that enjoy long walks outside."),
    ]
    chunks = []
    for d in docs:
        chunks.extend(chunk_document(d, strategy="fixed_size", chunk_size=200, overlap=0))
    return chunks


def test_bm25_retrieves_lexically_matching_doc_first():
    retriever = BM25Retriever()
    chunks = build_chunks()
    retriever.index(chunks)
    results = retriever.retrieve("retrieval augmented generation grounding", top_k=3)
    assert results[0].chunk.doc_id == "d2"


def test_bm25_returns_requested_top_k():
    retriever = BM25Retriever()
    chunks = build_chunks()
    retriever.index(chunks)
    results = retriever.retrieve("animals", top_k=2)
    assert len(results) <= 2


def test_bm25_scores_are_descending():
    retriever = BM25Retriever()
    chunks = build_chunks()
    retriever.index(chunks)
    results = retriever.retrieve("cat sat mat", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_raises_if_not_indexed():
    retriever = BM25Retriever()
    try:
        retriever.retrieve("test", top_k=3)
        assert False, "should have raised"
    except RuntimeError:
        pass
