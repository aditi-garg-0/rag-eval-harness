from rag.chunking import Chunk
from rag.retrieval import ScoredChunk
from rag.rerank import NoOpReranker, LexicalOverlapReranker, RERANKERS


def make_candidates(texts):
    return [
        ScoredChunk(
            chunk=Chunk(chunk_id=f"c{i}", doc_id=f"d{i}", text=t, start_char=0, end_char=len(t)),
            score=1.0 - i * 0.1,  # descending first-stage score, arbitrary order
        )
        for i, t in enumerate(texts)
    ]


def test_noop_reranker_preserves_order_and_truncates():
    candidates = make_candidates(["a", "b", "c", "d"])
    out = NoOpReranker().rerank("query", candidates, top_k=2)
    assert [c.chunk.chunk_id for c in out] == ["c0", "c1"]


def test_noop_reranker_empty():
    assert NoOpReranker().rerank("query", [], top_k=3) == []


def test_lexical_overlap_reranker_promotes_better_match():
    # c0 has no term overlap with the query; c1 fully overlaps. Even
    # though c0 has a higher first-stage score, lexical overlap reranking
    # should promote c1 to the top.
    candidates = [
        ScoredChunk(chunk=Chunk(chunk_id="c0", doc_id="d0", text="completely unrelated text",
                                 start_char=0, end_char=1), score=0.9),
        ScoredChunk(chunk=Chunk(chunk_id="c1", doc_id="d1", text="dense retrieval embeddings",
                                 start_char=0, end_char=1), score=0.1),
    ]
    out = LexicalOverlapReranker().rerank("dense retrieval embeddings", candidates, top_k=2)
    assert out[0].chunk.chunk_id == "c1"


def test_lexical_overlap_reranker_empty():
    assert LexicalOverlapReranker().rerank("query", [], top_k=3) == []


def test_reranker_registry_contains_expected_keys():
    assert set(RERANKERS) == {"none", "lexical_overlap", "cross_encoder"}
