from rag.chunking import Document
from rag.generation import MockGenerator
from eval.ablation import AblationConfig, run_ablation
from eval.dataset import EvalExample

DOCS = [
    Document(doc_id="bm25_doc", text="BM25 is a sparse lexical ranking function based on term frequency. " * 3),
    Document(doc_id="dense_doc", text="Dense retrieval encodes text into embeddings for semantic search. " * 3),
]

EXAMPLES = [
    EvalExample(query_id="q1", question="What is BM25?", relevant_doc_ids=["bm25_doc"]),
    EvalExample(query_id="q2", question="What is dense retrieval?", relevant_doc_ids=["dense_doc"]),
]


def test_run_ablation_sweeps_new_axes():
    config = AblationConfig(
        chunk_strategies=["recursive"],
        chunk_sizes=[128],
        retriever_names=["bm25"],
        top_k_values=[2],
        query_transform_names=["identity", "decompose"],
        reranker_names=["none", "lexical_overlap"],
    )
    results = run_ablation(DOCS, EXAMPLES, MockGenerator(), config, verbose=False)

    # 1 chunk_strategy x 1 size x 1 retriever x 1 top_k x 2 transforms x
    # 2 rerankers x 2 queries = 8 rows.
    assert len(results) == 8

    seen_combos = {(r["query_transform"], r["reranker"]) for r in results}
    assert seen_combos == {
        ("identity", "none"), ("identity", "lexical_overlap"),
        ("decompose", "none"), ("decompose", "lexical_overlap"),
    }


def test_run_ablation_reports_map_and_cost_columns():
    config = AblationConfig(
        chunk_strategies=["recursive"], chunk_sizes=[128],
        retriever_names=["bm25"], top_k_values=[2],
    )
    results = run_ablation(DOCS, EXAMPLES, MockGenerator(), config, verbose=False)
    row = results[0]
    assert "map_for_config" in row
    assert "estimated_tokens" in row
    assert "estimated_cost_usd" in row
    assert "retrieval_latency_sec" in row
    assert "generation_latency_sec" in row
    # Local model, default cost_per_1k_tokens=0.0 -> cost should be zero.
    assert row["estimated_cost_usd"] == 0.0


def test_run_ablation_nonzero_cost_when_configured():
    config = AblationConfig(
        chunk_strategies=["recursive"], chunk_sizes=[128],
        retriever_names=["bm25"], top_k_values=[2],
        cost_per_1k_tokens=1.0,
    )
    results = run_ablation(DOCS, EXAMPLES, MockGenerator(), config, verbose=False)
    assert all(r["estimated_cost_usd"] > 0 for r in results)


def test_run_ablation_without_generation_skips_generation_columns():
    config = AblationConfig(
        chunk_strategies=["recursive"], chunk_sizes=[128],
        retriever_names=["bm25"], top_k_values=[2],
    )
    results = run_ablation(DOCS, EXAMPLES, MockGenerator(), config, run_generation=False, verbose=False)
    assert "answer" not in results[0]
    assert "lexical_faithfulness" not in results[0]
