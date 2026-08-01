from rag.chunking import Chunk
from rag.retrieval import ScoredChunk
from eval.metrics import (
    precision_at_k, recall_at_k, mrr, ndcg_at_k,
    lexical_faithfulness, refusal_rate,
    average_precision, f1_at_k, mean_average_precision,
)


def make_scored(doc_ids):
    return [
        ScoredChunk(
            chunk=Chunk(chunk_id=f"c{i}", doc_id=d, text="x", start_char=0, end_char=1),
            score=1.0 - i * 0.1,
        )
        for i, d in enumerate(doc_ids)
    ]


def test_precision_at_k_all_relevant():
    retrieved = make_scored(["a", "b", "c"])
    assert precision_at_k(retrieved, {"a", "b", "c"}, k=3) == 1.0


def test_precision_at_k_none_relevant():
    retrieved = make_scored(["a", "b", "c"])
    assert precision_at_k(retrieved, {"x", "y"}, k=3) == 0.0


def test_precision_at_k_partial():
    retrieved = make_scored(["a", "b", "c", "d"])
    assert precision_at_k(retrieved, {"a", "c"}, k=4) == 0.5


def test_recall_at_k():
    retrieved = make_scored(["a", "b"])
    assert abs(recall_at_k(retrieved, {"a", "b", "c"}, k=2) - 2 / 3) < 1e-6


def test_mrr_first_position():
    retrieved = make_scored(["a", "b", "c"])
    assert mrr(retrieved, {"a"}) == 1.0


def test_mrr_third_position():
    retrieved = make_scored(["x", "y", "a"])
    assert abs(mrr(retrieved, {"a"}) - 1 / 3) < 1e-6


def test_mrr_not_found():
    retrieved = make_scored(["x", "y", "z"])
    assert mrr(retrieved, {"a"}) == 0.0


def test_ndcg_perfect_ranking():
    retrieved = make_scored(["a", "b"])
    assert ndcg_at_k(retrieved, {"a", "b"}, k=2) == 1.0


def test_ndcg_no_relevant_docs_returns_zero_not_error():
    retrieved = make_scored(["a", "b"])
    assert ndcg_at_k(retrieved, set(), k=2) == 0.0


def test_lexical_faithfulness_full_overlap():
    answer = "the cat sat on the mat"
    context = ["the cat sat on the mat today"]
    score = lexical_faithfulness(answer, context)
    assert score == 1.0


def test_lexical_faithfulness_no_overlap():
    answer = "quantum entanglement drives the reaction"
    context = ["the cat sat on the mat"]
    score = lexical_faithfulness(answer, context)
    assert score < 0.5


def test_refusal_rate():
    answers = [
        "I don't have enough information to answer that.",
        "The answer is 42.",
        "I don't have enough information to answer that.",
        "Paris is the capital of France.",
    ]
    assert refusal_rate(answers) == 0.5


def test_refusal_rate_empty_list():
    assert refusal_rate([]) == 0.0


def test_average_precision_perfect_ranking():
    retrieved = make_scored(["a", "b"])
    assert average_precision(retrieved, {"a", "b"}, k=2) == 1.0


def test_average_precision_rewards_earlier_hits():
    # relevant doc at rank 1 vs rank 2 (only one other relevant doc, at
    # rank 1 in each) -- earlier ranking should score strictly higher.
    early = make_scored(["a", "b", "c"])  # a,b relevant, both early
    late = make_scored(["b", "x", "a"])   # a relevant but pushed to rank 3
    ap_early = average_precision(early, {"a", "b"}, k=3)
    ap_late = average_precision(late, {"a", "b"}, k=3)
    assert ap_early > ap_late


def test_average_precision_no_relevant_docs():
    retrieved = make_scored(["a", "b"])
    assert average_precision(retrieved, set(), k=2) == 0.0


def test_f1_at_k_matches_precision_recall_harmonic_mean():
    retrieved = make_scored(["a", "b", "c", "d"])
    p = precision_at_k(retrieved, {"a", "c"}, k=4)
    r = recall_at_k(retrieved, {"a", "c"}, k=4)
    expected = 2 * p * r / (p + r)
    assert abs(f1_at_k(retrieved, {"a", "c"}, k=4) - expected) < 1e-9


def test_f1_at_k_zero_when_no_hits():
    retrieved = make_scored(["a", "b"])
    assert f1_at_k(retrieved, {"x"}, k=2) == 0.0


def test_mean_average_precision():
    assert mean_average_precision([1.0, 0.5, 0.0]) == 0.5


def test_mean_average_precision_empty():
    assert mean_average_precision([]) == 0.0
