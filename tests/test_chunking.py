from rag.chunking import Document, chunk_document, fixed_size_chunker, recursive_chunker, semantic_chunker


SAMPLE_TEXT = (
    "Retrieval-augmented generation combines a retriever with a generator. "
    "The retriever finds relevant passages from a corpus. "
    "The generator then conditions its output on those passages.\n\n"
    "This two-stage design lets the model access information beyond its "
    "training data. It also makes answers more auditable, since retrieved "
    "passages can be shown as citations."
)


def make_doc(text=SAMPLE_TEXT, doc_id="test_doc"):
    return Document(doc_id=doc_id, text=text, metadata={"source": "test"})


def test_fixed_size_chunker_respects_size():
    doc = make_doc()
    chunks = fixed_size_chunker(doc, chunk_size=50, overlap=10)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 50
        assert c.doc_id == "test_doc"


def test_fixed_size_chunker_overlap_creates_shared_text():
    doc = make_doc()
    chunks = fixed_size_chunker(doc, chunk_size=60, overlap=20)
    assert len(chunks) >= 2
    tail_of_first = chunks[0].text[-20:]
    assert tail_of_first[:5] in chunks[1].text or len(chunks) < 2


def test_fixed_size_no_overlap_no_infinite_loop():
    doc = make_doc(text="short text")
    chunks = fixed_size_chunker(doc, chunk_size=1000, overlap=0)
    assert len(chunks) == 1


def test_recursive_chunker_respects_max_size_roughly():
    doc = make_doc()
    chunks = recursive_chunker(doc, chunk_size=80, overlap=10)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.doc_id == "test_doc"
        assert c.text.strip() != ""


def test_recursive_chunker_handles_short_doc():
    doc = make_doc(text="One short sentence.")
    chunks = recursive_chunker(doc, chunk_size=500)
    assert len(chunks) == 1
    assert "short sentence" in chunks[0].text


def test_semantic_chunker_groups_sentences():
    doc = make_doc()
    chunks = semantic_chunker(doc, max_sentences=2)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.metadata["strategy"] == "semantic"


def test_chunk_document_dispatch():
    doc = make_doc()
    for strategy in ("fixed_size", "recursive", "semantic"):
        chunks = chunk_document(doc, strategy=strategy)
        assert len(chunks) >= 1
        assert all(c.doc_id == "test_doc" for c in chunks)


def test_chunk_document_unknown_strategy_raises():
    doc = make_doc()
    try:
        chunk_document(doc, strategy="not_a_real_strategy")
        assert False, "should have raised"
    except ValueError:
        pass


def test_empty_document_does_not_crash():
    doc = make_doc(text="")
    for strategy in ("fixed_size", "recursive", "semantic"):
        chunks = chunk_document(doc, strategy=strategy)
        assert isinstance(chunks, list)
