"""Loads a corpus.json (from fetch_corpus.py or the sample corpus) into
Document objects the pipeline can ingest."""
from __future__ import annotations

import json
from pathlib import Path

from rag.chunking import Document


def load_corpus(path: str | Path) -> list[Document]:
    with open(path) as f:
        raw = json.load(f)
    return [
        Document(
            doc_id=item["doc_id"],
            text=item["text"],
            metadata={k: v for k, v in item.items() if k not in ("doc_id", "text")},
        )
        for item in raw
    ]
