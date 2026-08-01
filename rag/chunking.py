"""
Chunking strategies for RAG ingestion.

Each strategy takes a document (with metadata) and returns a list of Chunk
objects. Keeping these as pluggable, independently testable units is what
lets the ablation runner sweep over chunking strategy as one axis of the
experiment grid.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)


def _sentence_split(text: str) -> list[str]:
    """Lightweight sentence splitter (no external model download required).

    Not linguistically perfect, but deterministic and dependency-free,
    which matters for a project meant to run fully offline.
    """
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []
    # Split on sentence-ending punctuation followed by whitespace + capital,
    # while trying to avoid breaking on common abbreviations / decimals.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"])", text)
    return [p.strip() for p in parts if p.strip()]


def fixed_size_chunker(
    doc: Document, chunk_size: int = 512, overlap: int = 64
) -> list[Chunk]:
    """Naive fixed-size character chunking with overlap.

    This is the baseline everyone starts with. Fast, simple, but can slice
    sentences (and ideas) in half -- one of the things the eval harness is
    meant to surface.
    """
    text = doc.text
    chunks = []
    start = 0
    idx = 0
    step = max(chunk_size - overlap, 1)
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        if chunk_text.strip():
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::fixed::{idx}",
                    doc_id=doc.doc_id,
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    metadata={**doc.metadata, "strategy": "fixed_size",
                              "chunk_size": chunk_size, "overlap": overlap},
                )
            )
            idx += 1
        if end == len(text):
            break
        start += step
    return chunks


def recursive_chunker(
    doc: Document,
    chunk_size: int = 512,
    overlap: int = 64,
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " "),
) -> list[Chunk]:
    """Recursively split on a hierarchy of separators (paragraph -> line ->
    sentence -> word) until pieces fit under chunk_size, then re-merge with
    overlap. Mirrors the common "RecursiveCharacterTextSplitter" pattern
    used in most production RAG stacks.
    """

    def split(text: str, seps: list[str]) -> list[str]:
        if len(text) <= chunk_size or not seps:
            return [text]
        sep = seps[0]
        pieces = text.split(sep) if sep else list(text)
        pieces = [p for p in pieces if p]
        out = []
        for p in pieces:
            if len(p) > chunk_size:
                out.extend(split(p, seps[1:]))
            else:
                out.append(p)
        return out

    raw_pieces = split(doc.text, list(separators))

    # Re-merge small pieces up to chunk_size, with overlap carried forward.
    merged: list[str] = []
    current = ""
    for piece in raw_pieces:
        candidate = (current + " " + piece).strip() if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                merged.append(current)
            current = piece
    if current:
        merged.append(current)

    # Apply overlap by prepending the tail of the previous chunk.
    chunks = []
    offset_cursor = 0
    prev_tail = ""
    for i, m in enumerate(merged):
        text_with_overlap = (prev_tail + " " + m).strip() if prev_tail else m
        start = doc.text.find(m, offset_cursor)
        if start == -1:
            start = offset_cursor
        end = start + len(m)
        offset_cursor = max(offset_cursor, end)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::recursive::{i}",
                doc_id=doc.doc_id,
                text=text_with_overlap,
                start_char=start,
                end_char=end,
                metadata={**doc.metadata, "strategy": "recursive",
                          "chunk_size": chunk_size, "overlap": overlap},
            )
        )
        prev_tail = m[-overlap:] if overlap > 0 else ""
    return chunks


def semantic_chunker(
    doc: Document,
    max_sentences: int = 6,
    min_sentences: int = 2,
) -> list[Chunk]:
    """Groups sentences into chunks. This is a simplified stand-in for
    embedding-based semantic chunking (which would compute sentence
    embeddings and split at topic-shift boundaries). Here we split at
    paragraph boundaries first, then cap group size by sentence count --
    a decent proxy that doesn't require downloading a sentence-embedding
    model to run offline.

    Swap `group_boundaries` for an embedding-similarity based splitter
    once you have model access; the Chunk interface stays the same.
    """
    paragraphs = [p.strip() for p in doc.text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [doc.text]

    chunks = []
    idx = 0
    cursor = 0
    for para in paragraphs:
        sentences = _sentence_split(para)
        if not sentences:
            continue
        group: list[str] = []
        for sent in sentences:
            group.append(sent)
            if len(group) >= max_sentences:
                chunk_text = " ".join(group)
                start = doc.text.find(chunk_text, cursor)
                start = start if start != -1 else cursor
                end = start + len(chunk_text)
                cursor = end
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}::semantic::{idx}",
                    doc_id=doc.doc_id, text=chunk_text,
                    start_char=start, end_char=end,
                    metadata={**doc.metadata, "strategy": "semantic"},
                ))
                idx += 1
                group = []
        if len(group) >= min_sentences or (group and not chunks):
            chunk_text = " ".join(group)
            if chunk_text.strip():
                start = doc.text.find(chunk_text, cursor)
                start = start if start != -1 else cursor
                end = start + len(chunk_text)
                cursor = end
                chunks.append(Chunk(
                    chunk_id=f"{doc.doc_id}::semantic::{idx}",
                    doc_id=doc.doc_id, text=chunk_text,
                    start_char=start, end_char=end,
                    metadata={**doc.metadata, "strategy": "semantic"},
                ))
                idx += 1
    return chunks


CHUNKERS: dict[str, Callable[..., list[Chunk]]] = {
    "fixed_size": fixed_size_chunker,
    "recursive": recursive_chunker,
    "semantic": semantic_chunker,
}


def chunk_document(doc: Document, strategy: str = "recursive", **kwargs) -> list[Chunk]:
    if strategy not in CHUNKERS:
        raise ValueError(f"Unknown chunking strategy: {strategy}. "
                          f"Available: {list(CHUNKERS)}")
    return CHUNKERS[strategy](doc, **kwargs)
