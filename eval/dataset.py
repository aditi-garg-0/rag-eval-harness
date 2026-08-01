"""
Labeled query set for evaluation. Each EvalExample pairs a question with
the doc_id(s) that actually contain the answer (needed for retrieval
metrics), and optionally a reference answer (useful for spot-checking the
judge, though we don't do answer-similarity scoring against it by default
since that tends to just reward paraphrase-matching).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalExample:
    query_id: str
    question: str
    relevant_doc_ids: list[str]
    reference_answer: str | None = None
    notes: str = ""


def load_eval_set(path: str | Path) -> list[EvalExample]:
    path = Path(path)
    with open(path) as f:
        raw = json.load(f)
    return [
        EvalExample(
            query_id=item["query_id"],
            question=item["question"],
            relevant_doc_ids=item["relevant_doc_ids"],
            reference_answer=item.get("reference_answer"),
            notes=item.get("notes", ""),
        )
        for item in raw
    ]


def save_eval_set(examples: list[EvalExample], path: str | Path) -> None:
    path = Path(path)
    payload = [
        {
            "query_id": e.query_id,
            "question": e.question,
            "relevant_doc_ids": e.relevant_doc_ids,
            "reference_answer": e.reference_answer,
            "notes": e.notes,
        }
        for e in examples
    ]
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
