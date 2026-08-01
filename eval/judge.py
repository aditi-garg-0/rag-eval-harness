from __future__ import annotations

import json
import re
from dataclasses import dataclass

from rag.generation import BaseGenerator

FAITHFULNESS_PROMPT = """Evaluating whether an AI-generated answer is faithful to (i.e. fully \
supported by) the provided context. An answer is faithful if every claim in it can be \
verified from the context. It is NOT faithful if it adds facts not present in the context.

Context:
{context}

Question: {question}

Answer to evaluate: {answer}

Respond with ONLY a JSON object in this exact format, no other text:
{{"score": <integer 1-5, where 5 is fully faithful and 1 is completely unsupported>, "reason": "<one sentence>"}}"""

RELEVANCE_PROMPT = """Evaluating whether an AI-generated answer actually addresses the question asked, \
regardless of whether it is factually correct.

Question: {question}

Answer to evaluate: {answer}

Respond with ONLY a JSON object in this exact format, no other text:
{{"score": <integer 1-5, where 5 is fully relevant and 1 is completely off-topic>, "reason": "<one sentence>"}}"""


@dataclass
class JudgeScore:
    score: int
    reason: str
    raw_response: str


def _parse_judge_json(raw: str) -> JudgeScore:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return JudgeScore(score=0, reason="PARSE_ERROR: no JSON found", raw_response=raw)
    try:
        data = json.loads(match.group(0))
        return JudgeScore(
            score=int(data.get("score", 0)),
            reason=str(data.get("reason", "")),
            raw_response=raw,
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return JudgeScore(score=0, reason="PARSE_ERROR: malformed JSON", raw_response=raw)


class LLMJudge:
    def __init__(self, judge_generator: BaseGenerator):
        self.judge = judge_generator

    def score_faithfulness(self, question: str, answer: str, context_chunks: list[str]) -> JudgeScore:
        context = "\n\n".join(context_chunks)
        prompt = FAITHFULNESS_PROMPT.format(context=context, question=question, answer=answer)
        raw = self.judge.generate(prompt)
        return _parse_judge_json(raw)

    def score_relevance(self, question: str, answer: str) -> JudgeScore:
        prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
        raw = self.judge.generate(prompt)
        return _parse_judge_json(raw)
