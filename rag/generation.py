"""
Generation backends. All local/open-source, no paid API required.

- OllamaGenerator: talks to a local Ollama server (easiest way to run
  something like llama3 or mistral locally with no ML setup).
- HFGenerator: loads a HuggingFace causal LM directly (more control,
  needs more RAM/VRAM and a model download).
- MockGenerator: deterministic, no model required. Used for testing the
  pipeline plumbing and eval harness logic in environments with no
  internet/model access. NOT a substitute for real generation quality
  eval -- swap in Ollama/HF before drawing conclusions from the ablation.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod

from rag.chunking import Chunk

RAG_PROMPT_TEMPLATE = """Answer the question using ONLY the information in the context below. \
If the context does not contain the answer, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""


class BaseGenerator(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, prompt: str) -> str:
        ...

    def answer(self, question: str, chunks: list[Chunk]) -> str:
        context = "\n\n".join(
            f"[{i+1}] {c.text}" for i, c in enumerate(chunks)
        )
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        return self.generate(prompt)


class OllamaGenerator(BaseGenerator):
    """Requires a local Ollama server: https://ollama.com
    Run `ollama pull llama3.2` (or similar) then `ollama serve` before use.
    """

    name = "ollama"

    def __init__(self, model: str = "llama3.2", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host

    def generate(self, prompt: str) -> str:
        import urllib.request

        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read())
                return data.get("response", "").strip()
        except Exception as e:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host}. Is `ollama serve` "
                f"running and have you pulled model '{self.model}'? "
                f"Original error: {e}"
            ) from e


class HFGenerator(BaseGenerator):
    """Loads a HuggingFace causal LM directly. Good for small models like
    Qwen2.5-1.5B-Instruct or Phi-3-mini on modest hardware. Requires
    `transformers` + `torch` and a model download on first use."""

    name = "huggingface"

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 max_new_tokens: int = 256):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            try:
                from transformers import pipeline
            except ImportError as e:
                raise ImportError(
                    "transformers not installed. Run: pip install transformers torch"
                ) from e
            self._pipe = pipeline("text-generation", model=self.model_name)
        return self._pipe

    def generate(self, prompt: str) -> str:
        pipe = self._load()
        out = pipe(prompt, max_new_tokens=self.max_new_tokens,
                    do_sample=False, return_full_text=False)
        return out[0]["generated_text"].strip()


class MockGenerator(BaseGenerator):
    """No model required. Produces a deterministic, template-based
    'answer' that quotes the top retrieved chunk. This exists purely so
    the pipeline and eval harness can be smoke-tested end-to-end without
    internet/model access -- swap in Ollama or HF for real experiments.
    """

    name = "mock"

    def generate(self, prompt: str) -> str:
        # Extract the question and first context snippet for a
        # deterministic, inspectable "answer".
        return "[MOCK GENERATION - not a real model response]"

    def answer(self, question: str, chunks: list[Chunk]) -> str:
        if not chunks:
            return "I don't have enough information to answer that."
        top = chunks[0].text.strip().split(". ")[0]
        return f"Based on the retrieved context: {top}."


GENERATORS = {
    "ollama": OllamaGenerator,
    "huggingface": HFGenerator,
    "mock": MockGenerator,
}
