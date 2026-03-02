from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Protocol
import hashlib
import time
from openai import OpenAI
import os

class Generator(Protocol):
    def generate(self, query: str, context: List[str]) -> "GenerationResult":
        ...

@dataclass(frozen=True)
class GenerationResult:
    text: str
    model_id: str
    latency_ms: float
    prompt_hash: str
    sampling: Dict[str, object]

def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

@dataclass
class SimpleLLMGenerator:
    """ A thin wrapper over an LLM model API. """
    model_id: str
    base_prompt: str
    temperature: float = 0.0
    max_tokens: int = int(os.getenv("GENERATOR_MAX_TOKENS", 512))

    def generate(self, query: str, context: List[str]) -> GenerationResult:
        start_time = time.perf_counter()

        joined_context = "\n\n---\n\n".join(context) if context else ""

        # Support template-style prompts with {context} and {question} placeholders
        if "{context}" in self.base_prompt and "{question}" in self.base_prompt:
            prompt = self.base_prompt.format(context=joined_context, question=query)
            system = "You are a strict QA system. Answer based ONLY on the provided context."
        else:
            prompt = (
                f"{self.base_prompt}\n\n"
                "Context:\n"
                + joined_context
                + f"\n\nQuestion: {query}\n\n"
                "Answer:"
            )
            system = self.base_prompt

        prompt_hash = _hash_prompt(prompt)

        # OpenAI API call
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        answer = response.choices[0].message.content if context else "No context provided."

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        return GenerationResult(
            text=answer,
            model_id=self.model_id,
            latency_ms=latency_ms,
            prompt_hash=prompt_hash,
            sampling={
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            },
        )