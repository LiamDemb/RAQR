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

        # Prompt template
        prompt = (
            f'{self.base_prompt}\n\n'
            f'Query: {query}\n\n'
            f'Context:\n' + '\n\n---\n\n'.join(context)
        )
        prompt_hash = _hash_prompt(prompt)

        # OpenAI API call
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": self.base_prompt},
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