"""OpenAI Batch API helpers for strategy-generation requests.

Builds chat completion request bodies matching SimpleLLMGenerator format,
so batch outputs are equivalent to synchronous generation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

from raqr.generator import GenerationResult
from raqr.prompts import GENERATOR_SYSTEM_MESSAGE


def build_generation_request(
    query: str,
    context: List[str],
    base_prompt: str | None = None,
    model_id: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Dict[str, Any]:
    """Build a chat completions request body for batch API. Uses the same prompt logic as SimpleLLMGenerator for parity."""
    model_id = model_id or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = (
        temperature
        if temperature is not None
        else float(os.getenv("GENERATOR_TEMPERATURE", "0"))
    )
    max_tokens = (
        max_tokens
        if max_tokens is not None
        else int(os.getenv("GENERATOR_MAX_TOKENS", "512"))
    )

    joined_context = "\n\n---\n\n".join(context) if context else ""

    if "{context}" in base_prompt and "{question}" in base_prompt:
        prompt = base_prompt.format(context=joined_context, question=query)
        system = GENERATOR_SYSTEM_MESSAGE
    else:
        prompt = (
            f"{base_prompt}\n\n"
            "Context:\n" + joined_context + f"\n\nQuestion: {query}\n\n"
            "Answer:"
        )
        system = base_prompt

    return {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def build_batch_line(custom_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """Build a single line for the Batch API input JSONL file."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def parse_generation_output(line: Dict[str, Any]) -> tuple[str, str]:
    """Parse a Batch API output line into (custom_id, answer_text).

    Returns ("", "") for failed or empty responses.
    """
    custom_id = line.get("custom_id") or ""

    if line.get("error"):
        return (custom_id, "")

    response = line.get("response")
    if not response or response.get("status_code") != 200:
        return (custom_id, "")

    body = response.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return (custom_id, "")

    msg = choices[0].get("message") or {}
    content = msg.get("content") or ""

    # Parse answer (strip "Answer:" prefix if present, matching SimpleLLMGenerator)
    if "Answer:" in content:
        content = content.split("Answer:", 1)[1].strip()
    return (custom_id, content.strip())


@dataclass
class BatchRecorderGenerator:
    """Generator that records (query, context) for batch prep instead of calling the API.

    Set next_custom_id before each strategy.retrieve_and_generate() call;
    the recorder captures the request and returns a stub. Use recorded_requests
    to build the batch JSONL.
    """

    base_prompt: str | None = None
    model_id: str = ""
    temperature: float = 0.0
    max_tokens: int = 512

    next_custom_id: str = ""
    recorded_requests: List[Dict[str, Any]] = field(default_factory=list)

    def generate(self, query: str, context: List[str]) -> GenerationResult:
        """Record the request and return a stub. No API call."""
        cid = self.next_custom_id or "unknown"
        body = build_generation_request(
            query=query,
            context=context,
            base_prompt=self.base_prompt,
            model_id=self.model_id or None,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self.recorded_requests.append(
            {
                "custom_id": cid,
                "body": body,
            }
        )
        return GenerationResult(
            text="",
            model_id=self.model_id or "batch",
            latency_ms=0.0,
            prompt_hash="",
            sampling={},
        )
