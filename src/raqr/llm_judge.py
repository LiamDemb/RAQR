"""LLM-as-judge module for semantic correctness evaluation of strategy outputs.

Used for oracle label generation and strategy comparison when F1/EM are
unreliable due to diverse answer formats or verbose model outputs.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI


DEFAULT_JUDGE_PROMPT_OLD = """You are an expert evaluator grading an AI's answer against a Ground Truth reference. 
Focus strictly on factual and semantic equivalence. Ignore differences in wording, length, or conversational boilerplate.

Grade the AI Answer on this strict 0-2 scale:
[0] Incorrect / Refusal: The answer is factually wrong, completely misses the core meaning, or states that the context is insufficient.
[1] Partial / Superficial: The answer contains relevant keywords or partial facts, but misses the specific reasoning, causality, or completeness of the Ground Truth.
[2] Comprehensive / Correct: The answer fully captures the core facts and logical meaning of the Ground Truth.

Original Question: {question}
Ground Truth: {gold_answers}
AI Answer: {predicted_answer}

Output ONLY a single integer (0, 1, or 2). Do not output any other text or explanation."""

DEFAULT_JUDGE_PROMPT = """You are an expert evaluator grading an AI's answer against a Ground Truth reference for a factual Question Answering task. 
Your only goal is to determine if the AI successfully retrieved the core fact requested by the question.

CRITICAL GRADING RULES:
- Entity Resolution: Treat aliases, initials, acronyms, and missing middle names as equivalent (e.g., "E.L. Doctorow" perfectly matches "Edgar Lawrence 'E. L.' Doctorow").
- Embedded Answers: If the Ground Truth is explicitly present in the AI Answer, it is correct. Ignore extra conversational text or hallucinated context as long as the exact answer to the question is provided.

Grade the AI Answer strictly on this 0-2 scale:
[2] Correct: The AI Answer contains the exact Ground Truth fact, entity, or date. It successfully answers the question, regardless of minor formatting differences or extra text.
[1] Partial: The AI Answer contains some correct elements (e.g., 1 out of 2 items in a list) but is incomplete, or it is slightly too broad to be a perfect match.
[0] Incorrect: The AI Answer completely misses the Ground Truth, contradicts it, or states "insufficient context".

Original Question: {question}
Ground Truth: {gold_answers}
AI Answer: {predicted_answer}

Output ONLY a single integer (0, 1, or 2). Do not output any other text or explanation."""

@dataclass
class LLMJudge:
    """LLM-based judge for semantic correctness of QA answers.

    Returns 0, 1, or 2:
    - 0: completely incorrect, irrelevant, or no sufficient context
    - 1: partially correct or superficial
    - 2: comprehensive / correct
    """

    model_id: str = ""
    prompt_template: str = ""
    temperature: float = 0.0
    max_tokens: int = 16
    _client: Optional[OpenAI] = None

    def __post_init__(self) -> None:
        if not self.model_id:
            self.model_id = os.getenv("LLM_JUDGE_MODEL", "gpt-4o-mini")
        if not self.prompt_template:
            self.prompt_template = DEFAULT_JUDGE_PROMPT
        env_temp = os.getenv("LLM_JUDGE_TEMPERATURE")
        if env_temp is not None:
            self.temperature = float(env_temp)

    def _get_client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return self._client

    def judge(
        self,
        question: str,
        gold_answers: List[str],
        predicted_answer: str,
    ) -> int:
        """Judge whether the predicted answer correctly addresses the question.

        Args:
            question: The question asked.
            gold_answers: Reference (gold) answer(s).
            predicted_answer: The model's predicted answer.

        Returns:
            0 (incorrect/irrelevant/no context), 1 (partial/superficial), or 2 (comprehensive/correct).
        """
        gold_str = "\n".join(f"- {a}" for a in gold_answers) if gold_answers else "(none)"
        prompt = self.prompt_template.format(
            question=question,
            gold_answers=gold_str,
            predicted_answer=predicted_answer or "(empty)",
        )
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        text = (response.choices[0].message.content or "").strip()
        return self._parse_response(text)

    def _parse_response(self, text: str) -> int:
        """Parse judge response to 0, 1, or 2."""
        if not text:
            return 0
        # Extract first digit 0, 1, or 2 from response (handles "2", "2.", "Score: 2", etc.)
        match = re.search(r"\b([012])\b", text.strip())
        if match:
            return int(match.group(1))
        return 0
