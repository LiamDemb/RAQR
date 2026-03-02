"""Central prompt definitions for generators and LLM-as-judge.

All strategies, evaluation scripts, and integration checks use these prompts
for consistency. To customize for testing:

- Edit the constants below, or
- Set GENERATOR_BASE_PROMPT_FILE / LLM_JUDGE_PROMPT_FILE to a .txt file path
  to load an alternate prompt from disk.
"""

from __future__ import annotations

import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Generator prompt (used by DenseStrategy, GraphStrategy, TemporalStrategy, etc.)
# ---------------------------------------------------------------------------

BASE_PROMPT_OLD = (
    "You are a strict factual answering system. Answer the question based ONLY on the provided context."
    "CRITICAL INSTRUCTIONS:"
    "- Be as concise as possible."
    "- Do NOT repeat the question."
    "- Do NOT use conversational filler like 'Based on the context...' or 'The answer is...'."
    "- If the context does not contain the answer, reply with exactly the word: 'INSUFFICIENT_CONTEXT'."
)

BASE_PROMPT = (
    "You are a strict QA system. Answer based ONLY on the provided context."
    "\n\n"
    "EXAMPLES:"
    "Context: 'Toy Story features a boy named Andy who has a younger sister named Molly.'\n"
    "Question: what is andy's sisters name in toy story\n"
    "Answer: Molly\n\n"
    "Context: 'The PUMA 560 was the first robot used in a surgery, assisting in a biopsy in 1983.'\n"
    "Question: when was the first robot used in surgery\n"
    "Answer: 1983\n\n"
    "Context: 'Donovan Mitchell was selected with the 13th overall pick in the 2017 NBA draft.'\n"
    "Question: where was donovan mitchell picked in the draft\n"
    "Answer: 13th\n\n"
    "Context: 'Gabriela Mistral was a Chilean poet. G. K. Chesterton was an English writer and philosopher.'\n"
    "Question: Were both Gabriela Mistral and G. K. Chesterton authors?\n"
    "Answer: yes"
    "\n\n"
    "YOUR TASK:"
    "Context: {context}\n"
    "Question: {question}\n"
    "Answer:"
)


def get_generator_prompt() -> str:
    """Return the generator base prompt. Override via GENERATOR_BASE_PROMPT_FILE env."""
    path = os.getenv("GENERATOR_BASE_PROMPT_FILE")
    if path and Path(path).is_file():
        return Path(path).read_text(encoding="utf-8")
    return BASE_PROMPT


# ---------------------------------------------------------------------------
# Judge prompt (used by LLMJudge for oracle labels and strategy comparison)
# ---------------------------------------------------------------------------

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


def get_judge_prompt() -> str:
    """Return the judge prompt template. Override via LLM_JUDGE_PROMPT_FILE env."""
    path = os.getenv("LLM_JUDGE_PROMPT_FILE")
    if path and Path(path).is_file():
        return Path(path).read_text(encoding="utf-8")
    return DEFAULT_JUDGE_PROMPT
