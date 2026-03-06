"""Query Graph RAG with a question, then judge the answer with LLM-as-judge.

Usage:
    python scripts/dev/query_graph_judge.py "When did Apollo 11 land?" --gold "1969"
    python scripts/dev/query_graph_judge.py "What is the capital of France?" --gold "Paris"

Output: Answer, then Score (0/1/2).
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _common import build_graph_strategy

from raqr.llm_judge import LLMJudge


SCORE_LABELS = {
    0: "Incorrect / Refusal",
    1: "Partial / Superficial",
    2: "Comprehensive / Correct",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Query Graph RAG and judge the answer with LLM-as-judge.",
    )
    parser.add_argument(
        "--question",
        help="The question to ask Graph RAG.",
    )
    parser.add_argument(
        "--gold",
        required=True,
        help="Golden (reference) answer for judging.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed"),
        help="Directory containing corpus, graph, alias_map.",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Set it in .env or environment.", file=sys.stderr)
        return 1

    gold_list = [args.gold.strip()] if args.gold.strip() else []
    if not gold_list:
        print("Gold answer cannot be empty.", file=sys.stderr)
        return 1

    print("Building Graph strategy...")
    strategy = build_graph_strategy(args.output_dir)
    result = strategy.retrieve_and_generate(args.question)

    if result.status == "ERROR":
        print("Error:", result.error or "Unknown error", file=sys.stderr)
        return 1

    answer = result.answer or ""

    print("Judging with LLM-as-judge...")
    judge = LLMJudge()
    score = judge.judge(
        question=args.question,
        gold_answers=gold_list,
        predicted_answer=answer,
    )

    label = SCORE_LABELS.get(score, f"Unknown ({score})")
    print()
    print("Answer:", answer)
    print("Score:", score, "—", label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
