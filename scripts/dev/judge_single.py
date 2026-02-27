"""Run LLM-as-judge on a single gold/ai answer pair.

Usage:
    python scripts/dev/judge_single.py --gold "Paris" --ai "The capital of France is Paris."
    python scripts/dev/judge_single.py --gold "1969" --ai "Apollo 11 landed in 1969." --question "When did Apollo 11 land?"
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from raqr.llm_judge import LLMJudge


SCORE_LABELS = {
    0: "Incorrect / Refusal",
    1: "Partial / Superficial",
    2: "Comprehensive / Correct",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run LLM-as-judge on a single gold vs AI answer pair.",
    )
    parser.add_argument(
        "--gold",
        required=True,
        help="Golden (reference) answer.",
    )
    parser.add_argument(
        "--ai",
        required=True,
        help="AI-predicted answer to judge.",
    )
    parser.add_argument(
        "--question",
        default="",
        help="Optional question context (for judge prompt).",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Set it in .env or environment.", file=sys.stderr)
        return 1

    gold_list = [args.gold.strip()] if args.gold.strip() else []
    if not gold_list:
        print("Gold answer cannot be empty.", file=sys.stderr)
        return 1

    judge = LLMJudge()
    score = judge.judge(
        question=args.question or "(no question provided)",
        gold_answers=gold_list,
        predicted_answer=args.ai or "(empty)",
    )

    label = SCORE_LABELS.get(score, f"Unknown ({score})")
    print(f"Score: {score} — {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
