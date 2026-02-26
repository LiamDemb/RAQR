"""GraphRAG evaluation: list questions where the LLM judge returns a specific score.

Evaluates GraphRAG on the benchmark using LLM-as-judge (0/1/2). Prints the list
of questions where the evaluation returned the specified score (default: 0).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _common import build_graph_strategy, normalize_gold_answers


def _progress_bar(current: int, total: int, width: int = 40) -> str:
    """Simple text progress bar: [=========>          ] 45/100"""
    if total <= 0:
        return ""
    pct = current / total
    filled = int(width * pct)
    bar = "=" * filled + ">" * (1 if filled < width else 0) + " " * (width - filled - 1)
    return f"[{bar}] {current}/{total}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GraphRAG eval: list questions where judge returns a specific score (0/1/2).",
    )
    parser.add_argument(
        "--benchmark",
        default=os.getenv("BENCHMARK_PATH", "data/processed/benchmark.jsonl"),
        help="Path to benchmark JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed"),
        help="Directory containing corpus, graph, alias_map.",
    )
    parser.add_argument(
        "--score",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="Filter questions where judge returned this score (0=incorrect, 1=partial, 2=correct).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions (for quick dev runs).",
    )
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        print(f"Benchmark not found: {benchmark_path}", file=sys.stderr)
        return 1

    samples = []
    with benchmark_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if args.limit and len(samples) >= args.limit:
                break

    if not samples:
        print("No benchmark samples.", file=sys.stderr)
        return 1

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Set it in .env or environment.", file=sys.stderr)
        return 1

    from raqr.llm_judge import LLMJudge

    print("Building GraphRAG strategy...")
    graph = build_graph_strategy(args.output_dir)
    judge = LLMJudge()
    print(f"Evaluating GraphRAG on {len(samples)} questions (filtering for score={args.score})")
    print("=" * 60)

    matching_questions: list[str] = []
    total = len(samples)

    for idx, sample in enumerate(samples):
        question = sample.get("question", "")
        gold_raw = sample.get("gold_answers", [])
        gold_list = normalize_gold_answers(gold_raw)

        if not question or not gold_list:
            continue

        r_graph = graph.retrieve_and_generate(question)
        pred_graph = r_graph.answer or ""
        score = judge.judge(question, gold_list, pred_graph)

        if score == args.score:
            matching_questions.append(question)

        # Progress bar
        print(f"\r{_progress_bar(idx + 1, total)}", end="", flush=True)

    print()  # newline after progress bar

    print(f"\nQuestions where judge returned score={args.score}: {len(matching_questions)}/{total}")
    print("-" * 60)
    for i, q in enumerate(matching_questions, 1):
        print(f"{i}. {q}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
