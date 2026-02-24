"""Mock-oracle evaluation: one winner per question (highest judge score; Dense wins ties).

Evaluates Dense and Graph on the benchmark using LLM-as-judge (0/1/2). For each
question, the strategy with the higher judge score wins; ties go to Dense.
Reports score breakdown (0/1/2), correct (2) counts, total wins, wins by source,
and score breakdown per strategy per source.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent))
from _common import (
    build_dense_strategy,
    build_graph_strategy,
    normalize_gold_answers,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Mock-oracle eval: one winner per question (highest judge score; Dense wins ties).",
    )
    parser.add_argument(
        "--benchmark",
        default=os.getenv("BENCHMARK_PATH", "data/processed/benchmark.jsonl"),
        help="Path to benchmark JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed"),
        help="Directory containing corpus, indexes, graph, alias_map.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions (for quick dev runs).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each question and winner.",
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

    print("Building strategies...")
    dense = build_dense_strategy(args.output_dir)
    graph = build_graph_strategy(args.output_dir)
    judge = LLMJudge()
    print("Mock Oracle Evaluation (Judge 0/1/2, Dense wins ties)")
    print("=" * 60)
    print(f"Total: {len(samples)} questions\n")

    # Per-strategy: score counts (0,1,2), correct (2), wins
    dense_scores: dict[int, int] = {0: 0, 1: 0, 2: 0}
    graph_scores: dict[int, int] = {0: 0, 1: 0, 2: 0}
    dense_wins = 0
    graph_wins = 0
    # Per source: total, dense_wins, graph_wins, dense_scores, graph_scores
    def _source_entry() -> dict:
        return {
            "total": 0,
            "dense_wins": 0,
            "graph_wins": 0,
            "dense_scores": {0: 0, 1: 0, 2: 0},
            "graph_scores": {0: 0, 1: 0, 2: 0},
        }

    by_source: dict[str, dict] = defaultdict(_source_entry)

    for idx, sample in enumerate(samples):
        question = sample.get("question", "")
        gold_raw = sample.get("gold_answers", [])
        source = sample.get("dataset_source", "unknown")
        gold_list = normalize_gold_answers(gold_raw)

        if not question or not gold_list:
            continue

        r_dense = dense.retrieve_and_generate(question)
        r_graph = graph.retrieve_and_generate(question)
        pred_dense = r_dense.answer or ""
        pred_graph = r_graph.answer or ""

        score_dense = judge.judge(question, gold_list, pred_dense)
        score_graph = judge.judge(question, gold_list, pred_graph)

        dense_scores[score_dense] += 1
        graph_scores[score_graph] += 1
        by_source[source]["total"] += 1
        by_source[source]["dense_scores"][score_dense] += 1
        by_source[source]["graph_scores"][score_graph] += 1

        # Winner: higher score wins; tie -> Dense
        if score_dense >= score_graph:
            dense_wins += 1
            by_source[source]["dense_wins"] += 1
            winner = "Dense"
        else:
            graph_wins += 1
            by_source[source]["graph_wins"] += 1
            winner = "Graph"

        if args.verbose:
            def _trunc(s: str, n: int = 50) -> str:
                return s[:n] + "..." if len(s) > n else s
            print(f"[{idx + 1}] {_trunc(question)} | D={score_dense} G={score_graph} → {winner}")

    n = sum(by_source[s]["total"] for s in by_source)

    # Score breakdown
    print("Score breakdown (0=incorrect, 1=partial, 2=correct):")
    d_correct = dense_scores[2]
    g_correct = graph_scores[2]
    d_pct = 100 * d_correct / n if n else 0
    g_pct = 100 * g_correct / n if n else 0
    print(f"  Dense:  0={dense_scores[0]:3d}  1={dense_scores[1]:3d}  2={dense_scores[2]:3d}  → Correct: {d_correct}/{n} ({d_pct:.1f}%)")
    print(f"  Graph:  0={graph_scores[0]:3d}  1={graph_scores[1]:3d}  2={graph_scores[2]:3d}  → Correct: {g_correct}/{n} ({g_pct:.1f}%)")
    print()

    # Wins
    print("Wins (highest score; tie → Dense):")
    print(f"  Dense: {dense_wins}   Graph: {graph_wins}")
    print()

    # Score breakdown by source (0, 1, 2 per strategy per source)
    print("Score breakdown by source (0=incorrect, 1=partial, 2=correct):")
    for key in sorted(by_source.keys()):
        r = by_source[key]
        t = r["total"]
        ds = r["dense_scores"]
        gs = r["graph_scores"]
        d_pct = 100 * ds[2] / t if t else 0
        g_pct = 100 * gs[2] / t if t else 0
        print(f"  {key}:")
        print(f"    Dense:  0={ds[0]:3d}  1={ds[1]:3d}  2={ds[2]:3d}  → Correct: {ds[2]}/{t} ({d_pct:.1f}%)")
        print(f"    Graph:  0={gs[0]:3d}  1={gs[1]:3d}  2={gs[2]:3d}  → Correct: {gs[2]}/{t} ({g_pct:.1f}%)")
    print()

    # Wins by source
    print("Wins by source:")
    for key in sorted(by_source.keys()):
        r = by_source[key]
        t = r["total"]
        dw = r["dense_wins"]
        gw = r["graph_wins"]
        print(f"  {key:20s}  Dense {dw}/{t}   Graph {gw}/{t}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
