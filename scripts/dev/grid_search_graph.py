"""Grid search to tune ScoringConfig parameters for GraphStrategy."""

import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

from raqr.embedder import SentenceTransformersEmbedder
from raqr.entity_alias_resolver import EntityAliasResolver
from raqr.entity_index_store import EntityIndexStore
from raqr.generator import SimpleLLMGenerator
from raqr.graph_store import NetworkXGraphStore
from raqr.loaders import JsonCorpusLoader
from raqr.prompts import get_generator_prompt
from raqr.strategies.graph import GraphStrategy, _default_query_entity_extractor
from raqr.scoring_config import ScoringConfig

# Import from dev scripts common utils
from _common import normalize_gold_answers


def compute_f1(predicted: str, gold_answers: List[str]) -> float:
    """Compute token-level F1 score against gold answers."""
    if not predicted or not gold_answers:
        return 0.0

    pred_tokens = predicted.lower().split()
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = gold.lower().split()
        if not gold_tokens:
            continue

        common = sum(1 for t in pred_tokens if t in gold_tokens)
        if common == 0:
            continue

        precision = common / len(pred_tokens)
        recall = common / len(gold_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1

    return best_f1


def compute_exact_match(predicted: str, gold_answers: List[str]) -> int:
    """Check if predicted answer exactly contains one of the gold answers."""
    if not predicted or not gold_answers:
        return 0

    pred_lower = predicted.lower()
    for gold in gold_answers:
        if gold.lower() in pred_lower:
            return 1
    return 0


def build_base_strategy(output_dir: str, top_k: int, max_hops: int) -> GraphStrategy:
    """Build a GraphStrategy whose scoring_config can be mutated later."""
    corpus_path = f"{output_dir}/corpus.jsonl"
    graph_path = f"{output_dir}/graph.pkl"
    alias_map_path = f"{output_dir}/alias_map.json"

    if not os.path.exists(alias_map_path):
        raise FileNotFoundError(f"Missing {alias_map_path}. Rebuild Phase 1 artifacts.")

    alias_resolver = EntityAliasResolver.from_artifacts(output_dir=output_dir)

    entity_index_store = None
    if os.path.exists(f"{output_dir}/entity_index.faiss") and os.path.exists(
        f"{output_dir}/entity_index_meta.parquet"
    ):
        entity_index_store = EntityIndexStore(
            f"{output_dir}/entity_index.faiss",
            f"{output_dir}/entity_index_meta.parquet",
        )
    return GraphStrategy(
        graph_store=NetworkXGraphStore(graph_path=graph_path),
        corpus=JsonCorpusLoader(jsonl_path=corpus_path),
        generator=SimpleLLMGenerator(
            model_id=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_prompt=get_generator_prompt(),
        ),
        entity_extractor=_default_query_entity_extractor(alias_resolver),
        top_k=top_k,
        max_hops=max_hops,
        entity_index_store=entity_index_store,
        embedder=SentenceTransformersEmbedder(model_name="all-MiniLM-L6-v2"),
    )


def load_benchmark(path: str, limit: int = 0) -> List[Dict[str, Any]]:
    """Load benchmark queries and corresponding gold answers."""
    data = []
    if not os.path.exists(path):
        print(f"Benchmark file not found: {path}")
        return data

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit > 0 and idx >= limit:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            if "question" in item and "gold_answers" in item:
                item["gold_normalized"] = normalize_gold_answers(item["gold_answers"])
                data.append(item)
    return data


def format_config(config: ScoringConfig) -> str:
    return f"L_Pred={config.local_pred_weight:.2f}, B_Pred={config.bundle_pred_weight:.2f}, Len_Pen={config.length_penalty:.2f}"


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Grid search for GraphStrategy ScoringConfig"
    )
    parser.add_argument(
        "--benchmark",
        default="data/processed/benchmark.jsonl",
        help="Path to benchmark.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed"),
        help="Artifacts directory",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Max queries to run per config"
    )
    parser.add_argument("--top-k", type=int, default=10, help="Retrieval Top K")
    parser.add_argument("--max-hops", type=int, default=2, help="Graph max hops")
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Use LLMJudge instead of F1 (expensive)",
    )
    parser.add_argument(
        "--report-file",
        default="data/processed/grid_search_results.csv",
        help="Output file",
    )
    args = parser.parse_args()

    # Grid values to test
    LOCAL_PRED_WEIGHTS = [0.5, 0.55, 0.6, 0.65, 0.7]
    BUNDLE_PRED_WEIGHTS = [0.4, 0.45, 0.5, 0.55, 0.6]
    LENGTH_PENALTIES = [0.00, 0.02, 0.04, 0.06]

    # Create grid of configs
    configs = [
        ScoringConfig(local_pred_weight=l, bundle_pred_weight=b, length_penalty=lp)
        for l, b, lp in itertools.product(
            LOCAL_PRED_WEIGHTS, BUNDLE_PRED_WEIGHTS, LENGTH_PENALTIES
        )
    ]

    print(f"Starting grid search over {len(configs)} configurations.")

    benchmark_data = load_benchmark(args.benchmark, args.limit)
    if not benchmark_data:
        print("No benchmark data loaded. Exiting.")
        return 1

    print(f"Loaded {len(benchmark_data)} queries from benchmark.")

    judge = None
    if args.use_llm_judge:
        from raqr.llm_judge import LLMJudge

        judge = LLMJudge()
        print("Using LLM Judge for evaluation.")
    else:
        print("Using F1/EM macro scoring for evaluation.")

    # Build once, mutate config later
    print("Building base GraphStrategy (Loading embedding model, graph, etc.)...")
    strategy = build_base_strategy(args.output_dir, args.top_k, args.max_hops)

    results = []

    start_time = time.time()
    for idx, config in enumerate(configs, 1):
        print(f"\n--- Running Config {idx}/{len(configs)}: {format_config(config)} ---")
        strategy.scoring_config = config

        total_f1 = 0.0
        total_em = 0.0
        total_judge = 0.0

        for q_idx, item in enumerate(benchmark_data, 1):
            q_text = item["question"]
            gold = item["gold_normalized"]

            # Retrieve & Generate
            res = strategy.retrieve_and_generate(q_text)
            pred = res.answer or ""

            if judge:
                score = judge.judge(q_text, gold, pred)
                total_judge += score
            else:
                f1 = compute_f1(pred, gold)
                em = compute_exact_match(pred, gold)
                total_f1 += f1
                total_em += em

            # Print brief progress if not using expensive judge
            if q_idx % 10 == 0:
                print(f"  Processed {q_idx}/{len(benchmark_data)} queries...")

        n = len(benchmark_data)
        res_dict = {
            "local_pred_weight": config.local_pred_weight,
            "bundle_pred_weight": config.bundle_pred_weight,
            "length_penalty": config.length_penalty,
        }

        if judge:
            mean_score = total_judge / n
            res_dict["judge_score"] = mean_score
            print(f"  Result: Mean Judge Score = {mean_score:.3f}")
        else:
            mean_f1 = total_f1 / n
            mean_em = total_em / n
            res_dict["f1"] = mean_f1
            res_dict["em"] = mean_em
            print(f"  Result: F1 = {mean_f1:.3f} | EM = {mean_em:.3f}")

        results.append(res_dict)

    print(f"\nGrid Search Complete in {time.time() - start_time:.2f}s")

    # Sort and Report Leaderboard
    if judge:
        results.sort(key=lambda x: x["judge_score"], reverse=True)
    else:
        results.sort(key=lambda x: x["f1"], reverse=True)

    print("\nLEADERBOARD:")
    for i, r in enumerate(results, 1):
        c_str = f"L_Pred={r['local_pred_weight']:.2f}, B_Pred={r['bundle_pred_weight']:.2f}, Len_Pen={r['length_penalty']:.2f}"
        if judge:
            print(f"{i}. {c_str} ==> Judge Score: {r['judge_score']:.3f}")
        else:
            print(f"{i}. {c_str} ==> F1: {r['f1']:.3f} | EM: {r['em']:.3f}")

    # Save to CSV
    import csv

    with open(args.report_file, "w", newline="") as f:
        keys = results[0].keys()
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nFull results saved to {args.report_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
