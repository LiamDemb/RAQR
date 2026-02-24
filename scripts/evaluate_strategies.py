"""Evaluate Dense and Graph strategies on the benchmark.

Runs each question through both strategies, compares predicted answers to gold,
and reports F1 (primary), EM, and per-source breakdown. Uses token-level F1
as the main metric for head-to-head comparison.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _normalize_gold_answers(raw: list) -> list[str]:
    """Parse gold_answers into a flat list of normalized strings."""
    answers = []
    for item in raw:
        s = str(item).strip()
        if not s:
            continue
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                answers.extend(str(x).strip() for x in parsed if str(x).strip())
            else:
                answers.append(s)
        except (ValueError, SyntaxError):
            answers.append(s)
    return answers


def _normalize_for_compare(text: str) -> str:
    """Lowercase, collapse whitespace, strip."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.lower().strip())


def _tokenize(text: str) -> set[str]:
    """Simple tokenization for F1: split on non-alphanumeric."""
    norm = _normalize_for_compare(text)
    return set(re.findall(r"\b\w+\b", norm)) if norm else set()


def exact_match(pred: str, gold_list: list[str]) -> bool:
    """True if normalized pred matches any gold, or pred contains gold (lenient QA style)."""
    pred_norm = _normalize_for_compare(pred)
    if not pred_norm:
        return False
    for gold in gold_list:
        gold_norm = _normalize_for_compare(gold)
        if not gold_norm:
            continue
        if pred_norm == gold_norm:
            return True
        # Lenient: gold answer contained in prediction (e.g. "Molly" in "Molly Davis")
        if gold_norm in pred_norm:
            return True
    return False


def token_f1(pred: str, gold_list: list[str]) -> float:
    """Token-level F1: max over gold answers (SQuAD-style)."""
    pred_tokens = _tokenize(pred)
    if not pred_tokens:
        return 0.0
    best_f1 = 0.0
    for gold in gold_list:
        gold_tokens = _tokenize(gold)
        if not gold_tokens:
            continue
        common = pred_tokens & gold_tokens
        precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(common) / len(gold_tokens) if gold_tokens else 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            best_f1 = max(best_f1, f1)
    return best_f1


def _build_dense_strategy(output_dir: str):
    from raqr.embedder import SentenceTransformersEmbedder
    from raqr.generator import SimpleLLMGenerator
    from raqr.index_store import FaissIndexStore
    from raqr.loaders import JsonCorpusLoader, VectorMetaMapper
    from raqr.strategies.dense import DenseStrategy

    corpus_path = f"{output_dir}/corpus.jsonl"
    index_path = f"{output_dir}/vector_index.faiss"
    meta_path = f"{output_dir}/vector_meta.parquet"
    return DenseStrategy(
        index_store=FaissIndexStore(index_path=index_path),
        meta=VectorMetaMapper(parquet_path=meta_path),
        embedder=SentenceTransformersEmbedder(model_name="all-MiniLM-L6-v2"),
        generator=SimpleLLMGenerator(
            model_id=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_prompt=(
                "Answer the question based only on the provided context. "
                "Give a concise, direct answer. "
                "If the context does not contain the answer, say so."
            ),
        ),
        corpus=JsonCorpusLoader(jsonl_path=corpus_path),
        top_k=int(os.getenv("DENSE_TOP_K", "10")),
    )


def _build_graph_strategy(output_dir: str):
    from raqr.entity_alias_resolver import EntityAliasResolver
    from raqr.generator import SimpleLLMGenerator
    from raqr.graph_store import NetworkXGraphStore
    from raqr.loaders import JsonCorpusLoader
    from raqr.strategies.graph import GraphStrategy, SpacyQueryEntityExtractor

    corpus_path = f"{output_dir}/corpus.jsonl"
    graph_path = f"{output_dir}/graph.pkl"
    lexicon_path = f"{output_dir}/entity_lexicon.parquet"
    alias_map_path = f"{output_dir}/alias_map.json"

    if not os.path.exists(alias_map_path):
        raise FileNotFoundError(
            f"Required artifact missing: {alias_map_path}. Rebuild corpus with Phase 1 pipeline."
        )

    alias_resolver = EntityAliasResolver.from_artifacts(output_dir=output_dir)
    entity_df_by_norm = EntityAliasResolver.load_df_map_from_lexicon(lexicon_path=lexicon_path)
    return GraphStrategy(
        graph_store=NetworkXGraphStore(graph_path=graph_path),
        corpus=JsonCorpusLoader(jsonl_path=corpus_path),
        generator=SimpleLLMGenerator(
            model_id=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_prompt=(
                "Answer the question based only on the provided context. "
                "Give a concise, direct answer. "
                "If the context does not contain the answer, say so."
            ),
        ),
        entity_extractor=SpacyQueryEntityExtractor(alias_resolver=alias_resolver),
        top_k=int(os.getenv("GRAPH_TOP_K", "10")),
        max_hops=int(os.getenv("GRAPH_MAX_HOPS", "1")),
        entity_df_by_norm=entity_df_by_norm,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Dense and Graph strategies on benchmark.")
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
        help="Print each question and result.",
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

    print("Building strategies...")
    dense = _build_dense_strategy(args.output_dir)
    graph = _build_graph_strategy(args.output_dir)
    print(f"Evaluating on {len(samples)} questions.\n")

    results_dense: dict[str, dict] = defaultdict(lambda: {"em": 0, "f1": 0.0, "correct": 0, "total": 0})
    results_graph: dict[str, dict] = defaultdict(lambda: {"em": 0, "f1": 0.0, "correct": 0, "total": 0})
    time_dense_sec = 0.0
    time_graph_sec = 0.0
    t_start = time.perf_counter()

    for idx, sample in enumerate(samples):
        question = sample.get("question", "")
        gold_raw = sample.get("gold_answers", [])
        source = sample.get("dataset_source", "unknown")
        gold_list = _normalize_gold_answers(gold_raw)

        if not question or not gold_list:
            continue

        # Dense
        t0 = time.perf_counter()
        r_dense = dense.retrieve_and_generate(question)
        time_dense_sec += time.perf_counter() - t0
        pred_dense = r_dense.answer or ""
        em_dense = exact_match(pred_dense, gold_list)
        f1_dense = token_f1(pred_dense, gold_list)
        results_dense[source]["em"] += 1 if em_dense else 0
        results_dense[source]["f1"] += f1_dense
        results_dense[source]["correct"] += 1 if em_dense else 0
        results_dense[source]["total"] += 1
        results_dense["_all"]["em"] += 1 if em_dense else 0
        results_dense["_all"]["f1"] += f1_dense
        results_dense["_all"]["correct"] += 1 if em_dense else 0
        results_dense["_all"]["total"] += 1

        # Graph
        t0 = time.perf_counter()
        r_graph = graph.retrieve_and_generate(question)
        time_graph_sec += time.perf_counter() - t0
        pred_graph = r_graph.answer or ""
        em_graph = exact_match(pred_graph, gold_list)
        f1_graph = token_f1(pred_graph, gold_list)
        results_graph[source]["em"] += 1 if em_graph else 0
        results_graph[source]["f1"] += f1_graph
        results_graph[source]["correct"] += 1 if em_graph else 0
        results_graph[source]["total"] += 1
        results_graph["_all"]["em"] += 1 if em_graph else 0
        results_graph["_all"]["f1"] += f1_graph
        results_graph["_all"]["correct"] += 1 if em_graph else 0
        results_graph["_all"]["total"] += 1

        if args.verbose:
            def _trunc(s: str, n: int = 60) -> str:
                return s[:n] + "..." if len(s) > n else s
            print(f"[{idx + 1}] {_trunc(question)}")
            print(f"  Gold: {_trunc(gold_list[0])}")
            print(f"  Dense: EM={'✓' if em_dense else '✗'} F1={f1_dense:.3f} {_trunc(pred_dense)}")
            print(f"  Graph: EM={'✓' if em_graph else '✗'} F1={f1_graph:.3f} {_trunc(pred_graph)}")
            print()

    # Print summary (F1 primary, EM secondary)
    def _print_strategy(name: str, res: dict):
        print(f"\n{'='*60}")
        print(f"  {name}")
        print("=" * 60)
        for key in sorted(res.keys()):
            if key == "_all":
                continue
            r = res[key]
            n = r["total"]
            if n == 0:
                continue
            f1_avg = r["f1"] / n
            em_pct = 100 * r["em"] / n
            print(f"  {key}: F1={f1_avg:.3f}, EM={r['correct']}/{n} ({em_pct:.1f}%)")
        r_all = res["_all"]
        n_all = r_all["total"]
        f1_avg = r_all["f1"] / n_all
        em_pct = 100 * r_all["em"] / n_all
        print(f"  ---")
        print(f"  OVERALL: F1={f1_avg:.3f}, EM={r_all['correct']}/{n_all} ({em_pct:.1f}%)")

    _print_strategy("Dense", results_dense)
    _print_strategy("Graph", results_graph)

    t_total_sec = time.perf_counter() - t_start

    # Head-to-head (F1 primary)
    n = results_dense["_all"]["total"]
    d_f1 = results_dense["_all"]["f1"] / n
    g_f1 = results_graph["_all"]["f1"] / n
    d_em = results_dense["_all"]["em"]
    g_em = results_graph["_all"]["em"]
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Dense F1: {d_f1:.3f}  |  Graph F1: {g_f1:.3f}")
    print(f"  Dense EM: {d_em}/{n}  |  Graph EM: {g_em}/{n}")
    print(f"  Dense time: {time_dense_sec:.1f}s  |  Graph time: {time_graph_sec:.1f}s")
    print(f"  Total elapsed: {t_total_sec:.1f}s")
    if d_f1 > g_f1:
        print("  → Dense has higher F1")
    elif g_f1 > d_f1:
        print("  → Graph has higher F1")
    else:
        print("  → Tie on F1")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
