"""Compute graph entity-match-rate diagnostic on benchmark questions."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from dotenv import load_dotenv

from raqr.entity_alias_resolver import EntityAliasResolver
from raqr.graph_store import NetworkXGraphStore
from raqr.strategies.graph import SpacyQueryEntityExtractor


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _question_from_item(item: dict) -> str:
    return str(item.get("question") or item.get("query") or "").strip()


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Graph entity match-rate diagnostic.")
    parser.add_argument(
        "--benchmark",
        default=os.getenv("BENCHMARK_PATH", "data/processed/benchmark.jsonl"),
        help="Path to benchmark questions JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed_rebel"),
        help="Directory containing graph.pkl and entity_lexicon.parquet.",
    )
    parser.add_argument(
        "--top-unmatched",
        type=int,
        default=20,
        help="How many unmatched entity keys to print.",
    )
    args = parser.parse_args()

    graph_path = f"{args.output_dir}/graph.pkl"
    lexicon_path = f"{args.output_dir}/entity_lexicon.parquet"
    graph = NetworkXGraphStore(graph_path=graph_path).load()
    alias_resolver = EntityAliasResolver.from_lexicon(lexicon_path=lexicon_path)
    extractor = SpacyQueryEntityExtractor(alias_resolver=alias_resolver)

    total_queries = 0
    queries_with_match = 0
    total_entities = 0
    unmatched_entity_counts: Counter[str] = Counter()

    for item in _iter_jsonl(args.benchmark):
        question = _question_from_item(item)
        if not question:
            continue
        total_queries += 1
        entity_norms = extractor.extract(question)
        total_entities += len(entity_norms)
        matched = False
        for norm in entity_norms:
            node_id = f"E:{norm}"
            if graph.has_node(node_id):
                matched = True
            else:
                unmatched_entity_counts[norm] += 1
        if matched:
            queries_with_match += 1

    if total_queries == 0:
        print("No benchmark queries found.")
        return 1

    avg_entities = total_entities / total_queries
    match_rate = queries_with_match / total_queries

    print("Graph Match-Rate Diagnostic")
    print(f"- benchmark: {args.benchmark}")
    print(f"- output_dir: {args.output_dir}")
    print(f"- queries_total: {total_queries}")
    print(f"- queries_with_entity_match: {queries_with_match}")
    print(f"- entity_match_rate: {match_rate:.2%}")
    print(f"- avg_extracted_entities_per_query: {avg_entities:.2f}")

    top_n = max(0, args.top_unmatched)
    if top_n > 0:
        print(f"\nTop {min(top_n, len(unmatched_entity_counts))} unmatched normalized entities:")
        for norm, count in unmatched_entity_counts.most_common(top_n):
            print(f"- {norm}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

