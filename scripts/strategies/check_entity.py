"""Check if entities extracted from a query string exist in the graph."""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv

from raqr.entity_alias_resolver import EntityAliasResolver
from raqr.graph_store import NetworkXGraphStore
from raqr.strategies.graph import _default_query_entity_extractor


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Check if query entities exist in the graph.")
    parser.add_argument("query", help="Query string to extract and check entities from.")
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed"),
        help="Directory containing graph.pkl and entity_lexicon.parquet.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    alias_map_path = f"{output_dir}/alias_map.json"
    if not os.path.exists(alias_map_path):
        raise FileNotFoundError(
            f"Required artifact missing: {alias_map_path}. Rebuild corpus with Phase 1 pipeline."
        )

    graph = NetworkXGraphStore(graph_path=f"{output_dir}/graph.pkl").load()
    alias_resolver = EntityAliasResolver.from_artifacts(output_dir=output_dir)
    extractor = _default_query_entity_extractor(alias_resolver)

    entity_norms = extractor.extract(args.query)
    if not entity_norms:
        print("No entities extracted.")
        return 0

    print(f"Query: {args.query!r}")
    print(f"Extracted entities: {entity_norms}")
    print()
    for norm in entity_norms:
        node_id = f"E:{norm}"
        found = graph.has_node(node_id)
        status = "✓" if found else "✗"
        print(f"  {status} {norm} -> {'in graph' if found else 'not in graph'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
