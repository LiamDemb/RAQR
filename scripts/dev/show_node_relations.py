"""Print all relations (incoming and outgoing) for a given node in the graph.

Usage:
    poetry run python scripts/graphrag/show_node_relations.py "barack obama"
    poetry run python scripts/graphrag/show_node_relations.py "E:united states" --output-dir data/processed
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show all graph relations connected to a given node.",
    )
    parser.add_argument(
        "node",
        help='Entity or node ID (e.g. "barack obama" or "E:barack obama").',
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed"),
        help="Directory containing graph.pkl.",
    )
    args = parser.parse_args()

    graph_path = f"{args.output_dir}/graph.pkl"
    if not os.path.isfile(graph_path):
        print(f"Graph not found: {graph_path}", file=sys.stderr)
        return 1

    import pandas as pd

    graph = pd.read_pickle(graph_path)
    node = args.node.strip()
    if not node.startswith("E:") and not node.startswith("C:"):
        node = f"E:{node}"

    if not graph.has_node(node):
        print(f"Node not found: {node}", file=sys.stderr)
        print("Tip: Use entity norm (e.g. 'barack obama') or full ID (e.g. 'E:barack obama').", file=sys.stderr)
        return 1

    print(f"Node: {node}")
    print("=" * 60)

    # Outgoing edges (this node -> others)
    out_edges = list(graph.out_edges(node, data=True))
    print(f"\nOutgoing ({len(out_edges)}):")
    if not out_edges:
        print("  (none)")
    else:
        for _, target, data in out_edges:
            kind = data.get("kind", "")
            label = data.get("label", "")
            if kind == "rel":
                print(f"  {node} --[{label}]--> {target}")
            else:
                print(f"  {node} --[{kind}]--> {target}")

    # Incoming edges (others -> this node)
    in_edges = list(graph.in_edges(node, data=True))
    print(f"\nIncoming ({len(in_edges)}):")
    if not in_edges:
        print("  (none)")
    else:
        for source, _, data in in_edges:
            kind = data.get("kind", "")
            label = data.get("label", "")
            if kind == "rel":
                print(f"  {source} --[{label}]--> {node}")
            else:
                print(f"  {source} --[{kind}]--> {node}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
