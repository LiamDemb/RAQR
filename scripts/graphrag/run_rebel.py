"""Run REBEL relation extraction on a chunk of text and print triples.

Usage:
    poetry run python scripts/graphrag/run_rebel.py "Barack Obama was born in Hawaii."
    poetry run python scripts/graphrag/run_rebel.py "text" --output-dir data/processed  # uses alias_map
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run REBEL on text and print extracted (subj, pred, obj) triples.",
    )
    parser.add_argument("text", nargs="?", help="Input text. Reads from stdin if omitted.")
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed"),
        help="Optional: load alias_map.json for entity normalization.",
    )
    args = parser.parse_args()

    text = args.text
    if text is None:
        text = sys.stdin.read().strip()
    if not text:
        print("No text provided.", file=sys.stderr)
        return 1

    alias_map = {}
    alias_path = f"{args.output_dir}/alias_map.json"
    if os.path.isfile(alias_path):
        with open(alias_path, encoding="utf-8") as f:
            alias_map = json.load(f)

    from raqr.data.enrich_relations import extract_relations_rebel, load_rebel

    print("Loading REBEL...", file=sys.stderr)
    tokenizer, model, device = load_rebel()
    results = extract_relations_rebel(
        [text],
        tokenizer,
        model,
        device,
        alias_map=alias_map,
        batch_size=1,
    )
    triples = results[0]

    print(f"Triples ({len(triples)}):")
    for t in triples:
        print(f"  ({t['subj_norm']}) --[{t['pred']}]--> ({t['obj_norm']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
