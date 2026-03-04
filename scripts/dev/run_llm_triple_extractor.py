"""Run LLM-based triple extraction on chunk text and print triples + evidence.

Usage:
    poetry run python scripts/dev/run_llm_triple_extractor.py "Eva Busch was a German cabaret artist."
    echo "text" | poetry run python scripts/dev/run_llm_triple_extractor.py
    poetry run python scripts/dev/run_llm_triple_extractor.py --text-file path/to/file.txt
    poetry run python scripts/dev/run_llm_triple_extractor.py "text" --output-dir data/processed --json --debug

Requires OPENAI_API_KEY. Model via LLM_TRIPLE_MODEL (default: gpt-4o-mini).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run LLM-based triple extraction on chunk text and print triples + evidence.",
    )
    parser.add_argument("text", nargs="?", help="Input text. Reads from stdin or --text-file if omitted.")
    parser.add_argument(
        "--text-file",
        metavar="PATH",
        help="Load input text from file instead of positional arg or stdin.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed"),
        help="Optional: load alias_map.json for entity normalization.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output triples as JSON (one object per line) instead of compact.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show extra trace info (raw tool args, normalization warnings).",
    )
    args = parser.parse_args()

    text = None
    if args.text_file:
        path = Path(args.text_file)
        if not path.is_file():
            print(f"File not found: {path}", file=sys.stderr)
            return 1
        text = path.read_text(encoding="utf-8").strip()
    elif args.text is not None:
        text = args.text
    else:
        text = sys.stdin.read().strip()
    if not text:
        print("No text provided.", file=sys.stderr)
        return 1

    alias_map: dict = {}
    alias_path = f"{args.output_dir}/alias_map.json"
    if os.path.isfile(alias_path):
        with open(alias_path, encoding="utf-8") as f:
            alias_map = json.load(f)

    from raqr.data.llm_relations import LLMTripleExtractor

    ext = LLMTripleExtractor()
    triples = ext.extract(text, alias_map=alias_map, debug=args.debug)

    print(f"Triples ({len(triples)}):")
    if not triples:
        print("  (none)")
    elif args.json:
        for t in triples:
            print(json.dumps(t, ensure_ascii=False))
    else:
        for t in triples:
            print(f"  ({t['subj_norm']}) --[{t['pred']}]--> ({t['obj_norm']})")
            print(f"      rule={t['rule_id']} conf={t['confidence']}")
            if t.get("match_text"):
                mt = t["match_text"]
                print(f"      match=\"{mt[:80]}{'...' if len(mt) > 80 else ''}\"")

    return 0


if __name__ == "__main__":
    sys.exit(main())
