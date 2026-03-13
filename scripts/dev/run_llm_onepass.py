"""Run one-pass LLM extraction (entities + triples) on chunk text.

Usage:
    poetry run python scripts/dev/run_llm_onepass.py --text-file path/to/chunk.txt
    poetry run python scripts/dev/run_llm_onepass.py --text-file path/to/chunk.txt --output-dir data/processed
    poetry run python scripts/dev/run_llm_onepass.py "Imhotep was high priest of Ra." --json

Requires OPENAI_API_KEY. Model via LLM_ONEPASS_MODEL (default: gpt-4o-mini).
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
        description="Run one-pass LLM extraction (entities + triples) on chunk text.",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="Input text. Uses --text-file or stdin if omitted.",
    )
    parser.add_argument(
        "--text-file",
        metavar="PATH",
        help="Load input text from .txt file.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed"),
        help="Optional: load wiki_titles.jsonl for seed anchoring.",
    )
    parser.add_argument(
        "--title",
        default="N/A",
        help="Page title for prompt context (e.g. Wikipedia article title).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output entities and relations as JSON.",
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

    seed_titles: list[str] = []
    wiki_path = Path(args.output_dir) / "wiki_titles.jsonl"
    alias_path = Path(args.output_dir) / "alias_map.json"
    if wiki_path.is_file():
        try:
            from raqr.data.wiki_title_matcher import build_wiki_title_matcher, load_wiki_titles
            titles = load_wiki_titles(wiki_path)
            alias_map = {}
            if alias_path.is_file():
                alias_map = json.loads(alias_path.read_text(encoding="utf-8"))
            matcher = build_wiki_title_matcher(titles, alias_map=alias_map)
            seed_titles = matcher.find_titles_in_text(text, max_results=20)
        except Exception as e:
            print(f"Warning: could not load wiki titles: {e}", file=sys.stderr)

    from raqr.data.llm_onepass import LLMOnepassExtractor

    ext = LLMOnepassExtractor()
    entities, relations = ext.extract(
        text,
        title=args.title,
        seed_titles_in_chunk=seed_titles,
    )

    if args.json:
        print(json.dumps({"entities": entities, "relations": relations}, indent=2, ensure_ascii=False))
        return 0

    print(f"Entities ({len(entities)}):")
    for e in entities:
        print(f"  {e.get('surface')} ({e.get('type')}) -> norm={e.get('norm')}")
    if not entities:
        print("  (none)")

    print(f"\nRelations ({len(relations)}):")
    for r in relations:
        print(f"  ({r['subj_norm']}) --[{r['pred']}]--> ({r['obj_norm']})")
        if r.get("match_text"):
            mt = r["match_text"]
            print(f"      evidence: \"{mt[:80]}{'...' if len(mt) > 80 else ''}\"")
    if not relations:
        print("  (none)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
