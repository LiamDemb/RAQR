"""Run two-stage LLM triple extraction (Discovery -> Validation) on a single input.

For dev/testing: runs the same Discovery and Validation prompts synchronously
on one piece of text and prints Stage 1 candidates and Stage 2 validated triples.

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
        description="Run two-stage LLM triple extraction (Discovery -> Validation) on a single input.",
    )
    parser.add_argument("text", nargs="?", help="Input text. Reads from stdin or --text-file if omitted.")
    parser.add_argument(
        "--text-file",
        metavar="PATH",
        help="Load input text from file instead of positional arg or stdin.",
    )
    parser.add_argument(
        "--title",
        default="N/A",
        help="Page title for context (default: N/A).",
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
        help="Show extra trace info.",
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
    alias_path = Path(args.output_dir) / "alias_map.json"
    if alias_path.is_file():
        with alias_path.open(encoding="utf-8") as f:
            alias_map = json.load(f)

    from raqr.data.llm_relations import call_llm_for_triples, _post_process_raw_triples
    from raqr.data.canonical_clean import normalize_text_for_extraction
    from raqr.prompts import get_triple_discovery_prompt, get_triple_validation_prompt

    text_norm = normalize_text_for_extraction(text)
    discovery_prompt = get_triple_discovery_prompt()
    validation_prompt = get_triple_validation_prompt()

    print("Stage 1 (Discovery)...")
    raw_candidates = call_llm_for_triples(
        discovery_prompt.format(title=args.title, text=text_norm)
    )
    print(f"  Candidates: {len(raw_candidates)}")
    if args.debug and raw_candidates:
        for t in raw_candidates:
            print(f"    {t.get('subj_surface')} --[{t.get('pred')}]--> {t.get('obj_surface')}")

    if not raw_candidates:
        print("Stage 2 (Validation): no candidates to validate.")
        print("Triples (final): 0")
        return 0

    print("Stage 2 (Validation)...")
    candidates_json = json.dumps(raw_candidates, ensure_ascii=False)
    raw_validated = call_llm_for_triples(
        validation_prompt.format(
            title=args.title,
            text=text_norm,
            candidates_from_stage_1=candidates_json,
        )
    )
    triples = _post_process_raw_triples(raw_validated, text, alias_map, None, debug=args.debug)

    print(f"\nTriples (validated, {len(triples)}):")
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
