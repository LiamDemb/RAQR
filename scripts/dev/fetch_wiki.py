"""Fetch a single Wikipedia page and run the full cleaning pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from raqr.data.canonical_clean import clean_html_to_structured_doc
from raqr.data.wikipedia_client import WikipediaClient


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch a Wikipedia page, run cleaning pipeline, write output to .txt"
    )
    parser.add_argument(
        "title",
        help="Wikipedia page title (e.g. 'Barack Obama', 'Paris')",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output .txt file path (default: <title>.txt in current dir)",
        default=None,
    )
    args = parser.parse_args()

    title = args.title.strip()
    if not title:
        print("Error: title cannot be empty", file=sys.stderr)
        return 1

    wiki = WikipediaClient()
    page = wiki.fetch_html(title)

    if not page.html:
        print(f"Error: no HTML returned for '{title}'", file=sys.stderr)
        return 1

    structured = clean_html_to_structured_doc(
        html=page.html,
        doc_id=page.page_id or title,
        title=page.title,
        url=page.url,
        anchors={"outgoing_titles": page.outgoing_titles, "incoming_stub": []},
        source="dev",
        dataset_origin="dev",
        page_id=page.page_id,
        revision_id=page.revision_id,
    )

    # Join blocks exactly as in corpus (chunking._join_blocks)
    text_out = "\n\n".join(
        block.text for block in structured.blocks if block.text
    ).rstrip()

    out_path = args.output
    if not out_path:
        safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)[:80]
        out_path = Path.cwd() / f"{safe_title.strip()}.txt"
    else:
        out_path = Path(out_path)

    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text_out, encoding="utf-8")

    print(f"Wrote {len(structured.blocks)} blocks to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
