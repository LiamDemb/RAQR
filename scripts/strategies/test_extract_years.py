"""Quick CLI test for enrich_years.extract_years."""

from __future__ import annotations

import argparse

from raqr.data.enrich_years import extract_years


def main() -> int:
    parser = argparse.ArgumentParser(description="Test year extraction on one query.")
    parser.add_argument("query", help="Query text to parse for years")
    args = parser.parse_args()

    years = extract_years(args.query)
    print("Extracted years:", years)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
