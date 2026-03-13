from __future__ import annotations

import argparse
import json
from pathlib import Path


def _default_alias_map_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data" / "processed" / "alias_map.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count top-level items (keys) in alias_map.json."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=_default_alias_map_path(),
        help="Path to alias_map.json (default: data/processed/alias_map.json).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a labeled output instead of a bare integer.",
    )
    args = parser.parse_args()

    with args.path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        n = len(data)
        if args.verbose:
            print(f"alias_map keys: {n}")
        else:
            print(n)
        return 0

    if isinstance(data, list):
        n = len(data)
        if args.verbose:
            print(f"alias_map items: {n}")
        else:
            print(n)
        return 0

    raise TypeError(
        f"Expected alias_map to be a JSON object or array, got {type(data).__name__}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
