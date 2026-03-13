"""Run debug_graph reasoning traces for a batch of queries from a file, including gold answers and node relations."""

# poetry run python scripts/strategies/batch_debug_graph.py --output temp/debug-output.txt --input temp/queries.txt --benchmark data/processed/benchmark.jsonl

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path


def normalize_gold_answers(raw: list) -> list[str]:
    """Parse gold_answers into a flat list of normalized strings. (Copied from _common.py)"""
    answers = []
    for item in raw:
        s = str(item).strip()
        if not s:
            continue
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                answers.extend(str(x).strip() for x in parsed if str(x).strip())
            else:
                answers.append(s)
        except (ValueError, SyntaxError):
            answers.append(s)
    return answers


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run debug_graph logic in batch mode with gold answers."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to text file containing one question per line.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the file where concatenated results should be saved.",
    )
    parser.add_argument(
        "--benchmark",
        default="data/processed/benchmark.jsonl",
        help="Path to benchmark.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "data/processed"),
        help="Output directory for artifacts.",
    )

    # Capture unknown args to pass them through to debug_graph.py
    args, pass_through_args = parser.parse_known_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    benchmark_path = Path(args.benchmark)

    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found.")
        return 1

    # Load benchmark data
    benchmark_data = {}
    if benchmark_path.exists():
        print(f"Loading benchmark from {benchmark_path}...")
        with open(benchmark_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                q = item.get("question")
                if q:
                    benchmark_data[q.strip().lower()] = item.get("gold_answers", [])
    else:
        print(
            f"Warning: Benchmark file '{args.benchmark}' not found. Gold answers will be missing."
        )

    # Read queries
    with open(input_path, "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    if not queries:
        print("No queries found in input file.")
        return 0

    print(f"Batch processing {len(queries)} queries...")
    separator = "\n" + "=" * 80 + "\n\n"

    # Paths to scripts
    script_dir = Path(__file__).parent
    debug_script = script_dir / "debug_graph.py"
    show_relations_script = script_dir.parent / "dev" / "show_node_relations.py"

    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, query in enumerate(queries):
            print(f"[{i+1}/{len(queries)}] Processing: {query}")

            # 1. Write Header
            out_f.write(f"BATCH QUERY {i+1}: {query}\n")
            out_f.write("-" * 40 + "\n")

            # 2. Add Gold Answers
            gold_answers_raw = benchmark_data.get(query.strip().lower(), [])
            normalized_answers = normalize_gold_answers(gold_answers_raw)
            out_f.write(f"GOLD STANDARD ANSWERS: {gold_answers_raw}\n")
            out_f.write(f"NORMALIZED ANSWERS: {normalized_answers}\n\n")

            # 3. Add Node Relations for Gold Answers
            if normalized_answers:
                out_f.write("GOLD ANSWER NODE RELATIONS:\n")
                for ans in normalized_answers:
                    # Run show_node_relations.py E:<ans>
                    rel_cmd = [
                        sys.executable,
                        str(show_relations_script),
                        f"E:{ans}",
                        "--output-dir",
                        args.output_dir,
                    ]
                    rel_res = subprocess.run(
                        rel_cmd, capture_output=True, text=True, check=False
                    )
                    if rel_res.returncode == 0:
                        out_f.write(rel_res.stdout)
                    else:
                        out_f.write(f"(No graph relations found for E:{ans})\n")
                out_f.write("\n")

            # 4. Run Debug Graph
            cmd = [sys.executable, str(debug_script), query] + pass_through_args
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, check=False
                )
                out_f.write("DEBUG GRAPH TRACE:\n")
                out_f.write(result.stdout)

                if result.stderr:
                    out_f.write("\nSTDERR / LOGS:\n")
                    out_f.write(result.stderr)

            except Exception as e:
                out_f.write(f"\nFAILED TO PROCESS DEBUG GRAPH: Error: {str(e)}\n")

            # Append separator if not the last item
            if i < len(queries) - 1:
                out_f.write(separator)

            out_f.flush()

    print(f"\nSuccessfully processed {len(queries)} queries.")
    print(f"Results saved to: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
