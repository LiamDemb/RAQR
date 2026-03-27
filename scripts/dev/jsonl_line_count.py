from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "2wikimultihop_2500.jsonl"
print(sum(1 for _ in PATH.open("r", encoding="utf-8")))
