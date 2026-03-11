import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file")
args = parser.parse_args()

total = 0

with open(args.file, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        titles = record.get("supporting_facts", {}).get("title", [])
        for title in titles:
            print(title)
            total += 1

print(f"\nTotal: {total}")
