import json

path = "/Users/liamdemb/Downloads/batch_69c2287613bc8190805560d136125cc6_output.jsonl"  # <- change this

tot_prompt = 0
tot_completion = 0
tot_total = 0
ok = 0
skipped = 0
failed = 0

with open(path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue

        # If the batch line has an error field populated, count as failed
        if obj.get("error") is not None:
            failed += 1
            continue

        resp = obj.get("response") or {}
        if resp.get("status_code") != 200:
            failed += 1
            continue

        usage = ((resp.get("body") or {}).get("usage")) or {}
        if not usage:
            skipped += 1
            continue

        p = usage.get("prompt_tokens", 0)
        c = usage.get("completion_tokens", 0)
        t = usage.get("total_tokens", p + c)

        tot_prompt += p
        tot_completion += c
        tot_total += t
        ok += 1

print(f"OK lines: {ok}")
print(f"Failed lines: {failed}")
print(f"Skipped (no usage / parse issues): {skipped}")
print(f"Prompt tokens: {tot_prompt:,}")
print(f"Completion tokens: {tot_completion:,}")
print(f"Total tokens: {tot_total:,}")

if ok:
    print(f"Avg total per OK request: {tot_total/ok:,.2f}")
