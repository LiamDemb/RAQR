from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI

from raqr.generation.batch import BatchRecorderGenerator, build_batch_line, parse_generation_output
from raqr.prompts import get_generator_prompt
from raqr.strategies.factory import build_dense_strategy, build_graph_strategy

logger = logging.getLogger(__name__)

BATCH_LIMIT_REQUESTS = 50_000
BATCH_LIMIT_BYTES = 200 * 1024 * 1024
SHARD_PREFIX = "batch_input_strategy"
STATE_FILENAME = "batch_state_strategy.json"
OUTPUT_FILENAME = "oracle_raw_scores.jsonl"


def _load_benchmark(path: Path, limit: int | None, exclude_test: bool) -> list[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if exclude_test and obj.get("split") == "test":
                continue
            samples.append(obj)
            if limit and len(samples) >= limit:
                break
    return samples


def submit_batches(
    benchmark_path: Path, 
    output_dir: Path, 
    limit: Optional[int] = None, 
    include_test: bool = False, 
    completion_window: str = "24h"
) -> int:
    """Submit OpenAI Batch for strategy generation and write state file. Returns exit code."""
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = _load_benchmark(
        benchmark_path,
        limit=limit,
        exclude_test=not include_test,
    )
    if not samples:
        logger.error("No samples to process (check --limit and splits).")
        return 1

    logger.info(
        "Loaded %d samples (Test excluded: %s)", len(samples), not include_test
    )

    base_prompt = get_generator_prompt()
    model_id = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    recorder = BatchRecorderGenerator(
        base_prompt=base_prompt,
        model_id=model_id,
        temperature=0.0,
        max_tokens=int(os.getenv("GENERATOR_MAX_TOKENS", "512")),
    )

    logger.info("Building strategies (with recorder)...")
    dense = build_dense_strategy(str(output_dir))
    graph = build_graph_strategy(str(output_dir))
    dense.generator = recorder
    graph.generator = recorder

    for idx, sample in enumerate(samples):
        question = sample.get("question", "").strip()
        if not question:
            continue
        recorder.next_custom_id = f"{idx}_dense"
        dense.retrieve_and_generate(question)
        recorder.next_custom_id = f"{idx}_graph"
        graph.retrieve_and_generate(question)

    if not recorder.recorded_requests:
        logger.error("No requests recorded. Check strategy retrieval.")
        return 1

    logger.info(
        "Recorded %d requests. Building batch JSONL...", len(recorder.recorded_requests)
    )

    shards: list[dict] = []
    shard_idx = 0
    shard_count = 0
    shard_bytes = 0
    shard_file = output_dir / f"{SHARD_PREFIX}_{shard_idx:03d}.jsonl"
    batch_out = shard_file.open("w", encoding="utf-8")

    def flush_shard():
        nonlocal shard_idx, shard_count, shard_bytes, shard_file, batch_out
        if shard_count == 0:
            return
        batch_out.close()
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("Uploading shard %d (%d requests)...", shard_idx, shard_count)
        with shard_file.open("rb") as r:
            uploaded = client.files.create(file=r, purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata={
                "description": "Strategy generation (Dense+Graph)",
                "benchmark": str(benchmark_path),
                "shard": str(shard_idx),
            },
        )
        shards.append(
            {
                "batch_id": batch.id,
                "input_path": str(shard_file),
                "request_count": shard_count,
            }
        )
        logger.info("Shard %d batch created: %s", shard_idx, batch.id)
        shard_idx += 1
        shard_count = 0
        shard_bytes = 0
        shard_file = output_dir / f"{SHARD_PREFIX}_{shard_idx:03d}.jsonl"
        batch_out = shard_file.open("w", encoding="utf-8")

    for req in recorder.recorded_requests:
        line = build_batch_line(req["custom_id"], req["body"])
        line_json = json.dumps(line, ensure_ascii=False) + "\n"
        line_bytes = len(line_json.encode("utf-8"))

        if shard_count > 0 and (
            shard_count >= BATCH_LIMIT_REQUESTS
            or shard_bytes + line_bytes > BATCH_LIMIT_BYTES
        ):
            flush_shard()

        batch_out.write(line_json)
        shard_count += 1
        shard_bytes += line_bytes

    if shard_count > 0:
        flush_shard()
    else:
        batch_out.close()

    state = {
        "shards": shards,
        "benchmark_path": str(benchmark_path),
        "samples_count": len(samples),
        "samples": samples,
        "total_requests": sum(s["request_count"] for s in shards),
    }
    state_path = output_dir / STATE_FILENAME
    with state_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    logger.info(
        "Submitted %d shard(s), %d requests. State: %s",
        len(shards),
        state["total_requests"],
        state_path,
    )
    for s in shards:
        print(s["batch_id"])
    return 0


def collect_batches(state_path: Path, output_dir: Path, output_file: Optional[Path] = None) -> int:
    """Collect strategy batch results and save to JSONL. Returns exit code."""
    import re
    
    CUSTOM_ID_PATTERN = re.compile(r"^(\d+)_(dense|graph)$")
    
    if not state_path.is_file():
        logger.error("State file not found: %s", state_path)
        return 1

    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)

    shards = state.get("shards") or []
    samples = state.get("samples") or []
    if not shards:
        logger.error("No shards in state file.")
        return 1
    if not samples:
        logger.error("No samples in state file. Re-submit with a state that includes samples.")
        return 1

    output_path = output_file if output_file else output_dir / OUTPUT_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY is not set.")
        return 1

    client = OpenAI(api_key=api_key)

    answers_by_id: dict[str, str] = {}
    completed = 0
    failed = 0

    for shard in shards:
        batch_id = shard.get("batch_id")
        if not batch_id:
            continue
        batch = client.batches.retrieve(batch_id)
        if batch.status != "completed":
            logger.warning("Shard batch %s not completed (status=%s). Skipping.", batch_id, batch.status)
            continue

        output_file_id = getattr(batch, "output_file_id", None) or getattr(batch, "output_file", None)
        if not output_file_id:
            logger.warning("Shard batch %s has no output file.", batch_id)
            continue

        logger.info("Downloading shard output for %s...", batch_id)
        content = client.files.content(output_file_id)
        text = content.read()
        if isinstance(text, bytes):
            text = text.decode("utf-8")

        for line in text.strip().split("\n"):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid, answer = parse_generation_output(obj)
            if obj.get("error"):
                failed += 1
                answers_by_id[cid] = ""
            else:
                completed += 1
                answers_by_id[cid] = answer

    by_idx: dict[int, dict[str, str]] = {}
    for cid, answer in answers_by_id.items():
        m = CUSTOM_ID_PATTERN.match(cid)
        if not m:
            continue
        idx = int(m.group(1))
        strategy = m.group(2)
        by_idx.setdefault(idx, {})[strategy] = answer

    logger.info("Writing %s (one line per question, pred_dense + pred_graph)...", output_path)
    with output_path.open("w", encoding="utf-8") as out:
        for idx, sample in enumerate(samples):
            row = dict(sample)
            preds = by_idx.get(idx, {})
            row["pred_dense"] = preds.get("dense", "")
            row["pred_graph"] = preds.get("graph", "")
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info(
        "Summary: completed=%d failed=%d; wrote %d rows to %s",
        completed,
        failed,
        len(samples),
        output_path,
    )
    print(f"Output: {output_path}")
    return 0
