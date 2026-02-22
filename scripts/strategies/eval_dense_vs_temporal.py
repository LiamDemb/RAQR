from __future__ import annotations

import argparse
import collections
import json
import os
import re
from dataclasses import dataclass
from typing import List

from dotenv import load_dotenv

from raqr.embedder import SentenceTransformersEmbedder
from raqr.generator import SimpleLLMGenerator
from raqr.index_store import FaissIndexStore
from raqr.loaders import JsonCorpusLoader, VectorMetaMapper, VectorMetaWithYears
from raqr.strategies.dense import DenseStrategy
from raqr.strategies.temporal import TemporalStrategy


@dataclass
class EvalResult:
    total: int = 0
    em_sum: float = 0.0
    f1_sum: float = 0.0
    errors: int = 0


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _tokens(s: str) -> List[str]:
    n = normalize_text(s)
    return n.split() if n else []


def parse_answers(row: dict) -> List[str]:
    raw = row.get("answer") or row.get("answers") or []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    if isinstance(raw, str) and raw.strip():
        return [raw.strip()]
    return []


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize_text(pred) == normalize_text(gold) else 0.0


def f1_score(pred: str, gold: str) -> float:
    pred_toks = _tokens(pred)
    gold_toks = _tokens(gold)
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0

    common = collections.Counter(pred_toks) & collections.Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def score_prediction(pred: str, gold_answers: List[str]) -> tuple[float, float]:
    if not pred or not gold_answers:
        return (0.0, 0.0)

    golds = [g for g in gold_answers if g.strip()]
    if not golds:
        return (0.0, 0.0)

    # SQuAD-style: take max score over gold aliases.
    best_em = max(exact_match(pred, g) for g in golds)
    best_f1 = max(f1_score(pred, g) for g in golds)
    return (best_em, best_f1)


def load_questions(path: str, limit: int | None) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = str(row.get("question", "")).strip()
            a = parse_answers(row)
            if q and a:
                rows.append({"question": q, "answers": a})
            if limit and len(rows) >= limit:
                break
    return rows


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Evaluate Dense vs Temporal on ComplexTempQA JSONL.")
    parser.add_argument("--dataset", default="data/raw/complex_tempqa_50.jsonl")
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "data/processed_rebel"))
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "all-MiniLM-L6-v2"))
    parser.add_argument("--openai-model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N rows.")
    parser.add_argument("--dense-top-k", type=int, default=int(os.getenv("DENSE_TOP_K", "10")))
    parser.add_argument("--temporal-top-k", type=int, default=int(os.getenv("TEMPORAL_TOP_K", "10")))
    parser.add_argument(
        "--temporal-candidate-multiplier",
        type=int,
        default=int(os.getenv("TEMPORAL_CANDIDATE_MULTIPLIER", "5")),
    )
    args = parser.parse_args()

    # Shared components
    index_path = f"{args.output_dir}/vector_index.faiss"
    meta_path = f"{args.output_dir}/vector_meta.parquet"
    corpus_path = f"{args.output_dir}/corpus.jsonl"

    embedder = SentenceTransformersEmbedder(model_name=args.model_name)
    generator = SimpleLLMGenerator(
        model_id=args.openai_model,
        base_prompt=(
            "Answer the question based only on the provided context. "
            "If the context does not contain the answer, say so."
        ),
    )
    corpus = JsonCorpusLoader(jsonl_path=corpus_path)

    dense = DenseStrategy(
        index_store=FaissIndexStore(index_path=index_path),
        meta=VectorMetaMapper(parquet_path=meta_path),
        embedder=embedder,
        generator=generator,
        corpus=corpus,
        top_k=args.dense_top_k,
    )

    temporal = TemporalStrategy(
        index_store=FaissIndexStore(index_path=index_path),
        meta=VectorMetaWithYears(parquet_path=meta_path),
        embedder=embedder,
        generator=generator,
        corpus=corpus,
        top_k=args.temporal_top_k,
        candidate_multiplier=args.temporal_candidate_multiplier,
    )

    rows = load_questions(args.dataset, args.limit)
    print(f"Loaded {len(rows)} questions from {args.dataset}")

    dense_res = EvalResult()
    temp_res = EvalResult()

    for i, row in enumerate(rows, start=1):
        q = row["question"]
        gold = row["answers"]

        # Dense
        dense_res.total += 1
        d = dense.retrieve_and_generate(q)
        if d.status == "ERROR":
            dense_res.errors += 1
        else:
            d_em, d_f1 = score_prediction(d.answer, gold)
            dense_res.em_sum += d_em
            dense_res.f1_sum += d_f1

        # Temporal
        temp_res.total += 1
        t = temporal.retrieve_and_generate(q)
        if t.status == "ERROR":
            temp_res.errors += 1
        else:
            t_em, t_f1 = score_prediction(t.answer, gold)
            temp_res.em_sum += t_em
            temp_res.f1_sum += t_f1

        print(
            f"[{i}/{len(rows)}] "
            f"Dense={d.status} (EM={dense_res.em_sum:.1f}, F1={dense_res.f1_sum:.2f}) | "
            f"Temporal={t.status} (EM={temp_res.em_sum:.1f}, F1={temp_res.f1_sum:.2f})"
        )

    def pct(v: float, n: int) -> float:
        return (100.0 * v / n) if n else 0.0

    print("\n=== Final Results ===")
    print(
        f"Dense:    EM={pct(dense_res.em_sum, dense_res.total):.2f} "
        f"F1={pct(dense_res.f1_sum, dense_res.total):.2f} "
        f"errors={dense_res.errors}/{dense_res.total}"
    )
    print(
        f"Temporal: EM={pct(temp_res.em_sum, temp_res.total):.2f} "
        f"F1={pct(temp_res.f1_sum, temp_res.total):.2f} "
        f"errors={temp_res.errors}/{temp_res.total}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())