from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .schemas import BenchmarkItem, sha256_text

logger = logging.getLogger(__name__)


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _first_non_empty(*values: Any) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _has_context_nq(row: Dict[str, Any]) -> bool:
    """Check if row has context from any NQ-style field."""
    doc_block = row.get("document")
    doc_html = doc_block.get("html") if isinstance(doc_block, dict) else None
    tokens = doc_block.get("tokens") if isinstance(doc_block, dict) else None
    if _first_non_empty(
        row.get("context"),
        row.get("document_text"),
        row.get("document") if isinstance(row.get("document"), str) else None,
        row.get("paragraph"),
    ):
        return True
    if _first_non_empty(row.get("document_html"), doc_html):
        return True
    if isinstance(tokens, list):
        for t in tokens:
            if isinstance(t, dict) and t.get("token") and not t.get("is_html"):
                return True
    return False


def _iter_json_records(path: Path) -> Iterator[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return
    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            for item in data:
                yield item
        elif isinstance(data, dict) and "data" in data:
            for item in data["data"]:
                yield item
        else:
            raise ValueError(f"Unsupported JSON structure in {path}")
        return
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def load_nq(
    path: str,
    dataset_version: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Iterator[BenchmarkItem]:
    """Load Natural Questions style data from JSON/JSONL."""
    source = "nq"
    count = 0
    for row in _iter_json_records(Path(path)):
        question_block = row.get("question")
        question = _first_non_empty(
            row.get("question_text"),
            row.get("questionText"),
            question_block.get("text") if isinstance(question_block, dict) else None,
            question_block if isinstance(question_block, str) else None,
        )
        if not question:
            continue

        answers: List[str] = []
        if "short_answers" in row:
            answers = _as_list(
                [item.get("text") for item in row.get("short_answers", []) if item]
            )
        if not answers and "answers" in row:
            answers = _as_list(row.get("answers"))
        if not answers and "short_answer" in row:
            answers = _as_list(row.get("short_answer"))
        if not answers and "answer" in row:
            answers = _as_list(row.get("answer"))
        if not answers and "annotations" in row:
            annotations = row.get("annotations")
            if isinstance(annotations, dict):
                annotations_iter = [annotations]
            elif isinstance(annotations, list):
                annotations_iter = annotations
            else:
                annotations_iter = []
            for annotation in annotations_iter:
                if not isinstance(annotation, dict):
                    continue
                short_answers = annotation.get("short_answers", [])
                if isinstance(short_answers, dict):
                    short_answers_iter = [short_answers]
                elif isinstance(short_answers, list):
                    short_answers_iter = short_answers
                else:
                    short_answers_iter = []
                for short_answer in short_answers_iter:
                    if isinstance(short_answer, dict) and short_answer.get("text"):
                        answers.append(str(short_answer["text"]).strip())
                    elif isinstance(short_answer, str) and short_answer.strip():
                        answers.append(short_answer.strip())
            answers = [a for a in answers if a]

        if not _has_context_nq(row) or not answers:
            continue

        yield BenchmarkItem(
            question_id=sha256_text(question),
            question=question,
            gold_answers=answers,
            dataset_source=source,
            split="",
            dataset_version=dataset_version,
        )
        count += 1
        if max_rows and count >= max_rows:
            break


def load_2wiki(
    path: str,
    dataset_version: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Iterator[BenchmarkItem]:
    """Load 2WikiMultiHopQA data from JSON/JSONL."""
    source = "2wiki"
    count = 0
    for row in _iter_json_records(Path(path)):
        question = row.get("question")
        answers = _as_list(row.get("answer"))
        if not question or not answers:
            continue

        supporting_facts = row.get("supporting_facts")
        if not supporting_facts:
            continue
        wiki_articles: List[str] = supporting_facts["title"]

        yield BenchmarkItem(
            question_id=sha256_text(question),
            question=question,
            gold_answers=answers,
            dataset_source=source,
            split="",
            dataset_version=dataset_version,
        )
        count += 1
        if max_rows and count >= max_rows:
            break
