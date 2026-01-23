from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from .schemas import BenchmarkItem, Document, sha256_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoaderRecord:
    document: Document
    benchmark_item: BenchmarkItem


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


def _strip_html(text: str) -> str:
    output: List[str] = []
    inside_tag = False
    for ch in text:
        if ch == "<":
            inside_tag = True
            continue
        if ch == ">":
            inside_tag = False
            continue
        if not inside_tag:
            output.append(ch)
    return "".join(output)


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
) -> Iterator[LoaderRecord]:
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

        document_block = row.get("document")
        document_html = None
        document_title = None
        tokens = None
        if isinstance(document_block, dict):
            document_html = document_block.get("html")
            document_title = document_block.get("title")
            tokens = document_block.get("tokens")

        context = _first_non_empty(
            row.get("context"),
            row.get("document_text"),
            row.get("document"),
            row.get("paragraph"),
        )
        if not context and "document_html" in row:
            context = _strip_html(str(row.get("document_html", "")))
        if not context and document_html:
            context = _strip_html(str(document_html))
        if not context and isinstance(tokens, list):
            token_text = [
                token.get("token", "")
                for token in tokens
                if isinstance(token, dict) and not token.get("is_html", False)
            ]
            context = " ".join([t for t in token_text if t.strip()])

        if not context or not answers:
            continue

        title = _first_non_empty(
            row.get("document_title"),
            row.get("title"),
            document_title,
        )
        timestamp = _first_non_empty(row.get("timestamp"), row.get("date"))
        metadata = {"source": source}
        if title:
            metadata["title"] = title
        if timestamp:
            metadata["timestamp"] = timestamp

        document = Document(id=sha256_text(context), content=context, metadata=metadata)
        benchmark_item = BenchmarkItem(
            question_id=sha256_text(question),
            question=question,
            gold_answers=answers,
            dataset_source=source,
            split="",
            dataset_version=dataset_version,
        )
        yield LoaderRecord(document=document, benchmark_item=benchmark_item)
        count += 1
        if max_rows and count >= max_rows:
            break


def load_complextempqa(
    path: str,
    dataset_version: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Iterator[LoaderRecord]:
    source = "complextempqa"
    count = 0
    for row in _iter_json_records(Path(path)):
        question = _first_non_empty(row.get("question"), row.get("query"))
        answers = _as_list(row.get("answers") or row.get("answer"))
        context = _first_non_empty(
            row.get("context"),
            row.get("passage"),
            row.get("document"),
            row.get("paragraph"),
        )
        if not context and question and answers:
            context = f"{question} {' '.join(answers)}".strip()
        if not question or not answers or not context:
            continue
        title = _first_non_empty(row.get("title"))
        timestamp = _first_non_empty(
            row.get("timestamp"),
            row.get("date"),
            row.get("year"),
        )
        metadata = {"source": source}
        if "context" not in row:
            metadata["context_fallback"] = True
        if title:
            metadata["title"] = title
        if timestamp:
            metadata["timestamp"] = timestamp
        document = Document(id=sha256_text(context), content=context, metadata=metadata)
        benchmark_item = BenchmarkItem(
            question_id=sha256_text(question),
            question=question,
            gold_answers=answers,
            dataset_source=source,
            split="",
            dataset_version=dataset_version,
        )
        yield LoaderRecord(document=document, benchmark_item=benchmark_item)
        count += 1
        if max_rows and count >= max_rows:
            break


def load_wikiwhy(
    path: str,
    dataset_version: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Iterator[LoaderRecord]:
    source = "wikiwhy"
    count = 0
    path_obj = Path(path)
    if path_obj.suffix == ".csv":
        with path_obj.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                question = _first_non_empty(row.get("question"), row.get("query"))
                answers = _as_list(row.get("answer") or row.get("answers"))
                context = _first_non_empty(
                    row.get("context"),
                    row.get("passage"),
                    row.get("paragraph"),
                    row.get("rationale"),
                )
                if not question or not answers or not context:
                    continue
                title = _first_non_empty(row.get("title"))
                timestamp = _first_non_empty(row.get("timestamp"), row.get("date"))
                metadata = {"source": source}
                if title:
                    metadata["title"] = title
                if timestamp:
                    metadata["timestamp"] = timestamp
                document = Document(
                    id=sha256_text(context), content=context, metadata=metadata
                )
                benchmark_item = BenchmarkItem(
                    question_id=sha256_text(question),
                    question=question,
                    gold_answers=answers,
                    dataset_source=source,
                    split="",
                    dataset_version=dataset_version,
                )
                yield LoaderRecord(document=document, benchmark_item=benchmark_item)
                count += 1
                if max_rows and count >= max_rows:
                    break
    else:
        for row in _iter_json_records(path_obj):
            question = _first_non_empty(row.get("question"), row.get("query"))
            answers = _as_list(
                row.get("answer")
                or row.get("answers")
                or row.get("cause")
                or row.get("effect")
            )
            context = _first_non_empty(
                row.get("ctx"),
                row.get("context"),
                row.get("passage"),
                row.get("paragraph"),
                row.get("rationale"),
            )
            if not question or not answers or not context:
                continue
            title = _first_non_empty(row.get("title"))
            timestamp = _first_non_empty(row.get("timestamp"), row.get("date"))
            metadata = {"source": source}
            if title:
                metadata["title"] = title
            if timestamp:
                metadata["timestamp"] = timestamp
            document = Document(
                id=sha256_text(context), content=context, metadata=metadata
            )
            benchmark_item = BenchmarkItem(
                question_id=sha256_text(question),
                question=question,
                gold_answers=answers,
                dataset_source=source,
                split="",
                dataset_version=dataset_version,
            )
            yield LoaderRecord(document=document, benchmark_item=benchmark_item)
            count += 1
            if max_rows and count >= max_rows:
                break
