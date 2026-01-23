from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def sha256_text(value: str) -> str:
    normalized = value.strip().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


@dataclass(frozen=True)
class Document:
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BenchmarkItem:
    question_id: str
    question: str
    gold_answers: List[str]
    dataset_source: str
    split: str
    dataset_version: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "question_id": self.question_id,
            "question": self.question,
            "gold_answers": list(self.gold_answers),
            "dataset_source": self.dataset_source,
            "split": self.split,
        }
        if self.dataset_version:
            payload["dataset_version"] = self.dataset_version
        return payload
