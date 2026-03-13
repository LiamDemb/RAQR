from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple


class BaseStrategy(ABC):
    name: str  # "Dense", "Temporal", "Graph"

    @abstractmethod
    def retrieve_and_generate(self, query: str, **kwargs) -> StrategyResult:
        """Run retrieval (and generation) for a single query.

        Implementations may accept optional kwargs for debugging or tracing.
        """
        raise NotImplementedError


@dataclass
class StrategyResult:
    answer: str
    context_scores: List[Tuple[str, float]]
    latency_ms: Dict[str, float]
    status: Literal["OK", "NO_CONTEXT", "ERROR"]
    error: Optional[str] = None
    debug_info: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        # Sorting context scores in descending order of score
        self.context_scores.sort(key=lambda x: x[1], reverse=True)

        if self.status == "OK":
            if self.context_scores is None or len(self.context_scores) == 0:
                raise ValueError("Status OK has an empty context_scores list")
            if self.error is not None:
                raise ValueError("Status OK has an error message")
        elif self.status == "NO_CONTEXT":
            if self.context_scores is not None and len(self.context_scores) > 0:
                raise ValueError(
                    "Status NO_CONTEXT has a non-empty context_scores list"
                )
            if self.error is not None:
                raise ValueError("Status NO_CONTEXT has an error message")
        elif self.status == "ERROR":
            if self.error is None:
                raise ValueError("Status ERROR has no error message")
