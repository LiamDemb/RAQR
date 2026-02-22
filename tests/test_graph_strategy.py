"""Tests for the Graph retrieval strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from raqr.data.build_graph import build_graph
from raqr.generator import GenerationResult
from raqr.strategies.base import StrategyResult
from raqr.strategies.graph import GraphStrategy


@dataclass
class _StaticExtractor:
    entities: List[str]

    def extract(self, query: str) -> List[str]:
        return list(self.entities)


@dataclass
class _GraphStoreStub:
    graph: object

    def load(self):
        return self.graph


class _CorpusStub:
    def __init__(self, texts: dict[str, str]):
        self._texts = texts

    def get_text(self, chunk_id: str) -> Optional[str]:
        return self._texts.get(chunk_id)


class _GeneratorStub:
    def generate(self, query: str, context: List[str]) -> GenerationResult:
        return GenerationResult(
            text=f"mocked answer for: {query}",
            model_id="test",
            latency_ms=1.0,
            prompt_hash="abc",
            sampling={},
        )


def _toy_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "c1",
            "metadata": {
                "entities": [{"norm": "a", "type": "ORG"}, {"norm": "b", "type": "ORG"}],
                "relations": [{"subj_norm": "a", "pred": "causes", "obj_norm": "b"}],
            },
        },
        {
            "chunk_id": "c2",
            "metadata": {
                "entities": [{"norm": "b", "type": "ORG"}, {"norm": "c", "type": "ORG"}],
                "relations": [{"subj_norm": "b", "pred": "causes", "obj_norm": "c"}],
            },
        },
    ]


def test_graph_strategy_returns_ok_with_contexts():
    graph = build_graph(_toy_chunks())
    strategy = GraphStrategy(
        graph_store=_GraphStoreStub(graph=graph),
        corpus=_CorpusStub({"c1": "A caused B.", "c2": "B caused C."}),
        generator=_GeneratorStub(),
        entity_extractor=_StaticExtractor(["a"]),
        top_k=3,
        max_hops=1,
    )

    result = strategy.retrieve_and_generate("How is A related to B?")

    assert isinstance(result, StrategyResult)
    assert result.status == "OK"
    assert result.answer
    assert len(result.context_scores) > 0
    assert "retrieval" in result.latency_ms
    assert "generation" in result.latency_ms
    assert "total" in result.latency_ms


def test_graph_strategy_returns_no_context_for_unmatched_entities():
    graph = build_graph(_toy_chunks())
    strategy = GraphStrategy(
        graph_store=_GraphStoreStub(graph=graph),
        corpus=_CorpusStub({"c1": "A caused B.", "c2": "B caused C."}),
        generator=_GeneratorStub(),
        entity_extractor=_StaticExtractor(["does-not-exist"]),
        top_k=3,
        max_hops=1,
    )

    result = strategy.retrieve_and_generate("Unknown entity query")

    assert result.status == "NO_CONTEXT"
    assert result.context_scores == []
    assert result.answer == ""


def test_graph_strategy_is_deterministic_for_same_query():
    graph = build_graph(_toy_chunks())
    strategy = GraphStrategy(
        graph_store=_GraphStoreStub(graph=graph),
        corpus=_CorpusStub({"c1": "A caused B.", "c2": "B caused C."}),
        generator=_GeneratorStub(),
        entity_extractor=_StaticExtractor(["a"]),
        top_k=3,
        max_hops=1,
    )

    result1 = strategy.retrieve_and_generate("How is A related to C?")
    result2 = strategy.retrieve_and_generate("How is A related to C?")

    assert result1.status == result2.status
    assert result1.context_scores == result2.context_scores
