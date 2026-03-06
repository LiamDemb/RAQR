"""Tests for the Graph retrieval strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from raqr.data.build_graph import build_graph
from raqr.entity_alias_resolver import EntityAliasResolver
from raqr.generator import GenerationResult
from raqr.strategies.base import StrategyResult
from raqr.strategies.graph import GraphStrategy, SpacyQueryEntityExtractor


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


def test_graph_strategy_df_downweighting_prefers_rare_entity_context():
    chunks = [
        {
            "chunk_id": "c_hub",
            "metadata": {
                "entities": [{"norm": "hub", "type": "ORG"}],
                "relations": [],
            },
        },
        {
            "chunk_id": "c_rare",
            "metadata": {
                "entities": [{"norm": "rare", "type": "ORG"}],
                "relations": [],
            },
        },
    ]
    graph = build_graph(chunks)
    strategy = GraphStrategy(
        graph_store=_GraphStoreStub(graph=graph),
        corpus=_CorpusStub({"c_hub": "Hub chunk", "c_rare": "Rare chunk"}),
        generator=_GeneratorStub(),
        entity_extractor=_StaticExtractor(["hub", "rare"]),
        top_k=2,
        max_hops=1,
        entity_df_by_norm={"hub": 10000, "rare": 0},
        synergy_gamma=0.0,
    )

    result = strategy.retrieve_and_generate("hub rare query")
    assert result.status == "OK"
    assert "Rare chunk" in result.context_scores[0][0]


def test_graph_strategy_synergy_bonus_prefers_joint_evidence():
    chunks = [
        {
            "chunk_id": "c_joint",
            "metadata": {
                "entities": [{"norm": "a", "type": "ORG"}, {"norm": "b", "type": "ORG"}],
                "relations": [],
            },
        },
        {
            "chunk_id": "c_a",
            "metadata": {
                "entities": [{"norm": "a", "type": "ORG"}],
                "relations": [],
            },
        },
        {
            "chunk_id": "c_b",
            "metadata": {
                "entities": [{"norm": "b", "type": "ORG"}],
                "relations": [],
            },
        },
    ]
    graph = build_graph(chunks)
    strategy = GraphStrategy(
        graph_store=_GraphStoreStub(graph=graph),
        corpus=_CorpusStub({"c_joint": "Joint", "c_a": "Only A", "c_b": "Only B"}),
        generator=_GeneratorStub(),
        entity_extractor=_StaticExtractor(["a", "b"]),
        top_k=3,
        max_hops=1,
        entity_df_by_norm={"a": 0, "b": 0},
        start_entity_weight=1.0,
        expanded_entity_weight=0.0,
        synergy_gamma=1.0,
    )

    result = strategy.retrieve_and_generate("a b query")
    assert result.status == "OK"
    assert "Joint" in result.context_scores[0][0]


def test_query_extractor_finds_entity_in_capitalized_query(monkeypatch):
    """Query extractor finds 'United States' via capitalization heuristic."""
    class _Doc:
        ents = []
        noun_chunks = []

    class _NlpStub:
        def __call__(self, text: str):
            return _Doc()

    monkeypatch.setattr("raqr.strategies.graph.load_spacy", lambda *args, **kwargs: _NlpStub())

    chunks = [
        {
            "chunk_id": "c_us",
            "metadata": {"entities": [{"norm": "united states", "type": "GPE"}], "relations": []},
        }
    ]
    graph = build_graph(chunks)
    extractor = SpacyQueryEntityExtractor(
        alias_resolver=EntityAliasResolver(alias_map={}),
    )
    strategy = GraphStrategy(
        graph_store=_GraphStoreStub(graph=graph),
        corpus=_CorpusStub({"c_us": "United States context."}),
        generator=_GeneratorStub(),
        entity_extractor=extractor,
        top_k=1,
        max_hops=1,
    )

    result = strategy.retrieve_and_generate("What happened in the United States?")
    assert result.status == "OK"
