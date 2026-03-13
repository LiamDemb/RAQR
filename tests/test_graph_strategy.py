"""Tests for the bundle-based Graph retrieval strategy."""

from __future__ import annotations

from raqr.data.build_graph import build_graph
from raqr.strategies.base import StrategyResult
from raqr.strategies.graph import GraphStrategy

from tests._graph_test_utils import CorpusStub, GeneratorStub, GraphStoreStub, StaticExtractor, strategy_embedder, toy_chunks



def test_graph_strategy_returns_ok_with_contexts():
    graph = build_graph(toy_chunks())
    strategy = GraphStrategy(
        graph_store=GraphStoreStub(graph=graph),
        corpus=CorpusStub({"c1": "A caused B.", "c2": "B caused C."}),
        generator=GeneratorStub(),
        entity_extractor=StaticExtractor(["a"]),
        embedder=strategy_embedder(),
        top_k=3,
        max_hops=2,
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
    graph = build_graph(toy_chunks())
    strategy = GraphStrategy(
        graph_store=GraphStoreStub(graph=graph),
        corpus=CorpusStub({"c1": "A caused B.", "c2": "B caused C."}),
        generator=GeneratorStub(),
        entity_extractor=StaticExtractor(["does-not-exist"]),
        embedder=strategy_embedder(),
        top_k=3,
        max_hops=2,
    )

    result = strategy.retrieve_and_generate("Unknown entity query")

    assert result.status == "NO_CONTEXT"
    assert result.context_scores == []
    assert result.answer == ""



def test_graph_strategy_is_deterministic_for_same_query():
    graph = build_graph(toy_chunks())
    strategy = GraphStrategy(
        graph_store=GraphStoreStub(graph=graph),
        corpus=CorpusStub({"c1": "A caused B.", "c2": "B caused C."}),
        generator=GeneratorStub(),
        entity_extractor=StaticExtractor(["a"]),
        embedder=strategy_embedder(),
        top_k=3,
        max_hops=2,
    )

    result1 = strategy.retrieve_and_generate("How is A related to C?")
    result2 = strategy.retrieve_and_generate("How is A related to C?")

    assert result1.status == result2.status
    assert result1.context_scores == result2.context_scores



def test_graph_strategy_debug_trace_contains_path_and_bundle_fields():
    graph = build_graph(toy_chunks())
    strategy = GraphStrategy(
        graph_store=GraphStoreStub(graph=graph),
        corpus=CorpusStub({"c1": "A caused B.", "c2": "B caused C."}),
        generator=GeneratorStub(),
        entity_extractor=StaticExtractor(["a"]),
        embedder=strategy_embedder(),
        top_k=3,
        max_hops=2,
        bidirectional=True,
    )

    result = strategy.retrieve_and_generate("How is A related to C?", debug=True)

    assert result.status == "OK"
    assert result.debug_info is not None
    assert "candidate_paths" in result.debug_info
    assert "bundle_trace" in result.debug_info
