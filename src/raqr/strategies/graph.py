from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Set, Tuple

import networkx as nx

from raqr.entity_alias_resolver import EntityAliasResolver
from raqr.generator import Generator
from raqr.graph_store import NetworkXGraphStore
from raqr.loaders import ChunkIdToText
from raqr.strategies.base import BaseStrategy, StrategyResult
from raqr.data.enrich_entities import extract_entities_spacy, load_spacy


logger = logging.getLogger(__name__)


class QueryEntityExtractor(Protocol):
    def extract(self, query: str) -> List[str]:
        ...


@dataclass
class SpacyQueryEntityExtractor:
    """Query entity extractor using the same normalization policy as corpus building."""

    alias_resolver: EntityAliasResolver
    spacy_model: str = os.getenv("SPACY_MODEL", "en_core_web_sm")
    _nlp: Optional[object] = None

    def _ensure_loaded(self) -> None:
        if self._nlp is None:
            self._nlp = load_spacy(self.spacy_model)

    def extract(self, query: str) -> List[str]:
        self._ensure_loaded()
        entities = extract_entities_spacy(
            text=query,
            nlp=self._nlp,
            alias_map=self.alias_resolver.alias_map,
        )
        return sorted({ent["norm"] for ent in entities if ent.get("norm")})


@dataclass
class GraphStrategy(BaseStrategy):
    name = "Graph"
    graph_store: NetworkXGraphStore = None
    corpus: ChunkIdToText = None
    generator: Generator = None
    entity_extractor: QueryEntityExtractor = None
    top_k: int = int(os.getenv("GRAPH_TOP_K", "10"))
    max_hops: int = 1

    _graph: Optional[nx.DiGraph] = None
    _queries_seen: int = 0
    _queries_with_match: int = 0

    def _ensure_loaded(self) -> None:
        if self._graph is None:
            self._graph = self.graph_store.load()

    @staticmethod
    def _chunk_id_from_node(node_id: str) -> Optional[str]:
        if node_id.startswith("C:"):
            return node_id[2:]
        return None

    def _expand_relational_entities(self, start_nodes: Set[str]) -> Set[str]:
        if self.max_hops <= 0:
            return set(start_nodes)

        frontier = set(start_nodes)
        visited = set(start_nodes)
        for _ in range(self.max_hops):
            next_frontier: Set[str] = set()
            for source in frontier:
                for _, target, data in self._graph.out_edges(source, data=True):
                    if data.get("kind") != "rel":
                        continue
                    if self._graph.nodes.get(target, {}).get("kind") != "entity":
                        continue
                    if target not in visited:
                        visited.add(target)
                        next_frontier.add(target)
            frontier = next_frontier
            if not frontier:
                break
        return visited

    def retrieve_and_generate(self, query: str) -> StrategyResult:
        t0 = time.perf_counter()
        timings: Dict[str, float] = {}
        stage = "retrieval"

        try:
            r0 = time.perf_counter()
            self._ensure_loaded()
            extracted_norms = self.entity_extractor.extract(query)
            candidate_nodes = {f"E:{norm}" for norm in extracted_norms}
            start_nodes = {node for node in candidate_nodes if self._graph.has_node(node)}

            self._queries_seen += 1
            if start_nodes:
                self._queries_with_match += 1
            match_rate = self._queries_with_match / max(1, self._queries_seen)
            logger.info(
                "Graph entity match rate: %.2f%% (%d/%d)",
                match_rate * 100.0,
                self._queries_with_match,
                self._queries_seen,
            )

            if not start_nodes:
                timings["retrieval"] = (time.perf_counter() - r0) * 1000.0
                timings["total"] = (time.perf_counter() - t0) * 1000.0
                return StrategyResult(
                    answer="",
                    context_scores=[],
                    latency_ms=timings,
                    status="NO_CONTEXT",
                )

            expanded_nodes = self._expand_relational_entities(start_nodes)
            chunk_scores: Dict[str, float] = {}
            for entity_node in expanded_nodes:
                weight = 2.0 if entity_node in start_nodes else 1.0
                for _, target, data in self._graph.out_edges(entity_node, data=True):
                    if data.get("kind") != "appears_in":
                        continue
                    chunk_id = self._chunk_id_from_node(target)
                    if not chunk_id:
                        continue
                    chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight

            if not chunk_scores:
                timings["retrieval"] = (time.perf_counter() - r0) * 1000.0
                timings["total"] = (time.perf_counter() - t0) * 1000.0
                return StrategyResult(
                    answer="",
                    context_scores=[],
                    latency_ms=timings,
                    status="NO_CONTEXT",
                )

            ranked_chunks = sorted(chunk_scores.items(), key=lambda item: item[1], reverse=True)
            pairs: List[Tuple[str, float]] = []
            for chunk_id, score in ranked_chunks:
                text = self.corpus.get_text(chunk_id)
                if not text:
                    continue
                pairs.append((text, float(score)))
                if len(pairs) >= self.top_k:
                    break

            timings["retrieval"] = (time.perf_counter() - r0) * 1000.0
            if not pairs:
                timings["total"] = (time.perf_counter() - t0) * 1000.0
                return StrategyResult(
                    answer="",
                    context_scores=[],
                    latency_ms=timings,
                    status="NO_CONTEXT",
                )

            stage = "generation"
            g0 = time.perf_counter()
            contexts = [ctx for (ctx, _) in pairs]
            generation = self.generator.generate(query=query, context=contexts)
            timings["generation"] = (time.perf_counter() - g0) * 1000.0
            timings["total"] = (time.perf_counter() - t0) * 1000.0
            return StrategyResult(
                answer=generation.text,
                context_scores=pairs,
                latency_ms=timings,
                status="OK",
            )
        except Exception as exc:
            timings["total"] = (time.perf_counter() - t0) * 1000.0
            return StrategyResult(
                answer="",
                context_scores=[],
                latency_ms=timings,
                status="ERROR",
                error=f"GraphStrategy failed during {stage}: {type(exc).__name__}: {exc}",
            )
