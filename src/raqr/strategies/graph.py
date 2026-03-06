from __future__ import annotations

import math
import os
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

import networkx as nx

from raqr.entity_alias_resolver import EntityAliasResolver
from raqr.entity_index_store import EntityIndexStore
from raqr.generator import Generator
from raqr.graph_store import NetworkXGraphStore
from raqr.loaders import ChunkIdToText
from raqr.query_entity_extractor import LLMQueryEntityExtractor
from raqr.strategies.base import BaseStrategy, StrategyResult
from raqr.data.enrich_entities import (
    extract_entities_capitalization,
    extract_entities_spacy,
    load_spacy,
)


logger = logging.getLogger(__name__)


class QueryEntityExtractor(Protocol):
    def extract(self, query: str) -> List[str]:
        ...


def _default_query_entity_extractor(alias_resolver: EntityAliasResolver) -> QueryEntityExtractor:
    """Return LLM-based query entity extractor."""
    return LLMQueryEntityExtractor(alias_resolver=alias_resolver)


@dataclass
class SpacyQueryEntityExtractor:
    """Legacy query entity extractor: capitalization heuristic + spaCy NER. Use QUERY_ENTITY_EXTRACTOR=spacy."""

    alias_resolver: EntityAliasResolver
    spacy_model: str = os.getenv("SPACY_MODEL", "en_core_web_sm")
    _nlp: Optional[object] = None

    def _ensure_loaded(self) -> None:
        if self._nlp is None:
            self._nlp = load_spacy(
                self.spacy_model,
                use_noun_chunks=False,
            )

    def extract(self, query: str) -> List[str]:
        self._ensure_loaded()
        alias_map = self.alias_resolver.alias_map
        cap_norms = set(extract_entities_capitalization(query, alias_map))
        spacy_ents = extract_entities_spacy(
            text=query,
            nlp=self._nlp,
            alias_map=alias_map,
            use_noun_chunks=False,
        )
        spacy_norms = {e["norm"] for e in spacy_ents if e.get("norm")}
        return sorted(cap_norms | spacy_norms)


@dataclass
class GraphStrategy(BaseStrategy):
    name = "Graph"
    graph_store: NetworkXGraphStore = None
    corpus: ChunkIdToText = None
    generator: Generator = None
    entity_extractor: QueryEntityExtractor = None
    top_k: int = int(os.getenv("GRAPH_TOP_K", "10"))
    max_hops: int = int(os.getenv("GRAPH_MAX_HOPS", "1"))
    entity_df_by_norm: Optional[Dict[str, int]] = None
    entity_index_store: Optional[EntityIndexStore] = None
    entity_vector_top_k: int = int(os.getenv("GRAPH_ENTITY_VECTOR_TOP_K", "3"))
    entity_vector_threshold: float = float(os.getenv("GRAPH_ENTITY_VECTOR_THRESHOLD", "0.5"))
    start_entity_weight: float = float(os.getenv("GRAPH_START_ENTITY_WEIGHT", "2.0"))
    expanded_entity_weight: float = float(os.getenv("GRAPH_EXPANDED_ENTITY_WEIGHT", "1.0"))
    synergy_gamma: float = float(os.getenv("GRAPH_SYNERGY_GAMMA", "0.5"))
    bidirectional: bool = field(
        default_factory=lambda: os.getenv("GRAPH_BIDIRECTIONAL", "false").lower()
        in ("1", "true", "yes")
    )

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

    def _expand_relational_entities(
        self,
        start_nodes: Set[str],
        rel_edges: Optional[List[Tuple[str, str, str]]] = None,
    ) -> Set[str]:
        if self.max_hops <= 0:
            return set(start_nodes)

        frontier = set(start_nodes)
        visited = set(start_nodes)  # Prevents backtracking: never re-enter A after A→B
        for _ in range(self.max_hops):
            next_frontier: Set[str] = set()
            for source in frontier:
                # Forward: follow outgoing relation edges
                for _, target, data in self._graph.out_edges(source, data=True):
                    if data.get("kind") != "rel":
                        continue
                    if self._graph.nodes.get(target, {}).get("kind") != "entity":
                        continue
                    if rel_edges is not None:
                        rel_edges.append((source, target, str(data.get("label") or "")))
                    if target not in visited:  # Skip: prevents A→B→A backtracking
                        visited.add(target)
                        next_frontier.add(target)
                # Backward: follow incoming relation edges when bidirectional
                if self.bidirectional:
                    for neighbor, _, data in self._graph.in_edges(source, data=True):
                        if data.get("kind") != "rel":
                            continue
                        if self._graph.nodes.get(neighbor, {}).get("kind") != "entity":
                            continue
                        if rel_edges is not None:
                            label = str(data.get("label") or "")
                            rel_edges.append((source, neighbor, f"inv:{label}"))
                        if neighbor not in visited:  # Skip: prevents A→B→A backtracking
                            visited.add(neighbor)
                            next_frontier.add(neighbor)
            frontier = next_frontier
            if not frontier:
                break
        return visited

    def _entity_df_weight(self, entity_node: str) -> float:
        if not entity_node.startswith("E:"):
            return 1.0
        norm = entity_node[2:]
        df = (self.entity_df_by_norm or {}).get(norm, 0)
        if df < 0:
            df = 0
        return 1.0 / math.sqrt(1.0 + float(df))

    @staticmethod
    def _entity_display(name: str) -> str:
        """Strip E: prefix for readable path formatting."""
        return name[2:] if name.startswith("E:") else name

    def _get_paths_to_entities(
        self,
        start_nodes: Set[str],
        rel_edges: List[Tuple[str, str, str]],
    ) -> Dict[str, List[str]]:
        """Build shortest path from each start to each entity reachable via rel_edges.
        Returns entity_node -> list of path strings (one per start that reaches it).
        """
        if not rel_edges:
            return {n: [self._entity_display(n)] for n in start_nodes}
        # Build adjacency: source -> [(target, label), ...]
        adj: Dict[str, List[Tuple[str, str]]] = {}
        for source, target, label in rel_edges:
            adj.setdefault(source, []).append((target, label))
        result: Dict[str, List[str]] = {}
        for start in start_nodes:
            # BFS from start, store (parent, edge_label) for backtracking
            prev: Dict[str, Tuple[str, str]] = {}  # node -> (parent, label)
            queue: deque = deque([start])
            prev[start] = (start, "")  # sentinel
            while queue:
                node = queue.popleft()
                for target, label in adj.get(node, []):
                    if target not in prev:
                        prev[target] = (node, label)
                        queue.append(target)
            for entity, (parent, label) in prev.items():
                if entity == start:
                    path_str = self._entity_display(start)
                else:
                    # Backtrack to build chain: start -[l1]-> n1 -[l2]-> ... -> entity
                    chain: List[Tuple[str, str]] = []  # (label, node)
                    cur = entity
                    while cur in prev:
                        p, l = prev[cur]
                        if p == cur:
                            break
                        chain.append((l, cur))
                        cur = p
                    # Format: start -[l1]-> n1 -[l2]-> n2 ...
                    segs = [self._entity_display(start)]
                    for l, n in reversed(chain):
                        segs.append(f"-[{l}]-> {self._entity_display(n)}")
                    path_str = " ".join(segs)
                result.setdefault(entity, []).append(path_str)
        return result

    def _format_chunk_with_paths(
        self,
        chunk_id: str,
        text: str,
        chunk_to_entities: Dict[str, Set[str]],
        entity_paths: Dict[str, List[str]],
    ) -> str:
        """Prepend path metadata to chunk text for generator context."""
        entities = chunk_to_entities.get(chunk_id, set())
        paths: List[str] = []
        seen: Set[str] = set()
        for entity in entities:
            for path_str in entity_paths.get(entity, [self._entity_display(entity)]):
                if path_str not in seen:
                    seen.add(path_str)
                    paths.append(path_str)
        if not paths:
            return text
        prefix = "Relevant because: " + "; ".join(paths[:3])  # cap at 3 paths
        return f"[{prefix}]\n\n{text}"

    def retrieve_and_generate(self, query: str, **kwargs) -> StrategyResult:
        t0 = time.perf_counter()
        timings: Dict[str, float] = {}
        stage = "retrieval"
        debug = bool(kwargs.get("debug", False))
        debug_info: Optional[Dict[str, Any]] = {} if debug else None

        try:
            r0 = time.perf_counter()
            self._ensure_loaded()
            extracted_norms = self.entity_extractor.extract(query)
            candidate_nodes = {f"E:{norm}" for norm in extracted_norms}
            start_nodes = {node for node in candidate_nodes if self._graph.has_node(node)}
            unmatched_norms = [norm for norm in extracted_norms if f"E:{norm}" not in start_nodes]
            if unmatched_norms and self.entity_index_store:
                for norm in unmatched_norms:
                    for match_norm, score in self.entity_index_store.search(
                        norm,
                        top_k=self.entity_vector_top_k,
                        threshold=self.entity_vector_threshold,
                    ):
                        node = f"E:{match_norm}"
                        if self._graph.has_node(node):
                            start_nodes.add(node)
            unmatched_entities = sorted(
                norm for norm in extracted_norms if f"E:{norm}" not in start_nodes
            )

            if debug_info is not None:
                debug_info["extracted_entities"] = sorted(extracted_norms)
                debug_info["start_nodes"] = sorted(start_nodes)
                debug_info["unmatched_entities"] = unmatched_entities

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
                    debug_info=debug_info,
                )

            rel_edges: List[Tuple[str, str, str]] = []
            expanded_nodes = self._expand_relational_entities(start_nodes, rel_edges=rel_edges)
            if debug_info is not None:
                debug_info["expanded_nodes"] = sorted(expanded_nodes)
                debug_info["rel_edges"] = [
                    {"source": source, "target": target, "label": label}
                    for source, target, label in (rel_edges or [])
                ]
            chunk_scores: Dict[str, float] = {}
            chunk_start_support: Dict[str, Set[str]] = {}
            chunk_to_entities: Dict[str, Set[str]] = {}
            for entity_node in expanded_nodes:
                role_weight = (
                    self.start_entity_weight
                    if entity_node in start_nodes
                    else self.expanded_entity_weight
                )
                weighted_vote = role_weight * self._entity_df_weight(entity_node)
                for _, target, data in self._graph.out_edges(entity_node, data=True):
                    if data.get("kind") != "appears_in":
                        continue
                    chunk_id = self._chunk_id_from_node(target)
                    if not chunk_id:
                        continue
                    chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weighted_vote
                    chunk_to_entities.setdefault(chunk_id, set()).add(entity_node)
                    if entity_node in start_nodes:
                        entity_norm = entity_node[2:] if entity_node.startswith("E:") else entity_node
                        chunk_start_support.setdefault(chunk_id, set()).add(entity_norm)

            if not chunk_scores:
                timings["retrieval"] = (time.perf_counter() - r0) * 1000.0
                timings["total"] = (time.perf_counter() - t0) * 1000.0
                return StrategyResult(
                    answer="",
                    context_scores=[],
                    latency_ms=timings,
                    status="NO_CONTEXT",
                    debug_info=debug_info,
                )

            final_scores: Dict[str, float] = {}
            for chunk_id, base_score in chunk_scores.items():
                support_count = len(chunk_start_support.get(chunk_id, set()))
                final_scores[chunk_id] = base_score + (self.synergy_gamma * support_count)

            ranked_chunks = sorted(
                final_scores.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if debug_info is not None:
                debug_info["chunk_trace"] = [
                    {
                        "chunk_id": chunk_id,
                        "score": float(score),
                        "supporting_entities": sorted(chunk_start_support.get(chunk_id, set())),
                    }
                    for chunk_id, score in ranked_chunks
                ]
            entity_paths = self._get_paths_to_entities(start_nodes, rel_edges)
            pairs: List[Tuple[str, float]] = []
            for chunk_id, score in ranked_chunks:
                text = self.corpus.get_text(chunk_id)
                if not text:
                    continue
                formatted = self._format_chunk_with_paths(
                    chunk_id, text, chunk_to_entities, entity_paths
                )
                pairs.append((formatted, float(score)))
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
                    debug_info=debug_info,
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
                debug_info=debug_info,
            )
        except Exception as exc:
            timings["total"] = (time.perf_counter() - t0) * 1000.0
            if debug_info is not None:
                debug_info["error_stage"] = stage
            return StrategyResult(
                answer="",
                context_scores=[],
                latency_ms=timings,
                status="ERROR",
                error=f"GraphStrategy failed during {stage}: {type(exc).__name__}: {exc}",
                debug_info=debug_info,
            )
