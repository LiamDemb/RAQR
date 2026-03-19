"""Factory for building fully instantiated strategies for the production pipeline."""

from __future__ import annotations

import os

from raqr.embedder import SentenceTransformersEmbedder
from raqr.generator import SimpleLLMGenerator
from raqr.prompts import get_generator_prompt
from raqr.scoring_config import DEFAULT_SCORING_CONFIG


def build_dense_strategy(output_dir: str):
    """Build DenseStrategy with corpus, index, embedder, generator for production pipeline."""
    from raqr.index_store import FaissIndexStore
    from raqr.loaders import JsonCorpusLoader, VectorMetaMapper
    from raqr.strategies.dense import DenseStrategy

    corpus_path = f"{output_dir}/corpus.jsonl"
    index_path = f"{output_dir}/vector_index.faiss"
    meta_path = f"{output_dir}/vector_meta.parquet"
    
    return DenseStrategy(
        index_store=FaissIndexStore(index_path=index_path),
        meta=VectorMetaMapper(parquet_path=meta_path),
        embedder=SentenceTransformersEmbedder(model_name="all-MiniLM-L6-v2"),
        generator=SimpleLLMGenerator(
            model_id=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_prompt=get_generator_prompt(),
        ),
        corpus=JsonCorpusLoader(jsonl_path=corpus_path),
        top_k=int(os.getenv("DENSE_TOP_K", "10")),
    )


def build_graph_strategy(output_dir: str):
    """Build GraphStrategy with corpus, graph, generator for production pipeline."""
    from raqr.entity_alias_resolver import EntityAliasResolver
    from raqr.entity_index_store import EntityIndexStore
    from raqr.graph_store import NetworkXGraphStore
    from raqr.loaders import JsonCorpusLoader
    from raqr.strategies.graph import GraphStrategy, _default_query_entity_extractor

    corpus_path = f"{output_dir}/corpus.jsonl"
    graph_path = f"{output_dir}/graph.pkl"
    alias_map_path = f"{output_dir}/alias_map.json"

    if not os.path.exists(alias_map_path):
        raise FileNotFoundError(
            f"Required artifact missing: {alias_map_path}. Rebuild corpus with Phase 1 pipeline."
        )

    alias_resolver = EntityAliasResolver.from_artifacts(output_dir=output_dir)

    entity_index_store = None
    if os.path.exists(f"{output_dir}/entity_index.faiss") and os.path.exists(
        f"{output_dir}/entity_index_meta.parquet"
    ):
        entity_index_store = EntityIndexStore(
            f"{output_dir}/entity_index.faiss",
            f"{output_dir}/entity_index_meta.parquet",
        )
        
    return GraphStrategy(
        graph_store=NetworkXGraphStore(graph_path=graph_path),
        corpus=JsonCorpusLoader(jsonl_path=corpus_path),
        generator=SimpleLLMGenerator(
            model_id=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_prompt=get_generator_prompt(),
        ),
        entity_extractor=_default_query_entity_extractor(alias_resolver),
        top_k=int(os.getenv("GRAPH_TOP_K", "10")),
        max_hops=int(os.getenv("GRAPH_MAX_HOPS", "1")),
        entity_index_store=entity_index_store,
        embedder=SentenceTransformersEmbedder(model_name="all-MiniLM-L6-v2"),
        scoring_config=DEFAULT_SCORING_CONFIG,
    )
