"""Shared helpers for dev scripts (evaluate_strategies, mock_oracle_eval)."""

from __future__ import annotations

import ast
import os


BASE_PROMPT = (
    "You are a strict factual answering system. Answer the question based ONLY on the provided context."
    "CRITICAL INSTRUCTIONS:"
    "- Be as concise as possible."
    "- Do NOT repeat the question."
    "- Do NOT use conversational filler like 'Based on the context...' or 'The answer is...'."
    "- If the context does not contain the answer, reply with exactly the word: 'INSUFFICIENT_CONTEXT'."
)


def normalize_gold_answers(raw: list) -> list[str]:
    """Parse gold_answers into a flat list of normalized strings."""
    answers = []
    for item in raw:
        s = str(item).strip()
        if not s:
            continue
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                answers.extend(str(x).strip() for x in parsed if str(x).strip())
            else:
                answers.append(s)
        except (ValueError, SyntaxError):
            answers.append(s)
    return answers


def build_dense_strategy(output_dir: str):
    """Build DenseStrategy with corpus, index, embedder, generator."""
    from raqr.embedder import SentenceTransformersEmbedder
    from raqr.generator import SimpleLLMGenerator
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
            base_prompt=BASE_PROMPT,
        ),
        corpus=JsonCorpusLoader(jsonl_path=corpus_path),
        top_k=int(os.getenv("DENSE_TOP_K", "10")),
    )


def build_graph_strategy(output_dir: str):
    """Build GraphStrategy with corpus, graph, generator, entity extractor."""
    from raqr.entity_alias_resolver import EntityAliasResolver
    from raqr.generator import SimpleLLMGenerator
    from raqr.graph_store import NetworkXGraphStore
    from raqr.loaders import JsonCorpusLoader
    from raqr.strategies.graph import GraphStrategy, SpacyQueryEntityExtractor

    corpus_path = f"{output_dir}/corpus.jsonl"
    graph_path = f"{output_dir}/graph.pkl"
    lexicon_path = f"{output_dir}/entity_lexicon.parquet"
    alias_map_path = f"{output_dir}/alias_map.json"

    if not os.path.exists(alias_map_path):
        raise FileNotFoundError(
            f"Required artifact missing: {alias_map_path}. Rebuild corpus with Phase 1 pipeline."
        )

    alias_resolver = EntityAliasResolver.from_artifacts(output_dir=output_dir)
    entity_df_by_norm = EntityAliasResolver.load_df_map_from_lexicon(lexicon_path=lexicon_path)
    return GraphStrategy(
        graph_store=NetworkXGraphStore(graph_path=graph_path),
        corpus=JsonCorpusLoader(jsonl_path=corpus_path),
        generator=SimpleLLMGenerator(
            model_id=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            base_prompt=BASE_PROMPT,
        ),
        entity_extractor=SpacyQueryEntityExtractor(alias_resolver=alias_resolver),
        top_k=int(os.getenv("GRAPH_TOP_K", "10")),
        max_hops=int(os.getenv("GRAPH_MAX_HOPS", "1")),
        entity_df_by_norm=entity_df_by_norm,
    )
