from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

from dotenv import load_dotenv
from raqr.data.build_faiss import build_faiss_index
from raqr.data.build_graph import build_graph
from raqr.data.alias_map import build_alias_map_from_redirects
from raqr.data.canonical_clean import clean_html_to_structured_doc
from raqr.data.chunking import chunk_blocks
from raqr.data.corpus_acquisition import Budgets, ingest_complextempqa, ingest_nq, ingest_wikiwhy
from raqr.data.corpus_schemas import CorpusChunk
from raqr.data.docstore import DocStore
from raqr.data.enrich_entities import extract_entities_spacy, load_spacy
from raqr.data.enrich_relations import extract_relations_rebel, load_rebel
from raqr.data.enrich_years import aggregate_year_fields, extract_years
from raqr.data.entity_lexicon import build_entity_lexicon
from raqr.data.quality_gates import run_quality_gates
from raqr.data.schemas import sha256_text
from raqr.data.wikipedia_client import WikipediaClient
from raqr.data.wikidata_client import WikidataClient


logger = logging.getLogger(__name__)


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_benchmark(path: str) -> Dict[str, Dict[str, dict]]:
    by_source: Dict[str, Dict[str, dict]] = {"nq": {}, "complextempqa": {}, "wikiwhy": {}}
    for item in _iter_jsonl(path):
        source = item["dataset_source"]
        by_source.setdefault(source, {})[item["question_id"]] = item
    return by_source


def _question_text_from_row(row: dict, source: str) -> str | None:
    if source == "nq":
        question_block = row.get("question")
        return (
            row.get("question_text")
            or row.get("questionText")
            or (question_block.get("text") if isinstance(question_block, dict) else None)
            or (question_block if isinstance(question_block, str) else None)
        )
    return row.get("question") or row.get("query")


def _build_samples(
    benchmark_by_source: Dict[str, Dict[str, dict]],
    nq_path: str,
    complextempqa_path: str,
    wikiwhy_path: str,
) -> List[dict]:
    samples: List[dict] = []

    for row in _iter_jsonl(nq_path):
        question = _question_text_from_row(row, "nq")
        if not question:
            continue
        qid = sha256_text(question)
        bench = benchmark_by_source.get("nq", {}).get(qid)
        if not bench:
            continue
        samples.append(
            {
                "source": "nq",
                "question_id": qid,
                "question": question,
                "gold_answers": bench.get("gold_answers", []),
                "document": row.get("document"),
                "document_html": row.get("document_html"),
                "document_title": row.get("document_title"),
                "title": row.get("title"),
            }
        )

    for row in _iter_jsonl(complextempqa_path):
        question = _question_text_from_row(row, "complextempqa")
        if not question:
            continue
        qid = sha256_text(question)
        bench = benchmark_by_source.get("complextempqa", {}).get(qid)
        if not bench:
            continue
        row_sample = {
            "source": "complextempqa",
            "question_id": qid,
            "question": question,
            "gold_answers": bench.get("gold_answers", []),
            "question_entity": row.get("question_entity"),
            "answer_entity": row.get("answer_entity"),
            "question_country_entity": row.get("question_country_entity"),
        }
        samples.append(row_sample)

    for row in _iter_jsonl(wikiwhy_path):
        question = _question_text_from_row(row, "wikiwhy")
        if not question:
            continue
        qid = sha256_text(question)
        bench = benchmark_by_source.get("wikiwhy", {}).get(qid)
        if not bench:
            continue
        samples.append(
            {
                "source": "wikiwhy",
                "question_id": qid,
                "question": question,
                "gold_answers": bench.get("gold_answers", []),
                "title": row.get("title"),
            }
        )

    return samples


def _write_jsonl(path: Path, items: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build unified corpus + indexes.")
    parser.add_argument("--benchmark", default=os.getenv("BENCHMARK_PATH"))
    parser.add_argument("--nq", default=os.getenv("NQ_PATH"))
    parser.add_argument("--complextempqa", default=os.getenv("COMPLEXTEMPQA_PATH"))
    parser.add_argument("--wikiwhy", default=os.getenv("WIKIWHY_PATH"))
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "data/processed"))
    parser.add_argument(
        "--docstore",
        default=os.getenv("DOCSTORE_PATH", "data/processed/docstore.sqlite"),
    )
    parser.add_argument("--model-name", default=os.getenv("MODEL_NAME", "all-MiniLM-L6-v2"))
    parser.add_argument(
        "--re-model-name",
        default=os.getenv("RE_MODEL_NAME", "Babelscape/rebel-large"),
    )
    parser.add_argument(
        "--re-batch-size",
        type=int,
        default=int(os.getenv("RE_BATCH_SIZE", "4")),
    )
    parser.add_argument(
        "--re-max-input-chars",
        type=int,
        default=int(os.getenv("RE_MAX_INPUT_CHARS", "2000")),
    )
    parser.add_argument(
        "--re-max-new-tokens",
        type=int,
        default=int(os.getenv("RE_MAX_NEW_TOKENS", "128")),
    )
    parser.add_argument("--max-pages", type=int, default=int(os.getenv("MAX_PAGES", "12")))
    parser.add_argument("--max-hops", type=int, default=int(os.getenv("MAX_HOPS", "2")))
    parser.add_argument(
        "--max-list-pages", type=int, default=int(os.getenv("MAX_LIST_PAGES", "2"))
    )
    parser.add_argument(
        "--max-country-pages",
        type=int,
        default=int(os.getenv("MAX_COUNTRY_PAGES", "1")),
    )
    args = parser.parse_args()

    missing = [
        name
        for name, value in [
            ("BENCHMARK_PATH", args.benchmark),
            ("NQ_PATH", args.nq),
            ("COMPLEXTEMPQA_PATH", args.complextempqa),
            ("WIKIWHY_PATH", args.wikiwhy),
        ]
        if not value
    ]
    if missing:
        raise ValueError(
            "Missing dataset paths. Provide CLI args or set: " + ", ".join(missing)
        )

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    benchmark_by_source = _load_benchmark(args.benchmark)
    samples = _build_samples(
        benchmark_by_source, args.nq, args.complextempqa, args.wikiwhy
    )

    budgets = Budgets(
        max_pages_per_question=args.max_pages,
        max_hops=args.max_hops,
        max_list_pages=args.max_list_pages,
        max_country_pages=args.max_country_pages,
    )
    docstore = DocStore(args.docstore)
    wiki = WikipediaClient()
    wikidata = WikidataClient()

    all_docs = {}
    for sample in samples:
        if sample["source"] == "complextempqa":
            docs = ingest_complextempqa(sample, budgets, docstore, wiki, wikidata)
        elif sample["source"] == "wikiwhy":
            docs = ingest_wikiwhy(sample, budgets, docstore, wiki)
        else:
            docs = ingest_nq(sample, budgets, docstore, wiki)
        for doc in docs:
            all_docs[doc.doc_key] = doc

    nlp = load_spacy()
    alias_map = build_alias_map_from_redirects(
        titles=[doc.title for doc in all_docs.values() if doc.title],
        wiki=wiki,
    )
    rebel_tokenizer, rebel_model, rebel_device = load_rebel(args.re_model_name)
    chunks: List[dict] = []
    chunk_texts: List[str] = []

    for doc in all_docs.values():
        if not doc.html:
            continue
        structured = clean_html_to_structured_doc(
            html=doc.html,
            doc_id=doc.doc_key,
            title=doc.title,
            url=doc.url,
            anchors=doc.anchors,
            source=doc.source,
            dataset_origin=doc.dataset_origin,
            page_id=doc.page_id,
            revision_id=doc.revision_id,
        )
        for idx, piece in enumerate(chunk_blocks(structured.blocks)):
            years = extract_years(piece.text)
            year_fields = aggregate_year_fields(years, piece.text, piece.token_count)
            entities = extract_entities_spacy(piece.text, nlp, alias_map)
            metadata = {
                "dataset_origin": structured.dataset_origin,
                "page_id": structured.page_id,
                "revision_id": structured.revision_id,
                "years": year_fields["years"],
                "year_min": year_fields["year_min"],
                "year_max": year_fields["year_max"],
                "temporal_density": year_fields["temporal_density"],
                "entities": entities,
                "anchors": structured.anchors,
            }
            chunk = CorpusChunk(
                chunk_id=sha256_text(f"{doc.doc_key}:{idx}:{piece.text}"),
                doc_id=doc.doc_key,
                source=doc.source,
                title=doc.title,
                url=doc.url,
                text=piece.text,
                section_path=piece.section_path,
                char_span_in_doc=piece.char_span_in_doc,
                metadata=metadata,
            )
            chunk_json = chunk.to_json()
            chunks.append(chunk_json)
            chunk_texts.append(piece.text)

    relations_by_chunk = extract_relations_rebel(
        chunk_texts,
        rebel_tokenizer,
        rebel_model,
        rebel_device,
        alias_map=alias_map,
        batch_size=args.re_batch_size,
        max_input_chars=args.re_max_input_chars,
        max_new_tokens=args.re_max_new_tokens,
    )
    for idx, rels in enumerate(relations_by_chunk):
        chunks[idx].setdefault("metadata", {})["relations"] = rels

    output_dir = Path(args.output_dir)
    corpus_path = output_dir / "corpus.jsonl"
    _write_jsonl(corpus_path, chunks)

    build_faiss_index(
        chunks,
        output_index_path=(output_dir / "vector_index.faiss").as_posix(),
        output_meta_path=(output_dir / "vector_meta.parquet").as_posix(),
        model_name=args.model_name,
    )

    graph = build_graph(chunks)
    graph_path = output_dir / "graph.pkl"
    pd.to_pickle(graph, graph_path)

    lexicon = build_entity_lexicon(chunks)
    lexicon_path = output_dir / "entity_lexicon.parquet"
    lexicon.to_parquet(lexicon_path, index=False)

    run_quality_gates(
        samples,
        chunks,
        output_path=(output_dir / "quality_report.json").as_posix(),
    )

    logger.info("Wrote corpus and artifacts to %s", output_dir)
    docstore.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
