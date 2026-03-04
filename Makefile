.PHONY: install setup-models lock test ingest build-corpus build-corpus-llm build-corpus-llm-batch build-corpus-simple eval-strategies mock-oracle debug-graph llm-triples-poc submit-llm-triples-batch collect-llm-triples-batch collect-and-build-graph build-graph-from-corpus

-include .env

NQ_PATH ?= data/raw/nq_300.jsonl
COMPLEXTEMPQA_PATH ?= data/raw/complex_tempqa_300.jsonl
WIKIWHY_PATH ?= data/raw/wikiwhy_300.jsonl
HOTPOTQA_PATH ?= data/raw/hotpotqa_300.jsonl
BENCHMARK_PATH ?= data/processed/benchmark.jsonl
OUTPUT_DIR ?= data/processed
HF_HOME ?= $(OUTPUT_DIR)/hf_cache

install:
	poetry install

setup-models:
	HF_HOME="$(HF_HOME)" poetry run python scripts/00_setup_models.py

lock:
	poetry lock

test:
	poetry run pytest

# Dataset paths: set in .env or override on CLI. Omit a path (e.g. NQ_PATH=) to exclude that dataset.
ingest:
	poetry run python scripts/01_ingest_data.py \
		--nq "$(NQ_PATH)" \
		--complextempqa "$(COMPLEXTEMPQA_PATH)" \
		--wikiwhy "$(WIKIWHY_PATH)" \
		--hotpotqa "$(HOTPOTQA_PATH)" \
		--output-dir "$(OUTPUT_DIR)"

# Relation extraction: RELATION_EXTRACTOR=rebel (default), llm (sync), or llm-batch (no extraction, auto-submit)
build-corpus:
	poetry run python scripts/01_build_corpus.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--nq "$(NQ_PATH)" \
		--complextempqa "$(COMPLEXTEMPQA_PATH)" \
		--wikiwhy "$(WIKIWHY_PATH)" \
		--hotpotqa "$(HOTPOTQA_PATH)" \
		--output-dir "$(OUTPUT_DIR)"

# Build corpus using LLM triple extractor (sync); same args as build-corpus
build-corpus-llm:
	RELATION_EXTRACTOR=llm $(MAKE) build-corpus

# Build corpus without extraction, auto-submit batch; run collect-and-build-graph when batch completes
build-corpus-llm-batch:
	RELATION_EXTRACTOR=llm-batch $(MAKE) build-corpus

build-corpus-simple:
	HF_HOME="$(HF_HOME)" poetry run python scripts/01_build_corpus.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--nq "$(NQ_PATH)" \
		--complextempqa "$(COMPLEXTEMPQA_PATH)" \
		--wikiwhy "$(WIKIWHY_PATH)" \
		--hotpotqa "$(HOTPOTQA_PATH)" \
		--output-dir "$(OUTPUT_DIR)" \
		--max-pages 3 \
		--max-hops 1 \
		--max-list-pages 0 \
		--max-country-pages 0 \
		--re-batch-size 1 \
		--re-max-input-tokens 512 \
		--re-max-new-tokens 64

eval-strategies:
	poetry run python scripts/dev/evaluate_strategies.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--output-dir "$(OUTPUT_DIR)" \
		--use-llm-judge

mock-oracle:
	poetry run python scripts/dev/mock_oracle_eval.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--output-dir "$(OUTPUT_DIR)"

# Debug GraphStrategy: pass your query via QUERY=... (e.g. make debug-graph QUERY="What happened in 2017?")
debug-graph:
	poetry run python scripts/strategies/debug_graph.py "$(QUERY)" --output-dir "$(OUTPUT_DIR)"

# LLM triple extractor POC: pass text via TEXT=... (e.g. make llm-triples TEXT="Eva Busch was a cabaret artist.")
llm-triples-poc:
	poetry run python scripts/dev/run_llm_triple_extractor.py "$(TEXT)" --output-dir "$(OUTPUT_DIR)"

# Batch API: submit LLM triple extraction job (50% cheaper). Uses corpus from OUTPUT_DIR.
submit-llm-triples-batch:
	poetry run python scripts/corpus/submit_llm_triple_batch.py \
		--corpus "$(OUTPUT_DIR)/corpus.jsonl" \
		--output-dir "$(OUTPUT_DIR)" \
		$(if $(LIMIT),--limit $(LIMIT))

# Batch API: collect results and merge into corpus_llm.jsonl. Pass BATCH_ID=... or STATE=.../batch_state.json
collect-llm-triples-batch:
	poetry run python scripts/corpus/collect_llm_triple_batch.py \
		--corpus "$(OUTPUT_DIR)/corpus.jsonl" \
		--output "$(OUTPUT_DIR)/corpus_llm.jsonl" \
		--output-dir "$(OUTPUT_DIR)" \
		$(if $(BATCH_ID),--batch-id $(BATCH_ID)) \
		$(if $(STATE),--state $(STATE))

# Rebuild graph from a corpus file. CORPUS=path GRAPH=path (e.g. corpus_llm.jsonl -> graph.pkl)
build-graph-from-corpus:
	poetry run python scripts/corpus/build_graph_from_corpus.py \
		--corpus "$(CORPUS)" \
		--graph-out "$(GRAPH)"

# Collect LLM batch results and rebuild graph in one step. Pass BATCH_ID=... or STATE=... if needed.
collect-and-build-graph:
	$(MAKE) collect-llm-triples-batch $(if $(BATCH_ID),BATCH_ID=$(BATCH_ID)) $(if $(STATE),STATE=$(STATE))
	$(MAKE) build-graph-from-corpus CORPUS="$(OUTPUT_DIR)/corpus_llm.jsonl" GRAPH="$(OUTPUT_DIR)/graph.pkl"