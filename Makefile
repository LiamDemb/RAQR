.PHONY: install setup-models lock test ingest build-corpus build-corpus-simple eval-strategies mock-oracle debug-graph debug-ie submit-ie-batch collect-ie-batch build-graph-from-corpus

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

# Build corpus: ingestion + chunking + LLM information extraction (entities+triples) via Batch API.
build-corpus:
	poetry run python scripts/01_build_corpus.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--nq "$(NQ_PATH)" \
		--complextempqa "$(COMPLEXTEMPQA_PATH)" \
		--wikiwhy "$(WIKIWHY_PATH)" \
		--hotpotqa "$(HOTPOTQA_PATH)" \
		--output-dir "$(OUTPUT_DIR)"

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
		--max-country-pages 0

eval-strategies:
	poetry run python scripts/dev/evaluate_strategies.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--output-dir "$(OUTPUT_DIR)" \
		--use-llm-judge

mock-oracle:
	poetry run python scripts/dev/mock_oracle_eval.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--output-dir "$(OUTPUT_DIR)" \
		$(if $(PRINT_DISAGREEMENTS),--print-disagreements)

# Same as mock-oracle but prints questions where Graph and Dense disagree (Graph 0/Dense 1, etc.)
mock-oracle-disagreements:
	$(MAKE) mock-oracle PRINT_DISAGREEMENTS=1

# Debug GraphStrategy: pass your query via QUERY=... (e.g. make debug-graph QUERY="What happened in 2017?")
debug-graph:
	poetry run python scripts/strategies/debug_graph.py "$(QUERY)" --output-dir "$(OUTPUT_DIR)"

# Dev: run LLM IE extraction on a single .txt file (e.g. make debug-ie TEXT_FILE=temp/sample.txt)
debug-ie:
	poetry run python scripts/dev/run_llm_onepass.py --text-file "$(TEXT_FILE)" --output-dir "$(OUTPUT_DIR)"

# IE batch: submit job (standalone; build-corpus runs submit+wait+collect internally)
submit-ie-batch:
	poetry run python scripts/corpus/submit_llm_ie_batch.py \
		--corpus "$(OUTPUT_DIR)/corpus.jsonl" \
		--output-dir "$(OUTPUT_DIR)" \
		$(if $(LIMIT),--limit $(LIMIT))

# IE batch: collect results and merge into corpus. Pass STATE=.../batch_state_ie.json if needed
collect-ie-batch:
	poetry run python scripts/corpus/collect_llm_ie_batch.py \
		--state "$(or $(STATE),$(OUTPUT_DIR)/batch_state_ie.json)" \
		--corpus "$(OUTPUT_DIR)/corpus.jsonl" \
		--output-dir "$(OUTPUT_DIR)"

# Rebuild graph from a corpus file. CORPUS=path GRAPH=path (e.g. corpus.jsonl -> graph.pkl)
build-graph-from-corpus:
	poetry run python scripts/corpus/build_graph_from_corpus.py \
		--corpus "$(CORPUS)" \
		--graph-out "$(GRAPH)"