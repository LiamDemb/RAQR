.PHONY: install setup-models lock test ingest build-corpus build-corpus-simple eval-strategies mock-oracle debug-graph debug-ie submit-ie-batch collect-ie-batch build-graph-from-corpus run-strategy-batch submit-strategy-batch collect-strategy-batch build-router-dataset train-classifier validate-classifier train-all-classifiers validate-all-classifiers

-include .env

NQ_PATH ?= data/raw/nq_100.jsonl
2WIKI_PATH ?= data/raw/2wikimultihop_100.jsonl
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
		--2wiki "$(2WIKI_PATH)" \
		--output-dir "$(OUTPUT_DIR)"

# Build corpus: ingestion + chunking + LLM information extraction (entities+triples) via Batch API.
build-corpus:
	poetry run python scripts/01_build_corpus.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--nq "$(NQ_PATH)" \
		--2wiki "$(2WIKI_PATH)" \
		--output-dir "$(OUTPUT_DIR)"

build-corpus-simple:
	HF_HOME="$(HF_HOME)" poetry run python scripts/01_build_corpus.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--nq "$(NQ_PATH)" \
		--2wiki "$(2WIKI_PATH)" \
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
		--corpus "$(OUTPUT_DIR)/corpus.jsonl" \
		--graph-out "$(OUTPUT_DIR)/graph.pkl"

# Strategy batch (Phase 3): submit -> wait -> collect. Dense + Graph answers per question.
# Output: oracle_raw_scores.jsonl with pred_dense, pred_graph paired per question.
# Use LIMIT=N for quick runs, NO_WAIT=1 to exit after submit.
run-strategy-batch:
	poetry run python scripts/oracle/run_strategy_batch.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--output-dir "$(OUTPUT_DIR)" \
		$(if $(LIMIT),--limit $(LIMIT)) \
		$(if $(NO_WAIT),--no-wait)

# Answer batch: submit only (standalone; run-strategy-batch does submit+wait+collect internally)
submit-answer-batch:
	poetry run python scripts/oracle/submit_answer_batch.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--output-dir "$(OUTPUT_DIR)" \
		$(if $(LIMIT),--limit $(LIMIT))

# Answer batch: collect results into oracle_raw_scores.jsonl (after batches complete)
collect-answer-batch:
	poetry run python scripts/oracle/collect_answer_batch.py \
		--state "$(or $(STATE),$(OUTPUT_DIR)/batch_state_strategy.json)" \
		--output-dir "$(OUTPUT_DIR)"

# Build router dataset from oracle_raw_scores.jsonl (after collect-answer-batch)
build-router-dataset:
	poetry run python scripts/oracle/build_router_dataset.py \
		--input "$(OUTPUT_DIR)/oracle_raw_scores.jsonl" \
		--output-dir "data/training" \
		$(if $(PROBE_TOP_K),--probe-top-k $(PROBE_TOP_K)) \
		$(if $(DELTA),--delta $(DELTA))

build-router-dataset-undersample:
	poetry run python scripts/oracle/build_router_dataset.py \
		--input "$(OUTPUT_DIR)/oracle_raw_scores.jsonl" \
		--output-dir "data/training" \
		--undersample \
		$(if $(PROBE_TOP_K),--probe-top-k $(PROBE_TOP_K)) \
		$(if $(DELTA),--delta $(DELTA))
		

# Phase 4: Train classifier router. SIGNALS=q_emb,q_feat,probe (comma-separated)
SIGNALS ?= q_emb,q_feat,probe
EPOCHS ?= 100
RESULTS_DIR ?= results

train-classifier:
	poetry run python scripts/04a_train_classifier.py \
		--signals "$(SIGNALS)" \
		--epochs $(EPOCHS) \
		$(if $(LR),--lr $(LR)) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE))

# Phase 4: Validate classifier against gate metrics
validate-classifier:
	poetry run python scripts/04b_validate_classifier.py \
		--signals "$(SIGNALS)" \
		--results-dir "$(RESULTS_DIR)"

# Phase 4: Train ALL ablation classifiers sequentially
ABLATIONS = q_emb q_feat probe q_emb,q_feat q_emb,probe q_feat,probe q_emb,q_feat,probe

train-all-classifiers:
	@echo "═══════════════════════════════════════════════════"
	@echo "  Training all 7 ablation classifiers"
	@echo "═══════════════════════════════════════════════════"
	@for sig in $(ABLATIONS); do \
		echo ""; \
		echo "─── Training: $$sig ───"; \
		poetry run python scripts/04a_train_classifier.py \
			--signals "$$sig" \
			--epochs $(EPOCHS) \
			--hidden-dim 64 \
			--weight-decay 0.05 \
			--lr 3e-4 \
			$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE)); \
	done
	@echo ""
	@echo "═══════════════════════════════════════════════════"
	@echo "  All classifiers trained.  Run: make validate-all-classifiers"
	@echo "═══════════════════════════════════════════════════"

# Phase 4: Validate ALL ablation classifiers and produce summary report
validate-all-classifiers:
	poetry run python scripts/04b_validate_classifier.py --all \
		--results-dir "$(RESULTS_DIR)"