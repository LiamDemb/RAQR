.PHONY: install setup-models lock test ingest build-corpus build-corpus-simple eval-strategies mock-oracle debug-graph debug-ie submit-ie-batch collect-ie-batch build-graph-from-corpus run-strategy-batch submit-strategy-batch collect-strategy-batch build-router-dataset build-router-dataset-undersample build-router-dataset-disagreement train-classifier validate-classifier train-all-classifiers validate-all-classifiers figures figure-oracle-dist figure-ablation-f1 figure-confusion figure-e2e figure-regret figure-permutation figures-e2e-preflight

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
		$(if $(NO_WAIT),--no-wait) \
		$(if $(ONLY_QUESTION_IDS_FROM),--only-question-ids-from $(ONLY_QUESTION_IDS_FROM))

# After build-router-dataset: backfill Dense+Graph batch answers for test IDs only (requires API).
figures-e2e-preflight:
	poetry run python scripts/oracle/run_strategy_batch.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--output-dir "$(OUTPUT_DIR)" \
		--only-question-ids-from "$(ROUTER_TEST_PATH)" \
		$(if $(NO_WAIT),--no-wait)

# Answer batch: submit only (standalone; run-strategy-batch does submit+wait+collect internally)
submit-answer-batch:
	poetry run python scripts/oracle/submit_answer_batch.py \
		--benchmark "$(BENCHMARK_PATH)" \
		--output-dir "$(OUTPUT_DIR)" \
		$(if $(LIMIT),--limit $(LIMIT))

# Answer batch: collect results into oracle_raw_scores.jsonl (merge; no new API submit).
# Use after batches complete, or alone if submit already ran but collect was skipped.
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

# Same as build-router-dataset but balances train only to 50/50 Dense vs Graph (dev/test natural split).
build-router-dataset-undersample:
	poetry run python scripts/oracle/build_router_dataset.py \
		--input "$(OUTPUT_DIR)/oracle_raw_scores.jsonl" \
		--output-dir "data/training" \
		--undersample \
		$(if $(PROBE_TOP_K),--probe-top-k $(PROBE_TOP_K)) \
		$(if $(DELTA),--delta $(DELTA))

# Train set only: rows where exactly one strategy is \"correct\" (F1 >= threshold). Dev/test unchanged.
# Override threshold: make build-router-dataset-disagreement DISAGREEMENT_THRESHOLD=0.4
DISAGREEMENT_THRESHOLD ?=
build-router-dataset-disagreement:
	poetry run python scripts/oracle/build_router_dataset.py \
		--input "$(OUTPUT_DIR)/oracle_raw_scores.jsonl" \
		--output-dir "data/training" \
		--train-disagreement \
		$(if $(DISAGREEMENT_THRESHOLD),--disagreement-threshold $(DISAGREEMENT_THRESHOLD)) \
		$(if $(PROBE_TOP_K),--probe-top-k $(PROBE_TOP_K)) \
		$(if $(DELTA),--delta $(DELTA))

# Phase 4: Train classifier router. SIGNALS=q_emb,q_feat,probe (comma-separated)
SIGNALS ?= q_emb,q_feat,probe
EPOCHS ?= 100
RESULTS_DIR ?= results

# Dissertation figures (see scripts/evaluation/). Oracle distribution needs two router builds:
# ROUTER_DIR_BEFORE = e.g. unbalanced labeled_* trees; ROUTER_DIR_AFTER = undersampled (default data/training).
FIGURES_DIR ?= figures
MODEL_DIR ?= models
ROUTER_TEST_PATH ?= data/training/labeled_test.jsonl
ROUTER_DIR_BEFORE ?= data/training_unbalanced
ROUTER_DIR_AFTER ?= data/training

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

# --- Evaluation figures (PDFs under FIGURES_DIR; single aggregate + per-figure targets) ---
figures: figure-oracle-dist figure-ablation-f1 figure-confusion figure-e2e figure-regret figure-permutation

figure-oracle-dist:
	poetry run python scripts/evaluation/plot_oracle_label_distribution.py \
		--before-dir "$(ROUTER_DIR_BEFORE)" \
		--after-dir "$(ROUTER_DIR_AFTER)" \
		--output "$(FIGURES_DIR)/oracle_label_distribution.pdf"

figure-ablation-f1:
	poetry run python scripts/evaluation/plot_ablation_macro_f1.py \
		--split-path "$(ROUTER_TEST_PATH)" \
		--model-dir "$(MODEL_DIR)" \
		--output "$(FIGURES_DIR)/ablation_macro_f1.pdf"

figure-confusion:
	poetry run python scripts/evaluation/plot_confusion_grids.py \
		--split-path "$(ROUTER_TEST_PATH)" \
		--model-dir "$(MODEL_DIR)" \
		--output-four "$(FIGURES_DIR)/confusion_matrix_grid_4.pdf" \
		--output-seven "$(FIGURES_DIR)/confusion_matrix_grid_7.pdf"

figure-e2e:
	poetry run python scripts/evaluation/plot_e2e_system_f1.py \
		--labeled-test "$(ROUTER_TEST_PATH)" \
		--model-dir "$(MODEL_DIR)" \
		--output "$(FIGURES_DIR)/e2e_system_f1.pdf" \
		--output-json "$(RESULTS_DIR)/e2e_test_f1.json"

figure-regret:
	poetry run python scripts/evaluation/plot_routing_regret_matrix.py \
		--labeled-test "$(ROUTER_TEST_PATH)" \
		--model-dir "$(MODEL_DIR)" \
		--output "$(FIGURES_DIR)/routing_regret_severity.pdf"

figure-permutation:
	poetry run python scripts/evaluation/plot_permutation_importance.py \
		--split-path "$(ROUTER_TEST_PATH)" \
		--model-dir "$(MODEL_DIR)" \
		--output "$(FIGURES_DIR)/router_permutation_importance.pdf"