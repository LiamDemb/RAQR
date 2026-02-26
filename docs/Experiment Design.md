# Experimental Design & Protocol

## 1. Experimental Overview
**Objective:** To isolate and measure the impact of specific input signals and model architectures on routing accuracy within a RAG system.
**Method:** "Gold Label" (Oracle) supervision. We generate ground-truth labels by running all strategies on a mixed benchmark, scoring answers via **LLM-as-judge** (semantic correctness), and selecting the winner; then we train/test routers to predict that winner.

## 2. The Dataset (Benchmark)
To ensure the system encounters diverse reasoning requirements, we construct a unified evaluation dataset merging samples from three sources:
*   **Natural Questions (NQ):** Factual/Lookup queries (Target: Dense RAG).
*   **ComplexTempQA:** Temporal/Chronological queries (Target: Temporal RAG).
*   **WikiWhy:** Causal/Explanation queries (Target: GraphRAG).

*Total Target Size: ~750–1,500 balanced samples.*

## 3. The Retrieval Strategies (Class Labels)
The system routes queries to one of three distinct retrieval pipelines. These act as the classification targets.

1.  **Dense RAG:** (Baseline) Vector search (`all-MiniLM-L6-v2`) + Chunk retrieval.
2.  **GraphRAG:** Relation-aware traversal (Subject-Predicate-Object triples) (NetworkX) for multi-hop reasoning.
3.  **Temporal RAG:** Metadata-filtered dense retrieval (vector search + year filter); no separate temporal Knowledge Graph.

### Label Generation Logic (The Oracle)
To generate training data, every question is processed by all 3 strategies. The Oracle produces a single "Gold Label" using a **margin-based simplicity bias** rule.

#### Definitions
Let:

- \(S_{dense}\), \(S_{temp}\), \(S_{graph}\) be the evaluation score for each strategy on that question. We use **LLM-as-judge** (0/1/2 scale: 0 = incorrect, 1 = partial, 2 = correct) rather than token-F1/EM, since answer formats vary and semantic comparison is more reliable.
- Complexity order: **Dense (simplest)** < **Temporal** < **Graph (most complex)**.
- A margin threshold \(\delta \ge 0\).
- A small tolerance \(\varepsilon\) for floating comparisons (e.g., \(10^{-6}\)).

#### Oracle selection rule (margin-based simplicity bias)
1.  **Compute the best-scoring strategy:** \(\text{best} = \arg\max S_*\).
2.  **Decide whether to “pay for complexity”:**
    - If \(\text{best} = \textbf{Dense}\): choose **Dense**.
    - If \(\text{best} = \textbf{Temporal}\): choose **Temporal** only if \(S_{temp} \ge S_{dense} + \delta\); otherwise choose **Dense**.
    - If \(\text{best} = \textbf{Graph}\): choose **Graph** only if \(S_{graph} \ge S_{dense} + \delta\) and (optionally, but used here for cleanliness) \(S_{graph} \ge S_{temp} + \delta\); otherwise fall back to the simplest strategy whose score is within \(\delta\) of \(S_{graph}\).
3.  **Deterministic tie-break (exact ties / within tolerance):**
    - If multiple strategies are effectively tied (within \(\varepsilon\)), break ties deterministically: **Dense > Temporal > Graph**.

This produces labels that encode the deployment goal: **do not spend complexity for tiny gains**.

Per-strategy correctness is computed via **LLM-as-judge** (semantic comparison). We prefer this over token-F1/EM because answer formats vary across strategies and datasets; the margin-based selection rule applies to these judge scores.

#### Choosing \(\delta\) (calibrated, not guessed)
We select \(\delta\) empirically on the **Dev** set:

- Compute \(\Delta = S_{best} - S_{second\_best}\) for each Dev example (using LLM-as-judge scores).
- Inspect the distribution of \(\Delta\), and set \(\delta\) to a stable, defensible value (e.g., the 10th–20th percentile of \(\Delta\)).
- Report an ablation over \(\delta \in \{0.00, 0.02, 0.05, 0.10\}\), including:
  - Label distribution shift as \(\delta\) changes
  - Router routing accuracy and downstream F1
  - Latency trade-offs (secondary objective)

#### Split usage policy (to avoid leakage)
- Use **Train** to fit router parameters.
- Use **Dev** for: selecting \(\delta\), early stopping, and model selection.
- Generate raw per-strategy scores for Train+Dev once; choose \(\delta\) on Dev only; then apply that fixed \(\delta\) to label Train+Dev.
- Keep **Test** untouched until the Phase 5 ablation runs.

## 4. The Input Signals (Features)
Routers consume three feature families, aligned with the two-stage ablation:

### A. Q-Emb (Query Embedding)
*   **DistilBERT [CLS] embedding:** 768-dimensional vector representation of the query (from `DistilBERT-base-uncased`). Used as the semantic representation of the query for routing.

### B. Q-Feat (Query Features, Engineered)
*   **Length / token count:** Query string length and token count (basic complexity proxy).
*   **Entity density:** Number of entities per token (via spaCy NER); higher density may indicate multi-hop or graph-relevant queries.
*   **Complexity keywords:** Presence of trigger terms (e.g., "when", "relationship", "cause", "how does X affect Y") via Regex.
*   **Optional syntax depth:** Parse-tree depth (e.g., spaCy dependency depth) for syntactic complexity; optional for Stage 1.

### C. Probe (Retrieval Feedback, Dynamic)
*Derived from a "Probe Search" (a fast top-10 Dense retrieval using the Dense index).*
*   **Max Score:** The cosine similarity of the top result (proxy for confidence).
*   **Score Skewness:** The statistical skew of the top-k score distribution. (High skew = precise; Low skew = confused/multi-hop).
*   **Semantic dispersion:** Average distance between the query embedding and the retrieved cluster centroid. (Alias: "semantic distance"—we use "semantic dispersion" as the canonical term.)

## 5. The Model Architectures (Routers)
We evaluate three distinct software architectures:

### Type 1: Heuristic Router (Baseline)
*   **Physical Form:** Python script with Regex and `if/else` logic.
*   **Mechanism:** Deterministic decision trees.
*   **Goal:** To test if AI is necessary, or if explicit rules suffice.

### Type 2: Lightweight Classifier (Core Contribution)
*   **Physical Form:** A fine-tuned PyTorch model (DistilBERT base) saved as `.pt`.
*   **Mechanism:** Supervised classification.
    *   *Text Input:* Tokens passed through Transformer layers.
    *   *Signal Input:* Numerical features concatenated to the classification head (MLP).
    *   *Head:* Layer 2: Linear (256 -> 3) (3 Classes).
*   **Goal:** To achieve maximum accuracy with minimum latency/cost.

### Type 3: LLM Router (Upper Bound)
*   **Physical Form:** A prompt template wrapping an API call (GPT-4o-mini / Llama-3).
*   **Mechanism:** In-context learning and reasoning.
*   **Goal:** To establish the performance ceiling.

Keywords and regex features are part of **Q-Feat** (engineered query features); they are not a separate router architecture.

## 6. Two-Stage Ablation

### Stage 1: Signal Ablation (Hold Model Class Fixed)
Stage 1 holds the **architecture fixed** (Classifier) and ablates **input channels** to determine which feature family(ies) best support routing:

| Input Set | Description |
| :--- | :--- |
| **Q-Emb only** | DistilBERT [CLS] embedding (768 dims) only. |
| **Q-Feat only** | Engineered features (length, entity density, complexity keywords, optional syntax depth) only. |
| **Probe only** | Top-10 Dense retrieval signals (max score, skewness, semantic dispersion) only. |
| **Q-Emb + Q-Feat** | Combined query-side signals (no probe). |
| **Q-Emb + Probe** | Embedding + probe stats. |
| **Q-Feat + Probe** | Engineered features + probe stats. |
| **Q-Emb + Q-Feat + Probe** | Full combined input. |

The winning input set (by routing accuracy on Dev) advances to Stage 2.

### Stage 2: Architecture Showdown (Use Winning Input Set)
Stage 2 compares **Heuristic vs Classifier vs LLM** using the winning input set from Stage 1:

| Model Class | Description |
| :--- | :--- |
| **Heuristic** | Deterministic rules (Regex + keyword logic; probe thresholds when Probe is in the input set). |
| **Classifier** | Supervised DistilBERT-based encoder with late fusion (text + numerical features). |
| **LLM** | Prompted Large Language Model with in-context reasoning. |

### Monolithic Baselines (RQ3 Extension)
For RQ3 (impact of routing errors), we also run **monolithic baselines**:
*   **Always-Dense:** Route every query to Dense RAG.
*   **Always-Temporal:** Route every query to Temporal RAG.
*   **Always-Graph:** Route every query to Graph RAG.

These establish performance floors and allow **routing regret** analysis.

### Routing Regret & Severity Taxonomy
*   **Routing regret** is defined as: \(\text{regret} = \text{oracleF1} - \text{selectedF1}\), i.e. the downstream F1 loss when the router selects a strategy different from the Oracle's best.
*   **Severity taxonomy:** We categorize routing errors by impact:
    *   **Forgivable:** Marginal F1 loss; Oracle and selected strategy perform similarly.
    *   **Moderate:** Noticeable degradation; selected strategy underperforms but remains usable.
    *   **Fatal:** Severe failure; selected strategy produces poor or empty answers where Oracle would succeed.

## 7. The Ablation Grid (Testing Matrix)
We perform a 3x3 ablation study. Each cell represents a distinct experiment script.

| Model Class | Input: Question Only | Input: Feedback Only | Input: Combined |
| :--- | :--- | :--- | :--- |
| **1. Heuristic** | **Test H-Q:** Keyword/Regex matching. | **Test H-F:** Thresholds on Skewness/Max Score. | **Test H-C:** Logic tree (Keywords $\rightarrow$ then Stats). |
| **2. Classifier** | **Test C-Q:** Standard DistilBERT (Text classification). | **Test C-F:** Simple MLP (Feed-forward network on stats). | **Test C-C:** DistilBERT + Concatenated Feature Vector. |
| **3. LLM** | **Test L-Q:** Standard Prompt ("Classify this text"). | **Test L-F:** Stats Prompt ("Given these scores..."). | **Test L-C:** Chain-of-Thought ("Analyze text and scores"). |

## 8. Evaluation Metrics
1.  **Routing Accuracy:** % of times the router picked the "Gold Label" strategy.
2.  **Downstream Performance:** F1 Score / Exact Match of the final answer produced by the routed strategy.
3.  **Efficiency:** Inference latency (ms) and cost per 1k queries.
4.  **Routing Macro Metrics:** Macro-F1 (or balanced accuracy) and per-class precision/recall for routing (ensures Graph/Temporal performance is visible under label imbalance).
5.  **Optional Utility Score:** A simple cost-aware summary, e.g. \(U = \text{DownstreamF1} - \lambda \cdot \text{end\_to\_end\_latency\_ms}\) (or cost), reported alongside raw metrics.
6.  **Graph Diagnostic (Match Rate):** “% of queries with \(\ge 1\) entity match in the graph” (after entity normalization). This is not a primary metric, but it helps interpret when GraphRAG underperforms due to surface-form mismatch rather than reasoning.

## 9. Reproducibility & Measurement Protocols
These are experimental controls required to make ablation results credible and reproducible.

### A. Experiment Tracking (Required Fields)
Each experiment run must log:

- **Config identity:** config file path/name (and/or a stable hash of the config contents)
- **Code identity:** git commit hash
- **Dataset identity:** dataset version tag (and/or hash of `benchmark.jsonl` / `corpus.jsonl`)
- **Randomness controls:** global random seed(s) used (Python/NumPy/PyTorch, plus any retrieval-specific seeding)
- **Run metadata:** timestamp, run ID, machine/CPU/GPU (optional but helpful)

### B. LLM Router Determinism
For the LLM router experiments:

- Fix **model name/version** (provider + model identifier)
- Set **`temperature = 0`** (and log other sampling params)
- Log the **prompt template version** and either:
  - the rendered prompts, or
  - a prompt hash + the exact inputs used (question + probe stats)
- Log the **parser version** (regex / extraction logic), since parser changes can change labels/routes

### D. Oracle Fairness Control (Labels depend on generation)
Oracle labels are determined by answer quality via **LLM-as-judge** (not F1/EM), so they are a function of **retrieval + generation**. To ensure fairness across strategies during Oracle labeling:

- Use the **same generator model** (provider + model identifier) for all strategies.
- Use the **same base prompt template** for all strategies (only the retrieved contexts differ).
- When using the **LLM-as-judge**, use the same judge model and prompt for all strategies.
- Record `model_id`, `prompt_hash`, and `sampling params` (e.g., temperature/top_p) in logs and cache keys.

### C. Latency & Cost Protocol (What “latency” means)
Latency must be logged as a breakdown and as a total:

- **probe_latency_ms**
- **router_latency_ms**
- **strategy_retrieval_latency_ms**
- **strategy_generation_latency_ms**
- **end_to_end_latency_ms = probe + router + retrieval + generation**

Measurement rules:

- **Probe usage per condition:** For **Q-only** router variants, the probe should be disabled (`run_probe = false`) so they are not unfairly penalized. In that case, log `probe_latency_ms = 0` and keep `end_to_end_latency_ms` consistent with actual executed components. For **F-only** and **Combined** variants, set `run_probe = true`.
- Specify **warm vs cold** runs (recommended: warm runs for steady-state; separately report cold-start if relevant).
- Use **multiple repeats** per condition and report **median** (and IQR) to reduce noise.
- For LLM calls, record token usage and compute a normalized **cost per 1k queries** if applicable.