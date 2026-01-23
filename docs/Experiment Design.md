# Experimental Design & Protocol

## 1. Experimental Overview
**Objective:** To isolate and measure the impact of specific input signals and model architectures on routing accuracy within a RAG system.
**Method:** "Gold Label" (Oracle) supervision. We generate ground-truth labels by running all strategies on a mixed benchmark and selecting the winner, then train/test routers to predict that winner.

## 2. The Dataset (Benchmark)
To ensure the system encounters diverse reasoning requirements, we construct a unified evaluation dataset merging samples from three sources:
*   **Natural Questions (NQ):** Factual/Lookup queries (Target: Dense RAG).
*   **ComplexTempQA:** Temporal/Chronological queries (Target: Temporal RAG).
*   **WikiWhy:** Causal/Explanation queries (Target: GraphRAG).

*Total Target Size: ~750–1,500 balanced samples.*

## 3. The Retrieval Strategies (Class Labels)
The system routes queries to one of three distinct retrieval pipelines. These act as the classification targets.

1.  **Dense RAG:** (Baseline) Vector search (`all-MiniLM-L6-v2`) + Chunk retrieval.
2.  **GraphRAG:** Entity-relation traversal (NetworkX/Neo4j) for multi-hop reasoning.
3.  **Temporal RAG:** Metadata-filtered retrieval for time-bound queries.

### Label Generation Logic (The Oracle)
To generate training data, every question is processed by all 3 strategies. The Oracle produces a single "Gold Label" using a **margin-based simplicity bias** rule.

#### Definitions
Let:

- \(S_{dense}\), \(S_{temp}\), \(S_{graph}\) be the evaluation score (e.g., token-F1 in \([0, 1]\)) for each strategy on that question.
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

#### Choosing \(\delta\) (calibrated, not guessed)
We select \(\delta\) empirically on the **Dev** set:

- Compute \(\Delta = S_{best} - S_{second\_best}\) for each Dev example (using raw scores).
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
The routers consume two categories of data:

### A. Question Features (Static)
*   **Raw Text:** The user's query string.
*   **Keywords:** Presence of trigger terms (e.g., "when", "relationship", "cause") via Regex.
*   **Embeddings:** A 384-dimensional vector representation of the query.

### B. Retrieval Feedback (Dynamic/Probe)
*Derived from a "Probe Search" (a fast top-k retrieval using the Dense index).*
*   **Max Score:** The cosine similarity of the top result (Proxy for confidence).
*   **Score Skewness:** The statistical skew of the top-k score distribution. (High skew = precise; Low skew = confused/multi-hop).
*   **Semantic Distance:** Average distance between the query and the retrieved cluster centroid.

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

## 6. The Ablation Grid (Testing Matrix)
We perform a 3x3 ablation study. Each cell represents a distinct experiment script.

| Model Class | Input: Question Only | Input: Feedback Only | Input: Combined |
| :--- | :--- | :--- | :--- |
| **1. Heuristic** | **Test H-Q:** Keyword/Regex matching. | **Test H-F:** Thresholds on Skewness/Max Score. | **Test H-C:** Logic tree (Keywords $\rightarrow$ then Stats). |
| **2. Classifier** | **Test C-Q:** Standard DistilBERT (Text classification). | **Test C-F:** Simple MLP (Feed-forward network on stats). | **Test C-C:** DistilBERT + Concatenated Feature Vector. |
| **3. LLM** | **Test L-Q:** Standard Prompt ("Classify this text"). | **Test L-F:** Stats Prompt ("Given these scores..."). | **Test L-C:** Chain-of-Thought ("Analyze text and scores"). |

## 7. Evaluation Metrics
1.  **Routing Accuracy:** % of times the router picked the "Gold Label" strategy.
2.  **Downstream Performance:** F1 Score / Exact Match of the final answer produced by the routed strategy.
3.  **Efficiency:** Inference latency (ms) and cost per 1k queries.
4.  **Routing Macro Metrics:** Macro-F1 (or balanced accuracy) and per-class precision/recall for routing (ensures Graph/Temporal performance is visible under label imbalance).
5.  **Optional Utility Score:** A simple cost-aware summary, e.g. \(U = \text{DownstreamF1} - \lambda \cdot \text{end\_to\_end\_latency\_ms}\) (or cost), reported alongside raw metrics.
6.  **Graph Diagnostic (Match Rate):** “% of queries with \(\ge 1\) entity match in the graph” (after entity normalization). This is not a primary metric, but it helps interpret when GraphRAG underperforms due to surface-form mismatch rather than reasoning.

## 8. Reproducibility & Measurement Protocols
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
Oracle labels are determined by answer quality (F1/EM), so they are a function of **retrieval + generation**. To ensure fairness across strategies during Oracle labeling:

- Use the **same generator model** (provider + model identifier) for all strategies.
- Use the **same base prompt template** for all strategies (only the retrieved contexts differ).
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