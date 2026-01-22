# Experimental Design & Protocol

## 1. Experimental Overview
**Objective:** To isolate and measure the impact of specific input signals and model architectures on routing accuracy within a RAG system.
**Method:** "Gold Label" (Oracle) supervision. We generate ground-truth labels by running all strategies on a mixed benchmark and selecting the winner, then train/test routers to predict that winner.

## 2. The Dataset (Benchmark)
To ensure the system encounters diverse reasoning requirements, we construct a unified evaluation dataset merging samples from four sources:
*   **Natural Questions (NQ):** Factual/Lookup queries (Target: Dense RAG).
*   **ComplexTempQA:** Temporal/Chronological queries (Target: Temporal RAG).
*   **WikiWhy:** Causal/Explanation queries (Target: GraphRAG).
*   **MEQA/Custom:** Event-centric and Aggregation queries (Target: Table/Hierarchical RAG).

*Total Target Size: ~1,000–2,000 balanced samples.*

## 3. The Retrieval Strategies (Class Labels)
The system routes queries to one of five distinct retrieval pipelines. These act as the classification targets.

1.  **Dense RAG:** (Baseline) Vector search (`all-MiniLM-L6-v2`) + Chunk retrieval.
2.  **Hierarchical RAG:** Recursive summarisation tree (RAPTOR-style) for global context.
3.  **GraphRAG:** Entity-relation traversal (NetworkX/Neo4j) for multi-hop reasoning.
4.  **Temporal RAG:** Metadata-filtered retrieval for time-bound queries.
5.  **Table RAG:** Schema-guided extraction for aggregation/SQL-like queries.

### Label Generation Logic (The Oracle)
To generate training data, every question is processed by all 5 strategies. The "Gold Label" is determined by:
1.  **Highest Performance:** The strategy with the highest Answer F1 Score.
2.  **Tie-Breaking:** If scores are identical, the "Simpler/Cheaper" strategy wins (Order of preference: Dense > Table > Temporal > Graph > Hierarchical).

## 4. The Input Signals (Features)
The routers consume two categories of data:

### A. Question Features (Static)
*   **Raw Text:** The user's query string.
*   **Keywords:** Presence of trigger terms (e.g., "when", "summarise", "average") via Regex.
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