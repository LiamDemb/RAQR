# RAQR Design Decisions & Specifications

**Purpose:** To document the specific logic, algorithms, and parameter choices that define the system's behaviour. This serves as the "truth" for development.

## 1. The Heuristic Router Logic (The "Waterfall" Model)

The Heuristic Router is implemented as a **Priority Cascade**. It evaluates rules in a specific order of precedence. If a strong signal is found, it routes immediately. If not, it falls through to the next check.

### Decision Hierarchy

1.  **Level 1: Strong Keyword Matches (Regex)**
    - _Logic:_ If the user explicitly asks for a format (e.g., "list", "timeline") or implies a specific reasoning mode, trust the language.
2.  **Level 2: Retrieval Signal Analysis (Statistical)**
    - _Logic:_ If the language is ambiguous, look at the Probe results. Is the retriever confused (low skew)? Is the confidence high (max score)?
3.  **Level 3: Default Fallback**
    - _Logic:_ If no signals trigger, default to **Dense RAG** (Safe/Cheapest).

### Heuristic Rule Specification

_Implement these exact logic gates in `src/routers/heuristic.py`._

| Priority | Check Type | Condition (Pseudocode) | Route To | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Regex | `matches(query, r\"(from|between|in) \\d{4}\")` OR `contains(query, \"timeline\", \"history of\")` | **Temporal RAG** | Explicit date constraints. |
| **2** | Regex | `contains(query, \"connection between\", \"relationship\", \"how does X affect Y\")` | **Graph RAG** | Multi-hop reasoning intent. |
| **3** | Signal | `Probe.skewness < 0.5` AND `Probe.max_score > 0.65` | **Graph RAG** | Flat distribution implies multiple relevant entities (multi-hop). |
| **4** | Fallback | `True` | **Dense RAG** | Standard factual lookup. |

_Note: Thresholds (0.5, 0.65, 0.4) are initial hyperparameters. These will be tuned on the Dev Set._

## 2. Classifier Model Architecture ("Late Fusion")

The Lightweight Classifier must process two distinct data modalities: **Text** (Tokens) and **Signals** (Floats). We will use a **Late Fusion** architecture.

### Physical Architecture

1.  **Text Branch:**
    - Input: Tokenized Query (max length 512).
    - Backbone: `DistilBERT-base-uncased` (frozen or fine-tuned).
    - Output: `[CLS]` token embedding (Dimension: 768).
2.  **Signal Branch:**
    - Input: Vector of 5 floats `[max, min, mean, skew, dist]`.
    - Layer: Batch Normalization (Crucial: scales inputs to 0-1 range).
    - Output: Signal Vector (Dimension: 5).
3.  **Fusion Layer:**
    - Operation: `torch.cat([CLS_Vector, Signal_Vector], dim=1)`
    - Result: Combined Vector (Dimension: 773).
4.  **Classification Head:**
    - Layer 1: Linear (773 -> 256) + ReLU + Dropout(0.2).
    - Layer 2: Linear (256 -> 3) (3 Classes).
    - Output: Softmax probability distribution.

### Keyword Baseline (C-Q-KW)
We include an additional shallow supervised baseline that uses **only keyword/regex features** (no embeddings).\n\n- **Model:** Logistic Regression (or small MLP)\n- **Input:** sparse binary flags (e.g., `has_when`, `has_digit`, `starts_with_why`)\n- **Output:** 3-class prediction (Dense / Temporal / Graph)

## 3. Feedback Signal Definitions

The Probe runs a standard Dense retrieval (top-k=10). We extract signals from the resulting `scores` list (cosine similarity).

| Signal                | Definition / Formula                                                                                 | Library Implementation        |
| :-------------------- | :--------------------------------------------------------------------------------------------------- | :---------------------------- |
| **Max Score**         | $S_{max} = \max(scores)$                                                                             | `np.max(scores)`              |
| **Min Score**         | $S_{min} = \min(scores)$                                                                             | `np.min(scores)`              |
| **Mean Score**        | $\mu = \frac{1}{k}\sum scores$                                                                       | `np.mean(scores)`             |
| **Skewness**          | Measure of asymmetry. High skew = sharp peak (good). Low skew = flat (ambiguous).                    | `scipy.stats.skew(scores)`    |
| **Semantic Distance** | Distance between the Query Embedding ($Q$) and the Centroid ($C$) of the retrieved chunk embeddings. | `1 - cosine_similarity(Q, C)` |

## 4. Strategy Implementation Specs

To keep the scope manageable, we will implement "Minimum Viable Versions" of the complex strategies. We are testing the _routing_, not building the world's best GraphRAG.

### A. GraphRAG (Simplified)

- **Representation:** Relation-aware GraphRAG (Triple Graph) using NetworkX (no Neo4j).
- **Implementation:**
    1.  **Ingestion:** Run a lightweight Relation Extraction model (e.g., `Babelscape/rebel-large` or a GLiNER-relation model) over each chunk to extract semantic triples `(subject, predicate, object)`.
    2.  **Storage:** Store a NetworkX **DiGraph**:
        - **Nodes:** Canonical Entities (normalized strings) and Chunks.
        - **Edges:**
            - **Semantic Edge:** `Entity --predicate--> Entity` (directed relation from extracted triples).
            - **Provenance Edge:** `Entity --> Chunk` (`appears_in`) to link evidence back to text.
    3.  **Retrieval:** Extract entities from query $\rightarrow$ map to graph entity nodes $\rightarrow$ traverse outgoing **relational** edges (1-hop) $\rightarrow$ collect chunks linked via provenance edges from the expanded entity set.

### B. TemporalRAG (Metadata Filter)

- **Ingestion:** Run a regex date extractor over the corpus. Save `year` into the vector store metadata.
- **Retrieval:**
    1.  Extract year range from query (e.g., "2019-2021").
    2.  Retrieve top-N from FAISS (N≫k), then filter by `year_min/year_max` metadata and refill until k contexts.

## 5. The "Oracle" Labeling Logic

When generating training data, we must decide which strategy is "Correct."

**The Problem:** Dense RAG might get F1 score 0.82, and Graph RAG might get 0.83. Is Graph actually better? Probably not worth the extra compute.

**The Solution: Simplicity Bias with Margin**
We define a **Simplicity Hierarchy** (Cheapest to Most Expensive). We only upgrade to a more expensive strategy if the performance gain exceeds a **Margin Threshold ($\delta = 0.05$)**.

### Hierarchy (Ranked 0 to 2)

0.  **Dense RAG** (Baseline)
1.  **Temporal RAG** (Filter is cheap)
2.  **Graph RAG** (Most expensive/complex)

### Selection Algorithm

```python
def select_gold_label(results: Dict[str, float]) -> str:
    # results = {'Dense': 0.80, 'Graph': 0.84, ...}

    # 1. Sort strategies by Simplicity Rank (0 to 2)
    sorted_strategies = ['Dense', 'Temporal', 'Graph']

    best_strategy = 'Dense'
    best_score = results['Dense']

    # 2. Iterate up the hierarchy
    for strategy in sorted_strategies[1:]:
        current_score = results[strategy]

        # 3. Only switch if improvement > Margin (0.05)
        if current_score > (best_score + 0.05):
            best_strategy = strategy
            best_score = current_score

    return best_strategy
```

_This ensures our router is trained to be efficient, not just accuracy-obsessed._
