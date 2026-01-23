# RAQR Model Design

## 1. The Heuristic Router Logic
The Heuristic Router is implemented as a **Priority Cascade**. It evaluates rules in a specific order of precedence. If a strong signal is found, it routes immediately. If not, it falls through to the next check.

### Decision Hierarchy
1.  **Level 1: Strong Keyword Matches (Regex)**
    *   *Logic:* If the user explicitly asks for a format (e.g., "list", "timeline") or implies a specific reasoning mode, trust the language.
2.  **Level 2: Retrieval Signal Analysis (Statistical)**
    *   *Logic:* If the language is ambiguous, look at the Probe results. Is the retriever confused (low skew)? Is the confidence high (max score)?
3.  **Level 3: Default Fallback**
    *   *Logic:* If no signals trigger, default to **Dense RAG** (Safe/Cheapest).

### Heuristic Rule Specification
*Implement these exact logic gates in `src/routers/heuristic.py`.*

| Priority | Check Type | Condition (Pseudocode) | Route To | Rationale |
| :--- | :--- | :--- | :--- | :--- |
| **1** | Regex | `matches(query, r"(from or between or in) \d{4}")` OR `contains(query, "timeline", "history of")` | **Temporal RAG** | Explicit date constraints. |
| **2** | Regex | `contains(query, "connection between", "relationship", "how does X affect Y")` | **Graph RAG** | Multi-hop reasoning intent. |
| **3** | Signal | `Probe.skewness < 0.5` AND `Probe.max_score > 0.65` | **Graph RAG** | "Flat" distribution implies multiple relevant entities (multi-hop). |
| **4** | Fallback | `True` | **Dense RAG** | Standard factual lookup. |

*Note: Thresholds (0.5, 0.65, 0.4) are initial hyperparameters. These will be tuned on the Dev Set.*

## 2. Classifier Model Architecture ("Late Fusion")
The Lightweight Classifier must process two distinct data modalities: **Text** (Tokens) and **Signals** (Floats). We will use a **Late Fusion** architecture.

### Physical Architecture
1.  **Text Branch:**
    *   Input: Tokenized Query (max length 512).
    *   Backbone: `DistilBERT-base-uncased` (frozen or fine-tuned).
    *   Output: `[CLS]` token embedding (Dimension: 768).
2.  **Signal Branch:**
    *   Input: Vector of 5 floats `[max, min, mean, skew, dist]`.
    *   Layer: Batch Normalization (Crucial: scales inputs to 0-1 range).
    *   Output: Signal Vector (Dimension: 5).
3.  **Fusion Layer:**
    *   Operation: `torch.cat([CLS_Vector, Signal_Vector], dim=1)`
    *   Result: Combined Vector (Dimension: 773).
4.  **Classification Head:**
    *   Layer 1: Linear (773 -> 256) + ReLU + Dropout(0.2).
    *   Layer 2: Linear (256 -> 3) (3 Classes).
    *   Output: Softmax probability distribution.

## 3. Feedback Signal Definitions
The Probe runs a standard Dense retrieval (top-k=10). We extract signals from the resulting `scores` list (cosine similarity).

| Signal | Definition / Formula | Library Implementation |
| :--- | :--- | :--- |
| **Max Score** | $S_{max} = \max(scores)$ | `np.max(scores)` |
| **Min Score** | $S_{min} = \min(scores)$ | `np.min(scores)` |
| **Mean Score** | $\mu = \frac{1}{k}\sum scores$ | `np.mean(scores)` |
| **Skewness** | Measure of asymmetry. High skew = sharp peak (good). Low skew = flat (ambiguous). | `scipy.stats.skew(scores)` |
| **Semantic Distance** | Distance between the Query Embedding ($Q$) and the Centroid ($C$) of the retrieved chunk embeddings. | `1 - cosine_similarity(Q, C)` |

## 4. Strategy Implementation Specs
To keep the scope manageable, we will implement "Minimum Viable Versions" of the complex strategies. We are testing the *routing*, not building the world's best GraphRAG.

### A. GraphRAG (Simplified)
*   **Representation:** We will not create a massive Neo4j instance.
*   **Implementation:**
    *   **Graph Schema (explicit):**
        *   **Nodes:** `Entity` nodes (from spaCy; optional GLiNER) and `Chunk` nodes (document chunk IDs).
        *   **Edges (minimum):** `Entity -> Chunk` membership if an entity occurs in that chunk.
        *   **Edges (optional):** `Entity <-> Entity` co-occurrence within the same chunk (or within a small window).
        *   **Retrieval:** extract entities from query → find matching `Entity` nodes → retrieve connected `Chunk` nodes (and optionally 1-hop expansion).
    *   **Entity Normalization Policy (to reduce surface-form mismatch):**
        *   Lowercase
        *   Strip punctuation / normalize whitespace
        *   Optional lightweight alias table for common entities (e.g., `{"us": "united states", "u.s.": "united states", "uk": "united kingdom"}`)
        *   Apply the *same* normalization to entities extracted from chunks and from queries.
    *   **Diagnostic to log:** entity match rate on the benchmark:
        *   “% of queries with \(\ge 1\) entity match in the graph” (after normalization).
        *   This helps distinguish “GraphRAG failed due to aliasing/matching” from “Graph reasoning not needed”.
    1.  **Ingestion:** Use `spaCy` to extract entities (ORG, PERSON, GPE) from chunks.
    2.  **Storage:** Store relations in a NetworkX graph (in-memory) or simple adjacency list JSON.
    3.  **Retrieval:** Extract entities from Query $\rightarrow$ Find nodes in Graph $\rightarrow$ Retrieve 1-hop neighbors $\rightarrow$ Return text chunks associated with those neighbors.

### B. TemporalRAG (Metadata Filter)
*   **Ingestion:** Run a regex date extractor over the corpus. Save `year` into the vector store metadata.
*   **Retrieval:**
    1.  Extract year range from query (e.g., "2019-2021").
    2.  **Implementation note (FAISS):** Raw FAISS does not natively support metadata filters. We implement filtering explicitly:
        *   Retrieve a deeper candidate set (e.g., top \(N\) where \(N \gg k\)).
        *   Filter candidates by `year` in metadata.
        *   If fewer than \(k\) contexts remain, keep pulling from deeper ranks until \(k\) contexts are filled (or candidates are exhausted).

## 5. The "Oracle" Labeling Logic
When generating training data, we must decide which strategy is "Correct."

**The Problem:** Dense RAG might get F1 score 0.82, and Graph RAG might get 0.83. Is Graph actually better? Probably not worth the extra compute.

**The Solution: Simplicity Bias with Margin**
We use a **margin-based simplicity bias** oracle: only choose a more complex strategy if it beats simpler alternatives by at least a margin \(\delta\). This reduces label noise from tiny score differences and trains routers toward the deployment goal: **don’t spend complexity for tiny gains**.

### Hierarchy (Ranked 0 to 2)
0.  **Dense RAG** (Baseline)
1.  **Temporal RAG** (Filter is cheap)
2.  **Graph RAG** (Most expensive/complex)

### Selection Algorithm
```python
def select_gold_label(results: Dict[str, float]) -> str:
    # results = {'Dense': 0.80, 'Graph': 0.84, ...}

    delta = 0.05  # Calibrated on Dev (ablated over {0.00, 0.02, 0.05, 0.10})
    eps = 1e-6

    s_dense = results['Dense']
    s_temp = results['Temporal']
    s_graph = results['Graph']

    # 1) Determine best by raw score (ties handled deterministically below)
    best_score = max(s_dense, s_temp, s_graph)
    tied = []
    if abs(s_dense - best_score) <= eps:
        tied.append('Dense')
    if abs(s_temp - best_score) <= eps:
        tied.append('Temporal')
    if abs(s_graph - best_score) <= eps:
        tied.append('Graph')

    # Deterministic tie-break: Dense > Temporal > Graph
    for name in ['Dense', 'Temporal', 'Graph']:
        if name in tied:
            best = name
            break

    # 2) Margin-based "upgrade" rule
    if best == 'Dense':
        return 'Dense'

    if best == 'Temporal':
        return 'Temporal' if (s_temp >= s_dense + delta) else 'Dense'

    # best == 'Graph'
    # Require Graph to beat Dense by delta, and (optionally but clean) Temporal by delta.
    if (s_graph >= s_dense + delta) and (s_graph >= s_temp + delta):
        return 'Graph'

    # Fall back to the simplest strategy within delta of Graph
    # (i.e., doesn't lose much compared to Graph)
    if s_dense >= s_graph - delta:
        return 'Dense'
    if s_temp >= s_graph - delta:
        return 'Temporal'
    return 'Graph'
```
*This ensures our router is trained to be efficient, not just accuracy-obsessed.*