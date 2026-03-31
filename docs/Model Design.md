# RAQR Model Design

## 1. The Heuristic Router Logic

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

| Priority | Check Type | Condition (Pseudocode)                                                         | Route To      | Rationale                                                           |
| :------- | :--------- | :----------------------------------------------------------------------------- | :------------ | :------------------------------------------------------------------ |
| **2**    | Regex      | `contains(query, "connection between", "relationship", "how does X affect Y")` | **Graph RAG** | Multi-hop reasoning intent.                                         |
| **3**    | Signal     | `Probe.skewness < 0.5` AND `Probe.max_score > 0.65`                            | **Graph RAG** | "Flat" distribution implies multiple relevant entities (multi-hop). |
| **4**    | Fallback   | `True`                                                                         | **Dense RAG** | Standard factual lookup.                                            |

_Note: Thresholds (0.5, 0.65, 0.4) are initial hyperparameters. These will be tuned on the Dev Set._

## 2. Classifier Model Architecture ("Late Fusion")

The Lightweight Classifier must process distinct **feature families**: **Q-Emb** (query embedding), **Q-Feat** (engineered query features), and **Probe** (retrieval feedback). We use a **Late Fusion** architecture, concatenating these modalities before the classification head.

**Stage 1 signal ablation:** Stage 1 holds the architecture fixed (classifier) and only swaps input channels (Q-Emb only, Q-Feat only, Probe only, or combinations) to determine the winning input set. Stage 2 then compares Heuristic vs Classifier vs LLM using that winning set.

### Physical Architecture (Late Fusion Expectations)

1.  **Q-Emb Branch (Text):**
    - Input: Pre-computed `all-MiniLM-L6-v2` embedding.
    - Output: Sentence embedding (Dimension: 384).
2.  **Q-Feat + Probe Branch (Signals):**
    - Q-Feat: length/token count, entity density (from query entity extraction), complexity keywords; optional syntax depth.
    - Probe: max score, skewness, semantic dispersion (alias: semantic distance).
    - Input: Vector of floats (dimension depends on active channels).
    - Layer: Batch Normalization (Crucial: scales inputs to 0-1 range).
    - Output: Signal Vector.
3.  **Fusion Layer:**
    - Operation: `torch.cat([Q_Emb_Vector, Signal_Vector], dim=1)`
    - Result: Combined Vector (dimension = 384 + signal dimension).
4.  **Classification Head:**
    - Layer 1: Linear (input_dim -> 128) + ReLU + Dropout(0.3).
    - Layer 2: Linear (128 -> 128) + ReLU + Dropout(0.3).
    - Layer 3: Linear (128 -> 2) (2 Classes: Dense, Graph).
    - Output: Softmax probability distribution.

## 3. Feature Families (Q-Emb / Q-Feat / Probe)

**Q-Emb:** `all-MiniLM-L6-v2` sentence embedding (384 dims).

**Q-Feat:** Engineered query features—length/token count, entity density (from query entity extraction), complexity keywords; optional syntax depth.

**Probe:** Top-10 Dense retrieval signals. The Probe runs a standard Dense retrieval (top-k=10). We extract signals from the resulting `scores` list (cosine similarity):

| Signal                  | Definition / Formula                                                                                         | Library Implementation        |
| :---------------------- | :----------------------------------------------------------------------------------------------------------- | :---------------------------- |
| **Max Score**           | $S_{max} = \max(scores)$                                                                                     | `np.max(scores)`              |
| **Min Score**           | $S_{min} = \min(scores)$                                                                                     | `np.min(scores)`              |
| **Mean Score**          | $\mu = \frac{1}{k}\sum scores$                                                                               | `np.mean(scores)`             |
| **Skewness**            | Measure of asymmetry. High skew = sharp peak (good). Low skew = flat (ambiguous).                            | `scipy.stats.skew(scores)`    |
| **Semantic dispersion** | Average distance between the Query Embedding ($Q$) and the Centroid ($C$) of the retrieved chunk embeddings. | `1 - cosine_similarity(Q, C)` |

## 4. Strategy Implementation Specs

To keep the scope manageable, we will implement "Minimum Viable Versions" of the complex strategies. We are testing the _routing_, not building the world's best GraphRAG.

### A. GraphRAG (Simplified)

- **Representation:** We will not create a massive Neo4j instance.
- **Implementation:**
    - **Graph Schema (explicit):**
        - **Nodes:** `Entity` nodes and `Chunk` nodes (document chunk IDs).
        - **Edges (primary):** Directed semantic relations from extracted triples: `Entity --predicate--> Entity`.
        - **Edges (provenance):** `Entity --> Chunk` (`appears_in`) to provide evidence text for relations.
        - **Retrieval:** extract entities from query → map to graph entity nodes → traverse 1-hop relational edges → resolve to chunks via provenance edges.
    - **Relation Extraction Policy:**
        - LLM extraction over chunk text to extract entities and `(subject, predicate, object)` triples.
    - **Entity Normalization Policy (to reduce surface-form mismatch):**
        - Lowercase
        - Strip punctuation / normalize whitespace
        - Optional lightweight alias table for common entities (e.g., `{"us": "united states", "u.s.": "united states", "uk": "united kingdom"}`)
        - Apply the _same_ normalization to entities extracted from chunks and from queries.
    - **Diagnostic to log:** entity match rate on the benchmark:
        - “% of queries with \(\ge 1\) entity match in the graph” (after normalization).
        - This helps distinguish “GraphRAG failed due to aliasing/matching” from “Graph reasoning not needed”.
    1.  **Ingestion:** Extract entities + semantic triples \((subject, predicate, object)\) from chunk text using LLM extraction via the Batch API.
    2.  **Storage:** Store a NetworkX **DiGraph** with directed `Entity --predicate--> Entity` semantic edges and `Entity --> Chunk` provenance edges.
    3.  **Retrieval:** Extract entities from query $\rightarrow$ map to entity nodes $\rightarrow$ traverse outgoing relational edges (1-hop) $\rightarrow$ collect evidence chunks via provenance edges. Candidate hop and path relevance scoring is managed via a configurable `ScoringConfig` (`local_pred_weight`, `bundle_pred_weight`, `length_penalty`).

## 5. The "Oracle" Labeling Logic

When generating training data, we must decide which strategy is "Correct."

**The Problem:** Dense RAG might get F1 score 0.82, and Graph RAG might get 0.83. Is Graph actually better? Probably not worth the extra compute.

**The Solution: Simplicity Bias with Margin**
We use a **margin-based simplicity bias** oracle: only choose a more complex strategy if it beats simpler alternatives by at least a margin \(\delta\). This reduces label noise from tiny score differences and trains routers toward the deployment goal: **don’t spend complexity for tiny gains**.

### Hierarchy (Ranked 0 to 2)

0.  **Dense RAG** (Baseline)
1.  **Graph RAG** (Most expensive/complex)

### Selection Algorithm

```python
def select_gold_label(results: Dict[str, float]) -> str:
    # results = {'Dense': 0.80, 'Graph': 0.84, ...}

    delta = 0.05  # Calibrated on Dev (ablated over {0.00, 0.02, 0.05, 0.10})
    eps = 1e-6

    s_dense = results['Dense']
    s_graph = results['Graph']

    # 1) Determine best by raw score (ties handled deterministically below)
    best_score = max(s_dense, s_temp, s_graph)
    tied = []
    if abs(s_dense - best_score) <= eps:
        tied.append('Dense')
    if abs(s_graph - best_score) <= eps:
        tied.append('Graph')

    # Deterministic tie-break: Dense > Graph
    for name in ['Dense', 'Graph']:
        if name in tied:
            best = name
            break

    # 2) Margin-based "upgrade" rule
    if best == 'Dense':
        return 'Dense'

    # best == 'Graph'
    if (s_graph >= s_dense + delta) and (s_graph >= s_temp + delta):
        return 'Graph'

    # Fall back to the simplest strategy within delta of Graph
    # (i.e., doesn't lose much compared to Graph)
    if s_dense >= s_graph - delta:
        return 'Dense'
    return 'Graph'
```

_This ensures our router is trained to be efficient, not just accuracy-obsessed._
