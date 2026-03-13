# RAQR Unified Corpus Creation Pipeline

**Goal:** Build a **Unified Mega Corpus** that enables a scientifically fair “Gold Label Oracle” evaluation across **Dense RAG** and **GraphRAG (NetworkX)** under a shared retrieval interface and shared artifacts (FAISS + NetworkX).

## 0) Design Objectives and Non-Goals

### Objectives

1. **Unified**: All strategies operate over the _same_ corpus and chunk inventory.
2. **Connective tissue**: The corpus must contain:
    - **multi-hop relational links** (to give GraphRAG a real advantage),

3. **Fairness**: No dataset gets special treatment; corpus expansion is driven by the same rules/budgets across datasets.
4. **Completeness**: For each sampled question, the corpus should contain answer-bearing evidence, even if it requires **1–2 hops**.
5. **Efficiency**: Must scale to ~1,500 questions without downloading full Wikipedia dumps.

### Non-goals

- Building a perfect KG (GraphRAG is intentionally “minimum viable,” NetworkX-based).
- Exhaustive Wikipedia ingestion.

## 1) Inputs, Outputs, and Physical Artifacts

### Inputs (sampled question sets)

- **2WikiMultiHopQA**: `supporting_facts.title` — a list of 2 Wikipedia page titles per question (the gold supporting documents).
- **Natural Questions (NQ)**: raw HTML/tokens with `is_html` flags.

### Outputs (offline artifacts)

1. `data/processed/corpus.jsonl` — unified chunked corpus (schema below).
2. `data/processed/docstore.sqlite` (or LMDB) — page-level cache + chunk metadata (for dedupe and reproducibility).
3. `data/processed/vector_index.faiss` + `data/processed/vector_meta.parquet` — embeddings + metadata mapping (FAISS has no native metadata filtering, so store metadata externally).
4. `data/processed/graph.pkl` — NetworkX graph (Entity-Relation-Entity edges + Entity-Chunk provenance edges).
5. `data/processed/entity_lexicon.parquet` — canonical entity strings, aliases, and optional QIDs (used for normalization).
6. `data/processed/entity_index.faiss` + `entity_index_meta.parquet` — FAISS index over entity norms for vector similarity matching at query time.
7. `data/processed/wiki_titles.jsonl` — Wikipedia page titles fetched for the corpus (used for LLM seed anchoring).
8. `data/processed/alias_map.json` — deterministic redirect-mined alias artifact (`Dict[str, str]`, normalized alias -> normalized canonical) produced at corpus-build time; used for FlashText expansion and query normalization.

These artifacts align with the modular “offline prep vs online routing & inference” separation.

## 2) Unified `corpus.jsonl` Schema

Each line is one **chunk**, not one document.

```json
{
    "chunk_id": "uuid",
    "doc_id": "uuid",
    "source": "wikipedia|nq|2wiki",
    "title": "string|null",
    "url": "string|null",
    "text": "string",
    "section_path": ["Lead", "Early life", "Career"],
    "char_span_in_doc": [1234, 1876],

    "metadata": {
        "dataset_origin": "nq|2wiki",
        "page_id": "string|null",
        "revision_id": "string|null",

        "entities": [
            {
                "surface": "United States",
                "norm": "united states",
                "type": "GPE",
                "qid": "Q30"
            },
            {
                "surface": "Barack Obama",
                "norm": "barack obama",
                "type": "PERSON",
                "qid": "Q76"
            }
        ],

        "relations": [
            {
                "subj_norm": "barack obama",
                "pred": "born_in",
                "obj_norm": "united states"
            }
        ],

        "anchors": {
            "outgoing_titles": [
                "List of ...",
                "France",
                "2012 Summer Olympics"
            ],
            "incoming_stub": []
        }
    }
}
```

**Notes**

- The corpus supports **relation extraction (triples)** for GraphRAG: `relations[]` enables relation-aware 1-hop expansion `Entity --predicate--> Entity`, with evidence resolved via `Entity -> Chunk` provenance edges.
- `entities[].norm` is the GraphRAG join key (string-canonical), with optional `qid` when resolvable.
- `anchors.outgoing_titles` is lightweight “connective tissue” to support multi-hop traversal without requiring a full hyperlink graph.

## 3) Pipeline Overview (Start-to-Finish)

### Stage A — Sampling & Budgeting

**Input:** ~750–1,500 questions balanced across datasets.
**Output:** `data/interim/benchmark_sample.jsonl` (question-only; no gold strategy label).

**Fairness rule:** Every question receives the _same_ maximum expansion budget:

- `MAX_PAGES_PER_QUESTION` (e.g., 12)
- `MAX_HOPS` (e.g., 2)
- `MAX_LIST_PAGES_PER_QUESTION` (e.g., 2)
- `MAX_COUNTRY_PAGES_PER_QUESTION` (e.g., 1)

### Stage B — Source-Specific Ingestion (Doc Acquisition)

Produce a set of **page candidates** per question. Dedupe globally by `page_id`/`title`.

### Stage C — Canonical Cleaning → Structured Document

Convert each acquired doc into:

- `doc.text` (clean prose, preserves headings/lists where possible),
- `doc.section_tree` (for chunk provenance),
- `doc.anchors` (outgoing titles).

### Stage D — Unified Enrichment Layer

Run **the same enrichment** on all docs:

- Regex date extraction → `years` metadata
- Entity + relation extraction (LLM IE batch) → `entities` + `relations` metadata (and graph edges)

### Stage E — Chunking & Serialization

Chunk to 500–800 tokens (tiktoken) with overlap; write `corpus.jsonl`. LLM IE extraction runs per chunk via the Batch API.

**Token limits:** Chunking uses configurable `CHUNK_MIN_TOKENS`, `CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP_TOKENS`. The embedder (all-MiniLM-L6-v2) truncates at 256 tokens internally.

### Stage F — Indexing

- Embed chunks → build FAISS
- Build NetworkX graph from extracted relations (triples) + provenance edges

### Stage G — Quality Gates (Required)

- Coverage and diagnostics to ensure GraphRAG has a fair chance (see Section 8).

## 4) Source-Specific Ingestion Specifications

#### 4.1.1 QID Resolution

For each question:

- Collect seed QIDs:
    - `question_entity` (and any additional linked QIDs present)
    - `answer_entity`

- Resolve each QID to:
    - Wikipedia title(s) via sitelinks
    - Optional redirects/aliases

**Acquisition:** Use Wikipedia API to fetch article plaintext (or HTML then clean) + section headers + links.

#### 4.1.2 Hop Expansion (Connective Tissue)

We create a _bounded_ expansion set per question:

**Hop-0 (seed pages):**

- page(question_entity)
- page(answer_entity)

**Hop-1 (relational/context pages):**

- For each seed QID, pull a small set of “context QIDs” from Wikidata properties (bounded):
    - geographic: `country (P17)`, `located in admin entity (P131)`
    - membership/affiliation: `member of (P463)`, `part of (P361)`
    - time-bearing: `inception (P571)`, `point in time (P585)` → used to prioritize pages likely containing dates

- Resolve those context QIDs to pages.

**Hop-2 (optional, strict budget):**

- Only if remaining budget and query appears multi-hop (e.g., multiple entities detected).
- Expand via outgoing Wikipedia links from Hop-0 pages (top-N links by lead/infobox presence).

#### 4.1.3 Parsing Strategy

1. **Reconstruct HTML** using token sequence and `is_html` flags (when available), otherwise use the provided HTML field directly.
2. Parse with an HTML parser (e.g., BeautifulSoup/lxml).
3. Remove non-content blocks:
    - `script`, `style`, navigation, footers, edit links

4. Preserve document structure:
    - headings `h1–h6` → section nodes
    - `p` paragraphs
    - `ul/ol` lists (flatten with bullet markers)
    - tables: serialize rows as `|`-delimited lines (bounded to avoid massive tables)

#### 4.1.4 NQ Cleaning Pseudocode

```python
def clean_nq_html(html):
    soup = BeautifulSoup(html, "lxml")

    drop_tags = ["script", "style", "nav", "footer", "aside"]
    for t in drop_tags:
        for node in soup.find_all(t):
            node.decompose()

    doc = StructuredDoc()
    current_path = ["Lead"]

    for node in soup.body.descendants:
        if node.name in ["h1","h2","h3","h4","h5","h6"]:
            current_path = update_section_path(current_path, node.get_text(" ", strip=True), level=node.name)
        elif node.name == "p":
            doc.add_block("paragraph", node.get_text(" ", strip=True), current_path)
        elif node.name in ["ul","ol"]:
            items = [li.get_text(" ", strip=True) for li in node.find_all("li", recursive=False)]
            doc.add_block("list", "\n".join(f"- {x}" for x in items), current_path)
        elif node.name == "table":
            text = table_to_text(node, max_rows=30, max_cols=8)
            doc.add_block("table", text, current_path)

    return doc
```

### 4.4 2WikiMultiHopQA (Supporting Facts → Wikipedia Pages Only)

**Problem:** 2WikiMultiHopQA provides `supporting_facts.title` — exactly 2 Wikipedia page titles that contain the answer evidence. No expansion is needed; the dataset already specifies the gold documents.

#### 4.4.1 Base Retrieval

- For each question, read `supporting_facts.title` (a list of 2 strings).
- Fetch each Wikipedia page via the API (cached in docstore).
- No outgoing-link or list-page expansion; 2WikiMultiHopQA uses only the supporting pages.

#### 4.4.2 2WikiMultiHopQA Ingestion Pseudocode

```python
def ingest_2wiki(example, budgets):
    titles = unique(example.supporting_facts["title"])
    return fetch_wikipedia_docs(titles)  # cached + deduped
```

## 5) Unified Enrichment Layer (Applied to All Docs)

### 5.2 Entity & Relation Extraction Policy (GraphRAG Join-Key Reliability)

**Goal:** GraphRAG success depends on matching query entities to corpus entities. Surface-form mismatch (“U.S.” vs “United States”) can collapse GraphRAG’s recall; we must normalize entities consistently across all sources.

#### 5.2.1 LLM Information Extraction (Pipeline Default)

- A single LLM call per chunk extracts **both** entities and triples via the Batch API.
- Wikipedia titles for fetched pages are pre-scanned with FlashText to pass a per-chunk **seed list** into the prompt; the LLM anchors to these when possible but may add new entities.
- Output: `metadata.entities` and `metadata.relations` with consistent join keys.
- Implemented by `scripts/corpus/run_llm_ie_batch.py` (submit → wait → collect → replace corpus).

#### 5.2.2 Normalization (deterministic)

Apply the same function everywhere:

1. Unicode normalize (NFKC), lowercase
2. Strip punctuation except internal hyphens
3. Collapse whitespace
4. Remove possessives: `"obama's" → "obama"`
5. Alias map (small, curated + auto-mined):
    - curated: `{ "u.s.": "united states", "us": "united states", "uk": "united kingdom" }`
    - auto-mined from Wikipedia redirects (when available from fetched pages)

**Optional (recommended if available without heavy infra):**

- If a chunk originates from a resolved Wikipedia page with a known QID, tag that page-title entity with `qid` and propagate.

#### 5.2.3 Entity Lexicon Build

Create `entity_lexicon.parquet`:

- `norm` (primary key)
- `surface_forms` (top-k frequent)
- `qid_candidates` (if any)
- `df` document frequency (for downweighting extremely common entities)

#### 5.2.4 Entity Policy Pseudocode

```python
def norm_entity(s, alias_map):
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"['’]s\b", "", s)            # possessive
    s = re.sub(r"[^\w\s-]", " ", s)          # drop punct
    s = re.sub(r"\s+", " ", s).strip()
    return alias_map.get(s, s)

def extract_entities_spacy(text, nlp, alias_map):
    ents = []
    doc = nlp(text)
    for e in doc.ents:
        if e.label_ in {"PERSON","ORG","GPE","LOC","EVENT","WORK_OF_ART"}:
            ents.append({
                "surface": e.text,
                "norm": norm_entity(e.text, alias_map),
                "type": e.label_,
                "qid": None
            })
    return dedupe_by_norm(ents)
```

## 6) Chunking Strategy (500–800 tokens + overlap)

### 6.1 Requirements

- Chunk size: **500–800 tokens** (tiktoken cl100k_base; aligns with OpenAI models)
- Overlap: **~10–15%** (e.g., 100 tokens)
- Preserve section boundaries where possible (don’t merge unrelated sections)

### 6.2 Algorithm

1. Split doc into blocks (paragraph/list/table) with section paths.
2. For each block, count tokens via tiktoken (cl100k_base).
3. Accumulate blocks until token budget reached.
4. When exceeding:
    - finalize chunk
    - start next chunk with overlap tail (token-based)
5. If a single block exceeds `max_tokens`, flush buffer and split the block into overlapping sub-chunks.

### 6.3 Implementation

Implemented in `src/raqr/data/chunking.py`. Uses `chunk_blocks(blocks, min_tokens=500, max_tokens=800, overlap_tokens=100)` with tiktoken cl100k_base. Chunk limits are configurable via `--chunk-min-tokens`, `--chunk-max-tokens`, `--chunk-overlap-tokens` or env vars.

## 7) Indexing Artifacts

### 7.1 FAISS Vector Index (Dense Candidate Pool)

- Embed every chunk using a single embedding model (e.g., `all-MiniLM-L6-v2`) for consistency.
- Store:
    - FAISS index (vectors)
    - Separate metadata table keyed by FAISS row id → `chunk_id`, etc.

### 7.2 NetworkX Relationship Graph (GraphRAG)

Graph schema (minimum viable):

- Nodes:
    - `E:{norm_entity}` (Entity nodes)
    - `C:{chunk_id}` (Chunk nodes)

- Edges:
    - **Semantic edges (primary):** `E --predicate--> E` from extracted triples \((subject, predicate, object)\)
    - **Provenance edges:** `E --> C` (`appears_in`) if entity occurs in chunk (optionally weighted by frequency)

**Build pseudocode**

```python
def build_graph(chunks):
    G = nx.DiGraph()
    for ch in chunks:
        cnode = f"C:{ch.chunk_id}"
        G.add_node(cnode, kind="chunk")

        for ent in ch.metadata["entities"]:
            enode = f"E:{ent['norm']}"
            G.add_node(enode, kind="entity", type=ent["type"])
            G.add_edge(enode, cnode, kind="appears_in")

        # semantic triples (subject, predicate, object)
        for rel in ch.metadata.get("relations", []):
            s = rel["subj_norm"]
            p = rel["pred"]
            o = rel["obj_norm"]
            G.add_node(f"E:{s}", kind="entity")
            G.add_node(f"E:{o}", kind="entity")
            G.add_edge(f"E:{s}", f"E:{o}", kind="rel", label=p)
    return G
```

## 8) Required Quality Gates (Prevents “Simplicity Bias” Collapse)

RAQR’s Oracle labels will otherwise default to Dense RAG if GraphRAG lack evidence or linkage. The following diagnostics must be computed after ingestion and before Oracle runs. This aligns with the project’s controlled, oracle-based methodology and the need to interpret Graph underperformance as either _reasoning_ failure or _matching_ failure.

### 8.1 Coverage Gate (Answer Evidence Presence)

For each question:

- Verify that at least one ingested doc corresponds to:
    - the answer entity page (for QID/title sources), **or**
    - an equivalent page containing answer string(s) (for NQ)

- If missing, allow a **single bounded fallback fetch**:
    - NQ: fetch the canonical Wikipedia page for the NQ topic if present
    - 2WIKIMULTIHOPQA: drop the question and move onto the next

### 8.3 Graph Gate (Entity Match Rate)

Compute:

- “% of queries with ≥1 entity match in the graph (post-normalization)”
- Average entities per chunk and per query

Low match rate indicates normalization/aliasing issues, not that Graph reasoning is unnecessary.

## 9) Rationale for Key Design Decisions (Why These Choices)

### 9.3 Why entity normalization is non-negotiable

GraphRAG’s failure mode is often _string mismatch_, not reasoning failure. A unified normalization policy (plus aliases/redirects) ensures GraphRAG has a fair chance to retrieve the right chunk neighbourhood.

## 10) Automation, Caching, and Reproducibility

### 10.1 Caching (Efficiency without Wikipedia dumps)

- Cache fetched Wikipedia pages by `page_id + revision_id`.
- Cache cleaned text + extracted anchors.
- Dedupe titles globally; many questions point to the same hubs.

### 10.2 Determinism Controls

- Fixed random seeds for sampling and any stochastic steps.
- Store corpus build config hash inside artifact metadata:
    - tokenization version
    - chunking parameters
    - regex versions
    - alias map version

### 10.3 Scripted Stages (recommended layout)

Matches the project’s staged workflow separation.

- `scripts/01_build_corpus.py` → produces corpus + indexes
- `scripts/02_run_oracle.py` → produces oracle labels (out of scope here)
- `scripts/03_train_router.py`
- `scripts/04_evaluate.py`

## 12) Deferred Enhancements (Not Implemented Yet)

The following items are intentionally deferred for Phase 1 to keep the pipeline
simple and reliable. Each item is valuable and can be added later without
changing the benchmark generation logic.

1. **NQ HTML reconstruction from tokens**: only needed when `document.html` is
   missing or corrupted.
2. **Caching of cleaned text + anchors**: docstore currently caches HTML and
   raw anchors only; cleaned text caching can be added later.

## 11) End-to-End Build Skeleton (Putting It All Together)

```python
def build_unified_corpus(samples):
    budgets = Budgets(max_pages_per_question=12, max_hops=2, max_list_pages=2, allow_country=True)

    all_docs = {}
    for ex in samples:
        elif ex.source == "2wiki":
            docs = ingest_2wiki(ex, budgets)  # supporting_facts.title only
        elif ex.source == "nq":
            docs = ingest_nq(ex, budgets)  # clean HTML, optionally add linked pages within same budget
        else:
            raise ValueError(ex.source)

        for d in docs:
            all_docs[d.doc_key] = d  # global dedupe

    alias_map = build_alias_map_from_redirects(all_docs.values())

    chunks = []
    for doc in all_docs.values():
        structured = canonical_clean(doc)                  # preserves headings/lists/tables
        blocks = structured.to_blocks()

        for ch in chunk_doc(blocks, tok=Tokenizer()):
            ch.metadata["years"] = extract_years(ch.text)
            ch.metadata.update(aggregate_year_fields(ch.metadata["years"], ch.text))

            ch.metadata["entities"] = extract_entities_spacy(ch.text, nlp, alias_map)
            chunks.append(ch)

    write_jsonl("data/processed/corpus.jsonl", chunks)

    build_faiss("data/processed/vector_index.faiss", chunks)
    write_metadata_table("data/processed/vector_meta.parquet", chunks)

    G = build_graph(chunks)
    pickle_dump("data/processed/graph.pkl", G)

    run_quality_gates(samples, chunks, G)
```
