# RAQR Unified Corpus Creation Pipeline

**Goal:** Build a **Unified Mega Corpus** that enables a scientifically fair “Gold Label Oracle” evaluation across **Dense RAG**, **GraphRAG (NetworkX)**, and **Temporal RAG (year-filtered retrieval)** under a shared retrieval interface and shared artifacts (FAISS + NetworkX).  


## 0) Design Objectives and Non-Goals

### Objectives

1. **Unified**: All strategies operate over the *same* corpus and chunk inventory.
2. **Connective tissue**: The corpus must contain:

   * **multi-hop relational links** (to give GraphRAG a real advantage),
   * **temporal anchors** (`year` metadata; to give Temporal RAG a real advantage).
3. **Fairness**: No dataset gets special treatment; corpus expansion is driven by the same rules/budgets across datasets.
4. **Completeness**: For each sampled question, the corpus should contain answer-bearing evidence, even if it requires **1–2 hops**.
5. **Efficiency**: Must scale to ~1,500 questions without downloading full Wikipedia dumps.

### Non-goals

* Building a perfect KG (GraphRAG is intentionally “minimum viable,” NetworkX-based). 
* Exhaustive Wikipedia ingestion.


## 1) Inputs, Outputs, and Physical Artifacts

### Inputs (sampled question sets)

* **ComplexTempQA**: Wikidata QIDs (`question_entity`, `answer_entity`, etc.), no raw text.
* **WikiWhy**: Wikipedia `title` per question.
* **Natural Questions (NQ)**: raw HTML/tokens with `is_html` flags.

### Outputs (offline artifacts)

1. `data/processed/corpus.jsonl` — unified chunked corpus (schema below).
2. `data/processed/docstore.sqlite` (or LMDB) — page-level cache + chunk metadata (for dedupe and reproducibility).
3. `data/processed/vector_index.faiss` + `data/processed/vector_meta.parquet` — embeddings + metadata mapping (FAISS has no native metadata filtering, so store metadata externally).  
4. `data/processed/graph.pkl` — NetworkX graph (Entity-Relation-Entity edges + Entity-Chunk provenance edges).  
5. `data/processed/entity_lexicon.parquet` — canonical entity strings, aliases, and optional QIDs (used for normalization).

These artifacts align with the modular “offline prep vs online routing & inference” separation. 


## 2) Unified `corpus.jsonl` Schema

Each line is one **chunk**, not one document.

```json
{
  "chunk_id": "uuid",
  "doc_id": "uuid",
  "source": "wikipedia|nq|wikiwhy|complextempqa",
  "title": "string|null",
  "url": "string|null",
  "text": "string",
  "section_path": ["Lead", "Early life", "Career"],
  "char_span_in_doc": [1234, 1876],

  "metadata": {
    "dataset_origin": "nq|wikiwhy|complextempqa|wiki",
    "page_id": "string|null",
    "revision_id": "string|null",

    "years": [1998, 2001],
    "year_min": 1998,
    "year_max": 2001,
    "temporal_density": 0.014,

    "entities": [
      {"surface": "United States", "norm": "united states", "type": "GPE", "qid": "Q30"},
      {"surface": "Barack Obama", "norm": "barack obama", "type": "PERSON", "qid": "Q76"}
    ],

    "relations": [
      {"subj_norm": "barack obama", "pred": "born_in", "obj_norm": "united states"}
    ],

    "anchors": {
      "outgoing_titles": ["List of ...", "France", "2012 Summer Olympics"],
      "incoming_stub": []
    }
  }
}
```

**Notes**

* `years/year_min/year_max` enable Temporal RAG filtering.
* `entities[].norm` is the GraphRAG join key (string-canonical), with optional `qid` when resolvable.
* `relations[]` enables relation-aware 1-hop expansion: `Entity --predicate--> Entity`, with evidence resolved via `Entity -> Chunk` provenance edges.
* `anchors.outgoing_titles` is lightweight “connective tissue” to support multi-hop traversal without requiring a full hyperlink graph.


## 3) Pipeline Overview (Start-to-Finish)

### Stage A — Sampling & Budgeting

**Input:** ~750–1,500 questions balanced across datasets. 
**Output:** `data/interim/benchmark_sample.jsonl` (question-only; no gold strategy label).

**Fairness rule:** Every question receives the *same* maximum expansion budget:

* `MAX_PAGES_PER_QUESTION` (e.g., 12)
* `MAX_HOPS` (e.g., 2)
* `MAX_LIST_PAGES_PER_QUESTION` (e.g., 2)
* `MAX_COUNTRY_PAGES_PER_QUESTION` (e.g., 1)

### Stage B — Source-Specific Ingestion (Doc Acquisition)

Produce a set of **page candidates** per question. Dedupe globally by `page_id`/`title`.

### Stage C — Canonical Cleaning → Structured Document

Convert each acquired doc into:

* `doc.text` (clean prose, preserves headings/lists where possible),
* `doc.section_tree` (for chunk provenance),
* `doc.anchors` (outgoing titles).

### Stage D — Unified Enrichment Layer

Run **the same enrichment** on all docs:

* Regex date extraction → `years` metadata
* Entity extraction + normalization policy → `entities` metadata
* Relation extraction (semantic triples) → `relations` metadata (and graph edges)

### Stage E — Chunking & Serialization

Chunk to 500–1000 tokens with overlap; write `corpus.jsonl`.

### Stage F — Indexing

* Embed chunks → build FAISS
* Build NetworkX graph from extracted relations (triples) + provenance edges

### Stage G — Quality Gates (Required)

* Coverage and diagnostics to ensure Graph/Temporal have a fair chance (see Section 8).


## 4) Source-Specific Ingestion Specifications

### 4.1 ComplexTempQA (Wikidata QIDs → Wikipedia Text + Expansion)

**Problem:** QIDs have no raw evidence text.
**Requirement:** Resolve QIDs to Wikipedia pages and add **strategy-critical expansions** so Temporal/Graph aren’t starved.

#### 4.1.1 QID Resolution

For each question:

* Collect seed QIDs:

  * `question_entity` (and any additional linked QIDs present)
  * `answer_entity`
* Resolve each QID to:

  * Wikipedia title(s) via sitelinks
  * Optional redirects/aliases

**Acquisition:** Use Wikipedia API to fetch article plaintext (or HTML then clean) + section headers + links.

#### 4.1.2 Hop Expansion (Connective Tissue)

We create a *bounded* expansion set per question:

**Hop-0 (seed pages):**

* page(question_entity)
* page(answer_entity)

**Hop-1 (relational/context pages):**

* For each seed QID, pull a small set of “context QIDs” from Wikidata properties (bounded):

  * geographic: `country (P17)`, `located in admin entity (P131)`
  * membership/affiliation: `member of (P463)`, `part of (P361)`
  * time-bearing: `inception (P571)`, `point in time (P585)` → used to prioritize pages likely containing dates
* Resolve those context QIDs to pages.

**Hop-2 (optional, strict budget):**

* Only if remaining budget and query appears multi-hop (e.g., multiple entities detected).
* Expand via outgoing Wikipedia links from Hop-0 pages (top-N links by lead/infobox presence).

#### 4.1.3 Country Page Expansion (Temporal + Context)

**Why:** Many temporal questions require geopolitical context (“in France in 19xx…”, “under US administration…”) and these facts often live on **country pages** or their “History of …” subpages. This reduces failure cases where Dense finds a local snippet but Temporal/Graph need broader context.

**Rule (bounded, fairness-preserving):**

* If any seed/context entity resolves to a page whose Wikidata `P17` exists:

  * include **one** country page maximum per question (and optionally “History of {Country}” if it exists and budget allows).

#### 4.1.4 “List of …” Page Expansion (Temporal + Multi-Instance Evidence)

**Why:** List pages provide **high recall** for year-anchored events and enumerations (“List of earthquakes in 2011”, “List of prime ministers…”) and are often essential for Temporal RAG to outperform pure dense similarity on time-bound queries.

**Rule (bounded):**
Include up to `MAX_LIST_PAGES_PER_QUESTION` list pages chosen by:

1. If the seed page title matches common list triggers (contains “Olympics”, “elections”, “earthquakes”, “awards”, “episodes”), search for “List of {title}” variants.
2. From seed page outgoing links, include list pages that match regex: `^List of\b|^Timeline of\b|^(\d{4}) in\b`.
3. Prefer list pages whose extracted `years` overlap the query’s year/range (if detectable from question text).

#### 4.1.5 ComplexTempQA Ingestion Pseudocode

```python
def ingest_complextempqa(example, budgets):
    seed_qids = unique([example.question_entity, example.answer_entity] + example.other_qids)

    pages = set()
    for qid in seed_qids:
        pages |= resolve_qid_to_wikipedia_pages(qid)

    # Hop expansion using bounded Wikidata properties
    context_qids = set()
    for qid in seed_qids:
        context_qids |= bounded_wikidata_context(qid, limit=budgets.max_context_qids)

    for qid in context_qids:
        if len(pages) >= budgets.max_pages_per_question: break
        pages |= resolve_qid_to_wikipedia_pages(qid)

    # Country expansion (max 1)
    if budgets.allow_country:
        country_title = pick_country_page_from(seed_qids | context_qids)
        if country_title and len(pages) < budgets.max_pages_per_question:
            pages.add(country_title)

    # List expansion (max L)
    list_titles = propose_list_pages(pages, example.question_text, limit=budgets.max_list_pages)
    for t in list_titles:
        if len(pages) >= budgets.max_pages_per_question: break
        pages.add(t)

    return fetch_wikipedia_docs(pages)  # cached + deduped
```


### 4.2 Natural Questions (NQ) HTML Cleaning While Preserving Structure

**Problem:** NQ provides HTML/tokens with `is_html`. Naive stripping destroys headings, lists, and tables (which are valuable for multi-hop and temporal signals).

#### 4.2.1 Parsing Strategy

1. **Reconstruct HTML** using token sequence and `is_html` flags (when available), otherwise use the provided HTML field directly.
2. Parse with an HTML parser (e.g., BeautifulSoup/lxml).
3. Remove non-content blocks:

   * `script`, `style`, navigation, footers, edit links
4. Preserve document structure:

   * headings `h1–h6` → section nodes
   * `p` paragraphs
   * `ul/ol` lists (flatten with bullet markers)
   * tables: serialize rows as `|`-delimited lines (bounded to avoid massive tables)

#### 4.2.2 NQ Cleaning Pseudocode

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


### 4.3 WikiWhy (Title-Based Lookups + Contextual Breadth)

**Problem:** WikiWhy provides a title; the answer may require adjacent context (linked pages, broader causal background).

#### 4.3.1 Base Retrieval

* Fetch the Wikipedia page for the provided `title`.
* Also fetch **redirect target** if title resolves to redirect.

#### 4.3.2 Contextual Breadth (bounded)

To support “why/how” causal questions:

* Add up to `B` linked pages from:

  1. **Lead section outgoing links** (high signal)
  2. **Infobox entity fields** (if extractable via HTML)
  3. “History”, “Background”, “Causes”, “Aftermath” subpages if present as separate pages

**Fairness constraint:** Same hop/budget mechanism as ComplexTempQA; WikiWhy does not get extra pages—only different seeding.

#### 4.3.3 WikiWhy Ingestion Pseudocode

```python
def ingest_wikiwhy(example, budgets):
    pages = {resolve_title_or_redirect(example.title)}

    # Add bounded context pages from outgoing links (lead-biased)
    out = outgoing_links(pages, lead_only=True, limit=budgets.max_outgoing)
    for t in out:
        if len(pages) >= budgets.max_pages_per_question: break
        pages.add(t)

    # Optional: pick one "List of" if strongly relevant and budget remains
    for t in propose_list_pages(pages, example.question_text, limit=budgets.max_list_pages):
        if len(pages) >= budgets.max_pages_per_question: break
        pages.add(t)

    return fetch_wikipedia_docs(pages)
```


## 5) Unified Enrichment Layer (Applied to All Docs)

### 5.1 Regex Date Extraction → `year` Metadata for Temporal RAG

**Goal:** Populate `years`, `year_min`, `year_max`, and `temporal_density` per chunk.

#### 5.1.1 Supported Patterns (examples)

* Year: `\b(1[6-9]\d{2}|20\d{2}|2100)\b`
* Month Day, Year: `\b(Jan(uary)?|...)\s+\d{1,2},\s+(19\d{2}|20\d{2})\b`
* Year ranges: `\b(19\d{2}|20\d{2})\s*(–|-|to)\s*(19\d{2}|20\d{2})\b`
* Decades/centuries (optional, conservative mapping):

  * `1990s` → expand to [1990..1999] (store as range)
  * `21st century` → skip unless you explicitly want coarse ranges (recommend: skip to avoid noise)

#### 5.1.2 Chunk-level Aggregation

For each chunk:

* Extract all matched years (including expanded ranges with cap, e.g., max 15 years).
* `year_min/max` from extracted set
* `temporal_density = (#year_mentions) / (#tokens)` (or per 1k tokens)

#### 5.1.3 Date Extraction Pseudocode

```python
YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2}|2100)\b")
RANGE_RE = re.compile(r"\b(19\d{2}|20\d{2})\s*(?:–|-|to)\s*(19\d{2}|20\d{2})\b")

def extract_years(text, max_range_expand=15):
    years = set(int(y) for y in YEAR_RE.findall(text))

    for y1, y2 in RANGE_RE.findall(text):
        a, b = int(y1), int(y2)
        if a > b: a, b = b, a
        if (b - a) <= max_range_expand:
            years |= set(range(a, b+1))
        else:
            years |= {a, b}  # conservative if huge range

    return sorted(years)
```

**Why this matters:** Temporal RAG relies on metadata filtering; without robust year signals, it degenerates into Dense RAG with extra steps. 


### 5.2 Entity & Relation Extraction Policy (GraphRAG Join-Key Reliability)

**Goal:** GraphRAG success depends on matching query entities to corpus entities. Surface-form mismatch (“U.S.” vs “United States”) can collapse GraphRAG’s recall; we must normalize entities consistently across all sources. 

#### 5.2.1 Extraction (Entities)

* Run spaCy NER over **chunk text** (and later over queries at retrieval time).
* Keep entity types: `PERSON`, `ORG`, `GPE`, `LOC`, `WORK_OF_ART`, `EVENT` (configurable).

#### 5.2.2 Relation Extraction (Semantic Triples)

**Goal:** Replace shallow “co-mentions” with explicit directed relations to support multi-hop reasoning.

* Run a lightweight Relation Extraction (RE) model over each chunk to extract triples \((subject, predicate, object)\).
* Normalize `subject` and `object` using the same entity normalization policy (below).
* Store triples in chunk metadata as `relations` (for auditability) and use them to build directed semantic edges in the graph.
* Example models:
  * `Babelscape/rebel-large` (transformers-based RE)
  * GLiNER relation models (spaCy pipeline integration, if preferred)

#### 5.2.3 Normalization (deterministic)

Apply the same function everywhere:

1. Unicode normalize (NFKC), lowercase
2. Strip punctuation except internal hyphens
3. Collapse whitespace
4. Remove possessives: `"obama's" → "obama"`
5. Alias map (small, curated + auto-mined):

   * curated: `{ "u.s.": "united states", "us": "united states", "uk": "united kingdom" }`
   * auto-mined from Wikipedia redirects (when available from fetched pages)

**Optional (recommended if available without heavy infra):**

* If a chunk originates from a resolved Wikipedia page with a known QID, tag that page-title entity with `qid` and propagate.

#### 5.2.4 Entity Lexicon Build

Create `entity_lexicon.parquet`:

* `norm` (primary key)
* `surface_forms` (top-k frequent)
* `qid_candidates` (if any)
* `df` document frequency (for downweighting extremely common entities)

#### 5.2.5 Entity Policy Pseudocode

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


## 6) Chunking Strategy (500–1000 tokens + overlap)

### 6.1 Requirements

* Chunk size: **500–1000 tokens**
* Overlap: **~10–15%** (e.g., 100–150 tokens)
* Preserve section boundaries where possible (don’t merge unrelated sections)

### 6.2 Algorithm

1. Split doc into blocks (paragraph/list/table) with section paths.
2. Accumulate blocks until token budget reached.
3. When exceeding:

   * finalize chunk
   * start next chunk with overlap tail (token-based, not block-based if needed)

### 6.3 Chunking Pseudocode

```python
def chunk_doc(blocks, tok, min_t=500, max_t=1000, overlap_t=120):
    chunks = []
    buf, buf_tokens = [], 0

    for b in blocks:
        t = tok.count(b.text)
        if buf_tokens + t > max_t and buf_tokens >= min_t:
            chunk_text = join_blocks(buf)
            chunks.append(make_chunk(chunk_text, buf))
            tail = tok.tail(chunk_text, overlap_t)
            buf = [Block(text=tail, section_path=buf[-1].section_path)]
            buf_tokens = tok.count(tail)

        buf.append(b)
        buf_tokens += t

    if buf:
        chunks.append(make_chunk(join_blocks(buf), buf))

    return chunks
```


## 7) Indexing Artifacts

### 7.1 FAISS Vector Index (Dense + Temporal Candidate Pool)

* Embed every chunk using a single embedding model (e.g., `all-MiniLM-L6-v2`) for consistency. 
* Store:

  * FAISS index (vectors)
  * Separate metadata table keyed by FAISS row id → `chunk_id`, `year_min/max`, `years`, etc.

**Temporal RAG compatibility:** retrieve top-N (N≫k) then filter by year; refill from deeper ranks as needed. 

### 7.2 NetworkX Relationship Graph (GraphRAG)

Graph schema (minimum viable): 

* Nodes:

  * `E:{norm_entity}` (Entity nodes)
  * `C:{chunk_id}` (Chunk nodes)
* Edges:

  * **Semantic edges (primary):** `E --predicate--> E` from extracted triples \((subject, predicate, object)\)
  * **Provenance edges:** `E --> C` (`appears_in`) if entity occurs in chunk (optionally weighted by frequency)

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

RAQR’s Oracle labels will otherwise default to Dense RAG if Graph/Temporal lack evidence or linkage. The following diagnostics must be computed after ingestion and before Oracle runs. This aligns with the project’s controlled, oracle-based methodology and the need to interpret Graph underperformance as either *reasoning* failure or *matching* failure.   

### 8.1 Coverage Gate (Answer Evidence Presence)

For each question:

* Verify that at least one ingested doc corresponds to:

  * the answer entity page (for QID/title sources), **or**
  * an equivalent page containing answer string(s) (for NQ)
* If missing, allow a **single bounded fallback fetch**:

  * ComplexTempQA: re-resolve `answer_entity` + redirects
  * WikiWhy: follow redirect + disambiguation pick (first non-disambig)
  * NQ: fetch the canonical Wikipedia page for the NQ topic if present

### 8.2 Temporal Gate (Year Signal Adequacy)

Compute:

* `% chunks with ≥1 year`
* Distribution of `temporal_density`
* For queries with explicit year/range terms: `% with at least one candidate chunk overlapping year range`

Failing this gate means Temporal RAG cannot win on merit.

### 8.3 Graph Gate (Entity Match Rate)

Compute:

* “% of queries with ≥1 entity match in the graph (post-normalization)”  
* Average entities per chunk and per query

Low match rate indicates normalization/aliasing issues, not that Graph reasoning is unnecessary.


## 9) Rationale for Key Design Decisions (Why These Choices)

### 9.1 Why “List of …” expansion is necessary

Without list/timeline pages, many temporal questions reduce to sparse mentions scattered across unrelated articles. Dense retrieval often “wins by convenience” (one good snippet), while Temporal RAG can’t exploit time filters because year-tagged enumerations aren’t present. Adding bounded list pages creates high-recall, year-anchored evidence—critical for a fair oracle comparison.

### 9.2 Why country expansion is necessary

Country pages and their history/politics sections act as *connectors* across events, offices, conflicts, and periods. They are “low-cost hubs” that improve both:

* Temporal reasoning (periodization, administrations)
* Graph traversal (shared geopolitical entities)

### 9.3 Why entity normalization is non-negotiable

GraphRAG’s failure mode is often *string mismatch*, not reasoning failure. A unified normalization policy (plus aliases/redirects) ensures GraphRAG has a fair chance to retrieve the right chunk neighborhood. 

### 9.4 Why unified enrichment prevents dataset favoritism

If only ComplexTempQA gets QID-based entity IDs, GraphRAG becomes biased toward that dataset. By extracting entities and years uniformly from *all* ingested text, every dataset contributes equally to graph and temporal signals.


## 10) Automation, Caching, and Reproducibility

### 10.1 Caching (Efficiency without Wikipedia dumps)

* Cache fetched Wikipedia pages by `page_id + revision_id`.
* Cache cleaned text + extracted anchors.
* Dedupe titles globally; many questions point to the same hubs.

### 10.2 Determinism Controls

* Fixed random seeds for sampling and any stochastic steps. 
* Store corpus build config hash inside artifact metadata:

  * tokenization version
  * chunking parameters
  * regex versions
  * alias map version

### 10.3 Scripted Stages (recommended layout)

Matches the project’s staged workflow separation. 

* `scripts/01_build_corpus.py` → produces corpus + indexes
* `scripts/02_run_oracle.py` → produces oracle labels (out of scope here)
* `scripts/03_train_router.py`
* `scripts/04_evaluate.py`

## 12) Deferred Enhancements (Not Implemented Yet)
The following items are intentionally deferred for Phase 1 to keep the pipeline
simple and reliable. Each item is valuable and can be added later without
changing the benchmark generation logic.

1. **ComplexTempQA expansion depth**: deeper hop logic and richer list/country
   selection rules beyond the current bounded approach.
2. **NQ HTML reconstruction from tokens**: only needed when `document.html` is
   missing or corrupted.
3. **Stricter WikiWhy selection logic**: lead/infobox‑biased link selection and
   targeted “Background/History/Causes” subpages.
4. **Caching of cleaned text + anchors**: docstore currently caches HTML and
   raw anchors only; cleaned text caching can be added later.
5. **Query‑aware quality gates**: current gates are chunk‑level only; query‑level
   overlap checks are deferred. Later upgrade should add (a) year‑range overlap
   checks for explicitly temporal questions and (b) per‑query entity‑match rates
   against the graph to distinguish normalization failures from reasoning failures.


## 11) End-to-End Build Skeleton (Putting It All Together)

```python
def build_unified_corpus(samples):
    budgets = Budgets(max_pages_per_question=12, max_hops=2, max_list_pages=2, allow_country=True)

    all_docs = {}
    for ex in samples:
        if ex.source == "complextempqa":
            docs = ingest_complextempqa(ex, budgets)
        elif ex.source == "wikiwhy":
            docs = ingest_wikiwhy(ex, budgets)
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