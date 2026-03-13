# Wikipedia Fetching and Cleaning Pipeline

This doc describes how Wikipedia articles are fetched, cleaned, and chunked for the RAQR corpus. It covers the fetch step, HTML cleaning rules, and the normalize-for-extraction policy for downstream NLP.

---

## 1. Fetch: Why `action=parse` HTML?

Wikipedia pages are fetched via the MediaWiki API using `action=parse` with `prop=text|links|revid`. This returns:

- **Rendered HTML** of the page content (what readers see)
- **Outgoing links** (ns=0 article links) for multi-hop expansion
- **Page/revision IDs** for caching and reproducibility

We use parsed HTML rather than plaintext extracts because:

1. We need **structure** (headings, paragraphs, lists, tables) for section-aware chunking.
2. Outgoing links come from the same API call, avoiding a second request.
3. The docstore caches raw HTML; cleaning runs at corpus-build time, so we can iterate on cleaning rules without re-fetching.

**Source:** `src/raqr/data/wikipedia_client.py` — `WikipediaClient.fetch_html()`

---

## 2. Cleaning Flow

```
Raw HTML (parse API)
    → Drop tags + decompose noisy nodes
    → Restrict to div.mw-parser-output
    → Extract blocks (p, ul, ol, table)
    → Skip lists inside tables
    → Chunk blocks
```

**Source:** `src/raqr/data/canonical_clean.py` — `clean_html_to_structured_doc()`

---

## 3. What Is Dropped (Decomposed Before Extraction)

### 3.1 Whole tags

- `script`, `style`, `nav`, `footer`, `aside`

### 3.2 Citations and references

- `sup.reference`, `sup.mw-ref`, `span.mw-ref`, `a.mw-ref`
- `ol.references`, `div.reflist`, `div#mw-references-wrap`

### 3.3 Table of contents and navigation

- `#toc`, `.toc`, `.mw-toc`, `.vector-toc`
- `.navbox`, `.vertical-navbox`, `.navbox-inner`

### 3.4 Metadata and maintenance

- `.metadata`, `.ambox`, `.mbox-small`, `.hatnote`, `.dablink`
- `#catlinks` (category links)
- `.mw-editsection` (edit section links)

### 3.5 Pronunciation and IPA

- `.IPA`, `.ext-phonos`, `span.IPA`, `sup.IPA`
- Links to `Help:IPA`, `Help:Pronunciation` (via `a[title^="..."]`)

---

## 4. What Is Preserved

- **Paragraphs** (`<p>`)
- **Headings** (`<h1>`–`<h6>`) — used for section paths
- **Lists** (`<ul>`, `<ol>`) — except when nested **inside** tables (e.g. infobox plainlists)
- **Tables** — converted to `cell | cell | …` rows, capped at 30 rows × 8 cols; supports `table > tbody > tr` structure

Extraction is restricted to `div.mw-parser-output` (main article content). If that container is missing, the parser falls back to `body`.

---

## 5. Normalize-for-Extraction Policy

**Stored chunk text** in `corpus.jsonl` keeps original formatting (paragraph breaks, list newlines) for readability and retrieval.

**Entity and relation extraction** (LLM IE batch) receive a normalized copy instead:

- Collapse whitespace (replace newlines/multiple spaces with a single space)
- Strip common Wikipedia artifacts (IPA brackets, `(listen)`)

This reduces noise for NLP models (e.g. stray IPA symbols, fragmented lines) while preserving readable text in the corpus.

**Helper:** `normalize_text_for_extraction()` in `src/raqr/data/canonical_clean.py`  
**Usage:** `scripts/01_build_corpus.py` — applied when running extraction (LLM IE batch)

---

## 6. Table Handling

- Tables use `table > tbody > tr` (and `thead`/`tfoot`) correctly; rows are identified as those whose nearest `table` ancestor is the current table (avoids nested-table leakage).
- Lists inside tables (e.g. infobox “Parents”, “Siblings”) are **skipped** and not emitted as corpus blocks.
- Navbox-style tables are dropped via `.navbox` selectors in the decompose step.

---

## 7. Configuration

All cleaning rules live in `canonical_clean.py` as constants:

- `DROP_TAGS`
- `CITATION_SELECTORS`
- `NOISE_SELECTORS` (TOC, navboxes, metadata, IPA)
- `NOISE_LINK_SELECTORS` (IPA/pronunciation links)

Tuning selectors can be done there without changing the overall pipeline logic.
