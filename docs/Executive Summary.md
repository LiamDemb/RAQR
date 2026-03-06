# RAQR Executive Summary

**Project Title:** Reasoning-Aware Query Routing (RAQR): Dynamic Retrieval Selection for Mixed-Reasoning QA

**Focus:** A rigorous ablation study of query routing mechanisms for different Retrieval-Augmented Generation (RAG) strategies.

## 1. Overview

The growing body of Retrieval-Augmented Generation (RAG) research has produced a range of frameworks tailored to specific reasoning patterns and retrieval objects. Although these systems have been successful in boosting performance at their given task, they tend to underperform when confronted with fundamentally different query structures. For example, graph-structured RAG, while effective for multi-hop reasoning, have been shown to underperform on simpler 0-hop factual questions. This motivates the exploration of adaptive retrieval strategies capable of aligning query characteristics with the most suitable retrieval mechanism.

RAQR investigates the **Router**, the component responsible for directing a user’s query to the most appropriate retrieval strategy.

The core objective is to move beyond static heuristics and prompt-based guesswork. I aim to empirically determine **what information** a router needs to make good decisions and **what model class** is most efficient at making them.

## 2. The Problem

RAG is not "one size fits all". A dense vector lookup works well for factual queries, while Knowledge Graphs (GraphRAG) excel at multi-hop reasoning.

- **Current State:** Most RAG systems force all queries through a single pipeline, or use routers to decide between different knowledge domain experts.
    - `RAG vs. GraphRAG` explored the idea of routing between static RAG and GraphRAG and showed impressive performance improvements when utilising routing
    - However, this was just a simple surface-level experiment – I want to dive deeper into optimising and exploring the routing mechanism
- **The Gap:** There is little systematic research on constructing a specialised router to route queries to different retrieval strategies / frameworks.

## 3. Approach

### Positioning

RAQR differs from prior work that routes purely on linguistic "query complexity" (e.g. EA-GraphRAG-style pipelines that classify queries by surface cues). Instead, RAQR routes by **reasoning needs** (temporal, causal, factual) combined with **dynamic probe signals**—metrics derived from a cheap top-10 Dense retrieval run. This allows the router to detect when a retrieval method is failing (e.g. low confidence scores or flat score distributions) rather than relying solely on query semantics.

The research project is framed as a controlled experiment using a "Gold Label" (Oracle) methodology.

1. **The "Oracle" Workflow:** Run every question in our benchmark through all available strategies.
2. **Scoring:** We score the answers against ground truth.
3. **Labelling:** The strategy that produces the best answer becomes the "target label" for that question.
4. **Training:** We train various routers to predict this label using different input signals.

### The Retrieval Strategies (The Choices)

To ensure broad coverage of reasoning types, the system routes between three distinct behaviours, each utilising a different knowledge representation:

1. **Dense RAG:** The baseline vector-chunk retrieval; optimal for specific, local factual lookups.
2. **GraphRAG:** Relation-aware traversal using **Subject-Predicate-Object triples** for complex, multi-hop reasoning across documents (no entity-only graph; triples are the ground truth).
3. **Temporal RAG:** **Metadata-filtered dense retrieval** (vector search + year filtering)—no separate temporal Knowledge Graph; Temporal RAG is vector search with year metadata filters.

## 4. The Variables (What’s Being Tested)

This research contains a two-dimensional ablation study:

### Input Signals

- **Q-Emb:** DistilBERT [CLS] embedding (768 dims) of the query
- **Q-Feat:** Engineered features (length/token count, entity density from query entity extraction, complexity keywords; optional syntax depth)
- **Probe:** Top-10 Dense retrieval signals—max score, skewness, semantic dispersion (alias: semantic distance)
- **Combined:** The router uses Q-Emb + Q-Feat + Probe together

### Model Class

- **Heuristic:** Deterministic, training-free rules (focus on Regex and keywords)
- **Lightweight Classifier:** A supervised BERT-style encoder trained on the Oracle labels
- **LLM Router:** A prompted Large Language Model relying on in-context reasoning

## 5. Research Questions & Hypotheses

- **RQ1:** Does adding retrieval feedback signals improve routing accuracy compared to looking at the query embedding and features alone?
    - _Hypothesis:_ **Yes.** Signal-aware routers will significantly outperform question-only routers because they can detect when a specific retrieval method is failing (e.g., low confidence scores).
- **RQ2:** Can a small, trained classifier match the performance of an expensive LLM router?
    - _Hypothesis:_ **Yes.** A lightweight classifier trained on combined signals will match LLM accuracy while being **faster and more stable**
- **RQ3:** How do routing errors impact downstream QA performance?
    - Test the performance of the whole system compared to just GraphRAG for e.g.?

## 6. Expected Contribution

- **A Systematic Evaluation of Information Sources:** Determining if query semantics or retrieval metadata is the superior signal for RAG orchestration.
- **Efficiency Frontier Analysis:** Demonstrating that lightweight classifiers can achieve near-SOTA accuracy at a fraction of the cost of LLM-based routing.
- **Impact & Regret Taxonomy:** A quantitative analysis of **Routing Regret**—defined as $\text{regret} = \text{oracle}_\text{F1} - \text{selected}_\text{F1}$, i.e. the F1 loss incurred when the router selects a strategy different from the Oracle’s best. We identify which reasoning mismatches are "forgivable" (marginal performance loss) versus "fatal" (severe system failure), providing a severity taxonomy and a safety-first roadmap for RAG design.
