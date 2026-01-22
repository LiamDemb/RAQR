# RAQR Executive Summary

**Project Title:** Reasoning-Aware Query Routing (RAQR): Dynamic Retrieval Selection for Mixed-Reasoning QA

**Focus:** A rigorous ablation study of query routing mechanisms for Retrieval-Augmented Generation (RAG).


## 1. Overview
RAQR investigates the decision-making process within composite AI systems. Rather than building a new RAG architecture, this project focuses on the **Router**, the component responsible for directing a user’s query to the most appropriate retrieval strategy.

The core objective is to move beyond static heuristics and prompt-based guesswork. We aim to empirically determine **what information** a router needs to make good decisions and **what model class** is most efficient at making them.

## 2. The Problem
Retrieval-Augmented Generation is not "one size fits all". A dense vector lookup works well for factual queries, while Knowledge Graphs (GraphRAG) excel at multi-hop reasoning.
*   **Current State:** Most RAG systems force all queries through a single pipeline, or use routers to decide between different knowledge domain experts.
*   **The Gap:** There is little systematic research on constructing a specialised router to route queries to different retrieval strategies / frameworks.

## 3. The Approach: A Controlled Ablation Study
We frame the research as a controlled experiment using a "Gold Label" (Oracle) methodology.

1.  **The "Oracle" Workflow:** We run every question in our benchmark through all available strategies.
2.  **Scoring:** We score the answers against ground truth.
3.  **Labelling:** The strategy that produces the best answer becomes the "target label" for that question.
4.  **Training:** We train various routers to predict this label using different input signals.

### The Retrieval Strategies (The Choices)
To ensure broad coverage of reasoning types, the system routes between five distinct behaviours, each utilising a different knowledge representation:

1.  **Dense RAG:** The baseline vector-chunk retrieval; optimal for specific, local factual lookups.
2.  **Hierarchical RAG:** Tree-structured summarisation (e.g., RAPTOR) for high-level global synthesis.
3.  **GraphRAG:** Entity-relationship traversal for complex, multi-hop reasoning across documents.
4.  **Temporal RAG:** Time-indexed retrieval for chronological, trend-based, or "as-of" queries.
5.  **Table RAG:** Structured schema extraction for aggregation, sorting, and statistical queries.

## 4. The Variables (What We Are Testing)

We are conducting a two-dimensional ablation study:

### Dimension A: Input Signals (Information Sources)
*   **Question Only:** The router sees only the user's query text.
*   **Retrieval Feedback:** The router sees metrics derived from a "cheap probe" retrieval. Specific signals tested include **Score Skewness** (distribution shape), **Max Score** (top result confidence), and **Semantic Distance**.
*   **Combined:** The router uses both question semantics and retrieval feedback signals.

### Dimension B: Model Class (Implementation)
*   **Heuristic:** Deterministic, training-free rules (e.g., "If skewness < X, use GraphRAG").
*   **Lightweight Classifier:** A supervised BERT-style encoder trained on the Oracle labels.
*   **LLM Router:** A prompted Large Language Model relying on in-context reasoning.

## 5. Research Questions & Hypotheses

*   **RQ1:** Does adding retrieval feedback signals improve routing accuracy compared to looking at the question text alone?
    *   *Hypothesis:* **Yes.** Signal-aware routers will significantly outperform question-only routers because they can detect when a specific retrieval method is failing (e.g., low confidence scores).
*   **RQ2:** Can a small, trained classifier match the performance of an expensive LLM router?
    *   *Hypothesis:* **Yes.** A lightweight classifier trained on combined signals will match LLM accuracy while being faster and more stable.
*   **RQ3:** How do routing errors impact downstream QA performance?

## 6. Technical Scope
*   **Language:** Python.
*   **Core Libraries:** PyTorch, Hugging Face Transformers (for the lightweight router), LangChain/LangGraph (orchestration).
*   **Data Stores:** FAISS (Vector), Neo4j/NetworkX (Graph), SQL/Pandas (Table).
*   **Evaluation:** Exact Match (EM), F1 Score, and Routing Accuracy.

## 7. Expected Contribution
This project provides a scientific foundation for building adaptive RAG systems. Instead of guessing which tool to use, we provide an evidence-based taxonomy of **when** to route, **why** specific strategies fail, and **how** to build efficient routers that don't rely on expensive LLM calls for every decision.