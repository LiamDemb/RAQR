"""Build router dataset rows from oracle_raw_scores items."""

from __future__ import annotations

import ast
from typing import Any, Iterable, Protocol

from raqr.evaluation.metrics import compute_max_em, compute_max_f1
from raqr.evaluation.oracle import determine_oracle_label


def _normalize_gold_answers(raw: list) -> list[str]:
    """Parse gold_answers into a flat list of strings."""
    result = []
    for item in raw or []:
        s = str(item).strip()
        if not s:
            continue
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                result.extend(str(x).strip() for x in parsed if str(x).strip())
            else:
                result.append(s)
        except (ValueError, SyntaxError):
            result.append(s)
    return result


class EmbedderLike(Protocol):
    def embed_query(self, text: str) -> "Any":
        ...


class ProbeLike(Protocol):
    def run(self, query: str) -> dict:
        ...


class QfeatNLPLike(Protocol):
    def __call__(self, query: str) -> Any:
        ...


def build_router_dataset_rows(
    items: Iterable[dict],
    *,
    embedder: EmbedderLike,
    probe: ProbeLike,
    compute_qfeat: "QfeatNLPLike",
    delta: float = 0.05,
) -> Iterable[dict]:
    """Build router dataset rows from oracle_raw_scores items.

    Each item must have: question_id, question, gold_answers, pred_dense, pred_graph,
    split, dataset_source (optional).

    Yields dicts with: question_id, question, split, dataset_source, question_embedding,
    gold_answers, gold_label, f1_dense, f1_graph, em_dense, em_graph, pred_dense, pred_graph,
    probe_scores, probe_max_score, probe_min_score, probe_score_sd, probe_skewness,
    probe_semantic_dispersion, entity_count, syntactic_depth, query_length_tokens,
    relational_keyword_flag.
    """
    for item in items:
        question = item.get("question", "").strip()
        if not question:
            continue

        question_id = item.get("question_id", "")
        split = item.get("split", "train")
        dataset_source = item.get("dataset_source", "")
        gold_raw = item.get("gold_answers", [])
        gold_answers = _normalize_gold_answers(gold_raw)
        pred_dense = item.get("pred_dense", "")
        pred_graph = item.get("pred_graph", "")

        # Embedding (normalized, as list of floats)
        q_emb = embedder.embed_query(question)
        question_embedding = q_emb.tolist() if hasattr(q_emb, "tolist") else list(q_emb)

        # Q-feat
        qfeat = compute_qfeat(question) if callable(compute_qfeat) else {}
        if isinstance(qfeat, dict):
            entity_count = qfeat.get("entity_count", 0)
            syntactic_depth = qfeat.get("syntactic_depth", 0)
            query_length_tokens = qfeat.get("query_length_tokens", 0)
            relational_keyword_flag = qfeat.get("relational_keyword_flag", 0)
        else:
            entity_count = syntactic_depth = query_length_tokens = 0
            relational_keyword_flag = 0

        # Probe signals
        probe_res = probe.run(question)

        # F1 / EM vs gold (max over golds)
        f1_dense = compute_max_f1(pred_dense, gold_answers)
        f1_graph = compute_max_f1(pred_graph, gold_answers)
        em_dense = compute_max_em(pred_dense, gold_answers)
        em_graph = compute_max_em(pred_graph, gold_answers)

        # Oracle label
        gold_label = determine_oracle_label(f1_dense, f1_graph, delta=delta)

        row = {
            "question_id": question_id,
            "question": question,
            "split": split,
            "dataset_source": dataset_source,
            "question_embedding": question_embedding,
            "gold_answers": gold_answers,
            "gold_label": gold_label,
            "f1_dense": f1_dense,
            "f1_graph": f1_graph,
            "em_dense": em_dense,
            "em_graph": em_graph,
            "pred_dense": pred_dense,
            "pred_graph": pred_graph,
            "probe_scores": probe_res.get("probe_scores", []),
            "probe_max_score": probe_res.get("probe_max_score", 0.0),
            "probe_min_score": probe_res.get("probe_min_score", 0.0),
            "probe_score_sd": probe_res.get("probe_score_sd", 0.0),
            "probe_skewness": probe_res.get("probe_skewness", 0.0),
            "probe_semantic_dispersion": probe_res.get("probe_semantic_dispersion", float("nan")),
            "entity_count": entity_count,
            "syntactic_depth": syntactic_depth,
            "query_length_tokens": query_length_tokens,
            "relational_keyword_flag": relational_keyword_flag,
        }
        yield row
