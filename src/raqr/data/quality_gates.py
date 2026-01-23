from __future__ import annotations

import json
from typing import Dict, Iterable, List


def coverage_gate(samples: List[dict], chunks: List[dict]) -> Dict[str, float]:
    matched = 0
    for sample in samples:
        answers = sample.get("gold_answers") or sample.get("answers") or []
        answers = [str(a).strip().lower() for a in answers if str(a).strip()]
        found = False
        if answers:
            for chunk in chunks:
                text = chunk.get("text", "").lower()
                if any(ans in text for ans in answers):
                    found = True
                    break
        if found:
            matched += 1
    return {"coverage_rate": matched / max(len(samples), 1)}


def temporal_gate(chunks: List[dict]) -> Dict[str, float]:
    with_year = 0
    densities = []
    for chunk in chunks:
        meta = chunk.get("metadata", {})
        years = meta.get("years") or []
        if years:
            with_year += 1
        density = meta.get("temporal_density")
        if density is not None:
            densities.append(float(density))
    return {
        "chunks_with_year_rate": with_year / max(len(chunks), 1),
        "temporal_density_avg": sum(densities) / max(len(densities), 1),
    }


def graph_gate(chunks: List[dict]) -> Dict[str, float]:
    ent_counts = []
    for chunk in chunks:
        ents = chunk.get("metadata", {}).get("entities", []) or []
        ent_counts.append(len(ents))
    avg_ents = sum(ent_counts) / max(len(ent_counts), 1)
    return {"avg_entities_per_chunk": avg_ents}


def run_quality_gates(
    samples: List[dict],
    chunks: List[dict],
    output_path: str,
    min_coverage: float = 0.2,
    min_year_rate: float = 0.1,
) -> Dict[str, float]:
    report = {}
    report.update(coverage_gate(samples, chunks))
    report.update(temporal_gate(chunks))
    report.update(graph_gate(chunks))

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    if report["coverage_rate"] < min_coverage:
        raise ValueError("Coverage gate failed. Low answer evidence rate.")
    if report["chunks_with_year_rate"] < min_year_rate:
        raise ValueError("Temporal gate failed. Too few year-tagged chunks.")
    return report
