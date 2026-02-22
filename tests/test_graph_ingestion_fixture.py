"""Golden fixture tests for graph ingestion schema."""

from __future__ import annotations

from raqr.data.build_graph import build_graph


def test_build_graph_creates_expected_nodes_and_edges():
    chunks = [
        {
            "chunk_id": "c100",
            "metadata": {
                "entities": [
                    {"norm": "a", "type": "ORG"},
                    {"norm": "b", "type": "ORG"},
                ],
                "relations": [
                    {"subj_norm": "a", "pred": "caused", "obj_norm": "b"},
                ],
            },
        },
        {
            "chunk_id": "c200",
            "metadata": {
                "entities": [{"norm": "c", "type": "ORG"}],
                "relations": [],
            },
        },
    ]

    graph = build_graph(chunks)

    assert graph.nodes["E:a"]["kind"] == "entity"
    assert graph.nodes["E:b"]["kind"] == "entity"
    assert graph.nodes["C:c100"]["kind"] == "chunk"
    assert graph.nodes["C:c200"]["kind"] == "chunk"

    assert graph.has_edge("E:a", "E:b")
    assert graph["E:a"]["E:b"]["kind"] == "rel"
    assert graph["E:a"]["E:b"]["label"] == "caused"

    assert graph.has_edge("E:a", "C:c100")
    assert graph["E:a"]["C:c100"]["kind"] == "appears_in"
    assert graph.has_edge("E:b", "C:c100")
    assert graph["E:b"]["C:c100"]["kind"] == "appears_in"
