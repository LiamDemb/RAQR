from __future__ import annotations

from typing import Iterable, List

import networkx as nx


def build_graph(chunks: Iterable[dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            continue
        cnode = f"C:{chunk_id}"
        G.add_node(cnode, kind="chunk")

        entities = chunk.get("metadata", {}).get("entities", [])
        for ent in entities:
            norm = ent.get("norm")
            if not norm:
                continue
            enode = f"E:{norm}"
            G.add_node(enode, kind="entity", type=ent.get("type"))
            G.add_edge(enode, cnode, kind="appears_in")

        relations = chunk.get("metadata", {}).get("relations", []) or []
        for rel in relations:
            subj = rel.get("subj_norm") or rel.get("subject_norm")
            obj = rel.get("obj_norm") or rel.get("object_norm")
            pred = rel.get("pred") or rel.get("predicate")
            if not subj or not obj or not pred:
                continue
            snode = f"E:{subj}"
            onode = f"E:{obj}"
            G.add_node(snode, kind="entity")
            G.add_node(onode, kind="entity")
            G.add_edge(snode, onode, kind="rel", label=pred)
    return G
