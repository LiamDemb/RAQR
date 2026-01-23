from __future__ import annotations

from typing import Iterable, List

import networkx as nx


def build_graph(chunks: Iterable[dict], max_pairs: int = 50) -> nx.Graph:
    G = nx.Graph()
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            continue
        cnode = f"C:{chunk_id}"
        G.add_node(cnode, kind="chunk")

        entities = chunk.get("metadata", {}).get("entities", [])
        norms = []
        for ent in entities:
            norm = ent.get("norm")
            if not norm:
                continue
            enode = f"E:{norm}"
            G.add_node(enode, kind="entity", type=ent.get("type"))
            G.add_edge(enode, cnode, kind="mentions")
            norms.append(norm)

        pairs = 0
        for i in range(len(norms)):
            for j in range(i + 1, len(norms)):
                if pairs >= max_pairs:
                    break
                G.add_edge(f"E:{norms[i]}", f"E:{norms[j]}", kind="cooccur")
                pairs += 1
            if pairs >= max_pairs:
                break
    return G
