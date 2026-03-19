"""Oracle label determination for router dataset (token-F1 with margin)."""


def determine_oracle_label(
    dense_f1: float,
    graph_f1: float,
    delta: float = 0.05,
) -> str:
    """Determine gold label (Dense or Graph) from token-level F1 scores.

    Graph is chosen only if graph_f1 beats dense_f1 by at least delta.
    Ties and near-ties go to Dense.

    Args:
        dense_f1: F1 of Dense prediction vs gold.
        graph_f1: F1 of Graph prediction vs gold.
        delta: Required margin for Graph to win (default 0.05).

    Returns:
        "Dense" or "Graph".
    """
    if graph_f1 - dense_f1 >= delta:
        return "Graph"
    return "Dense"
