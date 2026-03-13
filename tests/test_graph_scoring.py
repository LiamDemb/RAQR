from raqr.data.build_graph import build_graph
from raqr.graph_scoring import score_bundle
from raqr.graph_types import EvidenceBundle, GraphHop, GraphPath, GroundedHop


class TinyEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors

    def embed_query(self, text: str):
        if text not in self.vectors:
            raise KeyError(f"Missing vector for: {text}")
        return self.vectors[text]


class CorpusStub:
    def __init__(self, texts):
        self.texts = texts

    def get_text(self, chunk_id):
        return self.texts.get(chunk_id)


def _bundle(*hops: GraphHop, chunk_ids=None):
    if chunk_ids is None:
        chunk_ids = ["c1"]

    return EvidenceBundle(
        path=GraphPath(start_node=hops[0].source, hops=tuple(hops)),
        grounded_hops=[
            GroundedHop(
                hop=hop,
                chunk_id=chunk_ids[min(i, len(chunk_ids) - 1)],
                support_score=1.0,
            )
            for i, hop in enumerate(hops)
        ],
        supporting_chunk_ids=list(chunk_ids),
    )


def test_score_bundle_prefers_relation_semantically_closer_to_query():
    graph = build_graph([])
    corpus = CorpusStub(
        {
            "c1": "The film was directed by John Smith.",
            "c2": "The country of origin is France.",
        }
    )
    embedder = TinyEmbedder(
        {
            "director death question": [1.0, 0.0, 0.0],
            "directed_by": [1.0, 0.0, 0.0],
            "country": [0.0, 1.0, 0.0],
            "The film was directed by John Smith.": [1.0, 0.0, 0.0],
            "The country of origin is France.": [0.0, 1.0, 0.0],
        }
    )

    better = _bundle(
        GraphHop(source="E:film", relation="directed_by", target="E:person"),
        chunk_ids=["c1"],
    )
    worse = _bundle(
        GraphHop(source="E:film", relation="country", target="E:usa"),
        chunk_ids=["c2"],
    )

    better_score, _ = score_bundle(
        query="director death question",
        bundle=better,
        graph=graph,
        corpus=corpus,
        embedder=embedder,
        length_penalty=0.75,
        cache={},
    )
    worse_score, _ = score_bundle(
        query="director death question",
        bundle=worse,
        graph=graph,
        corpus=corpus,
        embedder=embedder,
        length_penalty=0.75,
        cache={},
    )

    assert better_score > worse_score


def test_score_bundle_penalizes_longer_paths_when_relation_relevance_is_equal():
    graph = build_graph([])
    corpus = CorpusStub(
        {
            "c1": "The film was directed by John Smith.",
            "c2": "John Smith died in 1970.",
        }
    )
    embedder = TinyEmbedder(
        {
            "same query": [1.0, 0.0, 0.0],
            "r": [1.0, 0.0, 0.0],
            "The film was directed by John Smith.": [1.0, 0.0, 0.0],
            "John Smith died in 1970.": [1.0, 0.0, 0.0],
        }
    )

    short_bundle = _bundle(
        GraphHop(source="E:a", relation="r", target="E:b"),
        chunk_ids=["c1"],
    )
    long_bundle = _bundle(
        GraphHop(source="E:a", relation="r", target="E:b"),
        GraphHop(source="E:b", relation="r", target="E:c"),
        chunk_ids=["c1", "c2"],
    )

    short_score, _ = score_bundle(
        query="same query",
        bundle=short_bundle,
        graph=graph,
        corpus=corpus,
        embedder=embedder,
        length_penalty=0.75,
        cache={},
    )
    long_score, _ = score_bundle(
        query="same query",
        bundle=long_bundle,
        graph=graph,
        corpus=corpus,
        embedder=embedder,
        length_penalty=0.75,
        cache={},
    )

    assert short_score > long_score


def test_score_bundle_returns_breakdown_with_expected_keys():
    graph = build_graph([])
    corpus = CorpusStub({"c1": "The film was directed by John Smith."})
    embedder = TinyEmbedder(
        {
            "query": [1.0, 0.0, 0.0],
            "directed_by": [1.0, 0.0, 0.0],
            "The film was directed by John Smith.": [1.0, 0.0, 0.0],
        }
    )
    bundle = _bundle(
        GraphHop(source="E:film", relation="directed_by", target="E:person"),
        chunk_ids=["c1"],
    )

    score, breakdown = score_bundle(
        query="query",
        bundle=bundle,
        graph=graph,
        corpus=corpus,
        embedder=embedder,
        length_penalty=0.75,
        cache={},
    )

    assert isinstance(score, float)
    assert "s_rel" in breakdown
    assert "s_len" in breakdown
