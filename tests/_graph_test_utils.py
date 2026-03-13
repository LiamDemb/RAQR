from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from raqr.generator import GenerationResult


@dataclass
class StaticExtractor:
    entities: List[str]

    def extract(self, query: str) -> List[str]:
        return list(self.entities)


@dataclass
class GraphStoreStub:
    graph: object

    def load(self):
        return self.graph


class CorpusStub:
    def __init__(self, texts: dict[str, str]):
        self._texts = texts

    def get_text(self, chunk_id: str) -> Optional[str]:
        return self._texts.get(chunk_id)


class GeneratorStub:
    def generate(self, query: str, context: List[str]) -> GenerationResult:
        return GenerationResult(
            text=f"mocked answer for: {query}",
            model_id="test",
            latency_ms=1.0,
            prompt_hash="abc",
            sampling={},
        )


class TinyEmbedder:
    def __init__(self, vectors: dict[str, list[float]]):
        self._vectors = vectors

    def embed_query(self, text: str):
        if text not in self._vectors:
            raise KeyError(f"Missing vector for: {text}")
        return self._vectors[text]



def toy_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "c1",
            "metadata": {
                "entities": [{"norm": "a", "type": "ORG"}, {"norm": "b", "type": "ORG"}],
                "relations": [{"subj_norm": "a", "pred": "causes", "obj_norm": "b"}],
            },
        },
        {
            "chunk_id": "c2",
            "metadata": {
                "entities": [{"norm": "b", "type": "ORG"}, {"norm": "c", "type": "ORG"}],
                "relations": [{"subj_norm": "b", "pred": "causes", "obj_norm": "c"}],
            },
        },
    ]



def comparison_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "c1",
            "metadata": {
                "entities": [
                    {"norm": "a daughter of two worlds"},
                    {"norm": "james young"},
                ],
                "relations": [
                    {
                        "subj_norm": "a daughter of two worlds",
                        "pred": "directed_by",
                        "obj_norm": "james young",
                    }
                ],
            },
        },
        {
            "chunk_id": "c2",
            "metadata": {
                "entities": [
                    {"norm": "james young"},
                    {"norm": "date_1948"},
                ],
                "relations": [
                    {
                        "subj_norm": "james young",
                        "pred": "died_on",
                        "obj_norm": "date_1948",
                    }
                ],
            },
        },
        {
            "chunk_id": "c3",
            "metadata": {
                "entities": [
                    {"norm": "valentin the good"},
                    {"norm": "martin fric"},
                ],
                "relations": [
                    {
                        "subj_norm": "valentin the good",
                        "pred": "directed_by",
                        "obj_norm": "martin fric",
                    }
                ],
            },
        },
        {
            "chunk_id": "c4",
            "metadata": {
                "entities": [
                    {"norm": "martin fric"},
                    {"norm": "date_1968"},
                ],
                "relations": [
                    {
                        "subj_norm": "martin fric",
                        "pred": "died_on",
                        "obj_norm": "date_1968",
                    }
                ],
            },
        },
    ]



def comparison_corpus() -> dict[str, str]:
    return {
        "c1": "A Daughter of Two Worlds was directed by James Young.",
        "c2": "James Young died on 1948-06-09.",
        "c3": "Valentin the Good was directed by Martin Fric.",
        "c4": "Martin Fric died on 1968-08-26.",
    }



def strategy_embedder() -> TinyEmbedder:
    return TinyEmbedder(
        {
            "query::How is A related to B?": [1.0, 0.0, 0.0],
            "query::How is A related to C?": [1.0, 0.0, 0.0],
            "query::Unknown entity query": [1.0, 0.0, 0.0],
            "query::Which film has the director who died later, Valentin the Good or A Daughter of Two Worlds?": [1.0, 0.0, 0.0],
            "causes": [1.0, 0.0, 0.0],
            "a causes b": [1.0, 0.0, 0.0],
            "b causes c": [1.0, 0.0, 0.0],
            "directed by": [1.0, 0.0, 0.0],
            "died on": [1.0, 0.0, 0.0],
            "a daughter of two worlds directed by james young": [1.0, 0.0, 0.0],
            "james young died on date_1948": [1.0, 0.0, 0.0],
            "valentin the good directed by martin fric": [1.0, 0.0, 0.0],
            "martin fric died on date_1968": [1.0, 0.0, 0.0],
            "country": [0.0, 1.0, 0.0],
            "r": [1.0, 0.0, 0.0],
        }
    )
