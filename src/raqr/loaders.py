from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple
import json
import pandas as pd
import numpy as np


class RowIdToChunkId(Protocol):
    """Row ID in FAISS index -> chunk ID in corpus."""

    def row_to_chunk(self, row_id: int) -> Optional[str]:
        ...


class ChunkIdToText(Protocol):
    """Chunk ID in corpus -> text."""

    def get_text(self, chunk_id: str) -> Optional[str]:
        ...


@dataclass
class VectorMetaMapper:
    """Loads vector_meta.parquet into memory and provides row_id -> chunk_id mapping."""

    parquet_path: str
    row_col: str = "row_id"
    chunk_col: str = "chunk_id"

    def __post_init__(self) -> None:
        df = pd.read_parquet(self.parquet_path, columns=[self.row_col, self.chunk_col])
        self._map: Dict[int, str] = dict(
            zip(df[self.row_col].astype(int), df[self.chunk_col].astype(str))
        )

    def row_to_chunk(self, row_id: int) -> Optional[str]:
        return self._map.get(row_id)


@dataclass
class VectorMetaWithYears:
    """Loads vector_meta.parquet with year fields for temporal filtering."""

    parquet_path: str
    row_col: str = "row_id"
    chunk_col: str = "chunk_id"
    year_min_col: str = "year_min"
    year_max_col: str = "year_max"
    years_col: str = "years"

    def __post_init__(self) -> None:
        cols = [
            self.row_col,
            self.chunk_col,
            self.year_min_col,
            self.year_max_col,
            self.years_col,
        ]
        df = pd.read_parquet(self.parquet_path, columns=cols)
        df[self.row_col] = df[self.row_col].astype(int)
        df[self.chunk_col] = df[self.chunk_col].astype(str)
        df["_ymin"] = df[self.year_min_col].apply(
            lambda x: int(x) if pd.notna(x) else None
        )
        df["_ymax"] = df[self.year_max_col].apply(
            lambda x: int(x) if pd.notna(x) else None
        )
        self._chunk: Dict[int, str] = dict(zip(df[self.row_col], df[self.chunk_col]))
        self._year_min: Dict[int, Optional[int]] = dict(zip(df[self.row_col], df["_ymin"]))
        self._year_max: Dict[int, Optional[int]] = dict(zip(df[self.row_col], df["_ymax"]))
        self._years: Dict[int, List[int]] = {}
        for row_id, raw_years in zip(df[self.row_col], df[self.years_col]):
            parsed: List[int] = []
            if isinstance(raw_years, (list, tuple, np.ndarray)):
                parsed = [int(y) for y in raw_years if pd.notna(y)]
            elif raw_years is not None and pd.notna(raw_years):
                # Defensive parse for non-list parquet representations.
                text = str(raw_years).strip()
                if text:
                    text = text.strip("[]")
                    if text:
                        parsed = [int(part.strip()) for part in text.split(",") if part.strip()]
            self._years[int(row_id)] = sorted(set(parsed))

    def row_to_chunk(self, row_id: int) -> Optional[str]:
        return self._chunk.get(row_id)

    def get_year_bounds(self, row_id: int) -> Optional[Tuple[Optional[int], Optional[int]]]:
        """Return (year_min, year_max) for the row, or None if row unknown."""
        if row_id not in self._chunk:
            return None
        return (self._year_min.get(row_id), self._year_max.get(row_id))

    def get_years(self, row_id: int) -> Optional[List[int]]:
        """Return explicit extracted years for the row, or None if row unknown."""
        if row_id not in self._chunk:
            return None
        return list(self._years.get(row_id, []))
    
@dataclass
class JsonCorpusLoader:
    """ Loads corpus.jsonl and provides chunk_id -> chunk text. """

    jsonl_path: str
    chunk_id_col: str = "chunk_id"
    text_key: str = "text"
    
    def __post_init__(self) -> None:
        self.text_by_id: Dict[str, str] = {}
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cid = str(obj[self.chunk_id_col])
                txt = str(obj[self.text_key])
                self.text_by_id[cid] = txt
    
    def get_text(self, chunk_id: str) -> Optional[str]:
        return self.text_by_id.get(chunk_id)