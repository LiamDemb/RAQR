from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .corpus_schemas import Block


@dataclass(frozen=True)
class ChunkPiece:
    text: str
    section_path: List[str]
    char_span_in_doc: Tuple[int, int]
    token_count: int


def _count_tokens(text: str) -> int:
    return len(text.split())


def _tail_tokens(text: str, token_count: int) -> str:
    tokens = text.split()
    return " ".join(tokens[-token_count:]) if tokens else ""


def chunk_blocks(
    blocks: Sequence[Block],
    min_tokens: int = 500,
    max_tokens: int = 1000,
    overlap_tokens: int = 120,
) -> List[ChunkPiece]:
    chunks: List[ChunkPiece] = []
    buf: List[Block] = []
    buf_tokens = 0
    char_cursor = 0

    def _join_blocks(blocks_to_join: Sequence[Block]) -> str:
        return "\n\n".join(block.text for block in blocks_to_join if block.text)

    for block in blocks:
        block_tokens = _count_tokens(block.text)
        if buf_tokens + block_tokens > max_tokens and buf_tokens >= min_tokens:
            chunk_text = _join_blocks(buf)
            chunk_tokens = _count_tokens(chunk_text)
            start = char_cursor
            end = start + len(chunk_text)
            chunks.append(
                ChunkPiece(
                    text=chunk_text,
                    section_path=buf[-1].section_path,
                    char_span_in_doc=(start, end),
                    token_count=chunk_tokens,
                )
            )
            char_cursor = end
            tail = _tail_tokens(chunk_text, overlap_tokens)
            buf = [Block(text=tail, section_path=buf[-1].section_path, block_type="overlap")]
            buf_tokens = _count_tokens(tail)

        buf.append(block)
        buf_tokens += block_tokens

    if buf:
        chunk_text = _join_blocks(buf)
        chunk_tokens = _count_tokens(chunk_text)
        start = char_cursor
        end = start + len(chunk_text)
        chunks.append(
            ChunkPiece(
                text=chunk_text,
                section_path=buf[-1].section_path,
                char_span_in_doc=(start, end),
                token_count=chunk_tokens,
            )
        )

    return chunks
