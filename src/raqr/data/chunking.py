"""Token-based chunking using a HuggingFace tokenizer (REBEL BPE)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .corpus_schemas import Block


@dataclass(frozen=True)
class ChunkPiece:
    text: str
    section_path: List[str]
    char_span_in_doc: Tuple[int, int]
    token_count: int


def _count_tokens(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    """Return BPE token count (no special tokens)."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def _tail_tokens(text: str, token_count: int, tokenizer: PreTrainedTokenizerBase) -> str:
    """Return the last token_count tokens as decoded text for overlap."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    tail_ids = ids[-token_count:] if len(ids) >= token_count else ids
    return tokenizer.decode(tail_ids) if tail_ids else ""


def _split_long_block(
    block: Block,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
    overlap_tokens: int,
    char_cursor: int,
) -> Tuple[List[ChunkPiece], int]:
    """Split a block exceeding max_tokens into overlapping ChunkPieces."""
    ids = tokenizer.encode(block.text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return [], char_cursor

    chunks: List[ChunkPiece] = []
    stride = max_tokens - overlap_tokens
    pos = 0
    while pos < len(ids):
        chunk_ids = ids[pos : pos + max_tokens]
        chunk_text = tokenizer.decode(chunk_ids)
        chunk_tokens = len(chunk_ids)
        start = char_cursor
        end = start + len(chunk_text)
        chunks.append(
            ChunkPiece(
                text=chunk_text,
                section_path=block.section_path,
                char_span_in_doc=(start, end),
                token_count=chunk_tokens,
            )
        )
        char_cursor = end
        pos += stride

    return chunks, char_cursor


def chunk_blocks(
    blocks: Sequence[Block],
    tokenizer: PreTrainedTokenizerBase,
    min_tokens: int = 500,
    max_tokens: int = 800,
    overlap_tokens: int = 100,
) -> List[ChunkPiece]:
    """Chunk blocks by BPE token count.

    Invariant: emitted chunks never exceed max_tokens (REBEL-safe).
    min_tokens is a best-effort target (used when it does not violate max_tokens).
    """
    chunks: List[ChunkPiece] = []
    buf: List[Block] = []
    buf_tokens = 0
    char_cursor = 0

    def _join_blocks(blocks_to_join: Sequence[Block]) -> str:
        return "\n\n".join(block.text for block in blocks_to_join if block.text)

    def _flush_buf() -> None:
        nonlocal buf, buf_tokens, char_cursor
        if not buf:
            return
        chunk_text = _join_blocks(buf)
        chunk_tokens = _count_tokens(chunk_text, tokenizer)
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
        tail = _tail_tokens(chunk_text, overlap_tokens, tokenizer)
        buf = [Block(text=tail, section_path=buf[-1].section_path, block_type="overlap")]
        buf_tokens = _count_tokens(tail, tokenizer)

    for block in blocks:
        if not block.text.strip():
            continue
        block_tokens = _count_tokens(block.text, tokenizer)

        if block_tokens > max_tokens:
            _flush_buf()
            sub_chunks, char_cursor = _split_long_block(
                block, tokenizer, max_tokens, overlap_tokens, char_cursor
            )
            chunks.extend(sub_chunks)
            buf = []
            buf_tokens = 0
            continue

        if not buf:
            buf.append(block)
            buf_tokens = block_tokens
            continue

        if buf_tokens + block_tokens <= max_tokens:
            buf.append(block)
            buf_tokens += block_tokens
            continue

        # Adding this block would exceed max_tokens. Flush current buffer even if it
        # hasn't reached min_tokens to maintain the hard cap.
        #
        # Special case: if the buffer is only an overlap tail and still prevents the
        # next block from fitting, drop it to avoid emitting tiny overlap-only chunks.
        if len(buf) == 1 and buf[0].block_type == "overlap" and buf_tokens < min_tokens:
            buf = []
            buf_tokens = 0
            buf.append(block)
            buf_tokens = block_tokens
            continue

        _flush_buf()

        # After flushing, we have an overlap tail in the buffer. If even overlap+block
        # can't fit, drop the overlap tail and start a fresh chunk from this block.
        if buf_tokens + block_tokens > max_tokens:
            buf = []
            buf_tokens = 0

        buf.append(block)
        buf_tokens += block_tokens

    if buf:
        chunk_text = _join_blocks(buf)
        chunk_tokens = _count_tokens(chunk_text, tokenizer)
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
