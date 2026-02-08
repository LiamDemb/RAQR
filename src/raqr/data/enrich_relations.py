from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import logging

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from raqr.data.enrich_entities import normalize_key

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RelationTriple:
    subj: str
    pred: str
    obj: str
    subj_norm: str
    obj_norm: str

    def to_json(self) -> Dict[str, str]:
        return {
            "subj_surface": self.subj,
            "pred": self.pred,
            "obj_surface": self.obj,
            "subj_norm": self.subj_norm,
            "obj_norm": self.obj_norm,
        }


def load_rebel(model_name: str = "Babelscape/rebel-large", device: Optional[str] = None):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
    )
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(resolved_device)
    model.eval()
    return tokenizer, model, resolved_device

def load_rebel(model_name: str = "Babelscape/rebel-large", device: Optional[str] = None):
    # Standard environment optimizations for avoiding thread-bloat
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    # Determine the best available device
    if device:
        resolved_device = device
    elif torch.cuda.is_available():
        resolved_device = "cuda"
    #elif torch.backends.mps.is_available():
        #resolved_device = "mps" # <--- Automatically picks your M1 Pro GPU
    else:
        resolved_device = "cpu"

    print(f"Loading REBEL onto: {resolved_device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
    )
    
    model.to(resolved_device)
    model.eval()
    
    return tokenizer, model, resolved_device


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def parse_rebel_output(text: str) -> List[Tuple[str, str, str]]:
    """
    Parse REBEL output tokens into (subj, pred, obj) triples.

    Expected token order (REBEL default):
      <triplet> subj <subj> obj <obj> pred
    """
    if not text:
        return []

    tokens = (
        text.replace("<triplet>", " <triplet> ")
        .replace("<subj>", " <subj> ")
        .replace("<obj>", " <obj> ")
        .split()
    )

    triples: List[Tuple[str, str, str]] = []
    current: Dict[str, str] = {}
    buf: List[str] = []
    state: Optional[str] = None

    def _flush_state():
        nonlocal buf, state
        if state and buf:
            current[state] = " ".join(buf).strip()
        buf = []

    def _emit_if_ready():
        if all(current.get(k) for k in ("subj", "obj", "pred")):
            triples.append((current["subj"], current["pred"], current["obj"]))

    for tok in tokens:
        # REBEL generations sometimes include special tokens (or tokens glued to text),
        # e.g. "part</s>" or "<pad>". Strip them deterministically so predicates don't
        # get polluted.
        tok = tok.replace("</s>", "").replace("<s>", "").replace("<pad>", "")
        tok = tok.strip()
        if not tok:
            continue
        if tok == "<triplet>":
            _flush_state()
            _emit_if_ready()
            current = {}
            state = "subj"
            continue
        if tok == "<subj>":
            _flush_state()
            state = "obj"
            continue
        if tok == "<obj>":
            _flush_state()
            state = "pred"
            continue
        buf.append(tok)

    _flush_state()
    _emit_if_ready()
    return triples


def _normalize_triple(
    subj: str, pred: str, obj: str, alias_map: Optional[Dict[str, str]] = None
) -> Optional[RelationTriple]:
    alias_map = alias_map or {}
    subj_norm = alias_map.get(normalize_key(subj), normalize_key(subj))
    obj_norm = alias_map.get(normalize_key(obj), normalize_key(obj))
    pred_norm = pred.strip()
    if not subj_norm or not obj_norm or not pred_norm:
        return None
    return RelationTriple(
        subj=subj.strip(),
        pred=pred_norm,
        obj=obj.strip(),
        subj_norm=subj_norm,
        obj_norm=obj_norm,
    )


def extract_relations_rebel(
    texts: Sequence[str],
    tokenizer,
    model,
    device: str,
    alias_map: Optional[Dict[str, str]] = None,
    batch_size: int = 4,
    max_input_chars: int = 2000,
    max_input_tokens: int = 512,
    max_new_tokens: int = 128,
) -> List[List[Dict[str, str]]]:
    alias_map = alias_map or {}
    results: List[List[Dict[str, str]]] = []
    total_truncated = 0

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        truncated = [_truncate_text(t, max_input_chars) for t in batch]
        total_truncated += sum(1 for i, t in enumerate(batch) if len(t) != len(truncated[i]))

        enc = tokenizer(
            truncated,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        ).to(device)

        with torch.inference_mode():
            gen = model.generate(
                **enc,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
            )

        decoded = tokenizer.batch_decode(gen, skip_special_tokens=False)
        for output in decoded:
            triples = []
            for subj, pred, obj in parse_rebel_output(output):
                triple = _normalize_triple(subj, pred, obj, alias_map)
                if triple is None:
                    continue
                triples.append(triple)

            # Dedupe by normalized triple
            seen = set()
            unique = []
            for tr in triples:
                key = (tr.subj_norm, tr.pred, tr.obj_norm)
                if key in seen:
                    continue
                seen.add(key)
                unique.append(tr.to_json())
            results.append(unique)

    if total_truncated:
        logger.info("Relation extraction truncated %s chunks for max_input_chars.", total_truncated)
    return results
