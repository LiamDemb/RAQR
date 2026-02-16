#!/usr/bin/env python3
"""Download and warm required models (spaCy, SentenceTransformers, REBEL).

Run once with network access. Models are cached for offline use.
Use HF_HOME / TRANSFORMERS_CACHE to control HuggingFace cache location.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SPACY_MODEL = os.environ.get("SPACY_MODEL", "en_core_web_sm")
MODEL_NAME = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")
RE_MODEL_NAME = os.environ.get("RE_MODEL_NAME", "Babelscape/rebel-large")


def main() -> int:
    hf_home = os.environ.get("HF_HOME", "data/processed/hf_cache")
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
    os.makedirs(hf_home, exist_ok=True)
    logger.info("HF cache: %s", os.environ.get("HF_HOME"))

    # spaCy
    logger.info("Downloading spaCy model: %s", SPACY_MODEL)
    rc = subprocess.run(
        [sys.executable, "-m", "spacy", "download", SPACY_MODEL],
        capture_output=True,
        text=True,
    )
    if rc.returncode != 0:
        logger.error("spaCy download failed: %s", rc.stderr)
        return 1
    logger.info("spaCy model ready.")

    # SentenceTransformer
    logger.info("Downloading SentenceTransformer model: %s", MODEL_NAME)
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(MODEL_NAME)
    logger.info("SentenceTransformer model ready.")

    # REBEL (HuggingFace transformers)
    logger.info("Downloading REBEL model: %s", RE_MODEL_NAME)
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    AutoTokenizer.from_pretrained(RE_MODEL_NAME)
    AutoModelForSeq2SeqLM.from_pretrained(RE_MODEL_NAME)
    logger.info("REBEL model ready.")

    logger.info("All models downloaded. Caches can be reused offline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
