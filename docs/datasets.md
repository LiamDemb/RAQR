## Dataset Provenance and Licensing

This document records the exact dataset sources, versions, and licensing
details used for Phase 1. Update the "Version/Snapshot" and "Checksum"
fields after downloading the final artifacts.

## Natural Questions (NQ)

- Canonical source: https://ai.google.com/research/NaturalQuestions/download
- Mirror (optional): https://huggingface.co/datasets/google-research-datasets/natural_questions
- License: Apache-2.0 (official repository), CC-BY-SA-3.0 (Hugging Face mirror)
- Version/Snapshot: TBD (record release tag/date)
- Download date: TBD
- Checksum (sha256): TBD
- Redistribution notes: Check license terms before redistributing processed data.

## 2WikiMultiHopQA

- Canonical source: https://huggingface.co/datasets/framolfese/2WikiMultihopQA
- Original paper: Ho et al., Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps, COLING 2020
- License: Apache-2.0 (original repo)
- Download script: `scripts/datasets/twowikimultihop_download.py`
- Schema: supporting_facts.title, supporting_facts.sent_id. Use `--2wiki` with the downloaded JSONL to run ingestion.
