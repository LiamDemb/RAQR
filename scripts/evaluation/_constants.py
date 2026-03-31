"""Shared constants for evaluation figures (keep aligned with 04b_validate_classifier)."""

from __future__ import annotations

# Same order as scripts/04b_validate_classifier.py ALL_ABLATIONS
ALL_ABLATIONS = [
    "q_emb",
    "q_feat",
    "probe",
    "q_emb,q_feat",
    "q_emb,probe",
    "q_feat,probe",
    "q_emb,q_feat,probe",
]

# Dissertation 2x2 grid: Q-feat, Q-emb, Q-emb+Q-probe, All combined
ABLATION_GRID_FOUR = [
    "q_feat",
    "q_emb",
    "q_emb,probe",
    "q_emb,q_feat,probe",
]

# Full combined router (RAQR end-to-end bar)
RAQR_SIGNALS = "q_emb,q_feat,probe"
