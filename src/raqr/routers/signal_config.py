"""Signal configuration for router ablation studies."""

from __future__ import annotations

from dataclasses import dataclass

Q_EMB_DIM = 384
Q_FEAT_DIM = 4
PROBE_DIM = 3

Q_FEAT_KEYS = [
    "entity_count",
    "syntactic_depth",
    "query_length_tokens",
    "relational_keyword_flag",
]

PROBE_KEYS = [
    "probe_max_score",
    "probe_skewness",
    "probe_semantic_dispersion",
]

LABEL_MAP = {"Dense": 0, "Graph": 1}
LABEL_NAMES = ["Dense", "Graph"]


@dataclass(frozen=True)
class SignalConfig:
    """Which input signal groups are active for an ablation run.

    Enables toggling Q-Emb, Q-Feat, and Probe channels independently to
    test different input combinations with a single classifier architecture.
    """

    use_q_emb: bool = False
    use_q_feat: bool = False
    use_probe: bool = False

    def __post_init__(self) -> None:
        if not (self.use_q_emb or self.use_q_feat or self.use_probe):
            raise ValueError("At least one signal must be enabled.")

    @property
    def identifier(self) -> str:
        parts = []
        if self.use_q_emb:
            parts.append("q_emb")
        if self.use_q_feat:
            parts.append("q_feat")
        if self.use_probe:
            parts.append("probe")
        return "-".join(parts)

    @property
    def input_dim(self) -> int:
        dim = 0
        if self.use_q_emb:
            dim += Q_EMB_DIM
        if self.use_q_feat:
            dim += Q_FEAT_DIM
        if self.use_probe:
            dim += PROBE_DIM
        return dim

    @property
    def num_scalar_features(self) -> int:
        """Number of scalar features that require normalization (Q-Feat + Probe)."""
        dim = 0
        if self.use_q_feat:
            dim += Q_FEAT_DIM
        if self.use_probe:
            dim += PROBE_DIM
        return dim

    @classmethod
    def from_str(cls, signals: str) -> SignalConfig:
        """Parse a comma-separated signal string like 'q_emb,probe'."""
        parts = {s.strip().lower() for s in signals.split(",") if s.strip()}
        return cls(
            use_q_emb="q_emb" in parts,
            use_q_feat="q_feat" in parts,
            use_probe="probe" in parts,
        )
