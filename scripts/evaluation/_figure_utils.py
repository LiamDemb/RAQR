"""Shared helpers for dissertation figure scripts (PDF export, styling)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def apply_default_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


def savefig_pdf(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
