"""Probe module: cheap top-k Dense retrieval and signal extraction."""

from .dense_probe import run_probe
from .signals import ProbeSignals

__all__ = ["ProbeSignals", "run_probe"]
