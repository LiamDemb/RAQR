"""Generation helpers (batch API, orchestration, answer parsing).

Import from submodules (e.g. ``raqr.generation.batch``) rather than this package
root so that ``raqr.generator`` can use leaf modules like ``answer_prefix``
without eager-loading batch (which depends on ``generator``).
"""
