"""Tiny synthetic systems used by the conformance suite.

Kept hermetic and CPU-only so the contract tests can run on every PR in
seconds, even on a laptop without a GPU.
"""
from __future__ import annotations

from evedesign.system import Protein, System


def tiny_protein_system() -> System:
    """A 12-residue single-chain protein.

    Cheap enough for a full single-mutation scan in the contract tests
    (12 positions x 20 amino acids = 240 mutants).
    """
    return System([
        Protein(id="toy", rep="MKLAVTSGGEFA", first_index=1),
    ])


def tiny_two_chain_system() -> System:
    """Two short chains for models that claim multi-entity support."""
    return System([
        Protein(id="A", rep="MKLAVT", first_index=1),
        Protein(id="B", rep="GGEFAY", first_index=10),
    ])
