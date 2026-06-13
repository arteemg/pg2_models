"""
Conformance contract mixins for evedesign models.

Each contract is a plain class with ``test_*`` methods that pytest discovers
through inheritance. Contributors mix in only the contracts that match the
interfaces their model implements (e.g. omit ``MutationScorerContract`` for a
pure ``Scorer``).

The assertions here are deliberately tied to specific promises in the
docstrings of ``evedesign.model``; if a docstring promise changes, the
corresponding test should change with it.
"""
from __future__ import annotations

import copy
import random

import numpy as np
import pandas as pd
import pytest

from evedesign.model import BaseModel, MutationScorer, Scorer
from evedesign.system import Mutation, SystemInstance
from evedesign.testing.fixtures import tiny_protein_system


# --- shared fixtures ---------------------------------------------------------

class _ContractBase:
    """Common fixtures inherited by every contract.

    Override ``system`` and/or ``model`` in your subclass to plug in your
    own configuration.
    """

    @pytest.fixture
    def system(self):
        return tiny_protein_system()

    @pytest.fixture
    def model(self, system):  # pragma: no cover - must be overridden
        raise NotImplementedError(
            "Subclasses must provide a built `model` fixture"
        )

    @pytest.fixture
    def instances(self, system) -> list[SystemInstance]:
        """Three valid instances: WT plus two single mutants, deterministic."""
        wt = system.rep_to_instance()
        first = system[0]
        ref0 = str(wt[0].rep[0])
        ref1 = str(wt[0].rep[1])
        # pick `to` symbols different from ref to ensure non-trivial mutants
        to0 = "A" if ref0 != "A" else "V"
        to1 = "V" if ref1 != "V" else "L"
        m1 = system.mutate(
            wt, [[Mutation(entity=0, pos=first.first_index, ref=ref0, to=to0)]]
        )[0]
        m2 = system.mutate(
            wt, [[Mutation(entity=0, pos=first.first_index + 1, ref=ref1, to=to1)]]
        )[0]
        return [wt, m1, m2]


# --- BaseModel ---------------------------------------------------------------

class BaseModelContract(_ContractBase):
    """Universal checks any ``BaseModel`` subclass must pass."""

    def test_is_basemodel(self, model):
        assert isinstance(model, BaseModel)

    def test_metadata_types(self, model):
        # `name` and `citations` are surfaced in the UI/server; bad types break consumers.
        assert isinstance(model.name, str) and model.name.strip()
        assert isinstance(model.citations, list)
        for c in model.citations:
            assert isinstance(c, str) and c.strip()

    def test_capability_flag_invariants(self, model):
        # cf. _Core docstrings: insertions imply variable length; gpu requires gpu support.
        if model.handles_insertions:
            assert not model.requires_fixed_length, (
                "handles_insertions=True implies requires_fixed_length=False"
            )
        if model.requires_gpu:
            assert model.supports_gpu, (
                "requires_gpu=True implies supports_gpu=True"
            )

    def test_build_assigns_system_and_is_ready(self, model, system):
        # `build()` must set self.system and flip ready=True (BaseModel.build docstring).
        assert model.ready, "Model fixture must already be built (call .build() in fixture)"
        assert model.system is system or model.system == system

    def test_positions_sorted_and_in_range(self, model, system):
        instance = None if model.requires_fixed_length else system.rep_to_instance()
        positions = model.positions(instance=instance)

        assert positions == sorted(positions), \
            "positions() must be ordered ascending by (entity_idx, pos)"
        for entity_idx, pos in positions:
            assert 0 <= entity_idx < len(system)
            entity = system[entity_idx]
            assert entity.first_index <= pos < entity.first_index + len(entity.rep), (
                f"position {pos} out of range for entity {entity_idx} "
                f"[{entity.first_index}, {entity.first_index + len(entity.rep)})"
            )


# --- Scorer ------------------------------------------------------------------

class ScorerContract(_ContractBase):
    """Checks for any class implementing the ``Scorer`` interface."""

    def test_is_scorer(self, model):
        assert isinstance(model, Scorer)

    def test_score_shape_and_dtype(self, model, instances):
        scores = model.score(instances)
        assert isinstance(scores, np.ndarray)
        assert scores.shape == (len(instances),)
        assert np.issubdtype(scores.dtype, np.floating)

    def test_score_is_deterministic(self, model, instances):
        a = model.score(instances)
        b = model.score(instances)
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)

    def test_score_preserves_order(self, model, instances):
        rng = random.Random(0)
        order = list(range(len(instances)))
        rng.shuffle(order)
        shuffled = [instances[i] for i in order]

        baseline = model.score(instances)
        permuted = model.score(shuffled)
        np.testing.assert_allclose(
            permuted, baseline[order], rtol=1e-4, atol=1e-5
        )

    def test_score_does_not_mutate_input(self, model, instances):
        snapshot = copy.deepcopy(instances)
        model.score(instances)
        assert len(instances) == len(snapshot)
        for before, after in zip(snapshot, instances):
            for e_before, e_after in zip(before, after):
                np.testing.assert_array_equal(e_before.rep, e_after.rep)


# --- MutationScorer ----------------------------------------------------------

class MutationScorerContract(_ContractBase):
    """Checks for any class implementing the ``MutationScorer`` interface."""

    def test_is_mutation_scorer(self, model):
        assert isinstance(model, MutationScorer)

    def test_score_mutants_empty_returns_empty(self, model, instances):
        out = model.score_mutants(instances[0], [])
        assert isinstance(out, np.ndarray) and out.shape == (0,)

    def test_score_mutants_self_mutation_is_zero(self, model, system, instances):
        # MutationScorer docstring: self-substitutions must score 0
        # (log-odds are relative to the given instance).
        wt = instances[0]
        first_pos = system[0].first_index
        ref = str(wt[0].rep[0])
        scores = model.score_mutants(
            wt, [[Mutation(entity=0, pos=first_pos, ref=ref, to=ref)]]
        )
        np.testing.assert_allclose(scores, [0.0], atol=1e-4)

    def test_single_mutation_scan_dataframe_contract(self, model, system, instances):
        df = model.single_mutation_scan(instances[0], entity=0)

        # row index after unstacking "to" is (entity, pos, ref); columns become "to".
        assert isinstance(df, pd.DataFrame)
        assert list(df.index.names) == ["entity", "pos", "ref"], (
            f"row index must be (entity, pos, ref), got {df.index.names}"
        )

        # columns must be a subset of the system's alphabet, in declared order
        expected = system[0].alphabet(
            include_gap=model.handles_deletions,
            include_inserts=model.handles_insertions,
        )
        assert list(df.columns) == expected, (
            f"columns must follow Entity.alphabet() order: "
            f"expected {expected}, got {list(df.columns)}"
        )

        # diagonal: every row's value at column == ref must be 0 (self-mutation).
        for (entity_idx, pos, ref), row in df.iterrows():
            if ref in row.index and not pd.isna(row[ref]):
                assert row[ref] == pytest.approx(0.0, abs=1e-4), (
                    f"diagonal score at (entity={entity_idx}, pos={pos}, ref={ref}) "
                    f"must be 0, got {row[ref]}"
                )

        # Missing predictions must be encoded as np.nan (not -inf, not 0, not sentinel).
        nonfinite_mask = ~np.isfinite(df.to_numpy(dtype=float))
        nonfinite_values = df.to_numpy(dtype=float)[nonfinite_mask]
        assert np.all(np.isnan(nonfinite_values)), (
            "non-finite scores must be encoded as np.nan, not -inf or other"
        )
