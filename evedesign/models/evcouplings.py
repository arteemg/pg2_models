"""
Wrapper class around EVcouplings (Hopf, Ingraham et al., Nature Biotechnology 2017,
doi:10.1038/nbt.3769).

EVcouplings is a pairwise undirected graphical model (Potts model) over
protein sequences trained from an MSA with the standalone `plmc` C tool.
Once trained, the resulting binary `.model` file fully describes the model
through per-position fields `h_i` and pairwise couplings `J_ij`.

The wrapper exposes the five scoring/sampling/transform interfaces of evedesign:

* `Scorer`                       — full statistical energy (sequence Hamiltonian)
* `MutationScorer`               — single and higher-order mutation effects
* `ConditionalMutationScorer`    — per-position log-conditional energies
* `Generator`                    — Gibbs sampling driven by `GibbsSampler`
* `Transformer`                  — per-residue Hamiltonian contributions
                                   ``h_i(x_i) + (1/2) Σ_{j≠i} J_ij(x_i, x_j)``
                                   stashed as ``entity_instance.embedding``;
                                   sums to ``score(instance)`` by construction.

Conventions inherited from EVcouplings:

* Hamiltonians are *higher* for more native-like sequences (parameters of an
  `exp(H)` distribution), so `delta_hamiltonian` returns positive values for
  beneficial substitutions and negative values for deleterious ones. This
  matches evedesign's `MutationScorer` contract (`self-mutation = 0`,
  `beneficial > 0`, `deleterious < 0`) without any sign flip.
* The model alphabet is loaded from the file; by default plmc uses
  ``"-ACDEFGHIKLMNPQRSTVWY"`` (gap first). The wrapper reindexes the
  output dataframes to evedesign's `Entity.alphabet()` order at the boundary.
* The model only models positions in its `index_list` (focus-mode alignments
  drop insertion-state positions); `positions()` reflects this so downstream
  samplers do not try to design unmodelled positions.

Note: this wrapper assumes the user has already trained the Potts model and
points us to the resulting `.model` file. Training from scratch needs to run
the plmc binary as a subprocess; that path is not (yet) implemented here.
"""
from __future__ import annotations

from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import Any, Self, Sequence

import numpy as np
import pandas as pd
from loguru import logger

from evedesign.constants import GAP
from evedesign.model import (
    BaseModel,
    ConditionalMutationScorer,
    Generator,
    MutationScorer,
    Scorer,
    Transformer,
)
from evedesign.samplers.gibbs import (
    GibbsSampler,
    InitStrategy,
    ScanOrder,
    TemperatureSchedule,
)
from evedesign.system import EntityInstance, Mutant, System, SystemInstance
from evedesign.types import BatchSize, EntityPosList, StatusCallback
from evedesign.utils import ensure_sequence, model_param_context

try:
    from evcouplings.couplings.model import (
        CouplingsModel,
        _single_mutant_hamiltonians,
        FULL,
    )
    IMPORT_AVAILABLE = True
except ImportError:
    CouplingsModel = None  # type: ignore[assignment]
    _single_mutant_hamiltonians = None  # type: ignore[assignment]
    FULL = 0
    IMPORT_AVAILABLE = False


class EVcouplings(
    BaseModel,
    Scorer,
    MutationScorer,
    ConditionalMutationScorer,
    Generator,
    Transformer,
):
    """
    Wrapper around a pre-trained EVcouplings Potts model.

    Parameters
    ----------
    model_file
        Path to a binary `.model` file produced by `plmc` (the EVcouplings
        training stage). Required.
    file_format
        Plmc binary format; defaults to ``"plmc_v2"`` (modern). Old `.eij`
        files use ``"plmc_v1"``.
    precision
        Numeric precision used to read the file (``"float32"`` or ``"float64"``).
    keep_model_after_build
        If True (default), keep the loaded `CouplingsModel` cached on the
        instance after `build()`. Set False to keep memory low between scoring
        bursts (model reloads on next call).
    keep_model_after_pred
        If True (default), keep the model loaded after scoring. Mirrors the
        same flag on the other wrappers; turn off when serializing.
    """

    available = IMPORT_AVAILABLE
    name: str = "EVcouplings"
    citations: list[str] = [
        "doi:10.1038/nbt.3769",
        "doi:10.1371/journal.pone.0028766",
    ]

    requires_target: bool = True
    requires_fixed_length: bool = True
    handles_deletions: bool = True
    handles_insertions: bool = False
    requires_gpu: bool = False
    supports_gpu: bool = False
    supports_gpu_parallel: bool = False
    supports_cpu_parallel: bool = False

    required_entity_attributes: list[str] | None = []
    optional_entity_attributes: list[str] | None = ["sequences"]

    def __init__(
        self,
        model_file: str | PathLike,
        file_format: str = "plmc_v2",
        precision: str = "float32",
        keep_model_after_build: bool = True,
        keep_model_after_pred: bool = True,
        # GibbsSampler hyperparameters used by generate()
        num_sweeps: int = 1000,
        init_strategy: InitStrategy = "system",
        scan_order: ScanOrder = "random",
        temperature_schedule: TemperatureSchedule | None = None,
        batch_size: BatchSize = 64,
    ):
        if not self.available:
            raise ValueError(
                "evcouplings package could not be imported. Install it with "
                "`pip install evedesign[evcouplings]` or `pip install evcouplings`."
            )

        self.model_file = Path(model_file)
        if not self.model_file.is_file():
            raise ValueError(
                f"Model file not found: {self.model_file}. Train one with the "
                f"plmc binary (see EVcouplings docs) and supply the resulting "
                f"`.model` file."
            )

        self.file_format = file_format
        self.precision = precision
        self.keep_model_after_build = keep_model_after_build
        self.keep_model_after_pred = keep_model_after_pred

        self.num_sweeps = num_sweeps
        self.init_strategy = init_strategy
        self.scan_order = scan_order
        self.temperature_schedule = temperature_schedule
        self.batch_size = batch_size

        self._system: System | None = None
        self.model: CouplingsModel | None = None

        # Cached after build(): maps biological positions (int) -> internal
        # model index, and the model's char alphabet -> column index in the
        # evedesign Entity.alphabet() ordering.
        self._pos_to_idx: dict[int, int] | None = None
        self._idx_to_pos: np.ndarray | None = None  # shape (L,), int
        self._model_to_evedesign_cols: np.ndarray | None = None  # shape (A_model,)
        self._evedesign_to_model_cols: dict[str, int] | None = None

    # ------------------------------------------------------------------ core

    @property
    def ready(self) -> bool:
        return self._system is not None and self._pos_to_idx is not None

    @property
    def system(self) -> System | None:
        return self._system

    @classmethod
    def can_model(cls, system: System, data: None = None) -> tuple[bool, str]:
        if data is not None:
            return False, "Model does not support data parameter (must be None)"
        if len(system) != 1 or system[0].type != "protein":
            return False, "Can only handle single-component protein system"
        target = system[0]
        if not target.defined_sequence():
            return False, "Entity must have defined rep sequence"
        return True, ""

    def _load_model(self) -> None:
        if self.model is not None:
            return
        self.model = CouplingsModel(
            str(self.model_file),
            precision=self.precision,
            file_format=self.file_format,
        )

    def _delete_model(self) -> None:
        self.model = None

    def build(
        self,
        system: System,
        data: None = None,
        status_callback: StatusCallback | None = None,  # noqa: ARG002
    ) -> Self:
        self.can_model_or_raise(system, data)
        self._system = system
        target = system[0]

        # Load model parameters and validate compatibility with the system.
        self._load_model()

        index_list = np.asarray(self.model.index_list, dtype=int)
        first_index = target.first_index
        rep_end = first_index + len(target.rep)

        out_of_range = (index_list < first_index) | (index_list >= rep_end)
        if out_of_range.any():
            raise ValueError(
                f"EVcouplings model covers positions outside the target entity "
                f"range [{first_index}, {rep_end - 1}]: "
                f"{index_list[out_of_range].tolist()}. Make sure the entity "
                f"`first_index` matches the numbering used to train the model."
            )

        # Cross-check: at every modelled position, the model's target_seq
        # symbol should agree with the entity rep, otherwise we are silently
        # scoring an unrelated protein.
        rep = np.asarray(target.rep, dtype="U1")
        model_target = np.asarray(self.model.target_seq, dtype="U1")
        mismatches = []
        for k, pos in enumerate(index_list):
            entity_aa = rep[pos - first_index]
            model_aa = model_target[k]
            if entity_aa != model_aa and model_aa != GAP and entity_aa != GAP:
                mismatches.append((int(pos), str(entity_aa), str(model_aa)))
        if mismatches:
            preview = ", ".join(f"pos{p}: rep={r} vs model={m}" for p, r, m in mismatches[:5])
            raise ValueError(
                f"EVcouplings model target sequence disagrees with the entity "
                f"rep at {len(mismatches)} position(s). First few: {preview}. "
                f"Either the model was trained on a different protein or the "
                f"alignment numbering is off."
            )

        # Cache index translations and alphabet reindexing.
        self._pos_to_idx = {int(p): k for k, p in enumerate(index_list)}
        self._idx_to_pos = index_list

        # Build a column mapping from the model alphabet to the evedesign
        # `Entity.alphabet(include_gap=True)` ordering, so dataframes line up
        # with the rest of the framework.
        evedesign_alphabet = target.alphabet(include_gap=True)
        evedesign_col = {aa: i for i, aa in enumerate(evedesign_alphabet)}
        self._model_to_evedesign_cols = np.array(
            [evedesign_col.get(aa, -1) for aa in self.model.alphabet],
            dtype=int,
        )
        self._evedesign_to_model_cols = dict(self.model.alphabet_map)

        logger.debug(
            f"EVcouplings model loaded: L={self.model.L}, "
            f"num_symbols={self.model.num_symbols}, alphabet={''.join(self.model.alphabet)}, "
            f"N_eff={self.model.N_eff:.1f}"
        )

        if not self.keep_model_after_build:
            self._delete_model()

        return self

    # -------------------------------------------------- positions (override)

    def positions(
        self,
        instance: SystemInstance | None,  # noqa: ARG002
    ) -> list[tuple[int, int]]:
        """
        Return only positions covered by the Potts model. In focus-mode
        alignments `plmc` drops insertion-state positions, so the model
        typically covers a subset of the entity rep.
        """
        if self._idx_to_pos is None:
            raise ValueError("Must call build() first")
        return [(0, int(p)) for p in self._idx_to_pos]

    # ---------------------------------------------------- helpers (private)

    def _convert_one(self, instance: SystemInstance) -> np.ndarray:
        """Map a SystemInstance sequence to the integer encoding the model uses."""
        target = self.system[0]
        first_index = target.first_index
        rep = np.asarray(instance[0].rep, dtype="U1")
        seq_chars = [rep[p - first_index] for p in self._idx_to_pos]
        try:
            return np.array(
                [self._evedesign_to_model_cols[aa] for aa in seq_chars], dtype=int
            )
        except KeyError as exc:
            raise ValueError(
                f"Sequence contains symbol {exc.args[0]!r} that is not in the "
                f"model alphabet {''.join(self.model.alphabet)!r}"
            ) from exc

    def _reindex_columns(self, model_matrix: np.ndarray) -> np.ndarray:
        """
        Reindex a (..., A_model) matrix into the evedesign alphabet order,
        padding missing symbols with NaN.
        """
        target = self.system[0]
        out_cols = len(target.alphabet(include_gap=True))
        out = np.full(model_matrix.shape[:-1] + (out_cols,), np.nan)
        for model_col, evedesign_col in enumerate(self._model_to_evedesign_cols):
            if evedesign_col >= 0:
                out[..., evedesign_col] = model_matrix[..., model_col]
        return out

    @contextmanager
    def _ensure_loaded(self):
        """Load the model if missing, then optionally release after exit."""
        with model_param_context(
            self._load_model, self._delete_model, self.keep_model_after_pred
        ):
            yield

    # ----------------------------------------------------------- Scorer API

    def score(
        self,
        instances: Sequence[SystemInstance],
        status_callback: StatusCallback | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        """
        Statistical energy (full Hamiltonian) of each instance under the
        Potts model. Higher = more native-like.
        """
        self.ready_or_raise()
        self._validate_instances(instances)

        with self._ensure_loaded():
            seqs = np.stack([self._convert_one(inst) for inst in instances], axis=0)
            h = self.model.hamiltonians(seqs)
        return h[:, FULL].astype(float)

    # ------------------------------------------------------- Transformer API

    def transform(
        self,
        instances: Sequence[SystemInstance],
        entity: int | None = None,
        status_callback: StatusCallback | None = None,  # noqa: ARG002
    ) -> list[SystemInstance]:
        """
        Decompose each instance's Hamiltonian by residue and stash the
        result as ``entity_instance.embedding``.

        The decomposition uses the symmetric per-residue contribution
        ``e_i = h_i(x_i) + (1/2) Σ_{j ≠ i} J_ij(x_i, x_j)`` so that
        ``Σ_i e_i ≡ H(x) ≡ score(x)`` exactly. Positions outside the
        model's index list (e.g. lower-case columns dropped from a focus
        alignment) are filled with NaN in the embedding so the caller can
        tell modeled vs. unmodeled positions apart.

        Returns shallow copies of each instance with ``embedding`` and
        ``score`` filled in; the input list is left unmodified per the
        Transformer contract.
        """
        self.ready_or_raise()
        self._validate_instances(instances)

        entity = 0 if entity is None else entity
        if entity != 0:
            raise ValueError("Model can only handle one single entity")

        target = self.system[0]
        first_index = target.first_index
        rep_len = len(target.rep)
        # per-instance dimension is the *full* entity rep length so the
        # embedding is interpretable position-by-position alongside rep.
        with self._ensure_loaded():
            L = self.model.L
            j_arange = np.arange(L)

            out: list[SystemInstance] = []
            for instance in instances:
                seq = self._convert_one(instance)  # (L,) model indices

                h_at_seq = self.model.h_i[j_arange, seq]  # (L,)
                # J_ij[i, j, x_i, x_j] for x = seq, vectorised:
                J_at_seq = self.model.J_ij[
                    j_arange[:, None], j_arange[None, :],
                    seq[:, None], seq[None, :],
                ]
                np.fill_diagonal(J_at_seq, 0.0)
                per_pos_model = h_at_seq + 0.5 * J_at_seq.sum(axis=1)  # (L,)

                # scatter back to full rep length, leaving unmodeled
                # positions as NaN.
                per_residue = np.full(rep_len, np.nan)
                for k, p in enumerate(self._idx_to_pos):
                    per_residue[int(p) - first_index] = float(per_pos_model[k])

                inst_copy = instance.copy()
                inst_copy[entity].embedding = per_residue
                inst_copy.score = float(np.nansum(per_residue))
                out.append(inst_copy)
        return out

    # --------------------------------------------------- MutationScorer API

    def score_mutants(
        self,
        instance: SystemInstance,
        mutants: Sequence[Mutant],
        status_callback: StatusCallback | None = None,  # noqa: ARG002
    ) -> np.ndarray:
        """
        Delta Hamiltonian of each (single or higher-order) mutant relative
        to the supplied instance. Self-mutations score 0 by construction.
        """
        self.ready_or_raise()
        self._validate_instances([instance])
        self.system.valid_mutants(
            instance,
            mutants,
            deletions=self.handles_deletions,
            insertions=False,
            raise_invalid=True,
        )

        out = np.zeros(len(mutants), dtype=float)
        with self._ensure_loaded():
            # Swap the model's reference sequence to the instance so
            # delta_hamiltonian computes deltas relative to it (not relative
            # to the sequence baked into the .model file).
            inst_chars = [
                str(c) for c in np.asarray(instance[0].rep, dtype="U1")[
                    np.asarray(self._idx_to_pos) - self.system[0].first_index
                ]
            ]
            previous_target = self.model.target_seq
            try:
                self.model.target_seq = inst_chars
                for i, mutant in enumerate(mutants):
                    subs = [(s.pos, s.ref, s.to) for s in mutant]
                    # Skip subs at unmodelled positions; if every sub is
                    # unmodelled, that's a zero-delta no-op for the model.
                    subs_in_model = [
                        s for s in subs if s[0] in self._pos_to_idx
                    ]
                    if len(subs_in_model) != len(subs):
                        missing = [s for s in subs if s[0] not in self._pos_to_idx]
                        logger.warning(
                            f"Mutant {i} has substitutions at positions not "
                            f"covered by the Potts model (skipped): {missing}"
                        )
                    if not subs_in_model:
                        out[i] = 0.0
                        continue
                    out[i] = self.model.delta_hamiltonian(
                        subs_in_model, verify_mutants=False
                    )[FULL]
            finally:
                # Restore so other consumers of the same model don't see
                # surprise side effects.
                self.model.target_seq = previous_target.tolist()
        return out

    def single_mutation_scan(
        self,
        instance: SystemInstance,
        entity: int | None = None,
        positions: Sequence[int] | None = None,
        status_callback: StatusCallback | None = None,  # noqa: ARG002
    ) -> pd.DataFrame:
        """
        Compute the full single-mutation effect matrix relative to the given
        instance. Returns a DataFrame indexed by (entity, pos, ref) with one
        column per amino acid in `Entity.alphabet(include_gap=True)`.
        """
        self.ready_or_raise()
        self._validate_instances([instance])

        if positions is not None and entity is None:
            raise ValueError(
                "Parameter entity must be explicitly specified if using parameter positions"
            )
        entity = 0 if entity is None else entity
        if entity != 0:
            raise ValueError("Model can only handle one single entity")

        target = self.system[0]
        if positions is not None:
            self.valid_positions(positions, entities=0, raise_invalid=True)
            pos_list = list(positions)
        else:
            pos_list = [int(p) for p in self._idx_to_pos]

        # internal model indices for each requested position
        internal_idx = np.array([self._pos_to_idx[p] for p in pos_list], dtype=int)

        with self._ensure_loaded():
            inst_mapped = self._convert_one(instance)
            # `_single_mutant_hamiltonians` returns (L, A_model, 3); take :, :, FULL
            full_smm = _single_mutant_hamiltonians(
                inst_mapped, self.model.J_ij, self.model.h_i
            )[:, :, FULL]
            # subset to requested positions, then reindex columns to evedesign order
            smm = full_smm[internal_idx]              # (P, A_model)
        scan = self._reindex_columns(smm)              # (P, A_evedesign)

        evedesign_alphabet = target.alphabet(include_gap=True)
        df = pd.DataFrame(scan, columns=evedesign_alphabet)
        # build the (pos, ref) index from the instance, then prepend entity
        rep = np.asarray(instance[0].rep, dtype="U1")
        first_index = target.first_index
        index = [(p, str(rep[p - first_index])) for p in pos_list]
        df.index = pd.MultiIndex.from_tuples(index, names=["pos", "ref"])
        df = pd.concat({entity: df}, names=["entity"])

        # If `handles_deletions` is False on the entity, drop the gap column
        # for contract conformance with that mode (kept here since we declare
        # handles_deletions=True; gap stays in).
        return df

    # ----------------------------------- ConditionalMutationScorer API

    def score_conditional(
        self,
        instances: Sequence[SystemInstance],
        entities: Sequence[int],
        positions: Sequence[int],
        status_callback: StatusCallback | None = None,  # noqa: ARG002
    ) -> pd.DataFrame:
        """
        Per-position log-conditional energies
        ``log p(A_i = a | A_{-i}) = h_i[i, a] + sum_{j != i} J_ij[i, j, a, inst[j]]``
        (up to a per-position normalising constant that does not affect
        substitution ratios, matching the contract that conditional scores
        are raw logits).
        """
        self.ready_or_raise()
        self._validate_instances(instances)

        if set(entities) != {0}:
            raise ValueError("Can only specify entities with index 0")
        if not len(instances) == len(entities) == len(positions):
            raise ValueError(
                "Sequences for instances, entities and positions must all have same length"
            )

        # Validate positions per instance (the contract says positions must be
        # in the supplied instance, not just in the model).
        for inst, pos in zip(instances, positions):
            self.valid_positions(
                [pos], instance=inst, entities=0, raise_invalid=True
            )

        rows = []
        row_index = []
        with self._ensure_loaded():
            L = self.model.L
            j_arange = np.arange(L)
            for instance_idx, (inst, entity, pos) in enumerate(
                zip(instances, entities, positions)
            ):
                inst_mapped = self._convert_one(inst)
                i = self._pos_to_idx[int(pos)]
                # We want logits[a] = h_i[i, a] + sum_j J_ij[i, j, a, inst[j]],
                # with the j = i self-term contributing 0 by plmc convention.
                # J_ij[i] has shape (L, A, A); for each j select the column
                # corresponding to inst[j] to get (L, A), then sum over j.
                J_at_i = self.model.J_ij[i]                          # (L, A, A)
                J_to_inst = J_at_i[j_arange, :, inst_mapped]         # (L, A)
                logits = self.model.h_i[i] + J_to_inst.sum(axis=0)   # (A_model,)
                rows.append(logits)
                row_index.append((instance_idx, entity, pos))

        cond = np.stack(rows, axis=0)                       # (N, A_model)
        cond = self._reindex_columns(cond)                  # (N, A_evedesign)

        target = self.system[0]
        df = pd.DataFrame(cond, columns=target.alphabet(include_gap=True))
        df.index = pd.MultiIndex.from_tuples(
            row_index, names=["instance", "entity", "pos"]
        )
        return df

    # -------------------------------------------------------- Generator API

    def generate(
        self,
        num_designs: int,
        entities: Sequence[int] | None = None,
        fixed_pos: EntityPosList | None = None,
        temperature: float = 1.0,
        status_callback: StatusCallback | None = None,
    ) -> list[SystemInstance]:
        """
        Generate designs by Gibbs sampling from the Potts model conditional
        distribution. Mirrors the ESM2 wrapper's pattern of delegating to
        `GibbsSampler` with self as the (only) scorer.
        """
        self.ready_or_raise()

        entities = entities if entities is not None else [0]
        entities = ensure_sequence(entities)
        if len(entities) != 1 or entities[0] != 0:
            raise ValueError(
                "Can only design single entity (entities = [0] | None)"
            )

        with self._ensure_loaded():
            sampler = GibbsSampler(
                scorers=[self],
                weights=None,
                num_sweeps=self.num_sweeps,
                init_strategy=self.init_strategy,
                scan_order=self.scan_order,
                temperature_schedule=self.temperature_schedule,
                require_strict_pos=True,
                record_full_chain=False,
            )
            instances = sampler.generate(
                num_designs=num_designs,
                entities=entities,
                fixed_pos=fixed_pos,
                temperature=temperature,
                status_callback=status_callback,
            )

        # Normalize scores relative to the WT reference, same as ESM2/EVmutation2.
        target = self.system[0]
        ref_instance = SystemInstance(EntityInstance(rep="".join(target.rep)))
        all_instances = [ref_instance] + list(instances)
        scores = self.score(all_instances)
        ref_score = scores[0]
        for i, inst in enumerate(instances):
            inst.score = float(scores[i + 1] - ref_score)
        return list(instances)

    # ------------------------------------------------------ bonus: contacts

    def evolutionary_couplings(self, apc: bool = True) -> pd.DataFrame:
        """
        Return the model's evolutionary couplings (EC scores). Useful for
        contact prediction; not part of any evedesign interface.

        Parameters
        ----------
        apc
            If True (default) return APC-corrected CN scores; otherwise the
            raw Frobenius norm (FN) scores.
        """
        self.ready_or_raise()
        with self._ensure_loaded():
            ecs = self.model.ecs.copy()
        # cn = APC-corrected, fn = raw Frobenius norm
        score_col = "cn" if apc else "fn"
        ecs["score"] = ecs[score_col]
        return ecs

    # ----------------------------------------------------- (de)serialization

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["model"] = None  # the binary model file is the source of truth
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.model = None
