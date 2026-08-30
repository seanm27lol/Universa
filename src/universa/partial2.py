"""Partial observation of 2-complexes: degraded boundary pairs.

:mod:`universa.partial` degrades the single boundary of a 1-complex; this
module extends the same regime to a 2-complex ``(B1, B2)``, degraded by

- **masked edges**: a fraction of the edge columns removed from ``B1``
  *and* the matching rows removed from ``B2`` — one consistent edge set,
  with the kept-edge indices carried as provenance;
- **corrupted signs**: a controlled fraction of the nonzero entries of the
  surviving ``B1`` (resp. ``B2``) sign-flipped, under two independent
  draws.

All three draws are deterministic from an integer seed via
:func:`universa.generators.subseed` (components ``"partial2-mask"``,
``"partial2-b1"``, ``"partial2-b2"``) and *nested*: each degraded set at
fraction ``f`` is a prefix of one master permutation, so a larger fraction
degrades strictly more of the same matrices. Counts are
``round(fraction * total)`` (round-half-even). Masking is applied first,
corruption to the surviving entries — the convention of
:mod:`universa.partial`.

**The observed object is a purported structure, not a chain complex.**
After sign corruption — and, away from the trivial endpoints, after edge
masking as well — the observed pair generally violates ``d^2 = 0``:
``B1' B2' != 0`` (masking drops terms of a sum that only cancels in
full). :class:`ObservedTwoComplex` therefore holds the raw boundary arrays
plus full provenance *without* :class:`~universa.structures.ChainComplex`'s
fail-closed ``d^2`` validation: it is a plain frozen dataclass whose
constructor checks shapes and bookkeeping only. That is the point of the
regime, not a validation hole, and it has an informative consequence:
under corruption the *true* target can stop being a valid complex while
every valid-complex decoy remains valid — a naive validity check would
anti-rank the truth. This is part of why partial observation of higher
complexes is hard, and why the reference signal here is the commutation
residual against the observed pair, never a validity test.

:func:`observed_commutation_residuals` evaluates both degrees of that
residual for a planted chain map; :func:`observed_d2_residue` measures how
far the observed pair itself is from composing to zero. Every matrix is
integer-valued in float64, so residuals stay exact, and the repo's
fail-closed discipline applies to everything that *is* claimed: shapes,
provenance ranges, fraction domains.

Documented behavior (default sizes, pinned study seed 2026, default 0.1
grid; :func:`ranking_study_two`). Under dual sign corruption
(``mode="corrupt"``) the true candidate's degree-1 residual rises
strictly from ``0.0`` at every grid step — the structural monotonicity of
:mod:`universa.partial`, each flipped ``B1`` entry adding a
Frobenius-orthogonal term — while its degree-2 residual is *not*
monotone on this seed (cross terms between ``f2`` rows inside one
observed edge row can cancel), so the max-degree misfit is not monotone
either: it dips at fraction 0.7. The true candidate keeps ranking first
through fraction 0.3, ties a decoy at 0.4 (margin exactly ``0.0`` — the
breakdown fraction), and loses strictly from 0.5 on. Under consistent
edge masking (``mode="mask"``) it ranks first through 0.9 and breaks
only at 1.0, where no edge is observed and every candidate ties at
``||f0 B1||_F``. The observed ``d^2`` residue of the true target is
exactly ``0.0`` at zero degradation, positive at every intermediate
probed level — while every decoy's own boundaries keep ``d^2 = 0`` —
and returns to exactly ``0.0`` at full dual corruption, where every
nonzero of both boundaries has been negated and
``(-B1)(-B2) = B1 B2 = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from universa.complexes2 import make_two_complex_switch_instance
from universa.generators import subseed
from universa.partial import (
    DEFAULT_FRACTIONS,
    RankingRow,
    breakdown_fraction,
)
from universa.structures import ChainComplex, ChainMap

__all__ = [
    "DEFAULT_FRACTIONS",
    "ObservedTwoComplex",
    "RankingRow",
    "TwoComplexObservationModel",
    "breakdown_fraction",
    "observed_commutation_residuals",
    "observed_d2_residue",
    "ranking_study_two",
]


@dataclass(frozen=True)
class ObservedTwoComplex:
    """A ground-truth 2-complex as actually observed, with provenance.

    ``b1`` and ``b2`` are the *raw* observed boundaries: the ground-truth
    ``B1`` restricted to the columns of ``kept_edges`` with the entries at
    ``flipped_b1`` sign-flipped, and the ground-truth ``B2`` restricted to
    the rows of ``kept_edges`` with the entries at ``flipped_b2``
    sign-flipped. They are stored as plain arrays, **not** as a
    :class:`~universa.structures.ChainComplex`: the pair generally
    violates ``d^2 = 0`` (see the module docstring), so ChainComplex's
    fail-closed validation would reject exactly the objects this module
    exists to describe. Only shapes and provenance are checked here —
    fail-closed on bookkeeping, silent on validity.

    ``truth_dims`` is ``(num_vertices, num_edges, num_faces)`` of the
    ground truth; ``kept_edges`` maps observed edge slots (columns of
    ``b1``, rows of ``b2``) back to ground-truth edge indices and is
    strictly increasing; ``flipped_b1`` and ``flipped_b2`` are
    ``(row, observed_position)`` sites of sign flips and must sit on
    nonzero entries of the respective observed array.
    """

    b1: np.ndarray
    b2: np.ndarray
    truth_dims: tuple[int, int, int]
    kept_edges: tuple[int, ...]
    flipped_b1: tuple[tuple[int, int], ...]
    flipped_b2: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        for name, matrix in (("b1", self.b1), ("b2", self.b2)):
            if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
                raise ValueError(f"{name} is not a 2-D array")
        num_vertices, num_edges, num_faces = self.truth_dims
        num_kept = len(self.kept_edges)
        if self.b1.shape != (num_vertices, num_kept):
            raise ValueError(
                f"dims contract violated: b1 is {self.b1.shape}, expected "
                f"{(num_vertices, num_kept)}"
            )
        if self.b2.shape != (num_kept, num_faces):
            raise ValueError(
                f"dims contract violated: b2 is {self.b2.shape}, expected "
                f"{(num_kept, num_faces)}"
            )
        if any(b <= a for a, b in zip(self.kept_edges, self.kept_edges[1:])):
            raise ValueError("kept_edges must be strictly increasing")
        if self.kept_edges and not 0 <= self.kept_edges[-1] < num_edges:
            raise ValueError("kept_edges out of range")
        for row, col in self.flipped_b1:
            if not (0 <= row < num_vertices and 0 <= col < num_kept):
                raise ValueError("flipped b1 entry out of range")
            if self.b1[row, col] == 0.0:
                raise ValueError("flipped b1 entry must be nonzero")
        for row, col in self.flipped_b2:
            if not (0 <= row < num_kept and 0 <= col < num_faces):
                raise ValueError("flipped b2 entry out of range")
            if self.b2[row, col] == 0.0:
                raise ValueError("flipped b2 entry must be nonzero")

    @property
    def num_masked(self) -> int:
        """How many ground-truth edges were removed."""
        return self.truth_dims[1] - len(self.kept_edges)


def _sign_flip(
    matrix: np.ndarray, flip_seed: int, fraction: float
) -> tuple[tuple[int, int], ...]:
    """Sign-flip a nested ``round(fraction * nnz)`` of the nonzero entries.

    The flipped set at any fraction is a prefix of one master permutation
    of the nonzero positions drawn under ``flip_seed``, so degradation is
    nested in the fraction. Mutates ``matrix``; returns the sorted
    ``(row, col)`` flip sites.
    """
    rows, cols = np.nonzero(matrix)
    rng = np.random.default_rng(flip_seed)
    entry_order = rng.permutation(len(rows))
    num_flipped = int(round(fraction * len(rows)))
    flipped = []
    for index in entry_order[:num_flipped]:
        row, col = int(rows[index]), int(cols[index])
        matrix[row, col] = -matrix[row, col]
        flipped.append((row, col))
    return tuple(sorted(flipped))


@dataclass(frozen=True)
class TwoComplexObservationModel:
    """A ground-truth 2-complex plus the fractions of it we do not see.

    ``truth`` must be a valid 2-complex (the ground truth *is* a chain
    complex — only its observation is not). ``mask_fraction`` of the edges
    are unobserved (their ``B1`` columns and ``B2`` rows removed,
    consistently); ``corrupt_b1_fraction`` and ``corrupt_b2_fraction`` of
    the nonzero entries that remain in each boundary are sign-flipped
    under independent draws. Masking is applied first, corruption to the
    surviving entries. All draws are deterministic functions of ``seed``
    and the fractions, via nested prefixes of master permutations drawn
    under ``subseed(seed, "partial2-mask" | "partial2-b1" | "partial2-b2")``.
    """

    truth: ChainComplex
    seed: int
    mask_fraction: float = 0.0
    corrupt_b1_fraction: float = 0.0
    corrupt_b2_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.truth.top_degree != 2:
            raise ValueError("partial observation is defined for 2-complexes")
        for name, value in (
            ("mask_fraction", self.mask_fraction),
            ("corrupt_b1_fraction", self.corrupt_b1_fraction),
            ("corrupt_b2_fraction", self.corrupt_b2_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}={value} outside [0, 1]")

    def observe(self) -> ObservedTwoComplex:
        """Draw the degraded observation, deterministically."""
        b1, b2 = self.truth.boundaries
        num_vertices, num_edges = b1.shape
        num_faces = b2.shape[1]
        mask_rng = np.random.default_rng(subseed(self.seed, "partial2-mask"))
        edge_order = mask_rng.permutation(num_edges)
        num_masked = int(round(self.mask_fraction * num_edges))
        masked = {int(e) for e in edge_order[:num_masked]}
        kept_edges = tuple(e for e in range(num_edges) if e not in masked)

        observed_b1 = b1[:, list(kept_edges)].copy()
        observed_b2 = b2[list(kept_edges), :].copy()
        flipped_b1 = _sign_flip(
            observed_b1, subseed(self.seed, "partial2-b1"),
            self.corrupt_b1_fraction,
        )
        flipped_b2 = _sign_flip(
            observed_b2, subseed(self.seed, "partial2-b2"),
            self.corrupt_b2_fraction,
        )
        return ObservedTwoComplex(
            b1=observed_b1,
            b2=observed_b2,
            truth_dims=(num_vertices, num_edges, num_faces),
            kept_edges=kept_edges,
            flipped_b1=flipped_b1,
            flipped_b2=flipped_b2,
        )


def observed_commutation_residuals(
    chain_map: ChainMap, observed: ObservedTwoComplex
) -> tuple[float, float]:
    """Commutation residuals of ``chain_map`` against the observed pair.

    Returns ``(residual_deg1, residual_deg2)`` with

    - degree 1: ``||B1' f1[kept] - f0 B1||_F``, where ``f1[kept]``
      restricts the degree-1 map to the rows of the kept target edges —
      the observed ``B1'`` has exactly those columns, so both sides are
      maps from source edges to target vertices;
    - degree 2: ``||B2' f2 - f1[kept] B2||_F`` — the **kept-row
      convention**: the observed ``B2'`` holds only the kept ground-truth
      edge rows, so the source side must be pushed through the same
      row-restricted ``f1[kept]`` for the two sides to be comparable.
      Both sides are then maps from source faces to kept target edges.

    At full observation (every edge kept, no flips) these are exactly the
    chain map's commutation residuals — ``(0.0, 0.0)`` for a true planted
    map, in exact float64 integer arithmetic.
    """
    if chain_map.source.top_degree != 2 or chain_map.target.top_degree != 2:
        raise ValueError(
            "observed commutation residuals are defined between 2-complexes"
        )
    if observed.truth_dims != chain_map.target.dims:
        raise ValueError(
            f"observed truth dims {observed.truth_dims} do not match "
            f"chain map target dims {chain_map.target.dims}"
        )
    f0, f1, f2 = chain_map.maps
    kept = np.asarray(observed.kept_edges, dtype=int)
    f1_kept = f1[kept, :]
    left_deg1 = observed.b1 @ f1_kept
    right_deg1 = f0 @ chain_map.source.boundaries[0]
    left_deg2 = observed.b2 @ f2
    right_deg2 = f1_kept @ chain_map.source.boundaries[1]
    return (
        float(np.linalg.norm(left_deg1 - right_deg1)),
        float(np.linalg.norm(left_deg2 - right_deg2)),
    )


def observed_d2_residue(observed: ObservedTwoComplex) -> float:
    """``||B1' B2'||_F`` of the observed pair: how far it is from ``d^2 = 0``.

    Exactly ``0.0`` at full observation (the ground truth is a valid
    complex); generally positive under corruption or partial masking. Not
    a validity check — the observed object is a purported structure and
    this residue is a *measurement* on it, never a gate.
    """
    return float(np.linalg.norm(observed.b1 @ observed.b2))


def ranking_study_two(
    seed: int,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    *,
    mode: str = "corrupt",
    num_vertices: int = 8,
    num_edges: int = 14,
    num_classes: int = 4,
    num_decoys: int = 3,
) -> tuple[RankingRow, ...]:
    """Max-degree residual of the planted map per candidate, under degradation.

    Builds the planted 2-complex switch instance
    (:func:`universa.complexes2.make_two_complex_switch_instance`) and, at
    every fraction of ``fractions`` (non-decreasing, in ``[0, 1]``),
    observes every candidate — true target and decoys — through one shared
    :class:`TwoComplexObservationModel` draw (``mode="corrupt"`` sweeps
    both sign-corruption fractions together, ``mode="mask"`` sweeps edge
    masking; the other knob stays at 0). A candidate's misfit is the max
    of its two observed commutation residuals. The true candidate should
    keep ranking first (strictly lowest misfit) until a breakdown
    fraction, with the decoys' misfits as the reference floor. Every row
    is deterministic from ``seed``; use
    :func:`~universa.partial.breakdown_fraction` (re-exported here) for
    the first fraction where the true candidate stops ranking first.
    """
    if mode not in ("mask", "corrupt"):
        raise ValueError(f"mode must be 'mask' or 'corrupt', got {mode!r}")
    fractions = tuple(float(f) for f in fractions)
    if any(not 0.0 <= f <= 1.0 for f in fractions):
        raise ValueError("fractions must lie in [0, 1]")
    if any(b < a for a, b in zip(fractions, fractions[1:])):
        raise ValueError("fractions must be non-decreasing")
    instance = make_two_complex_switch_instance(
        seed, num_vertices, num_edges, num_classes, num_decoys
    )
    observation_seed = subseed(seed, "partial2", mode)
    rows = []
    for fraction in fractions:
        mask_fraction = fraction if mode == "mask" else 0.0
        corrupt_fraction = fraction if mode == "corrupt" else 0.0
        misfits = []
        for candidate in instance.candidates:
            observed = TwoComplexObservationModel(
                candidate,
                observation_seed,
                mask_fraction=mask_fraction,
                corrupt_b1_fraction=corrupt_fraction,
                corrupt_b2_fraction=corrupt_fraction,
            ).observe()
            probe = ChainMap(instance.source, candidate, instance.chain_map.maps)
            misfits.append(max(observed_commutation_residuals(probe, observed)))
        true_misfit = misfits[0]
        decoy_misfits = tuple(misfits[1:])
        floor = min(decoy_misfits, default=float("inf"))
        rows.append(
            RankingRow(
                fraction=fraction,
                true_misfit=true_misfit,
                decoy_misfits=decoy_misfits,
                floor=floor,
                margin=floor - true_misfit,
                true_ranks_first=all(
                    true_misfit < decoy for decoy in decoy_misfits
                ),
            )
        )
    return tuple(rows)
