"""Partial observation of structure: degraded boundary operators.

HOMYMOLY's exact constrained estimator assumes the boundary operator is
given exactly. This module covers the regime where it is not — the regime
design doc section 6 flags as the one where HOMYMOLY showed learning is
motivated: an *observed* 1-complex whose boundary differs from the ground
truth by

- **masked edges**: entire boundary columns removed (unobserved edges),
  and/or
- **corrupted signs**: a controlled fraction of the nonzero entries that
  remain sign-flipped.

Both degradations are deterministic from an integer seed via
:func:`universa.generators.subseed`, with explicit fractions, and both are
*nested*: the masked (resp. flipped) set at fraction ``f`` is a prefix of
one master permutation, so a larger fraction degrades strictly more of the
same matrix. Counts are ``round(fraction * total)`` (round-half-even).

:func:`observed_misfit` evaluates the commutation residual of a planted
chain map against the observed boundary — the exact reference signal a
router would read when the true boundary is unavailable. Every matrix here
is integer-valued in float64, so residuals stay exact, and the repo's
fail-closed certificate discipline applies: shapes and bookkeeping are
validated at construction, never assumed.

Documented behavior (default sizes, pinned study seed 2026, default 0.1
grid): under sign corruption the true candidate's residual rises strictly
from ``0.0`` at every grid step and the true candidate keeps ranking first
through fraction 0.5, breaking at 0.6; under edge masking it ranks first
through 0.9 and breaks only at 1.0, where no edge is observed and every
candidate ties at ``||f0 B1||_F``. The monotonicity is structural, not
sampled luck: a planted quotient map sends each source edge to exactly one
target edge row, so the contributions of individual flipped entries (or
masked columns) to the squared residual are Frobenius-orthogonal and the
residual is a nested sum — non-decreasing by construction, strictly
increasing whenever the degradation count grows.

This exact evaluator is the seam where a *learned amortized misfit
estimator* would plug in (design doc section 6): a model trained to
predict the misfit of a candidate map from the observed boundary would
stand in for :func:`observed_misfit` behind the same signature, and
:func:`ranking_study` is the audit harness scoring whether the ranking of
candidates — true structure first, decoys as the reference floor —
survives the degradation up to a breakdown fraction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from universa.generators import make_switch_instance, subseed
from universa.structures import ChainComplex, ChainMap

DEFAULT_FRACTIONS = tuple(i / 10 for i in range(11))
"""Default degradation grid for :func:`ranking_study` (0.0 to 1.0)."""


@dataclass(frozen=True)
class ObservedComplex:
    """A ground-truth 1-complex as actually observed, with provenance.

    ``complex`` is the observed 1-complex: the ground-truth boundary with
    the edge columns outside ``kept_edges`` removed and the entries at
    ``flipped_entries`` sign-flipped. ``kept_edges`` maps observed columns
    back to ground-truth edge indices and is strictly increasing;
    ``flipped_entries`` are ``(row, observed_column)`` positions of
    sign-flipped entries. ``truth_dims`` is ``(num_vertices, num_edges)``
    of the ground truth, so the dims contract
    ``complex.boundaries[0].shape == (truth_dims[0], len(kept_edges))``
    is checked here, at construction — fail-closed.
    """

    complex: ChainComplex
    truth_dims: tuple[int, int]
    kept_edges: tuple[int, ...]
    flipped_entries: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if self.complex.top_degree != 1:
            raise ValueError("observed complexes are 1-complexes")
        num_vertices, num_edges = self.truth_dims
        expected = (num_vertices, len(self.kept_edges))
        if self.complex.boundaries[0].shape != expected:
            raise ValueError(
                f"dims contract violated: boundary is "
                f"{self.complex.boundaries[0].shape}, expected {expected}"
            )
        if any(
            b <= a for a, b in zip(self.kept_edges, self.kept_edges[1:])
        ):
            raise ValueError("kept_edges must be strictly increasing")
        if self.kept_edges and not 0 <= self.kept_edges[-1] < num_edges:
            raise ValueError("kept_edges out of range")
        for row, col in self.flipped_entries:
            in_range = 0 <= row < num_vertices and 0 <= col < len(
                self.kept_edges
            )
            if not in_range:
                raise ValueError("flipped entry out of range")
            if self.complex.boundaries[0][row, col] == 0.0:
                raise ValueError("flipped entry must be a nonzero position")

    @property
    def num_masked(self) -> int:
        """How many ground-truth edge columns were removed."""
        return self.truth_dims[1] - len(self.kept_edges)


@dataclass(frozen=True)
class ObservationModel:
    """A ground-truth 1-complex plus the fractions of it we do not see.

    ``mask_fraction`` of the edge columns are unobserved (removed);
    ``corrupt_fraction`` of the nonzero entries that remain are
    sign-flipped. Masking is applied first, corruption to the surviving
    entries. Both draws are deterministic functions of ``seed`` and the
    fractions, via nested prefixes of master permutations drawn under
    ``subseed(seed, "observe", "mask" | "corrupt")``.
    """

    truth: ChainComplex
    seed: int
    mask_fraction: float = 0.0
    corrupt_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.truth.top_degree != 1:
            raise ValueError("partial observation is defined for 1-complexes")
        for name, value in (
            ("mask_fraction", self.mask_fraction),
            ("corrupt_fraction", self.corrupt_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}={value} outside [0, 1]")

    def observe(self) -> ObservedComplex:
        """Draw the degraded observation, deterministically."""
        boundary = self.truth.boundaries[0]
        num_vertices, num_edges = boundary.shape
        mask_rng = np.random.default_rng(
            subseed(self.seed, "observe", "mask")
        )
        edge_order = mask_rng.permutation(num_edges)
        num_masked = int(round(self.mask_fraction * num_edges))
        masked = {int(e) for e in edge_order[:num_masked]}
        kept_edges = tuple(e for e in range(num_edges) if e not in masked)

        observed = boundary[:, list(kept_edges)].copy()
        rows, cols = np.nonzero(observed)
        corrupt_rng = np.random.default_rng(
            subseed(self.seed, "observe", "corrupt")
        )
        entry_order = corrupt_rng.permutation(len(rows))
        num_flipped = int(round(self.corrupt_fraction * len(rows)))
        flipped = []
        for index in entry_order[:num_flipped]:
            row, col = int(rows[index]), int(cols[index])
            observed[row, col] = -observed[row, col]
            flipped.append((row, col))
        return ObservedComplex(
            complex=ChainComplex((observed,)),
            truth_dims=(num_vertices, num_edges),
            kept_edges=kept_edges,
            flipped_entries=tuple(sorted(flipped)),
        )


def observed_misfit(
    chain_map: ChainMap, observed_complex: ObservedComplex
) -> float:
    """Commutation residual of ``chain_map`` against the observed boundary.

    Computes ``||B1_obs f1[kept] - f0 B1||_F``: the exact commutation
    residual, evaluated against the observed (masked, sign-corrupted)
    target boundary restricted to the edges that were actually seen. At
    full observation this is exactly the chain map's commutation residual
    — ``0.0`` for a true planted map, in exact float64 integer arithmetic.
    """
    if chain_map.source.top_degree != 1 or chain_map.target.top_degree != 1:
        raise ValueError("observed misfit is defined between 1-complexes")
    target_dims = (chain_map.target.dims[0], chain_map.target.dims[1])
    if observed_complex.truth_dims != target_dims:
        raise ValueError(
            f"observed truth dims {observed_complex.truth_dims} do not "
            f"match chain map target dims {target_dims}"
        )
    f0, f1 = chain_map.maps
    kept = np.asarray(observed_complex.kept_edges, dtype=int)
    left = observed_complex.complex.boundaries[0] @ f1[kept, :]
    right = f0 @ chain_map.source.boundaries[0]
    return float(np.linalg.norm(left - right))


@dataclass(frozen=True)
class RankingRow:
    """One degradation level of a :func:`ranking_study`."""

    fraction: float
    true_misfit: float
    decoy_misfits: tuple[float, ...]
    floor: float  # min decoy misfit (inf when there are no decoys)
    margin: float  # floor - true_misfit; positive while the true leads
    true_ranks_first: bool  # true misfit strictly below every decoy


def ranking_study(
    seed: int,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    *,
    mode: str = "corrupt",
    num_vertices: int = 8,
    num_edges: int = 14,
    num_classes: int = 4,
    num_decoys: int = 3,
) -> tuple[RankingRow, ...]:
    """Misfit of the planted map against each candidate, under degradation.

    Builds the planted switch instance
    (:func:`universa.generators.make_switch_instance`) and, at every
    fraction of ``fractions`` (non-decreasing, in ``[0, 1]``), observes
    every candidate — true target and decoys — through one shared
    :class:`ObservationModel` draw (``mode="corrupt"`` sweeps sign flips,
    ``mode="mask"`` sweeps edge masking; the other knob stays at 0). The
    true candidate should keep ranking first (strictly lowest misfit)
    until a breakdown fraction, with the decoys' misfits as the reference
    floor. Every row is deterministic from ``seed``.
    """
    if mode not in ("mask", "corrupt"):
        raise ValueError(f"mode must be 'mask' or 'corrupt', got {mode!r}")
    fractions = tuple(float(f) for f in fractions)
    if any(not 0.0 <= f <= 1.0 for f in fractions):
        raise ValueError("fractions must lie in [0, 1]")
    if any(b < a for a, b in zip(fractions, fractions[1:])):
        raise ValueError("fractions must be non-decreasing")
    instance = make_switch_instance(
        seed, num_vertices, num_edges, num_classes, num_decoys
    )
    observation_seed = subseed(seed, "partial", mode)
    rows = []
    for fraction in fractions:
        mask_fraction = fraction if mode == "mask" else 0.0
        corrupt_fraction = fraction if mode == "corrupt" else 0.0
        misfits = [
            observed_misfit(
                instance.chain_map,
                ObservationModel(
                    candidate,
                    observation_seed,
                    mask_fraction=mask_fraction,
                    corrupt_fraction=corrupt_fraction,
                ).observe(),
            )
            for candidate in instance.candidates
        ]
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


def breakdown_fraction(rows: tuple[RankingRow, ...]) -> float | None:
    """First fraction at which the true candidate stops ranking first.

    ``None`` when the true candidate ranks first at every probed fraction.
    """
    for row in rows:
        if not row.true_ranks_first:
            return row.fraction
    return None
