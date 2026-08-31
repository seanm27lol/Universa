"""Partial observation of cellular sheaves: degraded restriction maps.

:mod:`universa.partial` degrades the boundary of a 1-complex and
:mod:`universa.partial2` a boundary pair; this module covers the second
synthetic family — a cellular sheaf (:mod:`universa.sheaves`) — observed
through

- **masked edges**: a fraction of the edges unobserved; their restriction
  blocks vanish, so the corresponding block rows drop out of the observed
  coboundary, with the kept-edge indices carried as provenance; and
- **corrupted signs**: a controlled fraction of the nonzero entries of the
  surviving restriction blocks sign-flipped.

Both draws are deterministic from an integer seed via
:func:`universa.generators.subseed` (components ``"sheaf-observe"``,
``"mask"`` | ``"corrupt"``) and *nested*: each degraded set at fraction
``f`` is a prefix of one master permutation, so a larger fraction degrades
strictly more of the same blocks. Counts are ``round(fraction * total)``
(round-half-even). Masking is applied first, corruption to the surviving
entries — the convention of :mod:`universa.partial`.

**The observed object is a purported structure, not a sheaf.** With any
edge masked it does not cover the incidences of its own base graph, so
:class:`~universa.sheaves.Sheaf`'s fail-closed incidence validation would
reject exactly the objects this module exists to describe.
:class:`ObservedSheaf` therefore holds the degraded restriction blocks
plus full provenance and checks shapes and bookkeeping only — fail-closed
on provenance, silent on coherence. That is the point of the regime, not
a validation hole: the reference signal here is the naturality residual
of a planted morphism against the observed restrictions, never a validity
test on the observation.

:func:`observed_naturality_residual` evaluates
``||rho'_obs phi_v - phi_e rho||_F`` over the surviving incidences — the
planted morphism's naturality read against the degraded target sheaf.
Exactness is the dyadic-scaling argument of :mod:`universa.sheaves`,
extended by one exact step: planted target restrictions are integers
times a power of two, stalk maps are signed permutations scaled by +/-1
or +/-2, and a sign flip negates a float64 entry — an exact operation —
so a flipped entry's contribution ``2 * rho'[i, j] * scale_v`` to a
residual row is another dyadic value (an exponent shift, never mantissa
rounding). At zero degradation the residual is the planted morphism's
exact ``0.0``, and no arithmetic beyond dyadic fractions ever arises.

The two modes degrade that signal in structurally different ways. Under
corruption the true candidate's residual is *structurally monotone*:
``phi_v`` is a scaled permutation, so each flipped entry moves exactly
one position of the incidence residual, the moved positions of distinct
flips never coincide, and the squared residual is a nested sum of
positive terms — strictly increasing whenever the flip count grows (the
same Frobenius-orthogonality argument as :mod:`universa.partial`, one
stalk level down). Under masking the residual runs over *surviving*
incidences only, so the true candidate stays at exactly ``0.0`` (every
surviving block is exact) while each decoy's residual erodes
monotonically as its discriminating terms drop out; discrimination
survives only while every decoy keeps a discriminating incidence, and at
full masking every candidate ties at ``0.0``. Masking mode is therefore
a coverage study whose margin never grows — unlike
:mod:`universa.partial`'s mask mode, where the source side stays fully
observed and the true residual rises.

Documented behavior (default sizes, pinned study seed 70001, default 0.1
grid; :func:`ranking_study_sheaf`). Under sign corruption
(``mode="corrupt"``) the true candidate's residual rises strictly from
``0.0`` at every grid step (47 flippable entries; the flip count grows at
every step) and the true candidate keeps ranking first through fraction
0.4 — the margin shrinking from 25.36 at full observation to 0.76 at 0.4
— and loses strictly from 0.5 on (the breakdown fraction, margin
−0.66). Under edge masking (``mode="mask"``) the true candidate's
residual is exactly ``0.0`` at every level while the decoys' residuals
erode monotonically (fractions 0.4 and 0.5 mask the same
``round(f * 9) = 4`` edges, so their rows coincide); the true candidate
ranks first through 0.9 and breaks only at 1.0, where no incidence
survives and every candidate ties at ``0.0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from universa.generators import subseed
from universa.partial import (
    DEFAULT_FRACTIONS,
    RankingRow,
    breakdown_fraction,
)
from universa.sheaves import (
    Sheaf,
    SheafMorphism,
    _edge_endpoints,
    make_sheaf_switch_instance,
    planted_morphism,
    random_sheaf,
    to_chain_complex,
)
from universa.structures import ChainComplex

__all__ = [
    "DEFAULT_FRACTIONS",
    "ObservedSheaf",
    "RankingRow",
    "SheafObservationModel",
    "breakdown_fraction",
    "observed_coboundary",
    "observed_naturality_residual",
    "ranking_study_sheaf",
]


@dataclass(frozen=True)
class ObservedSheaf:
    """A ground-truth sheaf as actually observed, with provenance.

    ``restrictions`` holds the *degraded* restriction blocks of the kept
    edges only: the masked edges' blocks are gone (their block rows drop
    out of the observed coboundary), and the surviving blocks carry sign
    flips at the ``flipped_entries`` sites. ``base``, ``vertex_dims`` and
    ``edge_dims`` are the ground truth's, kept as the indexing reference
    the provenance maps back into; ``kept_edges`` is strictly increasing
    and ``flipped_entries`` are ``(vertex, edge, row, col)`` sites of
    sign-flipped entries, one per flip.

    This is a purported structure, not a
    :class:`~universa.sheaves.Sheaf`: with any edge masked it does not
    cover the incidences of its own base (see the module docstring), so
    only shapes and bookkeeping are checked here, at construction —
    fail-closed on provenance, silent on coherence.
    """

    base: ChainComplex
    vertex_dims: tuple[int, ...]
    edge_dims: tuple[int, ...]
    restrictions: dict[tuple[int, int], np.ndarray]
    kept_edges: tuple[int, ...]
    flipped_entries: tuple[tuple[int, int, int, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.base, ChainComplex) or self.base.top_degree != 1:
            raise ValueError("the base of an observed sheaf is a 1-complex")
        boundary = self.base.boundaries[0]
        num_vertices, num_edges = boundary.shape
        tails, heads = _edge_endpoints(boundary)
        for d in (*self.vertex_dims, *self.edge_dims):
            if not isinstance(d, (int, np.integer)):
                raise ValueError("stalk dimensions must be integers")
        vertex_dims = tuple(int(d) for d in self.vertex_dims)
        edge_dims = tuple(int(d) for d in self.edge_dims)
        if len(vertex_dims) != num_vertices or len(edge_dims) != num_edges:
            raise ValueError("stalk dims must cover every vertex and every edge")
        if any(d < 1 for d in (*vertex_dims, *edge_dims)):
            raise ValueError("stalk dimensions must be positive")
        kept_edges = tuple(int(e) for e in self.kept_edges)
        if any(b <= a for a, b in zip(kept_edges, kept_edges[1:])):
            raise ValueError("kept_edges must be strictly increasing")
        if kept_edges and not 0 <= kept_edges[-1] < num_edges:
            raise ValueError("kept_edges out of range")
        expected = {
            (int(v), e)
            for e in kept_edges
            for v in (int(tails[e]), int(heads[e]))
        }
        restrictions: dict[tuple[int, int], np.ndarray] = {}
        for (v, e), block in self.restrictions.items():
            v, e = int(v), int(e)
            block = np.asarray(block, dtype=np.float64)
            if block.ndim != 2:
                raise ValueError(f"restriction {(v, e)} is not a 2-D block")
            if (v, e) not in expected:
                raise ValueError(
                    f"restriction {(v, e)} is not a surviving incidence"
                )
            if block.shape != (edge_dims[e], vertex_dims[v]):
                raise ValueError(
                    f"restriction {(v, e)} has shape {block.shape}, "
                    f"expected {(edge_dims[e], vertex_dims[v])}"
                )
            if not np.isfinite(block).all():
                raise ValueError(f"restriction {(v, e)} has non-finite entries")
            restrictions[(v, e)] = block
        if set(restrictions) != expected:
            missing = expected - set(restrictions)
            extra = set(restrictions) - expected
            raise ValueError(
                "restrictions must cover exactly the surviving incidences "
                f"(missing {sorted(missing)}, extra {sorted(extra)})"
            )
        for v, e, row, col in self.flipped_entries:
            if (int(v), int(e)) not in restrictions:
                raise ValueError(
                    f"flipped entry {(v, e, row, col)} is not on a "
                    "surviving incidence"
                )
            if not (0 <= row < edge_dims[e] and 0 <= col < vertex_dims[v]):
                raise ValueError("flipped entry out of range")
            if restrictions[(int(v), int(e))][row, col] == 0.0:
                raise ValueError("flipped entry must be a nonzero position")
        object.__setattr__(self, "vertex_dims", vertex_dims)
        object.__setattr__(self, "edge_dims", edge_dims)
        object.__setattr__(self, "restrictions", restrictions)
        object.__setattr__(self, "kept_edges", kept_edges)

    @property
    def num_vertices(self) -> int:
        return self.base.boundaries[0].shape[0]

    @property
    def num_edges(self) -> int:
        """Ground-truth edge count (observed plus masked)."""
        return self.base.boundaries[0].shape[1]

    @property
    def truth_dims(self) -> tuple[int, int]:
        """``(num_vertices, num_edges)`` of the ground-truth sheaf."""
        return (self.num_vertices, self.num_edges)

    @property
    def num_masked(self) -> int:
        """How many ground-truth edges were removed."""
        return self.num_edges - len(self.kept_edges)

    @property
    def c0_dim(self) -> int:
        """Dimension of C^0 (the direct sum of vertex stalks)."""
        return sum(self.vertex_dims)

    @property
    def c1_dim(self) -> int:
        """Observed dimension of C^1: kept edge stalks only."""
        return sum(self.edge_dims[e] for e in self.kept_edges)

    @property
    def incidence_pairs(self) -> tuple[tuple[int, int], ...]:
        """Every surviving incidence ``(v, e)``, by edge, tail before head."""
        tails, heads = _edge_endpoints(self.base.boundaries[0])
        return tuple(
            (int(v), e)
            for e in self.kept_edges
            for v in (int(tails[e]), int(heads[e]))
        )


@dataclass(frozen=True)
class SheafObservationModel:
    """A ground-truth sheaf plus the fractions of it we do not see.

    ``truth`` must be a valid :class:`~universa.sheaves.Sheaf` (the ground
    truth *is* a sheaf — only its observation is not).
    ``mask_fraction`` of the edges are unobserved (their restriction
    blocks removed); ``corrupt_fraction`` of the nonzero restriction
    entries that remain are sign-flipped. Masking is applied first,
    corruption to the surviving entries. Both draws are deterministic
    functions of ``seed`` and the fractions, via nested prefixes of
    master permutations drawn under
    ``subseed(seed, "sheaf-observe", "mask" | "corrupt")``.
    """

    truth: Sheaf
    seed: int
    mask_fraction: float = 0.0
    corrupt_fraction: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.truth, Sheaf):
            raise ValueError(
                "truth must be a universa.sheaves.Sheaf, "
                f"got {type(self.truth).__name__}"
            )
        for name, value in (
            ("mask_fraction", self.mask_fraction),
            ("corrupt_fraction", self.corrupt_fraction),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name}={value} outside [0, 1]")

    def observe(self) -> ObservedSheaf:
        """Draw the degraded observation, deterministically."""
        truth = self.truth
        num_edges = truth.num_edges
        mask_rng = np.random.default_rng(
            subseed(self.seed, "sheaf-observe", "mask")
        )
        edge_order = mask_rng.permutation(num_edges)
        num_masked = int(round(self.mask_fraction * num_edges))
        masked = {int(e) for e in edge_order[:num_masked]}
        kept_edges = tuple(e for e in range(num_edges) if e not in masked)

        restrictions = {
            pair: truth.restrictions[pair].copy()
            for pair in truth.incidence_pairs
            if pair[1] in kept_edges
        }
        # Canonical ordering of the surviving nonzero positions: by kept
        # edge, tail before head, C-order within each block.
        positions = []
        for v, e in truth.incidence_pairs:
            if e not in kept_edges:
                continue
            rows, cols = np.nonzero(restrictions[(v, e)])
            positions.extend(
                (v, e, int(r), int(c)) for r, c in zip(rows, cols)
            )
        corrupt_rng = np.random.default_rng(
            subseed(self.seed, "sheaf-observe", "corrupt")
        )
        entry_order = corrupt_rng.permutation(len(positions))
        num_flipped = int(round(self.corrupt_fraction * len(positions)))
        flipped = []
        for index in entry_order[:num_flipped]:
            v, e, row, col = positions[int(index)]
            restrictions[(v, e)][row, col] = -restrictions[(v, e)][row, col]
            flipped.append((v, e, row, col))
        return ObservedSheaf(
            base=truth.base,
            vertex_dims=truth.vertex_dims,
            edge_dims=truth.edge_dims,
            restrictions=restrictions,
            kept_edges=kept_edges,
            flipped_entries=tuple(sorted(flipped)),
        )


def observed_naturality_residual(
    morphism: SheafMorphism, observed: ObservedSheaf
) -> float:
    """Naturality residual of ``morphism`` against the observed sheaf.

    The root-sum-square of ``||rho'_obs phi_v - phi_e rho||_F`` over the
    surviving incidences, where ``rho'_obs`` is the observed (degraded)
    target-side restriction and ``rho`` the morphism's source
    restriction. At full observation this is exactly the root-sum-square
    of the morphism's per-incidence naturality residuals — ``0.0`` for a
    true planted morphism, in exact float64 dyadic arithmetic.
    """
    if not isinstance(morphism, SheafMorphism):
        raise ValueError("observed naturality needs a SheafMorphism")
    if not np.array_equal(
        observed.base.boundaries[0], morphism.source.base.boundaries[0]
    ):
        raise ValueError("observed sheaf and morphism share no base graph")
    if (
        observed.vertex_dims != morphism.target.vertex_dims
        or observed.edge_dims != morphism.target.edge_dims
    ):
        raise ValueError(
            "observed stalk dims do not match the morphism's target"
        )
    residuals = []
    for v, e in observed.incidence_pairs:
        left = observed.restrictions[(v, e)] @ morphism.vertex_maps[v]
        right = morphism.edge_maps[e] @ morphism.source.restrictions[(v, e)]
        residuals.append(float(np.linalg.norm(left - right)))
    return float(np.sqrt(sum(r * r for r in residuals)))


def observed_coboundary(observed: ObservedSheaf) -> np.ndarray:
    """The observed block coboundary, shape ``(c1_dim, c0_dim)``.

    The block assembly of :func:`universa.sheaves.coboundary` restricted
    to the kept edges: ``(delta x)_e = rho_{h,e} x_h - rho_{t,e} x_t``
    for kept ``e = (t -> h)``, with the masked edges' block rows dropped.
    At full observation this is exactly the ground truth's coboundary.
    """
    tails, heads = _edge_endpoints(observed.base.boundaries[0])
    delta = np.zeros((observed.c1_dim, observed.c0_dim))
    vertex_offsets = np.concatenate(([0], np.cumsum(observed.vertex_dims)))
    kept_offsets = np.concatenate(
        ([0], np.cumsum([observed.edge_dims[e] for e in observed.kept_edges]))
    )
    for slot, e in enumerate(observed.kept_edges):
        rows = slice(kept_offsets[slot], kept_offsets[slot + 1])
        for v, sign in ((int(tails[e]), -1.0), (int(heads[e]), 1.0)):
            cols = slice(vertex_offsets[v], vertex_offsets[v + 1])
            delta[rows, cols] = sign * observed.restrictions[(v, e)]
    return delta


def _sheaf_from_coboundary(
    base: ChainComplex,
    vertex_dims: tuple[int, ...],
    edge_dims: tuple[int, ...],
    delta: np.ndarray,
) -> Sheaf:
    """Recover the sheaf whose coboundary is ``delta``, block by block.

    :func:`universa.sheaves.coboundary` places ``sign * rho_{v,e}`` in the
    ``(e, v)`` block, so ``sign * delta_block`` (signs +/-1, exact)
    restores each restriction exactly. This is how
    :func:`ranking_study_sheaf` reads
    :func:`~universa.sheaves.make_sheaf_switch_instance`'s compiled
    candidates back as sheaves without duplicating their generation.
    """
    if delta.shape != (sum(edge_dims), sum(vertex_dims)):
        raise ValueError(
            f"coboundary shape {delta.shape} does not match stalk dims "
            f"{(sum(edge_dims), sum(vertex_dims))}"
        )
    tails, heads = _edge_endpoints(base.boundaries[0])
    vertex_offsets = np.concatenate(([0], np.cumsum(vertex_dims)))
    edge_offsets = np.concatenate(([0], np.cumsum(edge_dims)))
    restrictions = {}
    for e in range(base.boundaries[0].shape[1]):
        rows = slice(edge_offsets[e], edge_offsets[e + 1])
        for v, sign in ((int(tails[e]), -1.0), (int(heads[e]), 1.0)):
            cols = slice(vertex_offsets[v], vertex_offsets[v + 1])
            restrictions[(v, e)] = sign * delta[rows, cols]
    return Sheaf(base, vertex_dims, edge_dims, restrictions)


def ranking_study_sheaf(
    seed: int,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    *,
    mode: str = "corrupt",
    num_vertices: int = 6,
    num_edges: int = 9,
    max_stalk_dim: int = 3,
    num_decoys: int = 3,
) -> tuple[RankingRow, ...]:
    """Naturality residual of the planted morphism per candidate, degraded.

    Builds the planted sheaf switch instance
    (:func:`universa.sheaves.make_sheaf_switch_instance`) and, at every
    fraction of ``fractions`` (non-decreasing, in ``[0, 1]``), observes
    every candidate — true target and decoys, read back as sheaves block
    by block from the instance's compiled coboundaries — through one
    shared :class:`SheafObservationModel` draw (``mode="corrupt"`` sweeps
    sign flips, ``mode="mask"`` sweeps edge masking; the other knob stays
    at 0). A candidate's misfit is the planted morphism's naturality
    residual against its observed restrictions. The true candidate should
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
    instance = make_sheaf_switch_instance(
        seed, num_vertices, num_edges, max_stalk_dim, num_decoys
    )
    source = random_sheaf(seed, num_vertices, num_edges, max_stalk_dim)
    _, morphism = planted_morphism(source, seed)
    # Fail-closed consistency: the replanted morphism must be the one the
    # instance compiled (deterministic regeneration is a design invariant,
    # not an assumption to leave unchecked).
    compiled = morphism.to_chain_map()
    if not np.array_equal(
        compiled.target.boundaries[0], instance.true_target.boundaries[0]
    ):
        raise RuntimeError("replanted morphism disagrees with the instance")
    candidates = tuple(
        _sheaf_from_coboundary(
            source.base, source.vertex_dims, source.edge_dims,
            candidate.boundaries[0],
        )
        for candidate in instance.candidates
    )
    observation_seed = subseed(seed, "partial-sheaf", mode)
    rows = []
    for fraction in fractions:
        mask_fraction = fraction if mode == "mask" else 0.0
        corrupt_fraction = fraction if mode == "corrupt" else 0.0
        misfits = [
            observed_naturality_residual(
                morphism,
                SheafObservationModel(
                    candidate,
                    observation_seed,
                    mask_fraction=mask_fraction,
                    corrupt_fraction=corrupt_fraction,
                ).observe(),
            )
            for candidate in candidates
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
