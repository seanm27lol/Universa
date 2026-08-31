"""Partial observation of group nerves: degraded multiplication tables.

:mod:`universa.partial` degrades the boundary of a graph and
:mod:`universa.partial2` the boundary pair of a 2-complex; this module
degrades the *multiplication table* of a finite group seen through its bar
construction (:mod:`universa.category_instances`). For a
``corrupt_fraction`` of the ordered pairs ``(g, f)`` the observed product
is replaced by a different uniformly-drawn element, and the observed nerve
boundaries are then assembled **directly from the corrupted table** — not
through :func:`universa.nerve.nerve_chain_complex`, whose input validation
the corrupted table would not survive.

**The observed object is a purported structure, not the nerve of a
category.** A corrupted table generally fails associativity, so no
:class:`~universa.nerve.FiniteCategory` has these boundaries:
:class:`ObservedGroupNerve` holds the raw boundary arrays plus full
provenance *without* FiniteCategory's fail-closed associativity and
inverse checks — a plain frozen dataclass whose constructor checks shapes
and bookkeeping only. That is exactly the ``d^2``-handling convention of
:mod:`universa.partial2`, with one twist: with one object ``d1`` is the
zero map, so ``d1 d2 = 0`` holds *trivially* for any table whatsoever —
:class:`~universa.structures.ChainComplex`'s fail-closed ``d^2`` gate
cannot see the damage at all. The invalidity lives one level up, at the
category axioms, which is precisely why the reference signal in this
regime is the commutation residual against the observed boundaries, never
a validity test.

Corruption is deterministic from an integer seed via
:func:`universa.generators.subseed` (components ``"group-observe"``,
``"corrupt"``) and *nested*: the corrupted set at fraction ``f`` is a
prefix of one master permutation of the ``n^2`` table entries (flat index
``g * n + f``, the nerve's own pair ordering), with replacement elements
drawn in the same order, so a larger fraction degrades strictly more of
the same table with the same replacement values. Counts are
``round(fraction * n^2)`` (round-half-even); each replacement is uniform
over the ``n - 1`` elements different from the true product.

:func:`observed_nerve_commutation_residual` evaluates the degree-2
residual ``||d2' N2 - N1 d2||_F`` of a homomorphism-induced nerve map
against the observed boundaries — the exact reference signal a router
would read when the true table is unavailable. Degree 1 is trivially zero
(one object on both sides: ``d1' N1 = 0 = N0 d1``), so only degree 2 is
returned. Every matrix here is integer-valued in float64, so residuals
stay exact: at zero corruption the observed ``d2`` is assembled by the
same operations as the nerve's own and the residual is the induced map's
exact ``0.0``, never numerical residue.

The true candidate's residual has a closed form. The planted
homomorphisms are surjective, so every target table entry is hit by
exactly ``(s / m)^2`` source pairs (fibers of a group homomorphism are
kernel cosets). Corruption touches only the composite row of an observed
``d2`` column, so each source pair landing on a corrupted entry
contributes one Frobenius-orthogonal column ``e_{phi(g.f)} - e_{c'}`` of
squared norm 2, and

    true residual(f) = sqrt(2 * (s / m)^2 * round(f * m^2))  exactly,

monotone non-decreasing in ``f`` on *every* seed — strictly increasing
whenever the corruption count grows. On the default ``z6_to_z3`` grid
(m = 3) this is ``sqrt(8 * round(9 f))``; the count stalls between
fractions 0.4 and 0.5 (``round(9 * 0.4) == round(9 * 0.5) == 4``), the
only flat step.

Documented behavior (pinned study seed 70001, default family
``z6_to_z3``, two decoys, default 0.1 grid; :func:`ranking_study_group`):
the true candidate's residual follows the closed form above exactly, from
``0.0`` at fraction 0.0 to ``sqrt(72)`` at full corruption, flat only
between 0.4 and 0.5 (both ``sqrt(32)``). The decoy floor starts at
``sqrt(72) ~= 8.485`` and erodes under heavy corruption; the true
candidate keeps ranking first (strictly below both decoys) through
fraction 0.7, where the margin has narrowed to ``+0.555112``, and the
breakdown fraction is 0.8 (margin ``-0.555112``), losing strictly from
there on. The margin is widest at fraction 0.0 (``+8.485281``). See the
test suite for the pinned numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from universa.category_instances import (
    group_as_category,
    make_group_switch_instance,
)
from universa.generators import subseed
from universa.nerve import FiniteCategory
from universa.partial import DEFAULT_FRACTIONS, RankingRow, breakdown_fraction
from universa.structures import ChainMap

__all__ = [
    "DEFAULT_FRACTIONS",
    "GroupObservationModel",
    "ObservedGroupNerve",
    "RankingRow",
    "breakdown_fraction",
    "observed_nerve_commutation_residual",
    "ranking_study_group",
]


def _identity_of(group: FiniteCategory) -> int:
    """The unique neutral element of a one-object category's table."""
    order = len(group.morphisms)
    neutrals = [
        e
        for e in range(order)
        if all(
            group.compose[(e, x)] == x and group.compose[(x, e)] == x
            for x in range(order)
        )
    ]
    if len(neutrals) != 1:
        raise ValueError("table must have exactly one identity element")
    return neutrals[0]


def _table_of(group: FiniteCategory) -> np.ndarray:
    """The multiplication table of a one-object (group) category."""
    order = len(group.morphisms)
    return np.array(
        [[group.compose[(g, f)] for f in range(order)] for g in range(order)]
    )


@dataclass(frozen=True)
class ObservedGroupNerve:
    """A ground-truth group nerve as actually observed, with provenance.

    ``d1`` and ``d2`` are the *raw* observed nerve boundaries: ``d1`` the
    zero map ``(1, n)`` of the single object, ``d2`` of shape
    ``(n, n^2)`` assembled directly from the corrupted multiplication
    table in the pair ordering of
    :func:`universa.nerve.nerve_chain_complex` — column ``g * n + f``
    holds the purported faces ``e_g + e_f - e_{c}`` with ``c`` the
    (possibly corrupted) observed product. They are stored as plain
    arrays, **not** run through :class:`~universa.nerve.FiniteCategory`:
    the corrupted table generally fails associativity, so FiniteCategory's
    fail-closed validation would reject exactly the objects this module
    exists to describe (the convention of :mod:`universa.partial2` — with
    the twist that here even ``d^2 = 0`` cannot see the damage, since
    ``d1`` is zero). Only shapes and bookkeeping are checked here —
    fail-closed on bookkeeping, silent on validity.

    ``truth_order`` is the ground-truth group order ``n``;
    ``corrupted_entries`` are ``(g, f, old, new)` provenance: the product
    at ``(g, f)`` was observed as ``new`` instead of ``old``, and each
    claimed entry must be in range, have ``new != old``, appear at most
    once, and match the ``d2`` column exactly — ``old`` itself is not
    checkable here (the truth table lives in the model), only its
    replacement is.
    """

    d1: np.ndarray
    d2: np.ndarray
    truth_order: int
    corrupted_entries: tuple[tuple[int, int, int, int], ...]

    def __post_init__(self) -> None:
        for name, matrix in (("d1", self.d1), ("d2", self.d2)):
            if not isinstance(matrix, np.ndarray) or matrix.ndim != 2:
                raise ValueError(f"{name} is not a 2-D array")
        order = self.truth_order
        if order < 1:
            raise ValueError("a group needs at least one element")
        if self.d1.shape != (1, order):
            raise ValueError(
                f"dims contract violated: d1 is {self.d1.shape}, expected "
                f"{(1, order)}"
            )
        if self.d2.shape != (order, order * order):
            raise ValueError(
                f"dims contract violated: d2 is {self.d2.shape}, expected "
                f"{(order, order * order)}"
            )
        if np.any(self.d1 != 0.0):
            raise ValueError("d1 must be the zero map (one object)")
        for column in range(order * order):
            col = self.d2[:, column]
            if not np.all(col == np.round(col)) or col.sum() != 1.0:
                raise ValueError(
                    f"d2 column {column} is not a purported face column "
                    "(integer entries summing to 1)"
                )
        seen = set()
        for g, f, old, new in self.corrupted_entries:
            in_range = (
                0 <= g < order
                and 0 <= f < order
                and 0 <= old < order
                and 0 <= new < order
            )
            if not in_range:
                raise ValueError("corrupted entry out of range")
            if new == old:
                raise ValueError("corrupted entry must change the product")
            if (g, f) in seen:
                raise ValueError(f"duplicate corrupted entry ({g}, {f})")
            seen.add((g, f))
            expected = np.zeros(order)
            expected[g] += 1.0
            expected[new] -= 1.0
            expected[f] += 1.0
            if not np.array_equal(self.d2[:, g * order + f], expected):
                raise ValueError(
                    f"provenance inconsistent with d2 at ({g}, {f})"
                )

    @property
    def num_corrupted(self) -> int:
        """How many table entries were corrupted."""
        return len(self.corrupted_entries)


@dataclass(frozen=True)
class GroupObservationModel:
    """A ground-truth group plus the fraction of its table we misread.

    ``truth`` is the group as a one-object
    :class:`~universa.nerve.FiniteCategory`. Fail-closed: the category
    must be one-object and satisfy the group inverse axiom — a monoid
    that is not a group is rejected here, not downstream — and corrupting
    a one-element group is an error (no different element exists to
    draw). ``corrupt_fraction`` of the ``n^2`` table entries have their
    product replaced by a different uniformly-drawn element, via nested
    prefixes of one master permutation drawn under
    ``subseed(seed, "group-observe", "corrupt")``.
    """

    truth: FiniteCategory
    seed: int
    corrupt_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.truth.num_objects != 1:
            raise ValueError(
                "group observation is defined for one-object categories"
            )
        order = len(self.truth.morphisms)
        identity = _identity_of(self.truth)
        for element in range(order):
            inverse_exists = any(
                self.truth.compose[(element, other)] == identity
                and self.truth.compose[(other, element)] == identity
                for other in range(order)
            )
            if not inverse_exists:
                raise ValueError(
                    f"element {element} has no inverse: truth is not a group"
                )
        if not 0.0 <= self.corrupt_fraction <= 1.0:
            raise ValueError(
                f"corrupt_fraction={self.corrupt_fraction} outside [0, 1]"
            )
        if self.corrupt_fraction > 0.0 and order < 2:
            raise ValueError(
                "a one-element group has no different element to draw"
            )

    def observe(self) -> ObservedGroupNerve:
        """Draw the degraded observation, deterministically."""
        order = len(self.truth.morphisms)
        table = _table_of(self.truth)
        rng = np.random.default_rng(
            subseed(self.seed, "group-observe", "corrupt")
        )
        entry_order = rng.permutation(order * order)
        num_corrupted = int(round(self.corrupt_fraction * order * order))
        corrupted = []
        for flat in entry_order[:num_corrupted]:
            g, f = divmod(int(flat), order)
            old = int(table[g, f])
            # Uniform over the order - 1 elements different from old.
            draw = int(rng.integers(0, order - 1))
            new = draw if draw < old else draw + 1
            table[g, f] = new
            corrupted.append((g, f, old, new))

        d1 = np.zeros((1, order))
        d2 = np.zeros((order, order * order))
        for g in range(order):
            for f in range(order):
                column = g * order + f  # nerve pair ordering: g outer
                d2[g, column] += 1.0  # face d0
                d2[int(table[g, f]), column] -= 1.0  # face d1 (composite)
                d2[f, column] += 1.0  # face d2
        return ObservedGroupNerve(
            d1=d1,
            d2=d2,
            truth_order=order,
            corrupted_entries=tuple(sorted(corrupted)),
        )


def observed_nerve_commutation_residual(
    chain_map: ChainMap, observed: ObservedGroupNerve
) -> float:
    """Degree-2 commutation residual against the observed boundaries.

    Computes ``||d2' N2 - N1 d2||_F``: the exact degree-2 commutation
    residual of a homomorphism-induced nerve map, evaluated against the
    observed (corrupted-table) target boundary. Degree 1 is trivially
    zero — one object on both sides makes ``d1' N1 = 0 = N0 d1`` — so it
    is not returned. At zero corruption the observed ``d2`` is the nerve's
    own boundary, so this is exactly the chain map's degree-2 residual:
    ``0.0`` for a true induced map, in exact float64 integer arithmetic.
    """
    if chain_map.source.top_degree != 2 or chain_map.target.top_degree != 2:
        raise ValueError(
            "observed nerve residuals are defined between 2-complexes "
            "(levels 0-2 of a nerve)"
        )
    order = observed.truth_order
    expected_dims = (1, order, order * order)
    if chain_map.target.dims != expected_dims:
        raise ValueError(
            f"observed truth order {order} does not match chain map "
            f"target dims {chain_map.target.dims}"
        )
    _, n1, n2 = chain_map.maps
    left = observed.d2 @ n2
    right = n1 @ chain_map.source.boundaries[1]
    return float(np.linalg.norm(left - right))


def _group_from_nerve(complex_) -> FiniteCategory:
    """Recover the group a nerve 2-complex came from, fail-closed.

    :func:`make_group_switch_instance` exposes its candidates only as
    nerves, but the observation model needs the group itself. In a group
    nerve the identity row is the unique row holding ``2n - 1``
    single-+1 columns (every pair involving the identity collapses to
    ``e_e``); for any other pair the composite sits at the column's
    unique -1 entry, and pairs involving the identity are forced. The
    recovered table is rebuilt through
    :func:`~universa.category_instances.group_as_category`, so an input
    that is not the nerve of a group is an error here, not downstream.
    """
    if complex_.top_degree != 2:
        raise ValueError("a group nerve is a 2-complex")
    d1, d2 = complex_.boundaries
    order = d1.shape[1]
    if d1.shape[0] != 1 or d2.shape != (order, order * order):
        raise ValueError(f"not a group nerve: dims {complex_.dims}")
    if np.any(d1 != 0.0):
        raise ValueError("a group nerve has zero d1 (one object)")
    counts = np.zeros(order, dtype=int)
    for column in range(order * order):
        col = d2[:, column]
        nonzero = np.nonzero(col)[0]
        if len(nonzero) == 1 and col[nonzero[0]] == 1.0:
            counts[nonzero[0]] += 1
    expected_singles = 2 * order - 1
    identities = [r for r in range(order) if counts[r] == expected_singles]
    if len(identities) != 1 or int(counts.sum()) != expected_singles:
        raise ValueError(
            "not a group nerve: identity row is not uniquely determined"
        )
    identity = identities[0]
    table = np.zeros((order, order), dtype=int)
    for g in range(order):
        for f in range(order):
            column = d2[:, g * order + f]
            if f == identity:
                composite = g
            elif g == identity:
                composite = f
            else:
                negatives = np.nonzero(column < 0.0)[0]
                if len(negatives) != 1 or column[negatives[0]] != -1.0:
                    raise ValueError("not a group nerve: bad face column")
                composite = int(negatives[0])
            expected = np.zeros(order)
            expected[g] += 1.0
            expected[composite] -= 1.0
            expected[f] += 1.0
            if not np.array_equal(column, expected):
                raise ValueError("not a group nerve: bad face column")
            table[g, f] = composite
    return group_as_category(table)


def ranking_study_group(
    seed: int,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    *,
    family: str = "z6_to_z3",
    num_decoys: int = 2,
) -> tuple[RankingRow, ...]:
    """Observed degree-2 residual of the induced map per candidate.

    Builds the planted group switch instance
    (:func:`universa.category_instances.make_group_switch_instance`) and,
    at every fraction of ``fractions`` (non-decreasing, in ``[0, 1]``),
    observes every candidate — true target and decoys — through one
    shared :class:`GroupObservationModel` draw (same observation seed,
    same group order: the same table positions are corrupted, with the
    same replacement draws, in every candidate). A candidate's misfit is
    :func:`observed_nerve_commutation_residual` of the planted induced
    map against that observation. The true candidate's misfit follows the
    closed form in the module docstring and should keep ranking first
    (strictly lowest) until a breakdown fraction, with the decoys'
    misfits as the reference floor. Every row is deterministic from
    ``seed``; use :func:`~universa.partial.breakdown_fraction`
    (re-exported here) for the first fraction where the true candidate
    stops ranking first.
    """
    fractions = tuple(float(f) for f in fractions)
    if any(not 0.0 <= f <= 1.0 for f in fractions):
        raise ValueError("fractions must lie in [0, 1]")
    if any(b < a for a, b in zip(fractions, fractions[1:])):
        raise ValueError("fractions must be non-decreasing")
    instance = make_group_switch_instance(
        seed, family=family, num_decoys=num_decoys
    )
    observation_seed = subseed(seed, "partial-group")
    rows = []
    for fraction in fractions:
        misfits = []
        for candidate in instance.candidates:
            observed = GroupObservationModel(
                _group_from_nerve(candidate),
                observation_seed,
                corrupt_fraction=fraction,
            ).observe()
            probe = ChainMap(
                instance.source, candidate, instance.chain_map.maps
            )
            misfits.append(observed_nerve_commutation_residual(probe, observed))
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
