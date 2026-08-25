"""Second synthetic family: finite groups through the bar construction.

A finite group is a one-object category, so its Grothendieck nerve is the
simplicial bar construction: N0 is the single object, N1 the elements, N2
all ordered pairs, with ``d2(f, g) = g - gf + f``. A group homomorphism
``phi: G -> H`` induces a chain map between the nerves — apply ``phi`` to
every simplex — and commutation is *exact*: the degree-2 square over the
pair ``(f, g)`` closes iff ``phi(gf) = phi(g) phi(f)``, an integer index
equality checked exhaustively on the multiplication tables before any
matrix is built, never a floating-point coincidence.

Planted instances — Z/6 -> Z/3 by reduction mod 3, Z/4 -> Z/2, and the
sign homomorphism S3 -> Z/2 — give auditable routing ground truth like
the graph-quotient family in :mod:`universa.generators`: the true
target's commutation residual is exactly 0.0, decoy target groups (the
target under a relabeled multiplication table, same order) stay positive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from universa.generators import subseed
from universa.nerve import FiniteCategory, nerve_chain_complex
from universa.structures import ChainComplex, ChainMap


def group_as_category(table: np.ndarray) -> FiniteCategory:
    """A finite group as a one-object category, from its multiplication
    table.

    ``table[g][f]`` is the product ``g . f`` ("g after f"), with elements
    numbered ``0..n-1``; element ``i`` becomes morphism ``i``. Fail-closed:
    the table must be square with integer entries in range, and the result
    must pass :class:`FiniteCategory`'s exhaustive associativity and
    identity checks plus the group inverse axiom — a monoid that is not a
    group is rejected here, not downstream.
    """
    array = np.asarray(table)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("a multiplication table must be square")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("table entries must be integers")
    order = array.shape[0]
    if order < 1:
        raise ValueError("a group needs at least one element")
    if int(array.min()) < 0 or int(array.max()) >= order:
        raise ValueError("table entries must be element indices 0..n-1")
    morphisms = tuple((0, 0) for _ in range(order))
    compose = {
        (g, f): int(array[g, f]) for g in range(order) for f in range(order)
    }
    category = FiniteCategory(1, morphisms, compose)
    identity = _derive_identity(array)
    for element in range(order):
        inverse_exists = any(
            int(array[element, other]) == identity
            and int(array[other, element]) == identity
            for other in range(order)
        )
        if not inverse_exists:
            raise ValueError(f"element {element} has no inverse: not a group")
    return category


def _derive_identity(table: np.ndarray) -> int:
    """The unique two-sided identity of a validated table."""
    order = table.shape[0]
    neutrals = [
        e
        for e in range(order)
        if all(
            int(table[e, x]) == x and int(table[x, e]) == x
            for x in range(order)
        )
    ]
    if len(neutrals) != 1:
        raise ValueError("table must have exactly one identity element")
    return neutrals[0]


def cyclic_group(n: int) -> FiniteCategory:
    """Z/n as a one-object category: ``table[g][f] = (g + f) % n``."""
    if n < 1:
        raise ValueError("need n >= 1")
    table = np.array([[(g + f) % n for f in range(n)] for g in range(n)])
    return group_as_category(table)


_S3 = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)


def symmetric_group_3() -> FiniteCategory:
    """S3, the permutations of three elements, as a one-object category.

    Elements are the six permutations in lexicographic order of their
    one-line notation ``(p(0), p(1), p(2))``; ``table[g][f]`` composes
    ``g`` after ``f``.
    """
    index = {perm: i for i, perm in enumerate(_S3)}
    table = np.array(
        [
            [index[tuple(g[f[x]] for x in range(3))] for f in _S3]
            for g in _S3
        ]
    )
    return group_as_category(table)


def _parity(perm: tuple[int, ...]) -> int:
    """The sign of a permutation as 0 (even) / 1 (odd), via inversions."""
    inversions = sum(
        1
        for i in range(len(perm))
        for j in range(i + 1, len(perm))
        if perm[i] > perm[j]
    )
    return inversions % 2


def induced_nerve_map(
    phi: np.ndarray,
    source_group: FiniteCategory,
    target_group: FiniteCategory,
) -> ChainMap:
    """The exact chain map a group homomorphism induces on the nerves.

    ``phi[x]`` is the image of element ``x``; any integer sequence of the
    source group's order works. N0 is the identity on the single object,
    N1 sends each element ``x`` to the basis vector of ``phi[x]``, and N2
    sends each composable pair ``(f, g)`` to ``(phi[f], phi[g])``.
    Fail-closed: ``phi`` is checked against the multiplication tables —
    ``phi(g . f) == phi(g) . phi(f)`` for every pair — before anything is
    built; a non-homomorphism is an error, not a warning.
    """
    if source_group.num_objects != 1 or target_group.num_objects != 1:
        raise ValueError(
            "nerve maps here are between one-object (group) categories"
        )
    source_order = len(source_group.morphisms)
    target_order = len(target_group.morphisms)
    images = np.asarray(phi)
    if images.ndim != 1 or images.shape[0] != source_order:
        raise ValueError(
            f"phi must map each of the {source_order} source elements"
        )
    if not np.issubdtype(images.dtype, np.integer):
        raise ValueError("phi must be integer-valued")
    if int(images.min()) < 0 or int(images.max()) >= target_order:
        raise ValueError("phi images must be target element indices")
    images = images.astype(np.int64)
    for g in range(source_order):
        for f in range(source_order):
            composite = source_group.compose[(g, f)]
            image = target_group.compose[(int(images[g]), int(images[f]))]
            if int(images[composite]) != image:
                raise ValueError(
                    f"phi is not a homomorphism: fails on (g={g}, f={f})"
                )
    source_complex = nerve_chain_complex(source_group)
    target_complex = nerve_chain_complex(target_group)
    n0 = np.ones((1, 1))
    n1 = np.zeros((target_order, source_order))
    n1[images, np.arange(source_order)] = 1.0
    # nerve_chain_complex orders pairs (f, g) with g in the outer loop:
    # the pair index is g * order + f.
    n2 = np.zeros((target_order**2, source_order**2))
    for g in range(source_order):
        for f in range(source_order):
            column = g * source_order + f
            row = int(images[g]) * target_order + int(images[f])
            n2[row, column] = 1.0
    return ChainMap(source_complex, target_complex, (n0, n1, n2))


@dataclass(frozen=True)
class GroupSwitchInstance:
    """One auditable routing instance over group nerves: a source group's
    bar construction, a planted homomorphism-induced chain map to the true
    target, and decoy target groups of the same order."""

    seed: int
    family: str
    source: ChainComplex
    true_target: ChainComplex
    chain_map: ChainMap
    decoy_targets: tuple[ChainComplex, ...]

    @property
    def candidates(self) -> tuple[ChainComplex, ...]:
        return (self.true_target, *self.decoy_targets)

    def commutation_scores(self) -> tuple[float, ...]:
        """Worst-degree commutation residual of the planted map against
        each candidate.

        Index 0 is the true target (exactly 0.0); decoys are positive. This
        is the structure-level misfit the router/discovery layer will read.
        """
        scores = []
        for candidate in self.candidates:
            probe = ChainMap(
                self.chain_map.source, candidate, self.chain_map.maps
            )
            scores.append(max(probe.commutation_residuals()))
        return tuple(scores)


def make_group_switch_instance(
    seed: int,
    family: str = "z6_to_z3",
    num_decoys: int = 2,
) -> GroupSwitchInstance:
    """A planted homomorphism between group nerves, plus decoy targets.

    Families: ``z6_to_z3`` (reduction mod 3), ``z4_to_z2`` (reduction mod
    2), ``s3_sign`` (the sign homomorphism S3 -> Z/2). Decoys are the
    target group under a relabeled multiplication table — a non-identity
    permutation of the element labels transported through the table — so
    every candidate has the same order and the planted map stays
    shape-compatible. A decoy is accepted only when the planted map's
    commutation residual against it exceeds 1e-9, so discrimination is
    guaranteed by construction; relabelings the planted map still commutes
    with (target automorphisms) are redrawn.

    The decoy pool is finite: a target of order ``m`` has
    ``m! / |Aut(target)|`` relabeled tables, minus the true one. Z/3
    admits two decoys, Z/2 exactly one — asking for more is an error,
    never a silent duplicate.
    """
    source_group, target_group, phi = _planted_homomorphism(family)
    chain_map = induced_nerve_map(phi, source_group, target_group)
    source = chain_map.source
    true_target = chain_map.target
    target_table = _table_of(target_group)
    order = len(target_group.morphisms)

    decoys: list[ChainComplex] = []
    seen_tables = [target_table]
    attempt = 0
    while len(decoys) < num_decoys:
        attempt += 1
        if attempt > 1000:
            raise RuntimeError("could not generate discriminating decoys")
        draw = subseed(seed, "decoy", family, str(attempt))
        rng = np.random.default_rng(draw)
        permutation = rng.permutation(order)
        if np.array_equal(permutation, np.arange(order)):
            continue  # the identity relabeling is the true target itself
        decoy_table = _relabel(target_table, permutation)
        if any(np.array_equal(decoy_table, seen) for seen in seen_tables):
            continue
        candidate = nerve_chain_complex(group_as_category(decoy_table))
        probe = ChainMap(source, candidate, chain_map.maps)
        if max(probe.commutation_residuals()) > 1e-9:
            decoys.append(candidate)
            seen_tables.append(decoy_table)
    return GroupSwitchInstance(
        seed, family, source, true_target, chain_map, tuple(decoys)
    )


_PLANTED_FAMILIES = ("z6_to_z3", "z4_to_z2", "s3_sign")


def _planted_homomorphism(
    family: str,
) -> tuple[FiniteCategory, FiniteCategory, tuple[int, ...]]:
    """The planted groups and homomorphism of one family."""
    if family == "z6_to_z3":
        return cyclic_group(6), cyclic_group(3), tuple(x % 3 for x in range(6))
    if family == "z4_to_z2":
        return cyclic_group(4), cyclic_group(2), tuple(x % 2 for x in range(4))
    if family == "s3_sign":
        return (
            symmetric_group_3(),
            cyclic_group(2),
            tuple(_parity(perm) for perm in _S3),
        )
    raise ValueError(
        f"unknown family {family!r}: expected one of {_PLANTED_FAMILIES}"
    )


def _table_of(group: FiniteCategory) -> np.ndarray:
    """The multiplication table of a one-object (group) category."""
    order = len(group.morphisms)
    return np.array(
        [
            [group.compose[(g, f)] for f in range(order)]
            for g in range(order)
        ]
    )


def _relabel(table: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    """The same group under relabeled elements.

    ``result[i][j] = permutation[table[inv(i)][inv(j)]]``: the transported
    table, whose identity element sits wherever the permutation moved it.
    """
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(len(permutation))
    return permutation[table[np.ix_(inverse, inverse)]]
