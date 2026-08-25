"""The Grothendieck nerve: finite categories enter the chain-complex format.

For a finite category, the nerve's 0-simplices are objects, 1-simplices are
morphisms, and 2-simplices are composable pairs ``(f, g)`` with
``target(f) = source(g)``. Taking free real vector spaces on each level and
alternating sums of faces as boundaries gives an ordinary chain complex:

- ``d1(m) = target(m) - source(m)``
- ``d2(f, g) = g - (g . f) + f``  (faces ``d0 - d1 + d2``)

and ``d1 d2 = 0`` telescopes. Posets are categories; their nerve is the
order complex, so simplicial complexes arise as a special case. This module
builds levels 0-2, which is all Universa's first suite needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from universa.structures import ChainComplex


@dataclass(frozen=True)
class FiniteCategory:
    """A finite category by its morphism table.

    ``morphisms[k] = (source, target)`` with object ids ``0..num_objects-1``.
    Identity morphisms must be included explicitly. ``compose[(g, f)]`` gives
    the index of the composite ``g . f`` whenever ``target(f) == source(g)``;
    it must be defined for every composable pair and satisfy associativity —
    checked exhaustively here, since the category is finite.
    """

    num_objects: int
    morphisms: tuple[tuple[int, int], ...]
    compose: dict[tuple[int, int], int]

    def __post_init__(self) -> None:
        if self.num_objects < 1:
            raise ValueError("a category needs at least one object")
        for index, (source, target) in enumerate(self.morphisms):
            if not (0 <= source < self.num_objects):
                raise ValueError(f"morphism {index} has bad source {source}")
            if not (0 <= target < self.num_objects):
                raise ValueError(f"morphism {index} has bad target {target}")
        for g, (gs, _) in enumerate(self.morphisms):
            for f, (_, ft) in enumerate(self.morphisms):
                if ft != gs:
                    continue
                if (g, f) not in self.compose:
                    raise ValueError(f"missing composite ({g}, {f})")
                h = self.compose[(g, f)]
                hs, ht = self.morphisms[h]
                if hs != self.morphisms[f][0] or ht != self.morphisms[g][1]:
                    raise ValueError(
                        f"composite ({g}, {f}) -> {h} has wrong endpoints"
                    )
        for a in range(len(self.morphisms)):
            for b in range(len(self.morphisms)):
                if (b, a) not in self.compose:
                    continue
                ab = self.compose[(b, a)]
                for c in range(len(self.morphisms)):
                    if (c, b) not in self.compose:
                        continue
                    bc = self.compose[(c, b)]
                    left = self.compose.get((c, ab))
                    right = self.compose.get((bc, a))
                    if left is None or right is None or left != right:
                        raise ValueError(
                            f"associativity failed on morphisms ({c}, {b}, {a})"
                        )
        # Identities are derived from the table, not counted: an identity at
        # an object is an endomorphism that composes as the neutral element
        # with every compatible morphism. A category has exactly one per
        # object; other endomorphisms (e.g. group elements) are fine.
        for obj in range(self.num_objects):
            candidates = []
            for e, (es, et) in enumerate(self.morphisms):
                if not (es == et == obj):
                    continue
                neutral = True
                for m, (ms, mt) in enumerate(self.morphisms):
                    if mt == obj and self.compose.get((e, m)) != m:
                        neutral = False
                    if ms == obj and self.compose.get((m, e)) != m:
                        neutral = False
                if neutral:
                    candidates.append(e)
            if len(candidates) != 1:
                raise ValueError(
                    f"object {obj} must have exactly one identity morphism "
                    f"under the composition table, found {len(candidates)}"
                )


def nerve_chain_complex(category: FiniteCategory) -> ChainComplex:
    """Build levels 0-2 of the category's nerve as a chain complex."""
    n0 = category.num_objects
    n1 = len(category.morphisms)
    pairs = [
        (f, g)
        for g, (gs, _) in enumerate(category.morphisms)
        for f, (_, ft) in enumerate(category.morphisms)
        if ft == gs
    ]

    d1 = np.zeros((n0, n1))
    for m, (source, target) in enumerate(category.morphisms):
        d1[source, m] -= 1.0
        d1[target, m] += 1.0

    d2 = np.zeros((n1, len(pairs)))
    for column, (f, g) in enumerate(pairs):
        gf = category.compose[(g, f)]
        d2[g, column] += 1.0  # face d0
        d2[gf, column] -= 1.0  # face d1 (the composite)
        d2[f, column] += 1.0  # face d2

    return ChainComplex((d1, d2))
