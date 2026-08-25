"""Third synthetic family: multi-hop switch tasks (chains of structures).

A single switch task asks the router to find the one target structure a
planted chain map commutes with. A *multi-hop* switch task transports a
quantity through a chain of structures ``C_0 -> C_1 -> ... -> C_hops``
along one planted chain map per hop, and the router must choose the right
target at *every* hop. The chain composes the existing families: each hop
is a graph quotient (:func:`universa.generators.quotient_chain_map`) whose
target is the next hop's source — a quotient onto a quotient — and the
whole chain can optionally be lifted to 2-complexes
(:func:`universa.complexes2.two_complex` +
:func:`universa.complexes2.induced_quotient_map`), so every degree of
every hop commutes with exact integer arithmetic.

Composition of chain maps is a chain map by construction: the commutation
defect of a composite factors through the defects of its factors,

    d (g f) - (g f) d = (d g - g d) f + g (d f - f d),

which is an exact zero matrix when both factors commute exactly, and with
integer-valued maps the degree products are exact in float64.
:func:`compose` still *checks* the composite's residuals are exactly 0.0
rather than assuming — fail-closed, like the rest of the scaffold.

The router-facing signal is :func:`localize_misfit`: per-hop commutation
residuals of the planted maps against the chosen candidate structures.
Hop ``k`` is always probed with the true map and the true source of that
hop, so a wrong choice at hop ``k`` raises exactly residual ``k`` while
every other hop — including all later hops, given the true earlier maps —
keeps an exactly zero residual: the misfit localizes to the wrong hop
(proved in the tests). Decoys follow the same contract as the single-hop
families: accepted only above a 1e-9 residual, and instances are
deterministic from integer seeds via SHA-256-derived sub-seeds.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from universa.complexes2 import induced_quotient_map, two_complex
from universa.generators import (
    _random_partition,
    _vertex_relabeling,
    quotient_chain_map,
    random_connected_graph,
    subseed,
)
from universa.structures import ChainComplex, ChainMap

DECOY_TOL = 1e-9
"""Minimum commutation residual for an accepted decoy candidate."""


def _same_complex(first: ChainComplex, second: ChainComplex) -> bool:
    """Boundary-by-boundary equality of two complexes."""
    return len(first.boundaries) == len(second.boundaries) and all(
        np.array_equal(b_first, b_second)
        for b_first, b_second in zip(first.boundaries, second.boundaries)
    )


def compose(first: ChainMap, second: ChainMap) -> ChainMap:
    """The composite ``second ∘ first`` of two exactly-commuting chain maps.

    Degree maps multiply, ``(g ∘ f)_k = g_k f_k``, and the target of
    ``first`` must be the source of ``second``. The composite is a chain
    map by construction — its defect factors through the defects of the
    two factors, both exactly zero — and that is *checked*, not assumed:
    both factors must commute exactly (a factor's defect can otherwise be
    annihilated by the other factor and slip through), and the composite
    is returned only if every commutation residual is exactly 0.0, never
    a tolerance.
    """
    if not _same_complex(first.target, second.source):
        raise ValueError("the target of first must be the source of second")
    if not first.is_chain_map(tol=0.0):
        raise ValueError("first does not commute exactly")
    if not second.is_chain_map(tol=0.0):
        raise ValueError("second does not commute exactly")
    composite = ChainMap(
        first.source,
        second.target,
        tuple(g @ f for f, g in zip(first.maps, second.maps)),
    )
    if not composite.is_chain_map(tol=0.0):
        raise ValueError(
            "composition failed to commute exactly: residuals "
            f"{composite.commutation_residuals()}"
        )
    return composite


@dataclass(frozen=True)
class HopChain:
    """One auditable multi-hop routing instance.

    ``complexes`` is the ordered chain ``C_0 -> ... -> C_hops`` of true
    structures, ``maps[k]`` the planted exact chain map
    ``C_k -> C_{k+1}``, and ``decoys[k]`` the decoy candidate targets for
    hop ``k`` — same dims as the true target, residual above
    :data:`DECOY_TOL` by construction. Validation is fail-closed: the
    links must match, every planted map must commute exactly, and every
    decoy must discriminate.
    """

    seed: int
    complexes: tuple[ChainComplex, ...]
    maps: tuple[ChainMap, ...]
    decoys: tuple[tuple[ChainComplex, ...], ...]

    def __post_init__(self) -> None:
        if len(self.complexes) < 2:
            raise ValueError("a hop chain needs at least two complexes")
        if len(self.maps) != len(self.complexes) - 1:
            raise ValueError("need exactly one chain map per hop")
        if len(self.decoys) != len(self.maps):
            raise ValueError("need one decoy set per hop")
        for hop, chain_map in enumerate(self.maps):
            if not _same_complex(
                chain_map.source, self.complexes[hop]
            ) or not _same_complex(chain_map.target, self.complexes[hop + 1]):
                raise ValueError(f"hop {hop} does not link the chain")
            if not chain_map.is_chain_map(tol=0.0):
                raise ValueError(f"hop {hop} map does not commute exactly")
            for decoy in self.decoys[hop]:
                if decoy.dims != self.complexes[hop + 1].dims:
                    raise ValueError(
                        f"hop {hop} decoy dims differ from the true target"
                    )
                probe = ChainMap(chain_map.source, decoy, chain_map.maps)
                if not any(r > DECOY_TOL for r in probe.commutation_residuals()):
                    raise ValueError(
                        f"hop {hop} decoy fails the {DECOY_TOL} residual "
                        "contract"
                    )

    @property
    def hops(self) -> int:
        return len(self.maps)

    def candidates(self, hop: int) -> tuple[ChainComplex, ...]:
        """Candidate targets for one hop; index 0 is the true target."""
        if not 0 <= hop < self.hops:
            raise ValueError(f"no hop {hop} in a {self.hops}-hop chain")
        return (self.complexes[hop + 1], *self.decoys[hop])

    @property
    def composed(self) -> ChainMap:
        """The end-to-end map ``C_0 -> C_hops``.

        Per-hop degree-map products, built by repeated :func:`compose`, so
        exactness is re-checked at every stage and the result commutes
        with residuals exactly 0.0.
        """
        composite = self.maps[0]
        for chain_map in self.maps[1:]:
            composite = compose(composite, chain_map)
        return composite


def localize_misfit(
    chain: HopChain, candidate_choices: Sequence[int]
) -> tuple[float, ...]:
    """Per-hop residuals for one candidate choice per hop — the router signal.

    ``candidate_choices[k]`` indexes :meth:`HopChain.candidates` for hop
    ``k`` (0 is the true target). Hop ``k`` is probed with the true map
    and the true source of that hop, so its score is the worst-degree
    commutation residual of the planted map against the chosen target:
    exactly 0.0 for the true target, above :data:`DECOY_TOL` for any
    decoy. A wrong choice at hop ``k`` therefore raises exactly residual
    ``k``; given the true maps at the other hops, every later hop's
    residual stays exactly 0.0.
    """
    if len(candidate_choices) != chain.hops:
        raise ValueError(
            f"need one candidate choice per hop ({chain.hops}), "
            f"got {len(candidate_choices)}"
        )
    residuals = []
    for hop, choice in enumerate(candidate_choices):
        candidates = chain.candidates(hop)
        if not 0 <= choice < len(candidates):
            raise ValueError(
                f"hop {hop} has {len(candidates)} candidates, "
                f"got choice {choice}"
            )
        probe = ChainMap(
            chain.maps[hop].source, candidates[choice], chain.maps[hop].maps
        )
        residuals.append(max(probe.commutation_residuals()))
    return tuple(residuals)


def make_multihop_instance(
    seed: int,
    hops: int = 2,
    num_vertices: int = 8,
    num_edges: int = 14,
    class_counts: tuple[int, ...] | None = None,
    num_decoys: int = 3,
    family: Literal["graph", "two_complex"] = "graph",
) -> HopChain:
    """A deterministic multi-hop chain with per-hop decoy candidates.

    The source graph is drawn from ``seed``; hop ``k`` quotients its
    source onto ``class_counts[k]`` classes (default: halve the vertex
    count each hop, never below 2), and the quotient is the next hop's
    source. With ``family="two_complex"`` every graph in the chain is
    lifted to its fundamental-cycle 2-complex and every hop map is lifted
    to degree 2 by the exact integer solve. Per-hop decoys share the true
    target's counts and are accepted only when the planted map's
    commutation residual against them exceeds :data:`DECOY_TOL` at some
    degree — mirroring :func:`universa.generators.make_switch_instance`.
    """
    if hops < 1:
        raise ValueError("need at least one hop")
    if family not in ("graph", "two_complex"):
        raise ValueError(f"unknown family {family!r}")
    if class_counts is None:
        counts = []
        vertices = num_vertices
        for _ in range(hops):
            vertices = max(2, vertices // 2)
            counts.append(vertices)
        class_counts = tuple(counts)
    if len(class_counts) != hops:
        raise ValueError(f"need {hops} class counts, got {len(class_counts)}")

    # The graph-level chain: each quotient is the next hop's source.
    graphs = [
        random_connected_graph(
            subseed(seed, "multihop", "source"), num_vertices, num_edges
        )
    ]
    quotient_maps = []
    for hop, classes in enumerate(class_counts):
        source_vertices = graphs[-1].dims[0]
        if not 2 <= classes <= source_vertices:
            raise ValueError(
                f"hop {hop}: {classes} classes infeasible for "
                f"{source_vertices} vertices"
            )
        partition = _random_partition(
            subseed(seed, "multihop", "hop", str(hop)),
            source_vertices,
            classes,
        )
        target, quotient_map = quotient_chain_map(graphs[-1], partition)
        graphs.append(target)
        quotient_maps.append(quotient_map)

    if family == "two_complex":
        complexes = tuple(two_complex(graph) for graph in graphs)
        maps = tuple(
            induced_quotient_map(
                complexes[hop], quotient_maps[hop], complexes[hop + 1]
            )
            for hop in range(hops)
        )
    else:
        complexes = tuple(graphs)
        maps = tuple(quotient_maps)

    decoys: list[tuple[ChainComplex, ...]] = []
    for hop in range(hops):
        quotient_graph = graphs[hop + 1]
        target_vertices, target_edges = quotient_graph.dims
        complete = (
            target_edges == target_vertices * (target_vertices - 1) // 2
        )
        hop_decoys: list[ChainComplex] = []
        attempt = 0
        while len(hop_decoys) < num_decoys:
            attempt += 1
            if attempt > 1000:
                raise RuntimeError("could not generate discriminating decoys")
            draw = subseed(
                seed, "multihop", "hop", str(hop), "decoy", str(attempt)
            )
            if complete:
                candidate_graph = _vertex_relabeling(quotient_graph, draw)
            else:
                candidate_graph = random_connected_graph(
                    draw, target_vertices, target_edges
                )
            candidate = (
                two_complex(candidate_graph)
                if family == "two_complex"
                else candidate_graph
            )
            probe = ChainMap(complexes[hop], candidate, maps[hop].maps)
            if any(r > DECOY_TOL for r in probe.commutation_residuals()):
                hop_decoys.append(candidate)
        decoys.append(tuple(hop_decoys))
    return HopChain(seed, complexes, maps, tuple(decoys))
