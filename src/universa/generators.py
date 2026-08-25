"""First synthetic families for the multi-structure suite.

Everything is deterministic from an integer seed via SHA-256-derived
sub-seeds (HOMYMOLY convention), and every planted chain map is built with
exact integer arithmetic: graph quotients commute with the boundaries
*exactly*, so a nonzero commutation residual is always signal, never
numerical residue.

The first family: a random connected source graph, a planted quotient chain
map onto a coarser graph, and decoy target structures. The routing ground
truth is auditable by construction — the true target has commutation
residual exactly zero, decoys do not.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from universa.structures import ChainComplex, ChainMap

SEED_PREFIX = "universa"


def subseed(seed: int, *components: str) -> int:
    """Deterministic 63-bit sub-seed for one component of one instance."""
    message = ":".join([SEED_PREFIX, str(seed), *components]).encode()
    digest = hashlib.sha256(message).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def incidence_matrix(
    num_vertices: int, edges: list[tuple[int, int]]
) -> np.ndarray:
    """Signed incidence ``B1`` (vertices x edges), tail -1 / head +1.

    Edges are stored with ``tail < head`` so signs are canonical.
    """
    boundary = np.zeros((num_vertices, len(edges)))
    for column, (tail, head) in enumerate(edges):
        if tail == head:
            raise ValueError("self-loops are not supported")
        boundary[tail, column] = -1.0
        boundary[head, column] = 1.0
    return boundary


def random_connected_graph(
    seed: int, num_vertices: int, num_edges: int
) -> ChainComplex:
    """Uniform-ish connected simple graph as a 1-complex.

    A random recursive tree guarantees connectivity; the remaining edges are
    drawn without replacement from the complement.
    """
    if num_vertices < 2:
        raise ValueError("need at least two vertices")
    if not num_vertices - 1 <= num_edges <= num_vertices * (num_vertices - 1) // 2:
        raise ValueError(
            f"num_edges={num_edges} infeasible for {num_vertices} vertices"
        )
    rng = np.random.default_rng(subseed(seed, "graph"))
    edges: set[tuple[int, int]] = set()
    for vertex in range(1, num_vertices):
        parent = int(rng.integers(0, vertex))
        edges.add((min(parent, vertex), max(parent, vertex)))
    while len(edges) < num_edges:
        a, b = int(rng.integers(0, num_vertices)), int(
            rng.integers(0, num_vertices)
        )
        if a != b:
            edges.add((min(a, b), max(a, b)))
    ordered = sorted(edges)
    return ChainComplex((incidence_matrix(num_vertices, ordered),))


def _random_partition(
    seed: int, num_vertices: int, num_classes: int
) -> np.ndarray:
    """A surjective assignment of vertices to classes ``0..num_classes-1``."""
    if not 1 <= num_classes <= num_vertices:
        raise ValueError("bad class count")
    rng = np.random.default_rng(subseed(seed, "partition"))
    partition = np.arange(num_vertices) % num_classes
    rng.shuffle(partition)
    return partition


def quotient_chain_map(
    source: ChainComplex, partition: np.ndarray
) -> tuple[ChainComplex, ChainMap]:
    """The exact chain map induced by a vertex partition (graph quotient).

    ``f0`` maps each vertex to its class; each edge maps to the edge between
    the classes of its endpoints (signed by orientation), or to zero if the
    edge collapses inside a class. Commutation ``B1' f1 = f0 B1`` holds with
    exact integer arithmetic.
    """
    if source.top_degree != 1:
        raise ValueError("quotients are defined for 1-complexes (graphs)")
    boundary = source.boundaries[0]
    num_vertices = boundary.shape[0]
    if partition.shape != (num_vertices,):
        raise ValueError("partition must assign every vertex a class")
    classes = sorted(set(int(c) for c in partition))
    if classes != list(range(len(classes))):
        raise ValueError("partition classes must be 0..k-1 and surjective")
    num_classes = len(classes)

    f0 = np.zeros((num_classes, num_vertices))
    f0[partition, np.arange(num_vertices)] = 1.0

    tails = np.argmin(boundary, axis=0)  # index of the -1 entry per column
    heads = np.argmax(boundary, axis=0)  # index of the +1 entry per column
    keyed: list[tuple[tuple[int, int] | None, float]] = []
    for edge in range(boundary.shape[1]):
        ct, ch = int(partition[tails[edge]]), int(partition[heads[edge]])
        if ct == ch:
            keyed.append((None, 0.0))  # collapsed edge
            continue
        key = (min(ct, ch), max(ct, ch))
        sign = 1.0 if (ct, ch) == key else -1.0
        keyed.append((key, sign))

    ordered_keys = sorted({key for key, _ in keyed if key is not None})
    row_of = {key: row for row, key in enumerate(ordered_keys)}
    f1 = np.zeros((len(ordered_keys), boundary.shape[1]))
    for edge, (key, sign) in enumerate(keyed):
        if key is not None:
            f1[row_of[key], edge] = sign

    target_boundary = incidence_matrix(num_classes, ordered_keys)
    target = ChainComplex((target_boundary,))
    return target, ChainMap(source, target, (f0, f1))


@dataclass(frozen=True)
class SwitchInstance:
    """One auditable routing instance: a source structure, a planted chain
    map to the true target, decoy targets, and per-candidate residuals."""

    seed: int
    source: ChainComplex
    true_target: ChainComplex
    chain_map: ChainMap
    decoy_targets: tuple[ChainComplex, ...]

    @property
    def candidates(self) -> tuple[ChainComplex, ...]:
        return (self.true_target, *self.decoy_targets)

    def commutation_scores(self) -> tuple[float, ...]:
        """Commutation residual of the planted map against each candidate.

        Index 0 is the true target (exactly 0.0); decoys are positive. For
        multi-degree instances the score is the max residual over degrees.
        This is the structure-level misfit the router/discovery layer reads.
        """
        scores = []
        for candidate in self.candidates:
            probe = ChainMap(
                self.chain_map.source, candidate, self.chain_map.maps
            )
            residuals = probe.commutation_residuals()
            scores.append(max(residuals) if residuals else 0.0)
        return tuple(scores)


def make_switch_instance(
    seed: int,
    num_vertices: int = 8,
    num_edges: int = 14,
    num_classes: int = 4,
    num_decoys: int = 3,
) -> SwitchInstance:
    """A source graph, its planted quotient chain map, and decoy targets.

    Decoys are candidate target structures on the same vertex and edge
    counts as the true quotient, so the planted map is shape-compatible with
    every candidate. A decoy is only accepted if the planted map fails to
    commute with it, so discrimination is guaranteed by construction. When
    the quotient is the complete graph (no other simple graph shares its
    counts), decoys are vertex relabelings of the true target — a different
    labeled structure, which the map still fails.
    """
    source = random_connected_graph(seed, num_vertices, num_edges)
    partition = _random_partition(seed, num_vertices, num_classes)
    true_target, chain_map = quotient_chain_map(source, partition)
    num_quotient_edges = true_target.boundaries[0].shape[1]
    complete = num_quotient_edges == num_classes * (num_classes - 1) // 2

    decoys: list[ChainComplex] = []
    attempt = 0
    while len(decoys) < num_decoys:
        attempt += 1
        if attempt > 1000:
            raise RuntimeError("could not generate discriminating decoys")
        draw = subseed(seed, "decoy", str(attempt))
        if complete:
            candidate = _vertex_relabeling(true_target, draw)
        else:
            candidate = random_connected_graph(
                draw, num_classes, num_quotient_edges
            )
        probe = ChainMap(source, candidate, chain_map.maps)
        (residual,) = probe.commutation_residuals()
        if residual > 1e-9:
            decoys.append(candidate)
    return SwitchInstance(
        seed, source, true_target, chain_map, tuple(decoys)
    )


def _vertex_relabeling(complex_: ChainComplex, seed: int) -> ChainComplex:
    """The same 1-complex under a random vertex relabeling (row permutation)."""
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(complex_.boundaries[0].shape[0])
    return ChainComplex((complex_.boundaries[0][permutation, :],))
