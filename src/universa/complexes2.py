"""Second synthetic family: 2-complexes with planted quotient chain maps.

From a connected graph (a 1-complex) we attach one 2-cell to every
fundamental cycle of a deterministic spanning tree — BFS from vertex 0 with
lexicographic edge order. The resulting ``B2`` is integer, ``B1 B2 = 0``
exactly, and its columns form a Z-basis of ``ker(B1)``: each fundamental
cycle is the only one touching its defining non-tree edge, so the non-tree
rows of ``B2`` form an identity matrix and every integer cycle is an exact
integer combination of the columns (proved in the tests).

That basis property is what makes the planted maps exact. A 1-complex
quotient map ``(f0, f1)`` lifts to degree 2 by solving ``B2' f2 = f1 B2``:
the columns of ``f1 B2`` are cycles in the target (``B1' f1 B2 = f0 B1 B2
= 0``), so the solution exists, is unique, and is integral. It is computed
by Gaussian elimination over ``fractions.Fraction`` — no least squares, no
tolerances on the answer — and the lift is rejected unless the residual is
exactly zero. Planted 2-complex maps therefore commute with residuals that
are exact zeros at both degrees, and decoys are accepted only when some
degree discriminates, keeping the routing ground truth auditable.
"""

from __future__ import annotations

from collections import deque
from fractions import Fraction

import numpy as np

from universa.generators import (
    SwitchInstance,
    _random_partition,
    _vertex_relabeling,
    quotient_chain_map,
    random_connected_graph,
    subseed,
)
from universa.structures import ChainComplex, ChainMap


def two_complex(graph: ChainComplex) -> ChainComplex:
    """Attach a face to every fundamental cycle of a deterministic tree.

    The spanning tree is BFS from vertex 0 with incident edges considered in
    lexicographic edge order, so the complex is a pure function of the
    graph. Faces are ordered by the column index of their defining non-tree
    edge, and each face column is the fundamental cycle: ``+1`` on the
    non-tree edge itself, ``+-1`` along the unique tree path closing it, so
    ``B1 B2`` telescopes to exact zeros. Fail-closed: ``graph`` must be a
    connected graph's signed incidence (one ``-1`` and one ``+1`` per
    column).
    """
    if graph.top_degree != 1:
        raise ValueError("2-complexes grow on 1-complexes (graphs)")
    b1 = graph.boundaries[0]
    num_vertices, num_edges = b1.shape
    if num_vertices < 1:
        raise ValueError("a graph needs at least one vertex")
    if not np.all(np.isin(b1, (-1.0, 0.0, 1.0))):
        raise ValueError("B1 must be a signed incidence matrix")
    if not (
        np.all((b1 == 1.0).sum(axis=0) == 1)
        and np.all((b1 == -1.0).sum(axis=0) == 1)
    ):
        raise ValueError("every edge column must hold one -1 and one +1")

    tails = np.argmin(b1, axis=0)  # index of the -1 entry per column
    heads = np.argmax(b1, axis=0)  # index of the +1 entry per column
    adjacency: list[list[tuple[tuple[int, int], int, int]]] = [
        [] for _ in range(num_vertices)
    ]
    for edge in range(num_edges):
        tail, head = int(tails[edge]), int(heads[edge])
        key = (min(tail, head), max(tail, head))  # lexicographic order key
        adjacency[tail].append((key, edge, head))
        adjacency[head].append((key, edge, tail))
    for entries in adjacency:
        entries.sort()

    visited = [False] * num_vertices
    parent_edge = [-1] * num_vertices
    depth = [0] * num_vertices
    visited[0] = True
    queue: deque[int] = deque([0])
    while queue:
        vertex = queue.popleft()
        for _, edge, neighbor in adjacency[vertex]:
            if not visited[neighbor]:
                visited[neighbor] = True
                parent_edge[neighbor] = edge
                depth[neighbor] = depth[vertex] + 1
                queue.append(neighbor)
    if not all(visited):
        raise ValueError("graph must be connected")

    tree_edges = {edge for edge in parent_edge if edge >= 0}
    non_tree = [edge for edge in range(num_edges) if edge not in tree_edges]

    b2 = np.zeros((num_edges, len(non_tree)))
    for column, edge in enumerate(non_tree):
        tail, head = int(tails[edge]), int(heads[edge])
        b2[edge, column] = 1.0
        for path_edge, sign in _tree_path(parent_edge, depth, tails, heads, head, tail):
            b2[path_edge, column] += float(sign)
    return _make_complex(b1, b2)


def _tree_path(
    parent_edge: list[int],
    depth: list[int],
    tails: np.ndarray,
    heads: np.ndarray,
    start: int,
    goal: int,
) -> list[tuple[int, int]]:
    """Signed tree edges of the unique path ``start -> goal``.

    Each entry is ``(edge, +1|-1)`` with ``+1`` when the path traverses the
    edge from its tail to its head, so the coefficients' boundary telescopes
    to ``e_goal - e_start``.
    """
    up: list[tuple[int, int]] = []  # start's side, traversed upward
    down: list[tuple[int, int]] = []  # goal's side, traversed downward
    a, b = start, goal
    while a != b:
        if depth[a] > depth[b]:
            edge = parent_edge[a]
            other = int(tails[edge]) + int(heads[edge]) - a
            up.append((edge, 1 if a == int(tails[edge]) else -1))
            a = other
        elif depth[b] > depth[a]:
            edge = parent_edge[b]
            other = int(tails[edge]) + int(heads[edge]) - b
            down.append((edge, 1 if other == int(tails[edge]) else -1))
            b = other
        else:  # equal depths, below the meeting point: lift both
            edge_a = parent_edge[a]
            other_a = int(tails[edge_a]) + int(heads[edge_a]) - a
            up.append((edge_a, 1 if a == int(tails[edge_a]) else -1))
            a = other_a
            edge_b = parent_edge[b]
            other_b = int(tails[edge_b]) + int(heads[edge_b]) - b
            down.append((edge_b, 1 if other_b == int(tails[edge_b]) else -1))
            b = other_b
    return up + down[::-1]


def _make_complex(b1: np.ndarray, b2: np.ndarray) -> ChainComplex:
    """``ChainComplex((b1, b2))``, including a zero-width ``b2`` (a tree's
    empty face set, where the ``d^2`` product is vacuous)."""
    return ChainComplex((b1, b2))


def _exact_integer_solve(coeff: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """The unique exact integer solution of ``coeff @ X = rhs``.

    ``coeff`` must be integer-valued with full column rank and ``rhs``
    integer-valued in its rational column span — the regime of
    fundamental-cycle bases of ``ker(B1)``. Solved by Gauss–Jordan
    elimination over ``fractions.Fraction``; inconsistency, rank
    deficiency, or a non-integral solution are errors, never results.
    """
    num_rows, num_cols = coeff.shape
    num_rhs = rhs.shape[1]
    if not (
        np.all(coeff == np.round(coeff)) and np.all(rhs == np.round(rhs))
    ):
        raise ValueError("exact solve requires integer-valued matrices")
    if num_cols == 0:
        if np.any(rhs != 0.0):
            raise ValueError("inconsistent: no unknowns but a nonzero column")
        return np.zeros((0, num_rhs))
    table = [
        [Fraction(int(v)) for v in coeff[row]]
        + [Fraction(int(v)) for v in rhs[row]]
        for row in range(num_rows)
    ]
    pivot_rows = 0
    for col in range(num_cols):
        pivot = next(
            (r for r in range(pivot_rows, num_rows) if table[r][col] != 0),
            None,
        )
        if pivot is None:
            raise ValueError("coeff does not have full column rank")
        table[pivot_rows], table[pivot] = table[pivot], table[pivot_rows]
        pivot_value = table[pivot_rows][col]
        table[pivot_rows] = [v / pivot_value for v in table[pivot_rows]]
        for r in range(num_rows):
            if r != pivot_rows and table[r][col] != 0:
                factor = table[r][col]
                table[r] = [
                    v - factor * w
                    for v, w in zip(table[r], table[pivot_rows])
                ]
        pivot_rows += 1
    for r in range(pivot_rows, num_rows):
        if any(table[r][num_cols + j] != 0 for j in range(num_rhs)):
            raise ValueError("inconsistent: a column lies outside the span")
    solution = np.zeros((num_cols, num_rhs))
    for i in range(num_cols):
        for j in range(num_rhs):
            value = table[i][num_cols + j]
            if value.denominator != 1:
                raise ValueError("solution is not integral")
            solution[i, j] = float(value)
    return solution


def induced_quotient_map(
    source: ChainComplex,
    quotient_map: ChainMap,
    target: ChainComplex,
) -> ChainMap:
    """Lift a 1-complex quotient map to the planted 2-complexes, exactly.

    ``quotient_map`` must be the 1-complex quotient chain map between the
    two 1-skeletons (as built by
    :func:`universa.generators.quotient_chain_map`). The degree-2 component
    is the unique solution of ``B2' f2 = f1 B2``: every column of
    ``f1 B2`` is a cycle in the target, and the target's fundamental-cycle
    ``B2'`` spans ``ker(B1')`` over Z, so an exact integer solution exists
    and is computed over the rationals. The lift is returned only if the
    degree-2 residual is exactly 0.0.
    """
    if source.top_degree != 2 or target.top_degree != 2:
        raise ValueError("source and target must be 2-complexes")
    if (
        quotient_map.source.top_degree != 1
        or quotient_map.target.top_degree != 1
    ):
        raise ValueError("quotient_map must map between 1-complexes")
    if not np.array_equal(
        quotient_map.source.boundaries[0], source.boundaries[0]
    ):
        raise ValueError("quotient_map source must be the 1-skeleton of source")
    if not np.array_equal(
        quotient_map.target.boundaries[0], target.boundaries[0]
    ):
        raise ValueError("quotient_map target must be the 1-skeleton of target")
    if not quotient_map.is_chain_map(tol=0.0):
        raise ValueError("quotient_map must commute exactly")

    f0, f1 = quotient_map.maps
    b2, b2_prime = source.boundaries[1], target.boundaries[1]
    pushed_cycles = f1 @ b2
    # B1' f1 B2 = f0 B1 B2 = 0: the pushed columns are cycles, checked
    # exactly rather than assumed.
    if np.any(target.boundaries[0] @ pushed_cycles != 0.0):
        raise ValueError("pushed cycles do not lie in ker(B1')")
    f2 = _exact_integer_solve(b2_prime, pushed_cycles)
    if np.any(b2_prime @ f2 - pushed_cycles != 0.0):
        raise ValueError("degree-2 lift failed to solve exactly")
    return ChainMap(source, target, (f0, f1, f2))


def commutation_scores(instance: SwitchInstance) -> tuple[float, ...]:
    """Worst-degree commutation residual of the planted map per candidate.

    Index 0 is the true target (exactly 0.0); decoys are positive at some
    degree by construction. This is the multi-degree analogue of
    ``SwitchInstance.commutation_scores``, which unpacks a single degree
    and so only reads 1-complex instances.
    """
    scores = []
    for candidate in instance.candidates:
        probe = ChainMap(
            instance.chain_map.source, candidate, instance.chain_map.maps
        )
        scores.append(max(probe.commutation_residuals()))
    return tuple(scores)


def make_two_complex_switch_instance(
    seed: int,
    num_vertices: int = 8,
    num_edges: int = 14,
    num_classes: int = 4,
    num_decoys: int = 3,
) -> SwitchInstance:
    """A source 2-complex, its planted quotient chain map, and decoys.

    The source graph is lifted to its fundamental-cycle 2-complex, the
    quotient map is lifted to degree 2 exactly, and decoy 2-complexes are
    built the same way from decoy 1-complexes on the same vertex and edge
    counts, so the planted map is shape-compatible with every candidate at
    every degree. A decoy is accepted only if the planted map's commutation
    residual against it exceeds ``1e-9`` at some degree — discrimination is
    guaranteed by construction, mirroring
    :func:`universa.generators.make_switch_instance`.
    """
    source_graph = random_connected_graph(seed, num_vertices, num_edges)
    partition = _random_partition(seed, num_vertices, num_classes)
    quotient_graph, quotient_map = quotient_chain_map(source_graph, partition)
    source = two_complex(source_graph)
    true_target = two_complex(quotient_graph)
    chain_map = induced_quotient_map(source, quotient_map, true_target)
    num_quotient_edges = quotient_graph.boundaries[0].shape[1]
    complete = num_quotient_edges == num_classes * (num_classes - 1) // 2

    decoys: list[ChainComplex] = []
    attempt = 0
    while len(decoys) < num_decoys:
        attempt += 1
        if attempt > 1000:
            raise RuntimeError("could not generate discriminating decoys")
        draw = subseed(seed, "decoy", str(attempt))
        if complete:
            candidate_graph = _vertex_relabeling(quotient_graph, draw)
        else:
            candidate_graph = random_connected_graph(
                draw, num_classes, num_quotient_edges
            )
        candidate = two_complex(candidate_graph)
        probe = ChainMap(source, candidate, chain_map.maps)
        if any(r > 1e-9 for r in probe.commutation_residuals()):
            decoys.append(candidate)
    return SwitchInstance(seed, source, true_target, chain_map, tuple(decoys))
