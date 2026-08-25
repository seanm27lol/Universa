from collections import deque
from fractions import Fraction

import numpy as np
import pytest

from universa.complexes2 import (
    commutation_scores,
    induced_quotient_map,
    make_two_complex_switch_instance,
    two_complex,
)
from universa.generators import quotient_chain_map, random_connected_graph
from universa.structures import ChainComplex, ChainMap


def exact_rank(matrix: np.ndarray) -> int:
    """Rank over the rationals, by exact Fraction Gauss–Jordan elimination."""
    table = [
        [Fraction(int(v)) for v in row]
        for row in np.asarray(matrix, dtype=int)
    ]
    num_rows = len(table)
    num_cols = len(table[0]) if table else 0
    rank = 0
    for col in range(num_cols):
        pivot = next(
            (r for r in range(rank, num_rows) if table[r][col] != 0), None
        )
        if pivot is None:
            continue
        table[rank], table[pivot] = table[pivot], table[rank]
        pivot_value = table[rank][col]
        table[rank] = [v / pivot_value for v in table[rank]]
        for r in range(num_rows):
            if r != rank and table[r][col] != 0:
                factor = table[r][col]
                table[r] = [
                    v - factor * w for v, w in zip(table[r], table[rank])
                ]
        rank += 1
    return rank


def independent_non_tree_edges(b1: np.ndarray) -> list[int]:
    """Non-tree edge indices under the documented deterministic rule.

    Independent re-derivation of the spanning tree (BFS from vertex 0,
    incident edges in lexicographic edge order) used to check that the
    module's ``b2`` really is the fundamental-cycle basis of that tree.
    """
    num_vertices, num_edges = b1.shape
    tails = np.argmin(b1, axis=0)
    heads = np.argmax(b1, axis=0)
    adjacency: list[list[tuple[tuple[int, int], int, int]]] = [
        [] for _ in range(num_vertices)
    ]
    for edge in range(num_edges):
        tail, head = int(tails[edge]), int(heads[edge])
        key = (min(tail, head), max(tail, head))
        adjacency[tail].append((key, edge, head))
        adjacency[head].append((key, edge, tail))
    for entries in adjacency:
        entries.sort()
    visited = [True] + [False] * (num_vertices - 1)
    tree_edges = set()
    queue: deque[int] = deque([0])
    while queue:
        vertex = queue.popleft()
        for _, edge, neighbor in adjacency[vertex]:
            if not visited[neighbor]:
                visited[neighbor] = True
                tree_edges.add(edge)
                queue.append(neighbor)
    return [edge for edge in range(num_edges) if edge not in tree_edges]


def cycle_avoiding_edge(b1: np.ndarray, skipped: int) -> np.ndarray | None:
    """An integer cycle through edge ``skipped``, built without ``b2``.

    BFS from the tail to the head of ``skipped`` over every other edge; the
    signed path vector minus the skipped edge's own column is an integer
    cycle. Returns ``None`` when the edge is a bridge.
    """
    num_vertices, _ = b1.shape
    tails = np.argmin(b1, axis=0)
    heads = np.argmax(b1, axis=0)
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(num_vertices)]
    for edge in range(b1.shape[1]):
        if edge == skipped:
            continue
        tail, head = int(tails[edge]), int(heads[edge])
        adjacency[tail].append((head, edge))
        adjacency[head].append((tail, edge))
    start, goal = int(tails[skipped]), int(heads[skipped])
    previous: dict[int, tuple[int, int] | None] = {start: None}
    queue: deque[int] = deque([start])
    while queue:
        vertex = queue.popleft()
        if vertex == goal:
            break
        for neighbor, edge in adjacency[vertex]:
            if neighbor not in previous:
                previous[neighbor] = (vertex, edge)
                queue.append(neighbor)
    if goal not in previous:
        return None
    cycle = np.zeros(b1.shape[1])
    vertex = goal
    while vertex != start:
        parent, edge = previous[vertex]
        cycle[edge] += 1.0 if parent == int(tails[edge]) else -1.0
        vertex = parent
    cycle[skipped] -= 1.0
    return cycle


def test_two_complex_dimensions_and_exact_d_squared():
    for seed, num_vertices, num_edges in [(0, 8, 14), (7, 6, 12), (2026, 10, 20)]:
        complex_ = two_complex(random_connected_graph(seed, num_vertices, num_edges))
        b1, b2 = complex_.boundaries
        assert complex_.dims == (num_vertices, num_edges, num_edges - num_vertices + 1)
        assert complex_.top_degree == 2
        product = b1 @ b2
        assert product.shape == (num_vertices, num_edges - num_vertices + 1)
        assert not np.any(product != 0.0)  # exact zero, not below a tolerance
        assert set(np.unique(b2)) <= {-1.0, 0.0, 1.0}


def test_two_complex_is_deterministic():
    first = two_complex(random_connected_graph(42, 8, 14))
    second = two_complex(random_connected_graph(42, 8, 14))
    assert np.array_equal(first.boundaries[0], second.boundaries[0])
    assert np.array_equal(first.boundaries[1], second.boundaries[1])


def test_fundamental_cycles_are_a_z_basis_of_the_cycle_space():
    complex_ = two_complex(random_connected_graph(5, 8, 14))
    b1, b2 = complex_.boundaries
    num_vertices, num_edges = b1.shape
    num_faces = num_edges - num_vertices + 1
    # Over Q: rank(B1) = V - 1 (connected) and rank(B2) = F, and
    # B1 B2 = 0, so the columns of B2 are a rational basis of ker(B1).
    assert exact_rank(b1) == num_vertices - 1
    assert exact_rank(b2) == num_faces
    # The non-tree rows (identified by an independent BFS rebuild of the
    # documented spanning tree) are an identity matrix: the coefficients of
    # any cycle in this basis are read off, integrally, from its own
    # non-tree entries — this is what makes the basis a Z-basis, not just
    # a Q-basis.
    rows = np.array(independent_non_tree_edges(b1), dtype=int)
    assert rows.shape == (num_faces,)
    assert np.array_equal(b2[rows], np.eye(num_faces))
    # Every integer cycle we can build independently of B2 is an exact
    # integer combination of its columns.
    cycles = []
    for edge in range(num_edges):
        cycle = cycle_avoiding_edge(b1, edge)
        if cycle is not None:
            cycles.append(cycle)
    assert cycles  # a graph with F > 0 has non-bridge edges
    coefficients = np.arange(1, num_faces + 1, dtype=float)
    cycles.append(b2 @ coefficients)
    cycles.append(b2 @ ((-1.0) ** np.arange(num_faces)))
    for cycle in cycles:
        assert not np.any(b1 @ cycle != 0.0)  # it really is a cycle
        read_off = cycle[rows]
        assert np.array_equal(read_off, np.round(read_off))  # integral coords
        assert np.array_equal(b2 @ read_off, cycle)  # exact reconstruction


def test_two_complex_rejects_non_graphs():
    graph = random_connected_graph(0, 6, 10)
    with pytest.raises(ValueError, match="1-complexes"):
        two_complex(two_complex(graph))
    disconnected = np.zeros((4, 2))
    disconnected[0, 0], disconnected[1, 0] = -1.0, 1.0
    disconnected[2, 1], disconnected[3, 1] = -1.0, 1.0
    with pytest.raises(ValueError, match="connected"):
        two_complex(ChainComplex((disconnected,)))
    not_incidence = np.array([[-1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="one -1 and one"):
        two_complex(ChainComplex((not_incidence,)))


def test_tree_source_has_an_empty_face_set_and_still_works():
    tree = random_connected_graph(3, 8, 7)  # V - 1 edges: a tree, F = 0
    source = two_complex(tree)
    assert source.dims == (8, 7, 0)
    assert source.boundaries[1].shape == (7, 0)
    partition = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    quotient_graph, quotient_map = quotient_chain_map(tree, partition)
    target = two_complex(quotient_graph)
    lifted = induced_quotient_map(source, quotient_map, target)
    f2 = lifted.maps[2]
    assert f2.shape == (target.dims[2], 0)  # the empty map
    assert lifted.is_chain_map(tol=0.0)
    assert lifted.commutation_residuals(tol=0.0) == (0.0, 0.0)


def test_single_vertex_graph_lifts_to_an_empty_complex():
    complex_ = two_complex(ChainComplex((np.zeros((1, 0)),)))
    assert complex_.dims == (1, 0, 0)


def test_induced_quotient_map_commutes_exactly_at_both_degrees():
    for seed in (7, 2026):
        graph = random_connected_graph(seed, 8, 14)
        partition = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        quotient_graph, quotient_map = quotient_chain_map(graph, partition)
        source = two_complex(graph)
        target = two_complex(quotient_graph)
        lifted = induced_quotient_map(source, quotient_map, target)
        assert lifted.is_chain_map(tol=0.0)
        assert lifted.commutation_residuals(tol=0.0) == (0.0, 0.0)
        f2 = lifted.maps[2]
        assert f2.shape == (target.dims[2], source.dims[2])
        assert source.dims[2] > 0 and target.dims[2] > 0  # non-vacuous
        assert np.array_equal(f2, np.round(f2))  # exact integer values
        residual = (
            target.boundaries[1] @ f2 - quotient_map.maps[1] @ source.boundaries[1]
        )
        assert not np.any(residual != 0.0)  # exactly zero, not tolerated


def test_induced_f2_is_read_off_the_target_non_tree_rows():
    # Blocky partition: edges inside a class collapse, and the lift stays
    # exact. Uniqueness of the solution of B2' f2 = f1 B2 plus the identity
    # block in B2' pins f2 to the non-tree rows of the pushed cycles.
    graph = random_connected_graph(11, 8, 14)
    partition = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    quotient_graph, quotient_map = quotient_chain_map(graph, partition)
    source = two_complex(graph)
    target = two_complex(quotient_graph)
    lifted = induced_quotient_map(source, quotient_map, target)
    assert target.dims[2] > 0
    pushed_cycles = quotient_map.maps[1] @ source.boundaries[1]
    rows = np.array(
        independent_non_tree_edges(target.boundaries[0]), dtype=int
    )
    assert np.array_equal(lifted.maps[2], pushed_cycles[rows])


def test_quotient_to_a_point_collapses_everything_and_still_works():
    graph = random_connected_graph(9, 6, 10)
    quotient_graph, quotient_map = quotient_chain_map(graph, np.zeros(6, dtype=int))
    assert quotient_graph.dims == (1, 0)
    source = two_complex(graph)
    target = two_complex(quotient_graph)
    lifted = induced_quotient_map(source, quotient_map, target)
    assert lifted.maps[1].shape == (0, 10)
    assert lifted.maps[2].shape == (0, source.dims[2])
    assert lifted.commutation_residuals(tol=0.0) == (0.0, 0.0)


def test_induced_quotient_map_rejects_broken_or_mismatched_maps():
    graph = random_connected_graph(4, 8, 14)
    partition = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    quotient_graph, quotient_map = quotient_chain_map(graph, partition)
    source = two_complex(graph)
    target = two_complex(quotient_graph)
    f0, f1 = quotient_map.maps
    broken_f1 = f1.copy()
    row, col = np.argwhere(broken_f1 != 0.0)[0]
    broken_f1[row, col] *= -1.0
    broken = ChainMap(graph, quotient_graph, (f0, broken_f1))
    with pytest.raises(ValueError, match="commute exactly"):
        induced_quotient_map(source, broken, target)
    other_source = two_complex(random_connected_graph(5, 8, 14))
    with pytest.raises(ValueError, match="1-skeleton of source"):
        induced_quotient_map(other_source, quotient_map, target)
    other_quotient, _ = quotient_chain_map(
        graph, np.array([0, 0, 1, 1, 2, 2, 3, 3])
    )
    with pytest.raises(ValueError, match="1-skeleton of target"):
        induced_quotient_map(source, quotient_map, two_complex(other_quotient))


def test_switch_instance_residuals_discriminate():
    instance = make_two_complex_switch_instance(seed=2026, num_decoys=4)
    scores = commutation_scores(instance)
    assert len(scores) == 5
    assert scores[0] == 0.0  # true target: exact chain map at both degrees
    assert instance.chain_map.commutation_residuals(tol=0.0) == (0.0, 0.0)
    assert all(score > 1e-6 for score in scores[1:])


def test_switch_instance_is_deterministic():
    first = make_two_complex_switch_instance(seed=99)
    second = make_two_complex_switch_instance(seed=99)
    for left, right in zip(first.candidates, second.candidates):
        for b_left, b_right in zip(left.boundaries, right.boundaries):
            assert np.array_equal(b_left, b_right)
    for m_left, m_right in zip(first.chain_map.maps, second.chain_map.maps):
        assert np.array_equal(m_left, m_right)
    assert commutation_scores(first) == commutation_scores(second)


def test_switch_instance_decoys_match_shapes_at_every_degree():
    instance = make_two_complex_switch_instance(seed=5, num_decoys=3)
    for decoy in instance.decoy_targets:
        assert decoy.dims == instance.true_target.dims
        probe = ChainMap(instance.source, decoy, instance.chain_map.maps)
        assert len(probe.commutation_residuals()) == 2


def test_switch_instance_with_tree_source():
    instance = make_two_complex_switch_instance(
        seed=17, num_vertices=8, num_edges=7, num_classes=4, num_decoys=3
    )
    assert instance.source.dims == (8, 7, 0)
    assert instance.chain_map.maps[2].shape == (instance.true_target.dims[2], 0)
    scores = commutation_scores(instance)
    assert scores[0] == 0.0
    assert all(score > 1e-9 for score in scores[1:])
