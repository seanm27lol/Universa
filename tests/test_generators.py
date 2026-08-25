import numpy as np

from universa.generators import (
    make_switch_instance,
    quotient_chain_map,
    random_connected_graph,
    subseed,
)
from universa.structures import ChainMap


def test_subseeds_are_stable_separated_and_63_bit():
    first = subseed(123, "graph")
    again = subseed(123, "graph")
    other = subseed(123, "partition")
    assert first == again
    assert first != other
    assert 0 <= first < 2**63


def test_random_graph_is_connected_and_deterministic():
    graph_a = random_connected_graph(42, 8, 14)
    graph_b = random_connected_graph(42, 8, 14)
    assert np.array_equal(graph_a.boundaries[0], graph_b.boundaries[0])
    boundary = graph_a.boundaries[0]
    # Connected <=> rank(B1) = V - 1 over the reals.
    rank = np.linalg.matrix_rank(boundary, tol=1e-10)
    assert rank == boundary.shape[0] - 1
    # Canonical signs: every column sums to zero with one +1 and one -1.
    assert np.allclose(boundary.sum(axis=0), 0.0)
    assert set(np.unique(np.abs(boundary))) == {0.0, 1.0}


def test_quotient_chain_map_commutes_exactly():
    source = random_connected_graph(7, 8, 14)
    partition = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    target, chain_map = quotient_chain_map(source, partition)
    assert chain_map.is_chain_map(tol=0.0)
    (residual,) = chain_map.commutation_residuals(tol=0.0)
    assert residual == 0.0
    assert target.boundaries[0].shape == (4, chain_map.maps[1].shape[0])


def test_quotient_collapsed_edges_vanish():
    source = random_connected_graph(11, 6, 10)
    partition = np.array([0, 0, 1, 1, 2, 2])
    _, chain_map = quotient_chain_map(source, partition)
    boundary = source.boundaries[0]
    f1 = chain_map.maps[1]
    for edge in range(boundary.shape[1]):
        tail = int(np.argmin(boundary[:, edge]))
        head = int(np.argmax(boundary[:, edge]))
        if partition[tail] == partition[head]:
            assert np.allclose(f1[:, edge], 0.0)


def test_switch_instance_residuals_discriminate():
    instance = make_switch_instance(seed=2026, num_decoys=4)
    scores = instance.commutation_scores()
    assert len(scores) == 5
    assert scores[0] == 0.0  # true target: exact chain map
    assert all(score > 1e-6 for score in scores[1:])


def test_switch_instance_is_deterministic():
    first = make_switch_instance(seed=99)
    second = make_switch_instance(seed=99)
    for left, right in zip(first.candidates, second.candidates):
        assert np.array_equal(left.boundaries[0], right.boundaries[0])
    assert first.commutation_scores() == second.commutation_scores()


def test_decoys_are_genuinely_different_structures():
    instance = make_switch_instance(seed=5, num_decoys=3)
    true_b1 = instance.true_target.boundaries[0]
    for decoy in instance.decoy_targets:
        assert not np.array_equal(true_b1, decoy.boundaries[0])


def test_chain_map_rejects_wrong_target_shape():
    instance = make_switch_instance(seed=3)
    bigger = random_connected_graph(1, 6, 12)  # 6 vertices, not 4
    try:
        ChainMap(instance.source, bigger, instance.chain_map.maps)
        raised = False
    except ValueError:
        raised = True
    assert raised
