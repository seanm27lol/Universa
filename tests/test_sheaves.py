import numpy as np
import pytest

from universa.generators import incidence_matrix
from universa.operators import CERT_TOL, nullspace_basis
from universa.sheaves import (
    Sheaf,
    SheafMorphism,
    coboundary,
    make_sheaf_switch_instance,
    planted_morphism,
    random_sheaf,
    to_chain_complex,
)
from universa.structures import ChainComplex

TRIANGLE_B1 = np.array(
    [[-1.0, 0.0, -1.0], [1.0, -1.0, 0.0], [0.0, 1.0, 1.0]]
)
# edges: e0 = 0 -> 1, e1 = 1 -> 2, e2 = 0 -> 2


def constant_sheaf(b1: np.ndarray) -> Sheaf:
    """The constant sheaf: every stalk R^1, every restriction the identity."""
    num_vertices, num_edges = b1.shape
    restrictions = {}
    for e in range(num_edges):
        for v in np.flatnonzero(b1[:, e]):
            restrictions[(int(v), e)] = np.ones((1, 1))
    return Sheaf(
        ChainComplex((b1,)),
        (1,) * num_vertices,
        (1,) * num_edges,
        restrictions,
    )


def hand_sheaf() -> Sheaf:
    """Triangle with vertex dims (2, 1, 1), edge dims (1, 1, 1)."""
    restrictions = {
        (0, 0): np.array([[1.0, -1.0]]),  # rho_{0,e0}: R^2 -> R^1
        (1, 0): np.array([[2.0]]),
        (1, 1): np.array([[3.0]]),
        (2, 1): np.array([[1.0]]),
        (0, 2): np.array([[0.0, 1.0]]),
        (2, 2): np.array([[2.0]]),
    }
    return Sheaf(
        ChainComplex((TRIANGLE_B1,)), (2, 1, 1), (1, 1, 1), restrictions
    )


def test_coboundary_matches_hand_computed_triangle():
    sheaf = hand_sheaf()
    # (delta x)_e = rho_{h,e} x_h - rho_{t,e} x_t; column blocks ordered by
    # vertex (dims 2, 1, 1), row blocks by edge (dims 1, 1, 1).
    expected = np.array(
        [
            [-1.0, 1.0, 2.0, 0.0],  # e0 = 0 -> 1
            [0.0, 0.0, -3.0, 1.0],  # e1 = 1 -> 2
            [0.0, -1.0, 0.0, 2.0],  # e2 = 0 -> 2
        ]
    )
    assert np.array_equal(coboundary(sheaf), expected)


def test_constant_sheaf_coboundary_is_b1_transpose():
    sheaf = constant_sheaf(TRIANGLE_B1)
    assert np.array_equal(coboundary(sheaf), TRIANGLE_B1.T)


def test_to_chain_complex_wraps_the_coboundary():
    sheaf = hand_sheaf()
    complex_ = to_chain_complex(sheaf)
    assert complex_.top_degree == 1
    assert complex_.dims == (sheaf.c1_dim, sheaf.c0_dim) == (3, 4)
    assert np.array_equal(complex_.boundaries[0], coboundary(sheaf))


def test_sheaf_validation_is_fail_closed():
    base = ChainComplex((TRIANGLE_B1,))
    good = {
        key: np.ones((1, 1))
        for key in [(0, 0), (1, 0), (1, 1), (2, 1), (0, 2), (2, 2)]
    }
    with pytest.raises(ValueError, match="shape"):
        Sheaf(base, (1, 1, 1), (1, 1, 1), {k: np.ones((2, 1)) for k in good})
    missing = dict(good)
    del missing[(2, 2)]
    with pytest.raises(ValueError, match="incidences"):
        Sheaf(base, (1, 1, 1), (1, 1, 1), missing)
    extra = dict(good)
    extra[(1, 2)] = np.ones((1, 1))  # vertex 1 is not incident to e2
    with pytest.raises(ValueError, match="incidences"):
        Sheaf(base, (1, 1, 1), (1, 1, 1), extra)
    with pytest.raises(ValueError, match="positive"):
        Sheaf(base, (1, 0, 1), (1, 1, 1), good)
    two_deep = ChainComplex((TRIANGLE_B1, np.array([[1.0], [1.0], [-1.0]])))
    with pytest.raises(ValueError, match="1-complex"):
        Sheaf(two_deep, (1, 1, 1), (1, 1, 1), good)
    not_a_graph = ChainComplex((np.array([[-1.0], [-1.0]]),))
    with pytest.raises(ValueError, match="incidence"):
        Sheaf(
            not_a_graph,
            (1, 1),
            (1,),
            {(0, 0): np.ones((1, 1)), (1, 0): np.ones((1, 1))},
        )


def test_sheaf_morphism_validation_is_fail_closed():
    b1_a = incidence_matrix(3, [(0, 1), (1, 2)])
    b1_b = incidence_matrix(3, [(0, 2), (1, 2)])
    sheaf_a = constant_sheaf(b1_a)
    sheaf_b = constant_sheaf(b1_b)
    vertex_maps = tuple(np.eye(d) for d in sheaf_a.vertex_dims)
    edge_maps = tuple(np.eye(d) for d in sheaf_a.edge_dims)
    with pytest.raises(ValueError, match="common base"):
        SheafMorphism(sheaf_a, sheaf_b, vertex_maps, edge_maps)
    with pytest.raises(ValueError, match="per vertex"):
        SheafMorphism(sheaf_a, sheaf_a, (np.eye(1),), edge_maps)
    with pytest.raises(ValueError, match="edge_maps"):
        SheafMorphism(sheaf_a, sheaf_a, vertex_maps, (np.eye(2),) * 2)


def test_planted_morphism_naturality_is_exactly_zero():
    for seed in (0, 1, 20260825):
        source = random_sheaf(seed, num_vertices=5, num_edges=8)
        target, morphism = planted_morphism(source, seed)
        assert target.vertex_dims == source.vertex_dims
        assert target.edge_dims == source.edge_dims
        residuals = morphism.naturality_residuals()
        assert len(residuals) == 2 * source.num_edges
        assert all(r == 0.0 for r in residuals)
        assert morphism.is_morphism(tol=0.0)
        (commutation,) = morphism.to_chain_map().commutation_residuals()
        assert commutation == 0.0


def test_planted_target_entries_are_simple_dyadic_fractions():
    source = random_sheaf(7)
    target, _ = planted_morphism(source, 7)
    # Scales in {+/-1, +/-2}: rho' entries are integers times a power of
    # two in {1/2, 1, 2}, so doubling clears every denominator exactly.
    for block in target.restrictions.values():
        assert np.array_equal(block * 2.0, np.round(block * 2.0))


def test_identity_morphism_is_exact_and_tampering_is_detected():
    source = random_sheaf(3, num_vertices=4, num_edges=6)
    identity = SheafMorphism(
        source,
        source,
        tuple(np.eye(d) for d in source.vertex_dims),
        tuple(np.eye(d) for d in source.edge_dims),
    )
    assert all(r == 0.0 for r in identity.naturality_residuals())
    # Tamper with one restriction entry: the same stalk maps stop commuting.
    tampered = dict(source.restrictions)
    key = source.incidence_pairs[0]
    tampered[key] = tampered[key] + 1.0
    other = Sheaf(source.base, source.vertex_dims, source.edge_dims, tampered)
    probe = SheafMorphism(source, other, identity.vertex_maps, identity.edge_maps)
    residuals = probe.naturality_residuals()
    assert not probe.is_morphism(tol=0.0)
    expected = float(np.sqrt(sum(r * r for r in residuals)))
    # Integer blocks: a nonzero residual is at least 1, never a sliver.
    assert expected >= 1.0
    (commutation,) = probe.to_chain_map().commutation_residuals()
    # Same value, summed in different block order: not a fit tolerance.
    assert abs(commutation - expected) <= 1e-12


def test_random_sheaf_is_deterministic_with_bounded_stalks():
    first = random_sheaf(17, num_vertices=5, num_edges=7, max_stalk_dim=3)
    second = random_sheaf(17, num_vertices=5, num_edges=7, max_stalk_dim=3)
    assert first.vertex_dims == second.vertex_dims
    assert first.edge_dims == second.edge_dims
    assert all(1 <= d <= 3 for d in (*first.vertex_dims, *first.edge_dims))
    for key in first.restrictions:
        assert np.array_equal(first.restrictions[key], second.restrictions[key])


def test_planted_morphism_is_deterministic():
    source = random_sheaf(11)
    target_a, morphism_a = planted_morphism(source, 11)
    target_b, morphism_b = planted_morphism(source, 11)
    for key in target_a.restrictions:
        assert np.array_equal(
            target_a.restrictions[key], target_b.restrictions[key]
        )
    for left, right in zip(morphism_a.vertex_maps, morphism_b.vertex_maps):
        assert np.array_equal(left, right)
    for left, right in zip(morphism_a.edge_maps, morphism_b.edge_maps):
        assert np.array_equal(left, right)


def test_sheaf_switch_instance_discriminates():
    instance = make_sheaf_switch_instance(seed=20260825, num_decoys=4)
    scores = instance.commutation_scores()
    assert len(scores) == 5
    assert scores[0] == 0.0  # true target: exact planted morphism
    assert all(score > 1e-6 for score in scores[1:])


def test_sheaf_switch_instance_is_deterministic():
    first = make_sheaf_switch_instance(seed=99)
    second = make_sheaf_switch_instance(seed=99)
    for left, right in zip(first.candidates, second.candidates):
        assert np.array_equal(left.boundaries[0], right.boundaries[0])
    assert first.commutation_scores() == second.commutation_scores()


def test_sheaf_decoys_are_genuinely_different_structures():
    instance = make_sheaf_switch_instance(seed=5, num_decoys=3)
    assert instance.source.dims == instance.true_target.dims
    true_delta = instance.true_target.boundaries[0]
    for decoy in instance.decoy_targets:
        assert decoy.dims == instance.true_target.dims  # same stalk dims
        assert not np.array_equal(true_delta, decoy.boundaries[0])


def test_harmonic_space_certificate_matches_hand_computed_dims():
    # Constant sheaf on the (connected) triangle: H^0 = constants, dim 1.
    cert = nullspace_basis(to_chain_complex(constant_sheaf(TRIANGLE_B1)).boundaries[0])
    assert cert.basis.shape == (3, 1)
    assert cert.observed_rank == 2
    assert cert.membership_residual <= CERT_TOL
    assert cert.orthonormality_residual <= CERT_TOL

    # Constant sheaf on a disconnected base (path 0-1-2 plus isolated 3):
    # H^0 has one dimension per connected component.
    b1 = incidence_matrix(4, [(0, 1), (1, 2)])
    cert = nullspace_basis(to_chain_complex(constant_sheaf(b1)).boundaries[0])
    assert cert.basis.shape == (4, 2)
    assert cert.observed_rank == 2

    # Zeroing rho_{2,e2} forces x0 = 0 via e2, then x1 = x0 and x2 = x1
    # collapse everything: H^0 = {0}.
    restrictions = {
        (0, 0): np.ones((1, 1)),
        (1, 0): np.ones((1, 1)),
        (1, 1): np.ones((1, 1)),
        (2, 1): np.ones((1, 1)),
        (0, 2): np.ones((1, 1)),
        (2, 2): np.zeros((1, 1)),
    }
    killed = Sheaf(ChainComplex((TRIANGLE_B1,)), (1, 1, 1), (1, 1, 1), restrictions)
    cert = nullspace_basis(to_chain_complex(killed).boundaries[0])
    assert cert.basis.shape == (3, 0)
    assert cert.observed_rank == 3
