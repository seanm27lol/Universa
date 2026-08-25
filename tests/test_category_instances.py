import numpy as np
import pytest

from universa.category_instances import (
    cyclic_group,
    group_as_category,
    induced_nerve_map,
    make_group_switch_instance,
    symmetric_group_3,
)
from universa.nerve import FiniteCategory, nerve_chain_complex


def hand_built_z2() -> FiniteCategory:
    """Z/2 as a one-object category, written out by hand."""
    morphisms = ((0, 0), (0, 0))  # identity e, generator g
    compose = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}
    return FiniteCategory(1, morphisms, compose)


def discrete_category(num_objects: int) -> FiniteCategory:
    """Identities only: a valid category that is not a group."""
    morphisms = tuple((k, k) for k in range(num_objects))
    compose = {(k, k): k for k in range(num_objects)}
    return FiniteCategory(num_objects, morphisms, compose)


def planted_homomorphisms():
    """The three planted families, with phi written out as ground truth."""
    return (
        ("z6_to_z3", cyclic_group(6), cyclic_group(3), (0, 1, 2, 0, 1, 2)),
        ("z4_to_z2", cyclic_group(4), cyclic_group(2), (0, 1, 0, 1)),
        # S3 in lexicographic one-line order; parity by inversion count.
        ("s3_sign", symmetric_group_3(), cyclic_group(2), (0, 1, 1, 0, 0, 1)),
    )


def test_group_as_category_matches_hand_built_z2():
    from_table = group_as_category(np.array([[0, 1], [1, 0]]))
    assert from_table == hand_built_z2()
    complex_ = nerve_chain_complex(from_table)
    assert complex_.dims == (1, 2, 4)
    # One object => d1 = 0; every pair composable => N2 = 4.
    assert np.abs(complex_.boundaries[0]).max() == 0.0


def test_group_as_category_is_fail_closed():
    with pytest.raises(ValueError, match="square"):
        group_as_category(np.array([[0, 1, 2], [1, 0, 2]]))
    with pytest.raises(ValueError, match="indices"):
        group_as_category(np.array([[0, 2], [1, 0]]))
    with pytest.raises(ValueError, match="integers"):
        group_as_category(np.array([[0.0, 1.0], [1.0, 0.0]]))
    # Z/3 with one broken product: (1.1).2 = 0 != 1 = 1.(1.2).
    broken = np.array([[0, 1, 2], [1, 0, 0], [2, 0, 1]])
    with pytest.raises(ValueError, match="associativity"):
        group_as_category(broken)
    # A monoid that is not a group: element 0 has no inverse.
    with pytest.raises(ValueError, match="inverse"):
        group_as_category(np.array([[0, 0], [0, 1]]))


def test_group_orders_feed_nerve_dimensions():
    for n in (1, 2, 6):
        complex_ = nerve_chain_complex(cyclic_group(n))
        assert complex_.dims == (1, n, n * n)
    s3 = symmetric_group_3()
    assert len(s3.morphisms) == 6
    assert nerve_chain_complex(s3).dims == (1, 6, 36)
    # S3 is nonabelian: the two transpositions do not commute.
    assert s3.compose[(1, 2)] != s3.compose[(2, 1)]


def test_planted_maps_commute_exactly_at_degrees_1_and_2():
    for name, source, target, phi in planted_homomorphisms():
        chain_map = induced_nerve_map(phi, source, target)
        residuals = chain_map.commutation_residuals(tol=0.0)
        assert residuals == (0.0, 0.0), name
        assert chain_map.is_chain_map(tol=0.0), name


def test_nerve_map_dimensions_track_group_orders():
    phi = (0, 1, 2, 0, 1, 2)
    chain_map = induced_nerve_map(phi, cyclic_group(6), cyclic_group(3))
    assert chain_map.source.dims == (1, 6, 36)
    assert chain_map.target.dims == (1, 3, 9)
    n0, n1, n2 = chain_map.maps
    assert n0.shape == (1, 1)
    assert n1.shape == (3, 6)
    assert n2.shape == (9, 36)
    # N2 sends the pair (f, g) = (2, 5) to (phi f, phi g) = (2, 2):
    # column g * 6 + f = 32, row phi(g) * 3 + phi(f) = 8.
    column = n2[:, 5 * 6 + 2]
    assert column[2 * 3 + 2] == 1.0
    assert column.sum() == 1.0

    sign = (0, 1, 1, 0, 0, 1)
    chain_map = induced_nerve_map(sign, symmetric_group_3(), cyclic_group(2))
    assert chain_map.source.dims == (1, 6, 36)
    assert chain_map.target.dims == (1, 2, 4)
    assert chain_map.maps[2].shape == (4, 36)


def test_non_homomorphisms_are_rejected():
    source, target = cyclic_group(4), cyclic_group(2)
    with pytest.raises(ValueError, match="homomorphism"):
        induced_nerve_map((1, 1, 1, 1), source, target)  # constant map
    # Near miss: phi(1 + 1) = phi(2) = 1 but phi(1) phi(1) = 0.
    with pytest.raises(ValueError, match="homomorphism"):
        induced_nerve_map((0, 1, 1, 0), source, target)
    # Reduction mod 3 is not a homomorphism Z/4 -> Z/3.
    with pytest.raises(ValueError, match="homomorphism"):
        induced_nerve_map((0, 1, 2, 0), cyclic_group(4), cyclic_group(3))
    with pytest.raises(ValueError, match="target element indices"):
        induced_nerve_map((0, 1, 2, 3), source, target)
    with pytest.raises(ValueError, match="one-object"):
        induced_nerve_map((0,), discrete_category(2), hand_built_z2())


def test_switch_instance_residuals_discriminate():
    instance = make_group_switch_instance(seed=2026)
    scores = instance.commutation_scores()
    assert len(scores) == 3  # true target plus two decoys
    assert scores[0] == 0.0  # true target: exact chain map
    assert all(score > 1e-9 for score in scores[1:])
    for family in ("z4_to_z2", "s3_sign"):
        instance = make_group_switch_instance(
            seed=2026, family=family, num_decoys=1
        )
        scores = instance.commutation_scores()
        assert scores[0] == 0.0, family
        assert scores[1] > 1e-9, family


def test_decoys_are_genuinely_different_structures():
    instance = make_group_switch_instance(seed=5)
    true_d2 = instance.true_target.boundaries[1]
    for decoy in instance.decoy_targets:
        # Same-order target group: the planted map stays shape-compatible.
        assert decoy.dims == instance.true_target.dims
        assert not np.array_equal(true_d2, decoy.boundaries[1])


def test_switch_instance_is_deterministic():
    first = make_group_switch_instance(seed=99)
    second = make_group_switch_instance(seed=99)
    for left, right in zip(first.candidates, second.candidates):
        for left_b, right_b in zip(left.boundaries, right.boundaries):
            assert np.array_equal(left_b, right_b)
    assert first.commutation_scores() == second.commutation_scores()


def test_decoy_pool_exhaustion_is_an_error():
    # Z/2 has exactly one relabeled table; a second decoy cannot exist.
    with pytest.raises(RuntimeError, match="discriminating decoys"):
        make_group_switch_instance(seed=7, family="s3_sign", num_decoys=2)


def test_unknown_family_rejected():
    with pytest.raises(ValueError, match="unknown family"):
        make_group_switch_instance(seed=1, family="z3_to_z2")
