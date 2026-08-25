import numpy as np
import pytest

from universa.multihop import (
    DECOY_TOL,
    HopChain,
    compose,
    localize_misfit,
    make_multihop_instance,
)
from universa.structures import ChainComplex, ChainMap

FAMILIES = ("graph", "two_complex")


def assert_complex_equal(first: ChainComplex, second: ChainComplex) -> None:
    assert len(first.boundaries) == len(second.boundaries)
    for b_first, b_second in zip(first.boundaries, second.boundaries):
        assert np.array_equal(b_first, b_second)


def assert_map_equal(first: ChainMap, second: ChainMap) -> None:
    for m_first, m_second in zip(first.maps, second.maps):
        assert np.array_equal(m_first, m_second)


def test_composition_commutes_exactly_end_to_end():
    for family in FAMILIES:
        for hops in (1, 2, 3):
            chain = make_multihop_instance(seed=2026, hops=hops, family=family)
            composed = chain.composed
            top_degree = chain.complexes[0].top_degree
            assert composed.is_chain_map(tol=0.0)
            assert composed.commutation_residuals(tol=0.0) == (0.0,) * top_degree
            # The defect matrices are exact zeros, not below a tolerance.
            for k in range(1, top_degree + 1):
                defect = (
                    composed.target.boundaries[k - 1] @ composed.maps[k]
                    - composed.maps[k - 1] @ composed.source.boundaries[k - 1]
                )
                assert not np.any(defect != 0.0)


def test_composed_degree_maps_are_products_of_hop_maps():
    for family in FAMILIES:
        chain = make_multihop_instance(seed=7, hops=3, family=family)
        composed = chain.composed
        assert_complex_equal(composed.source, chain.complexes[0])
        assert_complex_equal(composed.target, chain.complexes[-1])
        for k in range(len(composed.maps)):
            product = chain.maps[0].maps[k]
            for hop in range(1, chain.hops):
                product = chain.maps[hop].maps[k] @ product
            assert np.array_equal(composed.maps[k], product)


def test_localizability_per_hop():
    # Inject a decoy at each hop in turn: the residual signature must
    # identify exactly that hop, with later hops exactly clean.
    for family in FAMILIES:
        chain = make_multihop_instance(seed=11, hops=3, family=family)
        for wrong_hop in range(chain.hops):
            for decoy_index in (1, len(chain.decoys[wrong_hop])):
                choices = [0] * chain.hops
                choices[wrong_hop] = decoy_index
                residuals = localize_misfit(chain, choices)
                assert residuals[wrong_hop] > DECOY_TOL
                for hop, residual in enumerate(residuals):
                    if hop != wrong_hop:
                        assert residual == 0.0  # exact, later hops included


def test_all_true_choices_score_exact_zero():
    for family in FAMILIES:
        chain = make_multihop_instance(seed=5, hops=2, family=family)
        assert localize_misfit(chain, (0, 0)) == (0.0, 0.0)


def test_instance_is_deterministic():
    for family in FAMILIES:
        first = make_multihop_instance(seed=99, hops=3, family=family)
        second = make_multihop_instance(seed=99, hops=3, family=family)
        assert len(first.complexes) == len(second.complexes)
        for left, right in zip(first.complexes, second.complexes):
            assert_complex_equal(left, right)
        for left, right in zip(first.maps, second.maps):
            assert_map_equal(left, right)
        assert_map_equal(first.composed, second.composed)
        for decoys_left, decoys_right in zip(first.decoys, second.decoys):
            assert len(decoys_left) == len(decoys_right)
            for left, right in zip(decoys_left, decoys_right):
                assert_complex_equal(left, right)
        choices = tuple(range(1, first.hops + 1))
        assert localize_misfit(first, choices) == localize_misfit(
            second, choices
        )


def test_decoy_contract_holds_at_every_hop():
    chain = make_multihop_instance(
        seed=3, hops=3, num_decoys=4, family="two_complex"
    )
    for hop in range(chain.hops):
        top_degree = chain.complexes[hop].top_degree
        assert (
            chain.maps[hop].commutation_residuals(tol=0.0)
            == (0.0,) * top_degree
        )
        assert len(chain.decoys[hop]) == 4
        for decoy in chain.decoys[hop]:
            assert decoy.dims == chain.complexes[hop + 1].dims
            probe = ChainMap(
                chain.complexes[hop], decoy, chain.maps[hop].maps
            )
            assert any(r > DECOY_TOL for r in probe.commutation_residuals())


def test_single_hop_chain():
    chain = make_multihop_instance(seed=17, hops=1)
    assert chain.hops == 1
    assert len(chain.complexes) == 2
    assert_map_equal(chain.composed, chain.maps[0])
    assert localize_misfit(chain, (0,)) == (0.0,)
    (residual,) = localize_misfit(chain, (1,))
    assert residual > DECOY_TOL


def test_tree_source():
    for family in FAMILIES:
        chain = make_multihop_instance(
            seed=23, hops=2, num_vertices=8, num_edges=7, family=family
        )
        top_degree = chain.complexes[0].top_degree
        if family == "two_complex":
            assert chain.complexes[0].dims[2] == 0  # no faces on a tree
            assert chain.maps[0].maps[2].shape == (
                chain.complexes[1].dims[2],
                0,
            )
        assert chain.composed.commutation_residuals(tol=0.0) == (
            0.0,
        ) * top_degree
        for wrong_hop in range(chain.hops):
            choices = [0] * chain.hops
            choices[wrong_hop] = 1
            residuals = localize_misfit(chain, choices)
            assert residuals[wrong_hop] > DECOY_TOL
            assert all(
                residual == 0.0
                for hop, residual in enumerate(residuals)
                if hop != wrong_hop
            )


def test_seed_sweep_all_hops_exact_and_localizable():
    for seed in (0, 1, 7, 2026):
        for family in FAMILIES:
            chain = make_multihop_instance(seed=seed, hops=3, family=family)
            assert chain.composed.is_chain_map(tol=0.0)
            assert localize_misfit(chain, (0, 0, 0)) == (0.0, 0.0, 0.0)
            for wrong_hop in range(chain.hops):
                choices = [0] * chain.hops
                choices[wrong_hop] = 1
                residuals = localize_misfit(chain, choices)
                assert residuals[wrong_hop] > DECOY_TOL
                assert all(
                    residual == 0.0
                    for hop, residual in enumerate(residuals)
                    if hop != wrong_hop
                )


def test_default_class_counts_halve_vertices():
    chain = make_multihop_instance(seed=61, hops=3)
    assert [complex_.dims[0] for complex_ in chain.complexes] == [8, 4, 2, 2]


def test_candidates_index_zero_is_true_target():
    chain = make_multihop_instance(seed=51, hops=2)
    for hop in range(chain.hops):
        assert chain.candidates(hop)[0] is chain.complexes[hop + 1]
        assert len(chain.candidates(hop)) == len(chain.decoys[hop]) + 1
    with pytest.raises(ValueError, match="no hop 2"):
        chain.candidates(2)


def test_compose_rejects_unlinked_or_inexact_maps():
    chain = make_multihop_instance(seed=31, hops=2, family="two_complex")
    first, second = chain.maps
    with pytest.raises(ValueError, match="target of first"):
        compose(second, first)  # wrong order: the complexes do not link
    f0, f1, f2 = first.maps
    broken_f1 = f1.copy()
    row, col = np.argwhere(broken_f1 != 0.0)[0]
    broken_f1[row, col] *= -1.0
    broken = ChainMap(first.source, first.target, (f0, broken_f1, f2))
    with pytest.raises(ValueError, match="commute exactly"):
        compose(broken, second)


def test_localize_misfit_validates_choices():
    chain = make_multihop_instance(seed=41, hops=2)
    with pytest.raises(ValueError, match="one candidate choice per hop"):
        localize_misfit(chain, (0,))
    with pytest.raises(ValueError, match="candidates"):
        localize_misfit(chain, (0, 99))
    with pytest.raises(ValueError, match="candidates"):
        localize_misfit(chain, (0, -1))


def test_hopchain_validation_is_fail_closed():
    chain = make_multihop_instance(seed=43, hops=2)
    with pytest.raises(ValueError, match="one chain map per hop"):
        HopChain(chain.seed, chain.complexes, chain.maps[:1], chain.decoys)
    with pytest.raises(ValueError, match="one decoy set per hop"):
        HopChain(chain.seed, chain.complexes, chain.maps, chain.decoys[:1])
    with pytest.raises(ValueError, match="does not link the chain"):
        HopChain(
            chain.seed,
            chain.complexes,
            (chain.maps[1], chain.maps[0]),
            chain.decoys,
        )
    # A "decoy" that is the true target commutes exactly and must be
    # rejected: it fails the residual contract.
    with pytest.raises(ValueError, match="residual contract"):
        HopChain(
            chain.seed,
            chain.complexes,
            chain.maps,
            ((chain.complexes[1],), chain.decoys[1]),
        )
    # A decoy with the wrong dims is rejected before any probing.
    with pytest.raises(ValueError, match="dims"):
        HopChain(
            chain.seed,
            chain.complexes,
            chain.maps,
            ((chain.complexes[2],), chain.decoys[1]),
        )


def test_make_multihop_instance_validates_parameters():
    with pytest.raises(ValueError, match="at least one hop"):
        make_multihop_instance(seed=1, hops=0)
    with pytest.raises(ValueError, match="unknown family"):
        make_multihop_instance(seed=1, family="sheaf")
    with pytest.raises(ValueError, match="class counts"):
        make_multihop_instance(seed=1, hops=2, class_counts=(4,))
    with pytest.raises(ValueError, match="infeasible"):
        make_multihop_instance(seed=1, hops=1, class_counts=(99,))
    with pytest.raises(ValueError, match="infeasible"):
        make_multihop_instance(seed=1, hops=1, class_counts=(1,))
