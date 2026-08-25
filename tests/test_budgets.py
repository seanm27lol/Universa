import numpy as np
import pytest

from universa.budgets import (
    Probes,
    budget_curve,
    identifiability_threshold,
    identification_residual,
    make_budget_instance,
    probe_instance,
    probe_operator,
)
from universa.generators import incidence_matrix, quotient_chain_map
from universa.operators import CERT_TOL
from universa.structures import ChainComplex

SEED = 2026
BUDGETS = (0, 1, 2, 3, 4, 5, 6, 7, 8)
# make_budget_instance(SEED) has threshold 6: sub-threshold at N in 0..5,
# uniqueness from N = 6 on. Verified empirically and by the formula test.
THRESHOLD = 6


def test_below_threshold_truth_admitted_but_nonunique():
    curve = budget_curve(SEED, BUDGETS)
    assert curve.threshold == THRESHOLD
    for i, budget in enumerate(BUDGETS):
        if budget >= curve.threshold:
            continue
        # The true candidate still admits the truth...
        assert curve.residuals[0][i] <= CERT_TOL
        # ...but the feasible set is positive-dimensional, with nullity
        # exactly threshold - N (generic probes: rank grows one per probe).
        assert curve.nullities[0][i] == curve.threshold - budget > 0


def test_above_threshold_uniqueness():
    curve = budget_curve(SEED, BUDGETS)
    for i, budget in enumerate(BUDGETS):
        if budget < curve.threshold:
            continue
        # Nullity 0: the feasible set is a singleton, and since the truth is
        # still admitted (residual 0), that singleton is the truth.
        assert curve.nullities[0][i] == 0
        assert curve.residuals[0][i] <= CERT_TOL


def test_nullity_formula_holds_for_every_candidate():
    curve = budget_curve(SEED, BUDGETS)
    for nullities in curve.nullities:
        for i, budget in enumerate(BUDGETS):
            assert nullities[i] == max(curve.threshold - budget, 0)


def test_decoys_never_admit_the_truth():
    curve = budget_curve(SEED, BUDGETS)
    for candidate_residuals in curve.residuals[1:]:
        for residual in candidate_residuals:
            assert residual > 1e-9


def test_budget_curve_is_deterministic():
    first = budget_curve(SEED, BUDGETS)
    second = budget_curve(SEED, BUDGETS)
    assert first.budgets == second.budgets
    assert first.threshold == second.threshold
    assert first.residuals == second.residuals
    assert first.nullities == second.nullities
    instance_a = make_budget_instance(SEED)
    instance_b = make_budget_instance(SEED)
    assert np.array_equal(instance_a.transported, instance_b.transported)
    assert np.array_equal(instance_a.source_cycle, instance_b.source_cycle)


def test_nested_budgets_use_probe_prefixes():
    # The budget-N row of a wider grid equals the same row of a narrower grid.
    wide = budget_curve(SEED, (0, 1, 2, 3, 4, 5))
    narrow = budget_curve(SEED, (0, 1, 2))
    assert wide.residuals[0][:3] == narrow.residuals[0]
    assert wide.nullities[0][:3] == narrow.nullities[0]


def hand_sized_target():
    """A 6-cycle quotiented onto the triangle by classes [0,1,2,0,1,2].

    The planted source cycle s = (1,1,1,1,1,-1) (the 6-cycle itself, with the
    last edge traversed against its stored orientation) transports to
    a_* = (2,-2,2) = 2 * (the triangle cycle) by exact integer arithmetic.
    """
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
    source = ChainComplex((incidence_matrix(6, edges),))
    partition = np.array([0, 1, 2, 0, 1, 2])
    target, chain_map = quotient_chain_map(source, partition)
    source_cycle = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0])
    return source, target, chain_map, source_cycle


def test_threshold_matches_documented_formula_on_hand_example():
    source, target, chain_map, source_cycle = hand_sized_target()
    boundary = target.boundaries[0]
    num_vertices, num_edges = boundary.shape
    assert (num_vertices, num_edges) == (3, 3)  # the triangle
    # Documented formula: N* = nullity(B1) = E - V + 1 for a connected target.
    assert identifiability_threshold(target) == num_edges - num_vertices + 1 == 1
    # The planted transport is exactly the doubled triangle cycle.
    transported = chain_map.maps[1] @ source_cycle
    assert np.array_equal(transported, np.array([2.0, -2.0, 2.0]))
    assert np.linalg.norm(boundary @ transported) == 0.0

    # N = 0 < N*: truth admitted (residual 0), feasible set 1-dimensional.
    empty = Probes(np.zeros((0, 3)), np.zeros(0))
    recovered = probe_instance(empty, boundary)
    assert recovered.nullity == 1
    assert recovered.distance(transported) <= CERT_TOL

    # N = 1 = N*: one generic probe x = (1,0,0), y = <x, a_*> = 2.
    # By hand: a in ker(B1) means a = t * (1,-1,1); x.a = t = 2 pins a = a_*.
    probes = Probes(np.array([[1.0, 0.0, 0.0]]), np.array([2.0]))
    recovered = probe_instance(probes, boundary)
    assert recovered.nullity == 0
    assert recovered.distance(transported) <= CERT_TOL
    assert np.allclose(recovered.particular, transported, atol=CERT_TOL)


def test_identification_residual_matches_probe_instance():
    instance = make_budget_instance(SEED)
    dim = instance.true_target.boundaries[0].shape[1]

    operator = probe_operator(SEED, dim, THRESHOLD)
    observations = operator @ instance.transported
    probes = Probes(operator, observations)
    for candidate in instance.candidates:
        residual = identification_residual(
            probes, candidate.boundaries[0], instance.transported
        )
        recovered = probe_instance(probes, candidate.boundaries[0])
        assert residual == recovered.distance(instance.transported)
    assert residual > 0.0  # last candidate is a decoy


def test_feasible_set_is_the_stacked_affine_solution_set():
    instance = make_budget_instance(SEED)
    dim = instance.true_target.boundaries[0].shape[1]

    operator = probe_operator(SEED, dim, 1)  # below threshold: nullity THRESHOLD - 1
    probes = Probes(operator, operator @ instance.transported)
    recovered = probe_instance(probes, instance.true_target.boundaries[0])
    assert recovered.nullity == THRESHOLD - 1
    # Every point particular + Q c is feasible (distance 0) and attains the
    # same stacked residual; the truth is one of them up to float64.
    rng = np.random.default_rng(0)
    coeffs = rng.standard_normal(recovered.nullity)
    point = recovered.particular + recovered.certificate.basis @ coeffs
    assert recovered.distance(point) <= CERT_TOL
    assert recovered.stacked_residual <= CERT_TOL


def test_fail_closed_validation():
    with pytest.raises(ValueError):
        Probes(np.zeros((2, 3)), np.zeros(3))  # count mismatch
    with pytest.raises(ValueError):
        Probes(np.zeros(3), np.zeros(3))  # operator not 2-D
    probes = Probes(np.zeros((1, 3)), np.zeros(1))
    with pytest.raises(ValueError):
        probe_instance(probes, np.zeros((2, 4)))  # constraint dimension mismatch
    with pytest.raises(ValueError):
        identifiability_threshold(
            ChainComplex((np.eye(2), np.eye(2)))
        )  # not a 1-complex
    instance = make_budget_instance(SEED)
    dim = instance.true_target.boundaries[0].shape[1]
    recovered = probe_instance(
        Probes(np.zeros((0, dim)), np.zeros(0)),
        instance.true_target.boundaries[0],
    )
    with pytest.raises(ValueError):
        recovered.distance(np.zeros(dim - 1))  # wrong ambient dimension
