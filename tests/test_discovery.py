import numpy as np
import pytest

from universa.discovery import (
    DiscoveredConstraint,
    DiscoveryInsufficient,
    admit_to_library,
    discover_constraint,
    discovery_quality,
    run_discovery,
    synthesize_observations,
)
from universa.generators import make_switch_instance, subseed
from universa.operators import CERT_TOL, nullspace_basis, projector

SEED = 2026
# make_switch_instance(SEED, 8, 14, 6, 3): the true quotient target has
# E_t = 11 edges and a cycle space (kernel) of dimension 6, and the planted
# f1 maps the source cycle space ONTO it, so enough transported cycles
# support full recovery. The stabilization holdout first accepts at M = 7
# (prefix of 6 observations spans the 6-dimensional subspace). Verified
# empirically and asserted below.
KERNEL_DIM = 6
AMBIENT_DIM = 11


def make_instance(seed=SEED):
    return make_switch_instance(seed, 8, 14, 6, 3)


def make_observations(count=16, seed=SEED):
    return synthesize_observations(make_instance(seed), count)


def test_exact_recovery_with_sufficient_observations():
    switch = make_instance()
    observations = make_observations()
    assert observations.shape == (AMBIENT_DIM, 16)
    result = discover_constraint(observations, AMBIENT_DIM, seeds=(SEED,))
    assert isinstance(result, DiscoveredConstraint)

    # The discovered constraint annihilates every observed vector to the
    # certificate tolerance (per-column, not just in aggregate).
    assert result.certificate_residual <= CERT_TOL
    per_column = np.linalg.norm(result.boundary @ observations, axis=0)
    assert float(per_column.max()) <= CERT_TOL

    # S_disc is contained in ker(B1_true) with a certified projector
    # residual, and covers it completely at the default sizes.
    proj = projector(nullspace_basis(switch.true_target.boundaries[0]))
    containment = float(
        np.linalg.norm(result.support_basis - proj @ result.support_basis)
    )
    assert containment <= CERT_TOL
    assert result.coverage == KERNEL_DIM
    quality = discovery_quality(result, switch.true_target.boundaries[0])
    assert quality.containment_residual <= CERT_TOL
    assert quality.coverage_fraction == 1.0
    assert quality.discovered_dim == KERNEL_DIM
    assert quality.kernel_dim == KERNEL_DIM

    # Provenance.
    assert result.seeds == (SEED,)
    assert result.num_observations == 16


def test_certificate_residual_is_the_truncation_energy():
    # The annihilator rows and support columns together form an orthonormal
    # basis of the ambient space, so by Pythagoras ||C_disc Y||_F^2 equals
    # the discarded-singular-value energy. The certificate is computed
    # independently of the SVD, so agreement to float64 is a real check.
    observations = make_observations()
    result = discover_constraint(observations, AMBIENT_DIM)
    assert isinstance(result, DiscoveredConstraint)
    singulars = np.linalg.svd(observations, compute_uv=False)
    discarded = float(np.linalg.norm(singulars[result.coverage:]))
    assert abs(result.certificate_residual - discarded) <= CERT_TOL


def test_insufficient_observations_refused_without_exception():
    switch = make_instance()
    ambient = switch.true_target.boundaries[0].shape[1]
    # M = 0..6 cannot support the 6-dimensional kernel: the observed
    # dimension is still growing when the final 25% is added.
    for count in range(7):
        observations = synthesize_observations(switch, count)
        result = discover_constraint(observations, ambient, seeds=(SEED,))
        assert isinstance(result, DiscoveryInsufficient)
        assert result.num_observations == count
        assert result.observed_dim == count  # ranks still growing
        assert result.seeds == (SEED,)
    # The documented rule first accepts at M = 7 (prefix of 6 stabilizes).
    result = discover_constraint(
        synthesize_observations(switch, 7), ambient
    )
    assert isinstance(result, DiscoveredConstraint)
    assert result.coverage == KERNEL_DIM


def test_zero_signal_and_trivial_span_refused():
    ambient = AMBIENT_DIM
    zeros = np.zeros((ambient, 16))
    result = discover_constraint(zeros, ambient)
    assert isinstance(result, DiscoveryInsufficient)
    assert result.observed_dim == 0
    # A full-rank span leaves a trivial annihilator: no constraint to
    # discover, so this is a refusal, not the "everything" structure.
    rng = np.random.default_rng(subseed(SEED, "full-rank-test"))
    full = rng.standard_normal((ambient, 16))
    result = discover_constraint(full, ambient)
    assert isinstance(result, DiscoveryInsufficient)
    assert result.observed_dim == ambient


def test_uncertifiable_tolerance_refused_not_admitted():
    # Below float64's certifiable floor the residual cannot pass: the
    # outcome is an explicit refusal with the residual as diagnostics.
    observations = make_observations()
    result = discover_constraint(observations, AMBIENT_DIM, tol=1e-16)
    assert isinstance(result, DiscoveryInsufficient)
    assert result.certificate_residual is not None
    assert result.certificate_residual > 1e-16
    assert "certificate residual" in result.reason


def test_novelty_gate_blocks_duplicate_and_admits_new():
    switch = make_instance()
    result = discover_constraint(make_observations(), AMBIENT_DIM)
    assert isinstance(result, DiscoveredConstraint)
    true_boundary = switch.true_target.boundaries[0]
    decoys = [d.boundaries[0] for d in switch.decoy_targets]

    # Duplicate: the true boundary (same kernel as S_disc) is in the library.
    blocked = admit_to_library(
        [true_boundary, *decoys], result, CERT_TOL, 1e-6
    )
    assert not blocked.admitted
    assert len(blocked.distances) == 4
    assert blocked.min_distance == blocked.distances[0]
    assert blocked.min_distance <= 1e-12  # same subspace: float64 noise only
    assert "duplicates" in blocked.reason

    # Novel: decoys only — a genuinely new consistent subspace is admitted.
    admitted = admit_to_library(decoys, result, CERT_TOL, 1e-6)
    assert admitted.admitted
    assert admitted.min_distance > 1e-6
    assert admitted.min_distance > 1.0  # O(1) distance, documented behavior

    # Certification gate: a forged weak residual is refused admission.
    forged = DiscoveredConstraint(
        result.boundary,
        result.support_basis,
        certificate_residual=1.0,
        num_observations=result.num_observations,
        seeds=result.seeds,
    )
    refused = admit_to_library(decoys, forged, CERT_TOL, 1e-6)
    assert not refused.admitted
    assert "certificate residual" in refused.reason
    # An empty library admits anything certified (nothing to duplicate).
    empty = admit_to_library([], result, CERT_TOL, 1e-6)
    assert empty.admitted
    assert empty.distances == ()
    assert empty.min_distance == float("inf")


def test_partial_coverage_is_reported_honestly():
    # Observations supported on a 2-dimensional subspace of ker(B1_true):
    # discovery must return exactly that subspace — certified, contained,
    # and honestly reported as partial coverage, never inflated.
    switch = make_instance()
    kernel_basis = nullspace_basis(switch.true_target.boundaries[0]).basis
    rng = np.random.default_rng(subseed(SEED, "partial-coverage-test"))
    columns = [
        kernel_basis[:, :2] @ rng.standard_normal(2) for _ in range(8)
    ]
    observations = np.column_stack(columns)
    result = discover_constraint(observations, AMBIENT_DIM)
    assert isinstance(result, DiscoveredConstraint)
    assert result.coverage == 2
    quality = discovery_quality(result, switch.true_target.boundaries[0])
    assert quality.containment_residual <= CERT_TOL
    assert quality.coverage_fraction == pytest.approx(2 / KERNEL_DIM)


def test_discovery_is_bit_deterministic():
    first = run_discovery(SEED)
    second = run_discovery(SEED)
    assert np.array_equal(first.observations, second.observations)
    result_a, result_b = first.result, second.result
    assert isinstance(result_a, DiscoveredConstraint)
    assert isinstance(result_b, DiscoveredConstraint)
    assert np.array_equal(result_a.boundary, result_b.boundary)
    assert np.array_equal(result_a.support_basis, result_b.support_basis)
    assert result_a.certificate_residual == result_b.certificate_residual
    assert first.admission == second.admission
    assert first.quality == second.quality
    assert first.map_misfit == second.map_misfit


def test_end_to_end_three_seeds():
    for seed in (2026, 7, 99):
        run = run_discovery(seed)
        # The true target is withheld: the library handed to admission is
        # decoys only.
        assert len(run.library_boundaries) == 3
        for boundary in run.library_boundaries:
            assert not np.array_equal(
                boundary, run.switch.true_target.boundaries[0]
            )
        # Discovery succeeds, certifies, and is admitted over the decoys.
        assert isinstance(run.result, DiscoveredConstraint)
        assert run.result.certificate_residual <= CERT_TOL
        assert run.admission is not None and run.admission.admitted
        # Against the withheld truth: exact containment, full coverage.
        assert run.quality is not None
        assert run.quality.containment_residual <= CERT_TOL
        assert run.quality.coverage_fraction == 1.0
        # The planted map's misfit against the discovered constraint: the
        # router would now accept the discovered structure.
        assert run.map_misfit is not None
        assert run.map_misfit <= 1e-9


def test_fail_closed_validation():
    observations = make_observations()
    with pytest.raises(ValueError):
        discover_constraint(observations, AMBIENT_DIM + 1)  # ambient mismatch
    with pytest.raises(ValueError):
        discover_constraint(observations.ravel(), AMBIENT_DIM)  # not 2-D
    with pytest.raises(ValueError):
        discover_constraint(observations, 0)  # bad ambient_dim
    nonfinite = observations.copy()
    nonfinite[0, 0] = np.inf
    with pytest.raises(ValueError):
        discover_constraint(nonfinite, AMBIENT_DIM)  # non-finite data
    with pytest.raises(ValueError):
        discover_constraint(observations, AMBIENT_DIM, tol=0.0)

    result = discover_constraint(observations, AMBIENT_DIM)
    assert isinstance(result, DiscoveredConstraint)
    insufficient = discover_constraint(observations[:, :1], AMBIENT_DIM)
    assert isinstance(insufficient, DiscoveryInsufficient)
    # An insufficient discovery is never admissible — a type error, not a
    # quiet "not admitted".
    with pytest.raises(ValueError):
        admit_to_library([], insufficient, CERT_TOL, 1e-6)
    with pytest.raises(ValueError):
        admit_to_library(
            [np.zeros((2, AMBIENT_DIM + 1))], result, CERT_TOL, 1e-6
        )
    with pytest.raises(ValueError):
        discovery_quality(result, np.zeros((2, AMBIENT_DIM + 1)))
    with pytest.raises(ValueError):
        discovery_quality(insufficient, np.eye(AMBIENT_DIM))
    # Bookkeeping validation on the result dataclass itself.
    with pytest.raises(ValueError):
        DiscoveredConstraint(
            result.boundary,
            result.support_basis[:-1],  # wrong ambient dimension
            result.certificate_residual,
            result.num_observations,
            (),
        )
    with pytest.raises(ValueError):
        DiscoveredConstraint(
            result.boundary,
            result.support_basis,
            -1.0,  # negative residual
            result.num_observations,
            (),
        )
    with pytest.raises(ValueError):
        synthesize_observations(make_instance(), -1)
