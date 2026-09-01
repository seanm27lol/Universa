from dataclasses import replace

import numpy as np
import pytest

from universa.discovery import (
    DiscoveredConstraint,
    discover_constraint,
    discovery_quality,
    run_discovery,
    synthesize_observations,
)
from universa.generators import (
    make_switch_instance,
    random_connected_graph,
    subseed,
)
from universa.loop import (
    ALARM_TOL,
    LibraryLoop,
    acquisition_correct,
    make_loop_instance,
    null_observations,
    routing_correct,
    run_loop,
)
from universa.operators import CERT_TOL
from universa.structures import ChainComplex, ChainMap

SEED = 70002
SEEDS = (70001, 70002, 70003, 70004, 70005)
# make_loop_instance(SEED): the true quotient target has a 6-edge-class
# quotient with a kernel the transported cycles cover fully (coverage
# fraction 1.0), so the novelty-blocking test below can rely on S_disc ==
# ker(B1_true). Verified empirically; SEED=70001 has partial (5/6) coverage
# and is used only where coverage does not matter.


def test_no_sealed_seeds():
    # The sealed eval blocks (and the reserved 70101 block) are off-limits.
    sealed_blocks = tuple(
        range(start, start + 100)
        for start in (20101, 30101, 40101, 60101, 70101, 80101, 90101)
    )
    for seed in SEEDS:
        for block in sealed_blocks:
            assert seed not in block


def test_frozen_exp4_constants():
    loop = LibraryLoop()
    assert ALARM_TOL == 1e-9  # re-pinned universa.router.RESIDUAL_TOL
    assert loop.alarm_tol == 1e-9
    assert loop.misfit_tol == CERT_TOL == 1e-10
    assert loop.novelty_tol == 1e-6
    assert loop.num_observations == 16


def test_make_loop_instance_paired_conditions():
    instance, in_library, out_library = make_loop_instance(SEED)
    # Two views of the SAME instance: the paired conditions.
    assert len(in_library) == 4 and len(out_library) == 3
    assert in_library[0] is instance.true_target  # truth at index 0
    assert all(c is d for c, d in zip(in_library[1:], instance.decoy_targets))
    assert all(c is d for c, d in zip(out_library, instance.decoy_targets))
    # The truth is withheld from the out-of-library view.
    true_boundary = instance.true_target.boundaries[0]
    for decoy in out_library:
        assert not np.array_equal(decoy.boundaries[0], true_boundary)
    # Deterministic construction.
    _, in_again, out_again = make_loop_instance(SEED)
    for first, second in zip(in_library, in_again):
        assert np.array_equal(first.boundaries[0], second.boundaries[0])
    for first, second in zip(out_library, out_again):
        assert np.array_equal(first.boundaries[0], second.boundaries[0])


def test_scores_are_exact_certified_residuals():
    for seed in SEEDS:
        instance, in_library, _ = make_loop_instance(seed)
        outcome = run_loop(LibraryLoop(), instance, in_library, seed=seed)
        # Exactly the make_switch_instance commutation_scores semantics:
        # bit-identical to a fresh probe run over the same instance.
        switch = make_switch_instance(seed, 8, 14, 6, 3)
        assert outcome.scores == switch.commutation_scores()
        # Exact zeros asserted, not tolerated; decoys clear the alarm.
        assert outcome.scores[0] == 0.0
        assert outcome.best_score == 0.0
        assert all(s > ALARM_TOL for s in outcome.scores[1:])
        # And the out-of-library scores are the decoy probe residuals.
        _, _, out_library = make_loop_instance(seed)
        out_outcome = run_loop(
            LibraryLoop(), instance, out_library, seed=seed
        )
        for candidate, score in zip(out_library, out_outcome.scores):
            probe = ChainMap(
                instance.source, candidate, instance.chain_map.maps
            )
            (residual,) = probe.commutation_residuals()
            assert score == residual
            assert score > ALARM_TOL


def test_route_mode_in_library():
    for seed in SEEDS:
        instance, in_library, _ = make_loop_instance(seed)
        outcome = run_loop(LibraryLoop(), instance, in_library, seed=seed)
        assert outcome.mode == "route"
        assert not outcome.alarm_fired
        assert outcome.routed_index == 0
        assert routing_correct(outcome)
        assert not acquisition_correct(outcome, instance)
        assert outcome.best_score == 0.0
        assert outcome.scores[outcome.routed_index] == outcome.best_score
        assert outcome.initial_library_size == len(in_library)
        assert outcome.final_library_size == len(in_library)
        assert outcome.seed == seed
        # No discovery phase happened.
        assert outcome.discovery_verdict is None
        assert outcome.discovery_reason is None
        assert outcome.certificate_residual is None
        assert outcome.map_misfit is None
        assert outcome.num_observations is None
        assert not outcome.admitted


def test_discovery_mode_out_of_library():
    loop = LibraryLoop()
    for seed in SEEDS:
        instance, _, out_library = make_loop_instance(seed)
        outcome = run_loop(loop, instance, out_library, seed=seed)
        assert outcome.mode == "discover"
        assert outcome.alarm_fired
        assert outcome.discovery_verdict == "discovered"
        assert outcome.certificate_residual is not None
        assert outcome.certificate_residual <= loop.misfit_tol
        assert outcome.admitted
        assert outcome.admission_min_distance is not None
        assert outcome.admission_min_distance > loop.novelty_tol
        assert outcome.admission_distances is not None
        assert len(outcome.admission_distances) == len(out_library)
        assert min(outcome.admission_distances) == (
            outcome.admission_min_distance
        )
        assert outcome.map_misfit is not None
        assert outcome.map_misfit <= ALARM_TOL
        assert acquisition_correct(outcome, instance)
        assert not routing_correct(outcome)
        # The library grew by one and the instance re-routed to the new
        # index of the FINAL library.
        assert outcome.final_library_size == len(out_library) + 1
        assert outcome.routed_index == len(out_library)
        assert outcome.num_observations == loop.num_observations
        # The loop reuses the certified head verbatim: bit-identical to
        # run_discovery's structured condition on the same seed.
        run = run_discovery(seed)
        assert isinstance(run.result, DiscoveredConstraint)
        assert outcome.certificate_residual == (
            run.result.certificate_residual
        )
        assert outcome.map_misfit == run.map_misfit
        assert outcome.admission_distances == run.admission.distances


def test_alarm_does_not_fire_in_library():
    # Precision: no false alarms when the library contains the truth.
    for seed in SEEDS:
        instance, in_library, _ = make_loop_instance(seed)
        outcome = run_loop(LibraryLoop(), instance, in_library, seed=seed)
        assert not outcome.alarm_fired


def test_alarm_fires_out_of_library():
    # Recall: the alarm always fires when the truth is withheld.
    for seed in SEEDS:
        instance, _, out_library = make_loop_instance(seed)
        outcome = run_loop(LibraryLoop(), instance, out_library, seed=seed)
        assert outcome.alarm_fired


def test_null_observations_refused():
    # The H4 false-admission control: structure-free observations must be
    # refused, never certified into a false constraint.
    loop = LibraryLoop()
    for seed in SEEDS:
        instance, _, out_library = make_loop_instance(seed)
        ambient = instance.true_target.boundaries[0].shape[1]
        nulls = null_observations(seed, ambient)
        outcome = run_loop(
            loop, instance, out_library, seed=seed, observations=nulls
        )
        assert outcome.mode == "refused"
        assert outcome.alarm_fired
        assert outcome.discovery_verdict == "insufficient"
        assert outcome.discovery_reason
        assert not outcome.admitted
        assert outcome.routed_index is None
        assert outcome.map_misfit is None
        assert outcome.admission_distances is None
        assert outcome.final_library_size == len(out_library)
        assert outcome.num_observations == loop.num_observations
        assert not acquisition_correct(outcome, instance)


def test_null_observations_schedule():
    # Bit-equality with the frozen exp4 H4 schedule.
    ambient = 11
    nulls = null_observations(SEED, ambient)
    assert nulls.shape == (ambient, 16)
    for j in range(16):
        expected = np.random.default_rng(
            subseed(SEED, "discovery-null", str(j))
        ).standard_normal(ambient)
        assert np.array_equal(nulls[:, j], expected)
    assert np.array_equal(nulls, null_observations(SEED, ambient))
    assert null_observations(SEED, ambient, count=3).shape == (ambient, 3)
    assert null_observations(SEED, ambient, count=0).shape == (ambient, 0)


def test_tie_breaking_first_index():
    instance, _, _ = make_loop_instance(SEED)
    library = (
        instance.true_target,
        instance.true_target,
        *instance.decoy_targets,
    )
    outcome = run_loop(LibraryLoop(), instance, library, seed=SEED)
    assert outcome.mode == "route"
    assert outcome.scores[0] == 0.0 == outcome.scores[1]
    assert outcome.routed_index == 0  # first index, not the duplicate


def test_loop_is_bit_deterministic():
    loop = LibraryLoop()
    for seed in SEEDS:
        instance, in_library, out_library = make_loop_instance(seed)
        first = run_loop(loop, instance, in_library, seed=seed)
        second = run_loop(loop, instance, in_library, seed=seed)
        assert first == second
        first = run_loop(loop, instance, out_library, seed=seed)
        second = run_loop(loop, instance, out_library, seed=seed)
        assert first == second
        ambient = instance.true_target.boundaries[0].shape[1]
        nulls = null_observations(seed, ambient)
        first = run_loop(
            loop, instance, out_library, seed=seed, observations=nulls
        )
        second = run_loop(
            loop, instance, out_library, seed=seed, observations=nulls
        )
        assert first == second


def test_novelty_gate_blocks_rediscovery():
    # A vertex-relabeled copy of the true boundary has the SAME kernel (the
    # budgets design finding) but still fails the planted map, so the alarm
    # fires and the novelty gate is what must block re-admission.
    instance, _, _ = make_loop_instance(SEED)
    true_boundary = instance.true_target.boundaries[0]
    ambient = int(true_boundary.shape[1])
    # Precondition at this frozen seed: the transported cycles support the
    # full target kernel, so S_disc == ker(B1_true) exactly and the
    # relabeled boundary sits at projector distance ~0 from the discovery.
    switch = make_switch_instance(SEED, 8, 14, 6, 3)
    discovered = discover_constraint(
        synthesize_observations(switch, 16), ambient
    )
    assert isinstance(discovered, DiscoveredConstraint)
    quality = discovery_quality(discovered, true_boundary)
    assert quality.coverage_fraction == 1.0

    rng = np.random.default_rng(subseed(SEED, "loop-relabeling"))
    permutation = rng.permutation(true_boundary.shape[0])
    assert not np.array_equal(
        permutation, np.arange(true_boundary.shape[0])
    )
    relabeled = ChainComplex((true_boundary[permutation, :],))
    library = (relabeled, *instance.decoy_targets)
    loop = LibraryLoop()
    outcome = run_loop(loop, instance, library, seed=SEED)
    assert outcome.scores[0] > ALARM_TOL  # the relabeling does not commute
    assert outcome.alarm_fired
    assert outcome.mode == "refused"
    assert outcome.discovery_verdict == "discovered"
    assert not outcome.admitted
    assert outcome.admission_min_distance is not None
    assert outcome.admission_min_distance <= loop.novelty_tol
    assert "duplicates" in outcome.admission_reason
    # The certified discovery itself was router-acceptable; only the
    # novelty gate kept it out.
    assert outcome.map_misfit is not None
    assert outcome.map_misfit <= ALARM_TOL
    assert outcome.routed_index is None
    assert outcome.final_library_size == len(library)
    assert not acquisition_correct(outcome, instance)


def test_fail_closed_loop_config():
    with pytest.raises(ValueError):
        LibraryLoop(alarm_tol=0.0)
    with pytest.raises(ValueError):
        LibraryLoop(alarm_tol=-1e-9)
    with pytest.raises(ValueError):
        LibraryLoop(alarm_tol=float("nan"))
    with pytest.raises(ValueError):
        LibraryLoop(alarm_tol=float("inf"))
    with pytest.raises(ValueError):
        LibraryLoop(misfit_tol=0.0)
    with pytest.raises(ValueError):
        LibraryLoop(novelty_tol=-1e-6)
    with pytest.raises(ValueError):
        LibraryLoop(num_observations=0)


def test_fail_closed_run_loop_inputs():
    loop = LibraryLoop()
    instance, in_library, out_library = make_loop_instance(SEED)
    with pytest.raises(ValueError):
        run_loop(loop, instance, (), seed=SEED)  # empty library
    with pytest.raises(ValueError):
        run_loop(loop, instance, in_library, seed=SEED + 1)  # provenance
    with pytest.raises(ValueError):
        run_loop(loop, instance, ("not-a-complex",), seed=SEED)
    # Mismatched instance/library dims: a graph on other counts does not
    # share this instance's target edge space.
    other = random_connected_graph(SEED, 7, 12)
    with pytest.raises(ValueError):
        run_loop(loop, instance, (other,), seed=SEED)
    ambient = instance.true_target.boundaries[0].shape[1]
    with pytest.raises(ValueError):
        run_loop(
            loop,
            instance,
            out_library,
            seed=SEED,
            observations=np.zeros((ambient + 1, 16)),  # ambient mismatch
        )
    with pytest.raises(ValueError):
        run_loop(
            loop,
            instance,
            out_library,
            seed=SEED,
            observations=np.zeros(ambient),  # not 2-D
        )
    nonfinite = null_observations(SEED, ambient)
    nonfinite[0, 0] = np.inf
    with pytest.raises(ValueError):
        run_loop(
            loop,
            instance,
            out_library,
            seed=SEED,
            observations=nonfinite,
        )


def test_fail_closed_outcome_coherence():
    loop = LibraryLoop()
    instance, in_library, out_library = make_loop_instance(SEED)
    route_outcome = run_loop(loop, instance, in_library, seed=SEED)
    with pytest.raises(ValueError):
        replace(route_outcome, alarm_fired=True)
    with pytest.raises(ValueError):
        replace(route_outcome, admitted=True)
    with pytest.raises(ValueError):
        replace(route_outcome, routed_index=len(in_library))  # out of range
    with pytest.raises(ValueError):
        replace(route_outcome, routed_index=1)  # not the argmin
    with pytest.raises(ValueError):
        replace(route_outcome, best_score=1.0)  # not min(scores)
    with pytest.raises(ValueError):
        replace(route_outcome, map_misfit=0.0)  # no discovery phase

    discover_outcome = run_loop(loop, instance, out_library, seed=SEED)
    with pytest.raises(ValueError):
        replace(discover_outcome, mode="route")  # alarm fired
    with pytest.raises(ValueError):
        replace(discover_outcome, routed_index=0)  # must be the appended
    with pytest.raises(ValueError):
        replace(discover_outcome, admitted=False)
    with pytest.raises(ValueError):
        replace(discover_outcome, final_library_size=len(out_library))
    with pytest.raises(ValueError):
        replace(discover_outcome, map_misfit=None)  # certified discovery
    with pytest.raises(ValueError):
        replace(discover_outcome, admission_min_distance=1.0)  # not the min

    other_instance, _, _ = make_loop_instance(SEEDS[2])
    assert other_instance.seed != discover_outcome.seed
    with pytest.raises(ValueError):
        acquisition_correct(discover_outcome, other_instance)


def test_fail_closed_null_observations():
    with pytest.raises(ValueError):
        null_observations(SEED, 0)
    with pytest.raises(ValueError):
        null_observations(SEED, 11, count=-1)
