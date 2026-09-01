import ast
import inspect
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only, before the torch import

import numpy as np
import pytest
import torch

import universa
import universa.loop_v2
import universa.partial
from universa.discovery import synthesize_observations
from universa.generators import SwitchInstance, subseed
from universa.loop import make_loop_instance, null_observations
from universa.loop_v2 import (
    ALARM_THRESHOLD,
    ARMS,
    CONDITIONS,
    GENERIC_FEATURE_DIM,
    GENERIC_FEATURE_NAMES,
    MAP_ACCEPT_TOL,
    ArmOutcome,
    GenericMLP,
    LearnedAlarm,
    alarm_decision,
    alarm_features,
    arch_candidate_features,
    arch_row_features,
    arm_arch_full,
    arm_discovery_only,
    arm_generic,
    arm_routing_only,
    generic_candidate_features,
    generic_decision,
    generic_row_features,
    operating_grid_point,
    router_argmax,
    router_gates,
    train_alarm,
    train_generic,
)
from universa.operators import CERT_TOL, nullspace_basis
from universa.partial import ObservationModel, observed_misfit
from universa.router import StructureRouter, train_router
from universa.router_v2 import (
    DEFAULT_MASK_FRACTION,
    NO_ANCHOR_GRID,
    build_no_anchor_dataset,
    no_anchor_candidate_features,
    no_anchor_feature_dim,
)
from universa.structures import ChainMap

SEED = 70002
SEEDS = (70001, 70002, 70003, 70004, 70005)
# Non-sealed training blocks for the learned models, disjoint from the
# fixture seeds; fixtures (loop rows) are 70001..70005 only.
TRAIN_SEEDS = (70501, 70502, 70503, 70504, 70505, 70506)
VAL_SEEDS = (70601, 70602, 70603)

NUM_DECOYS = 3  # the frozen loop instance family (8, 14, 6, 3)
K = NUM_DECOYS  # the equal-K paired views: (truth, d0, d1) vs (d0, d1, d2)
ROUTER_DATASET_DECOYS = K - 1  # the router trains on K-candidate rows
GRID_LEN = len(NO_ANCHOR_GRID)
ARCH_DIM = 2 * GRID_LEN - 1 + 3
NUM_OBSERVATIONS = 16  # the frozen exp4 observation count


def observation_seed(seed: int) -> int:
    """The row's shared observation draw: the router-v2 regime's canonical
    observation family, so the learned router's training rows and the loop
    rows come from one observation distribution on disjoint seeds (the v2
    protocol's train/eval split discipline)."""
    return subseed(seed, "router-v2-observe")


def loop_views(seed: int):
    """The equal-K paired views of one loop instance (module docstring):
    in-library = (true_target, *decoys[:-1]) with the truth at index 0,
    out-of-library = decoy_targets with the truth withheld."""
    instance = make_loop_instance(seed)[0]
    in_library = (instance.true_target, *instance.decoy_targets[:-1])
    out_library = instance.decoy_targets
    assert len(in_library) == len(out_library) == K
    return instance, in_library, out_library


def row_data(seed: int):
    """One full row: instance, both views, exact transported observations,
    and the structure-free null observations (the v1 H4 schedule)."""
    instance, in_library, out_library = loop_views(seed)
    switch_view = SwitchInstance(
        instance.seed,
        instance.source,
        instance.true_target,
        instance.chain_map,
        instance.decoy_targets,
    )
    observations = synthesize_observations(switch_view, NUM_OBSERVATIONS)
    ambient = int(instance.true_target.boundaries[0].shape[1])
    nulls = null_observations(seed, ambient)
    return instance, in_library, out_library, observations, nulls


@pytest.fixture(scope="module")
def router():
    """The profile router: universa.router machinery, reused unmodified."""
    train_data = build_no_anchor_dataset(
        TRAIN_SEEDS, (0.2, 0.3, 0.4, 0.5), num_decoys=ROUTER_DATASET_DECOYS
    )
    val_data = build_no_anchor_dataset(
        VAL_SEEDS, (0.6, 0.7, 0.8), num_decoys=ROUTER_DATASET_DECOYS
    )
    model, _ = train_router(
        train_data,
        val_data,
        epochs=150,
        lr=1e-3,
        seed=0,
        feature_dim=no_anchor_feature_dim(NO_ANCHOR_GRID),
    )
    return model


@pytest.fixture(scope="module")
def alarm_rows(router):
    """The alarm's training rows: in-library views labeled fit, the paired
    out-of-library views labeled no-fit, over the training seeds."""
    fit_rows, nofit_rows = [], []
    for seed in TRAIN_SEEDS:
        instance, in_library, out_library, _, _ = row_data(seed)
        block, raw = arch_row_features(instance, in_library, observation_seed(seed))
        fit_rows.append(alarm_features(router_gates(router, block), raw))
        block_o, raw_o = arch_row_features(
            instance, out_library, observation_seed(seed)
        )
        nofit_rows.append(alarm_features(router_gates(router, block_o), raw_o))
    return np.stack(fit_rows), np.stack(nofit_rows)


@pytest.fixture(scope="module")
def alarm(alarm_rows):
    model, _ = train_alarm(
        lambda: LearnedAlarm(K), alarm_rows[0], alarm_rows[1]
    )
    return model


@pytest.fixture(scope="module")
def generic_rows():
    """The generic model's training rows: in-library blocks labeled with
    the true candidate's index (0 by the unpermuted convention), the
    paired out-of-library blocks labeled K (no-fit)."""
    features, labels = [], []
    for seed in TRAIN_SEEDS:
        instance, in_library, out_library, observations, _ = row_data(seed)
        point = operating_grid_point(seed)
        features.append(
            generic_row_features(
                instance, in_library, observation_seed(seed), observations, point
            )
        )
        labels.append(0)
        features.append(
            generic_row_features(
                instance, out_library, observation_seed(seed), observations, point
            )
        )
        labels.append(K)
    return np.stack(features), np.asarray(labels, dtype=np.int64)


@pytest.fixture(scope="module")
def generic_model(generic_rows):
    torch.manual_seed(4244)
    model = GenericMLP(K)
    model, _ = train_generic(model, generic_rows[0], generic_rows[1])
    return model


def constant_generic_model(cls: int) -> GenericMLP:
    """A stub generic model whose decision is constantly ``cls`` (zero
    weights, one-hot head bias) — deterministic arm-semantics checks that
    do not depend on training quality."""
    torch.manual_seed(0)
    model = GenericMLP(K)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head.bias[cls] = 1.0
    return model


def test_no_sealed_seeds():
    # The sealed eval blocks (and reserved blocks) are off-limits.
    sealed_blocks = tuple(
        range(start, start + 100)
        for start in (20101, 30101, 40101, 60101, 70101, 80101, 90101)
    ) + (
        range(130001, 130101),
        range(140001, 140101),
        range(150001, 150201),
        range(160001, 160037),
    )
    used = set(SEEDS) | set(TRAIN_SEEDS) | set(VAL_SEEDS)
    assert used == set(range(70001, 70006)) | set(range(70501, 70507)) | set(
        range(70601, 70604)
    )
    for seed in used:
        for block in sealed_blocks:
            assert seed not in block


def test_paired_views_equal_k_truth_at_zero():
    for seed in SEEDS:
        instance, in_library, out_library = loop_views(seed)
        assert len(in_library) == len(out_library) == K
        assert in_library[0] is instance.true_target
        assert all(c is d for c, d in zip(in_library[1:], instance.decoy_targets))
        assert all(c is d for c, d in zip(out_library, instance.decoy_targets))
        true_boundary = instance.true_target.boundaries[0]
        for decoy in out_library:
            assert not np.array_equal(decoy.boundaries[0], true_boundary)


def test_arch_feature_layout_and_raw_retention():
    instance, in_library, _ = loop_views(SEED)
    seed = observation_seed(SEED)
    for candidate in in_library:
        features, raw = arch_candidate_features(
            instance.chain_map, candidate, seed
        )
        assert features.shape == (ARCH_DIM,) == (18,)
        assert features.dtype == np.float64
        assert np.isfinite(features).all()
        assert len(raw) == GRID_LEN
        # The raw profile is exactly the observed misfit per grid point...
        expected_raw = tuple(
            observed_misfit(
                instance.chain_map,
                ObservationModel(
                    candidate,
                    seed,
                    mask_fraction=DEFAULT_MASK_FRACTION,
                    corrupt_fraction=g,
                ).observe(),
            )
            for g in NO_ANCHOR_GRID
        )
        assert raw == expected_raw
        # ...the profile block is exactly its log1p...
        assert np.array_equal(features[:GRID_LEN], np.log1p(np.asarray(raw)))
        # ...then the slopes, then the masked structural dims.
        assert np.array_equal(
            features[GRID_LEN : 2 * GRID_LEN - 1],
            np.diff(features[:GRID_LEN]),
        )
        observed_boundary = (
            ObservationModel(candidate, seed, mask_fraction=DEFAULT_MASK_FRACTION)
            .observe()
            .complex.boundaries[0]
        )
        vertices, edges = observed_boundary.shape
        assert features[-3] == vertices
        assert features[-2] == edges
        assert edges < candidate.boundaries[0].shape[1]  # the mask removed edges
        assert features[-1] == int(
            nullspace_basis(observed_boundary).basis.shape[1]
        )
        # Bit-compatible with the router-v2 no-anchor builder at the same
        # observation seed, grid, and mask.
        reference = no_anchor_candidate_features(
            instance.chain_map, candidate, seed
        )
        assert np.array_equal(features, reference)


def test_arch_feature_determinism_and_finiteness_all_seeds():
    for seed in SEEDS:
        instance, in_library, _ = loop_views(seed)
        for candidate in in_library:
            first, raw_first = arch_candidate_features(
                instance.chain_map, candidate, observation_seed(seed)
            )
            second, raw_second = arch_candidate_features(
                instance.chain_map, candidate, observation_seed(seed)
            )
            assert np.array_equal(first, second)
            assert raw_first == raw_second
            assert np.isfinite(first).all()


def test_observation_draw_shared_across_candidates():
    # One draw per (row, grid point), shared across candidates: the mask
    # draw is fixed per seed, so every candidate misses the SAME columns
    # at every grid point, and corruption is nested per fraction.
    instance, in_library, _ = loop_views(SEED)
    seed = observation_seed(SEED)
    kept_by_candidate = []
    flipped_sets = []
    for g in NO_ANCHOR_GRID:
        flipped = []
        for candidate in in_library:
            observed = ObservationModel(
                candidate,
                seed,
                mask_fraction=DEFAULT_MASK_FRACTION,
                corrupt_fraction=g,
            ).observe()
            kept_by_candidate.append(observed.kept_edges)
            flipped.append(set(observed.flipped_entries))
        flipped_sets.append(flipped)
    assert len(set(kept_by_candidate)) == 1
    for earlier, later in zip(flipped_sets, flipped_sets[1:]):
        for first, second in zip(earlier, later):
            assert first <= second
    # The kept-edge feature column agrees across candidates of a row.
    block, _ = arch_row_features(instance, in_library, seed)
    assert len(set(block[:, -2])) == 1
    # A different observation seed draws a different degradation.
    other_block, other_raw = arch_row_features(
        instance, in_library, observation_seed(SEED) + 1
    )
    assert not np.array_equal(block, other_block)
    assert not np.array_equal(arch_row_features(instance, in_library, seed)[1], other_raw)


def test_generic_feature_layout_values_and_determinism():
    assert GENERIC_FEATURE_DIM == 18
    assert len(GENERIC_FEATURE_NAMES) == GENERIC_FEATURE_DIM
    instance, in_library, _, observations, _ = row_data(SEED)
    point = operating_grid_point(SEED)
    for candidate in in_library:
        features = generic_candidate_features(
            candidate, observations, observation_seed(SEED), (point,)
        )
        assert features.shape == (GENERIC_FEATURE_DIM,)
        assert features.dtype == np.float64
        assert np.isfinite(features).all()
        # Recompute every column from first principles.
        observed_boundary = (
            ObservationModel(
                candidate,
                observation_seed(SEED),
                mask_fraction=DEFAULT_MASK_FRACTION,
                corrupt_fraction=point,
            )
            .observe()
            .complex.boundaries[0]
        )
        vertices, edges = observed_boundary.shape
        assert features[0] == vertices
        assert features[1] == edges
        assert features[2] == observations.shape[1] == NUM_OBSERVATIONS
        y_sv = np.linalg.svd(observations, compute_uv=False)
        b_sv = np.linalg.svd(observed_boundary, compute_uv=False)
        assert np.allclose(features[3:7], np.log1p(y_sv[:4]), rtol=0, atol=0)
        assert np.allclose(features[7:11], np.log1p(b_sv[:4]), rtol=0, atol=0)

        def rank(sv, shape):
            tol = max(shape) * np.finfo(float).eps * sv[0]
            return int((sv > tol).sum())

        y_rank, b_rank = rank(y_sv, observations.shape), rank(b_sv, observed_boundary.shape)
        assert features[11] == np.log1p(y_sv[0] / y_sv[y_rank - 1])
        assert features[12] == np.log1p(b_sv[0] / b_sv[b_rank - 1])
        assert features[13] == np.log1p(np.linalg.norm(observations) ** 2)
        assert features[14] == np.log1p(np.linalg.norm(observed_boundary) ** 2)
        assert features[15] == y_rank
        assert features[16] == b_rank
        assert features[17] == edges - b_rank
        # Deterministic rebuild.
        again = generic_candidate_features(
            candidate, observations, observation_seed(SEED), (point,)
        )
        assert np.array_equal(features, again)


def test_generic_builder_avoids_misfit_machinery_by_construction():
    # Structural proof: the builder's CODE references no commutation
    # residual, no misfit evaluation, no profile builder, and no chain map
    # (AST-level: docstrings are not code).
    tree = ast.parse(inspect.getsource(generic_candidate_features))
    referenced = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } | {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    for forbidden in (
        "observed_misfit",
        "commutation_residuals",
        "arch_candidate_features",
        "no_anchor_candidate_features",
        "degraded_candidate_features",
        "chain_map",
        "ChainMap",
    ):
        assert forbidden not in referenced
    # Behavioral proof: sabotage the misfit machinery — the generic builder
    # is untouched (it never calls it), while the arch builder dies.
    instance, in_library, _, observations, _ = row_data(SEED)
    point = operating_grid_point(SEED)
    baseline = generic_candidate_features(
        in_library[0], observations, observation_seed(SEED), (point,)
    )

    def sabotage(*args, **kwargs):
        raise RuntimeError("misfit machinery called")

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(universa.loop_v2, "observed_misfit", sabotage)
        monkeypatch.setattr(universa.partial, "observed_misfit", sabotage)
        monkeypatch.setattr(ChainMap, "commutation_residuals", sabotage)
        under_sabotage = generic_candidate_features(
            in_library[0], observations, observation_seed(SEED), (point,)
        )
        assert np.array_equal(baseline, under_sabotage)
        with pytest.raises(RuntimeError):
            arch_candidate_features(
                instance.chain_map, in_library[0], observation_seed(SEED)
            )
    # Behavioral separation: a perturbed map that changes the commutation
    # residual moves the arch features; the generic features never see any
    # map, so they are bit-identical. Flip one planted entry of f1 in a
    # target-edge row that survives the mask, so the flip is observable.
    kept = ObservationModel(
        in_library[0], observation_seed(SEED), mask_fraction=DEFAULT_MASK_FRACTION
    ).observe().kept_edges
    f0, f1 = instance.chain_map.maps
    rows, cols = np.nonzero(f1)
    pick = next(i for i, row in enumerate(rows) if row in kept)
    f1_perturbed = f1.copy()
    f1_perturbed[rows[pick], cols[pick]] *= -1.0
    perturbed = ChainMap(instance.source, in_library[0], (f0, f1_perturbed))
    arch_base, raw_base = arch_candidate_features(
        instance.chain_map, in_library[0], observation_seed(SEED)
    )
    arch_moved, raw_moved = arch_candidate_features(
        perturbed, in_library[0], observation_seed(SEED)
    )
    assert not np.array_equal(arch_base, arch_moved)
    assert raw_base != raw_moved
    assert np.array_equal(
        baseline,
        generic_candidate_features(
            in_library[0], observations, observation_seed(SEED), (point,)
        ),
    )
    # The generic features DO read the exact observation matrix Y.
    _, _, _, _, nulls = row_data(SEED)
    with_nulls = generic_candidate_features(
        in_library[0], nulls, observation_seed(SEED), (point,)
    )
    assert not np.array_equal(baseline, with_nulls)


def test_operating_grid_point():
    points = [operating_grid_point(seed) for seed in SEEDS]
    assert all(p in NO_ANCHOR_GRID for p in points)
    assert len(set(points)) > 1  # rows spread across degradation levels
    assert operating_grid_point(SEED) == operating_grid_point(SEED)
    with pytest.raises(ValueError):
        operating_grid_point(SEED, (0.0, 0.2))
    with pytest.raises(ValueError):
        operating_grid_point(SEED, ())


def test_alarm_features_layout():
    gates = np.array([0.7, 0.2, 0.1])
    raw = np.array([[1.0, 2.0, 4.0], [3.0, 3.0, 3.0], [5.0, 1.0, 2.0]])
    features = alarm_features(gates, raw)
    assert features.shape == (K + 3,)
    assert np.array_equal(features[:K], gates)
    assert features[K] == pytest.approx(
        -float((gates * np.log(gates)).sum())
    )
    assert features[K + 1] == np.log1p(1.0)  # min raw profile value
    slopes = np.diff(np.log1p(raw), axis=1)
    assert features[K + 2] == np.log1p(np.abs(slopes).max())
    with pytest.raises(ValueError):
        alarm_features(np.array([0.5, 0.4]), np.ones((2, 3)))  # sum != 1
    with pytest.raises(ValueError):
        alarm_features(gates, -np.ones((3, 2)))  # negative misfits
    with pytest.raises(ValueError):
        alarm_features(gates, np.ones((2, 3)))  # K mismatch
    with pytest.raises(ValueError):
        alarm_features(np.array([1.0, np.nan, 0.0]), np.ones((3, 2)))


def test_router_fixture_routes_true_candidate(router):
    # Pinned fixture behavior: the trained router's hard argmax is the true
    # candidate (index 0) on every in-library fixture row.
    for seed in SEEDS:
        instance, in_library, _ = loop_views(seed)
        block, _ = arch_row_features(instance, in_library, observation_seed(seed))
        assert router_argmax(router, block) == 0
        gates = router_gates(router, block)
        assert gates.shape == (K,)
        assert gates.sum() == pytest.approx(1.0, abs=1e-5)
        assert int(gates.argmax()) == 0


def test_arch_arm_in_library_routes_to_zero(router, alarm):
    for seed in SEEDS:
        instance, in_library, _, observations, _ = row_data(seed)
        outcome = arm_arch_full(
            seed,
            instance,
            in_library,
            router=router,
            alarm=alarm,
            observation_seed=observation_seed(seed),
            observations=observations,
            condition="in_library",
        )
        assert outcome.arm == "arch_full"
        assert outcome.condition == "in_library"
        assert outcome.action == "route"
        assert outcome.routed_index == 0
        assert outcome.correct
        assert outcome.discovery_invocations == 0  # the alarm stayed quiet
        assert not outcome.admitted
        assert outcome.map_misfit is None
        assert outcome.initial_library_size == K == outcome.final_library_size
        assert outcome.detail.startswith("alarm=fit")
        assert outcome.seed == seed


def test_arch_arm_out_of_library_discovers_and_acquires(router, alarm):
    # Pinned fixture behavior: the learned alarm classifies no-fit on
    # every out-of-library fixture row, and the certified discovery path
    # then acquires on the exact observations.
    for seed in SEEDS:
        instance, _, out_library, observations, _ = row_data(seed)
        outcome = arm_arch_full(
            seed,
            instance,
            out_library,
            router=router,
            alarm=alarm,
            observation_seed=observation_seed(seed),
            observations=observations,
            condition="out_of_library",
        )
        assert outcome.action == "discover"
        assert outcome.routed_index == K  # the appended structure's index
        assert outcome.correct
        assert outcome.discovery_invocations == 1
        assert outcome.admitted
        assert outcome.map_misfit is not None
        assert outcome.map_misfit <= MAP_ACCEPT_TOL
        assert outcome.final_library_size == K + 1
        assert "no-fit" in outcome.detail


def test_arch_arm_null_control_admits_nothing(router, alarm):
    for seed in SEEDS:
        instance, _, out_library, _, nulls = row_data(seed)
        outcome = arm_arch_full(
            seed,
            instance,
            out_library,
            router=router,
            alarm=alarm,
            observation_seed=observation_seed(seed),
            observations=nulls,
            condition="null_control",
        )
        assert outcome.action == "refused"
        assert outcome.routed_index is None
        assert not outcome.admitted
        assert outcome.correct  # the false-admission control
        assert outcome.discovery_invocations == 1
        assert outcome.final_library_size == K


def test_routing_only_forced_choice(router):
    for seed in SEEDS:
        instance, in_library, out_library, _, _ = row_data(seed)
        in_outcome = arm_routing_only(
            seed,
            instance,
            in_library,
            router=router,
            observation_seed=observation_seed(seed),
            condition="in_library",
        )
        assert in_outcome.action == "route"
        assert in_outcome.routed_index == 0
        assert in_outcome.correct
        assert in_outcome.discovery_invocations == 0
        out_outcome = arm_routing_only(
            seed,
            instance,
            out_library,
            router=router,
            observation_seed=observation_seed(seed),
            condition="out_of_library",
        )
        # No alarm, no discovery: a forced choice, so on out-of-library it
        # must pick a decoy — an honest failure.
        assert out_outcome.action == "route"
        assert 0 <= out_outcome.routed_index < K
        assert not out_outcome.correct
        assert not out_outcome.admitted
        assert out_outcome.discovery_invocations == 0
        null_outcome = arm_routing_only(
            seed,
            instance,
            out_library,
            router=router,
            observation_seed=observation_seed(seed),
            condition="null_control",
        )
        assert null_outcome.correct  # nothing admitted (no admission channel)
    # The v1 full-candidates view remains a valid in-library view (truth
    # at index 0 followed by the full decoy prefix).
    instance = make_loop_instance(SEED)[0]
    full_view = arm_routing_only(
        SEED,
        instance,
        instance.candidates,
        router=router,
        observation_seed=observation_seed(SEED),
        condition="in_library",
    )
    assert full_view.action == "route" and full_view.routed_index == 0


def test_discovery_only_in_library_and_null_refusal():
    for seed in SEEDS:
        instance, _, _, observations, nulls = row_data(seed)
        for condition in ("in_library", "out_of_library"):
            outcome = arm_discovery_only(
                seed, instance, observations=observations, condition=condition
            )
            assert outcome.action == "discover"
            assert outcome.correct
            assert outcome.admitted
            assert outcome.discovery_invocations == 1
            assert outcome.map_misfit is not None
            assert outcome.map_misfit <= MAP_ACCEPT_TOL
            # Novelty is checked against the decoy library; the appended
            # structure sits at index len(decoys).
            assert outcome.initial_library_size == NUM_DECOYS
            assert outcome.routed_index == NUM_DECOYS
            assert outcome.final_library_size == NUM_DECOYS + 1
        null_outcome = arm_discovery_only(
            seed, instance, observations=nulls, condition="null_control"
        )
        assert null_outcome.action == "refused"
        assert null_outcome.correct  # refusal is the specificity
        assert not null_outcome.admitted
        assert null_outcome.routed_index is None
        assert null_outcome.map_misfit is None
        assert null_outcome.final_library_size == NUM_DECOYS


def test_generic_arm_refusal_semantics_with_stub_models():
    route_stub = constant_generic_model(0)
    nofit_stub = constant_generic_model(K)
    assert generic_decision(route_stub, np.ones((K, GENERIC_FEATURE_DIM))) == 0
    assert generic_decision(nofit_stub, np.ones((K, GENERIC_FEATURE_DIM))) == K
    for seed in SEEDS:
        instance, in_library, out_library, observations, nulls = row_data(seed)
        routed = arm_generic(
            seed,
            instance,
            in_library,
            generic_model=route_stub,
            observation_seed=observation_seed(seed),
            observations_y=observations,
            condition="in_library",
        )
        assert routed.action == "route" and routed.routed_index == 0
        assert routed.correct  # the index-0 certified gate, as arch route
        assert routed.discovery_invocations == 0 and not routed.admitted
        forced = arm_generic(
            seed,
            instance,
            out_library,
            generic_model=route_stub,
            observation_seed=observation_seed(seed),
            observations_y=observations,
            condition="out_of_library",
        )
        assert forced.action == "route" and not forced.correct
        noise_routed = arm_generic(
            seed,
            instance,
            out_library,
            generic_model=route_stub,
            observation_seed=observation_seed(seed),
            observations_y=nulls,
            condition="null_control",
        )
        assert noise_routed.action == "route"
        assert not noise_routed.correct  # refusal is the only specificity
        refused = arm_generic(
            seed,
            instance,
            out_library,
            generic_model=nofit_stub,
            observation_seed=observation_seed(seed),
            observations_y=nulls,
            condition="null_control",
        )
        assert refused.action == "refused" and refused.correct
        assert refused.routed_index is None and not refused.admitted
        assert "no-fit" in refused.detail
        nofit_out = arm_generic(
            seed,
            instance,
            out_library,
            generic_model=nofit_stub,
            observation_seed=observation_seed(seed),
            observations_y=observations,
            condition="out_of_library",
        )
        # Nothing is ever synthesized by this arm: out-of-library it
        # cannot acquire, even when it correctly says no-fit.
        assert nofit_out.action == "refused" and not nofit_out.correct


def test_generic_arm_real_model_smoke(generic_model, generic_rows):
    # Above-chance classification after the tiny training run (chance is
    # 1/(K+1)); the smoke assertion is on the training rows, exactly the
    # router suite's convention.
    features, labels = generic_rows
    predictions = [
        generic_decision(generic_model, block) for block in features
    ]
    accuracy = float(np.mean([p == l for p, l in zip(predictions, labels)]))
    assert accuracy > 1.0 / (K + 1) + 0.4
    # On fixture rows the outcomes are coherent records whatever the
    # (deliberately weak, no-architecture) model decides.
    for seed in SEEDS:
        instance, in_library, _, observations, _ = row_data(seed)
        outcome = arm_generic(
            seed,
            instance,
            in_library,
            generic_model=generic_model,
            observation_seed=observation_seed(seed),
            observations_y=observations,
            condition="in_library",
        )
        assert isinstance(outcome, ArmOutcome)
        if outcome.action == "route":
            assert 0 <= outcome.routed_index < K
            assert outcome.correct == (outcome.routed_index == 0)
        else:
            assert not outcome.correct


def test_alarm_training_determinism(alarm_rows):
    model_a, history_a = train_alarm(
        lambda: LearnedAlarm(K), alarm_rows[0], alarm_rows[1]
    )
    model_b, history_b = train_alarm(
        lambda: LearnedAlarm(K), alarm_rows[0], alarm_rows[1]
    )
    assert history_a == history_b
    for value_a, value_b in zip(
        model_a.state_dict().values(), model_b.state_dict().values()
    ):
        assert torch.equal(value_a, value_b)
    model_c, _ = train_alarm(
        lambda: LearnedAlarm(K), alarm_rows[0], alarm_rows[1], seed=4244
    )
    assert any(
        not torch.equal(value_a, value_c)
        for value_a, value_c in zip(
            model_a.state_dict().values(), model_c.state_dict().values()
        )
    )


def test_alarm_separates_fit_nofit_on_fixture_rows(router, alarm_rows, alarm):
    _, history = train_alarm(
        lambda: LearnedAlarm(K), alarm_rows[0], alarm_rows[1]
    )
    assert history["train_accuracy"][-1] >= 0.9  # above chance 0.5
    # Pinned fixture behavior: fit on every in-library row, no-fit on
    # every out-of-library row (10/10 after the tiny training run).
    decisions = []
    for seed in SEEDS:
        instance, in_library, out_library, _, _ = row_data(seed)
        block, raw = arch_row_features(instance, in_library, observation_seed(seed))
        block_o, raw_o = arch_row_features(
            instance, out_library, observation_seed(seed)
        )
        decisions.append(alarm_decision(alarm, router_gates(router, block), raw))
        decisions.append(
            alarm_decision(alarm, router_gates(router, block_o), raw_o)
        )
    assert decisions == [True, False] * len(SEEDS)
    assert ALARM_THRESHOLD == 0.5


def test_generic_training_determinism(generic_rows):
    torch.manual_seed(4244)
    model_a, history_a = train_generic(
        GenericMLP(K), generic_rows[0], generic_rows[1]
    )
    torch.manual_seed(4244)
    model_b, history_b = train_generic(
        GenericMLP(K), generic_rows[0], generic_rows[1]
    )
    assert history_a == history_b
    for value_a, value_b in zip(
        model_a.state_dict().values(), model_b.state_dict().values()
    ):
        assert torch.equal(value_a, value_b)


def test_arms_end_to_end_determinism(router, alarm, generic_model):
    for seed in (70001, 70002):
        instance, in_library, out_library, observations, nulls = row_data(seed)

        def all_arms():
            return (
                arm_arch_full(
                    seed, instance, in_library,
                    router=router, alarm=alarm,
                    observation_seed=observation_seed(seed),
                    observations=observations, condition="in_library",
                ),
                arm_arch_full(
                    seed, instance, out_library,
                    router=router, alarm=alarm,
                    observation_seed=observation_seed(seed),
                    observations=observations, condition="out_of_library",
                ),
                arm_arch_full(
                    seed, instance, out_library,
                    router=router, alarm=alarm,
                    observation_seed=observation_seed(seed),
                    observations=nulls, condition="null_control",
                ),
                arm_routing_only(
                    seed, instance, out_library, router=router,
                    observation_seed=observation_seed(seed),
                    condition="out_of_library",
                ),
                arm_discovery_only(
                    seed, instance, observations=observations,
                    condition="in_library",
                ),
                arm_discovery_only(
                    seed, instance, observations=nulls,
                    condition="null_control",
                ),
                arm_generic(
                    seed, instance, in_library, generic_model=generic_model,
                    observation_seed=observation_seed(seed),
                    observations_y=observations, condition="in_library",
                ),
                arm_generic(
                    seed, instance, out_library, generic_model=generic_model,
                    observation_seed=observation_seed(seed),
                    observations_y=nulls, condition="null_control",
                ),
            )

        first = all_arms()
        second = all_arms()
        assert first == second  # pure scalar records: plain equality


def test_fail_closed_feature_builder_validation():
    instance, in_library, _, observations, _ = row_data(SEED)
    seed = observation_seed(SEED)
    with pytest.raises(ValueError):
        arch_candidate_features(
            instance.chain_map, in_library[0], seed, profile_grid=(0.0, 0.2)
        )
    with pytest.raises(ValueError):
        arch_candidate_features(
            instance.chain_map, in_library[0], seed, profile_grid=()
        )
    with pytest.raises(ValueError):
        arch_candidate_features(
            instance.chain_map, in_library[0], seed, profile_grid=(0.3, 0.2)
        )
    with pytest.raises(ValueError):
        arch_candidate_features(
            instance.chain_map, in_library[0], seed, mask_fraction=0.0
        )
    with pytest.raises(ValueError):
        arch_candidate_features("not a chain map", in_library[0], seed)
    with pytest.raises(ValueError):
        arch_candidate_features(instance.chain_map, "not a complex", seed)
    with pytest.raises(ValueError):
        generic_candidate_features(
            in_library[0], observations, seed, (0.2, 0.3)  # not one point
        )
    with pytest.raises(ValueError):
        generic_candidate_features(
            in_library[0], observations, seed, (0.0,)  # anchored
        )
    with pytest.raises(ValueError):
        generic_candidate_features(
            in_library[0], observations, seed, (operating_grid_point(SEED),),
            mask_fraction=1.5,
        )
    with pytest.raises(ValueError):
        generic_candidate_features(
            in_library[0], np.zeros(3), seed, (0.2,)  # Y not 2-D
        )
    bad_y = np.full((observations.shape[0], 4), np.inf)
    with pytest.raises(ValueError):
        generic_candidate_features(in_library[0], bad_y, seed, (0.2,))
    with pytest.raises(ValueError):
        generic_candidate_features(
            in_library[0],
            np.zeros((observations.shape[0] + 1, 4)),
            seed,
            (0.2,),  # ambient mismatch
        )


def test_fail_closed_learned_model_validation(alarm_rows, generic_rows):
    with pytest.raises(ValueError):
        LearnedAlarm(0)
    with pytest.raises(ValueError):
        GenericMLP(0)
    with pytest.raises(ValueError):
        train_alarm(lambda: LearnedAlarm(K), alarm_rows[0], alarm_rows[1][:, :2])
    with pytest.raises(ValueError):
        train_alarm(lambda: LearnedAlarm(K), alarm_rows[0][:0], alarm_rows[1])
    with pytest.raises(ValueError):
        train_alarm(lambda: "not an alarm", alarm_rows[0], alarm_rows[1])
    with pytest.raises(ValueError):
        train_alarm(
            lambda: LearnedAlarm(K), alarm_rows[0], alarm_rows[1], epochs=0
        )
    with pytest.raises(ValueError):
        train_alarm(lambda: LearnedAlarm(K), alarm_rows[0], alarm_rows[1], lr=0.0)
    with pytest.raises(ValueError):
        train_generic(GenericMLP(K), generic_rows[0][:, :, :2], generic_rows[1])
    with pytest.raises(ValueError):
        train_generic(
            GenericMLP(K),
            generic_rows[0],
            np.full(len(generic_rows[1]), K + 1),  # out of range
        )
    with pytest.raises(ValueError):
        train_generic(GenericMLP(K), generic_rows[0], generic_rows[1], epochs=0)
    model = LearnedAlarm(K)
    with pytest.raises(ValueError):
        alarm_decision(model, np.array([1.0, 0.0]), np.ones((2, GRID_LEN)))
    with pytest.raises(ValueError):
        generic_decision(GenericMLP(K), np.ones((K + 1, GENERIC_FEATURE_DIM)))
    with pytest.raises(ValueError):
        router_gates(StructureRouter(feature_dim=8), np.ones((K, 18)))


def test_fail_closed_arm_validation(router, alarm, generic_model):
    instance, in_library, out_library, observations, nulls = row_data(SEED)
    ambient = int(instance.true_target.boundaries[0].shape[1])
    seed = observation_seed(SEED)
    with pytest.raises(ValueError):
        arm_arch_full(  # provenance: seed must equal instance.seed
            SEED + 1, instance, in_library, router=router, alarm=alarm,
            observation_seed=seed, observations=observations,
            condition="in_library",
        )
    with pytest.raises(ValueError):
        arm_arch_full(  # view/condition mismatch
            SEED, instance, out_library, router=router, alarm=alarm,
            observation_seed=seed, observations=observations,
            condition="in_library",
        )
    with pytest.raises(ValueError):
        arm_arch_full(
            SEED, instance, in_library, router=router, alarm=alarm,
            observation_seed=seed, observations=observations,
            condition="out_of_library",
        )
    with pytest.raises(ValueError):
        arm_arch_full(
            SEED, instance, (), router=router, alarm=alarm,
            observation_seed=seed, observations=observations,
            condition="in_library",
        )
    with pytest.raises(ValueError):
        arm_arch_full(
            SEED, instance, in_library, router=router, alarm=alarm,
            observation_seed=seed, observations=observations,
            condition="not a condition",
        )
    with pytest.raises(ValueError):
        arm_arch_full(  # alarm K mismatch
            SEED, instance, in_library, router=router, alarm=LearnedAlarm(K + 1),
            observation_seed=seed, observations=observations,
            condition="in_library",
        )
    with pytest.raises(ValueError):
        arm_arch_full(  # router feature-dim mismatch
            SEED, instance, in_library,
            router=StructureRouter(feature_dim=8), alarm=alarm,
            observation_seed=seed, observations=observations,
            condition="in_library",
        )
    with pytest.raises(ValueError):
        arm_arch_full(  # ambient mismatch
            SEED, instance, out_library, router=router, alarm=alarm,
            observation_seed=seed,
            observations=np.zeros((ambient + 1, 4)), condition="out_of_library",
        )
    with pytest.raises(ValueError):
        arm_arch_full(  # non-finite observations
            SEED, instance, out_library, router=router, alarm=alarm,
            observation_seed=seed,
            observations=np.full((ambient, 4), np.nan),
            condition="out_of_library",
        )
    with pytest.raises(ValueError):
        arm_arch_full(
            SEED, instance, out_library, router="not a router", alarm=alarm,
            observation_seed=seed, observations=observations,
            condition="out_of_library",
        )
    with pytest.raises(ValueError):
        arm_routing_only(
            SEED, instance, out_library, router=router,
            observation_seed=seed, condition="in_library",  # wrong view
        )
    with pytest.raises(ValueError):
        arm_discovery_only(SEED + 1, instance, observations=observations,
                           condition="in_library")
    with pytest.raises(ValueError):
        arm_discovery_only(SEED, instance, observations=observations,
                           novelty_tol=0.0, condition="in_library")
    with pytest.raises(ValueError):
        arm_discovery_only(SEED, instance, observations=np.zeros(ambient),
                           condition="in_library")
    with pytest.raises(ValueError):
        arm_generic(
            SEED, instance, in_library, generic_model=GenericMLP(K + 1),
            observation_seed=seed, observations_y=observations,
            condition="in_library",
        )
    with pytest.raises(ValueError):
        arm_generic(
            SEED, instance, in_library, generic_model="not a model",
            observation_seed=seed, observations_y=observations,
            condition="in_library",
        )
    with pytest.raises(ValueError):
        arm_generic(
            SEED, instance, in_library, generic_model=generic_model,
            observation_seed=seed,
            observations_y=np.zeros((ambient + 1, 4)), condition="in_library",
        )


def test_fail_closed_outcome_coherence(router):
    instance, in_library, _, observations, _ = row_data(SEED)
    route_outcome = arm_routing_only(
        SEED, instance, in_library,
        router=router,
        observation_seed=observation_seed(SEED), condition="in_library",
    )
    with pytest.raises(ValueError):
        replace(route_outcome, admitted=True)
    with pytest.raises(ValueError):
        replace(route_outcome, routed_index=None)
    with pytest.raises(ValueError):
        replace(route_outcome, routed_index=K)  # out of the library
    with pytest.raises(ValueError):
        replace(route_outcome, map_misfit=0.0)  # route has no discovery
    with pytest.raises(ValueError):
        replace(route_outcome, discovery_invocations=1)
    with pytest.raises(ValueError):
        replace(route_outcome, action="not an action")
    with pytest.raises(ValueError):
        replace(route_outcome, condition="not a condition")
    with pytest.raises(ValueError):
        replace(route_outcome, arm="not an arm")
    with pytest.raises(ValueError):
        replace(route_outcome, detail="")
    discover_outcome = arm_discovery_only(
        SEED, instance, observations=observations, condition="in_library"
    )
    with pytest.raises(ValueError):
        replace(discover_outcome, admitted=False)
    with pytest.raises(ValueError):
        replace(discover_outcome, routed_index=0)  # must be the appended
    with pytest.raises(ValueError):
        replace(discover_outcome, final_library_size=NUM_DECOYS)
    with pytest.raises(ValueError):
        replace(discover_outcome, map_misfit=None)
    with pytest.raises(ValueError):
        replace(discover_outcome, map_misfit=-1.0)
    with pytest.raises(ValueError):
        replace(discover_outcome, action="refused")  # admitted incoherent


def test_module_constants():
    assert MAP_ACCEPT_TOL == 1e-9
    assert ALARM_THRESHOLD == 0.5
    assert CONDITIONS == ("in_library", "out_of_library", "null_control")
    assert ARMS == ("arch_full", "routing_only", "discovery_only", "generic")
    assert CERT_TOL == 1e-10


def test_plain_import_of_universa_does_not_import_loop_v2_or_torch():
    # Run in a fresh interpreter: this test module itself imports torch.
    src = Path(universa.__file__).resolve().parent.parent
    code = (
        "import sys\n"
        "import universa\n"
        "bad = [m for m in sys.modules if m == 'torch' or m.startswith('torch.')]\n"
        "bad += [m for m in ('universa.router', 'universa.router_v2', 'universa.loop_v2') if m in sys.modules]\n"
        "sys.exit(1 if bad else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(src)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
