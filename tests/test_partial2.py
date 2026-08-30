import numpy as np
import pytest

from universa.complexes2 import (
    commutation_scores,
    make_two_complex_switch_instance,
)
from universa.generators import subseed
from universa.partial2 import (
    ObservedTwoComplex,
    TwoComplexObservationModel,
    breakdown_fraction,
    observed_commutation_residuals,
    observed_d2_residue,
    ranking_study_two,
)
from universa.structures import ChainComplex, ChainMap

STUDY_SEED = 2026
# Documented in universa.partial2's docstring: on the default sizes and
# the default 0.1 grid, the true candidate keeps ranking first strictly
# below these fractions and stops ranking first at them (a tie at 0.4
# under corruption, a full tie at 1.0 under masking).
CORRUPT_BREAKDOWN = 0.4
MASK_BREAKDOWN = 1.0


def _observed_true(fraction, mode="corrupt", seed=STUDY_SEED):
    """The study's shared observation of the true target at one fraction."""
    instance = make_two_complex_switch_instance(seed)
    observation_seed = subseed(seed, "partial2", mode)
    kwargs = {
        "mask_fraction": 0.0,
        "corrupt_b1_fraction": 0.0,
        "corrupt_b2_fraction": 0.0,
    }
    if mode == "mask":
        kwargs["mask_fraction"] = fraction
    else:
        kwargs["corrupt_b1_fraction"] = fraction
        kwargs["corrupt_b2_fraction"] = fraction
    model = TwoComplexObservationModel(
        instance.true_target, observation_seed, **kwargs
    )
    return instance, model.observe()


def test_full_observation_reproduces_exact_zero_residuals_at_both_degrees():
    instance = make_two_complex_switch_instance(STUDY_SEED)
    observed = TwoComplexObservationModel(instance.true_target, seed=123).observe()
    truth_b1, truth_b2 = instance.true_target.boundaries
    assert np.array_equal(observed.b1, truth_b1)
    assert np.array_equal(observed.b2, truth_b2)
    num_edges = instance.true_target.dims[1]
    assert observed.kept_edges == tuple(range(num_edges))
    assert observed.flipped_b1 == ()
    assert observed.flipped_b2 == ()
    residuals = observed_commutation_residuals(instance.chain_map, observed)
    assert residuals == (0.0, 0.0)  # exact, not numerical residue
    assert residuals == instance.chain_map.commutation_residuals(tol=0.0)
    assert observed_d2_residue(observed) == 0.0


def test_study_at_zero_degradation_reproduces_commutation_scores():
    instance = make_two_complex_switch_instance(STUDY_SEED)
    first = ranking_study_two(STUDY_SEED, mode="corrupt")[0]
    scores = commutation_scores(instance)
    assert first.fraction == 0.0
    assert first.true_misfit == scores[0] == 0.0
    assert first.decoy_misfits == scores[1:]
    assert first.floor == min(scores[1:])


def test_corruption_degree1_residual_degrades_monotonically_fixed_seed():
    instance = make_two_complex_switch_instance(STUDY_SEED)
    observation_seed = subseed(STUDY_SEED, "partial2", "corrupt")
    deg1, deg2 = [], []
    for i in range(11):
        observed = TwoComplexObservationModel(
            instance.true_target,
            observation_seed,
            corrupt_b1_fraction=i / 10,
            corrupt_b2_fraction=i / 10,
        ).observe()
        residuals = observed_commutation_residuals(
            instance.chain_map, observed
        )
        deg1.append(residuals[0])
        deg2.append(residuals[1])
    assert deg1[0] == 0.0
    # The quotient map sends each source edge to exactly one target edge
    # row, so flipped B1 entries add Frobenius-orthogonal terms and the
    # degree-1 residual is strictly increasing whenever the flip count
    # grows — which the 0.1 grid guarantees on this seed (12 nonzeros).
    assert all(b > a for a, b in zip(deg1, deg1[1:]))
    # Degree 2 carries no such guarantee: f2-row cross terms inside one
    # observed edge row can cancel, and on this seed they do.
    assert not all(b >= a for a, b in zip(deg2, deg2[1:]))
    assert all(value >= 0.0 for value in deg2)


def test_d2_violation_under_corruption_while_decoys_stay_valid():
    instance, observed = _observed_true(0.1)
    # One flipped entry in each boundary already breaks d^2 = 0 ...
    assert observed_d2_residue(observed) > 0.0
    # ... to the point that the fail-closed ChainComplex constructor
    # refuses the observed pair outright ...
    with pytest.raises(ValueError, match=r"d\^2"):
        ChainComplex((observed.b1, observed.b2))
    # ... while every decoy remains a perfectly valid complex: a naive
    # validity check would anti-rank the truth.
    for decoy in instance.decoy_targets:
        product = decoy.boundaries[0] @ decoy.boundaries[1]
        assert not np.any(product != 0.0)
    # The residue is positive at every intermediate probed level on this
    # seed, and returns to exact zero only at the trivial endpoints.
    observation_seed = subseed(STUDY_SEED, "partial2", "corrupt")
    residues = [
        observed_d2_residue(
            TwoComplexObservationModel(
                instance.true_target,
                observation_seed,
                corrupt_b1_fraction=i / 10,
                corrupt_b2_fraction=i / 10,
            ).observe()
        )
        for i in range(11)
    ]
    assert residues[0] == 0.0
    assert all(r > 0.0 for r in residues[1:-1])
    # Full dual corruption negates both boundaries: (-B1)(-B2) = B1 B2.
    assert residues[-1] == 0.0


def test_observed_two_complex_skips_d2_validation_by_design():
    instance, observed = _observed_true(0.1)
    assert observed_d2_residue(observed) > 0.0
    # Reconstructing the same purported structure directly — valid
    # bookkeeping, invalid d^2 — must succeed: shapes and provenance are
    # all this dataclass gates on.
    rebuilt = ObservedTwoComplex(
        b1=observed.b1.copy(),
        b2=observed.b2.copy(),
        truth_dims=observed.truth_dims,
        kept_edges=observed.kept_edges,
        flipped_b1=observed.flipped_b1,
        flipped_b2=observed.flipped_b2,
    )
    assert np.array_equal(rebuilt.b1, observed.b1)
    with pytest.raises(ValueError):
        ChainComplex((rebuilt.b1, rebuilt.b2))


def test_kept_row_masking_consistency_bookkeeping_contract():
    instance, observed = _observed_true(0.5, mode="mask")
    num_vertices, num_edges, num_faces = instance.true_target.dims  # (4, 6, 3)
    truth_b1, truth_b2 = instance.true_target.boundaries
    assert observed.truth_dims == (num_vertices, num_edges, num_faces)
    assert observed.num_masked == round(0.5 * num_edges)
    assert len(observed.kept_edges) == num_edges - observed.num_masked
    assert observed.b1.shape == (num_vertices, len(observed.kept_edges))
    # The contract: B2 loses exactly the rows of the edges whose B1
    # columns were removed — one consistent kept-edge set.
    assert observed.b2.shape == (len(observed.kept_edges), num_faces)
    assert all(
        b > a for a, b in zip(observed.kept_edges, observed.kept_edges[1:])
    )
    for slot, edge in enumerate(observed.kept_edges):
        assert np.array_equal(observed.b1[:, slot], truth_b1[:, edge])
        assert np.array_equal(observed.b2[slot, :], truth_b2[edge, :])
    assert observed.flipped_b1 == () and observed.flipped_b2 == ()
    # Partial masking also breaks d^2 (dropped terms only cancel in full).
    assert observed_d2_residue(observed) > 0.0


def test_full_masking_removes_every_edge():
    instance, observed = _observed_true(1.0, mode="mask")
    assert observed.kept_edges == ()
    assert observed.num_masked == instance.true_target.dims[1]
    assert observed.b1.shape == (4, 0)
    assert observed.b2.shape == (0, 3)
    # No edge observed: degree 1 ties at ||f0 B1||_F for every candidate,
    # and the kept-row convention leaves degree 2 exactly zero.
    f0 = instance.chain_map.maps[0]
    expected = float(np.linalg.norm(f0 @ instance.source.boundaries[0]))
    residuals = observed_commutation_residuals(instance.chain_map, observed)
    assert residuals[0] == expected
    assert residuals[1] == 0.0
    assert observed_d2_residue(observed) == 0.0


def test_ranking_preserved_below_documented_corrupt_breakdown():
    rows = ranking_study_two(STUDY_SEED, mode="corrupt")
    below = [row for row in rows if row.fraction < CORRUPT_BREAKDOWN]
    assert below and all(row.true_ranks_first for row in below)
    assert all(row.margin > 0.0 for row in below)
    assert breakdown_fraction(rows) == CORRUPT_BREAKDOWN
    lost = next(row for row in rows if row.fraction == CORRUPT_BREAKDOWN)
    # The breakdown is a tie: margin exactly zero, so the true candidate
    # no longer ranks strictly first; from 0.5 on it loses strictly.
    assert not lost.true_ranks_first and lost.margin == 0.0
    assert all(
        row.margin < 0.0
        for row in rows
        if row.fraction > CORRUPT_BREAKDOWN
    )


def test_masking_preserves_ranking_until_total_masking():
    rows = ranking_study_two(STUDY_SEED, mode="mask")
    assert all(
        row.true_ranks_first for row in rows if row.fraction < MASK_BREAKDOWN
    )
    assert breakdown_fraction(rows) == MASK_BREAKDOWN
    # At full masking no edge is observed: every candidate ties at
    # ||f0 B1||_F and the margin collapses to exactly zero.
    assert rows[-1].margin == 0.0


def test_observation_and_study_are_deterministic():
    instance = make_two_complex_switch_instance(STUDY_SEED)
    first = TwoComplexObservationModel(
        instance.true_target,
        7,
        mask_fraction=0.4,
        corrupt_b1_fraction=0.3,
        corrupt_b2_fraction=0.5,
    ).observe()
    second = TwoComplexObservationModel(
        instance.true_target,
        7,
        mask_fraction=0.4,
        corrupt_b1_fraction=0.3,
        corrupt_b2_fraction=0.5,
    ).observe()
    assert np.array_equal(first.b1, second.b1)
    assert np.array_equal(first.b2, second.b2)
    assert first.kept_edges == second.kept_edges
    assert first.flipped_b1 == second.flipped_b1
    assert first.flipped_b2 == second.flipped_b2
    assert ranking_study_two(STUDY_SEED, mode="corrupt") == ranking_study_two(
        STUDY_SEED, mode="corrupt"
    )
    assert ranking_study_two(STUDY_SEED, mode="mask") == ranking_study_two(
        STUDY_SEED, mode="mask"
    )


def test_degradation_sets_are_nested_in_fraction():
    instance = make_two_complex_switch_instance(STUDY_SEED)
    low = TwoComplexObservationModel(
        instance.true_target, 7, corrupt_b1_fraction=0.2,
        corrupt_b2_fraction=0.3,
    )
    high = TwoComplexObservationModel(
        instance.true_target, 7, corrupt_b1_fraction=0.6,
        corrupt_b2_fraction=0.8,
    )
    assert set(low.observe().flipped_b1) <= set(high.observe().flipped_b1)
    assert set(low.observe().flipped_b2) <= set(high.observe().flipped_b2)
    low_mask = TwoComplexObservationModel(
        instance.true_target, 7, mask_fraction=0.2
    )
    high_mask = TwoComplexObservationModel(
        instance.true_target, 7, mask_fraction=0.6
    )
    assert set(high_mask.observe().kept_edges) <= set(
        low_mask.observe().kept_edges
    )


def test_corruption_bookkeeping_counts_and_values():
    instance, observed = _observed_true(0.5)
    truth_b1, truth_b2 = instance.true_target.boundaries
    assert len(observed.flipped_b1) == round(0.5 * np.count_nonzero(truth_b1))
    assert len(observed.flipped_b2) == round(0.5 * np.count_nonzero(truth_b2))
    flipped_b1 = set(observed.flipped_b1)
    for row in range(truth_b1.shape[0]):
        for col in range(truth_b1.shape[1]):
            expected = truth_b1[row, col]
            if (row, col) in flipped_b1:
                expected = -expected
            assert observed.b1[row, col] == expected
    flipped_b2 = set(observed.flipped_b2)
    for row in range(truth_b2.shape[0]):
        for col in range(truth_b2.shape[1]):
            expected = truth_b2[row, col]
            if (row, col) in flipped_b2:
                expected = -expected
            assert observed.b2[row, col] == expected


def test_fractions_outside_unit_interval_rejected():
    instance = make_two_complex_switch_instance(STUDY_SEED)
    for bad in (-0.1, 1.1):
        for kwargs in (
            {"mask_fraction": bad},
            {"corrupt_b1_fraction": bad},
            {"corrupt_b2_fraction": bad},
        ):
            with pytest.raises(ValueError, match="outside"):
                TwoComplexObservationModel(instance.true_target, 0, **kwargs)


def test_observation_model_rejects_non_2_complexes():
    instance = make_two_complex_switch_instance(STUDY_SEED)
    graph = ChainComplex((instance.true_target.boundaries[0],))
    with pytest.raises(ValueError, match="2-complexes"):
        TwoComplexObservationModel(graph, 0)


def test_ranking_study_two_rejects_bad_mode_and_fractions():
    with pytest.raises(ValueError, match="mode"):
        ranking_study_two(STUDY_SEED, mode="bogus")
    with pytest.raises(ValueError, match="non-decreasing"):
        ranking_study_two(STUDY_SEED, (0.5, 0.25))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ranking_study_two(STUDY_SEED, (0.0, 1.5))


def test_observed_commutation_residuals_rejects_mismatches():
    instance, observed = _observed_true(0.3)
    # Observation of a different complex than the chain map's target.
    wrong = TwoComplexObservationModel(instance.source, 7).observe()
    assert wrong.truth_dims != instance.true_target.dims
    with pytest.raises(ValueError, match="truth dims"):
        observed_commutation_residuals(instance.chain_map, wrong)
    # A chain map between 1-complexes has no degrees to compare here.
    graph = ChainComplex((instance.true_target.boundaries[0],))
    v, e = graph.dims
    identity = ChainMap(graph, graph, (np.eye(v), np.eye(e)))
    with pytest.raises(ValueError, match="2-complexes"):
        observed_commutation_residuals(identity, observed)


def test_observed_two_complex_validates_its_own_dims_contract():
    instance = make_two_complex_switch_instance(STUDY_SEED)
    truth_b1, truth_b2 = instance.true_target.boundaries
    dims = instance.true_target.dims
    base = dict(
        b1=truth_b1.copy(),
        b2=truth_b2.copy(),
        truth_dims=dims,
        kept_edges=tuple(range(dims[1])),
        flipped_b1=(),
        flipped_b2=(),
    )
    # Six claimed kept edges but only three columns provided.
    with pytest.raises(ValueError, match="dims contract"):
        ObservedTwoComplex(**{**base, "b1": truth_b1[:, :3].copy()})
    # kept_edges must be strictly increasing and in range.
    with pytest.raises(ValueError, match="strictly increasing"):
        ObservedTwoComplex(**{**base, "kept_edges": (0, 2, 1, 3, 4, 5)})
    with pytest.raises(ValueError, match="out of range"):
        ObservedTwoComplex(
            **{**base, "kept_edges": tuple(range(dims[1] - 1)) + (dims[1],)}
        )
    # Flip sites must sit on nonzero entries of the right array.
    zero_b1 = next(
        (r, c)
        for r in range(truth_b1.shape[0])
        for c in range(truth_b1.shape[1])
        if truth_b1[r, c] == 0.0
    )
    with pytest.raises(ValueError, match="b1 entry must be nonzero"):
        ObservedTwoComplex(**{**base, "flipped_b1": (zero_b1,)})
    zero_b2 = next(
        (r, c)
        for r in range(truth_b2.shape[0])
        for c in range(truth_b2.shape[1])
        if truth_b2[r, c] == 0.0
    )
    with pytest.raises(ValueError, match="b2 entry must be nonzero"):
        ObservedTwoComplex(**{**base, "flipped_b2": (zero_b2,)})
    with pytest.raises(ValueError, match="out of range"):
        ObservedTwoComplex(**{**base, "flipped_b2": ((dims[1], 0),)})
