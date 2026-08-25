import math

import numpy as np

from universa.complexes2 import two_complex
from universa.generators import make_switch_instance, subseed
from universa.partial import (
    ObservationModel,
    ObservedComplex,
    breakdown_fraction,
    observed_misfit,
    ranking_study,
)
from universa.structures import ChainComplex

STUDY_SEED = 2026
# Documented in universa.partial's docstring: on the default sizes and the
# default 0.1 grid, the true candidate keeps ranking first strictly below
# these fractions and stops ranking first at them.
CORRUPT_BREAKDOWN = 0.6
MASK_BREAKDOWN = 1.0


def _observed_true(fraction, mode="corrupt", seed=STUDY_SEED):
    """The study's shared observation of the true target at one fraction."""
    instance = make_switch_instance(seed)
    observation_seed = subseed(seed, "partial", mode)
    kwargs = {"mask_fraction": 0.0, "corrupt_fraction": 0.0}
    kwargs[f"{mode}_fraction"] = fraction
    model = ObservationModel(instance.true_target, observation_seed, **kwargs)
    return instance, model.observe()


def test_full_observation_reproduces_exact_zero_residual():
    instance = make_switch_instance(STUDY_SEED)
    observed = ObservationModel(instance.true_target, seed=123).observe()
    assert np.array_equal(
        observed.complex.boundaries[0], instance.true_target.boundaries[0]
    )
    num_edges = instance.true_target.dims[1]
    assert observed.kept_edges == tuple(range(num_edges))
    assert observed.flipped_entries == ()
    residual = observed_misfit(instance.chain_map, observed)
    assert residual == 0.0  # exact, not numerical residue
    assert residual == instance.commutation_scores()[0]


def test_study_at_zero_degradation_reproduces_commutation_scores():
    instance = make_switch_instance(STUDY_SEED)
    first = ranking_study(STUDY_SEED, mode="corrupt")[0]
    scores = instance.commutation_scores()
    assert first.fraction == 0.0
    assert first.true_misfit == scores[0] == 0.0
    assert first.decoy_misfits == scores[1:]
    assert first.floor == min(scores[1:])


def test_corruption_residual_degrades_monotonically_fixed_seed():
    rows = ranking_study(STUDY_SEED, mode="corrupt")
    residuals = [row.true_misfit for row in rows]
    assert residuals[0] == 0.0
    # The quotient target has 6 edges (12 flippable entries) and the 0.1
    # grid grows the flip count at every step, so the structural
    # monotonicity is strict at every step on this seed.
    assert all(b > a for a, b in zip(residuals, residuals[1:]))


def test_corruption_residual_matches_independent_recomputation():
    instance, observed = _observed_true(0.5)
    residual = observed_misfit(instance.chain_map, observed)
    # Each flipped entry (i, e) contributes ||f1[e, :]||^2 to the squared
    # residual; the contributions are Frobenius-orthogonal because every
    # source edge lands in exactly one target edge row.
    f1 = instance.chain_map.maps[1]
    squared = sum(
        4.0 * float(f1[col] @ f1[col]) for _, col in observed.flipped_entries
    )
    assert abs(residual - math.sqrt(squared)) <= 1e-12


def test_ranking_preserved_below_documented_corrupt_breakdown():
    rows = ranking_study(STUDY_SEED, mode="corrupt")
    below = [row for row in rows if row.fraction < CORRUPT_BREAKDOWN]
    assert below and all(row.true_ranks_first for row in below)
    assert all(row.margin > 0.0 for row in below)
    assert breakdown_fraction(rows) == CORRUPT_BREAKDOWN
    lost = next(row for row in rows if row.fraction == CORRUPT_BREAKDOWN)
    assert not lost.true_ranks_first and lost.margin < 0.0


def test_masking_preserves_ranking_until_total_masking():
    rows = ranking_study(STUDY_SEED, mode="mask")
    assert all(
        row.true_ranks_first for row in rows if row.fraction < MASK_BREAKDOWN
    )
    assert breakdown_fraction(rows) == MASK_BREAKDOWN
    # At full masking no edge is observed: every candidate ties at
    # ||f0 B1||_F and the margin collapses to exactly zero.
    assert rows[-1].margin == 0.0


def test_observation_and_study_are_deterministic():
    instance = make_switch_instance(STUDY_SEED)
    first = ObservationModel(
        instance.true_target, 7, mask_fraction=0.4, corrupt_fraction=0.3
    ).observe()
    second = ObservationModel(
        instance.true_target, 7, mask_fraction=0.4, corrupt_fraction=0.3
    ).observe()
    assert np.array_equal(
        first.complex.boundaries[0], second.complex.boundaries[0]
    )
    assert first.kept_edges == second.kept_edges
    assert first.flipped_entries == second.flipped_entries
    assert ranking_study(STUDY_SEED, mode="corrupt") == ranking_study(
        STUDY_SEED, mode="corrupt"
    )


def test_degradation_sets_are_nested_in_fraction():
    instance = make_switch_instance(STUDY_SEED)
    low = ObservationModel(instance.true_target, 7, corrupt_fraction=0.2)
    high = ObservationModel(instance.true_target, 7, corrupt_fraction=0.6)
    assert set(low.observe().flipped_entries) <= set(
        high.observe().flipped_entries
    )
    low_mask = ObservationModel(instance.true_target, 7, mask_fraction=0.2)
    high_mask = ObservationModel(instance.true_target, 7, mask_fraction=0.6)
    assert set(high_mask.observe().kept_edges) <= set(
        low_mask.observe().kept_edges
    )


def test_masked_edge_bookkeeping_dims_contract():
    instance = make_switch_instance(STUDY_SEED)
    num_vertices, num_edges = instance.true_target.dims  # (4, 6)
    observed = ObservationModel(
        instance.true_target, 7, mask_fraction=0.5
    ).observe()
    assert observed.truth_dims == (num_vertices, num_edges)
    assert observed.num_masked == round(0.5 * num_edges)
    assert len(observed.kept_edges) == num_edges - observed.num_masked
    assert observed.complex.boundaries[0].shape == (
        num_vertices,
        len(observed.kept_edges),
    )
    assert all(
        b > a for a, b in zip(observed.kept_edges, observed.kept_edges[1:])
    )
    # Kept columns agree with the ground truth; masked ones are gone.
    truth = instance.true_target.boundaries[0]
    for col, edge in enumerate(observed.kept_edges):
        assert np.array_equal(
            observed.complex.boundaries[0][:, col], truth[:, edge]
        )


def test_full_masking_removes_every_edge():
    instance, observed = _observed_true(1.0, mode="mask")
    assert observed.kept_edges == ()
    assert observed.num_masked == instance.true_target.dims[1]
    assert observed.complex.boundaries[0].shape == (4, 0)
    f0 = instance.chain_map.maps[0]
    expected = float(np.linalg.norm(f0 @ instance.source.boundaries[0]))
    assert observed_misfit(instance.chain_map, observed) == expected


def test_corruption_bookkeeping_counts_and_values():
    instance, observed = _observed_true(0.5)
    truth = instance.true_target.boundaries[0]
    nnz = int(np.count_nonzero(truth))
    assert len(observed.flipped_entries) == round(0.5 * nnz)
    flipped = set(observed.flipped_entries)
    boundary = observed.complex.boundaries[0]
    for row in range(truth.shape[0]):
        for col in range(truth.shape[1]):
            expected = truth[row, col]
            if (row, col) in flipped:
                expected = -expected
            assert boundary[row, col] == expected


def test_fractions_outside_unit_interval_rejected():
    instance = make_switch_instance(STUDY_SEED)
    for bad in (-0.1, 1.1):
        for kwargs in (
            {"mask_fraction": bad},
            {"corrupt_fraction": bad},
        ):
            try:
                ObservationModel(instance.true_target, 0, **kwargs)
                raised = False
            except ValueError:
                raised = True
            assert raised


def test_observation_model_rejects_non_graphs():
    instance = make_switch_instance(STUDY_SEED)
    try:
        ObservationModel(two_complex(instance.source), 0)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_ranking_study_rejects_bad_mode_and_fractions():
    calls = (
        lambda: ranking_study(STUDY_SEED, mode="bogus"),
        lambda: ranking_study(STUDY_SEED, (0.5, 0.25)),  # decreasing
        lambda: ranking_study(STUDY_SEED, (0.0, 1.5)),  # outside [0, 1]
    )
    for call in calls:
        try:
            call()
            raised = False
        except ValueError:
            raised = True
        assert raised


def test_observed_misfit_rejects_mismatched_truth():
    instance = make_switch_instance(STUDY_SEED)
    wrong = ObservationModel(instance.source, 7).observe()  # dims (8, 14)
    assert wrong.truth_dims != instance.true_target.dims
    try:
        observed_misfit(instance.chain_map, wrong)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_observed_complex_validates_its_own_dims_contract():
    instance = make_switch_instance(STUDY_SEED)
    truth = instance.true_target.boundaries[0]
    try:  # three columns but six claimed kept edges
        ObservedComplex(
            complex=ChainComplex((truth[:, :3].copy(),)),
            truth_dims=instance.true_target.dims,
            kept_edges=(0, 1, 2, 3, 4, 5),
            flipped_entries=(),
        )
        raised = False
    except ValueError:
        raised = True
    assert raised
    zero = next(
        (r, c)
        for r in range(truth.shape[0])
        for c in range(truth.shape[1])
        if truth[r, c] == 0.0
    )
    try:  # a flipped entry must sit on a nonzero position
        ObservedComplex(
            complex=ChainComplex((truth.copy(),)),
            truth_dims=instance.true_target.dims,
            kept_edges=tuple(range(truth.shape[1])),
            flipped_entries=(zero,),
        )
        raised = False
    except ValueError:
        raised = True
    assert raised
