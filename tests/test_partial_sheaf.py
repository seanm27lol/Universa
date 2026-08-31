import math

import numpy as np
import pytest

from universa.generators import subseed
from universa.partial_sheaf import (
    ObservedSheaf,
    SheafObservationModel,
    breakdown_fraction,
    observed_coboundary,
    observed_naturality_residual,
    ranking_study_sheaf,
)
from universa.sheaves import (
    Sheaf,
    coboundary,
    make_sheaf_switch_instance,
    planted_morphism,
    random_sheaf,
)
from universa.structures import ChainComplex

STUDY_SEED = 70001
# Documented in universa.partial_sheaf's docstring: on the default sizes
# and the default 0.1 grid, the true candidate keeps ranking first
# strictly below these fractions and stops ranking first at them (a
# strict loss at 0.5 under corruption, a full tie at 1.0 under masking).
CORRUPT_BREAKDOWN = 0.5
MASK_BREAKDOWN = 1.0


def _planted(seed=STUDY_SEED):
    """The study's source sheaf, true target sheaf, and planted morphism."""
    source = random_sheaf(seed)
    target, morphism = planted_morphism(source, seed)
    return source, target, morphism


def _observed_true(fraction, mode="corrupt", seed=STUDY_SEED):
    """The study's shared observation of the true target at one fraction."""
    _, target, morphism = _planted(seed)
    observation_seed = subseed(seed, "partial-sheaf", mode)
    kwargs = {"mask_fraction": 0.0, "corrupt_fraction": 0.0}
    kwargs[f"{mode}_fraction"] = fraction
    model = SheafObservationModel(target, observation_seed, **kwargs)
    return morphism, model.observe()


def test_full_observation_reproduces_exact_zero_residual():
    _, target, morphism = _planted()
    observed = SheafObservationModel(target, seed=123).observe()
    assert observed.truth_dims == (target.num_vertices, target.num_edges)
    assert observed.kept_edges == tuple(range(target.num_edges))
    assert observed.flipped_entries == ()
    for pair, block in observed.restrictions.items():
        assert np.array_equal(block, target.restrictions[pair])
    assert np.array_equal(observed_coboundary(observed), coboundary(target))
    residual = observed_naturality_residual(morphism, observed)
    assert residual == 0.0  # exact, not numerical residue
    rss = math.sqrt(sum(r * r for r in morphism.naturality_residuals()))
    assert residual == rss


def test_study_at_zero_degradation_reproduces_commutation_scores():
    instance = make_sheaf_switch_instance(STUDY_SEED)
    first = ranking_study_sheaf(STUDY_SEED, mode="corrupt")[0]
    scores = instance.commutation_scores()
    assert first.fraction == 0.0
    assert first.true_misfit == scores[0] == 0.0
    # Dyadic entries make the blockwise root-sum-square order-independent,
    # so the per-incidence assembly matches the compiled scores exactly.
    assert first.decoy_misfits == scores[1:]
    assert first.floor == min(scores[1:])


def test_corruption_residual_degrades_monotonically_fixed_seed():
    rows = ranking_study_sheaf(STUDY_SEED, mode="corrupt")
    residuals = [row.true_misfit for row in rows]
    assert residuals[0] == 0.0
    # The true target has 47 flippable entries and the 0.1 grid grows the
    # flip count at every step, so the structural monotonicity (each flip
    # moves one Frobenius-orthogonal position) is strict at every step.
    assert all(b > a for a, b in zip(residuals, residuals[1:]))


def test_corruption_residual_matches_independent_recomputation():
    morphism, observed = _observed_true(0.5)
    _, target, _ = _planted()
    residual = observed_naturality_residual(morphism, observed)
    # A flipped entry (v, e, i, j) contributes (2 rho'[i, j] scale_v)^2 to
    # the squared residual; phi_v is a scaled permutation, so the
    # contributions are Frobenius-orthogonal and add.
    squared = 0.0
    for v, e, row, col in observed.flipped_entries:
        rho = target.restrictions[(v, e)][row, col]
        scale_v = float(np.abs(morphism.vertex_maps[v]).max())
        squared += 4.0 * rho * rho * scale_v * scale_v
    assert abs(residual - math.sqrt(squared)) <= 1e-12


def test_ranking_preserved_below_documented_corrupt_breakdown():
    rows = ranking_study_sheaf(STUDY_SEED, mode="corrupt")
    below = [row for row in rows if row.fraction < CORRUPT_BREAKDOWN]
    assert below and all(row.true_ranks_first for row in below)
    assert all(row.margin > 0.0 for row in below)
    assert breakdown_fraction(rows) == CORRUPT_BREAKDOWN
    lost = next(row for row in rows if row.fraction == CORRUPT_BREAKDOWN)
    assert not lost.true_ranks_first and lost.margin < 0.0


def test_masking_preserves_ranking_until_total_masking():
    rows = ranking_study_sheaf(STUDY_SEED, mode="mask")
    assert all(
        row.true_ranks_first for row in rows if row.fraction < MASK_BREAKDOWN
    )
    assert breakdown_fraction(rows) == MASK_BREAKDOWN
    # At full masking no incidence survives: every candidate ties at 0.0
    # and the margin collapses to exactly zero.
    assert rows[-1].margin == 0.0
    assert rows[-1].floor == 0.0


def test_masking_true_residual_is_exactly_zero_and_decoys_erode():
    rows = ranking_study_sheaf(STUDY_SEED, mode="mask")
    # Surviving blocks are exact, so the true candidate never moves ...
    assert all(row.true_misfit == 0.0 for row in rows)
    # ... while each decoy's residual is a nested sum of nonnegative
    # per-incidence terms losing terms as edges drop: non-increasing.
    for index in range(len(rows[0].decoy_misfits)):
        series = [row.decoy_misfits[index] for row in rows]
        assert all(b <= a for a, b in zip(series, series[1:]))


def test_observation_and_study_are_deterministic():
    _, target, _ = _planted()
    first = SheafObservationModel(
        target, 7, mask_fraction=0.4, corrupt_fraction=0.3
    ).observe()
    second = SheafObservationModel(
        target, 7, mask_fraction=0.4, corrupt_fraction=0.3
    ).observe()
    assert first.kept_edges == second.kept_edges
    assert first.flipped_entries == second.flipped_entries
    assert first.restrictions.keys() == second.restrictions.keys()
    for pair in first.restrictions:
        assert np.array_equal(
            first.restrictions[pair], second.restrictions[pair]
        )
    assert ranking_study_sheaf(STUDY_SEED, mode="corrupt") == (
        ranking_study_sheaf(STUDY_SEED, mode="corrupt")
    )
    assert ranking_study_sheaf(STUDY_SEED, mode="mask") == (
        ranking_study_sheaf(STUDY_SEED, mode="mask")
    )


def test_degradation_sets_are_nested_in_fraction():
    _, target, _ = _planted()
    low = SheafObservationModel(target, 7, corrupt_fraction=0.2)
    high = SheafObservationModel(target, 7, corrupt_fraction=0.6)
    assert set(low.observe().flipped_entries) <= set(
        high.observe().flipped_entries
    )
    low_mask = SheafObservationModel(target, 7, mask_fraction=0.2)
    high_mask = SheafObservationModel(target, 7, mask_fraction=0.6)
    assert set(high_mask.observe().kept_edges) <= set(
        low_mask.observe().kept_edges
    )


def test_masked_edge_bookkeeping_dims_contract():
    _, target, _ = _planted()  # 6 vertices, 9 edges
    morphism, observed = _observed_true(0.5, mode="mask")
    num_edges = target.num_edges
    assert observed.truth_dims == (target.num_vertices, num_edges)
    assert observed.num_masked == round(0.5 * num_edges)  # 4, half-even
    assert len(observed.kept_edges) == num_edges - observed.num_masked
    assert all(
        b > a for a, b in zip(observed.kept_edges, observed.kept_edges[1:])
    )
    assert observed.flipped_entries == ()
    # Kept blocks agree with the ground truth; masked ones are gone.
    assert set(observed.restrictions) == set(observed.incidence_pairs)
    for pair, block in observed.restrictions.items():
        assert pair[1] in observed.kept_edges
        assert np.array_equal(block, target.restrictions[pair])
    # The observed coboundary drops exactly the masked edges' block rows.
    delta = observed_coboundary(observed)
    assert delta.shape == (observed.c1_dim, observed.c0_dim)
    truth_delta = coboundary(target)
    truth_offsets = np.concatenate(([0], np.cumsum(target.edge_dims)))
    kept_offsets = np.concatenate(
        ([0], np.cumsum([target.edge_dims[e] for e in observed.kept_edges]))
    )
    for slot, e in enumerate(observed.kept_edges):
        kept_rows = slice(kept_offsets[slot], kept_offsets[slot + 1])
        truth_rows = slice(truth_offsets[e], truth_offsets[e + 1])
        assert np.array_equal(delta[kept_rows, :], truth_delta[truth_rows, :])
    assert observed_naturality_residual(morphism, observed) == 0.0


def test_full_masking_removes_every_edge():
    _, target, morphism = _planted()
    observed = SheafObservationModel(target, 7, mask_fraction=1.0).observe()
    assert observed.kept_edges == ()
    assert observed.num_masked == target.num_edges
    assert observed.restrictions == {}
    assert observed.incidence_pairs == ()
    assert observed.c1_dim == 0
    assert observed_coboundary(observed).shape == (0, observed.c0_dim)
    assert observed_naturality_residual(morphism, observed) == 0.0


def test_corruption_bookkeeping_counts_and_values():
    _, target, _ = _planted()
    _, observed = _observed_true(0.5)
    nnz = sum(
        int(np.count_nonzero(block))
        for block in target.restrictions.values()
    )
    assert len(observed.flipped_entries) == round(0.5 * nnz)
    flipped = set(observed.flipped_entries)
    for pair, truth_block in target.restrictions.items():
        block = observed.restrictions[pair]
        assert block.shape == truth_block.shape
        for row in range(block.shape[0]):
            for col in range(block.shape[1]):
                expected = truth_block[row, col]
                if (*pair, row, col) in flipped:
                    assert expected != 0.0
                    expected = -expected
                assert block[row, col] == expected
    # The observed coboundary differs from the truth's at exactly the
    # flipped positions (offset-mapped), each value negated.
    delta = observed_coboundary(observed)
    truth_delta = coboundary(target)
    vertex_offsets = np.concatenate(([0], np.cumsum(target.vertex_dims)))
    edge_offsets = np.concatenate(([0], np.cumsum(target.edge_dims)))
    expected_sites = {
        (int(edge_offsets[e]) + row, int(vertex_offsets[v]) + col)
        for v, e, row, col in flipped
    }
    differing = set(zip(*np.nonzero(delta != truth_delta)))
    assert differing == expected_sites
    for row, col in expected_sites:
        assert delta[row, col] == -truth_delta[row, col]


def test_fractions_outside_unit_interval_rejected():
    _, target, _ = _planted()
    for bad in (-0.1, 1.1):
        for kwargs in (
            {"mask_fraction": bad},
            {"corrupt_fraction": bad},
        ):
            with pytest.raises(ValueError, match="outside"):
                SheafObservationModel(target, 0, **kwargs)


def test_observation_model_rejects_non_sheaves():
    _, target, _ = _planted()
    with pytest.raises(ValueError, match="Sheaf"):
        SheafObservationModel(target.base, 0)


def test_ranking_study_sheaf_rejects_bad_mode_and_fractions():
    with pytest.raises(ValueError, match="mode"):
        ranking_study_sheaf(STUDY_SEED, mode="bogus")
    with pytest.raises(ValueError, match="non-decreasing"):
        ranking_study_sheaf(STUDY_SEED, (0.5, 0.25))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ranking_study_sheaf(STUDY_SEED, (0.0, 1.5))


def test_observed_naturality_residual_rejects_mismatches():
    source, target, morphism = _planted()
    observed = SheafObservationModel(target, 7).observe()
    with pytest.raises(ValueError, match="SheafMorphism"):
        observed_naturality_residual(morphism.to_chain_map(), observed)
    # A morphism planted on a different base graph.
    other_source = random_sheaf(70002, num_vertices=5, num_edges=8)
    _, other_morphism = planted_morphism(other_source, 70002)
    with pytest.raises(ValueError, match="base"):
        observed_naturality_residual(other_morphism, observed)
    # Same base graph, but stalk dims that are not the morphism's target's.
    bumped = (source.vertex_dims[0] + 1, *source.vertex_dims[1:])
    restrictions = {
        pair: np.zeros((source.edge_dims[pair[1]], bumped[pair[0]]))
        for pair in source.incidence_pairs
    }
    other = Sheaf(source.base, bumped, source.edge_dims, restrictions)
    with pytest.raises(ValueError, match="stalk dims"):
        observed_naturality_residual(
            morphism, SheafObservationModel(other, 7).observe()
        )


def test_observed_sheaf_validates_its_own_bookkeeping():
    _, target, _ = _planted()
    _, observed = _observed_true(0.5, mode="mask")
    base = dict(
        base=observed.base,
        vertex_dims=observed.vertex_dims,
        edge_dims=observed.edge_dims,
        restrictions=dict(observed.restrictions),
        kept_edges=observed.kept_edges,
        flipped_entries=observed.flipped_entries,
    )
    # The purported structure itself — masked incidences missing — builds
    # fine: only bookkeeping is gated on.
    rebuilt = ObservedSheaf(**base)
    assert rebuilt.kept_edges == observed.kept_edges
    with pytest.raises(ValueError, match="strictly increasing"):
        ObservedSheaf(
            **{**base, "kept_edges": tuple(reversed(base["kept_edges"]))}
        )
    with pytest.raises(ValueError, match="out of range"):
        ObservedSheaf(
            **{
                **base,
                "kept_edges": base["kept_edges"][:-1] + (target.num_edges,),
            }
        )
    missing = dict(base["restrictions"])
    missing.pop(observed.incidence_pairs[0])
    with pytest.raises(ValueError, match="surviving incidences"):
        ObservedSheaf(**{**base, "restrictions": missing})
    extra = dict(base["restrictions"])
    masked = next(
        e for e in range(target.num_edges) if e not in observed.kept_edges
    )
    extra_pair = next(p for p in target.restrictions if p[1] == masked)
    extra[extra_pair] = target.restrictions[extra_pair]
    with pytest.raises(ValueError, match="not a surviving incidence"):
        ObservedSheaf(**{**base, "restrictions": extra})
    bad_shape = dict(base["restrictions"])
    pair = observed.incidence_pairs[0]
    expected = (observed.edge_dims[pair[1]], observed.vertex_dims[pair[0]])
    bad_shape[pair] = np.ones((expected[0] + 1, expected[1]))
    with pytest.raises(ValueError, match="shape"):
        ObservedSheaf(**{**base, "restrictions": bad_shape})


def test_observed_sheaf_rejects_bad_flip_provenance():
    _, target, _ = _planted()
    _, observed = _observed_true(0.3)  # pure corruption, all edges kept
    base = dict(
        base=observed.base,
        vertex_dims=observed.vertex_dims,
        edge_dims=observed.edge_dims,
        restrictions=dict(observed.restrictions),
        kept_edges=observed.kept_edges,
        flipped_entries=observed.flipped_entries,
    )
    pair, zero = next(
        (pair, (r, c))
        for pair in observed.incidence_pairs
        for r in range(target.restrictions[pair].shape[0])
        for c in range(target.restrictions[pair].shape[1])
        if target.restrictions[pair][r, c] == 0.0
    )
    block = target.restrictions[pair]
    with pytest.raises(ValueError, match="nonzero"):
        ObservedSheaf(**{**base, "flipped_entries": ((*pair, *zero),)})
    with pytest.raises(ValueError, match="out of range"):
        ObservedSheaf(
            **{**base, "flipped_entries": ((*pair, block.shape[0], 0),)}
        )
    with pytest.raises(ValueError, match="surviving incidence"):
        ObservedSheaf(
            **{**base, "flipped_entries": ((0, target.num_edges, 0, 0),)}
        )
