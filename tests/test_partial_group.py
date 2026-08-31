import math

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
from universa.partial import breakdown_fraction
from universa.partial_group import (
    GroupObservationModel,
    ObservedGroupNerve,
    observed_nerve_commutation_residual,
    ranking_study_group,
)
from universa.structures import ChainComplex, ChainMap

STUDY_SEED = 70001
# Documented in universa.partial_group's docstring: on the default family
# (z6_to_z3, two decoys) and the default 0.1 grid, the true candidate
# keeps ranking first strictly below this fraction and stops at it.
CORRUPT_BREAKDOWN = 0.8


def _planted_homomorphisms():
    """The three planted families, with phi written out as ground truth."""
    return (
        ("z6_to_z3", cyclic_group(6), cyclic_group(3), (0, 1, 2, 0, 1, 2)),
        ("z4_to_z2", cyclic_group(4), cyclic_group(2), (0, 1, 0, 1)),
        # S3 in lexicographic one-line order; parity by inversion count.
        ("s3_sign", symmetric_group_3(), cyclic_group(2), (0, 1, 1, 0, 0, 1)),
    )


def _table_of(group):
    order = len(group.morphisms)
    return np.array(
        [[group.compose[(g, f)] for f in range(order)] for g in range(order)]
    )


def test_full_observation_reproduces_exact_zero_residual_at_degree_2():
    for name, source, target, phi in _planted_homomorphisms():
        chain_map = induced_nerve_map(phi, source, target)
        observed = GroupObservationModel(
            target, 70003, corrupt_fraction=0.0
        ).observe()
        truth_d1, truth_d2 = nerve_chain_complex(target).boundaries
        # Same assembly operations as the nerve: bitwise equal boundaries.
        assert np.array_equal(observed.d1, truth_d1), name
        assert np.array_equal(observed.d2, truth_d2), name
        assert np.abs(observed.d1).max() == 0.0, name  # one object: d1 = 0
        assert observed.corrupted_entries == (), name
        residual = observed_nerve_commutation_residual(chain_map, observed)
        assert residual == 0.0, name  # exact, not numerical residue
        assert residual == chain_map.commutation_residuals(tol=0.0)[1], name


def test_study_at_zero_degradation_reproduces_commutation_scores():
    instance = make_group_switch_instance(STUDY_SEED)
    first = ranking_study_group(STUDY_SEED)[0]
    scores = instance.commutation_scores()
    assert first.fraction == 0.0
    # Degree 1 is identically zero for one-object nerves, so the scores'
    # max over degrees IS the degree-2 residual the study evaluates.
    assert first.true_misfit == scores[0] == 0.0
    assert first.decoy_misfits == scores[1:]
    assert first.floor == min(scores[1:])


def test_true_residual_matches_closed_form_and_is_monotone():
    rows = ranking_study_group(STUDY_SEED)
    residuals = [row.true_misfit for row in rows]
    # Closed form (module docstring): each corrupted target entry is hit
    # by (s / m)^2 = 4 source pairs and adds a Frobenius-orthogonal
    # column of squared norm 2, so the residual is sqrt(8 * round(9 f)).
    for row in rows:
        expected = math.sqrt(8 * int(round(row.fraction * 9)))
        assert abs(row.true_misfit - expected) <= 1e-12
    # Monotone non-decreasing on every seed by construction ...
    assert all(b >= a for a, b in zip(residuals, residuals[1:]))
    # ... and strict wherever the corruption count grows: the only flat
    # step on the 0.1 grid is 0.4 -> 0.5 (round(9 f) = 4 at both).
    flat = [
        (a.fraction, b.fraction)
        for a, b in zip(rows, rows[1:])
        if b.true_misfit == a.true_misfit
    ]
    assert flat == [(0.4, 0.5)]


def test_ranking_preserved_below_documented_breakdown():
    rows = ranking_study_group(STUDY_SEED)
    below = [row for row in rows if row.fraction < CORRUPT_BREAKDOWN]
    assert below and all(row.true_ranks_first for row in below)
    assert all(row.margin > 0.0 for row in below)
    assert breakdown_fraction(rows) == CORRUPT_BREAKDOWN
    lost = next(row for row in rows if row.fraction == CORRUPT_BREAKDOWN)
    # The true candidate loses strictly at the breakdown and beyond.
    assert not lost.true_ranks_first and lost.margin < 0.0
    assert all(
        row.margin < 0.0
        for row in rows
        if row.fraction > CORRUPT_BREAKDOWN
    )


def test_corruption_bookkeeping_counts_values_and_provenance():
    truth = cyclic_group(3)
    table = _table_of(truth)
    observed = GroupObservationModel(
        truth, 70013, corrupt_fraction=0.5
    ).observe()
    assert observed.truth_order == 3
    # round-half-even: round(0.5 * 9) = round(4.5) = 4.
    assert observed.num_corrupted == 4 == int(round(0.5 * 9))
    entries = observed.corrupted_entries
    assert entries == tuple(sorted(entries))
    assert len({(g, f) for g, f, _, _ in entries}) == len(entries)
    corrupted_table = table.copy()
    for g, f, old, new in entries:
        assert table[g, f] == old  # provenance's old matches the truth
        assert new != old  # the product is actually changed
        corrupted_table[g, f] = new
    # Provenance is complete: every d2 column agrees with the truth table
    # plus exactly the claimed corruptions, nothing else.
    expected_d2 = np.zeros((3, 9))
    for g in range(3):
        for f in range(3):
            column = g * 3 + f
            expected_d2[g, column] += 1.0
            expected_d2[int(corrupted_table[g, f]), column] -= 1.0
            expected_d2[f, column] += 1.0
    assert np.array_equal(observed.d2, expected_d2)
    assert not np.array_equal(observed.d2, nerve_chain_complex(truth).boundaries[1])


def test_observed_nerve_bypasses_category_validation():
    truth = cyclic_group(3)
    observed = GroupObservationModel(
        truth, 70011, corrupt_fraction=0.34
    ).observe()
    assert observed.num_corrupted == 3
    # Rebuild the corrupted table from provenance: it fails
    # FiniteCategory's associativity check, so the observed boundaries
    # are the nerve of no category — a purported structure.
    corrupted_table = _table_of(truth)
    for g, f, old, new in observed.corrupted_entries:
        corrupted_table[g, f] = new
    with pytest.raises(ValueError, match="associativity"):
        group_as_category(corrupted_table)
    # ... yet ObservedGroupNerve constructs fine on the same arrays ...
    rebuilt = ObservedGroupNerve(
        d1=observed.d1.copy(),
        d2=observed.d2.copy(),
        truth_order=observed.truth_order,
        corrupted_entries=observed.corrupted_entries,
    )
    assert np.array_equal(rebuilt.d2, observed.d2)
    # ... and, unlike partial2's observed pairs, even ChainComplex's d^2
    # gate cannot see the damage: d1 = 0 makes d1 d2 = 0 trivially.
    complex_ = ChainComplex((rebuilt.d1, rebuilt.d2))
    assert complex_.dims == (1, 3, 9)


def test_full_corruption_replaces_every_entry():
    truth = cyclic_group(3)
    observed = GroupObservationModel(
        truth, 70016, corrupt_fraction=1.0
    ).observe()
    assert observed.num_corrupted == 9
    assert len({(g, f) for g, f, _, _ in observed.corrupted_entries}) == 9
    chain_map = induced_nerve_map(
        (0, 1, 2, 0, 1, 2), cyclic_group(6), truth
    )
    residual = observed_nerve_commutation_residual(chain_map, observed)
    assert abs(residual - math.sqrt(72.0)) <= 1e-12


def test_observation_and_study_are_deterministic():
    first = GroupObservationModel(
        cyclic_group(3), 70007, corrupt_fraction=0.4
    ).observe()
    second = GroupObservationModel(
        cyclic_group(3), 70007, corrupt_fraction=0.4
    ).observe()
    assert np.array_equal(first.d1, second.d1)
    assert np.array_equal(first.d2, second.d2)
    assert first.corrupted_entries == second.corrupted_entries
    assert ranking_study_group(STUDY_SEED) == ranking_study_group(STUDY_SEED)


def test_degradation_is_nested_in_fraction():
    truth = cyclic_group(3)
    low = GroupObservationModel(truth, 70012, corrupt_fraction=0.2).observe()
    high = GroupObservationModel(truth, 70012, corrupt_fraction=0.6).observe()
    # Nested including replacement values: the low-fraction entries are a
    # prefix of one master permutation with the same draws.
    assert set(low.corrupted_entries) <= set(high.corrupted_entries)
    assert low.num_corrupted == int(round(0.2 * 9))
    assert high.num_corrupted == int(round(0.6 * 9))


def test_fractions_outside_unit_interval_rejected():
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError, match="outside"):
            GroupObservationModel(cyclic_group(3), 0, corrupt_fraction=bad)


def test_observation_model_rejects_non_groups():
    # A category with more than one object.
    discrete = FiniteCategory(2, ((0, 0), (1, 1)), {(0, 0): 0, (1, 1): 1})
    with pytest.raises(ValueError, match="one-object"):
        GroupObservationModel(discrete, 70015)
    # A one-object monoid that is not a group: element 1 has no inverse.
    monoid = FiniteCategory(
        1, ((0, 0), (0, 0)), {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1}
    )
    with pytest.raises(ValueError, match="not a group"):
        GroupObservationModel(monoid, 70015)
    # A one-element group observes fine at zero corruption ...
    trivial = GroupObservationModel(cyclic_group(1), 70014).observe()
    assert trivial.d2.shape == (1, 1) and trivial.d2[0, 0] == 1.0
    # ... but cannot be corrupted: no different element exists to draw.
    with pytest.raises(ValueError, match="one-element"):
        GroupObservationModel(cyclic_group(1), 70014, corrupt_fraction=0.5)


def test_ranking_study_group_rejects_bad_fractions():
    with pytest.raises(ValueError, match="non-decreasing"):
        ranking_study_group(STUDY_SEED, (0.5, 0.25))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ranking_study_group(STUDY_SEED, (0.0, 1.5))


def test_residual_rejects_mismatches():
    # Observed order-3 group vs a chain map targeting an order-2 group.
    instance = make_group_switch_instance(
        70002, family="z4_to_z2", num_decoys=1
    )
    observed = GroupObservationModel(
        cyclic_group(3), 70017, corrupt_fraction=0.2
    ).observe()
    with pytest.raises(ValueError, match="truth order"):
        observed_nerve_commutation_residual(instance.chain_map, observed)
    # A chain map between 1-complexes has no degree 2 to compare here.
    graph = ChainComplex((np.array([[-1.0], [1.0]]),))
    identity = ChainMap(graph, graph, (np.eye(2), np.eye(1)))
    with pytest.raises(ValueError, match="2-complexes"):
        observed_nerve_commutation_residual(identity, observed)


def test_observed_group_nerve_validates_its_own_bookkeeping():
    observed = GroupObservationModel(
        cyclic_group(3), 70011, corrupt_fraction=0.34
    ).observe()
    base = dict(
        d1=observed.d1.copy(),
        d2=observed.d2.copy(),
        truth_order=3,
        corrupted_entries=observed.corrupted_entries,
    )
    with pytest.raises(ValueError, match="dims contract"):
        ObservedGroupNerve(**{**base, "d1": np.zeros((2, 3))})
    with pytest.raises(ValueError, match="dims contract"):
        ObservedGroupNerve(**{**base, "d2": observed.d2[:, :8].copy()})
    with pytest.raises(ValueError, match="dims contract"):
        ObservedGroupNerve(**{**base, "truth_order": 2})
    with pytest.raises(ValueError, match="zero map"):
        ObservedGroupNerve(**{**base, "d1": np.ones((1, 3))})
    bad_face = observed.d2.copy()
    bad_face[:, 0] = 0.0  # a column that no longer sums to 1
    with pytest.raises(ValueError, match="purported face"):
        ObservedGroupNerve(**{**base, "d2": bad_face})
    g, f, old, new = observed.corrupted_entries[0]
    with pytest.raises(ValueError, match="out of range"):
        ObservedGroupNerve(
            **{**base, "corrupted_entries": ((0, 3, 0, 1),)}
        )
    with pytest.raises(ValueError, match="must change"):
        ObservedGroupNerve(
            **{**base, "corrupted_entries": ((g, f, old, old),)}
        )
    with pytest.raises(ValueError, match="duplicate"):
        ObservedGroupNerve(
            **{
                **base,
                "corrupted_entries": observed.corrupted_entries
                + observed.corrupted_entries[:1],
            }
        )
    claimed = (new + 1) % 3  # a wrong replacement value, still != old
    if claimed == old:
        claimed = (claimed + 1) % 3
    with pytest.raises(ValueError, match="inconsistent"):
        ObservedGroupNerve(
            **{**base, "corrupted_entries": ((g, f, old, claimed),)}
        )
