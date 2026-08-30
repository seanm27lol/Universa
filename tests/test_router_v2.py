import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only, before the torch import

import numpy as np
import pytest
import torch

import universa
from universa.budgets import make_budget_instance
from universa.generators import subseed
from universa.operators import nullspace_basis
from universa.partial import ObservationModel, observed_misfit
from universa.router import DEGRADED_STRUCTURAL_NAMES
from universa.router_v2 import (
    DEFAULT_MASK_FRACTION,
    NO_ANCHOR_GRID,
    NoAnchorFractionReport,
    NoAnchorInstanceMetadata,
    NoAnchorRegimeReport,
    build_no_anchor_dataset,
    evaluate_no_anchor,
    mean_profile_heuristic_accuracy,
    no_anchor_candidate_features,
    no_anchor_feature_dim,
    no_anchor_feature_names,
    observed_residual_oracle_accuracy,
)
from universa.structures import ChainMap

# Seeds: disjoint from the v1 sealed eval block 30101..30136 (never
# instantiated anywhere) and from the v1 train block; 50001+ throughout.
SEALED_V1_EVAL_BLOCK = range(30101, 30137)
BUILD_SEEDS = (50001, 50002, 50003)
BUILD_FRACTIONS = (0.2, 0.3)
NUM_DECOYS = 2  # K = 3
K = 1 + NUM_DECOYS
GRID_LEN = len(NO_ANCHOR_GRID)
V2_DIM = no_anchor_feature_dim(NO_ANCHOR_GRID)


def small_dataset():
    return build_no_anchor_dataset(
        BUILD_SEEDS, BUILD_FRACTIONS, num_decoys=NUM_DECOYS
    )


def test_dataset_shapes_finiteness_determinism():
    features, labels, metadata = small_dataset()
    rows = len(BUILD_SEEDS) * len(BUILD_FRACTIONS)
    assert features.shape == (rows, K, V2_DIM)
    assert V2_DIM == len(no_anchor_feature_names(NO_ANCHOR_GRID))
    assert V2_DIM == 2 * GRID_LEN + 2  # profile + slopes + 3 dims
    assert features.dtype == np.float64
    assert np.isfinite(features).all()
    assert labels.shape == (rows,) and labels.dtype == np.int64
    assert 0 <= labels.min() and labels.max() < K
    assert len(metadata) == rows
    assert {meta.seed for meta in metadata} == set(BUILD_SEEDS)
    assert {meta.fraction for meta in metadata} == set(BUILD_FRACTIONS)
    # Deterministic rebuild: identical features, labels, and metadata.
    again = small_dataset()
    assert np.array_equal(features, again[0])
    assert np.array_equal(labels, again[1])
    assert metadata == again[2]
    # Audited permutation labels, exactly as v0/v1.
    for i, meta in enumerate(metadata):
        assert sorted(meta.permutation) == list(range(K))
        assert meta.permutation[meta.true_index] == 0
        assert labels[i] == meta.true_index
    assert len(set(int(label) for label in labels)) > 1


def test_rows_repeat_per_instance_across_operating_fractions():
    # The observation draw and permutation are per instance, so the rows of
    # one instance repeat the same permuted feature block; the operating
    # fraction selects the oracle column and the split, never a feature.
    features, labels, metadata = small_dataset()
    for seed in BUILD_SEEDS:
        rows = [i for i, meta in enumerate(metadata) if meta.seed == seed]
        assert len(rows) == len(BUILD_FRACTIONS)
        for row in rows[1:]:
            assert np.array_equal(features[rows[0]], features[row])
            assert labels[rows[0]] == labels[row]


def test_feature_names_layout():
    names = no_anchor_feature_names(NO_ANCHOR_GRID)
    assert names[0] == "log1p_observed_misfit_fraction_0.2"
    assert names[GRID_LEN - 1] == "log1p_observed_misfit_fraction_0.9"
    assert names[GRID_LEN] == "profile_slope_0.2_to_0.3"
    assert names[2 * GRID_LEN - 2] == "profile_slope_0.8_to_0.9"
    assert names[-3:] == DEGRADED_STRUCTURAL_NAMES
    with pytest.raises(ValueError):
        no_anchor_feature_names(())  # empty grid
    with pytest.raises(ValueError):
        no_anchor_feature_names((0.0, 0.2, 0.3))  # anchored grid rejected
    with pytest.raises(ValueError):
        no_anchor_feature_names((0.2, 0.2))  # not strictly increasing
    with pytest.raises(ValueError):
        no_anchor_feature_names((0.2, 1.2))  # outside (0, 1]
    with pytest.raises(ValueError):
        no_anchor_feature_names((-0.1, 0.2))  # outside (0, 1]


def test_grid_excludes_zero_and_fraction_zero_row_rejected():
    assert len(NO_ANCHOR_GRID) == 8
    assert NO_ANCHOR_GRID[0] == 0.2 and NO_ANCHOR_GRID[-1] == 0.9
    assert 0.0 not in NO_ANCHOR_GRID
    assert all(0.0 < g <= 1.0 for g in NO_ANCHOR_GRID)
    # The clean anchor is not a row fraction either: 0.0 names no column.
    with pytest.raises(ValueError):
        build_no_anchor_dataset(
            BUILD_SEEDS, (0.0,), num_decoys=NUM_DECOYS
        )
    with pytest.raises(ValueError):
        build_no_anchor_dataset(
            BUILD_SEEDS,
            (0.2,),
            profile_grid=(0.0, 0.2, 0.4, 0.6),
            num_decoys=NUM_DECOYS,
        )


def test_mask_applied_observed_boundary_narrower_and_nested():
    instance = make_budget_instance(BUILD_SEEDS[0], num_decoys=NUM_DECOYS)
    observation_seed = subseed(BUILD_SEEDS[0], "router-v2-observe")
    boundary = instance.true_target.boundaries[0]
    kept_by_fraction = []
    flipped_by_fraction = []
    for g in NO_ANCHOR_GRID:
        observed = ObservationModel(
            instance.true_target,
            observation_seed,
            mask_fraction=DEFAULT_MASK_FRACTION,
            corrupt_fraction=g,
        ).observe()
        # The observed boundary is strictly narrower than the truth.
        assert observed.complex.boundaries[0].shape[0] == boundary.shape[0]
        assert observed.complex.boundaries[0].shape[1] < boundary.shape[1]
        assert observed.num_masked == round(
            DEFAULT_MASK_FRACTION * boundary.shape[1]
        )
        kept_by_fraction.append(observed.kept_edges)
        flipped_by_fraction.append(set(observed.flipped_entries))
    # The mask draw is fixed per instance: same kept columns at every grid
    # point; corruption is nested per fraction (prefix-growing flip sets).
    assert len(set(kept_by_fraction)) == 1
    for earlier, later in zip(flipped_by_fraction, flipped_by_fraction[1:]):
        assert earlier <= later
    assert len(flipped_by_fraction[0]) < len(flipped_by_fraction[-1])


def test_feature_values_match_masked_corrupted_misfits():
    features, _, metadata = small_dataset()
    for row, meta in enumerate(metadata):
        instance = make_budget_instance(meta.seed, num_decoys=NUM_DECOYS)
        observation_seed = subseed(meta.seed, "router-v2-observe")
        for position, original in enumerate(meta.permutation):
            candidate = instance.candidates[original]
            misfits = [
                observed_misfit(
                    instance.chain_map,
                    ObservationModel(
                        candidate,
                        observation_seed,
                        mask_fraction=DEFAULT_MASK_FRACTION,
                        corrupt_fraction=g,
                    ).observe(),
                )
                for g in NO_ANCHOR_GRID
            ]
            for column, misfit in enumerate(misfits):
                assert np.isclose(
                    np.expm1(features[row, position, column]),
                    misfit,
                    rtol=1e-9,
                    atol=1e-12,
                )
            log_profile = np.log1p(np.asarray(misfits))
            assert np.array_equal(
                features[row, position, GRID_LEN : 2 * GRID_LEN - 1],
                np.diff(log_profile),
            )
            # Structural dims come from the MASKED observed boundary.
            observed_boundary = ObservationModel(
                candidate,
                observation_seed,
                mask_fraction=DEFAULT_MASK_FRACTION,
            ).observe().complex.boundaries[0]
            vertices, edges = observed_boundary.shape
            assert features[row, position, -3] == vertices
            assert features[row, position, -2] == edges
            assert edges < candidate.boundaries[0].shape[1]
            assert features[row, position, -1] == int(
                nullspace_basis(observed_boundary).basis.shape[1]
            )


def test_no_clean_column_anywhere_bookkeeping_not_in_features():
    features, labels, metadata = small_dataset()
    # No exact-zero residual anywhere in the profile block — not even for
    # the true candidate: the features carry no clean anchor.
    profile = np.expm1(features[..., :GRID_LEN])
    assert (features[..., :GRID_LEN] != 0.0).all()
    assert profile.min() > 1e-9
    for row, meta in enumerate(metadata):
        assert profile[row, meta.true_index].min() > 1e-9
    # The eligibility audit that certifies the labels lives only in the
    # bookkeeping: recomputed on the UNDEGRADED instance it separates
    # cleanly, but no feature column has that property.
    for meta in metadata[: len(BUILD_SEEDS)]:
        instance = make_budget_instance(meta.seed, num_decoys=NUM_DECOYS)
        clean = [
            ChainMap(
                instance.source, candidate, instance.chain_map.maps
            ).commutation_residuals()[0]
            for candidate in instance.candidates
        ]
        assert clean[0] <= 1e-9
        assert min(clean[1:]) > 1e-9
        assert labels[metadata.index(meta)] == meta.true_index


def test_true_profile_monotone_nondecreasing_fixed_seeds():
    features, _, metadata = small_dataset()
    for row, meta in enumerate(metadata):
        profile = np.expm1(features[row, meta.true_index, :GRID_LEN])
        # Structural monotonicity (universa.partial): at a fixed mask the
        # nested sign-flips add Frobenius-orthogonal contributions to the
        # squared misfit of the planted map against the true candidate.
        assert (np.diff(profile) >= -1e-9).all()
        # No anchor: the first column is already degraded.
        assert profile[0] > 1e-9


def test_fail_closed_validation():
    with pytest.raises(ValueError):
        build_no_anchor_dataset([], BUILD_FRACTIONS, num_decoys=NUM_DECOYS)
    with pytest.raises(ValueError):
        build_no_anchor_dataset(BUILD_SEEDS, (), num_decoys=NUM_DECOYS)
    with pytest.raises(ValueError):
        build_no_anchor_dataset(BUILD_SEEDS, (1.1,), num_decoys=NUM_DECOYS)
    with pytest.raises(ValueError):
        build_no_anchor_dataset(
            BUILD_SEEDS, (0.45,), num_decoys=NUM_DECOYS
        )  # not a grid point
    with pytest.raises(ValueError):
        build_no_anchor_dataset(
            BUILD_SEEDS, (0.2,), mask_fraction=0.0, num_decoys=NUM_DECOYS
        )  # the regime masks edges
    with pytest.raises(ValueError):
        build_no_anchor_dataset(
            BUILD_SEEDS, (0.2,), mask_fraction=1.5, num_decoys=NUM_DECOYS
        )
    with pytest.raises(ValueError):
        build_no_anchor_dataset(
            BUILD_SEEDS, (0.2,), num_decoys=0
        )  # need K >= 2
    with pytest.raises(ValueError):
        NoAnchorInstanceMetadata(0, 0.2, 1, (1, 1), 0)  # not a permutation
    with pytest.raises(ValueError):
        NoAnchorInstanceMetadata(0, 0.2, 1, (0, 1), 1)  # must locate 0
    with pytest.raises(ValueError):
        NoAnchorInstanceMetadata(0, 1.5, 1, (0, 1), 0)  # bad fraction
    with pytest.raises(ValueError):
        NoAnchorInstanceMetadata(0, 0.2, 0, (0, 1), 0)  # bad threshold
    instance = make_budget_instance(BUILD_SEEDS[0], num_decoys=NUM_DECOYS)
    with pytest.raises(ValueError):
        no_anchor_candidate_features(
            instance.chain_map, instance.true_target, 0, mask_fraction=0.0
        )
    with pytest.raises(ValueError):
        no_anchor_candidate_features(
            "not a chain map", instance.true_target, 0
        )
    features, labels, _ = small_dataset()
    with pytest.raises(ValueError):
        observed_residual_oracle_accuracy(features, labels, 0.45)  # off grid
    with pytest.raises(ValueError):
        observed_residual_oracle_accuracy(features, labels[1:], 0.2)
    with pytest.raises(ValueError):
        mean_profile_heuristic_accuracy(features[..., :-1], labels)
    with pytest.raises(ValueError):
        NoAnchorFractionReport(0.6, 4, 1.5, 0.0, 0.0)  # accuracy > 1
    with pytest.raises(ValueError):
        NoAnchorRegimeReport(
            (1, 2), (2, 3), (0.2,), (0.6,), NO_ANCHOR_GRID, 0.25, 0.5, 0.5,
            (NoAnchorFractionReport(0.6, 1, 0.5, 0.5, 0.5),), {},
        )  # seed overlap


def test_oracle_and_heuristic_column_semantics():
    # Crafted features: oracle reads exactly the operating fraction's
    # column; the heuristic reads the uniform grid mean.
    features = np.ones((1, K, V2_DIM))
    features[0, 2, 0] = 0.0  # column 0 (fraction 0.2): candidate 2 lowest
    features[0, 1, GRID_LEN - 1] = 0.0  # fraction 0.9: candidate 1 lowest
    assert observed_residual_oracle_accuracy(features, [2], 0.2) == 1.0
    assert observed_residual_oracle_accuracy(features, [2], 0.9) == 0.0
    assert observed_residual_oracle_accuracy(features, [1], 0.9) == 1.0
    # Heuristic: candidate 1 has the lowest grid mean without winning
    # column 0: means are 1.0 / 0.6875 / 1.5, column 0 is 1.0 / 2.0 / 1.5.
    features = np.ones((1, K, V2_DIM))
    features[0, 1, :GRID_LEN] = 0.5
    features[0, 1, 0] = 2.0  # loses column 0 but wins the grid mean
    features[0, 2, :GRID_LEN] = 1.5
    assert mean_profile_heuristic_accuracy(features, [1]) == 1.0
    assert observed_residual_oracle_accuracy(features, [0], 0.2) == 1.0
    # On the real dataset the heuristic agrees with a manual mean argmin.
    real_features, real_labels, _ = small_dataset()
    manual = (
        real_features[..., :GRID_LEN].mean(axis=-1).argmin(axis=1)
        == real_labels
    ).mean()
    assert mean_profile_heuristic_accuracy(real_features, real_labels) == float(
        manual
    )


def tiny_no_anchor_protocol(**overrides):
    kwargs = dict(
        train_seeds=(50101, 50102, 50103, 50104),
        eval_seeds=(50201, 50202, 50203),
        train_fractions=(0.2, 0.3, 0.4, 0.5),
        eval_fractions=(0.6, 0.7, 0.8),
        epochs=80,
        lr=3e-3,
        seed=5,
        num_decoys=NUM_DECOYS,
    )
    kwargs.update(overrides)
    return evaluate_no_anchor(**kwargs)


def test_protocol_split_hygiene():
    with pytest.raises(ValueError):
        tiny_no_anchor_protocol(eval_seeds=(50104, 50205))  # seed overlap
    with pytest.raises(ValueError):
        tiny_no_anchor_protocol(eval_fractions=(0.5, 0.6))  # fraction overlap
    with pytest.raises(ValueError):
        tiny_no_anchor_protocol(train_seeds=())  # empty block
    _, report = tiny_no_anchor_protocol()
    assert set(report.train_seeds).isdisjoint(report.eval_seeds)
    assert set(report.train_fractions).isdisjoint(report.eval_fractions)
    assert [row.fraction for row in report.per_fraction] == [0.6, 0.7, 0.8]
    assert all(row.num_instances == 3 for row in report.per_fraction)
    assert report.mask_fraction == DEFAULT_MASK_FRACTION


def test_no_anchor_training_smoke_above_chance():
    _, report = tiny_no_anchor_protocol()
    # Learns above chance 1/K at the train operating fractions.
    assert report.final_train_accuracy > 1.0 / K + 0.2


def test_no_anchor_pipeline_determinism():
    model_a, report_a = tiny_no_anchor_protocol()
    model_b, report_b = tiny_no_anchor_protocol()
    assert report_a == report_b  # history excluded from equality
    for value_a, value_b in zip(
        model_a.state_dict().values(), model_b.state_dict().values()
    ):
        assert torch.equal(value_a, value_b)


def test_full_protocol_report_only():
    train_seeds = tuple(range(50301, 50309))
    eval_seeds = tuple(range(50401, 50405))
    train_fractions = (0.2, 0.3, 0.4, 0.5)
    eval_fractions = (0.6, 0.7, 0.8)  # held-out degradation, no anchor
    _, report = evaluate_no_anchor(
        train_seeds,
        eval_seeds,
        train_fractions,
        eval_fractions,
        epochs=60,
        lr=3e-3,
        seed=11,
        num_decoys=NUM_DECOYS,
    )
    assert len(report.per_fraction) == len(eval_fractions)
    for row in report.per_fraction:
        assert row.num_instances == len(eval_seeds)
        assert 0.0 <= row.learned_accuracy <= 1.0
        assert 0.0 <= row.oracle_accuracy <= 1.0
        assert 0.0 <= row.heuristic_accuracy <= 1.0
    # Deterministic rerun of the whole protocol.
    _, report_again = evaluate_no_anchor(
        train_seeds,
        eval_seeds,
        train_fractions,
        eval_fractions,
        epochs=60,
        lr=3e-3,
        seed=11,
        num_decoys=NUM_DECOYS,
    )
    assert report == report_again
    # Report-only: print the learned-vs-baselines table; no assertion that
    # the learned router beats the polluted oracle — the numbers stand
    # either way.
    for row in report.per_fraction:
        print(
            f"fraction {row.fraction:.1f}: learned={row.learned_accuracy:.4f}"
            f" oracle={row.oracle_accuracy:.4f}"
            f" heuristic={row.heuristic_accuracy:.4f}"
            f" (n={row.num_instances})"
        )


def test_no_sealed_seeds_used():
    used = (
        set(BUILD_SEEDS)
        | set(range(50101, 50105))
        | set(range(50201, 50204))
        | set(range(50301, 50309))
        | set(range(50401, 50405))
    )
    assert used.isdisjoint(SEALED_V1_EVAL_BLOCK)
    assert min(used) >= 50001


def test_plain_import_of_universa_does_not_import_router_v2_or_torch():
    # Run in a fresh interpreter: this test module itself imports torch.
    src = Path(universa.__file__).resolve().parent.parent
    code = (
        "import sys\n"
        "import universa\n"
        "bad = [m for m in sys.modules if m == 'torch' or m.startswith('torch.')]\n"
        "bad += [m for m in ('universa.router', 'universa.router_v2') if m in sys.modules]\n"
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
