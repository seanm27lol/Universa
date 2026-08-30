"""Learned structure routing, v2: the no-anchor regime.

The sealed v1 experiment (``docs/03-router-v1-sealed-1-results.md``) won with
a declared asymmetry: the learned router read each candidate's degradation
profile *anchored at the exact fraction-0 residual* — one clean column —
while the polluted oracle read only the operating fraction's column. The
declared next hardening is removing that clean anchor. This module builds
it, on top of :mod:`universa.router`'s machinery (the same
:class:`~universa.router.StructureRouter`, :func:`~universa.router.train_router`,
annealing, and load-balancing — imported, not reimplemented).

**Regime.** Every candidate boundary is observed through an
:class:`~universa.partial.ObservationModel` with BOTH
``mask_fraction = 0.25`` (edge columns removed) AND ``corrupt_fraction``
swept over a profile grid that EXCLUDES 0.0: 0.2..0.9 by 0.1 (8 points) by
default. There is no clean column anywhere: every profile entry is
``log1p`` of an observed misfit against a degraded operator, and the
structural dims are read off the masked observation too. Ground truth never
enters features: the eligibility audit (true misfit at most
:data:`~universa.router.RESIDUAL_TOL`, every decoy strictly above) is
computed on the UNDEGRADED instance, once per seed, purely as bookkeeping
for seed accounting and label certification.

**Observation sharing.** One observation draw per instance,
``subseed(seed, "router-v2-observe")``. The mask draw is fixed per
instance: ``ObservationModel`` derives the mask permutation from its seed
alone and ``mask_fraction`` is constant, so the same edge columns are
missing at every grid point; corruption is nested per grid fraction exactly
as :mod:`universa.partial` does (prefixes of one master permutation over
the surviving entries). The draw at each (instance, grid point) is shared
across candidates, exactly as :func:`~universa.partial.ranking_study` and
the v1 builder shared theirs. The candidate permutation is likewise per
instance (``subseed(seed, "router-v2-permutation")``). Consequently every
row of one instance carries the SAME permuted feature block; the operating
fraction of a row selects which column the polluted oracle reads and which
side of the train/eval split the row belongs to — it is never a feature.

**Features** (:func:`no_anchor_feature_names`): for a grid of ``G``
fractions, ``2G - 1 + 3`` columns — ``log1p`` of the observed commutation
misfit at every grid point, the profile slopes (first differences), and
the v1 structural dims (``V'``, ``E'``, boundary cycle nullity) read off
the MASKED observed boundary. Unlike v1, where those dims were computed on
the clean candidate (and constant across the equal-sized candidates), the
masked cycle nullity can differ across candidates — masking changes the
cycle rank of the observed operator.

**Baselines**, reported side by side per eval fraction: the polluted
argmin observed-residual oracle (:func:`observed_residual_oracle_accuracy`
— argmin of the single operating-fraction column, polluted at EVERY
fraction now that no clean column exists) and, descriptively, a
non-learned grid-mean profile heuristic
(:func:`mean_profile_heuristic_accuracy` — argmin over candidates of the
uniform average of the ``log1p`` profile across the grid: the simplest
whole-trajectory integration with no learning).

**Protocol** (:func:`evaluate_no_anchor`): train on seed block A x the
train operating fractions {0.2, 0.3, 0.4, 0.5}, evaluate on a DISJOINT
seed block B x the held-out fractions {0.6, 0.7, 0.8} — split hygiene is
fail-closed. The honest question: does the learned router still beat the
polluted oracle with NO exact anchor and partially masked edges? Numbers
are reported per fraction, whichever way they go — a null or a loss is as
publishable internally as a win.

**Import boundary.** Same convention as :mod:`universa.router`: CUDA is
hidden below, then importing this module pulls in torch transitively via
:mod:`universa.router`; ``universa/__init__.py`` does not import this
module, so a plain ``import universa`` stays torch-free. All exact suite
mathematics (observed misfits, nullities) is computed in float64 by the
certified numpy machinery; torch sees only the resulting feature vectors,
in float32, on CPU.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only, before any torch import

from dataclasses import dataclass, field

import numpy as np

from universa.budgets import make_budget_instance
from universa.generators import subseed
from universa.operators import nullspace_basis
from universa.partial import ObservationModel, observed_misfit
from universa.router import (
    DEFAULT_LAMBDA_AUX,
    DEGRADED_STRUCTURAL_NAMES,
    RESIDUAL_TOL,
    StructureRouter,
    hard_accuracy,
    train_router,
)
from universa.structures import ChainMap

NO_ANCHOR_GRID = tuple((i + 2) / 10 for i in range(8))
"""Default no-anchor profile grid: fractions 0.2 to 0.9 by 0.1 (8 points).

The grid EXCLUDES 0.0 by construction: there is no exact observation
anywhere in this regime — the first column is already masked+corrupted.
"""

DEFAULT_MASK_FRACTION = 0.25
"""Default edge-masking fraction: a quarter of every candidate's boundary
columns are unobserved (removed before sign-corruption is applied)."""


def _validated_no_anchor_grid(profile_grid) -> tuple[float, ...]:
    """Fail-closed no-anchor grid validation: nonempty, inside ``(0, 1]``,
    strictly increasing, and excluding 0.0 — the defining constraint of the
    regime is that no grid point is the exact (undegraded) observation, so
    a grid containing 0.0 is an error, never a warning."""
    grid = tuple(float(g) for g in profile_grid)
    if not grid:
        raise ValueError("profile grid must be nonempty")
    if any(g == 0.0 for g in grid):
        raise ValueError(
            "no-anchor regime: the profile grid must exclude 0.0"
        )
    if any(not 0.0 < g <= 1.0 for g in grid):
        raise ValueError("profile grid fractions must lie in (0, 1]")
    if any(b <= a for a, b in zip(grid, grid[1:])):
        raise ValueError("profile grid must be strictly increasing")
    return grid


def _fraction_key(fraction: float) -> str:
    """Deterministic string key for one fraction in subseed components."""
    return f"{fraction:.6f}"


def no_anchor_feature_names(
    profile_grid=NO_ANCHOR_GRID,
) -> tuple[str, ...]:
    """Per-candidate v2 feature layout, in column order (float64 at build).

    For a grid of ``G`` fractions the layout is ``2G - 1 + 3`` columns:
    ``log1p`` of the observed commutation misfit at every grid fraction,
    then the profile slopes (first differences, ``slope_i = profile[i+1] -
    profile[i]``, named by the grid fractions they span), then the
    structural dims :data:`universa.router.DEGRADED_STRUCTURAL_NAMES` read
    off the masked observed boundary.
    """
    grid = _validated_no_anchor_grid(profile_grid)
    profile = tuple(f"log1p_observed_misfit_fraction_{g:.1f}" for g in grid)
    slopes = tuple(
        f"profile_slope_{a:.1f}_to_{b:.1f}" for a, b in zip(grid, grid[1:])
    )
    return profile + slopes + DEGRADED_STRUCTURAL_NAMES


def no_anchor_feature_dim(profile_grid=NO_ANCHOR_GRID) -> int:
    """Feature width of a no-anchor dataset row over ``profile_grid``."""
    return len(no_anchor_feature_names(profile_grid))


def no_anchor_candidate_features(
    chain_map: ChainMap,
    candidate,
    observation_seed: int,
    profile_grid=NO_ANCHOR_GRID,
    mask_fraction: float = DEFAULT_MASK_FRACTION,
) -> np.ndarray:
    """No-anchor degradation-profile feature vector for one candidate.

    Columns follow :func:`no_anchor_feature_names`. The candidate boundary
    is observed through :class:`ObservationModel` draws with BOTH
    ``mask_fraction`` (edge columns removed) and ``corrupt_fraction = g``
    for every grid fraction ``g``, under the shared ``observation_seed`` —
    one draw per (instance, grid point), reused across candidates exactly
    as v1 did. The mask draw is fixed per instance (constant
    ``mask_fraction``, seed-derived permutation), so the same columns are
    missing at every grid point and corruption is nested per fraction.
    Every profile entry is ``log1p`` of
    :func:`universa.partial.observed_misfit` of the planted chain map
    against the observed candidate — polluted at every grid point, with no
    exact anchor anywhere.

    The structural dims (``V'``, ``E'``, boundary cycle nullity) are read
    off the MASKED observed boundary (``corrupt_fraction = 0``, the
    instance-fixed degraded operator): every feature is computed against a
    degraded operator. The masked cycle nullity is certified by
    :func:`universa.operators.nullspace_basis` and can differ across
    candidates, unlike v1's clean dims.
    """
    grid = _validated_no_anchor_grid(profile_grid)
    if not isinstance(chain_map, ChainMap):
        raise ValueError("chain_map must be a ChainMap instance")
    if candidate.top_degree != 1:
        raise ValueError("routing candidates are 1-complexes (graphs)")
    if not 0.0 < mask_fraction <= 1.0:
        raise ValueError("mask_fraction must lie in (0, 1]")
    misfits = [
        observed_misfit(
            chain_map,
            ObservationModel(
                candidate,
                observation_seed,
                mask_fraction=mask_fraction,
                corrupt_fraction=g,
            ).observe(),
        )
        for g in grid
    ]
    profile = np.log1p(np.asarray(misfits, dtype=np.float64))
    slopes = np.diff(profile)
    observed_boundary = ObservationModel(
        candidate, observation_seed, mask_fraction=mask_fraction
    ).observe().complex.boundaries[0]
    cycle_nullity = int(nullspace_basis(observed_boundary).basis.shape[1])
    num_vertices, num_edges = observed_boundary.shape
    return np.concatenate(
        [
            profile,
            slopes,
            np.array(
                [float(num_vertices), float(num_edges), float(cycle_nullity)],
                dtype=np.float64,
            ),
        ]
    )


@dataclass(frozen=True)
class NoAnchorInstanceMetadata:
    """Provenance for one no-anchor dataset row (instance x fraction).

    Same contract as v1's ``DegradedInstanceMetadata``: ``permutation[i]``
    is the original candidate index sitting at permuted position ``i``, so
    ``permutation[true_index] == 0`` always holds and the label can be
    reconstructed from the stored permutation. The permutation is drawn
    once per instance; ``fraction`` is the row's operating fraction (split
    bookkeeping and oracle-column selection, never a feature).
    """

    seed: int
    fraction: float
    threshold: int
    permutation: tuple[int, ...]
    true_index: int

    def __post_init__(self) -> None:
        k = len(self.permutation)
        if k < 2:
            raise ValueError("need at least two candidates")
        if sorted(self.permutation) != list(range(k)):
            raise ValueError("permutation must be a permutation of 0..K-1")
        if not 0 <= self.true_index < k:
            raise ValueError("true_index out of range")
        if self.permutation[self.true_index] != 0:
            raise ValueError("true_index must locate original candidate 0")
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError("fraction outside [0, 1]")
        if self.threshold < 1:
            raise ValueError("bad threshold")


def build_no_anchor_dataset(
    seeds,
    fractions,
    *,
    profile_grid=NO_ANCHOR_GRID,
    mask_fraction: float = DEFAULT_MASK_FRACTION,
    num_vertices: int = 8,
    num_edges: int = 14,
    num_classes: int = 6,
    num_decoys: int = 3,
) -> tuple[np.ndarray, np.ndarray, tuple[NoAnchorInstanceMetadata, ...]]:
    """Certified routing dataset over (seed, operating fraction) rows, with
    no clean anchor anywhere in the features.

    Returns ``(features, labels, metadata)`` with ``features`` of shape
    ``(M, K, F)`` in float64 where ``M = len(seeds) * len(fractions)``,
    ``K = 1 + num_decoys``, and ``F = no_anchor_feature_dim(profile_grid)``;
    ``labels`` holds the permuted true index. Every row of ``fractions``
    must be a point of ``profile_grid`` — the row's fraction is what the
    polluted oracle reads, so it must name a profile column.

    Per instance there is ONE observation draw
    (``subseed(seed, "router-v2-observe")``, shared across candidates, mask
    fixed at ``mask_fraction``, corruption nested per grid fraction) and
    ONE candidate permutation (``subseed(seed, "router-v2-permutation")``),
    so the ``len(fractions)`` rows of an instance repeat the same permuted
    feature block — the operating fraction selects the oracle's column and
    the split, never a feature.

    Fail-closed audits: the eligibility audit runs on the UNDEGRADED
    instance (the true candidate's clean commutation residual must be at
    most ``RESIDUAL_TOL`` and every decoy's strictly above it) — this is
    bookkeeping for seed accounting and label certification only and never
    enters features; the recorded features themselves must be finite. The
    polluted profile values are recorded, never audit-failed.
    """
    grid = _validated_no_anchor_grid(profile_grid)
    seeds = tuple(int(s) for s in seeds)
    fractions = tuple(float(f) for f in fractions)
    if not seeds:
        raise ValueError("seeds must be nonempty")
    if not fractions:
        raise ValueError("fractions must be nonempty")
    for fraction in fractions:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction {fraction} outside [0, 1]")
        if not any(abs(g - fraction) <= 1e-9 for g in grid):
            raise ValueError(
                f"fraction {fraction} is not a profile grid point; the "
                "row fraction must name a profile column"
            )
    if not 0.0 < mask_fraction <= 1.0:
        raise ValueError("mask_fraction must lie in (0, 1]")
    if num_decoys < 1:
        raise ValueError("need at least one decoy (K >= 2)")
    k = num_decoys + 1
    features: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[NoAnchorInstanceMetadata] = []
    for seed in seeds:
        instance = make_budget_instance(
            seed, num_vertices, num_edges, num_classes, num_decoys
        )
        # Eligibility bookkeeping on the UNDEGRADED instance: seed
        # accounting and label certification only — never a feature.
        clean = np.array(
            [
                ChainMap(
                    instance.source, candidate, instance.chain_map.maps
                ).commutation_residuals()[0]
                for candidate in instance.candidates
            ]
        )
        if clean[0] > RESIDUAL_TOL:
            raise RuntimeError(
                f"seed {seed}: true candidate clean misfit "
                f"{clean[0]:.3e} exceeds {RESIDUAL_TOL}"
            )
        if float(clean[1:].min()) <= RESIDUAL_TOL:
            raise RuntimeError(
                f"seed {seed}: a decoy clean misfit "
                f"{float(clean[1:].min()):.3e} is at or below "
                f"{RESIDUAL_TOL}; ground truth is not separable"
            )
        observation_seed = subseed(seed, "router-v2-observe")
        per_candidate = [
            no_anchor_candidate_features(
                instance.chain_map,
                candidate,
                observation_seed,
                grid,
                mask_fraction,
            )
            for candidate in instance.candidates
        ]
        rng = np.random.default_rng(subseed(seed, "router-v2-permutation"))
        permutation = tuple(int(p) for p in rng.permutation(k))
        true_index = permutation.index(0)
        block = np.stack([per_candidate[p] for p in permutation])
        if not np.isfinite(block).all():
            raise RuntimeError(f"seed {seed}: non-finite features")
        for fraction in fractions:
            features.append(block)
            labels.append(true_index)
            metadata.append(
                NoAnchorInstanceMetadata(
                    seed=seed,
                    fraction=fraction,
                    threshold=instance.threshold,
                    permutation=permutation,
                    true_index=true_index,
                )
            )
    return (
        np.stack(features),
        np.asarray(labels, dtype=np.int64),
        tuple(metadata),
    )


def observed_residual_oracle_accuracy(
    features,
    labels,
    fraction: float,
    *,
    profile_grid=NO_ANCHOR_GRID,
) -> float:
    """The polluted oracle: argmin over candidates of the observed misfit.

    Reads the single profile column at ``fraction`` (log1p of the observed
    commutation misfit; log1p is strictly increasing, so this is exactly
    argmin of the misfit) and predicts its argmin — no learning. Unlike v1,
    NO fraction is exact in this regime: the grid excludes 0.0 and every
    observation is masked+corrupted, so the oracle is polluted at every
    operating fraction. This is the reference the learned router is
    honestly compared against at every eval fraction.
    """
    grid = _validated_no_anchor_grid(profile_grid)
    features = np.asarray(features)
    labels = np.asarray(labels)
    expected = no_anchor_feature_dim(grid)
    if features.ndim != 3 or features.shape[-1] != expected:
        raise ValueError(f"features must have shape (M, K, {expected})")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must be one per instance")
    matches = [i for i, g in enumerate(grid) if abs(g - fraction) <= 1e-9]
    if not matches:
        raise ValueError(f"fraction {fraction} is not a profile grid point")
    predictions = features[..., matches[0]].argmin(axis=1)
    return float((predictions == labels).mean())


def mean_profile_heuristic_accuracy(
    features,
    labels,
    *,
    profile_grid=NO_ANCHOR_GRID,
) -> float:
    """Descriptive non-learned baseline: argmin of the grid-mean profile.

    Averages each candidate's ``log1p`` observed-misfit profile uniformly
    over the grid and predicts the argmin — the simplest whole-trajectory
    integration with no learning and no operating-fraction choice. Reported
    descriptively beside the learned router and the myopic polluted oracle;
    it is not the comparison of record.
    """
    grid = _validated_no_anchor_grid(profile_grid)
    features = np.asarray(features)
    labels = np.asarray(labels)
    expected = no_anchor_feature_dim(grid)
    if features.ndim != 3 or features.shape[-1] != expected:
        raise ValueError(f"features must have shape (M, K, {expected})")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must be one per instance")
    predictions = features[..., : len(grid)].mean(axis=-1).argmin(axis=1)
    return float((predictions == labels).mean())


@dataclass(frozen=True)
class NoAnchorFractionReport:
    """Learned-vs-baseline accuracies at one eval fraction."""

    fraction: float
    num_instances: int
    learned_accuracy: float
    oracle_accuracy: float
    heuristic_accuracy: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError("fraction outside [0, 1]")
        if self.num_instances < 1:
            raise ValueError("num_instances must be >= 1")
        for name, value in (
            ("learned_accuracy", self.learned_accuracy),
            ("oracle_accuracy", self.oracle_accuracy),
            ("heuristic_accuracy", self.heuristic_accuracy),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} outside [0, 1]")


@dataclass(frozen=True)
class NoAnchorRegimeReport:
    """Result of a no-anchor held-out-regime evaluation (router v2).

    ``per_fraction`` carries one :class:`NoAnchorFractionReport` per eval
    fraction — the deliverable: the learned router's hard-inference
    accuracy beside the polluted oracle's and the grid-mean heuristic's, at
    each held-out degradation level. ``history`` is the
    :func:`universa.router.train_router` history (excluded from equality:
    it holds numpy arrays).
    """

    train_seeds: tuple[int, ...]
    eval_seeds: tuple[int, ...]
    train_fractions: tuple[float, ...]
    eval_fractions: tuple[float, ...]
    profile_grid: tuple[float, ...]
    mask_fraction: float
    final_train_accuracy: float
    final_eval_accuracy: float
    per_fraction: tuple[NoAnchorFractionReport, ...]
    history: dict = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if set(self.train_seeds) & set(self.eval_seeds):
            raise ValueError("train/eval seed blocks overlap")
        if set(self.train_fractions) & set(self.eval_fractions):
            raise ValueError("train/eval fractions overlap")
        if not self.per_fraction:
            raise ValueError("per_fraction must be nonempty")
        if not 0.0 < self.mask_fraction <= 1.0:
            raise ValueError("mask_fraction must lie in (0, 1]")
        for name, value in (
            ("final_train_accuracy", self.final_train_accuracy),
            ("final_eval_accuracy", self.final_eval_accuracy),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} outside [0, 1]")


def evaluate_no_anchor(
    train_seeds,
    eval_seeds,
    train_fractions,
    eval_fractions,
    *,
    profile_grid=NO_ANCHOR_GRID,
    mask_fraction: float = DEFAULT_MASK_FRACTION,
    epochs: int = 150,
    lr: float = 1e-3,
    seed: int = 0,
    lambda_aux: float = DEFAULT_LAMBDA_AUX,
    tau_start: float = 2.0,
    tau_end: float = 0.25,
    hidden_dim: int = 64,
    num_vertices: int = 8,
    num_edges: int = 14,
    num_classes: int = 6,
    num_decoys: int = 3,
) -> tuple[StructureRouter, NoAnchorRegimeReport]:
    """The v2 protocol: generalization with no exact anchor.

    Trains a :class:`universa.router.StructureRouter`
    (:func:`universa.router.train_router`, annealed, load-balanced,
    deterministic) on the no-anchor dataset over ``train_seeds`` x
    ``train_fractions`` and evaluates on the DISJOINT ``eval_seeds`` x the
    HELD-OUT ``eval_fractions`` — split hygiene is fail-closed: any seed or
    fraction overlap is an error. At every eval fraction the learned
    router's hard-inference (strictly discrete argmax) accuracy is reported
    beside the polluted oracle's
    (:func:`observed_residual_oracle_accuracy` at that fraction) and,
    descriptively, the grid-mean heuristic's
    (:func:`mean_profile_heuristic_accuracy`). The honest v2 question —
    does the learned router still beat the polluted oracle with no exact
    anchor and partially masked edges? — is answered by the numbers in the
    report, whichever way they go.
    """
    train_seeds = tuple(int(s) for s in train_seeds)
    eval_seeds = tuple(int(s) for s in eval_seeds)
    train_fractions = tuple(float(f) for f in train_fractions)
    eval_fractions = tuple(float(f) for f in eval_fractions)
    if not train_seeds or not eval_seeds:
        raise ValueError("train and eval seed blocks must be nonempty")
    if set(train_seeds) & set(eval_seeds):
        raise ValueError("train/eval seed blocks must be disjoint")
    if not train_fractions or not eval_fractions:
        raise ValueError("train and eval fractions must be nonempty")
    if set(train_fractions) & set(eval_fractions):
        raise ValueError("train/eval fractions must be disjoint")
    grid = _validated_no_anchor_grid(profile_grid)
    dim = no_anchor_feature_dim(grid)
    train_data = build_no_anchor_dataset(
        train_seeds,
        train_fractions,
        profile_grid=grid,
        mask_fraction=mask_fraction,
        num_vertices=num_vertices,
        num_edges=num_edges,
        num_classes=num_classes,
        num_decoys=num_decoys,
    )
    eval_data = build_no_anchor_dataset(
        eval_seeds,
        eval_fractions,
        profile_grid=grid,
        mask_fraction=mask_fraction,
        num_vertices=num_vertices,
        num_edges=num_edges,
        num_classes=num_classes,
        num_decoys=num_decoys,
    )
    model, history = train_router(
        train_data,
        eval_data,
        epochs,
        lr=lr,
        seed=seed,
        lambda_aux=lambda_aux,
        tau_start=tau_start,
        tau_end=tau_end,
        hidden_dim=hidden_dim,
        feature_dim=dim,
    )
    features, labels, metadata = eval_data
    rows: list[NoAnchorFractionReport] = []
    for fraction in eval_fractions:
        mask = np.array([m.fraction == fraction for m in metadata])
        block_features, block_labels = features[mask], labels[mask]
        rows.append(
            NoAnchorFractionReport(
                fraction=fraction,
                num_instances=int(mask.sum()),
                learned_accuracy=hard_accuracy(
                    model, block_features, block_labels
                ),
                oracle_accuracy=observed_residual_oracle_accuracy(
                    block_features, block_labels, fraction, profile_grid=grid
                ),
                heuristic_accuracy=mean_profile_heuristic_accuracy(
                    block_features, block_labels, profile_grid=grid
                ),
            )
        )
    report = NoAnchorRegimeReport(
        train_seeds=train_seeds,
        eval_seeds=eval_seeds,
        train_fractions=train_fractions,
        eval_fractions=eval_fractions,
        profile_grid=grid,
        mask_fraction=mask_fraction,
        final_train_accuracy=history["train_accuracy"][-1],
        final_eval_accuracy=history["val_accuracy"][-1],
        per_fraction=tuple(rows),
        history=history,
    )
    return model, report
