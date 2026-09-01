"""The degraded-regime loop: the route-or-discover cycle meets the learned router.

The sealed loop experiment (``docs/21-router-loop-sealed-1-results.md``,
:mod:`universa.loop`) closed the route-or-discover cycle in the CLEAN regime:
the certified commutation residual separates exactly there (true candidate at
exactly 0.0, decoys strictly above 1e-9), so the fit/no-fit ALARM —
``best residual <= 1e-9`` — is a certified decision, not a judgment. This
module is the degraded-regime counterpart: the loop where routing itself
must be learned.

**Regime** (the router-v2 no-anchor regime, lifted into the loop). The
degradation applies to the OBSERVED STRUCTURES — the boundary operators the
model sees — never to the transported data. Every candidate boundary is
observed through an :class:`~universa.partial.ObservationModel` with BOTH
``mask_fraction = 0.25`` (edge columns removed) AND ``corrupt_fraction``
swept over a profile grid that EXCLUDES 0.0 (0.2..0.9 by 0.1, 8 points,
:data:`universa.router_v2.NO_ANCHOR_GRID`): no clean column anywhere, so no
residual the loop reads is exact and the v1 certified alarm cannot separate
— the fit/no-fit alarm MUST be learned. The vector observations
``y = f1 a`` are exact under every regime (the planted chain map commutes
with the true boundary in exact integer arithmetic whatever the model
sees), so the certified discovery head of :mod:`universa.discovery` works
UNCHANGED in the degraded regime: discovery always runs on the exact
transported observations, at the frozen gates (certification 1e-10, novelty
1e-6, router acceptance ``map_misfit <= 1e-9``).

**The four arms** (one row = one seeded instance, one library view, one
shared observation draw — paired across arms):

* :func:`arm_arch_full` — the full system. Candidates are scored by a
  TRAINED :class:`~universa.router.StructureRouter` over the no-anchor
  degradation-profile feature blocks (hard argmax); a TRAINED
  :class:`LearnedAlarm` decides fit/no-fit from the soft gates and the raw
  profile block. Fit -> route to the router's argmax. No-fit -> certified
  discovery on the EXACT transported observations
  (:func:`universa.discovery.discover_constraint` at 1e-10) gated through
  :func:`universa.discovery.admit_to_library` (1e-10, 1e-6); on admission
  the instance routes to the appended structure, else the row is refused.
* :func:`arm_routing_only` — ablation with NO alarm and NO discovery:
  always routes to the router's argmax candidate (forced choice; on
  out-of-library rows it must pick a decoy — an honest failure).
* :func:`arm_discovery_only` — ablation that ALWAYS runs certified
  discovery on the exact observations (the library is unused for routing;
  novelty is checked against the instance's decoy library, exactly the
  :func:`universa.discovery.run_discovery` construction).
* :func:`arm_generic` — the NO-ARCHITECTURE arm: a :class:`GenericMLP`
  over generic spectral features that use NO commutation residual and NO
  degradation profile (see :func:`generic_candidate_features`). Classes
  0..K-1 route to that candidate; class K is "no-fit" and synthesizes
  nothing.

**Correctness semantics** (frozen; :class:`ArmOutcome.correct`):

* in-library — the arm's final structure is the true target: for routing
  arms ``routed_index == 0`` (the generator places the truth at library
  index 0 and this module never permutes — the :mod:`universa.loop`
  convention); for the discovery arm, a certified novel structure admitted
  with ``map_misfit <= 1e-9``.
* out-of-library — a certified novel structure admitted with
  ``map_misfit <= 1e-9``. Routing-only and generic arms can never acquire,
  so they score incorrect there by construction.
* null-control (structure-free observations) — for arms with a discovery
  channel, nothing is admitted (the v1 H4 false-admission control); for
  the alarm-less/generic arms, whose only specificity mechanism is refusal,
  correct iff refused.

**Conditions and views.** The condition of a row is an explicit provenance
declaration (``condition=``), validated fail-closed against the library
view actually passed. ``"out_of_library"`` and ``"null_control"`` require
exactly the ``instance.decoy_targets`` view (the truth withheld).
``"in_library"`` requires the true target at index 0 followed by a PREFIX
of the decoys — the full ``instance.candidates`` view qualifies, and so
does the equal-K paired view this regime needs: the learned alarm and the
generic head have FIXED input widths (K gates, K+1 classes), so the paired
in-/out-of-library rows of this experiment must share one candidate count
K, which forces the in-library view to drop at least one decoy (the
out-of-library view cannot contain the truth). The canonical paired views
of one instance are therefore ``(true_target, *decoy_targets[:-1])`` and
``decoy_targets`` — same K = ``num_decoys``, overlapping decoys, truth at
index 0 in-library and withheld out-of-library. The arm never infers the
condition from residuals — it records and scores it.

**Features** (deterministic, float64, numpy-only):

* :func:`arch_candidate_features` — the router-v2 no-anchor degradation
  profile per candidate: ``log1p`` of the observed commutation misfit at
  each of the 8 grid points (one shared observation draw per (row, grid
  point), reused across candidates; mask fixed, corruption nested per
  fraction), the 7 first-difference slopes, and 3 masked structural dims
  (``V'``, kept edge count, masked cycle nullity) = 18 dims, PLUS the raw
  8-value misfit profile retained for audit and for the alarm's summary
  statistics. The layout is bit-compatible with
  :func:`universa.router_v2.no_anchor_candidate_features`.
* :func:`generic_candidate_features` — the no-architecture arm's input:
  spectral statistics of the observed boundary (same shared draw, at the
  row's operating grid point) and of the EXACT transported observation
  matrix ``Y`` — provably no commutation residual and no profile (see the
  function docstring for the fixed 18-column layout).
* :func:`alarm_features` — the :class:`LearnedAlarm` input: the K soft
  gate values (trained router, tau = 1.0), the gate entropy (nats),
  ``log1p`` of the minimum raw profile value across candidates, and
  ``log1p`` of the maximum profile slope magnitude.

**The alarm-v2 redesign (additive — the v1 alarm above is frozen and
unchanged).** The sealed loop-v2 experiment
(``docs/24-router-loop-v2-sealed-1-results.md``) found the v1 alarm's two
error modes: false QUIETS (fit declared on out-of-library rows — no
discovery runs, the acquisition is lost) and false ALARMS (no-fit
declared on in-library rows — discovery correctly refuses as non-novel,
the accuracy is lost). The v1 layout cannot see "all decoys look alike"
(small margins) vs "one candidate stands out" (a large margin). The
redesign is purely additive: :func:`alarm_features_v2` appends five
margin features to the frozen v1 layout (bit-exact prefix),
:class:`LearnedAlarmV2` reads the widened ``K + 8`` vector,
:func:`train_alarm_v2` trains it under the v1 recipe,
:func:`calibrate_threshold` picks the decision threshold on the TRAIN
block by maximizing balanced accuracy subject to a frozen false-quiet
bound, :func:`alarm_decision_v2` applies the calibrated threshold, and
:func:`arm_arch_full_v2` runs the full loop with the v2 alarm under the
module's frozen correctness semantics (the ``detail`` strings carry an
``alarm_v2=`` marker for provenance). Nothing certified changes:
discovery, admission, the router-acceptance gate, and the correctness
semantics are exactly the v1 paths.

**Import boundary and torch discipline.** Same convention as
:mod:`universa.router`: CUDA is hidden below, then this module imports
torch; ``universa/__init__.py`` does not import this module, so a plain
``import universa`` stays torch-free. Torch appears ONLY in the
learned-model section (the :class:`LearnedAlarm`/:class:`GenericMLP`
definitions, their training and decision functions, and the router
gate/argmax helpers) — CPU, float32, ``torch.manual_seed``, full-batch
Adam (no shuffling RNG), train-only standardization stored in buffers.
Everything else — the feature builders, the certified discovery path, the
arm control flow, and the outcome records — is certified float64 numpy or
plain Python. Determinism: every draw keys on :func:`universa.generators.subseed`
and every learned model on its frozen training seed, so two runs of any
arm on the same row produce bit-identical :class:`ArmOutcome` records.

Documented non-features: the arms never read the withheld truth (the
condition is declared provenance, and correctness reads only the outcome
record and the index-0 convention); the grown library is not returned
(``ArmOutcome`` is a pure scalar record, so two runs compare bit-identical
with plain equality — the :mod:`universa.loop` retention split); no
anti-leak permutation (the experiment runner's concern, as in v1).
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only, before any torch import

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from universa.budgets import BudgetInstance
from universa.discovery import (
    DEFAULT_NOVELTY_TOL,
    DiscoveryInsufficient,
    admit_to_library,
    discover_constraint,
)
from universa.generators import subseed
from universa.operators import CERT_TOL, nullspace_basis
from universa.partial import ObservationModel, observed_misfit
from universa.router import RESIDUAL_TOL, StructureRouter
from universa.router_v2 import DEFAULT_MASK_FRACTION, NO_ANCHOR_GRID
from universa.structures import ChainComplex, ChainMap

MAP_ACCEPT_TOL = RESIDUAL_TOL
"""Router-acceptance threshold for a discovered constraint (1e-9).

Exactly ``universa.router.RESIDUAL_TOL``, re-pinned as
``universa.loop.ALARM_TOL`` in the clean-regime loop: the planted map's
misfit ``||C_disc f1 Z||_F`` against the discovered constraint must pass
it for the discovered structure to count as an acquisition.
"""

ALARM_THRESHOLD = 0.5
"""Frozen decision threshold of the learned alarm (sigmoid probability)."""

CONDITIONS = ("in_library", "out_of_library", "null_control")
"""The paired row conditions (declared provenance, view-validated)."""

ARMS = ("arch_full", "routing_only", "discovery_only", "generic")
"""The four arm names recorded on :class:`ArmOutcome`."""

_DISCOVERY_MISFIT_TOL = CERT_TOL
"""Certification gate of the discovery path (1e-10), the loop convention."""


# ---------------------------------------------------------------------------
# Feature builders (deterministic, float64, numpy-only — no torch here).
# ---------------------------------------------------------------------------


def _validated_profile_grid(profile_grid) -> tuple[float, ...]:
    """Fail-closed no-anchor grid validation: nonempty, inside ``(0, 1]``,
    strictly increasing, and excluding 0.0 — the defining constraint of the
    degraded regime is that no grid point is the exact observation."""
    grid = tuple(float(g) for g in profile_grid)
    if not grid:
        raise ValueError("profile grid must be nonempty")
    if any(g == 0.0 for g in grid):
        raise ValueError("no-anchor regime: the profile grid must exclude 0.0")
    if any(not 0.0 < g <= 1.0 for g in grid):
        raise ValueError("profile grid fractions must lie in (0, 1]")
    if any(b <= a for a, b in zip(grid, grid[1:])):
        raise ValueError("profile grid must be strictly increasing")
    return grid


def _validate_candidate(chain_map, candidate, mask_fraction: float) -> None:
    """Shared fail-closed checks of the per-candidate feature builders."""
    if not isinstance(chain_map, ChainMap):
        raise ValueError("chain_map must be a ChainMap instance")
    if not isinstance(candidate, ChainComplex) or candidate.top_degree != 1:
        raise ValueError("candidates are 1-complexes (graphs)")
    if not 0.0 < float(mask_fraction) <= 1.0:
        raise ValueError("mask_fraction must lie in (0, 1]")


def arch_candidate_features(
    chain_map: ChainMap,
    candidate: ChainComplex,
    observation_seed: int,
    profile_grid=NO_ANCHOR_GRID,
    mask_fraction: float = DEFAULT_MASK_FRACTION,
) -> tuple[np.ndarray, tuple[float, ...]]:
    """No-anchor degradation-profile features for one candidate, plus the raw profile.

    Returns ``(features, raw_profile)``. ``features`` is the 18-dim float64
    vector of the router-v2 no-anchor layout (bit-compatible with
    :func:`universa.router_v2.no_anchor_candidate_features` at the same
    observation seed, grid, and mask): ``log1p`` of the observed commutation
    misfit (:func:`universa.partial.observed_misfit` of the planted chain
    map against the observed candidate) at every grid fraction, then the
    ``G - 1`` profile slopes (first differences), then the 3 masked
    structural dims — ``V'``, the kept edge count, and the masked boundary
    cycle nullity — read off the masked observed boundary
    (``corrupt_fraction = 0``). ``raw_profile`` is the tuple of the ``G``
    raw misfit values (pre-``log1p``), retained for audit and for the
    alarm's summary statistics (:func:`alarm_features`).

    The candidate boundary is observed through
    :class:`~universa.partial.ObservationModel` draws with BOTH
    ``mask_fraction`` and ``corrupt_fraction = g`` for every grid fraction
    ``g``, under the shared ``observation_seed``: one draw per (row, grid
    point), reused across candidates (the caller passes one seed per row);
    the mask draw is fixed per seed so the same columns are missing at
    every grid point and corruption is nested per fraction. No grid point
    is exact — the grid excludes 0.0 by construction.
    """
    grid = _validated_profile_grid(profile_grid)
    _validate_candidate(chain_map, candidate, mask_fraction)
    observation_seed = int(observation_seed)
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
    raw_profile = tuple(float(m) for m in misfits)
    profile = np.log1p(np.asarray(raw_profile, dtype=np.float64))
    slopes = np.diff(profile)
    observed_boundary = (
        ObservationModel(candidate, observation_seed, mask_fraction=mask_fraction)
        .observe()
        .complex.boundaries[0]
    )
    cycle_nullity = int(nullspace_basis(observed_boundary).basis.shape[1])
    num_vertices, num_edges = observed_boundary.shape
    features = np.concatenate(
        [
            profile,
            slopes,
            np.array(
                [float(num_vertices), float(num_edges), float(cycle_nullity)],
                dtype=np.float64,
            ),
        ]
    )
    return features, raw_profile


GENERIC_FEATURE_NAMES = (
    "observed_num_vertices",  # V' — rows of the observed boundary
    "observed_kept_edges",  # columns of the observed boundary (mask 0.25)
    "num_observations",  # M — columns of the exact observation matrix Y
    "log1p_sv_observations_1",  # singular values of Y, largest first, log1p
    "log1p_sv_observations_2",  # (zero-padded when fewer than four exist)
    "log1p_sv_observations_3",
    "log1p_sv_observations_4",
    "log1p_sv_observed_boundary_1",  # singular values of the observed boundary
    "log1p_sv_observed_boundary_2",
    "log1p_sv_observed_boundary_3",
    "log1p_sv_observed_boundary_4",
    "log1p_condition_observations",  # sigma_0 / sigma_{rank-1} of Y, log1p
    "log1p_condition_observed_boundary",  # same for the observed boundary
    "log1p_frobenius_energy_observations",  # log1p(||Y||_F^2)
    "log1p_frobenius_energy_observed_boundary",  # log1p(||B_obs||_F^2)
    "rank_estimate_observations",  # numerical rank of Y (operators.py rule)
    "rank_estimate_observed_boundary",  # numerical rank of the boundary
    "observed_cycle_nullity",  # kept edges - boundary rank (masked nullity)
)
"""The generic arm's fixed 18-column feature layout, in column order."""

GENERIC_FEATURE_DIM = len(GENERIC_FEATURE_NAMES)


def _numerical_rank(matrix: np.ndarray) -> int:
    """Numerical rank under the operators.py SVD tolerance convention
    (``max(shape) * eps * sigma_0``), the :mod:`universa.discovery` rule."""
    singulars = np.linalg.svd(matrix, compute_uv=False)
    if singulars.size == 0:
        return 0
    rank_tol = max(matrix.shape) * np.finfo(float).eps * singulars[0]
    return int((singulars > rank_tol).sum())


def _first_singulars_log1p(matrix: np.ndarray, count: int = 4) -> np.ndarray:
    """``log1p`` of the ``count`` largest singular values, zero-padded."""
    singulars = np.linalg.svd(matrix, compute_uv=False)
    padded = np.zeros(count, dtype=np.float64)
    take = min(count, singulars.size)
    if take:
        padded[:take] = singulars[:take]
    return np.log1p(padded)


def _rank_condition(matrix: np.ndarray, rank: int) -> float:
    """Condition number over the certified numerical rank:
    ``sigma_0 / sigma_{rank-1}``; exactly 1.0 when the rank is zero."""
    if rank < 1:
        return 1.0
    singulars = np.linalg.svd(matrix, compute_uv=False)
    return float(singulars[0] / singulars[rank - 1])


def generic_candidate_features(
    candidate: ChainComplex,
    observations_y: np.ndarray,
    observation_seed: int,
    profile_grid,
    mask_fraction: float = DEFAULT_MASK_FRACTION,
) -> np.ndarray:
    """Generic spectral features for one candidate — the NO-ARCHITECTURE input.

    Returns the fixed 18-dim float64 vector of
    :data:`GENERIC_FEATURE_NAMES`. The candidate boundary is observed under
    the SAME shared draw the arch arm uses
    (:class:`~universa.partial.ObservationModel` with ``mask_fraction``
    and corruption at the row's operating grid point — ``profile_grid``
    must be exactly that 1-point grid, so the generic arm reads one
    degradation level, for parity with the arch arm's degradation family),
    and ``observations_y`` is the EXACT transported observation matrix
    ``Y = f1 A`` (columns the transported vectors). The columns: the three
    dims (``V'``, kept edge count, observation count), the first four
    singular values of ``Y`` (``log1p``, zero-padded), the first four of
    the observed boundary (``log1p``), the two rank-restricted condition
    numbers (``log1p``), the two Frobenius energies (``log1p`` of the
    squared norms), and the rank estimates (numerical rank of ``Y``,
    numerical rank of the observed boundary, and its cycle nullity), all
    under the operators.py SVD tolerance convention.

    BY CONSTRUCTION this builder uses NO commutation residual and NO
    degradation profile: it never receives the chain map, never calls
    :func:`universa.partial.observed_misfit` or
    :meth:`~universa.structures.ChainMap.commutation_residuals`, and never
    sweeps a profile — the observed boundary enters only through raw
    spectral statistics. The tests assert this structurally (source
    inspection) and behaviorally (the misfit machinery may be sabotaged
    without changing this builder's output).
    """
    grid = _validated_profile_grid(profile_grid)
    if len(grid) != 1:
        raise ValueError(
            "the generic builder reads exactly one degradation level: pass "
            "the row's operating grid point as a 1-point profile grid"
        )
    if not isinstance(candidate, ChainComplex) or candidate.top_degree != 1:
        raise ValueError("candidates are 1-complexes (graphs)")
    if not 0.0 < float(mask_fraction) <= 1.0:
        raise ValueError("mask_fraction must lie in (0, 1]")
    boundary = candidate.boundaries[0]
    if not isinstance(observations_y, np.ndarray) or observations_y.ndim != 2:
        raise ValueError("observations_y must be a 2-D array (columns are y_j)")
    if not np.all(np.isfinite(observations_y)):
        raise ValueError("observations_y must be finite")
    if observations_y.shape[0] != boundary.shape[1]:
        raise ValueError(
            f"observations_y have ambient dimension {observations_y.shape[0]}, "
            f"expected {boundary.shape[1]} (the candidate's edge count)"
        )
    observed_boundary = (
        ObservationModel(
            candidate,
            int(observation_seed),
            mask_fraction=mask_fraction,
            corrupt_fraction=grid[0],
        )
        .observe()
        .complex.boundaries[0]
    )
    y_rank = _numerical_rank(observations_y)
    b_rank = _numerical_rank(observed_boundary)
    num_vertices, kept_edges = observed_boundary.shape
    return np.concatenate(
        [
            np.array(
                [
                    float(num_vertices),
                    float(kept_edges),
                    float(observations_y.shape[1]),
                ],
                dtype=np.float64,
            ),
            _first_singulars_log1p(observations_y),
            _first_singulars_log1p(observed_boundary),
            np.log1p(
                np.array(
                    [
                        _rank_condition(observations_y, y_rank),
                        _rank_condition(observed_boundary, b_rank),
                        float(np.linalg.norm(observations_y) ** 2),
                        float(np.linalg.norm(observed_boundary) ** 2),
                    ],
                    dtype=np.float64,
                )
            ),
            np.array(
                [
                    float(y_rank),
                    float(b_rank),
                    float(kept_edges - b_rank),
                ],
                dtype=np.float64,
            ),
        ]
    )


def operating_grid_point(seed: int, profile_grid=NO_ANCHOR_GRID) -> float:
    """The row's operating grid point, derived deterministically from the
    row seed: ``grid[subseed(seed, "loop-v2-operating") % len(grid)]``.

    The generic arm observes its boundary at this single corruption
    fraction (the arch arm's profile spans the whole grid, so the point
    selects nothing there); deriving it per row spreads the generic arm's
    rows across degradation levels while keeping every arm of a row paired
    on the same instance, library view, and observation draw.
    """
    grid = _validated_profile_grid(profile_grid)
    index = int(subseed(int(seed), "loop-v2-operating")) % len(grid)
    return grid[index]


def alarm_features(gates, profile_block) -> np.ndarray:
    """The :class:`LearnedAlarm` input vector from the router gates and the
    raw profile block: ``[K gates, gate entropy (nats), log1p(min raw
    profile value across candidates), log1p(max profile slope magnitude)]``
    — ``K + 3`` dims, float64.

    ``gates`` are the trained router's SOFT gates at ``tau = 1.0`` (they
    must be nonnegative and sum to 1, checked fail-closed);
    ``profile_block`` is the ``(K, G)`` block of raw observed-misfit
    profiles retained by :func:`arch_candidate_features`. The slopes are
    first differences of the ``log1p`` profile per candidate (the same
    transform the feature blocks carry).
    """
    gates = np.asarray(gates, dtype=np.float64)
    block = np.asarray(profile_block, dtype=np.float64)
    if gates.ndim != 1 or gates.size < 1:
        raise ValueError("gates must be a nonempty 1-D array")
    if not np.all(np.isfinite(gates)) or (gates < 0.0).any():
        raise ValueError("gates must be finite and nonnegative")
    if abs(float(gates.sum()) - 1.0) > 1e-4:
        raise ValueError("gates must sum to 1 (soft router gates)")
    if block.ndim != 2 or block.shape[0] != gates.size:
        raise ValueError(
            "profile_block must have shape (K, G) with K = len(gates)"
        )
    if block.shape[1] < 1:
        raise ValueError("profile_block must have at least one grid column")
    if not np.all(np.isfinite(block)) or (block < 0.0).any():
        raise ValueError("raw misfit profiles must be finite and nonnegative")
    entropy = float(-(gates * np.log(np.clip(gates, 1e-12, None))).sum())
    min_profile = float(block.min())
    slopes = np.diff(np.log1p(block), axis=1)
    max_slope = float(np.abs(slopes).max()) if slopes.size else 0.0
    return np.concatenate(
        [gates, np.array([entropy, np.log1p(min_profile), np.log1p(max_slope)])]
    )


ALARM_V2_MARGIN_FEATURE_NAMES = (
    "log1p_profile_margin_second_minus_best",  # second-smallest minus smallest raw profile value
    "gate_top2_gap",  # largest minus second-largest soft gate
    "log1p_mean_profile",  # mean raw profile value across the block
    "best_profile_curvature",  # mean |second difference| of the best candidate's log1p profile
    "log1p_best_profile",  # absolute level of the smallest raw profile value
)
"""The five margin-feature columns :func:`alarm_features_v2` appends to the
frozen v1 :func:`alarm_features` layout, in column order (K+3 .. K+7)."""


def alarm_features_v2(gates, profile_block) -> np.ndarray:
    """The :class:`LearnedAlarmV2` input vector: the frozen v1
    :func:`alarm_features` layout (``K + 3`` dims, reused unchanged — the
    additive contract is bit-exact) followed by the five margin features of
    :data:`ALARM_V2_MARGIN_FEATURE_NAMES` (``K + 8`` dims total), all
    computed from the raw ``(K, G)`` profile block and the gates:

    * ``log1p_profile_margin_second_minus_best`` — ``log1p`` of the
      second-smallest minus the smallest raw profile value over the whole
      block (nonnegative by construction): the "all decoys look alike"
      margin — small when no candidate stands out, large when one does.
    * ``gate_top2_gap`` — the largest minus the second-largest soft gate,
      in ``[0, 1]`` (exactly 1.0 for a single candidate: the top gate
      minus 0.0).
    * ``log1p_mean_profile`` — ``log1p`` of the mean raw profile value
      across candidates and grid points: the block's overall misfit level.
    * ``best_profile_curvature`` — the mean absolute second difference of
      the best (minimum) candidate's ``log1p`` profile across the grid —
      the candidate row holding the block minimum, first row index on
      ties; exactly 0.0 when the grid has fewer than three points (the v1
      degenerate-difference convention).
    * ``log1p_best_profile`` — ``log1p`` of the smallest raw profile value
      (the block's absolute best-misfit level).

    Same fail-closed validation as :func:`alarm_features` (which runs
    first and supplies the leading ``K + 3`` columns), plus one v2 check:
    the block must hold at least two profile values for the
    second-minus-best margin.
    """
    gates = np.asarray(gates, dtype=np.float64)
    block = np.asarray(profile_block, dtype=np.float64)
    base = alarm_features(gates, block)  # the frozen v1 layout, validated
    if block.size < 2:
        raise ValueError(
            "alarm_features_v2 needs at least two profile values for the "
            "second-minus-best margin"
        )
    ordered = np.sort(block, axis=None)
    margin = float(ordered[1] - ordered[0])
    sorted_gates = np.sort(gates)[::-1]
    top2_gap = float(
        sorted_gates[0] - (sorted_gates[1] if gates.size > 1 else 0.0)
    )
    best_row = np.log1p(
        block[np.unravel_index(block.argmin(), block.shape)[0]]
    )
    second_differences = np.diff(best_row, n=2)
    curvature = (
        float(np.abs(second_differences).mean())
        if second_differences.size
        else 0.0
    )
    return np.concatenate(
        [
            base,
            np.array(
                [
                    np.log1p(margin),
                    top2_gap,
                    float(np.log1p(block.mean())),
                    curvature,
                    float(np.log1p(block.min())),
                ]
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Learned models (torch, CPU, float32, deterministic). This is the ONLY
# section of the module that touches torch.
# ---------------------------------------------------------------------------


def _as_float32(features) -> torch.Tensor:
    return torch.as_tensor(np.asarray(features), dtype=torch.float32)


def router_gates(router: StructureRouter, feature_block) -> np.ndarray:
    """The trained router's SOFT gates at ``tau = 1.0`` over one row's
    ``(K, F)`` feature block — the alarm's gate input. Deterministic."""
    if not isinstance(router, StructureRouter):
        raise ValueError("router must be a universa.router.StructureRouter")
    block = np.asarray(feature_block, dtype=np.float64)
    if block.ndim != 2 or block.shape[1] != router.feature_dim:
        raise ValueError(
            f"feature_block must have shape (K, {router.feature_dim})"
        )
    if not np.all(np.isfinite(block)):
        raise ValueError("feature_block must be finite")
    router.eval()
    with torch.no_grad():
        gates = router(_as_float32(block).unsqueeze(0), tau=1.0, hard=False)
    return gates.squeeze(0).cpu().numpy().astype(np.float64)


def router_argmax(router: StructureRouter, feature_block) -> int:
    """The trained router's hard decision: argmax over candidate logits,
    first-index tie-breaking — strictly discrete inference."""
    if not isinstance(router, StructureRouter):
        raise ValueError("router must be a universa.router.StructureRouter")
    block = np.asarray(feature_block, dtype=np.float64)
    if block.ndim != 2 or block.shape[1] != router.feature_dim:
        raise ValueError(
            f"feature_block must have shape (K, {router.feature_dim})"
        )
    if not np.all(np.isfinite(block)):
        raise ValueError("feature_block must be finite")
    router.eval()
    with torch.no_grad():
        logits = router.logits(_as_float32(block).unsqueeze(0))
    return int(logits.squeeze(0).argmax(dim=-1).item())


class LearnedAlarm(nn.Module):
    """The learned fit/no-fit alarm: ``Linear(K+3, 32) GELU Linear(32, 1)``.

    Input: :func:`alarm_features` — the K soft gate values (trained router,
    ``tau = 1.0``), the gate entropy in nats, ``log1p`` of the minimum raw
    profile value across candidates, and ``log1p`` of the maximum profile
    slope magnitude. Output: one logit; the decision is
    ``sigmoid(logit) >= ALARM_THRESHOLD`` (0.5, frozen) — "fits" True/False.

    The train-only standardization statistics (measured by
    :func:`train_alarm` on the training rows) live in buffers, so the model
    is self-contained: raw float features in, logit out.
    """

    def __init__(self, num_candidates: int, hidden_dim: int = 32):
        super().__init__()
        if int(num_candidates) < 1 or int(hidden_dim) < 1:
            raise ValueError("num_candidates and hidden_dim must be >= 1")
        self.num_candidates = int(num_candidates)
        self.hidden_dim = int(hidden_dim)
        self.input_dim = self.num_candidates + 3
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.register_buffer("input_mean", torch.zeros(self.input_dim))
        self.register_buffer("input_std", torch.ones(self.input_dim))

    def logit(self, features: torch.Tensor) -> torch.Tensor:
        """The alarm logit, shape ``(B,)`` from ``(B, K+3)``."""
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"features must have shape (B, {self.input_dim})"
            )
        standardized = (features - self.input_mean) / self.input_std
        return self.network(standardized).squeeze(-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.logit(features)


def train_alarm(
    model_factory,
    rows_fit,
    rows_nofit,
    epochs: int = 150,
    lr: float = 1e-3,
    seed: int = 4243,
) -> tuple[LearnedAlarm, dict]:
    """Train a :class:`LearnedAlarm`, deterministically, on CPU.

    ``model_factory`` is a zero-argument callable returning a fresh
    :class:`LearnedAlarm`; ``torch.manual_seed(seed)`` runs immediately
    before construction, so initialization is governed by the frozen seed.
    ``rows_fit`` / ``rows_nofit`` are ``(M, K+3)`` arrays of
    :func:`alarm_features` vectors — in-library rows are labeled fit (1),
    out-of-library rows no-fit (0). Binary cross-entropy (on the logit),
    full-batch Adam (every epoch is one step; no shuffling RNG exists).
    Standardization statistics are measured on the TRAINING rows only and
    stored in the model's buffers. The decision threshold is the frozen
    :data:`ALARM_THRESHOLD` (0.5), never tuned. History records the loss
    and the thresholded train accuracy per epoch; a non-finite loss is an
    error, never a warning.
    """
    rows_fit = np.asarray(rows_fit, dtype=np.float64)
    rows_nofit = np.asarray(rows_nofit, dtype=np.float64)
    for name, rows in (("rows_fit", rows_fit), ("rows_nofit", rows_nofit)):
        if rows.ndim != 2 or rows.shape[0] < 1:
            raise ValueError(f"{name} must be a nonempty 2-D array")
        if not np.all(np.isfinite(rows)):
            raise ValueError(f"{name} must be finite")
    if rows_fit.shape[1] != rows_nofit.shape[1]:
        raise ValueError("rows_fit/rows_nofit width mismatch")
    if int(epochs) < 1:
        raise ValueError("epochs must be >= 1")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if not callable(model_factory):
        raise ValueError("model_factory must be callable")
    torch.manual_seed(int(seed))
    model = model_factory()
    if not isinstance(model, LearnedAlarm):
        raise ValueError("model_factory must return a LearnedAlarm")
    if model.input_dim != rows_fit.shape[1]:
        raise ValueError(
            f"alarm input dim {model.input_dim} does not match the rows' "
            f"width {rows_fit.shape[1]} (K gates + 3)"
        )
    x = _as_float32(np.vstack([rows_fit, rows_nofit]))
    y = torch.cat(
        [torch.ones(rows_fit.shape[0]), torch.zeros(rows_nofit.shape[0])]
    )
    model.input_mean.copy_(x.mean(dim=0))
    model.input_std.copy_(x.std(dim=0, unbiased=False).clamp_min(1e-8))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list] = {"loss": [], "train_accuracy": []}
    for epoch in range(int(epochs)):
        model.train()
        logits = model.logit(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
        if not np.isfinite(loss_value):
            raise FloatingPointError(
                f"non-finite training loss at epoch {epoch}"
            )
        model.eval()
        with torch.no_grad():
            predictions = (
                torch.sigmoid(model.logit(x)) >= ALARM_THRESHOLD
            ).float()
        history["loss"].append(loss_value)
        history["train_accuracy"].append(float((predictions == y).double().mean()))
    return model, history


def alarm_decision(
    model: LearnedAlarm, gates, profile_block
) -> bool:
    """The alarm's decision: True = "fits" (route), False = "no-fit"
    (discover). Sigmoid of the logit against the frozen
    :data:`ALARM_THRESHOLD`; deterministic."""
    if not isinstance(model, LearnedAlarm):
        raise ValueError("model must be a LearnedAlarm")
    features = alarm_features(gates, profile_block)
    if features.shape[0] != model.input_dim:
        raise ValueError(
            f"alarm input dim {features.shape[0]} does not match the "
            f"model's {model.input_dim}"
        )
    model.eval()
    with torch.no_grad():
        probability = float(
            torch.sigmoid(model.logit(_as_float32(features).unsqueeze(0)))
            .squeeze()
            .item()
        )
    return bool(probability >= ALARM_THRESHOLD)


class LearnedAlarmV2(nn.Module):
    """The v2 learned fit/no-fit alarm: ``Linear(K+8, 32) GELU Linear(32, 1)``.

    Same shape as :class:`LearnedAlarm` but over the
    :func:`alarm_features_v2` layout (the frozen v1 ``K + 3`` columns plus
    the five margin columns). Output: one logit; the decision is
    ``sigmoid(logit) >= threshold`` with the threshold supplied EXPLICITLY
    (calibrated on the train block by :func:`calibrate_threshold` — the v2
    design freezes no 0.5).

    The train-only standardization statistics (measured by
    :func:`train_alarm_v2` on the training rows) live in buffers, so the
    model is self-contained: raw float features in, logit out.
    """

    def __init__(self, num_candidates: int, hidden_dim: int = 32):
        super().__init__()
        if int(num_candidates) < 1 or int(hidden_dim) < 1:
            raise ValueError("num_candidates and hidden_dim must be >= 1")
        self.num_candidates = int(num_candidates)
        self.hidden_dim = int(hidden_dim)
        self.input_dim = self.num_candidates + 8
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.register_buffer("input_mean", torch.zeros(self.input_dim))
        self.register_buffer("input_std", torch.ones(self.input_dim))

    def logit(self, features: torch.Tensor) -> torch.Tensor:
        """The alarm logit, shape ``(B,)`` from ``(B, K+8)``."""
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(
                f"features must have shape (B, {self.input_dim})"
            )
        standardized = (features - self.input_mean) / self.input_std
        return self.network(standardized).squeeze(-1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.logit(features)


def train_alarm_v2(
    model: LearnedAlarmV2,
    rows_fit,
    rows_nofit,
    epochs: int = 150,
    lr: float = 1e-3,
    seed: int = 4243,
) -> tuple[LearnedAlarmV2, dict]:
    """Train a :class:`LearnedAlarmV2`, deterministically, on CPU.

    The :func:`train_alarm` recipe over the v2 features: ``rows_fit`` /
    ``rows_nofit`` are ``(M, K+8)`` arrays of :func:`alarm_features_v2`
    vectors — in-library rows are labeled fit (1), out-of-library rows
    no-fit (0). Binary cross-entropy (on the logit), full-batch Adam
    (every epoch is one step; no shuffling RNG exists). Standardization
    statistics are measured on the TRAINING rows only and stored in the
    model's buffers. ``torch.manual_seed(seed)`` runs before the
    optimization loop; construct ``model`` itself under a manual seed for
    deterministic initialization (the caller's concern — the model is
    passed in, exactly the :func:`train_generic` convention, unlike
    :func:`train_alarm`'s factory). The decision threshold is NOT set
    here: it is calibrated afterwards by :func:`calibrate_threshold`;
    the history's train accuracy is a diagnostic at the 0.5 score. A
    non-finite loss is an error, never a warning.
    """
    rows_fit = np.asarray(rows_fit, dtype=np.float64)
    rows_nofit = np.asarray(rows_nofit, dtype=np.float64)
    for name, rows in (("rows_fit", rows_fit), ("rows_nofit", rows_nofit)):
        if rows.ndim != 2 or rows.shape[0] < 1:
            raise ValueError(f"{name} must be a nonempty 2-D array")
        if not np.all(np.isfinite(rows)):
            raise ValueError(f"{name} must be finite")
    if rows_fit.shape[1] != rows_nofit.shape[1]:
        raise ValueError("rows_fit/rows_nofit width mismatch")
    if int(epochs) < 1:
        raise ValueError("epochs must be >= 1")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if not isinstance(model, LearnedAlarmV2):
        raise ValueError("model must be a LearnedAlarmV2")
    if model.input_dim != rows_fit.shape[1]:
        raise ValueError(
            f"alarm input dim {model.input_dim} does not match the rows' "
            f"width {rows_fit.shape[1]} (K gates + 8)"
        )
    torch.manual_seed(int(seed))
    x = _as_float32(np.vstack([rows_fit, rows_nofit]))
    y = torch.cat(
        [torch.ones(rows_fit.shape[0]), torch.zeros(rows_nofit.shape[0])]
    )
    model.input_mean.copy_(x.mean(dim=0))
    model.input_std.copy_(x.std(dim=0, unbiased=False).clamp_min(1e-8))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list] = {"loss": [], "train_accuracy": []}
    for epoch in range(int(epochs)):
        model.train()
        logits = model.logit(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
        if not np.isfinite(loss_value):
            raise FloatingPointError(
                f"non-finite training loss at epoch {epoch}"
            )
        model.eval()
        with torch.no_grad():
            predictions = (
                torch.sigmoid(model.logit(x)) >= ALARM_THRESHOLD
            ).float()
        history["loss"].append(loss_value)
        history["train_accuracy"].append(float((predictions == y).double().mean()))
    return model, history


def calibrate_threshold(
    model: LearnedAlarmV2,
    rows_fit,
    rows_nofit,
    *,
    max_false_quiet_rate: float = 0.02,
) -> dict:
    """The frozen train-block calibration rule for the v2 alarm's threshold.

    Computes the model's sigmoid scores on both row sets and sweeps the
    candidate thresholds — the sorted unique scores plus 0.0 and 1.0,
    deduplicated. For each candidate ``t`` the decision is
    ``sigmoid >= t`` => fit (exactly :func:`alarm_decision_v2`), and the
    rule records:

    * balanced accuracy — ``(TPR + TNR) / 2`` with ``TPR`` the fraction of
      fit rows scored ``>= t`` and ``TNR`` the fraction of no-fit rows
      scored ``< t``;
    * false-quiet rate — the fraction of NO-FIT rows scored ``>= t``: the
      alarm stays quiet on an out-of-library row, no discovery runs, the
      acquisition is lost (the sealed experiment's dominant error mode,
      ``docs/24-router-loop-v2-sealed-1-results.md``);
    * false-alarm rate — the fraction of FIT rows scored ``< t``: the
      alarm fires on an in-library row, discovery correctly refuses as
      non-novel, the accuracy is lost.

    Selection: the threshold maximizing balanced accuracy SUBJECT TO
    ``false-quiet rate <= max_false_quiet_rate``; ties break toward the
    smaller false-quiet rate, then toward the LARGER threshold (the
    false-quiet rate is non-increasing in the threshold, so the larger
    candidate never reintroduces a quiet). If NO candidate satisfies the
    bound, the rule falls back to the threshold with the smallest
    false-quiet rate, tie-broken toward the larger balanced accuracy and
    then the larger threshold, and records ``bound_satisfied: False``.

    Returns the frozen record ``{"threshold", "balanced_accuracy",
    "false_quiet_rate", "false_alarm_rate", "bound_satisfied",
    "num_candidates"}`` — the selected threshold, its metrics, and the
    swept candidate count. Deterministic (no RNG anywhere). Fail-closed
    on empty or non-finite row sets, a width mismatch, non-finite scores,
    or a bound outside ``[0, 1]``.
    """
    if not isinstance(model, LearnedAlarmV2):
        raise ValueError("model must be a LearnedAlarmV2")
    rows_fit = np.asarray(rows_fit, dtype=np.float64)
    rows_nofit = np.asarray(rows_nofit, dtype=np.float64)
    for name, rows in (("rows_fit", rows_fit), ("rows_nofit", rows_nofit)):
        if rows.ndim != 2 or rows.shape[0] < 1:
            raise ValueError(f"{name} must be a nonempty 2-D array")
        if not np.all(np.isfinite(rows)):
            raise ValueError(f"{name} must be finite")
        if rows.shape[1] != model.input_dim:
            raise ValueError(
                f"{name} width {rows.shape[1]} does not match the model's "
                f"input dim {model.input_dim} (K gates + 8)"
            )
    bound = float(max_false_quiet_rate)
    if not np.isfinite(bound) or not 0.0 <= bound <= 1.0:
        raise ValueError("max_false_quiet_rate must lie in [0, 1]")
    model.eval()
    with torch.no_grad():
        scores_fit = (
            torch.sigmoid(model.logit(_as_float32(rows_fit)))
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        scores_nofit = (
            torch.sigmoid(model.logit(_as_float32(rows_nofit)))
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    if not (
        np.all(np.isfinite(scores_fit)) and np.all(np.isfinite(scores_nofit))
    ):
        raise FloatingPointError("the alarm's scores must be finite")
    candidates = np.unique(
        np.concatenate([scores_fit, scores_nofit, [0.0, 1.0]])
    )
    records = []
    for threshold in candidates:
        threshold = float(threshold)
        tpr = float(np.mean(scores_fit >= threshold))
        tnr = float(np.mean(scores_nofit < threshold))
        records.append(
            {
                "threshold": threshold,
                "balanced_accuracy": 0.5 * (tpr + tnr),
                "false_quiet_rate": float(np.mean(scores_nofit >= threshold)),
                "false_alarm_rate": float(np.mean(scores_fit < threshold)),
            }
        )
    feasible = [
        record
        for record in records
        if record["false_quiet_rate"] <= bound
    ]
    if feasible:
        chosen = max(
            feasible,
            key=lambda record: (
                record["balanced_accuracy"],
                -record["false_quiet_rate"],
                record["threshold"],
            ),
        )
        bound_satisfied = True
    else:
        chosen = min(
            records,
            key=lambda record: (
                record["false_quiet_rate"],
                -record["balanced_accuracy"],
                -record["threshold"],
            ),
        )
        bound_satisfied = False
    return {
        "threshold": chosen["threshold"],
        "balanced_accuracy": chosen["balanced_accuracy"],
        "false_quiet_rate": chosen["false_quiet_rate"],
        "false_alarm_rate": chosen["false_alarm_rate"],
        "bound_satisfied": bound_satisfied,
        "num_candidates": int(candidates.size),
    }


def calibrate_threshold_cost_aware(
    model: LearnedAlarmV2,
    rows_fit,
    rows_nofit,
    *,
    false_quiet_cost: float = 1.0,
    false_alarm_cost: float = 1.0,
) -> dict:
    """The cost-aware train-block calibration rule for the v2 alarm.

    The counterpart to :func:`calibrate_threshold`, which maximizes
    balanced accuracy SUBJECT TO a one-sided bound on the false-quiet
    rate. The sealed loop-v3 result
    (``docs/27-router-loop-v3-sealed-1-results.md``) measured what that
    bound costs: it eliminated false quiets entirely (out-of-library
    acquisition 0.833 -> 1.000) and paid a 0.41 train false-alarm rate
    that materialized as 14/36 in-library harm. This rule prices BOTH
    error modes instead of bounding one.

    Scores and the candidate sweep are identical to
    :func:`calibrate_threshold` — the sorted unique sigmoid scores of
    both row sets plus 0.0 and 1.0, deduplicated, with the decision
    ``sigmoid >= t`` => fit (exactly :func:`alarm_decision_v2`) and the
    same three metrics per candidate.

    Selection: the threshold MINIMIZING the weighted total error

    ``total_cost = false_quiet_cost * FQ + false_alarm_cost * FA``

    with ties broken toward the LARGER balanced accuracy and then the
    LARGER threshold. There is no feasibility bound, so there is no
    fallback branch and no ``bound_satisfied`` key: the rule always
    selects the sweep's cost minimizer.

    **The equal-cost identity (recorded because it makes the frozen rule
    analyzable, not because it is a defect).** ``FQ = 1 - TNR`` and
    ``FA = 1 - TPR`` while ``balanced_accuracy = (TPR + TNR) / 2``, so at
    equal unit costs

    ``FQ + FA = 2 - 2 * balanced_accuracy``

    exactly. With ``false_quiet_cost == false_alarm_cost`` this rule is
    therefore identical to UNCONSTRAINED balanced-accuracy maximization,
    the balanced-accuracy tiebreak can never fire (equal cost implies
    equal balanced accuracy), and the larger-threshold tiebreak does all
    the tie-breaking work. That is the intended frozen instantiation:
    the two error modes carry equal measured task cost in this
    experiment — a false quiet costs one out-of-library seed, a false
    alarm costs one in-library seed, and the two conditions enter the
    end-to-end mean with equal weight. The costs are parameters rather
    than constants so that an asymmetric pricing is expressible without
    a second rule; under unequal costs the identity breaks and the
    balanced-accuracy tiebreak becomes live.

    Returns the frozen record ``{"threshold", "balanced_accuracy",
    "false_quiet_rate", "false_alarm_rate", "total_cost",
    "false_quiet_cost", "false_alarm_cost", "num_candidates"}``.
    Deterministic (no RNG anywhere). Fail-closed on empty or non-finite
    row sets, a width mismatch, non-finite scores, or costs that are
    non-finite, negative, or both zero.
    """
    if not isinstance(model, LearnedAlarmV2):
        raise ValueError("model must be a LearnedAlarmV2")
    rows_fit = np.asarray(rows_fit, dtype=np.float64)
    rows_nofit = np.asarray(rows_nofit, dtype=np.float64)
    for name, rows in (("rows_fit", rows_fit), ("rows_nofit", rows_nofit)):
        if rows.ndim != 2 or rows.shape[0] < 1:
            raise ValueError(f"{name} must be a nonempty 2-D array")
        if not np.all(np.isfinite(rows)):
            raise ValueError(f"{name} must be finite")
        if rows.shape[1] != model.input_dim:
            raise ValueError(
                f"{name} width {rows.shape[1]} does not match the model's "
                f"input dim {model.input_dim} (K gates + 8)"
            )
    costs = {
        "false_quiet_cost": float(false_quiet_cost),
        "false_alarm_cost": float(false_alarm_cost),
    }
    for name, cost in costs.items():
        if not np.isfinite(cost) or cost < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if costs["false_quiet_cost"] + costs["false_alarm_cost"] <= 0.0:
        raise ValueError("the two costs must not both be zero")
    model.eval()
    with torch.no_grad():
        scores_fit = (
            torch.sigmoid(model.logit(_as_float32(rows_fit)))
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        scores_nofit = (
            torch.sigmoid(model.logit(_as_float32(rows_nofit)))
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    if not (
        np.all(np.isfinite(scores_fit)) and np.all(np.isfinite(scores_nofit))
    ):
        raise FloatingPointError("the alarm's scores must be finite")
    candidates = np.unique(
        np.concatenate([scores_fit, scores_nofit, [0.0, 1.0]])
    )
    records = []
    for threshold in candidates:
        threshold = float(threshold)
        tpr = float(np.mean(scores_fit >= threshold))
        tnr = float(np.mean(scores_nofit < threshold))
        false_quiet_rate = float(np.mean(scores_nofit >= threshold))
        false_alarm_rate = float(np.mean(scores_fit < threshold))
        records.append(
            {
                "threshold": threshold,
                "balanced_accuracy": 0.5 * (tpr + tnr),
                "false_quiet_rate": false_quiet_rate,
                "false_alarm_rate": false_alarm_rate,
                "total_cost": (
                    costs["false_quiet_cost"] * false_quiet_rate
                    + costs["false_alarm_cost"] * false_alarm_rate
                ),
            }
        )
    chosen = min(
        records,
        key=lambda record: (
            record["total_cost"],
            -record["balanced_accuracy"],
            -record["threshold"],
        ),
    )
    return {
        "threshold": chosen["threshold"],
        "balanced_accuracy": chosen["balanced_accuracy"],
        "false_quiet_rate": chosen["false_quiet_rate"],
        "false_alarm_rate": chosen["false_alarm_rate"],
        "total_cost": chosen["total_cost"],
        "false_quiet_cost": costs["false_quiet_cost"],
        "false_alarm_cost": costs["false_alarm_cost"],
        "num_candidates": int(candidates.size),
    }


def alarm_decision_v2(
    model: LearnedAlarmV2, features_v2, threshold: float
) -> bool:
    """The v2 alarm's decision: True = "fits" (route), False = "no-fit"
    (discover). Sigmoid of the logit against the explicitly supplied
    ``threshold`` (calibrated by :func:`calibrate_threshold`); a score
    exactly AT the threshold decides fit. Deterministic."""
    if not isinstance(model, LearnedAlarmV2):
        raise ValueError("model must be a LearnedAlarmV2")
    features = np.asarray(features_v2, dtype=np.float64)
    if features.ndim != 1 or features.shape[0] != model.input_dim:
        raise ValueError(
            f"features_v2 must have shape ({model.input_dim},)"
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("features_v2 must be finite")
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    model.eval()
    with torch.no_grad():
        probability = float(
            torch.sigmoid(model.logit(_as_float32(features).unsqueeze(0)))
            .squeeze()
            .item()
        )
    return bool(probability >= threshold)


class GenericMLP(nn.Module):
    """The no-architecture model: a DeepSets head over the candidate block.

    Architecture: ``Linear(18, 64) GELU Linear(64, 64) GELU`` applied
    independently to every candidate's generic feature vector (a shared
    per-candidate encoder), then a MEAN over the candidate axis, then
    ``Linear(64, K+1)`` — classes 0..K-1 are the candidates, class K is
    "no-fit". ``forward(features)`` takes ``(B, K, 18)`` and returns row
    logits ``(B, K+1)``; inference is :func:`generic_decision` (argmax).

    The mean-pooling makes the head permutation-invariant over candidates
    — an honest, documented limitation of the no-architecture arm: with
    the loop's unpermuted convention (truth at index 0) the index classes
    degenerate to "route to 0", and the model's real task is the
    fit/no-fit separation from spectral statistics alone. The train-only
    standardization lives in buffers (measured by :func:`train_generic`).
    """

    def __init__(
        self,
        num_candidates: int,
        feature_dim: int = GENERIC_FEATURE_DIM,
        hidden_dim: int = 64,
    ):
        super().__init__()
        if int(num_candidates) < 1 or int(feature_dim) < 1 or int(hidden_dim) < 1:
            raise ValueError(
                "num_candidates, feature_dim, hidden_dim must be >= 1"
            )
        self.num_candidates = int(num_candidates)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.head = nn.Linear(self.hidden_dim, self.num_candidates + 1)
        self.register_buffer("input_mean", torch.zeros(self.feature_dim))
        self.register_buffer("input_std", torch.ones(self.feature_dim))

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        """Row logits, shape ``(B, K+1)`` from ``(B, K, feature_dim)``."""
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features must have shape (B, K, {self.feature_dim})"
            )
        standardized = (features - self.input_mean) / self.input_std
        encoded = self.encoder(standardized)  # (B, K, hidden)
        return self.head(encoded.mean(dim=1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.logits(features)


def train_generic(
    model: GenericMLP,
    features,
    labels,
    epochs: int = 150,
    lr: float = 1e-3,
    seed: int = 4244,
) -> tuple[GenericMLP, dict]:
    """Train a :class:`GenericMLP`, deterministically, on CPU.

    ``features`` is the ``(M, K, 18)`` float64 block of per-candidate
    :func:`generic_candidate_features` rows; ``labels`` holds the class per
    row — the true candidate's index (0..K-1) for in-library rows, K
    ("no-fit") for out-of-library rows. Cross-entropy, full-batch Adam
    (every epoch is one step; no shuffling RNG exists), train-only
    standardization stored in the buffers. ``torch.manual_seed(seed)``
    runs before the optimization loop; construct ``model`` itself under a
    manual seed for deterministic initialization (the caller's concern —
    the model is passed in, unlike :func:`train_alarm`'s factory). History
    records the loss and the argmax train accuracy per epoch; a non-finite
    loss is an error, never a warning.
    """
    if not isinstance(model, GenericMLP):
        raise ValueError("model must be a GenericMLP")
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    if features.ndim != 3 or features.shape[1:] != (
        model.num_candidates,
        model.feature_dim,
    ):
        raise ValueError(
            f"features must have shape (M, {model.num_candidates}, "
            f"{model.feature_dim})"
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("features must be finite")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must be one per row")
    if labels.size and (labels.min() < 0 or labels.max() > model.num_candidates):
        raise ValueError("labels must lie in 0..K (K is the no-fit class)")
    if int(epochs) < 1:
        raise ValueError("epochs must be >= 1")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    torch.manual_seed(int(seed))
    x = _as_float32(features)
    y = torch.as_tensor(labels, dtype=torch.long)
    flat = x.reshape(-1, x.shape[-1])
    model.input_mean.copy_(flat.mean(dim=0))
    model.input_std.copy_(flat.std(dim=0, unbiased=False).clamp_min(1e-8))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list] = {"loss": [], "train_accuracy": []}
    for epoch in range(int(epochs)):
        model.train()
        logits = model.logits(x)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
        if not np.isfinite(loss_value):
            raise FloatingPointError(
                f"non-finite training loss at epoch {epoch}"
            )
        model.eval()
        with torch.no_grad():
            predictions = model.logits(x).argmax(dim=-1)
        history["loss"].append(loss_value)
        history["train_accuracy"].append(
            float((predictions == y).double().mean())
        )
    return model, history


def generic_decision(model: GenericMLP, features) -> int:
    """The generic model's discrete decision over one row's ``(K, 18)``
    feature block: argmax of the row logits — classes 0..K-1 route to that
    candidate, class K is "no-fit". Deterministic."""
    if not isinstance(model, GenericMLP):
        raise ValueError("model must be a GenericMLP")
    block = np.asarray(features, dtype=np.float64)
    if block.ndim != 2 or block.shape != (
        model.num_candidates,
        model.feature_dim,
    ):
        raise ValueError(
            f"features must have shape ({model.num_candidates}, "
            f"{model.feature_dim})"
        )
    if not np.all(np.isfinite(block)):
        raise ValueError("features must be finite")
    model.eval()
    with torch.no_grad():
        logits = model.logits(_as_float32(block).unsqueeze(0))
    return int(logits.squeeze(0).argmax(dim=-1).item())


# ---------------------------------------------------------------------------
# Arm pipelines (certified numpy control flow; learned models only through
# the decision functions above).
# ---------------------------------------------------------------------------


def arch_row_features(
    instance: BudgetInstance,
    library,
    observation_seed: int,
    profile_grid=NO_ANCHOR_GRID,
    mask_fraction: float = DEFAULT_MASK_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """One row's arch feature block and raw profile block.

    Returns ``(block, raw)`` with ``block`` of shape ``(K, F)`` (the
    per-candidate :func:`arch_candidate_features` vectors, in library
    order) and ``raw`` of shape ``(K, G)`` (the retained raw misfit
    profiles). One shared ``observation_seed`` for every candidate — the
    paired observation draw of the row.
    """
    if not isinstance(instance, BudgetInstance):
        raise ValueError("instance must be a BudgetInstance")
    blocks = []
    raws = []
    for candidate in tuple(library):
        features, raw = arch_candidate_features(
            instance.chain_map,
            candidate,
            observation_seed,
            profile_grid,
            mask_fraction,
        )
        blocks.append(features)
        raws.append(raw)
    if not blocks:
        raise ValueError("the library must be nonempty")
    return np.stack(blocks), np.stack(raws)


def generic_row_features(
    instance: BudgetInstance,
    library,
    observation_seed: int,
    observations_y: np.ndarray,
    operating_point: float,
    mask_fraction: float = DEFAULT_MASK_FRACTION,
) -> np.ndarray:
    """One row's generic feature block, shape ``(K, 18)``: the per-candidate
    :func:`generic_candidate_features` vectors at the row's operating grid
    point, in library order, under the row's shared observation draw."""
    if not isinstance(instance, BudgetInstance):
        raise ValueError("instance must be a BudgetInstance")
    blocks = [
        generic_candidate_features(
            candidate,
            observations_y,
            observation_seed,
            (float(operating_point),),
            mask_fraction,
        )
        for candidate in tuple(library)
    ]
    if not blocks:
        raise ValueError("the library must be nonempty")
    return np.stack(blocks)


@dataclass(frozen=True)
class ArmOutcome:
    """The auditable record of one arm's pass over one row.

    ``arm`` is one of ``"arch_full"``, ``"routing_only"``,
    ``"discovery_only"``, ``"generic"``; ``condition`` is the declared row
    condition (``"in_library"`` / ``"out_of_library"`` / ``"null_control"``,
    view-validated by the arm). ``correct`` is the condition-correctness
    bit under the module's frozen semantics (see the module docstring).
    ``action`` is the action record: ``"route"`` (committed to a library
    candidate), ``"discover"`` (certified discovery admitted; routed to the
    appended structure), or ``"refused"`` (nothing routed, nothing
    admitted). ``routed_index`` indexes the FINAL library: in discover
    mode it is exactly ``initial_library_size`` (the appended structure's
    index), in route mode an index into the unchanged library, in refused
    mode ``None``. The arm's FINAL STRUCTURE is therefore
    ``library[routed_index]`` for route mode, the appended discovered
    constraint for discover mode, and nothing for refused mode.
    ``discovery_invocations`` counts the actual
    :func:`universa.discovery.discover_constraint` calls (0 or 1 per row
    here). ``admitted`` records whether a discovered constraint passed the
    novelty gate; ``map_misfit`` is the router-acceptance residual
    ``||C_disc f1 Z||_F``, set whenever discovery certified (including a
    certified-but-blocked refusal). ``detail`` is the human-readable
    decision provenance (alarm verdict, router/generic class, discovery
    reason). ``seed``, ``initial_library_size``, and
    ``final_library_size`` pin the run's provenance.

    Construction is fail-closed: cross-field coherence (action vs routed
    index vs admission vs library growth) is validated here, so an
    incoherent record is an error, never a silent state. The record holds
    only scalars and strings, so two runs of one row compare bit-identical
    with plain equality.
    """

    arm: str
    condition: str
    correct: bool
    action: str  # "route" | "discover" | "refused"
    routed_index: int | None
    discovery_invocations: int
    admitted: bool
    map_misfit: float | None
    detail: str
    seed: int
    initial_library_size: int
    final_library_size: int

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}")
        if self.condition not in CONDITIONS:
            raise ValueError(f"unknown condition {self.condition!r}")
        if self.action not in ("route", "discover", "refused"):
            raise ValueError(f"unknown action {self.action!r}")
        object.__setattr__(self, "correct", bool(self.correct))
        object.__setattr__(self, "admitted", bool(self.admitted))
        object.__setattr__(self, "seed", int(self.seed))
        if not self.detail:
            raise ValueError("the outcome carries its decision provenance")
        if self.initial_library_size < 1:
            raise ValueError("initial_library_size must be >= 1")
        if self.final_library_size < self.initial_library_size:
            raise ValueError("the library never shrinks")
        if not 0 <= int(self.discovery_invocations) <= 1:
            raise ValueError("discovery_invocations is 0 or 1 per row here")
        object.__setattr__(
            self, "discovery_invocations", int(self.discovery_invocations)
        )
        if self.map_misfit is not None:
            misfit = float(self.map_misfit)
            if not np.isfinite(misfit) or misfit < 0.0:
                raise ValueError("map_misfit must be nonnegative and finite")
            object.__setattr__(self, "map_misfit", misfit)
        if self.action == "route":
            if self.routed_index is None or not (
                0 <= self.routed_index < self.final_library_size
            ):
                raise ValueError("route mode needs a routed index")
            if self.admitted:
                raise ValueError("route mode admits nothing")
            if self.final_library_size != self.initial_library_size:
                raise ValueError("route mode leaves the library unchanged")
            if self.discovery_invocations != 0:
                raise ValueError("route mode never invokes discovery")
            if self.map_misfit is not None:
                raise ValueError("route mode has no discovery misfit")
        elif self.action == "discover":
            if not self.admitted:
                raise ValueError("discover mode requires admission")
            if self.routed_index != self.initial_library_size:
                raise ValueError(
                    "discover mode routes to the appended structure: "
                    "routed_index must equal initial_library_size"
                )
            if self.final_library_size != self.initial_library_size + 1:
                raise ValueError("discover mode grows the library by one")
            if self.map_misfit is None:
                raise ValueError(
                    "discover mode carries the router-acceptance misfit"
                )
            if self.discovery_invocations != 1:
                raise ValueError("discover mode invoked discovery once")
        else:  # refused
            if self.routed_index is not None:
                raise ValueError("a refused arm routes nowhere")
            if self.admitted:
                raise ValueError("a refused arm admits nothing")
            if self.final_library_size != self.initial_library_size:
                raise ValueError("a refused arm leaves the library unchanged")


def _same_library_view(library, view) -> bool:
    """Structural view identity: same count, bit-equal boundaries in order."""
    return len(library) == len(view) and all(
        np.array_equal(a.boundaries[0], b.boundaries[0])
        for a, b in zip(library, view)
    )


def _validate_row(seed, instance, library, condition):
    """Shared fail-closed row validation: provenance (``seed`` must equal
    ``instance.seed``, the :func:`universa.loop.run_loop` discipline),
    library shape, and condition/view coherence."""
    if not isinstance(instance, BudgetInstance):
        raise ValueError("instance must be a BudgetInstance")
    if int(seed) != int(instance.seed):
        raise ValueError(
            f"seed {seed} does not match instance.seed {instance.seed}: "
            "the record and the instance cannot disagree"
        )
    library = tuple(library)
    if not library:
        raise ValueError("the library must be nonempty")
    true_boundary = instance.true_target.boundaries[0]
    for candidate in library:
        if not isinstance(candidate, ChainComplex) or candidate.top_degree != 1:
            raise ValueError("library candidates must be 1-complexes (graphs)")
        if candidate.boundaries[0].shape != true_boundary.shape:
            raise ValueError(
                f"candidate boundary has shape "
                f"{candidate.boundaries[0].shape} but the instance's target "
                f"boundary has shape {true_boundary.shape}: the library and "
                "the instance must share the target dimensions"
            )
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    if condition == "in_library":
        # The truth at index 0 followed by a PREFIX of the decoys: the
        # full instance.candidates view qualifies, and so does the
        # equal-K paired view of this regime (see the module docstring).
        expected = (
            instance.true_target,
            *instance.decoy_targets[: len(library) - 1],
        )
        if not _same_library_view(library, expected):
            raise ValueError(
                "condition 'in_library' requires the true target at index "
                "0 followed by a prefix of the instance's decoys"
            )
    elif not _same_library_view(library, instance.decoy_targets):
        raise ValueError(
            f"condition {condition!r} requires the instance.decoy_targets "
            "view (the truth withheld)"
        )
    return int(seed), library


def _validate_observations(observations, ambient_dim: int) -> np.ndarray:
    """Fail-closed observation-matrix validation (the loop convention)."""
    if not isinstance(observations, np.ndarray) or observations.ndim != 2:
        raise ValueError("observations must be a 2-D array")
    if not np.all(np.isfinite(observations)):
        raise ValueError("observations must be finite")
    if observations.shape[0] != ambient_dim:
        raise ValueError(
            f"observations have ambient dimension {observations.shape[0]}, "
            f"expected {ambient_dim}"
        )
    return observations


def _discovery_path(
    instance: BudgetInstance,
    library_boundaries,
    observations: np.ndarray,
    seed: int,
    novelty_tol: float,
) -> tuple[bool, float | None, str]:
    """The certified discovery path, exactly the :mod:`universa.loop`
    construction: :func:`universa.discovery.discover_constraint` at
    ``CERT_TOL`` (1e-10) on the EXACT transported observations, gated
    through :func:`universa.discovery.admit_to_library` at
    ``(CERT_TOL, novelty_tol)``, with the router-acceptance residual
    ``map_misfit = ||C_disc f1 Z||_F`` over the certified source cycle
    basis. Returns ``(admitted, map_misfit, detail)``; ``map_misfit`` is
    set whenever discovery certified (even if admission blocked it).
    Certified-machinery errors propagate, never swallowed.
    """
    ambient_dim = int(instance.true_target.boundaries[0].shape[1])
    result = discover_constraint(
        observations, ambient_dim, tol=_DISCOVERY_MISFIT_TOL, seeds=(seed,)
    )
    if isinstance(result, DiscoveryInsufficient):
        return False, None, f"discovery insufficient: {result.reason}"
    admission = admit_to_library(
        tuple(library_boundaries), result, _DISCOVERY_MISFIT_TOL, novelty_tol
    )
    cycle_basis = nullspace_basis(instance.source.boundaries[0]).basis
    map_misfit = float(
        np.linalg.norm(
            result.boundary @ instance.chain_map.maps[1] @ cycle_basis
        )
    )
    return admission.admitted, map_misfit, admission.reason


def _condition_correct(
    arm: str,
    condition: str,
    *,
    action: str,
    routed_index: int | None,
    admitted: bool,
    map_misfit: float | None,
) -> bool:
    """The module's frozen condition-correctness semantics (module docstring).

    Reads only the outcome record and the index-0 generator convention —
    never the withheld truth.
    """
    if condition == "null_control":
        if arm in ("arch_full", "routing_only"):
            return not admitted  # the false-admission control
        return action == "refused"  # refusal is the only specificity
    if arm == "discovery_only":
        return bool(
            action == "discover"
            and admitted
            and map_misfit is not None
            and map_misfit <= MAP_ACCEPT_TOL
        )
    if condition == "in_library":
        return bool(action == "route" and routed_index == 0)
    # out_of_library, arms with a discovery channel vs without:
    if arm == "arch_full":
        return bool(
            action == "discover"
            and admitted
            and map_misfit is not None
            and map_misfit <= MAP_ACCEPT_TOL
        )
    return False  # routing_only / generic can never acquire


def arm_arch_full(
    seed: int,
    instance: BudgetInstance,
    library,
    *,
    router: StructureRouter,
    alarm: LearnedAlarm,
    observation_seed: int,
    observations: np.ndarray,
    condition: str,
    profile_grid=NO_ANCHOR_GRID,
    mask_fraction: float = DEFAULT_MASK_FRACTION,
    novelty_tol: float = DEFAULT_NOVELTY_TOL,
) -> ArmOutcome:
    """The full degraded-regime loop: learned routing, learned alarm,
    certified discovery.

    Scores every candidate with the trained ``router`` over the no-anchor
    profile blocks (hard argmax — strictly discrete inference); the
    trained ``alarm`` reads the soft gates (``tau = 1.0``) and the raw
    profile block and decides fit/no-fit. FIT -> route to the router's
    argmax (discovery is never invoked). NO-FIT -> certified discovery on
    the EXACT transported ``observations`` (the degradation never touches
    the data) at the frozen gates (1e-10 certification, 1e-6 novelty); on
    admission the row routes to the appended structure (discover mode),
    else it is refused. Correctness follows the module's frozen semantics.
    """
    seed, library = _validate_row(seed, instance, library, condition)
    if not isinstance(router, StructureRouter):
        raise ValueError("router must be a universa.router.StructureRouter")
    if not isinstance(alarm, LearnedAlarm):
        raise ValueError("alarm must be a LearnedAlarm")
    if alarm.num_candidates != len(library):
        raise ValueError(
            f"alarm expects {alarm.num_candidates} candidates but the "
            f"library has {len(library)}"
        )
    if not np.isfinite(float(novelty_tol)) or novelty_tol <= 0.0:
        raise ValueError("novelty_tol must be positive and finite")
    ambient_dim = int(instance.true_target.boundaries[0].shape[1])
    _validate_observations(observations, ambient_dim)
    block, raw = arch_row_features(
        instance, library, observation_seed, profile_grid, mask_fraction
    )
    if block.shape[1] != router.feature_dim:
        raise ValueError(
            f"router feature dim {router.feature_dim} does not match the "
            f"profile block width {block.shape[1]}"
        )
    gates = router_gates(router, block)
    argmax = router_argmax(router, block)
    if alarm_decision(alarm, gates, raw):
        return ArmOutcome(
            arm="arch_full",
            condition=condition,
            correct=_condition_correct(
                "arch_full",
                condition,
                action="route",
                routed_index=argmax,
                admitted=False,
                map_misfit=None,
            ),
            action="route",
            routed_index=argmax,
            discovery_invocations=0,
            admitted=False,
            map_misfit=None,
            detail=f"alarm=fit router_argmax={argmax}",
            seed=seed,
            initial_library_size=len(library),
            final_library_size=len(library),
        )
    admitted, map_misfit, reason = _discovery_path(
        instance,
        [candidate.boundaries[0] for candidate in library],
        observations,
        seed,
        float(novelty_tol),
    )
    if admitted:
        return ArmOutcome(
            arm="arch_full",
            condition=condition,
            correct=_condition_correct(
                "arch_full",
                condition,
                action="discover",
                routed_index=len(library),
                admitted=True,
                map_misfit=map_misfit,
            ),
            action="discover",
            routed_index=len(library),
            discovery_invocations=1,
            admitted=True,
            map_misfit=map_misfit,
            detail=f"alarm=no-fit discovery=admitted ({reason})",
            seed=seed,
            initial_library_size=len(library),
            final_library_size=len(library) + 1,
        )
    return ArmOutcome(
        arm="arch_full",
        condition=condition,
        correct=_condition_correct(
            "arch_full",
            condition,
            action="refused",
            routed_index=None,
            admitted=False,
            map_misfit=map_misfit,
        ),
        action="refused",
        routed_index=None,
        discovery_invocations=1,
        admitted=False,
        map_misfit=map_misfit,
        detail=f"alarm=no-fit discovery=refused ({reason})",
        seed=seed,
        initial_library_size=len(library),
        final_library_size=len(library),
    )


def arm_arch_full_v2(
    seed: int,
    instance: BudgetInstance,
    library,
    *,
    router: StructureRouter,
    alarm: LearnedAlarmV2,
    threshold: float,
    observation_seed: int,
    observations: np.ndarray,
    condition: str,
    profile_grid=NO_ANCHOR_GRID,
    mask_fraction: float = DEFAULT_MASK_FRACTION,
    novelty_tol: float = DEFAULT_NOVELTY_TOL,
) -> ArmOutcome:
    """The full degraded-regime loop with the v2 alarm: learned routing,
    calibrated learned alarm, certified discovery.

    Exactly :func:`arm_arch_full`'s structure and frozen semantics, with
    the v1 alarm swapped for the redesign: every candidate is scored by
    the trained ``router`` over the no-anchor profile blocks (hard
    argmax); the trained :class:`LearnedAlarmV2` reads
    :func:`alarm_features_v2` of the soft gates (``tau = 1.0``) and the
    raw profile block and decides fit/no-fit against the explicitly
    supplied ``threshold`` (calibrated on the train block by
    :func:`calibrate_threshold`). FIT -> route to the router's argmax
    (discovery is never invoked). NO-FIT -> certified discovery on the
    EXACT transported ``observations`` at the frozen gates (1e-10
    certification, 1e-6 novelty); on admission the row routes to the
    appended structure (discover mode), else it is refused. Correctness
    delegates to the module's frozen :func:`_condition_correct` path
    (``arch_full`` semantics — the outcome vocabulary is unchanged); the
    ``detail`` strings carry an ``alarm_v2=`` marker for provenance.
    """
    seed, library = _validate_row(seed, instance, library, condition)
    if not isinstance(router, StructureRouter):
        raise ValueError("router must be a universa.router.StructureRouter")
    if not isinstance(alarm, LearnedAlarmV2):
        raise ValueError("alarm must be a LearnedAlarmV2")
    if alarm.num_candidates != len(library):
        raise ValueError(
            f"alarm expects {alarm.num_candidates} candidates but the "
            f"library has {len(library)}"
        )
    if not np.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    if not np.isfinite(float(novelty_tol)) or novelty_tol <= 0.0:
        raise ValueError("novelty_tol must be positive and finite")
    ambient_dim = int(instance.true_target.boundaries[0].shape[1])
    _validate_observations(observations, ambient_dim)
    block, raw = arch_row_features(
        instance, library, observation_seed, profile_grid, mask_fraction
    )
    if block.shape[1] != router.feature_dim:
        raise ValueError(
            f"router feature dim {router.feature_dim} does not match the "
            f"profile block width {block.shape[1]}"
        )
    gates = router_gates(router, block)
    argmax = router_argmax(router, block)
    features = alarm_features_v2(gates, raw)
    if alarm_decision_v2(alarm, features, float(threshold)):
        return ArmOutcome(
            arm="arch_full",
            condition=condition,
            correct=_condition_correct(
                "arch_full",
                condition,
                action="route",
                routed_index=argmax,
                admitted=False,
                map_misfit=None,
            ),
            action="route",
            routed_index=argmax,
            discovery_invocations=0,
            admitted=False,
            map_misfit=None,
            detail=f"alarm_v2=fit router_argmax={argmax}",
            seed=seed,
            initial_library_size=len(library),
            final_library_size=len(library),
        )
    admitted, map_misfit, reason = _discovery_path(
        instance,
        [candidate.boundaries[0] for candidate in library],
        observations,
        seed,
        float(novelty_tol),
    )
    if admitted:
        return ArmOutcome(
            arm="arch_full",
            condition=condition,
            correct=_condition_correct(
                "arch_full",
                condition,
                action="discover",
                routed_index=len(library),
                admitted=True,
                map_misfit=map_misfit,
            ),
            action="discover",
            routed_index=len(library),
            discovery_invocations=1,
            admitted=True,
            map_misfit=map_misfit,
            detail=f"alarm_v2=no-fit discovery=admitted ({reason})",
            seed=seed,
            initial_library_size=len(library),
            final_library_size=len(library) + 1,
        )
    return ArmOutcome(
        arm="arch_full",
        condition=condition,
        correct=_condition_correct(
            "arch_full",
            condition,
            action="refused",
            routed_index=None,
            admitted=False,
            map_misfit=map_misfit,
        ),
        action="refused",
        routed_index=None,
        discovery_invocations=1,
        admitted=False,
        map_misfit=map_misfit,
        detail=f"alarm_v2=no-fit discovery=refused ({reason})",
        seed=seed,
        initial_library_size=len(library),
        final_library_size=len(library),
    )


def arm_routing_only(
    seed: int,
    instance: BudgetInstance,
    library,
    *,
    router: StructureRouter,
    observation_seed: int,
    condition: str,
    profile_grid=NO_ANCHOR_GRID,
    mask_fraction: float = DEFAULT_MASK_FRACTION,
) -> ArmOutcome:
    """Ablation: routing with NO alarm and NO discovery.

    Always routes to the trained router's argmax candidate — a forced
    choice. On out-of-library rows it must pick a decoy (an honest
    failure: the frozen semantics score it incorrect, since nothing
    certified is ever acquired); on null-control rows it admits nothing
    (vacuously correct — this arm has no admission channel at all, which
    is exactly the mechanism the ablation removes).
    """
    seed, library = _validate_row(seed, instance, library, condition)
    if not isinstance(router, StructureRouter):
        raise ValueError("router must be a universa.router.StructureRouter")
    block, _ = arch_row_features(
        instance, library, observation_seed, profile_grid, mask_fraction
    )
    if block.shape[1] != router.feature_dim:
        raise ValueError(
            f"router feature dim {router.feature_dim} does not match the "
            f"profile block width {block.shape[1]}"
        )
    argmax = router_argmax(router, block)
    return ArmOutcome(
        arm="routing_only",
        condition=condition,
        correct=_condition_correct(
            "routing_only",
            condition,
            action="route",
            routed_index=argmax,
            admitted=False,
            map_misfit=None,
        ),
        action="route",
        routed_index=argmax,
        discovery_invocations=0,
        admitted=False,
        map_misfit=None,
        detail=f"forced route router_argmax={argmax} (no alarm, no discovery)",
        seed=seed,
        initial_library_size=len(library),
        final_library_size=len(library),
    )


def arm_discovery_only(
    seed: int,
    instance: BudgetInstance,
    *,
    observations: np.ndarray,
    novelty_tol: float = DEFAULT_NOVELTY_TOL,
    condition: str,
) -> ArmOutcome:
    """Ablation: ALWAYS run certified discovery on the exact observations.

    The library is unused for routing; novelty is checked against the
    instance's decoy library (``instance.decoy_targets`` boundaries),
    exactly the :func:`universa.discovery.run_discovery` construction.
    Correct iff discovery certifies, is novel against that library, and
    the router-acceptance misfit passes 1e-9 — on null-control rows,
    correct iff refused (a false constraint certified from noise is a
    specificity failure, the v1 H4 control).
    """
    if not isinstance(instance, BudgetInstance):
        raise ValueError("instance must be a BudgetInstance")
    if int(seed) != int(instance.seed):
        raise ValueError(
            f"seed {seed} does not match instance.seed {instance.seed}: "
            "the record and the instance cannot disagree"
        )
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    if not np.isfinite(float(novelty_tol)) or novelty_tol <= 0.0:
        raise ValueError("novelty_tol must be positive and finite")
    seed = int(seed)
    ambient_dim = int(instance.true_target.boundaries[0].shape[1])
    _validate_observations(observations, ambient_dim)
    library_boundaries = tuple(
        decoy.boundaries[0] for decoy in instance.decoy_targets
    )
    admitted, map_misfit, reason = _discovery_path(
        instance, library_boundaries, observations, seed, float(novelty_tol)
    )
    initial_size = len(library_boundaries)
    if admitted:
        return ArmOutcome(
            arm="discovery_only",
            condition=condition,
            correct=_condition_correct(
                "discovery_only",
                condition,
                action="discover",
                routed_index=initial_size,
                admitted=True,
                map_misfit=map_misfit,
            ),
            action="discover",
            routed_index=initial_size,
            discovery_invocations=1,
            admitted=True,
            map_misfit=map_misfit,
            detail=f"always-discover admitted ({reason})",
            seed=seed,
            initial_library_size=initial_size,
            final_library_size=initial_size + 1,
        )
    return ArmOutcome(
        arm="discovery_only",
        condition=condition,
        correct=_condition_correct(
            "discovery_only",
            condition,
            action="refused",
            routed_index=None,
            admitted=False,
            map_misfit=map_misfit,
        ),
        action="refused",
        routed_index=None,
        discovery_invocations=1,
        admitted=False,
        map_misfit=map_misfit,
        detail=f"always-discover refused ({reason})",
        seed=seed,
        initial_library_size=initial_size,
        final_library_size=initial_size,
    )


def arm_generic(
    seed: int,
    instance: BudgetInstance,
    library,
    *,
    generic_model: GenericMLP,
    observation_seed: int,
    observations_y: np.ndarray,
    condition: str,
    mask_fraction: float = DEFAULT_MASK_FRACTION,
) -> ArmOutcome:
    """The NO-ARCHITECTURE arm: the generic MLP's decision, nothing certified.

    Builds the per-candidate generic spectral blocks (no commutation
    residual, no profile — see :func:`generic_candidate_features`) at the
    row's operating grid point and reads :func:`generic_decision`:
    classes 0..K-1 route to that candidate (correctness by the same
    certified index-0 gate as the arch route mode); class K ("no-fit")
    synthesizes nothing — out-of-library that scores incorrect (nothing
    certified is ever acquired by this arm), null-control it scores
    correct (refusal is this arm's only specificity mechanism).
    """
    seed, library = _validate_row(seed, instance, library, condition)
    if not isinstance(generic_model, GenericMLP):
        raise ValueError("generic_model must be a GenericMLP")
    if generic_model.num_candidates != len(library):
        raise ValueError(
            f"generic model expects {generic_model.num_candidates} "
            f"candidates but the library has {len(library)}"
        )
    ambient_dim = int(instance.true_target.boundaries[0].shape[1])
    _validate_observations(observations_y, ambient_dim)
    point = operating_grid_point(seed)
    block = generic_row_features(
        instance, library, observation_seed, observations_y, point, mask_fraction
    )
    if block.shape[1] != generic_model.feature_dim:
        raise ValueError(
            f"generic feature dim {generic_model.feature_dim} does not "
            f"match the block width {block.shape[1]}"
        )
    decision = generic_decision(generic_model, block)
    no_fit = decision == len(library)
    action = "refused" if no_fit else "route"
    routed_index = None if no_fit else decision
    return ArmOutcome(
        arm="generic",
        condition=condition,
        correct=_condition_correct(
            "generic",
            condition,
            action=action,
            routed_index=routed_index,
            admitted=False,
            map_misfit=None,
        ),
        action=action,
        routed_index=routed_index,
        discovery_invocations=0,
        admitted=False,
        map_misfit=None,
        detail=(
            f"generic_class={decision}"
            f"{' (no-fit)' if no_fit else ''}"
            f" operating_fraction={point:.1f}"
        ),
        seed=seed,
        initial_library_size=len(library),
        final_library_size=len(library),
    )
