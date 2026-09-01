#!/usr/bin/env python3
"""Sealed runner for experiment ``universa-loop-v4-sealed-1``.

The frozen design: the DEGRADED-regime route-or-discover loop of
:mod:`universa.loop_v2` under COST-AWARE THRESHOLD CALIBRATION — routing
and the fit/no-fit alarm are LEARNED in the no-anchor regime (every
candidate boundary observed through
:class:`universa.partial.ObservationModel` draws with
``mask_fraction = 0.25`` AND corruption swept over the 0.2..0.9 profile
grid that excludes 0.0, so no residual the loop reads is exact), the
alarm is the :class:`universa.loop_v2.LearnedAlarmV2` over the
:func:`universa.loop_v2.alarm_features_v2` layout (the frozen v1 columns
plus five margin features) — architecture UNCHANGED from loop-v3 — with
its decision threshold CALIBRATED on the train block by
:func:`universa.loop_v2.calibrate_threshold_cost_aware` at the frozen
unit costs ``false_quiet_cost = false_alarm_cost = 1.0``, while discovery
stays certified on the
exact transported observations. THREE models are trained inside the
sealed run on the train seed block 220001..220400 (never on a sealed eval
seed) and the alarm threshold is calibrated on the same block, then FOUR
arms are compared on the sealed eval seed block 230001..230036 x THREE
paired conditions. Each eligible seed contributes TWELVE raw rows — one
per (condition, arm):

* **conditions** — the equal-K paired views of
  :mod:`universa.loop_v2` (K = 3): the in-library view
  ``(instance.true_target, *instance.decoy_targets[:-1])`` (the truth at
  index 0, the loop's unpermuted convention) and the out-of-library view
  ``instance.decoy_targets`` (the truth withheld), plus the null-control
  condition — the same out-of-library view fed structure-free
  observations (``universa.loop.null_observations(seed, ambient_dim,
  16)``, the frozen exp4 H4 ``"discovery-null"`` schedule). The
  structured conditions consume the exact transported observations
  ``Y = f1 A`` of ``universa.discovery.synthesize_observations`` (the
  frozen ``"discovery-observation"`` schedule, M = 16 columns).
* **arms** — ``arch_full`` (learned router + learned v2 alarm under the
  calibrated threshold + certified discovery), ``routing_only`` (learned
  router, no alarm, no discovery — a forced choice), ``discovery_only``
  (always certified discovery, the library unused for routing), and
  ``generic`` (the no-architecture
  :class:`universa.loop_v2.GenericMLP` over generic spectral features:
  no commutation residual, no degradation profile). Correctness follows
  the frozen :mod:`universa.loop_v2` semantics: in-library, the arm's
  final structure is the true target (route to index 0, or a certified
  admitted discovery with ``map_misfit <= 1e-9``); out-of-library, a
  certified novel structure admitted with ``map_misfit <= 1e-9``;
  null-control, nothing admitted (the alarm arms) or a refusal (the
  arms whose only specificity mechanism is refusal).

**Train block (frozen).** One row per train seed x BOTH views: the arch
no-anchor degradation-profile features AND the generic spectral features
per candidate, under the seed's canonical shared observation draw
``subseed(seed, "router-v2-observe")`` — the router-v2 regime's
observation family, so the learned models' training rows and the loop
rows come from ONE observation distribution on disjoint seed blocks —
with the generic arm's operating grid point
``NO_ANCHOR_GRID[subseed(seed, "loop-v2-operating") % 8]``. The
:class:`universa.router.StructureRouter` (feature_dim 18, hidden 64,
seed 4242) trains via :func:`universa.router.train_router` on the
in-library rows, label ``0`` for every row — the loop's unpermuted
index-0 convention: nothing in the router's training signal distinguishes
the candidates' positions, so the router's learned role is profile-shape
scoring under the index-0 commitment and the loop's learned SELECTIVITY
lives in the alarm, not the argmax (the protocol's declared consequences
of the index-0 convention; the out-of-library view carries no routing
label and contributes no router rows). The
:class:`universa.loop_v2.LearnedAlarmV2` (seed 4243) trains via
:func:`universa.loop_v2.train_alarm_v2` on fit/no-fit labels from the
in-library vs out-of-library rows (the gates of the trained router,
tau = 1.0) over the :func:`universa.loop_v2.alarm_features_v2` layout,
and its decision threshold is then CALIBRATED on the same train rows by
:func:`universa.loop_v2.calibrate_threshold_cost_aware` at the frozen
unit costs (the threshold MINIMIZING ``false_quiet_cost *
false_quiet_rate + false_alarm_cost * false_alarm_rate``, no feasibility
bound — this is the experiment's ONE changed component, replacing
loop-v3's binding ``false_quiet_rate <= 0.02`` constraint) — the full
eight-key calibration record is retained in the training provenance; the
:class:`universa.loop_v2.GenericMLP` (seed 4244) trains on index/no-fit
labels (in-library rows labeled the true candidate's index 0 — the
unpermuted loop convention — out-of-library rows labeled K). Training
order is pinned — router (4242), alarm (4243) and its calibration,
generic (4244) — because the alarm's features read the trained router's
soft gates. Every model state's SHA-256 and the final training scalars
are recorded.

**Eval pairing (frozen).** One shared observation draw per seed —
``subseed(seed, "router-v2-observe")`` — feeds every candidate of every
arm of every condition of the row (exact pairing), so the arch and
routing arms' feature blocks are IDENTICAL in the out-of-library and
null-control conditions per seed (the alarm's decision basis precedes
observations: the observation matrix is consulted only by the discovery
path and the generic arm's spectral features). The arch raw profiles
(the 8 raw observed-misfit values per candidate) are retained per row.
The claims are four paired per-seed differences against the full
architecture, decided by one-sided Bonferroni Student-t lower bounds
(per-claim alpha 0.05/4 = 0.0125) against their frozen thresholds
(0.0 / 0.0 / 0.0 / -0.05): H1/H2 are the end-to-end comparisons
(per-seed e2e = the mean of an arm's correctness bits over the three
paired conditions) against the generic and routing-only arms; H3 is the
strict in-library comparison against the generic arm (a tie FAILS);
H4 is the in-library HARM non-inferiority bound against the
discovery-only arm (the architecture must not lose by more than 0.05).
The generator seed is the inference unit; the mechanical ``SE = 0``
case applies (the lower bound then equals the estimate).

**Determinism, verified.** Every train row, every eval row, every model
hash, and the calibration record is a deterministic function of the
frozen seeds: the runner executes every eval (seed, condition, arm) row
TWICE and requires bit-identical rows — every field except the retained
``wall_time_seconds`` (execution metadata; the first execution's wall
time is kept) — and any mismatch is a whole-run ``design_failure``.

**Retention (frozen).** Every row retains the full
:class:`universa.loop_v2.ArmOutcome` scalar record (correctness, action,
routed index, discovery invocations, admission, router-acceptance map
misfit, decision detail, library sizes), the row's observation seed, the
per-candidate arch raw profiles (8 floats each), and the wall time. The
audit block recomputes every correctness decision and every claim
statistic from the raw rows alone WHERE THE SEMANTICS ALLOW (the exact
scope is stated in the audit block): the routing arms' decisions
recompute from the retained action/routed_index, the certified-gate
semantics from the retained admitted/map_misfit, and the observation
schedule from the retained observation seed; the router's argmax, the
calibrated v2 alarm's fit/no-fit, and the generic model's class are NOT
recomputable from the rows — the three learned models are hash-pinned in
``training`` instead. No matrices are retained.

**Declared caveat (frozen).** Demo-scale numbers are not evidence: the
arms' correctness rates and their paired differences are properties of
this one synthetic generator family at the frozen exp4 sizes (8
vertices, 14 edges, 6 classes, 3 decoys, M = 16 observations) in the
no-anchor degraded regime, with small MLPs trained on 400 train seeds
of the same family — not guarantees for other families, other library
sizes, other training budgets, or real data; the generic arm is
deliberately architecture-free (no commutation residual, no degradation
profile); the null control is an i.i.d. Gaussian null matched to the
target edge space, not a worst-case adversary; and the wall times are
descriptive execution metadata, never analyzed.

This runner must not be executed on the sealed eval seed block until the
protocol, this runner, and the seal record
(``docs/31-router-loop-v4-seal.json``) have been committed and pushed. It
keeps the seed block inert at import time: no instance is constructed
until every preflight check in :func:`_preflight` has passed, in the
frozen order — output missing, clean worktree, seal committed at HEAD,
seal validation, file hashes, runtime hash, environment — and only then
any data construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# The operator-provided CUDA visibility is captured BEFORE the runner hides
# CUDA from torch, so the result records both the operator-provided and the
# effective value.
OPERATOR_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only, before the torch import

import torch

import universa
from universa.budgets import make_budget_instance
from universa.discovery import synthesize_observations
from universa.generators import SwitchInstance, subseed
from universa.loop import ALARM_TOL, null_observations
from universa.loop_v2 import (
    ALARM_V2_MARGIN_FEATURE_NAMES,
    ARMS as _LOOP_V2_ARMS,
    CONDITIONS as _LOOP_V2_CONDITIONS,
    MAP_ACCEPT_TOL,
    ArmOutcome,
    GenericMLP,
    LearnedAlarmV2,
    alarm_features_v2,
    arch_row_features,
    arm_arch_full_v2,
    arm_discovery_only,
    arm_generic,
    arm_routing_only,
    calibrate_threshold_cost_aware,
    generic_row_features,
    operating_grid_point,
    router_gates,
    train_alarm_v2,
    train_generic,
)
from universa.router import StructureRouter, train_router
from universa.router_v2 import (
    DEFAULT_MASK_FRACTION,
    NO_ANCHOR_GRID,
    no_anchor_feature_dim,
)
from universa.structures import ChainMap

EXPERIMENT_ID = "universa-loop-v4-sealed-1"
RESULT_SCHEMA = "universa-router-loop-v4-sealed-result/1"

# The one canonical execution command, recorded verbatim in every result.
CANONICAL_COMMAND = (
    "env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src python "
    "scripts/run_loop_v4_sealed_1.py --output "
    "results/experiments/router-loop-v4-sealed-1.json"
)

PROTOCOL = "docs/30-sealed-router-loop-v4-protocol.md"
RUNNER_SOURCE = "scripts/run_loop_v4_sealed_1.py"
DEFAULT_SEAL = "docs/31-router-loop-v4-seal.json"
SEAL_SCHEMA = "universa-seal/10"

PROTOCOL_SHA256 = (
    "93ee35e00b764bb26c9321cd8f81ca7939b9349f675efdc1ac21ed909faa36b5"
)
"""Frozen fingerprint of the sealed protocol.

The committed runner embeds the sealed protocol's SHA-256 as this constant;
the value must equal the seal record's ``protocol_sha256``. Any protocol edit
after the seal requires re-pinning this constant (and a new seal record) —
the fail-closed refusal in :func:`_frozen_protocol_sha256` then catches the
mismatch. While the value is the placeholder ``PENDING_PROTOCOL_SHA256`` the
runner refuses every execution.
"""

# Frozen task family: graph-quotient budget instances via
# universa.budgets.make_budget_instance at the exp4 sizes (8 vertices, 14
# edges, 6 classes, 3 decoys), ONE instance per seed with the equal-K paired
# views of universa.loop_v2: the in-library view (true target at index 0
# followed by the decoy prefix) and the out-of-library view (the decoys, the
# truth withheld) — both of size K = num_decoys, so the learned models'
# fixed input widths match across the paired conditions.
NUM_VERTICES = 8
NUM_EDGES = 14
NUM_CLASSES = 6
NUM_DECOYS = 3
NUM_VIEW_CANDIDATES = NUM_DECOYS  # K = 3 for BOTH paired views

TRAIN_SEEDS = tuple(range(220001, 220401))  # 220001..220400
SEALED_EVAL_SEEDS = tuple(range(230001, 230037))  # 230001..230036

CONDITIONS = ("in_library", "out_of_library", "null_control")
"""The frozen per-seed conditions, in campaign build order (loop_v2 names)."""

ARMS = ("arch_full", "routing_only", "discovery_only", "generic")
"""The frozen arms, in campaign build order (loop_v2 names)."""

PROFILE_GRID = NO_ANCHOR_GRID  # 0.2..0.9 by 0.1 (8 points, excludes 0.0)
MASK_FRACTION = DEFAULT_MASK_FRACTION  # 0.25
NUM_OBSERVATIONS = 16  # M observation vectors per seed, frozen exp4 count

FEATURE_DIM = 18
ROUTER_HIDDEN_DIM = 64
EPOCHS = 150
LEARNING_RATE = 1e-3
LAMBDA_AUX = 0.01
TAU_START = 2.0
TAU_END = 0.25
TORCH_SEED_ROUTER = 4242
TORCH_SEED_ALARM = 4243
TORCH_SEED_GENERIC = 4244
FALSE_QUIET_COST = 1.0
FALSE_ALARM_COST = 1.0
"""The frozen unit costs of the train-block threshold calibration.

The alarm's decision threshold is picked on the train block by
:func:`universa.loop_v2.calibrate_threshold_cost_aware`: the threshold
MINIMIZING ``false_quiet_cost * false_quiet_rate + false_alarm_cost *
false_alarm_rate``, ties toward the larger balanced accuracy then the
larger threshold. This replaces loop-v3's rule, which maximized balanced
accuracy subject to a one-sided bound ``false_quiet_rate <= 0.02``; that
bound was binding (the sealed loop-v3 record: threshold 0.8897 at
false-quiet rate exactly 0.02 with a 0.41 false-alarm rate), and the
published consequence was 14/36 in-library false alarms.

Both costs are frozen at 1.0 because each error mode costs exactly one
seed in one condition — a false quiet loses an out-of-library
acquisition, a false alarm loses an in-library route — and the two
conditions enter the end-to-end statistic with equal weight. As the
protocol (§1, §5) and the module docstring declare openly, at equal unit
costs ``false_quiet_rate + false_alarm_rate = 2 - 2 * balanced_accuracy``
exactly, so this rule is identical to UNCONSTRAINED balanced-accuracy
maximization and the balanced-accuracy tiebreak can never fire. The
experiment's alarm-side change is precisely the removal of loop-v3's
binding constraint, and it is described that way throughout.
"""

FAMILYWISE_ALPHA = 0.05
NUM_CLAIMS = 4
PER_CLAIM_ALPHA = FAMILYWISE_ALPHA / NUM_CLAIMS  # 0.0125
MIN_ELIGIBLE = 30
MAX_TRAIN_EXCLUDED = 20
"""The frozen ceiling on excluded (non-instance) train seeds.

Protocol errata 1 (docs/30): a train seed whose instance fails to build is
a recorded EXCLUSION, not a whole-run failure — the generator's own
discriminability guard refuses a decoy sharing the true target's kernel,
and such a seed is not an instance of the family at all. This constant is
the fail-closed guard over that exclusion: more than 20 non-instances in
the declared block means something systematic is wrong rather than the
rare generator collision the errata describes, and the run stops as a
whole-run design failure. The first attempt's measured rate was one
non-instance in roughly a thousand seeds built, so 20 out of 400 is a
ceiling the frozen design expects never to approach.

The guard counts EXCLUSIONS rather than survivors so that it means the
same thing at any declared block size — the runner's tests exercise it on
monkeypatched fixture blocks of a handful of seeds.
"""
AUDIT_TOL = ALARM_TOL  # 1e-9, the undegraded-instance audit tolerance

# One-sided Bonferroni Student-t critical values t.ppf(1 - 0.05/4, n - 1) for
# the eligible n, computed once at design time with scipy:
#     from scipy.stats import t
#     {n: t.ppf(1 - 0.05 / 4, n - 1) for n in range(30, 37)}
# Numerically identical to the router, discovery, and loop experiments' table
# (same alpha, same eligible-n range); hard-coded so scipy is not a runtime
# dependency.
_T_ONE_SIDED_BONFERRONI = {
    30: 2.36384607320831,
    31: 2.359562458700931,
    32: 2.3555682821599135,
    33: 2.3518351803763706,
    34: 2.348338377257479,
    35: 2.3450561343451817,
    36: 2.3419692993010397,
}

# The frozen four-claim confirmatory family. The decision logic keys off this
# table alone; the seal must carry exactly these claim ids, in this order.
# Every claim is a paired per-seed difference against the full architecture:
# ``arms`` names the minuend/subtrahend arms, ``scope`` the per-seed values
# the difference is taken on (``e2e``: the mean of the arm's correctness bits
# over the three paired conditions; ``in_library``: the in-library condition
# bit), ``statistic`` the frozen statistic name of the protocol, and
# ``difference`` the per-seed record key the claim is decided on.
_CLAIM_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "h1-loopv4-arch-vs-generic-e2e",
        "statistic": "arch_vs_generic_e2e",
        "difference": "d_arch_generic_e2e",
        "arms": ("arch_full", "generic"),
        "scope": "e2e",
        "threshold": 0.0,
        "role": "primary",
        "description": (
            "the full architecture's per-seed end-to-end correctness — the "
            "mean of its correctness bits over the three paired conditions "
            "— beats the no-architecture generic arm's"
        ),
    },
    {
        "id": "h2-loopv4-arch-vs-routing-only-e2e",
        "statistic": "arch_vs_routing_only_e2e",
        "difference": "d_arch_routing_only_e2e",
        "arms": ("arch_full", "routing_only"),
        "scope": "e2e",
        "threshold": 0.0,
        "role": "secondary",
        "description": (
            "the full architecture's per-seed end-to-end correctness beats "
            "the routing-only ablation's (no alarm, no discovery: a forced "
            "choice that can never acquire out-of-library)"
        ),
    },
    {
        "id": "h3-loopv4-arch-vs-generic-inlibrary",
        "statistic": "arch_vs_generic_inlibrary",
        "difference": "d_arch_generic_inlibrary",
        "arms": ("arch_full", "generic"),
        "scope": "in_library",
        "threshold": 0.0,
        "role": "secondary",
        "description": (
            "on the in-library condition the full architecture's "
            "correctness bit strictly beats the no-architecture generic "
            "arm's (the strict paired comparison at 0)"
        ),
    },
    {
        "id": "h4-loopv4-arch-vs-discovery-only-inlibrary-harm",
        "statistic": "arch_vs_discovery_only_inlibrary_harm",
        "difference": "d_arch_discovery_only_inlibrary_harm",
        "arms": ("arch_full", "discovery_only"),
        "scope": "in_library",
        "threshold": -0.05,
        "role": "secondary",
        "description": (
            "on the in-library condition the full architecture is not "
            "worse than the discovery-only ablation by more than the 0.05 "
            "harm margin (a non-inferiority bound: always-discovering "
            "must not materially beat route-or-discover when the library "
            "fits)"
        ),
    },
)
CLAIM_IDS = tuple(definition["id"] for definition in _CLAIM_DEFINITIONS)

# The per-seed statistic provenance named in each claim's estimand text.
_DIFFERENCE_PROVENANCE = {
    "d_arch_generic_e2e": (
        "arch_full_e2e - generic_e2e, where an arm's e2e is the mean of "
        "its correctness bits over the three paired conditions "
        "(in_library, out_of_library, null_control)"
    ),
    "d_arch_routing_only_e2e": (
        "arch_full_e2e - routing_only_e2e, where an arm's e2e is the mean "
        "of its correctness bits over the three paired conditions "
        "(in_library, out_of_library, null_control)"
    ),
    "d_arch_generic_inlibrary": (
        "arch_full's in_library correctness bit minus generic's"
    ),
    "d_arch_discovery_only_inlibrary_harm": (
        "arch_full's in_library correctness bit minus discovery_only's; "
        "the harm margin is -0.05"
    ),
}

# The subfields the seal record must carry for each frozen claim (id,
# statistic, theta, null, alternative, bound direction, threshold, support
# rule). Schema universa-seal/10 claim objects carry ``statistic`` (the loop
# seals' convention: there is no operating fraction and no baseline arm —
# every claim is a paired arm-vs-architecture difference). A claim object
# must carry exactly these keys, plus the optional ``role``; any other key
# is refused.
_CLAIM_SEAL_KEYS = (
    "id",
    "statistic",
    "theta",
    "null",
    "alternative",
    "bound_direction",
    "threshold",
    "support_rule",
)

DECLARED_CAVEAT = (
    "demo-scale numbers are not evidence: the arms' correctness rates and "
    "their paired differences are properties of this one synthetic "
    "generator family at the frozen exp4 sizes (8 vertices, 14 edges, 6 "
    "classes, 3 decoys, M = 16 observations) in the no-anchor degraded "
    "regime, with small MLPs trained on 400 train seeds of the same "
    "family — not guarantees for other families, other library sizes, "
    "other training budgets, or real data; the generic arm is "
    "deliberately architecture-free (no commutation residual, no "
    "degradation profile); the null control is an i.i.d. Gaussian null "
    "matched to the target edge space, not a worst-case adversary; and "
    "the wall times are descriptive execution metadata, never analyzed"
)

if CONDITIONS != tuple(_LOOP_V2_CONDITIONS):
    raise RuntimeError(
        f"frozen CONDITIONS={CONDITIONS} disagree with universa.loop_v2: "
        f"{tuple(_LOOP_V2_CONDITIONS)}"
    )
if ARMS != tuple(_LOOP_V2_ARMS):
    raise RuntimeError(
        f"frozen ARMS={ARMS} disagree with universa.loop_v2: "
        f"{tuple(_LOOP_V2_ARMS)}"
    )
if FEATURE_DIM != no_anchor_feature_dim(PROFILE_GRID):
    raise RuntimeError(
        f"frozen FEATURE_DIM={FEATURE_DIM} disagrees with "
        f"universa.router_v2: no_anchor_feature_dim(PROFILE_GRID)="
        f"{no_anchor_feature_dim(PROFILE_GRID)}"
    )


class DesignFailureError(RuntimeError):
    """A frozen-design validation failure: the whole run is a design failure."""

    def __init__(self, message: str, *, seed: int | None = None) -> None:
        super().__init__(message)
        self.seed = seed


@dataclass(frozen=True)
class _TrainBlock:
    """The runner-built train block (400 train seeds x both views).

    ``in_blocks``/``in_raws`` and ``out_blocks``/``out_raws`` are the
    UNPERMUTED arch blocks and raw profiles of the two views — the
    StructureRouter trains on the in-library blocks with label ``0`` for
    every row (the loop's unpermuted index-0 convention; the
    out-of-library view carries no routing label and contributes no
    router rows), and the LearnedAlarmV2 rows are computed from both
    views once the router is trained (the alarm reads the trained
    router's gates). ``generic_features``/``generic_labels`` are the
    GenericMLP's rows: both views' generic spectral blocks, in-library
    labeled 0 (the true candidate's index under the unpermuted loop
    convention) and out-of-library labeled K (no-fit).
    """

    in_blocks: np.ndarray  # (M, K, FEATURE_DIM), unpermuted in-library
    in_raws: np.ndarray  # (M, K, G), raw misfit profiles
    out_blocks: np.ndarray  # (M, K, FEATURE_DIM), unpermuted out-of-library
    out_raws: np.ndarray  # (M, K, G)
    generic_features: np.ndarray  # (2M, K, GENERIC_FEATURE_DIM)
    generic_labels: np.ndarray  # (2M,), 0 for in-library, K for out
    built_seeds: tuple[int, ...] = ()  # the train seeds that produced rows
    excluded_seeds: tuple[dict[str, Any], ...] = ()  # non-instances, with reasons


# ---------------------------------------------------------------------------
# Hashing, git, and atomic publication helpers.


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_lower_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_sha256(value: Any, *, label: str) -> None:
    if not _is_lower_hex(value, 64):
        raise RuntimeError(f"stop condition: {label} must be a lowercase SHA-256")


def _verified_sha256(
    project_root: Path, relative_path: str, expected: str, *, label: str
) -> str:
    _validate_sha256(expected, label=f"expected {label} fingerprint")
    path = project_root / relative_path
    try:
        actual = _sha256(path)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"stop condition: {label} is missing at {relative_path}"
        ) from error
    if actual != expected:
        raise RuntimeError(
            f"stop condition: {label} SHA-256 is {actual}, expected {expected}"
        )
    return actual


def _git_checked(project_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"stop condition: git {' '.join(args)} timed out after 15s"
        ) from error
    if result.returncode != 0:
        detail = result.stderr.strip()
        if not detail:
            detail = (
                f"git exited {result.returncode} with no stderr "
                "(e.g. merge-base --is-ancestor reports a non-ancestor "
                "this way)"
            )
        raise RuntimeError(
            f"stop condition: git {' '.join(args)} failed: {detail}"
        )
    return result.stdout.strip()


def _git_head_blob(project_root: Path, relative_path: str) -> bytes:
    """The exact bytes of a blob committed at HEAD (no text decoding)."""
    result = subprocess.run(
        ("git", "show", f"HEAD:{relative_path}"),
        cwd=project_root,
        check=False,
        capture_output=True,
        timeout=15,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"stop condition: git show HEAD:{relative_path} failed: "
            f"{detail or 'unknown git error'}"
        )
    return result.stdout


def _code_manifest(project_root: Path) -> dict[str, str]:
    """Per-file SHA-256 of every ``src/universa/*.py``, sorted by path."""
    directory = project_root / "src" / "universa"
    files = sorted(directory.glob("*.py"), key=lambda path: path.name)
    if not files:
        raise RuntimeError(
            "stop condition: no src/universa/*.py files found under the "
            "project root"
        )
    return {
        file.relative_to(project_root).as_posix(): _sha256(file)
        for file in files
    }


def _manifest_digest(manifest: dict[str, str]) -> str:
    canonical = json.dumps(manifest, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _atomic_json_new(path: Path, payload: dict[str, Any]) -> None:
    """Atomically publish JSON without ever replacing an existing result."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError(
                f"stop condition: output appeared during execution: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Seal parsing and validation.


def _validate_seed_block(block: Any, expected: tuple[int, ...], *, label: str) -> None:
    if (
        not isinstance(block, dict)
        or set(block) - {"first", "last"}
        or block.get("first") != expected[0]
        or block.get("last") != expected[-1]
    ):
        raise RuntimeError(
            f"stop condition: design seal {label} must be exactly "
            f"{{'first': {expected[0]}, 'last': {expected[-1]}}}"
        )


def _load_seal(project_root: Path, seal_relative: str) -> dict[str, Any]:
    """Parse the committed design seal and validate its frozen structure."""

    path = project_root / seal_relative
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise RuntimeError(
            f"stop condition: design seal is missing at {seal_relative}"
        ) from error
    try:
        seal = json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"stop condition: design seal is not valid JSON: {error}"
        ) from error
    if not isinstance(seal, dict):
        raise RuntimeError("stop condition: design seal must be a JSON object")
    if seal.get("schema") != SEAL_SCHEMA:
        raise RuntimeError(
            f"stop condition: design seal schema must be {SEAL_SCHEMA!r}, got "
            f"{seal.get('schema')!r}"
        )
    required = (
        "schema",
        "design_commit",
        "protocol_sha256",
        "runner_sha256",
        "code_manifest",
        "train_seed_block",
        "eval_seed_block",
        "no_preview_declaration",
        "primary_family",
        "stop_rules",
        "output_path",
    )
    missing = [key for key in required if key not in seal]
    if missing:
        raise RuntimeError(
            f"stop condition: design seal is missing keys: {missing}"
        )
    unknown = sorted(key for key in seal if key not in required)
    if unknown:
        raise RuntimeError(
            f"stop condition: design seal has unknown keys: {unknown}"
        )
    if not _is_lower_hex(seal["design_commit"], 40):
        raise RuntimeError(
            "stop condition: design seal design_commit must be a full "
            "lowercase commit hash"
        )
    # The pinned design commit must exist in this repository AND be an
    # ancestor of HEAD: existence alone admits a dangling commit object
    # (e.g. one produced by git commit-tree) that no reachable history
    # contains.
    _git_checked(
        project_root, "cat-file", "-e", seal["design_commit"] + "^{commit}"
    )
    _git_checked(
        project_root,
        "merge-base",
        "--is-ancestor",
        seal["design_commit"],
        "HEAD",
    )
    _validate_sha256(seal["protocol_sha256"], label="design seal protocol_sha256")
    _validate_sha256(seal["runner_sha256"], label="design seal runner_sha256")
    _validate_seed_block(
        seal["train_seed_block"], TRAIN_SEEDS, label="train_seed_block"
    )
    _validate_seed_block(
        seal["eval_seed_block"], SEALED_EVAL_SEEDS, label="eval_seed_block"
    )
    declaration = seal["no_preview_declaration"]
    if not isinstance(declaration, str) or not declaration.strip():
        raise RuntimeError(
            "stop condition: design seal no_preview_declaration must be a "
            "nonempty string"
        )
    manifest = seal["code_manifest"]
    if not isinstance(manifest, dict) or not manifest:
        raise RuntimeError(
            "stop condition: design seal code_manifest must be a nonempty "
            "object of per-file SHA-256 values"
        )
    for key, value in manifest.items():
        if (
            not isinstance(key, str)
            or not key.startswith("src/universa/")
            or not key.endswith(".py")
            or key.count("/") != 2
        ):
            raise RuntimeError(
                "stop condition: design seal code_manifest keys must be "
                f"src/universa/*.py paths, got {key!r}"
            )
        _validate_sha256(value, label=f"design seal code_manifest[{key!r}]")
    family = seal["primary_family"]
    if not isinstance(family, list) or len(family) != NUM_CLAIMS:
        raise RuntimeError(
            "stop condition: design seal primary_family must list exactly "
            "the four frozen claims"
        )
    for claim in family:
        if not isinstance(claim, dict):
            raise RuntimeError(
                "stop condition: every design seal primary_family entry "
                "must be a claim object"
            )
        missing_claim_keys = [key for key in _CLAIM_SEAL_KEYS if key not in claim]
        if missing_claim_keys:
            raise RuntimeError(
                "stop condition: design seal primary_family entry "
                f"{claim.get('id')!r} is missing keys: {missing_claim_keys}"
            )
        unknown_claim_keys = sorted(
            key
            for key in claim
            if key not in _CLAIM_SEAL_KEYS and key != "role"
        )
        if unknown_claim_keys:
            raise RuntimeError(
                "stop condition: design seal primary_family entry "
                f"{claim.get('id')!r} has unknown keys: {unknown_claim_keys}"
            )
        if not isinstance(claim["id"], str):
            raise RuntimeError(
                "stop condition: every design seal primary_family entry "
                "needs a string id"
            )
        if any(
            not isinstance(claim[key], str) or not claim[key].strip()
            for key in ("theta", "null", "alternative", "support_rule")
        ):
            raise RuntimeError(
                "stop condition: design seal primary_family entry "
                f"{claim.get('id')!r} must carry the descriptive subfields "
                "(theta, null, alternative, support_rule) as nonempty "
                "strings"
            )
    if tuple(claim["id"] for claim in family) != CLAIM_IDS:
        raise RuntimeError(
            "stop condition: design seal primary_family claim ids must be "
            f"exactly {list(CLAIM_IDS)} in order"
        )
    for claim, definition in zip(family, _CLAIM_DEFINITIONS):
        if (
            claim["statistic"] != definition["statistic"]
            or claim["threshold"] != definition["threshold"]
            or claim["bound_direction"] != "greater"
            or ("role" in claim and claim["role"] != definition["role"])
        ):
            raise RuntimeError(
                "stop condition: design seal primary_family claim "
                f"{claim['id']!r} does not match the frozen definition "
                "(statistic, threshold, bound_direction, role)"
            )
    stop_rules = seal["stop_rules"]
    if (
        not isinstance(stop_rules, list)
        or not stop_rules
        or any(not isinstance(rule, str) or not rule.strip() for rule in stop_rules)
    ):
        raise RuntimeError(
            "stop condition: design seal stop_rules must be a nonempty list "
            "of nonempty strings"
        )
    if not isinstance(seal["output_path"], str) or not seal["output_path"]:
        raise RuntimeError(
            "stop condition: design seal output_path must be a nonempty string"
        )
    return seal


# ---------------------------------------------------------------------------
# Environment and preflight.


def _frozen_protocol_sha256() -> str:
    """The runner's pinned protocol fingerprint.

    The committed runner embeds the sealed protocol hash; any protocol edit
    requires re-pinning it. The refusal below is what fires while the value
    is still the placeholder.
    """
    if not _is_lower_hex(PROTOCOL_SHA256, 64):
        raise RuntimeError(
            "stop condition: the runner's PROTOCOL_SHA256 is still the "
            "placeholder PENDING_PROTOCOL_SHA256; the committed runner must "
            "embed the sealed protocol hash, and any protocol edit requires "
            "re-pinning it"
        )
    return PROTOCOL_SHA256


def _execution_environment() -> dict[str, Any]:
    """Pin single-thread CPU float32 execution and refuse a visible CUDA.

    The three learned models compute in torch float32 on CPU; the certified
    machinery (features, discovery, audits) is numpy float64. Both are
    probed here, fail-closed.
    """
    torch.set_num_threads(1)
    if torch.get_num_threads() != 1:
        raise RuntimeError(
            "stop condition: PyTorch must run with exactly one thread, got "
            f"{torch.get_num_threads()}"
        )
    if torch.cuda.is_available():
        raise RuntimeError(
            "stop condition: CUDA must be unavailable or hidden "
            "(run with CUDA_VISIBLE_DEVICES=-1)"
        )
    probe = torch.tensor([1.0, -2.0], dtype=torch.float32, device="cpu")
    reduced = float((probe * 3.0).sum())
    if (
        probe.dtype is not torch.float32
        or probe.device.type != "cpu"
        or reduced != -3.0
    ):
        raise RuntimeError("stop condition: CPU float32 tensor operation failed")
    numpy_probe = np.array([1.0, -2.0], dtype=np.float64)
    numpy_reduced = float((numpy_probe * 3.0).sum())
    if numpy_probe.dtype != np.float64 or numpy_reduced != -3.0:
        raise RuntimeError(
            "stop condition: CPU float64 numpy operation failed"
        )
    return {
        "compute": (
            "torch float32 on CPU for the three learned models "
            "(StructureRouter, LearnedAlarmV2, GenericMLP); numpy float64 for "
            "the certified machinery (features, discovery, audits)"
        ),
        "tensor_device": "cpu",
        "tensor_dtype": "float32",
        "array_dtype": "float64",
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_visible_devices_operator": OPERATOR_CUDA_VISIBLE_DEVICES,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "argv": list(sys.argv),
    }


def _environment_provenance() -> dict[str, Any]:
    uname = platform.uname()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": {
            "system": uname.system,
            "node": uname.node,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "processor": uname.processor,
        },
        "torch_cuda_build": torch.version.cuda,
    }


def _preflight(project_root: Path, output: Path, *, seal: str) -> dict[str, Any]:
    """Fail closed before constructing any instance from a sealed seed.

    Frozen order: output missing -> clean worktree -> seal committed at HEAD
    -> seal validation -> file hashes -> runtime hash -> environment. No data
    construction of any kind may happen before this returns.
    """

    # 1. The output must not exist and must lie inside the project root.
    if output.exists():
        raise RuntimeError(f"stop condition: output already exists: {output}")
    try:
        output_relative = output.relative_to(project_root).as_posix()
    except ValueError as error:
        raise RuntimeError(
            "stop condition: output must lie inside the project root"
        ) from error

    # 2. The worktree must be clean; nothing is opened on a dirty tree.
    status = _git_checked(
        project_root, "status", "--porcelain", "--untracked-files=all"
    )
    if status:
        raise RuntimeError(
            "stop condition: working tree is dirty; no seed was opened"
        )

    # 3. The seal itself must be committed at HEAD, and the committed blob
    # must equal the on-disk bytes: a clean worktree alone does not catch an
    # assume-unchanged/skip-worktree swap. A symlinked seal is refused
    # because git stores symlinks as target text, which would defeat the
    # blob comparison.
    seal_path = Path(seal)
    if seal_path.is_absolute():
        try:
            seal_path = seal_path.relative_to(project_root)
        except ValueError as error:
            raise RuntimeError(
                "stop condition: the design seal must lie inside the project root"
            ) from error
    seal_relative = seal_path.as_posix()
    _git_checked(project_root, "cat-file", "-e", f"HEAD:{seal_relative}")
    if (project_root / seal_relative).is_symlink():
        raise RuntimeError(
            f"stop condition: the design seal must not be a symlink: "
            f"{seal_relative}"
        )
    if _git_head_blob(project_root, seal_relative) != (
        project_root / seal_relative
    ).read_bytes():
        raise RuntimeError(
            "stop condition: the on-disk design seal differs from the seal "
            "blob committed at HEAD"
        )

    # 4. Seal structure and frozen content.
    seal_record = _load_seal(project_root, seal_relative)
    if seal_record["output_path"] != output_relative:
        raise RuntimeError(
            "stop condition: design seal output_path "
            f"{seal_record['output_path']!r} does not match --output "
            f"{output_relative!r}"
        )

    # 5. File hashes: the sealed protocol and the universa code manifest.
    # The imported universa package must be the project's own copy: with a
    # relative PYTHONPATH from a foreign working directory, a shadow
    # package could satisfy the import while the manifest check below
    # hashes the project's files.
    package_root = (project_root / "src" / "universa").resolve()
    try:
        Path(universa.__file__).resolve().relative_to(package_root)
    except ValueError as error:
        raise RuntimeError(
            "stop condition: the imported universa package is not the "
            f"project's own copy: {universa.__file__} is not under "
            f"{package_root}"
        ) from error
    protocol_constant = _frozen_protocol_sha256()
    if seal_record["protocol_sha256"] != protocol_constant:
        raise RuntimeError(
            "stop condition: embedded protocol fingerprint differs from the "
            "design seal"
        )
    if (project_root / PROTOCOL).is_symlink():
        raise RuntimeError(
            f"stop condition: the sealed protocol must not be a symlink: "
            f"{PROTOCOL}"
        )
    protocol_hash = _verified_sha256(
        project_root, PROTOCOL, protocol_constant, label="sealed protocol"
    )
    manifest = _code_manifest(project_root)
    if seal_record["code_manifest"] != manifest:
        raise RuntimeError(
            "stop condition: the design seal code_manifest does not match "
            "the on-disk src/universa/*.py files"
        )

    # 6. The running file must be exactly the sealed runner.
    if (project_root / RUNNER_SOURCE).is_symlink():
        raise RuntimeError(
            f"stop condition: the sealed runner must not be a symlink: "
            f"{RUNNER_SOURCE}"
        )
    runner_hash = _sha256(Path(__file__).resolve())
    if runner_hash != seal_record["runner_sha256"]:
        raise RuntimeError(
            "stop condition: the running file's SHA-256 differs from the "
            "sealed runner"
        )

    # 7. Environment: CUDA hidden, one torch thread, CPU float32 torch and
    # CPU float64 numpy.
    execution = _execution_environment()
    environment = _environment_provenance()

    revision = _git_checked(project_root, "rev-parse", "HEAD")
    return {
        "git_revision": revision,
        "execution_revision": revision,
        "clean_worktree": True,
        "git_status_porcelain": status,
        "seal": {
            "path": seal_relative,
            "schema": SEAL_SCHEMA,
            "sha256": _sha256(project_root / seal_relative),
            "design_commit": seal_record["design_commit"],
            "committed_at_head": True,
        },
        "protocol": {"path": PROTOCOL, "sha256": protocol_hash},
        "runner": {"path": RUNNER_SOURCE, "sha256": runner_hash},
        "code_manifest": {
            "glob": "src/universa/*.py",
            "sha256": _manifest_digest(manifest),
            "files": manifest,
        },
        "environment": environment,
        "execution": execution,
    }


# ---------------------------------------------------------------------------
# Frozen schedules, views, and the undegraded audit.


def _observation_seed(seed: int) -> int:
    """The canonical shared observation draw of one seed's row.

    ``subseed(seed, "router-v2-observe")`` — the router-v2 regime's
    canonical observation family: ONE draw per seed, reused across every
    candidate at every grid point, across every arm, and across all three
    conditions of the row (exact pairing), for train and eval rows alike —
    so the learned models' training rows and the loop rows come from one
    observation distribution on disjoint seed blocks. The mask permutation
    derives from the seed alone and the mask fraction is constant, so the
    same edge columns are missing at every grid point, and corruption is
    nested per fraction. Consequently the arch and routing arms' feature
    blocks are identical in the out-of-library and null-control conditions
    per seed (the alarm's decision basis precedes observations).
    """
    return subseed(seed, "router-v2-observe")


def _paired_views(instance: Any) -> tuple[Any, Any]:
    """The equal-K paired library views of one instance (loop_v2 canonical).

    ``in_library = (instance.true_target, *instance.decoy_targets[:-1])``
    (the truth at index 0) and ``out_library = instance.decoy_targets``
    (the truth withheld) — both of size K = ``NUM_VIEW_CANDIDATES``, the
    fixed input width the learned alarm and the generic head require.
    Fail-closed on any deviation from the frozen family shape.
    """
    decoys = tuple(instance.decoy_targets)
    if len(decoys) != NUM_DECOYS:
        raise DesignFailureError(
            f"seed {instance.seed}: expected {NUM_DECOYS} decoys, got "
            f"{len(decoys)}",
            seed=instance.seed,
        )
    in_library = (instance.true_target, *decoys[:-1])
    out_library = decoys
    if len(in_library) != NUM_VIEW_CANDIDATES or len(out_library) != (
        NUM_VIEW_CANDIDATES
    ):
        raise DesignFailureError(
            f"seed {instance.seed}: the paired views must both have size "
            f"K = {NUM_VIEW_CANDIDATES}",
            seed=instance.seed,
        )
    return in_library, out_library


def _undegraded_misfits(instance: Any) -> tuple[float, ...]:
    """Per-candidate max-degree clean commutation residual.

    The frozen eligibility/train audit: the true candidate's score must be
    exactly 0.0 and every decoy strictly above AUDIT_TOL. Deterministic per
    seed, computed on the UNDEGRADED instance purely as bookkeeping for seed
    accounting — it never enters any feature (every feature is computed
    against a degraded operator).
    """
    misfits = []
    for candidate in instance.candidates:
        residuals = ChainMap(
            instance.source, candidate, instance.chain_map.maps
        ).commutation_residuals()
        if not residuals:
            raise DesignFailureError(
                f"seed {instance.seed}: expected at least one commutation "
                "residual for the undegraded audit",
                seed=instance.seed,
            )
        misfits.append(float(max(residuals)))
    return tuple(misfits)


def _audit_undegraded(seed: int, misfits: tuple[float, ...]) -> str | None:
    """The frozen audit decision: a reason string, or None when the seed
    passes (true misfit exactly 0.0, every decoy strictly above AUDIT_TOL)."""
    if misfits[0] != 0.0:
        return f"true candidate undegraded misfit {misfits[0]} != 0.0"
    for misfit in misfits[1:]:
        if misfit <= AUDIT_TOL:
            return f"decoy undegraded misfit {misfit} <= {AUDIT_TOL}"
    return None


def _structured_observations(seed: int, instance: Any) -> np.ndarray:
    """The exact transported observations ``Y = f1 A`` of one seed.

    The frozen exp4 ``"discovery-observation"`` schedule:
    :func:`universa.discovery.synthesize_observations` of a
    :class:`~universa.generators.SwitchInstance` view of the budget
    instance (the same underlying arrays), M = 16 columns. A violation of
    the certified schedule is a whole-run design failure.
    """
    switch_view = SwitchInstance(
        instance.seed,
        instance.source,
        instance.true_target,
        instance.chain_map,
        instance.decoy_targets,
    )
    try:
        observations = synthesize_observations(switch_view, NUM_OBSERVATIONS)
    except DesignFailureError:
        raise
    except Exception as error:
        raise DesignFailureError(
            f"seed {seed}: structured observation schedule failed: "
            f"{type(error).__name__}: {error}",
            seed=seed,
        ) from error
    return observations


def _null_observations(seed: int, instance: Any) -> np.ndarray:
    """The structure-free null observations of one seed.

    The frozen exp4 H4 schedule: 16 i.i.d. standard Gaussian columns on the
    seed's target edge space, column ``j`` drawn from
    ``np.random.default_rng(subseed(seed, "discovery-null", str(j)))`` — a
    disjoint subseed family from the structured condition's
    ``"discovery-observation"`` draws.
    """
    ambient_dim = int(instance.true_target.boundaries[0].shape[1])
    try:
        return null_observations(seed, ambient_dim, NUM_OBSERVATIONS)
    except DesignFailureError:
        raise
    except Exception as error:
        raise DesignFailureError(
            f"seed {seed}: null observation schedule failed: "
            f"{type(error).__name__}: {error}",
            seed=seed,
        ) from error


def _condition_inputs(
    seed: int, instance: Any
) -> dict[str, tuple[Any, np.ndarray]]:
    """One row's (library view, observations) per condition, computed once
    per seed: the structured conditions share the seed's exact transported
    observations; the null control replaces them with the structure-free
    schedule on the same out-of-library view."""
    in_library, out_library = _paired_views(instance)
    structured = _structured_observations(seed, instance)
    nulls = _null_observations(seed, instance)
    return {
        "in_library": (in_library, structured),
        "out_of_library": (out_library, structured),
        "null_control": (out_library, nulls),
    }


# ---------------------------------------------------------------------------
# Train block construction and the three frozen trainings.


def _build_train_block() -> _TrainBlock:
    """The frozen train block: one row per train seed x both views.

    Built by the runner itself with the frozen schedules — per seed the
    equal-K paired views, the arch no-anchor profile blocks and raw
    profiles under the seed's shared ``"router-v2-observe"`` draw, and
    the generic spectral blocks at the seed's operating grid point over
    the exact transported observations.

    **Train-side build exclusion (protocol errata 1, docs/30).** A train
    seed whose INSTANCE fails to build is not a training example — it is
    not an instance of the family at all: the generator's own
    discriminability guard refuses a decoy that shares the true target's
    kernel, exactly as ``docs/00-design.md`` §6 records. Such a seed is
    recorded with its reason and EXCLUDED, mirroring how the eval side
    already treats an ineligible seed; the models train on the seeds that
    build, and the excluded seeds and the built count are reported in the
    training provenance. At most :data:`MAX_TRAIN_EXCLUDED` of the
    declared train seeds may fail to build, or the run stops as a
    whole-run design failure — the fail-closed guard over the exclusion itself.

    Everything else keeps the original discipline: a FEATURE-construction
    exception, a shape violation, or a non-finite feature is a whole-run
    design failure, never an exclusion. A seed that builds but whose
    features fail indicates a pipeline fault, not a non-instance.
    """
    in_blocks: list[np.ndarray] = []
    in_raws: list[np.ndarray] = []
    out_blocks: list[np.ndarray] = []
    out_raws: list[np.ndarray] = []
    generic_features: list[np.ndarray] = []
    generic_labels: list[int] = []
    built_seeds: list[int] = []
    excluded_seeds: list[dict[str, Any]] = []
    for seed in TRAIN_SEEDS:
        try:
            instance = make_budget_instance(
                seed, NUM_VERTICES, NUM_EDGES, NUM_CLASSES, NUM_DECOYS
            )
        except Exception as error:
            # Protocol errata 1: a non-instance is excluded and recorded,
            # never silently dropped and never fatal on its own.
            excluded_seeds.append(
                {
                    "seed": seed,
                    "reason": "instance_build_failed",
                    "exception_type": type(error).__name__,
                    "message": str(error),
                }
            )
            continue
        in_library, out_library = _paired_views(instance)
        observation_seed = _observation_seed(seed)
        try:
            in_block, in_raw = arch_row_features(
                instance, in_library, observation_seed
            )
            out_block, out_raw = arch_row_features(
                instance, out_library, observation_seed
            )
        except DesignFailureError:
            raise
        except Exception as error:
            raise DesignFailureError(
                f"train seed {seed}: certified feature construction "
                f"failed: {type(error).__name__}: {error}"
            ) from error
        observations = _structured_observations(seed, instance)
        point = operating_grid_point(seed)
        try:
            in_generic = generic_row_features(
                instance, in_library, observation_seed, observations, point
            )
            out_generic = generic_row_features(
                instance, out_library, observation_seed, observations, point
            )
        except DesignFailureError:
            raise
        except Exception as error:
            raise DesignFailureError(
                f"train seed {seed}: generic feature construction failed: "
                f"{type(error).__name__}: {error}"
            ) from error
        for label, block, raw in (
            ("in-library", in_block, in_raw),
            ("out-of-library", out_block, out_raw),
        ):
            if block.shape != (NUM_VIEW_CANDIDATES, FEATURE_DIM):
                raise DesignFailureError(
                    f"train seed {seed}: {label} arch block has shape "
                    f"{block.shape}, expected "
                    f"{(NUM_VIEW_CANDIDATES, FEATURE_DIM)}"
                )
            if raw.shape != (NUM_VIEW_CANDIDATES, len(PROFILE_GRID)):
                raise DesignFailureError(
                    f"train seed {seed}: {label} raw profile block has "
                    f"shape {raw.shape}, expected "
                    f"{(NUM_VIEW_CANDIDATES, len(PROFILE_GRID))}"
                )
            if not np.isfinite(block).all() or not np.isfinite(raw).all():
                raise DesignFailureError(
                    f"train seed {seed}: non-finite {label} arch features"
                )
        for label, block in (
            ("in-library", in_generic),
            ("out-of-library", out_generic),
        ):
            if block.ndim != 2 or block.shape[0] != NUM_VIEW_CANDIDATES:
                raise DesignFailureError(
                    f"train seed {seed}: {label} generic block has shape "
                    f"{block.shape}, expected K = {NUM_VIEW_CANDIDATES} rows"
                )
            if not np.isfinite(block).all():
                raise DesignFailureError(
                    f"train seed {seed}: non-finite {label} generic features"
                )
        in_blocks.append(in_block)
        in_raws.append(in_raw)
        out_blocks.append(out_block)
        out_raws.append(out_raw)
        generic_features.append(in_generic)
        generic_labels.append(0)  # the true candidate's index, unpermuted
        generic_features.append(out_generic)
        generic_labels.append(NUM_VIEW_CANDIDATES)  # K: the no-fit class
        built_seeds.append(seed)
    num_seeds = len(built_seeds)
    if len(excluded_seeds) > MAX_TRAIN_EXCLUDED:
        raise DesignFailureError(
            f"{len(excluded_seeds)} of {len(TRAIN_SEEDS)} declared train "
            f"seeds failed to build an instance, above the frozen ceiling "
            f"{MAX_TRAIN_EXCLUDED}; "
            f"excluded: {[record['seed'] for record in excluded_seeds]}"
        )
    if not built_seeds:
        raise DesignFailureError(
            "no declared train seed built an instance"
        )
    block = _TrainBlock(
        in_blocks=np.stack(in_blocks),
        in_raws=np.stack(in_raws),
        out_blocks=np.stack(out_blocks),
        out_raws=np.stack(out_raws),
        generic_features=np.stack(generic_features),
        generic_labels=np.asarray(generic_labels, dtype=np.int64),
        built_seeds=tuple(built_seeds),
        excluded_seeds=tuple(excluded_seeds),
    )
    expected_router = (num_seeds, NUM_VIEW_CANDIDATES, FEATURE_DIM)
    if block.in_blocks.shape != expected_router:
        raise DesignFailureError(
            f"router train block has shape {block.in_blocks.shape}, "
            f"expected {expected_router}"
        )
    expected_generic = (2 * num_seeds, NUM_VIEW_CANDIDATES, block.generic_features.shape[-1])
    if block.generic_features.shape != expected_generic:
        raise DesignFailureError(
            f"generic train block has shape {block.generic_features.shape}, "
            f"expected {expected_generic}"
        )
    return block


def _model_state_sha256(model: Any) -> str:
    """Canonical model-state fingerprint of the frozen design.

    SHA-256 over the concatenation, in lexicographic name order, of each
    ``state_dict`` entry's UTF-8 name, a NUL byte, and its little-endian
    float32 bytes.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(array.astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _validate_calibration_record(record: Any) -> None:
    """Fail-closed shape check of the frozen eight-key calibration record.

    The record is what
    :func:`universa.loop_v2.calibrate_threshold_cost_aware` returns; the
    runner re-validates it before the threshold is allowed anywhere near
    an eval row: exactly the eight frozen keys, the threshold and the
    three rates finite floats in ``[0, 1]``, the two costs equal to this
    experiment's frozen constants, ``total_cost`` a finite nonnegative
    float consistent with the rates and costs, and ``num_candidates`` an
    int of at least 2 (the sweep always includes the 0.0 and 1.0
    candidates).

    There is no ``bound_satisfied`` key: the cost-aware rule has no
    feasibility bound and therefore no fallback branch — it always
    selects the sweep's cost minimizer.
    """
    expected = {
        "threshold",
        "balanced_accuracy",
        "false_quiet_rate",
        "false_alarm_rate",
        "total_cost",
        "false_quiet_cost",
        "false_alarm_cost",
        "num_candidates",
    }
    if not isinstance(record, dict) or set(record) != expected:
        raise DesignFailureError(
            "the calibration record must carry exactly the eight frozen "
            f"keys {sorted(expected)}"
        )
    threshold = record["threshold"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, float)
        or not math.isfinite(threshold)
        or not 0.0 <= threshold <= 1.0
    ):
        raise DesignFailureError(
            "the calibration record's threshold must be a float in [0, 1]"
        )
    for key in ("balanced_accuracy", "false_quiet_rate", "false_alarm_rate"):
        value = record[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, float)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise DesignFailureError(
                f"the calibration record's {key} must be a float in [0, 1]"
            )
    for key, frozen in (
        ("false_quiet_cost", FALSE_QUIET_COST),
        ("false_alarm_cost", FALSE_ALARM_COST),
    ):
        value = record[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, float)
            or not math.isfinite(value)
            or value != frozen
        ):
            raise DesignFailureError(
                f"the calibration record's {key} must be the frozen "
                f"constant {frozen}"
            )
    total_cost = record["total_cost"]
    if (
        isinstance(total_cost, bool)
        or not isinstance(total_cost, float)
        or not math.isfinite(total_cost)
        or total_cost < 0.0
    ):
        raise DesignFailureError(
            "the calibration record's total_cost must be a nonnegative "
            "finite float"
        )
    recomputed = (
        record["false_quiet_cost"] * record["false_quiet_rate"]
        + record["false_alarm_cost"] * record["false_alarm_rate"]
    )
    if abs(total_cost - recomputed) > 1e-12:
        raise DesignFailureError(
            "the calibration record's total_cost must equal "
            "false_quiet_cost * false_quiet_rate + false_alarm_cost * "
            f"false_alarm_rate (recorded {total_cost}, recomputed "
            f"{recomputed})"
        )
    num_candidates = record["num_candidates"]
    if (
        isinstance(num_candidates, bool)
        or not isinstance(num_candidates, int)
        or num_candidates < 2
    ):
        raise DesignFailureError(
            "the calibration record's num_candidates must be an int >= 2"
        )


def _train_models(
    block: _TrainBlock,
) -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any]]:
    """Train the THREE frozen models on the train block, in the frozen
    order — StructureRouter (seed 4242), LearnedAlarmV2 (seed 4243,
    reading the trained router's gates), GenericMLP (seed 4244) — then
    calibrate the alarm's decision threshold on the same train rows
    under the frozen cost-aware rule at unit costs, and record every
    model state SHA-256, the final training scalars, and the full
    eight-key calibration record.

    A violation inside the certified training machinery (e.g. a non-finite
    loss) is a whole-run design failure, never an execution failure — the
    same classification :func:`_build_train_block` applies.
    """
    # 1. The StructureRouter on the in-library rows, label 0 for every row
    # — the loop's unpermuted index-0 convention (the protocol's declared
    # consequences: the router's learned role is profile-shape scoring
    # under the index-0 commitment, and the loop's learned selectivity
    # lives in the alarm; the out-of-library view carries no routing label
    # and contributes no router rows). The frozen design defines no
    # validation split; the train block itself is passed as the validation
    # monitor (history-only — it cannot affect the fitted parameters), so
    # only train seeds are ever seen.
    router_labels = np.zeros(block.in_blocks.shape[0], dtype=np.int64)
    dataset = (block.in_blocks, router_labels, ())
    try:
        router, router_history = train_router(
            dataset,
            dataset,
            EPOCHS,
            lr=LEARNING_RATE,
            seed=TORCH_SEED_ROUTER,
            lambda_aux=LAMBDA_AUX,
            tau_start=TAU_START,
            tau_end=TAU_END,
            hidden_dim=ROUTER_HIDDEN_DIM,
            feature_dim=FEATURE_DIM,
        )
    except Exception as error:
        raise DesignFailureError(
            f"the frozen router training call failed: "
            f"{type(error).__name__}: {error}"
        ) from error
    # 2. The LearnedAlarmV2: fit/no-fit labels from the in-library vs
    # out-of-library rows, reading the TRAINED router's soft gates
    # (tau = 1.0) over the unpermuted blocks — the rows exactly as the loop
    # sees them at eval time — in the alarm_features_v2 layout (the frozen
    # v1 columns plus the five margin features). Constructed under the
    # frozen seed for deterministic initialization (train_alarm_v2's
    # documented caller concern), then trained under the same seed.
    try:
        rows_fit = np.stack(
            [
                alarm_features_v2(router_gates(router, in_block), in_raw)
                for in_block, in_raw in zip(block.in_blocks, block.in_raws)
            ]
        )
        rows_nofit = np.stack(
            [
                alarm_features_v2(router_gates(router, out_block), out_raw)
                for out_block, out_raw in zip(block.out_blocks, block.out_raws)
            ]
        )
        torch.manual_seed(TORCH_SEED_ALARM)
        alarm = LearnedAlarmV2(NUM_VIEW_CANDIDATES)
        alarm, alarm_history = train_alarm_v2(
            alarm,
            rows_fit,
            rows_nofit,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            seed=TORCH_SEED_ALARM,
        )
    except DesignFailureError:
        raise
    except Exception as error:
        raise DesignFailureError(
            f"the frozen alarm training call failed: "
            f"{type(error).__name__}: {error}"
        ) from error
    # 2b. The frozen train-block calibration: the alarm's decision
    # threshold is picked on the SAME train rows — the threshold
    # MINIMIZING false_quiet_cost * false_quiet_rate + false_alarm_cost *
    # false_alarm_rate at the frozen unit costs, ties toward the larger
    # balanced accuracy then the larger threshold. No feasibility bound
    # and no fallback branch. Deterministic (no RNG anywhere).
    try:
        calibration = calibrate_threshold_cost_aware(
            alarm,
            rows_fit,
            rows_nofit,
            false_quiet_cost=FALSE_QUIET_COST,
            false_alarm_cost=FALSE_ALARM_COST,
        )
    except DesignFailureError:
        raise
    except Exception as error:
        raise DesignFailureError(
            f"the frozen threshold calibration call failed: "
            f"{type(error).__name__}: {error}"
        ) from error
    _validate_calibration_record(calibration)
    # 3. The GenericMLP: index/no-fit labels (in-library rows labeled 0,
    # the true candidate's index under the unpermuted loop convention;
    # out-of-library rows labeled K, no-fit). Constructed under the frozen
    # seed for deterministic initialization (train_generic's documented
    # caller concern), then trained under the same seed.
    try:
        torch.manual_seed(TORCH_SEED_GENERIC)
        generic = GenericMLP(NUM_VIEW_CANDIDATES)
        generic, generic_history = train_generic(
            generic,
            block.generic_features,
            block.generic_labels,
            epochs=EPOCHS,
            lr=LEARNING_RATE,
            seed=TORCH_SEED_GENERIC,
        )
    except DesignFailureError:
        raise
    except Exception as error:
        raise DesignFailureError(
            f"the frozen generic training call failed: "
            f"{type(error).__name__}: {error}"
        ) from error
    provenance = {
        "train_block": {
            "declared_seeds": len(TRAIN_SEEDS),
            "declared_first": TRAIN_SEEDS[0],
            "declared_last": TRAIN_SEEDS[-1],
            "built_seeds": len(block.built_seeds),
            "excluded_seeds": [dict(record) for record in block.excluded_seeds],
            "max_excluded_ceiling": MAX_TRAIN_EXCLUDED,
            "exclusion_rule": (
                "protocol errata 1: a train seed whose INSTANCE fails to "
                "build is not an instance of the family (the budgets "
                "discriminability guard refuses a decoy sharing the true "
                "target's kernel) and is recorded here and excluded, "
                "mirroring the eval side's ineligibility record; a "
                "feature-construction failure, a shape violation, or a "
                "non-finite feature remains a whole-run design_failure, "
                "never an exclusion; more than max_excluded_ceiling "
                "non-instances is itself a whole-run design_failure"
            ),
        },
        "router": {
            "torch_seed": TORCH_SEED_ROUTER,
            "epochs": EPOCHS,
            "lr": LEARNING_RATE,
            "lambda_aux": LAMBDA_AUX,
            "tau_start": TAU_START,
            "tau_end": TAU_END,
            "hidden_dim": ROUTER_HIDDEN_DIM,
            "feature_dim": FEATURE_DIM,
            "optimizer": "full-batch Adam",
            "dtype_device": "CPU float32",
            "num_train_rows": int(router_labels.shape[0]),
            "labels": (
                "0 for every row — the loop's unpermuted index-0 "
                "convention (the true target sits at library index 0; the "
                "out-of-library view carries no routing label and "
                "contributes no router rows)"
            ),
            "standardization": "input mean/std measured on the train block only",
            "validation_monitor": (
                "the train block itself; the frozen design defines no "
                "validation split"
            ),
            "final": {
                key: float(router_history[key][-1])
                for key in (
                    "loss",
                    "cross_entropy",
                    "aux_loss",
                    "tau",
                    "train_accuracy",
                    "val_accuracy",
                    "gate_entropy",
                )
            },
            "model_state_sha256": _model_state_sha256(router),
        },
        "alarm": {
            "torch_seed": TORCH_SEED_ALARM,
            "epochs": EPOCHS,
            "lr": LEARNING_RATE,
            "hidden_dim": int(alarm.hidden_dim),
            "num_candidates": NUM_VIEW_CANDIDATES,
            "input": (
                "the alarm_features_v2 layout (K + 8 dims): the K soft "
                "router gates (tau = 1.0), the gate entropy (nats), "
                "log1p of the minimum raw profile value, log1p of the "
                "maximum profile slope magnitude, and the five margin "
                "features (log1p of the second-minus-best raw profile "
                "margin, the top-2 gate gap, log1p of the mean raw "
                "profile value, the best candidate's profile curvature, "
                "log1p of the best raw profile value)"
            ),
            "margin_feature_names": list(ALARM_V2_MARGIN_FEATURE_NAMES),
            "optimizer": "full-batch Adam",
            "dtype_device": "CPU float32",
            "num_fit_rows": int(rows_fit.shape[0]),
            "num_nofit_rows": int(rows_nofit.shape[0]),
            "labels": (
                "in-library rows labeled fit (1), out-of-library rows "
                "labeled no-fit (0)"
            ),
            "standardization": "input mean/std measured on the train rows only",
            "final": {
                key: float(alarm_history[key][-1])
                for key in ("loss", "train_accuracy")
            },
            "model_state_sha256": _model_state_sha256(alarm),
            "calibration": calibration,
            "calibration_costs": {
                "false_quiet_cost": FALSE_QUIET_COST,
                "false_alarm_cost": FALSE_ALARM_COST,
            },
            "calibration_rule": (
                "universa.loop_v2.calibrate_threshold_cost_aware on the "
                "SAME train rows: the threshold MINIMIZING "
                "false_quiet_cost * false_quiet_rate + false_alarm_cost "
                "* false_alarm_rate at the frozen unit costs (1.0, 1.0), "
                "ties toward the larger balanced accuracy then the larger "
                "threshold; no feasibility bound and no fallback branch. "
                "Replaces loop-v3's bounded rule, whose "
                "false_quiet_rate <= 0.02 constraint was binding. At "
                "equal unit costs false_quiet_rate + false_alarm_rate = "
                "2 - 2 * balanced_accuracy exactly, so the rule is "
                "identical to unconstrained balanced-accuracy "
                "maximization and the balanced-accuracy tiebreak never "
                "fires — declared in the protocol, not discovered"
            ),
        },
        "generic": {
            "torch_seed": TORCH_SEED_GENERIC,
            "epochs": EPOCHS,
            "lr": LEARNING_RATE,
            "hidden_dim": int(generic.hidden_dim),
            "feature_dim": int(generic.feature_dim),
            "num_candidates": NUM_VIEW_CANDIDATES,
            "optimizer": "full-batch Adam",
            "dtype_device": "CPU float32",
            "num_train_rows": int(block.generic_labels.shape[0]),
            "labels": (
                "in-library rows labeled the true candidate's index (0, "
                "the unpermuted loop convention); out-of-library rows "
                f"labeled K ({NUM_VIEW_CANDIDATES}), the no-fit class"
            ),
            "standardization": "input mean/std measured on the train rows only",
            "final": {
                key: float(generic_history[key][-1])
                for key in ("loss", "train_accuracy")
            },
            "model_state_sha256": _model_state_sha256(generic),
        },
    }
    return router, alarm, generic, calibration, provenance


# ---------------------------------------------------------------------------
# Row construction: one row per (seed, condition, arm).


def _execute_arm(
    seed: int,
    instance: Any,
    library: Any,
    observations: np.ndarray,
    *,
    condition: str,
    arm: str,
    observation_seed: int,
    router: Any,
    alarm: Any,
    threshold: float,
    generic: Any,
) -> tuple[ArmOutcome, float]:
    """One arm's pass over one row, design-failure wrapped and wall-timed.

    A violation inside the arm machinery — the certified feature builders,
    the learned models' decision functions, the certified discovery path —
    is a whole-run design failure, never an execution failure; a foreign
    outcome type or an outcome whose provenance disagrees with the request
    is one too (the symmetric check: without it a bad outcome would
    AttributeError downstream and classify as an execution failure).
    """
    try:
        start = time.perf_counter()
        if arm == "arch_full":
            outcome = arm_arch_full_v2(
                seed,
                instance,
                library,
                router=router,
                alarm=alarm,
                threshold=threshold,
                observation_seed=observation_seed,
                observations=observations,
                condition=condition,
            )
        elif arm == "routing_only":
            outcome = arm_routing_only(
                seed,
                instance,
                library,
                router=router,
                observation_seed=observation_seed,
                condition=condition,
            )
        elif arm == "discovery_only":
            outcome = arm_discovery_only(
                seed,
                instance,
                observations=observations,
                condition=condition,
            )
        elif arm == "generic":
            outcome = arm_generic(
                seed,
                instance,
                library,
                generic_model=generic,
                observation_seed=observation_seed,
                observations_y=observations,
                condition=condition,
            )
        else:
            raise DesignFailureError(
                f"seed {seed}: unknown arm {arm!r}", seed=seed
            )
        wall_time_seconds = time.perf_counter() - start
    except DesignFailureError:
        raise
    except Exception as error:
        raise DesignFailureError(
            f"seed {seed}: arm {arm} failed under condition {condition}: "
            f"{type(error).__name__}: {error}",
            seed=seed,
        ) from error
    if not isinstance(outcome, ArmOutcome):
        raise DesignFailureError(
            f"seed {seed}: arm {arm} returned a foreign outcome type: "
            f"{type(outcome).__name__}",
            seed=seed,
        )
    if (
        outcome.arm != arm
        or outcome.condition != condition
        or outcome.seed != seed
    ):
        raise DesignFailureError(
            f"seed {seed}: arm {arm} returned an outcome whose provenance "
            f"disagrees with the request (arm={outcome.arm!r}, "
            f"condition={outcome.condition!r}, seed={outcome.seed})",
            seed=seed,
        )
    return outcome, wall_time_seconds


def _build_arm_row(
    seed: int,
    instance: Any,
    library: Any,
    observations: np.ndarray,
    *,
    condition: str,
    arm: str,
    observation_seed: int,
    router: Any,
    alarm: Any,
    threshold: float,
    generic: Any,
) -> dict[str, Any]:
    """One raw row: the arm's ArmOutcome scalar record plus the retention
    block — the row's observation seed, the per-candidate arch raw profiles
    (the 8 raw observed-misfit values per candidate under the row's shared
    draw), and the wall time.

    The row records the raw outcome and the frozen correctness bit — it
    does NOT enforce them: a wrong route, a false admission, or a refusal
    is an honest negative result scored by the claims, not a design
    failure.
    """
    outcome, wall_time_seconds = _execute_arm(
        seed,
        instance,
        library,
        observations,
        condition=condition,
        arm=arm,
        observation_seed=observation_seed,
        router=router,
        alarm=alarm,
        threshold=threshold,
        generic=generic,
    )
    try:
        _, raw = arch_row_features(instance, library, observation_seed)
    except DesignFailureError:
        raise
    except Exception as error:
        raise DesignFailureError(
            f"seed {seed}: retained arch raw profile construction failed "
            f"under condition {condition}: {type(error).__name__}: {error}",
            seed=seed,
        ) from error
    if raw.shape != (NUM_VIEW_CANDIDATES, len(PROFILE_GRID)):
        raise DesignFailureError(
            f"seed {seed}: retained raw profile block has shape "
            f"{raw.shape}, expected {(NUM_VIEW_CANDIDATES, len(PROFILE_GRID))}",
            seed=seed,
        )
    if not np.isfinite(raw).all():
        raise DesignFailureError(
            f"seed {seed}: non-finite retained raw profiles under "
            f"condition {condition}",
            seed=seed,
        )
    return {
        "seed": seed,
        "condition": condition,
        "arm": arm,
        "correct": bool(outcome.correct),
        "action": outcome.action,
        "routed_index": outcome.routed_index,
        "discovery_invocations": int(outcome.discovery_invocations),
        "admitted": bool(outcome.admitted),
        "map_misfit": outcome.map_misfit,
        "detail": outcome.detail,
        "initial_library_size": int(outcome.initial_library_size),
        "final_library_size": int(outcome.final_library_size),
        "observation_seed": int(observation_seed),
        "arch_raw_profiles": [
            [float(value) for value in candidate] for candidate in raw
        ],
        "wall_time_seconds": float(wall_time_seconds),
    }


def _bit_identity_fields(row: dict[str, Any]) -> dict[str, Any]:
    """The fields the frozen double execution requires bit-identical:
    everything except ``wall_time_seconds`` (execution metadata; the first
    execution's wall time is retained)."""
    return {
        key: value for key, value in row.items() if key != "wall_time_seconds"
    }


def _campaign_rows(
    eligible: list[tuple[int, Any]],
    raw_rows: list[dict[str, Any]],
    *,
    router: Any,
    alarm: Any,
    threshold: float,
    generic: Any,
) -> None:
    """One row per (seed, condition, arm), in seed/condition/arm order.

    The pipeline is a deterministic function of the frozen seeds: every row
    is built TWICE and the two executions must be bit-identical (every
    field except ``wall_time_seconds``), else the whole run is a design
    failure.
    Rows are appended to the caller's list as soon as each row's
    bit-identity check passes — BEFORE the seed's later conditions and arms
    are built — so a mid-campaign failure preserves every completed row
    (the per-seed view downstream covers only seeds complete in all twelve
    rows).
    """
    for seed, instance in eligible:
        try:
            inputs = _condition_inputs(seed, instance)
            observation_seed = _observation_seed(seed)
            for condition in CONDITIONS:
                library, observations = inputs[condition]
                for arm in ARMS:
                    row = _build_arm_row(
                        seed,
                        instance,
                        library,
                        observations,
                        condition=condition,
                        arm=arm,
                        observation_seed=observation_seed,
                        router=router,
                        alarm=alarm,
                        threshold=threshold,
                        generic=generic,
                    )
                    again = _build_arm_row(
                        seed,
                        instance,
                        library,
                        observations,
                        condition=condition,
                        arm=arm,
                        observation_seed=observation_seed,
                        router=router,
                        alarm=alarm,
                        threshold=threshold,
                        generic=generic,
                    )
                    if _bit_identity_fields(row) != _bit_identity_fields(again):
                        raise DesignFailureError(
                            f"seed {seed}: {condition}/{arm} row is not "
                            "bit-identical across the frozen double "
                            "execution",
                            seed=seed,
                        )
                    raw_rows.append(row)
        except BaseException as error:
            if getattr(error, "seed", None) is None:
                error.seed = seed  # type: ignore[attr-defined]
            raise


# ---------------------------------------------------------------------------
# Inference: per-seed statistics and the four frozen claims.


def _estimand(definition: dict[str, Any]) -> str:
    statistic = definition["statistic"]
    return (
        f"mean over eligible seeds of {statistic}(seed) with "
        f"{statistic}(seed) = {_DIFFERENCE_PROVENANCE[definition['difference']]}"
    )


def _claim_summary(
    definition: dict[str, Any], values: list[float]
) -> dict[str, Any]:
    """One-sided Student-t lower bound for one claim's per-seed differences.

    Applied mechanically to the frozen constants, including when SE is
    exactly zero: the lower bound then equals the estimate and support
    reduces to the estimate strictly exceeding the threshold.
    """
    n = len(values)
    if n not in _T_ONE_SIDED_BONFERRONI:
        raise DesignFailureError(
            f"no frozen Bonferroni critical value for n={n}; the design "
            "pins n = 30..36"
        )
    critical = _T_ONE_SIDED_BONFERRONI[n]
    estimate = statistics.fmean(values)
    sd = statistics.stdev(values)  # sample SD, ddof=1
    se = sd / math.sqrt(n)
    lower_bound = estimate - critical * se
    threshold = definition["threshold"]
    return {
        "id": definition["id"],
        "role": definition["role"],
        "description": definition["description"],
        "statistic": definition["statistic"],
        "arms": list(definition["arms"]),
        "scope": definition["scope"],
        "estimand": _estimand(definition),
        "direction": "greater",
        "threshold": threshold,
        "n": n,
        "estimate": estimate,
        "standard_deviation": sd,
        "standard_error": se,
        "critical_value": critical,
        "lower_bound": lower_bound,
        "supported": bool(lower_bound > threshold),
    }


def _claims_family(claims: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "family_size": NUM_CLAIMS,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "per_claim_alpha": PER_CLAIM_ALPHA,
        "method": (
            "one-sided Student-t lower bounds with Bonferroni correction; "
            "the generator seed is the inference unit"
        ),
        "critical_values_n30_to_n36": {
            str(n): value for n, value in sorted(_T_ONE_SIDED_BONFERRONI.items())
        },
        "claims": claims,
    }


def _recompute_per_seed(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-seed condition indicators, condition-means, and the four paired
    differences, joined from the raw rows alone.

    Every statistic is read off the recorded correctness bits: each arm's
    e2e is the mean of its three condition bits, the in-library statistics
    read the in-library rows, and the four differences subtract the
    comparison arm from the full architecture — there is no conditioning on
    loop success and no deletion of seeds. A seed must contribute exactly
    one row per (condition, arm); duplicates and gaps are design failures.
    """
    expected = {(condition, arm) for condition in CONDITIONS for arm in ARMS}
    groups: dict[int, dict[tuple[str, str], dict[str, Any]]] = {}
    order: list[int] = []
    for row in raw_rows:
        seed = row["seed"]
        key = (row["condition"], row["arm"])
        if key not in expected:
            raise DesignFailureError(
                f"seed {seed}: unknown (condition, arm) {key} in a raw row"
            )
        group = groups.setdefault(seed, {})
        if seed not in order:
            order.append(seed)
        if key in group:
            raise DesignFailureError(
                f"seed {seed}: duplicate raw row for {key}"
            )
        group[key] = row
    per_seed = []
    for seed in order:
        group = groups[seed]
        missing = sorted(expected - set(group))
        if missing:
            raise DesignFailureError(
                f"seed {seed}: missing raw rows for {missing}"
            )

        def bit(condition: str, arm: str) -> float:
            return 1.0 if group[(condition, arm)]["correct"] else 0.0

        e2e = {
            arm: statistics.fmean(
                bit(condition, arm) for condition in CONDITIONS
            )
            for arm in ARMS
        }
        per_seed.append(
            {
                "seed": seed,
                "arch_full_e2e": e2e["arch_full"],
                "routing_only_e2e": e2e["routing_only"],
                "discovery_only_e2e": e2e["discovery_only"],
                "generic_e2e": e2e["generic"],
                "arch_full_in_library": bit("in_library", "arch_full"),
                "routing_only_in_library": bit("in_library", "routing_only"),
                "discovery_only_in_library": bit(
                    "in_library", "discovery_only"
                ),
                "generic_in_library": bit("in_library", "generic"),
                "d_arch_generic_e2e": e2e["arch_full"] - e2e["generic"],
                "d_arch_routing_only_e2e": (
                    e2e["arch_full"] - e2e["routing_only"]
                ),
                "d_arch_generic_inlibrary": (
                    bit("in_library", "arch_full")
                    - bit("in_library", "generic")
                ),
                "d_arch_discovery_only_inlibrary_harm": (
                    bit("in_library", "arch_full")
                    - bit("in_library", "discovery_only")
                ),
            }
        )
    return per_seed


def _claim_inference(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    claims = [
        _claim_summary(
            definition, [row[definition["difference"]] for row in per_seed]
        )
        for definition in _CLAIM_DEFINITIONS
    ]
    return _claims_family(claims)


def _descriptive_block(
    raw_rows: list[dict[str, Any]],
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The frozen descriptive block: reported, NEVER claim-tested.

    Per-(condition, arm) accuracies (means of the correctness bits over
    eligible seeds), per-(condition, arm) discovery-invocation counts, and
    wall-time summaries per (condition, arm) and per arm — the protocol's
    descriptive quantities, read by no decision rule. The wall-time
    summaries are execution metadata: they are reproducible from the raw
    rows but are not bit-identical across executions of the runner.

    The block also carries the frozen three-point frontier baseline: the
    published loop-v2 and loop-v3 operating points against which this
    experiment's cost-aware point is read (protocol §1, §11). Those are
    quoted constants from the sealed records, never recomputed here and
    never claim inputs.
    """
    accuracy: dict[str, float | None] = {}
    invocations: dict[str, int] = {}
    wall_mean: dict[str, float | None] = {}
    wall_per_arm: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for condition in CONDITIONS:
        for arm in ARMS:
            rows = [
                row
                for row in raw_rows
                if row["condition"] == condition and row["arm"] == arm
            ]
            key = f"{condition}/{arm}"
            accuracy[key] = (
                statistics.fmean(1.0 if row["correct"] else 0.0 for row in rows)
                if rows
                else None
            )
            invocations[key] = sum(row["discovery_invocations"] for row in rows)
            walls = [row["wall_time_seconds"] for row in rows]
            wall_mean[key] = statistics.fmean(walls) if walls else None
            wall_per_arm[arm].extend(walls)
    return {
        "note": (
            "descriptive only, read by no decision rule: the accuracies "
            "are the per-(condition, arm) means of the correctness bits "
            "over the eligible seeds; the discovery-invocation counts are "
            "per (condition, arm) totals over the eligible seeds; the "
            "wall-time summaries are execution metadata (reproducible "
            "from the raw rows, not bit-identical across executions)"
        ),
        "accuracy_per_condition_arm": accuracy,
        "discovery_invocations_per_condition_arm": invocations,
        "wall_time_seconds": {
            "mean_per_condition_arm": wall_mean,
            "per_arm": {
                arm: {
                    "mean": (
                        statistics.fmean(times) if times else None
                    ),
                    "min": min(times) if times else None,
                    "max": max(times) if times else None,
                }
                for arm, times in wall_per_arm.items()
            },
        },
        "frontier_baseline": {
            "note": (
                "the two documented operating points of the alarm's "
                "Pareto frontier, quoted verbatim from the published "
                "records: loop-v2 (docs/24, the v1 alarm at the frozen "
                "0.5 threshold) and loop-v3 (docs/27, the v2 alarm at a "
                "threshold bounded to false_quiet_rate <= 0.02). This "
                "experiment is the third point, at symmetric cost-aware "
                "pricing. descriptive only, read by no decision rule."
            ),
            "loop_v2": {
                "source": "docs/24-router-loop-v2-sealed-1-results.md",
                "alarm": "the v1 alarm at the frozen threshold 0.5",
                "accuracy_per_condition": {
                    "in_library": 0.9722222222222222,
                    "out_of_library": 0.8333333333333334,
                    "null_control": 1.0,
                },
                "discovery_invocations": {
                    "arch_full_total": 61,
                    "in_library": 1,
                    "out_of_library": 30,
                    "null_control": 30,
                    "discovery_only_total": 108,
                },
                "bounded_harm_estimate": -1.0 / 36.0,
                "harm_seed_count": 1,
            },
            "loop_v3": {
                "source": "docs/27-router-loop-v3-sealed-1-results.md",
                "alarm": (
                    "LearnedAlarmV2 at the bounded-false-quiet calibrated "
                    "threshold (false-quiet bound 0.02, binding)"
                ),
                "accuracy_per_condition": {
                    "in_library": 0.6111111111111112,
                    "out_of_library": 1.0,
                    "null_control": 1.0,
                },
                "discovery_invocations": {
                    "arch_full_total": 86,
                    "in_library": 14,
                    "out_of_library": 36,
                    "null_control": 36,
                    "discovery_only_total": 108,
                },
                "bounded_harm_estimate": -14.0 / 36.0,
                "harm_seed_count": 14,
                "calibration": {
                    "threshold": 0.889712393283844,
                    "balanced_accuracy": 0.7849999999999999,
                    "false_quiet_rate": 0.02,
                    "false_alarm_rate": 0.41,
                    "num_candidates": 802,
                    "bound_satisfied": True,
                },
            },
        },
        "calibration": calibration,
    }


def _null_claims() -> dict[str, Any]:
    """All four frozen claims with a null decision after a failed run."""
    claims = []
    for definition in _CLAIM_DEFINITIONS:
        claims.append(
            {
                "id": definition["id"],
                "role": definition["role"],
                "description": definition["description"],
                "statistic": definition["statistic"],
                "arms": list(definition["arms"]),
                "scope": definition["scope"],
                "estimand": _estimand(definition),
                "direction": "greater",
                "threshold": definition["threshold"],
                "n": None,
                "estimate": None,
                "standard_deviation": None,
                "standard_error": None,
                "critical_value": None,
                "lower_bound": None,
                "supported": None,
            }
        )
    family = _claims_family(claims)
    family["critical_values_n30_to_n36"] = None
    family["role"] = "no confirmatory decision: the run did not complete"
    return family


# ---------------------------------------------------------------------------
# Audit and design records.


def _recomputed_correct(
    arm: str,
    condition: str,
    *,
    action: str,
    routed_index: int | None,
    admitted: bool,
    map_misfit: float | None,
) -> bool:
    """The frozen condition-correctness semantics (the
    :mod:`universa.loop_v2` module docstring), recomputed from a row's
    retained scalar record alone — never the withheld truth, never the
    learned models.
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


def _audit_row_decisions(row: dict[str, Any]) -> dict[str, Any]:
    """Recompute one row's decisions from its retained scalar record alone.

    The audit never re-runs an arm, never rebuilds an instance, and never
    touches the trained models: the condition/arm naming, the observation
    schedule (the retained observation seed against the frozen subseed
    schedule), the retention block's shape and finiteness (K x 8 raw
    profiles), the action/library coherence (route leaves the library
    unchanged, invokes no discovery, and admits nothing; discover grows the
    library by one and routes to the appended index with the
    router-acceptance misfit retained; a refusal routes nowhere and admits
    nothing), the arm's discovery-invocation discipline (routing_only and
    generic never invoke discovery; discovery_only always does; arch_full
    does exactly when it does not route), and the frozen correctness
    semantics are all recomputed exactly. Any disagreement is a whole-run
    design failure.
    """
    seed = row["seed"]
    condition = row["condition"]
    arm = row["arm"]
    if condition not in CONDITIONS:
        raise DesignFailureError(
            f"seed {seed}: unknown condition {condition!r} in a raw row",
            seed=seed,
        )
    if arm not in ARMS:
        raise DesignFailureError(
            f"seed {seed}: unknown arm {arm!r} in a raw row",
            seed=seed,
        )
    if row["observation_seed"] != _observation_seed(seed):
        raise DesignFailureError(
            f"seed {seed}: retained observation_seed "
            f"{row['observation_seed']} disagrees with the frozen schedule",
            seed=seed,
        )
    profiles = row["arch_raw_profiles"]
    if (
        not isinstance(profiles, list)
        or len(profiles) != NUM_VIEW_CANDIDATES
        or any(
            not isinstance(profile, list)
            or len(profile) != len(PROFILE_GRID)
            or any(
                not isinstance(value, float)
                or not math.isfinite(value)
                or value < 0.0
                for value in profile
            )
            for profile in profiles
        )
    ):
        raise DesignFailureError(
            f"seed {seed}: retained arch raw profiles must be "
            f"{NUM_VIEW_CANDIDATES} x {len(PROFILE_GRID)} finite "
            "nonnegative floats",
            seed=seed,
        )
    action = row["action"]
    initial = row["initial_library_size"]
    final = row["final_library_size"]
    if initial != NUM_VIEW_CANDIDATES:
        raise DesignFailureError(
            f"seed {seed}: retained initial_library_size {initial} is not "
            f"the frozen K = {NUM_VIEW_CANDIDATES}",
            seed=seed,
        )
    invocations = row["discovery_invocations"]
    if arm in ("routing_only", "generic") and invocations != 0:
        raise DesignFailureError(
            f"seed {seed}: arm {arm} never invokes discovery, retained "
            f"{invocations}",
            seed=seed,
        )
    if arm == "discovery_only" and invocations != 1:
        raise DesignFailureError(
            f"seed {seed}: discovery_only always invokes discovery once, "
            f"retained {invocations}",
            seed=seed,
        )
    if action == "route":
        if (
            row["admitted"]
            or final != initial
            or invocations != 0
            or row["map_misfit"] is not None
        ):
            raise DesignFailureError(
                f"seed {seed}: route mode must admit nothing, invoke no "
                "discovery, carry no map misfit, and leave the library "
                "unchanged",
                seed=seed,
            )
        if not isinstance(row["routed_index"], int) or not (
            0 <= row["routed_index"] < final
        ):
            raise DesignFailureError(
                f"seed {seed}: route mode needs a routed index into the "
                "library",
                seed=seed,
            )
    elif action == "discover":
        if (
            not row["admitted"]
            or row["routed_index"] != initial
            or final != initial + 1
            or invocations != 1
            or row["map_misfit"] is None
        ):
            raise DesignFailureError(
                f"seed {seed}: discover mode must admit, grow the library "
                "by one, route to the appended index, invoke discovery "
                "once, and carry the router-acceptance misfit",
                seed=seed,
            )
    elif action == "refused":
        if (
            row["routed_index"] is not None
            or row["admitted"]
            or final != initial
        ):
            raise DesignFailureError(
                f"seed {seed}: a refused arm routes nowhere, admits "
                "nothing, and leaves the library unchanged",
                seed=seed,
            )
        if arm in ("arch_full", "discovery_only") and invocations != 1:
            raise DesignFailureError(
                f"seed {seed}: a refused {arm} invoked discovery once, "
                f"retained {invocations}",
                seed=seed,
            )
    else:
        raise DesignFailureError(
            f"seed {seed}: unknown action {action!r} in a raw row",
            seed=seed,
        )
    correct = _recomputed_correct(
        arm,
        condition,
        action=action,
        routed_index=row["routed_index"],
        admitted=row["admitted"],
        map_misfit=row["map_misfit"],
    )
    if correct != row["correct"]:
        raise DesignFailureError(
            f"seed {seed}: retained correctness bit disagrees with the "
            "recomputed frozen-semantics decision",
            seed=seed,
        )
    return {
        "seed": seed,
        "condition": condition,
        "arm": arm,
        "action": action,
        "correct": correct,
    }


def _audit_block(
    eligibility: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    *,
    complete: bool,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Eligibility counts plus raw-row-recomputable decisions and claims."""
    block: dict[str, Any] = {
        "recompute_scope": (
            "(a) every claim statistic and every correctness decision "
            "below is recomputed from the raw rows alone: the "
            "condition/arm naming, the observation schedule (the retained "
            "observation_seed against subseed(seed, 'router-v2-observe')), "
            "the action/library coherence (route leaves the "
            "library unchanged and admits nothing; discover grows it by "
            "one and routes to the appended index; a refusal routes "
            "nowhere), the discovery-invocation discipline per arm, and "
            "the frozen condition-correctness semantics — the routing "
            "arms' decisions recompute from the retained "
            "action/routed_index, and the certified-gate semantics from "
            "the retained admitted/map_misfit against 1e-9 — followed by "
            "the per-seed condition indicators, condition-means, paired "
            "differences, means, standard deviations, SEs, one-sided "
            "lower bounds, decisions, and the descriptive recomputes "
            "(per-(condition, arm) accuracies and discovery-invocation "
            "counts; the wall-time summaries are execution metadata, "
            "recomputed from the rows but not bit-identical across "
            "executions); the eligibility counts and seed "
            "ids come from the eligibility pass, not from the raw rows. "
            "(b) what is NOT recomputable from the rows: the learned "
            "models' internal decisions — the router's argmax and soft "
            "gates, the calibrated v2 alarm's fit/no-fit, and the "
            "generic model's class "
            "— because the trained weights are not retained; the three "
            "models are hash-pinned in training (model_state_sha256), so "
            "re-running the frozen training on the frozen train block "
            "reproduces them bit-exactly. The retained per-candidate arch "
            "raw profiles (8 floats each) are the row's shared "
            "degradation-profile observation record (the alarm's "
            "profile-summary inputs are functions of them); no matrices "
            "are retained. This audit does NOT re-verify the discovery "
            "head's internal SVD certificates: the per-row certified "
            "residuals (map_misfit) are reproducible only by re-running "
            "the manifest-pinned deterministic pipeline, and that "
            "assurance is pinned by the code manifest and enforced by the "
            "per-row double-execution bit-identity check."
        ),
        "declared_seeds": eligibility["declared"],
        "eligible_seeds": eligibility["eligible"],
        "eligible_seed_ids": list(eligibility["eligible_seeds"]),
        "ineligible_seed_rows": len(eligibility["ineligible"]),
        "build_failure_rows": len(eligibility["build_failures"]),
        "raw_rows": len(raw_rows),
        "rows_per_seed": len(ARMS) * len(CONDITIONS),
        "arms": list(ARMS),
        "conditions": list(CONDITIONS),
    }
    if complete:
        block["decision_recompute"] = [
            _audit_row_decisions(row) for row in raw_rows
        ]
        per_seed = _recompute_per_seed(raw_rows)
        block["per_seed"] = per_seed
        claims: dict[str, Any] = {}
        for definition in _CLAIM_DEFINITIONS:
            values = [row[definition["difference"]] for row in per_seed]
            n = len(values)
            if n not in _T_ONE_SIDED_BONFERRONI:
                raise DesignFailureError(
                    f"no frozen Bonferroni critical value for n={n}; the "
                    "design pins n = 30..36"
                )
            estimate = statistics.fmean(values)
            standard_deviation = statistics.stdev(values)
            standard_error = standard_deviation / math.sqrt(n)
            lower_bound = (
                estimate - _T_ONE_SIDED_BONFERRONI[n] * standard_error
            )
            claims[definition["id"]] = {
                "estimate": estimate,
                "standard_deviation": standard_deviation,
                "standard_error": standard_error,
                "lower_bound": lower_bound,
                "supported": bool(lower_bound > definition["threshold"]),
            }
        block["claims"] = claims
        block["descriptive"] = _descriptive_block(
            raw_rows, calibration=calibration
        )
    return block


def _design_record() -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "canonical_command": CANONICAL_COMMAND,
        "task_family": (
            "graph-quotient switch instances via "
            "universa.budgets.make_budget_instance(num_vertices=8, "
            "num_edges=14, num_classes=6, num_decoys=3); ONE budget "
            "instance per seed with the equal-K paired views of "
            "universa.loop_v2: the in-library view (instance.true_target, "
            "*instance.decoy_targets[:-1]) with the true target at index "
            "0, and the out-of-library view instance.decoy_targets with "
            "the truth withheld — both of size K = 3, the fixed input "
            "width of the learned alarm and the generic head"
        ),
        "observation_regime": (
            "the router-v2 no-anchor regime lifted into the loop: every "
            "candidate boundary is observed through "
            "universa.partial.ObservationModel draws with mask_fraction = "
            "0.25 AND sign corruption swept over the profile grid "
            "0.2..0.9 by 0.1, which EXCLUDES 0.0 — no clean column exists "
            "anywhere, so no residual the loop reads is exact and the "
            "fit/no-fit alarm must be learned; the transported vector "
            "observations y = f1 a are exact under every regime, so the "
            "certified discovery head runs unchanged at the frozen gates "
            "(certification 1e-10, novelty 1e-6, router acceptance "
            "map_misfit <= 1e-9)"
        ),
        "observation_schedules": {
            "observation_seed": (
                "subseed(seed, 'router-v2-observe'): ONE shared "
                "ObservationModel draw per seed, feeding every candidate "
                "at every grid point of every arm of every condition of "
                "the row (exact pairing), for train and eval rows alike — "
                "the router-v2 regime's canonical observation family, so "
                "the learned models' training rows and the loop rows come "
                "from one observation distribution on disjoint seed "
                "blocks; the arch and routing arms' feature blocks are "
                "therefore identical in the out-of-library and "
                "null-control conditions per seed (the alarm's decision "
                "basis precedes observations)"
            ),
            "structured_observations": (
                "universa.discovery.synthesize_observations of a "
                "SwitchInstance view of the budget instance, M = 16 "
                "columns — the frozen exp4 'discovery-observation' "
                "schedule keyed on the instance seed"
            ),
            "null_observations": (
                "universa.loop.null_observations(seed, ambient_dim, 16) — "
                "column j drawn from np.random.default_rng(subseed(seed, "
                "'discovery-null', str(j))), the frozen exp4 H4 schedule"
            ),
            "operating_grid_point": (
                "universa.loop_v2.operating_grid_point(seed) — "
                "grid[subseed(seed, 'loop-v2-operating') % 8], the generic "
                "arm's single degradation level per row"
            ),
        },
        "arms": {
            "arch_full": (
                "the full system: the trained StructureRouter scores the "
                "no-anchor profile blocks (hard argmax), the trained "
                "LearnedAlarmV2 decides fit/no-fit from the soft gates "
                "and the raw profile block against the train-block "
                "calibrated threshold; fit routes to the argmax, no-fit "
                "runs certified discovery on the exact observations, "
                "gated (1e-10 certification, 1e-6 novelty); on admission "
                "the row routes to the appended structure, else refused"
            ),
            "routing_only": (
                "ablation with no alarm and no discovery: always routes "
                "to the trained router's argmax (a forced choice; "
                "out-of-library it must pick a decoy — an honest failure)"
            ),
            "discovery_only": (
                "ablation that always runs certified discovery on the "
                "exact observations, novelty checked against the "
                "instance's decoy library; the library is unused for "
                "routing"
            ),
            "generic": (
                "the no-architecture arm: the trained GenericMLP over "
                "generic spectral features (NO commutation residual, NO "
                "degradation profile); classes 0..K-1 route, class K is "
                "no-fit and synthesizes nothing"
            ),
        },
        "conditions": {
            "in_library": (
                "one row per arm over the in-library view with the exact "
                "transported observations"
            ),
            "out_of_library": (
                "one row per arm over the out-of-library view (the truth "
                "withheld) with the same exact transported observations"
            ),
            "null_control": (
                "one row per arm over the out-of-library view with the "
                "structure-free 'discovery-null' observations — the "
                "false-admission / refusal specificity control"
            ),
        },
        "correctness_semantics": (
            "the frozen universa.loop_v2 semantics: in-library, the arm's "
            "final structure is the true target (route to index 0, or a "
            "certified admitted discovery with map_misfit <= 1e-9); "
            "out-of-library, a certified novel structure admitted with "
            "map_misfit <= 1e-9 (routing_only and generic can never "
            "acquire); null-control, the alarm arms must admit nothing "
            "and the refusal-only arms must refuse"
        ),
        "train_seed_block": {
            "first": TRAIN_SEEDS[0],
            "last": TRAIN_SEEDS[-1],
        },
        "training": {
            "router": (
                "StructureRouter(feature_dim=18, hidden_dim=64) via "
                "universa.router.train_router, torch seed 4242, 150 "
                "epochs, lr 1e-3, lambda_aux 0.01, tau 2.0 -> 0.25, "
                "full-batch Adam, CPU float32; the in-library arch rows, "
                "label 0 for every row — the loop's unpermuted index-0 "
                "convention (the declared consequences: the router's "
                "learned role is profile-shape scoring under the index-0 "
                "commitment; the loop's learned selectivity lives in the "
                "alarm; the out-of-library view contributes no router "
                "rows)"
            ),
            "alarm": (
                "LearnedAlarmV2(K=3, hidden_dim=32) via "
                "universa.loop_v2.train_alarm_v2, torch seed 4243, 150 "
                "epochs, lr 1e-3; fit/no-fit labels from the in-library "
                "vs out-of-library rows, reading the trained router's "
                "soft gates (tau = 1.0) over the alarm_features_v2 "
                "layout (K + 8 dims: the frozen v1 columns plus the "
                "five margin features)"
            ),
            "calibration": (
                "the alarm's decision threshold is calibrated on the "
                "train block by "
                "universa.loop_v2.calibrate_threshold_cost_aware at the "
                "frozen unit costs false_quiet_cost = 1.0 and "
                "false_alarm_cost = 1.0 (the threshold MINIMIZING the "
                "weighted total error, ties toward the larger balanced "
                "accuracy then the larger threshold, no feasibility "
                "bound); the full eight-key record (threshold, "
                "balanced_accuracy, false_quiet_rate, false_alarm_rate, "
                "total_cost, false_quiet_cost, false_alarm_cost, "
                "num_candidates) is retained in the training provenance. "
                "This replaces loop-v3's bounded rule; at equal unit "
                "costs the objective equals 2 - 2 * balanced_accuracy "
                "exactly, so the rule is unconstrained balanced-accuracy "
                "maximization — declared in the protocol, not discovered"
            ),
            "generic": (
                "GenericMLP(K=3, feature_dim=18, hidden_dim=64) via "
                "universa.loop_v2.train_generic, torch seed 4244, 150 "
                "epochs, lr 1e-3; index/no-fit labels (in-library labeled "
                "0, out-of-library labeled K)"
            ),
            "rows_per_train_seed": (
                "both views: arch profile features AND generic spectral "
                "features per candidate"
            ),
            "train_build_exclusion": (
                "protocol errata 1: a train seed whose INSTANCE fails to "
                "build is recorded and EXCLUDED (it is not an instance of "
                "the family — the budgets discriminability guard refuses a "
                "decoy sharing the true target's kernel), mirroring the "
                "eval side's ineligibility record; the excluded seeds and "
                "the built count are reported in the training provenance. "
                "A feature-construction failure, a shape violation, or a "
                "non-finite feature remains a whole-run design_failure. "
                "More than 20 non-instances among the 400 declared train "
                "seeds is itself a whole-run design_failure"
            ),
        },
        "eval_seeds": {
            "first": SEALED_EVAL_SEEDS[0],
            "last": SEALED_EVAL_SEEDS[-1],
        },
        "determinism": (
            "every eval (seed, condition, arm) row is executed twice and "
            "required bit-identical in every field except "
            "wall_time_seconds (execution metadata; the first execution's "
            "wall time is retained); any mismatch is a whole-run "
            "design_failure; every train row and every model hash is "
            "reproducible from the frozen seeds"
        ),
        "inference_unit": "the generator seed",
        "estimand": (
            "per-seed condition indicators (the correctness bits per arm "
            "per condition), per-arm condition-means (e2e: the mean of an "
            "arm's bits over the three paired conditions), and the four "
            "paired per-seed differences against the full architecture: "
            "arch_vs_generic_e2e, arch_vs_routing_only_e2e, "
            "arch_vs_generic_inlibrary, and "
            "arch_vs_discovery_only_inlibrary_harm; each claim is the "
            "mean of its per-seed difference over the eligible seeds"
        ),
        "familywise_alpha": FAMILYWISE_ALPHA,
        "per_claim_alpha": PER_CLAIM_ALPHA,
        "minimum_eligible": MIN_ELIGIBLE,
        "descriptive": (
            "reported, never claim-tested: per-(condition, arm) "
            "accuracies (means of the correctness bits over eligible "
            "seeds), per-(condition, arm) discovery-invocation counts, "
            "wall-time summaries per (condition, arm) and per arm, the "
            "calibration record, and the three-point frontier baseline "
            "against the published docs/24 and docs/27 records (loop-v2's "
            "arch arm at 0.972/0.833/1.000 with 61/108 discovery "
            "invocations and a -1/36 bounded-harm estimate; loop-v3's at "
            "0.611/1.000/1.000 with 86/108 invocations and a -14/36 "
            "estimate, plus its calibration record) — the price of "
            "selectivity at two sealed operating points, read by no "
            "decision rule"
        ),
        "eligibility_rule": (
            "the instance builds AND the undegraded audit passes: "
            "universa.budgets.make_budget_instance(seed, 8, 14, 6, 3) "
            "returns (a build exception is a whole-run design_failure, "
            "never an exclusion), the true candidate's undegraded misfit "
            "is exactly 0.0, and every decoy's undegraded misfit is "
            "strictly above 1e-9; an audit failure makes that one seed "
            "ineligible with a recorded reason (never a stop)"
        ),
        "row_count_invariant": (
            "4 x 3 x n: exactly one row per (condition, arm) — 4 arms x 3 "
            "conditions = 12 rows per eligible seed (432 rows over the "
            "declared block when every declared seed is eligible), "
            "checked before any summary"
        ),
        "no_outcome_dependent_stopping": True,
        "no_seed_deletion": True,
        "declared_caveat": DECLARED_CAVEAT,
    }


# ---------------------------------------------------------------------------
# The sealed campaign.


def run(project_root: Path, output: Path, *, seal: str = DEFAULT_SEAL) -> dict[str, Any]:
    """Run the complete sealed design after fail-closed preflight checks.

    The terminal status is one of ``complete``, ``design_failure``,
    ``design_failure_insufficient_eligible``, ``execution_failure``, or
    ``interrupted``. Every failure path preserves all completed raw rows and
    emits all four claims with ``supported: null``.
    """

    provenance = _preflight(project_root, output, seal=seal)

    # Eligibility pass over ALL sealed eval seeds, before any training: an
    # instance build exception is a whole-run design failure (never an
    # exclusion); an undegraded-instance audit failure makes the seed
    # ineligible with a recorded reason. No outcome-dependent stopping, no
    # seed deletion. Any other failure of the pass itself is classified into
    # the frozen status vocabulary below, preserving every completed seed
    # record.
    seed_records: list[dict[str, Any]] = []
    eligible: list[tuple[int, Any]] = []
    build_failure: dict[str, Any] | None = None
    eligibility_failure: tuple[str, BaseException] | None = None
    try:
        for seed in SEALED_EVAL_SEEDS:
            try:
                instance = make_budget_instance(
                    seed, NUM_VERTICES, NUM_EDGES, NUM_CLASSES, NUM_DECOYS
                )
            except Exception as error:  # a build exception stops the whole run
                build_failure = {
                    "seed": seed,
                    "type": type(error).__name__,
                    "message": str(error),
                }
                break
            misfits = _undegraded_misfits(instance)
            reason = _audit_undegraded(seed, misfits)
            seed_records.append(
                {
                    "seed": seed,
                    "eligible": reason is None,
                    "reason": reason,
                    "true_misfit_undegraded": misfits[0],
                    "decoy_misfits_undegraded": list(misfits[1:]),
                }
            )
            if reason is None:
                eligible.append((seed, instance))
    except DesignFailureError as error:
        eligibility_failure = ("design_failure", error)
    except KeyboardInterrupt as error:
        eligibility_failure = ("interrupted", error)
    except Exception as error:  # unexpected faults are preserved
        eligibility_failure = ("execution_failure", error)

    eligibility = {
        "declared": len(SEALED_EVAL_SEEDS),
        "attempted": len(seed_records) + (1 if build_failure else 0),
        "eligible": len(eligible),
        "eligible_seeds": [seed for seed, _ in eligible],
        "ineligible": [
            {"seed": record["seed"], "reason": record["reason"]}
            for record in seed_records
            if not record["eligible"]
        ],
        "build_failures": [build_failure] if build_failure else [],
    }
    base: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "provenance": provenance,
        "design": _design_record(),
        "eligibility": eligibility,
        "seed_records": seed_records,
    }
    if eligibility_failure is not None:
        status, error = eligibility_failure
        base["status"] = status
        base["failure"] = {
            "seed": getattr(error, "seed", None),
            "phase": "eligibility",
            "type": type(error).__name__,
            "message": str(error) or type(error).__name__,
        }
        base["stop_condition"] = (
            "the eligibility pass itself failed; every completed seed "
            "record is preserved and every claim carries a null decision"
        )
        base["training"] = None
        base["raw_rows"] = []
        base["per_seed"] = []
        base["claims"] = _null_claims()
        base["descriptive"] = None
        base["audit"] = _audit_block(eligibility, [], complete=False)
        return base
    if build_failure is not None:
        base["status"] = "design_failure"
        base["failure"] = {
            "seed": build_failure["seed"],
            "phase": "eligibility",
            "type": build_failure["type"],
            "message": build_failure["message"],
        }
        base["stop_condition"] = (
            "an instance build exception is a whole-run design failure, "
            "never an exclusion; no training ran and no rows were built"
        )
        base["training"] = None
        base["raw_rows"] = []
        base["per_seed"] = []
        base["claims"] = _null_claims()
        base["descriptive"] = None
        base["audit"] = _audit_block(eligibility, [], complete=False)
        return base
    if len(eligible) < MIN_ELIGIBLE:
        base["status"] = "design_failure_insufficient_eligible"
        base["stop_condition"] = (
            f"only {len(eligible)} seeds were eligible; minimum is "
            f"{MIN_ELIGIBLE}; no training ran and no claims were decided"
        )
        base["training"] = None
        base["raw_rows"] = []
        base["per_seed"] = []
        base["claims"] = _null_claims()
        base["descriptive"] = None
        base["audit"] = _audit_block(eligibility, [], complete=False)
        return base

    raw_rows: list[dict[str, Any]] = []
    training: dict[str, Any] | None = None
    failure: tuple[str, BaseException] | None = None
    try:
        train_block = _build_train_block()
        router, alarm, generic, calibration, training = _train_models(train_block)
        _campaign_rows(
            eligible,
            raw_rows,
            router=router,
            alarm=alarm,
            threshold=calibration["threshold"],
            generic=generic,
        )
        # Fail-closed row-count invariant (4 x 3 x n): every eligible seed
        # must have contributed exactly one row per (condition, arm) before
        # any summary.
        expected_rows = len(ARMS) * len(CONDITIONS) * len(eligible)
        if len(raw_rows) != expected_rows:
            raise DesignFailureError(
                f"row-count invariant violated: {len(raw_rows)} raw rows, "
                f"expected {expected_rows} (4 arms x 3 conditions x "
                f"{len(eligible)} eligible seeds — 4 x 3 x n: one row per "
                "(condition, arm) per eligible seed; 432 rows over the "
                "declared block when every declared seed is eligible)"
            )
        # Summaries are computed only after every required raw row has
        # completed (frozen stop rule: no interim analysis).
        per_seed = _recompute_per_seed(raw_rows)
        base["claims"] = _claim_inference(per_seed)
    except DesignFailureError as error:
        failure = ("design_failure", error)
    except KeyboardInterrupt as error:
        failure = ("interrupted", error)
    except Exception as error:  # unexpected faults are preserved
        failure = ("execution_failure", error)

    base["training"] = training
    base["raw_rows"] = raw_rows
    if failure is not None:
        # A mid-seed stop can leave rows of a seed whose later conditions
        # or arms were never built; the per-seed view then covers only the
        # seeds complete in all twelve rows (every completed raw row is
        # preserved regardless).
        rows_per_seed = len(ARMS) * len(CONDITIONS)
        counts: dict[int, int] = {}
        for row in raw_rows:
            counts[row["seed"]] = counts.get(row["seed"], 0) + 1
        complete_seeds = {
            seed for seed, count in counts.items() if count == rows_per_seed
        }
        base["per_seed"] = _recompute_per_seed(
            [row for row in raw_rows if row["seed"] in complete_seeds]
        )
        status, error = failure
        base["status"] = status
        base["failure"] = {
            "seed": getattr(error, "seed", None),
            "phase": "campaign",
            "type": type(error).__name__,
            "message": str(error) or type(error).__name__,
        }
        base["stop_condition"] = (
            "the campaign stopped before completion; all completed raw rows "
            "are preserved and every claim carries a null decision"
        )
        base["claims"] = _null_claims()
        base["descriptive"] = None
        base["audit"] = _audit_block(eligibility, raw_rows, complete=False)
        return base

    # TOCTOU guard, just before a COMPLETE artifact is published: the
    # worktree must still be clean, the on-disk seal must still equal its
    # HEAD blob, and the protocol, runner, and universa code files must be
    # unchanged since preflight. The post-run status is recorded next to
    # the pre-run one in the provenance. A violation here is a classified
    # design_failure with a preserved artifact, never a bare traceback.
    try:
        post_run_status = _git_checked(
            project_root, "status", "--porcelain", "--untracked-files=all"
        )
        if post_run_status:
            raise RuntimeError(
                "stop condition: working tree became dirty during execution"
            )
        seal_relative = provenance["seal"]["path"]
        if _git_head_blob(project_root, seal_relative) != (
            project_root / seal_relative
        ).read_bytes():
            raise RuntimeError(
                "stop condition: the on-disk design seal changed during execution"
            )
        if (
            _sha256(project_root / provenance["protocol"]["path"])
            != provenance["protocol"]["sha256"]
        ):
            raise RuntimeError(
                "stop condition: the sealed protocol changed during execution"
            )
        if _sha256(Path(__file__).resolve()) != provenance["runner"]["sha256"]:
            raise RuntimeError(
                "stop condition: the running file changed during execution"
            )
        if _code_manifest(project_root) != provenance["code_manifest"]["files"]:
            raise RuntimeError(
                "stop condition: src/universa/*.py changed during execution"
            )
    except (RuntimeError, OSError) as error:
        base["per_seed"] = _recompute_per_seed(raw_rows)
        base["status"] = "design_failure"
        base["failure"] = {
            "seed": None,
            "phase": "publication",
            "type": type(error).__name__,
            "message": str(error),
        }
        base["stop_condition"] = (
            "the end-of-run publication guard refused; all completed raw "
            "rows are preserved and every claim carries a null decision"
        )
        base["claims"] = _null_claims()
        base["descriptive"] = None
        base["audit"] = _audit_block(eligibility, raw_rows, complete=False)
        return base
    provenance["git_status_porcelain_post_run"] = post_run_status
    base["per_seed"] = _recompute_per_seed(raw_rows)
    base["descriptive"] = _descriptive_block(raw_rows, calibration=calibration)
    base["audit"] = _audit_block(
        eligibility, raw_rows, complete=True, calibration=calibration
    )
    base["status"] = "complete"
    return base


def main(argv: list[str] | None = None) -> int:
    default_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seal",
        default=DEFAULT_SEAL,
        help=(
            "repository-relative path of the committed design-seal JSON "
            f"(default: {DEFAULT_SEAL})"
        ),
    )
    args = parser.parse_args(argv)
    root = args.project_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    report = run(root, output, seal=args.seal)
    if report["status"] == "complete":
        publish_path = output
    else:
        # Failure artifacts never occupy the canonical output path; the
        # deterministic per-status failure name is likewise no-clobber.
        publish_path = (
            output.parent
            / "failures"
            / f"{output.stem}.{report['status']}{output.suffix}"
        )
    try:
        _atomic_json_new(publish_path, report)
    except RuntimeError as error:
        # The target was occupied or appeared mid-run: preserve the attempt
        # under a distinct secondary path and still exit non-zero.
        publish_path = publish_path.with_name(
            f"{publish_path.name}.{os.getpid()}.failed.json"
        )
        _atomic_json_new(publish_path, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "eligible": report["eligibility"]["eligible"],
                    "output": str(publish_path),
                    "publish_error": str(error),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "status": report["status"],
                "eligible": report["eligibility"]["eligible"],
                "output": str(publish_path),
                "supported_claims": [
                    claim["id"]
                    for claim in report.get("claims", {}).get("claims", [])
                    if claim["supported"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
