#!/usr/bin/env python3
"""Sealed runner for experiment ``universa-router-loop-sealed-1``.

The frozen design: the certified route-or-discover loop of
:mod:`universa.loop` — no learned component, no training, numpy float64
only — is evaluated on the graph-quotient switch family at the frozen
exp4 sizes (``make_loop_instance(seed)`` =
``make_budget_instance(seed, 8, 14, 6, 3)``: ONE budget instance with
two library views, the paired conditions) over the sealed eval seed
block 130001..130036. Each eligible seed contributes THREE raw rows, one
per condition:

* **in-library** — one :func:`universa.loop.run_loop` pass over the
  instance with the in-library view ``instance.candidates`` (the true
  target sits at library index 0 by generator convention; the loop never
  permutes, and there is no learned component that could fit a
  position): the certified commutation scores must keep the misfit
  alarm quiet (the true candidate scores exactly 0.0) and the loop must
  commit to index 0. This condition is the source of the H3 (routing)
  and H4 (alarm silence) per-seed statistics.
* **out-of-library** — one :func:`universa.loop.run_loop` pass over the
  SAME instance with the out-of-library view ``instance.decoy_targets``
  (the truth withheld): the alarm must fire, the certified discovery
  head must certify a constraint from the frozen ``"discovery-
  observation"`` schedule, the novelty gate must admit it, and the
  planted map must commute with it (``map_misfit <= 1e-9``, the
  router-acceptance residual). This condition is the source of the H1
  (acquisition) per-seed statistic.
* **null-control** — the H2 false-admission (specificity) control: the
  same out-of-library pass with the observations override
  ``null_observations(seed, ambient_dim, 16)`` — 16 i.i.d. standard
  Gaussian columns on the seed's target edge space, column ``j`` drawn
  from ``np.random.default_rng(subseed(seed, "discovery-null", str(j)))``
  (the frozen exp4 H4 schedule, a disjoint subseed family from the
  structured condition's ``"discovery-observation"`` draws). The alarm
  fires on the decoy library and the structure-free data must then be
  REFUSED: a constraint certified and admitted from noise is a
  specificity failure.

Because the loop trains nothing, the train seed block is the
documented-empty sentinel ``{"first": 0, "last": 0}`` — seed ``0``
belongs to no Universa seed block and ``first > last`` encodes the empty
set. There is no fit, no train rows, and no train-side audits. The
inference unit is the generator seed for the four one-sided Bonferroni
claims (per-claim alpha 0.05/4 = 0.0125), each decided by a one-sided
Student-t lower bound against its frozen threshold (0.95 / 0.95 / 0.95 /
0.95), including the mechanical ``SE = 0`` case (the lower bound then
equals the estimate). H1's statistic is
:func:`universa.loop.acquisition_correct`; H3's is
:func:`universa.loop.routing_correct` (index 0 by the frozen generator
convention); H2's is the refusal indicator of the null-control row; H4's
is the alarm-silence indicator of the in-library row.

**Determinism, verified.** The pipeline is a deterministic float64
function of the seed: the runner executes the full per-seed pipeline —
all three conditions — TWICE for every eligible seed and requires
bit-identical raw rows; any mismatch is a whole-run ``design_failure``.

**Retention (frozen).** Every row retains the full
:class:`universa.loop.LoopOutcome` scalar record — the mode, the alarm
flag, the best score, the per-candidate certified commutation scores,
the routed index, the discovery verdict and reason, the certificate
residual, the admission flag, the per-entry admission distances and
their minimum, the router-acceptance map misfit, the initial and final
library sizes, and the observation count — plus the condition's decision
indicator. No matrices are retained: every claim statistic and every
alarm/routing/acquisition/refusal decision is a function of this scalar
record, and the audit block recomputes all of them from the raw rows
alone, without re-running the loop or rebuilding any instance (the exact
scope is stated in the audit block). Because nothing is trained, there
is no model to pin.

**No torch.** The loop is numpy-only: this runner computes no torch
tensor anywhere. For uniformity with the other sealed runners the
environment check still requires CUDA hidden, and sets/verifies exactly
one torch thread *when torch is importable in the environment*; torch is
not a dependency of this runner, and when it is absent the CUDA-hidden
requirement is enforced by the ``CUDA_VISIBLE_DEVICES`` override at
import time. See :func:`_execution_environment`.

**Declared caveat (frozen).** Demo-scale numbers are not evidence: the
acquisition, routing, refusal, and alarm-silence rates are properties of
this one synthetic generator family at the frozen exp4 sizes under the
certified route-or-discover loop, not guarantees for other families,
other library sizes, or real data; and the false-admission control is an
i.i.d. Gaussian null matched to the target edge space, not a worst-case
adversary.

This runner must not be executed on the sealed eval seed block until the
protocol, this runner, and the seal record
(``docs/20-router-loop-seal.json``) have been committed and pushed. It
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
from pathlib import Path
from typing import Any

import numpy as np

# The operator-provided CUDA visibility is captured BEFORE the runner hides
# CUDA, so the result records both the operator-provided and the effective
# value.
OPERATOR_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only, before any torch import

# torch is NOT a dependency of this runner: the loop is numpy-only and no
# torch tensor is ever computed here. The guarded import exists solely so
# the environment check can pin one torch thread and verify a hidden CUDA
# for uniformity with the other sealed runners when torch happens to be
# installed; when it is absent, the CUDA-hidden requirement rests on the
# CUDA_VISIBLE_DEVICES override above. See _execution_environment.
try:
    import torch
except ImportError:  # pragma: no cover - torch may be absent
    torch = None  # type: ignore[assignment]

import universa
from universa.loop import (
    ALARM_TOL,
    LibraryLoop,
    LoopOutcome,
    acquisition_correct,
    make_loop_instance,
    null_observations,
    routing_correct,
    run_loop,
)
from universa.structures import ChainMap

EXPERIMENT_ID = "universa-router-loop-sealed-1"
RESULT_SCHEMA = "universa-router-loop-sealed-result/1"

# The one canonical execution command, recorded verbatim in every result.
CANONICAL_COMMAND = (
    "env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src python "
    "scripts/run_loop_sealed_1.py --output "
    "results/experiments/router-loop-sealed-1.json"
)

PROTOCOL = "docs/19-sealed-router-loop-protocol.md"
RUNNER_SOURCE = "scripts/run_loop_sealed_1.py"
DEFAULT_SEAL = "docs/20-router-loop-seal.json"
SEAL_SCHEMA = "universa-seal/7"

PROTOCOL_SHA256 = (
    "e4965933a1b372a266c0768d52ab491b56232db01ad00aee6e11949e99dbf157"
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
# universa.loop.make_loop_instance (make_budget_instance at the exp4 sizes
# 8 vertices, 14 edges, 6 classes, 3 decoys), ONE instance per seed with
# two library views: instance.candidates (the in-library view, true
# target at index 0) and instance.decoy_targets (the out-of-library view,
# the truth withheld).
NUM_OBSERVATIONS = 16  # M observation vectors per seed, frozen exp4 count

LOOP = LibraryLoop()
"""The frozen loop configuration (the exp4 constants).

``alarm_tol`` = :data:`universa.loop.ALARM_TOL` = 1e-9 (re-pinned
``universa.router.RESIDUAL_TOL``), ``misfit_tol`` =
:data:`universa.operators.CERT_TOL` = 1e-10, ``novelty_tol`` =
:data:`universa.discovery.DEFAULT_NOVELTY_TOL` = 1e-6,
``num_observations`` = 16. Constructing it builds no seed instance; the
sealed block stays inert until preflight passes.
"""

# The loop trains nothing: the train block is the documented-empty
# sentinel {"first": 0, "last": 0} — seed 0 belongs to no Universa seed
# block and first > last encodes the empty set.
TRAIN_SEEDS: tuple[int, ...] = ()
TRAIN_SEED_BLOCK = {"first": 0, "last": 0}
SEALED_EVAL_SEEDS = tuple(range(130001, 130037))  # 130001..130036

CONDITIONS = ("in-library", "out-of-library", "null-control")
"""The frozen per-seed conditions, in campaign build order."""

FAMILYWISE_ALPHA = 0.05
NUM_CLAIMS = 4
PER_CLAIM_ALPHA = FAMILYWISE_ALPHA / NUM_CLAIMS  # 0.0125
MIN_ELIGIBLE = 30

# One-sided Bonferroni Student-t critical values t.ppf(1 - 0.05/4, n - 1) for
# the eligible n, computed once at design time with scipy:
#     from scipy.stats import t
#     {n: t.ppf(1 - 0.05 / 4, n - 1) for n in range(30, 37)}
# Numerically identical to the router and discovery experiments' table (same
# alpha, same eligible-n range); hard-coded so scipy is not a runtime
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
# ``condition`` names the condition the claim's per-seed statistics come
# from (``in-library``: the route condition; ``out-of-library``: the
# discover condition; ``null-control``: the Gaussian-null specificity
# control), ``statistic`` the frozen per-seed statistic of the protocol,
# and ``value`` the paired per-seed value the claim is decided on.
_CLAIM_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "h1-loop-acquisition-rate",
        "condition": "out-of-library",
        "statistic": "acquisition",
        "value": "acquired",
        "threshold": 0.95,
        "role": "primary",
        "description": (
            "the per-seed acquisition rate — the loop detects the "
            "out-of-library misfit, the certified discovery head "
            "certifies a constraint, the novelty gate admits it, and the "
            "planted map commutes with it (map_misfit <= 1e-9) — clears "
            "0.95"
        ),
    },
    {
        "id": "h2-loop-false-admission-control",
        "condition": "null-control",
        "statistic": "refusal",
        "value": "control_refused",
        "threshold": 0.95,
        "role": "secondary",
        "description": (
            "the per-seed null-control refusal rate — structure-free "
            "i.i.d. Gaussian observations through the same discovery "
            "path are refused, never certified and admitted into the "
            "library — clears 0.95, so the false-admission rate stays "
            "below 0.05"
        ),
    },
    {
        "id": "h3-loop-in-library-routing",
        "condition": "in-library",
        "statistic": "routing",
        "value": "routed_correctly",
        "threshold": 0.95,
        "role": "secondary",
        "description": (
            "the per-seed in-library routing rate — the alarm stays "
            "quiet and the loop commits the instance to the true target "
            "at library index 0 — clears 0.95"
        ),
    },
    {
        "id": "h4-loop-alarm-precision",
        "condition": "in-library",
        "statistic": "alarm_silence",
        "value": "alarm_silent",
        "threshold": 0.95,
        "role": "secondary",
        "description": (
            "the per-seed in-library alarm-silence rate — the misfit "
            "alarm does not fire when the library explains the instance "
            "— clears 0.95, so the false-alarm rate stays below 0.05"
        ),
    },
)
CLAIM_IDS = tuple(definition["id"] for definition in _CLAIM_DEFINITIONS)

# The per-seed statistic provenance named in each claim's estimand text.
_VALUE_PROVENANCE = {
    "acquired": (
        "1.0 iff universa.loop.acquisition_correct(outcome, instance) on "
        "the out-of-library row — the alarm fired, the discovery head "
        "certified, the novelty gate admitted, and map_misfit <= 1e-9 — "
        "else 0.0"
    ),
    "control_refused": (
        "1.0 iff the null-control row admitted nothing (NOT admitted: "
        "no constraint was certified and admitted into the library; a "
        "route-mode outcome admits nothing), else 0.0"
    ),
    "routed_correctly": (
        "1.0 iff universa.loop.routing_correct(outcome) on the "
        "in-library row — mode 'route' with routed_index 0, the true "
        "target's frozen library index — else 0.0"
    ),
    "alarm_silent": (
        "1.0 iff the in-library row's misfit alarm did not fire "
        "(best_score <= 1e-9), else 0.0"
    ),
}

# The subfields the seal record must carry for each frozen claim (id,
# statistic, theta, null, alternative, bound direction, threshold, support
# rule). Schema universa-seal/7 claim objects carry ``statistic`` in place
# of the router seals' fraction/reference: there is no operating fraction
# and no baseline arm. A claim object must carry exactly these keys, plus
# the optional ``role``; any other key is refused.
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
    "demo-scale numbers are not evidence: the acquisition, routing, "
    "refusal, and alarm-silence rates are properties of this one "
    "synthetic generator family at the frozen exp4 sizes (8 vertices, "
    "14 edges, 6 classes, 3 decoys, M = 16 observations) under the "
    "certified route-or-discover loop, not guarantees for other "
    "families, other library sizes, or real data; and the "
    "false-admission control is an i.i.d. Gaussian null matched to the "
    "target edge space, not a worst-case adversary"
)


class DesignFailureError(RuntimeError):
    """A frozen-design validation failure: the whole run is a design failure."""

    def __init__(self, message: str, *, seed: int | None = None) -> None:
        super().__init__(message)
        self.seed = seed


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
    result = subprocess.run(
        ("git", *args),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or "unknown git error"
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


def _validate_empty_train_block(block: Any) -> None:
    """The documented-empty train-block sentinel: ``{"first": 0, "last": 0}``.

    The loop trains nothing; seed ``0`` belongs to no Universa seed block
    and ``first > last`` encodes the empty set. Any other value is a
    seal-validation failure.
    """
    if (
        not isinstance(block, dict)
        or set(block) - {"first", "last"}
        or block.get("first") != 0
        or block.get("last") != 0
    ):
        raise RuntimeError(
            "stop condition: design seal train_seed_block must be exactly "
            "{'first': 0, 'last': 0} (the documented empty-block sentinel: "
            "this experiment trains nothing)"
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
    _validate_empty_train_block(seal["train_seed_block"])
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
    """Pin single-thread CPU float64 execution and refuse a visible CUDA.

    This runner computes with numpy float64 only — no torch tensor is ever
    created. For uniformity with the other sealed runners, when torch IS
    importable it is still pinned to exactly one thread and required to see
    no CUDA device; torch is not a dependency of this runner, so when it is
    absent those two checks are skipped and the CUDA-hidden requirement
    rests on the ``CUDA_VISIBLE_DEVICES`` override at import time (recorded
    below, alongside ``torch_importable: false``).
    """
    if torch is not None:
        torch.set_num_threads(1)
        if torch.get_num_threads() != 1:
            raise RuntimeError(
                "stop condition: PyTorch must run with exactly one thread, "
                f"got {torch.get_num_threads()}"
            )
        if torch.cuda.is_available():
            raise RuntimeError(
                "stop condition: CUDA must be unavailable or hidden "
                "(run with CUDA_VISIBLE_DEVICES=-1)"
            )
    probe = np.array([1.0, -2.0], dtype=np.float64)
    reduced = float((probe * 3.0).sum())
    if probe.dtype != np.float64 or reduced != -3.0:
        raise RuntimeError(
            "stop condition: CPU float64 numpy operation failed"
        )
    return {
        "compute": (
            "numpy float64 on CPU; no torch tensors anywhere in this runner"
        ),
        "array_dtype": "float64",
        "torch_importable": torch is not None,
        "torch_num_threads": (
            torch.get_num_threads() if torch is not None else None
        ),
        "torch_num_interop_threads": (
            torch.get_num_interop_threads() if torch is not None else None
        ),
        "cuda_available": (
            bool(torch.cuda.is_available()) if torch is not None else None
        ),
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
        "numpy": np.__version__,
        "torch": torch.__version__ if torch is not None else None,
        "platform": {
            "system": uname.system,
            "node": uname.node,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "processor": uname.processor,
        },
        "torch_cuda_build": (
            torch.version.cuda if torch is not None else None
        ),
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

    # 7. Environment: CUDA hidden, one torch thread when torch is
    # importable, CPU float64 numpy.
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
# Row construction: one row per condition per eligible seed.


def _undegraded_misfits(instance: Any) -> tuple[float, ...]:
    """Per-candidate max-degree clean commutation residual.

    The second half of the frozen eligibility gate (protocol §10): the true
    candidate's score must be exactly 0.0 and every decoy strictly above
    ALARM_TOL. Deterministic per seed, computed once per seed, purely as
    bookkeeping for seed accounting — it never enters the loop or any row
    (every condition row is computed by universa.loop.run_loop itself).
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


def _outcome_fields(outcome: LoopOutcome) -> dict[str, Any]:
    """The full :class:`universa.loop.LoopOutcome` scalar record, JSON-ready.

    Every claim statistic and every alarm/routing/acquisition/refusal
    decision is a function of these fields alone; no matrices are retained.
    """
    return {
        "mode": outcome.mode,
        "alarm_fired": bool(outcome.alarm_fired),
        "best_score": float(outcome.best_score),
        "scores": [float(score) for score in outcome.scores],
        "routed_index": outcome.routed_index,
        "discovery_verdict": outcome.discovery_verdict,
        "discovery_reason": outcome.discovery_reason,
        "certificate_residual": outcome.certificate_residual,
        "admitted": bool(outcome.admitted),
        "admission_min_distance": outcome.admission_min_distance,
        "map_misfit": outcome.map_misfit,
        "final_library_size": int(outcome.final_library_size),
        "initial_library_size": int(outcome.initial_library_size),
        "num_observations": outcome.num_observations,
        "admission_distances": (
            [float(distance) for distance in outcome.admission_distances]
            if outcome.admission_distances is not None
            else None
        ),
        "admission_reason": outcome.admission_reason,
    }


def _run_condition(
    seed: int,
    instance: Any,
    library: Any,
    *,
    observations: np.ndarray | None = None,
) -> LoopOutcome:
    """One frozen-loop pass over one condition, design-failure wrapped.

    A violation inside the certified loop machinery — scoring, the alarm,
    the discovery head, admission — is a whole-run design failure, never an
    execution failure; a foreign outcome type is one too (the symmetric
    check: without it a bad outcome would AttributeError downstream and
    classify as an execution failure).
    """
    try:
        outcome = run_loop(
            LOOP, instance, library, seed=seed, observations=observations
        )
    except DesignFailureError:
        raise
    except Exception as error:
        raise DesignFailureError(
            f"seed {seed}: certified loop pass failed: "
            f"{type(error).__name__}: {error}",
            seed=seed,
        ) from error
    if not isinstance(outcome, LoopOutcome):
        raise DesignFailureError(
            f"seed {seed}: run_loop returned a foreign outcome type: "
            f"{type(outcome).__name__}",
            seed=seed,
        )
    return outcome


def _build_in_library_row(
    seed: int, instance: Any, in_library: Any
) -> dict[str, Any]:
    """One in-library row: the H3 routing and H4 alarm-silence statistics.

    The loop scores the in-library view (the true target at index 0). The
    row records the raw outcome and the condition's two decision
    indicators — it does NOT enforce them: a fired alarm or a wrong
    commitment is an honest negative result scored by H3/H4, not a design
    failure.
    """
    outcome = _run_condition(seed, instance, in_library)
    return {
        "seed": seed,
        "condition": "in-library",
        **_outcome_fields(outcome),
        "routing_correct": bool(routing_correct(outcome)),
        "alarm_silent": not outcome.alarm_fired,
    }


def _build_out_of_library_row(
    seed: int, instance: Any, out_library: Any
) -> dict[str, Any]:
    """One out-of-library row: the H1 acquisition statistic.

    The loop scores the decoy-only view, the alarm fires, and the
    discovery path runs the frozen ``"discovery-observation"`` schedule.
    The row records the raw outcome and
    :func:`universa.loop.acquisition_correct` — a refusal or a certified
    discovery the planted map does not commute with scores 0, honestly.
    """
    outcome = _run_condition(seed, instance, out_library)
    return {
        "seed": seed,
        "condition": "out-of-library",
        **_outcome_fields(outcome),
        "acquisition_correct": bool(acquisition_correct(outcome, instance)),
    }


def _build_null_control_row(
    seed: int, instance: Any, out_library: Any
) -> dict[str, Any]:
    """One null-control row: the H2 false-admission specificity control.

    The frozen exp4 H4 schedule — 16 i.i.d. Gaussian columns on the
    seed's target edge space via :func:`universa.loop.null_observations` —
    goes through the SAME out-of-library loop pass as the observations
    override. The alarm fires on the decoy library; structure-free data
    must then be refused. The row records the raw outcome and the refusal
    indicator — a false admission scores 0, honestly.
    """
    ambient_dim = int(instance.true_target.boundaries[0].shape[1])
    try:
        observations = null_observations(seed, ambient_dim, NUM_OBSERVATIONS)
    except DesignFailureError:
        raise
    except Exception as error:
        raise DesignFailureError(
            f"seed {seed}: null observation schedule failed: "
            f"{type(error).__name__}: {error}",
            seed=seed,
        ) from error
    outcome = _run_condition(
        seed, instance, out_library, observations=observations
    )
    return {
        "seed": seed,
        "condition": "null-control",
        **_outcome_fields(outcome),
        "refused": not outcome.admitted,
    }


def _campaign_rows(
    eligible: list[tuple[int, Any, Any, Any]],
    in_library_rows: list[dict[str, Any]],
    out_of_library_rows: list[dict[str, Any]],
    null_control_rows: list[dict[str, Any]],
) -> None:
    """One row per condition per eligible seed, in seed order.

    The pipeline is a deterministic float64 function of the seed: every row
    is built TWICE and the two executions must be bit-identical, else the
    whole run is a design failure. Rows are appended to the caller's lists
    so a mid-campaign failure preserves every completed row: each row is
    appended as soon as its bit-identity check passes — BEFORE the seed's
    later conditions are built — so a null-control failure still preserves
    the seed's in-library and out-of-library rows (the per-seed view
    downstream covers only seeds complete in all three conditions).
    """
    for seed, instance, in_library, out_library in eligible:
        try:
            in_row = _build_in_library_row(seed, instance, in_library)
            if in_row != _build_in_library_row(seed, instance, in_library):
                raise DesignFailureError(
                    f"seed {seed}: in-library row is not bit-identical "
                    "across the frozen double execution",
                    seed=seed,
                )
            in_library_rows.append(in_row)
            out_row = _build_out_of_library_row(seed, instance, out_library)
            if out_row != _build_out_of_library_row(seed, instance, out_library):
                raise DesignFailureError(
                    f"seed {seed}: out-of-library row is not bit-identical "
                    "across the frozen double execution",
                    seed=seed,
                )
            out_of_library_rows.append(out_row)
            null_row = _build_null_control_row(seed, instance, out_library)
            if null_row != _build_null_control_row(seed, instance, out_library):
                raise DesignFailureError(
                    f"seed {seed}: null-control row is not bit-identical "
                    "across the frozen double execution",
                    seed=seed,
                )
        except BaseException as error:
            if getattr(error, "seed", None) is None:
                error.seed = seed  # type: ignore[attr-defined]
            raise
        null_control_rows.append(null_row)


# ---------------------------------------------------------------------------
# Inference: per-seed statistics and the four frozen claims.


def _estimand(definition: dict[str, Any]) -> str:
    statistic = definition["statistic"]
    return (
        f"mean over eligible seeds of {statistic}(seed) from the "
        f"{definition['condition']} condition; {statistic}(seed) is "
        f"{_VALUE_PROVENANCE[definition['value']]}"
    )


def _claim_summary(
    definition: dict[str, Any], values: list[float]
) -> dict[str, Any]:
    """One-sided Student-t lower bound for one claim's per-seed statistics.

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
        "condition": definition["condition"],
        "statistic": definition["statistic"],
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


def _recompute_per_seed(
    in_library_rows: list[dict[str, Any]],
    out_of_library_rows: list[dict[str, Any]],
    null_control_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Per-seed claim statistics, joined from the three conditions' rows.

    Every statistic is read off the recorded decision indicator of its
    condition's row: a fired in-library alarm scores routing 0.0 and alarm
    silence 0.0, a refused or non-commuting out-of-library pass scores
    acquisition 0.0, and a false admission on the null control scores
    refusal 0.0 — there is no conditioning on loop success and no deletion
    of seeds.
    """
    out_by_seed = {row["seed"]: row for row in out_of_library_rows}
    if len(out_by_seed) != len(out_of_library_rows):
        raise DesignFailureError("duplicate seed in the out-of-library rows")
    null_by_seed = {row["seed"]: row for row in null_control_rows}
    if len(null_by_seed) != len(null_control_rows):
        raise DesignFailureError("duplicate seed in the null-control rows")
    per_seed = []
    for in_row in in_library_rows:
        seed = in_row["seed"]
        if seed not in out_by_seed:
            raise DesignFailureError(
                f"seed {seed}: in-library row has no matching "
                "out-of-library row"
            )
        if seed not in null_by_seed:
            raise DesignFailureError(
                f"seed {seed}: in-library row has no matching "
                "null-control row"
            )
        out_row = out_by_seed[seed]
        null_row = null_by_seed[seed]
        per_seed.append(
            {
                "seed": seed,
                "acquired": 1.0 if out_row["acquisition_correct"] else 0.0,
                "control_refused": 1.0 if null_row["refused"] else 0.0,
                "routed_correctly": 1.0 if in_row["routing_correct"] else 0.0,
                "alarm_silent": 1.0 if in_row["alarm_silent"] else 0.0,
            }
        )
    return per_seed


def _claim_inference(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    claims = [
        _claim_summary(
            definition, [row[definition["value"]] for row in per_seed]
        )
        for definition in _CLAIM_DEFINITIONS
    ]
    return _claims_family(claims)


def _null_claims() -> dict[str, Any]:
    """All four frozen claims with a null decision after a failed run."""
    claims = []
    for definition in _CLAIM_DEFINITIONS:
        claims.append(
            {
                "id": definition["id"],
                "role": definition["role"],
                "description": definition["description"],
                "condition": definition["condition"],
                "statistic": definition["statistic"],
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


def _audit_row_decisions(row: dict[str, Any]) -> dict[str, Any]:
    """Recompute one row's decisions from its retained scalar record alone.

    The audit never re-runs the loop and never rebuilds an instance: the
    alarm decision (retained ``best_score`` against ``ALARM_TOL``, with
    ``best_score`` required to be the minimum of the retained scores), the
    mode/library coherence (route and refused leave the library unchanged
    and admit nothing; discover grows it by one and routes to the appended
    index; route commits to the first argmin of the retained scores), and
    the condition's decision indicator (routing and alarm silence for
    in-library rows, acquisition for out-of-library rows, refusal for
    null-control rows) are all recomputed exactly. Any disagreement is a
    whole-run design failure.
    """
    seed = row["seed"]
    condition = row["condition"]
    if condition not in CONDITIONS:
        raise DesignFailureError(
            f"seed {seed}: unknown condition {condition!r} in a raw row",
            seed=seed,
        )
    scores = row["scores"]
    if not isinstance(scores, list) or not scores:
        raise DesignFailureError(
            f"seed {seed}: retained scores must be a nonempty list",
            seed=seed,
        )
    if row["best_score"] != min(scores):
        raise DesignFailureError(
            f"seed {seed}: retained best_score {row['best_score']!r} is "
            "not the minimum of the retained scores",
            seed=seed,
        )
    alarm = row["best_score"] > ALARM_TOL
    if row["alarm_fired"] != alarm:
        raise DesignFailureError(
            f"seed {seed}: retained alarm flag {row['alarm_fired']} "
            f"disagrees with the recomputed alarm (best_score "
            f"{row['best_score']!r} against alarm_tol {ALARM_TOL})",
            seed=seed,
        )
    mode = row["mode"]
    initial = row["initial_library_size"]
    final = row["final_library_size"]
    if initial != len(scores):
        raise DesignFailureError(
            f"seed {seed}: retained initial_library_size {initial} does "
            f"not cover the {len(scores)} retained scores",
            seed=seed,
        )
    if mode == "route":
        if alarm:
            raise DesignFailureError(
                f"seed {seed}: route mode with a fired alarm is incoherent",
                seed=seed,
            )
        if row["routed_index"] != scores.index(row["best_score"]):
            raise DesignFailureError(
                f"seed {seed}: retained routed_index "
                f"{row['routed_index']!r} is not the first argmin of the "
                "retained scores (the frozen first-index tie-break)",
                seed=seed,
            )
        if row["admitted"] or final != initial:
            raise DesignFailureError(
                f"seed {seed}: route mode must admit nothing and leave "
                "the library unchanged",
                seed=seed,
            )
    elif mode == "discover":
        if not alarm:
            raise DesignFailureError(
                f"seed {seed}: discover mode without a fired alarm is "
                "incoherent",
                seed=seed,
            )
        if row["routed_index"] != initial or final != initial + 1:
            raise DesignFailureError(
                f"seed {seed}: discover mode must grow the library by one "
                "and route to the appended index",
                seed=seed,
            )
        if not row["admitted"]:
            raise DesignFailureError(
                f"seed {seed}: discover mode requires admission",
                seed=seed,
            )
    elif mode == "refused":
        if not alarm:
            raise DesignFailureError(
                f"seed {seed}: refused mode without a fired alarm is "
                "incoherent",
                seed=seed,
            )
        if row["routed_index"] is not None or row["admitted"] or final != initial:
            raise DesignFailureError(
                f"seed {seed}: a refused loop routes nowhere, admits "
                "nothing, and leaves the library unchanged",
                seed=seed,
            )
    else:
        raise DesignFailureError(
            f"seed {seed}: unknown loop mode {mode!r} in a raw row",
            seed=seed,
        )
    entry: dict[str, Any] = {
        "seed": seed,
        "condition": condition,
        "mode": mode,
        "alarm_fired": alarm,
    }
    if condition == "in-library":
        routing = mode == "route" and row["routed_index"] == 0
        if routing != row["routing_correct"]:
            raise DesignFailureError(
                f"seed {seed}: retained routing indicator disagrees with "
                "the recomputed routing decision",
                seed=seed,
            )
        silent = not alarm
        if silent != row["alarm_silent"]:
            raise DesignFailureError(
                f"seed {seed}: retained alarm-silence indicator disagrees "
                "with the recomputed alarm decision",
                seed=seed,
            )
        entry.update(routing_correct=routing, alarm_silent=silent)
    elif condition == "out-of-library":
        acquired = bool(
            mode == "discover"
            and alarm
            and row["discovery_verdict"] == "discovered"
            and row["certificate_residual"] is not None
            and row["admitted"]
            and row["map_misfit"] is not None
            and row["map_misfit"] <= ALARM_TOL
        )
        if acquired != row["acquisition_correct"]:
            raise DesignFailureError(
                f"seed {seed}: retained acquisition indicator disagrees "
                "with the recomputed acquisition decision",
                seed=seed,
            )
        entry["acquisition_correct"] = acquired
    else:  # null-control
        refused = not row["admitted"]
        if refused != row["refused"]:
            raise DesignFailureError(
                f"seed {seed}: retained refusal indicator disagrees with "
                "the recomputed refusal decision (NOT admitted)",
                seed=seed,
            )
        entry["refused"] = refused
    return entry


def _audit_block(
    eligibility: dict[str, Any],
    in_library_rows: list[dict[str, Any]],
    out_of_library_rows: list[dict[str, Any]],
    null_control_rows: list[dict[str, Any]],
    *,
    complete: bool,
) -> dict[str, Any]:
    """Eligibility counts plus raw-row-recomputable decisions and claims."""
    block: dict[str, Any] = {
        "recompute_scope": (
            "(a) every claim statistic and every "
            "alarm/routing/acquisition/refusal decision below is "
            "recomputed from the raw rows alone: the alarm decision "
            "(retained best_score against alarm_tol = 1e-9, and "
            "best_score as the minimum of the retained per-candidate "
            "scores), the routing decision (route mode commits to the "
            "first argmin of the retained scores; the routing indicator "
            "is mode 'route' to library index 0), the acquisition "
            "decision (mode 'discover' with a fired alarm, verdict "
            "'discovered', a certificate residual, admission, and "
            "map_misfit <= 1e-9), the refusal indicator (mode "
            "'refused'), and the mode/library coherence (route and "
            "refused leave the library unchanged; discover grows it by "
            "one and routes to the appended index), followed by the "
            "per-seed claim statistics, means, standard deviations, SEs, "
            "one-sided lower bounds, and decisions; the eligibility "
            "counts and seed ids come from the eligibility pass, not "
            "from the raw rows. (b) no matrices are retained or needed: "
            "every claim statistic is a function of the retained "
            "LoopOutcome scalar record, and no instance is rebuilt — "
            "the audit never re-runs the loop. Unlike the router "
            "experiments, no hash-pinned model exists or is needed, "
            "because nothing was trained."
        ),
        "declared_seeds": eligibility["declared"],
        "eligible_seeds": eligibility["eligible"],
        "eligible_seed_ids": list(eligibility["eligible_seeds"]),
        "ineligible_seed_rows": len(eligibility["ineligible"]),
        "build_failure_rows": len(eligibility["build_failures"]),
        "in_library_rows": len(in_library_rows),
        "out_of_library_rows": len(out_of_library_rows),
        "null_control_rows": len(null_control_rows),
        "rows_per_seed_per_condition": 1,
    }
    if complete:
        block["decision_recompute"] = [
            *(_audit_row_decisions(row) for row in in_library_rows),
            *(_audit_row_decisions(row) for row in out_of_library_rows),
            *(_audit_row_decisions(row) for row in null_control_rows),
        ]
        per_seed = _recompute_per_seed(
            in_library_rows, out_of_library_rows, null_control_rows
        )
        block["per_seed"] = per_seed
        claims: dict[str, Any] = {}
        for definition in _CLAIM_DEFINITIONS:
            values = [row[definition["value"]] for row in per_seed]
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
    return block


def _design_record() -> dict[str, Any]:
    return {
        "experiment_id": EXPERIMENT_ID,
        "canonical_command": CANONICAL_COMMAND,
        "task_family": (
            "graph-quotient switch instances via "
            "universa.loop.make_loop_instance "
            "(universa.budgets.make_budget_instance at the frozen exp4 "
            "sizes: 8 vertices, 14 edges, 6 classes, 3 decoys); ONE "
            "budget instance per seed with two library views — the "
            "in-library view instance.candidates (true target at index "
            "0) and the out-of-library view instance.decoy_targets (the "
            "truth withheld)"
        ),
        "observation_model": (
            "the vector-observation regime of universa.discovery, "
            "consumed through universa.loop.run_loop: M = 16 transported "
            "vectors y_j = f1 a_j over certified random source cycles, "
            "synthesized deterministically from the seed by "
            "universa.discovery.synthesize_observations (the frozen "
            "'discovery-observation' schedule); the null-control "
            "condition replaces them with 16 i.i.d. standard Gaussian "
            "columns from universa.loop.null_observations (the frozen "
            "'discovery-null' schedule)"
        ),
        "procedure": {
            "loop": (
                "universa.loop.LibraryLoop() — the frozen exp4 "
                "configuration: alarm_tol = 1e-9 (re-pinned "
                "universa.router.RESIDUAL_TOL), misfit_tol = CERT_TOL = "
                "1e-10, novelty_tol = 1e-6, num_observations = 16"
            ),
            "in_library_call": (
                "universa.loop.run_loop(LOOP, instance, "
                "instance.candidates, seed=seed)"
            ),
            "out_of_library_call": (
                "universa.loop.run_loop(LOOP, instance, "
                "instance.decoy_targets, seed=seed)"
            ),
            "null_control_call": (
                "universa.loop.run_loop(LOOP, instance, "
                "instance.decoy_targets, seed=seed, "
                "observations=universa.loop.null_observations(seed, "
                "ambient_dim, 16))"
            ),
            "num_observations": NUM_OBSERVATIONS,
            "alarm_tol": LOOP.alarm_tol,
            "misfit_tol": LOOP.misfit_tol,
            "novelty_tol": LOOP.novelty_tol,
            "map_misfit_tol": ALARM_TOL,
        },
        "conditions": {
            "in-library": (
                "one run_loop row per eligible seed over "
                "instance.candidates (the H3 routing and H4 "
                "alarm-silence statistics), retaining the full "
                "LoopOutcome scalar record plus the routing and "
                "alarm-silence indicators"
            ),
            "out-of-library": (
                "one run_loop row per eligible seed over "
                "instance.decoy_targets (the H1 acquisition statistic), "
                "retaining the full LoopOutcome scalar record plus the "
                "acquisition indicator"
            ),
            "null-control": (
                "one run_loop row per eligible seed over "
                "instance.decoy_targets with the observations override "
                "null_observations(seed, ambient_dim, 16) — column j "
                "drawn from np.random.default_rng(subseed(seed, "
                "'discovery-null', str(j))) — (the H2 refusal "
                "statistic), retaining the full LoopOutcome scalar "
                "record plus the refusal indicator"
            ),
        },
        "train_seed_block": dict(TRAIN_SEED_BLOCK),
        "training": (
            "no training provenance exists because no training occurred: "
            "the route-or-discover loop has no learned component; the "
            "train block is the documented-empty sentinel "
            "{'first': 0, 'last': 0}"
        ),
        "eval_seeds": {
            "first": SEALED_EVAL_SEEDS[0],
            "last": SEALED_EVAL_SEEDS[-1],
        },
        "determinism": (
            "the full per-seed pipeline — all three conditions — is "
            "executed twice and required bit-identical; any mismatch is "
            "a whole-run design_failure"
        ),
        "inference_unit": "the generator seed",
        "estimand": (
            "per-seed statistics from the three conditions: acquisition "
            "(universa.loop.acquisition_correct) from the out-of-library "
            "condition; refusal (NOT admitted) from the null-control "
            "condition; routing (universa.loop.routing_correct) and "
            "alarm silence (the alarm did not fire) from the in-library "
            "condition; each claim is the mean of its per-seed "
            "statistic over the eligible seeds"
        ),
        "familywise_alpha": FAMILYWISE_ALPHA,
        "per_claim_alpha": PER_CLAIM_ALPHA,
        "minimum_eligible": MIN_ELIGIBLE,
        "eligibility_rule": (
            "the instance builds AND the undegraded audit passes: "
            "universa.loop.make_loop_instance(seed) returns (a build "
            "exception is a whole-run design_failure, never an exclusion), "
            "the true candidate's undegraded misfit is exactly 0.0, and "
            "every decoy's undegraded misfit is strictly above 1e-9; an "
            "audit failure makes that one seed ineligible with a recorded "
            "reason (never a stop)"
        ),
        "row_count_invariant": (
            "3 x n: exactly one in-library row, one out-of-library row, "
            "and one null-control row per eligible seed (108 rows over "
            "the declared block when every declared seed is eligible), "
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

    # Eligibility pass over ALL sealed eval seeds, before any row: the
    # build is the sole gate, and an instance build exception is a
    # whole-run design failure (never an exclusion). No outcome-dependent
    # stopping, no seed deletion. Any other failure of the pass itself is
    # classified into the frozen status vocabulary below, preserving every
    # completed seed record.
    seed_records: list[dict[str, Any]] = []
    eligible: list[tuple[int, Any, Any, Any]] = []
    build_failure: dict[str, Any] | None = None
    eligibility_failure: tuple[str, BaseException] | None = None
    try:
        for seed in SEALED_EVAL_SEEDS:
            try:
                instance, in_library, out_library = make_loop_instance(seed)
            except Exception as error:  # a build exception stops the whole run
                build_failure = {
                    "seed": seed,
                    "type": type(error).__name__,
                    "message": str(error),
                }
                break
            misfits = _undegraded_misfits(instance)
            reason = None
            if misfits[0] != 0.0:
                reason = (
                    f"true candidate undegraded misfit {misfits[0]} != 0.0"
                )
            else:
                for misfit in misfits[1:]:
                    if misfit <= ALARM_TOL:
                        reason = (
                            f"decoy undegraded misfit {misfit} <= "
                            f"{ALARM_TOL}"
                        )
                        break
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
                eligible.append((seed, instance, in_library, out_library))
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
        "eligible_seeds": [seed for seed, _, _, _ in eligible],
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
        base["in_library_rows"] = []
        base["out_of_library_rows"] = []
        base["null_control_rows"] = []
        base["per_seed"] = []
        base["claims"] = _null_claims()
        base["audit"] = _audit_block(eligibility, [], [], [], complete=False)
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
            "never an exclusion; no rows were built"
        )
        base["in_library_rows"] = []
        base["out_of_library_rows"] = []
        base["null_control_rows"] = []
        base["per_seed"] = []
        base["claims"] = _null_claims()
        base["audit"] = _audit_block(eligibility, [], [], [], complete=False)
        return base
    if len(eligible) < MIN_ELIGIBLE:
        base["status"] = "design_failure_insufficient_eligible"
        base["stop_condition"] = (
            f"only {len(eligible)} seeds were eligible; minimum is "
            f"{MIN_ELIGIBLE}; no rows were built and no claims were decided"
        )
        base["in_library_rows"] = []
        base["out_of_library_rows"] = []
        base["null_control_rows"] = []
        base["per_seed"] = []
        base["claims"] = _null_claims()
        base["audit"] = _audit_block(eligibility, [], [], [], complete=False)
        return base

    in_library_rows: list[dict[str, Any]] = []
    out_of_library_rows: list[dict[str, Any]] = []
    null_control_rows: list[dict[str, Any]] = []
    failure: tuple[str, BaseException] | None = None
    try:
        _campaign_rows(
            eligible, in_library_rows, out_of_library_rows, null_control_rows
        )
        # Fail-closed row-count invariant (3 x n): every eligible seed must
        # have contributed exactly one row per condition before any summary.
        expected_rows = len(eligible)
        if (
            len(in_library_rows) != expected_rows
            or len(out_of_library_rows) != expected_rows
            or len(null_control_rows) != expected_rows
        ):
            raise DesignFailureError(
                f"row-count invariant violated: {len(in_library_rows)} "
                f"in-library rows, {len(out_of_library_rows)} "
                f"out-of-library rows, and {len(null_control_rows)} "
                f"null-control rows, expected {expected_rows} of each "
                "(3 x n: one row per condition per eligible seed; 108 "
                "rows over the declared block when every declared seed "
                "is eligible)"
            )
        # Summaries are computed only after every required raw row has
        # completed (frozen stop rule: no interim analysis).
        per_seed = _recompute_per_seed(
            in_library_rows, out_of_library_rows, null_control_rows
        )
        base["claims"] = _claim_inference(per_seed)
    except DesignFailureError as error:
        failure = ("design_failure", error)
    except KeyboardInterrupt as error:
        failure = ("interrupted", error)
    except Exception as error:  # unexpected faults are preserved
        failure = ("execution_failure", error)

    base["in_library_rows"] = in_library_rows
    base["out_of_library_rows"] = out_of_library_rows
    base["null_control_rows"] = null_control_rows
    if failure is not None:
        # A mid-seed stop can leave rows of a seed whose later conditions
        # were never built; the per-seed view then covers only the seeds
        # complete in all three conditions (every completed raw row is
        # preserved regardless).
        complete_seeds = (
            {row["seed"] for row in in_library_rows}
            & {row["seed"] for row in out_of_library_rows}
            & {row["seed"] for row in null_control_rows}
        )
        base["per_seed"] = _recompute_per_seed(
            [row for row in in_library_rows if row["seed"] in complete_seeds],
            [
                row
                for row in out_of_library_rows
                if row["seed"] in complete_seeds
            ],
            [
                row
                for row in null_control_rows
                if row["seed"] in complete_seeds
            ],
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
        base["audit"] = _audit_block(
            eligibility,
            in_library_rows,
            out_of_library_rows,
            null_control_rows,
            complete=False,
        )
        return base

    # TOCTOU guard, just before a COMPLETE artifact is published: the
    # worktree must still be clean, the on-disk seal must still equal its
    # HEAD blob, and the protocol, runner, and universa code files must be
    # unchanged since preflight. The post-run status is recorded next to
    # the pre-run one in the provenance.
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
    provenance["git_status_porcelain_post_run"] = post_run_status
    base["per_seed"] = _recompute_per_seed(
        in_library_rows, out_of_library_rows, null_control_rows
    )
    base["audit"] = _audit_block(
        eligibility,
        in_library_rows,
        out_of_library_rows,
        null_control_rows,
        complete=True,
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
