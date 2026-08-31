#!/usr/bin/env python3
"""Sealed runner for experiment ``universa-discovery-sealed-1``.

The frozen design: the certified discovery machinery of
:mod:`universa.discovery` — no learned component, no training, numpy
float64 only — is evaluated on the graph-quotient switch family
(``make_switch_instance(seed, num_vertices=8, num_edges=14, num_classes=6,
num_decoys=3)``) over the sealed eval seed block 90101..90136, with the
true quotient target WITHHELD from the library (decoys only). Each
eligible seed contributes TWO raw rows, one per condition:

* **structured** — one end-to-end :func:`universa.discovery.run_discovery`
  call with ``num_observations=16``, ``misfit_tol=1e-10`` (``CERT_TOL``)
  and ``novelty_tol=1e-6``: 16 transported vectors ``y_j = f1 a_j`` are
  synthesized deterministically from the seed, the constraint is
  discovered and certified, admitted (or refused) against the decoy
  library, and audited against the withheld truth. This condition is the
  source of the H1–H3 per-seed statistics.
* **null** — the H4 false-discovery (specificity) control: 16 i.i.d.
  standard Gaussian columns on the seed's target edge space, column ``j``
  drawn from ``np.random.default_rng(subseed(seed, "discovery-null",
  str(j)))``, fed through the SAME
  :func:`universa.discovery.discover_constraint` call at the module
  default tolerance. Structure-free data must be refused; a false
  constraint certified from noise is a specificity failure. The row
  retains the dimension trajectory (the numerical rank of the first
  ``j + 1`` columns, ``j = 0..15``).

Because discovery trains nothing, the train seed block is the
documented-empty sentinel ``{"first": 0, "last": 0}`` — seed ``0``
belongs to no Universa seed block and ``first > last`` encodes the empty
set. There is no fit, no train rows, and no train-side audits. The
inference unit is the generator seed for the four one-sided Bonferroni
claims (per-claim alpha 0.05/4 = 0.0125), each decided by a one-sided
Student-t lower bound against its frozen threshold (0.9 / 0.95 / 0.95 /
0.95), including the mechanical ``SE = 0`` case (the lower bound then
equals the estimate). Refusals count against H1–H3 (coverage 0.0,
indicators 0); H2 is mechanically the discovery-success rate (the
discovery gate's tolerance and the claim threshold are the same
constant); H3's statistic is the planted-map misfit indicator at the
frozen threshold ``1e-9``, distinct from the admission gate's ``1e-10``.

**Determinism, verified.** The pipeline is a deterministic float64
function of the seed: the runner executes the full per-seed pipeline —
both conditions — TWICE for every eligible seed and requires bit-identical
raw rows; any mismatch is a whole-run ``design_failure``.

**Retention (frozen).** Every structured row retains the discovered
``boundary`` and ``support_basis`` matrices (small dims), the observation
matrix, and the transported certified source-cycle image; every null row
retains the null observation matrix (and the matrices of any false
discovery). Because nothing is trained, there is no model to pin: the
audit block recomputes the certificate residual, the boundary/support
annihilation, the support orthonormality, the planted map's misfit, the
coverage column count, and the null dimension trajectory from the
retained matrices alone, re-derives the observations from the frozen seed
schedules (bit-equality required), and re-runs admission and quality from
the reconstructed :class:`universa.discovery.DiscoveredConstraint` — the
exact recompute scope is stated in the audit block.

**No torch.** Discovery is numpy-only: this runner computes no torch
tensor anywhere. For uniformity with the other sealed runners the
environment check still requires CUDA hidden, and sets/verifies exactly
one torch thread *when torch is importable in the environment*; torch is
not a dependency of this runner, and when it is absent the CUDA-hidden
requirement is enforced by the ``CUDA_VISIBLE_DEVICES`` override at
import time. See :func:`_execution_environment`.

**Declared caveat (frozen).** Demo-scale numbers are not evidence: the
certification, coverage, and misfit rates are properties of this one
synthetic generator family at M = 16 observations under the certified
discovery head, not guarantees for other families, other observation
counts, or real data; and the H4 control is an i.i.d. Gaussian null
matched to the target edge space, not a worst-case adversary.

This runner must not be executed on the sealed eval seed block until the
protocol, this runner, and the seal record (``docs/11-discovery-seal.json``)
have been committed and pushed. It keeps the seed block inert at import
time: no instance is constructed until every preflight check in
:func:`_preflight` has passed, in the frozen order — output missing, clean
worktree, seal committed at HEAD, seal validation, file hashes, runtime
hash, environment — and only then any data construction.
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

# torch is NOT a dependency of this runner: discovery is numpy-only and no
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
from universa.discovery import (
    CERT_TOL,
    DEFAULT_NOVELTY_TOL,
    MIN_OBSERVATION_NORM,
    STABILITY_FRACTION,
    DiscoveredConstraint,
    DiscoveryInsufficient,
    admit_to_library,
    discover_constraint,
    discovery_quality,
    run_discovery,
    synthesize_observations,
)
from universa.generators import make_switch_instance, subseed
from universa.operators import nullspace_basis

EXPERIMENT_ID = "universa-discovery-sealed-1"
RESULT_SCHEMA = "universa-discovery-sealed-result/1"

# The one canonical execution command, recorded verbatim in every result.
CANONICAL_COMMAND = (
    "env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src python "
    "scripts/run_discovery_sealed_1.py --output "
    "results/experiments/discovery-sealed-1.json"
)

PROTOCOL = "docs/10-sealed-discovery-protocol.md"
RUNNER_SOURCE = "scripts/run_discovery_sealed_1.py"
DEFAULT_SEAL = "docs/11-discovery-seal.json"
SEAL_SCHEMA = "universa-seal/4"

PROTOCOL_SHA256 = (
    "41c25ee7f0a1a03d5cbe45fe4b89f24cc43c6aebb43fb440dbfa521588cf54d7"
)
"""Frozen fingerprint of the sealed protocol.

The committed runner embeds the sealed protocol's SHA-256 as this constant;
the value must equal the seal record's ``protocol_sha256``. Any protocol edit
after the seal requires re-pinning this constant (and a new seal record) —
the fail-closed refusal in :func:`_frozen_protocol_sha256` then catches the
mismatch. While the value is the placeholder ``PENDING_PROTOCOL_SHA256`` the
runner refuses every execution.
"""

# Frozen task family: graph-quotient switch instances; the true target is
# withheld from the library (decoys only) and must be discovered.
NUM_VERTICES = 8
NUM_EDGES = 14
NUM_CLASSES = 6
NUM_DECOYS = 3

NUM_OBSERVATIONS = 16  # M observation vectors per seed, both conditions
MISFIT_TOL = CERT_TOL  # 1e-10; discovery gate and admission misfit gate
NOVELTY_TOL = DEFAULT_NOVELTY_TOL  # 1e-6; admission novelty gate
MAP_MISFIT_TOL = 1e-9
"""Frozen H3 claim threshold on the planted-map misfit ``||C_disc f1 Z||_F``
(the router-acceptance residual) — distinct from the admission gate's
``misfit_tol = 1e-10`` on the certificate residual."""

# Discovery trains nothing: the train block is the documented-empty
# sentinel {"first": 0, "last": 0} — seed 0 belongs to no Universa seed
# block and first > last encodes the empty set.
TRAIN_SEEDS: tuple[int, ...] = ()
TRAIN_SEED_BLOCK = {"first": 0, "last": 0}
SEALED_EVAL_SEEDS = tuple(range(90101, 90137))  # 90101..90136

FAMILYWISE_ALPHA = 0.05
NUM_CLAIMS = 4
PER_CLAIM_ALPHA = FAMILYWISE_ALPHA / NUM_CLAIMS  # 0.0125
MIN_ELIGIBLE = 30

# Tolerance for the audit's recomputed float64 norms against the recorded
# values: the recompute redoes the identical operation on the retained
# matrices, so anything beyond BLAS-level variation is tampering.
AUDIT_RECOMPUTE_TOL = 1e-12

# One-sided Bonferroni Student-t critical values t.ppf(1 - 0.05/4, n - 1) for
# the eligible n, computed once at design time with scipy:
#     from scipy.stats import t
#     {n: t.ppf(1 - 0.05 / 4, n - 1) for n in range(30, 37)}
# Numerically identical to the router experiments' table (same alpha, same
# eligible-n range); hard-coded so scipy is not a runtime dependency.
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
# from (``structured``: the end-to-end discovery run; ``null``: the
# Gaussian null control), ``statistic`` the frozen per-seed statistic of
# the protocol, and ``value`` the paired per-seed value the claim is
# decided on.
_CLAIM_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "h1-discovery-coverage-floor",
        "condition": "structured",
        "statistic": "coverage",
        "value": "coverage_fraction",
        "threshold": 0.9,
        "role": "primary",
        "description": (
            "the mean per-seed coverage fraction dim(S_disc)/dim(ker "
            "B1_true) of the discovery pipeline clears 0.9 (a refused "
            "discovery scores 0)"
        ),
    },
    {
        "id": "h2-discovery-certification-rate",
        "condition": "structured",
        "statistic": "cert",
        "value": "certified",
        "threshold": 0.95,
        "role": "secondary",
        "description": (
            "the per-seed certification rate — discovery succeeds and the "
            "certificate residual passes 1e-10, mechanically the "
            "discovery-success rate — clears 0.95"
        ),
    },
    {
        "id": "h3-discovery-admission-ready",
        "condition": "structured",
        "statistic": "misfit",
        "value": "misfit_ok",
        "threshold": 0.95,
        "role": "secondary",
        "description": (
            "the per-seed router-acceptance rate — the planted map's "
            "misfit ||C_disc f1 Z||_F against the discovered constraint "
            "passes 1e-9 — clears 0.95"
        ),
    },
    {
        "id": "h4-discovery-false-discovery-control",
        "condition": "null",
        "statistic": "refusal",
        "value": "control_refused",
        "threshold": 0.95,
        "role": "secondary",
        "description": (
            "the per-seed null-condition refusal rate — "
            "discover_constraint refuses the i.i.d. Gaussian control "
            "observations on the target edge space — clears 0.95, so the "
            "false-discovery rate stays below 0.05"
        ),
    },
)
CLAIM_IDS = tuple(definition["id"] for definition in _CLAIM_DEFINITIONS)

# The per-seed statistic provenance named in each claim's estimand text.
_VALUE_PROVENANCE = {
    "coverage_fraction": (
        "quality.coverage_fraction when discovery succeeds, 0.0 for a "
        "refused discovery"
    ),
    "certified": (
        "1.0 iff discovery succeeds and certificate_residual <= 1e-10, "
        "else 0.0 (mechanically the discovery-success rate: the discovery "
        "gate's tolerance is the same constant)"
    ),
    "misfit_ok": (
        "1.0 iff discovery succeeds and map_misfit <= 1e-9, else 0.0"
    ),
    "control_refused": (
        "1.0 iff discover_constraint returns DiscoveryInsufficient on "
        "the null observations, else 0.0"
    ),
}

# The subfields the seal record must carry for each frozen claim (id,
# statistic, theta, null, alternative, bound direction, threshold, support
# rule). Schema universa-seal/4 claim objects carry ``statistic`` in place
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
    "demo-scale numbers are not evidence: the certification, coverage, and "
    "misfit rates are properties of this one synthetic generator family "
    "at M = 16 observations under the certified discovery head, not "
    "guarantees for other families, other observation counts, or real "
    "data; and the H4 control is an i.i.d. Gaussian null matched to the "
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

    Discovery trains nothing; seed ``0`` belongs to no Universa seed block
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
# Row construction: one structured row and one null row per eligible seed.


def _target_kernel_dim(instance: Any) -> int:
    """Certified dimension of ``ker(B1_true)``, recorded for audit.

    Deterministic per seed and observation-independent: computed from the
    undegraded true target alone via
    :func:`universa.operators.nullspace_basis` (a certification failure
    raises, and is classified by the frozen failure mapping).
    """
    return int(nullspace_basis(instance.true_target.boundaries[0]).basis.shape[1])


def _null_observations(
    seed: int, ambient_dim: int, num_observations: int
) -> np.ndarray:
    """The frozen H4 null draw: i.i.d. Gaussians on the target edge space.

    Column ``j`` is ``np.random.default_rng(subseed(seed, "discovery-null",
    str(j))).standard_normal(ambient_dim)`` — one independent generator per
    column, exactly the frozen schedule, a disjoint subseed family from the
    structured condition's ``"discovery-observation"`` draws.
    """
    if num_observations < 0:
        raise ValueError("num_observations must be nonnegative")
    columns = [
        np.random.default_rng(subseed(seed, "discovery-null", str(j)))
        .standard_normal(ambient_dim)
        for j in range(num_observations)
    ]
    if not columns:
        return np.zeros((ambient_dim, 0))
    return np.column_stack(columns)


def _numerical_rank(matrix: np.ndarray) -> int:
    """Numerical rank under the operators.py SVD tolerance convention.

    Duplicated here (rather than imported from the discovery module's
    private helper) so the audit's recomputations — the null dimension
    trajectory and the refusal-consistency ranks — are independent of the
    module under test: ``max(shape) * eps * sigma_0``.
    """
    singulars = np.linalg.svd(matrix, compute_uv=False)
    if singulars.size == 0:
        return 0
    rank_tol = max(matrix.shape) * np.finfo(float).eps * singulars[0]
    return int((singulars > rank_tol).sum())


def _dimension_trajectory(observations: np.ndarray) -> list[int]:
    """Rank of the first ``j + 1`` columns for ``j = 0..M-1`` (M integers)."""
    return [
        _numerical_rank(observations[:, : j + 1])
        for j in range(observations.shape[1])
    ]


def _build_structured_row(seed: int) -> dict[str, Any]:
    """One structured-condition row: the end-to-end certified discovery run.

    Calls :func:`universa.discovery.run_discovery` with the frozen sizes and
    tolerances and retains everything the audit recomputes from: the
    observation matrix, the transported certified source-cycle image, and —
    when discovery succeeds — the discovered ``boundary`` and
    ``support_basis`` matrices. A violation inside the certified discovery
    machinery — including a nonzero-observation draw failure — is a
    whole-run design failure, never an execution failure.
    """
    try:
        run = run_discovery(
            seed,
            num_observations=NUM_OBSERVATIONS,
            num_vertices=NUM_VERTICES,
            num_edges=NUM_EDGES,
            num_classes=NUM_CLASSES,
            num_decoys=NUM_DECOYS,
            misfit_tol=MISFIT_TOL,
            novelty_tol=NOVELTY_TOL,
        )
    except DesignFailureError:
        raise
    except Exception as error:
        raise DesignFailureError(
            f"seed {seed}: certified discovery run failed: "
            f"{type(error).__name__}: {error}",
            seed=seed,
        ) from error
    ambient_dim = int(run.switch.true_target.boundaries[0].shape[1])
    observations = run.observations
    if observations.shape != (ambient_dim, NUM_OBSERVATIONS):
        raise DesignFailureError(
            f"seed {seed}: observations have shape {observations.shape}, "
            f"expected {(ambient_dim, NUM_OBSERVATIONS)}",
            seed=seed,
        )
    if not np.isfinite(observations).all():
        raise DesignFailureError(
            f"seed {seed}: non-finite observations", seed=seed
        )
    try:
        cycle_basis = nullspace_basis(run.switch.source.boundaries[0]).basis
        transported = run.switch.chain_map.maps[1] @ cycle_basis
        ker_dim = _target_kernel_dim(run.switch)
    except Exception as error:
        raise DesignFailureError(
            f"seed {seed}: certified audit-basis construction failed: "
            f"{type(error).__name__}: {error}",
            seed=seed,
        ) from error
    result = run.result
    discovered = isinstance(result, DiscoveredConstraint)
    if not discovered and not isinstance(result, DiscoveryInsufficient):
        raise DesignFailureError(
            f"seed {seed}: discovery returned neither a DiscoveredConstraint "
            f"nor a DiscoveryInsufficient: {type(result).__name__}",
            seed=seed,
        )
    row: dict[str, Any] = {
        "seed": seed,
        "condition": "structured",
        "verdict": "discovered" if discovered else "insufficient",
        "reason": None,
        "ambient_dim": ambient_dim,
        "ker_dim": ker_dim,
        "num_observations": NUM_OBSERVATIONS,
        "discovered": discovered,
        "certified": False,
        "admitted": False,
        "discovered_dim": None,
        "coverage_fraction": None,
        "containment_residual": None,
        "certificate_residual": None,
        "admission_min_distance": None,
        "map_misfit": None,
        "observations": observations.tolist(),
        "transported_cycle_image": transported.tolist(),
        "boundary": None,
        "support_basis": None,
    }
    if discovered:
        if run.admission is None or run.quality is None or run.map_misfit is None:
            raise DesignFailureError(
                f"seed {seed}: a successful discovery must carry admission, "
                "quality, and map misfit audits",
                seed=seed,
            )
        if result.certificate_residual > MISFIT_TOL:
            raise DesignFailureError(
                f"seed {seed}: discovery returned a structure with "
                f"certificate residual {result.certificate_residual:.3e} "
                f"above misfit_tol {MISFIT_TOL:.3e}",
                seed=seed,
            )
        if run.quality.kernel_dim != ker_dim:
            raise DesignFailureError(
                f"seed {seed}: quality kernel dimension "
                f"{run.quality.kernel_dim} disagrees with the certified "
                f"audit {ker_dim}",
                seed=seed,
            )
        row.update(
            reason=run.admission.reason,
            certified=True,
            admitted=bool(run.admission.admitted),
            discovered_dim=result.coverage,
            coverage_fraction=float(run.quality.coverage_fraction),
            containment_residual=float(run.quality.containment_residual),
            certificate_residual=float(result.certificate_residual),
            admission_min_distance=float(run.admission.min_distance),
            map_misfit=float(run.map_misfit),
            boundary=result.boundary.tolist(),
            support_basis=result.support_basis.tolist(),
        )
    else:
        row["reason"] = result.reason
        if result.certificate_residual is not None:
            row["certificate_residual"] = float(result.certificate_residual)
    return row


def _null_row_from_observations(
    seed: int, observations: np.ndarray
) -> dict[str, Any]:
    """The H4 null-condition outcome of one observation matrix, with retention.

    The matrix goes through :func:`universa.discovery.discover_constraint`
    directly — no library, no quality audit: certification on
    structure-free data would be a false discovery, and that indicator is
    all H4 reads. The row retains the dimension trajectory (the rank of the
    first ``j + 1`` columns, ``j = 0..M-1``), the null matrix, and — for a
    false discovery — the falsely certified matrices, so the audit can
    recompute the outcome from the row alone.
    """
    ambient_dim = int(observations.shape[0])
    if observations.shape != (ambient_dim, NUM_OBSERVATIONS):
        raise DesignFailureError(
            f"seed {seed}: null observations have shape "
            f"{observations.shape}, expected "
            f"{(ambient_dim, NUM_OBSERVATIONS)}",
            seed=seed,
        )
    if not np.isfinite(observations).all():
        raise DesignFailureError(
            f"seed {seed}: non-finite null observations", seed=seed
        )
    try:
        result = discover_constraint(observations, ambient_dim, seeds=(seed,))
    except DesignFailureError:
        raise
    except Exception as error:
        raise DesignFailureError(
            f"seed {seed}: certified null discovery failed: "
            f"{type(error).__name__}: {error}",
            seed=seed,
        ) from error
    false_discovery = isinstance(result, DiscoveredConstraint)
    if not false_discovery and not isinstance(result, DiscoveryInsufficient):
        raise DesignFailureError(
            f"seed {seed}: null discovery returned neither a "
            f"DiscoveredConstraint nor a DiscoveryInsufficient: "
            f"{type(result).__name__}",
            seed=seed,
        )
    if false_discovery and result.certificate_residual > MISFIT_TOL:
        raise DesignFailureError(
            f"seed {seed}: null discovery returned a structure with "
            f"certificate residual {result.certificate_residual:.3e} above "
            f"misfit_tol {MISFIT_TOL:.3e}",
            seed=seed,
        )
    row: dict[str, Any] = {
        "seed": seed,
        "condition": "null",
        "refusal": not false_discovery,
        "reason": None,
        "ambient_dim": ambient_dim,
        "num_observations": NUM_OBSERVATIONS,
        "dimension_trajectory": _dimension_trajectory(observations),
        "false_discovery": false_discovery,
        "certificate_residual": None,
        "discovered_dim": None,
        "observations": observations.tolist(),
        "boundary": None,
        "support_basis": None,
    }
    if false_discovery:
        row.update(
            certificate_residual=float(result.certificate_residual),
            discovered_dim=result.coverage,
            boundary=result.boundary.tolist(),
            support_basis=result.support_basis.tolist(),
        )
    else:
        row["reason"] = result.reason
        if result.certificate_residual is not None:
            row["certificate_residual"] = float(result.certificate_residual)
    return row


def _build_null_row(seed: int, instance: Any) -> dict[str, Any]:
    """One null-condition row: the frozen Gaussian null schedule on the
    seed's target edge space, through ``discover_constraint`` directly."""
    ambient_dim = int(instance.true_target.boundaries[0].shape[1])
    observations = _null_observations(seed, ambient_dim, NUM_OBSERVATIONS)
    return _null_row_from_observations(seed, observations)


def _campaign_rows(
    eligible: list[tuple[int, Any]],
    structured_rows: list[dict[str, Any]],
    null_rows: list[dict[str, Any]],
) -> None:
    """One structured row and one null row per eligible seed, in seed order.

    The pipeline is a deterministic float64 function of the seed: every row
    is built TWICE and the two executions must be bit-identical, else the
    whole run is a design failure. Rows are appended to the caller's lists
    so a mid-campaign failure preserves every completed row: the structured
    row is appended as soon as its bit-identity check passes — BEFORE its
    null row is built — so a null-row failure still preserves it (the
    paired view downstream covers only seeds complete in both conditions).
    """
    for seed, instance in eligible:
        try:
            structured = _build_structured_row(seed)
            if structured != _build_structured_row(seed):
                raise DesignFailureError(
                    f"seed {seed}: structured row is not bit-identical "
                    "across the frozen double execution",
                    seed=seed,
                )
            structured_rows.append(structured)
            null = _build_null_row(seed, instance)
            if null != _build_null_row(seed, instance):
                raise DesignFailureError(
                    f"seed {seed}: null row is not bit-identical across "
                    "the frozen double execution",
                    seed=seed,
                )
        except BaseException as error:
            if getattr(error, "seed", None) is None:
                error.seed = seed  # type: ignore[attr-defined]
            raise
        null_rows.append(null)


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


def _recompute_paired(
    structured_rows: list[dict[str, Any]], null_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Per-seed claim statistics, joined from the two conditions' rows.

    Refusal counts against H1–H3: a refused structured discovery scores
    coverage 0.0, cert 0.0, misfit 0.0 — there is no conditioning on
    discovery success and no deletion of refused seeds. A null-condition
    false discovery scores refusal 0.0.
    """
    null_by_seed = {row["seed"]: row for row in null_rows}
    if len(null_by_seed) != len(null_rows):
        raise DesignFailureError("duplicate seed in the null condition rows")
    paired = []
    for structured in structured_rows:
        seed = structured["seed"]
        if seed not in null_by_seed:
            raise DesignFailureError(
                f"seed {seed}: structured row has no matching null row"
            )
        null = null_by_seed[seed]
        discovered = structured["discovered"]
        paired.append(
            {
                "seed": seed,
                "coverage_fraction": (
                    float(structured["coverage_fraction"])
                    if discovered
                    else 0.0
                ),
                "certified": (
                    1.0
                    if discovered
                    and structured["certificate_residual"] <= MISFIT_TOL
                    else 0.0
                ),
                "misfit_ok": (
                    1.0
                    if discovered
                    and structured["map_misfit"] <= MAP_MISFIT_TOL
                    else 0.0
                ),
                "control_refused": 1.0 if null["refusal"] else 0.0,
            }
        )
    return paired


def _claim_inference(paired: list[dict[str, Any]]) -> dict[str, Any]:
    claims = [
        _claim_summary(
            definition, [row[definition["value"]] for row in paired]
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


def _as_matrix(payload: Any, *, label: str, seed: int) -> np.ndarray:
    matrix = np.asarray(payload, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise DesignFailureError(
            f"seed {seed}: retained {label} is not a finite 2-D matrix",
            seed=seed,
        )
    return matrix


def _require_close(actual: float, expected: float, *, label: str, seed: int) -> None:
    if abs(actual - expected) > AUDIT_RECOMPUTE_TOL * max(1.0, abs(expected)):
        raise DesignFailureError(
            f"seed {seed}: audit recomputation of {label} gives {actual!r}, "
            f"the recorded value is {expected!r}",
            seed=seed,
        )


def _rank_pair(observations: np.ndarray) -> tuple[int, int]:
    """(full rank, stabilized-prefix rank) under the frozen stability rule."""
    num_observations = observations.shape[1]
    holdout = max(1, int(STABILITY_FRACTION * num_observations))
    return (
        _numerical_rank(observations),
        _numerical_rank(observations[:, : num_observations - holdout]),
    )


def _audit_structured_row_recompute(row: dict[str, Any]) -> dict[str, Any]:
    """Recompute one structured row's certificates from its retained data.

    Layer 1 (raw rows alone): certificate residual, boundary/support
    annihilation, support orthonormality, planted-map misfit, and coverage
    column count from the retained ``boundary`` / ``support_basis`` /
    ``observations`` / ``transported_cycle_image`` — without re-running the
    discovery SVD. Layer 2 (the deterministic, manifest-pinned generator
    rebuild): the retained observations must equal the
    :func:`universa.discovery.synthesize_observations` draw of the seed's
    rebuilt instance (bit-equality), and admission/quality are re-run from
    the reconstructed :class:`DiscoveredConstraint`. Any disagreement is a
    whole-run design failure.
    """
    seed = row["seed"]
    observations = _as_matrix(
        row["observations"], label="observations", seed=seed
    )
    entry: dict[str, Any] = {
        "seed": seed,
        "condition": "structured",
        "verdict": row["verdict"],
    }
    try:
        instance = make_switch_instance(
            seed, NUM_VERTICES, NUM_EDGES, NUM_CLASSES, NUM_DECOYS
        )
        scheduled = synthesize_observations(instance, NUM_OBSERVATIONS)
    except Exception as error:
        raise DesignFailureError(
            f"seed {seed}: audit could not rebuild the seed-schedule "
            f"observations: {type(error).__name__}: {error}",
            seed=seed,
        ) from error
    if not np.array_equal(observations, scheduled):
        raise DesignFailureError(
            f"seed {seed}: retained observations differ from the frozen "
            "seed-schedule draw",
            seed=seed,
        )
    entry["observations_match_seed_schedule"] = True
    if not row["discovered"]:
        rank, prefix_rank = _rank_pair(observations)
        if row["certificate_residual"] is not None:
            # A certification refusal: the module computed a boundary whose
            # residual exceeded tol, but a refused boundary is never
            # returned, so the recorded residual is not row-recomputable.
            # The rank configuration must be consistent with reaching
            # certification at all.
            consistent = (
                prefix_rank == rank
                and 0 < rank < row["ambient_dim"]
                and row["certificate_residual"] > CERT_TOL
            )
        else:
            consistent = (
                prefix_rank != rank
                or rank == 0
                or rank == row["ambient_dim"]
            )
        if not consistent:
            raise DesignFailureError(
                f"seed {seed}: retained observations are inconsistent with "
                f"the recorded refusal (rank {rank}, prefix {prefix_rank}, "
                f"ambient {row['ambient_dim']})",
                seed=seed,
            )
        entry.update(observed_rank=rank, prefix_rank=prefix_rank)
        return entry
    boundary = _as_matrix(row["boundary"], label="boundary", seed=seed)
    support = _as_matrix(
        row["support_basis"], label="support_basis", seed=seed
    )
    transported = _as_matrix(
        row["transported_cycle_image"],
        label="transported_cycle_image",
        seed=seed,
    )
    certificate = float(np.linalg.norm(boundary @ observations))
    _require_close(
        certificate,
        row["certificate_residual"],
        label="certificate residual",
        seed=seed,
    )
    annihilation = float(np.linalg.norm(boundary @ support))
    orthonormality = float(
        np.linalg.norm(support.T @ support - np.eye(support.shape[1]))
    )
    if annihilation > CERT_TOL or orthonormality > CERT_TOL:
        raise DesignFailureError(
            f"seed {seed}: retained matrices fail certification "
            f"(annihilation {annihilation:.3e}, orthonormality "
            f"{orthonormality:.3e}, tol {CERT_TOL})",
            seed=seed,
        )
    map_misfit = float(np.linalg.norm(boundary @ transported))
    _require_close(
        map_misfit, row["map_misfit"], label="map misfit", seed=seed
    )
    if int(support.shape[1]) != row["discovered_dim"]:
        raise DesignFailureError(
            f"seed {seed}: retained support basis has {support.shape[1]} "
            f"columns, the row records {row['discovered_dim']}",
            seed=seed,
        )
    reconstructed = DiscoveredConstraint(
        boundary,
        support,
        certificate,
        row["num_observations"],
        (seed,),
    )
    quality = discovery_quality(reconstructed, instance.true_target.boundaries[0])
    _require_close(
        quality.containment_residual,
        row["containment_residual"],
        label="containment residual",
        seed=seed,
    )
    _require_close(
        quality.coverage_fraction,
        row["coverage_fraction"],
        label="coverage fraction",
        seed=seed,
    )
    admission = admit_to_library(
        tuple(d.boundaries[0] for d in instance.decoy_targets),
        reconstructed,
        MISFIT_TOL,
        NOVELTY_TOL,
    )
    if admission.admitted != row["admitted"]:
        raise DesignFailureError(
            f"seed {seed}: audit admission verdict "
            f"{admission.admitted} disagrees with the recorded "
            f"{row['admitted']}",
            seed=seed,
        )
    _require_close(
        admission.min_distance,
        row["admission_min_distance"],
        label="admission min distance",
        seed=seed,
    )
    entry.update(
        certificate_residual=certificate,
        annihilation_residual=annihilation,
        support_orthonormality=orthonormality,
        map_misfit=map_misfit,
        discovered_dim=int(support.shape[1]),
        coverage_fraction=quality.coverage_fraction,
        containment_residual=quality.containment_residual,
        admitted=admission.admitted,
        admission_min_distance=admission.min_distance,
    )
    return entry


def _audit_null_row_recompute(row: dict[str, Any]) -> dict[str, Any]:
    """Recompute one null row's outcome from its retained null matrix.

    The retained observations must equal the frozen ``discovery-null``
    subseed schedule (bit-equality, no instance needed); the dimension
    trajectory recomputes exactly; the rank configuration must be
    consistent with the recorded refusal, and a false discovery's
    certificate recomputes from its retained matrices.
    """
    seed = row["seed"]
    observations = _as_matrix(
        row["observations"], label="null observations", seed=seed
    )
    scheduled = _null_observations(
        seed, row["ambient_dim"], row["num_observations"]
    )
    if not np.array_equal(observations, scheduled):
        raise DesignFailureError(
            f"seed {seed}: retained null observations differ from the "
            "frozen discovery-null schedule",
            seed=seed,
        )
    trajectory = _dimension_trajectory(observations)
    if trajectory != row["dimension_trajectory"]:
        raise DesignFailureError(
            f"seed {seed}: recomputed dimension trajectory "
            f"{trajectory} differs from the recorded "
            f"{row['dimension_trajectory']}",
            seed=seed,
        )
    entry: dict[str, Any] = {
        "seed": seed,
        "condition": "null",
        "refusal": row["refusal"],
        "observations_match_seed_schedule": True,
        "dimension_trajectory": trajectory,
    }
    rank, prefix_rank = _rank_pair(observations)
    if row["false_discovery"]:
        if not (prefix_rank == rank and 0 < rank < row["ambient_dim"]):
            raise DesignFailureError(
                f"seed {seed}: retained null observations are "
                f"inconsistent with the recorded false discovery (rank "
                f"{rank}, prefix {prefix_rank}, ambient "
                f"{row['ambient_dim']})",
                seed=seed,
            )
        boundary = _as_matrix(row["boundary"], label="boundary", seed=seed)
        support = _as_matrix(
            row["support_basis"], label="support_basis", seed=seed
        )
        certificate = float(np.linalg.norm(boundary @ observations))
        _require_close(
            certificate,
            row["certificate_residual"],
            label="null certificate residual",
            seed=seed,
        )
        annihilation = float(np.linalg.norm(boundary @ support))
        orthonormality = float(
            np.linalg.norm(support.T @ support - np.eye(support.shape[1]))
        )
        if annihilation > CERT_TOL or orthonormality > CERT_TOL:
            raise DesignFailureError(
                f"seed {seed}: retained false-discovery matrices fail "
                f"certification (annihilation {annihilation:.3e}, "
                f"orthonormality {orthonormality:.3e}, tol {CERT_TOL})",
                seed=seed,
            )
        entry.update(
            certificate_residual=certificate,
            annihilation_residual=annihilation,
            support_orthonormality=orthonormality,
            discovered_dim=int(support.shape[1]),
        )
    else:
        if row["certificate_residual"] is not None:
            consistent = (
                prefix_rank == rank
                and 0 < rank < row["ambient_dim"]
                and row["certificate_residual"] > CERT_TOL
            )
        else:
            consistent = (
                prefix_rank != rank
                or rank == 0
                or rank == row["ambient_dim"]
            )
        if not consistent:
            raise DesignFailureError(
                f"seed {seed}: retained null observations are "
                f"inconsistent with the recorded refusal (rank {rank}, "
                f"prefix {prefix_rank}, ambient {row['ambient_dim']})",
                seed=seed,
            )
        entry.update(observed_rank=rank, prefix_rank=prefix_rank)
    return entry


def _audit_block(
    eligibility: dict[str, Any],
    structured_rows: list[dict[str, Any]],
    null_rows: list[dict[str, Any]],
    *,
    complete: bool,
) -> dict[str, Any]:
    """Eligibility counts plus raw-row-recomputable certificates and claims."""
    block: dict[str, Any] = {
        "recompute_scope": (
            "(a) every claim statistic and decision below is recomputed "
            "from the raw rows alone: the refusal-to-zero coverage "
            "convention, the indicator recomputation against the frozen "
            "thresholds (map misfit at 1e-9, certificate at 1e-10), the "
            "means, standard deviations, SEs, one-sided lower bounds, and "
            "decisions; the eligibility counts and seed ids come from the "
            "eligibility pass, not from the raw rows. (b) the per-row "
            "float quantities are recomputed from the retained matrices "
            "plus the deterministic, manifest-pinned generator rebuild — "
            "without re-running the discovery SVD: the certificate "
            "residual (retained boundary x retained observations), the "
            "boundary/support annihilation and support orthonormality, "
            "the planted map's misfit (retained boundary x retained "
            "transported certified source-cycle image), the coverage "
            "column count (retained support basis), and the null "
            "dimension trajectory (retained null matrix); every "
            "observation matrix is re-derived from the frozen seed "
            "schedules (synthesize_observations for structured rows, the "
            "discovery-null subseed schedule for null rows) and required "
            "bit-equal to the retained matrices, and for discovered "
            "structured rows admission and quality are re-run from the "
            "reconstructed DiscoveredConstraint against the rebuilt "
            "instance's decoy library and true boundary. Unlike the "
            "router experiments, no hash-pinned model exists or is "
            "needed, because nothing was trained. A certification "
            "refusal's recorded residual is module-computed and NOT "
            "row-recomputable (a refused boundary is never returned); "
            "the audit verifies the rank configuration is consistent "
            "with the recorded refusal class instead, under the "
            "operators.py tolerance convention duplicated in this runner."
        ),
        "declared_seeds": eligibility["declared"],
        "eligible_seeds": eligibility["eligible"],
        "eligible_seed_ids": list(eligibility["eligible_seeds"]),
        "ineligible_seed_rows": len(eligibility["ineligible"]),
        "build_failure_rows": len(eligibility["build_failures"]),
        "structured_rows": len(structured_rows),
        "null_rows": len(null_rows),
        "rows_per_seed_per_condition": 1,
    }
    if complete:
        block["matrix_recompute"] = [
            *(_audit_structured_row_recompute(row) for row in structured_rows),
            *(_audit_null_row_recompute(row) for row in null_rows),
        ]
        paired = _recompute_paired(structured_rows, null_rows)
        block["per_seed"] = paired
        claims: dict[str, Any] = {}
        for definition in _CLAIM_DEFINITIONS:
            values = [row[definition["value"]] for row in paired]
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
            "universa.generators.make_switch_instance(num_vertices=8, "
            "num_edges=14, num_classes=6, num_decoys=3); the true target "
            "is withheld from the library (decoys only) and must be "
            "discovered"
        ),
        "observation_model": (
            "the vector-observation regime of universa.discovery: M = 16 "
            "transported vectors y_j = f1 a_j over certified random source "
            "cycles, synthesized deterministically from the seed by "
            "universa.discovery.synthesize_observations"
        ),
        "procedure": {
            "structured_call": (
                "universa.discovery.run_discovery(seed, "
                "num_observations=16, num_vertices=8, num_edges=14, "
                "num_classes=6, num_decoys=3, misfit_tol=1e-10, "
                "novelty_tol=1e-6)"
            ),
            "null_call": (
                "universa.discovery.discover_constraint(Y_null, "
                "ambient_dim, seeds=(seed,)) at the module-default tol = "
                "CERT_TOL = 1e-10"
            ),
            "num_observations": NUM_OBSERVATIONS,
            "misfit_tol": MISFIT_TOL,
            "novelty_tol": NOVELTY_TOL,
            "map_misfit_tol": MAP_MISFIT_TOL,
            "stability_fraction": STABILITY_FRACTION,
            "min_observation_norm": MIN_OBSERVATION_NORM,
            "stability_rule": (
                "the numerical rank of the first M - max(1, floor(M/4)) "
                "observations must equal the rank of all M (holdout 4 of "
                "16)"
            ),
        },
        "conditions": {
            "structured": (
                "one universa.discovery.run_discovery row per eligible "
                "seed (the H1-H3 statistics), retaining the observation "
                "matrix, the transported certified source-cycle image, "
                "and — when discovery succeeds — the discovered boundary "
                "and support basis"
            ),
            "null": (
                "one structure-free row per eligible seed (the H4 "
                "statistic): column j drawn from "
                "np.random.default_rng(subseed(seed, 'discovery-null', "
                "str(j))) on the target edge space, fed through "
                "discover_constraint directly, retaining the dimension "
                "trajectory (16 ranks) and the null matrix"
            ),
        },
        "train_seed_block": dict(TRAIN_SEED_BLOCK),
        "training": (
            "no training provenance exists because no training occurred: "
            "the discovery head has no learned component; the train block "
            "is the documented-empty sentinel {'first': 0, 'last': 0}"
        ),
        "eval_seeds": {
            "first": SEALED_EVAL_SEEDS[0],
            "last": SEALED_EVAL_SEEDS[-1],
        },
        "determinism": (
            "the full per-seed pipeline — both conditions — is executed "
            "twice and required bit-identical; any mismatch is a whole-run "
            "design_failure"
        ),
        "inference_unit": "the generator seed",
        "estimand": (
            "per-seed statistics from the two conditions: coverage "
            "(quality.coverage_fraction, 0.0 on refusal), cert (success "
            "and certificate_residual <= 1e-10, mechanically the success "
            "rate), misfit (success and map_misfit <= 1e-9) from the "
            "structured condition; refusal (DiscoveryInsufficient "
            "returned) from the null condition; each claim is the mean "
            "of its per-seed statistic over the eligible seeds"
        ),
        "familywise_alpha": FAMILYWISE_ALPHA,
        "per_claim_alpha": PER_CLAIM_ALPHA,
        "minimum_eligible": MIN_ELIGIBLE,
        "eligibility_rule": (
            "the instance builds: make_switch_instance(seed, 8, 14, 6, 3) "
            "returns; a build exception is a whole-run design_failure, "
            "never an exclusion; there are no per-seed audit gates beyond "
            "the build"
        ),
        "row_count_invariant": (
            "2 x n: exactly one structured row and one null row per "
            "eligible seed (72 rows over the declared block when every "
            "declared seed is eligible), checked before any summary"
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
    eligible: list[tuple[int, Any]] = []
    build_failure: dict[str, Any] | None = None
    eligibility_failure: tuple[str, BaseException] | None = None
    try:
        for seed in SEALED_EVAL_SEEDS:
            try:
                instance = make_switch_instance(
                    seed, NUM_VERTICES, NUM_EDGES, NUM_CLASSES, NUM_DECOYS
                )
            except Exception as error:  # a build exception stops the whole run
                build_failure = {
                    "seed": seed,
                    "type": type(error).__name__,
                    "message": str(error),
                }
                break
            seed_records.append(
                {"seed": seed, "eligible": True, "reason": None}
            )
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
        base["structured_rows"] = []
        base["null_rows"] = []
        base["paired"] = []
        base["claims"] = _null_claims()
        base["audit"] = _audit_block(eligibility, [], [], complete=False)
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
        base["structured_rows"] = []
        base["null_rows"] = []
        base["paired"] = []
        base["claims"] = _null_claims()
        base["audit"] = _audit_block(eligibility, [], [], complete=False)
        return base
    if len(eligible) < MIN_ELIGIBLE:
        base["status"] = "design_failure_insufficient_eligible"
        base["stop_condition"] = (
            f"only {len(eligible)} seeds were eligible; minimum is "
            f"{MIN_ELIGIBLE}; no rows were built and no claims were decided"
        )
        base["structured_rows"] = []
        base["null_rows"] = []
        base["paired"] = []
        base["claims"] = _null_claims()
        base["audit"] = _audit_block(eligibility, [], [], complete=False)
        return base

    structured_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    failure: tuple[str, BaseException] | None = None
    try:
        _campaign_rows(eligible, structured_rows, null_rows)
        # Fail-closed row-count invariant (2 x n): every eligible seed must
        # have contributed exactly one row per condition before any summary.
        expected_rows = len(eligible)
        if (
            len(structured_rows) != expected_rows
            or len(null_rows) != expected_rows
        ):
            raise DesignFailureError(
                f"row-count invariant violated: {len(structured_rows)} "
                f"structured rows and {len(null_rows)} null rows, expected "
                f"{expected_rows} of each (2 x n: one row per condition "
                "per eligible seed; 72 rows over the declared block when "
                "every declared seed is eligible)"
            )
        # Summaries are computed only after every required raw row has
        # completed (frozen stop rule: no interim analysis).
        paired = _recompute_paired(structured_rows, null_rows)
        base["claims"] = _claim_inference(paired)
    except DesignFailureError as error:
        failure = ("design_failure", error)
    except KeyboardInterrupt as error:
        failure = ("interrupted", error)
    except Exception as error:  # unexpected faults are preserved
        failure = ("execution_failure", error)

    base["structured_rows"] = structured_rows
    base["null_rows"] = null_rows
    if failure is not None:
        # A mid-seed stop can leave a structured row whose null row was
        # never built; the paired view then covers only the seeds complete
        # in both conditions (every completed raw row is preserved
        # regardless).
        paired_seeds = {row["seed"] for row in null_rows}
        base["paired"] = _recompute_paired(
            [row for row in structured_rows if row["seed"] in paired_seeds],
            null_rows,
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
            eligibility, structured_rows, null_rows, complete=False
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
    base["paired"] = _recompute_paired(structured_rows, null_rows)
    base["audit"] = _audit_block(
        eligibility, structured_rows, null_rows, complete=True
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
