#!/usr/bin/env python3
"""Sealed runner for experiment ``universa-router-sheaf-sealed-1``.

The frozen design: one :class:`universa.router.StructureRouter`
(``feature_dim=18``, ``hidden_dim=64``) trained on the 18-dim NO-ANCHOR
degradation profiles of cellular sheaf switch instances over the train seed
block 11001..11200 x operating fractions {0.2, 0.3, 0.4, 0.5} (one
replicate, index 0, per row; 800 rows), then compared on the sealed eval
seed block 20101..20136 x held-out fractions {0.6, 0.7, 0.8} against TWO
baselines: the polluted argmin observed-residual oracle (argmin of the
operating-fraction observed naturality-residual column) and, descriptively,
the non-learned grid-mean ``log1p`` naturality-profile heuristic. Every
observation is a :class:`universa.partial_sheaf.SheafObservationModel` draw
with ``mask_fraction = 0.25`` (edge restriction blocks removed, the
surviving coboundary losing their block rows) AND sign corruption
(``corrupt_fraction = g`` at grid point ``g``); the profile grid 0.2..0.9 by
0.1 EXCLUDES 0.0 — no clean column exists anywhere. Each data row (seed,
fraction, replicate) gets ``observation_seed = subseed(seed,
"sealed-replicate", str(fraction), str(replicate_index))``, driving the
mask draw AND the nested corruption draws (component ``sheaf-observe``
with keys ``mask`` and ``corrupt``); the SAME seed feeds all arms within a row
(exactly paired), with R=4 replicates per eval (seed, fraction). The
per-(seed, fraction) candidate permutation is
``subseed(seed, "router-v2-permutation", f"{fraction:.6f}")`` (the
component literal retained verbatim from the v2 schedule as a frozen string
constant), shared across the row's replicates, with audited labels. The
generator seed is the inference unit for the four one-sided Bonferroni
claims (per-claim alpha 0.05/4 = 0.0125).

**Retention (frozen).** Every eval raw row retains, per candidate, the FULL
naturality-residual profile — the planted morphism's observed naturality
residual at all 8 grid points, 8 raw floats in permuted candidate order —
plus the operating-fraction residual. The oracle arm's per-row bits are
recomputable from the retained operating residuals and the heuristic arm's
per-row bits from the retained full profiles: BOTH non-learned arms
recompute from the raw rows alone, and the audit block does exactly that.
Only the learned arm needs the retained model (hash-pinned in
``training.model_state_sha256``).

**Declared caveat (frozen).** The comparison is asymmetric by design: the
learned router integrates the full degraded naturality profile; the oracle
reads only one polluted column (the operating fraction's observed
naturality residual). The heuristic baseline in H4 is the profile-shape
non-learned comparator. Demo-scale numbers are not evidence.

This runner must not be executed on the sealed eval seed block until the
protocol, this runner, and the seal record
(``docs/14-router-sheaf-seal.json``) have been committed and pushed.
It keeps the seed block inert at import time: no instance is constructed
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
from universa.generators import subseed
from universa.partial_sheaf import (
    SheafObservationModel,
    _sheaf_from_coboundary,
    observed_naturality_residual,
)
from universa.router import RESIDUAL_TOL, hard_predictions, train_router
from universa.sheaves import (
    Sheaf,
    SheafMorphism,
    make_sheaf_switch_instance,
    planted_morphism,
    random_sheaf,
)
from universa.structures import ChainMap

EXPERIMENT_ID = "universa-router-sheaf-sealed-1"
RESULT_SCHEMA = "universa-router-sheaf-sealed-result/1"

# The one canonical execution command, recorded verbatim in every result.
CANONICAL_COMMAND = (
    "env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src python "
    "scripts/run_sheaf_sealed_1.py --output "
    "results/experiments/router-sheaf-sealed-1.json"
)

PROTOCOL = "docs/13-sealed-router-sheaf-protocol.md"
RUNNER_SOURCE = "scripts/run_sheaf_sealed_1.py"
DEFAULT_SEAL = "docs/14-router-sheaf-seal.json"
SEAL_SCHEMA = "universa-seal/5"

PROTOCOL_SHA256 = (
    "7b8434509864a294274620a10cd7e7ab2c7dd2743f1dd8ebc0b4929c00e6ced3"
)
"""Frozen fingerprint of the sealed protocol.

The committed runner embeds the sealed protocol's SHA-256 as this constant;
the value must equal the seal record's ``protocol_sha256``. Any protocol edit
after the seal requires re-pinning this constant (and a new seal record) —
the fail-closed refusal in :func:`_frozen_protocol_sha256` then catches the
mismatch. While the value is the placeholder ``PENDING_PROTOCOL_SHA256`` the
runner refuses every execution.
"""

# Frozen task family: cellular sheaf switch instances, K = 4 candidates with
# the true target at index 0 pre-permutation.
NUM_VERTICES = 6
NUM_EDGES = 9
MAX_STALK_DIM = 3
NUM_DECOYS = 3
NUM_CANDIDATES = 1 + NUM_DECOYS

TRAIN_SEEDS = tuple(range(11001, 11201))  # 11001..11200
SEALED_EVAL_SEEDS = tuple(range(20101, 20137))  # 20101..20136
TRAIN_FRACTIONS = (0.2, 0.3, 0.4, 0.5)
HELD_OUT_FRACTIONS = (0.6, 0.7, 0.8)
EVAL_FRACTIONS = HELD_OUT_FRACTIONS  # no clean anchor exists in this regime
PROFILE_GRID = tuple((i + 2) / 10 for i in range(8))  # 0.2..0.9, excludes 0.0
MASK_FRACTION = 0.25
REPLICATES = 4
TRAIN_REPLICATE_INDEX = 0  # one replicate per train (seed, fraction)

FEATURE_DIM = 18  # 8 naturality profile + 7 slopes + 3 masked structural dims
HIDDEN_DIM = 64
EPOCHS = 150
LEARNING_RATE = 1e-3
TORCH_SEED = 4242
LAMBDA_AUX = 0.01
TAU_START = 2.0
TAU_END = 0.25

FAMILYWISE_ALPHA = 0.05
NUM_CLAIMS = 4
PER_CLAIM_ALPHA = FAMILYWISE_ALPHA / NUM_CLAIMS  # 0.0125
MIN_ELIGIBLE = 30
AUDIT_TOL = RESIDUAL_TOL  # 1e-9

# One-sided Bonferroni Student-t critical values t.ppf(1 - 0.05/4, n - 1) for
# the eligible n, computed once at design time with scipy:
#     from scipy.stats import t
#     {n: t.ppf(1 - 0.05 / 4, n - 1) for n in range(30, 37)}
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
# ``baseline`` selects the paired difference the claim is decided on:
# ``d_oracle`` (learned minus polluted argmin observed-residual oracle) for
# H1..H3, ``d_heur`` (learned minus the non-learned grid-mean log1p
# naturality-profile heuristic) for H4.
_CLAIM_DEFINITIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "h1-sheaf-0.7-primary",
        "fraction": 0.7,
        "threshold": 0.0,
        "role": "primary",
        "baseline": "oracle",
        "difference": "d_oracle",
        "description": (
            "the learned router beats the polluted argmin "
            "naturality-residual oracle at the held-out fraction 0.7"
        ),
    },
    {
        "id": "h2-sheaf-0.6",
        "fraction": 0.6,
        "threshold": 0.0,
        "role": "secondary",
        "baseline": "oracle",
        "difference": "d_oracle",
        "description": (
            "the learned router beats the polluted argmin "
            "naturality-residual oracle at the held-out fraction 0.6"
        ),
    },
    {
        "id": "h3-sheaf-0.8",
        "fraction": 0.8,
        "threshold": 0.0,
        "role": "secondary",
        "baseline": "oracle",
        "difference": "d_oracle",
        "description": (
            "the learned router beats the polluted argmin "
            "naturality-residual oracle at the held-out fraction 0.8"
        ),
    },
    {
        "id": "h4-sheaf-heuristic-0.7",
        "fraction": 0.7,
        "threshold": 0.0,
        "role": "secondary",
        "baseline": "heuristic",
        "difference": "d_heur",
        "description": (
            "the learned router beats the non-learned grid-mean "
            "naturality-residual profile heuristic at the held-out "
            "fraction 0.7"
        ),
    },
)
CLAIM_IDS = tuple(definition["id"] for definition in _CLAIM_DEFINITIONS)

# The subfields the seal record must carry for each frozen claim (id,
# fraction, theta, null, alternative, reference, bound direction, threshold,
# support rule). A claim object must carry exactly these keys, plus the
# optional ``role``; any other key is refused.
_CLAIM_SEAL_KEYS = (
    "id",
    "fraction",
    "theta",
    "null",
    "alternative",
    "reference",
    "bound_direction",
    "threshold",
    "support_rule",
)

DECLARED_CAVEAT = (
    "the comparison is asymmetric by design: the learned router integrates "
    "the full degraded naturality profile; the oracle reads only one "
    "polluted column (the operating fraction's observed naturality "
    "residual). The heuristic baseline in H4 is the profile-shape "
    "non-learned comparator. Demo-scale numbers are not evidence."
)


class DesignFailureError(RuntimeError):
    """A frozen-design validation failure: the whole run is a design failure."""

    def __init__(self, message: str, *, seed: int | None = None) -> None:
        super().__init__(message)
        self.seed = seed


@dataclass(frozen=True)
class _ReplicateRow:
    """One (seed, fraction, replicate) observation draw, all arms' input.

    ``features`` is the permuted (K, FEATURE_DIM) float64 no-anchor
    degradation-profile block exactly as :func:`sheaf_candidate_features`
    computes it (per-candidate ``log1p`` of the observed naturality residual
    over the 0.2..0.9 grid under mask_fraction 0.25 and sign corruption,
    first-difference slopes, masked structural dims). ``residual_profiles``
    holds, per candidate, the raw naturality-residual profile (the residual
    at all grid points, 8 floats, permuted order); ``operating_residuals``
    the per-candidate residual at the operating fraction — the column the
    polluted oracle reads. All derive from ONE shared observation draw (the
    same seed drives the mask draw and the nested corruption draws), so the
    arms are exactly paired.
    """

    seed: int
    fraction: float
    replicate: int
    observation_seed: int
    features: np.ndarray
    true_index: int
    permutation: tuple[int, ...]
    residual_profiles: tuple[tuple[float, ...], ...] | None
    operating_residuals: tuple[float, ...] | None


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
            for key in ("theta", "null", "alternative", "reference", "support_rule")
        ):
            raise RuntimeError(
                "stop condition: design seal primary_family entry "
                f"{claim.get('id')!r} must carry the descriptive subfields "
                "(theta, null, alternative, reference, support_rule) as "
                "nonempty strings"
            )
    if tuple(claim["id"] for claim in family) != CLAIM_IDS:
        raise RuntimeError(
            "stop condition: design seal primary_family claim ids must be "
            f"exactly {list(CLAIM_IDS)} in order"
        )
    for claim, definition in zip(family, _CLAIM_DEFINITIONS):
        if (
            claim["fraction"] != definition["fraction"]
            or claim["threshold"] != definition["threshold"]
            or claim["bound_direction"] != "greater"
            or ("role" in claim and claim["role"] != definition["role"])
        ):
            raise RuntimeError(
                "stop condition: design seal primary_family claim "
                f"{claim['id']!r} does not match the frozen definition "
                "(fraction, threshold, bound_direction, role)"
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
    """Pin single-thread CPU float32 execution and refuse a visible CUDA."""

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
    return {
        "tensor_device": "cpu",
        "tensor_dtype": "float32",
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

    # 7. Environment: CUDA hidden, one torch thread, CPU float32.
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
# Instance construction, feature machinery, audits, and replicate rows.


def _replicate_observation_seed(seed: int, fraction: float, replicate: int) -> int:
    """The frozen replicate draw, SHARED across all arms within a row.

    One observation seed per data row (seed, fraction, replicate); it drives
    the mask draw AND the nested corruption draws (``sheaf-observe``,
    ``mask`` | ``corrupt``) of every
    :class:`universa.partial_sheaf.SheafObservationModel` the row constructs.
    """
    if replicate < 0:
        raise ValueError("replicate must be nonnegative")
    return subseed(seed, "sealed-replicate", str(fraction), str(replicate))


def _fraction_key(fraction: float) -> str:
    """The frozen permutation subseed component: ``f"{fraction:.6f}"``."""
    return f"{fraction:.6f}"


def _validated_profile_grid(profile_grid: Any) -> tuple[float, ...]:
    """Fail-closed no-anchor grid validation: nonempty, inside ``(0, 1]``,
    strictly increasing, and excluding 0.0 — the defining constraint of the
    regime is that no grid point is the exact (undegraded) observation, so a
    grid containing 0.0 is an error, never a warning."""
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


def _grid_index(fraction: float) -> int:
    """The profile-grid column of an operating fraction (fail-closed)."""
    matches = [
        index
        for index, point in enumerate(PROFILE_GRID)
        if abs(point - fraction) <= 1e-9
    ]
    if not matches:
        raise DesignFailureError(
            f"fraction {fraction} is not a profile grid point"
        )
    return matches[0]


def _feature_names(profile_grid: Any = PROFILE_GRID) -> tuple[str, ...]:
    """Per-candidate feature layout, in column order (float64 at build).

    For a grid of ``G`` fractions the layout is ``G + (G - 1) + 3`` columns:
    ``log1p`` of the observed naturality residual at every grid point, then
    the first-difference slopes (named by the grid fractions they span),
    then the masked structural dims — ``num_base_vertices``, the kept edge
    count, and ``c0_dim`` (the total vertex-stalk dimension) — read off the
    masked (uncorrupted) observation.
    """
    grid = _validated_profile_grid(profile_grid)
    naturality = tuple(
        f"log1p_observed_naturality_residual_fraction_{g:.1f}" for g in grid
    )
    slopes = tuple(
        f"naturality_profile_slope_{a:.1f}_to_{b:.1f}"
        for a, b in zip(grid, grid[1:])
    )
    structural = (
        "num_base_vertices",
        "kept_edge_count",
        "c0_dim",
    )
    return naturality + slopes + structural


def sheaf_candidate_features(
    morphism: SheafMorphism,
    candidate: Sheaf,
    observation_seed: int,
    profile_grid: Any = PROFILE_GRID,
    mask_fraction: float = MASK_FRACTION,
) -> tuple[np.ndarray, tuple[float, ...]]:
    """No-anchor naturality-profile feature vector for one sheaf candidate.

    Returns ``(features, residuals)``: the FEATURE_DIM-wide float64 feature
    vector laid out as :func:`_feature_names` plus the RAW per-grid-point
    observed naturality residuals, so the row builder can retain the full
    profile (the retention contract: both non-learned arms recompute from
    raw rows alone). The feature profile itself is ``log1p`` of the
    residual per grid point — the frozen score of the protocol.

    The candidate is observed through
    :class:`universa.partial_sheaf.SheafObservationModel` draws with BOTH
    ``mask_fraction`` (edge restriction blocks removed) and
    ``corrupt_fraction = g`` for every grid fraction ``g``, under the shared
    ``observation_seed`` — one draw per (row, grid point), reused across
    candidates. The mask draw is fixed per row (constant ``mask_fraction``,
    seed-derived permutation), so the same edges are missing at every grid
    point and the corruption sweep is nested per fraction (prefixes of one
    master permutation over the surviving nonzero entries). Every profile
    entry is ``log1p`` of the
    :func:`universa.partial_sheaf.observed_naturality_residual` of the
    planted morphism against the observed candidate — polluted at every
    grid point, with no exact anchor anywhere.

    The structural dims (``num_base_vertices``, kept edge count,
    ``c0_dim`` — the total vertex-stalk dimension) are read off the MASKED
    observed sheaf (no corruption, the row-fixed degraded operator): every
    feature is computed against a degraded operator.
    """
    grid = _validated_profile_grid(profile_grid)
    if not isinstance(morphism, SheafMorphism):
        raise ValueError("morphism must be a SheafMorphism instance")
    if not isinstance(candidate, Sheaf):
        raise ValueError("routing candidates are cellular sheaves")
    if not 0.0 < mask_fraction <= 1.0:
        raise ValueError("mask_fraction must lie in (0, 1]")
    residuals = tuple(
        float(
            observed_naturality_residual(
                morphism,
                SheafObservationModel(
                    candidate,
                    observation_seed,
                    mask_fraction=mask_fraction,
                    corrupt_fraction=g,
                ).observe(),
            )
        )
        for g in grid
    )
    profile = np.log1p(np.asarray(residuals, dtype=np.float64))
    slopes = np.diff(profile)
    masked = SheafObservationModel(
        candidate, observation_seed, mask_fraction=mask_fraction
    ).observe()
    features = np.concatenate(
        [
            profile,
            slopes,
            np.array(
                [
                    float(masked.num_vertices),
                    float(len(masked.kept_edges)),
                    float(masked.c0_dim),
                ],
                dtype=np.float64,
            ),
        ]
    )
    return features, residuals


def _instance_sheaves(instance: Any) -> tuple[SheafMorphism, tuple[Sheaf, ...]]:
    """The planted morphism and per-candidate sheaves of one instance.

    Regenerates the source sheaf and replants the morphism by the same
    deterministic draws :func:`universa.sheaves.make_sheaf_switch_instance`
    compiled, then reads every compiled candidate back as a sheaf block by
    block from its coboundary — the certified recovery of
    :func:`universa.partial_sheaf.ranking_study_sheaf`. Fail-closed: a
    replanted morphism that disagrees with the instance is a design
    failure, never an assumption left unchecked.
    """
    source = random_sheaf(instance.seed, NUM_VERTICES, NUM_EDGES, MAX_STALK_DIM)
    _, morphism = planted_morphism(source, instance.seed)
    compiled = morphism.to_chain_map()
    if not (
        np.array_equal(
            compiled.source.boundaries[0], instance.source.boundaries[0]
        )
        and np.array_equal(
            compiled.target.boundaries[0], instance.true_target.boundaries[0]
        )
        and len(compiled.maps) == len(instance.chain_map.maps)
        and all(
            np.array_equal(drawn, planted)
            for drawn, planted in zip(compiled.maps, instance.chain_map.maps)
        )
    ):
        raise DesignFailureError(
            f"seed {instance.seed}: the replanted morphism disagrees with "
            "the compiled instance",
            seed=instance.seed,
        )
    candidates = tuple(
        _sheaf_from_coboundary(
            source.base,
            source.vertex_dims,
            source.edge_dims,
            candidate.boundaries[0],
        )
        for candidate in instance.candidates
    )
    return morphism, candidates


def _undegraded_misfits(instance: Any) -> tuple[float, ...]:
    """Per-candidate clean commutation residual of the compiled morphism.

    The sole eligibility audit (the undegraded naturality audit): by the
    documented :meth:`universa.sheaves.SheafMorphism.to_chain_map` equality
    the compiled commutation residual equals the root-sum-square of the
    per-incidence clean naturality residuals. Deterministic per seed,
    computed once per seed, purely as bookkeeping for seed accounting and
    label certification — it never enters features (every feature is
    computed against a degraded operator).
    """
    misfits = []
    for candidate in instance.candidates:
        residuals = ChainMap(
            instance.source, candidate, instance.chain_map.maps
        ).commutation_residuals()
        if len(residuals) != 1:
            raise DesignFailureError(
                f"seed {instance.seed}: expected 1 commutation residual "
                f"(a compiled sheaf), got {len(residuals)}",
                seed=instance.seed,
            )
        misfits.append(float(residuals[0]))
    return tuple(misfits)


def _build_replicate_row(
    instance: Any,
    seed: int,
    fraction: float,
    replicate: int,
    *,
    with_residuals: bool,
) -> _ReplicateRow:
    """One paired replicate row: features for the learned arm, the retained
    naturality-residual profile for the heuristic, and the
    operating-fraction residuals for the oracle, from ONE shared
    observation draw.

    The feature block is computed exactly as
    :func:`sheaf_candidate_features` computes it — per candidate under the
    row's shared observation seed with ``mask_fraction = 0.25`` and the
    nested corruption sweep over the 0.2..0.9 grid — permuted by the
    per-(seed, fraction) permutation from
    ``subseed(seed, "router-v2-permutation", f"{fraction:.6f}")``
    (shared across the row's replicates). The eligibility audit is per-seed
    on the UNDEGRADED instance (see :func:`_undegraded_misfits`) and is not
    repeated here; the row-level fail-closed checks are candidate count,
    grid membership, block shape, and finiteness.
    """
    candidates = instance.candidates
    if len(candidates) != NUM_CANDIDATES:
        raise DesignFailureError(
            f"seed {seed}: expected {NUM_CANDIDATES} candidates, got "
            f"{len(candidates)}",
            seed=seed,
        )
    if not any(abs(g - fraction) <= 1e-9 for g in PROFILE_GRID):
        raise DesignFailureError(
            f"seed {seed}: fraction {fraction} is not a profile grid point",
            seed=seed,
        )
    observation_seed = _replicate_observation_seed(seed, fraction, replicate)
    try:
        morphism, sheaves = _instance_sheaves(instance)
        per_candidate = [
            sheaf_candidate_features(
                morphism,
                candidate,
                observation_seed,
                PROFILE_GRID,
                MASK_FRACTION,
            )
            for candidate in sheaves
        ]
    except DesignFailureError:
        raise
    except Exception as error:
        # A violation inside the certified feature machinery is a
        # whole-run design failure, never an execution failure.
        raise DesignFailureError(
            f"seed {seed}, fraction {fraction}: certified feature "
            f"construction failed: {type(error).__name__}: {error}",
            seed=seed,
        ) from error
    rng = np.random.default_rng(
        subseed(seed, "router-v2-permutation", _fraction_key(fraction))
    )
    permutation = tuple(int(p) for p in rng.permutation(NUM_CANDIDATES))
    true_index = permutation.index(0)
    block = np.stack([per_candidate[p][0] for p in permutation])
    if block.shape != (NUM_CANDIDATES, FEATURE_DIM):
        raise DesignFailureError(
            f"seed {seed}, fraction {fraction}: feature block has shape "
            f"{block.shape}, expected {(NUM_CANDIDATES, FEATURE_DIM)}",
            seed=seed,
        )
    if not np.isfinite(block).all():
        raise DesignFailureError(
            f"seed {seed}, fraction {fraction}: non-finite features",
            seed=seed,
        )
    residual_profiles = None
    operating_residuals = None
    if with_residuals:
        grid_index = _grid_index(fraction)
        residual_profiles = tuple(
            per_candidate[p][1] for p in permutation
        )
        operating_residuals = tuple(
            float(per_candidate[p][1][grid_index]) for p in permutation
        )
    return _ReplicateRow(
        seed=seed,
        fraction=fraction,
        replicate=replicate,
        observation_seed=observation_seed,
        features=block,
        true_index=true_index,
        permutation=permutation,
        residual_profiles=residual_profiles,
        operating_residuals=operating_residuals,
    )


def _build_train_block() -> tuple[np.ndarray, np.ndarray]:
    """The frozen train block: one row per (seed, fraction), 800 rows.

    Built by the runner itself with the frozen row convention — one
    replicate (index 0) per (seed, fraction) with the sealed-replicate
    observation draw and the per-(seed, fraction) permutation. The per-seed
    eligibility audit on the UNDEGRADED instance applies to every train
    seed; any violation is a whole-run design failure, never a dropped row.
    """
    features: list[np.ndarray] = []
    labels: list[int] = []
    for seed in TRAIN_SEEDS:
        try:
            instance = make_sheaf_switch_instance(
                seed, NUM_VERTICES, NUM_EDGES, MAX_STALK_DIM, NUM_DECOYS
            )
        except Exception as error:
            raise DesignFailureError(
                f"train block construction failed at seed {seed}: "
                f"{type(error).__name__}: {error}"
            ) from error
        misfits = _undegraded_misfits(instance)
        if misfits[0] > AUDIT_TOL:
            raise DesignFailureError(
                f"train seed {seed}: true candidate misfit "
                f"{misfits[0]:.3e} on the undegraded instance exceeds "
                f"{AUDIT_TOL}"
            )
        decoy_floor = min(misfits[1:])
        if decoy_floor <= AUDIT_TOL:
            raise DesignFailureError(
                f"train seed {seed}: a decoy misfit {decoy_floor:.3e} on "
                f"the undegraded instance is at or below {AUDIT_TOL}; "
                "ground truth is not separable"
            )
        for fraction in TRAIN_FRACTIONS:
            row = _build_replicate_row(
                instance,
                seed,
                fraction,
                TRAIN_REPLICATE_INDEX,
                with_residuals=False,
            )
            features.append(row.features)
            labels.append(row.true_index)
    block = np.stack(features)
    label_array = np.asarray(labels, dtype=np.int64)
    expected_rows = len(TRAIN_SEEDS) * len(TRAIN_FRACTIONS)
    if block.shape != (expected_rows, NUM_CANDIDATES, FEATURE_DIM):
        raise DesignFailureError(
            f"train block has shape {block.shape}, expected "
            f"{(expected_rows, NUM_CANDIDATES, FEATURE_DIM)}"
        )
    return block, label_array


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


def _train_model(
    features: np.ndarray, labels: np.ndarray
) -> tuple[Any, dict[str, Any]]:
    """Train the one frozen StructureRouter and record training provenance."""
    # The frozen design defines no validation split; the train block itself
    # is passed as the validation monitor (history-only — it cannot affect
    # the fitted parameters), so only train seeds are ever seen.
    dataset = (features, labels, ())
    try:
        model, history = train_router(
            dataset,
            dataset,
            EPOCHS,
            lr=LEARNING_RATE,
            seed=TORCH_SEED,
            lambda_aux=LAMBDA_AUX,
            tau_start=TAU_START,
            tau_end=TAU_END,
            hidden_dim=HIDDEN_DIM,
            feature_dim=FEATURE_DIM,
        )
    except Exception as error:
        # A violation inside the certified training machinery (e.g. a
        # non-finite loss) is a whole-run design failure, never an
        # execution failure — the same classification _build_train_block
        # applies.
        raise DesignFailureError(
            f"the frozen training call failed: {type(error).__name__}: "
            f"{error}"
        ) from error
    provenance = {
        "torch_seed": TORCH_SEED,
        "epochs": EPOCHS,
        "lr": LEARNING_RATE,
        "lambda_aux": LAMBDA_AUX,
        "tau_start": TAU_START,
        "tau_end": TAU_END,
        "hidden_dim": HIDDEN_DIM,
        "feature_dim": FEATURE_DIM,
        "optimizer": "full-batch Adam",
        "dtype_device": "CPU float32",
        "num_train_rows": int(labels.shape[0]),
        "standardization": "input mean/std measured on the train block only",
        "validation_monitor": (
            "the train block itself; the frozen design defines no "
            "validation split"
        ),
        "final": {
            key: float(history[key][-1])
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
        "model_state_sha256": _model_state_sha256(model),
    }
    return model, provenance


if FEATURE_DIM != len(_feature_names(PROFILE_GRID)):
    raise RuntimeError(
        f"frozen FEATURE_DIM={FEATURE_DIM} disagrees with the runner's "
        f"feature layout: len(_feature_names(PROFILE_GRID))="
        f"{len(_feature_names(PROFILE_GRID))}"
    )


# ---------------------------------------------------------------------------
# Inference: per-seed paired differences and the four frozen claims.


def _oracle_index(operating_residuals: Any) -> int:
    """The polluted oracle's pick: argmin of the operating-fraction
    observed naturality-residual column (log1p is strictly increasing, so
    this is exactly argmin of the profile column the oracle reads)."""
    return int(np.argmin(np.asarray(operating_residuals, dtype=np.float64)))


def _heuristic_index(residual_profiles: Any) -> int:
    """The non-learned heuristic's pick: argmin over candidates of the
    uniform grid-mean of the ``log1p`` observed naturality-residual profile
    — the simplest whole-trajectory integration with no learning and no
    operating-fraction choice. Reads only the retained raw profiles (8
    floats per candidate)."""
    grid = len(PROFILE_GRID)
    scores = []
    for profile in residual_profiles:
        values = np.asarray(profile, dtype=np.float64)
        if values.shape != (grid,):
            raise DesignFailureError(
                f"a retained residual profile has width {values.shape}, "
                f"expected {(grid,)}"
            )
        scores.append(float(np.log1p(values).mean()))
    return int(np.argmin(np.asarray(scores, dtype=np.float64)))


def _score_row(
    model: Any,
    features: np.ndarray,
    residual_profiles: tuple[tuple[float, ...], ...],
    operating_residuals: tuple[float, ...],
    true_index: int,
) -> tuple[bool, bool, bool]:
    """Score one replicate row under all arms.

    Learned: strictly discrete hard argmax over candidate logits. Oracle:
    argmin over the per-candidate observed naturality residuals at the
    operating fraction (the row's profile column; log1p is strictly
    increasing, so this is exactly argmin of the misfit). Heuristic
    (descriptive, non-learned): argmin over candidates of the uniform
    grid-mean of the ``log1p`` naturality profile. All three read the same
    row, so the correctness triple is exactly paired.

    Returns ``(learned_correct, oracle_correct, heuristic_correct)``.
    """
    learned_index = int(hard_predictions(model, features[np.newaxis])[0])
    oracle_index = _oracle_index(operating_residuals)
    heuristic_index = _heuristic_index(residual_profiles)
    return (
        learned_index == true_index,
        oracle_index == true_index,
        heuristic_index == true_index,
    )


def _estimand(definition: dict[str, Any]) -> str:
    difference = definition["difference"]
    baseline_acc = (
        "oracle_acc" if definition["baseline"] == "oracle" else "heuristic_acc"
    )
    return (
        f"mean over eligible seeds of {difference}(seed, "
        f"{definition['fraction']}) with {difference} = learned_acc - "
        f"{baseline_acc}, accuracies over R={REPLICATES} shared replicate "
        "draws"
    )


def _claim_summary(
    definition: dict[str, Any], values: list[float]
) -> dict[str, Any]:
    """One-sided Student-t lower bound for one claim's per-seed differences."""
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
        "fraction": definition["fraction"],
        "baseline": definition["baseline"],
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


def _claim_inference(paired: list[dict[str, Any]]) -> dict[str, Any]:
    claims = [
        _claim_summary(
            definition,
            [
                row[definition["difference"]]
                for row in paired
                if row["fraction"] == definition["fraction"]
            ],
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
                "fraction": definition["fraction"],
                "baseline": definition["baseline"],
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


def _recompute_paired(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-(seed, fraction) accuracies and differences, from raw rows only.

    The oracle arm's bits are RECOMPUTED from the retained per-row
    operating-fraction residuals and the heuristic arm's bits from the
    retained full naturality-residual profiles (8 floats per candidate) —
    both non-learned arms recompute from the raw rows alone, and the
    recomputed bits must equal the stored ones (a mismatch is a design
    failure, never a warning). The learned arm's bits are taken from the
    stored row (recomputable only with the retained hash-pinned model).
    """
    groups: dict[tuple[int, float], list[dict[str, Any]]] = {}
    order: list[tuple[int, float]] = []
    for row in raw_rows:
        oracle_bit = _oracle_index(row["operating_residuals"]) == row["true_index"]
        heuristic_bit = (
            _heuristic_index(row["residual_profiles"]) == row["true_index"]
        )
        if oracle_bit != row["oracle_correct"] or heuristic_bit != row["heuristic_correct"]:
            raise DesignFailureError(
                "raw-row retention inconsistency at seed "
                f"{row['seed']}, fraction {row['fraction']}, replicate "
                f"{row['replicate_index']}: the stored arm bits are not "
                "recomputable from the retained residual profiles"
            )
        key = (row["seed"], row["fraction"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)
    recomputed = []
    for seed, fraction in order:
        rows = groups[(seed, fraction)]
        learned = sum(1 for row in rows if row["learned_correct"]) / len(rows)
        oracle = sum(1 for row in rows if row["oracle_correct"]) / len(rows)
        heuristic = sum(1 for row in rows if row["heuristic_correct"]) / len(rows)
        recomputed.append(
            {
                "seed": seed,
                "fraction": fraction,
                "replicates": len(rows),
                "learned_acc": learned,
                "oracle_acc": oracle,
                "heuristic_acc": heuristic,
                "d_oracle": learned - oracle,
                "d_heur": learned - heuristic,
            }
        )
    return recomputed


def _audit_block(
    eligibility: dict[str, Any],
    raw_rows: list[dict[str, Any]],
    *,
    complete: bool,
) -> dict[str, Any]:
    """Eligibility counts plus raw-row-recomputable claim aggregation."""
    block: dict[str, Any] = {
        "recompute_scope": (
            "the eligibility counts and seed ids come from the eligibility "
            "pass, not from the raw rows; the oracle arm's per-row bits are "
            "recomputed from the retained per-row operating-fraction "
            "observed naturality residuals; the heuristic arm's per-row "
            "bits are recomputed from the retained per-row full "
            "naturality-residual profiles (the residual at all 8 grid "
            "points, 8 floats per candidate) — BOTH non-learned arms "
            "recompute from the raw rows alone; the learned arm's per-row "
            "bits are recomputable only with the retained model "
            "(hash-pinned in training.model_state_sha256); this block "
            "recomputes the aggregation from the retained raw rows"
        ),
        "declared_seeds": eligibility["declared"],
        "eligible_seeds": eligibility["eligible"],
        "eligible_seed_ids": list(eligibility["eligible_seeds"]),
        "ineligible_seed_rows": len(eligibility["ineligible"]),
        "build_failure_rows": len(eligibility["build_failures"]),
        "raw_rows": len(raw_rows),
        "replicates_per_seed_fraction": REPLICATES,
    }
    if complete:
        recomputed = _recompute_paired(raw_rows)
        block["per_seed_fraction"] = recomputed
        claims: dict[str, Any] = {}
        for definition in _CLAIM_DEFINITIONS:
            values = [
                row[definition["difference"]]
                for row in recomputed
                if row["fraction"] == definition["fraction"]
            ]
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
            "cellular sheaf switch instances via "
            "universa.sheaves.make_sheaf_switch_instance("
            "num_vertices=6, num_edges=9, max_stalk_dim=3, num_decoys=3); "
            "K=4 candidates, true target at index 0 pre-permutation"
        ),
        "observation_degradation": (
            "universa.partial_sheaf.SheafObservationModel with "
            "mask_fraction = 0.25 (edge restriction blocks removed, their "
            "block rows dropped from the observed coboundary) AND sign "
            "corruption (corrupt_fraction = g) at every grid point; the "
            "profile grid 0.2..0.9 by 0.1 excludes 0.0 — no clean column "
            "exists anywhere (the no-anchor regime, lifted to sheaf "
            "naturality)"
        ),
        "feature": (
            "the 18-dim no-anchor degradation-profile feature vector "
            "exactly as this runner's sheaf_candidate_features computes "
            "it: log1p of the planted morphism's observed naturality "
            "residual against the observed candidate at the 8 grid points "
            "under the row's observation draw (8 columns), the 7 "
            "first-difference slopes, and 3 masked structural dims "
            "(num_base_vertices, kept edge count, c0_dim — the total "
            "vertex-stalk dimension) read off the masked observation"
        ),
        "profile_grid": list(PROFILE_GRID),
        "mask_fraction": MASK_FRACTION,
        "train_seeds": {"first": TRAIN_SEEDS[0], "last": TRAIN_SEEDS[-1]},
        "eval_seeds": {
            "first": SEALED_EVAL_SEEDS[0],
            "last": SEALED_EVAL_SEEDS[-1],
        },
        "train_fractions": list(TRAIN_FRACTIONS),
        "eval_fractions": list(EVAL_FRACTIONS),
        "held_out_fractions": list(HELD_OUT_FRACTIONS),
        "train_replicates": (
            "one replicate (index 0) per train (seed, fraction); 800 rows"
        ),
        "replicates": REPLICATES,
        "replicate_subseed": (
            "subseed(seed, 'sealed-replicate', str(fraction), "
            "str(replicate_index)) per data row, driving the mask draw AND "
            "the nested corruption draws (sheaf-observe mask, "
            "sheaf-observe corrupt); the same observation_seed feeds all "
            "arms within a row (paired)"
        ),
        "permutation": (
            "per (seed, fraction) from subseed(seed, "
            "'router-v2-permutation', f'{fraction:.6f}') — the component "
            "literal retained verbatim from the v2 schedule as a frozen "
            "string constant — shared across the row's replicates, with "
            "audited labels"
        ),
        "raw_row_retention": (
            "each eval raw row retains, per candidate, the full "
            "naturality-residual profile (the observed naturality residual "
            "at all 8 grid points; 8 raw floats, permuted candidate "
            "order) plus the operating-fraction residual, so the oracle "
            "AND heuristic arms' per-row bits are recomputable from the "
            "raw rows alone (the audit block recomputes them); only the "
            "learned arm needs the retained hash-pinned model"
        ),
        "training": {
            "model": "StructureRouter(feature_dim=18, hidden_dim=64)",
            "epochs": EPOCHS,
            "lr": LEARNING_RATE,
            "torch_seed": TORCH_SEED,
            "lambda_aux": LAMBDA_AUX,
            "tau_start": TAU_START,
            "tau_end": TAU_END,
            "optimizer": "full-batch Adam",
            "dtype_device": "CPU float32",
            "standardization": "train block only",
        },
        "baselines": {
            "oracle": (
                "polluted argmin observed-residual oracle: argmin of the "
                "operating-fraction observed naturality-residual column "
                "(polluted at every fraction — no clean column exists)"
            ),
            "heuristic": (
                "descriptive non-learned heuristic: argmin of the "
                "grid-mean log1p naturality profile (the profile-shape "
                "comparator of H4)"
            ),
        },
        "inference_unit": "the generator seed",
        "estimand": (
            "per (seed, fraction) accuracies over R=4 shared replicates; "
            "paired differences d_oracle = learned_acc - oracle_acc and "
            "d_heur = learned_acc - heuristic_acc"
        ),
        "familywise_alpha": FAMILYWISE_ALPHA,
        "per_claim_alpha": PER_CLAIM_ALPHA,
        "minimum_eligible": MIN_ELIGIBLE,
        "eligibility_rule": (
            "the instance builds and the undegraded naturality audit on "
            "the UNDEGRADED instance passes (true candidate clean "
            "naturality residual <= 1e-9, every decoy > 1e-9); "
            "deterministic per seed and never entering features"
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

    # Eligibility pass over ALL sealed eval seeds, before any fit: an
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
                instance = make_sheaf_switch_instance(
                    seed, NUM_VERTICES, NUM_EDGES, MAX_STALK_DIM, NUM_DECOYS
                )
            except Exception as error:  # a build exception stops the whole run
                build_failure = {
                    "seed": seed,
                    "type": type(error).__name__,
                    "message": str(error),
                }
                break
            misfits = _undegraded_misfits(instance)
            problems = []
            if misfits[0] > AUDIT_TOL:
                problems.append(
                    f"true candidate misfit {misfits[0]:.3e} on the "
                    f"undegraded instance exceeds {AUDIT_TOL}"
                )
            bad_decoys = [
                (index, value)
                for index, value in enumerate(misfits[1:])
                if value <= AUDIT_TOL
            ]
            for index, value in bad_decoys:
                problems.append(
                    f"decoy {index} misfit {value:.3e} on the undegraded "
                    f"instance is at or below {AUDIT_TOL}"
                )
            seed_records.append(
                {
                    "seed": seed,
                    "eligible": not problems,
                    "reason": None if not problems else "; ".join(problems),
                    "true_misfit_undegraded": misfits[0],
                    "decoy_misfits_undegraded": list(misfits[1:]),
                }
            )
            if not problems:
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
        base["paired"] = []
        base["claims"] = _null_claims()
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
            "never an exclusion; no fits ran"
        )
        base["training"] = None
        base["raw_rows"] = []
        base["paired"] = []
        base["claims"] = _null_claims()
        base["audit"] = _audit_block(eligibility, [], complete=False)
        return base
    if len(eligible) < MIN_ELIGIBLE:
        base["status"] = "design_failure_insufficient_eligible"
        base["stop_condition"] = (
            f"only {len(eligible)} seeds were eligible; minimum is "
            f"{MIN_ELIGIBLE}; no fits ran and no claims were decided"
        )
        base["training"] = None
        base["raw_rows"] = []
        base["paired"] = []
        base["claims"] = _null_claims()
        base["audit"] = _audit_block(eligibility, [], complete=False)
        return base

    raw_rows: list[dict[str, Any]] = []
    training: dict[str, Any] | None = None
    paired: list[dict[str, Any]] = []
    failure: tuple[str, BaseException] | None = None
    try:
        train_features, train_labels = _build_train_block()
        model, training = _train_model(train_features, train_labels)
        for seed, instance in eligible:
            try:
                for fraction in EVAL_FRACTIONS:
                    for replicate in range(REPLICATES):
                        row = _build_replicate_row(
                            instance, seed, fraction, replicate,
                            with_residuals=True,
                        )
                        assert row.residual_profiles is not None
                        assert row.operating_residuals is not None
                        (
                            learned_correct,
                            oracle_correct,
                            heuristic_correct,
                        ) = _score_row(
                            model,
                            row.features,
                            row.residual_profiles,
                            row.operating_residuals,
                            row.true_index,
                        )
                        raw_rows.append(
                            {
                                "seed": seed,
                                "fraction": fraction,
                                "replicate_index": replicate,
                                "observation_seed": row.observation_seed,
                                "learned_correct": learned_correct,
                                "oracle_correct": oracle_correct,
                                "heuristic_correct": heuristic_correct,
                                "residual_profiles": [
                                    list(profile)
                                    for profile in row.residual_profiles
                                ],
                                "operating_residuals": list(
                                    row.operating_residuals
                                ),
                                "permutation": list(row.permutation),
                                "true_index": row.true_index,
                            }
                        )
            except BaseException as error:
                if getattr(error, "seed", None) is None:
                    error.seed = seed
                raise
        # Fail-closed row-count invariant: every eligible seed must have
        # contributed exactly R rows per eval fraction before any summary.
        expected_rows = REPLICATES * len(EVAL_FRACTIONS) * len(eligible)
        if len(raw_rows) != expected_rows:
            raise DesignFailureError(
                f"row-count invariant violated: {len(raw_rows)} raw rows, "
                f"expected {expected_rows} (R={REPLICATES} x "
                f"{len(EVAL_FRACTIONS)} fractions x {len(eligible)} "
                "eligible seeds)"
            )
        # Summaries are computed only after every required raw row has
        # completed (frozen stop rule: no interim analysis).
        paired = _recompute_paired(raw_rows)
        base["claims"] = _claim_inference(paired)
    except DesignFailureError as error:
        failure = ("design_failure", error)
    except KeyboardInterrupt as error:
        failure = ("interrupted", error)
    except Exception as error:  # unexpected faults are preserved
        failure = ("execution_failure", error)

    base["training"] = training
    base["raw_rows"] = raw_rows
    try:
        base["paired"] = _recompute_paired(raw_rows)
    except DesignFailureError:
        # The paired-recompute audit itself was the campaign failure cause;
        # do not raise it again out of the artifact assembly — fall back to
        # whatever the campaign try-block produced (possibly empty).
        base["paired"] = paired
    if failure is not None:
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
        base["audit"] = _audit_block(eligibility, raw_rows, complete=False)
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
    base["audit"] = _audit_block(eligibility, raw_rows, complete=True)
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
