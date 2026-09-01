"""Tests for the sealed router-loop-v4 runner
(scripts/run_loop_v4_sealed_1.py).

Seed discipline: the sealed eval block 210001..210036 — like the v1 block
30101..30136, the v2 block 60101..60136, the reserved block 80101..80136,
the discovery block 90101..90136, the sheaf block 20101..20136, the group
block 40101..40136, the 2-complex block 70101..70136, the first loop block
130001..130036, the loop block 140001..140036, the loop-v2 block
160001..160036, and the loop-v3 block 190001..190036 — is NEVER
instantiated in this suite: every campaign test
monkeypatches the runner's eval seed block down to explicit non-sealed
fixture seeds (70001..70005) and every direct eval-instance construction
uses those same fixture seeds. The train seed block 200001..200400 (the
runner's own TRAIN block, never an eval seed) is likewise NEVER
instantiated in this suite: campaign tests monkeypatch TRAIN_SEEDS down
to the protocol's sanctioned train fixture seeds (70501..70502 for the
tiny campaigns), and the trained-models fixture trains on the sanctioned
fixture block 70501..70520. The loop-v2 train block 170001..170200 and
the loop-v3 train block 180001..180400 (previous experiments' consumed
blocks) are never instantiated here either.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only, before the torch import

import numpy as np
import pytest
import torch
from scipy.stats import t as t_dist

PROJECT_ROOT = Path(__file__).parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_loop_v4_sealed_1.py"
SPEC = importlib.util.spec_from_file_location("run_loop_v4_sealed_1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

torch.set_num_threads(1)

import universa  # noqa: E402
from universa.budgets import make_budget_instance  # noqa: E402
from universa.generators import subseed  # noqa: E402
from universa.loop import ALARM_TOL  # noqa: E402
from universa.loop_v2 import (  # noqa: E402
    GENERIC_FEATURE_DIM,
    MAP_ACCEPT_TOL,
    GenericMLP,
    LearnedAlarmV2,
    operating_grid_point,
)
from universa.router_v2 import NO_ANCHOR_GRID  # noqa: E402

# Explicit non-sealed fixture seeds; no test may instantiate a sealed seed.
FIXTURE_SEEDS = (70001, 70002, 70003, 70004, 70005)
# The sanctioned train fixture block: the trained-models fixture trains the
# three frozen models (and calibrates the v2 alarm's threshold) on
# 70501..70520; tiny campaigns monkeypatch TRAIN_SEEDS down to a prefix.
TRAIN_FIXTURE_SEEDS = tuple(range(70501, 70521))
V1_SEALED_BLOCK = range(30101, 30137)
V2_SEALED_BLOCK = range(60101, 60137)
RESERVED_SEALED_BLOCK = range(80101, 80137)
DISCOVERY_SEALED_BLOCK = range(90101, 90137)
SHEAF_SEALED_BLOCK = range(20101, 20137)
GROUP_SEALED_BLOCK = range(40101, 40137)
COMPLEX2_SEALED_BLOCK = range(70101, 70137)
FIRST_LOOP_SEALED_BLOCK = range(130001, 130037)
LOOP_SEALED_BLOCK = range(140001, 140037)
LOOP_V2_TRAIN_BLOCK = range(170001, 170201)
LOOP_V2_SEALED_BLOCK = range(160001, 160037)
# loop-v3's consumed blocks: sealed history, never reused here.
LOOP_V3_TRAIN_BLOCK = range(180001, 180401)
LOOP_V3_SEALED_BLOCK = range(190001, 190037)
# This experiment's own declared blocks.
LOOP_V4_TRAIN_BLOCK = range(200001, 200401)
LOOP_V4_SEALED_BLOCK = range(210001, 210037)

ALL_SEALED_BLOCKS = (
    V1_SEALED_BLOCK,
    V2_SEALED_BLOCK,
    RESERVED_SEALED_BLOCK,
    DISCOVERY_SEALED_BLOCK,
    SHEAF_SEALED_BLOCK,
    GROUP_SEALED_BLOCK,
    COMPLEX2_SEALED_BLOCK,
    FIRST_LOOP_SEALED_BLOCK,
    LOOP_SEALED_BLOCK,
    LOOP_V2_TRAIN_BLOCK,
    LOOP_V2_SEALED_BLOCK,
    LOOP_V3_TRAIN_BLOCK,
    LOOP_V3_SEALED_BLOCK,
    LOOP_V4_TRAIN_BLOCK,
    LOOP_V4_SEALED_BLOCK,
)

K = MODULE.NUM_VIEW_CANDIDATES
GRID_LEN = len(MODULE.PROFILE_GRID)


# ---------------------------------------------------------------------------
# Helpers.


def _claim_object(definition: dict) -> dict:
    """A full seal-record claim object matching one frozen definition."""
    return {
        "id": definition["id"],
        "statistic": definition["statistic"],
        "theta": f"E_seed[{definition['statistic']}(seed)]",
        "null": f"theta <= {definition['threshold']}",
        "alternative": f"theta > {definition['threshold']}",
        "bound_direction": "greater",
        "threshold": definition["threshold"],
        "support_rule": (
            "supported iff the one-sided Bonferroni lower bound exceeds "
            f"{definition['threshold']}"
        ),
    }


def _seal_payload(output_path: str = "result.json", **overrides: object) -> dict:
    payload: dict[str, object] = {
        "schema": MODULE.SEAL_SCHEMA,
        "design_commit": "0" * 40,
        "protocol_sha256": "1" * 64,
        "runner_sha256": "2" * 64,
        "code_manifest": {"src/universa/alpha.py": "3" * 64},
        "train_seed_block": {"first": 200001, "last": 200400},
        "eval_seed_block": {"first": 210001, "last": 210036},
        "no_preview_declaration": "no sealed seed was previewed before the seal",
        "primary_family": [
            _claim_object(definition) for definition in MODULE._CLAIM_DEFINITIONS
        ],
        "stop_rules": ["fewer than 30 eligible seeds stops the design"],
        "output_path": output_path,
    }
    payload.update(overrides)
    return payload


def _write_seal(root: Path, payload: dict, seal: str = "seal.json") -> str:
    path = root / seal
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return seal


def _stage_sealed_tree(tmp_path: Path) -> dict[str, object]:
    """A minimal fake repository: package files, protocol, runner, seal."""
    package = tmp_path / "src" / "universa"
    package.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, content in (("alpha.py", "A = 1\n"), ("beta.py", "B = 2\n")):
        (package / name).write_text(content, encoding="utf-8")
        manifest[f"src/universa/{name}"] = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()
    protocol = tmp_path / MODULE.PROTOCOL
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_text("frozen protocol\n", encoding="utf-8")
    protocol_hash = hashlib.sha256(b"frozen protocol\n").hexdigest()
    runner = tmp_path / MODULE.RUNNER_SOURCE
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("runner source\n", encoding="utf-8")
    runner_hash = hashlib.sha256(b"runner source\n").hexdigest()
    _write_seal(
        tmp_path,
        _seal_payload(
            protocol_sha256=protocol_hash,
            runner_sha256=runner_hash,
            code_manifest=manifest,
        ),
    )
    return {
        "protocol_hash": protocol_hash,
        "runner_hash": runner_hash,
        "manifest": manifest,
    }


def _fake_git_clean(_root: Path, *args: str) -> str:
    if args[0] in ("status", "cat-file", "merge-base"):
        return ""
    if args[:2] == ("rev-parse", "HEAD"):
        return "f" * 40
    raise AssertionError(f"unexpected git call: {args}")


@pytest.fixture(autouse=True)
def _no_real_git(monkeypatch):
    """No test ever shells out to the real git binary."""
    monkeypatch.setattr(MODULE, "_git_checked", _fake_git_clean)


def _patch_staged_preflight(monkeypatch, tmp_path: Path, hashes: dict) -> None:
    monkeypatch.setattr(MODULE, "_git_checked", _fake_git_clean)
    monkeypatch.setattr(
        MODULE,
        "_git_head_blob",
        lambda root, relative: (root / relative).read_bytes(),
    )
    monkeypatch.setattr(MODULE, "PROTOCOL_SHA256", hashes["protocol_hash"])
    monkeypatch.setattr(
        MODULE, "__file__", str(tmp_path / MODULE.RUNNER_SOURCE)
    )
    # The staged tree's own "installed" package: preflight requires the
    # imported universa to resolve under <root>/src/universa.
    monkeypatch.setattr(
        universa, "__file__", str(tmp_path / "src" / "universa" / "__init__.py")
    )


def _staged_payload(hashes: dict, **overrides: object) -> dict:
    """A seal payload consistent with the staged tree, minus the overrides."""
    merged = {
        "protocol_sha256": hashes["protocol_hash"],
        "runner_sha256": hashes["runner_hash"],
        "code_manifest": hashes["manifest"],
    }
    merged.update(overrides)
    return _seal_payload(**merged)


# The on-disk seal content the stub preflight below stages.
_STUB_SEAL_JSON = '{"stub": true}\n'


def _patch_tiny_campaign(
    monkeypatch,
    *,
    eval_seeds=(70003, 70004, 70005),
    min_eligible=3,
    train_seeds=(70501, 70502),
    epochs=2,
) -> None:
    """Shrink the frozen design to a fast, fully non-sealed campaign."""

    def _stub_preflight(project_root, *_args, **_kwargs):
        # A self-consistent stub provenance: the post-run TOCTOU re-check
        # re-reads the seal and protocol from disk and re-hashes the
        # running file, so stage real files under the project root.
        root = Path(project_root)
        (root / "seal.json").write_text(_STUB_SEAL_JSON, encoding="utf-8")
        (root / "protocol.md").write_text("stub protocol\n", encoding="utf-8")
        return {
            "stub": True,
            "git_status_porcelain": "",
            "code_manifest": {"files": {}},
            "seal": {"path": "seal.json"},
            "protocol": {
                "path": "protocol.md",
                "sha256": MODULE._sha256(root / "protocol.md"),
            },
            "runner": {
                "path": MODULE.RUNNER_SOURCE,
                "sha256": MODULE._sha256(Path(MODULE.__file__).resolve()),
            },
        }

    monkeypatch.setattr(MODULE, "_preflight", _stub_preflight)
    monkeypatch.setattr(
        MODULE,
        "_git_head_blob",
        lambda root, relative: (root / relative).read_bytes(),
    )
    monkeypatch.setattr(MODULE, "_code_manifest", lambda _root: {})
    monkeypatch.setattr(MODULE, "TRAIN_SEEDS", tuple(train_seeds))
    monkeypatch.setattr(MODULE, "SEALED_EVAL_SEEDS", tuple(eval_seeds))
    monkeypatch.setattr(MODULE, "MIN_ELIGIBLE", min_eligible)
    monkeypatch.setattr(MODULE, "EPOCHS", epochs)
    for n in range(2, 8):
        monkeypatch.setitem(
            MODULE._T_ONE_SIDED_BONFERRONI,
            n,
            float(t_dist.ppf(1 - 0.05 / 4, n - 1)),
        )


def _definition(claim_id: str) -> dict:
    return next(
        definition
        for definition in MODULE._CLAIM_DEFINITIONS
        if definition["id"] == claim_id
    )


@pytest.fixture(scope="module")
def trained_models():
    """The three frozen models trained on a small non-sealed train block
    (70501..70520, the protocol's sanctioned train fixture seeds) with the
    full frozen recipe (150 epochs) plus the frozen train-block threshold
    calibration — the pinned-behavior arm fixtures. Returns
    ``(router, alarm, generic, calibration)``."""
    saved = MODULE.TRAIN_SEEDS
    MODULE.TRAIN_SEEDS = TRAIN_FIXTURE_SEEDS
    try:
        block = MODULE._build_train_block()
        router, alarm, generic, calibration, _ = MODULE._train_models(block)
    finally:
        MODULE.TRAIN_SEEDS = saved
    return router, alarm, generic, calibration


def _fixture_instance(seed: int):
    return make_budget_instance(
        seed,
        MODULE.NUM_VERTICES,
        MODULE.NUM_EDGES,
        MODULE.NUM_CLASSES,
        MODULE.NUM_DECOYS,
    )


def _arm_row(seed, instance, condition, arm, models):
    """One raw row on a fixture instance with the trained fixture models."""
    router, alarm, generic, calibration = models
    inputs = MODULE._condition_inputs(seed, instance)
    library, observations = inputs[condition]
    return MODULE._build_arm_row(
        seed,
        instance,
        library,
        observations,
        condition=condition,
        arm=arm,
        observation_seed=MODULE._observation_seed(seed),
        router=router,
        alarm=alarm,
        threshold=calibration["threshold"],
        generic=generic,
    )


def _constant_generic_model(cls: int) -> GenericMLP:
    """A stub generic model whose decision is constantly ``cls`` (zero
    weights, one-hot head bias) — deterministic arm-semantics checks that
    do not depend on training quality."""
    torch.manual_seed(0)
    model = GenericMLP(K)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head.bias[cls] = 1.0
    return model


# ---------------------------------------------------------------------------
# Frozen constants and seed-block discipline.


def test_sealed_seed_blocks_declared_disjoint_and_never_instantiated() -> None:
    assert MODULE.SEALED_EVAL_SEEDS == tuple(range(210001, 210037))
    assert MODULE.TRAIN_SEEDS == tuple(range(200001, 200401))
    assert not set(MODULE.SEALED_EVAL_SEEDS) & set(MODULE.TRAIN_SEEDS)
    # The fixture seeds belong to no seed block: eval fixtures sit outside
    # every block, and the sanctioned train fixtures (70501..70520,
    # 70601..70603) sit OUTSIDE the runner's own reserved train block.
    for seed in FIXTURE_SEEDS:
        for block in ALL_SEALED_BLOCKS:
            assert seed not in block
    for seed in TRAIN_FIXTURE_SEEDS:
        assert seed not in LOOP_V4_TRAIN_BLOCK
        assert seed not in LOOP_V4_SEALED_BLOCK
        for block in ALL_SEALED_BLOCKS:
            assert seed not in block
    # This experiment's own blocks are disjoint from every block the series
    # has ever declared — including loop-v3's consumed 180001..180400 and
    # 190001..190036, which this suite must never reuse.
    for block in ALL_SEALED_BLOCKS:
        if block not in (LOOP_V4_TRAIN_BLOCK, LOOP_V4_SEALED_BLOCK):
            assert not set(MODULE.SEALED_EVAL_SEEDS) & set(block)
            assert not set(MODULE.TRAIN_SEEDS) & set(block)
    assert not set(MODULE.TRAIN_SEEDS) & set(LOOP_V3_TRAIN_BLOCK)
    assert not set(MODULE.SEALED_EVAL_SEEDS) & set(LOOP_V3_SEALED_BLOCK)


def test_frozen_design_constants() -> None:
    assert (MODULE.NUM_VERTICES, MODULE.NUM_EDGES) == (8, 14)
    assert (MODULE.NUM_CLASSES, MODULE.NUM_DECOYS) == (6, 3)
    assert MODULE.NUM_VIEW_CANDIDATES == 3  # the equal-K paired views
    assert MODULE.CONDITIONS == ("in_library", "out_of_library", "null_control")
    assert MODULE.ARMS == (
        "arch_full",
        "routing_only",
        "discovery_only",
        "generic",
    )
    assert MODULE.PROFILE_GRID == tuple((i + 2) / 10 for i in range(8))
    assert 0.0 not in MODULE.PROFILE_GRID  # the no-anchor regime
    assert MODULE.MASK_FRACTION == 0.25
    assert MODULE.NUM_OBSERVATIONS == 16
    assert MODULE.FEATURE_DIM == 18
    assert MODULE.ROUTER_HIDDEN_DIM == 64
    assert MODULE.EPOCHS == 150
    assert MODULE.LEARNING_RATE == 1e-3
    assert MODULE.TORCH_SEED_ROUTER == 4242
    assert MODULE.TORCH_SEED_ALARM == 4243
    assert MODULE.TORCH_SEED_GENERIC == 4244
    assert MODULE.FALSE_QUIET_COST == 1.0
    assert MODULE.FALSE_ALARM_COST == 1.0
    # loop-v3's bounded rule is gone: this experiment prices both errors.
    assert not hasattr(MODULE, "MAX_FALSE_QUIET_RATE")
    assert (MODULE.LAMBDA_AUX, MODULE.TAU_START, MODULE.TAU_END) == (
        0.01,
        2.0,
        0.25,
    )
    assert MODULE.PER_CLAIM_ALPHA == 0.0125
    assert MODULE.MIN_ELIGIBLE == 30
    assert MODULE.AUDIT_TOL == 1e-9 == ALARM_TOL
    assert MODULE.EXPERIMENT_ID == "universa-loop-v4-sealed-1"
    assert MODULE.RESULT_SCHEMA == "universa-router-loop-v4-sealed-result/1"
    assert MODULE.SEAL_SCHEMA == "universa-seal/10"
    assert MODULE.PROTOCOL == "docs/30-sealed-router-loop-v4-protocol.md"
    assert MODULE.DEFAULT_SEAL == "docs/31-router-loop-v4-seal.json"
    assert MODULE.RUNNER_SOURCE == "scripts/run_loop_v4_sealed_1.py"
    assert MODULE.CLAIM_IDS == (
        "h1-loopv4-arch-vs-generic-e2e",
        "h2-loopv4-arch-vs-routing-only-e2e",
        "h3-loopv4-arch-vs-generic-inlibrary",
        "h4-loopv4-arch-vs-discovery-only-inlibrary-harm",
    )
    assert [d["statistic"] for d in MODULE._CLAIM_DEFINITIONS] == [
        "arch_vs_generic_e2e",
        "arch_vs_routing_only_e2e",
        "arch_vs_generic_inlibrary",
        "arch_vs_discovery_only_inlibrary_harm",
    ]
    assert [d["threshold"] for d in MODULE._CLAIM_DEFINITIONS] == [
        0.0,
        0.0,
        0.0,
        -0.05,
    ]
    assert [d["difference"] for d in MODULE._CLAIM_DEFINITIONS] == [
        "d_arch_generic_e2e",
        "d_arch_routing_only_e2e",
        "d_arch_generic_inlibrary",
        "d_arch_discovery_only_inlibrary_harm",
    ]
    assert [d["scope"] for d in MODULE._CLAIM_DEFINITIONS] == [
        "e2e",
        "e2e",
        "in_library",
        "in_library",
    ]
    assert [d["arms"] for d in MODULE._CLAIM_DEFINITIONS] == [
        ("arch_full", "generic"),
        ("arch_full", "routing_only"),
        ("arch_full", "generic"),
        ("arch_full", "discovery_only"),
    ]
    by_id = {d["id"]: d for d in MODULE._CLAIM_DEFINITIONS}
    assert by_id["h1-loopv4-arch-vs-generic-e2e"]["role"] == "primary"
    assert all(
        d["role"] == "secondary"
        for d in MODULE._CLAIM_DEFINITIONS
        if d["id"] != "h1-loopv4-arch-vs-generic-e2e"
    )
    assert "not evidence" in MODULE.DECLARED_CAVEAT
    assert "architecture-free" in MODULE.DECLARED_CAVEAT
    # The runner ships with the placeholder until the protocol hash is
    # pinned; in either state the fail-closed behavior is pinned.
    if MODULE.PROTOCOL_SHA256 == "PENDING_PROTOCOL_SHA256":
        with pytest.raises(RuntimeError, match="placeholder"):
            MODULE._frozen_protocol_sha256()
    else:
        assert len(MODULE.PROTOCOL_SHA256) == 64
        assert MODULE._frozen_protocol_sha256() == MODULE.PROTOCOL_SHA256


def test_pinned_protocol_hash_matches_the_protocol_on_disk() -> None:
    """A stale pin must fail here, not at run time after the seal.

    The runner's preflight compares the frozen constant against the
    protocol bytes, but that check only fires during a canonical run —
    by which point the seal is committed and pushed. Pinning the hash is
    the last hand-edited step before commit A, so the pin is verified
    against the actual file here, where a mistake is still cheap.
    """
    if MODULE.PROTOCOL_SHA256 == "PENDING_PROTOCOL_SHA256":
        pytest.skip("the protocol hash is not pinned yet")
    protocol = PROJECT_ROOT / MODULE.PROTOCOL
    assert protocol.is_file(), MODULE.PROTOCOL
    digest = hashlib.sha256(protocol.read_bytes()).hexdigest()
    assert digest == MODULE.PROTOCOL_SHA256, (
        f"the pinned PROTOCOL_SHA256 is stale: {MODULE.PROTOCOL} hashes to "
        f"{digest}"
    )


def test_critical_value_table_matches_scipy() -> None:
    assert sorted(MODULE._T_ONE_SIDED_BONFERRONI) == list(range(30, 37))
    for n in range(30, 37):
        expected = float(t_dist.ppf(1 - 0.05 / 4, n - 1))
        assert MODULE._T_ONE_SIDED_BONFERRONI[n] == pytest.approx(
            expected, rel=1e-12
        )


def test_observation_seed_schedule_is_the_shared_router_v2_draw() -> None:
    # The canonical shared observation seed: ONE draw per seed — the
    # router-v2 regime's observation family — reused across every
    # candidate, every condition, and every arm of the row, for train and
    # eval rows alike.
    for seed in (70001, 70002, 70003):
        assert MODULE._observation_seed(seed) == subseed(
            seed, "router-v2-observe"
        )
        assert MODULE._observation_seed(seed) == MODULE._observation_seed(seed)
    assert len({MODULE._observation_seed(seed) for seed in FIXTURE_SEEDS}) == 5


def test_operating_grid_point_derivation() -> None:
    # The generic arm's single degradation level per row:
    # grid[subseed(seed, "loop-v2-operating") % 8].
    for seed in FIXTURE_SEEDS:
        point = operating_grid_point(seed)
        assert point == NO_ANCHOR_GRID[subseed(seed, "loop-v2-operating") % 8]
        assert point in NO_ANCHOR_GRID
        assert point != 0.0  # the no-anchor regime
        assert point == operating_grid_point(seed)  # deterministic
    assert len({operating_grid_point(seed) for seed in FIXTURE_SEEDS}) > 1


def test_paired_views_are_equal_k_with_the_truth_at_zero() -> None:
    for seed in FIXTURE_SEEDS:
        instance = _fixture_instance(seed)
        in_library, out_library = MODULE._paired_views(instance)
        assert len(in_library) == len(out_library) == K
        assert in_library[0] is instance.true_target
        assert all(
            candidate is decoy
            for candidate, decoy in zip(in_library[1:], instance.decoy_targets)
        )
        assert all(
            candidate is decoy
            for candidate, decoy in zip(out_library, instance.decoy_targets)
        )


# ---------------------------------------------------------------------------
# Seal validation.


def test_load_seal_accepts_a_valid_frozen_seal(tmp_path: Path) -> None:
    _write_seal(tmp_path, _seal_payload())
    seal = MODULE._load_seal(tmp_path, "seal.json")
    assert seal["schema"] == "universa-seal/10"
    # The frozen v4 claim ids (loopv4 prefix) and their statistics.
    assert [claim["id"] for claim in seal["primary_family"]] == [
        "h1-loopv4-arch-vs-generic-e2e",
        "h2-loopv4-arch-vs-routing-only-e2e",
        "h3-loopv4-arch-vs-generic-inlibrary",
        "h4-loopv4-arch-vs-discovery-only-inlibrary-harm",
    ]
    assert [claim["statistic"] for claim in seal["primary_family"]] == [
        "arch_vs_generic_e2e",
        "arch_vs_routing_only_e2e",
        "arch_vs_generic_inlibrary",
        "arch_vs_discovery_only_inlibrary_harm",
    ]


def test_load_seal_rejects_missing_file_invalid_json_and_non_dict(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="missing"):
        MODULE._load_seal(tmp_path, "seal.json")
    (tmp_path / "seal.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not valid JSON"):
        MODULE._load_seal(tmp_path, "seal.json")
    (tmp_path / "seal.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="JSON object"):
        MODULE._load_seal(tmp_path, "seal.json")


def test_load_seal_rejects_wrong_schema(tmp_path: Path) -> None:
    _write_seal(tmp_path, _seal_payload(schema="universa-seal/8"))
    with pytest.raises(RuntimeError, match="schema"):
        MODULE._load_seal(tmp_path, "seal.json")


def test_load_seal_rejects_missing_keys_and_bad_commit(tmp_path: Path) -> None:
    payload = _seal_payload()
    del payload["stop_rules"]
    _write_seal(tmp_path, payload)
    with pytest.raises(RuntimeError, match="missing keys"):
        MODULE._load_seal(tmp_path, "seal.json")
    _write_seal(tmp_path, _seal_payload(design_commit="0" * 39))
    with pytest.raises(RuntimeError, match="design_commit"):
        MODULE._load_seal(tmp_path, "seal.json")
    _write_seal(tmp_path, _seal_payload(design_commit="G" * 40))
    with pytest.raises(RuntimeError, match="design_commit"):
        MODULE._load_seal(tmp_path, "seal.json")


def test_load_seal_rejects_bad_hash_formats(tmp_path: Path) -> None:
    _write_seal(tmp_path, _seal_payload(protocol_sha256="1" * 63))
    with pytest.raises(RuntimeError, match="protocol_sha256"):
        MODULE._load_seal(tmp_path, "seal.json")
    _write_seal(tmp_path, _seal_payload(runner_sha256="Z" * 64))
    with pytest.raises(RuntimeError, match="runner_sha256"):
        MODULE._load_seal(tmp_path, "seal.json")
    _write_seal(
        tmp_path,
        _seal_payload(code_manifest={"src/universa/alpha.py": "3" * 63}),
    )
    with pytest.raises(RuntimeError, match="code_manifest"):
        MODULE._load_seal(tmp_path, "seal.json")
    _write_seal(tmp_path, _seal_payload(code_manifest={}))
    with pytest.raises(RuntimeError, match="code_manifest"):
        MODULE._load_seal(tmp_path, "seal.json")
    _write_seal(
        tmp_path,
        _seal_payload(code_manifest={"src/universa/sub/alpha.py": "3" * 64}),
    )
    with pytest.raises(RuntimeError, match="code_manifest"):
        MODULE._load_seal(tmp_path, "seal.json")


def test_load_seal_rejects_wrong_seed_blocks(tmp_path: Path) -> None:
    _write_seal(
        tmp_path,
        _seal_payload(train_seed_block={"first": 200001, "last": 200401}),
    )
    with pytest.raises(RuntimeError, match="train_seed_block"):
        MODULE._load_seal(tmp_path, "seal.json")
    _write_seal(
        tmp_path,
        _seal_payload(eval_seed_block={"first": 190000, "last": 210036}),
    )
    with pytest.raises(RuntimeError, match="eval_seed_block"):
        MODULE._load_seal(tmp_path, "seal.json")
    _write_seal(tmp_path, _seal_payload(train_seed_block=[200001, 200400]))
    with pytest.raises(RuntimeError, match="train_seed_block"):
        MODULE._load_seal(tmp_path, "seal.json")
    # The v1 loop's empty-train-block sentinel does NOT apply here: this
    # experiment trains three models on 200001..200400.
    _write_seal(
        tmp_path, _seal_payload(train_seed_block={"first": 0, "last": 0})
    )
    with pytest.raises(RuntimeError, match="train_seed_block"):
        MODULE._load_seal(tmp_path, "seal.json")
    _write_seal(
        tmp_path,
        _seal_payload(
            eval_seed_block={"first": 210001, "last": 210036, "note": "x"}
        ),
    )
    with pytest.raises(RuntimeError, match="eval_seed_block"):
        MODULE._load_seal(tmp_path, "seal.json")


def test_load_seal_rejects_missing_no_preview_declaration(
    tmp_path: Path,
) -> None:
    _write_seal(tmp_path, _seal_payload(no_preview_declaration=""))
    with pytest.raises(RuntimeError, match="no_preview_declaration"):
        MODULE._load_seal(tmp_path, "seal.json")
    _write_seal(tmp_path, _seal_payload(no_preview_declaration=123))
    with pytest.raises(RuntimeError, match="no_preview_declaration"):
        MODULE._load_seal(tmp_path, "seal.json")


def test_load_seal_rejects_altered_claims(tmp_path: Path) -> None:
    family = [_claim_object(d) for d in MODULE._CLAIM_DEFINITIONS]
    _write_seal(
        tmp_path, _seal_payload(primary_family=list(reversed(family)))
    )
    with pytest.raises(RuntimeError, match="claim ids"):
        MODULE._load_seal(tmp_path, "seal.json")
    _write_seal(tmp_path, _seal_payload(primary_family=family[:3]))
    with pytest.raises(RuntimeError, match="four"):
        MODULE._load_seal(tmp_path, "seal.json")
    _write_seal(
        tmp_path, _seal_payload(primary_family=["h1", "h2", "h3", "h4"])
    )
    with pytest.raises(RuntimeError, match="claim object"):
        MODULE._load_seal(tmp_path, "seal.json")
    missing_key = _claim_object(MODULE._CLAIM_DEFINITIONS[0])
    del missing_key["theta"]
    _write_seal(
        tmp_path, _seal_payload(primary_family=[missing_key, *family[1:]])
    )
    with pytest.raises(RuntimeError, match="missing keys"):
        MODULE._load_seal(tmp_path, "seal.json")
    bad_id = _claim_object(MODULE._CLAIM_DEFINITIONS[0])
    bad_id["id"] = 123
    _write_seal(tmp_path, _seal_payload(primary_family=[bad_id, *family[1:]]))
    with pytest.raises(RuntimeError, match="string id"):
        MODULE._load_seal(tmp_path, "seal.json")


def test_load_seal_rejects_tampered_claim_values_and_unknown_keys(
    tmp_path: Path,
) -> None:
    family = [_claim_object(d) for d in MODULE._CLAIM_DEFINITIONS]
    for key, bad_value in (
        ("statistic", "arch_vs_generic_e2e_typo"),
        ("threshold", 0.01),
        ("bound_direction", "less"),
    ):
        tampered = _claim_object(MODULE._CLAIM_DEFINITIONS[0])
        tampered[key] = bad_value
        _write_seal(
            tmp_path, _seal_payload(primary_family=[tampered, *family[1:]])
        )
        with pytest.raises(RuntimeError, match="frozen definition"):
            MODULE._load_seal(tmp_path, "seal.json")
    # The h4 harm threshold is -0.05 exactly; a watered-down margin fails.
    tampered = _claim_object(MODULE._CLAIM_DEFINITIONS[3])
    tampered["threshold"] = -0.1
    _write_seal(
        tmp_path, _seal_payload(primary_family=[*family[:3], tampered])
    )
    with pytest.raises(RuntimeError, match="frozen definition"):
        MODULE._load_seal(tmp_path, "seal.json")
    # A tampered role fails as well.
    tampered = _claim_object(MODULE._CLAIM_DEFINITIONS[0])
    tampered["role"] = "secondary"
    _write_seal(
        tmp_path, _seal_payload(primary_family=[tampered, *family[1:]])
    )
    with pytest.raises(RuntimeError, match="frozen definition"):
        MODULE._load_seal(tmp_path, "seal.json")
    # An unknown top-level seal key is rejected.
    _write_seal(tmp_path, _seal_payload(unexpected="extra"))
    with pytest.raises(RuntimeError, match="unknown keys"):
        MODULE._load_seal(tmp_path, "seal.json")
    # An unknown key inside a seed-block dict is rejected.
    _write_seal(
        tmp_path,
        _seal_payload(
            train_seed_block={"first": 200001, "last": 200400, "note": "x"}
        ),
    )
    with pytest.raises(RuntimeError, match="train_seed_block"):
        MODULE._load_seal(tmp_path, "seal.json")


def test_load_seal_rejects_unknown_keys_inside_a_claim(tmp_path: Path) -> None:
    family = [_claim_object(d) for d in MODULE._CLAIM_DEFINITIONS]
    tampered = _claim_object(MODULE._CLAIM_DEFINITIONS[0])
    tampered["p_value"] = 0.01  # a foreign analysis field
    _write_seal(
        tmp_path, _seal_payload(primary_family=[tampered, *family[1:]])
    )
    with pytest.raises(RuntimeError, match="unknown keys"):
        MODULE._load_seal(tmp_path, "seal.json")
    # The optional role is the ONLY extra key a claim may carry, and it
    # must match the frozen definition when present.
    with_role = _claim_object(MODULE._CLAIM_DEFINITIONS[0])
    with_role["role"] = "primary"
    _write_seal(
        tmp_path, _seal_payload(primary_family=[with_role, *family[1:]])
    )
    seal = MODULE._load_seal(tmp_path, "seal.json")
    assert seal["primary_family"][0]["role"] == "primary"


def test_load_seal_rejects_empty_descriptive_claim_subfields(
    tmp_path: Path,
) -> None:
    family = [_claim_object(d) for d in MODULE._CLAIM_DEFINITIONS]
    for key in ("theta", "null", "alternative", "support_rule"):
        for bad in ("", "   ", 0):
            tampered = _claim_object(MODULE._CLAIM_DEFINITIONS[1])
            tampered[key] = bad
            _write_seal(
                tmp_path,
                _seal_payload(
                    primary_family=[family[0], tampered, *family[2:]]
                ),
            )
            with pytest.raises(RuntimeError, match="descriptive subfields"):
                MODULE._load_seal(tmp_path, "seal.json")


def test_load_seal_rejects_watered_down_stop_rules_and_bad_output(
    tmp_path: Path,
) -> None:
    _write_seal(tmp_path, _seal_payload(stop_rules=[]))
    with pytest.raises(RuntimeError, match="stop_rules"):
        MODULE._load_seal(tmp_path, "seal.json")
    _write_seal(tmp_path, _seal_payload(stop_rules=["  "]))
    with pytest.raises(RuntimeError, match="stop_rules"):
        MODULE._load_seal(tmp_path, "seal.json")
    _write_seal(tmp_path, _seal_payload(stop_rules="none"))
    with pytest.raises(RuntimeError, match="stop_rules"):
        MODULE._load_seal(tmp_path, "seal.json")
    _write_seal(tmp_path, _seal_payload(output_path=""))
    with pytest.raises(RuntimeError, match="output_path"):
        MODULE._load_seal(tmp_path, "seal.json")


# ---------------------------------------------------------------------------
# Preflight ceremony: order and guards.


def test_preflight_accepts_a_staged_sealed_tree(
    monkeypatch, tmp_path: Path
) -> None:
    hashes = _stage_sealed_tree(tmp_path)
    _patch_staged_preflight(monkeypatch, tmp_path, hashes)
    provenance = MODULE._preflight(
        tmp_path, tmp_path / "result.json", seal="seal.json"
    )
    assert provenance["clean_worktree"] is True
    assert provenance["git_status_porcelain"] == ""
    assert "git_status_short" not in provenance
    assert provenance["execution_revision"] == "f" * 40
    assert provenance["seal"]["design_commit"] == "0" * 40
    assert provenance["seal"]["schema"] == "universa-seal/10"
    assert provenance["seal"]["committed_at_head"] is True
    assert (
        provenance["protocol"]["path"]
        == "docs/30-sealed-router-loop-v4-protocol.md"
    )
    assert provenance["protocol"]["sha256"] == hashes["protocol_hash"]
    assert provenance["runner"]["path"] == "scripts/run_loop_v4_sealed_1.py"
    assert provenance["runner"]["sha256"] == hashes["runner_hash"]
    assert provenance["code_manifest"]["files"] == hashes["manifest"]
    assert len(provenance["code_manifest"]["sha256"]) == 64
    execution = provenance["execution"]
    assert execution["torch_num_threads"] == 1
    assert execution["cuda_available"] is False
    assert execution["tensor_dtype"] == "float32"
    assert execution["array_dtype"] == "float64"
    assert "argv" in execution


def test_preflight_refuses_existing_output_before_any_git_access(
    monkeypatch, tmp_path: Path
) -> None:
    output = tmp_path / "result.json"
    output.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        MODULE,
        "_git_checked",
        lambda *_args: pytest.fail("git was touched before the output check"),
    )
    with pytest.raises(RuntimeError, match="output already exists"):
        MODULE._preflight(tmp_path, output, seal="seal.json")


def test_preflight_refuses_output_outside_project_root(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="inside the project root"):
        MODULE._preflight(tmp_path, Path("/tmp/elsewhere.json"), seal="seal.json")


def test_preflight_refuses_dirty_tree_before_seal_access(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        MODULE, "_git_checked", lambda *_args: " M dirty.py"
    )
    monkeypatch.setattr(
        MODULE,
        "_load_seal",
        lambda *_args, **_kwargs: pytest.fail(
            "the seal was read on a dirty tree"
        ),
    )
    with pytest.raises(RuntimeError, match="dirty"):
        MODULE._preflight(tmp_path, tmp_path / "result.json", seal="seal.json")


def test_preflight_rejects_seal_not_committed_at_head(
    monkeypatch, tmp_path: Path
) -> None:
    _stage_sealed_tree(tmp_path)

    def fake_git(_root: Path, *args: str) -> str:
        if args[0] == "status":
            return ""
        if args[0] == "cat-file":
            raise RuntimeError(
                "stop condition: git cat-file -e HEAD:seal.json failed: "
                "path exists on disk but not in HEAD"
            )
        raise AssertionError(args)

    monkeypatch.setattr(MODULE, "_git_checked", fake_git)
    with pytest.raises(RuntimeError, match="cat-file"):
        MODULE._preflight(tmp_path, tmp_path / "result.json", seal="seal.json")


def test_preflight_rejects_output_path_mismatch_with_seal(
    monkeypatch, tmp_path: Path
) -> None:
    hashes = _stage_sealed_tree(tmp_path)
    _write_seal(tmp_path, _staged_payload(hashes, output_path="other.json"))
    _patch_staged_preflight(monkeypatch, tmp_path, hashes)
    with pytest.raises(RuntimeError, match="does not match"):
        MODULE._preflight(tmp_path, tmp_path / "result.json", seal="seal.json")


def test_preflight_refuses_the_protocol_placeholder(
    monkeypatch, tmp_path: Path
) -> None:
    # With the placeholder (re)stored, the fail-closed guard must bite.
    hashes = _stage_sealed_tree(tmp_path)
    _patch_staged_preflight(monkeypatch, tmp_path, hashes)
    monkeypatch.setattr(
        MODULE, "PROTOCOL_SHA256", "PENDING_PROTOCOL_SHA256"
    )
    assert MODULE.PROTOCOL_SHA256 == "PENDING_PROTOCOL_SHA256"
    assert hashes is not None
    with pytest.raises(RuntimeError, match="placeholder"):
        MODULE._preflight(tmp_path, tmp_path / "result.json", seal="seal.json")


def test_preflight_rejects_protocol_hash_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    hashes = _stage_sealed_tree(tmp_path)
    _write_seal(tmp_path, _staged_payload(hashes, protocol_sha256="9" * 64))
    _patch_staged_preflight(monkeypatch, tmp_path, hashes)
    with pytest.raises(RuntimeError, match="embedded protocol fingerprint"):
        MODULE._preflight(tmp_path, tmp_path / "result.json", seal="seal.json")
    # A tampered protocol file fails the on-disk verification.
    hashes = _stage_sealed_tree(tmp_path)
    (tmp_path / MODULE.PROTOCOL).write_text("tampered\n", encoding="utf-8")
    _patch_staged_preflight(monkeypatch, tmp_path, hashes)
    with pytest.raises(RuntimeError, match="sealed protocol SHA-256"):
        MODULE._preflight(tmp_path, tmp_path / "result.json", seal="seal.json")


def test_preflight_rejects_manifest_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    hashes = _stage_sealed_tree(tmp_path)
    tampered = dict(hashes["manifest"])
    tampered["src/universa/alpha.py"] = "0" * 64
    _write_seal(tmp_path, _staged_payload(hashes, code_manifest=tampered))
    _patch_staged_preflight(monkeypatch, tmp_path, hashes)
    with pytest.raises(RuntimeError, match="code_manifest"):
        MODULE._preflight(tmp_path, tmp_path / "result.json", seal="seal.json")
    # An extra on-disk module not covered by the seal also fails.
    hashes = _stage_sealed_tree(tmp_path)
    (tmp_path / "src" / "universa" / "gamma.py").write_text(
        "G = 3\n", encoding="utf-8"
    )
    _patch_staged_preflight(monkeypatch, tmp_path, hashes)
    with pytest.raises(RuntimeError, match="code_manifest"):
        MODULE._preflight(tmp_path, tmp_path / "result.json", seal="seal.json")


def test_preflight_rejects_runner_hash_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    hashes = _stage_sealed_tree(tmp_path)
    _write_seal(tmp_path, _staged_payload(hashes, runner_sha256="0" * 64))
    _patch_staged_preflight(monkeypatch, tmp_path, hashes)
    with pytest.raises(RuntimeError, match="running file's SHA-256"):
        MODULE._preflight(tmp_path, tmp_path / "result.json", seal="seal.json")


def test_preflight_refuses_a_foreign_universa_package(
    monkeypatch, tmp_path: Path
) -> None:
    # A shadow universa package — e.g. resolved from a relative PYTHONPATH
    # from a foreign working directory — must be refused even though the
    # on-disk manifest of the project's own files hashes clean.
    hashes = _stage_sealed_tree(tmp_path)
    _patch_staged_preflight(monkeypatch, tmp_path, hashes)
    foreign = tmp_path / "foreign" / "universa"
    foreign.mkdir(parents=True)
    (foreign / "__init__.py").write_text("# shadow package\n", encoding="utf-8")
    monkeypatch.setattr(universa, "__file__", str(foreign / "__init__.py"))
    with pytest.raises(RuntimeError, match="imported universa"):
        MODULE._preflight(tmp_path, tmp_path / "result.json", seal="seal.json")


def test_preflight_refuses_a_seal_swapped_under_assume_unchanged(
    monkeypatch, tmp_path: Path
) -> None:
    hashes = _stage_sealed_tree(tmp_path)
    _patch_staged_preflight(monkeypatch, tmp_path, hashes)
    # git reports a clean tree and the seal present at HEAD, but the
    # committed blob is not what is on disk (an assume-unchanged swap).
    monkeypatch.setattr(
        MODULE, "_git_head_blob", lambda _root, _rel: b'{"schema": "swapped"}\n'
    )
    with pytest.raises(RuntimeError, match="differs from the seal blob"):
        MODULE._preflight(tmp_path, tmp_path / "result.json", seal="seal.json")


def test_preflight_refuses_symlinked_seal_protocol_and_runner(
    monkeypatch, tmp_path: Path
) -> None:
    for index, relative in enumerate(
        ("seal.json", MODULE.PROTOCOL, MODULE.RUNNER_SOURCE)
    ):
        case_root = tmp_path / f"case{index}"
        hashes = _stage_sealed_tree(case_root)
        _patch_staged_preflight(monkeypatch, case_root, hashes)
        original = case_root / relative
        moved = original.with_name(original.name + ".real")
        original.rename(moved)
        original.symlink_to(moved.name)
        with pytest.raises(RuntimeError, match="symlink"):
            MODULE._preflight(
                case_root, case_root / "result.json", seal="seal.json"
            )


def test_load_seal_requires_the_design_commit_to_exist(
    monkeypatch, tmp_path: Path
) -> None:
    fabricated = "a" * 40
    _write_seal(tmp_path, _seal_payload(design_commit=fabricated))

    def fake_git(root: Path, *args: str) -> str:
        if args[0] == "cat-file" and args[-1] == fabricated + "^{commit}":
            raise RuntimeError(
                f"stop condition: git cat-file -e {fabricated}^{{commit}} "
                "failed: not a valid object name"
            )
        return _fake_git_clean(root, *args)

    monkeypatch.setattr(MODULE, "_git_checked", fake_git)
    with pytest.raises(RuntimeError, match="cat-file"):
        MODULE._load_seal(tmp_path, "seal.json")


def test_load_seal_requires_the_design_commit_to_be_an_ancestor_of_head(
    monkeypatch, tmp_path: Path
) -> None:
    # A commit that EXISTS (e.g. produced by git commit-tree) but no
    # reachable history contains is not a design commit.
    dangling = "b" * 40
    _write_seal(tmp_path, _seal_payload(design_commit=dangling))

    def fake_git(root: Path, *args: str) -> str:
        if args[0] == "merge-base":
            raise RuntimeError(
                f"stop condition: git merge-base --is-ancestor {dangling} "
                "HEAD failed: not an ancestor"
            )
        return _fake_git_clean(root, *args)

    monkeypatch.setattr(MODULE, "_git_checked", fake_git)
    with pytest.raises(RuntimeError, match="merge-base"):
        MODULE._load_seal(tmp_path, "seal.json")


def test_worktree_checks_include_untracked_files(
    monkeypatch, tmp_path: Path
) -> None:
    hashes = _stage_sealed_tree(tmp_path)
    _patch_staged_preflight(monkeypatch, tmp_path, hashes)
    calls = []

    def recording(root: Path, *args: str) -> str:
        calls.append(args)
        return _fake_git_clean(root, *args)

    monkeypatch.setattr(MODULE, "_git_checked", recording)
    MODULE._preflight(tmp_path, tmp_path / "result.json", seal="seal.json")
    assert ("status", "--porcelain", "--untracked-files=all") in calls


# ---------------------------------------------------------------------------
# Environment probes and the atomic writer.


def test_execution_environment_pins_single_thread_cpu_float32() -> None:
    env = MODULE._execution_environment()
    assert env["torch_num_threads"] == 1
    assert env["cuda_available"] is False
    assert env["tensor_device"] == "cpu"
    assert env["tensor_dtype"] == "float32"
    assert env["array_dtype"] == "float64"


def test_execution_environment_refuses_a_visible_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="CUDA"):
        MODULE._execution_environment()


def test_execution_environment_records_operator_and_effective_cuda() -> None:
    env = MODULE._execution_environment()
    assert env["cuda_visible_devices"] == ""
    assert (
        env["cuda_visible_devices_operator"]
        == MODULE.OPERATOR_CUDA_VISIBLE_DEVICES
    )


def test_preflight_refuses_a_visible_cuda_device(
    monkeypatch, tmp_path: Path
) -> None:
    hashes = _stage_sealed_tree(tmp_path)
    _patch_staged_preflight(monkeypatch, tmp_path, hashes)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="CUDA"):
        MODULE._preflight(tmp_path, tmp_path / "result.json", seal="seal.json")


def test_atomic_writer_never_overwrites_an_existing_record(
    tmp_path: Path,
) -> None:
    target = tmp_path / "result.json"
    MODULE._atomic_json_new(target, {"a": 1})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}
    with pytest.raises(RuntimeError, match="output appeared"):
        MODULE._atomic_json_new(target, {"a": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1}


# ---------------------------------------------------------------------------
# Train block construction and the three frozen trainings.


def test_train_block_uses_the_frozen_schedules(monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "TRAIN_SEEDS", (70501, 70502))
    block = MODULE._build_train_block()
    assert block.in_blocks.shape == (2, K, MODULE.FEATURE_DIM)
    assert block.in_raws.shape == (2, K, GRID_LEN)
    assert block.out_blocks.shape == (2, K, MODULE.FEATURE_DIM)
    assert block.out_raws.shape == (2, K, GRID_LEN)
    assert block.generic_features.shape == (4, K, GENERIC_FEATURE_DIM)
    assert block.generic_labels.tolist() == [0, K, 0, K]
    from universa.discovery import synthesize_observations
    from universa.generators import SwitchInstance
    from universa.loop_v2 import arch_row_features, generic_row_features

    for index, seed in enumerate((70501, 70502)):
        instance = _fixture_instance(seed)
        in_library, out_library = MODULE._paired_views(instance)
        observation_seed = subseed(seed, "router-v2-observe")
        # The arch blocks come from the frozen shared draw over the
        # equal-K views.
        expected_in, expected_in_raw = arch_row_features(
            instance, in_library, observation_seed
        )
        expected_out, _ = arch_row_features(
            instance, out_library, observation_seed
        )
        assert np.array_equal(block.in_blocks[index], expected_in)
        assert np.array_equal(block.in_raws[index], expected_in_raw)
        assert np.array_equal(block.out_blocks[index], expected_out)
        # The generic rows read the exact transported matrix at the seed's
        # operating grid point.
        switch_view = SwitchInstance(
            instance.seed,
            instance.source,
            instance.true_target,
            instance.chain_map,
            instance.decoy_targets,
        )
        observations = synthesize_observations(switch_view, 16)
        point = operating_grid_point(seed)
        assert np.array_equal(
            block.generic_features[2 * index],
            generic_row_features(
                instance, in_library, observation_seed, observations, point
            ),
        )
        assert np.array_equal(
            block.generic_features[2 * index + 1],
            generic_row_features(
                instance, out_library, observation_seed, observations, point
            ),
        )
    # The whole block is deterministic.
    again = MODULE._build_train_block()
    for first, second in (
        (block.in_blocks, again.in_blocks),
        (block.in_raws, again.in_raws),
        (block.out_blocks, again.out_blocks),
        (block.out_raws, again.out_raws),
        (block.generic_features, again.generic_features),
    ):
        assert np.array_equal(first, second)
    assert np.array_equal(block.generic_labels, again.generic_labels)


def test_train_block_has_no_train_side_eligibility_gate(
    monkeypatch, tmp_path: Path
) -> None:
    """The frozen design has NO train-side eligibility gate: every train
    seed is used as-is (build/feature exceptions excepted). Sabotaging the
    undegraded audit for the train seeds must not stop the run — the audit
    applies to the eval eligibility pass only."""
    _patch_tiny_campaign(monkeypatch)
    real_misfits = MODULE._undegraded_misfits

    def poisoned_train(instance):
        misfits = list(real_misfits(instance))
        if instance.seed in (70501, 70502):  # train seeds, not eval seeds
            misfits[0] = 1.0  # would fail ANY audit — but none runs here
        return tuple(misfits)

    monkeypatch.setattr(MODULE, "_undegraded_misfits", poisoned_train)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "complete"
    assert report["training"] is not None
    assert len(report["raw_rows"]) == 36


def test_train_block_build_exception_is_a_design_failure(monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "TRAIN_SEEDS", (70501,))

    def exploding(*_args, **_kwargs):
        raise ValueError("generator blew up")

    monkeypatch.setattr(MODULE, "make_budget_instance", exploding)
    with pytest.raises(
        MODULE.DesignFailureError, match="train block construction failed"
    ):
        MODULE._build_train_block()


def test_three_models_train_deterministically_on_a_tiny_block(
    monkeypatch,
) -> None:
    """Identical model hashes across two runs on a tiny block — all THREE
    models (router, alarm, generic) — plus identical final training
    scalars AND an identical calibration record."""
    monkeypatch.setattr(MODULE, "TRAIN_SEEDS", (70501, 70502))
    monkeypatch.setattr(MODULE, "EPOCHS", 2)
    first_block = MODULE._build_train_block()
    router_a, alarm_a, generic_a, calibration_a, prov_a = MODULE._train_models(
        first_block
    )
    second_block = MODULE._build_train_block()
    router_b, alarm_b, generic_b, calibration_b, prov_b = MODULE._train_models(
        second_block
    )
    for name in ("router", "alarm", "generic"):
        assert prov_a[name]["model_state_sha256"] == prov_b[name][
            "model_state_sha256"
        ]
        assert len(prov_a[name]["model_state_sha256"]) == 64
        assert prov_a[name]["final"] == prov_b[name]["final"]
        assert all(
            math.isfinite(value) for value in prov_a[name]["final"].values()
        )
    # The three models are distinct artifacts.
    hashes = {
        prov_a[name]["model_state_sha256"] for name in ("router", "alarm", "generic")
    }
    assert len(hashes) == 3
    # Row accounting: one router row per train seed, one fit and one no-fit
    # alarm row per train seed, two generic rows per train seed.
    assert prov_a["router"]["num_train_rows"] == 2
    assert prov_a["alarm"]["num_fit_rows"] == 2
    assert prov_a["alarm"]["num_nofit_rows"] == 2
    assert prov_a["generic"]["num_train_rows"] == 4
    assert prov_a["router"]["torch_seed"] == 4242
    assert prov_a["alarm"]["torch_seed"] == 4243
    assert prov_a["generic"]["torch_seed"] == 4244
    # The calibration record is deterministic, is returned alongside the
    # models, and is retained verbatim in the training provenance under the
    # frozen bound.
    assert calibration_a == calibration_b
    assert prov_a["alarm"]["calibration"] == calibration_a
    assert prov_b["alarm"]["calibration"] == calibration_b
    assert prov_a["alarm"]["calibration_costs"] == {
        "false_quiet_cost": 1.0,
        "false_alarm_cost": 1.0,
    }
    # The v1 frozen 0.5 threshold key is gone: the threshold is calibrated.
    assert "threshold" not in prov_a["alarm"]
    # The trained models are usable by the arm machinery (fixture types).
    from universa.router import StructureRouter

    assert isinstance(router_a, StructureRouter)
    assert isinstance(alarm_a, LearnedAlarmV2)
    assert isinstance(generic_a, GenericMLP)


def test_router_trains_on_unpermuted_in_library_rows_labeled_zero(
    monkeypatch,
) -> None:
    """The frozen index-0 convention: the router's training rows are the
    UNPERMUTED in-library arch blocks with label 0 for every row; the
    out-of-library view contributes no router rows."""
    monkeypatch.setattr(MODULE, "TRAIN_SEEDS", (70501, 70502))
    monkeypatch.setattr(MODULE, "EPOCHS", 2)
    captured = {}
    real_train_router = MODULE.train_router

    def capturing(train_dataset, val_dataset, *args, **kwargs):
        captured["features"], captured["labels"], _ = train_dataset
        return real_train_router(train_dataset, val_dataset, *args, **kwargs)

    monkeypatch.setattr(MODULE, "train_router", capturing)
    block = MODULE._build_train_block()
    MODULE._train_models(block)
    assert np.array_equal(captured["features"], block.in_blocks)
    assert captured["labels"].tolist() == [0, 0]


# ---------------------------------------------------------------------------
# The four arms' correctness semantics on fixtures (pinned trained models).


def test_arch_arm_v2_routes_discovers_and_refuses_on_fixtures(
    trained_models,
) -> None:
    """Pinned fixture behavior of the full architecture under the v2 alarm
    (models trained on the sanctioned fixture block 70501..70520, threshold
    calibrated on that block): the calibrated learned alarm stays quiet
    in-library (route to the true target at index 0), fires out-of-library
    (certified discovery acquires with map_misfit <= 1e-9), and the null
    control is refused."""
    calibration = trained_models[3]
    assert 0.0 <= calibration["threshold"] <= 1.0
    for seed in FIXTURE_SEEDS:
        instance = _fixture_instance(seed)
        row = _arm_row(seed, instance, "in_library", "arch_full", trained_models)
        assert row["action"] == "route"
        assert row["routed_index"] == 0
        assert row["correct"] is True
        assert row["discovery_invocations"] == 0
        assert row["admitted"] is False
        assert row["map_misfit"] is None
        assert row["initial_library_size"] == K == row["final_library_size"]
        assert row["detail"].startswith("alarm_v2=fit")
        assert len(row["arch_raw_profiles"]) == K
        assert all(len(profile) == GRID_LEN for profile in row["arch_raw_profiles"])
        row = _arm_row(
            seed, instance, "out_of_library", "arch_full", trained_models
        )
        assert row["action"] == "discover"
        assert row["routed_index"] == K  # the appended structure's index
        assert row["correct"] is True
        assert row["discovery_invocations"] == 1
        assert row["admitted"] is True
        assert row["map_misfit"] is not None
        assert row["map_misfit"] <= MAP_ACCEPT_TOL
        assert row["final_library_size"] == K + 1
        assert "alarm_v2=no-fit" in row["detail"]
        row = _arm_row(seed, instance, "null_control", "arch_full", trained_models)
        assert row["action"] == "refused"
        assert row["routed_index"] is None
        assert row["admitted"] is False
        assert row["correct"] is True  # the false-admission control
        assert row["discovery_invocations"] == 1
        assert row["final_library_size"] == K
        # The audit recomputes every one of these decisions from the row.
        for condition in MODULE.CONDITIONS:
            entry = MODULE._audit_row_decisions(
                _arm_row(seed, instance, condition, "arch_full", trained_models)
            )
            assert entry["correct"] is True


def test_routing_only_forced_choice_fails_out_of_library(
    trained_models,
) -> None:
    for seed in FIXTURE_SEEDS:
        instance = _fixture_instance(seed)
        row = _arm_row(
            seed, instance, "in_library", "routing_only", trained_models
        )
        assert row["action"] == "route"
        assert row["routed_index"] == 0
        assert row["correct"] is True
        assert row["discovery_invocations"] == 0
        row = _arm_row(
            seed, instance, "out_of_library", "routing_only", trained_models
        )
        # No alarm, no discovery: a forced choice, so out-of-library it
        # must pick a decoy — an honest failure.
        assert row["action"] == "route"
        assert 0 <= row["routed_index"] < K
        assert row["correct"] is False
        assert row["admitted"] is False
        assert row["discovery_invocations"] == 0
        row = _arm_row(
            seed, instance, "null_control", "routing_only", trained_models
        )
        assert row["correct"] is True  # nothing admitted (no channel)
        assert row["admitted"] is False


def test_discovery_only_semantics(trained_models) -> None:
    for seed in FIXTURE_SEEDS:
        instance = _fixture_instance(seed)
        for condition in ("in_library", "out_of_library"):
            row = _arm_row(seed, instance, condition, "discovery_only", trained_models)
            assert row["action"] == "discover"
            assert row["correct"] is True
            assert row["admitted"] is True
            assert row["discovery_invocations"] == 1
            assert row["map_misfit"] is not None
            assert row["map_misfit"] <= MAP_ACCEPT_TOL
            # Novelty is checked against the decoy library; the appended
            # structure sits at index len(decoys).
            assert row["initial_library_size"] == MODULE.NUM_DECOYS
            assert row["routed_index"] == MODULE.NUM_DECOYS
            assert row["final_library_size"] == MODULE.NUM_DECOYS + 1
        row = _arm_row(seed, instance, "null_control", "discovery_only", trained_models)
        assert row["action"] == "refused"
        assert row["correct"] is True  # refusal is the specificity
        assert row["admitted"] is False
        assert row["routed_index"] is None
        assert row["map_misfit"] is None
        assert row["final_library_size"] == MODULE.NUM_DECOYS


def test_generic_arm_no_fit_and_refusal_semantics(trained_models) -> None:
    router, alarm, _, calibration = trained_models
    route_stub = _constant_generic_model(0)
    nofit_stub = _constant_generic_model(K)
    for seed in FIXTURE_SEEDS:
        instance = _fixture_instance(seed)
        inputs = MODULE._condition_inputs(seed, instance)

        def stub_row(condition, model):
            library, observations = inputs[condition]
            return MODULE._build_arm_row(
                seed,
                instance,
                library,
                observations,
                condition=condition,
                arm="generic",
                observation_seed=MODULE._observation_seed(seed),
                router=router,
                alarm=alarm,
                threshold=calibration["threshold"],
                generic=model,
            )

        routed = stub_row("in_library", route_stub)
        assert routed["action"] == "route" and routed["routed_index"] == 0
        assert routed["correct"] is True  # the index-0 certified gate
        assert routed["discovery_invocations"] == 0
        forced = stub_row("out_of_library", route_stub)
        assert forced["action"] == "route" and forced["correct"] is False
        noise_routed = stub_row("null_control", route_stub)
        assert noise_routed["action"] == "route"
        assert noise_routed["correct"] is False  # refusal is the specificity
        refused = stub_row("null_control", nofit_stub)
        assert refused["action"] == "refused" and refused["correct"] is True
        assert refused["routed_index"] is None and refused["admitted"] is False
        assert "no-fit" in refused["detail"]
        nofit_out = stub_row("out_of_library", nofit_stub)
        # Nothing is ever synthesized by this arm: out-of-library it
        # cannot acquire, even when it correctly says no-fit.
        assert nofit_out["action"] == "refused"
        assert nofit_out["correct"] is False
        in_refused = stub_row("in_library", nofit_stub)
        assert in_refused["action"] == "refused"
        assert in_refused["correct"] is False
        # The audit agrees with every stub row's retained bit.
        for condition in MODULE.CONDITIONS:
            for model in (route_stub, nofit_stub):
                entry = MODULE._audit_row_decisions(stub_row(condition, model))
                assert isinstance(entry["correct"], bool)


def test_arm_rows_are_deterministic_end_to_end(trained_models) -> None:
    for seed in FIXTURE_SEEDS[:2]:
        instance = _fixture_instance(seed)
        for condition in MODULE.CONDITIONS:
            for arm in MODULE.ARMS:
                first = _arm_row(seed, instance, condition, arm, trained_models)
                second = _arm_row(seed, instance, condition, arm, trained_models)
                assert MODULE._bit_identity_fields(
                    first
                ) == MODULE._bit_identity_fields(second)


# ---------------------------------------------------------------------------
# The frozen correctness semantics, both sides (hand fixtures).


def test_recomputed_correct_both_sides_of_the_frozen_semantics() -> None:
    check = MODULE._recomputed_correct
    # null_control: the alarm arms must admit nothing; the refusal-only
    # arms must refuse.
    assert check("arch_full", "null_control", action="route", routed_index=0, admitted=False, map_misfit=None) is True
    assert check("arch_full", "null_control", action="discover", routed_index=3, admitted=True, map_misfit=0.0) is False
    assert check("routing_only", "null_control", action="route", routed_index=2, admitted=False, map_misfit=None) is True
    assert check("routing_only", "null_control", action="route", routed_index=2, admitted=True, map_misfit=None) is False
    assert check("discovery_only", "null_control", action="refused", routed_index=None, admitted=False, map_misfit=None) is True
    assert check("discovery_only", "null_control", action="discover", routed_index=3, admitted=True, map_misfit=0.0) is False
    assert check("generic", "null_control", action="refused", routed_index=None, admitted=False, map_misfit=None) is True
    assert check("generic", "null_control", action="route", routed_index=0, admitted=False, map_misfit=None) is False
    # in_library: the routing arms must route to index 0; discovery_only
    # needs the certified gate.
    assert check("arch_full", "in_library", action="route", routed_index=0, admitted=False, map_misfit=None) is True
    assert check("arch_full", "in_library", action="route", routed_index=1, admitted=False, map_misfit=None) is False
    assert check("arch_full", "in_library", action="refused", routed_index=None, admitted=False, map_misfit=None) is False
    assert check("routing_only", "in_library", action="route", routed_index=0, admitted=False, map_misfit=None) is True
    assert check("generic", "in_library", action="route", routed_index=0, admitted=False, map_misfit=None) is True
    assert check("generic", "in_library", action="refused", routed_index=None, admitted=False, map_misfit=None) is False
    assert check("discovery_only", "in_library", action="discover", routed_index=3, admitted=True, map_misfit=1e-10) is True
    # The certified gate is inclusive at exactly 1e-9 and closed above it.
    assert check("discovery_only", "in_library", action="discover", routed_index=3, admitted=True, map_misfit=MAP_ACCEPT_TOL) is True
    assert check("discovery_only", "in_library", action="discover", routed_index=3, admitted=True, map_misfit=1e-8) is False
    assert check("discovery_only", "in_library", action="refused", routed_index=None, admitted=False, map_misfit=None) is False
    # out_of_library: only arms with a discovery channel can acquire.
    assert check("arch_full", "out_of_library", action="discover", routed_index=3, admitted=True, map_misfit=1e-10) is True
    assert check("arch_full", "out_of_library", action="discover", routed_index=3, admitted=True, map_misfit=1e-8) is False
    assert check("arch_full", "out_of_library", action="refused", routed_index=None, admitted=False, map_misfit=1e-10) is False
    assert check("arch_full", "out_of_library", action="route", routed_index=0, admitted=False, map_misfit=None) is False
    assert check("routing_only", "out_of_library", action="route", routed_index=0, admitted=False, map_misfit=None) is False
    assert check("generic", "out_of_library", action="refused", routed_index=None, admitted=False, map_misfit=None) is False
    assert check("discovery_only", "out_of_library", action="discover", routed_index=3, admitted=True, map_misfit=MAP_ACCEPT_TOL) is True
    assert check("discovery_only", "out_of_library", action="discover", routed_index=3, admitted=True, map_misfit=2e-9) is False


def test_audit_recompute_fails_closed_on_tampered_rows(trained_models) -> None:
    instance = _fixture_instance(70003)
    in_row = _arm_row(70003, instance, "in_library", "arch_full", trained_models)
    assert in_row["action"] == "route" and in_row["correct"] is True
    out_row = _arm_row(70003, instance, "out_of_library", "arch_full", trained_models)
    assert out_row["action"] == "discover" and out_row["correct"] is True
    # A tampered correctness bit contradicts the recompute.
    tampered = dict(in_row)
    tampered["correct"] = False
    with pytest.raises(MODULE.DesignFailureError, match="seed 70003"):
        MODULE._audit_row_decisions(tampered)
    # A tampered routed index flips the recomputed routing decision.
    tampered = dict(in_row)
    tampered["routed_index"] = 1
    with pytest.raises(MODULE.DesignFailureError, match="seed 70003"):
        MODULE._audit_row_decisions(tampered)
    # A tampered map misfit breaks the certified gate.
    tampered = dict(out_row)
    tampered["map_misfit"] = 1.0
    with pytest.raises(MODULE.DesignFailureError, match="seed 70003"):
        MODULE._audit_row_decisions(tampered)
    # A dropped map misfit breaks discover-mode coherence.
    tampered = dict(out_row)
    tampered["map_misfit"] = None
    with pytest.raises(MODULE.DesignFailureError, match="seed 70003"):
        MODULE._audit_row_decisions(tampered)
    # A tampered observation seed breaks the frozen schedule.
    tampered = dict(in_row)
    tampered["observation_seed"] = in_row["observation_seed"] + 1
    with pytest.raises(MODULE.DesignFailureError, match="schedule"):
        MODULE._audit_row_decisions(tampered)
    # Tampered retention blocks are refused.
    tampered = dict(in_row)
    tampered["arch_raw_profiles"] = in_row["arch_raw_profiles"][:2]
    with pytest.raises(MODULE.DesignFailureError, match="raw profiles"):
        MODULE._audit_row_decisions(tampered)
    tampered = dict(in_row)
    tampered["arch_raw_profiles"] = [
        list(profile) for profile in in_row["arch_raw_profiles"]
    ]
    tampered["arch_raw_profiles"][0][0] = -1.0
    with pytest.raises(MODULE.DesignFailureError, match="raw profiles"):
        MODULE._audit_row_decisions(tampered)
    # Unknown names and incoherent records are refused.
    for key, value, match in (
        ("condition", "sideways", "unknown condition"),
        ("arm", "teleport", "unknown arm"),
        ("action", "teleport", "unknown action"),
        ("initial_library_size", 4, "frozen K"),
    ):
        tampered = dict(in_row)
        tampered[key] = value
        with pytest.raises(MODULE.DesignFailureError, match=match):
            MODULE._audit_row_decisions(tampered)
    # A route row carrying a discovery invocation is incoherent; a
    # discovery_only row without one is too.
    tampered = dict(in_row)
    tampered["discovery_invocations"] = 1
    with pytest.raises(MODULE.DesignFailureError, match="seed 70003"):
        MODULE._audit_row_decisions(tampered)
    discovery_row = _arm_row(
        70003, instance, "in_library", "discovery_only", trained_models
    )
    tampered = dict(discovery_row)
    tampered["discovery_invocations"] = 0
    with pytest.raises(MODULE.DesignFailureError, match="seed 70003"):
        MODULE._audit_row_decisions(tampered)


# ---------------------------------------------------------------------------
# Eligibility: the build and the undegraded audit.


def test_eligibility_real_fixture_seeds_all_pass(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(
        monkeypatch, eval_seeds=FIXTURE_SEEDS, min_eligible=6
    )
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    # Five real eligible seeds, but the frozen minimum is six in this patch.
    assert report["status"] == "design_failure_insufficient_eligible"
    assert report["eligibility"]["declared"] == 5
    assert report["eligibility"]["eligible"] == 5
    assert report["eligibility"]["eligible_seeds"] == list(FIXTURE_SEEDS)
    assert report["eligibility"]["ineligible"] == []
    assert [record["seed"] for record in report["seed_records"]] == list(
        FIXTURE_SEEDS
    )
    for record in report["seed_records"]:
        assert record["eligible"] is True
        assert record["reason"] is None
        assert record["true_misfit_undegraded"] == 0.0  # exact, never tolerated
        assert min(record["decoy_misfits_undegraded"]) > MODULE.AUDIT_TOL


def test_undegraded_audit_failure_marks_seed_ineligible_with_reason(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(
        monkeypatch, eval_seeds=(70003, 70004, 70005), min_eligible=3
    )
    real_misfits = MODULE._undegraded_misfits

    def poisoned(instance):
        misfits = list(real_misfits(instance))
        if instance.seed == 70004:
            misfits[1] = 0.0  # a decoy that admits the truth
        return tuple(misfits)

    monkeypatch.setattr(MODULE, "_undegraded_misfits", poisoned)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure_insufficient_eligible"
    assert report["eligibility"]["eligible_seeds"] == [70003, 70005]
    ineligible = report["eligibility"]["ineligible"]
    assert len(ineligible) == 1
    assert ineligible[0]["seed"] == 70004
    assert "decoy undegraded misfit" in ineligible[0]["reason"]
    record = next(
        record
        for record in report["seed_records"]
        if record["seed"] == 70004
    )
    assert record["eligible"] is False
    assert record["reason"] == ineligible[0]["reason"]
    # No training ran and no rows were built.
    assert report["training"] is None
    assert report["raw_rows"] == []


def test_true_misfit_nonzero_marks_seed_ineligible(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(
        monkeypatch, eval_seeds=(70003, 70004, 70005), min_eligible=3
    )
    real_misfits = MODULE._undegraded_misfits

    def poisoned(instance):
        misfits = list(real_misfits(instance))
        if instance.seed == 70004:
            misfits[0] = 1e-12  # the true candidate must be EXACTLY 0.0
        return tuple(misfits)

    monkeypatch.setattr(MODULE, "_undegraded_misfits", poisoned)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure_insufficient_eligible"
    assert report["eligibility"]["eligible_seeds"] == [70003, 70005]
    assert "!= 0.0" in report["eligibility"]["ineligible"][0]["reason"]


def test_build_exception_is_a_whole_run_design_failure(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(
        monkeypatch, eval_seeds=(70003, 70004, 70005), min_eligible=1
    )
    real_make = MODULE.make_budget_instance

    def exploding(seed, *args, **kwargs):
        if seed == 70004:
            raise ValueError("generator blew up")
        return real_make(seed, *args, **kwargs)

    monkeypatch.setattr(MODULE, "make_budget_instance", exploding)
    monkeypatch.setattr(
        MODULE,
        "_build_train_block",
        lambda: pytest.fail("training must not run after a build failure"),
    )
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure"
    assert report["failure"]["seed"] == 70004
    assert report["failure"]["phase"] == "eligibility"
    assert report["failure"]["type"] == "ValueError"
    assert "generator blew up" in report["failure"]["message"]
    assert report["eligibility"]["build_failures"] == [
        {"seed": 70004, "type": "ValueError", "message": "generator blew up"}
    ]
    assert report["eligibility"]["attempted"] == 2
    # Seed 70003 was fully audited before the failure and is preserved.
    assert [record["seed"] for record in report["seed_records"]] == [70003]
    assert report["training"] is None
    assert report["raw_rows"] == []
    assert report["per_seed"] == []
    claims = report["claims"]["claims"]
    assert len(claims) == 4
    assert all(claim["supported"] is None for claim in claims)


def test_eligibility_pass_failures_enter_the_status_vocabulary(
    monkeypatch, tmp_path: Path
) -> None:
    real_misfits = MODULE._undegraded_misfits
    cases = (
        (RuntimeError("audit exploded"), "execution_failure"),
        (
            MODULE.DesignFailureError("audit rejected the seed", seed=70004),
            "design_failure",
        ),
        (KeyboardInterrupt(), "interrupted"),
    )
    for index, (raised, expected_status) in enumerate(cases):
        _patch_tiny_campaign(
            monkeypatch, eval_seeds=(70003, 70004, 70005), min_eligible=1
        )

        def exploding(instance, _raised=raised):
            if instance.seed == 70004:
                raise _raised
            return real_misfits(instance)

        monkeypatch.setattr(MODULE, "_undegraded_misfits", exploding)
        report = MODULE.run(
            tmp_path, tmp_path / f"out-{index}.json", seal="seal.json"
        )
        assert report["status"] == expected_status
        assert report["failure"]["phase"] == "eligibility"
        assert report["failure"]["type"] == type(raised).__name__
        # Seed 70003 was fully audited before the failure and is preserved.
        assert [record["seed"] for record in report["seed_records"]] == [70003]
        assert report["training"] is None
        assert report["raw_rows"] == []
        assert report["per_seed"] == []
        claims = report["claims"]["claims"]
        assert len(claims) == 4
        assert all(claim["supported"] is None for claim in claims)


def test_insufficient_eligible_runs_no_training_and_nulls_claims(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(
        monkeypatch, eval_seeds=(70003, 70004, 70005), min_eligible=4
    )
    monkeypatch.setattr(
        MODULE,
        "_build_train_block",
        lambda: pytest.fail("training must not run after eligibility failure"),
    )
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure_insufficient_eligible"
    assert report["eligibility"]["eligible"] == 3
    assert report["training"] is None
    assert report["raw_rows"] == []
    assert report["per_seed"] == []
    claims = report["claims"]["claims"]
    assert len(claims) == 4
    assert {claim["id"] for claim in claims} == set(MODULE.CLAIM_IDS)
    assert all(claim["supported"] is None for claim in claims)
    assert all(claim["estimate"] is None for claim in claims)
    assert report["descriptive"] is None
    assert report["audit"]["raw_rows"] == 0
    assert "claims" not in report["audit"]


def test_run_never_constructs_data_when_preflight_fails(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        MODULE,
        "_preflight",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("dirty")
        ),
    )
    monkeypatch.setattr(
        MODULE,
        "make_budget_instance",
        lambda *_args, **_kwargs: pytest.fail(
            "an instance was built before preflight passed"
        ),
    )
    with pytest.raises(RuntimeError, match="dirty"):
        MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")


# ---------------------------------------------------------------------------
# Complete tiny campaigns: structure, determinism, audit recompute.


def test_complete_tiny_campaign_structure_and_audit_recompute(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "complete"
    assert report["schema"] == "universa-router-loop-v4-sealed-result/1"
    assert report["experiment_id"] == "universa-loop-v4-sealed-1"
    assert report["eligibility"]["eligible_seeds"] == [70003, 70004, 70005]
    # The TOCTOU guard records the post-run status next to the pre-run one.
    assert report["provenance"]["git_status_porcelain"] == ""
    assert report["provenance"]["git_status_porcelain_post_run"] == ""
    # The whole report must be JSON-serializable (no numpy leaks).
    json.dumps(report, sort_keys=True)

    rows = report["raw_rows"]
    assert len(rows) == 4 * 3 * 3  # 4 arms x 3 conditions x 3 seeds
    for row in rows:
        assert set(row) == {
            "seed",
            "condition",
            "arm",
            "correct",
            "action",
            "routed_index",
            "discovery_invocations",
            "admitted",
            "map_misfit",
            "detail",
            "initial_library_size",
            "final_library_size",
            "observation_seed",
            "arch_raw_profiles",
            "wall_time_seconds",
        }
        assert row["seed"] in (70003, 70004, 70005)
        assert row["condition"] in MODULE.CONDITIONS
        assert row["arm"] in MODULE.ARMS
        assert isinstance(row["correct"], bool)
        assert row["action"] in ("route", "discover", "refused")
        assert isinstance(row["admitted"], bool)
        assert row["discovery_invocations"] in (0, 1)
        assert row["initial_library_size"] == K
        assert row["observation_seed"] == subseed(row["seed"], "router-v2-observe")
        assert len(row["arch_raw_profiles"]) == K
        assert all(
            len(profile) == GRID_LEN for profile in row["arch_raw_profiles"]
        )
        assert isinstance(row["wall_time_seconds"], float)
        assert row["wall_time_seconds"] >= 0.0
    # Pairing structure: the four arms of one (seed, condition) share the
    # observation draw AND the retained arch raw profiles; all twelve rows
    # of one seed share the seed's single shared draw.
    groups: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["seed"], row["condition"]), []).append(row)
    assert len(groups) == 9
    for group_rows in groups.values():
        assert len(group_rows) == len(MODULE.ARMS)
        assert {row["arm"] for row in group_rows} == set(MODULE.ARMS)
        assert len({row["observation_seed"] for row in group_rows}) == 1
        assert len(
            {
                json.dumps(row["arch_raw_profiles"], sort_keys=True)
                for row in group_rows
            }
        ) == 1
    for seed in (70003, 70004, 70005):
        assert len(
            {
                row["observation_seed"]
                for row in rows
                if row["seed"] == seed
            }
        ) == 1
        # The declared convention: the arch observation record is identical
        # in the out-of-library and null-control conditions per seed (the
        # alarm's decision basis precedes observations); the in-library
        # view differs (the truth replaces a decoy).
        for arm in ("arch_full", "routing_only"):
            out_row = next(
                row for row in rows
                if row["seed"] == seed
                and row["condition"] == "out_of_library"
                and row["arm"] == arm
            )
            null_row = next(
                row for row in rows
                if row["seed"] == seed
                and row["condition"] == "null_control"
                and row["arm"] == arm
            )
            in_row = next(
                row for row in rows
                if row["seed"] == seed
                and row["condition"] == "in_library"
                and row["arm"] == arm
            )
            assert out_row["arch_raw_profiles"] == null_row["arch_raw_profiles"]
            assert in_row["arch_raw_profiles"] != out_row["arch_raw_profiles"]

    per_seed = report["per_seed"]
    assert len(per_seed) == 3
    for entry in per_seed:
        assert set(entry) == {
            "seed",
            "arch_full_e2e",
            "routing_only_e2e",
            "discovery_only_e2e",
            "generic_e2e",
            "arch_full_in_library",
            "routing_only_in_library",
            "discovery_only_in_library",
            "generic_in_library",
            "d_arch_generic_e2e",
            "d_arch_routing_only_e2e",
            "d_arch_generic_inlibrary",
            "d_arch_discovery_only_inlibrary_harm",
        }
        for arm in MODULE.ARMS:
            assert 0.0 <= entry[f"{arm}_e2e"] <= 1.0
            assert entry[f"{arm}_in_library"] in (0.0, 1.0)
        assert entry["d_arch_generic_e2e"] == pytest.approx(
            entry["arch_full_e2e"] - entry["generic_e2e"]
        )
        assert entry["d_arch_routing_only_e2e"] == pytest.approx(
            entry["arch_full_e2e"] - entry["routing_only_e2e"]
        )
        assert entry["d_arch_generic_inlibrary"] == pytest.approx(
            entry["arch_full_in_library"] - entry["generic_in_library"]
        )
        assert entry["d_arch_discovery_only_inlibrary_harm"] == pytest.approx(
            entry["arch_full_in_library"] - entry["discovery_only_in_library"]
        )

    claims = {claim["id"]: claim for claim in report["claims"]["claims"]}
    assert set(claims) == set(MODULE.CLAIM_IDS)
    for claim_id, claim in claims.items():
        assert claim["n"] == 3
        assert isinstance(claim["supported"], bool)
        assert claim["threshold"] == _definition(claim_id)["threshold"]
        assert claim["statistic"] == _definition(claim_id)["statistic"]
        assert claim["arms"] == list(_definition(claim_id)["arms"])
        assert claim["scope"] == _definition(claim_id)["scope"]
        assert _definition(claim_id)["statistic"] in claim["estimand"]
        assert claim["lower_bound"] == pytest.approx(
            claim["estimate"]
            - claim["critical_value"] * claim["standard_error"]
        )

    audit = report["audit"]
    assert audit["declared_seeds"] == 3
    assert audit["eligible_seeds"] == 3
    assert audit["raw_rows"] == len(rows)
    assert audit["rows_per_seed"] == 12
    assert audit["arms"] == list(MODULE.ARMS)
    assert audit["conditions"] == list(MODULE.CONDITIONS)
    assert audit["per_seed"] == per_seed
    assert "claim_estimates" not in audit
    recompute = audit["decision_recompute"]
    assert len(recompute) == len(rows)
    by_key = {
        (entry["seed"], entry["condition"], entry["arm"]): entry
        for entry in recompute
    }
    for row in rows:
        entry = by_key[(row["seed"], row["condition"], row["arm"])]
        assert entry["action"] == row["action"]
        assert entry["correct"] == row["correct"]
    for definition in MODULE._CLAIM_DEFINITIONS:
        estimate = statistics.fmean(
            entry[definition["difference"]] for entry in per_seed
        )
        audit_claim = audit["claims"][definition["id"]]
        assert audit_claim["estimate"] == pytest.approx(estimate)
        assert claims[definition["id"]]["estimate"] == pytest.approx(estimate)
        assert audit_claim["standard_error"] == pytest.approx(
            claims[definition["id"]]["standard_error"]
        )
        assert audit_claim["lower_bound"] == pytest.approx(
            claims[definition["id"]]["lower_bound"]
        )
        assert audit_claim["supported"] == claims[definition["id"]]["supported"]

    # The descriptive block: reported, never claim-tested; the audit's
    # recompute equals the result's block (same raw rows).
    descriptive = report["descriptive"]
    assert audit["descriptive"] == descriptive
    assert set(descriptive["accuracy_per_condition_arm"]) == {
        f"{condition}/{arm}"
        for condition in MODULE.CONDITIONS
        for arm in MODULE.ARMS
    }
    for condition in MODULE.CONDITIONS:
        for arm in MODULE.ARMS:
            key = f"{condition}/{arm}"
            expected_accuracy = statistics.fmean(
                1.0 if row["correct"] else 0.0
                for row in rows
                if row["condition"] == condition and row["arm"] == arm
            )
            assert descriptive["accuracy_per_condition_arm"][key] == (
                pytest.approx(expected_accuracy)
            )
            assert descriptive["discovery_invocations_per_condition_arm"][
                key
            ] == sum(
                row["discovery_invocations"]
                for row in rows
                if row["condition"] == condition and row["arm"] == arm
            )
    wall = descriptive["wall_time_seconds"]
    assert set(wall["per_arm"]) == set(MODULE.ARMS)
    for arm in MODULE.ARMS:
        times = [
            row["wall_time_seconds"] for row in rows if row["arm"] == arm
        ]
        assert wall["per_arm"][arm]["mean"] == pytest.approx(
            statistics.fmean(times)
        )
        assert wall["per_arm"][arm]["min"] == min(times)
        assert wall["per_arm"][arm]["max"] == max(times)

    # The frozen three-point frontier baseline (docs/24 and docs/27, quoted
    # verbatim) and the duplicated calibration record, both descriptive-only.
    baseline = descriptive["frontier_baseline"]
    v2 = baseline["loop_v2"]
    assert v2["source"] == "docs/24-router-loop-v2-sealed-1-results.md"
    assert v2["accuracy_per_condition"] == {
        "in_library": 0.9722222222222222,
        "out_of_library": 0.8333333333333334,
        "null_control": 1.0,
    }
    assert v2["discovery_invocations"] == {
        "arch_full_total": 61,
        "in_library": 1,
        "out_of_library": 30,
        "null_control": 30,
        "discovery_only_total": 108,
    }
    assert v2["bounded_harm_estimate"] == pytest.approx(-1.0 / 36.0)
    assert v2["harm_seed_count"] == 1
    v3 = baseline["loop_v3"]
    assert v3["source"] == "docs/27-router-loop-v3-sealed-1-results.md"
    assert v3["accuracy_per_condition"] == {
        "in_library": 0.6111111111111112,
        "out_of_library": 1.0,
        "null_control": 1.0,
    }
    assert v3["discovery_invocations"] == {
        "arch_full_total": 86,
        "in_library": 14,
        "out_of_library": 36,
        "null_control": 36,
        "discovery_only_total": 108,
    }
    assert v3["bounded_harm_estimate"] == pytest.approx(-14.0 / 36.0)
    assert v3["harm_seed_count"] == 14
    # loop-v3's calibration record, carried as the bounded-rule reference
    # point: its bound was binding at a 0.41 false-alarm rate.
    assert v3["calibration"]["threshold"] == 0.889712393283844
    assert v3["calibration"]["false_quiet_rate"] == 0.02
    assert v3["calibration"]["false_alarm_rate"] == 0.41
    assert v3["calibration"]["bound_satisfied"] is True
    assert "descriptive only" in baseline["note"]
    assert descriptive["calibration"] == report["training"]["alarm"][
        "calibration"
    ]
    margin_names = report["training"]["alarm"]["margin_feature_names"]
    assert margin_names == list(MODULE.ALARM_V2_MARGIN_FEATURE_NAMES)
    assert len(margin_names) == 5

    training = report["training"]
    assert set(training) == {"router", "alarm", "generic"}
    assert training["router"]["torch_seed"] == 4242
    assert training["alarm"]["torch_seed"] == 4243
    assert training["generic"]["torch_seed"] == 4244
    assert training["router"]["epochs"] == 2  # the monkeypatched tiny value
    assert training["router"]["num_train_rows"] == 2  # 2 train seeds
    assert training["alarm"]["num_fit_rows"] == 2
    assert training["alarm"]["num_nofit_rows"] == 2
    assert training["generic"]["num_train_rows"] == 4
    assert set(training["router"]["final"]) == {
        "loss",
        "cross_entropy",
        "aux_loss",
        "tau",
        "train_accuracy",
        "val_accuracy",
        "gate_entropy",
    }
    for name in ("router", "alarm", "generic"):
        assert len(training[name]["model_state_sha256"]) == 64
        assert all(
            math.isfinite(value) for value in training[name]["final"].values()
        )


def test_complete_tiny_campaign_is_deterministic_including_model_hashes(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    first = MODULE.run(tmp_path, tmp_path / "a.json", seal="seal.json")
    second = MODULE.run(tmp_path, tmp_path / "b.json", seal="seal.json")
    assert first["status"] == second["status"] == "complete"
    # The eval rows are bit-identical in every field except
    # wall_time_seconds.
    strip = MODULE._bit_identity_fields
    assert [strip(row) for row in first["raw_rows"]] == [
        strip(row) for row in second["raw_rows"]
    ]
    assert first["per_seed"] == second["per_seed"]
    assert first["claims"] == second["claims"]
    # The audit and the descriptive block are bit-identical except the
    # wall-time summaries (execution metadata, never claim-tested).
    def strip_wall_times(block: dict) -> dict:
        stripped = dict(block)
        descriptive = dict(stripped.get("descriptive") or {})
        descriptive.pop("wall_time_seconds", None)
        stripped["descriptive"] = descriptive
        return stripped

    assert strip_wall_times(first["audit"]) == strip_wall_times(second["audit"])
    for block in (first["descriptive"], second["descriptive"]):
        assert set(block["wall_time_seconds"]["per_arm"]) == set(MODULE.ARMS)
    without_wall_first = dict(first["descriptive"])
    without_wall_second = dict(second["descriptive"])
    without_wall_first.pop("wall_time_seconds")
    without_wall_second.pop("wall_time_seconds")
    assert without_wall_first == without_wall_second
    # ALL THREE model hashes and final training scalars are reproducible.
    for name in ("router", "alarm", "generic"):
        assert (
            first["training"][name]["model_state_sha256"]
            == second["training"][name]["model_state_sha256"]
        )
        assert (
            first["training"][name]["final"]
            == second["training"][name]["final"]
        )
    # The full eight-key calibration record is reproducible across runs too.
    assert (
        first["training"]["alarm"]["calibration"]
        == second["training"]["alarm"]["calibration"]
    )


def test_tiny_campaign_instantiates_only_declared_test_seeds(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    seen = []
    real_make = MODULE.make_budget_instance

    def recording(seed, *args, **kwargs):
        seen.append(seed)
        return real_make(seed, *args, **kwargs)

    # Both the eval eligibility pass and the runner-built train block
    # construct instances through the runner module.
    monkeypatch.setattr(MODULE, "make_budget_instance", recording)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "complete"
    assert set(seen) == {70003, 70004, 70005, 70501, 70502}
    # No instantiated seed belongs to ANY sealed block — in particular not
    # to the runner's own blocks 200001..200400 / 210001..210036.
    for block in ALL_SEALED_BLOCKS:
        assert not set(seen) & set(block)


def test_nondeterministic_rows_are_a_whole_run_design_failure(
    monkeypatch, tmp_path: Path
) -> None:
    # The frozen double execution: the two executions of a row must be
    # bit-identical (every field except wall_time_seconds), else the whole
    # run is a design failure.
    _patch_tiny_campaign(monkeypatch)
    real_build = MODULE._build_arm_row
    calls = {"count": 0}

    def drifting(seed, instance, library, observations, **kwargs):
        row = real_build(seed, instance, library, observations, **kwargs)
        calls["count"] += 1
        if seed == 70003 and calls["count"] == 2:
            row = dict(row)
            row["routed_index"] = (
                None if row["routed_index"] is not None else 0
            )  # a second, different draw
        return row

    monkeypatch.setattr(MODULE, "_build_arm_row", drifting)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure"
    assert report["failure"]["seed"] == 70003
    assert report["failure"]["phase"] == "campaign"
    assert "bit-identical" in report["failure"]["message"]
    assert all(
        claim["supported"] is None for claim in report["claims"]["claims"]
    )


# ---------------------------------------------------------------------------
# Failure paths: statuses, preserved rows, null claims.


def test_execution_failure_preserves_rows_and_nulls_claims(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    real_build = MODULE._build_arm_row

    def flaky(seed, instance, library, observations, **kwargs):
        if seed == 70004 and kwargs["condition"] == "out_of_library":
            raise FloatingPointError("unexpected numeric fault")
        return real_build(seed, instance, library, observations, **kwargs)

    monkeypatch.setattr(MODULE, "_build_arm_row", flaky)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "execution_failure"
    assert report["failure"]["seed"] == 70004
    assert report["failure"]["phase"] == "campaign"
    assert report["failure"]["type"] == "FloatingPointError"
    assert "unexpected numeric fault" in report["failure"]["message"]
    # Seed 70003 completed all twelve rows; seed 70004's in-library rows
    # completed and are preserved (a completed row is appended as soon as
    # its bit-identity check passes, before the seed's later conditions);
    # 70005 never ran.
    assert [row["seed"] for row in report["raw_rows"]] == [70003] * 12 + [
        70004
    ] * 4
    assert [entry["seed"] for entry in report["per_seed"]] == [70003]
    claims = report["claims"]["claims"]
    assert len(claims) == 4
    assert all(claim["supported"] is None for claim in claims)
    assert all(claim["estimate"] is None for claim in claims)
    assert report["training"] is not None  # completed training is preserved
    assert report["descriptive"] is None
    assert report["audit"]["raw_rows"] == 16
    assert "claims" not in report["audit"]


def test_interrupted_status_preserves_rows_and_nulls_claims(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    real_build = MODULE._build_arm_row

    def interrupting(seed, instance, library, observations, **kwargs):
        if (
            seed == 70004
            and kwargs["condition"] == "null_control"
            and kwargs["arm"] == "discovery_only"
        ):
            raise KeyboardInterrupt
        return real_build(seed, instance, library, observations, **kwargs)

    monkeypatch.setattr(MODULE, "_build_arm_row", interrupting)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "interrupted"
    assert report["failure"]["seed"] == 70004
    assert report["failure"]["type"] == "KeyboardInterrupt"
    # Seed 70003 completed all twelve rows; seed 70004's in-library and
    # out-of-library rows plus its first two null-control rows completed
    # and are preserved — so the per-seed view covers only 70003.
    assert [row["seed"] for row in report["raw_rows"]] == [70003] * 12 + [
        70004
    ] * 10
    assert [entry["seed"] for entry in report["per_seed"]] == [70003]
    claims = report["claims"]["claims"]
    assert len(claims) == 4
    assert all(claim["supported"] is None for claim in claims)


def test_design_failure_during_campaign_nulls_claims(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    real_build = MODULE._build_arm_row

    def flaky(seed, instance, library, observations, **kwargs):
        if (
            seed == 70004
            and kwargs["condition"] == "in_library"
            and kwargs["arm"] == "routing_only"
        ):
            raise MODULE.DesignFailureError(
                "seed 70004: incoherent arm outcome",
                seed=70004,
            )
        return real_build(seed, instance, library, observations, **kwargs)

    monkeypatch.setattr(MODULE, "_build_arm_row", flaky)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure"
    assert report["failure"]["seed"] == 70004
    assert report["failure"]["phase"] == "campaign"
    assert report["failure"]["type"] == "DesignFailureError"
    assert "incoherent" in report["failure"]["message"]
    assert [row["seed"] for row in report["raw_rows"]] == [70003] * 12 + [70004]
    claims = report["claims"]["claims"]
    assert len(claims) == 4
    assert all(claim["supported"] is None for claim in claims)


def test_training_machinery_failure_is_a_design_failure(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)

    def exploding_train(*_args, **_kwargs):
        raise FloatingPointError("non-finite loss at epoch 1")

    monkeypatch.setattr(MODULE, "train_router", exploding_train)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure"
    assert report["failure"]["type"] == "DesignFailureError"
    assert report["failure"]["phase"] == "campaign"
    assert "FloatingPointError" in report["failure"]["message"]
    assert report["training"] is None
    assert report["raw_rows"] == []
    assert all(
        claim["supported"] is None for claim in report["claims"]["claims"]
    )


def test_train_block_failure_during_campaign_is_a_design_failure(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    real_make = MODULE.make_budget_instance

    def exploding(seed, *args, **kwargs):
        if seed in (70501, 70502):  # train seeds, not eval seeds
            raise ValueError("generator blew up")
        return real_make(seed, *args, **kwargs)

    monkeypatch.setattr(MODULE, "make_budget_instance", exploding)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure"
    assert report["failure"]["phase"] == "campaign"
    assert report["failure"]["type"] == "DesignFailureError"
    assert "train block construction failed at seed 70501" in report[
        "failure"
    ]["message"]
    assert report["training"] is None
    assert report["raw_rows"] == []
    assert report["descriptive"] is None
    assert all(
        claim["supported"] is None for claim in report["claims"]["claims"]
    )


def test_row_count_invariant_fails_closed_before_claim_inference(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    real_campaign = MODULE._campaign_rows

    def shrinking(eligible, raw_rows, **kwargs):
        real_campaign(eligible, raw_rows, **kwargs)
        raw_rows.pop()  # a silently dropped null-control row

    monkeypatch.setattr(MODULE, "_campaign_rows", shrinking)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure"
    assert report["failure"]["type"] == "DesignFailureError"
    assert "row-count invariant" in report["failure"]["message"]
    assert "4 arms x 3 conditions" in report["failure"]["message"]
    assert all(
        claim["supported"] is None for claim in report["claims"]["claims"]
    )


def test_recompute_per_seed_fails_closed_on_duplicates_gaps_and_unknowns(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "complete"
    rows = report["raw_rows"]
    # A duplicate (condition, arm) of a seed is a design failure.
    with pytest.raises(MODULE.DesignFailureError, match="duplicate"):
        MODULE._recompute_per_seed(rows + [rows[0]])
    # A missing (condition, arm) of a seed is a design failure.
    with pytest.raises(MODULE.DesignFailureError, match="missing raw rows"):
        MODULE._recompute_per_seed(
            [row for row in rows if row is not rows[0]]
        )
    # An unknown (condition, arm) is a design failure.
    tampered = dict(rows[0])
    tampered["arm"] = "teleport"
    with pytest.raises(MODULE.DesignFailureError, match="unknown"):
        MODULE._recompute_per_seed([tampered, *rows[1:]])


# ---------------------------------------------------------------------------
# Claim decision logic on synthetic per-seed differences.


def test_claim_decision_logic_both_sides_of_every_threshold() -> None:
    h1 = _definition("h1-loopv4-arch-vs-generic-e2e")
    h2 = _definition("h2-loopv4-arch-vs-routing-only-e2e")
    h3 = _definition("h3-loopv4-arch-vs-generic-inlibrary")
    h4 = _definition("h4-loopv4-arch-vs-discovery-only-inlibrary-harm")
    n = 30
    # Clearly positive: supported; arithmetic on a constant block is exact
    # (SE exactly 0, so the lower bound equals the estimate).
    for definition in (h1, h2, h3):
        claim = MODULE._claim_summary(definition, [0.1] * n)
        assert claim["supported"] is True
        assert claim["n"] == n
        assert claim["critical_value"] == MODULE._T_ONE_SIDED_BONFERRONI[30]
        assert claim["estimate"] == pytest.approx(0.1)
        assert claim["standard_deviation"] == pytest.approx(0.0)
        assert claim["standard_error"] == pytest.approx(0.0)
        assert claim["lower_bound"] == pytest.approx(0.1)
        # Exactly at the 0.0 threshold: the strict > rule does not support
        # (the strict h3 among them).
        assert MODULE._claim_summary(definition, [0.0] * n)["supported"] is False
        # Clearly negative: not supported.
        negative = MODULE._claim_summary(definition, [-0.1] * n)
        assert negative["supported"] is False
        assert negative["lower_bound"] == pytest.approx(-0.1)
    # H4's harm margin is -0.05: inside the margin is supported...
    inside = MODULE._claim_summary(h4, [-0.04] * n)
    assert inside["supported"] is True
    assert inside["lower_bound"] == pytest.approx(-0.04)
    assert inside["threshold"] == -0.05
    # ...exactly AT the margin is not (the strict > rule)...
    assert MODULE._claim_summary(h4, [-0.05] * n)["supported"] is False
    # ...and outside the margin is harm, not supported.
    outside = MODULE._claim_summary(h4, [-0.06] * n)
    assert outside["supported"] is False
    assert outside["lower_bound"] == pytest.approx(-0.06)
    # Zero difference (the typical no-harm outcome) is supported.
    assert MODULE._claim_summary(h4, [0.0] * n)["supported"] is True
    # One full-harm seed in 30 kills the non-inferiority claim: the lower
    # bound of 29 x 0.0 + 1 x -1.0 falls below the -0.05 margin.
    one_harm = [0.0] * 29 + [-1.0]
    claim = MODULE._claim_summary(h4, one_harm)
    assert claim["estimate"] == pytest.approx(-1 / 30)
    assert claim["standard_error"] > 0.0
    assert claim["lower_bound"] < -0.05
    assert claim["supported"] is False


def test_claim_decision_logic_arithmetic_and_critical_value_by_n() -> None:
    h1 = _definition("h1-loopv4-arch-vs-generic-e2e")
    n = 30
    values = [0.05 + 0.001 * index for index in range(n)]
    claim = MODULE._claim_summary(h1, values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values)
    se = sd / math.sqrt(n)
    assert claim["estimate"] == pytest.approx(mean)
    assert claim["standard_deviation"] == pytest.approx(sd)
    assert claim["standard_error"] == pytest.approx(se)
    assert claim["lower_bound"] == pytest.approx(
        mean - MODULE._T_ONE_SIDED_BONFERRONI[30] * se
    )
    assert claim["supported"] is True
    # Noise around zero: the one-sided bound crosses below zero.
    noisy = [0.02 * ((-1) ** index) for index in range(n)]
    claim = MODULE._claim_summary(h1, noisy)
    assert claim["estimate"] == pytest.approx(0.0)
    assert claim["lower_bound"] < 0.0
    assert claim["supported"] is False
    # The critical value tracks the eligible n.
    claim31 = MODULE._claim_summary(h1, [0.1] * 31)
    assert claim31["n"] == 31
    assert claim31["critical_value"] == MODULE._T_ONE_SIDED_BONFERRONI[31]


def test_claim_summary_refuses_an_unpinned_n() -> None:
    with pytest.raises(MODULE.DesignFailureError, match="critical value"):
        MODULE._claim_summary(
            _definition("h1-loopv4-arch-vs-generic-e2e"), [0.1] * 29
        )


def test_claim_inference_routes_each_paired_difference_to_its_claim() -> None:
    # 30 synthetic seeds; the four differences disagree in sign, so the
    # routing of each claim to its paired difference is fully exposed.
    def per_seed(d1: float, d2: float, d3: float, d4: float) -> list[dict]:
        return [
            {
                "seed": seed,
                "arch_full_e2e": 0.0,
                "routing_only_e2e": 0.0,
                "discovery_only_e2e": 0.0,
                "generic_e2e": 0.0,
                "arch_full_in_library": 0.0,
                "routing_only_in_library": 0.0,
                "discovery_only_in_library": 0.0,
                "generic_in_library": 0.0,
                "d_arch_generic_e2e": d1,
                "d_arch_routing_only_e2e": d2,
                "d_arch_generic_inlibrary": d3,
                "d_arch_discovery_only_inlibrary_harm": d4,
            }
            for seed in range(30)
        ]

    claims = {
        claim["id"]: claim
        for claim in MODULE._claim_inference(
            per_seed(0.1, -0.1, 0.1, -0.1)
        )["claims"]
    }
    assert claims["h1-loopv4-arch-vs-generic-e2e"]["supported"] is True
    assert claims["h2-loopv4-arch-vs-routing-only-e2e"]["supported"] is False
    assert claims["h3-loopv4-arch-vs-generic-inlibrary"]["supported"] is True
    # -0.1 is outside the -0.05 harm margin: not supported.
    assert claims["h4-loopv4-arch-vs-discovery-only-inlibrary-harm"][
        "supported"
    ] is False
    claims = {
        claim["id"]: claim
        for claim in MODULE._claim_inference(
            per_seed(-0.1, 0.1, -0.1, 0.04)
        )["claims"]
    }
    assert claims["h1-loopv4-arch-vs-generic-e2e"]["supported"] is False
    assert claims["h2-loopv4-arch-vs-routing-only-e2e"]["supported"] is True
    assert claims["h3-loopv4-arch-vs-generic-inlibrary"]["supported"] is False
    # 0.04 is inside the -0.05 harm margin: supported.
    assert claims["h4-loopv4-arch-vs-discovery-only-inlibrary-harm"][
        "supported"
    ] is True
    # Estimand text names the routed difference and its provenance.
    family = MODULE._claim_inference(per_seed(0.1, 0.1, 0.1, 0.1))["claims"]
    assert "arch_vs_generic_e2e" in family[0]["estimand"]
    assert "arch_full_e2e - generic_e2e" in family[0]["estimand"]
    assert "arch_vs_routing_only_e2e" in family[1]["estimand"]
    assert "arch_vs_generic_inlibrary" in family[2]["estimand"]
    assert "in_library" in family[2]["estimand"]
    assert "arch_vs_discovery_only_inlibrary_harm" in family[3]["estimand"]
    assert "-0.05" in family[3]["estimand"]


def test_null_claims_emit_all_four_with_null_decisions() -> None:
    family = MODULE._null_claims()
    assert family["family_size"] == 4
    assert family["per_claim_alpha"] == 0.0125
    claims = family["claims"]
    assert [claim["id"] for claim in claims] == list(MODULE.CLAIM_IDS)
    for claim in claims:
        assert claim["supported"] is None
        assert claim["estimate"] is None
        assert claim["standard_error"] is None
        assert claim["lower_bound"] is None
        assert claim["critical_value"] is None
        assert claim["direction"] == "greater"
    thresholds = {claim["id"]: claim["threshold"] for claim in claims}
    assert thresholds["h1-loopv4-arch-vs-generic-e2e"] == 0.0
    assert thresholds["h2-loopv4-arch-vs-routing-only-e2e"] == 0.0
    assert thresholds["h3-loopv4-arch-vs-generic-inlibrary"] == 0.0
    assert thresholds["h4-loopv4-arch-vs-discovery-only-inlibrary-harm"] == -0.05
    statistics_ = {claim["id"]: claim["statistic"] for claim in claims}
    assert statistics_["h1-loopv4-arch-vs-generic-e2e"] == "arch_vs_generic_e2e"
    assert (
        statistics_["h4-loopv4-arch-vs-discovery-only-inlibrary-harm"]
        == "arch_vs_discovery_only_inlibrary_harm"
    )


# ---------------------------------------------------------------------------
# CLI and status vocabulary.


def test_cli_requires_output_defaults_seal_and_reports_status(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    captured = {}

    def fake_run(root, output, *, seal):
        captured.update(root=root, output=output, seal=seal)
        return {
            "status": "complete",
            "eligibility": {"eligible": 34},
            "claims": {"claims": [{"id": "h1", "supported": True}]},
        }

    monkeypatch.setattr(MODULE, "run", fake_run)
    monkeypatch.setattr(MODULE, "_atomic_json_new", lambda *_args: None)
    code = MODULE.main(
        ["--project-root", str(tmp_path), "--output", str(tmp_path / "r.json")]
    )
    assert code == 0
    assert captured["seal"] == MODULE.DEFAULT_SEAL == (
        "docs/31-router-loop-v4-seal.json"
    )
    assert captured["output"] == (tmp_path / "r.json").resolve()
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "complete"
    assert printed["supported_claims"] == ["h1"]
    with pytest.raises(SystemExit):
        MODULE.main(["--project-root", str(tmp_path)])

    def failed_run(root, output, *, seal):
        return {
            "status": "design_failure",
            "eligibility": {"eligible": 0},
            "claims": {"claims": []},
        }

    monkeypatch.setattr(MODULE, "run", failed_run)
    capsys.readouterr()
    code = MODULE.main(
        ["--project-root", str(tmp_path), "--output", str(tmp_path / "r.json")]
    )
    assert code == 2


def test_terminal_status_vocabulary_is_exactly_the_five_pinned_tokens() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    pinned = {
        "complete",
        "design_failure",
        "design_failure_insufficient_eligible",
        "execution_failure",
        "interrupted",
    }
    tokens = set(
        re.findall(
            r'"(complete|design_failure|design_failure_insufficient_eligible'
            r'|execution_failure|interrupted)"',
            source,
        )
    )
    assigned = set(re.findall(r'\["status"\]\s*=\s*"([a-z_]+)"', source))
    assert tokens == pinned
    assert assigned <= pinned


# ---------------------------------------------------------------------------
# TOCTOU guard: a mid-run change voids the complete artifact.


def test_complete_run_refuses_a_tree_dirtied_mid_run(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)

    def dirty_git(_root: Path, *args: str) -> str:
        if args[0] == "status":
            return " M src/universa/loop_v2.py"
        return _fake_git_clean(_root, *args)

    monkeypatch.setattr(MODULE, "_git_checked", dirty_git)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure"
    assert report["failure"]["phase"] == "publication"
    assert "dirty" in report["failure"]["message"]
    assert all(
        claim["supported"] is None for claim in report["claims"]["claims"]
    )


def test_complete_run_refuses_a_manifest_changed_mid_run(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    monkeypatch.setattr(
        MODULE,
        "_code_manifest",
        lambda _root: {"src/universa/loop_v2.py": "0" * 64},
    )
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure"
    assert report["failure"]["phase"] == "publication"
    assert "changed during execution" in report["failure"]["message"]


def test_complete_run_refuses_a_seal_swapped_mid_run(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    # git keeps reporting the ORIGINAL committed seal blob while the on-disk
    # seal is swapped mid-run (assume-unchanged style): the post-run
    # re-check must refuse to publish a complete artifact.
    monkeypatch.setattr(
        MODULE,
        "_git_head_blob",
        lambda _root, _rel: _STUB_SEAL_JSON.encode("utf-8"),
    )
    real_campaign = MODULE._campaign_rows

    def swapping(eligible, raw_rows, **kwargs):
        real_campaign(eligible, raw_rows, **kwargs)
        (tmp_path / "seal.json").write_text(
            '{"stub": "swapped"}\n', encoding="utf-8"
        )

    monkeypatch.setattr(MODULE, "_campaign_rows", swapping)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure"
    assert report["failure"]["phase"] == "publication"
    assert "design seal changed during execution" in report["failure"][
        "message"
    ]


def test_complete_run_refuses_a_protocol_swapped_mid_run(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    real_campaign = MODULE._campaign_rows

    def swapping(eligible, raw_rows, **kwargs):
        real_campaign(eligible, raw_rows, **kwargs)
        (tmp_path / "protocol.md").write_text(
            "swapped protocol\n", encoding="utf-8"
        )

    monkeypatch.setattr(MODULE, "_campaign_rows", swapping)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure"
    assert report["failure"]["phase"] == "publication"
    assert "protocol changed during execution" in report["failure"][
        "message"
    ]


def test_complete_run_refuses_the_runner_file_swapped_mid_run(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    # The running file is edited after preflight recorded its hash —
    # simulated by a _sha256 that reports the true hash at preflight time
    # and a different one afterwards (the real script is never touched).
    real_sha256 = MODULE._sha256
    runner_path = Path(MODULE.__file__).resolve()
    calls = {"count": 0}

    def swapped(path):
        if Path(path) == runner_path:
            calls["count"] += 1
            if calls["count"] > 1:  # preflight recorded the true hash
                return "e" * 64
        return real_sha256(path)

    monkeypatch.setattr(MODULE, "_sha256", swapped)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure"
    assert report["failure"]["phase"] == "publication"
    assert "running file changed during execution" in report["failure"][
        "message"
    ]


# ---------------------------------------------------------------------------
# Audit-block recomputation from the retained raw rows.


def test_audit_claims_match_an_independent_raw_row_recomputation(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "complete"
    assert "eligibility pass" in report["audit"]["recompute_scope"]
    assert "raw rows alone" in report["audit"]["recompute_scope"]
    assert "hash-pinned" in report["audit"]["recompute_scope"]
    rows = report["raw_rows"]
    by_seed: dict[int, list[dict]] = {}
    for row in rows:
        by_seed.setdefault(row["seed"], []).append(row)
    claims = {claim["id"]: claim for claim in report["claims"]["claims"]}
    for definition in MODULE._CLAIM_DEFINITIONS:
        values = []
        for seed, seed_rows in by_seed.items():
            bits = {
                (row["condition"], row["arm"]): 1.0 if row["correct"] else 0.0
                for row in seed_rows
            }
            e2e = {
                arm: statistics.fmean(
                    bits[(condition, arm)] for condition in MODULE.CONDITIONS
                )
                for arm in MODULE.ARMS
            }
            arm_a, arm_b = definition["arms"]
            if definition["scope"] == "e2e":
                values.append(e2e[arm_a] - e2e[arm_b])
            else:
                values.append(
                    bits[("in_library", arm_a)] - bits[("in_library", arm_b)]
                )
        n = len(values)
        estimate = statistics.fmean(values)
        standard_error = statistics.stdev(values) / math.sqrt(n)
        lower_bound = (
            estimate - MODULE._T_ONE_SIDED_BONFERRONI[n] * standard_error
        )
        audit_claim = report["audit"]["claims"][definition["id"]]
        assert audit_claim["estimate"] == pytest.approx(estimate)
        assert audit_claim["standard_error"] == pytest.approx(standard_error)
        assert audit_claim["lower_bound"] == pytest.approx(lower_bound)
        assert audit_claim["supported"] == (
            lower_bound > definition["threshold"]
        )
        # The audit agrees with the decision-path claim.
        assert (
            audit_claim["supported"] == claims[definition["id"]]["supported"]
        )


# ---------------------------------------------------------------------------
# Publication paths: failures never occupy the canonical path.


def test_main_writes_failure_artifacts_only_to_the_failures_path(
    monkeypatch, tmp_path: Path
) -> None:
    report = {
        "status": "design_failure",
        "eligibility": {"eligible": 0},
        "claims": {"claims": []},
    }
    monkeypatch.setattr(MODULE, "run", lambda *_args, **_kwargs: report)
    output = (
        tmp_path / "results" / "experiments" / "router-loop-v4-sealed-1.json"
    )
    code = MODULE.main(
        ["--project-root", str(tmp_path), "--output", str(output)]
    )
    assert code == 2
    assert not output.exists()  # the canonical path stays reserved
    failure_path = (
        output.parent
        / "failures"
        / "router-loop-v4-sealed-1.design_failure.json"
    )
    assert json.loads(failure_path.read_text(encoding="utf-8")) == report


def test_main_preserves_the_report_when_publishing_fails(
    monkeypatch, tmp_path: Path
) -> None:
    report = {
        "status": "complete",
        "eligibility": {"eligible": 34},
        "claims": {"claims": []},
    }
    monkeypatch.setattr(MODULE, "run", lambda *_args, **_kwargs: report)
    output = (
        tmp_path / "results" / "experiments" / "router-loop-v4-sealed-1.json"
    )
    output.parent.mkdir(parents=True)
    output.write_text("occupied", encoding="utf-8")  # appeared mid-run
    code = MODULE.main(
        ["--project-root", str(tmp_path), "--output", str(output)]
    )
    assert code == 2
    assert output.read_text(encoding="utf-8") == "occupied"  # never clobbered
    fallback = output.with_name(f"{output.name}.{os.getpid()}.failed.json")
    assert json.loads(fallback.read_text(encoding="utf-8")) == report


# ---------------------------------------------------------------------------
# Design record, the canonical command, and source wording.


def test_design_record_pins_the_canonical_command() -> None:
    assert MODULE._design_record()["canonical_command"] == (
        "env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src python "
        "scripts/run_loop_v4_sealed_1.py --output "
        "results/experiments/router-loop-v4-sealed-1.json"
    )


def test_design_record_declares_the_train_block_schedules_and_caveat() -> None:
    design = MODULE._design_record()
    assert design["train_seed_block"] == {"first": 200001, "last": 200400}
    assert design["eval_seeds"] == {"first": 210001, "last": 210036}
    assert "4242" in design["training"]["router"]
    assert "4243" in design["training"]["alarm"]
    assert "4244" in design["training"]["generic"]
    assert "label 0 for every row" in design["training"]["router"]
    assert "LearnedAlarmV2" in design["training"]["alarm"]
    assert "alarm_features_v2" in design["training"]["alarm"]
    assert (
        "calibrate_threshold_cost_aware" in design["training"]["calibration"]
    )
    assert "false_quiet_cost = 1.0" in design["training"]["calibration"]
    assert "false_alarm_cost = 1.0" in design["training"]["calibration"]
    assert "eight-key" in design["training"]["calibration"]
    # The declared identity is recorded in the design block, not hidden.
    assert (
        "2 - 2 * balanced_accuracy" in design["training"]["calibration"]
    )
    schedules = design["observation_schedules"]
    assert "router-v2-observe" in schedules["observation_seed"]
    assert "router_permutation" not in schedules  # the loop never permutes
    assert "discovery-observation" in schedules["structured_observations"]
    assert "discovery-null" in schedules["null_observations"]
    assert "loop-v2-operating" in schedules["operating_grid_point"]
    assert set(design["arms"]) == set(MODULE.ARMS)
    assert set(design["conditions"]) == set(MODULE.CONDITIONS)
    assert "no clean column" in design["observation_regime"]
    assert "EXACTLY" in design["eligibility_rule"].upper()
    assert "build" in design["eligibility_rule"]
    assert "4 x 3 x n" in design["row_count_invariant"]
    assert "bit-identical" in design["determinism"]
    assert "wall_time_seconds" in design["determinism"]
    assert "never claim-tested" in design["descriptive"]
    assert design["declared_caveat"] == MODULE.DECLARED_CAVEAT
    assert design["no_outcome_dependent_stopping"] is True
    assert design["no_seed_deletion"] is True


def test_source_wording_and_module_set() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    # The placeholder wording is stale once the hash is pinned: the
    # committed runner embeds the sealed protocol hash and any protocol
    # edit requires re-pinning.
    assert "fail-closed placeholder" not in source
    assert "re-pinning" in source
    # The recorded key names match what the runner actually runs.
    assert "git_status_short" not in source
    assert "git_status_porcelain" in source
    # This runner computes with torch: the three models are trained and
    # hashed here (unlike the numpy-only v1 loop runner).
    assert "torch.manual_seed" in source
    assert "torch.tensor(" in source
    # The v2 alarm machinery is what this runner runs: the widened
    # features, the calibrated threshold, the v2 arch arm.
    assert "alarm_features_v2" in source
    assert "calibrate_threshold_cost_aware" in source
    # loop-v3's bounded rule must not be reachable from this runner.
    assert "max_false_quiet_rate" not in source
    assert "arm_arch_full_v2(" in source
    assert "LearnedAlarmV2" in source
    # The runner never reaches outside its frozen module set (a docstring
    # may NAME another module to say why it is not used; a call is what is
    # forbidden).
    assert "build_no_anchor_dataset(" not in source
    assert "import scripts" not in source
    assert "run_loop_sealed_1" not in source
    assert "run_loop_v2_sealed_1" not in source
    assert "run_router_v2_sealed_1" not in source
    assert "run_router_v1_sealed_1" not in source
    assert "run_discovery_sealed_1" not in source
    assert "run_2complex_sealed_1" not in source
    assert "run_sheaf_sealed_1" not in source
    assert "run_group_sealed_1" not in source


# ---------------------------------------------------------------------------
# The frozen calibration step: record shape, fail-closed validation, plumbing.


def test_calibration_record_is_present_and_well_formed_in_a_tiny_campaign(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "complete"
    alarm = report["training"]["alarm"]
    assert alarm["calibration_costs"] == {
        "false_quiet_cost": 1.0,
        "false_alarm_cost": 1.0,
    }
    assert MODULE.FALSE_QUIET_COST == 1.0
    assert MODULE.FALSE_ALARM_COST == 1.0
    assert "calibrate_threshold_cost_aware" in alarm["calibration_rule"]
    record = alarm["calibration"]
    # Exactly the eight frozen keys, well-formed.
    assert set(record) == {
        "threshold",
        "balanced_accuracy",
        "false_quiet_rate",
        "false_alarm_rate",
        "total_cost",
        "false_quiet_cost",
        "false_alarm_cost",
        "num_candidates",
    }
    assert isinstance(record["threshold"], float)
    assert 0.0 <= record["threshold"] <= 1.0
    for key in ("balanced_accuracy", "false_quiet_rate", "false_alarm_rate"):
        assert isinstance(record[key], float)
        assert 0.0 <= record[key] <= 1.0
    assert record["false_quiet_cost"] == 1.0
    assert record["false_alarm_cost"] == 1.0
    assert record["total_cost"] == pytest.approx(
        record["false_quiet_rate"] + record["false_alarm_rate"], abs=1e-12
    )
    # The declared equal-cost identity, live on a real campaign record.
    assert record["total_cost"] == pytest.approx(
        2.0 - 2.0 * record["balanced_accuracy"], abs=1e-12
    )
    assert isinstance(record["num_candidates"], int)
    assert record["num_candidates"] >= 2
    # Every arch row of the campaign ran under the v2 arm — the alarm_v2
    # provenance marker in the detail strings.
    arch_rows = [row for row in report["raw_rows"] if row["arm"] == "arch_full"]
    assert len(arch_rows) == 3 * 3  # 3 conditions x 3 seeds
    assert all("alarm_v2=" in row["detail"] for row in arch_rows)


def test_validate_calibration_record_fails_closed() -> None:
    good = {
        "threshold": 0.5,
        "balanced_accuracy": 0.895,
        "false_quiet_rate": 0.01,
        "false_alarm_rate": 0.2,
        "total_cost": 0.21,
        "false_quiet_cost": 1.0,
        "false_alarm_cost": 1.0,
        "num_candidates": 7,
    }
    MODULE._validate_calibration_record(good)  # accepts the frozen shape
    for key in ("balanced_accuracy", "false_quiet_rate", "false_alarm_rate"):
        for bad in (-0.1, 1.1, float("nan"), "high", True):
            tampered = dict(good)
            tampered[key] = bad
            with pytest.raises(MODULE.DesignFailureError, match=key):
                MODULE._validate_calibration_record(tampered)
    for bad in (-0.1, 1.1, float("nan"), "0.5", True, 0):
        tampered = dict(good)
        tampered["threshold"] = bad
        with pytest.raises(MODULE.DesignFailureError, match="threshold"):
            MODULE._validate_calibration_record(tampered)
    # The costs must be exactly the experiment's frozen constants: a run
    # calibrated under any other pricing is refused, not reported.
    for key in ("false_quiet_cost", "false_alarm_cost"):
        for bad in (0.0, 2.0, 10.0, float("nan"), "1.0", True):
            tampered = dict(good)
            tampered[key] = bad
            with pytest.raises(MODULE.DesignFailureError, match=key):
                MODULE._validate_calibration_record(tampered)
    for bad in (-0.1, float("nan"), "0.21", True):
        tampered = dict(good)
        tampered["total_cost"] = bad
        with pytest.raises(MODULE.DesignFailureError, match="total_cost"):
            MODULE._validate_calibration_record(tampered)
    # total_cost must agree with the rates and costs it claims to summarize.
    tampered = dict(good)
    tampered["total_cost"] = 0.42
    with pytest.raises(MODULE.DesignFailureError, match="total_cost"):
        MODULE._validate_calibration_record(tampered)
    for bad in (1, 0, 2.5, "7", True):
        tampered = dict(good)
        tampered["num_candidates"] = bad
        with pytest.raises(MODULE.DesignFailureError, match="num_candidates"):
            MODULE._validate_calibration_record(tampered)
    # A missing key, an extra key, and a non-dict are all refused — including
    # loop-v3's bound_satisfied, which this rule must never carry.
    for key in good:
        tampered = dict(good)
        del tampered[key]
        with pytest.raises(MODULE.DesignFailureError, match="eight"):
            MODULE._validate_calibration_record(tampered)
    tampered = dict(good)
    tampered["p_value"] = 0.01  # a foreign analysis field
    with pytest.raises(MODULE.DesignFailureError, match="eight"):
        MODULE._validate_calibration_record(tampered)
    tampered = dict(good)
    tampered["bound_satisfied"] = True  # loop-v3's key, not this rule's
    with pytest.raises(MODULE.DesignFailureError, match="eight"):
        MODULE._validate_calibration_record(tampered)
    with pytest.raises(MODULE.DesignFailureError, match="eight"):
        MODULE._validate_calibration_record("not a record")


def test_arch_arm_receives_the_calibrated_threshold(
    monkeypatch, tmp_path: Path
) -> None:
    """The arch arm runs under the threshold the calibration step picked:
    the value threaded into arm_arch_full_v2 is exactly the retained
    calibration record's threshold, on every arch row of the campaign."""
    _patch_tiny_campaign(monkeypatch)
    captured = []
    real_arm = MODULE.arm_arch_full_v2

    def capturing(seed, instance, library, **kwargs):
        captured.append(kwargs["threshold"])
        return real_arm(seed, instance, library, **kwargs)

    monkeypatch.setattr(MODULE, "arm_arch_full_v2", capturing)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "complete"
    calibrated = report["training"]["alarm"]["calibration"]["threshold"]
    assert len(captured) == 2 * 3 * 3  # the double execution, 3 x 3 arch rows
    assert all(threshold == calibrated for threshold in captured)


def test_v1_alarm_untouched_and_loop_v2_runner_suite_intact() -> None:
    """The sealed loop-v2 experiment is immutable history: its runner and
    its test suite exist, tracked and unmodified, and the v1 alarm
    machinery they pin is still importable with its frozen constant — the
    v3 runner adds the v2 alarm without touching any of it."""
    v2_runner = PROJECT_ROOT / "scripts" / "run_loop_v2_sealed_1.py"
    v2_tests = PROJECT_ROOT / "tests" / "test_run_loop_v2_sealed_1.py"
    assert v2_runner.is_file()
    assert v2_tests.is_file()
    status = subprocess.run(
        (
            "git",
            "status",
            "--porcelain",
            "--",
            "scripts/run_loop_v2_sealed_1.py",
            "tests/test_run_loop_v2_sealed_1.py",
        ),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert status.stdout.strip() == ""  # both untouched relative to HEAD
    # The v1 alarm API the loop-v2 runner pins is unchanged.
    from universa.loop_v2 import (
        ALARM_THRESHOLD,
        LearnedAlarm,
        alarm_features,
        arm_arch_full,
        train_alarm,
    )

    assert ALARM_THRESHOLD == 0.5
    assert issubclass(LearnedAlarm, torch.nn.Module)
    assert callable(alarm_features)
    assert callable(arm_arch_full)
    assert callable(train_alarm)
    # The v3 runner never touches the v1 alarm entry points (word
    # boundaries: the v1 names are strict prefixes of the v2 names).
    source = SCRIPT.read_text(encoding="utf-8")
    for name in (
        "arm_arch_full",
        "LearnedAlarm",
        "alarm_features",
        "train_alarm",
        "ALARM_THRESHOLD",
    ):
        assert not re.search(rf"\b{name}\b", source), name
