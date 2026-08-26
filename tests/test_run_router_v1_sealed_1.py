"""Tests for the sealed router-v1 runner (scripts/run_router_v1_sealed_1.py).

Seed discipline: the sealed eval block 30101..30136 is NEVER instantiated in
this suite — every campaign test monkeypatches the runner's seed blocks down
to explicit non-sealed train seeds (10001..10005) and every direct instance
construction uses those same train seeds.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only, before the torch import

import numpy as np
import pytest
import torch
from scipy.stats import t as t_dist

PROJECT_ROOT = Path(__file__).parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_router_v1_sealed_1.py"
SPEC = importlib.util.spec_from_file_location("run_router_v1_sealed_1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

torch.set_num_threads(1)

from universa.budgets import make_budget_instance
from universa.generators import subseed
from universa.router import StructureRouter, degraded_feature_dim

# Explicit non-sealed fixture seeds; no test may instantiate a sealed seed.
FIXTURE_SEEDS = (10001, 10002, 10003, 10004, 10005)


# ---------------------------------------------------------------------------
# Helpers.


def _claim_object(definition: dict) -> dict:
    """A full seal-record claim object matching one frozen definition."""
    return {
        "id": definition["id"],
        "fraction": definition["fraction"],
        "theta": f"E_seed[d(., {definition['fraction']})]",
        "null": f"theta <= {definition['threshold']}",
        "alternative": f"theta > {definition['threshold']}",
        "reference": "the polluted observed-residual oracle",
        "bound_direction": "greater",
        "threshold": definition["threshold"],
        "support_rule": (
            "supported iff the one-sided Bonferroni lower bound is above "
            "the threshold"
        ),
    }


def _seal_payload(output_path: str = "result.json", **overrides: object) -> dict:
    payload: dict[str, object] = {
        "schema": MODULE.SEAL_SCHEMA,
        "design_commit": "0" * 40,
        "protocol_sha256": "1" * 64,
        "runner_sha256": "2" * 64,
        "code_manifest": {"src/universa/alpha.py": "3" * 64},
        "train_seed_block": {"first": 10001, "last": 10200},
        "eval_seed_block": {"first": 30101, "last": 30136},
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
    if args[0] in ("status", "cat-file"):
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


def _staged_payload(hashes: dict, **overrides: object) -> dict:
    """A seal payload consistent with the staged tree, minus the overrides."""
    merged = {
        "protocol_sha256": hashes["protocol_hash"],
        "runner_sha256": hashes["runner_hash"],
        "code_manifest": hashes["manifest"],
    }
    merged.update(overrides)
    return _seal_payload(**merged)


def _patch_tiny_campaign(
    monkeypatch,
    *,
    eval_seeds=(10003, 10004, 10005),
    min_eligible=3,
    train_seeds=(10001, 10002),
    train_fractions=(0.0,),
    epochs=2,
) -> None:
    """Shrink the frozen design to a fast, fully non-sealed campaign."""
    monkeypatch.setattr(
        MODULE,
        "_preflight",
        lambda *_args, **_kwargs: {
            "stub": True,
            "git_status_porcelain": "",
            "code_manifest": {"files": {}},
        },
    )
    monkeypatch.setattr(MODULE, "_code_manifest", lambda _root: {})
    monkeypatch.setattr(MODULE, "TRAIN_SEEDS", tuple(train_seeds))
    monkeypatch.setattr(MODULE, "TRAIN_FRACTIONS", tuple(train_fractions))
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


# ---------------------------------------------------------------------------
# Frozen constants and seed-block discipline.


def test_sealed_seed_block_declared_disjoint_and_never_instantiated() -> None:
    assert MODULE.SEALED_EVAL_SEEDS == tuple(range(30101, 30137))
    assert MODULE.TRAIN_SEEDS == tuple(range(10001, 10201))
    assert not set(MODULE.SEALED_EVAL_SEEDS) & set(MODULE.TRAIN_SEEDS)
    assert not set(MODULE.SEALED_EVAL_SEEDS) & set(FIXTURE_SEEDS)


def test_frozen_design_constants() -> None:
    assert (MODULE.NUM_VERTICES, MODULE.NUM_EDGES) == (8, 14)
    assert (MODULE.NUM_CLASSES, MODULE.NUM_DECOYS) == (6, 3)
    assert MODULE.NUM_CANDIDATES == 4
    assert MODULE.TRAIN_FRACTIONS == (0.0, 0.1, 0.2, 0.3, 0.4)
    assert MODULE.HELD_OUT_FRACTIONS == (0.5, 0.6, 0.7)
    assert MODULE.EVAL_FRACTIONS == (0.5, 0.6, 0.7, 0.0)
    assert MODULE.PROFILE_GRID == tuple(i / 10 for i in range(8))
    assert MODULE.REPLICATES == 4
    assert MODULE.FEATURE_DIM == 18
    assert MODULE.FEATURE_DIM == degraded_feature_dim(MODULE.PROFILE_GRID)
    assert MODULE.HIDDEN_DIM == 64
    assert MODULE.EPOCHS == 150
    assert MODULE.LEARNING_RATE == 1e-3
    assert MODULE.TORCH_SEED == 4242
    assert MODULE.LAMBDA_AUX == 0.01
    assert (MODULE.TAU_START, MODULE.TAU_END) == (2.0, 0.25)
    assert MODULE.PER_CLAIM_ALPHA == 0.0125
    assert MODULE.MIN_ELIGIBLE == 30
    assert MODULE.AUDIT_TOL == 1e-9
    assert MODULE.H4_MARGIN == -0.05
    assert MODULE.CLAIM_IDS == (
        "h1-held-out-0.6-primary",
        "h2-held-out-0.5",
        "h3-held-out-0.7",
        "h4-clean-anchor-bounded-harm",
    )
    # The committed runner embeds the sealed protocol hash (the placeholder
    # was replaced before commit A); the placeholder path is covered by
    # test_preflight_refuses_the_protocol_placeholder via monkeypatching.
    assert MODULE.PROTOCOL_SHA256 != "PENDING_PROTOCOL_SHA256"
    assert len(MODULE.PROTOCOL_SHA256) == 64
    assert "asymmetric by design" in MODULE.DECLARED_CAVEAT


def test_replicate_subseed_derivation_is_exact_and_component_separated() -> None:
    seed = MODULE._replicate_observation_seed(10001, 0.5, 0)
    assert seed == subseed(10001, "sealed-replicate", str(0.5), str(0))
    assert seed != MODULE._replicate_observation_seed(10001, 0.5, 1)
    assert seed != MODULE._replicate_observation_seed(10001, 0.6, 0)
    assert seed != MODULE._replicate_observation_seed(10002, 0.5, 0)


def test_critical_value_table_matches_scipy() -> None:
    assert sorted(MODULE._T_ONE_SIDED_BONFERRONI) == list(range(30, 37))
    for n in range(30, 37):
        expected = float(t_dist.ppf(1 - 0.05 / 4, n - 1))
        assert MODULE._T_ONE_SIDED_BONFERRONI[n] == pytest.approx(
            expected, rel=1e-12
        )


# ---------------------------------------------------------------------------
# Seal validation.


def test_load_seal_accepts_a_valid_frozen_seal(tmp_path: Path) -> None:
    _write_seal(tmp_path, _seal_payload())
    seal = MODULE._load_seal(tmp_path, "seal.json")
    assert seal["schema"] == "universa-seal/1"
    assert [claim["id"] for claim in seal["primary_family"]] == list(
        MODULE.CLAIM_IDS
    )


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
    _write_seal(tmp_path, _seal_payload(schema="universa-seal/2"))
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
        _seal_payload(train_seed_block={"first": 10001, "last": 10201}),
    )
    with pytest.raises(RuntimeError, match="train_seed_block"):
        MODULE._load_seal(tmp_path, "seal.json")
    _write_seal(
        tmp_path,
        _seal_payload(eval_seed_block={"first": 30100, "last": 30136}),
    )
    with pytest.raises(RuntimeError, match="eval_seed_block"):
        MODULE._load_seal(tmp_path, "seal.json")
    _write_seal(tmp_path, _seal_payload(train_seed_block=[10001, 10200]))
    with pytest.raises(RuntimeError, match="train_seed_block"):
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
    family = [_claim_object(definition) for definition in MODULE._CLAIM_DEFINITIONS]
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
    assert provenance["seal"]["committed_at_head"] is True
    assert provenance["protocol"]["sha256"] == hashes["protocol_hash"]
    assert provenance["runner"]["sha256"] == hashes["runner_hash"]
    assert provenance["code_manifest"]["files"] == hashes["manifest"]
    assert len(provenance["code_manifest"]["sha256"]) == 64
    execution = provenance["execution"]
    assert execution["torch_num_threads"] == 1
    assert execution["cuda_available"] is False
    assert execution["tensor_dtype"] == "float32"
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
    # With the placeholder restored, the fail-closed guard must bite.
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


def test_preflight_refuses_a_visible_cuda_device(
    monkeypatch, tmp_path: Path
) -> None:
    hashes = _stage_sealed_tree(tmp_path)
    _patch_staged_preflight(monkeypatch, tmp_path, hashes)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="CUDA"):
        MODULE._preflight(tmp_path, tmp_path / "result.json", seal="seal.json")


def test_execution_environment_pins_single_thread_cpu_float32() -> None:
    env = MODULE._execution_environment()
    assert env["torch_num_threads"] == 1
    assert env["cuda_available"] is False
    assert env["tensor_device"] == "cpu"
    assert env["tensor_dtype"] == "float32"


def test_execution_environment_refuses_a_visible_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="CUDA"):
        MODULE._execution_environment()


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
# Pairing, audits, and eligibility (real instances, non-sealed seeds only).


def test_paired_replicates_are_shared_across_both_arms() -> None:
    instance = make_budget_instance(10001, 8, 14, 6, 3)
    row = MODULE._build_replicate_row(
        instance, 10001, 0.5, 0, with_residuals=True
    )
    assert row.observation_seed == subseed(
        10001, "sealed-replicate", str(0.5), str(0)
    )
    assert row.features.shape == (MODULE.NUM_CANDIDATES, MODULE.FEATURE_DIM)
    assert row.permutation[row.true_index] == 0
    # The learned features and the oracle residuals come from ONE draw: the
    # profile column at the operating fraction is log1p of those residuals.
    grid_index = next(
        index
        for index, point in enumerate(MODULE.PROFILE_GRID)
        if abs(point - 0.5) <= 1e-9
    )
    column = np.expm1(row.features[:, grid_index])
    assert np.allclose(
        column, np.asarray(row.observed_residuals), rtol=0.0, atol=1e-12
    )
    assert int(np.argmin(row.observed_residuals)) == int(np.argmin(column))
    # The draw and the row construction are exactly deterministic.
    again = MODULE._build_replicate_row(
        instance, 10001, 0.5, 0, with_residuals=True
    )
    assert np.array_equal(row.features, again.features)
    assert row.observed_residuals == again.observed_residuals
    assert row.permutation == again.permutation


def test_fraction0_audit_is_exact_for_real_instances() -> None:
    for seed in FIXTURE_SEEDS[:3]:
        instance = make_budget_instance(seed, 8, 14, 6, 3)
        misfits = MODULE._fraction0_misfits(instance)
        assert len(misfits) == MODULE.NUM_CANDIDATES
        assert misfits[0] <= MODULE.AUDIT_TOL
        assert all(misfit > MODULE.AUDIT_TOL for misfit in misfits[1:])


def test_both_arms_score_hand_fixtures_by_the_frozen_rules() -> None:
    # Learned arm: strictly discrete argmax over candidate logits. A
    # zero-initialized router produces identical logits for every candidate,
    # so the argmax rule must pick candidate 0.
    model = StructureRouter(MODULE.FEATURE_DIM, MODULE.HIDDEN_DIM)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    features = np.zeros((MODULE.NUM_CANDIDATES, MODULE.FEATURE_DIM))
    residuals = (3.0, 1.0, 2.0, 4.0)
    # Oracle arm: argmin of the observed residuals is candidate 1.
    learned_correct, oracle_correct = MODULE._score_row(
        model, features, residuals, true_index=0
    )
    assert (learned_correct, oracle_correct) == (True, False)
    learned_correct, oracle_correct = MODULE._score_row(
        model, features, residuals, true_index=1
    )
    assert (learned_correct, oracle_correct) == (False, True)
    learned_correct, oracle_correct = MODULE._score_row(
        model, features, residuals, true_index=2
    )
    assert (learned_correct, oracle_correct) == (False, False)


def test_eligibility_real_train_seeds_all_pass(
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
        assert record["true_misfit_fraction0"] <= MODULE.AUDIT_TOL
        assert min(record["decoy_misfits_fraction0"]) > MODULE.AUDIT_TOL


def test_fraction0_audit_failure_marks_seed_ineligible_with_reason(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(
        monkeypatch, eval_seeds=(10001, 10002, 10003), min_eligible=3
    )
    real_misfits = MODULE._fraction0_misfits

    def poisoned(instance):
        misfits = list(real_misfits(instance))
        if instance.seed == 10002:
            misfits[1] = 0.0  # a decoy that admits the truth
        return tuple(misfits)

    monkeypatch.setattr(MODULE, "_fraction0_misfits", poisoned)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure_insufficient_eligible"
    assert report["eligibility"]["eligible_seeds"] == [10001, 10003]
    ineligible = report["eligibility"]["ineligible"]
    assert len(ineligible) == 1
    assert ineligible[0]["seed"] == 10002
    assert "decoy 0" in ineligible[0]["reason"]
    record = next(
        record
        for record in report["seed_records"]
        if record["seed"] == 10002
    )
    assert record["eligible"] is False
    assert record["reason"] == ineligible[0]["reason"]


def test_build_exception_is_a_whole_run_design_failure(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(
        monkeypatch, eval_seeds=(10001, 10002, 10003), min_eligible=1
    )
    real_make = MODULE.make_budget_instance

    def exploding(seed, *args, **kwargs):
        if seed == 10002:
            raise ValueError("generator blew up")
        return real_make(seed, *args, **kwargs)

    monkeypatch.setattr(MODULE, "make_budget_instance", exploding)
    monkeypatch.setattr(
        MODULE,
        "_build_train_block",
        lambda: pytest.fail("fits must not run after a build failure"),
    )
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure"
    assert report["failure"]["seed"] == 10002
    assert report["failure"]["phase"] == "eligibility"
    assert report["failure"]["type"] == "ValueError"
    assert "generator blew up" in report["failure"]["message"]
    assert report["eligibility"]["build_failures"] == [
        {"seed": 10002, "type": "ValueError", "message": "generator blew up"}
    ]
    assert report["eligibility"]["attempted"] == 2
    # Seed 10001 was fully audited before the failure and is preserved.
    assert [record["seed"] for record in report["seed_records"]] == [10001]
    assert report["training"] is None
    assert report["raw_rows"] == []
    assert report["paired"] == []
    claims = report["claims"]["claims"]
    assert len(claims) == 4
    assert all(claim["supported"] is None for claim in claims)


def test_insufficient_eligible_runs_no_fits_and_nulls_claims(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(
        monkeypatch, eval_seeds=(10001, 10002, 10003), min_eligible=4
    )
    monkeypatch.setattr(
        MODULE,
        "_build_train_block",
        lambda: pytest.fail("fits must not run after eligibility failure"),
    )
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure_insufficient_eligible"
    assert report["eligibility"]["eligible"] == 3
    assert report["training"] is None
    assert report["raw_rows"] == []
    assert report["paired"] == []
    claims = report["claims"]["claims"]
    assert len(claims) == 4
    assert {claim["id"] for claim in claims} == set(MODULE.CLAIM_IDS)
    assert all(claim["supported"] is None for claim in claims)
    assert all(claim["estimate"] is None for claim in claims)
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
    assert report["schema"] == "universa-router-v1-sealed-result/1"
    assert report["experiment_id"] == "universa-router-v1-sealed-1"
    assert report["eligibility"]["eligible_seeds"] == [10003, 10004, 10005]
    # The TOCTOU guard records the post-run status next to the pre-run one.
    assert report["provenance"]["git_status_porcelain"] == ""
    assert report["provenance"]["git_status_porcelain_post_run"] == ""
    # The whole report must be JSON-serializable (no numpy leaks).
    json.dumps(report, sort_keys=True)

    rows = report["raw_rows"]
    assert len(rows) == 3 * len(MODULE.EVAL_FRACTIONS) * MODULE.REPLICATES
    for row in rows:
        assert set(row) == {
            "seed",
            "fraction",
            "replicate_index",
            "observation_seed",
            "learned_correct",
            "oracle_correct",
            "observed_residuals",
            "permutation",
            "true_index",
        }
        assert row["seed"] in (10003, 10004, 10005)
        assert row["fraction"] in MODULE.EVAL_FRACTIONS
        assert 0 <= row["replicate_index"] < MODULE.REPLICATES
        assert isinstance(row["learned_correct"], bool)
        assert isinstance(row["oracle_correct"], bool)
        assert len(row["observed_residuals"]) == MODULE.NUM_CANDIDATES
        assert sorted(row["permutation"]) == list(range(MODULE.NUM_CANDIDATES))
        assert row["permutation"][row["true_index"]] == 0

    paired = report["paired"]
    assert len(paired) == 3 * len(MODULE.EVAL_FRACTIONS)
    for entry in paired:
        assert set(entry) == {
            "seed",
            "fraction",
            "replicates",
            "learned_acc",
            "oracle_acc",
            "d",
        }
        assert 0.0 <= entry["learned_acc"] <= 1.0
        assert 0.0 <= entry["oracle_acc"] <= 1.0
        assert entry["replicates"] == MODULE.REPLICATES
        assert entry["d"] == pytest.approx(
            entry["learned_acc"] - entry["oracle_acc"]
        )
    # The clean anchor: the exact oracle must be perfect.
    anchor = [entry for entry in paired if entry["fraction"] == 0.0]
    assert len(anchor) == 3
    assert all(entry["oracle_acc"] == 1.0 for entry in anchor)

    claims = {claim["id"]: claim for claim in report["claims"]["claims"]}
    assert set(claims) == set(MODULE.CLAIM_IDS)
    for claim in claims.values():
        assert claim["n"] == 3
        assert isinstance(claim["supported"], bool)
        assert claim["lower_bound"] == pytest.approx(
            claim["estimate"]
            - claim["critical_value"] * claim["standard_error"]
        )

    audit = report["audit"]
    assert audit["declared_seeds"] == 3
    assert audit["eligible_seeds"] == 3
    assert audit["raw_rows"] == len(rows)
    assert audit["per_seed_fraction"] == paired
    assert "claim_estimates" not in audit
    for definition in MODULE._CLAIM_DEFINITIONS:
        estimate = statistics.fmean(
            entry["d"]
            for entry in paired
            if entry["fraction"] == definition["fraction"]
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

    training = report["training"]
    assert training["torch_seed"] == MODULE.TORCH_SEED
    assert training["epochs"] == 2  # the monkeypatched tiny value
    # One train row per (seed, fraction): 2 seeds x 1 fraction here.
    assert training["num_train_rows"] == 2
    assert set(training["final"]) == {
        "loss",
        "cross_entropy",
        "aux_loss",
        "tau",
        "train_accuracy",
        "val_accuracy",
        "gate_entropy",
    }
    assert all(math.isfinite(v) for v in training["final"].values())
    assert len(training["model_state_sha256"]) == 64


def test_complete_tiny_campaign_is_deterministic(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    first = MODULE.run(tmp_path, tmp_path / "a.json", seal="seal.json")
    second = MODULE.run(tmp_path, tmp_path / "b.json", seal="seal.json")
    assert first["status"] == second["status"] == "complete"
    assert first["raw_rows"] == second["raw_rows"]
    assert first["paired"] == second["paired"]
    assert first["claims"] == second["claims"]
    assert (
        first["training"]["model_state_sha256"]
        == second["training"]["model_state_sha256"]
    )
    assert first["training"]["final"] == second["training"]["final"]


def test_tiny_campaign_instantiates_only_declared_test_seeds(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    seen = []
    real_make = MODULE.make_budget_instance

    def recording(seed, *args, **kwargs):
        seen.append(seed)
        return real_make(seed, *args, **kwargs)

    # The eval eligibility pass builds through the runner module; the train
    # block builds inside universa.router.build_degraded_dataset.
    import universa.router as router_module

    monkeypatch.setattr(MODULE, "make_budget_instance", recording)
    monkeypatch.setattr(router_module, "make_budget_instance", recording)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "complete"
    assert set(seen) == set(FIXTURE_SEEDS)
    assert not set(seen) & set(range(30101, 30137))


# ---------------------------------------------------------------------------
# Failure paths: statuses, preserved rows, null claims.


def test_execution_failure_preserves_rows_and_nulls_claims(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    real_build_row = MODULE._build_replicate_row

    def flaky(instance, seed, fraction, replicate, *, with_residuals):
        if seed == 10004:
            raise FloatingPointError("unexpected numeric fault")
        return real_build_row(
            instance, seed, fraction, replicate,
            with_residuals=with_residuals,
        )

    monkeypatch.setattr(MODULE, "_build_replicate_row", flaky)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "execution_failure"
    assert report["failure"]["seed"] == 10004
    assert report["failure"]["phase"] == "campaign"
    assert report["failure"]["type"] == "FloatingPointError"
    assert "unexpected numeric fault" in report["failure"]["message"]
    # Seed 10003 completed all of its rows before the failure; 10005 never ran.
    expected = len(MODULE.EVAL_FRACTIONS) * MODULE.REPLICATES
    assert len(report["raw_rows"]) == expected
    assert {row["seed"] for row in report["raw_rows"]} == {10003}
    assert len(report["paired"]) == len(MODULE.EVAL_FRACTIONS)
    claims = report["claims"]["claims"]
    assert len(claims) == 4
    assert all(claim["supported"] is None for claim in claims)
    assert all(claim["estimate"] is None for claim in claims)
    assert report["training"] is not None  # completed training is preserved
    assert report["audit"]["raw_rows"] == expected
    assert "claims" not in report["audit"]


def test_interrupted_status_preserves_rows_and_nulls_claims(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    real_build_row = MODULE._build_replicate_row

    def interrupting(instance, seed, fraction, replicate, *, with_residuals):
        if seed == 10003 and fraction == 0.7:
            raise KeyboardInterrupt
        return real_build_row(
            instance, seed, fraction, replicate,
            with_residuals=with_residuals,
        )

    monkeypatch.setattr(MODULE, "_build_replicate_row", interrupting)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "interrupted"
    assert report["failure"]["seed"] == 10003
    assert report["failure"]["type"] == "KeyboardInterrupt"
    # Fractions 0.5 and 0.6 of seed 10003 completed before the interrupt.
    expected = 2 * MODULE.REPLICATES
    assert len(report["raw_rows"]) == expected
    assert {row["fraction"] for row in report["raw_rows"]} == {0.5, 0.6}
    claims = report["claims"]["claims"]
    assert len(claims) == 4
    assert all(claim["supported"] is None for claim in claims)


def test_design_failure_during_campaign_nulls_claims(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    real_build_row = MODULE._build_replicate_row

    def flaky(instance, seed, fraction, replicate, *, with_residuals):
        if seed == 10004:
            raise MODULE.DesignFailureError(
                "seed 10004: true candidate misfit 1.0 at fraction 0 "
                "exceeds 1e-09",
                seed=10004,
            )
        return real_build_row(
            instance, seed, fraction, replicate,
            with_residuals=with_residuals,
        )

    monkeypatch.setattr(MODULE, "_build_replicate_row", flaky)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure"
    assert report["failure"]["seed"] == 10004
    assert report["failure"]["phase"] == "campaign"
    assert report["failure"]["type"] == "DesignFailureError"
    assert "fraction 0" in report["failure"]["message"]
    # Seed 10003 completed all of its rows before the design failure.
    expected = len(MODULE.EVAL_FRACTIONS) * MODULE.REPLICATES
    assert len(report["raw_rows"]) == expected
    assert {row["seed"] for row in report["raw_rows"]} == {10003}
    claims = report["claims"]["claims"]
    assert len(claims) == 4
    assert all(claim["supported"] is None for claim in claims)


def test_row_fraction0_audit_fails_closed_on_tainted_features(
    monkeypatch,
) -> None:
    instance = make_budget_instance(10001, 8, 14, 6, 3)
    real_features = MODULE.degraded_candidate_features

    def tainted(chain_map, candidate, observation_seed, grid):
        vector = real_features(chain_map, candidate, observation_seed, grid)
        if candidate is instance.true_target:
            vector = vector.copy()
            vector[0] = float(np.log1p(1.0))  # true misfit 1.0 at fraction 0
        return vector

    monkeypatch.setattr(MODULE, "degraded_candidate_features", tainted)
    with pytest.raises(MODULE.DesignFailureError, match="true candidate misfit"):
        MODULE._build_replicate_row(
            instance, 10001, 0.5, 0, with_residuals=True
        )


# ---------------------------------------------------------------------------
# Claim decision logic on synthetic per-seed differences.


def test_claim_decision_logic_supported_and_not_supported_sides() -> None:
    h1 = _definition("h1-held-out-0.6-primary")
    h4 = _definition("h4-clean-anchor-bounded-harm")
    n = 30
    # Clearly positive: supported; arithmetic on a constant block is exact.
    claim = MODULE._claim_summary(h1, [0.1] * n)
    assert claim["supported"] is True
    assert claim["n"] == n
    assert claim["critical_value"] == MODULE._T_ONE_SIDED_BONFERRONI[30]
    assert claim["estimate"] == pytest.approx(0.1)
    assert claim["standard_deviation"] == pytest.approx(0.0)
    assert claim["standard_error"] == pytest.approx(0.0)
    assert claim["lower_bound"] == pytest.approx(0.1)
    # Exactly at the threshold: the strict > rule does not support.
    assert MODULE._claim_summary(h1, [0.0] * n)["supported"] is False
    # Clearly negative: not supported.
    negative = MODULE._claim_summary(h1, [-0.1] * n)
    assert negative["supported"] is False
    assert negative["lower_bound"] == pytest.approx(-0.1)
    # H4 bounded-harm margin at -0.05: both sides and the boundary.
    assert MODULE._claim_summary(h4, [-0.03] * n)["supported"] is True
    assert MODULE._claim_summary(h4, [-0.05] * n)["supported"] is False
    assert MODULE._claim_summary(h4, [-0.07] * n)["supported"] is False


def test_claim_decision_logic_arithmetic_and_critical_value_by_n() -> None:
    h1 = _definition("h1-held-out-0.6-primary")
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
            _definition("h1-held-out-0.6-primary"), [0.1] * 29
        )


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
    assert thresholds["h4-clean-anchor-bounded-harm"] == -0.05
    assert thresholds["h1-held-out-0.6-primary"] == 0.0


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
    assert captured["seal"] == MODULE.DEFAULT_SEAL == "docs/02-router-v1-seal.json"
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
# Seal byte-binding, symlink refusal, and design-commit anchor.


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


def test_load_seal_rejects_tampered_claim_values_and_unknown_keys(
    tmp_path: Path,
) -> None:
    family = [_claim_object(d) for d in MODULE._CLAIM_DEFINITIONS]
    for key, bad_value in (
        ("fraction", 0.7),
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
    # An unknown top-level seal key is rejected.
    _write_seal(tmp_path, _seal_payload(unexpected="extra"))
    with pytest.raises(RuntimeError, match="unknown keys"):
        MODULE._load_seal(tmp_path, "seal.json")
    # An unknown key inside a seed-block dict is rejected.
    _write_seal(
        tmp_path,
        _seal_payload(
            train_seed_block={"first": 10001, "last": 10200, "note": "x"}
        ),
    )
    with pytest.raises(RuntimeError, match="train_seed_block"):
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
# Eligibility-phase classification and the row-count invariant.


def test_eligibility_pass_failures_enter_the_status_vocabulary(
    monkeypatch, tmp_path: Path
) -> None:
    real_misfits = MODULE._fraction0_misfits
    cases = (
        (RuntimeError("audit exploded"), "execution_failure"),
        (
            MODULE.DesignFailureError("audit rejected the seed", seed=10002),
            "design_failure",
        ),
        (KeyboardInterrupt(), "interrupted"),
    )
    for index, (raised, expected_status) in enumerate(cases):
        _patch_tiny_campaign(
            monkeypatch, eval_seeds=(10001, 10002, 10003), min_eligible=1
        )

        def exploding(instance, _raised=raised):
            if instance.seed == 10002:
                raise _raised
            return real_misfits(instance)

        monkeypatch.setattr(MODULE, "_fraction0_misfits", exploding)
        report = MODULE.run(
            tmp_path, tmp_path / f"out-{index}.json", seal="seal.json"
        )
        assert report["status"] == expected_status
        assert report["failure"]["phase"] == "eligibility"
        assert report["failure"]["type"] == type(raised).__name__
        # Seed 10001 was fully audited before the failure and is preserved.
        assert [record["seed"] for record in report["seed_records"]] == [
            10001
        ]
        assert report["training"] is None
        assert report["raw_rows"] == []
        assert report["paired"] == []
        claims = report["claims"]["claims"]
        assert len(claims) == 4
        assert all(claim["supported"] is None for claim in claims)


def test_row_count_invariant_fails_closed_before_claim_inference(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)

    class _ShrinkingFractions(tuple):
        """len() reports four fractions; iteration silently yields three."""

        def __iter__(self):
            full = tuple(super().__iter__())
            return iter(full[: len(full) - 1])

    monkeypatch.setattr(
        MODULE, "EVAL_FRACTIONS", _ShrinkingFractions(MODULE.EVAL_FRACTIONS)
    )
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure"
    assert report["failure"]["type"] == "DesignFailureError"
    assert "row-count invariant" in report["failure"]["message"]
    assert all(
        claim["supported"] is None for claim in report["claims"]["claims"]
    )


# ---------------------------------------------------------------------------
# Certified-machinery failures classify as design failures.


def test_training_machinery_failure_is_a_design_failure(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)

    def exploding_train(*_args, **_kwargs):
        raise FloatingPointError("non-finite loss at epoch 3")

    monkeypatch.setattr(MODULE, "train_router", exploding_train)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure"
    assert report["failure"]["type"] == "DesignFailureError"
    assert "FloatingPointError" in report["failure"]["message"]
    assert report["training"] is None
    assert all(
        claim["supported"] is None for claim in report["claims"]["claims"]
    )


def test_eval_row_machinery_failure_is_a_design_failure(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)

    def broken_features(*_args, **_kwargs):
        raise ValueError("certified machinery rejected the observation")

    monkeypatch.setattr(MODULE, "degraded_candidate_features", broken_features)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure"
    assert report["failure"]["seed"] == 10003
    assert report["failure"]["type"] == "DesignFailureError"
    assert "ValueError" in report["failure"]["message"]
    assert all(
        claim["supported"] is None for claim in report["claims"]["claims"]
    )


# ---------------------------------------------------------------------------
# TOCTOU guard: a mid-run change voids the complete artifact.


def test_complete_run_refuses_a_tree_dirtied_mid_run(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)

    def dirty_git(_root: Path, *args: str) -> str:
        if args[0] == "status":
            return " M src/universa/router.py"
        return _fake_git_clean(_root, *args)

    monkeypatch.setattr(MODULE, "_git_checked", dirty_git)
    with pytest.raises(RuntimeError, match="dirty"):
        MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")


def test_complete_run_refuses_a_manifest_changed_mid_run(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    monkeypatch.setattr(
        MODULE,
        "_code_manifest",
        lambda _root: {"src/universa/router.py": "0" * 64},
    )
    with pytest.raises(RuntimeError, match="changed during execution"):
        MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")


# ---------------------------------------------------------------------------
# Audit-block recomputation from the retained raw rows.


def test_audit_claims_match_an_independent_raw_row_recomputation(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "complete"
    assert "eligibility pass" in report["audit"]["recompute_scope"]
    rows = report["raw_rows"]
    claims = {claim["id"]: claim for claim in report["claims"]["claims"]}
    for definition in MODULE._CLAIM_DEFINITIONS:
        fraction = definition["fraction"]
        per_seed: dict[int, list[dict]] = {}
        for row in rows:
            if row["fraction"] == fraction:
                per_seed.setdefault(row["seed"], []).append(row)
        differences = [
            (
                sum(1 for row in seed_rows if row["learned_correct"])
                - sum(1 for row in seed_rows if row["oracle_correct"])
            )
            / len(seed_rows)
            for seed_rows in per_seed.values()
        ]
        n = len(differences)
        estimate = statistics.fmean(differences)
        standard_error = statistics.stdev(differences) / math.sqrt(n)
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
    output = tmp_path / "results" / "experiments" / "router-v1-sealed-1.json"
    code = MODULE.main(
        ["--project-root", str(tmp_path), "--output", str(output)]
    )
    assert code == 2
    assert not output.exists()  # the canonical path stays reserved
    failure_path = (
        output.parent / "failures" / "router-v1-sealed-1.design_failure.json"
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
    output = tmp_path / "results" / "experiments" / "router-v1-sealed-1.json"
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
# Environment provenance, the canonical command, and source wording.


def test_execution_environment_records_operator_and_effective_cuda() -> None:
    env = MODULE._execution_environment()
    assert env["cuda_visible_devices"] == ""
    assert (
        env["cuda_visible_devices_operator"]
        == MODULE.OPERATOR_CUDA_VISIBLE_DEVICES
    )


def test_design_record_pins_the_canonical_command() -> None:
    assert MODULE._design_record()["canonical_command"] == (
        "env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src python "
        "scripts/run_router_v1_sealed_1.py --output "
        "results/experiments/router-v1-sealed-1.json"
    )


def test_source_wording_and_git_key_names_are_current() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    # The placeholder wording is stale: the committed runner embeds the
    # sealed protocol hash and any protocol edit requires re-pinning.
    assert "fail-closed placeholder" not in source
    assert "re-pinning" in source
    # The recorded key names match what the runner actually runs.
    assert "git_status_short" not in source
    assert "git_status_porcelain" in source
