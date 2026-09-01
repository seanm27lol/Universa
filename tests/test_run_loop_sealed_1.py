"""Tests for the sealed router-loop runner (scripts/run_loop_sealed_1.py).

Seed discipline: the sealed eval block 140001..140036 — like the v1 block
30101..30136, the v2 block 60101..60136, the reserved block 80101..80136,
the discovery block 90101..90136, the sheaf block 20101..20136, and the
group block 40101..40136 — is NEVER instantiated in this suite: every
campaign test monkeypatches the runner's seed block down to explicit
non-sealed fixture seeds (70001..70005) and every direct instance
construction uses those same fixture seeds.
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

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only, before any torch import

import numpy as np
import pytest
from scipy.stats import t as t_dist

PROJECT_ROOT = Path(__file__).parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "run_loop_sealed_1.py"
SPEC = importlib.util.spec_from_file_location("run_loop_sealed_1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

import universa  # noqa: E402
from universa.loop import (  # noqa: E402
    ALARM_TOL,
    LibraryLoop,
    make_loop_instance,
    null_observations,
    run_loop,
)
from universa.operators import CERT_TOL  # noqa: E402

# Explicit non-sealed fixture seeds; no test may instantiate a sealed seed.
FIXTURE_SEEDS = (70001, 70002, 70003, 70004, 70005)
V1_SEALED_BLOCK = range(30101, 30137)
V2_SEALED_BLOCK = range(60101, 60137)
RESERVED_SEALED_BLOCK = range(80101, 80137)
DISCOVERY_SEALED_BLOCK = range(90101, 90137)
SHEAF_SEALED_BLOCK = range(20101, 20137)
GROUP_SEALED_BLOCK = range(40101, 40137)
LOOP_SEALED_BLOCK = range(140001, 140037)


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
        "train_seed_block": {"first": 0, "last": 0},
        "eval_seed_block": {"first": 140001, "last": 140036},
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
    monkeypatch.setattr(MODULE, "SEALED_EVAL_SEEDS", tuple(eval_seeds))
    monkeypatch.setattr(MODULE, "MIN_ELIGIBLE", min_eligible)
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
    assert MODULE.SEALED_EVAL_SEEDS == tuple(range(140001, 140037))
    assert MODULE.TRAIN_SEEDS == ()
    assert not set(MODULE.SEALED_EVAL_SEEDS) & set(FIXTURE_SEEDS)
    assert not set(MODULE.SEALED_EVAL_SEEDS) & set(V1_SEALED_BLOCK)
    assert not set(MODULE.SEALED_EVAL_SEEDS) & set(V2_SEALED_BLOCK)
    assert not set(MODULE.SEALED_EVAL_SEEDS) & set(RESERVED_SEALED_BLOCK)
    assert not set(MODULE.SEALED_EVAL_SEEDS) & set(DISCOVERY_SEALED_BLOCK)
    assert not set(MODULE.SEALED_EVAL_SEEDS) & set(SHEAF_SEALED_BLOCK)
    assert not set(MODULE.SEALED_EVAL_SEEDS) & set(GROUP_SEALED_BLOCK)
    assert not set(FIXTURE_SEEDS) & set(V1_SEALED_BLOCK)
    assert not set(FIXTURE_SEEDS) & set(V2_SEALED_BLOCK)
    assert not set(FIXTURE_SEEDS) & set(RESERVED_SEALED_BLOCK)
    assert not set(FIXTURE_SEEDS) & set(DISCOVERY_SEALED_BLOCK)
    assert not set(FIXTURE_SEEDS) & set(SHEAF_SEALED_BLOCK)
    assert not set(FIXTURE_SEEDS) & set(GROUP_SEALED_BLOCK)
    assert not set(FIXTURE_SEEDS) & set(LOOP_SEALED_BLOCK)


def test_frozen_design_constants() -> None:
    assert MODULE.NUM_OBSERVATIONS == 16
    assert MODULE.LOOP.alarm_tol == ALARM_TOL == 1e-9
    assert MODULE.LOOP.misfit_tol == CERT_TOL == 1e-10
    assert MODULE.LOOP.novelty_tol == 1e-6
    assert MODULE.LOOP.num_observations == 16
    assert MODULE.CONDITIONS == ("in-library", "out-of-library", "null-control")
    assert MODULE.TRAIN_SEED_BLOCK == {"first": 0, "last": 0}
    assert MODULE.PER_CLAIM_ALPHA == 0.0125
    assert MODULE.MIN_ELIGIBLE == 30
    assert MODULE.CLAIM_IDS == (
        "h1-loop-acquisition-rate",
        "h2-loop-false-admission-control",
        "h3-loop-in-library-routing",
        "h4-loop-alarm-precision",
    )
    by_id = {d["id"]: d for d in MODULE._CLAIM_DEFINITIONS}
    assert by_id["h1-loop-acquisition-rate"]["role"] == "primary"
    for claim_id in MODULE.CLAIM_IDS:
        assert by_id[claim_id]["threshold"] == 0.95
    assert by_id["h2-loop-false-admission-control"]["role"] == "secondary"
    assert by_id["h3-loop-in-library-routing"]["role"] == "secondary"
    assert by_id["h4-loop-alarm-precision"]["role"] == "secondary"
    assert [d["condition"] for d in MODULE._CLAIM_DEFINITIONS] == [
        "out-of-library",
        "null-control",
        "in-library",
        "in-library",
    ]
    assert [d["statistic"] for d in MODULE._CLAIM_DEFINITIONS] == [
        "acquisition",
        "refusal",
        "routing",
        "alarm_silence",
    ]
    assert [d["value"] for d in MODULE._CLAIM_DEFINITIONS] == [
        "acquired",
        "control_refused",
        "routed_correctly",
        "alarm_silent",
    ]
    assert "demo-scale" in MODULE.DECLARED_CAVEAT
    assert "Gaussian null" in MODULE.DECLARED_CAVEAT
    # The runner ships with the placeholder until the protocol hash is
    # pinned; in either state the fail-closed behavior is pinned.
    if MODULE.PROTOCOL_SHA256 == "PENDING_PROTOCOL_SHA256":
        with pytest.raises(RuntimeError, match="placeholder"):
            MODULE._frozen_protocol_sha256()
    else:
        assert len(MODULE.PROTOCOL_SHA256) == 64
        assert MODULE._frozen_protocol_sha256() == MODULE.PROTOCOL_SHA256


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
    assert seal["schema"] == "universa-seal/7"
    assert seal["train_seed_block"] == {"first": 0, "last": 0}
    assert seal["eval_seed_block"] == {"first": 140001, "last": 140036}
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
    _write_seal(tmp_path, _seal_payload(schema="universa-seal/4"))
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
    # The train block is the documented-empty sentinel and nothing else.
    _write_seal(
        tmp_path, _seal_payload(train_seed_block={"first": 0, "last": 1})
    )
    with pytest.raises(RuntimeError, match="train_seed_block"):
        MODULE._load_seal(tmp_path, "seal.json")
    _write_seal(
        tmp_path,
        _seal_payload(train_seed_block={"first": 50001, "last": 50200}),
    )
    with pytest.raises(RuntimeError, match="train_seed_block"):
        MODULE._load_seal(tmp_path, "seal.json")
    _write_seal(tmp_path, _seal_payload(train_seed_block=[0, 0]))
    with pytest.raises(RuntimeError, match="train_seed_block"):
        MODULE._load_seal(tmp_path, "seal.json")
    _write_seal(
        tmp_path,
        _seal_payload(eval_seed_block={"first": 140000, "last": 140036}),
    )
    with pytest.raises(RuntimeError, match="eval_seed_block"):
        MODULE._load_seal(tmp_path, "seal.json")
    _write_seal(
        tmp_path,
        _seal_payload(eval_seed_block={"first": 140001, "last": 140037}),
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
        ("statistic", "coverage"),
        ("threshold", 0.9),
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
        _seal_payload(train_seed_block={"first": 0, "last": 0, "note": "x"}),
    )
    with pytest.raises(RuntimeError, match="train_seed_block"):
        MODULE._load_seal(tmp_path, "seal.json")


def test_load_seal_rejects_unknown_keys_inside_a_claim(tmp_path: Path) -> None:
    family = [_claim_object(d) for d in MODULE._CLAIM_DEFINITIONS]
    extra = _claim_object(MODULE._CLAIM_DEFINITIONS[0])
    extra["surprise"] = "outside the pinned claim key set"
    _write_seal(tmp_path, _seal_payload(primary_family=[extra, *family[1:]]))
    with pytest.raises(RuntimeError, match="unknown keys"):
        MODULE._load_seal(tmp_path, "seal.json")
    # The optional role key stays allowed when it matches the frozen
    # definition, and the well-formed seal still validates.
    with_role = _claim_object(MODULE._CLAIM_DEFINITIONS[0])
    with_role["role"] = MODULE._CLAIM_DEFINITIONS[0]["role"]
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
        emptied = _claim_object(MODULE._CLAIM_DEFINITIONS[0])
        emptied[key] = "   "
        _write_seal(
            tmp_path,
            _seal_payload(primary_family=[emptied, *family[1:]]),
        )
        with pytest.raises(RuntimeError, match="descriptive subfields"):
            MODULE._load_seal(tmp_path, "seal.json")
        nonstring = _claim_object(MODULE._CLAIM_DEFINITIONS[0])
        nonstring[key] = 0
        _write_seal(
            tmp_path,
            _seal_payload(primary_family=[nonstring, *family[1:]]),
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
    assert provenance["seal"]["schema"] == "universa-seal/7"
    assert provenance["seal"]["committed_at_head"] is True
    assert provenance["protocol"]["path"] == "docs/19-sealed-router-loop-protocol.md"
    assert provenance["protocol"]["sha256"] == hashes["protocol_hash"]
    assert provenance["runner"]["path"] == "scripts/run_loop_sealed_1.py"
    assert provenance["runner"]["sha256"] == hashes["runner_hash"]
    assert provenance["code_manifest"]["files"] == hashes["manifest"]
    assert len(provenance["code_manifest"]["sha256"]) == 64
    execution = provenance["execution"]
    assert execution["array_dtype"] == "float64"
    assert execution["cuda_available"] is False
    assert execution["torch_importable"] is True
    assert execution["torch_num_threads"] == 1
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
    # A fabricated-but-dangling 40-hex commit — an object that EXISTS in the
    # repository (cat-file passes) but hangs off no reachable history —
    # must be refused: existence alone does not pin the sealed design into
    # the committed history.
    dangling = "b" * 40
    _write_seal(tmp_path, _seal_payload(design_commit=dangling))

    def fake_git(root: Path, *args: str) -> str:
        if args[:2] == ("merge-base", "--is-ancestor"):
            raise RuntimeError(
                f"stop condition: git merge-base --is-ancestor {dangling} "
                "HEAD failed: not an ancestor"
            )
        return _fake_git_clean(root, *args)

    monkeypatch.setattr(MODULE, "_git_checked", fake_git)
    with pytest.raises(RuntimeError, match="ancestor"):
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
# Environment: numpy-only CPU float64, guarded torch uniformity.


def test_execution_environment_pins_cpu_float64_numpy() -> None:
    env = MODULE._execution_environment()
    assert env["array_dtype"] == "float64"
    assert "no torch tensors" in env["compute"]
    assert env["cuda_visible_devices"] == ""
    assert (
        env["cuda_visible_devices_operator"]
        == MODULE.OPERATOR_CUDA_VISIBLE_DEVICES
    )
    if MODULE.torch is not None:
        assert env["torch_importable"] is True
        assert env["torch_num_threads"] == 1
        assert env["cuda_available"] is False
    else:  # pragma: no cover - torch is installed in this environment
        assert env["torch_importable"] is False
        assert env["torch_num_threads"] is None
        assert env["cuda_available"] is None


def test_execution_environment_refuses_a_visible_cuda(monkeypatch) -> None:
    if MODULE.torch is None:  # pragma: no cover - torch is installed here
        pytest.skip("torch is not importable in this environment")
    monkeypatch.setattr(MODULE.torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="CUDA"):
        MODULE._execution_environment()


def test_execution_environment_without_torch_skips_torch_checks(
    monkeypatch,
) -> None:
    # The guarded-import convention: without torch the runner still pins
    # CPU float64 numpy and records the torch checks as not applicable.
    monkeypatch.setattr(MODULE, "torch", None)
    env = MODULE._execution_environment()
    assert env["torch_importable"] is False
    assert env["torch_num_threads"] is None
    assert env["torch_num_interop_threads"] is None
    assert env["cuda_available"] is None
    assert env["array_dtype"] == "float64"
    provenance = MODULE._environment_provenance()
    assert provenance["torch"] is None
    assert provenance["torch_cuda_build"] is None


def test_preflight_refuses_a_visible_cuda_device(
    monkeypatch, tmp_path: Path
) -> None:
    if MODULE.torch is None:  # pragma: no cover - torch is installed here
        pytest.skip("torch is not importable in this environment")
    hashes = _stage_sealed_tree(tmp_path)
    _patch_staged_preflight(monkeypatch, tmp_path, hashes)
    monkeypatch.setattr(MODULE.torch.cuda, "is_available", lambda: True)
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
# Row construction and the three frozen conditions (fixture seeds only).


def test_in_library_row_routes_to_the_truth_with_a_silent_alarm() -> None:
    # The route path: the library contains the truth, so the certified
    # scores keep the alarm quiet and the loop commits to index 0.
    for seed in FIXTURE_SEEDS:
        instance, in_library, _ = make_loop_instance(seed)
        row = MODULE._build_in_library_row(seed, instance, in_library)
        assert row["seed"] == seed
        assert row["condition"] == "in-library"
        assert row["mode"] == "route"
        assert row["alarm_fired"] is False
        assert row["alarm_silent"] is True
        assert row["routing_correct"] is True
        assert row["routed_index"] == 0
        # Exact zeros asserted, never tolerated; decoys clear the alarm.
        assert row["best_score"] == 0.0
        assert row["scores"][0] == 0.0
        assert all(score > ALARM_TOL for score in row["scores"][1:])
        assert len(row["scores"]) == len(in_library) == 4
        assert row["initial_library_size"] == 4
        assert row["final_library_size"] == 4
        assert row["admitted"] is False
        # No discovery phase happened.
        assert row["discovery_verdict"] is None
        assert row["discovery_reason"] is None
        assert row["certificate_residual"] is None
        assert row["map_misfit"] is None
        assert row["num_observations"] is None
        assert row["admission_distances"] is None
        assert row["admission_reason"] is None
        # The row records exactly what the certified loop reports.
        outcome = run_loop(LibraryLoop(), instance, in_library, seed=seed)
        assert row["scores"] == list(outcome.scores)
        assert row["mode"] == outcome.mode
        # The row construction is exactly deterministic.
        assert row == MODULE._build_in_library_row(seed, instance, in_library)


def test_out_of_library_row_acquires_the_discovered_structure() -> None:
    # The alarm path into discovery: the truth withheld, the alarm fires,
    # the head certifies, admission accepts, and the planted map commutes.
    loop = LibraryLoop()
    for seed in FIXTURE_SEEDS:
        instance, _, out_library = make_loop_instance(seed)
        row = MODULE._build_out_of_library_row(seed, instance, out_library)
        assert row["seed"] == seed
        assert row["condition"] == "out-of-library"
        assert row["mode"] == "discover"
        assert row["alarm_fired"] is True
        assert row["acquisition_correct"] is True
        assert row["discovery_verdict"] == "discovered"
        assert row["discovery_reason"]
        assert row["certificate_residual"] <= loop.misfit_tol
        assert row["admitted"] is True
        assert row["admission_distances"] is not None
        assert len(row["admission_distances"]) == len(out_library) == 3
        assert min(row["admission_distances"]) == row["admission_min_distance"]
        assert row["admission_min_distance"] > loop.novelty_tol
        assert row["map_misfit"] <= ALARM_TOL
        # The library grew by one and the instance re-routed to it.
        assert row["initial_library_size"] == len(out_library)
        assert row["final_library_size"] == len(out_library) + 1
        assert row["routed_index"] == row["initial_library_size"]
        assert row["num_observations"] == loop.num_observations
        # Every decoy score cleared the alarm threshold.
        assert all(score > ALARM_TOL for score in row["scores"])
        assert row["best_score"] == min(row["scores"])
        # The row records exactly what the certified loop reports.
        outcome = run_loop(loop, instance, out_library, seed=seed)
        assert row["scores"] == list(outcome.scores)
        assert row["certificate_residual"] == outcome.certificate_residual
        assert row["map_misfit"] == outcome.map_misfit
        # The row construction is exactly deterministic.
        assert row == MODULE._build_out_of_library_row(seed, instance, out_library)


def test_null_control_row_refuses_structure_free_observations() -> None:
    # The refusal path: the alarm fires on the decoy library, then the
    # frozen Gaussian null schedule must be refused — never certified and
    # admitted into the library.
    for seed in FIXTURE_SEEDS:
        instance, _, out_library = make_loop_instance(seed)
        row = MODULE._build_null_control_row(seed, instance, out_library)
        assert row["seed"] == seed
        assert row["condition"] == "null-control"
        assert row["mode"] == "refused"
        assert row["refused"] is True
        assert row["alarm_fired"] is True  # the alarm correctly fires
        assert row["discovery_verdict"] == "insufficient"
        assert isinstance(row["discovery_reason"], str)
        assert row["discovery_reason"]
        assert row["admitted"] is False
        assert row["routed_index"] is None
        assert row["map_misfit"] is None
        assert row["admission_distances"] is None
        assert row["admission_reason"] is None
        assert row["num_observations"] == MODULE.NUM_OBSERVATIONS
        assert row["initial_library_size"] == len(out_library)
        assert row["final_library_size"] == len(out_library)
        # The null schedule itself is the frozen exp4 H4 draw.
        ambient = instance.true_target.boundaries[0].shape[1]
        assert np.array_equal(
            null_observations(seed, ambient, MODULE.NUM_OBSERVATIONS),
            null_observations(seed, ambient),
        )
        # The row construction is exactly deterministic.
        assert row == MODULE._build_null_control_row(seed, instance, out_library)


def test_row_builders_fail_closed_on_certified_machinery_violations(
    monkeypatch,
) -> None:
    def broken_loop(*_args, **_kwargs):
        raise ValueError("certified machinery rejected the instance")

    monkeypatch.setattr(MODULE, "run_loop", broken_loop)
    instance, in_library, out_library = make_loop_instance(70001)
    with pytest.raises(MODULE.DesignFailureError, match="ValueError") as main:
        MODULE._build_in_library_row(70001, instance, in_library)
    assert main.value.seed == 70001
    with pytest.raises(MODULE.DesignFailureError, match="ValueError") as ctrl:
        MODULE._build_out_of_library_row(70001, instance, out_library)
    assert ctrl.value.seed == 70001
    with pytest.raises(MODULE.DesignFailureError, match="ValueError") as null:
        MODULE._build_null_control_row(70001, instance, out_library)
    assert null.value.seed == 70001


def test_null_control_row_fails_closed_on_a_schedule_violation(
    monkeypatch,
) -> None:
    def broken_null(*_args, **_kwargs):
        raise ValueError("null schedule rejected the ambient dimension")

    monkeypatch.setattr(MODULE, "null_observations", broken_null)
    instance, _, out_library = make_loop_instance(70001)
    with pytest.raises(MODULE.DesignFailureError, match="ValueError") as null:
        MODULE._build_null_control_row(70001, instance, out_library)
    assert null.value.seed == 70001


def test_foreign_outcome_type_is_a_design_failure_not_an_execution_failure(
    monkeypatch, tmp_path: Path
) -> None:
    # The symmetric check to the discovery runner's foreign-result check: a
    # run_loop result that is not a LoopOutcome is a whole-run design
    # failure, never an AttributeError downstream.
    _patch_tiny_campaign(monkeypatch)
    monkeypatch.setattr(MODULE, "run_loop", lambda *_args, **_kwargs: object())
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure"
    assert report["failure"]["seed"] == 70003
    assert report["failure"]["phase"] == "campaign"
    assert report["failure"]["type"] == "DesignFailureError"
    assert "foreign" in report["failure"]["message"]
    # No row of any condition completed for seed 70003.
    assert report["in_library_rows"] == []
    assert report["out_of_library_rows"] == []
    assert report["null_control_rows"] == []


def test_per_seed_statistics_read_the_recorded_indicators() -> None:
    # The frozen convention: each statistic is the recorded decision
    # indicator of its condition's row — no conditioning on loop success.
    in_row = {"seed": 1, "routing_correct": True, "alarm_silent": False}
    out_row = {"seed": 1, "acquisition_correct": False}
    null_row = {"seed": 1, "refused": True}
    (entry,) = MODULE._recompute_per_seed([in_row], [out_row], [null_row])
    assert entry == {
        "seed": 1,
        "acquired": 0.0,
        "control_refused": 1.0,
        "routed_correctly": 1.0,
        "alarm_silent": 0.0,
    }
    in_row = {"seed": 2, "routing_correct": False, "alarm_silent": True}
    out_row = {"seed": 2, "acquisition_correct": True}
    null_row = {"seed": 2, "refused": False}
    (entry,) = MODULE._recompute_per_seed([in_row], [out_row], [null_row])
    assert entry == {
        "seed": 2,
        "acquired": 1.0,
        "control_refused": 0.0,
        "routed_correctly": 0.0,
        "alarm_silent": 1.0,
    }


def test_recompute_per_seed_fails_closed_on_duplicates_and_gaps() -> None:
    in_row = {"seed": 1, "routing_correct": True, "alarm_silent": True}
    out_row = {"seed": 1, "acquisition_correct": True}
    null_row = {"seed": 1, "refused": True}
    with pytest.raises(MODULE.DesignFailureError, match="duplicate"):
        MODULE._recompute_per_seed([in_row], [out_row, out_row], [null_row])
    with pytest.raises(MODULE.DesignFailureError, match="duplicate"):
        MODULE._recompute_per_seed([in_row], [out_row], [null_row, null_row])
    with pytest.raises(MODULE.DesignFailureError, match="no matching"):
        MODULE._recompute_per_seed([in_row], [], [null_row])
    with pytest.raises(MODULE.DesignFailureError, match="no matching"):
        MODULE._recompute_per_seed([in_row], [out_row], [])


# ---------------------------------------------------------------------------
# Eligibility (real instances, non-sealed seeds only).


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
        assert set(record) == {
            "seed",
            "eligible",
            "reason",
            "true_misfit_undegraded",
            "decoy_misfits_undegraded",
        }
        assert record["eligible"] is True
        assert record["reason"] is None
        assert record["true_misfit_undegraded"] == 0.0
        assert all(
            misfit > MODULE.ALARM_TOL
            for misfit in record["decoy_misfits_undegraded"]
        )


def test_undegraded_audit_failure_marks_one_seed_ineligible_with_reason(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(
        monkeypatch, eval_seeds=(70001, 70002, 70003), min_eligible=1
    )
    real_misfits = MODULE._undegraded_misfits

    def poisoned(instance):
        misfits = list(real_misfits(instance))
        if instance.seed == 70002:
            misfits[0] = 8.8e-3
        return tuple(misfits)

    monkeypatch.setattr(MODULE, "_undegraded_misfits", poisoned)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["eligibility"]["attempted"] == 3
    assert report["eligibility"]["eligible"] == 2
    assert report["eligibility"]["eligible_seeds"] == [70001, 70003]
    (entry,) = report["eligibility"]["ineligible"]
    assert entry["seed"] == 70002
    assert "true candidate undegraded misfit" in entry["reason"]
    # An audit-ineligible seed contributes no rows at any condition.
    row_seeds = set()
    for key in ("in_library_rows", "out_of_library_rows", "null_control_rows"):
        row_seeds.update(row["seed"] for row in report[key])
    assert row_seeds == {70001, 70003}


def test_undegraded_decoy_failure_marks_one_seed_ineligible(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(
        monkeypatch, eval_seeds=(70001, 70002, 70003), min_eligible=1
    )
    real_misfits = MODULE._undegraded_misfits

    def poisoned(instance):
        misfits = list(real_misfits(instance))
        if instance.seed == 70002:
            misfits[1] = 0.0
        return tuple(misfits)

    monkeypatch.setattr(MODULE, "_undegraded_misfits", poisoned)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    (entry,) = report["eligibility"]["ineligible"]
    assert entry["seed"] == 70002
    assert "decoy undegraded misfit" in entry["reason"]
    assert report["eligibility"]["eligible_seeds"] == [70001, 70003]


def test_routing_and_alarm_silence_coincide_on_real_fixture_seeds(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(
        monkeypatch, eval_seeds=FIXTURE_SEEDS, min_eligible=1
    )
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    in_library_rows = [
        row
        for row in report["in_library_rows"]
        if row["condition"] == "in-library"
    ]
    assert len(in_library_rows) == len(FIXTURE_SEEDS)
    for row in in_library_rows:
        assert row["routing_correct"] is row["alarm_silent"]
        assert row["routing_correct"] is True


def test_alarm_is_independent_of_the_observation_override() -> None:
    from universa.loop import LibraryLoop as _Loop
    from universa.loop import null_observations as _nulls

    seed = 70001
    instance, in_library, _ = MODULE.make_loop_instance(seed)
    ambient = instance.true_target.boundaries[0].shape[1]
    plain = MODULE.run_loop(_Loop(), instance, in_library, seed=seed)
    overridden = MODULE.run_loop(
        _Loop(),
        instance,
        in_library,
        seed=seed,
        observations=_nulls(seed, ambient),
    )
    assert plain == overridden
    assert plain.alarm_fired is False
    assert overridden.mode == "route"


def test_build_exception_is_a_whole_run_design_failure(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(
        monkeypatch, eval_seeds=(70001, 70002, 70003), min_eligible=1
    )
    real_make = MODULE.make_loop_instance

    def exploding(seed, *args, **kwargs):
        if seed == 70002:
            raise ValueError("generator blew up")
        return real_make(seed, *args, **kwargs)

    monkeypatch.setattr(MODULE, "make_loop_instance", exploding)
    monkeypatch.setattr(
        MODULE,
        "_campaign_rows",
        lambda *_args: pytest.fail("rows must not be built after a build failure"),
    )
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure"
    assert report["failure"]["seed"] == 70002
    assert report["failure"]["phase"] == "eligibility"
    assert report["failure"]["type"] == "ValueError"
    assert "generator blew up" in report["failure"]["message"]
    assert report["eligibility"]["build_failures"] == [
        {"seed": 70002, "type": "ValueError", "message": "generator blew up"}
    ]
    assert report["eligibility"]["attempted"] == 2
    # Seed 70001 was fully audited before the failure and is preserved.
    assert [record["seed"] for record in report["seed_records"]] == [70001]
    assert report["in_library_rows"] == []
    assert report["out_of_library_rows"] == []
    assert report["null_control_rows"] == []
    assert report["per_seed"] == []
    claims = report["claims"]["claims"]
    assert len(claims) == 4
    assert all(claim["supported"] is None for claim in claims)


def test_interrupt_during_eligibility_is_classified_and_preserved(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(
        monkeypatch, eval_seeds=(70001, 70002, 70003), min_eligible=1
    )
    real_make = MODULE.make_loop_instance

    def interrupting(seed, *args, **kwargs):
        if seed == 70002:
            raise KeyboardInterrupt
        return real_make(seed, *args, **kwargs)

    monkeypatch.setattr(MODULE, "make_loop_instance", interrupting)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "interrupted"
    assert report["failure"]["phase"] == "eligibility"
    assert report["failure"]["type"] == "KeyboardInterrupt"
    assert [record["seed"] for record in report["seed_records"]] == [70001]
    assert report["in_library_rows"] == []
    assert report["out_of_library_rows"] == []
    assert report["null_control_rows"] == []
    assert all(
        claim["supported"] is None for claim in report["claims"]["claims"]
    )


def test_insufficient_eligible_builds_no_rows_and_nulls_claims(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(
        monkeypatch, eval_seeds=(70001, 70002, 70003), min_eligible=4
    )
    monkeypatch.setattr(
        MODULE,
        "_campaign_rows",
        lambda *_args: pytest.fail(
            "rows must not be built after eligibility failure"
        ),
    )
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure_insufficient_eligible"
    assert report["eligibility"]["eligible"] == 3
    assert report["in_library_rows"] == []
    assert report["out_of_library_rows"] == []
    assert report["null_control_rows"] == []
    assert report["per_seed"] == []
    claims = report["claims"]["claims"]
    assert len(claims) == 4
    assert {claim["id"] for claim in claims} == set(MODULE.CLAIM_IDS)
    assert all(claim["supported"] is None for claim in claims)
    assert all(claim["estimate"] is None for claim in claims)
    assert report["audit"]["in_library_rows"] == 0
    assert report["audit"]["out_of_library_rows"] == 0
    assert report["audit"]["null_control_rows"] == 0
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
        "make_loop_instance",
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
    assert report["schema"] == "universa-router-loop-sealed-result/1"
    assert report["experiment_id"] == "universa-router-loop-sealed-1"
    assert report["eligibility"]["eligible_seeds"] == [70003, 70004, 70005]
    assert "training" not in report  # the loop trains nothing
    # The TOCTOU guard records the post-run status next to the pre-run one.
    assert report["provenance"]["git_status_porcelain"] == ""
    assert report["provenance"]["git_status_porcelain_post_run"] == ""
    # The whole report must be JSON-serializable (no numpy leaks).
    json.dumps(report, sort_keys=True)

    in_library_rows = report["in_library_rows"]
    out_of_library_rows = report["out_of_library_rows"]
    null_control_rows = report["null_control_rows"]
    assert (
        len(in_library_rows)
        == len(out_of_library_rows)
        == len(null_control_rows)
        == 3
    )
    for row in in_library_rows:
        assert set(row) == {
            "seed",
            "condition",
            "mode",
            "alarm_fired",
            "best_score",
            "scores",
            "routed_index",
            "discovery_verdict",
            "discovery_reason",
            "certificate_residual",
            "admitted",
            "admission_min_distance",
            "map_misfit",
            "final_library_size",
            "initial_library_size",
            "num_observations",
            "admission_distances",
            "admission_reason",
            "routing_correct",
            "alarm_silent",
        }
        assert row["seed"] in (70003, 70004, 70005)
        assert row["condition"] == "in-library"
        assert row["mode"] == "route"
        assert row["alarm_fired"] is False
        assert row["alarm_silent"] is True
        assert row["routing_correct"] is True
        assert row["routed_index"] == 0
        assert len(row["scores"]) == 4
    for row in out_of_library_rows:
        assert set(row) == {
            "seed",
            "condition",
            "mode",
            "alarm_fired",
            "best_score",
            "scores",
            "routed_index",
            "discovery_verdict",
            "discovery_reason",
            "certificate_residual",
            "admitted",
            "admission_min_distance",
            "map_misfit",
            "final_library_size",
            "initial_library_size",
            "num_observations",
            "admission_distances",
            "admission_reason",
            "acquisition_correct",
        }
        assert row["condition"] == "out-of-library"
        assert row["mode"] == "discover"
        assert row["alarm_fired"] is True
        assert row["acquisition_correct"] is True
        assert row["certificate_residual"] <= MODULE.LOOP.misfit_tol
        assert row["map_misfit"] <= ALARM_TOL
        assert len(row["scores"]) == 3
        assert len(row["admission_distances"]) == 3
    for row in null_control_rows:
        assert set(row) == {
            "seed",
            "condition",
            "mode",
            "alarm_fired",
            "best_score",
            "scores",
            "routed_index",
            "discovery_verdict",
            "discovery_reason",
            "certificate_residual",
            "admitted",
            "admission_min_distance",
            "map_misfit",
            "final_library_size",
            "initial_library_size",
            "num_observations",
            "admission_distances",
            "admission_reason",
            "refused",
        }
        assert row["condition"] == "null-control"
        assert row["mode"] == "refused"
        assert row["refused"] is True
        assert row["alarm_fired"] is True
        assert row["routed_index"] is None

    per_seed = report["per_seed"]
    assert len(per_seed) == 3
    for entry in per_seed:
        assert set(entry) == {
            "seed",
            "acquired",
            "control_refused",
            "routed_correctly",
            "alarm_silent",
        }
        # On the fixture seeds every loop pass succeeds.
        assert entry["acquired"] == 1.0
        assert entry["control_refused"] == 1.0
        assert entry["routed_correctly"] == 1.0
        assert entry["alarm_silent"] == 1.0

    claims = {claim["id"]: claim for claim in report["claims"]["claims"]}
    assert set(claims) == set(MODULE.CLAIM_IDS)
    for claim_id, claim in claims.items():
        assert claim["n"] == 3
        assert claim["estimate"] == pytest.approx(1.0)
        assert claim["standard_error"] == pytest.approx(0.0)  # SE = 0 mechanics
        assert claim["lower_bound"] == pytest.approx(1.0)
        assert claim["supported"] is True
        assert claim["threshold"] == _definition(claim_id)["threshold"] == 0.95
        assert claim["condition"] == _definition(claim_id)["condition"]
        assert claim["statistic"] == _definition(claim_id)["statistic"]
        assert _definition(claim_id)["statistic"] in claim["estimand"]
        assert claim["lower_bound"] == pytest.approx(
            claim["estimate"]
            - claim["critical_value"] * claim["standard_error"]
        )

    audit = report["audit"]
    assert audit["declared_seeds"] == 3
    assert audit["eligible_seeds"] == 3
    assert audit["in_library_rows"] == 3
    assert audit["out_of_library_rows"] == 3
    assert audit["null_control_rows"] == 3
    assert audit["rows_per_seed_per_condition"] == 1
    assert audit["per_seed"] == per_seed
    assert "claim_estimates" not in audit
    recompute = audit["decision_recompute"]
    assert len(recompute) == 9
    by_key = {(entry["seed"], entry["condition"]): entry for entry in recompute}
    for row in in_library_rows:
        entry = by_key[(row["seed"], "in-library")]
        assert entry["mode"] == "route"
        assert entry["alarm_fired"] is False
        assert entry["routing_correct"] is True
        assert entry["alarm_silent"] is True
    for row in out_of_library_rows:
        entry = by_key[(row["seed"], "out-of-library")]
        assert entry["mode"] == "discover"
        assert entry["alarm_fired"] is True
        assert entry["acquisition_correct"] is True
    for row in null_control_rows:
        entry = by_key[(row["seed"], "null-control")]
        assert entry["mode"] == "refused"
        assert entry["refused"] is True
    for definition in MODULE._CLAIM_DEFINITIONS:
        estimate = statistics.fmean(
            entry[definition["value"]] for entry in per_seed
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


def test_complete_tiny_campaign_is_deterministic(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    first = MODULE.run(tmp_path, tmp_path / "a.json", seal="seal.json")
    second = MODULE.run(tmp_path, tmp_path / "b.json", seal="seal.json")
    assert first["status"] == second["status"] == "complete"
    assert first["in_library_rows"] == second["in_library_rows"]
    assert first["out_of_library_rows"] == second["out_of_library_rows"]
    assert first["null_control_rows"] == second["null_control_rows"]
    assert first["per_seed"] == second["per_seed"]
    assert first["claims"] == second["claims"]
    assert first["audit"] == second["audit"]


def test_tiny_campaign_instantiates_only_declared_test_seeds(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    seen = []
    real_make = MODULE.make_loop_instance

    def recording(seed, *args, **kwargs):
        seen.append(seed)
        return real_make(seed, *args, **kwargs)

    # Only the eligibility pass constructs instances (the audit recomputes
    # decisions from the raw rows alone and rebuilds nothing).
    monkeypatch.setattr(MODULE, "make_loop_instance", recording)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "complete"
    assert set(seen) == {70003, 70004, 70005}
    assert not set(seen) & set(LOOP_SEALED_BLOCK)
    assert not set(seen) & set(V1_SEALED_BLOCK)
    assert not set(seen) & set(V2_SEALED_BLOCK)
    assert not set(seen) & set(RESERVED_SEALED_BLOCK)
    assert not set(seen) & set(DISCOVERY_SEALED_BLOCK)
    assert not set(seen) & set(SHEAF_SEALED_BLOCK)
    assert not set(seen) & set(GROUP_SEALED_BLOCK)


def test_nondeterministic_rows_are_a_whole_run_design_failure(
    monkeypatch, tmp_path: Path
) -> None:
    # The frozen double execution: the two executions of a row must be
    # bit-identical, else the whole run is a design failure.
    _patch_tiny_campaign(monkeypatch)
    real_build = MODULE._build_out_of_library_row
    calls = {"count": 0}

    def drifting(seed, instance, out_library):
        row = real_build(seed, instance, out_library)
        calls["count"] += 1
        if seed == 70003 and calls["count"] == 2:
            row = dict(row)
            row["map_misfit"] = 1.0  # a second, different draw
        return row

    monkeypatch.setattr(MODULE, "_build_out_of_library_row", drifting)
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
    real_build = MODULE._build_out_of_library_row

    def flaky(seed, instance, out_library):
        if seed == 70004:
            raise FloatingPointError("unexpected numeric fault")
        return real_build(seed, instance, out_library)

    monkeypatch.setattr(MODULE, "_build_out_of_library_row", flaky)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "execution_failure"
    assert report["failure"]["seed"] == 70004
    assert report["failure"]["phase"] == "campaign"
    assert report["failure"]["type"] == "FloatingPointError"
    assert "unexpected numeric fault" in report["failure"]["message"]
    # Seed 70003 completed all three rows before the failure; 70004's
    # in-library row completed and is preserved (a completed row is
    # appended as soon as its bit-identity check passes, before the seed's
    # later conditions are built); 70005 never ran.
    assert [row["seed"] for row in report["in_library_rows"]] == [70003, 70004]
    assert [row["seed"] for row in report["out_of_library_rows"]] == [70003]
    assert [row["seed"] for row in report["null_control_rows"]] == [70003]
    assert [entry["seed"] for entry in report["per_seed"]] == [70003]
    claims = report["claims"]["claims"]
    assert len(claims) == 4
    assert all(claim["supported"] is None for claim in claims)
    assert all(claim["estimate"] is None for claim in claims)
    assert report["audit"]["in_library_rows"] == 2
    assert report["audit"]["out_of_library_rows"] == 1
    assert report["audit"]["null_control_rows"] == 1
    assert "claims" not in report["audit"]


def test_interrupted_status_preserves_rows_and_nulls_claims(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    real_build_null = MODULE._build_null_control_row

    def interrupting(seed, instance, out_library):
        if seed == 70004:
            raise KeyboardInterrupt
        return real_build_null(seed, instance, out_library)

    monkeypatch.setattr(MODULE, "_build_null_control_row", interrupting)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "interrupted"
    assert report["failure"]["seed"] == 70004
    assert report["failure"]["type"] == "KeyboardInterrupt"
    # Seed 70003 completed all three rows; seed 70004's in-library and
    # out-of-library rows completed and are preserved, while its
    # null-control row was never built — so the per-seed view covers only
    # the seed complete in all three conditions.
    assert [row["seed"] for row in report["in_library_rows"]] == [70003, 70004]
    assert [row["seed"] for row in report["out_of_library_rows"]] == [70003, 70004]
    assert [row["seed"] for row in report["null_control_rows"]] == [70003]
    assert [entry["seed"] for entry in report["per_seed"]] == [70003]
    claims = report["claims"]["claims"]
    assert len(claims) == 4
    assert all(claim["supported"] is None for claim in claims)


def test_null_control_row_failure_preserves_the_completed_rows(
    monkeypatch, tmp_path: Path
) -> None:
    # A null-control failure after completed in-library and out-of-library
    # rows must still yield a failure artifact containing them: the
    # protocol's failure paths preserve every completed raw row.
    _patch_tiny_campaign(monkeypatch)
    real_build_null = MODULE._build_null_control_row

    def flaky(seed, instance, out_library):
        if seed == 70004:
            raise FloatingPointError("null machinery fault")
        return real_build_null(seed, instance, out_library)

    monkeypatch.setattr(MODULE, "_build_null_control_row", flaky)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "execution_failure"
    assert report["failure"]["seed"] == 70004
    assert report["failure"]["phase"] == "campaign"
    assert report["failure"]["type"] == "FloatingPointError"
    # Seeds 70003 and 70004 both completed their in-library and
    # out-of-library rows; 70004's null-control row was never built, so
    # the per-seed view covers only 70003.
    assert [row["seed"] for row in report["in_library_rows"]] == [70003, 70004]
    assert [row["seed"] for row in report["out_of_library_rows"]] == [70003, 70004]
    assert [row["seed"] for row in report["null_control_rows"]] == [70003]
    assert [entry["seed"] for entry in report["per_seed"]] == [70003]
    claims = report["claims"]["claims"]
    assert len(claims) == 4
    assert all(claim["supported"] is None for claim in claims)


def test_design_failure_during_campaign_nulls_claims(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    real_build = MODULE._build_in_library_row

    def flaky(seed, instance, in_library):
        if seed == 70004:
            raise MODULE.DesignFailureError(
                "seed 70004: incoherent loop outcome",
                seed=70004,
            )
        return real_build(seed, instance, in_library)

    monkeypatch.setattr(MODULE, "_build_in_library_row", flaky)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure"
    assert report["failure"]["seed"] == 70004
    assert report["failure"]["phase"] == "campaign"
    assert report["failure"]["type"] == "DesignFailureError"
    assert "incoherent" in report["failure"]["message"]
    assert [row["seed"] for row in report["in_library_rows"]] == [70003]
    claims = report["claims"]["claims"]
    assert len(claims) == 4
    assert all(claim["supported"] is None for claim in claims)


def test_row_count_invariant_fails_closed_before_claim_inference(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    real_campaign = MODULE._campaign_rows

    def shrinking(eligible, in_rows, out_rows, null_rows):
        real_campaign(eligible, in_rows, out_rows, null_rows)
        null_rows.pop()  # a silently dropped null-control row

    monkeypatch.setattr(MODULE, "_campaign_rows", shrinking)
    report = MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")
    assert report["status"] == "design_failure"
    assert report["failure"]["type"] == "DesignFailureError"
    assert "row-count invariant" in report["failure"]["message"]
    assert all(
        claim["supported"] is None for claim in report["claims"]["claims"]
    )


# ---------------------------------------------------------------------------
# Claim decision logic on synthetic per-seed statistics.


def test_claim_decision_logic_supported_and_not_supported_sides() -> None:
    h1 = _definition("h1-loop-acquisition-rate")
    h3 = _definition("h3-loop-in-library-routing")
    h4 = _definition("h4-loop-alarm-precision")
    n = 30
    # A perfect indicator block is supported; a constant block has SE
    # exactly 0, so the lower bound equals the estimate (the indicator-mean
    # mechanics).
    claim = MODULE._claim_summary(h1, [1.0] * n)
    assert claim["supported"] is True
    assert claim["n"] == n
    assert claim["critical_value"] == MODULE._T_ONE_SIDED_BONFERRONI[30]
    assert claim["estimate"] == pytest.approx(1.0)
    assert claim["standard_deviation"] == pytest.approx(0.0)
    assert claim["standard_error"] == pytest.approx(0.0)
    assert claim["lower_bound"] == pytest.approx(1.0)
    # Exactly at the threshold: the strict > rule does not support.
    at = MODULE._claim_summary(h1, [0.95] * n)
    assert at["supported"] is False
    assert at["standard_error"] == pytest.approx(0.0)
    assert at["lower_bound"] == pytest.approx(0.95)
    # Below the threshold: not supported.
    below = MODULE._claim_summary(h1, [0.94] * n)
    assert below["supported"] is False
    assert below["lower_bound"] == pytest.approx(0.94)
    assert MODULE._claim_summary(h1, [0.0] * n)["supported"] is False
    # H3's routing indicator obeys the same one-sided rule at 0.95.
    assert MODULE._claim_summary(h3, [1.0] * n)["supported"] is True
    assert MODULE._claim_summary(h3, [0.95] * n)["supported"] is False
    # H4's alarm-silence indicator obeys the same one-sided rule at 0.95.
    assert MODULE._claim_summary(h4, [1.0] * n)["supported"] is True
    assert MODULE._claim_summary(h4, [0.95] * n)["supported"] is False
    # One false alarm in 30 kills the rate claim: the lower bound of
    # 29 x 1.0 + 1 x 0.0 falls below 0.95.
    one_flake = [1.0] * 29 + [0.0]
    flaky = MODULE._claim_summary(h4, one_flake)
    assert flaky["estimate"] == pytest.approx(29 / 30)
    assert flaky["standard_error"] > 0.0
    assert flaky["lower_bound"] < 0.95
    assert flaky["supported"] is False


def test_claim_decision_logic_arithmetic_and_critical_value_by_n() -> None:
    h1 = _definition("h1-loop-acquisition-rate")
    n = 30
    values = [0.95 + 0.001 * index for index in range(n)]
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
    # Noise around the threshold: the one-sided bound crosses below 0.95.
    noisy = [0.95 + 0.02 * ((-1) ** index) for index in range(n)]
    claim = MODULE._claim_summary(h1, noisy)
    assert claim["estimate"] == pytest.approx(0.95)
    assert claim["lower_bound"] < 0.95
    assert claim["supported"] is False
    # The critical value tracks the eligible n.
    claim31 = MODULE._claim_summary(h1, [1.0] * 31)
    assert claim31["n"] == 31
    assert claim31["critical_value"] == MODULE._T_ONE_SIDED_BONFERRONI[31]


def test_claim_summary_refuses_an_unpinned_n() -> None:
    with pytest.raises(MODULE.DesignFailureError, match="critical value"):
        MODULE._claim_summary(
            _definition("h1-loop-acquisition-rate"), [1.0] * 29
        )


def test_claim_inference_routes_each_condition_statistic_to_its_claim() -> None:
    # 30 synthetic seeds; the four statistics disagree, so the routing of
    # each claim to its condition's statistic is fully exposed.
    def per_seed(
        acquired: float,
        control_refused: float,
        routed_correctly: float,
        alarm_silent: float,
    ) -> list[dict]:
        return [
            {
                "seed": seed,
                "acquired": acquired,
                "control_refused": control_refused,
                "routed_correctly": routed_correctly,
                "alarm_silent": alarm_silent,
            }
            for seed in range(30)
        ]

    claims = {
        claim["id"]: claim
        for claim in MODULE._claim_inference(
            per_seed(1.0, 0.0, 1.0, 0.0)
        )["claims"]
    }
    assert claims["h1-loop-acquisition-rate"]["supported"] is True
    assert claims["h2-loop-false-admission-control"]["supported"] is False
    assert claims["h3-loop-in-library-routing"]["supported"] is True
    assert claims["h4-loop-alarm-precision"]["supported"] is False
    claims = {
        claim["id"]: claim
        for claim in MODULE._claim_inference(
            per_seed(0.0, 1.0, 0.0, 1.0)
        )["claims"]
    }
    assert claims["h1-loop-acquisition-rate"]["supported"] is False
    assert claims["h2-loop-false-admission-control"]["supported"] is True
    assert claims["h3-loop-in-library-routing"]["supported"] is False
    assert claims["h4-loop-alarm-precision"]["supported"] is True
    # Estimand text names the routed statistic and its condition.
    family = MODULE._claim_inference(per_seed(1.0, 1.0, 1.0, 1.0))["claims"]
    assert "acquisition" in family[0]["estimand"]
    assert "out-of-library" in family[0]["estimand"]
    assert "refusal" in family[1]["estimand"]
    assert "null-control" in family[1]["estimand"]
    assert "routing" in family[2]["estimand"]
    assert "in-library" in family[2]["estimand"]
    assert "alarm_silence" in family[3]["estimand"]
    assert "in-library" in family[3]["estimand"]


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
        assert claim["threshold"] == 0.95
    statistics_ = {claim["id"]: claim["statistic"] for claim in claims}
    assert statistics_["h1-loop-acquisition-rate"] == "acquisition"
    assert statistics_["h2-loop-false-admission-control"] == "refusal"
    assert statistics_["h3-loop-in-library-routing"] == "routing"
    assert statistics_["h4-loop-alarm-precision"] == "alarm_silence"
    conditions = {claim["id"]: claim["condition"] for claim in claims}
    assert conditions["h1-loop-acquisition-rate"] == "out-of-library"
    assert conditions["h2-loop-false-admission-control"] == "null-control"
    assert conditions["h3-loop-in-library-routing"] == "in-library"
    assert conditions["h4-loop-alarm-precision"] == "in-library"


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
    assert captured["seal"] == MODULE.DEFAULT_SEAL == "docs/20-router-loop-seal.json"
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
            return " M src/universa/loop.py"
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
        lambda _root: {"src/universa/loop.py": "0" * 64},
    )
    with pytest.raises(RuntimeError, match="changed during execution"):
        MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")


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

    def swapping(eligible, in_rows, out_rows, null_rows):
        real_campaign(eligible, in_rows, out_rows, null_rows)
        (tmp_path / "seal.json").write_text(
            '{"stub": "swapped"}\n', encoding="utf-8"
        )

    monkeypatch.setattr(MODULE, "_campaign_rows", swapping)
    with pytest.raises(
        RuntimeError, match="design seal changed during execution"
    ):
        MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")


def test_complete_run_refuses_a_protocol_swapped_mid_run(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_tiny_campaign(monkeypatch)
    real_campaign = MODULE._campaign_rows

    def swapping(eligible, in_rows, out_rows, null_rows):
        real_campaign(eligible, in_rows, out_rows, null_rows)
        (tmp_path / "protocol.md").write_text(
            "swapped protocol\n", encoding="utf-8"
        )

    monkeypatch.setattr(MODULE, "_campaign_rows", swapping)
    with pytest.raises(
        RuntimeError, match="protocol changed during execution"
    ):
        MODULE.run(tmp_path, tmp_path / "out.json", seal="seal.json")


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
    with pytest.raises(
        RuntimeError, match="running file changed during execution"
    ):
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
    assert "raw rows alone" in report["audit"]["recompute_scope"]
    assert "no hash-pinned model exists" in report["audit"]["recompute_scope"]
    in_library_rows = report["in_library_rows"]
    out_by_seed = {row["seed"]: row for row in report["out_of_library_rows"]}
    null_by_seed = {row["seed"]: row for row in report["null_control_rows"]}
    claims = {claim["id"]: claim for claim in report["claims"]["claims"]}
    for definition in MODULE._CLAIM_DEFINITIONS:
        values = []
        for in_row in in_library_rows:
            out_row = out_by_seed[in_row["seed"]]
            null_row = null_by_seed[in_row["seed"]]
            per_seed = {
                "acquired": 1.0 if out_row["acquisition_correct"] else 0.0,
                "control_refused": 1.0 if null_row["refused"] else 0.0,
                "routed_correctly": 1.0 if in_row["routing_correct"] else 0.0,
                "alarm_silent": 1.0 if in_row["alarm_silent"] else 0.0,
            }
            values.append(per_seed[definition["value"]])
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


def test_audit_recompute_fails_closed_on_a_tampered_alarm_flag() -> None:
    instance, in_library, _ = make_loop_instance(70003)
    row = MODULE._build_in_library_row(70003, instance, in_library)
    assert row["mode"] == "route"
    tampered = dict(row)
    tampered["alarm_fired"] = True  # contradicts the retained best_score
    with pytest.raises(MODULE.DesignFailureError, match="seed 70003"):
        MODULE._audit_row_decisions(tampered)


def test_audit_recompute_fails_closed_on_a_tampered_score() -> None:
    instance, _, out_library = make_loop_instance(70003)
    row = MODULE._build_out_of_library_row(70003, instance, out_library)
    assert row["mode"] == "discover"
    tampered = dict(row)
    tampered["scores"] = list(row["scores"])
    tampered["scores"][0] = 0.0  # the alarm would no longer fire
    with pytest.raises(MODULE.DesignFailureError, match="seed 70003"):
        MODULE._audit_row_decisions(tampered)


def test_audit_recompute_fails_closed_on_a_tampered_indicator() -> None:
    instance, _, out_library = make_loop_instance(70003)
    row = MODULE._build_out_of_library_row(70003, instance, out_library)
    assert row["acquisition_correct"] is True
    tampered = dict(row)
    tampered["acquisition_correct"] = False
    with pytest.raises(MODULE.DesignFailureError, match="seed 70003"):
        MODULE._audit_row_decisions(tampered)
    # A tampered routed index contradicts the first-argmin recompute.
    instance, in_library, _ = make_loop_instance(70003)
    row = MODULE._build_in_library_row(70003, instance, in_library)
    tampered = dict(row)
    tampered["routed_index"] = 1
    with pytest.raises(MODULE.DesignFailureError, match="seed 70003"):
        MODULE._audit_row_decisions(tampered)
    # An unknown mode and an unknown condition are refused as well.
    tampered = dict(row)
    tampered["mode"] = "teleport"
    with pytest.raises(MODULE.DesignFailureError, match="unknown loop mode"):
        MODULE._audit_row_decisions(tampered)
    tampered = dict(row)
    tampered["condition"] = "sideways"
    with pytest.raises(MODULE.DesignFailureError, match="unknown condition"):
        MODULE._audit_row_decisions(tampered)


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
    output = tmp_path / "results" / "experiments" / "router-loop-sealed-1.json"
    code = MODULE.main(
        ["--project-root", str(tmp_path), "--output", str(output)]
    )
    assert code == 2
    assert not output.exists()  # the canonical path stays reserved
    failure_path = (
        output.parent / "failures" / "router-loop-sealed-1.design_failure.json"
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
    output = tmp_path / "results" / "experiments" / "router-loop-sealed-1.json"
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
        "scripts/run_loop_sealed_1.py --output "
        "results/experiments/router-loop-sealed-1.json"
    )


def test_design_record_declares_the_empty_train_block_and_caveat() -> None:
    design = MODULE._design_record()
    assert design["train_seed_block"] == {"first": 0, "last": 0}
    assert "no training occurred" in design["training"]
    assert design["eval_seeds"] == {"first": 140001, "last": 140036}
    assert design["procedure"]["num_observations"] == 16
    assert design["procedure"]["alarm_tol"] == 1e-9
    assert design["procedure"]["misfit_tol"] == CERT_TOL
    assert design["procedure"]["novelty_tol"] == 1e-6
    assert design["procedure"]["map_misfit_tol"] == 1e-9
    assert "instance.candidates" in design["procedure"]["in_library_call"]
    assert "instance.decoy_targets" in design["procedure"]["out_of_library_call"]
    assert "null_observations" in design["procedure"]["null_control_call"]
    assert "discovery-null" in design["conditions"]["null-control"]
    assert "build" in design["eligibility_rule"]
    assert "3 x n" in design["row_count_invariant"]
    assert "bit-identical" in design["determinism"]
    assert design["declared_caveat"] == MODULE.DECLARED_CAVEAT
    assert design["no_outcome_dependent_stopping"] is True
    assert design["no_seed_deletion"] is True


def test_source_wording_and_no_torch_computation() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    # The placeholder wording is stale once the hash is pinned: the
    # committed runner embeds the sealed protocol hash and any protocol
    # edit requires re-pinning.
    assert "fail-closed placeholder" not in source
    assert "re-pinning" in source
    # The recorded key names match what the runner actually runs.
    assert "git_status_short" not in source
    assert "git_status_porcelain" in source
    # torch appears exactly once, as the guarded import for the environment
    # check; nothing computes with torch.
    assert source.count("import torch") == 1
    assert "torch.tensor(" not in source
    assert "torch.no_grad" not in source
    assert "torch.optim" not in source
    assert "torch.manual_seed" not in source
    # The runner never reaches outside its frozen module set.
    assert "run_discovery_sealed_1" not in source
    assert "run_router_v2_sealed_1" not in source
    assert "run_router_v1_sealed_1" not in source
    assert "run_2complex_sealed_1" not in source
    assert "run_sheaf_sealed_1" not in source
    assert "run_group_sealed_1" not in source
    assert "import scripts" not in source
