# Universa continuation handoff — 2026-09-01

Status time: 2026-09-01. Repository: `seanm27lol/Universa` (private).
Branch: `main`. HEAD at writing: `6f78d04` ("Record sealed router-loop-v3
results, validation, and frozen h3/h4 boundaries"). Worktree clean.
Suite: **999 passed** (~30s with
`PYTHONPATH=src /home/seanjazm27/HOMYMOLY/.venv/bin/python -m pytest -q`).
Read this before running or changing anything.

## 0. What exists

Nine sealed experiments, all under the two-commit ceremony, all
independently validated, all artifacts committed. Evidence spine
(protocol -> seal -> results): docs/01-03 (router-v1), 04-06 (router-v2),
07-09 (router-2complex), 10-12 (discovery), 13-15 (router-sheaf),
16-18 (router-group), 19-21 (router-loop), 22-24 (router-loop-v2),
25-27 (router-loop-v3).

Library: 19 modules under `src/universa/` (chain-complex format, nerve,
operators, generators, 4 structure families, 4 observation-model modules,
budgets, multihop, discovery, loop, loop_v2, router, router_v2,
partial*, sheaves, structures). Runners under `scripts/run_*_sealed_1.py`
— **sealed runners from completed experiments are immutable; never edit
them or their tests.**

## 1. Results so far (percent form — user prefers percents)

| # | experiment | claims | headline numbers |
|---|---|---|---|
| 1 | router-v1 (graph-quotient, corrupted obs) | 4/4 | arch beats polluted oracle by 73-99 pts at held-out fractions; 0 harm at clean anchor |
| 2 | router-v2 (no-anchor) | 4/4 | beats oracle (+90 to +99 pts) and profile heuristic (+67 pts) |
| 3 | router-2complex | 4/4 | beats oracle (+82/+88/+94), heuristic (+58) |
| 4 | discovery | 4/4 unanimous | 100% coverage/certification/admission/refusal on 36/36 |
| 5 | router-sheaf | 4/4 | arch 100% at every fraction; +61 pts vs heuristic |
| 6 | router-group | 4/4 | arch 100% everywhere; narrowest baseline margins |
| 7 | router-loop (certified regime) | 4/4 unanimous | 100% acquisition/routing/alarm-precision/refusal on 36/36 |
| 8 | router-loop-v2 (comparison) | 3/4 | arch vs generic-no-architecture: **+74 pts e2e** (94% vs 19%), +56 pts in-library, +28 pts vs routing-only; generic fails everywhere (0% acquisition, 42% in-library, 17% refusal). **h4 NOT supported**: learned alarm costs exactly 1/36 seeds in-library accuracy vs always-discovering (seed 160031 named), traded against 44% fewer discovery calls (61 vs 108) |
| 9 | router-loop-v3 (alarm redesign) | 2/4 | h1 +44 pts vs generic, h2 +20 pts vs routing-only supported; **h3/h4 NOT supported**: the frozen 2% false-quiet bound eliminated false quiets completely (acquisition 83% -> 100%) but produced 41% train false-alarm rate -> 14/36 in-library harm. All 14 harm seeds named with decision chains |

The arc: win (v2), boundary (h4), redesign (v3), and a **measured Pareto
frontier**: the alarm threshold is a one-parameter dial; loop-v2 sat at
(83% acquisition, -1 seed harm), loop-v3 at (100% acquisition, -14 seeds
harm). A one-sided bound is a choice of which error to pay.

## 2. The open thread: loop-v4 (cost-aware calibration)

If continuing the alarm arc, the frozen design direction is a
**cost-aware calibration**: choose the threshold on the train block to
minimize total measured error (false_quiet_rate + false_alarm_rate),
tie-broken toward larger balanced accuracy — pricing both error modes
symmetrically instead of bounding one. Everything else (features,
models, conditions, claims) unchanged from loop-v3. Expectation: a
mid-frontier operating point; either h4 outcome is reported frozen.

- Blocks **RESERVED** (loop-v3 consumed `180001..180400`): train
  `200001..200400`, eval `210001..210036`. Word-boundary absence scan run
  2026-09-01 over the whole repo (106 text files, excluding `.git`,
  `.venv`, `.pytest_cache`): the only hits are this file's own mentions of
  the four block endpoints — inert text constants, the tolerated category.
  No runner, test, fixture, or artifact instantiates either block. Both
  are clear. Re-run the scan before the protocol anyway (cheap, and the
  guarantee is only as fresh as its last run); expect the same inert hits
  from this section plus `docs/29-writeup.md` §8.
- The write-up path is **done**: `docs/29-writeup.md` synthesizes all nine
  experiments (33/36 claims, the four-family sweep, the discovery head,
  and the alarm's Pareto frontier) from the sealed record. It quotes the
  artifacts only — no new numbers.

## 3. Ceremony playbook (any sealed experiment)

1. Absence scan for BOTH blocks:
   `for s in $(seq FIRST LAST); do grep -rIl --exclude-dir=.git --exclude-dir=.venv "$s" .; done | sort | uniq -c`
   (tolerate only inert text constants + coincidental float/hash digit collisions)
2. Protocol (12 sections, mirror docs/25) + runner (mirror
   `scripts/run_loop_v3_sealed_1.py`) + tests, written by parallel agents.
3. Insert real protocol SHA-256 into runner's `PROTOCOL_SHA256`
   (placeholder refuses all execution). Full suite green + `git diff --check`.
4. Pre-seal adversarial audits (consistency + adversarial, explore agents).
   Fix every finding before commit A.
5. Commit A (design files). Get the FULL commit hash with
   `git rev-parse HEAD` — **never invent a 40-hex hash** (this mistake
   was made once; preflight caught it).
6. Seal JSON (schema `universa-seal/N+1`: design_commit, protocol/runner
   hashes, code_manifest of all src/universa/*.py, both seed blocks,
   no_preview_declaration, claims with the 8-subfield objects, stop_rules,
   output_path). Commit B, **push before any seed is instantiated**.
7. Canonical run (background):
   `env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src /home/seanjazm27/HOMYMOLY/.venv/bin/python scripts/run_<EXP>.py --output results/experiments/<exp>.json`
8. Independent validation from raw rows (coder agent, /tmp-only).
9. Commit immutable result; results record doc; README list; push.
10. On any failure: retain the artifact under
    `results/experiments/failures/`, diagnose without inspecting
    successful outcomes, fix behavior-preservingly, void the block,
    amend the protocol, re-seal, retry once on a verified-absent
    replacement block.

## 4. Guardrails (hard rules)

- **Never instantiate a sealed/reserved block** in tests, demos, REPLs,
  or previews: 30101..30136, 60101..60136, 80101..80136, 90101..90136,
  20101..20136, 40101..40136, 70101..70136 (discarded), 130001..130036
  (void), 140001..140036, 150001..150200 (void), 160001..160036,
  170001..170200, 180001..180400, 190001..190036,
  **200001..200400 (reserved, loop-v4 train)**,
  **210001..210036 (reserved, loop-v4 eval)**. Sanctioned fixtures:
  70001..70005, 70501..70540, 70601..70620, hand-built fixtures.
- Tests monkeypatch `SEALED_EVAL_SEEDS`/`TRAIN_SEEDS` to fixtures.
- Sealed runners/tests from completed experiments are read-only history.
- `execution_revision` = HEAD at execution time; interleaved commits
  between seal and execution are normal (validators verify the seal is
  byte-identical at that revision).
- Report every claim outcome frozen; never reinterpret a failure, never
  retune after seeing results.
- The runner's preflight is fail-closed by design; a refusal is the
  ceremony working, not a bug.

## 5. Current scientific state (one paragraph)

The degraded-regime structure router beats classical and generic
baselines on all four structure families (16/16 claims). The certified
discovery head is unanimous (4/4). The route-or-discover loop is
unanimous in the certified regime (4/4). In the comparison experiment
the architecture beats a generic no-architecture model by 74 percentage
points end-to-end, and its one learned heuristic — the alarm — is the
measured ceiling: two operating points on its Pareto frontier are on
record with full decision chains. Everything is auditable at
`github.com/seanm27lol/Universa` (private).

## 6. Resume

```bash
cd /home/seanjazm27/Universa
git status --short --branch && git log --oneline -5
PYTHONPATH=src /home/seanjazm27/HOMYMOLY/.venv/bin/python -m pytest -q
sed -n '1,120p' docs/28-handoff.md   # this file
```
