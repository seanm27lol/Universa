# Universa continuation handoff — 2026-09-01

Status time: 2026-09-01. Repository: `seanm27lol/Universa` (private).
Branch: `main`. Worktree clean.
Suite: **1100 passed** (~32s with
`PYTHONPATH=src /home/seanjazm27/HOMYMOLY/.venv/bin/python -m pytest -q`).
Read this before running or changing anything.

## 0. What exists

Ten sealed experiments, all under the two-commit ceremony, all
independently validated, all artifacts committed. Evidence spine
(protocol -> seal -> results): docs/01-03 (router-v1), 04-06 (router-v2),
07-09 (router-2complex), 10-12 (discovery), 13-15 (router-sheaf),
16-18 (router-group), 19-21 (router-loop), 22-24 (router-loop-v2),
25-27 (router-loop-v3), 30-32 (router-loop-v4). Series synthesis:
docs/29-writeup.md (36/40 claims).

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

| 10 | router-loop-v4 (cost-aware calibration) | 3/4 | h1 +49 pts vs generic, h2 +32 vs routing-only, **h3 +39 supported where v3's failed**; **h4 NOT supported** at exactly -1/36 (seed 230011), same mechanism as v2's. 98.2% e2e, 97.2% acquisition, 97.2% in-library, 71/108 discovery calls. Calibration picked threshold 0.5307 at 94.0% balanced accuracy, FQ 7.75%, FA 4.25%. **First attempt failed** at train seed 200058 (generator discriminability guard); errata 1 made a train-side non-instance a recorded exclusion, both blocks voided, re-sealed, completed on replacements |

The arc: win (v2), boundary (h4), redesign (v3), and a **measured Pareto
frontier** the fourth point finally pushed rather than merely traversed.
loop-v2 sat at (83% acquisition, -1 seed harm), loop-v3 at (100%, -14),
loop-v4 at (97%, -1) — **dominating loop-v2**: same in-library accuracy,
same harm, +14 pts acquisition. A one-sided bound is a choice of which
error to pay; symmetric pricing was the right instrument for a symmetric
cost structure.

**h4 has now failed three times with one mechanism** (-1/36, -14/36,
-1/36): the alarm fires in-library, certified discovery correctly refuses
as non-novel, the route credit is forfeited. Always-discovering is perfect
in-library by construction, so the claim may be unwinnable as posed.

## 2. The open threads (loop-v4 is DONE — docs/30-32)

Cost-aware calibration is sealed and reported. What it left open, in
rough order of value:

1. **h4 may be unwinnable as posed.** Three failures, one mechanism.
   Always-discovering is perfect in-library by construction, so any
   learned alarm with a nonzero false-alarm rate loses to it there. A
   bounded-harm claim that prices the INVOCATION SAVINGS against the
   accuracy cost (loop-v4: 71 discovery calls vs 108) would test the real
   engineering tradeoff instead of a comparison the architecture cannot
   win. This needs a new claim family, so a new experiment.
2. **The train-block confound now limits every alarm-arc comparison.**
   Each experiment must draw a fresh block, and the alarm's intrinsic
   separation moved from 78.5% (loop-v3's block) to 94.0% (loop-v4's).
   Isolating a rule change from a block change needs several blocks per
   design — a materially bigger experiment.
3. **The loop has no sealed experiment outside graph quotients.** The
   four-family sweep that made the router result convincing
   (2-complexes, sheaves, group nerves) has never been run for the
   route-or-discover loop.

- **Blocks: none reserved.** loop-v4 consumed `220001..220400` /
  `230001..230036` and voided `200001..200400` / `210001..210036`. Pick
  fresh ones and run the §3 scan before writing any protocol.
- The write-up `docs/29-writeup.md` covers all ten experiments (36/40
  claims) and quotes the artifacts only — no new numbers. Update it
  alongside any new result.

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
  200001..200400 (VOID, loop-v4 first train block),
  210001..210036 (VOID, loop-v4 first eval block),
  220001..220400 (consumed, loop-v4 train),
  230001..230036 (consumed, loop-v4 eval),
  500001..500600 (BURNED — the loop-v4 errata diagnostic sample; never
  declare it as a train or eval block). Sanctioned fixtures:
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
unanimous in the certified regime (4/4). In the comparison experiments
the architecture beats a generic no-architecture model by 74 and 49
percentage points end-to-end, and its one learned heuristic — the alarm —
is the measured ceiling: three operating points on its Pareto frontier
are on record with full decision chains, and the third (cost-aware,
symmetric unit costs) dominates the first at 98.2% end-to-end. h4 has
failed at all three points with one identical mechanism, which is now the
most interesting open question in the series. 36/40 claims supported.
Everything is auditable at `github.com/seanm27lol/Universa` (private).

## 6. Resume

```bash
cd /home/seanjazm27/Universa
git status --short --branch && git log --oneline -5
PYTHONPATH=src /home/seanjazm27/HOMYMOLY/.venv/bin/python -m pytest -q
sed -n '1,120p' docs/28-handoff.md   # this file
```
