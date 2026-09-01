# Sealed experiment router-loop-v4-sealed-1: results record

Status: complete, independently validated. Universa's tenth sealed
experiment: the alarm's threshold chosen by **cost-aware calibration** —
pricing both error modes symmetrically instead of bounding one. The
immutable machine artifact is
`results/experiments/router-loop-v4-sealed-1.json`; the frozen (amended)
protocol is `docs/30-sealed-router-loop-v4-protocol.md`; the seal is
`docs/31-router-loop-v4-seal.json`. The failed first attempt is retained at
`results/experiments/failures/router-loop-v4-sealed-1.design_failure.json`.

## 0. Lineage

- design commit A: `63ebafc2edba70bed831f7ef5c5fdb9c423c31db` (first seal,
  blocks `200001..200400` / `210001..210036`, seal commit B
  `e6869ee3725afee2c2c0ff18f2dc2d9e5cb145eb`, pushed before instantiation)
- **FIRST ATTEMPT FAILED** during train-block construction at seed 200058
  with status `design_failure`:
  `ValueError: decoy admits every transported cycle (B1_decoy f1 vanishes
  on the source cycle space, e.g. a vertex-relabeled complete quotient)`.
  That is the `universa.budgets` discriminability guard of
  `docs/00-design.md` §6 refusing to construct an instance whose decoy
  shares the true target's kernel. The original §5 declared any train-seed
  build exception a whole-run failure, so the runner stopped exactly as
  frozen — **the runner was correct and the frozen rule was too strict.**
  The attempt produced no science: 0 raw rows, no model trained, no arm
  executed, no claim computed. Both first blocks are void for claims.
- amended commit A′: `64f38eeca94a51398029cfbbc021f0c5c7b4e9d6` (protocol
  errata 1 + the train-side build exclusion + replacement blocks
  `220001..220400` / `230001..230036`, verified absent with zero hits)
- amended seal commit B′: `b1c54b6f4202fb10320051f5148bafac3baf68c7` —
  pushed before any replacement seed was instantiated
- execution at `b1c54b6`; result commit: `31cb434`
- protocol SHA-256 `93ee35e0…`, runner SHA-256 `900b8f29…`, 19-file code
  manifest **byte-identical to the first seal** (the amendment changed the
  protocol, the runner, and the tests, and no `src/universa` file)
- canonical command: `env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src python
  scripts/run_loop_v4_sealed_1.py --output
  results/experiments/router-loop-v4-sealed-1.json`

## 1. What was asked

Loop-v3 measured what a one-sided error bound costs. Its rule maximized
balanced accuracy subject to `false_quiet_rate <= 0.02`; the bound was
binding, selecting threshold `0.8897` at a **0.41 false-alarm rate**, which
materialized as 14/36 in-library false alarms. This experiment changes
exactly one component — the rule that selects the threshold — to
`universa.loop_v2.calibrate_threshold_cost_aware` at frozen unit costs
`false_quiet_cost = false_alarm_cost = 1.0`: minimize
`FQ_cost * FQ + FA_cost * FA`, no feasibility bound, no fallback branch.
Everything else (features, architecture, models, seeds, arms, conditions,
gates, claim family) is unchanged.

**The equal-cost identity was declared before any data** (§1 and §5 of the
protocol, and the module docstring): since `FQ = 1 - TNR` and
`FA = 1 - TPR`, at equal unit costs

```text
FQ + FA = 2 - 2 * balanced_accuracy
```

exactly, so this rule **is** unconstrained balanced-accuracy maximization
and the rule's balanced-accuracy tiebreak can never fire. The experiment's
alarm-side change is precisely the removal of loop-v3's binding constraint,
and it is described that way throughout rather than dressed as a new
objective.

## 2. Execution accounting

- 36/36 eval seeds eligible (`230001..230036`); 432 raw rows = 36 seeds x 3
  conditions x 4 arms, paired on shared observation draws; every row
  double-executed bit-identically.
- Models trained inside the sealed run on the replacement train block
  `220001..220400`: router (seed 4242, final train accuracy 1.000),
  LearnedAlarmV2 (seed 4243, train accuracy 0.940), GenericMLP (seed 4244,
  train accuracy 0.546). Model states hash-pinned.
- **Errata-1 exclusions: zero.** All 400 declared train seeds built an
  instance, well inside the `MAX_TRAIN_EXCLUDED = 20` ceiling. The
  exclusion path was insurance; its value is that the run no longer depends
  on the replacement block being lucky, not that it was needed here.
- Calibration (train block only, frozen rule): threshold `0.5306756…`,
  balanced accuracy `0.940`, false-quiet rate `0.0775`, false-alarm rate
  `0.0425`, total cost `0.12`, 801 candidates.
- No outcome-dependent stopping; the retry after the errata was executed
  exactly once on the replacement blocks.

## 3. The four frozen claims (n = 36, critical value 2.3419692993010397,
Bonferroni alpha = 0.0125 one-sided)

| claim | comparison | estimate | lower bound | threshold | supported |
|---|---|---:|---:|---:|---|
| h1-loopv4-arch-vs-generic-e2e (primary) | arch − generic, e2e | +49.07 pts | +41.79 | 0 | **yes** |
| h2-loopv4-arch-vs-routing-only-e2e | arch − routing-only, e2e | +32.41 pts | +30.24 | 0 | **yes** |
| h3-loopv4-arch-vs-generic-inlibrary | arch − generic, in-library | +38.89 pts | +17.45 | 0 | **yes** |
| h4-loopv4-arch-vs-discovery-only-inlibrary-harm | arch − discovery-only, in-library | −2.78 pts | −9.28 | −0.05 | **no (reported frozen)** |

**h3 is the claim loop-v3 failed** (+8.33 pts, lower bound −15.23, sunk by
the variance of its 14 false alarms). Here it clears its bound by a wide
margin.

**h4's mechanism, exactly.** One seed — 230011 — costs the claim, the same
shape as loop-v2's 160031. On that in-library instance the alarm said
no-fit; certified discovery ran and correctly **refused as non-novel** (the
true structure was in view; min projector distance 1.361e-15 against a
1e-6 novelty tolerance), so arch-full scored incorrect while discovery-only
scored correct. The estimate is exactly −1/36 and the one-sided lower bound
(−9.28 pts) dips below the −5.0 pt margin. **Reported exactly as frozen,
never reinterpreted.**

## 4. The comparison table (per-condition per-arm accuracy, 36 seeds)

| condition | arch-full-v4 | generic | routing-only | discovery-only |
|---|---:|---:|---:|---:|
| in-library | 97.22% | 58.33% | 97.22% | 100% |
| out-of-library | **97.22%** | 0% | 0% | 100% |
| null-control | 100% | 88.89% | 100% | 100% |
| **end-to-end** | **98.15%** | 49.07% | 65.74% | 100% |

Discovery invocations: arch-full **71/108** (1 in-library + 35
out-of-library + 35 null-control) against discovery-only's 108. The
architecture's two alarm errors on this block are symmetric and rare: one
false alarm (in-library, seed 230011) and one false quiet (out-of-library,
the single acquisition miss).

## 5. The frontier, now three sealed points

| experiment | alarm rule | threshold | train balanced acc. | out-of-library acquisition | in-library | harm vs always-discovering | e2e |
|---|---|---:|---:|---:|---:|---:|---:|
| loop-v2 (docs/24) | v1 alarm, frozen 0.5 | 0.5 | — | 83.3% | 97.2% | −1/36 | 93.5% |
| loop-v3 (docs/27) | v2 alarm, bounded FQ ≤ 0.02 | 0.8897 | 0.785 | 100% | 61.1% | −14/36 | 87.0% |
| **loop-v4 (this)** | v2 alarm, cost-aware 1:1 | 0.5307 | 0.940 | 97.2% | 97.2% | −1/36 | **98.15%** |

**Loop-v4 dominates loop-v2**: identical in-library accuracy and identical
−1/36 harm, with **+13.9 pts of acquisition** and a 4.65-pt higher
end-to-end mean. Against loop-v3 it trades 2.8 pts of acquisition for 36.1
pts of in-library accuracy and reduces the harm from 14 seeds to 1. The
handoff predicted a mid-frontier operating point; the measured point is
mid-frontier on the acquisition axis and at the good end on the harm axis.

**The honest confound, stated plainly.** The v3→v4 comparison changes two
things at once, and it cannot do otherwise: the calibration rule AND the
train block, because a consumed block is never reused. Loop-v3's alarm
reached 0.785 balanced accuracy on `180001..180400`; loop-v4's reached
0.940 on `220001..220400`. Part of loop-v4's advantage is therefore a
better-separated alarm, not only a better threshold rule — the alarm's
intrinsic separation evidently varies materially across train blocks. What
the rule change alone demonstrably did is move the threshold off the
constraint (0.8897 → 0.5307) and cut the train false-alarm rate from 0.41
to 0.0425. Attributing the whole margin to the rule would overclaim, and
this record does not.

## 6. Independent validation (post-run, pre-commit)

67/67 checks passed, recomputed from the retained raw rows alone by a
second implementation of the frozen semantics written from the protocol
rather than from the runner's audit code. All 432 correctness bits
reproduced with 0 discrepancies; row counts, the 4x3xn invariant, per-arm
and per-condition balance, and ArmOutcome cross-field coherence verified;
all four claim estimates, SDs, SEs, critical values, lower bounds, and
decisions recomputed to ≤1e-12; the critical value checked against
`scipy.stats.t.ppf`; the eight-field calibration record verified for shape,
frozen costs, internal consistency of `total_cost`, and the declared
equal-cost identity; the h4 harm seed identified and its decision chain
confirmed; hashes, commits, seal design-commit ancestry, environment, and
seed accounting verified against git history. The learned models' internal
decisions are hash-pinned, not recomputable — the artifact's scope text
says exactly that.

## 7. What this adds to the series, and its boundary

The alarm arc now has three sealed operating points and, for the first
time, one that is not a pure trade: loop-v4 dominates loop-v2 outright and
beats loop-v3 on every axis except 2.8 pts of acquisition. Symmetric
pricing was the right instrument for a symmetric cost structure, and the
one-sided bound of loop-v3 is now measurably the wrong one for this task.

The alarm remains the ceiling. h4 has now failed three times running — at
−1/36, −14/36, and −1/36 — and the mechanism has been identical every time:
the alarm fires on an in-library instance, certified discovery correctly
refuses as non-novel, and the route credit is forfeited. A bounded-harm
claim against always-discovering may simply be unwinnable while the alarm
is a learned component with any error rate at all, since always-discovering
is perfect in-library by construction. That is a statement about the claim's
design, not a reinterpretation of its outcome.

Declared scope: the graph-quotient family at frozen sizes under the
no-anchor degraded regime; the generic arm is architecture-free by
construction; the v3→v4 comparison carries the train-block confound of §5;
and every failed claim is reported frozen and never reinterpreted.
