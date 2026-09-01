# Sealed experiment router-loop-v3-sealed-1: results record

Status: complete, independently validated. Universa's ninth sealed
experiment: the alarm redesign — margin features, a doubled declared
train block, and a train-block-calibrated threshold, built from the
published mechanism of loop-v2's h4 failure and frozen before any eval
outcome was seen. The immutable machine artifact is
`results/experiments/router-loop-v3-sealed-1.json`; the frozen protocol
is `docs/25-sealed-router-loop-v3-protocol.md`; the seal is
`docs/26-router-loop-v3-seal.json`.

## 0. Lineage

- design commit A: `86e8861892e4b4b8586906a1572716fff892c1ec`
- seal commit B: `28892b21c1d822223c242ed1718646f1aa1f32a9` — pushed
  before any seed of either block was instantiated
- train block: `180001..180400` (400 seeds — the doubled declared data
  budget, verified absent from history)
- eval block: `190001..190036` (verified absent from history)
- result commit: `ba8becd8f93bc76969bed4d180620348c63509a5`
- protocol SHA-256 `5151bbb9…`, runner SHA-256 `3d7bb5f2…`, 19-file code
  manifest
- canonical command: `env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src python
  scripts/run_loop_v3_sealed_1.py --output
  results/experiments/router-loop-v3-sealed-1.json`

## 1. What was asked

Loop-v2's h4 failed by one seed: the learned alarm, the loop's only
uncertified component, was its accuracy ceiling, with two error modes —
false quiets (6/36, acquisition lost) and false alarms (1/36,
in-library accuracy lost). This experiment redesigns the alarm from
that published mechanism — margin features ("do all decoys look
alike?"), 800 training rows, and a threshold calibrated on the train
block to maximize balanced accuracy subject to a frozen false-quiet
bound of 0.02 — and re-runs the identical claim family so the
before/after is directly measurable. The certified machinery
(discovery, admission, gates, eligibility) is untouched.

## 2. Execution accounting

- 36/36 eval seeds eligible (190001..190036); 432 raw rows = 36 seeds
  x 3 conditions x 4 arms, paired on shared observation draws.
- Models trained inside the sealed run (train block 180001..180400):
  router (seed 4242, train accuracy 0.995), LearnedAlarmV2 (seed 4243,
  accuracy 0.9225), GenericMLP (seed 4244, accuracy 0.5175).
- Calibration (train block only, frozen rule): threshold 0.8897,
  balanced accuracy 0.785, false-quiet rate 0.020 (bound satisfied),
  false-alarm rate 0.41, 802 candidates.
- No outcome-dependent stopping; single canonical attempt.

## 3. The four frozen claims (n = 36, critical value 2.3419692993010397,
Bonferroni alpha = 0.0125 one-sided)

| claim | comparison | estimate | SD | SE | lower bound | threshold | supported |
|---|---|---:|---:|---:|---:|---:|---|
| h1-loopv3-arch-vs-generic-e2e (primary) | arch-v2 − generic, e2e | +0.4352 | 0.2366 | 0.0394 | +0.3428 | 0 | **yes** |
| h2-loopv3-arch-vs-routing-only-e2e | arch-v2 − routing-only, e2e | +0.2037 | 0.1648 | 0.0275 | +0.1394 | 0 | **yes** |
| h3-loopv3-arch-vs-generic-inlibrary | arch-v2 − generic, in-library | +0.0833 | 0.6036 | 0.1006 | −0.1523 | 0 | **no (reported frozen)** |
| h4-loopv3-arch-vs-discovery-only-inlibrary-harm | arch-v2 − discovery-only, in-library | −0.3889 | 0.4944 | 0.0824 | −0.5819 | −0.05 | **no (reported frozen)** |

## 4. The mechanism, exactly

The frozen calibration rule did precisely what it was designed to do —
and the design's mistake is now measurable. Bounding false quiets at
0.02 pushed the threshold high enough to **eliminate them entirely**:
out-of-library acquisition went from 0.833 (loop-v2) to **1.000**
(36/36, every alarm fired correctly). But the same threshold produces
a 0.41 false-alarm rate on the train block, which materialized on eval
as **14/36 false alarms on in-library instances** — the alarm said
no-fit where the truth was in view, and on each of the 14 the
discovery head correctly refused admission as non-novel (support
subspace duplicates a library kernel, projector distance ~1e-15), so
arch-full scored incorrect while discovery-only scored correct.

The 14 harm seeds (190002, 190012, 190013, 190015, 190018, 190019,
190022, 190024, 190027, 190029, 190030, 190031, 190032, 190034) are
named in the artifact; each carries the full decision chain in its
detail string. The harm estimate is exactly −14/36 and the one-sided
lower bound is −0.582 against a −0.05 margin.

**The tradeoff moved along the alarm's Pareto frontier; it did not
close.** Loop-v2 sat at (0.833 acquisition, −1/36 harm); loop-v3 sits
at (1.000 acquisition, −14/36 harm). The one-sided false-quiet bound
bought the acquisition side completely and paid for it on the harm
side, because the alarm's intrinsic separation (balanced accuracy
0.785 on the train block) is not good enough to make both sides cheap
at once. h3's failure is the same phenomenon seen from the other side:
arch-v2 still beats the generic model on the in-library estimate
(+0.083), but the variance from the 14 false alarms pushes the lower
bound below zero.

## 5. The comparison table (per-condition per-arm accuracy, 36 seeds)

| condition | arch-full-v2 | generic (no architecture) | routing-only | discovery-only |
|---|---:|---:|---:|---:|
| in-library | 0.611 | 0.528 | 1.000 | 1.000 |
| out-of-library | 1.000 | 0.000 | 0.000 | 1.000 |
| null-control | 1.000 | 0.778 | 1.000 | 1.000 |
| end-to-end mean | 0.870 | 0.435 | 0.667 | 1.000 |

Notable in passing: the generic model itself improved with the doubled
train block (0.435 e2e vs 0.194 in loop-v2) — more data helps generic
models too, and it still fails acquisition structurally (0/36).

## 6. Independent validation (post-run, pre-commit)

102/102 checks passed. All 432 correctness bits recomputed with an
independent implementation of the frozen semantics (0 discrepancies);
pairing verified per (seed, condition) with the pinned observation
schedule recomputed from stdlib SHA-256; all four claim statistics to
≤1e-12; critical table identical to scipy; the 14 harm seeds
identified and each one's mechanism confirmed from its detail string;
the calibration record, margin feature names, v1-vs-v3 baseline block,
and duplicated descriptive calibration verified; hashes, commits,
environment, and seed accounting verified against the seal and git
history. The learned models' internal decisions are hash-pinned, not
recomputable — the artifact's scope text says exactly that.

## 7. What this adds to the series, and its boundary

The series now has the full three-point arc of the learned alarm:
loop-v2's simplest alarm (0.972 in-library, 0.833 acquisition,
−1/36 harm), loop-v3's calibrated alarm (0.611 in-library, 1.000
acquisition, −14/36 harm), and the mechanism connecting them (the
threshold is a one-parameter Pareto dial, and a one-sided bound is a
choice of which error to pay). The architecture's advantage over the
no-architecture model persists across both designs (h1 supported in
both), as does the discovery head's contribution (h2 supported in
both). The alarm remains the ceiling — now demonstrably twice, with
two different operating points, each measured to the seed. A
cost-aware calibration (pricing both error modes by their measured
task costs rather than bounding one) is the documented next design —
as a new sealed experiment, never a retune of this one. Declared
scope: the graph-quotient family at frozen sizes under the no-anchor
degraded regime; every failed claim is reported frozen and never
reinterpreted.
