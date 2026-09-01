# Sealed experiment router-loop-v2-sealed-1: results record

Status: complete, independently validated. Universa's eighth sealed
experiment: the degraded-regime route-or-discover loop **compared against
a generic learned model without the architecture and against two
ablations** — the comparison this series was built toward. The immutable
machine artifact is `results/experiments/router-loop-v2-sealed-1.json`;
the frozen (amended) protocol is
`docs/22-sealed-router-loop-v2-protocol.md`; the seal is
`docs/23-router-loop-v2-seal.json`.

## 0. Lineage

- design commit A: `cd6b61305f8e312eaf672a22823cbf5fcf314a18`
- seal commit B: `8dca13bc910eb62c3cbbf33ab6861ade47858d17` — pushed
  before any seed of either block was instantiated
- train-block errata (declared in the protocol): the first declared train
  block `150001..150200` was **voided pre-seal** — the runner's test
  suite, written under an instruction that predated the protocol's
  train-block no-preview rule, instantiated `150001..150010` in a
  training fixture before any commit. The pre-seal consistency audit
  caught it before commit A; no seal existed, no run happened, no claims
  were computed. The block was replaced by `170001..170200`, verified
  absent from history, and the tests were retargeted to the protocol's
  sanctioned fixture seeds.
- result commit: `274086e5dfcd497f4d3fcab7e72f9d0e7484f178`
- protocol SHA-256 `31493402…`, runner SHA-256 `b66c05e8…`, 19-file code
  manifest
- canonical command: `env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src python
  scripts/run_loop_v2_sealed_1.py --output
  results/experiments/router-loop-v2-sealed-1.json`

## 1. What was asked

On the graph-quotient family under no-anchor degraded observation (mask
0.25 + sign corruption, grid excluding 0.0 — the regime where routing
must be learned), per sealed eval seed and three paired conditions
(in-library, out-of-library, null-control): does the full architecture
(learned router + learned alarm + certified discovery + admission) beat
a deliberately architecture-free learned model and its own ablations,
and what does its selectivity cost against always-discovering? All
three models (router, alarm, generic) were trained inside the sealed run
on the reserved train block only.

## 2. Execution accounting

- 36/36 eval seeds eligible (160001..160036); 432 raw rows = 36 seeds x
  3 conditions x 4 arms, paired on shared observation draws.
- Models trained inside the sealed run (train block 170001..170200):
  router (seed 4242, final train accuracy 1.0), LearnedAlarm (seed 4243,
  accuracy 0.9275 on its train rows), GenericMLP (seed 4244, accuracy
  0.5525 on its train rows). Model states hash-pinned.
- No outcome-dependent stopping; single canonical attempt.

## 3. The four frozen claims (n = 36, critical value 2.3419692993010397,
Bonferroni alpha = 0.0125 one-sided)

| claim | comparison | estimate | SD | SE | lower bound | threshold | supported |
|---|---|---:|---:|---:|---:|---:|---|
| h1-loopv2-arch-vs-generic-e2e (primary) | arch − generic, e2e | +0.7407 | 0.2405 | 0.0401 | +0.6469 | 0 | **yes** |
| h2-loopv2-arch-vs-routing-only-e2e | arch − routing-only, e2e | +0.2778 | 0.1260 | 0.0210 | +0.2286 | 0 | **yes** |
| h3-loopv2-arch-vs-generic-inlibrary | arch − generic, in-library | +0.5556 | 0.5040 | 0.0840 | +0.3588 | 0 | **yes** |
| h4-loopv2-arch-vs-discovery-only-inlibrary-harm | arch − discovery-only, in-library | −0.0278 | 0.1667 | 0.0278 | −0.0928 | −0.05 | **no (reported frozen)** |

**h4's mechanism, exactly:** exactly one seed (160031) costs the claim.
There, on an in-library instance, the learned alarm said no-fit; the
discovery head ran and correctly **refused as non-novel** (the true
structure was in view), so arch-full scored incorrect while
discovery-only scored correct (map_misfit 8.65e-16). The estimate is
−1/36 and the one-sided lower bound dips below the −0.05 margin: at
n=36 the selectivity's measured price is one seed of in-library
accuracy, and the frozen bounded-harm claim cannot be sworn. This is
the honest boundary of the loop's learned alarm, and it is reported
exactly as frozen, never reinterpreted.

## 4. The comparison table (per-condition per-arm accuracy, 36 seeds)

| condition | arch-full | generic (no architecture) | routing-only | discovery-only |
|---|---:|---:|---:|---:|
| in-library | 0.972 | 0.417 | 0.972 | 1.000 |
| out-of-library | 0.833 | 0.000 | 0.000 | 1.000 |
| null-control | 1.000 | 0.167 | 1.000 | 1.000 |
| end-to-end mean | 0.935 | 0.194 | 0.657 | 1.000 |

The no-architecture model fails decisively everywhere: it cannot acquire
a missing structure at all (0/36 out-of-library), routes poorly even
when the answer is in the library (15/36), and refuses structure-free
input only 6/36. The ablations isolate the architecture's pieces:
routing-only wins where routing suffices but cannot acquire (0/36
out-of-library), and discovery-only is accuracy-perfect at 108
discovery invocations. The full architecture acquires 30/36
out-of-library with **61** invocations (1 in-library + 30 null-alarm +
30 out-of-library alarms) — a 44% reduction versus always-discovering,
traded against one seed of in-library accuracy (h4).

## 5. Independent validation (post-run, pre-commit)

81/81 checks passed. All 432 correctness bits recomputed with an
independent implementation of the frozen semantics (0 discrepancies);
pairing verified per (seed, condition) with the pinned observation
schedule; all four claim statistics bit-identical (max diff 0.0);
critical table identical to scipy; the h4 harm seed identified and its
mechanism verified; descriptive accuracies/invocations/wall times
recomputed exactly; hashes, commits, environment, and seed accounting
verified against the seal and git history. The learned models'
internal decisions are hash-pinned, not recomputable — the artifact's
scope text says exactly that.

## 6. What this adds to the series, and its boundary

This is the series' first comparison-grade result: the architecture
beats a generic learned model by +0.74 end-to-end and +0.56 on
in-library routing, the discovery head's measured contribution is
+0.28 end-to-end, and the honest costs are a 0.833 (not perfect)
learned-alarm acquisition rate and one seed of selectivity price
against always-discovering. Declared scope: the graph-quotient family
at frozen sizes under the no-anchor degraded regime; the generic arm
is deliberately architecture-free by construction (its out-of-library
failure is structural, not learned); and h4's failure is a result, not
a defect — the selectivity-accuracy tradeoff is the experiment's most
instructive number.
