# Structure-switching under seal: nine experiments

Status: write-up, 2026-09-01. This document is the synthesis of the sealed
experiment series recorded in `docs/01`–`27`. It introduces no new results
and no new numbers: every figure below is quoted from an immutable result
artifact under `results/experiments/`, and every claim outcome is reported
exactly as it was frozen — including the three that failed. Percentages are
used throughout; margins are in percentage points (pts).

---

## Abstract

Universa asks whether a model that keeps mathematical structure in
domain-native form and *switches* between structures internally can beat
models that do not. The switching layer is made concrete: every structure
compiles to a chain complex, the atomic move is transport-along-a-chain-map
followed by projection onto the target's consistent subspace, and the
residual left over is both the routing score and the discovery signal.

Nine experiments were run under a frozen sealing ceremony on untouched seed
blocks. **33 of 36 pre-registered claims were supported.** The learned
router beats the exact classical reading of the same degraded operator on
all four structure families (16/16 claims); the certified discovery head is
unanimous at 100% on coverage, certification, router-readiness, and
false-discovery refusal; and the composed route-or-discover loop beats a
deliberately architecture-free learned model by **74 pts end-to-end**.

The three failed claims all concern one component — the learned alarm that
decides *whether the library fits at all* — and they are the most
informative result in the series. Two sealed operating points now bound its
Pareto frontier: (83% acquisition, −1 seed of in-library harm) and (100%
acquisition, −14 seeds of harm). The alarm's threshold is a one-parameter
dial, and a one-sided error bound is a choice of which error to pay.

---

## 1. The question

The predecessor project, HOMYMOLY, established under its own sealed protocol
that **known structure collapses estimation to closed form**. When a boundary
operator `B1` is given exactly, least squares restricted to `ker(B1)` beats
the soft penalty, generic ridge, and a dimension-matched random subspace. The
mechanism is Pythagoras: for any estimate `A` with truth in `S = ker(B1)`,

```
||P_S A − A_*||_F^2  =  ||A − A_*||_F^2 − ||(I − P_S) A||_F^2
```

so the off-subspace component is *exactly* removable. There is nothing to
learn.

That result is a boundary, not an endpoint. It says learning is motivated
only where structure is **unknown, uncertain, intractable, or shared across
instances**. Universa studies the switching layer for exactly those regimes,
and treats HOMYMOLY's exact projection as its atomic operation rather than
its conclusion.

## 2. The architecture

**One tensor format.** Every structure compiles to a chain complex — graded
vector spaces with boundary matrices satisfying `d² = 0`, validated
fail-closed. Graphs are 1-complexes; cellular complexes are themselves;
sheaves are stalks plus restriction maps on the cells; categories enter via
the Grothendieck nerve. Four generator families implement this and share one
contract: planted maps commute with residual **exactly 0.0**, decoys are
accepted only above `1e-9`, and every instance is deterministic from an
integer seed.

**The complex lives in the computation graph, not the prompt.** Boundary
operators are instance-constant and ultra-sparse (a graph incidence matrix
has two nonzeros per column), so they enter as wiring and projection layers.
Tokens carry only cochain coefficients. The operator is never serialized.

**The atomic move.** Transport values along a chain map, project onto the
target's consistent subspace, measure what had to be removed. Two residuals
carry all the information: the *commutation residual*
`||d_D f_k − f_{k−1} d_C||_F` (does this map respect both structures?) and
the *misfit residual* `||(I − P_S) A||_F` (how much did the target have to
throw away?).

**Misfit is the discovery signal.** If no library structure fits, misfit
stays high after every projection. That is a quantitative "none of my
structures explains this" — not a hand-built novelty heuristic. The model is
never given an explicit wrong-structure flag; it sees only residuals.

**Routing is temperature-annealed, discrete at inference.** A shared
per-candidate MLP over certified features, soft gates annealed from τ = 2.0
to 0.25 during training, strictly discrete argmax with straight-through
gradients at inference, and a Switch-style load-balancing loss against
collapse. Hard routing at inference because exactness composes only under
commitment: an average of projectors is not a projector.

## 3. The ceremony

No claim in this series rests on engineering checks. Every experiment ran
under the same frozen protocol:

1. **Absence scan.** Both seed blocks are verified absent from the entire
   repository history by word-boundary scan before the protocol is written.
2. **Frozen protocol** (12 sections): conditions, features, models, arms,
   claims with pre-registered thresholds, stop rules, retention contract.
3. **Two-commit seal.** Commit A carries the design; commit B carries a seal
   JSON with the design commit, protocol and runner SHA-256s, a code manifest
   of every `src/universa/*.py`, both seed blocks, a no-preview declaration,
   and the claim objects. **Commit B is pushed before any sealed seed is
   instantiated.**
4. **Fail-closed preflight.** The runner refuses to execute unless the
   protocol hash, design commit, and code manifest all verify.
5. **One canonical run.** No outcome-dependent stopping, no rerun.
6. **Independent recomputation.** A separate validator rebuilds every claim
   from the retained raw rows alone, with no access to the runner's logic.
7. **Immutable artifact**, then the results record, then the README.

Statistics are uniform: paired differences over n = 36 seeds, one-sided lower
bounds at Bonferroni-corrected α = 0.0125 (four claims), critical value
`2.3419692993010397` (t, df 35), verified against `scipy.stats.t.ppf` at
every df from 29 to 35. Where SE = 0 mechanically (unanimous outcomes), the
lower bound equals the estimate, per protocol.

**The ceremony bit five times, on record:**

| # | what happened | when it was caught |
|---|---|---|
| 1 | Malformed `design_commit` hash in the 2-complex seal | Preflight, before any seed opened |
| 2 | Seal `statistic` keys misaligned in the discovery seal | Preflight, before any seed opened |
| 3 | Discovery crashed at seed 130006 on its *refusal* path (a borderline singular value counted in the prefix rank but not the full rank) | Canonical run; artifact retained, block voided, behavior-preserving fix, one retry on a verified-absent replacement |
| 4 | Loop-v2's first train block instantiated `150001..150010` in a test fixture | Pre-seal consistency audit, before commit A existed |
| 5 | Three claims failed | Reported frozen; never reinterpreted, never retuned |

Item 5 is the load-bearing one. A protocol that only ever confirms is not a
protocol.

---

## 4. Result group A — the router across four structure families

**Question.** Under degraded observation of the boundary operator, does a
learned router beat the *exact classical reading of the same degraded
operator* (the argmin-observed-residual oracle), and does it beat a
non-learned profile heuristic?

The oracle is a strong and honest baseline: it computes the true residual of
what it can actually see. The comparison's asymmetry is declared in every
protocol — the learned router reads the whole degradation profile, the oracle
reads one polluted column.

### Margins against the baselines (all supported)

| # | family | regime | vs oracle (3 fractions) | vs heuristic |
|---|---|---|---|---|
| 1 | graph-quotient | corruption, clean anchor | +72.9 / +91.0 / +99.3 pts | — (h4 was the anchor test) |
| 2 | graph-quotient | mask + corruption, **no anchor** | +90.3 / +93.8 / +99.3 pts | **+67.4 pts** |
| 3 | 2-complex | dual corruption, two constrained degrees | +81.9 / +87.5 / +94.4 pts | **+58.3 pts** |
| 5 | cellular sheaf | corrupted restriction entries | +88.2 / +96.5 / +99.3 pts | **+61.1 pts** |
| 6 | group nerve | corrupted multiplication table | +14.6 / +70.1 / +95.1 pts | **+4.9 pts** |

Every one-sided lower bound sits above zero: the narrowest in the series is
experiment 6's heuristic claim at **+0.9 pts**, and the widest is
experiment 1's +97.7 pts.

### Arm accuracies (at held-out fractions 0.6 / 0.7 / 0.8)

| experiment | learned router | polluted oracle | profile heuristic |
|---|---|---|---|
| 2 — graph, no anchor | 98.6 / 97.2 / 99.3% | 8.3 / 3.5 / 0.0% | 29.2 / 29.9 / 22.9% |
| 3 — 2-complex | 90.3 / 92.4 / 95.8% | 8.3 / 4.9 / 1.4% | 39.6 / 34.0 / 37.5% |
| 5 — sheaf | **100 / 100 / 100%** | 11.8 / 3.5 / 0.7% | 38.2 / 38.9 / 35.4% |
| 6 — group nerve | **100 / 100 / 100%** | 85.4 / 29.9 / 4.9% | 97.2 / 95.1 / 96.5% |

Chance is 25% in the first three rows (K = 4 candidates) and 33% in the group
row (K = 3). The oracle does not merely degrade — it goes *far below chance*,
because a corrupted operator's residual ordering is actively misleading
rather than merely noisy.

**The family variation is the interesting part.** Margins are widest where
the baselines are weakest (sheaf) and narrowest where the true structure's
degradation follows a closed form that a heuristic can read directly (group
nerve, where the heuristic itself reaches 97%). In every family the learned
router is at or near perfect. Experiment 6 also verified the declared
degradation law `sqrt(8 · round(9g))` **bit-exactly at all 3456 grid
entries**, confirming the residuals are genuine Z/6 → Z/3 nerve residuals.

**Experiment 1's fourth claim is the bounded-harm test**, and it is the one
that makes the rest meaningful: at the clean anchor, where the classical
procedure is exact, the learned router matched the oracle on all 144
replicate rows — **0.0 pts of harm, with zero variance.** The router does not
buy its degraded-regime win by giving anything up where exactness is
available.

## 5. Result group B — the certified discovery head

**Question.** With the true structure *withheld from the library*, can the
system recover a constraint from 16 transported vector observations that
covers the true kernel, certifies, would be accepted by the router, and
refuses structure-free input instead of inventing a constraint?

No learned model — this is a deterministic certified procedure: SVD
estimation of the data-supported consistent subspace, an annihilator
constraint with a certificate residual, fail-closed insufficiency via
dimensional stabilization, and a novelty gate on certified projector
distance.

| claim | pre-registered floor | result |
|---|---|---|
| Kernel coverage | 90% | **100%** (36/36) |
| Certification rate | 95% | **100%** (36/36) |
| Router-ready misfit (< 1e-9) | 95% | **100%** (36/36) |
| False-discovery refusal on null controls | 95% | **100%** (36/36) |

Unanimous on all four — the strongest form these claims can take. Every
structure-free control saturated at ambient rank and produced the certified
"trivial annihilator" refusal. Validation included a discriminating negative
control: a tampered copy of the artifact fails with 17 precise mismatches.

Unanimity at these floors does not mean the procedure works everywhere. It
means the frozen regime sits comfortably inside its working envelope.

## 6. Result group C — the loop, and where it breaks

The loop composes everything: route when the library fits, alarm and discover
when it does not, admit only certified novel structure, and know the
difference.

### Experiment 7 — the loop in the certified regime (4/4, unanimous)

Three conditions per seed: in-library (route, stay quiet), out-of-library
(alarm, discover, admit, re-route), null-control (refuse everything). All
four claims at **100% on 36/36 seeds**: every out-of-library instance was
acquired, every null control admitted nothing, every in-library instance
routed correctly with no false alarm.

This is also the experiment whose first attempt crashed at seed 130006. The
fault was diagnosed *without inspecting any successful outcome*, fixed
behavior-preservingly, the block voided, and the experiment re-sealed on a
verified-absent replacement — the failure path of the ceremony exercised end
to end. The failed artifact is retained at
`results/experiments/failures/`.

### Experiment 8 — the comparison the series was built toward (3/4)

The same loop under the **no-anchor degraded regime**, against a deliberately
architecture-free learned model and two ablations.

| condition | arch-full | generic (no architecture) | routing-only | discovery-only |
|---|---|---|---|---|
| in-library | 97.2% | 41.7% | 97.2% | 100% |
| out-of-library | 83.3% | **0%** | **0%** | 100% |
| null-control | 100% | 16.7% | 100% | 100% |
| **end-to-end** | **93.5%** | **19.4%** | 65.7% | 100% |

- **+74.1 pts end-to-end vs generic** (lower bound +64.7) — primary claim.
- **+27.8 pts vs routing-only** (lower bound +22.9) — the discovery head's
  measured contribution.
- **+55.6 pts on in-library routing vs generic** (lower bound +35.9).

The generic model fails *structurally*, not marginally: it cannot acquire a
missing structure at all (0/36), and it refuses structure-free input only
6/36 times. Both ablations confirm the decomposition — routing-only wins
where routing suffices and acquires nothing; discovery-only is perfect at the
cost of running discovery on all 108 invocations.

**The fourth claim failed.** Against always-discovering, the full
architecture's selectivity cost **exactly one seed** (160031) of in-library
accuracy: the learned alarm said no-fit, discovery ran and correctly refused
as non-novel, and the arm scored incorrect. The estimate is −1/36 = −2.8 pts
and the one-sided lower bound (−9.3) dips below the −5.0 pt margin. The claim
could not be sworn.

What that one seed bought: **61 discovery invocations instead of 108, a 44%
reduction.** That is the trade, measured.

### Experiment 9 — the alarm redesign (2/4)

Loop-v2's alarm had two error modes: 6/36 false quiets (acquisition lost) and
1/36 false alarms (in-library accuracy lost). Experiment 9 redesigned it from
that published mechanism — margin features ("do all the decoys look alike?"),
a doubled 400-seed train block, and a threshold calibrated on the train block
to maximize balanced accuracy **subject to a frozen false-quiet bound of
2%** — then re-ran the identical claim family.

The calibration selected threshold 0.8897: balanced accuracy 78.5%,
false-quiet rate 2.0% (bound satisfied), **false-alarm rate 41%.**

| condition | arch-full-v2 | generic | routing-only | discovery-only |
|---|---|---|---|---|
| in-library | 61.1% | 52.8% | 100% | 100% |
| out-of-library | **100%** | 0% | 0% | 100% |
| null-control | 100% | 77.8% | 100% | 100% |
| **end-to-end** | **87.0%** | 43.5% | 66.7% | 100% |

The frozen rule did exactly what it was designed to do. **False quiets were
eliminated completely — out-of-library acquisition went from 83.3% to
100%.** And the same threshold's 41% train false-alarm rate materialized on
eval as **14/36 false alarms on in-library instances.** On each of the 14,
the alarm said no-fit where the truth was in view, and discovery correctly
refused admission as non-novel (projector distance ~1e-15). All 14 seeds are
named in the artifact with their full decision chains.

h1 (+43.5 pts vs generic) and h2 (+20.4 pts vs routing-only) held. h3 failed
on variance — arch-v2 still leads the generic model in-library (+8.3 pts) but
the 14 false alarms push the lower bound to −15.2. h4 failed decisively:
−38.9 pts against a −5.0 margin.

### The frontier

| operating point | out-of-library acquisition | in-library harm vs always-discovering |
|---|---|---|
| loop-v2 (simplest alarm) | 83.3% | −1/36 |
| loop-v3 (calibrated alarm) | **100%** | −14/36 |

**The tradeoff moved along the alarm's Pareto frontier; it did not close.**
The threshold is a one-parameter dial and the alarm's intrinsic separation
(78.5% balanced accuracy) is not sharp enough to make both sides cheap at
once. Bounding one error one-sidedly is a *choice of which error to pay* —
and the series now has that choice measured to the seed at two points.

The architecture's advantage over the no-architecture model survives both
designs (h1 supported in both), as does the discovery head's contribution
(h2 supported in both). Everything that failed, failed at the alarm.

---

## 7. What is not claimed

Stated plainly, because the ceremony's value depends on it:

- **Family and size.** Every claim holds for the frozen generator families at
  frozen instance sizes (e.g. 8 vertices, 14 edges, 6 quotient classes,
  K = 4 for graph-quotient). Nothing is claimed for other sizes.
- **Synthetic only.** No real data anywhere in this series.
- **Not clean-regime superiority.** Where structure is given exactly, the
  classical estimator is optimal and the router merely does not lose
  (experiment 1, h4). That is the whole point of HOMYMOLY's boundary.
- **Not a representation claim.** Profile features suffice here; they are not
  shown to be the right representation.
- **The learned arms are hash-pinned, not recomputable.** Independent
  validators reproduce every non-learned bit from raw rows and every
  downstream aggregation bit-exactly, but the learned models' internal
  decisions are anchored to retained hash-pinned model states. Each artifact's
  scope text says so.
- **Unanimity is not a guarantee.** 100% on 36 seeds means the frozen regime
  is inside the envelope, not that the envelope has no edge.

## 8. What is next

The documented next design is **cost-aware calibration** (loop-v4): choose
the alarm threshold on the train block to minimize total measured error
(false-quiet rate + false-alarm rate), tie-broken toward larger balanced
accuracy — pricing both error modes symmetrically instead of bounding one.
Everything else stays frozen from loop-v3. The expectation is a mid-frontier
operating point; either h4 outcome will be reported frozen.

Seed blocks `200001..200400` (train) and `210001..210036` (eval) were scanned
and verified absent on 2026-09-01, and are reserved in `docs/28-handoff.md`.

This must be a new sealed experiment. It is never a retune of loop-v3.

---

## 9. Reproduction

```bash
PYTHONPATH=src python -m pytest -q        # 999 tests
python examples/quickstart.py             # misfit-discrimination demo
```

The library is numpy-only; torch is an optional dependency needed only by
`universa.router` (`pip install -e ".[router]"`). A plain `import universa`
stays torch-free.

**Evidence spine** — each experiment is a protocol → seal → results triple:

| # | experiment | protocol | seal | results | claims |
|---|---|---|---|---|---|
| 1 | router-v1 | `docs/01` | `docs/02` | `docs/03` | 4/4 |
| 2 | router-v2 (no anchor) | `docs/04` | `docs/05` | `docs/06` | 4/4 |
| 3 | router-2complex | `docs/07` | `docs/08` | `docs/09` | 4/4 |
| 4 | discovery | `docs/10` | `docs/11` | `docs/12` | 4/4 |
| 5 | router-sheaf | `docs/13` | `docs/14` | `docs/15` | 4/4 |
| 6 | router-group | `docs/16` | `docs/17` | `docs/18` | 4/4 |
| 7 | router-loop | `docs/19` | `docs/20` | `docs/21` | 4/4 |
| 8 | router-loop-v2 | `docs/22` | `docs/23` | `docs/24` | 3/4 |
| 9 | router-loop-v3 | `docs/25` | `docs/26` | `docs/27` | 2/4 |

Immutable artifacts: `results/experiments/*.json`. Retained failure:
`results/experiments/failures/router-loop-sealed-1.design_failure.json`.
Design record: `docs/00-design.md`. Continuation state: `docs/28-handoff.md`.

**Total: 33 of 36 pre-registered claims supported across nine sealed
experiments.** The three that failed all name the same component, and each
one names its seeds.
