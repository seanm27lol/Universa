# Frozen-design protocol: sealed evaluation of the degraded-regime route-or-discover loop with a cost-aware calibrated alarm threshold — pricing both error modes instead of bounding one — on the graph-quotient family

Status: **complete and freeze-ready, but not yet sealed or executable**. This
document becomes immutable only when the runner, tests, and design-seal record
described in §12 have been committed. Until then, the declared eval seeds must
not be instantiated. After sealing, any substantive change creates a new
experiment with a new untouched seed block; it must not be called
router-loop-v4-sealed-1.

Protocol date: 2026-09-01.
Experiment id: `universa-loop-v4-sealed-1`.

## 0. Errata — first attempt, and the amendment it forced

**Errata 1 (2026-09-01): the first canonical attempt failed, both first
blocks are void, and the train-side build rule is amended.**

The first sealed attempt — design commit `63ebafc`, seal commit `e6869ee`,
train block `200001..200400`, eval block `210001..210036` — ran and stopped
with status `design_failure` during train-block construction:

```text
train block construction failed at seed 200058: ValueError: decoy admits
every transported cycle (B1_decoy f1 vanishes on the source cycle space,
e.g. a vertex-relabeled complete quotient); choose parameters with a
non-complete quotient so decoys have different kernels
```

That is the `universa.budgets` family's own discriminability guard, the
design finding recorded in `docs/00-design.md` §6: a decoy sharing the true
target's kernel is indiscriminable by any transported quantity, so the
generator refuses to construct the instance. The original §5 declared any
train-seed build exception a whole-run `design_failure` and never an
exclusion, so the runner stopped exactly as frozen. **The runner was
correct; the frozen rule was too strict.**

The failed attempt produced no science and is retained immutably at
`results/experiments/failures/router-loop-v4-sealed-1.design_failure.json`:
0 raw rows, no model trained, no arm executed, no claim computed. All 36
eval seeds passed the eligibility gate that runs before training — a
deterministic build-and-audit property that carries no information about
any claim — and no eval outcome was ever computed or seen. The canonical
output path was never occupied.

**Both first blocks are void for claims** and are never reused:
`200001..200400` (train, consumed to seed 200058) and `210001..210036`
(eval, instantiated by the eligibility gate). The replacement blocks are
declared in §3 and were verified absent from the working tree and all Git
history before this amendment.

**The amendment, declared before any replacement seed was instantiated.** A
train seed whose INSTANCE fails to build is not a training example — it is
not an instance of the family at all. Such a seed is now a recorded,
counted EXCLUSION rather than a whole-run failure, mirroring exactly how
the eval side already records an ineligible seed (§10). The exclusion is
narrow and the rest of the discipline is unchanged:

- only a failure of `make_budget_instance` is an exclusion;
- a feature-construction failure, a shape violation, or a non-finite
  feature remains a whole-run `design_failure` — a seed that builds but
  whose features fail signals a pipeline fault, not a non-instance;
- the eval side is untouched: an eval-block build exception remains a
  whole-run `design_failure` (§9, §10);
- more than `MAX_TRAIN_EXCLUDED = 20` non-instances in the declared train
  block, or a block where nothing builds, is itself a whole-run
  `design_failure` — the fail-closed guard over the exclusion;
- the excluded seeds, with their reasons, and the built count are recorded
  in the training provenance and reported (§5, §11).

**Why this is not outcome-tuning.** The amendment cannot move any claim in
any direction: it changes only whether a run reaches the point of computing
claims at all, never what any arm does, what any correctness bit is, or how
any statistic is formed. It was chosen from the failure's published
mechanism alone — no eval quantity existed to tune against, since the
attempt produced none. The measured rate of the collision is one
non-instance in roughly a thousand seeds built (the first attempt's 400
train seeds plus 36 eval seeds, plus a 600-seed diagnostic sample on the
scratch range `500001..500600`, which is burned for measurement and must
never be declared as a train or eval block).

Per §9 the amended experiment is retried exactly once, on the
verified-absent replacement blocks of §3, under a re-sealed design.

## 1. Purpose and precise claim scope

This experiment is an untouched-seed evaluation of the Universa
degraded-regime route-or-discover loop with a **cost-aware calibrated alarm
threshold** (`universa.loop_v2`: `arm_arch_full_v2`, `arm_routing_only`,
`arm_discovery_only`, `arm_generic`, `LearnedAlarmV2`, `GenericMLP`,
`alarm_features_v2`, `train_alarm_v2`, `calibrate_threshold_cost_aware`,
`alarm_decision_v2`, `train_generic`, the arch and generic feature builders,
`router_gates`, `router_argmax`, `generic_decision`, `operating_grid_point`,
`ArmOutcome`) on the graph-quotient switch-instance family of
`universa.generators` / `universa.budgets` **only**. It is the third member
of the alarm arc — after the sealed loop v2
(`docs/22-sealed-router-loop-v2-protocol.md`, results `docs/24`) and the
sealed loop v3 (`docs/25-sealed-router-loop-v3-protocol.md`, results
`docs/27`): the SAME task family, the SAME no-anchor degraded regime, the
SAME paired views, the SAME three conditions, the SAME routing-only /
discovery-only / generic arms, the SAME margin features and alarm
architecture, and a claim family intentionally IDENTICAL in shape. **The one
changed component is the calibration rule that selects the alarm's decision
threshold.** Everything else, and in particular **the certified machinery —
the discovery head, admission, the gates, and eligibility — is untouched.**

**Why the calibration rule is changed — the published mechanism, declared as
the whole basis of the design.** Loop-v3's sealed result (`docs/27`, §3–§4)
measured what a one-sided error bound costs. Its frozen rule
(`universa.loop_v2.calibrate_threshold` with `max_false_quiet_rate = 0.02`)
maximized balanced accuracy SUBJECT TO a bound on the false-quiet rate. The
bound was binding: the selected threshold `0.8897` sat exactly at
`false_quiet_rate = 0.02` with `false_alarm_rate = 0.41` and balanced
accuracy `0.785`. On the eval block that eliminated false quiets entirely —
out-of-library acquisition rose from loop-v2's `0.833` to `1.000` (36/36) —
and produced **14/36 false alarms on in-library instances**, each one a row
where the alarm fired with the truth in view, the discovery head correctly
refused admission as non-novel, and the route credit was forfeited. Loop-v3
reported h3 and h4 as failed, frozen.

The two sealed results are therefore two measured operating points of one
one-parameter dial:

| operating point | out-of-library acquisition | in-library harm vs always-discovering |
|---|---:|---:|
| loop-v2 (`LearnedAlarm`, frozen threshold `0.5`) | 0.833 | −1/36 |
| loop-v3 (`LearnedAlarmV2`, bounded-false-quiet threshold) | 1.000 | −14/36 |

**The tradeoff moved along the alarm's Pareto frontier; it did not close.**
The design response, declared here, is to stop bounding one error mode and
price BOTH: the threshold is selected on the train block by the frozen rule
`universa.loop_v2.calibrate_threshold_cost_aware` at equal unit costs (§5).

**The equal-cost identity, declared openly before any data.** The
cost-aware objective at equal unit costs is not a new quantity. Writing
`TPR` for the fraction of fit rows scored at or above the threshold and
`TNR` for the fraction of no-fit rows scored below it, the rule's two error
rates are `false_quiet_rate = 1 − TNR` and `false_alarm_rate = 1 − TPR`,
while `balanced_accuracy = (TPR + TNR) / 2`. Hence, exactly,

```text
false_quiet_rate + false_alarm_rate = 2 − 2 * balanced_accuracy
```

so at equal unit costs this rule is **identical to unconstrained
balanced-accuracy maximization**, and the rule's balanced-accuracy tiebreak
can never fire (equal total cost implies equal balanced accuracy); the
larger-threshold tiebreak does all the tie-breaking work. This is stated
here, before any data, for three reasons. (a) It is true and would otherwise
be discovered later and look like a defect. (b) It makes the frozen rule
analyzable: the experiment's alarm-side change is exactly "drop loop-v3's
binding constraint", and loop-v3's own record shows the constraint was
binding, so the selected threshold is expected to differ. (c) It is the
correct pricing for THIS experiment, on the design's own terms: a false
quiet costs one out-of-library seed and a false alarm costs one in-library
seed, the two conditions enter the end-to-end statistic with equal weight
(§7), and no measured quantity in the sealed record prices one above the
other. The costs are rule PARAMETERS rather than constants so that an
asymmetric pricing is expressible without a second rule; this experiment
freezes them at `1.0` and `1.0` and uses no other values.

**Designed from the published mechanism, never tuned against any eval
outcome.** The change's entire justification is the PUBLISHED loop-v2 and
loop-v3 records (docs/24, docs/27) and the frozen module semantics of
`universa.loop_v2`. No seed of this experiment's eval block
(`230001..230036`) or train block (`220001..220400`) has ever been
instantiated for any purpose (§3); the design was fixed before any eval data
exists; no eval-side quantity — accuracy, margin, invocation count,
threshold behavior — played any role in choosing the rule, the costs, or the
block sizes. The claim family is intentionally IDENTICAL in shape to
loop-v2's and loop-v3's — same four paired-difference statistics, same
thresholds, same directions, same alpha — precisely so that the three-point
before/after is directly measurable. **Any claim may pass or fail; every
outcome is reported as frozen, never reinterpreted**, exactly as loop-v2 and
loop-v3 reported theirs.

**The expected outcome is a mid-frontier point, and that expectation binds
nothing.** Dropping a binding constraint that forced the threshold high is
expected to lower it, trading some acquisition back for some in-library
accuracy. Nothing in this design assumes that happens, assumes the total
error falls on eval, or assumes h3 or h4 passes. A null or negative
before/after is a reportable frozen outcome. In particular, the rule
minimizes total error **on the train block**; its eval-block behavior is
measured, not guaranteed.

**The published tradeoff is the documented baseline.** The descriptive block
of the result (§11) carries the three-point frontier comparison against the
published numbers of docs/24 and docs/27: loop-v2's arch arm (in-library
`0.972`, out-of-library `0.833`, null-control `1.000`; 61/108 discovery
invocations; −1/36 bounded-harm estimate on seed 160031) and loop-v3's
(in-library `0.611`, out-of-library `1.000`, null-control `1.000`; 86/108
discovery invocations — 14 in-library + 36 out-of-library + 36 null-control;
−14/36 bounded-harm estimate on the 14 named seeds; calibration record
threshold `0.889712393283844`, balanced accuracy `0.785`, false-quiet rate
`0.02`, false-alarm rate `0.41`, 802 candidates). These are descriptive
context, never claim inputs.

**The regime — unchanged from loop-v2 and loop-v3.** Every candidate
boundary is observed through an `universa.partial.ObservationModel` with
BOTH `mask_fraction = 0.25` (edge columns removed;
`universa.router_v2.DEFAULT_MASK_FRACTION`) AND sign corruption swept over
the profile grid `0.2, 0.3, ..., 0.9` (8 points,
`universa.router_v2.NO_ANCHOR_GRID`) that EXCLUDES `0.0`: there is no clean
column anywhere, so the v1 certified alarm cannot separate and the fit/no-fit
decision is a learned component. **The degradation applies to the observed
structures — the boundary operators the models see — never to the
transported data** (declared caveat 4 below): the vector observations
`y = f1 a` are exact under every regime, so the certified discovery head of
`universa.discovery` works UNCHANGED in this regime — discovery always runs
on the exact transported observations at the frozen gates (certification
`1e-10` = `universa.operators.CERT_TOL`, novelty `1e-6` =
`universa.discovery.DEFAULT_NOVELTY_TOL`, router acceptance
`map_misfit <= 1e-9` = `universa.loop_v2.MAP_ACCEPT_TOL`, re-pinned at
`universa.router.RESIDUAL_TOL`). What changes in this experiment is only
HOW the alarm's decision threshold is selected.

**The four arms.** One row is one (seed, condition) pair; the four arms run
paired on the same instance, the same library view, and the same observation
draws (§4):

* **arch-full-v4** — the full system with the alarm-v2 architecture and the
  cost-aware calibrated threshold (`universa.loop_v2.arm_arch_full_v2`).
  Candidates are scored by a TRAINED `StructureRouter` over the
  18-dimensional no-anchor degradation-profile feature blocks (hard argmax);
  the TRAINED `LearnedAlarmV2` reads `alarm_features_v2` of the router's soft
  gates (`tau = 1.0`) and the raw profile block and decides fit/no-fit
  against the **cost-aware train-block-calibrated threshold** of §5. Fit ->
  route to the router's argmax. No-fit -> certified discovery on the EXACT
  transported observations, gated through admission; on admission the row
  routes to the appended structure, else it is refused. This is the SAME arm
  FUNCTION as loop-v3 (`arm_arch_full_v2`) with a differently-calibrated
  threshold argument; its `ArmOutcome` records carry the module's unchanged
  `arm = "arch_full"` vocabulary with the `alarm_v2=` marker in `detail`
  (§6). The design name **arch-full-v4** refers to that arm function driven
  by this experiment's calibrated threshold.
* **routing-only** — ablation with NO alarm and NO discovery: the trained
  router's forced argmax, always. Unchanged.
* **discovery-only** — ablation that ALWAYS runs certified discovery on the
  exact transported observations; the library is unused. Unchanged.
* **generic-mlp** — the NO-ARCHITECTURE arm: a `GenericMLP` over the 18-dim
  generic spectral features; classes `0..K-1` route to that candidate, class
  `K` is "no-fit" and synthesizes nothing. There is no certified synthesis
  channel anywhere in this arm. Unchanged.

**Declared caveats — part of the frozen design, stated before any data.**

1. **The generic arm is deliberately architecture-free.** Its features use NO
   commutation residual and NO degradation profile (provably — the builder
   never receives the chain map and never calls the misfit machinery,
   asserted structurally and behaviorally in `tests/test_loop_v2.py`); it has
   no alarm logic and no certified synthesis channel. Its out-of-library
   failure is therefore STRUCTURAL BY CONSTRUCTION, not an empirical finding.
   The informative comparisons are the in-library and null-control conditions
   plus the two ablations; the out-of-library leg of the arch-vs-generic
   margins is a design constant, and h1/h3 are read with that declared.
2. **h4 is a bounded-harm claim, not a superiority claim.** Always-discovering
   is expected to be near-perfect in-library (the certified head on exact
   data). The claim is that the loop's SELECTIVITY costs no accuracy against
   always-discovering beyond a frozen `0.05` margin. **It may pass or fail;
   either outcome is reported as frozen, never reinterpreted** — loop-v2's h4
   failed by one seed and loop-v3's by fourteen, and this experiment's h4 is
   read against exactly those baselines. The price of selectivity is reported
   descriptively: discovery-invocation counts per condition per arm and wall
   times per arm, never claim-tested.
3. **The alarm remains a LEARNED component with a CALIBRATED threshold.** Its
   eval-block precision is MEASURED, not assumed: the calibration is a
   train-block procedure whose record is reported in full, and nothing in
   this design certifies that the alarm fires exactly when the library cannot
   explain the instance. The claims are constructed so that alarm errors show
   up as correctness losses (a false "fit" out-of-library strands the row on a
   decoy; a false "no-fit" in-library sends the row down the discovery path
   and forfeits the route credit) — the same construction as loop-v2 and
   loop-v3, so all three generations' measured error modes compare directly.
4. **Degradation applies to the observed structures, never to the transported
   data** — restated because it is load-bearing: every certified quantity the
   discovery path reads is computed from the exact `Y`, so the discovery gate
   semantics mean exactly what they meant in the clean-regime experiments and
   in loop-v2/v3. Only the routing and alarm layers read degraded inputs.
5. **The rule change is an a-priori response to a published tradeoff, not a
   guaranteed improvement.** The cost-aware rule and the equal unit costs are
   DECLARED, not validated. Nothing assumes the eval-block false-quiet rate,
   false-alarm rate, or total error improves on loop-v3's, and nothing assumes
   any claim passes. The experiment exists to MEASURE where symmetric pricing
   lands on the frontier whose two other points are already sealed.
6. **The equal-cost identity is declared, not discovered** (see above): at
   equal unit costs this rule IS unconstrained balanced-accuracy
   maximization. The experiment's alarm-side change is precisely the removal
   of loop-v3's binding false-quiet constraint, and it is described that way
   throughout rather than dressed as a new objective.

**What this experiment cannot establish.** The claims are claims about the
frozen pipeline — instance family, feature builders, three trained models
with frozen seeds plus the frozen calibration rule, four arms, frozen gates —
over the declared seed blocks, and nothing else. In particular it cannot
establish: anything about the other Universa families (2-complexes, cellular
sheaves, category nerves) or transfer across families; anything about the
CLEAN regime (loop v1's territory, sealed separately); anything about the v1
or bounded-threshold alarm designs (loop-v2's and loop-v3's territory — their
published results are this experiment's documented baselines, not evidence
for or against any claim here); anything about other alarm designs, feature
sets, calibration rules, cost weights, or thresholds; anything about other
observation regimes, mask fractions, or grids; anything about other training
seeds, architectures, or hyperparameters; or anything about real data, real
graphs, or any deployed system. Given the seed blocks and the frozen torch
seeds the whole pipeline is deterministic, so intervals quantify variation
over the seed mechanism, not sampling noise of the procedure.

All prior numbers — the published loop-v2 and loop-v3 results, the documented
behavior of `universa.loop_v2` on its fixture seeds, the router-v2 sealed
result, the loop-v1 sealed result, the README figures, and any engineering
check — are sanity evidence only. They are not evidence for or against any
claim here; only the sealed run decides the four claims of §8.

## 2. Immutable inputs and hashes

Known immutable inputs at protocol drafting are the Universa code manifest —
one SHA-256 per file of `src/universa/*.py` at the design commit — the
runner, and this protocol. No fingerprint is available at draft time: the
design commit does not exist yet. Every entry below is the literal
placeholder **`PENDING-DESIGN-SEAL`**, to be replaced by actual 64-hex
fingerprints in the machine-readable seal record of §12 (commit B), computed
at the design commit (commit A). The runner refuses to start while any
placeholder remains.

The manifest has 19 entries: the same 19 files sealed in the loop-v2 and
loop-v3 experiments. `loop_v2.py` at this design commit carries, in addition
to the alarm-v2 machinery, the new `calibrate_threshold_cost_aware` rule —
an ADDITIVE change: `calibrate_threshold` is untouched, so loop-v3's frozen
semantics are preserved exactly and its sealed result remains reproducible
under its own manifest. This experiment adds no `src/universa` file. The
prior seals and their manifests remain the frozen records of those
experiments and are not modified by this one.

| object | path | SHA-256 |
|---|---|---|
| universa package init | `src/universa/__init__.py` | `PENDING-DESIGN-SEAL` |
| probe budgets and budget instances | `src/universa/budgets.py` | `PENDING-DESIGN-SEAL` |
| category/nerve instances | `src/universa/category_instances.py` | `PENDING-DESIGN-SEAL` |
| 2-complex family | `src/universa/complexes2.py` | `PENDING-DESIGN-SEAL` |
| certified discovery head | `src/universa/discovery.py` | `PENDING-DESIGN-SEAL` |
| graph generators and `subseed` | `src/universa/generators.py` | `PENDING-DESIGN-SEAL` |
| route-or-discover loop (clean regime) | `src/universa/loop.py` | `PENDING-DESIGN-SEAL` |
| degraded-regime loop (learned router/alarm arms, alarm v2, cost-aware calibration) | `src/universa/loop_v2.py` | `PENDING-DESIGN-SEAL` |
| multi-hop transport | `src/universa/multihop.py` | `PENDING-DESIGN-SEAL` |
| Grothendieck nerve | `src/universa/nerve.py` | `PENDING-DESIGN-SEAL` |
| exact linear-algebra operators | `src/universa/operators.py` | `PENDING-DESIGN-SEAL` |
| partial observation models | `src/universa/partial.py` | `PENDING-DESIGN-SEAL` |
| 2-complex observation models | `src/universa/partial2.py` | `PENDING-DESIGN-SEAL` |
| group-nerve observation models | `src/universa/partial_group.py` | `PENDING-DESIGN-SEAL` |
| sheaf observation models | `src/universa/partial_sheaf.py` | `PENDING-DESIGN-SEAL` |
| structure router v0/v1 | `src/universa/router.py` | `PENDING-DESIGN-SEAL` |
| no-anchor router v2 | `src/universa/router_v2.py` | `PENDING-DESIGN-SEAL` |
| cellular sheaf family | `src/universa/sheaves.py` | `PENDING-DESIGN-SEAL` |
| chain-complex format | `src/universa/structures.py` | `PENDING-DESIGN-SEAL` |
| sealed runner | `scripts/run_loop_v4_sealed_1.py` | `PENDING-DESIGN-SEAL` |
| this protocol | `docs/30-sealed-router-loop-v4-protocol.md` | recorded externally in the seal record |

This protocol's own SHA-256 cannot appear inside itself — a file cannot
contain its own hash — so it is recorded externally in the seal record
(`protocol_sha256`, §12) and embedded as a frozen constant in the runner.

The runner must fail before any instance construction unless every fingerprint
above matches. It must also require this execution environment, recorded in
full in the result:

| component | required value |
|---|---|
| exact mathematics | numpy `float64` throughout the certified machinery (features, residuals, gates, calibration metrics); torch tensors carry only the learned models' float32 parameters and feature blocks |
| tensor device and dtype | CPU, `torch.float32`, for the three learned models only |
| PyTorch threads | exactly `1` |
| CUDA | unavailable or hidden (`torch.cuda.is_available()` is `False`) |

The runner records the Python, PyTorch, and NumPy versions, the operating
system and machine, `CUDA_VISIBLE_DEVICES`, the thread setting, the Git
revision, the initial `git status --porcelain --untracked-files=all` (which
must be empty), every verified hash, and the canonical command. No dependency
may be installed or upgraded between the seal and the run.

## 3. Seed blocks and the no-preview declaration

**Train block (declared, unsealed, reserved):** the 400 consecutive generator
seeds

```text
220001..220400
```

This block trains the three learned models of §5 and calibrates the alarm's
threshold, inside the sealed run, after the seal. It is the same 400-seed
data budget loop-v3 declared, carried forward unchanged: this experiment
changes the calibration RULE, not the data budget, and holding the budget
fixed is what makes the loop-v3/loop-v4 before/after a clean one-factor
comparison. It is the REPLACEMENT declaration of errata 1 (§0); the first
declared train block `200001..200400` is void. It is unsealed in the
series' sense — the confirmatory claims rest solely on the untouched eval
block below — but it is reserved against any preview until the canonical
run consumes it: at the time this amendment was written it was absent from
the working tree and from all Git history (verified by a repository-wide
word-boundary search over every commit, with ZERO hits), and no project run
had instantiated or previewed any member **on any generator family or in
any pipeline stage**. The block is disjoint from every train and eval block
and every demo or fixture seed previously declared in this series,
including both voided first blocks and the burned diagnostic range
`500001..500600` of §0.

**Sealed eval block:** the 36 consecutive generator seeds

```text
230001, 230002, 230003, 230004, 230005, 230006, 230007, 230008, 230009,
230010, 230011, 230012, 230013, 230014, 230015, 230016, 230017, 230018,
230019, 230020, 230021, 230022, 230023, 230024, 230025, 230026, 230027,
230028, 230029, 230030, 230031, 230032, 230033, 230034, 230035, 230036
```

This is the REPLACEMENT eval declaration of errata 1 (§0); the first
declared eval block `210001..210036` is void. At the time this amendment
was written, this block was absent from the working tree and from all Git
history (verified by a repository-wide word-boundary search over every
commit, with ZERO hits), and no project run had instantiated or previewed any
member **on any generator family or in any pipeline stage**. In particular,
no one has built an instance, computed a residual, drawn an observation,
trained or evaluated a model, calibrated a threshold, printed a dimension,
smoke-tested, or debugged with any seed in the block. The block is disjoint
from every train and eval block and every demo or fixture seed previously
declared in this series.

**Block history, declared in full.** This experiment carries one erratum of
its own (§0): its FIRST declared blocks `200001..200400` (train) and
`210001..210036` (eval) were consumed by the failed first attempt and are
**void for claims**, never to be reused or partially salvaged; the blocks
above are their verified-absent replacements. The scratch range
`500001..500600`, used once to measure the generator-collision rate during
the errata diagnosis, is burned and must never be declared as a train or
eval block. The series' prior voided blocks remain void and untouched, and
are not used here: `130001..130036` (void for claims after the loop-v1
design failure), `70101..70136` (discarded unopened), and `150001..150200`
(voided pre-seal by test-side instantiation during the loop-v2 design,
never used). The consumed blocks of prior experiments are their sealed
history, are never reused here, and this experiment reads their behavior
only from the PUBLISHED records: loop-v2's eval `160001..160036` and train
`170001..170200` (docs/24), and loop-v3's eval `190001..190036` and train
`180001..180400` (docs/27).

The no-preview rule remains in force until commit B of §12 — the
machine-readable seal record — is committed **and pushed to the private
remote**. Until then:

- tests may use the documented loop fixture seeds (`70001..70005`) and the
  learned-model fixture blocks of `tests/test_loop_v2.py` and the runner's
  tests (`70501..70540` train fixtures, `70601..70620` validation fixtures),
  or hand-built fixtures, only — runner tests monkeypatch the runner's frozen
  seed constants to such seeds — and may not parameterize over, import,
  derive data from, or otherwise touch the sealed eval block, the reserved
  train block, or any other sealed block (`20101..20136`, `30101..30136`,
  `40101..40136`, `60101..60136`, `70101..70136`, `80101..80136`,
  `90101..90136`, `130001..130036`, `140001..140036`, `150001..150200`,
  `160001..160036`, `170001..170200`, `180001..180400`, `190001..190036`);
- source-code constants containing the declared integers (e.g. the runner's
  frozen seed-list constants) are permitted;
- merely calling `make_budget_instance`, `make_loop_instance`,
  `make_switch_instance`, or any `universa.loop_v2` arm, feature builder,
  training function, `calibrate_threshold`, or
  `calibrate_threshold_cost_aware` with a declared eval or train seed counts
  as a preview, even if the result is discarded.

If any declared eval seed is instantiated before commit B is pushed, the
entire 36-seed block is void. It will not be partially salvaged. A new
consecutive block must be chosen, verified absent from history, and declared
in a newly committed protocol before any execution. The same rule applies to
the reserved train block: any premature instantiation voids it, and a
replacement train block must be declared in a newly committed protocol.

## 4. Data and generation protocol

**The instance.** For a generator seed `s`, the instance is exactly

```text
instance = universa.budgets.make_budget_instance(s, 8, 14, 6, 3)
```

— a random connected source graph, a planted quotient chain map onto
`num_classes=6` classes (keeping the quotient non-complete so decoys have
genuinely different kernels), and 3 certified discriminating decoys, all
exact integer arithmetic, float64, with a planted transported quantity and a
certified identifiability threshold. Identical to loop-v2 and loop-v3.

**The canonical paired views — K = 3 in EVERY condition.** Per
`universa.loop_v2` (its module docstring and fail-closed view validation):

```text
in_library   = instance.candidates[:3]
             = (instance.true_target, *instance.decoy_targets[:2])
out_library  = instance.decoy_targets          # all three decoys
```

The in-library view carries the true target at index 0 (generator convention,
never permuted) plus the FIRST TWO decoys; the out-of-library and
null-control views are all three decoys with the truth withheld. K = 3 in
every condition is REQUIRED by the fixed input widths of the learned models:
the `LearnedAlarmV2` reads `K + 8` alarm-v2 features and the `GenericMLP`
emits K + 1 classes — so the paired in-/out-of-library rows must share one
candidate count, which forces the in-library view to drop one decoy. Each arm
validates the view against the declared condition fail-closed. Identical
construction to loop-v2 and loop-v3.

**The degraded observation draws.** Every candidate boundary enters the
learned layers through `universa.partial.ObservationModel` with
`mask_fraction = 0.25` (edge columns removed, mask applied first) and
`corrupt_fraction = g` (sign flips of surviving nonzero entries) for every
grid fraction `g` of the 8-point no-anchor grid. All draws of one row key on
the canonical shared observation seed

```text
observation_seed(s) = subseed(s, "router-v2-observe")
```

— one draw per (row, grid point), reused across candidates within the row:
the mask permutation derives from the seed alone and `mask_fraction` is
constant, so the same edge columns are missing at every grid point, and
corruption is nested per fraction (prefixes of one master permutation over
the surviving entries), exactly the router-v2 regime's canonical schedule.
The row's operating grid point — the single corruption fraction at which the
generic arm observes its boundaries — is derived deterministically:

```text
operating_grid_point(s) = NO_ANCHOR_GRID[subseed(s, "loop-v2-operating") % 8]
```

(`universa.loop_v2.operating_grid_point`). The arch arm's profile spans the
whole grid, so the point selects nothing there; deriving it per row spreads
the generic arm's rows across degradation levels while keeping every arm of a
row paired on the same instance, view, and observation draw. Identical to
loop-v2 and loop-v3.

**The exact transported observations.** For the discovery path and the
generic arm's spectral features, the exact observation matrix is the frozen
exp4 schedule of 16 columns over the instance's `SwitchInstance` view:

```text
switch_view  = universa.generators.SwitchInstance(
                   instance.seed, instance.source, instance.true_target,
                   instance.chain_map, instance.decoy_targets)
Y            = universa.discovery.synthesize_observations(switch_view, 16)
```

Column `j` is `f1 a_j` with `a_j` a certified random source cycle drawn from
`subseed(s, "discovery-observation", str(j), str(attempt))`, `attempt`
counting from 1, redrawn while `||y_j|| < 1e-9`
(`universa.discovery.MIN_OBSERVATION_NORM`), at most 1000 attempts. Every
column lies in `im(f1)`, contained in `ker(B1_true)`, exactly — the
degradation never touches the data (caveat 4). Identical to loop-v2/v3.

**The null-control observations.** The specificity control replaces `Y` with
structure-free noise, the frozen exp4 H4 / loop-v1 schedule
(`universa.loop.null_observations`): column `j` is

```text
np.random.default_rng(subseed(s, "discovery-null", str(j)))
      .standard_normal(ambient_dim)        for j = 0..15
```

with `ambient_dim` the number of edges of the seed's true quotient target;
`Y_null` is the column stack (`ambient_dim x 16`, float64). The null matrix
is consulted only where an arm reads observations. The arch and routing arms'
feature blocks are identical in the out-of-library and null-control
conditions per seed. Identical to loop-v2/v3.

**The three conditions per seed, in build order.** Exactly three conditions,
all deterministic end-to-end, applied to every eligible seed:

1. **in-library** — view `in_library` (the truth at index 0), observations
   `Y`. The expected arch behavior is alarm-fit and route to index 0.
2. **out-of-library** — view `out_library` (the truth withheld), observations
   `Y`. The expected arch behavior is alarm-no-fit, certified discovery,
   admission, and re-route to the appended structure.
3. **null-control** — view `out_library`, observations `Y_null`. The expected
   behavior is that nothing is admitted by any arm with a discovery channel,
   and refusal from the arms whose only specificity mechanism is refusal.

**The four arms per condition, paired.** Each condition of each eligible seed
runs all four arms, in the module's `ARMS` order (`arch_full`,
`routing_only`, `discovery_only`, `generic`), all sharing the same
observation draws:

```text
arm_arch_full_v2(s, instance, view, router=router, alarm=alarm,
                 threshold=threshold,
                 observation_seed=observation_seed(s),
                 observations=Y_or_null, condition=condition)
arm_routing_only(s, instance, view, router=router,
                 observation_seed=observation_seed(s), condition=condition)
arm_discovery_only(s, instance, observations=Y_or_null, condition=condition)
arm_generic(s, instance, view, generic_model=generic,
            observation_seed=observation_seed(s), observations_y=Y_or_null,
            condition=condition)
```

where `threshold` is the cost-aware calibrated threshold of §5 — computed
once on the train block inside the sealed run, frozen for the whole eval, and
recorded in the calibration record. Every defaulted argument stays at its
module default — the profile grid `NO_ANCHOR_GRID`, `mask_fraction = 0.25`,
`novelty_tol = 1e-6`; the discovery certification gate is the module-frozen
`1e-10` and the router acceptance gate `1e-9`. There is no per-seed or
per-condition choice of any kind.

**Deterministic random streams.** Every stochastic component uses
`universa.generators.subseed`:

```text
message = ":".join(["universa", str(seed), *components]).encode("utf-8")
digest  = SHA256(message)
subseed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
```

The disjoint subseed families of one row are `"router-v2-observe"`,
`"loop-v2-operating"`, `"discovery-observation"`, and `"discovery-null"` —
identical to loop-v2 and loop-v3.

**Determinism, verified.** Given the trained models and the calibrated
threshold, every arm is a deterministic function of the seed, and
`ArmOutcome` is a pure scalar record precisely so that two runs compare
bit-identical with plain equality; the models themselves are deterministic
functions of the train block and the frozen torch seeds, and the calibrated
threshold is a deterministic function of the train block and the trained
alarm (the cost-aware calibration rule uses no RNG anywhere, §5). The runner
executes the full per-seed eval pipeline — all three conditions x all four
arms — TWICE for every eligible seed with the same trained models and the
same calibrated threshold, and requires bit-identical raw rows; any mismatch
is a whole-run `design_failure`. Any NaN or non-finite feature, residual,
misfit, loss, score, threshold, cost, or statistic, and any violation inside
the certified machinery, the training loops, or the calibration rule, is
likewise a whole-run `design_failure`, never a dropped row.

## 5. Model and training

Exactly three models are trained — inside the sealed runner, after the seal,
using only the declared train block `220001..220400` — and no other learned
parameter exists anywhere in the design. The alarm's decision threshold is
not a learned parameter and not a frozen constant: it is computed by the
frozen cost-aware calibration rule below on the same train rows, inside the
same sealed run. Training happens only AFTER the eval-block eligibility gate
of §10 passes (an insufficient-eligible run fits nothing, §9).

**The train rows.** Per train seed `s` in `220001..220400`, in seed order,
the runner builds the instance, BOTH paired views' rows, the shared
observation draw, and the exact observation matrix, exactly as §4 pins for
eval rows:

```text
instance = make_budget_instance(s, 8, 14, 6, 3)
in_library, out_library = instance.candidates[:3], instance.decoy_targets
obs = subseed(s, "router-v2-observe"); point = operating_grid_point(s)
block_in,  raw_in  = arch_row_features(instance, in_library,  obs)
block_out, raw_out = arch_row_features(instance, out_library, obs)
gen_in  = generic_row_features(instance, in_library,  obs, Y, point)
gen_out = generic_row_features(instance, out_library, obs, Y, point)
```

with `Y` the exact 16-column transported matrix of §4.

**Train-side build exclusion (errata 1, §0).** A train seed whose INSTANCE
fails to build is not a training example — it is not an instance of the
family at all, because the generator's discriminability guard refuses a
decoy that shares the true target's kernel (`docs/00-design.md` §6). Such a
seed is RECORDED with its reason and EXCLUDED, mirroring exactly how §10
records an ineligible eval seed; the models train on the seeds that build.
The exclusion is narrow, and everything else keeps the original discipline:

- only a failure of `make_budget_instance` is an exclusion;
- a feature-construction exception, a shape violation, or a non-finite
  feature is a whole-run `design_failure`, never an exclusion — a seed that
  builds but whose features fail signals a pipeline fault, not a
  non-instance;
- more than `MAX_TRAIN_EXCLUDED = 20` non-instances in the declared block,
  or a block in which nothing builds, is itself a whole-run
  `design_failure` — the fail-closed guard over the exclusion. The guard
  counts exclusions rather than survivors so that it means the same thing
  at any block size, including the runner's monkeypatched test fixtures;
- the excluded seeds with their reasons, the built count, the declared
  count, and the ceiling are recorded in the training provenance and
  reported (§11).

There is still no train-side ELIGIBILITY gate in the §10 sense: nothing
about a train seed's audit, residuals, or difficulty excludes it. Only the
non-existence of the instance does.

**Model 1 — the router.** `StructureRouter(feature_dim=18, hidden_dim=64)`,
the shared per-candidate MLP `18 -> 64 -> 64 -> 1` of `universa.router`,
constructed and trained by `universa.router.train_router` on the 400
in-library arch blocks (`(400, 3, 18)`, label `0` for every row) with:

- `epochs=150` (full-batch Adam: every epoch is one step, no shuffling RNG);
- `lr=1e-3`, PyTorch `torch.optim.Adam` defaults for every unspecified option;
- torch seed `4242` (`torch.manual_seed(4242)`, governing initialization);
- `lambda_aux=0.01` (Switch-style load-balancing pressure);
- temperature annealed exponentially `tau: 2.0 -> 0.25` (exactly `2.0` at
  epoch 0, exactly `0.25` from epoch 149 on, monotone non-increasing);
- CPU, `torch.float32`, exactly one PyTorch thread;
- input standardization statistics (mean and population std, clamped at
  `1e-8`) measured on the **train rows only** and stored in the model's
  buffers.

The validation argument of `train_router` is the training block itself. It
feeds only the `val_accuracy` history diagnostic — no early stopping, no
model selection, no gradient — and uses only train seeds.

**Model 2 — the learned alarm.** `LearnedAlarmV2(num_candidates=3)` —
`Linear(11, 32) GELU Linear(32, 1)` — trained by
`universa.loop_v2.train_alarm_v2` on the alarm-v2-feature rows of BOTH views:

```text
rows_fit   = stack(alarm_features_v2(router_gates(router, block_in[s]),
                                     raw_in[s])   for s in train block)  # (400, 11)
rows_nofit = stack(alarm_features_v2(router_gates(router, block_out[s]),
                                     raw_out[s])  for s in train block)  # (400, 11)
```

— in-library rows labeled fit (1), out-of-library rows no-fit (0) — with
`epochs=150`, `lr=1e-3`, torch seed `4243` (`torch.manual_seed(4243)` runs
immediately before `LearnedAlarmV2(3)` is constructed AND is passed as
`train_alarm_v2`'s `seed`), binary cross-entropy on the logit, full-batch
Adam, and train-only standardization stored in the model's buffers. The
alarm's architecture and features are UNCHANGED from loop-v3 — the five
margin columns of `ALARM_V2_MARGIN_FEATURE_NAMES`
(`log1p_profile_margin_second_minus_best`, `gate_top2_gap`,
`log1p_mean_profile`, `best_profile_curvature`, `log1p_best_profile`) on top
of the frozen v1 `K + 3` layout, `K + 8` = 11 dims at `K = 3`. Training sets
NO threshold; the history's train accuracy is a diagnostic at the `0.5` score
and decides nothing.

**The frozen calibration rule — the threshold. This is the experiment's one
changed component.** Immediately after alarm training, on the SAME train rows
(`rows_fit`, `rows_nofit`), the runner calls

```text
universa.loop_v2.calibrate_threshold_cost_aware(model, rows_fit, rows_nofit,
                                                false_quiet_cost=1.0,
                                                false_alarm_cost=1.0)
```

and the returned threshold becomes the alarm's frozen decision threshold for
the whole eval (`alarm_decision_v2`: `sigmoid(logit) >= threshold` decides
fit; a score exactly AT the threshold decides fit). The rule, pinned verbatim
from the module: the model's sigmoid scores are computed on both row sets;
the candidate thresholds are the sorted unique scores plus `0.0` and `1.0`,
deduplicated — the SAME sweep as loop-v3's rule; for each candidate `t` the
decision is `score >= t` => fit and the rule records

* balanced accuracy — `(TPR + TNR) / 2` with `TPR` the fraction of fit rows
  scored `>= t` and `TNR` the fraction of no-fit rows scored `< t`;
* **false-quiet rate** — the fraction of NO-FIT rows scored `>= t`: the alarm
  stays quiet on an out-of-library row when it should fire, no discovery
  runs, the acquisition is lost;
* **false-alarm rate** — the fraction of FIT rows scored `< t`: the alarm
  fires on an in-library row when it should stay quiet, discovery correctly
  refuses as non-novel, the route credit is lost;
* **total cost** — `false_quiet_cost * false_quiet_rate +
  false_alarm_cost * false_alarm_rate`.

Selection: the threshold **MINIMIZING total cost**, ties broken toward the
LARGER balanced accuracy and then the LARGER threshold. There is no
feasibility bound, therefore no fallback branch and no `bound_satisfied`
field — the rule always selects the sweep's cost minimizer. This is the sole
difference from loop-v3's rule, which maximized balanced accuracy subject to
`false_quiet_rate <= 0.02`.

**The frozen costs and the identity.** The costs are frozen at
`false_quiet_cost = 1.0` and `false_alarm_cost = 1.0`, declared in §1 with
their justification (each error mode costs exactly one seed in one condition,
and the two conditions carry equal weight in the end-to-end statistic of §7).
As §1 states and as the module's docstring records, at equal unit costs
`FQ + FA = 2 − 2 * balanced_accuracy` exactly, so this rule is identical to
UNCONSTRAINED balanced-accuracy maximization and the balanced-accuracy
tiebreak can never fire. That is declared, intended, and reported; no other
cost values are used anywhere in this experiment.

The full calibration record — `threshold`, `balanced_accuracy`,
`false_quiet_rate`, `false_alarm_rate`, `total_cost`, `false_quiet_cost`,
`false_alarm_cost`, `num_candidates` — is recorded in the result's training
provenance (§11). The rule is deterministic (no RNG anywhere) and fail-closed
(empty or non-finite row sets, a width mismatch, non-finite scores, or costs
that are non-finite, negative, or both zero are errors); any failure inside
it is a whole-run `design_failure`.

**Model 3 — the generic model.** `GenericMLP(num_candidates=3)` — the
DeepSets head `Linear(18, 64) GELU Linear(64, 64) GELU` per candidate, a mean
over the candidate axis, then `Linear(64, 4)` — trained by
`universa.loop_v2.train_generic` on the 800 generic rows (the 400 in-library
blocks labeled `0`, the 400 out-of-library blocks labeled `3` = K, the no-fit
class) with `epochs=150`, `lr=1e-3`, torch seed `4244`
(`torch.manual_seed(4244)` runs immediately before `GenericMLP(3)` is
constructed AND is passed as `train_generic`'s `seed`), cross-entropy,
full-batch Adam, and train-only standardization stored in the buffers.

**Training order is pinned** — router (4242), then alarm (4243) with its
calibration, then generic (4244) — because the alarm's features read the
trained router's soft gates, and the calibration reads the trained alarm's
scores on the train rows. A non-finite loss at any epoch of any model is an
error, never a warning, and a whole-run `design_failure`.

**Declared consequences of the index-0 convention, stated before any data.**
The loop never permutes candidates: the truth sits at index 0 of the
in-library view and the views are validated against exactly that. Two
consequences follow mechanically and are declared, not discovered after the
fact. (a) The router's training rows all carry label `0`, so nothing in the
router's training signal distinguishes the candidates' positions — the
router's learned role is profile-shape scoring under the index-0 commitment
convention, and the loop's learned SELECTIVITY lives in the alarm, not the
argmax. (b) The generic arm's index classes degenerate toward "route to 0"
for the same reason; its real task is the fit/no-fit separation from spectral
statistics alone. The in-library claims are declared against exactly this
construction: h3 is a strict-margin claim that a tie FAILS (§8), and h4 is a
bounded-harm claim against the always-discover ablation, not a
routing-quality claim. Identical to loop-v2 and loop-v3.

Inference is always strictly discrete: the router's hard argmax over
candidate logits (first-index tie-breaking), the alarm's thresholded sigmoid
against the calibrated threshold, the generic model's argmax — never sampled
or mixed.

Training provenance recorded in the result, per model: the torch seed, the
epoch count, the learning rate, the frozen objective (including `lambda_aux`,
`tau_start`, `tau_end`, `hidden_dim`, `feature_dim` for the router; the
`K + 8` feature layout and the five margin column names for the alarm), the
row counts, the final epoch's loss(es) and train accuracy, the optimizer
(`full-batch Adam`), the dtype/device (`CPU float32`), the standardization
note, and the model state SHA-256 — computed over the canonical
concatenation, in lexicographic name order, of each `state_dict` entry's
UTF-8 name, a NUL byte, and its little-endian float32 bytes. For the alarm,
the provenance additionally carries the FULL eight-field calibration record
above, with the frozen costs noted and the rule named
`calibrate_threshold_cost_aware`.

## 6. Arms and conditions

Exactly three conditions per seed, in the frozen build order of §4 —
**in-library**, **out-of-library**, **null-control** — and exactly four arms
per condition, paired on the same observation draws. The arm semantics are
the module's frozen ones:

1. **arch-full-v4** (`universa.loop_v2.arm_arch_full_v2` driven by this
   experiment's cost-aware calibrated threshold) — the trained router scores
   every candidate over the 18-dim arch profile blocks (hard argmax); the
   trained `LearnedAlarmV2` reads `alarm_features_v2` of the soft gates
   (`tau = 1.0`) and the raw profile block and decides fit/no-fit against the
   calibrated threshold of §5. FIT -> route to the router's argmax (discovery
   never invoked). NO-FIT -> certified discovery on the EXACT transported
   observations at the frozen gates (certification `1e-10`, novelty `1e-6`);
   on admission the row routes to the appended structure (discover mode),
   else it is refused. Its `ArmOutcome` records carry the module's unchanged
   outcome vocabulary — `arm = "arch_full"`, the same `action` values, the
   same fields — with an `alarm_v2=` marker in `detail` for provenance; the
   result's raw rows are keyed by that frozen vocabulary (§11).
2. **routing-only** (`arm_routing_only`) — NO alarm, NO discovery: the
   trained router's forced argmax, always a route. On out-of-library rows it
   must pick a decoy — an honest failure by the frozen semantics.
3. **discovery-only** (`arm_discovery_only`) — ALWAYS certified discovery on
   the exact observations, the library unused for routing; novelty is checked
   against the instance's decoy library.
4. **generic-mlp** (`arm_generic`) — the NO-ARCHITECTURE arm: the
   `GenericMLP`'s decision over the 18-dim generic spectral blocks at the
   row's operating grid point. Classes `0..2` route to that candidate; class
   `3` ("no-fit") synthesizes nothing and records a refusal.

**Correctness semantics (frozen; the module's `_condition_correct` gate,**
reading only the outcome record and the index-0 convention, never the
withheld truth — IDENTICAL to loop-v2 and loop-v3):

| arm | in-library correct iff | out-of-library correct iff | null-control correct iff |
|---|---|---|---|
| arch-full-v4 | `action == "route"` and `routed_index == 0` | `action == "discover"` and `admitted` and `map_misfit <= 1e-9` | NOT `admitted` |
| routing-only | `action == "route"` and `routed_index == 0` | never (no acquisition channel) | NOT `admitted` (vacuous: it admits nothing) |
| discovery-only | `action == "discover"` and `admitted` and `map_misfit <= 1e-9` | `action == "discover"` and `admitted` and `map_misfit <= 1e-9` | `action == "refused"` |
| generic | `action == "route"` and `routed_index == 0` | never (no certified synthesis channel) | `action == "refused"` |

That is: **in-library** correct means the arm's final structure is the true
target. **Out-of-library** correct means a certified novel structure is
admitted with `map_misfit <= 1e-9` — only the two arms with a certified
discovery channel can ever be correct there, by construction (caveat 1).
**Null-control** correct means nothing is admitted: for the arms with a
discovery channel the false-admission control of loop v1's H4; for the
alarm-less/generic arms, whose only specificity mechanism is refusal, correct
iff refused. Note the two refusal shapes, declared: for arch-full-v4 and
routing-only a route on null-control rows admits nothing and is therefore
correct — the control penalizes false ADMISSION, not commitment; for
discovery-only and generic only an explicit refusal is correct.

There is no fifth arm, no fourth condition, no ensemble, no eval-time tuning,
and no per-seed choice of any kind.

## 7. Estimands and unit of inference

For each eligible seed `s`, condition `c` in `{in_library, out_of_library,
null_control}`, and arm `a` in `{arch_full, routing_only, discovery_only,
generic}` (the frozen outcome vocabulary; `arch_full` rows are delivered by
arch-full-v4, §6), write `correct(s, c, a)` in `{0, 1}` for the frozen
correctness indicator of §6. The per-seed condition-mean correctness of an
arm is

```text
cm(s, a) = (correct(s, in_library, a) + correct(s, out_of_library, a)
            + correct(s, null_control, a)) / 3
```

The four frozen per-seed claim statistics are the paired differences

```text
d1(s) = cm(s, arch_full) - cm(s, generic)                      # arch_vs_generic_e2e
d2(s) = cm(s, arch_full) - cm(s, routing_only)                 # arch_vs_routing_only_e2e
d3(s) = correct(s, in_library, arch_full)
        - correct(s, in_library, generic)                      # arch_vs_generic_inlibrary
d4(s) = correct(s, in_library, arch_full)
        - correct(s, in_library, discovery_only)               # arch_vs_discovery_only_inlibrary_harm
```

— intentionally IDENTICAL in shape to loop-v2's and loop-v3's four
statistics, so the three-point before/after is directly measurable.

Declared conventions, fixed here:

- **The alarm's decision basis precedes observations.** The arch and routing
  arms' feature blocks — and therefore the router's gates and the alarm's
  verdict — are computed from the degraded candidate boundaries and the
  shared observation draw alone; the observation matrix (exact `Y` or
  `Y_null`) is consulted only by the discovery path and the generic arm's
  spectral features. Per seed, arch-full-v4 and routing-only therefore
  present identical feature blocks in the out-of-library and null-control
  conditions — and the alarm remains LEARNED (caveat 3): its verdicts are
  measured behavior, not certified decisions.
- **The paired design is the point.** All four arms of a (seed, condition)
  share the instance, the view, and the observation draws, so each difference
  `d1..d4` cancels per-seed instance difficulty and degradation realization.
- **The out-of-library legs of d1 are structural, declared.** Routing-only
  and generic can never acquire out-of-library (caveat 1), so their
  out-of-library indicators are 0 by construction; the informative legs of
  the end-to-end margins are the in-library and null-control conditions.
- **The descriptive quantities are never claim inputs.** Discovery-invocation
  counts per condition per arm, per-arm wall times, per-condition per-arm
  accuracies, the calibration record, and the three-point frontier comparison
  against the published loop-v2 and loop-v3 numbers (§1) are reported (§11)
  and read by no decision rule.

The **generator seed is the sole unit of inference**. It jointly determines
the instance, both library views, the shared observation draw, the operating
grid point, the exact observation matrix, and the null observation matrix;
the four arms and three conditions within a seed are paired measurements,
never independent units, and are never pooled as such.

Each claim's estimand is `theta = E_seed[d_k]` over the eligible-seed
mechanism, estimated by the arithmetic mean across the `n` eligible seeds,
with

```text
SE = sample_standard_deviation(d_k) / sqrt(n)
```

(sample standard deviation, `n - 1` denominator). Student-t inference assumes
eligible seed-level realizations are exchangeable draws from this
deterministic generator-and-subseed scheme; it conditions on the frozen
eligibility rule of §10. Because the whole pipeline — training and
calibration included — is deterministic given the declared seed blocks and
the frozen torch seeds, intervals quantify Monte Carlo variation over the
seed mechanism only, and say nothing about uncertainty for real data.

## 8. The four frozen claims and decision rules

The following four and only these four claims form the confirmatory family.
H1 is primary; H2–H4 are secondary members of the same family. The family is
intentionally IDENTICAL in shape to loop-v2's and loop-v3's — same
statistics, same thresholds, same directions, same alpha — with the arch arm
now arch-full-v4, so the before/after is directly measurable against the
published records (docs/24, docs/27).

| id | scope | statistic | prediction and governing decision |
|---|---|---|---|
| `h1-loopv4-arch-vs-generic-e2e` | condition-mean (all three conditions) | `arch_vs_generic_e2e` = `d1` | mean `> 0`; supported iff the one-sided lower bound exceeds `0` |
| `h2-loopv4-arch-vs-routing-only-e2e` | condition-mean (all three conditions) | `arch_vs_routing_only_e2e` = `d2` | mean `> 0`; supported iff the one-sided lower bound exceeds `0` |
| `h3-loopv4-arch-vs-generic-inlibrary` | in-library | `arch_vs_generic_inlibrary` = `d3` | mean `> 0`; supported iff the one-sided lower bound exceeds `0` (strict; a tie fails) |
| `h4-loopv4-arch-vs-discovery-only-inlibrary-harm` | in-library | `arch_vs_discovery_only_inlibrary_harm` = `d4` | mean `> -0.05`; supported iff the one-sided lower bound exceeds `-0.05` (bounded harm) |

The formal hypotheses are fixed as:

- **H1** (id `h1-loopv4-arch-vs-generic-e2e`, primary): `theta =
  E_seed[d1]`; H0: `theta <= 0`; H_A: `theta > 0`; supported iff the
  one-sided Bonferroni lower bound is above `0`. This is the system's
  end-to-end claim, with the out-of-library leg structural by construction
  (caveat 1) and the informative legs in-library and null-control.
- **H2** (id `h2-loopv4-arch-vs-routing-only-e2e`): `theta = E_seed[d2]`;
  H0: `theta <= 0`; H_A: `theta > 0`; supported iff the one-sided Bonferroni
  lower bound is above `0`. This isolates the alarm + discovery channel.
- **H3** (id `h3-loopv4-arch-vs-generic-inlibrary`): `theta = E_seed[d3]`;
  H0: `theta <= 0`; H_A: `theta > 0`; supported iff the one-sided Bonferroni
  lower bound is above `0`. A STRICT margin: a tie fails the claim and is
  reported as frozen. Given the declared degeneration of both arms' routing
  layers toward the index-0 convention (§5), a tie is a genuinely possible
  frozen outcome; the mechanical rule below decides it, and the outcome is
  reported either way. **Loop-v3's h3 failed** (estimate `+0.0833`, lower
  bound `-0.1523`) on the variance induced by its 14 false alarms; this
  experiment's h3 is read against exactly that baseline and may pass or fail.
- **H4** (id `h4-loopv4-arch-vs-discovery-only-inlibrary-harm`): `theta =
  E_seed[d4]`; H0: `theta <= -0.05`; H_A: `theta > -0.05`; supported iff the
  one-sided Bonferroni lower bound is above `-0.05`. A BOUNDED-HARM claim,
  not superiority (caveat 2). **This claim may pass or fail; either outcome
  is reported as frozen, never reinterpreted** — loop-v2's h4 failed at
  `-1/36` and loop-v3's at `-14/36`. Discovery-invocation counts and wall
  times are reported descriptively and play no role in the decision.

Family-wise type-I error is controlled at `0.05` by Bonferroni over four
claims, one-sided Student-t with `n` eligible seeds:

```text
alpha_per_claim = 0.05 / 4 = 0.0125
q = 1 - alpha_per_claim = 0.9875
lower = mean(d_k) - t_(0.9875, n-1) * SE
```

The direction-specific bound above is the sole support rule for each claim,
applied mechanically to the frozen constants — **including when `SE` is
exactly zero**. All four statistics are bounded paired differences and can
hit this mechanical case: when the sample standard deviation is `0`,
`SE = 0` and the one-sided lower bound **equals the estimate**; support then
reduces to the estimate itself strictly exceeding the threshold (`0` for
H1–H3, `-0.05` for H4). This case is specified here in advance and is decided
by the same frozen rule, not by discretion.

The hard-coded one-sided critical values, computed once with SciPy for design
only, are:

| eligible n | df | `t_(0.9875,df)` |
|---:|---:|---:|
| 30 | 29 | `2.36384607320831` |
| 31 | 30 | `2.359562458700931` |
| 32 | 31 | `2.3555682821599135` |
| 33 | 32 | `2.3518351803763706` |
| 34 | 33 | `2.348338377257479` |
| 35 | 34 | `2.3450561343451817` |
| 36 | 35 | `2.3419692993010397` |

Exact design-time computation (SciPy 1.18.0, HOMYMOLY venv):

```python
from scipy import stats
[stats.t.ppf(1 - 0.05 / 4, n - 1) for n in range(30, 37)]
```

The table is numerically identical to the earlier experiments' (same alpha,
same eligible-n range); it is hard-coded and tested independently in this
runner. SciPy is not a runtime dependency of the runner. Each of the four
claim summaries records: the estimate; its sample standard deviation; its
standard error; the governing one-sided lower bound; the critical value; the
direction (`greater`); the threshold (`0` for H1–H3, `-0.05` for H4); and the
support decision. No hypothesis may be added, dropped, reversed,
re-thresholded, or moved between families after execution. All nulls and
failed predictions are reported. The before/after reading of these decisions
against the published loop-v2 and loop-v3 decisions is descriptive context
for the results record, never a claim input, and no claim is ever
reinterpreted in its light.

## 9. Stop rules and status vocabulary

The run's terminal status is exactly one of:

- `complete` — every stage finished and all validation passed; the only
  status that permits non-null claim decisions;
- `design_failure` — any frozen-design violation;
- `design_failure_insufficient_eligible` — fewer than 30 eligible seeds;
- `execution_failure` — any other unexpected exception;
- `interrupted` — `KeyboardInterrupt` or equivalent manual cancellation.

The mapping is frozen:

- all 36 eval seeds are attempted: instance builds run in seed order, and no
  seed is ever skipped for outcome-dependent reasons short of a build
  failure;
- an EVAL-block instance-build exception is a **whole-run**
  `design_failure`, never an exclusion — the offending seed is recorded, the
  run stops, and no seed is ever deleted alone;
- a TRAIN-block instance-build exception is a recorded EXCLUSION (errata 1,
  §0, §5), not a stop, unless more than `MAX_TRAIN_EXCLUDED = 20` seeds are
  excluded or none builds, which are whole-run `design_failure`s; a
  train-side FEATURE-construction exception remains a whole-run
  `design_failure`;
- the eval build and the undegraded audit of §10 are the sole eligibility
  gate, and it is evaluated BEFORE any training. Every *other* failed check —
  any determinism (including the double-execution bit-identity check),
  NaN/non-finite value, audit (other than the per-seed eligibility gate),
  rank, orthonormality, certificate, or validation failure, any violation
  inside the certified discovery machinery, the training loops (a non-finite
  loss), or the calibration rule, an incoherent `ArmOutcome`, and any failed
  result-audit recomputation — is a whole-run `design_failure`;
- fewer than 30 eligible seeds yields `design_failure_insufficient_eligible`
  with **no fits, no calibration, no evaluation, and no claims**: the run
  stops before any model is trained;
- a caught failure preserves every completed raw row, identifies the failing
  seed and stage, records the exception type and message and the preflight
  provenance, and emits all four claims with `supported: null`;
- there is no outcome-dependent stopping, no interim analysis, no sample-size
  extension, and no condition-, arm-, or seed-level deletion;
- the run is never rerun because the results are surprising, weak, null, or
  apparently decisive;
- the runner must not compute or print scores, accuracies, margins, or claim
  summaries until every required raw row has completed; progress output is
  limited to seed identifiers, stage, completion state, and timing.

Mandatory stop conditions (any one stops the run before or during execution):

- any preflight mismatch: seal-validation failure (including an unknown seal
  key at any level including inside a claim object, a tampered claim field, a
  `train_seed_block` other than `{first: 220001, last: 220400}`, an
  `eval_seed_block` other than `{first: 230001, last: 230036}`, a
  `design_commit` absent from the repository's history or not an ancestor of
  HEAD, or the imported `universa` package not resolving under the project's
  `src`), hash mismatch, a seal not committed at HEAD, an on-disk seal not
  byte-identical to its HEAD blob, a symlinked seal, protocol, or runner
  file, a dirty worktree, an existing output path, available CUDA, or a torch
  thread count other than one;
- a worktree found dirty, or a code-manifest hash, the on-disk seal,
  protocol, or runner file found changed, at the end-of-run re-check before
  publishing a `complete` result;
- a raw-row count that violates the frozen invariant `4 x 3 x n`;
- an eval-block instance-build exception; a train-block instance-build
  exception only when it breaches the `MAX_TRAIN_EXCLUDED` ceiling or
  empties the block; a train-block feature-construction exception;
- fewer than 30 eligible seeds;
- any determinism, NaN, audit, rank, training, calibration, or validation
  failure;
- aggregation failure or a non-finite estimand; or
- any manual cancellation or infrastructure interruption.

If infrastructure fails after execution begins, the attempt is preserved
under a distinct non-overwriting failure path
`results/experiments/failures/router-loop-v4-sealed-1.<status>.json`; the
canonical output path is reserved for status `complete` and a failed attempt
never occupies it. An exact-code retry is allowed only for a documented
non-scientific failure, before any summaries are inspected, with identical
hashes. Every attempt remains in the evidence bundle. The first complete,
fully validated attempt is canonical; no choice among complete attempts is
allowed. Any code, tolerance, dependency, or analysis change after a partial
run invalidates the seed blocks and requires a new protocol and new seeds.

## 10. Eligibility

An eval seed is **eligible** if and only if both hold:

1. its instance builds — `make_budget_instance(seed, 8, 14, 6, 3)` returns (a
   build exception is a whole-run `design_failure` per §9, never an
   exclusion); and
2. the **undegraded audit** passes: computed on the undegraded instance
   without any observation draw, the true candidate's certified commutation
   score — the max residual over degrees of the planted-map `ChainMap` probe;
   one degree in this family — is exactly `0.0` (and therefore `<= 1e-9`),
   and every decoy's score is strictly `> 1e-9`.

The audit is **bookkeeping only, never a feature**: no arm, model, alarm,
router, or generic feature ever reads it — the learned layers see only the
degraded-observation feature blocks and the exact (or null) observation
matrix; the discovery path reads only the exact observations. The audit
certifies the library's make-up per seed, which is what licenses reading a
route to index 0 as correct (the frozen correctness gate of §6) and what
licenses the no-permutation stance of §5. It is observation-independent and
therefore a deterministic per-seed property, identical across the three
conditions and the four arms. Identical to loop-v2 and loop-v3 — eligibility
is part of the certified machinery this experiment leaves untouched.

Ineligible seeds are recorded with their explicit reasons and are never
replaced, never deleted, and never scored. `n`, the number of eligible seeds,
is the inference sample size of §7–§8. If `n < 30`, the run stops before any
training and any loop execution with
`design_failure_insufficient_eligible` and no claims. All 36 seeds are
attempted, in seed order. There is no train-side ELIGIBILITY gate — nothing
about a train seed's audit, residuals, or difficulty excludes it — but per
errata 1 (§0, §5) a train seed whose INSTANCE fails to build is recorded
and excluded as a non-instance, subject to the `MAX_TRAIN_EXCLUDED = 20`
ceiling; a train-side feature-construction exception remains a whole-run
`design_failure`.

## 11. Result JSON schema

The runner writes exactly one result file, to
`results/experiments/router-loop-v4-sealed-1.json`, through a temporary file
followed by an atomic no-clobber hard-link publish (create-if-absent, never
replace), and refuses an existing path. Non-`complete` statuses are published
instead to
`results/experiments/failures/router-loop-v4-sealed-1.<status>.json` under
the same no-clobber rule, and failure artifacts are written **only** there.
The JSON contains, at minimum:

- `schema`: the literal tag `universa-router-loop-v4-sealed-result/1`, and
  the frozen configuration (the exact `make_budget_instance` call and sizes,
  the paired-view construction with `K = 3`, the no-anchor grid and mask
  fraction, the observation-seed and operating-point subseed schedule, the
  exact and null observation schedules with `num_observations = 16`, the
  three conditions in build order, the four arms in `ARMS` order with the
  arch arm delivered by `arm_arch_full_v2`, the frozen gates — certification
  `1e-10`, novelty `1e-6`, router acceptance `1e-9` — the alarm-v2 feature
  layout (`K + 8`: the frozen v1 `K + 3` columns plus the five named margin
  columns), the frozen calibration rule named
  `calibrate_threshold_cost_aware` with `false_quiet_cost = 1.0` and
  `false_alarm_cost = 1.0`, and the claim family with thresholds);
- the seal SHA-256, protocol SHA-256, runner SHA-256, and every code-manifest
  SHA-256, plus `design_commit` (commit A) and the execution revision (HEAD
  at execution time);
- environment and provenance: Python/PyTorch/NumPy versions, OS and machine,
  thread count, the operator-provided and effective `CUDA_VISIBLE_DEVICES`,
  the recorded `git status --porcelain --untracked-files=all` output (key
  `git_status_porcelain`, must be empty, recorded pre- and post-run), the
  canonical command, and `sys.argv`;
- the train-block record of errata 1 (§5): the declared seed count and
  range, the built count, the excluded seeds each with `seed`, `reason`,
  `exception_type`, and `message`, the `MAX_TRAIN_EXCLUDED` ceiling, and
  the exclusion rule in words;
- training provenance per §5 for all three models — for the alarm
  additionally the FULL eight-field calibration record (`threshold`,
  `balanced_accuracy`, `false_quiet_rate`, `false_alarm_rate`, `total_cost`,
  `false_quiet_cost`, `false_alarm_cost`, `num_candidates`) with the rule
  name recorded;
- all 36 candidate seed records, each with its eligibility verdict (and its
  explicit reason when ineligible);
- raw rows per (seed, condition, arm) — one row per arm per condition per
  eligible seed — each carrying `seed`, `condition`, `arm` (the module's
  frozen outcome vocabulary), the full `ArmOutcome` scalars — `correct`,
  `action`, `routed_index`, `discovery_invocations`, `admitted`,
  `map_misfit`, `detail`, `initial_library_size`, `final_library_size` — and
  the row's `wall_time_seconds` (descriptive). Fields undefined for the row's
  action are `null`;
- the descriptive block: discovery-invocation counts per condition per arm,
  wall-time summaries per arm (and per condition), the per-condition per-arm
  accuracies, the calibration record, and the **three-point frontier
  comparison** against the published numbers of docs/24 and docs/27 — the
  loop-v2 and loop-v3 arch arms' per-condition accuracies, their discovery
  invocations (61/108 and 86/108), their bounded-harm estimates (−1/36 and
  −14/36), and loop-v3's calibration record — quoted as the documented
  baselines of §1. All of it reported, never claim-tested;
- the four claim summaries, each with estimate, sample standard deviation,
  SE, one-sided lower bound, critical value, direction, threshold, and
  `supported` (or `supported: null` for all four on any non-`complete`
  status); and
- an audit block recomputing **every claim statistic and every correctness
  decision from the raw rows alone**: row counts (the `4 x 3 x n`
  invariant), eligible-seed count, the correctness-gate recompute per the
  frozen semantics of §6, with each recomputed bit required equal to the
  recorded `correct`, plus the `ArmOutcome` cross-field coherence contract
  revalidated from the retained fields; then the paired differences,
  condition means, means, standard deviations, SEs, bounds, and decisions of
  §7–§8, and the descriptive recomputes.

**The audit's scope, stated plainly.** (a) Every claim statistic, every
correctness decision, and every coherence check is recomputable from the raw
rows alone — **no matrices and no models are needed**. (b) What the audit
does NOT re-derive is the arm DECISIONS behind those scalars — the router's
argmax, the alarm's verdict, the generic model's class: those are reproducible
only with the retained trained models and the recorded calibrated threshold,
which are pinned by the frozen torch seeds, the train block, the frozen
calibration rule, the code manifest, the recorded state SHA-256s, and the
recorded calibration record; the rows' determinism given the models and the
threshold is enforced by the double-execution bit-identity check of §4. (c)
As in loop v1, v2, and v3, the audit does not re-verify the discovery head's
internal SVD certificates from retained matrices (none are retained).

An independent validator must recompute all summaries solely from the
retained raw rows and fail closed on missing rows, duplicate (seed,
condition, arm) keys, condition or arm imbalance, ineligible seeds carrying
rows, `correct` fields inconsistent with the frozen gate recompute,
incoherent (action, routed_index, admitted, map_misfit, library-size)
combinations, a calibration record that is missing or inconsistent with its
frozen eight-field schema, wrong t constants, or any decision inconsistent
with §8.

## 12. The two-commit seal procedure

The following order is mandatory:

1. **Commit A (design commit):** this protocol, the runner
   `scripts/run_loop_v4_sealed_1.py`, all its tests, and the additive
   `calibrate_threshold_cost_aware` rule with its unit tests — committed
   without instantiating any declared eval or train seed. Tests use the
   documented loop fixture seeds (`70001..70005`) and the learned-model
   fixture blocks (`70501..70540`, `70601..70620`) or hand-built fixtures
   only — runner tests monkeypatch the runner's frozen seed constants to such
   seeds — and must cover, at minimum: the `"router-v2-observe"` and
   `"loop-v2-operating"` subseed derivations and the shared-draw schedule, no
   sealed-seed fixture use (the eval block, the reserved train block, and
   every other sealed block), the equal-K paired-view construction and its
   fail-closed condition/view validation, all four arms on hand fixtures (the
   arch arm exercised through `arm_arch_full_v2` with an explicitly supplied
   threshold), the `alarm_features_v2` additive layout and the five margin
   columns, **the frozen cost-aware calibration rule — its cost-minimizing
   selection, its tie-breaks, the equal-cost identity
   `FQ + FA = 2 − 2 * balanced_accuracy`, its divergence from the bounded
   rule on a constructed inseparable case, its behavior under asymmetric
   costs, and its fail-closed cases including both-zero costs** — and the
   calibrated-threshold decision including the at-threshold-decides-fit case,
   the frozen correctness gate per (condition, arm) including both
   null-control refusal shapes, **the errata-1 train-side build exclusion —
   a non-instance recorded and skipped, the `MAX_TRAIN_EXCLUDED` ceiling
   and the empty-block guard both fail-closed, a feature-construction
   failure still fatal, and the train-block record present in the training
   provenance** — all four claim decisions including the `SE = 0`
   mechanical case and the H3 tie-fails case, the hard-coded t table,
   eligibility and insufficient-eligible handling (no fits, no
   calibration), training determinism under the frozen torch seeds (including
   the calibrated threshold's determinism), failure-artifact shape
   (`supported: null`), seal parsing and validation including the declared
   train seed block, dirty-worktree refusal, and output-exists refusal. The
   runner embeds this protocol's SHA-256 as a frozen constant — initially the
   fail-closed placeholder `PENDING_PROTOCOL_SHA256`, replaced by the real
   hash before commit A — and never embeds its own hash: it computes its
   runtime SHA-256 and requires equality with the seal's `runner_sha256`.
2. **Compute fingerprints at commit A:** the protocol SHA-256, the runner
   SHA-256, and the per-file SHA-256 of every `src/universa/*.py` (the
   19-file code manifest of §2).
3. **Commit B (seal):** the machine-readable seal record
   `docs/31-router-loop-v4-seal.json` with frozen contents:
   - `schema`: the literal tag `universa-seal/10` — the `universa-seal/9`
     envelope carried forward with exactly two declared differences: the new
     seed blocks and the loop-v4 claim family, whose claim ids carry the
     `loopv4` marker while the four `statistic` keys, the bound directions,
     and the thresholds are unchanged (the family is intentionally identical
     in shape, §8);
   - `design_commit`: the full hash of commit A;
   - `protocol_sha256`, `runner_sha256`;
   - `code_manifest`: the 19 per-file SHA-256 pairs of §2;
   - `train_seed_block`: `{first: 220001, last: 220400}`;
   - `eval_seed_block`: `{first: 230001, last: 230036}`;
   - `no_preview_declaration`: the renewed attestation of §3, covering BOTH
     blocks, including the void-block rule;
   - `primary_family`: the four claims of §8 verbatim, as four claim objects
     with the eight subfields `id`, `statistic`, `theta`, `null`,
     `alternative`, `bound_direction`, `threshold`, `support_rule` — the
     `statistic` values `arch_vs_generic_e2e`, `arch_vs_routing_only_e2e`,
     `arch_vs_generic_inlibrary`, `arch_vs_discovery_only_inlibrary_harm`,
     `bound_direction` `greater` for all four, `threshold` `0.0` for the
     first three and `-0.05` for the bounded-harm claim;
   - `stop_rules`: every stop rule of §9; and
   - `output_path`: exactly `results/experiments/router-loop-v4-sealed-1.json`.
4. **Push commit B to the private remote before any sealed seed is
   instantiated**, and verify the remote contains it.
5. Confirm a clean worktree and the pushed seal, and only then execute the
   canonical command once:

   ```bash
   env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src python \
     scripts/run_loop_v4_sealed_1.py \
     --output results/experiments/router-loop-v4-sealed-1.json
   ```

   `--seal` defaults to `docs/31-router-loop-v4-seal.json` and therefore does
   not appear; no hash is ever passed on the command line.
6. Independently validate the retained raw rows (§11) before editing any
   result prose, then commit the immutable result artifact.

The runner accepts a `--seal` argument defaulting to that path. It parses and
validates the machine-readable seal file rather than trusting a naked
command-line hash; there is no `--expected-runner-sha256`-style flag.
Validation requires the schema tag and every field above with no unknown keys
at any level (including inside each claim object), each claim's `id`,
`statistic`, `threshold`, and `bound_direction` (and `role` where present)
value-checked against the runner's frozen claim definitions and the
descriptive subfields required present as nonempty strings, `design_commit`
present in the repository's history (`git cat-file -e <design_commit>^{commit}`)
AND an ancestor of HEAD (`git merge-base --is-ancestor <design_commit> HEAD`),
the imported `universa` package resolving under the project's `src`, equality
of each embedded hash with both the runner's frozen constants and the actual
file bytes, equality of the runner's own runtime SHA-256 with
`runner_sha256`, equality of every `code_manifest` hash with the actual file
bytes, the seal file's presence in the HEAD commit and byte-identity of the
on-disk seal with its HEAD blob, no symlink at the seal, protocol, or runner
path, a clean worktree per `git status --porcelain --untracked-files=all`,
and equality of `output_path` with the `--output` argument. Only after all of
this — and after setting and verifying one PyTorch thread and a hidden/absent
CUDA — may the runner attempt the 36 eval instance builds, apply eligibility,
and (only if at least 30 seeds are eligible) build the train rows, train the
three models in the pinned order of §5, calibrate the alarm's threshold on
the train block by the frozen cost-aware rule, execute the three conditions x
four arms for every eligible seed, verify determinism bit-identically,
aggregate, and write the result. Before publishing a `complete` result it
re-checks the clean worktree, the code-manifest hashes, and the on-disk seal,
protocol, and runner files, and records the post-run status. It records
`design_commit` (commit A) and HEAD at execution time as the execution
revision.
