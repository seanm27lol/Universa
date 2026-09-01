# Frozen-design protocol: sealed evaluation of the degraded-regime route-or-discover loop — learned router, learned alarm, certified discovery — on the graph-quotient family

Status: **complete and freeze-ready, but not yet sealed or executable**. This
document becomes immutable only when the runner, tests, and design-seal record
described in §12 have been committed. Until then, the declared eval seeds must
not be instantiated. After sealing, any substantive change creates a new
experiment with a new untouched seed block; it must not be called
router-loop-v2-sealed-1.

Protocol date: 2026-09-01.
Experiment id: `universa-loop-v2-sealed-1`.

## 1. Purpose and precise claim scope

This experiment is an untouched-seed evaluation of the Universa
degraded-regime route-or-discover loop (`universa.loop_v2`: `arm_arch_full`,
`arm_routing_only`, `arm_discovery_only`, `arm_generic`, `LearnedAlarm`,
`GenericMLP`, `train_alarm`, `train_generic`, the arch and generic feature
builders, `router_gates`, `router_argmax`, `generic_decision`,
`alarm_decision`, `operating_grid_point`, `ArmOutcome`) on the graph-quotient
switch-instance family of `universa.generators` / `universa.budgets` **only**.
It is the degraded-regime counterpart of the sealed loop v1
(`docs/19-sealed-router-loop-protocol.md`, results
`docs/21-router-loop-sealed-1-results.md`): v1 closed the route-or-discover
cycle in the CLEAN regime, where every score the loop reads is an exact
certified commutation residual and the fit/no-fit alarm (`best residual <=
1e-9`) is a certified decision. Here the loop closes in the **no-anchor
degraded regime** of the sealed router v2 (`docs/04-sealed-router-v2-protocol.md`,
results `docs/06-router-v2-sealed-1-results.md`), lifted into the loop by
`universa.loop_v2`: no residual the loop reads is exact, so **routing and the
fit/no-fit alarm must be learned**, while discovery remains certified.

**The regime.** Every candidate boundary is observed through an
`universa.partial.ObservationModel` with BOTH `mask_fraction = 0.25` (edge
columns removed; `universa.router_v2.DEFAULT_MASK_FRACTION`) AND sign
corruption swept over the profile grid `0.2, 0.3, ..., 0.9` (8 points,
`universa.router_v2.NO_ANCHOR_GRID`) that EXCLUDES `0.0`: there is no clean
column anywhere, so the v1 certified alarm cannot separate and the fit/no-fit
decision is a learned component. **The degradation applies to the observed
structures — the boundary operators the models see — never to the transported
data** (declared caveat 4 below): the vector observations `y = f1 a` are exact
under every regime (the planted chain map commutes with the true boundary in
exact integer arithmetic whatever the model sees), so the certified discovery
head of `universa.discovery` works UNCHANGED in this regime — discovery always
runs on the exact transported observations at the frozen gates (certification
`1e-10` = `universa.operators.CERT_TOL`, novelty `1e-6` =
`universa.discovery.DEFAULT_NOVELTY_TOL`, router acceptance `map_misfit <=
1e-9` = `universa.loop_v2.MAP_ACCEPT_TOL`, re-pinned at
`universa.router.RESIDUAL_TOL`). Certified discovery is therefore unchanged in
this regime; what changes is that the loop must LEARN when to trust the
library and when to invoke it.

**The four arms.** One row is one (seed, condition) pair; the four arms run
paired on the same instance, the same library view, and the same observation
draws (§4):

* **arch-full** — the full system. Candidates are scored by a TRAINED
  `StructureRouter` over the 18-dimensional no-anchor degradation-profile
  feature blocks (hard argmax); a TRAINED `LearnedAlarm` decides fit/no-fit
  from the router's soft gates (`tau = 1.0`) and the raw profile block. Fit
  -> route to the router's argmax. No-fit -> certified discovery on the EXACT
  transported observations, gated through admission; on admission the row
  routes to the appended structure, else it is refused.
* **routing-only** — ablation with NO alarm and NO discovery: the trained
  router's forced argmax, always.
* **discovery-only** — ablation that ALWAYS runs certified discovery on the
  exact transported observations; the library is unused.
* **generic-mlp** — the NO-ARCHITECTURE arm: a `GenericMLP` over the 18-dim
  generic spectral features; classes `0..K-1` route to that candidate, class
  `K` is "no-fit" and synthesizes nothing. There is no certified synthesis
  channel anywhere in this arm.

**Declared caveats — part of the frozen design, stated before any data.**

1. **The generic arm is deliberately architecture-free.** Its features use NO
   commutation residual and NO degradation profile (provably — the builder
   never receives the chain map and never calls the misfit machinery, asserted
   structurally and behaviorally in `tests/test_loop_v2.py`); it has no alarm
   logic and no certified synthesis channel. Its out-of-library failure is
   therefore STRUCTURAL BY CONSTRUCTION — nothing certified is ever acquired
   by that arm — not an empirical finding. The informative comparisons are the
   in-library and null-control conditions plus the two ablations
   (routing-only, discovery-only); the out-of-library leg of the
   arch-vs-generic margins is a design constant, and h1/h3 are read with that
   declared.
2. **h4 is a bounded-harm claim, not a superiority claim.** Always-discovering
   is expected to be near-perfect in-library (the certified head on exact
   data, exactly the sealed exp4 pipeline). The claim is that the loop's
   SELECTIVITY — routing when the alarm says the library fits, discovering
   only when it says otherwise — costs no accuracy against always-discovering
   beyond a frozen 0.05 margin. The price of selectivity is reported
   descriptively: discovery-invocation counts per condition per arm and wall
   times per arm, never claim-tested.
3. **Unlike loop v1's exact certified alarm, the loop-v2 alarm is a LEARNED
   component** — a `LearnedAlarm` over the trained router's soft gates plus
   profile summary statistics, trained on the declared train block inside the
   sealed run. Its precision is MEASURED, not assumed: nothing in this design
   certifies that the alarm fires exactly when the library cannot explain the
   instance, and the claims are constructed so that alarm errors show up as
   correctness losses (a false "fit" out-of-library strands the row on a
   decoy; a false "no-fit" in-library sends the row down the discovery path
   and forfeits the route credit).
4. **Degradation applies to the observed structures, never to the transported
   data** — restated from the regime paragraph because it is load-bearing:
   every certified quantity the discovery path reads is computed from the
   exact `Y`, so the discovery gate semantics (`map_misfit <= 1e-9`,
   certification `1e-10`, novelty `1e-6`) mean exactly what they meant in the
   clean-regime experiments. Only the routing and alarm layers read degraded
   inputs.

**What this experiment cannot establish.** The claims are claims about the
frozen pipeline — instance family, feature builders, three trained models with
frozen seeds, four arms, frozen gates — over the declared seed blocks, and
nothing else. In particular it cannot establish: anything about the other
Universa families (2-complexes, cellular sheaves, category nerves) or transfer
across families; anything about the CLEAN regime (loop v1's territory, sealed
separately and not evidence here, as this result is not evidence for it);
anything about other observation regimes, mask fractions, or grids; anything
about other training seeds, architectures, or hyperparameters; or anything
about real data, real graphs, or any deployed system. Given the seed blocks
and the frozen torch seeds the whole pipeline is deterministic, so intervals
quantify variation over the seed mechanism, not sampling noise of the
procedure.

All prior numbers — the documented behavior of `universa.loop_v2` on its
fixture seeds, the router-v2 sealed result, the loop-v1 sealed result, the
README figures, and any engineering check — are sanity evidence only. They are
not evidence for or against any claim here; only the sealed run decides the
four claims of §8.

## 2. Immutable inputs and hashes

Known immutable inputs at protocol drafting are the Universa code manifest —
one SHA-256 per file of `src/universa/*.py` at the design commit — the
runner, and this protocol. No fingerprint is available at draft time: the
design commit does not exist yet. Every entry below is the literal
placeholder **`PENDING-DESIGN-SEAL`**, to be replaced by actual 64-hex
fingerprints in the machine-readable seal record of §12 (commit B), computed
at the design commit (commit A). The runner refuses to start while any
placeholder remains.

The manifest has 19 entries: the 18 files sealed in the loop-v1 experiment
plus `loop_v2.py`, the degraded-regime loop added for this experiment series.
The prior seals and their manifests remain the frozen records of those
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
| degraded-regime loop (learned router/alarm arms) | `src/universa/loop_v2.py` | `PENDING-DESIGN-SEAL` |
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
| sealed runner | `scripts/run_loop_v2_sealed_1.py` | `PENDING-DESIGN-SEAL` |
| this protocol | `docs/22-sealed-router-loop-v2-protocol.md` | recorded externally in the seal record |

This protocol's own SHA-256 cannot appear inside itself — a file cannot
contain its own hash — so it is recorded externally in the seal record
(`protocol_sha256`, §12) and embedded as a frozen constant in the runner.

The runner must fail before any instance construction unless every fingerprint
above matches. It must also require this execution environment, recorded in
full in the result:

| component | required value |
|---|---|
| exact mathematics | numpy `float64` throughout the certified machinery (features, residuals, gates); torch tensors carry only the learned models' float32 parameters and feature blocks |
| tensor device and dtype | CPU, `torch.float32`, for the three learned models only |
| PyTorch threads | exactly `1` |
| CUDA | unavailable or hidden (`torch.cuda.is_available()` is `False`) |

The runner records the Python, PyTorch, and NumPy versions, the operating
system and machine, `CUDA_VISIBLE_DEVICES`, the thread setting, the Git
revision, the initial `git status --porcelain --untracked-files=all` (which
must be empty), every verified hash, and the canonical command. No dependency
may be installed or upgraded between the seal and the run.

## 3. Seed blocks and the no-preview declaration

**Train block (second, replacement — declared, unsealed, reserved):** the 200
consecutive generator seeds

```text
170001..170200
```

This block trains the three learned models of §5, inside the sealed run, after
the seal. It is unsealed in the series' sense — the confirmatory claims rest
solely on the untouched eval block below — but it is reserved against any
preview until the canonical run consumes it: at the time this amendment was
drafted it was absent from the working tree and from all Git history
(verified by a repository-wide word-boundary search over every commit), its
only occurrences anywhere being the inert declarations of this protocol and
the reserved-block guard of `tests/test_loop_v2.py`, and no project run had
instantiated or previewed any member **on any generator family or in any
pipeline stage**. The block is disjoint from every train and eval block and
every demo or fixture seed previously declared in this series.

**Train-block errata, declared in full.** The first declared train block,
`150001..150200`, was **voided pre-seal**: the runner's test suite, written
under an instruction that predated this section's no-preview rule for train
blocks, instantiated the prefix `150001..150010` in a module-scoped training
fixture before any commit was made. The pre-seal consistency audit caught it
before commit A — no seal existed, no run happened, no claims were computed.
Per the no-preview rule of this section the block is void; it is replaced by
`170001..170200`, verified absent as above. Tests now instantiate only the
sanctioned fixture seeds of this section (`70001..70005`, `70501..70540`,
`70601..70620`) or hand-built fixtures; the voided block is never used in
this experiment.

**Sealed eval block:** the 36 consecutive generator seeds

```text
160001, 160002, 160003, 160004, 160005, 160006, 160007, 160008, 160009,
160010, 160011, 160012, 160013, 160014, 160015, 160016, 160017, 160018,
160019, 160020, 160021, 160022, 160023, 160024, 160025, 160026, 160027,
160028, 160029, 160030, 160031, 160032, 160033, 160034, 160035, 160036
```

At the time this protocol was drafted, this block was absent from the working
tree and from all Git history (verified by a repository-wide word-boundary
search over every commit; its only occurrences anywhere are the inert
declarations of this protocol and the reserved-block guard of
`tests/test_loop_v2.py`), and no project run had instantiated or previewed
any member **on any generator family or in any pipeline stage**. In
particular, no one has built an instance, computed a residual, drawn an
observation, trained or evaluated a model, printed a dimension, smoke-tested,
or debugged with any seed in the block. The block is disjoint from every
train and eval block and every demo or fixture seed previously declared in
this series.

**Block history, declared in full.** The train-block errata above
(`150001..150200` voided pre-seal by test-side instantiation, replaced by
`170001..170200`). The series' prior voided blocks (`130001..130036`, void
for claims after the loop-v1 design failure; `70101..70136`, discarded
unopened) remain void and untouched, and are not used here.

The no-preview rule remains in force until commit B of §12 — the
machine-readable seal record — is committed **and pushed to the private
remote**. Until then:

- tests may use the documented loop fixture seeds (`70001..70005`) and the
  learned-model fixture blocks of `tests/test_loop_v2.py` and the runner's
  tests (`70501..70540` train fixtures, `70601..70620` validation
  fixtures), or hand-built
  fixtures, only — runner tests monkeypatch the runner's frozen seed
  constants to such seeds — and may not parameterize over, import, derive
  data from, or otherwise touch the sealed eval block, the reserved train
  block, or any other sealed block (`20101..20136`, `30101..30136`,
  `40101..40136`, `60101..60136`, `70101..70136`, `80101..80136`,
  `90101..90136`, `130001..130036`, `140001..140036`);
- source-code constants containing the declared integers (e.g. the runner's
  frozen seed-list constants) are permitted;
- merely calling `make_budget_instance`, `make_loop_instance`,
  `make_switch_instance`, or any `universa.loop_v2` arm or feature builder
  with a declared eval or train seed counts as a preview, even if the result
  is discarded.

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
`num_classes=6` classes (the budgets choice, keeping the quotient
non-complete so decoys have genuinely different kernels), and 3 certified
discriminating decoys, all exact integer arithmetic, float64, with a planted
transported quantity and a certified identifiability threshold. This is the
frozen loop instance family: `universa.loop.make_loop_instance(s)` returns
this same instance with its two library views.

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
the `LearnedAlarm` reads K gates + 3 summary statistics, and the `GenericMLP`
emits K + 1 classes — so the paired in-/out-of-library rows must share one
candidate count, which forces the in-library view to drop one decoy (the
out-of-library view cannot contain the truth). Each arm validates the view
against the declared condition fail-closed (`in_library` requires the truth
at index 0 followed by a prefix of the decoys; `out_of_library` and
`null_control` require exactly `instance.decoy_targets`).

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
the generic arm's rows across degradation levels while keeping every arm of
a row paired on the same instance, view, and observation draw.

**The exact transported observations.** For the discovery path and the
generic arm's spectral features, the exact observation matrix is the frozen
exp4 schedule of 16 columns over the instance's `SwitchInstance` view (the
same underlying arrays; the budget instance only adds the planted quantity
and threshold, which the schedule does not read):

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
degradation never touches the data (caveat 4).

**The null-control observations.** The specificity control replaces `Y` with
structure-free noise, the frozen exp4 H4 / loop-v1 schedule
(`universa.loop.null_observations`): column `j` is

```text
np.random.default_rng(subseed(s, "discovery-null", str(j)))
      .standard_normal(ambient_dim)        for j = 0..15
```

with `ambient_dim` the number of edges of the seed's true quotient target;
`Y_null` is the column stack (`ambient_dim x 16`, float64). The null matrix
is consulted only where an arm reads observations: the discovery path
(arch-full on no-fit; discovery-only always) and the generic arm's spectral
features. The arch and routing arms' feature blocks — built from the
degraded candidate boundaries under the shared observation draw — are
identical in the out-of-library and null-control conditions per seed.

**The three conditions per seed, in build order.** Exactly three conditions,
all deterministic end-to-end, applied to every eligible seed:

1. **in-library** — view `in_library` (the truth at index 0), observations
   `Y`. The expected arch-full behavior is alarm-fit and route to index 0.
2. **out-of-library** — view `out_library` (the truth withheld),
   observations `Y`. The expected arch-full behavior is alarm-no-fit,
   certified discovery, admission, and re-route to the appended structure.
3. **null-control** — view `out_library`, observations `Y_null`. The
   expected behavior is that nothing is admitted by any arm with a discovery
   channel, and refusal from the arms whose only specificity mechanism is
   refusal.

**The four arms per condition, paired.** Each condition of each eligible
seed runs all four arms, in the module's `ARMS` order (`arch_full`,
`routing_only`, `discovery_only`, `generic`), all sharing the same
observation draws — the same `observation_seed(s)`, the same operating grid
point, and the same observation matrix (`Y` for in-library and
out-of-library, `Y_null` for null-control):

```text
arm_arch_full(s, instance, view, router=router, alarm=alarm,
              observation_seed=observation_seed(s), observations=Y_or_null,
              condition=condition)
arm_routing_only(s, instance, view, router=router,
                 observation_seed=observation_seed(s), condition=condition)
arm_discovery_only(s, instance, observations=Y_or_null, condition=condition)
arm_generic(s, instance, view, generic_model=generic,
            observation_seed=observation_seed(s), observations_y=Y_or_null,
            condition=condition)
```

Every defaulted argument stays at its module default — the profile grid
`NO_ANCHOR_GRID`, `mask_fraction = 0.25`, `novelty_tol = 1e-6`
(`universa.discovery.DEFAULT_NOVELTY_TOL`); the discovery certification gate
is the module-frozen `1e-10` (`universa.operators.CERT_TOL`) and the router
acceptance gate `1e-9` (`universa.loop_v2.MAP_ACCEPT_TOL`). There is no
per-seed or per-condition choice of any kind.

**Deterministic random streams.** Every stochastic component uses
`universa.generators.subseed`:

```text
message = ":".join(["universa", str(seed), *components]).encode("utf-8")
digest  = SHA256(message)
subseed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
```

The disjoint subseed families of one row are `"router-v2-observe"` (the
shared boundary-observation draw, which `ObservationModel` further derives
as `subseed(seed, "observe", "mask" | "corrupt")`), `"loop-v2-operating"`
(the generic arm's operating grid point), `"discovery-observation"` (the
exact transported columns), and `"discovery-null"` (the null columns).

**Determinism, verified.** Given the trained models, every arm is a
deterministic function of the seed, and `ArmOutcome` is a pure scalar record
precisely so that two runs compare bit-identical with plain equality; the
models themselves are deterministic functions of the train block and the
frozen torch seeds (§5). The runner executes the full per-seed eval
pipeline — all three conditions x all four arms — TWICE for every eligible
seed with the same trained models and requires bit-identical raw rows; any
mismatch is a whole-run `design_failure`. Any NaN or non-finite feature,
residual, misfit, loss, or statistic, and any violation inside the certified
machinery (rank, orthonormality, or certificate failure) or inside the
training loops (a non-finite loss), is likewise a whole-run
`design_failure`, never a dropped row.

## 5. Model and training

Exactly three models are trained — inside the sealed runner, after the seal,
using only the declared train block `170001..170200` — and no other learned
parameter exists anywhere in the design. Training happens only AFTER the
eval-block eligibility gate of §10 passes (an insufficient-eligible run fits
nothing, §9).

**The train rows.** Per train seed `s` in `170001..170200`, in seed order,
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

with `Y` the exact 16-column transported matrix of §4. Every train seed is
used as-is; there is no train-side eligibility gate and no train-side
dropping — any train-seed build or feature-construction exception is a
whole-run `design_failure` (the same discipline as the eval builds), never
an exclusion.

**Model 1 — the router.** `StructureRouter(feature_dim=18, hidden_dim=64)`,
the shared per-candidate MLP `18 -> 64 -> 64 -> 1` of `universa.router`,
constructed and trained by `universa.router.train_router` on the 200
in-library arch blocks (`(200, 3, 18)`, label `0` for every row — the loop's
unpermuted index-0 convention; the out-of-library view carries no routing
label and contributes no router rows) with:

- `epochs=150` (full-batch Adam: every epoch is one step, no shuffling RNG);
- `lr=1e-3`, PyTorch `torch.optim.Adam` defaults for every unspecified
  option;
- torch seed `4242` (`torch.manual_seed(4242)`, governing initialization);
- `lambda_aux=0.01` (Switch-style load-balancing pressure);
- temperature annealed exponentially `tau: 2.0 -> 0.25`
  (`anneal_temperature`: exactly `2.0` at epoch 0, exactly `0.25` from epoch
  149 on, monotone non-increasing);
- CPU, `torch.float32`, exactly one PyTorch thread;
- input standardization statistics (mean and population std, clamped at
  `1e-8`) measured on the **train rows only** and stored in the model's
  buffers, so the model is self-contained at inference.

The validation argument of `train_router` is the training block itself. It
feeds only the `val_accuracy` history diagnostic — no early stopping, no
model selection, no gradient — and therefore cannot influence the fitted
parameters; this choice uses only train seeds.

**Model 2 — the learned alarm.** `LearnedAlarm(num_candidates=3)` —
`Linear(6, 32) GELU Linear(32, 1)` — trained by
`universa.loop_v2.train_alarm` on the alarm-feature rows of BOTH views:

```text
rows_fit   = stack(alarm_features(router_gates(router, block_in[s]),
                                  raw_in[s])   for s in train block)  # (200, 6)
rows_nofit = stack(alarm_features(router_gates(router, block_out[s]),
                                  raw_out[s])  for s in train block)  # (200, 6)
```

— in-library rows labeled fit (1), out-of-library rows no-fit (0) — with
`epochs=150`, `lr=1e-3`, torch seed `4243` (`torch.manual_seed(4243)` runs
inside `train_alarm` immediately before the factory constructs the model),
binary cross-entropy on the logit, full-batch Adam, and train-only
standardization stored in the model's buffers. The decision threshold is the
frozen `ALARM_THRESHOLD = 0.5` on the sigmoid probability, never tuned.

**Model 3 — the generic model.** `GenericMLP(num_candidates=3)` — the
DeepSets head `Linear(18, 64) GELU Linear(64, 64) GELU` per candidate, a
mean over the candidate axis, then `Linear(64, 4)` — trained by
`universa.loop_v2.train_generic` on the 400 generic rows (the 200 in-library
blocks labeled `0`, the 200 out-of-library blocks labeled `3` = K, the
no-fit class) with `epochs=150`, `lr=1e-3`, torch seed `4244`
(`torch.manual_seed(4244)` runs immediately before `GenericMLP(3)` is
constructed AND is passed as `train_generic`'s `seed`), cross-entropy,
full-batch Adam, and train-only standardization stored in the buffers.

**Training order is pinned** — router (4242), then alarm (4243), then
generic (4244) — because the alarm's features read the trained router's soft
gates. A non-finite loss at any epoch of any model is an error, never a
warning, and a whole-run `design_failure`.

**Declared consequences of the index-0 convention, stated before any data.**
The loop never permutes candidates: the truth sits at index 0 of the
in-library view and the views are validated against exactly that. Two
consequences follow mechanically and are declared, not discovered after the
fact. (a) The router's training rows all carry label `0`, so nothing in the
router's training signal distinguishes the candidates' positions — the
router's learned role in this design is profile-shape scoring under the
index-0 commitment convention, and the loop's learned SELECTIVITY lives in
the alarm, not the argmax. (b) The generic arm's index classes degenerate
toward "route to 0" for the same reason (its own docstring declares this);
its real task is the fit/no-fit separation from spectral statistics alone.
The in-library claims are declared against exactly this construction: h3 is
a strict-margin claim that a tie FAILS (§8), and h4 is a bounded-harm claim
against the always-discover ablation, not a routing-quality claim.

Inference is always strictly discrete: the router's hard argmax over
candidate logits (first-index tie-breaking), the alarm's thresholded
sigmoid, the generic model's argmax — never sampled or mixed.

Training provenance recorded in the result, per model: the torch seed, the
epoch count, the learning rate, the frozen objective (including
`lambda_aux`, `tau_start`, `tau_end`, `hidden_dim`, `feature_dim` for the
router; the threshold for the alarm), the row counts, the final epoch's
loss(es) and train accuracy, the optimizer (`full-batch Adam`), the
dtype/device (`CPU float32`), the standardization note, and the model state
SHA-256 — computed over the canonical concatenation, in lexicographic name
order, of each `state_dict` entry's UTF-8 name, a NUL byte, and its
little-endian float32 bytes.

## 6. Arms and conditions

Exactly three conditions per seed, in the frozen build order of §4 —
**in-library**, **out-of-library**, **null-control** — and exactly four arms
per condition, paired on the same observation draws. The arm semantics are
the module's frozen ones:

1. **arch-full** (`universa.loop_v2.arm_arch_full`) — the trained router
   scores every candidate over the 18-dim arch profile blocks (hard argmax);
   the trained alarm reads the soft gates (`tau = 1.0`) and the raw profile
   block. FIT -> route to the router's argmax (discovery never invoked).
   NO-FIT -> certified discovery on the EXACT transported observations at
   the frozen gates (certification `1e-10`, novelty `1e-6`); on admission
   the row routes to the appended structure (discover mode), else it is
   refused.
2. **routing-only** (`arm_routing_only`) — NO alarm, NO discovery: the
   trained router's forced argmax, always a route. On out-of-library rows it
   must pick a decoy — an honest failure by the frozen semantics.
3. **discovery-only** (`arm_discovery_only`) — ALWAYS certified discovery on
   the exact observations, the library unused for routing; novelty is
   checked against the instance's decoy library, exactly the
   `universa.discovery.run_discovery` construction.
4. **generic-mlp** (`arm_generic`) — the NO-ARCHITECTURE arm: the
   `GenericMLP`'s decision over the 18-dim generic spectral blocks at the
   row's operating grid point. Classes `0..2` route to that candidate; class
   `3` ("no-fit") synthesizes nothing and records a refusal. There is no
   certified synthesis channel anywhere in this arm.

**Correctness semantics (frozen; the module's `_condition_correct` gate,**
reading only the outcome record and the index-0 convention, never the
withheld truth):

| arm | in-library correct iff | out-of-library correct iff | null-control correct iff |
|---|---|---|---|
| arch-full | `action == "route"` and `routed_index == 0` | `action == "discover"` and `admitted` and `map_misfit <= 1e-9` | NOT `admitted` |
| routing-only | `action == "route"` and `routed_index == 0` | never (no acquisition channel) | NOT `admitted` (vacuous: it admits nothing) |
| discovery-only | `action == "discover"` and `admitted` and `map_misfit <= 1e-9` | `action == "discover"` and `admitted` and `map_misfit <= 1e-9` | `action == "refused"` |
| generic | `action == "route"` and `routed_index == 0` | never (no certified synthesis channel) | `action == "refused"` |

That is: **in-library** correct means the arm's final structure is the true
target — for the routing arms (arch-full, routing-only, generic) the route
lands on index 0; for the discovery arm a certified novel structure is
admitted with `map_misfit <= 1e-9`. **Out-of-library** correct means a
certified novel structure is admitted with `map_misfit <= 1e-9` — only the
two arms with a certified discovery channel can ever be correct there, by
construction (caveat 1). **Null-control** correct means nothing is admitted:
for the arms with a discovery channel the false-admission control of loop
v1's H4; for the alarm-less/generic arms, whose only specificity mechanism
is refusal, correct iff refused. Note the two refusal shapes, declared: for
arch-full and routing-only a route on null-control rows admits nothing and
is therefore correct — the control penalizes false ADMISSION, not
commitment; for discovery-only and generic only an explicit refusal is
correct, since a discovery-arm "admit" from noise or a generic route is
exactly the specificity failure the condition exists to catch.

There is no fifth arm, no fourth condition, no ensemble, no eval-time
tuning, and no per-seed choice of any kind.

## 7. Estimands and unit of inference

For each eligible seed `s`, condition `c` in `{in_library, out_of_library,
null_control}`, and arm `a` in `{arch_full, routing_only, discovery_only,
generic}`, write `correct(s, c, a)` in `{0, 1}` for the frozen correctness
indicator of §6, read off the row's `ArmOutcome`. The per-seed
condition-mean correctness of an arm is

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

Declared conventions, fixed here:

- **The alarm's decision basis precedes observations.** The arch and routing
  arms' feature blocks — and therefore the router's gates and the alarm's
  verdict — are computed from the degraded candidate boundaries and the
  shared observation draw alone; the observation matrix (exact `Y` or
  `Y_null`) is consulted only by the discovery path and the generic arm's
  spectral features. Per seed, arch-full and routing-only therefore present
  identical feature blocks in the out-of-library and null-control
  conditions, mirroring loop v1's "the alarm precedes observations"
  convention — but here the alarm is LEARNED (caveat 3): its verdicts are
  measured behavior, not certified decisions.
- **The paired design is the point.** All four arms of a (seed, condition)
  share the instance, the view, and the observation draws, so each
  difference `d1..d4` cancels per-seed instance difficulty and degradation
  realization; the margins isolate the arms' machinery differences (alarm +
  certified channel; no channel; discovery always; no architecture).
- **The out-of-library legs of d1 are structural, declared.** Routing-only
  and generic can never acquire out-of-library (caveat 1), so their
  out-of-library indicators are 0 by construction; the informative legs of
  the end-to-end margins are the in-library and null-control conditions.
- **The descriptive quantities are never claim inputs.** Discovery-invocation
  counts per condition per arm, per-arm wall times, and per-condition
  per-arm accuracies are reported (§11) and read by no decision rule.

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
eligibility rule of §10. Because the whole pipeline — training included — is
deterministic given the declared seed blocks and the frozen torch seeds,
intervals quantify Monte Carlo variation over the seed mechanism only —
there is no algorithmic sampling noise to average over — and say nothing
about uncertainty for real data.

## 8. The four frozen claims and decision rules

The following four and only these four claims form the confirmatory family.
H1 is primary; H2–H4 are secondary members of the same family.

| id | scope | statistic | prediction and governing decision |
|---|---|---|---|
| `h1-loopv2-arch-vs-generic-e2e` | condition-mean (all three conditions) | `arch_vs_generic_e2e` = `d1` | mean `> 0`; supported iff the one-sided lower bound exceeds `0` |
| `h2-loopv2-arch-vs-routing-only-e2e` | condition-mean (all three conditions) | `arch_vs_routing_only_e2e` = `d2` | mean `> 0`; supported iff the one-sided lower bound exceeds `0` |
| `h3-loopv2-arch-vs-generic-inlibrary` | in-library | `arch_vs_generic_inlibrary` = `d3` | mean `> 0`; supported iff the one-sided lower bound exceeds `0` (strict; a tie fails) |
| `h4-loopv2-arch-vs-discovery-only-inlibrary-harm` | in-library | `arch_vs_discovery_only_inlibrary_harm` = `d4` | mean `> -0.05`; supported iff the one-sided lower bound exceeds `-0.05` (bounded harm) |

The formal hypotheses are fixed as:

- **H1** (id `h1-loopv2-arch-vs-generic-e2e`, primary): `theta =
  E_seed[d1]`; H0: `theta <= 0`; H_A: `theta > 0`; supported iff the
  one-sided Bonferroni lower bound is above `0`. This is the system's
  end-to-end claim: the full architecture — learned router, learned alarm,
  certified discovery channel — outperforms the no-architecture arm over the
  whole condition mix, with the out-of-library leg of the margin structural
  by construction (caveat 1) and the informative legs in-library and
  null-control.
- **H2** (id `h2-loopv2-arch-vs-routing-only-e2e`): `theta = E_seed[d2]`;
  H0: `theta <= 0`; H_A: `theta > 0`; supported iff the one-sided
  Bonferroni lower bound is above `0`. This isolates the alarm + discovery
  channel: routing-only has the same learned router and no selectivity, so
  the margin measures what the loop's fit/no-fit decision and certified
  synthesis add end-to-end.
- **H3** (id `h3-loopv2-arch-vs-generic-inlibrary`): `theta = E_seed[d3]`;
  H0: `theta <= 0`; H_A: `theta > 0`; supported iff the one-sided
  Bonferroni lower bound is above `0`. A STRICT margin: a tie fails the
  claim and is reported as frozen. Given the declared degeneration of both
  arms' routing layers toward the index-0 convention (§5), a tie is a
  genuinely possible frozen outcome; the mechanical rule below decides it
  (with `SE = 0` and mean difference exactly `0`, the lower bound equals
  `0`, which does not exceed `0` — the claim fails), and the outcome is
  reported either way.
- **H4** (id `h4-loopv2-arch-vs-discovery-only-inlibrary-harm`): `theta =
  E_seed[d4]`; H0: `theta <= -0.05`; H_A: `theta > -0.05`; supported iff
  the one-sided Bonferroni lower bound is above `-0.05`. A BOUNDED-HARM
  claim, not superiority (caveat 2): the loop's selectivity — routing when
  the alarm says fit instead of always discovering — is claimed to cost no
  accuracy against always-discovering beyond the frozen `0.05` margin.
  Discovery-invocation counts and wall times are reported descriptively and
  play no role in the decision.

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
H1–H3, `-0.05` for H4). This case is specified here in advance and is
decided by the same frozen rule, not by discretion.

The hard-coded one-sided critical values, computed once with SciPy for
design only, are:

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
claim summaries records: the estimate (mean of the per-seed statistic); its
sample standard deviation; its standard error; the governing one-sided lower
bound; the critical value; the direction (`greater`); the threshold (`0` for
H1–H3, `-0.05` for H4); and the support decision. No hypothesis may be
added, dropped, reversed, re-thresholded, or moved between families after
execution. All nulls and failed predictions are reported.

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
- an instance-build exception — eval block or train block — is a
  **whole-run** `design_failure`, never an exclusion — the offending seed is
  recorded, the run stops, and no seed is ever deleted alone;
- the eval build and the undegraded audit of §10 are the sole eligibility
  gate, and it is evaluated BEFORE any training. Every *other* failed check
  — any determinism (including the double-execution bit-identity check),
  NaN/non-finite value, audit (other than the per-seed eligibility gate),
  rank, orthonormality, certificate, or validation failure, any violation
  inside the certified discovery machinery or the training loops (a
  non-finite loss), an incoherent `ArmOutcome`, and any failed result-audit
  recomputation — is a whole-run `design_failure`;
- fewer than 30 eligible seeds yields `design_failure_insufficient_eligible`
  with **no fits, no evaluation, and no claims**: the run stops before any
  model is trained. Under the frozen build rule a build problem stops the
  run as `design_failure` before eligibility accounting, so this status is
  the fail-closed guard over the eligibility count itself;
- a caught failure preserves every completed raw row, identifies the failing
  seed and stage, records the exception type and message and the preflight
  provenance, and emits all four claims with `supported: null`;
- there is no outcome-dependent stopping, no interim analysis, no
  sample-size extension, and no condition-, arm-, or seed-level deletion;
- the run is never rerun because the results are surprising, weak, null, or
  apparently decisive;
- the runner must not compute or print scores, accuracies, margins, or claim
  summaries until every required raw row has completed; progress output is
  limited to seed identifiers, stage, completion state, and timing.

Mandatory stop conditions (any one stops the run before or during
execution):

- any preflight mismatch: seal-validation failure (including an unknown seal
  key at any level including inside a claim object, a tampered claim field,
  a `train_seed_block` other than `{first: 170001, last: 170200}`, an
  `eval_seed_block` other than `{first: 160001, last: 160036}`, a
  `design_commit` absent from the repository's history or not an ancestor of
  HEAD, or the imported `universa` package not resolving under the project's
  `src`), hash mismatch, a seal not committed at HEAD, an on-disk seal not
  byte-identical to its HEAD blob, a symlinked seal, protocol, or runner
  file, a dirty worktree, an existing output path, available CUDA, or a
  torch thread count other than one;
- a worktree found dirty, or a code-manifest hash, the on-disk seal,
  protocol, or runner file found changed, at the end-of-run re-check before
  publishing a `complete` result;
- a raw-row count that violates the frozen invariant `4 x 3 x n` — exactly
  four arm rows per condition and three conditions per eligible seed;
- an instance-build exception (eval or train block);
- fewer than 30 eligible seeds;
- any determinism, NaN, audit, rank, training, or validation failure;
- aggregation failure or a non-finite estimand; or
- any manual cancellation or infrastructure interruption.

If infrastructure fails after execution begins, the attempt is preserved
under a distinct non-overwriting failure path
`results/experiments/failures/router-loop-v2-sealed-1.<status>.json`; the
canonical output path is reserved for status `complete` and a failed
attempt never occupies it. An exact-code retry is allowed
only for a documented non-scientific failure, before any summaries are
inspected, with identical hashes. Every attempt remains in the evidence
bundle. The first complete, fully validated attempt is canonical; no choice
among complete attempts is allowed. Any code, tolerance, dependency, or
analysis change after a partial run invalidates the seed blocks and requires
a new protocol and new seeds.

## 10. Eligibility

An eval seed is **eligible** if and only if both hold:

1. its instance builds — `make_budget_instance(seed, 8, 14, 6, 3)` returns
   (a build exception is a whole-run `design_failure` per §9, never an
   exclusion); and
2. the **undegraded audit** passes: computed on the undegraded instance
   without any observation draw, the true candidate's certified commutation
   score — the max residual over degrees of the planted-map `ChainMap`
   probe; one degree in this family — is exactly `0.0` (and therefore
   `<= 1e-9`), and every decoy's score is strictly `> 1e-9`.

The audit is **bookkeeping only, never a feature**: no arm, model, alarm,
router, or generic feature ever reads it — the learned layers see only the
degraded-observation feature blocks and the exact (or null) observation
matrix; the discovery path reads only the exact observations. The audit
certifies the library's make-up per seed, which is what licenses reading a
route to index 0 as correct (the frozen correctness gate of §6) and what
licenses the no-permutation stance of §5. It is observation-independent and
therefore a deterministic per-seed property, identical across the three
conditions and the four arms.

Ineligible seeds are recorded with their explicit reasons and are never
replaced, never deleted, and never scored. `n`, the number of eligible
seeds, is the inference sample size of §7–§8. If `n < 30`, the run stops
before any training and any loop execution with
`design_failure_insufficient_eligible` and no claims. All 36 seeds are
attempted, in seed order. There is no train-side eligibility gate: all 200
train rows must build, and any train-seed build or feature exception is a
whole-run `design_failure`.

## 11. Result JSON schema

The runner writes exactly one result file, to
`results/experiments/router-loop-v2-sealed-1.json`, through a temporary file
followed by an atomic no-clobber hard-link publish (create-if-absent, never
replace — stronger than a rename), and refuses an existing path. Non-
`complete` statuses are published instead to
`results/experiments/failures/router-loop-v2-sealed-1.<status>.json` under
the same no-clobber rule, and failure artifacts are written **only** there.
The JSON contains, at minimum:

- `schema`: the literal tag `universa-router-loop-v2-sealed-result/1`, and
  the frozen configuration (the exact `make_budget_instance` call and sizes,
  the paired-view construction with `K = 3`, the no-anchor grid and mask
  fraction, the observation-seed and operating-point subseed schedule, the
  exact and null observation schedules with `num_observations = 16`, the
  three conditions in build order, the four arms in `ARMS` order, the frozen
  gates — certification `1e-10`, novelty `1e-6`, router acceptance `1e-9`,
  alarm threshold `0.5` — and the claim family with thresholds);
- the seal SHA-256, protocol SHA-256, runner SHA-256, and every
  code-manifest SHA-256, plus `design_commit` (commit A) and the execution
  revision (HEAD at execution time);
- environment and provenance: Python/PyTorch/NumPy versions, OS and machine,
  thread count, the operator-provided and effective `CUDA_VISIBLE_DEVICES`,
  the recorded `git status --porcelain --untracked-files=all` output (key
  `git_status_porcelain`, must be empty, recorded pre- and post-run), the
  canonical command, and `sys.argv`;
- training provenance per §5 for all three models (torch seeds, epoch
  counts, objectives and hyperparameters, row counts, final losses and train
  accuracies, optimizer, dtype/device, standardization note, and each
  model's state SHA-256);
- all 36 candidate seed records, each with its eligibility verdict (and its
  explicit reason when ineligible);
- raw rows per (seed, condition, arm) — one row per arm per condition per
  eligible seed — each carrying `seed`, `condition` (`in_library` /
  `out_of_library` / `null_control`), `arm` (`arch_full` / `routing_only` /
  `discovery_only` / `generic`), the full `ArmOutcome` scalars — `correct`,
  `action`, `routed_index`, `discovery_invocations`, `admitted`,
  `map_misfit`, `detail`, `initial_library_size`, `final_library_size` —
  and the row's `wall_time_seconds` (descriptive). Fields undefined for the
  row's action (`routed_index` on a refusal; `map_misfit` in route mode and
  on a discovery insufficiency) are `null`;
- the descriptive block: discovery-invocation counts per condition per arm,
  wall-time summaries per arm (and per condition), and the per-condition
  per-arm accuracies (means of the correctness indicators over eligible
  seeds) — reported, never claim-tested;
- the four claim summaries, each with estimate, sample standard deviation,
  SE, one-sided lower bound, critical value, direction, threshold, and
  `supported` (or `supported: null` for all four on any non-`complete`
  status); and
- an audit block recomputing **every claim statistic and every correctness
  decision from the raw rows alone**: row counts (the `4 x 3 x n`
  invariant), eligible-seed count, the correctness-gate recompute per the
  frozen semantics of §6 — for the routing arms (arch-full, routing-only,
  generic) `action == "route"` and `routed_index == 0` in-library; the
  certified-gate semantics `action == "discover"` AND `admitted` AND
  `map_misfit <= 1e-9` from the retained fields for the discovery-channel
  arms; the null-control NOT-`admitted` / `action == "refused"` shapes per
  arm — with each recomputed bit required equal to the recorded `correct`,
  plus the `ArmOutcome` cross-field coherence contract (action vs routed
  index vs admission vs library growth) revalidated from the retained
  fields; then the paired differences, condition means, means, standard
  deviations, SEs, bounds, and decisions of §7–§8, and the descriptive
  recomputes.

**The audit's scope, stated plainly.** (a) Every claim statistic, every
correctness decision, and every coherence check is recomputable from the raw
rows alone — **no matrices and no models are needed**: `ArmOutcome` is a
pure scalar record and every frozen predicate reads only those scalars, so
no matrices are retained and none are required by the audit. (b) What the
audit does NOT re-derive is the arm DECISIONS behind those scalars — the
router's argmax, the alarm's verdict, the generic model's class: those are
reproducible only with the retained trained models, which live in the
runner's process and are pinned by the frozen torch seeds, the train block,
the code manifest, and their recorded state SHA-256s (the result carries the
hashes, not the weights); the rows' determinism given the models is enforced
by the double-execution bit-identity check of §4. (c) As in loop v1, the
audit does not re-verify the discovery head's internal SVD certificates from
retained matrices (none are retained); the per-row certified residuals are
reproducible only by re-running the manifest-pinned deterministic pipeline,
and that assurance is the sealed run's own — pinned by the code manifest and
enforced by the bit-identity check.

An independent validator must recompute all summaries solely from the
retained raw rows and fail closed on missing rows, duplicate (seed,
condition, arm) keys, condition or arm imbalance, ineligible seeds carrying
rows, `correct` fields inconsistent with the frozen gate recompute,
incoherent (action, routed_index, admitted, map_misfit, library-size)
combinations, wrong t constants, or any decision inconsistent with §8.

## 12. The two-commit seal procedure

The following order is mandatory:

1. **Commit A (design commit):** this protocol, the runner
   `scripts/run_loop_v2_sealed_1.py`, and all its tests — committed without
   instantiating any declared eval or train seed. Tests use the documented
   loop fixture seeds (`70001..70005`) and the learned-model fixture blocks
   (`70501..70540`, `70601..70620`) or hand-built fixtures only — runner
   tests monkeypatch the runner's frozen seed constants to such seeds — and
   must cover, at minimum: the `"router-v2-observe"` and
   `"loop-v2-operating"` subseed derivations and the shared-draw schedule
   (one draw per (row, grid point), reused across candidates), no
   sealed-seed fixture use (the eval block, the reserved train block, and
   every other sealed block), the equal-K paired-view construction and its
   fail-closed condition/view validation, all four arms on hand fixtures,
   the frozen correctness gate per (condition, arm) including both
   null-control refusal shapes, all four claim decisions including the
   `SE = 0` mechanical case (lower bound equals the estimate) and the H3
   tie-fails case, the hard-coded t table, eligibility (build plus the
   undegraded audit) and insufficient-eligible handling (no fits), training
   determinism under the frozen torch seeds, failure-artifact shape
   (`supported: null`), seal parsing and validation including the declared
   train seed block, dirty-worktree refusal, and output-exists refusal. The
   runner embeds this protocol's SHA-256 as a frozen constant — initially
   the fail-closed placeholder `PENDING_PROTOCOL_SHA256`, replaced by the
   real hash before commit A — and never embeds its own hash: it computes
   its runtime SHA-256 and requires equality with the seal's
   `runner_sha256`. This avoids a self-hash cycle.
2. **Compute fingerprints at commit A:** the protocol SHA-256, the runner
   SHA-256, and the per-file SHA-256 of every `src/universa/*.py` (the
   19-file code manifest of §2).
3. **Commit B (seal):** the machine-readable seal record
   `docs/23-router-loop-v2-seal.json` with frozen contents:
   - `schema`: the literal tag `universa-seal/8` — the `universa-seal/7`
     envelope carried forward with exactly three declared differences: a
     real (non-sentinel) `train_seed_block` naming this experiment's
     declared unsealed train block, the 19-file code manifest including
     `loop_v2.py`, and the loop-v2 paired-difference claim family whose
     claim objects carry the `statistic` keys below;
   - `design_commit`: the full hash of commit A;
   - `protocol_sha256`, `runner_sha256`;
   - `code_manifest`: the 19 per-file SHA-256 pairs of §2;
   - `train_seed_block`: `{first: 170001, last: 170200}` — the declared,
     unsealed, reserved train block of §3 (the three models train on it
     inside the sealed run);
   - `eval_seed_block`: `{first: 160001, last: 160036}`;
   - `no_preview_declaration`: the renewed attestation of §3, covering BOTH
     blocks (never instantiated, previewed, counted for eligibility,
     smoke-tested, or debugged with, on any generator family or in any
     pipeline stage; verified absent from the working tree and all Git
     history but for the inert declarations; no prior draft blocks for this
     experiment), including the void-block rule;
   - `primary_family`: the four claims of §8 verbatim, as four claim
     objects with the eight subfields `id`, `statistic`, `theta`, `null`,
     `alternative`, `bound_direction`, `threshold`, `support_rule` — the
     `statistic` values `arch_vs_generic_e2e`,
     `arch_vs_routing_only_e2e`, `arch_vs_generic_inlibrary`,
     `arch_vs_discovery_only_inlibrary_harm` (paired-difference statistics:
     there is no single-arm rate in this family, as in `universa-seal/7`
     there was no baseline arm), `bound_direction` `greater` for all four,
     `threshold` `0.0` for the first three and `-0.05` for the bounded-harm
     claim;
   - `stop_rules`: every stop rule of §9; and
   - `output_path`: exactly `results/experiments/router-loop-v2-sealed-1.json`.
4. **Push commit B to the private remote before any sealed seed is
   instantiated**, and verify the remote contains it.
5. Confirm a clean worktree and the pushed seal, and only then execute the
   canonical command once:

   ```bash
   env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src python \
     scripts/run_loop_v2_sealed_1.py \
     --output results/experiments/router-loop-v2-sealed-1.json
   ```

   `--seal` defaults to `docs/23-router-loop-v2-seal.json` and therefore
   does not appear; no hash is ever passed on the command line.
6. Independently validate the retained raw rows (§11) before editing any
   result prose, then commit the immutable result artifact.

The runner accepts a `--seal` argument defaulting to that path. It parses
and validates the machine-readable seal file rather than trusting a naked
command-line hash; there is no `--expected-runner-sha256`-style flag.
Validation requires the schema tag and every field above with no unknown
keys at any level (including inside each claim object), each claim's
`id`, `statistic`, `threshold`, and `bound_direction` (and `role` where
present) value-checked against the runner's frozen claim definitions and
the descriptive subfields required present as nonempty strings,
`design_commit` present in the repository's history
(`git cat-file -e <design_commit>^{commit}`) AND an ancestor of HEAD
(`git merge-base --is-ancestor <design_commit> HEAD`), the imported
`universa` package resolving under the project's `src`, equality of each
embedded hash with both the runner's frozen constants and the actual file
bytes, equality of the runner's own runtime SHA-256 with `runner_sha256`,
equality of every `code_manifest` hash with the actual file bytes, the seal
file's presence in the HEAD commit and byte-identity of the on-disk seal
with its HEAD blob, no symlink at the seal, protocol, or runner path, a
clean worktree per `git status --porcelain --untracked-files=all`, and
equality of `output_path` with the `--output` argument. Only after all of
this — and after setting and verifying one PyTorch thread and a
hidden/absent CUDA — may the runner attempt the 36 eval instance builds,
apply eligibility, and (only if at least 30 seeds are eligible) build the
train rows, train the three models in the pinned order of §5, execute the
three conditions x four arms for every eligible seed, verify determinism
bit-identically, aggregate, and write the result. Before publishing a
`complete` result it re-checks the clean worktree, the code-manifest
hashes, and the on-disk seal, protocol, and runner files, and records the
post-run status. It records `design_commit` (commit A) and HEAD at
execution time as the execution revision.
