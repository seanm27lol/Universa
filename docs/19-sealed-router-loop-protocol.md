# Frozen-design protocol: sealed evaluation of the route-or-discover loop on the graph-quotient family

Status: **complete and freeze-ready, but not yet sealed or executable**. This
document becomes immutable only when the runner, tests, and design-seal record
described in §12 have been committed. Until then, the declared eval seeds must
not be instantiated. After sealing, any substantive change creates a new
experiment with a new untouched seed block; it must not be called
router-loop-sealed-1.

Protocol date: 2026-09-01.
Experiment id: `universa-router-loop-sealed-1`.

## 1. Purpose and precise claim scope

This experiment is an untouched-seed evaluation of the Universa
route-or-discover loop (`universa.loop`: `LibraryLoop`, `run_loop`,
`make_loop_instance`, `null_observations`, `acquisition_correct`,
`routing_correct`) on the graph-quotient switch-instance family of
`universa.generators` / `universa.budgets` **only**. The regime is the
integration of `docs/00-design.md` §4 and §5: the loop holds a seeded library
of candidate structures; per instance it scores every candidate by the
certified commutation residual of the instance's planted chain map against it
and reads the best (minimum) score. Best score at or below the alarm
tolerance: the library explains the instance and the loop commits to the
argmin candidate (ROUTE). Best score above the tolerance: the misfit alarm
FIRES, and the loop runs the certified discovery head over transported
observations, gates the result through admission against the current library,
and — only on admission — appends the discovered constraint and re-routes the
instance to it (DISCOVER); a discovery that cannot certify, or that certifies
but is blocked at admission, is REFUSED and nothing is routed.

**The certified regime, and what it isolates.** Every score the loop reads is
an exact certified commutation residual: the planted quotient map commutes
with exact integer arithmetic, so the true candidate's score is exactly `0.0`
in float64, while the generator accepts decoys only above a `1e-9` residual.
There is no degraded observation anywhere in this design — the alarm, the
discovery trigger, the admission gate, and the re-route are all decided by
certified float64 residuals. The certified regime thereby **isolates the
loop's decision flow** — alarm → discovery → admission → re-route — from
every question of observation quality: whether the loop fires the alarm when
the library cannot explain the instance, stays silent when it can, acquires
the missing structure end-to-end, and refuses to invent one from noise. A
**learned-router drop-in for degraded observation is v2 territory**: the
no-anchor degraded regime belongs to the sealed v2, 2-complex, sheaf, and
group router experiments, which stand separately; their results are not
evidence here, and this result is not evidence for them.

**There is no learned model and no training in this experiment.** The
procedure under evaluation is a certified deterministic pipeline: candidate
scoring is the argmin of certified commutation residuals, discovery is the
certified SVD head of `universa.discovery`, and every quantity is a float64
numpy function of the generator seed, pinned by the code manifest of §2.
Nothing is fit, annealed, or selected; the train block is the empty sentinel
of §3. The sealed claims are therefore claims about the **procedure's
behavior over the seed mechanism** — the distribution induced by drawing
generator seeds from the declared block and running the frozen deterministic
pipeline — and nothing else. In particular this experiment cannot establish:

- anything about the other Universa families (2-complexes, cellular sheaves,
  category nerves) or about transfer across families;
- anything about stochastic or learned loop variants — there is no
  algorithmic randomness given the seed, so intervals quantify variation over
  the seed mechanism, not sampling noise of the procedure;
- anything about degraded-observation routing or any learned router — the
  degraded regime is the v2 router family's territory, declared above; or
- anything about real data, real graphs, or any deployed system.

**H1 is the system's end-to-end claim, declared in full.** H1 credits the
whole loop — misfit alarm, certified discovery, novelty admission, and
re-route to the appended structure with a router-acceptable map misfit — as
one conjunctive per-seed predicate on the out-of-library condition. It is
**not** a discovery-alone claim: the discovery-alone question (coverage,
certification, admission-readiness of the discovery head with the truth
withheld) was exp4, `universa-discovery-sealed-1`
(`docs/10-sealed-discovery-protocol.md`), whose sealed result stands
separately and is not evidence here. H1 subsumes that pipeline as one phase
of the loop and adds the alarm that triggers it, the admission gate, and the
re-route the loop exists to perform.

**H3 and H4 are complementary by design, declared in full.** Both read the
same in-library condition: H3 credits routing to the true target
(`routed_index == 0`) and H4 credits the alarm staying silent when the
library explains the instance. Routing correct and alarm-silent are expected
together — routing to the true target is possible only when the alarm is
quiet — and §7 states the mechanical coincidence of the two indicators under
the frozen eligibility audit, declared here in advance, not discovered after
the fact. They remain separate claims of the family because they name
distinct behaviors (correct commitment; alarm precision).

**The loop does not permute candidates, declared in full.** The router
experiments permuted library order as an anti-leak device against a learned
model that could exploit position. Here **there is no learned model that
could exploit position**: scoring is the certified argmin of commutation
residuals, computed from the instance and the library alone. The generators
place the true target at library index 0, `universa.loop` never permutes,
and the eligibility audit of §10 certifies the library's make-up per seed
(true score exactly `0.0`, every decoy strictly above `1e-9`), so position
carries no exploitable signal. The runner never permutes, and
`routing_correct` reads index 0 by frozen convention.

All prior numbers — the documented behavior for the loop module's demo seeds,
the README figures, and any engineering check — are demo-scale sanity
evidence only. They are not evidence for or against any claim here; only the
sealed run decides the four claims of §8.

## 2. Immutable inputs and hashes

Known immutable inputs at protocol drafting are the Universa code manifest —
one SHA-256 per file of `src/universa/*.py` at the design commit — the
runner, and this protocol. No fingerprint is available at draft time: the
design commit does not exist yet. Every entry below is the literal
placeholder **`PENDING-DESIGN-SEAL`**, to be replaced by actual 64-hex
fingerprints in the machine-readable seal record of §12 (commit B), computed
at the design commit (commit A). The runner refuses to start while any
placeholder remains.

The manifest has 18 entries: the 17 files sealed in the discovery
experiment plus `loop.py`, the route-or-discover loop added for this
experiment series. The prior seals and their manifests remain the frozen
records of those experiments and are not modified by this one.

| object | path | SHA-256 |
|---|---|---|
| universa package init | `src/universa/__init__.py` | `PENDING-DESIGN-SEAL` |
| probe budgets and budget instances | `src/universa/budgets.py` | `PENDING-DESIGN-SEAL` |
| category/nerve instances | `src/universa/category_instances.py` | `PENDING-DESIGN-SEAL` |
| 2-complex family | `src/universa/complexes2.py` | `PENDING-DESIGN-SEAL` |
| certified discovery head | `src/universa/discovery.py` | `PENDING-DESIGN-SEAL` |
| graph generators and `subseed` | `src/universa/generators.py` | `PENDING-DESIGN-SEAL` |
| route-or-discover loop | `src/universa/loop.py` | `PENDING-DESIGN-SEAL` |
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
| sealed runner | `scripts/run_loop_sealed_1.py` | `PENDING-DESIGN-SEAL` |
| this protocol | `docs/19-sealed-router-loop-protocol.md` | recorded externally in the seal record |

This protocol's own SHA-256 cannot appear inside itself — a file cannot
contain its own hash — so it is recorded externally in the seal record
(`protocol_sha256`, §12) and embedded as a frozen constant in the runner.

The runner must fail before any instance construction unless every fingerprint
above matches. It must also require this execution environment, recorded in
full in the result:

| component | required value |
|---|---|
| exact mathematics | numpy `float64` throughout; no torch tensor carries any quantity (there is nothing to learn) |
| tensor device and dtype | CPU; torch imported only to pin and verify the single-thread CPU environment |
| PyTorch threads | exactly `1` (when torch is importable) |
| CUDA | unavailable or hidden (`torch.cuda.is_available()` is `False`) |

The runner records the Python, PyTorch, and NumPy versions, the operating
system and machine, `CUDA_VISIBLE_DEVICES`, the thread setting, the Git
revision, the initial `git status --porcelain --untracked-files=all` (which must be empty), every
verified hash, and the canonical command. No dependency may be installed or
upgraded between the seal and the run.

## 3. Seed blocks and the no-preview declaration

**Train block: none.** This experiment has no training and therefore no train
seed block. The pinned convention — chosen here, validated by the runner —
is that the seal record carries the key `train_seed_block` with exactly the
documented empty-block sentinel

```json
{"first": 0, "last": 0}
```

Seed `0` belongs to no Universa seed block, and `first > last` encodes the
empty set. The runner must require the key's presence with exactly this
value (any other value is a seal-validation failure) and must never
instantiate a train seed: there are no train rows, no fit, and no
train-side audits in this design.

**Sealed eval block:** the 36 consecutive generator seeds

```text
130001, 130002, 130003, 130004, 130005, 130006, 130007, 130008, 130009,
130010, 130011, 130012, 130013, 130014, 130015, 130016, 130017, 130018,
130019, 130020, 130021, 130022, 130023, 130024, 130025, 130026, 130027,
130028, 130029, 130030, 130031, 130032, 130033, 130034, 130035, 130036
```

At the time this protocol was drafted, this block was absent from the working
tree and from all Git history (verified by a repository-wide word-boundary
search over every commit; incidental digit collisions inside unrelated float
strings of prior result artifacts are not seed uses, and no `"seed"` field or
seed literal in the block exists anywhere), and no project run had
instantiated or previewed any member **on any generator family or in any
pipeline stage**. In particular, no one has built a loop, budget, switch, or
2-complex instance, synthesized an observation, run the loop, run discovery,
computed a commutation score, certificate residual, or map misfit, printed a
dimension, smoke-tested, or debugged with any seed in the block. The block is
disjoint from the v1 train block `10001..10200`, the v1 sealed eval block
`30101..30136`, the v2 train block `50001..50200`, the v2 sealed eval block
`60101..60136`, the 2-complex train block `70001..70200`, the 2-complex sealed
eval block `80101..80136`, the discovery sealed eval block `90101..90136`,
the sheaf train block `11001..11200`, the sheaf sealed eval block
`20101..20136`, the group train block `21001..21200`, the group sealed eval
block `40101..40136`, and every demo seed.

**Why this block and not `70101..70136` (declared).** An earlier draft of this
protocol named `70101..70136` as the eval block. That block is numerically
contained in the 2-complex experiment's train block `70001..70200`
(`docs/08-router-2complex-seal.json`): those seeds were instantiated as
2-complex-family train instances — a different structure family and task —
before the loop experiment was designed. Under the series' strict no-preview
standard (a sealed block must never have been instantiated in any dataset
construction, for any purpose), a merely-declared containment is weaker than
verified absence, so the block was discarded unopened and replaced with
`130001..130036`, which is verified absent from every dataset construction in
the repository's history. No member of `70101..70136` will be used in this
experiment.

The no-preview rule remains in force until commit B of §12 — the
machine-readable seal record — is committed **and pushed to the private
remote**. Until then:

- tests may use the documented loop demo seeds (`70001..70005`) or
  hand-built fixtures only — runner tests monkeypatch the runner's
  `SEALED_EVAL_SEEDS` constant to such seeds — and may not parameterize
  over, import, derive data from, or otherwise touch the sealed block or any
  other sealed block (`20101..20136`, `30101..30136`, `40101..40136`,
  `60101..60136`, `80101..80136`, `90101..90136`);
- source-code constants containing the declared integers (e.g. the runner's
  frozen seed-list constant) are permitted;
- merely calling `make_loop_instance`, `make_budget_instance`,
  `make_switch_instance`, or `run_loop` with a declared eval seed counts as a
  preview, even if the result is discarded.

If any declared eval seed is instantiated before commit B is pushed, the
entire 36-seed block is void. It will not be partially salvaged. A new
consecutive block must be chosen, verified absent from history, and declared
in a newly committed protocol before any execution.

## 4. Data and generation protocol

**The instance and the two library views.** For a generator seed `s`, the
instance and its paired library views are exactly

```text
instance, in_library, out_library = universa.loop.make_loop_instance(s)
```

where `make_loop_instance(s)` builds `universa.budgets.make_budget_instance(
s, 8, 14, 6, 3)` — a random connected source graph, a planted quotient chain
map onto `num_classes=6` classes (the budgets choice, keeping the quotient
non-complete so decoys have genuinely different kernels), and 3 certified
discriminating decoys, all exact integer arithmetic, float64, with a planted
transported quantity and a certified identifiability threshold — and returns
two views of the SAME instance: `in_library = instance.candidates`, the
**K = 4** candidates with the true target at index 0 by generator convention,
and `out_library = instance.decoy_targets`, the 3 decoys with the truth
withheld.

**The loop configuration.** Every condition runs with the module defaults,

```text
loop = universa.loop.LibraryLoop()
# alarm_tol=1e-9, misfit_tol=1e-10, novelty_tol=1e-6, num_observations=16
```

the frozen exp4 constants (`alarm_tol` is `ALARM_TOL = 1e-9`, re-pinned at
`universa.router.RESIDUAL_TOL`; `misfit_tol` is
`universa.operators.CERT_TOL = 1e-10`; `novelty_tol` is
`universa.discovery.DEFAULT_NOVELTY_TOL = 1e-6`).

**One loop pass.** `run_loop(loop, instance, library, seed=s,
observations=None)` executes the frozen phases, all in certified float64:

1. SCORE — the certified commutation score of the planted map against every
   candidate of the given library (exactly
   `SwitchInstance.commutation_scores` semantics: a `ChainMap` probe per
   candidate, max residual over degrees; one degree in this family);
2. ALARM — `best = min(scores)`; the alarm fires iff
   `best > loop.alarm_tol`;
3. ROUTE (alarm quiet) — commit to the argmin candidate, first-index
   tie-breaking; the library is untouched; there is no discovery phase;
4. DISCOVERY (alarm fired) — the observations are
   `universa.discovery.synthesize_observations` of a `SwitchInstance` view of
   the budget instance (the same underlying arrays), i.e. the frozen exp4
   schedule: column `j` is `f1 a_j` with `a_j` a certified random source
   cycle drawn from `subseed(s, "discovery-observation", str(j),
   str(attempt))`, `attempt` counting from 1, redrawn while `||y_j|| < 1e-9`
   (`MIN_OBSERVATION_NORM`), at most 1000 attempts; the head certifies via
   `discover_constraint(observations, ambient_dim, tol=loop.misfit_tol,
   seeds=(s,))` under the `operators.py` rank convention, the
   dimensional-stabilization rule with `STABILITY_FRACTION = 0.25`, and the
   annihilator boundary certified by `nullspace_basis`; every insufficiency
   is a REFUSED outcome — never a weak structure. When the `observations`
   override is given it replaces the synthesized schedule — the control
   channel of the null-control condition; it is consulted only on the
   discovery path, never when the alarm stays quiet;
5. ADMISSION + RE-ROUTE — a certified discovery goes through
   `admit_to_library` against the current library boundaries at
   `(loop.misfit_tol, loop.novelty_tol)`, and the router-acceptance residual
   `map_misfit = ||C_disc f1 Z||_F` is computed over the certified source
   cycle basis `Z` (the `run_discovery` formula). On admission the library
   grows by one and the instance re-routes to the appended index (DISCOVER);
   a blocked admission is REFUSED.

`run_loop` requires `seed == instance.seed`: the observation schedule keys on
the instance seed, so the provenance record and the consumed schedule cannot
disagree.

**The three conditions.** Exactly three conditions, all deterministic
end-to-end, applied to every eligible seed:

1. **in-library** — `run_loop(loop, instance, instance.candidates, seed=s)`:
   the library contains the truth at index 0; the expected behavior is a
   quiet alarm and ROUTE to index 0. Source of the H3 and H4 per-seed
   statistics;
2. **out-of-library** — `run_loop(loop, instance, instance.decoy_targets,
   seed=s)`: the truth is withheld; the expected behavior is a fired alarm,
   certified discovery, admission, and re-route to the appended structure.
   Source of the H1 per-seed statistic;
3. **null-control** — `run_loop(loop, instance, instance.decoy_targets,
   seed=s, observations=null_observations(s, ambient_dim))`: the same
   out-of-library run with the observation override replaced by
   structure-free noise — the exp4 H4 schedule. Column `j` is

   ```text
   np.random.default_rng(subseed(s, "discovery-null", str(j)))
         .standard_normal(ambient_dim)        for j = 0..15
   ```

   with `ambient_dim` the number of edges of the seed's true quotient target
   (the target edge space), `Y_null` the column stack (`ambient_dim x 16`,
   float64). The override is consulted only on the discovery path: the alarm
   reads the decoy library scores exactly as in condition 2, and under the
   eligibility audit of §10 the alarm fires; the expected behavior is REFUSED
   — a false constraint admitted from noise is a specificity failure. Source
   of the H2 per-seed statistic.

**Deterministic random streams.** Every stochastic component uses
`universa.generators.subseed`:

```text
message = ":".join(["universa", str(seed), *components]).encode("utf-8")
digest  = SHA256(message)
subseed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
```

The structured and null observation draws are disjoint subseed families
(`"discovery-observation"` vs `"discovery-null"`).

**Determinism, verified.** The pipeline is a deterministic float64 function
of the seed, and `LoopOutcome` is a pure scalar record precisely so that two
runs compare bit-identical with plain equality. The runner executes the full
per-seed pipeline — all three conditions — twice for every eligible seed and
requires bit-identical raw rows; any mismatch is a whole-run
`design_failure`. Any NaN or non-finite score, residual, or statistic, and
any violation inside the certified machinery (rank, orthonormality, or
certificate failure), is likewise a whole-run `design_failure`, never a
dropped row.

## 5. Certified deterministic procedure — no learned parameters

**This experiment trains nothing.** There is no model, no fit, no gradient,
no annealing, no model selection, no torch parameter, and no train seed
block (§3). The procedure under evaluation is the certified deterministic
pipeline of §4 — one `run_loop` pass per condition, with certified argmin
scoring and the certified discovery head on the alarm path — pinned
end-to-end by the code manifest of §2. Every quantity it emits is a
deterministic float64 numpy function of the generator seed.

Where the router protocols record training provenance, this design records
**procedure provenance** instead: the exact frozen calls of §4; the loop
tolerances `alarm_tol = ALARM_TOL = 1e-9` (re-pinned at
`universa.router.RESIDUAL_TOL` — duplicated and documented in `loop.py`
rather than imported, because the loop module is numpy-only by contract and
`universa.router` imports torch), `misfit_tol = CERT_TOL = 1e-10`, and
`novelty_tol = DEFAULT_NOVELTY_TOL = 1e-6`; the discovery module constants
`STABILITY_FRACTION = 0.25` and `MIN_OBSERVATION_NORM = 1e-9`; the
observation count `num_observations = 16`; the NumPy version; and the
per-file code-manifest SHA-256s. The result states plainly, in place of any
training record, that no training provenance exists because no training
occurred (§11). A non-finite value at any point is an error, never a
warning.

## 6. Conditions

Exactly three conditions, all applied to every eligible seed:

1. **in-library** — `run_loop` on `instance.candidates` (K = 4, the true
   target at index 0): the source of the H3 and H4 per-seed statistics;
2. **out-of-library** — `run_loop` on `instance.decoy_targets` (the truth
   withheld): the source of the H1 per-seed statistic;
3. **null-control** — `run_loop` on `instance.decoy_targets` with the
   `null_observations` override of §4: the source of the H2 per-seed
   statistic.

There are no arms and no baselines: the claims are absolute floor/rate
claims about the procedure's behavior over the seed mechanism, not margins
over a comparator, and H2 is the specificity control condition of the same
family. Conditions 2 and 3 share the instance and the decoy library; their
observation streams are independent subseed families
(`"discovery-observation"` vs `"discovery-null"`). The alarm decision basis
is identical in all three conditions: the library scores are computed from
the planted map and the library alone, and observations — synthesized or
overridden — are consulted only on the discovery path, never by the alarm.
Condition 1 shares the instance with conditions 2 and 3 but reads a
different library view; the two views of one instance are the paired
conditions of the loop design. There is no fourth condition, no eval-time
tuning, and no per-seed choice of any kind.

## 7. Estimands and unit of inference

For each eligible seed `s`, write `out_in(s)`, `out_out(s)`, and
`out_null(s)` for the `LoopOutcome` records of the in-library,
out-of-library, and null-control conditions. The four frozen per-seed
statistics are:

```text
acquisition(s)   = 1{acquisition_correct(out_out(s), instance)}
                 = 1{alarm fired AND discovery certified AND admitted
                     AND map_misfit <= 1e-9}
refusal(s)       = 1{not out_null(s).admitted}
routing(s)       = 1{routing_correct(out_in(s))}
                 = 1{mode == "route" AND routed_index == 0}
alarm_silence(s) = 1{not out_in(s).alarm_fired}
```

Declared conventions, fixed here:

- **The alarm precedes observations.** The library scores — the alarm's sole
  decision basis — are computed from the planted map and the library alone;
  the `observations` override of the null-control condition is consulted only
  on the discovery path (the `run_loop` contract), so conditions 2 and 3 have
  identical alarm decisions per seed. Under the frozen eligibility audit of
  §10 (true score exactly `0.0`, every decoy strictly above `1e-9`), the
  audit certifies the loop's decision flow per seed: the alarm is quiet in
  the in-library condition and fires in the out-of-library and null-control
  conditions.
- **H3 and H4 coincide mechanically under the audit — declared, not
  discovered after the fact.** `routing(s) = 1` requires `mode == "route"`,
  which holds iff the alarm stayed quiet, i.e. iff `alarm_silence(s) = 1`;
  and when the alarm is quiet the audit forces the argmin to be index 0
  uniquely (true score `0.0`, every decoy strictly above `1e-9`), so
  `routed_index == 0` follows. Conversely a fired alarm gives a non-route
  mode and both indicators are `0`. Hence `routing(s) == alarm_silence(s)`
  on every eligible seed. This is the complementarity of §1: routing correct
  and alarm-silent are expected together; both remain in the family as
  separate claims crediting distinct behaviors (correct commitment; alarm
  precision).
- **The null-control refusal statistic is "NOT admitted".** `refusal(s)`
  covers both correct refusal modes on structure-free observations —
  `DiscoveryInsufficient` (the expected path) and a certified-but-blocked
  admission — and it is well-defined on every `LoopOutcome` (a route-mode
  outcome admits nothing). A false constraint admitted from noise is a
  specificity failure of the procedure.
- **`acquisition` is the loop's own end-to-end predicate.**
  `universa.loop.acquisition_correct` is fail-closed on incoherent records
  (a discover-mode outcome that does not route to the appended structure is
  an error, not a `False`) and reads only certified residuals, never the
  withheld truth. Its map-misfit threshold `1e-9` is the router-acceptance
  residual on `||C_disc f1 Z||_F`, distinct from the admission gate's
  `misfit_tol = 1e-10` on the certificate residual; both are frozen.

The **generator seed is the sole unit of inference**. It jointly determines
the instance, both library views, the structured observation draw, and the
null observation draw. Individual observations are not independent units and
are never pooled as such.

Each claim's estimand is `theta = E_seed[statistic]` over the eligible-seed
mechanism, estimated by the arithmetic mean across the `n` eligible seeds,
with

```text
SE = sample_standard_deviation(statistic) / sqrt(n)
```

(sample standard deviation, `n - 1` denominator). Student-t inference assumes
eligible seed-level realizations are exchangeable draws from this
deterministic generator-and-subseed scheme; it conditions on the frozen
eligibility rule of §10. Because the procedure is deterministic given the
seed, intervals quantify Monte Carlo variation over the seed mechanism only —
there is no algorithmic sampling noise to average over — and say nothing
about uncertainty for real data.

## 8. The four frozen claims and decision rules

The following four and only these four claims form the confirmatory family.
H1 is primary; H2–H4 are secondary members of the same family.

| id | condition | statistic | prediction and governing decision |
|---|---|---|---|
| `h1-loop-acquisition-rate` | out-of-library | `acquisition` | mean `>= 0.95`; supported iff the one-sided lower bound exceeds `0.95` |
| `h2-loop-false-admission-control` | null-control | `refusal` | mean `>= 0.95`; supported iff the one-sided lower bound exceeds `0.95` |
| `h3-loop-in-library-routing` | in-library | `routing` | mean `>= 0.95`; supported iff the one-sided lower bound exceeds `0.95` |
| `h4-loop-alarm-precision` | in-library | `alarm_silence` | mean `>= 0.95`; supported iff the one-sided lower bound exceeds `0.95` |

The formal hypotheses are fixed as:

- **H1** (id `h1-loop-acquisition-rate`, primary): `theta =
  E_seed[acquisition]`; H0: `theta <= 0.95`; H_A: `theta > 0.95`; supported
  iff the one-sided Bonferroni lower bound is above `0.95`. This is the
  system's **end-to-end** claim of §1 — alarm + discovery + admission +
  re-route with a router-acceptable map misfit — not a discovery-alone claim
  (that was exp4).
- **H2** (id `h2-loop-false-admission-control`): `theta = E_seed[refusal]`;
  H0: `theta <= 0.95`; H_A: `theta > 0.95`; supported iff the one-sided
  Bonferroni lower bound is above `0.95`. Non-admission is the correct
  behavior on structure-free data; a false constraint admitted from noise is
  a specificity failure. This is the false-admission control of the family,
  not a power claim.
- **H3** (id `h3-loop-in-library-routing`): `theta = E_seed[routing]`; H0:
  `theta <= 0.95`; H_A: `theta > 0.95`; supported iff the one-sided
  Bonferroni lower bound is above `0.95`. A supported H3 means the loop
  commits to the true target when the library contains it, at the required
  rate over the seed mechanism.
- **H4** (id `h4-loop-alarm-precision`): `theta = E_seed[alarm_silence]`;
  H0: `theta <= 0.95`; H_A: `theta > 0.95`; supported iff the one-sided
  Bonferroni lower bound is above `0.95`. A false alarm on an explainable
  instance sends the loop down the discovery path it should not take; H4 is
  the alarm-precision side of the in-library condition, complementary to H3
  by design (§1, §7).

Family-wise type-I error is controlled at `0.05` by Bonferroni over four
claims, one-sided Student-t with `n` eligible seeds:

```text
alpha_per_claim = 0.05 / 4 = 0.0125
q = 1 - alpha_per_claim = 0.9875
lower = mean(statistic) - t_(0.9875, n-1) * SE
```

The direction-specific bound above is the sole support rule for each claim,
applied mechanically to the frozen constants — **including when `SE` is
exactly zero**. All four claims are indicator means and can hit this
mechanical case: when the sample standard deviation is `0`, `SE = 0` and the
one-sided lower bound **equals the estimate**; support then reduces to the
estimate itself strictly exceeding the threshold. This case is specified
here in advance and is decided by the same frozen rule, not by discretion.

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
bound; the critical value; the direction (`greater`); the threshold (`0.95`);
and the support decision. No hypothesis may be added, dropped, reversed,
re-thresholded, or moved between families after execution. All nulls and
failed predictions are reported.

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
- an instance-build exception is a **whole-run** `design_failure`, never an
  exclusion — the offending seed is recorded, the run stops, and no seed is
  ever deleted alone;
- the build and the undegraded audit of §10 are the sole eligibility gate.
  Every *other* failed check — any determinism, NaN/non-finite value, audit,
  rank, orthonormality, certificate, or validation failure, including any
  violation inside the certified discovery machinery, a nonzero-observation
  draw failure, an incoherent `LoopOutcome`, and any failed result-audit
  recomputation — is a whole-run `design_failure`;
- fewer than 30 eligible seeds yields `design_failure_insufficient_eligible`
  with **no evaluation and no claims**: the run stops before any loop
  execution. Under the frozen build rule a build problem stops the run as
  `design_failure` before eligibility accounting, so this status is the
  fail-closed guard over the eligibility count itself;
- a caught failure preserves every completed raw row, identifies the failing
  seed and stage, records the exception type and message and the preflight
  provenance, and emits all four claims with `supported: null`;
- there is no outcome-dependent stopping, no interim analysis, no sample-size
  extension, and no condition- or seed-level deletion;
- the run is never rerun because the results are surprising, weak, null, or
  apparently decisive;
- the runner must not compute or print scores, residuals, rates, or claim
  summaries until every required raw row has completed; progress output is
  limited to seed identifiers, stage, completion state, and timing.

Mandatory stop conditions (any one stops the run before or during execution):

- any preflight mismatch: seal-validation failure (including an unknown seal
  key at any level including inside a claim object, a tampered claim field, a
  `train_seed_block` other than the empty sentinel of §3, a `design_commit`
  absent from the repository's history or not an ancestor of HEAD, or the
  imported `universa` package not resolving under the project's `src`), hash
  mismatch, a seal not committed at HEAD, an on-disk seal not byte-identical
  to its HEAD blob, a symlinked seal, protocol, or runner file, a dirty
  worktree, an existing output path, available CUDA, or a torch thread count
  other than one (when torch is importable);
- a worktree found dirty, or a code-manifest hash, the on-disk seal,
  protocol, or runner file found changed, at the end-of-run re-check before
  publishing a `complete` result;
- a raw-row count that violates the frozen invariant `3 x n` — exactly one
  in-library row, one out-of-library row, and one null-control row per
  eligible seed;
- an instance-build exception;
- fewer than 30 eligible seeds;
- any determinism, NaN, audit, rank, or validation failure;
- aggregation failure or a non-finite estimand; or
- any manual cancellation or infrastructure interruption.

If infrastructure fails after execution begins, the attempt is preserved
under a distinct non-overwriting failure path
`results/experiments/failures/router-loop-sealed-1.<status>.json`; the
canonical output path is reserved for status `complete` and a failed
attempt never occupies it. An exact-code retry is allowed
only for a documented non-scientific failure, before any summaries are
inspected, with identical hashes. Every attempt remains in the evidence
bundle. The first complete, fully validated attempt is canonical; no choice
among complete attempts is allowed. Any code, tolerance, dependency, or
analysis change after a partial run invalidates the seed block and requires a
new protocol and new seeds.

## 10. Eligibility

An eval seed is **eligible** if and only if both hold:

1. its instance builds — `make_loop_instance(seed)`, i.e.
   `make_budget_instance(seed, 8, 14, 6, 3)`, returns (a build exception is
   a whole-run `design_failure` per §9, never an exclusion); and
2. the **undegraded audit** passes: computed on the undegraded instance
   without any observation draw, the true candidate's certified commutation
   score — the max residual over degrees of the planted-map `ChainMap`
   probe; one degree in this family — is exactly `0.0` (and therefore
   `<= 1e-9`), and every decoy's score is strictly `> 1e-9`.

The audit is **bookkeeping only, never a loop input**: the loop never reads
it — `run_loop` re-derives every score from the instance and the library on
each pass. The audit certifies the library's make-up per seed, which is what
licenses reading routing to index 0 as correct (H3), what certifies the
alarm's decision basis per condition (quiet in-library; fired out-of-library
and null-control — the decision flow of §1), and what licenses the
no-permutation stance of §1. It is observation-independent and therefore a
deterministic per-seed property, identical across the three conditions.

Ineligible seeds are recorded with their explicit reasons and are never
replaced, never deleted, and never scored. `n`, the number of eligible
seeds, is the inference sample size of §7–§8. If `n < 30`, the run stops
before any loop execution with `design_failure_insufficient_eligible` and no
claims. All 36 seeds are attempted, in seed order. There is no train-side
eligibility because there are no train rows.

## 11. Result JSON schema

The runner writes exactly one result file, to
`results/experiments/router-loop-sealed-1.json`, through a temporary file
followed by an atomic no-clobber hard-link publish (create-if-absent, never
replace — stronger than a rename), and refuses an existing path. Non-
`complete` statuses are published instead to
`results/experiments/failures/router-loop-sealed-1.<status>.json` under the
same no-clobber rule, and failure artifacts are written **only** there. The
JSON contains, at minimum:

- `schema`: the literal tag `universa-router-loop-sealed-result/1`, and the
  frozen configuration (the exact `make_loop_instance` call and sizes, the
  `LibraryLoop` defaults with all tolerances, `num_observations`, `K = 4`,
  the null-observation schedule, the claim family with thresholds);
- the seal SHA-256, protocol SHA-256, runner SHA-256, and every code-manifest
  SHA-256, plus `design_commit` (commit A) and the execution revision (HEAD
  at execution time);
- environment and provenance: Python/PyTorch/NumPy versions, OS and machine,
  thread count, the operator-provided and effective `CUDA_VISIBLE_DEVICES`,
  the recorded `git status --porcelain --untracked-files=all` output (key
  `git_status_porcelain`, must be empty, recorded pre- and post-run), the
  canonical command, and `sys.argv`;
- procedure provenance per §5 — and, in place of any training provenance,
  the plain recorded statement that no training occurred and no learned
  parameters exist;
- all 36 candidate seed records, each with its eligibility verdict (and its
  explicit reason when ineligible);
- raw rows per (seed, condition) — one row per condition per eligible seed —
  each carrying `seed`, `condition` (`in-library` / `out-of-library` /
  `null-control`), and the full `LoopOutcome` scalars: `mode`,
  `alarm_fired`, `best_score`, `scores` (per-candidate, in library order),
  `routed_index`, `discovery_verdict`, `discovery_reason`,
  `certificate_residual`, `admitted`, `admission_min_distance`,
  `map_misfit`, `final_library_size`, and `num_observations`. Fields
  undefined for the row's mode (all discovery-phase fields in route mode;
  the admission and misfit fields on an insufficiency) are `null`;
- the four claim summaries, each with estimate, sample standard deviation,
  SE, one-sided lower bound, critical value, direction, threshold, and
  `supported` (or `supported: null` for all four on any non-`complete`
  status); and
- an audit block recomputing **every claim statistic from the raw rows
  alone**: row counts (the `3 x n` invariant), eligible-seed count, the
  indicator recomputation against the frozen predicates and thresholds
  (`acquisition` per the frozen conjunctive predicate, `refusal` as
  NOT-admitted, `routing` as route-to-index-0, `alarm_silence` as
  alarm-not-fired), means, standard deviations, SEs, bounds, and decisions.

**The audit's scope, stated plainly.** (a) Every claim statistic and decision
is recomputable from the raw rows alone — **no matrices are needed**:
`LoopOutcome` is a pure scalar record and every frozen predicate reads only
those scalars, so no matrices are retained and none are required by the
audit. (b) Unlike exp4, the audit does **not** re-verify the discovery head's
internal SVD certificates from retained matrices (none are retained); the
per-row certified residuals are reproducible only by re-running the
manifest-pinned deterministic pipeline, and that assurance is the sealed
run's own — pinned by the code manifest and enforced by the double-execution
bit-identity check of §4. (c) As in exp4, no hash-pinned model exists or is
needed, because nothing was trained.

An independent validator must recompute all summaries solely from the
retained raw rows and fail closed on missing rows, duplicate (seed,
condition) keys, condition imbalance, ineligible seeds carrying rows,
`scores` inconsistent with `best_score` or with the mode/routed-index
coherence of §4, wrong t constants, or any decision inconsistent with §8.

## 12. The two-commit seal procedure

The following order is mandatory:

1. **Commit A (design commit):** this protocol, the runner
   `scripts/run_loop_sealed_1.py`, and all its tests — committed without
   instantiating any declared eval seed. Tests use the documented loop demo
   seeds (`70001..70005`) or hand-built fixtures only — runner tests
   monkeypatch the runner's `SEALED_EVAL_SEEDS` constant to such seeds — and
   must cover, at minimum: the `discovery-null` subseed derivation and that
   it drives the per-vector Gaussian draws of the null-control condition, no
   sealed-seed fixture use (this block and every other sealed block), the
   H3/H4 mechanical coincidence on eligible fixtures, the alarm's
   independence of the observation override, all three conditions on hand
   fixtures, all four claim decisions including the `SE = 0` mechanical case
   (lower bound equals the estimate), the hard-coded t table, eligibility
   (build plus the undegraded audit) and insufficient-eligible handling,
   failure-artifact shape (`supported: null`), seal parsing and validation
   including the empty `train_seed_block` sentinel, dirty-worktree refusal,
   and output-exists refusal. The runner embeds this protocol's SHA-256 as a
   frozen constant — initially the fail-closed placeholder
   `PENDING_PROTOCOL_SHA256`, replaced by the real hash before commit A —
   and never embeds its own hash: it computes its runtime SHA-256 and
   requires equality with the seal's `runner_sha256`. This avoids a
   self-hash cycle.
2. **Compute fingerprints at commit A:** the protocol SHA-256, the runner
   SHA-256, and the per-file SHA-256 of every `src/universa/*.py` (the
   18-file code manifest of §2).
3. **Commit B (seal):** the machine-readable seal record
   `docs/20-router-loop-seal.json` with frozen contents:
   - `schema`: the literal tag `universa-seal/7`;
   - `design_commit`: the full hash of commit A;
   - `protocol_sha256`, `runner_sha256`;
   - `code_manifest`: the 18 per-file SHA-256 pairs of §2;
   - `train_seed_block`: exactly `{first: 0, last: 0}` — the documented
     empty-block sentinel of §3 (this experiment trains nothing);
   - `eval_seed_block`: `{first: 130001, last: 130036}`;
   - `no_preview_declaration`: the renewed attestation of §3, including the
     discarded-block declaration recorded there;
   - `primary_family`: the four claims of §8 verbatim, as four claim objects
     with the eight subfields `id`, `statistic`, `theta`, `null`,
     `alternative`, `bound_direction`, `threshold`, `support_rule` (schema
     `universa-seal/7` claim objects carry `statistic` — with values
     `acquisition`, `refusal`, `routing`, `alarm_silence` — in place of the
     router seals' `fraction`/`reference`: there is no operating fraction
     and no baseline arm, as in `universa-seal/4`);
   - `stop_rules`: every stop rule of §9; and
   - `output_path`: exactly `results/experiments/router-loop-sealed-1.json`.
4. **Push commit B to the private remote before any sealed seed is
   instantiated**, and verify the remote contains it.
5. Confirm a clean worktree and the pushed seal, and only then execute the
   canonical command once:

   ```bash
   env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src python \
     scripts/run_loop_sealed_1.py \
     --output results/experiments/router-loop-sealed-1.json
   ```

   `--seal` defaults to `docs/20-router-loop-seal.json` and therefore does
   not appear; no hash is ever passed on the command line.
6. Independently validate the retained raw rows (§11) before editing any
   result prose, then commit the immutable result artifact.

The runner accepts a `--seal` argument defaulting to that path. It parses and
validates the machine-readable seal file rather than trusting a naked
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
clean worktree per `git status --porcelain --untracked-files=all`, and equality of
`output_path` with the `--output` argument. Only after all of this — and
after setting and verifying one PyTorch thread and a hidden/absent CUDA — may
the runner attempt the 36 instance builds, apply eligibility, and (only if at
least 30 seeds are eligible) execute the in-library, out-of-library, and
null-control conditions for every eligible seed, verify determinism
bit-identically, aggregate, and write the result. Before publishing a
`complete` result it re-checks the clean worktree, the code-manifest hashes,
and the on-disk seal, protocol, and runner files, and records the post-run
status. It records `design_commit` (commit A) and HEAD at execution time as
the execution revision.
