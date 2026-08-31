# Frozen-design protocol: sealed evaluation of certified discovery on the graph-quotient family

Status: **complete and freeze-ready, but not yet sealed or executable**. This
document becomes immutable only when the runner, tests, and design-seal record
described in §12 have been committed. Until then, the declared eval seeds must
not be instantiated. After sealing, any substantive change creates a new
experiment with a new untouched seed block; it must not be called
discovery-sealed-1.

Protocol date: 2026-08-30.
Experiment id: `universa-discovery-sealed-1`.

## 1. Purpose and precise claim scope

This experiment is an untouched-seed evaluation of the Universa certified
discovery machinery (`universa.discovery`: `run_discovery`,
`discover_constraint`, `admit_to_library`, `discovery_quality`) on the
graph-quotient switch-instance family of `universa.generators` **only**. The
regime is the seeded-library-plus-discovery regime of `docs/00-design.md` §5:
the library is architecturally incomplete — the true quotient target is
**withheld** (decoys only) — and the discovery head must propose the missing
constraint from transported vectors `y_j = f1 a_j`, certify it, and only then
admit it. The question is whether the deterministic discovery *procedure*
recovers the withheld kernel constraint on generator seeds it has never seen:
whether it covers the true kernel (H1), certifies what it returns (H2),
produces a constraint a router would accept under the planted transport (H3),
and — the specificity side — refuses to invent a constraint from
structure-free noise (H4).

**There is no learned model and no training in this experiment.** The
procedure under evaluation is a certified deterministic pipeline: every
quantity is a float64 numpy function of the generator seed, pinned by the
code manifest of §2. Nothing is fit, annealed, or selected. The sealed claims
are therefore claims about the **procedure's behavior over the seed
mechanism** — the distribution induced by drawing generator seeds from the
declared block and running the frozen deterministic pipeline — and nothing
else. In particular this experiment cannot establish:

- anything about the other Universa families (2-complexes, cellular sheaves,
  category nerves) or about transfer across families;
- anything about stochastic or learned discovery variants — there is no
  algorithmic randomness given the seed, so intervals quantify variation over
  the seed mechanism, not sampling noise of the procedure;
- anything about real data, real graphs, or any deployed system; or
- anything about routing margins: the router experiments (v1, v2) stand
  separately, and their results are not evidence here.

All prior numbers — the documented behavior for seeds 2026/7/99 quoted in
`universa.discovery`'s module docstring, the README figures, and any
engineering check — are demo-scale sanity evidence only. They are not
evidence for or against any claim here; only the sealed run decides the four
claims of §8.

**The H4 caveat, declared in full.** H4 is a **specificity / false-discovery
control**, not a power claim. Its observations are i.i.d. standard Gaussian
vectors on the target edge space — structure-free by construction — fed
through the *same* `discover_constraint` call. The correct behavior is
refusal (`DiscoveryInsufficient`); a false constraint admitted from noise is
a specificity failure of the procedure. H4 bounds the false-discovery side of
the same machinery that H1–H3 credit on the power side; it is a member of the
confirmatory family, decided by the same mechanical rule, and its failure is
reported exactly like any other failed claim.

**The observation-provenance caveat, declared in full.** The observations for
H1–H3 come from the certified generator machinery (`run_discovery` →
`make_switch_instance` → `synthesize_observations`): transported vectors
`y_j = f1 a_j` over planted source cycles, exact in integer arithmetic at the
planted chain map, with the true target **withheld from the library** the
discovery process sees. Ground truth (`discovery_quality`, the true boundary)
is evaluation-only and never enters the discovery procedure.

**The retention improvement, declared.** Raw rows retain, per seed, the
discovered boundary matrix and the support-basis matrix (small dimensions:
the target edge space has at most 15 dimensions at these instance sizes).
Because nothing is trained, there is no model to pin: certification
(`||C_disc Y||_F`), coverage (`dim(S_disc)/dim(ker B1_true)`), and misfit
(`||C_disc f1 Z||_F`) are recomputable from the raw rows alone — the retained
matrices plus the deterministic, manifest-pinned generator rebuild of the
instance and observations — without re-running the discovery SVD. This
closes the v2 audit gap, where the learned arm's per-row bits were checkable
only with a retained hash-pinned model; here the audit block states plainly
that no such pinned artifact exists or is needed (§11).

## 2. Immutable inputs and hashes

Known immutable inputs at protocol drafting are the Universa code manifest —
one SHA-256 per file of `src/universa/*.py` at the design commit — the
runner, and this protocol. No fingerprint is available at draft time: the
design commit does not exist yet. Every entry below is the literal
placeholder **`PENDING-DESIGN-SEAL`**, to be replaced by actual 64-hex
fingerprints in the machine-readable seal record of §12 (commit B), computed
at the design commit (commit A). The runner refuses to start while any
placeholder remains.

The manifest has 17 entries: the 15 files sealed in the router-v2
experiment plus the two observation modules added for this experiment
series (`partial_group.py` and `partial_sheaf.py`). The v1 and v2 seals
and their manifests remain the frozen records
of those experiments and are not modified by this one.

| object | path | SHA-256 |
|---|---|---|
| universa package init | `src/universa/__init__.py` | `PENDING-DESIGN-SEAL` |
| probe budgets and budget instances | `src/universa/budgets.py` | `PENDING-DESIGN-SEAL` |
| category/nerve instances | `src/universa/category_instances.py` | `PENDING-DESIGN-SEAL` |
| 2-complex family | `src/universa/complexes2.py` | `PENDING-DESIGN-SEAL` |
| certified discovery head | `src/universa/discovery.py` | `PENDING-DESIGN-SEAL` |
| graph generators and `subseed` | `src/universa/generators.py` | `PENDING-DESIGN-SEAL` |
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
| sealed runner | `scripts/run_discovery_sealed_1.py` | `PENDING-DESIGN-SEAL` |
| this protocol | `docs/10-sealed-discovery-protocol.md` | recorded externally in the seal record |

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
| PyTorch threads | exactly `1` |
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
90101, 90102, 90103, 90104, 90105, 90106, 90107, 90108, 90109,
90110, 90111, 90112, 90113, 90114, 90115, 90116, 90117, 90118,
90119, 90120, 90121, 90122, 90123, 90124, 90125, 90126, 90127,
90128, 90129, 90130, 90131, 90132, 90133, 90134, 90135, 90136
```

At the time this protocol was drafted, this block was absent from the working
tree and from all Git history (verified by a repository-wide word-boundary
search over every commit; incidental digit collisions inside unrelated hash
strings in `results/experiments/router-v1-sealed-1.json` and
`results/experiments/router-v2-sealed-1.json` are not seed uses, and no
`"seed"` field or seed literal in the block exists anywhere), and no project
run had instantiated or previewed any member. In particular, no one has built
an instance, synthesized an observation, run discovery, computed a residual,
coverage, or misfit, printed a dimension, smoke-tested, or debugged with any
seed in the block. The block is disjoint from the v1 train block
`10001..10200`, the v1 sealed eval block `30101..30136`, the v2 train block
`50001..50200`, the v2 sealed eval block `60101..60136`, the likewise
untouched reserved block `80101..80136`, and every demo seed.

The no-preview rule remains in force until commit B of §12 — the
machine-readable seal record — is committed **and pushed to the private
remote**. Until then:

- tests may use the documented discovery demo seeds (`2026`, `7`, `99`) or
  hand-built fixtures only — runner tests monkeypatch the runner's
  `SEALED_EVAL_SEEDS` constant to such seeds — and may not parameterize
  over, import, derive data from, or otherwise touch the sealed block or any
  other sealed block (`30101..30136`, `60101..60136`, `80101..80136`,
  `90101..90136`);
- source-code constants containing the declared integers (e.g. the runner's
  frozen seed-list constant) are permitted;
- merely calling `make_switch_instance` or `run_discovery` with a declared
  eval seed counts as a preview, even if the result is discarded.

If any declared eval seed is instantiated before commit B is pushed, the
entire 36-seed block is void. It will not be partially salvaged. A new
consecutive block must be chosen, verified absent from history, and declared
in a newly committed protocol before any execution.

## 4. Data and generation protocol

**The seeded task (H1–H3 condition).** For a generator seed `s`, the task is
exactly

```text
universa.discovery.run_discovery(s, num_observations=16, num_classes=6,
                                 misfit_tol=1e-10, novelty_tol=1e-6)
```

with the remaining parameters at their module defaults (`num_vertices=8`,
`num_edges=14`, `num_decoys=3`). Concretely, `run_discovery`:

1. builds the switch instance `make_switch_instance(s, 8, 14, 6, 3)`: a
   random connected source graph, a planted quotient chain map onto
   `num_classes=6` classes (the budgets choice, keeping the quotient
   non-complete so decoys have genuinely different kernels), and 3 certified
   discriminating decoys — all exact integer arithmetic, float64;
2. **withholds the true target**: the library is the 3 decoy boundaries
   only — discovery must propose what routing cannot select;
3. synthesizes `M = 16` transported vectors via `synthesize_observations`:
   column `j` is `f1 a_j` with `a_j` a certified random source cycle drawn
   from `subseed(s, "discovery-observation", str(j), str(attempt))`,
   `attempt` counting from 1, redrawn while `||y_j|| < 1e-9`
   (`MIN_OBSERVATION_NORM`), at most 1000 attempts — every column lies in
   `im(f1) ⊆ ker(B1_true)` with exactness inherited from the planted map;
4. discovers via `discover_constraint(observations, ambient_dim,
   seeds=(s,))` at the module-default tolerance `tol = CERT_TOL = 1e-10`:
   certified SVD accumulation under the `operators.py` rank convention
   (`max(shape) * eps * sigma_0`), the dimensional-stabilization rule with
   `STABILITY_FRACTION = 0.25` (holdout `max(1, floor(16/4)) = 4`; the rank
   of the first 12 observations must equal the rank of all 16), the
   annihilator boundary certified by `nullspace_basis`, and the certificate
   residual `||C_disc Y||_F`; every insufficiency returns
   `DiscoveryInsufficient` — never a weak structure;
5. on success, admits against the decoy library (`misfit_tol=1e-10`,
   `novelty_tol=1e-6`), audits against the withheld truth
   (`discovery_quality`: containment and `coverage_fraction =
   dim(S_disc)/dim(ker B1_true)` — evaluation-only, never available to the
   discovery process), and reports the planted map's misfit against the
   discovered constraint, `||C_disc f1 Z||_F` over the certified source
   cycle basis `Z` — the residual a router would read.

**The structure-free control (H4 condition).** For the same seed `s`, the
null observation set is 16 i.i.d. standard Gaussian vectors on the seed's
target edge space:

```text
ambient_dim = number of edges of the seed's true quotient target
y_j = np.random.default_rng(subseed(s, "discovery-null", str(j)))
        .standard_normal(ambient_dim)        for j = 0..15
Y_null = column stack of y_0..y_15           (ambient_dim x 16, float64)
```

fed through the **same** discovery call,

```text
discover_constraint(Y_null, ambient_dim, seeds=(s,))
```

at the module-default `tol = CERT_TOL`. The null condition reads the instance
only for `ambient_dim`; its random stream is a disjoint subseed family from
the structured condition's. With 16 generic Gaussian columns in an ambient
space of at most 15 dimensions (6-class quotient), the expected certified
behavior is refusal — the span fills the ambient space and the annihilator
is trivial, or stabilization/certification fails first. The per-seed null
record additionally retains the **dimension trajectory**: the numerical rank
of the first `j + 1` columns of `Y_null` for `j = 0..15` (16 integers), under
the same `operators.py` SVD tolerance convention, making the growth-then-refusal
path auditable.

**Deterministic random streams.** Every stochastic component uses
`universa.generators.subseed`:

```text
message = ":".join(["universa", str(seed), *components]).encode("utf-8")
digest  = SHA256(message)
subseed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
```

**Determinism, verified.** The pipeline is a deterministic float64 function
of the seed. The runner executes the full per-seed pipeline — structured and
null conditions — twice for every eligible seed and requires bit-identical
raw rows; any mismatch is a whole-run `design_failure`. Any NaN or
non-finite residual, matrix entry, or statistic, and any violation inside the
certified machinery (rank, orthonormality, or certificate failure), is
likewise a whole-run `design_failure`, never a dropped row.

## 5. Certified deterministic procedure — no learned parameters

**This experiment trains nothing.** There is no model, no fit, no gradient,
no annealing, no model selection, no torch parameter, and no train seed
block (§3). The procedure under evaluation is the certified deterministic
pipeline of §4 — `run_discovery` for the structured condition and the single
`discover_constraint` call for the null condition — pinned end-to-end by the
code manifest of §2. Every quantity it emits is a deterministic float64
numpy function of the generator seed.

Where the router protocols record training provenance, this design records
**procedure provenance** instead: the exact frozen call signature of §4; the
tolerances `misfit_tol=1e-10`, `novelty_tol=1e-6`, and the module-default
discovery `tol = CERT_TOL = 1e-10`; the module constants
`STABILITY_FRACTION = 0.25` and `MIN_OBSERVATION_NORM = 1e-9`; the NumPy
version; and the per-file code-manifest SHA-256s. The result states plainly,
in place of any training record, that no training provenance exists because
no training occurred (§11). A non-finite value at any point is an error,
never a warning.

## 6. Conditions

Exactly two conditions, both applied to every eligible seed:

1. **structured** — the seeded task of §4 (`run_discovery` on the
   decoy-only library, truth withheld): the source of the H1–H3 per-seed
   statistics;
2. **structure-free null** — the `discover_constraint` call on the 16
   i.i.d. Gaussian vectors of §4: the source of the H4 per-seed statistic.

There are no arms and no baselines: the claims are absolute floor/rate
claims about the procedure's behavior over the seed mechanism, not margins
over a comparator, and H4 is the specificity control condition of the same
family. The two conditions share only the seed's instance (the null condition
reads only its target edge-space dimension); their observation streams are
independent subseed families (`"discovery-observation"` vs
`"discovery-null"`). There is no third condition, no eval-time tuning, and
no per-seed choice of any kind.

## 7. Estimands and unit of inference

For each eligible seed `s`, the four frozen per-seed statistics are:

```text
coverage(s) = quality.coverage_fraction        if discovery succeeds
            = 0.0                              if discovery refuses
cert(s)     = 1{discovery succeeds and certificate_residual <= 1e-10}
misfit(s)   = 1{discovery succeeds and map_misfit <= 1e-9}
refusal(s)  = 1{DiscoveryInsufficient returned on the null observations}
```

Declared conventions, fixed here:

- **Refusal counts against H1–H3.** A refusal discovers nothing, so it
  covers 0 of the true kernel, certifies nothing, and admits nothing:
  `coverage(s) = 0.0`, `cert(s) = 0`, `misfit(s) = 0`. The raw row records
  the raw nullable fields (`null` where undefined); the claim statistics
  apply this convention. There is no conditioning on discovery success and
  no deletion of refused seeds.
- **H2 is mechanically the discovery-success rate.** The discovery gate's
  tolerance and the claim threshold are the same constant (`CERT_TOL =
  1e-10`), so every `DiscoveredConstraint` certifies by construction and
  `cert(s) == 1{discovery succeeds}`. This is declared, not discovered after
  the fact: H2 tests that the certified machinery succeeds — and therefore
  certifies — at the required rate over the seed mechanism.
- `misfit(s)` uses the claim threshold `1e-9` on the planted-map misfit
  `||C_disc f1 Z||_F` (the router-acceptance residual), which is distinct
  from the admission gate's `misfit_tol = 1e-10` on the certificate
  residual; both are frozen.

The **generator seed is the sole unit of inference**. It jointly determines
the instance, the withheld library, the structured observation draw, and the
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
| `h1-discovery-coverage-floor` | structured | `coverage` | mean `>= 0.9`; supported iff the one-sided lower bound exceeds `0.9` |
| `h2-discovery-certification-rate` | structured | `cert` | mean `>= 0.95`; supported iff the one-sided lower bound exceeds `0.95` |
| `h3-discovery-admission-ready` | structured | `misfit` | mean `>= 0.95`; supported iff the one-sided lower bound exceeds `0.95` |
| `h4-discovery-false-discovery-control` | null | `refusal` | mean `>= 0.95`; supported iff the one-sided lower bound exceeds `0.95` |

The formal hypotheses are fixed as:

- **H1** (id `h1-discovery-coverage-floor`, primary): `theta =
  E_seed[coverage]`; H0: `theta <= 0.9`; H_A: `theta > 0.9`; supported iff
  the one-sided Bonferroni lower bound is above `0.9`.
- **H2** (id `h2-discovery-certification-rate`): `theta = E_seed[cert]`;
  H0: `theta <= 0.95`; H_A: `theta > 0.95`; supported iff the one-sided
  Bonferroni lower bound is above `0.95`.
- **H3** (id `h3-discovery-admission-ready`): `theta = E_seed[misfit]`; H0:
  `theta <= 0.95`; H_A: `theta > 0.95`; supported iff the one-sided
  Bonferroni lower bound is above `0.95`. A supported H3 means the router
  would accept the discovered structure under the planted transport at the
  required rate.
- **H4** (id `h4-discovery-false-discovery-control`): `theta =
  E_seed[refusal]`; H0: `theta <= 0.95`; H_A: `theta > 0.95`; supported iff
  the one-sided Bonferroni lower bound is above `0.95`. Refusal is the
  correct behavior on structure-free data; a false constraint admitted from
  noise is a specificity failure. This is the false-discovery control of §1,
  not a power claim.

Family-wise type-I error is controlled at `0.05` by Bonferroni over four
claims, one-sided Student-t with `n` eligible seeds:

```text
alpha_per_claim = 0.05 / 4 = 0.0125
q = 1 - alpha_per_claim = 0.9875
lower = mean(statistic) - t_(0.9875, n-1) * SE
```

The direction-specific bound above is the sole support rule for each claim,
applied mechanically to the frozen constants — **including when `SE` is
exactly zero**. The indicator-mean claims (H2–H4, and H1 if every coverage
coincides) can hit this mechanical case: when the sample standard deviation
is `0`, `SE = 0` and the one-sided lower bound **equals the estimate**;
support then reduces to the estimate itself strictly exceeding the
threshold. This case is specified here in advance and is decided by the same
frozen rule, not by discretion.

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

The table is numerically identical to the router experiments' (same alpha,
same eligible-n range); it is hard-coded and tested independently in this
runner. SciPy is not a runtime dependency of the runner. Each of the four
claim summaries records: the estimate (mean of the per-seed statistic); its
sample standard deviation; its standard error; the governing one-sided lower
bound; the critical value; the direction (`greater`); the threshold (`0.9`
for H1, `0.95` for H2–H4); and the support decision. No hypothesis may be
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
- an instance-build exception is a **whole-run** `design_failure`, never an
  exclusion — the offending seed is recorded, the run stops, and no seed is
  ever deleted alone;
- the build of §10 is the sole eligibility gate. Every *other* failed check —
  any determinism, NaN/non-finite value, audit, rank, orthonormality,
  certificate, or validation failure, including any violation inside the
  certified discovery machinery, a nonzero-observation draw failure, and any
  failed result-audit recomputation — is a whole-run `design_failure`;
- fewer than 30 eligible seeds yields `design_failure_insufficient_eligible`
  with **no evaluation and no claims**: the run stops before any discovery
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
- the runner must not compute or print coverages, residuals, rates, or claim
  summaries until every required raw row has completed; progress output is
  limited to seed identifiers, stage, completion state, and timing.

Mandatory stop conditions (any one stops the run before or during execution):

- any preflight mismatch: seal-validation failure (including an unknown seal
  key, a tampered claim field, a `train_seed_block` other than the empty
  sentinel of §3, or a `design_commit` absent from the repository's
  history), hash mismatch, a seal not committed at HEAD, an on-disk seal not
  byte-identical to its HEAD blob, a symlinked seal, protocol, or runner
  file, a dirty worktree, an existing output path, available CUDA, or a
  torch thread count other than one;
- a worktree found dirty, or a code-manifest hash found changed, at the
  end-of-run re-check before publishing a `complete` result;
- a raw-row count that violates the frozen invariant `2 x n` — exactly one
  structured row and one null row per eligible seed;
- an instance-build exception;
- fewer than 30 eligible seeds;
- any determinism, NaN, audit, rank, or validation failure;
- aggregation failure or a non-finite estimand; or
- any manual cancellation or infrastructure interruption.

If infrastructure fails after execution begins, the attempt is preserved
under a distinct non-overwriting failure path
`results/experiments/failures/discovery-sealed-1.<status>.json`; the
canonical output path is reserved for status `complete` and a failed
attempt never occupies it. An exact-code retry is allowed
only for a documented non-scientific failure, before any summaries are
inspected, with identical hashes. Every attempt remains in the evidence
bundle. The first complete, fully validated attempt is canonical; no choice
among complete attempts is allowed. Any code, tolerance, dependency, or
analysis change after a partial run invalidates the seed block and requires a
new protocol and new seeds.

## 10. Eligibility

An eval seed is **eligible** if and only if its instance builds —
`make_switch_instance(seed, 8, 14, 6, 3)` (inside the `run_discovery` call
of §4) returns. A build exception is a whole-run `design_failure` per §9,
never an exclusion. All 36 seeds are attempted, in seed order.

`n`, the number of eligible seeds, is the inference sample size of §7–§8. If
`n < 30`, the run stops before any discovery execution with
`design_failure_insufficient_eligible` and no claims. There are no per-seed
audit gates beyond the build — the planted map commutes exactly by
construction, so there is nothing stochastic to audit per seed — and no
train-side eligibility exists because there are no train rows.

## 11. Result JSON schema

The runner writes exactly one result file, to
`results/experiments/discovery-sealed-1.json`, through a temporary file
followed by an atomic no-clobber hard-link publish (create-if-absent, never
replace — stronger than a rename), and refuses an existing path. Non-
`complete` statuses are published instead to
`results/experiments/failures/discovery-sealed-1.<status>.json` under the
same no-clobber rule, and failure artifacts are written **only** there. The
JSON contains, at minimum:

- `schema`: the literal tag `universa-discovery-sealed-result/1`, and the
  frozen configuration (the exact task call of §4 with all tolerances and
  module constants, `num_observations`, the claim family with thresholds);
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
- all 36 candidate seed records, each with its eligibility verdict;
- raw rows per eligible seed, **structured condition**: `seed`, `verdict`
  (`discovered` / `insufficient`), `reason` (the refusal reason, or the
  admission decision's reason on success), `coverage_fraction`,
  `certificate_residual`, `map_misfit`, `discovered_dim`, `ker_dim`,
  `num_observations`, and the retained matrices — the discovered `boundary`
  (annihilator rows, `(ambient_dim - discovered_dim) x ambient_dim`) and
  `support_basis` (orthonormal columns, `ambient_dim x discovered_dim`) as
  nested float64 arrays. Fields undefined on refusal are `null`;
- raw rows per eligible seed, **null condition**: `seed`, `refusal` (bool),
  `reason`, and the `dimension_trajectory` (16 integers, §4);
- the four claim summaries, each with estimate, sample standard deviation,
  SE, one-sided lower bound, critical value, direction, threshold, and
  `supported` (or `supported: null` for all four on any non-`complete`
  status); and
- an audit block recomputing **every claim statistic from the raw rows
  alone**: row counts, eligible-seed count, the refusal-to-zero coverage
  convention, the indicator recomputation against the frozen thresholds,
  means, standard deviations, SEs, bounds, and decisions.

**The retention improvement, made precise.** Because the raw rows retain the
discovered boundary and support-basis matrices and nothing was trained, the
per-row float quantities are independently recomputable **without re-running
the discovery SVD**: from the retained matrices and the deterministic,
manifest-pinned generator rebuild of the instance and observations,
`certificate_residual = ||C_disc Y||_F`, `coverage_fraction` against
`nullspace_basis(B1_true)`, and `map_misfit = ||C_disc f1 Z||_F` follow
directly. The audit block states both clauses plainly: (a) every claim
statistic and decision is recomputable from the raw rows alone; (b) the
per-row float quantities are recomputable from the retained matrices plus
the deterministic generator rebuild — and, unlike the router experiments, no
hash-pinned model exists or is needed, because nothing was trained.

An independent validator must recompute all summaries solely from the
retained raw rows and fail closed on missing rows, duplicate (seed,
condition) keys, condition imbalance, ineligible seeds carrying rows,
matrices inconsistent with the recorded `discovered_dim`/`ker_dim`, wrong t
constants, or any decision inconsistent with §8.

## 12. The two-commit seal procedure

The following order is mandatory:

1. **Commit A (design commit):** this protocol, the runner
   `scripts/run_discovery_sealed_1.py`, and all its tests — committed without
   instantiating any declared eval seed. Tests use the documented discovery
   demo seeds (`2026`, `7`, `99`) or hand-built fixtures only — runner tests
   monkeypatch the runner's `SEALED_EVAL_SEEDS` constant to such seeds — and
   must cover, at minimum: the `discovery-null` subseed derivation and that
   it drives the per-vector Gaussian draws of the null condition, no
   sealed-seed fixture use (this block and every other sealed block), the
   refusal-to-zero coverage convention and the refusal indicators, the H2
   mechanical success-rate identity, both conditions on hand fixtures, all
   four claim decisions including the `SE = 0` mechanical case (lower bound
   equals the estimate), the hard-coded t table, eligibility and
   insufficient-eligible handling, failure-artifact shape
   (`supported: null`), seal parsing and validation including the empty
   `train_seed_block` sentinel, dirty-worktree refusal, and output-exists
   refusal. The runner embeds this protocol's SHA-256 as a frozen constant —
   initially the fail-closed placeholder `PENDING_PROTOCOL_SHA256`, replaced
   by the real hash before commit A — and never embeds its own hash: it
   computes its runtime SHA-256 and requires equality with the seal's
   `runner_sha256`. This avoids a self-hash cycle.
2. **Compute fingerprints at commit A:** the protocol SHA-256, the runner
   SHA-256, and the per-file SHA-256 of every `src/universa/*.py` (the
   15-file code manifest of §2).
3. **Commit B (seal):** the machine-readable seal record
   `docs/11-discovery-seal.json` with frozen contents:
   - `schema`: the literal tag `universa-seal/4`;
   - `design_commit`: the full hash of commit A;
   - `protocol_sha256`, `runner_sha256`;
   - `code_manifest`: the 17 per-file SHA-256 pairs of §2;
   - `train_seed_block`: exactly `{first: 0, last: 0}` — the documented
     empty-block sentinel of §3 (this experiment trains nothing);
   - `eval_seed_block`: `{first: 90101, last: 90136}`;
   - `no_preview_declaration`: the renewed attestation of §3;
   - `primary_family`: the four claims of §8 verbatim, as four claim objects
     with the eight subfields `id`, `statistic`, `theta`, `null`,
     `alternative`, `bound_direction`, `threshold`, `support_rule` (schema
     `universa-seal/4` claim objects carry `statistic` in place of the
     router seals' `fraction`/`reference`: there is no operating fraction
     and no baseline arm);
   - `stop_rules`: every stop rule of §9; and
   - `output_path`: exactly `results/experiments/discovery-sealed-1.json`.
4. **Push commit B to the private remote before any sealed seed is
   instantiated**, and verify the remote contains it.
5. Confirm a clean worktree and the pushed seal, and only then execute the
   canonical command once:

   ```bash
   env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src python \
     scripts/run_discovery_sealed_1.py \
     --output results/experiments/discovery-sealed-1.json
   ```

   `--seal` defaults to `docs/11-discovery-seal.json` and therefore does not
   appear; no hash is ever passed on the command line.
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
(`git merge-base --is-ancestor <design_commit> HEAD`), equality of each
embedded hash with both the runner's frozen constants and the actual file
bytes, equality of the runner's own runtime SHA-256 with `runner_sha256`,
equality of every `code_manifest` hash with the actual file bytes, the seal
file's presence in the HEAD commit and byte-identity of the on-disk seal
with its HEAD blob, no symlink at the seal, protocol, or runner path, a
clean worktree per `git status --porcelain --untracked-files=all`, and equality of
`output_path` with the `--output` argument. Only after all of this — and
after setting and verifying one PyTorch thread and a hidden/absent CUDA — may
the runner attempt the 36 instance builds, apply eligibility, and (only if at
least 30 seeds are eligible) execute the structured and null conditions for
every eligible seed, verify determinism bit-identically, aggregate, and write
the result. Before publishing a `complete` result it re-checks the clean
worktree and the code-manifest hashes and records the post-run status. It
records `design_commit` (commit A) and HEAD at execution time as the
execution revision.
