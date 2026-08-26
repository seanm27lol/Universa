# Frozen-design protocol: sealed evaluation of structure router v1 under degraded observation

Status: **complete and freeze-ready, but not yet sealed or executable**. This
document becomes immutable only when the runner, tests, and design-seal record
described in §12 have been committed. Until then, the declared eval seeds must
not be instantiated. After sealing, any substantive change creates a new
experiment with a new untouched seed block; it must not be called
router-v1-sealed-1.

Protocol date: 2026-08-26.
Experiment id: `universa-router-v1-sealed-1`.

## 1. Purpose and precise claim scope

This experiment is an untouched-seed evaluation of the Universa structure
router v1 (`universa.router`, degraded-regime half) in the degraded-observation
regime, on the graph-quotient switch-instance family of
`universa.budgets.make_budget_instance` **only**. The question is whether one
learned router, trained on low degradation fractions, beats the polluted
argmin observed-residual oracle at held-out higher degradation fractions, on
generator seeds neither side has ever seen, under a comparison whose
asymmetry is declared below in full.

The sealed claims are about the margin under this exact design and nothing
else. In particular this experiment cannot establish:

- anything about the other Universa families (2-complexes, cellular sheaves,
  category nerves) or about transfer across families;
- anything about masked (missing-edge) observation — masking is always off
  here — or about regimes without an exact fraction-0 anchor;
- anything about real data, real graphs, or any deployed system; or
- anything about a fair-information comparison (see the caveat).

All prior numbers — the demo script `examples/router_demo.py`, the v1 figures
quoted in `docs/00-design.md` §4, and any engineering check — are demo-scale
sanity evidence only. They are not evidence for or against any claim here;
only the sealed run decides the four claims of §8.

**The asymmetric-comparison caveat, declared in full.** The comparison is
asymmetric by design. The learned router reads the whole 18-dimensional
degradation profile of each candidate — the observed commutation misfit at
every fraction of the grid 0.0..0.7, anchored at the *exact* fraction-0
residual, plus the profile slopes and structural dims. The oracle reads only
the single polluted operating-fraction column of the same profile. This is
the design, not a defect to be apologized for after the fact: the sealed
claims of §8 are claims about the margin *under this asymmetry*. A supported
claim means the learned router exploits information the myopic oracle is
structurally denied; it does not mean the router wins a symmetric contest,
and no such symmetric claim is made anywhere in this experiment.

## 2. Immutable inputs and hashes

Known immutable inputs at protocol drafting are the Universa code manifest —
one SHA-256 per file of `src/universa/*.py` at the design commit — the
runner, and this protocol. No fingerprint is available at draft time: the
design commit does not exist yet. Every entry below is the literal
placeholder **`PENDING-DESIGN-SEAL`**, to be replaced by actual 64-hex
fingerprints in the machine-readable seal record of §12 (commit B), computed
at the design commit (commit A). The runner refuses to start while any
placeholder remains.

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
| structure router v0/v1 | `src/universa/router.py` | `PENDING-DESIGN-SEAL` |
| cellular sheaf family | `src/universa/sheaves.py` | `PENDING-DESIGN-SEAL` |
| chain-complex format | `src/universa/structures.py` | `PENDING-DESIGN-SEAL` |
| sealed runner | `scripts/run_router_v1_sealed_1.py` | `PENDING-DESIGN-SEAL` |
| this protocol | `docs/01-sealed-router-v1-protocol.md` | recorded externally in the seal record |

This protocol's own SHA-256 cannot appear inside itself — a file cannot
contain its own hash — so it is recorded externally in the seal record
(`protocol_sha256`, §12) and embedded as a frozen constant in the runner.

The runner must fail before any instance construction unless every fingerprint
above matches. It must also require this execution environment, recorded in
full in the result:

| component | required value |
|---|---|
| tensor device and dtype | CPU, `torch.float32` |
| PyTorch threads | exactly `1` |
| CUDA | unavailable or hidden (`torch.cuda.is_available()` is `False`) |

The runner records the Python, PyTorch, and NumPy versions, the operating
system and machine, `CUDA_VISIBLE_DEVICES`, the thread setting, the Git
revision, the initial `git status --porcelain --untracked-files=all` (which must be empty), every
verified hash, and the canonical command. No dependency may be installed or
upgraded between the seal and the run.

## 3. Seed blocks and the no-preview declaration

**Train block (declared, unsealed):** the 200 consecutive generator seeds

```text
10001..10200
```

This block is disjoint from all prior demo seeds — `1000..1031` (demo train)
and `2000..2015` (demo eval), both in `examples/router_demo.py` — and from
the sealed eval block. Train seeds may be used freely in tests, previews,
and debugging, before and after the seal.

**Sealed eval block:** the 36 consecutive generator seeds

```text
30101, 30102, 30103, 30104, 30105, 30106, 30107, 30108, 30109,
30110, 30111, 30112, 30113, 30114, 30115, 30116, 30117, 30118,
30119, 30120, 30121, 30122, 30123, 30124, 30125, 30126, 30127,
30128, 30129, 30130, 30131, 30132, 30133, 30134, 30135, 30136
```

At the time this protocol was drafted, this block was absent from the working
tree and from all Git history (verified by a repository-wide search over
every commit), and no project run had instantiated or previewed any member.
In particular, no one has built an instance, computed a residual, drawn an
observation, trained or evaluated a model, printed a dimension, smoke-tested,
or debugged with any seed in the block.

The no-preview rule remains in force until commit B of §12 — the
machine-readable seal record — is committed **and pushed to the private
remote**. Until then:

- tests may use train seeds or hand-built fixtures only; they may not
  parameterize over, import, derive data from, or otherwise touch the sealed
  block;
- source-code constants containing the declared integers (e.g. the runner's
  frozen seed-list constant) are permitted;
- merely calling `make_budget_instance` with a declared eval seed counts as a
  preview, even if the resulting instance is discarded.

If any declared eval seed is instantiated before commit B is pushed, the
entire 36-seed block is void. It will not be partially salvaged. A new
consecutive block must be chosen, verified absent from history, and declared
in a newly committed protocol before any execution.

## 4. Data and generation protocol

**Instances.** For a generator seed `s`, the switch instance is exactly

```text
make_budget_instance(s, num_vertices=8, num_edges=14,
                     num_classes=6, num_decoys=3)
```

from `universa.budgets`: a source graph, a planted quotient chain map, and
`K = 4` candidates — the true target plus 3 certified decoys — with the true
target at candidate index 0 pre-permutation. All exact suite mathematics
(commutation residuals, nullities) is the certified float64 numpy machinery;
torch sees only the resulting feature vectors.

**Observation degradation.** All observation is through
`universa.partial.ObservationModel` with sign-corruption only:
`mask_fraction = 0.0` always and everywhere, so every edge column is kept
and `corrupt_fraction` is the sole degradation knob.

**The frozen feature.** The per-candidate feature vector is exactly the
18-dimensional degradation-profile layout of
`universa.router.build_degraded_dataset` over the default profile grid
`0.0, 0.1, ..., 0.7` (8 points): `log1p` of the observed commutation misfit
of the planted chain map against the observed candidate boundary at each of
the 8 grid fractions, then the 7 consecutive profile slopes, then the 3
structural dims (`num_target_vertices`, `num_target_edges`,
`boundary_cycle_nullity`). At grid fraction 0 the observation is lossless
(zero corruptions), so column 0 is `log1p` of the exact commutation residual.

**Deterministic random streams.** Every stochastic component uses
`universa.generators.subseed`:

```text
message = ":".join(["universa", str(seed), *components]).encode("utf-8")
digest  = SHA256(message)
subseed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
```

**Train rows.** One row per (seed, fraction) over train seeds `10001..10200`
x operating fractions `{0.0, 0.1, 0.2, 0.3, 0.4}` — 1000 rows — built by
`build_degraded_dataset` with all defaults: one observation draw per
(seed, fraction) shared across candidates
(`subseed(seed, "router-v1-observe", key)` with `key = f"{fraction:.6f}"`),
and the per-row candidate permutation from
`subseed(seed, "router-v1-permutation", key)`, with the permuted true index
recorded as the audited label.

**Eval rows and pairing.** For each eval seed and each operating fraction in
`{0.0, 0.5, 0.6, 0.7}`, exactly `R = 4` replicate observation draws, indexed
`replicate_index` in `{0, 1, 2, 3}`, with

```text
observation_seed = subseed(seed, "sealed-replicate",
                           str(fraction), str(replicate_index))
```

where `str()` is Python's built-in float/int string conversion (e.g.
`str(0.6) == "0.6"`, so one literal message is
`"universa:30101:sealed-replicate:0.6:2"`). Each replicate row is the full
18-dimensional feature block built under that observation seed, exactly as
`build_degraded_dataset` builds a row: one draw shared across the 4
candidates, nested over the profile grid. The operating fraction names the
profile column the oracle reads (§6). The per-(seed, fraction) candidate
permutation is exactly as in `build_degraded_dataset` —
`subseed(seed, "router-v1-permutation", f"{fraction:.6f}")` — so all four
replicates of one (seed, fraction) share one permutation, and the permuted
true index is recorded as the label with `permutation[true_index] == 0`.
Both arms read the **same** replicate rows: the replicate draws are shared
across arms, making the comparison paired at the level of individual
observation realizations.

**Audited labels.** Every built row is audited fail-closed at its fraction-0
column (the exact residual): the true candidate's misfit must be at most
`1e-9` and every decoy's strictly above `1e-9`. Because a zero corrupt
fraction flips nothing, the fraction-0 column is observation-independent and
this audit is a deterministic per-seed property, identical across replicates
and arms.

## 5. Model and training

Exactly one model is trained, inside the sealed runner, after the seal, using
only train seeds:

```text
StructureRouter(feature_dim=18, hidden_dim=64)
```

the shared per-candidate MLP `18 -> 64 -> 64 -> 1` of `universa.router`,
trained by `universa.router.train_router` on the 1000 train rows with:

- `epochs=150` (full-batch Adam: every epoch is one step, no shuffling RNG);
- `lr=1e-3`, PyTorch `torch.optim.Adam` defaults for every unspecified option;
- torch seed `4242` (`torch.manual_seed(4242)`, governing initialization);
- `lambda_aux=0.01` (Switch-style load-balancing pressure);
- temperature annealed exponentially `tau: 2.0 -> 0.25`
  (`anneal_temperature`: exactly `2.0` at epoch 0, exactly `0.25` from epoch
  149 on, monotone non-increasing);
- CPU, `torch.float32`, exactly one PyTorch thread;
- input standardization statistics (mean and population std, clamped at
  `1e-8`) measured on the **train block only** and stored in the model's
  buffers, so the model is self-contained at inference.

The validation argument of `train_router` is the training block itself. It
feeds only the `val_accuracy` history diagnostic — no early stopping, no
model selection, no gradient — and therefore cannot influence the fitted
parameters; this choice uses only train seeds. A non-finite loss at any epoch
is an error, never a warning.

Inference is always strictly discrete: argmax over candidate logits
(`hard_predictions`), never sampled or mixed.

Training provenance recorded in the result: the torch seed, the epoch count,
the final total loss, final cross-entropy, final auxiliary loss, final train
hard accuracy, and the model state SHA-256 — computed over the canonical
concatenation, in lexicographic name order, of each `state_dict` entry's
UTF-8 name, a NUL byte, and its little-endian float32 bytes.

## 6. Arms

Exactly two arms, both applied to every eval row of every eligible seed:

1. **learned** — the single trained model of §5, hard argmax inference,
   reading the whole 18-dimensional feature of each candidate;
2. **oracle** — the polluted argmin observed-residual oracle: argmin over
   candidates of the observed misfit at the operating fraction, i.e. argmin
   of the row's profile column at that fraction (log1p is strictly
   increasing, so this is exactly argmin of the misfit). No learning.

At operating fraction 0.0 the oracle reads the exact fraction-0 column and is
the clean-residual oracle; at 0.5/0.6/0.7 it reads a polluted column. Both
arms read the same replicate rows (§4), so each (seed, fraction, replicate)
yields one paired correctness pair. There is no third arm, no ensemble, no
eval-time tuning.

## 7. Estimands and unit of inference

For each eligible seed `s` and each operating fraction `f` in
`{0.0, 0.5, 0.6, 0.7}`:

```text
learned_acc(s, f) = mean over the R=4 replicates of learned_correct
oracle_acc(s, f)  = mean over the R=4 replicates of oracle_correct
d(s, f)           = learned_acc(s, f) - oracle_acc(s, f)
```

where `learned_correct`/`oracle_correct` are 0/1 per replicate row, so each
accuracy is a multiple of `0.25` and `d` is a multiple of `0.25` in
`[-1, 1]`.

The **generator seed is the sole unit of inference**. It jointly determines
the instance, the candidate permutation, and — through the `sealed-replicate`
subseed schedule — the four shared observation draws per operating fraction;
the paired contrast shares all of these within a seed. Replicates are not
independent units and are never pooled as such.

Each claim's estimand is `theta_f = E_seed[d(., f)]` over the eligible-seed
mechanism, estimated by the arithmetic mean of `d(., f)` across the `n`
eligible seeds, with

```text
SE = sample_standard_deviation(d(., f)) / sqrt(n)
```

(sample standard deviation, `n - 1` denominator). Student-t inference assumes
eligible seed-level realizations are exchangeable draws from this
deterministic generator-and-subseed scheme; it does not separate instance
heterogeneity from observation-realization heterogeneity, and it conditions
on the frozen eligibility rule of §10. Intervals quantify Monte Carlo
variation over the seed mechanism, not uncertainty for real data.

## 8. The four frozen claims and decision rules

The following four and only these four claims form the confirmatory family.
H1 is primary; H2–H4 are secondary members of the same family. All claims
are about the margin under the asymmetric comparison declared in §1.

| id | fraction | prediction and governing decision |
|---|---:|---|
| `h1-held-out-0.6-primary` | 0.6 | learned beats oracle; one-sided lower bound `> 0` |
| `h2-held-out-0.5` | 0.5 | learned beats oracle; one-sided lower bound `> 0` |
| `h3-held-out-0.7` | 0.7 | learned beats oracle; one-sided lower bound `> 0` |
| `h4-clean-anchor-bounded-harm` | 0.0 | bounded harm; one-sided lower bound `> -0.05` |

With `theta_f = E_seed[d(., f)]`, the formal hypotheses are fixed as:

- **H1** (id `h1-held-out-0.6-primary`): `theta = E_seed[d(., 0.6)]`; H0:
  `theta <= 0`; H_A: `theta > 0`; reference: the polluted oracle at fraction
  0.6; supported iff the one-sided Bonferroni lower bound is above `0`.
- **H2** (id `h2-held-out-0.5`): `theta = E_seed[d(., 0.5)]`; H0:
  `theta <= 0`; H_A: `theta > 0`; reference: the polluted oracle at fraction
  0.5; supported iff the one-sided Bonferroni lower bound is above `0`.
- **H3** (id `h3-held-out-0.7`): `theta = E_seed[d(., 0.7)]`; H0:
  `theta <= 0`; H_A: `theta > 0`; reference: the polluted oracle at fraction
  0.7; supported iff the one-sided Bonferroni lower bound is above `0`.
- **H4** (id `h4-clean-anchor-bounded-harm`): `theta = E_seed[d(., 0.0)]`;
  H0: `theta <= -0.05`; H_A: `theta > -0.05`; reference: the exact
  fraction-0 oracle; supported iff the one-sided Bonferroni lower bound is
  above `-0.05`. This is a bounded-harm statement at one prespecified margin
  against the clean anchor: it rules out harm of 5 accuracy points or more.
  It is not an equivalence test and cannot establish absence of every harm.

Family-wise type-I error is controlled at `0.05` by Bonferroni over four
claims, one-sided Student-t with `n` eligible seeds:

```text
alpha_per_claim = 0.05 / 4 = 0.0125
q = 1 - alpha_per_claim = 0.9875
lower = mean(d) - t_(0.9875, n-1) * SE
```

The direction-specific bound above is the sole support rule for each claim,
applied mechanically to the frozen constants — including when `SE` is exactly
zero. The hard-coded one-sided critical values, computed once with SciPy for
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

The runner must contain and test this table; SciPy is not a runtime
dependency of the runner. Each of the four claim summaries records: the
estimate `mean(d)`; its standard error; the governing one-sided lower bound;
the critical value; the direction (`greater`); the threshold (`0` or
`-0.05`); and the support decision. No hypothesis may be added, dropped,
reversed, re-thresholded, or moved between families after execution. All
nulls and failed predictions are reported.

## 9. Stop rules and status vocabulary

The run's terminal status is exactly one of:

- `complete` — every stage finished and all validation passed; the only
  status that permits non-null claim decisions;
- `design_failure` — any frozen-design violation;
- `design_failure_insufficient_eligible` — fewer than 30 eligible seeds;
- `execution_failure` — any other unexpected exception;
- `interrupted` — `KeyboardInterrupt` or equivalent manual cancellation.

The mapping is frozen:

- all 36 eval seeds are attempted: instance builds and eligibility audits run
  in seed order, and no seed is ever skipped for outcome-dependent reasons
  short of a build failure;
- an instance-build exception is a **whole-run** `design_failure`, never an
  exclusion — the offending seed is recorded, the run stops, and no seed is
  ever deleted alone;
- the fraction-0 audit of §4/§10 is the sole eligibility gate: its failure
  makes that one seed ineligible (recorded, with its reason). Every *other*
  failed check — any determinism, NaN/non-finite feature or residual, audit,
  rank, or validation failure, including any violation inside the certified
  feature machinery and any failed result-audit recomputation — is a
  whole-run `design_failure`;
- fewer than 30 eligible seeds yields `design_failure_insufficient_eligible`
  with **no fits and no claims**: the run stops before training;
- a caught failure preserves every completed raw row, identifies the failing
  seed and stage, records the exception type and message and the preflight
  provenance, and emits all four claims with `supported: null`;
- there is no outcome-dependent stopping, no interim analysis, no sample-size
  extension, and no arm- or seed-level deletion;
- the run is never rerun because the results are surprising, weak, null, or
  apparently decisive;
- the runner must not compute or print accuracies, differences, or claim
  summaries until every required raw row has completed; progress output is
  limited to seed identifiers, stage, completion state, and timing.

Mandatory stop conditions (any one stops the run before or during execution):

- any preflight mismatch: seal-validation failure (including an unknown seal
  key, a tampered claim field, or a `design_commit` absent from the
  repository's history), hash mismatch, a seal not committed at HEAD, an
  on-disk seal not byte-identical to its HEAD blob, a symlinked seal,
  protocol, or runner file, a dirty worktree, an existing output path,
  available CUDA, or a torch thread count other than one;
- a worktree found dirty, or a code-manifest hash found changed, at the
  end-of-run re-check before publishing a `complete` result;
- a raw-row count that violates the frozen invariant
  `R x |eval fractions| x |eligible seeds|`;
- an instance-build exception;
- fewer than 30 eligible seeds;
- any determinism, NaN, audit (other than the per-seed eligibility gate),
  rank, or validation failure;
- aggregation failure or a non-finite estimand; or
- any manual cancellation or infrastructure interruption.

If infrastructure fails after execution begins, the attempt is preserved
under a distinct non-overwriting failure path
`results/experiments/failures/router-v1-sealed-1.<status>.json`; the
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

1. its instance builds — `make_budget_instance(seed, 8, 14, 6, 3)` returns
   (a build exception is a whole-run `design_failure` per §9, never an
   exclusion); and
2. the fraction-0 audit passes: the true candidate's misfit at profile
   fraction 0 is `<= 1e-9` and every decoy's is `> 1e-9`. As noted in §4,
   this audit is observation-independent and therefore a deterministic
   per-seed property.

Ineligible seeds are recorded with their explicit reasons and are never
replaced, never deleted, and never scored. `n`, the number of eligible
seeds, is the inference sample size of §7–§8. If `n < 30`, the run stops
before training with `design_failure_insufficient_eligible`. Train rows are
built with `build_degraded_dataset`'s own fail-closed audits; any violation
there raises and is a whole-run `design_failure`, never a dropped row.

## 11. Result JSON schema

The runner writes exactly one result file, to
`results/experiments/router-v1-sealed-1.json`, through a temporary file
followed by an atomic no-clobber hard-link publish (create-if-absent, never
replace — stronger than a rename), and refuses an existing path. Non-
`complete` statuses are published instead to
`results/experiments/failures/router-v1-sealed-1.<status>.json` under the
same no-clobber rule. The JSON contains,
at minimum:

- `schema`: the literal tag `universa-router-v1-sealed-result/1`, and the
  frozen configuration (instance constants, profile grid, fraction blocks,
  `R`, training constants, claim family);
- the seal SHA-256, protocol SHA-256, runner SHA-256, and every code-manifest
  SHA-256, plus `design_commit` (commit A) and the execution revision (HEAD
  at execution time);
- environment and provenance: Python/PyTorch/NumPy versions, OS and machine,
  thread count, the operator-provided and effective `CUDA_VISIBLE_DEVICES`,
  the recorded `git status --porcelain --untracked-files=all` output (key
  `git_status_porcelain`, must be empty, recorded pre- and post-run), and
  `sys.argv`;
- training provenance: torch seed, epochs, final total loss, final
  cross-entropy, final auxiliary loss, final train hard accuracy, and the
  model state SHA-256 (canonical form per §5);
- all 36 candidate seed records, each with its eligibility verdict and
  explicit ineligibility reason where applicable;
- raw rows per (seed, fraction, replicate) for every eligible seed: `seed`,
  `fraction`, `replicate_index`, `learned_correct`, `oracle_correct`, the
  per-candidate observed residuals at the operating fraction (float64, in
  permuted candidate order), the `permutation`, and `true_index`;
- per-(seed, fraction) accuracies `learned_acc`/`oracle_acc` and paired
  differences `d`, for all four operating fractions;
- the four claim summaries, each with estimate, SE, one-sided lower bound,
  critical value, direction, threshold, and `supported` (or
  `supported: null` for all four on any non-`complete` status); and
- an audit block recomputing the aggregation from the raw rows alone:
  row counts, eligible-seed count, per-fraction means, SEs, bounds, and
  decisions. Eligibility counts and seed ids come from the eligibility pass,
  and the learned arm's per-row bits are checkable only with the retained
  hash-pinned model; the audit block's contract is that every summary of the
  raw rows is independently recomputable from them.

An independent validator must recompute all summaries solely from the
retained raw rows and fail closed on missing rows, duplicate (seed, fraction,
replicate) keys, arm imbalance, ineligible seeds carrying rows, wrong t
constants, or any decision inconsistent with §8.

## 12. The two-commit seal procedure

The following order is mandatory:

1. **Commit A (design commit):** this protocol, the runner
   `scripts/run_router_v1_sealed_1.py`, and all its tests — committed without
   instantiating any declared eval seed. Tests use train seeds or hand
   fixtures only and must cover, at minimum: the `sealed-replicate` subseed
   derivation, no sealed-seed fixture use, the pairing of replicate draws
   across arms, the permutation and audited-label contract, both arms on hand
   fixtures, all four claim decisions, the hard-coded t table, eligibility
   and insufficient-eligible handling, failure-artifact shape
   (`supported: null`), seal parsing and validation, dirty-worktree refusal,
   and output-exists refusal. The runner embeds this protocol's SHA-256 as a
   frozen constant — initially the fail-closed placeholder
   `PENDING_PROTOCOL_SHA256`, replaced by the real hash before commit A — and
   never embeds its own hash: it computes its runtime SHA-256 and requires
   equality with the seal's `runner_sha256`. This avoids a self-hash cycle.
2. **Compute fingerprints at commit A:** the protocol SHA-256, the runner
   SHA-256, and the per-file SHA-256 of every `src/universa/*.py` (the code
   manifest of §2).
3. **Commit B (seal):** the machine-readable seal record
   `docs/02-router-v1-seal.json` with frozen contents:
   - `schema`: the literal tag `universa-seal/1`;
   - `design_commit`: the full hash of commit A;
   - `protocol_sha256`, `runner_sha256`;
   - `code_manifest`: the 13 per-file SHA-256 pairs of §2;
   - `train_seed_block`: `{first: 10001, last: 10200}`;
   - `eval_seed_block`: `{first: 30101, last: 30136}`;
   - `no_preview_declaration`: the renewed attestation of §3;
   - `primary_family`: the four claims of §8 verbatim, as four claim objects
     (id, fraction, theta, null, alternative, reference, bound direction,
     threshold, support rule);
   - `stop_rules`: every stop rule of §9; and
   - `output_path`: exactly `results/experiments/router-v1-sealed-1.json`.
4. **Push commit B to the private remote before any sealed seed is
   instantiated**, and verify the remote contains it.
5. Confirm a clean worktree and the pushed seal, and only then execute the
   canonical command once:

   ```bash
   env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src python3 \
     scripts/run_router_v1_sealed_1.py \
     --output results/experiments/router-v1-sealed-1.json
   ```

   `--seal` defaults to `docs/02-router-v1-seal.json` and therefore does not
   appear; no hash is ever passed on the command line.
6. Independently validate the retained raw rows (§11) before editing any
   result prose, then commit the immutable result artifact.

The runner accepts a `--seal` argument defaulting to that path. It parses and
validates the machine-readable seal file rather than trusting a naked
command-line hash; there is no `--expected-runner-sha256`-style flag.
Validation requires the schema tag and every field above with no unknown
keys, each claim object checked field-by-field against the runner's frozen
claim definitions, `design_commit` present in the repository's history
(`git cat-file -e <design_commit>^{commit}`), equality of each
embedded hash with both the runner's frozen constants and the actual file
bytes, equality of the runner's own runtime SHA-256 with `runner_sha256`,
equality of every `code_manifest` hash with the actual file bytes, the seal
file's presence in the HEAD commit and byte-identity of the on-disk seal
with its HEAD blob, no symlink at the seal, protocol, or runner path, a
clean worktree per `git status --porcelain --untracked-files=all`, and equality of
`output_path` with the `--output` argument. Only after all of this — and
after setting and verifying one PyTorch thread and a hidden/absent CUDA — may
the runner attempt the 36 instance builds, apply eligibility, and (only if at
least 30 seeds are eligible) build train rows, train the single model of §5,
build eval replicate rows for eligible seeds, score both arms, aggregate, and
write the result. Before publishing a `complete` result it re-checks the
clean worktree and the code-manifest hashes and records the post-run status.
It records `design_commit` (commit A) and HEAD at execution
time as the execution revision.
