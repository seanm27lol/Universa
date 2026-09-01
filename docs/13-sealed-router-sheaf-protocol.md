# Frozen-design protocol: sealed evaluation of the structure router on cellular-sheaf switch instances in the no-anchor degraded regime

Status: **complete and freeze-ready, but not yet sealed or executable**. This
document becomes immutable only when the runner, tests, and design-seal record
described in §12 have been committed. Until then, the declared eval seeds must
not be instantiated. After sealing, any substantive change creates a new
experiment with a new untouched seed block; it must not be called
router-sheaf-sealed-1.

Protocol date: 2026-08-31.
Experiment id: `universa-router-sheaf-sealed-1`.

## 1. Purpose and precise claim scope

This experiment is an untouched-seed evaluation of the Universa structure
router (the model and training machinery of `universa.router`, in the
no-anchor degraded regime established by the sealed v2 experiment) on the
cellular-sheaf switch-instance family of
`universa.sheaves.make_sheaf_switch_instance` **only**. It is the fifth
sealed experiment: the sealed v1 experiment
(`docs/01-sealed-router-v1-protocol.md`, results in
`docs/03-router-v1-sealed-1-results.md`) established the anchored regime on
graph quotients; the sealed v2 experiment
(`docs/04-sealed-router-v2-protocol.md`, results in
`docs/06-router-v2-sealed-1-results.md`) removed the exact fraction-0 anchor
on the same graph-quotient family; the sealed 2-complex experiment
(`docs/07-sealed-router-2complex-protocol.md`, results in
`docs/09-router-2complex-sealed-1-results.md`) carried the no-anchor regime
to 2-complexes; the sealed discovery experiment
(`docs/10-sealed-discovery-protocol.md`, results in
`docs/12-discovery-sealed-1-results.md`) evaluated the certified discovery
procedure, with no learned model, on graph quotients. This experiment keeps
the v2 no-anchor regime — `mask_fraction = 0.25` always, corruption sweeping
over a profile grid that excludes 0.0, so no feature anywhere is computed
against a clean structure — and **changes the family**: targets are now
cellular sheaves on graphs, where the constraint lives in the **stalk
restriction maps**, not a plain boundary. A sheaf attaches a vector space to
every vertex and every edge and a restriction map `rho_{v,e}` to every
incidence; a sheaf morphism is natural when
`rho'_{v,e} phi_v = phi_e rho_{v,e}` at every incidence. Observation is
through `universa.partial_sheaf.SheafObservationModel` with masking and
sign-corruption both active (`mask_fraction = 0.25` always,
`corrupt_fraction == g` at grid fraction `g`), and the frozen score is the
**observed naturality residual** of the planted morphism against the
observed candidate sheaf. The question is whether one learned router,
trained on low degradation fractions, beats the polluted argmin
naturality-residual oracle at held-out higher degradation fractions — and
whether it also beats a non-learned whole-profile heuristic — on generator
seeds neither side has ever seen, under a comparison whose asymmetry is
declared below in full.

The sealed claims are about the margin under this exact design and nothing
else. In particular this experiment cannot establish:

- anything about the other Universa families (graph quotients, 2-complexes,
  category nerves) or about transfer across families — the family change
  cuts both ways, and the v1/v2 graph-quotient, 2-complex, and discovery
  results stand separately;
- anything about regimes with an exact fraction-0 anchor — that was the v1
  experiment — or about masking-only or corruption-only degradation
  (masking is fixed at `0.25` and corruption sweeps; the two knobs'
  effects are not separated here — see the regime note below);
- anything about real data, real sheaves, or any deployed system; or
- anything about a fair-information comparison (see the caveat).

All prior numbers — the demo scripts, the figures quoted in
`docs/00-design.md`, the documented ranking-study behavior of
`universa.partial_sheaf` (pinned study seed `70001`), the four sealed
results, and any engineering check — are demo-scale sanity evidence only.
They are not evidence for or against any claim here; only the sealed run
decides the four claims of §8.

**The row-specific draws, retained from the v2 sealed design.** Each data
row — one (seed, operating fraction, replicate) triple — gets its own
observation draw (§4), exactly as the v2 sealed design mandated: features
are row-specific and replicates are genuine, distinct observation
realizations. There is no per-instance shared-draw default for this family
and none is introduced; the sealed schedule below is the only schedule.

**The asymmetric-comparison caveat, declared in full.** The comparison
remains asymmetric by design, unchanged from the v2 and 2-complex
experiments. The learned router reads the whole 18-dimensional degraded
profile of each candidate — the observed naturality residual of the planted
morphism at every fraction of the grid 0.2..0.9 (every column polluted:
masked and sign-corrupted, with no exact anchor anywhere), plus the profile
slopes and the masked structural dims. The oracle reads only the single
polluted operating-fraction column of the same profile. This is the design,
not a defect to be apologized for after the fact: the sealed claims H1–H3 of
§8 are claims about the margin *under this asymmetry*. A supported claim
means the learned router exploits information the myopic oracle is
structurally denied; it does not mean the router wins a symmetric contest,
and no such symmetric claim is made anywhere in this experiment. H4 keeps
the strengthened baseline side of the v2 and 2-complex experiments: it
requires the learned router to also beat the non-learned grid-mean profile
heuristic, a comparator that reads the same whole-profile shape with uniform
integration and no learning. Even a supported H4 establishes only
superiority over that one frozen non-learned comparator under this design —
still not a fair-information claim.

**The sheaf regime note, declared.** `universa.partial_sheaf` documents a
structural asymmetry between its two degradation modes, and it is part of
this design's declared caveats: **masking alone never grows the true
candidate's residual** — the observed naturality residual runs over the
surviving incidences only, and every surviving restriction block of the true
target is exact, so under masking the true candidate stays at exactly `0.0`
while each decoy's residual erodes monotonically as its discriminating terms
drop out (a coverage study whose margin never grows — unlike
`universa.partial`'s mask mode, where the source side stays fully observed
and the true residual rises). **The discrimination pressure in this
experiment comes from the corruption mode**: sign flips of surviving entries
are what move the true candidate's residual off zero (structurally monotone
in the flip count, by the scaled-permutation orthogonality argument of
`universa.partial_sheaf`), with masking fixed at `0.25` reducing coverage —
the residual is read over fewer incidences at every grid point. The two
knobs' effects are not separated by this design, and no claim here is a
claim about masking-only or corruption-only degradation.

**The retention improvement, continued.** The 2-complex experiment fixed the
v2 result's declared recompute limit (raw rows retained only the
operating-fraction residuals) by retaining the full 16-value two-degree
profile per candidate. This experiment continues that improvement by
construction: every raw row retains, per candidate, the **full 8-value
profile** — the observed naturality residual at all 8 grid points — plus
the operating-fraction residual (§11). The oracle arm at any grid operating
fraction, the heuristic arm, and every profile statistic (slopes, grid
means) are therefore recomputable from the raw rows alone, with no feature
machinery involved.

## 2. Immutable inputs and hashes

Known immutable inputs at protocol drafting are the Universa code manifest —
one SHA-256 per file of `src/universa/*.py` at the design commit — the
runner, and this protocol. No fingerprint is available at draft time: the
design commit does not exist yet. Every entry below is the literal
placeholder **`PENDING-DESIGN-SEAL`**, to be replaced by actual 64-hex
fingerprints in the machine-readable seal record of §12 (commit B), computed
at the design commit (commit A). The runner refuses to start while any
placeholder remains.

The manifest has 17 entries: the same 17 files sealed in the 2-complex and
discovery experiments (the 15 files sealed in the v2 experiment — the 13
files of the v1 experiment plus `partial2.py` and `router_v2.py` — plus
`partial_group.py` and `partial_sheaf.py`); no file is added or removed.
The v1, v2, 2-complex, and discovery seals and their manifests remain the
frozen records of those experiments and are not modified by this one.

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
| sealed runner | `scripts/run_sheaf_sealed_1.py` | `PENDING-DESIGN-SEAL` |
| this protocol | `docs/13-sealed-router-sheaf-protocol.md` | recorded externally in the seal record |

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
11001..11200
```

This block is disjoint from the sealed eval block below, from the v1 train
block `10001..10200` and sealed eval block `30101..30136`, from the v2 train
block `50001..50200` and sealed eval block `60101..60136`, from the
2-complex train block `70001..70200` and sealed eval block `80101..80136`,
from the discovery sealed eval block `90101..90136` (the discovery
experiment has no train block), from the demo seeds `1000..1031` and
`2000..2015` (`examples/router_demo.py`), and from the pinned study seeds
`2026` (`universa.partial2`) and `70001` (`universa.partial_sheaf`, itself
inside the 2-complex train block). It was verified absent from the working
tree and from all Git history at draft time (a repository-wide word-boundary
search over the tree and every commit): no member of this block has ever
been instantiated by any project run, test, or demo. Train seeds are
unsealed and may be used freely in tests, previews, and debugging, before
and after the seal; the confirmatory claims rest solely on the untouched
eval block.

**Sealed eval block:** the 36 consecutive generator seeds

```text
20101, 20102, 20103, 20104, 20105, 20106, 20107, 20108, 20109,
20110, 20111, 20112, 20113, 20114, 20115, 20116, 20117, 20118,
20119, 20120, 20121, 20122, 20123, 20124, 20125, 20126, 20127,
20128, 20129, 20130, 20131, 20132, 20133, 20134, 20135, 20136
```

At the time this protocol was drafted, this block was absent from the
working tree and from all Git history (verified by a repository-wide
word-boundary search over the tree and every one of the 24 commits; no
match exists anywhere, including inside hash and float strings of the
retained result artifacts — a looser raw-substring search finds only
coincidental digit-run collisions inside the float mantissas of the four
retained result artifacts, never an instantiated seed), and no project run
had instantiated or previewed any member. In particular, no one has built
an instance, computed a residual, drawn an observation, trained or
evaluated a model, printed a dimension, smoke-tested, or debugged with any
seed in the block.

The no-preview rule remains in force until commit B of §12 — the
machine-readable seal record — is committed **and pushed to the private
remote**. Until then:

- tests may use train seeds or hand-built fixtures only; they may not
  parameterize over, import, derive data from, or otherwise touch the
  sealed block;
- source-code constants containing the declared integers (e.g. the runner's
  frozen seed-list constant) are permitted;
- merely calling `make_sheaf_switch_instance` with a declared eval seed
  counts as a preview, even if the resulting instance is discarded.

If any declared eval seed is instantiated before commit B is pushed, the
entire 36-seed block is void. It will not be partially salvaged. A new
consecutive block must be chosen, verified absent from history, and declared
in a newly committed protocol before any execution.

## 4. Data and generation protocol

**Instances.** For a generator seed `s`, the switch instance is exactly

```text
make_sheaf_switch_instance(s, num_vertices=6, num_edges=9,
                           max_stalk_dim=3, num_decoys=3)
```

from `universa.sheaves`: a random source cellular sheaf on a connected
6-vertex 9-edge base graph with stalk dimensions in `1..3` and integer
restriction entries in `{-2..2}`; a planted morphism whose stalk maps are
signed permutations scaled by `+/-1` or `+/-2`, with target restrictions
`rho'_{v,e} := phi_e rho_{v,e} phi_v^{-1}` — every entry a permuted source
entry times a power of two, an exponent shift with no mantissa rounding, so
the per-incidence naturality residuals are exactly `0.0` in float64,
verified fail-closed at planting; and `K = 4` candidates — the compiled
true target plus 3 certified decoys — with the true target at candidate
index 0 pre-permutation. Decoys share the base graph and stalk dimensions,
redraw the restriction maps, and are accepted only when the planted map's
compiled commutation residual against them exceeds `1e-9`, so
discrimination is guaranteed by construction. All exact suite mathematics
(naturality residuals) is the certified float64 numpy machinery; torch sees
only the resulting feature vectors.

**Sheaves, compiled candidates, and the recovered morphism.** The instance
stores its candidates compiled to 1-complexes (the sheaf coboundary as the
single boundary, `C_0 := C^1`, `C_1 := C^0`). To observe and score them as
sheaves, the runner reads each compiled candidate back as a
`universa.sheaves.Sheaf` block by block from its coboundary — the
`_sheaf_from_coboundary` convention of `universa.partial_sheaf`, an exact
sign-flip block read-off — and recovers the planted `SheafMorphism` by
deterministic regeneration (`random_sheaf(s, 6, 9, 3)` then
`planted_morphism(source, s)`), with the fail-closed consistency check of
`ranking_study_sheaf`: the replanted morphism's compiled target must equal
the instance's compiled true target exactly (`np.array_equal`); any
disagreement is a whole-run `design_failure`, not a result.

**Observation degradation.** All observation is through
`universa.partial_sheaf.SheafObservationModel` with masking and
sign-corruption both active: `mask_fraction = 0.25` always and everywhere,
and `corrupt_fraction == g` at profile grid fraction `g`. Masking is
applied first — the masked edges' restriction blocks vanish, so the
corresponding block rows drop out of the observed coboundary, with the
kept-edge indices carried as provenance — corruption to the surviving
nonzero entries (sign flips only; nothing is added or removed). The mask
draw is derived from the observation seed alone
(`subseed(observation_seed, "sheaf-observe", "mask")`) and `mask_fraction`
is constant, so the same edges are missing at every grid point of a row;
the corruption draw is nested per grid fraction
(`subseed(observation_seed, "sheaf-observe", "corrupt")`, a prefix of one
master permutation over the surviving nonzero positions, counts
`round(fraction * total)` round-half-even), exactly as
`SheafObservationModel` defines. One draw at each (row, grid point) is
shared across all 4 candidates, exactly as the v1, v2, and 2-complex
builders shared theirs.

**The frozen score.** A candidate's misfit at a grid point is the
**observed naturality residual**
`universa.partial_sheaf.observed_naturality_residual(morphism, observed)`
of the planted morphism against the observed candidate sheaf: the
root-sum-square of `||rho'_obs phi_v - phi_e rho||_F` over the surviving
incidences, where `rho'_obs` is the observed (degraded) candidate-side
restriction. This is the declared family change of §1: the constraint
lives in the stalk restriction maps, and the score is the planted
morphism's naturality read against the degraded candidate. At full
observation the true candidate sits at exactly `0.0` by the dyadic-exact
planted construction (sign flips under degradation stay dyadic-exact as
well — an exponent shift, never mantissa rounding). Under this regime's
degradation no column is exact: masking alone would leave the true
residual at `0.0` (the regime note of §1); it is the corruption mode,
active from the first grid point on, that moves it off zero.

**The profile grid.** The grid is `0.2, 0.3, ..., 0.9` — 8 points, step
0.1. It **excludes 0.0** by construction: there is no clean column anywhere
in this regime, and the first column is already masked+corrupted. All
operating fractions (train and eval) are grid points, since a row's
operating fraction names the profile column the oracle reads; grid point
0.9 is a profile column only and is never an operating fraction.

**The frozen feature.** The per-candidate feature vector is exactly
18-dimensional: `log1p` of the observed naturality residual of the planted
morphism against the observed candidate sheaf at each of the 8 grid
fractions (8 dims), then the 7 consecutive first-difference profile slopes
(7 dims), then the 3 masked structural dims — `num_base_vertices`, the
kept edge count, and `c0_dim` (the total vertex-stalk dimension) — read
off the masked observed sheaf under the row's observation seed (3 dims).
All three structural dims are sign-corruption-invariant by construction
(corruption only flips signs of surviving entries; it adds or removes
nothing), so they are well-defined on the row's masked observation without
any separate draw. Because the mask draw is shared across candidates and
all candidates of an instance share the base graph and stalk dimensions,
the three structural dims are identical across the four candidates of a
row (`num_base_vertices` is `6` and the kept edge count is
`9 - round(0.25 * 9) = 7` for every row; only `c0_dim` varies, across
seeds); they carry no within-row candidate signal, and none is claimed.
Every profile feature entry is computed against a degraded observed sheaf.
Ground truth never enters features: the eligibility bookkeeping below is
computed on the undegraded instance purely for seed accounting and label
certification.

**Deterministic random streams.** Every stochastic component uses
`universa.generators.subseed`:

```text
message = ":".join(["universa", str(seed), *components]).encode("utf-8")
digest  = SHA256(message)
subseed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
```

**Row-specific observation draws.** Each data row — one (seed, operating
fraction, replicate) triple — gets

```text
observation_seed = subseed(seed, "sealed-replicate",
                           str(fraction), str(replicate_index))
```

where `str()` is Python's built-in float/int string conversion (e.g.
`str(0.6) == "0.6"`, so one literal message is
`"universa:20101:sealed-replicate:0.6:2"`). That one seed drives **both**
draws — the mask draw and the nested corruption draw — of every
`SheafObservationModel` built for the row (all 8 grid columns and the
masked structural dims of all 4 candidates), and the same seed feeds all
arms within the row. This is the sealed schedule, carried over verbatim
from the v2 experiment's row-specific design (§1).

**Train rows.** One row per (seed, fraction, replicate) over train seeds
`11001..11200` x operating fractions `{0.2, 0.3, 0.4, 0.5}` x a single
replicate with `replicate_index = 0` — 800 rows — each built under its own
row-specific observation seed (e.g. the row `(11001, 0.2, 0)` uses
`subseed(11001, "sealed-replicate", "0.2", "0")`).

**Eval rows and pairing.** For each eval seed and each operating fraction in
`{0.6, 0.7, 0.8}`, exactly `R = 4` replicate observation draws, indexed
`replicate_index` in `{0, 1, 2, 3}` — 432 rows over the block — each built
under its row-specific observation seed as above. Each replicate row is the
full 18-dimensional feature block built under that observation seed. The
operating fraction names the profile column the oracle reads (§6). All
three arms read the **same** replicate rows: the same `observation_seed`
feeds every arm within a row, making the comparison paired at the level of
individual observation realizations.

**Candidate permutations and audited labels.** The per-(seed, fraction)
candidate permutation is drawn from

```text
subseed(seed, "router-v2-permutation", f"{fraction:.6f}")
```

(the `%.6f` key, e.g. `"0.600000"` — a distinct string convention from the
observation subseed's `str(fraction)`, both frozen; the component literal
`"router-v2-permutation"` is retained verbatim from the v2 schedule as a
frozen string constant), via `np.random.default_rng(...).permutation(4)`.
It is shared across all replicates of one (seed, fraction) and across arms,
and the permuted true index is recorded as the label with
`permutation[true_index] == 0`.

**Eligibility bookkeeping.** For every built seed, train or eval, the clean
audit runs on the **undegraded** instance — no observation draw involved.
The planted morphism's clean naturality residual against a candidate is the
root-sum-square of the per-incidence residuals
`||rho'_{v,e} phi_v - phi_e rho_{v,e}||_F` of the probe morphism carrying
the planted stalk maps against that candidate's undegraded restrictions; by
the documented `SheafMorphism.to_chain_map` equality (the base-graph signs
only flip block signs) this equals the compiled `ChainMap` commutation
residual that decoy certification used. The audit requires the true
candidate's clean naturality residual to be at most
`universa.router.RESIDUAL_TOL` (`1e-9`) — it is an exact zero by the
dyadic planted construction of `universa.sheaves.planted_morphism` — and
every decoy's clean naturality residual strictly above `1e-9`. This is the
**undegraded naturality audit**: the sheaf analogue of the 2-complex
experiment's undegraded-both-degrees audit (an exact residual at zero
degradation, now at the stalk restriction maps), computed directly because
this regime's grid has no 0.0 column. It is a deterministic per-seed
property, identical across fractions, replicates, and arms, and it never
enters features.

## 5. Model and training

Exactly one model is trained, inside the sealed runner, after the seal, using
only train seeds:

```text
StructureRouter(feature_dim=18, hidden_dim=64)
```

the shared per-candidate MLP `18 -> 64 -> 64 -> 1` of `universa.router`,
trained by `universa.router.train_router` on the 800 train rows with:

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

Exactly three arms, all applied to every eval row of every eligible seed:

1. **learned** — the single trained model of §5, hard argmax inference,
   reading the whole 18-dimensional feature of each candidate;
2. **oracle** — the polluted argmin naturality-residual oracle: argmin over
   candidates of the observed naturality residual at the operating
   fraction, i.e. argmin of the row's profile column at that fraction
   (log1p is strictly increasing, so this is exactly argmin of the misfit).
   No learning. As in v2 and the 2-complex experiment, **no fraction is
   exact here**: the grid excludes 0.0, so the oracle reads a polluted
   column at every operating fraction;
3. **heuristic** — the descriptive non-learned grid-mean profile heuristic:
   argmin over candidates of the uniform average of the candidate's 8
   `log1p` naturality-residual profile values across the grid — the simplest
   whole-trajectory integration with no learning and no operating-fraction
   choice. It is the profile-shape non-learned comparator, and it is the
   reference of H4 only.

All three arms read the same replicate rows (§4), so each (seed, fraction,
replicate) yields one paired correctness triple. There is no fourth arm, no
ensemble, no eval-time tuning.

## 7. Estimands and unit of inference

For each eligible seed `s` and each operating fraction `f` in
`{0.6, 0.7, 0.8}`:

```text
learned_acc(s, f)   = mean over the R=4 replicates of learned_correct
oracle_acc(s, f)    = mean over the R=4 replicates of oracle_correct
heuristic_acc(s, f) = mean over the R=4 replicates of heuristic_correct
d_oracle(s, f)      = learned_acc(s, f) - oracle_acc(s, f)
d_heur(s, f)        = learned_acc(s, f) - heuristic_acc(s, f)
```

where `learned_correct`/`oracle_correct`/`heuristic_correct` are 0/1 per
replicate row, so each accuracy is a multiple of `0.25` and each paired
difference is a multiple of `0.25` in `[-1, 1]`.

The **generator seed is the sole unit of inference**. It jointly determines
the instance, the per-(seed, fraction) candidate permutations, and — through
the `sealed-replicate` subseed schedule — the four shared observation draws
per operating fraction; the paired contrasts share all of these within a
(seed, fraction). Replicates are not independent units and are never pooled
as such.

Each claim's estimand is `theta = E_seed[d(., f)]` over the eligible-seed
mechanism, with `d` either `d_oracle` (H1–H3) or `d_heur` (H4), estimated by
the arithmetic mean of `d(., f)` across the `n` eligible seeds, with

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

| id | fraction | contrast | prediction and governing decision |
|---|---:|---|---|
| `h1-sheaf-0.7-primary` | 0.7 | `d_oracle` | learned beats oracle; one-sided lower bound `> 0` |
| `h2-sheaf-0.6` | 0.6 | `d_oracle` | learned beats oracle; one-sided lower bound `> 0` |
| `h3-sheaf-0.8` | 0.8 | `d_oracle` | learned beats oracle; one-sided lower bound `> 0` |
| `h4-sheaf-heuristic-0.7` | 0.7 | `d_heur` | learned beats heuristic; one-sided lower bound `> 0` |

With `theta_oracle,f = E_seed[d_oracle(., f)]` and
`theta_heur,f = E_seed[d_heur(., f)]`, the formal hypotheses are fixed as:

- **H1** (id `h1-sheaf-0.7-primary`): `theta = theta_oracle,0.7`; H0:
  `theta <= 0`; H_A: `theta > 0`; reference: the polluted argmin
  naturality-residual oracle at fraction 0.7; supported iff the one-sided
  Bonferroni lower bound is above `0`.
- **H2** (id `h2-sheaf-0.6`): `theta = theta_oracle,0.6`; H0:
  `theta <= 0`; H_A: `theta > 0`; reference: the polluted argmin
  naturality-residual oracle at fraction 0.6; supported iff the one-sided
  Bonferroni lower bound is above `0`.
- **H3** (id `h3-sheaf-0.8`): `theta = theta_oracle,0.8`; H0:
  `theta <= 0`; H_A: `theta > 0`; reference: the polluted argmin
  naturality-residual oracle at fraction 0.8; supported iff the one-sided
  Bonferroni lower bound is above `0`.
- **H4** (id `h4-sheaf-heuristic-0.7`): `theta = theta_heur,0.7`; H0:
  `theta <= 0`; H_A: `theta > 0`; reference: the non-learned grid-mean
  naturality-residual profile heuristic; supported iff the one-sided
  Bonferroni lower bound is above `0`. This is a superiority statement
  against the frozen profile-shape comparator at the primary fraction — it
  keeps the strengthened baseline side of the v2 and 2-complex
  experiments' H4 beyond H1's myopic oracle. It is not a bounded-harm
  statement (v1's H4 was), and it is not a fair-information claim.

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

The table is numerically identical to those of the v1, v2, and 2-complex
experiments (same alpha, same eligible-n range); it is hard-coded and tested
independently in this runner. SciPy is not a runtime dependency of the
runner. Each of the four claim summaries records: the estimate `mean(d)`;
its sample standard deviation; its standard error; the governing one-sided
lower bound; the critical value; the direction (`greater`); the threshold
(`0`); and the support decision. No hypothesis may be added, dropped,
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
- the undegraded naturality audit of §4/§10 is the sole eligibility gate:
  its failure makes that one seed ineligible (recorded, with its reason).
  Every *other* failed check — any determinism, NaN/non-finite feature or
  residual, audit, rank, or validation failure, including any violation
  inside the certified feature machinery and any failed result-audit
  recomputation — is a whole-run `design_failure`;
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
  protocol, or runner file, the imported `universa` package not resolving
  under the project's `src`, a dirty worktree, an existing output path,
  available CUDA, or a torch thread count other than one;
- a worktree found dirty, or a code-manifest hash found changed, or the
  on-disk seal, protocol, or runner file found changed, at the end-of-run
  re-check before publishing a `complete` result;
- a raw-row count that violates the frozen invariant
  `R x |eval fractions| x |eligible seeds|` (`4 x 3 x n`);
- an instance-build exception;
- fewer than 30 eligible seeds;
- any determinism, NaN, audit (other than the per-seed eligibility gate),
  rank, or validation failure;
- aggregation failure or a non-finite estimand; or
- any manual cancellation or infrastructure interruption.

If infrastructure fails after execution begins, the attempt is preserved
under a distinct non-overwriting failure path
`results/experiments/failures/router-sheaf-sealed-1.<status>.json`; the
canonical output path is reserved for status `complete` and a failed
attempt never occupies it. Failure artifacts are written **only** to that
failure path. An exact-code retry is allowed only for a documented
non-scientific failure, before any summaries are inspected, with identical
hashes. Every attempt remains in the evidence bundle. The first complete,
fully validated attempt is canonical; no choice among complete attempts is
allowed. Any code, tolerance, dependency, or analysis change after a
partial run invalidates the seed block and requires a new protocol and new
seeds.

## 10. Eligibility

An eval seed is **eligible** if and only if both hold:

1. its instance builds — `make_sheaf_switch_instance(seed, 6, 9, 3, 3)`
   returns (a build exception is a whole-run `design_failure` per §9,
   never an exclusion); and
2. the undegraded naturality audit passes: computed on the undegraded
   instance without any observation draw, the true candidate's clean
   naturality residual is `<= 1e-9` (`RESIDUAL_TOL`) and every decoy's
   clean naturality residual is `> 1e-9`. As noted in §4, this audit is
   observation-independent and therefore a deterministic per-seed property,
   identical across fractions, replicates, and arms.

Ineligible seeds are recorded with their explicit reasons and are never
replaced, never deleted, and never scored. `n`, the number of eligible
seeds, is the inference sample size of §7–§8. If `n < 30`, the run stops
before training with `design_failure_insufficient_eligible`. Train rows are
built with the same fail-closed audits (build plus the undegraded
naturality audit, and finite-feature checks); any violation there raises
and is a whole-run `design_failure`, never a dropped row.

## 11. Result JSON schema

The runner writes exactly one result file, to
`results/experiments/router-sheaf-sealed-1.json`, through a temporary file
followed by an atomic no-clobber hard-link publish (create-if-absent, never
replace — stronger than a rename), and refuses an existing path.
Non-`complete` statuses are published instead to
`results/experiments/failures/router-sheaf-sealed-1.<status>.json` under
the same no-clobber rule, and failure artifacts are written **only** there.
The JSON contains, at minimum:

- `schema`: the literal tag `universa-router-sheaf-sealed-result/1`, and
  the frozen configuration (instance constants, `mask_fraction`, the
  profile grid, fraction blocks, `R`, training constants, claim family);
- the seal SHA-256, protocol SHA-256, runner SHA-256, and every code-manifest
  SHA-256, plus `design_commit` (commit A) and the execution revision (HEAD
  at execution time);
- environment and provenance: Python/PyTorch/NumPy versions, OS and machine,
  thread count, the operator-provided and effective `CUDA_VISIBLE_DEVICES`,
  the recorded `git status --porcelain --untracked-files=all` output (key
  `git_status_porcelain`, must be empty, recorded pre- and post-run), the
  canonical command, and `sys.argv`;
- training provenance: torch seed, epochs, final total loss, final
  cross-entropy, final auxiliary loss, final train hard accuracy, and the
  model state SHA-256 (canonical form per §5);
- all 36 candidate seed records, each with its eligibility verdict and
  explicit ineligibility reason where applicable;
- raw rows per (seed, fraction, replicate) for every eligible seed: `seed`,
  `fraction`, `replicate_index`, `learned_correct`, `oracle_correct`,
  `heuristic_correct`, `observation_seed`, the **full profile** per
  candidate — the observed naturality residual at all 8 grid points, 8
  float64 values per candidate, in permuted candidate order — plus the
  per-candidate observed naturality residuals at the operating fraction
  (float64, same order), the `permutation`, and `true_index`;
- per-(seed, fraction) accuracies `learned_acc`/`oracle_acc`/
  `heuristic_acc` and paired differences `d_oracle` and `d_heur` against
  both baselines, for all three operating fractions;
- the four claim summaries, each with estimate, sample standard deviation,
  SE, one-sided lower bound, critical value, direction, threshold, and
  `supported` (or `supported: null` for all four on any non-`complete`
  status); and
- an audit block recomputing **everything** from the raw rows alone: row
  counts, eligible-seed count, the oracle arm's bits at every operating
  fraction, the heuristic arm's bits, per-fraction means, standard
  deviations, SEs, bounds, and decisions. This is the retention improvement
  declared in §1: because each raw row retains the full 8-value profile per
  candidate plus the operating-fraction residuals, the oracle and heuristic
  arms and every profile statistic are recomputable from raw rows alone,
  with no feature machinery involved — the v2 experiment's declared
  recompute limit (its raw rows retained only operating-fraction residuals)
  does not apply here. Eligibility counts and seed ids come from the
  eligibility pass, and the learned arm's per-row bits remain checkable only
  with the retained hash-pinned model — the audit block states this
  plainly; its contract is that every arm bit except the learned arm's, and
  every summary of the raw rows, is independently recomputable from them.

An independent validator must recompute all summaries solely from the
retained raw rows and fail closed on missing rows, duplicate (seed, fraction,
replicate) keys, arm imbalance, ineligible seeds carrying rows, any oracle or
heuristic arm bit inconsistent with the retained profiles, wrong t
constants, or any decision inconsistent with §8.

## 12. The two-commit seal procedure

The following order is mandatory:

1. **Commit A (design commit):** this protocol, the runner
   `scripts/run_sheaf_sealed_1.py`, and all its tests — committed without
   instantiating any declared eval seed. Tests use train seeds or hand
   fixtures only (with `SEALED_EVAL_SEEDS` monkeypatched where the runner's
   frozen constant would otherwise be read) and must cover, at minimum: the
   row-specific `sealed-replicate` subseed derivation and that it drives
   the mask draw and the nested corruption draw (`sheaf-observe` `mask`,
   `sheaf-observe` `corrupt`), no sealed-seed fixture use, the pairing of
   replicate draws across all three arms, the fraction-keyed permutation
   and audited-label contract, the observed naturality score and the
   undegraded naturality eligibility audit on hand fixtures, all three arms
   on hand fixtures, all four claim decisions, the hard-coded t table,
   eligibility and insufficient-eligible handling, failure-artifact shape
   (`supported: null`), seal parsing and validation, dirty-worktree
   refusal, and output-exists refusal. The runner embeds this protocol's
   SHA-256 as a frozen constant — initially the fail-closed placeholder
   `PENDING_PROTOCOL_SHA256`, replaced by the real hash before commit A —
   and never embeds its own hash: it computes its runtime SHA-256 and
   requires equality with the seal's `runner_sha256`. This avoids a
   self-hash cycle.
2. **Compute fingerprints at commit A:** the protocol SHA-256, the runner
   SHA-256, and the per-file SHA-256 of every `src/universa/*.py` (the
   17-file code manifest of §2).
3. **Commit B (seal):** the machine-readable seal record
   `docs/14-router-sheaf-seal.json` with frozen contents:
   - `schema`: the literal tag `universa-seal/5`;
   - `design_commit`: the full hash of commit A;
   - `protocol_sha256`, `runner_sha256`;
   - `code_manifest`: the 17 per-file SHA-256 pairs of §2;
   - `train_seed_block`: `{first: 11001, last: 11200}`;
   - `eval_seed_block`: `{first: 20101, last: 20136}`;
   - `no_preview_declaration`: the renewed attestation of §3;
   - `primary_family`: the four claims of §8 verbatim, as four claim objects
     with the nine subfields `id`, `fraction`, `theta`, `null`,
     `alternative`, `reference`, `bound_direction`, `threshold`,
     `support_rule`;
   - `stop_rules`: every stop rule of §9; and
   - `output_path`: exactly `results/experiments/router-sheaf-sealed-1.json`.
4. **Push commit B to the private remote before any sealed seed is
   instantiated**, and verify the remote contains it.
5. Confirm a clean worktree and the pushed seal, and only then execute the
   canonical command once:

   ```bash
   env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src python \
     scripts/run_sheaf_sealed_1.py \
     --output results/experiments/router-sheaf-sealed-1.json
   ```

   `--seal` defaults to `docs/14-router-sheaf-seal.json` and therefore does
   not appear; no hash is ever passed on the command line.
6. Independently validate the retained raw rows (§11) before editing any
   result prose, then commit the immutable result artifact.

The runner accepts a `--seal` argument defaulting to that path. It parses and
validates the machine-readable seal file rather than trusting a naked
command-line hash; there is no `--expected-runner-sha256`-style flag.
Validation requires the schema tag and every field above with no unknown
keys at any level (including inside each claim object), each claim's
`id`, `fraction`, `threshold`, and `bound_direction` (and `role` where
present) value-checked against the runner's frozen claim definitions and
the descriptive subfields required present as nonempty strings,
`design_commit` present in the repository's history
(`git cat-file -e <design_commit>^{commit}`) AND an ancestor of HEAD
(`git merge-base --is-ancestor <design_commit> HEAD`), equality of each
embedded hash with both the runner's frozen constants and the actual file
bytes, equality of the runner's own runtime SHA-256 with `runner_sha256`,
equality of every `code_manifest` hash with the actual file bytes, the seal
file's presence in the HEAD commit and byte-identity of the on-disk seal
with its HEAD blob, no symlink at the seal, protocol, or runner path, the
imported `universa` package resolving under the project's `src`, a clean
worktree per `git status --porcelain --untracked-files=all`, and equality of
`output_path` with the `--output` argument. Only after all of this — and
after setting and verifying one PyTorch thread and a hidden/absent CUDA — may
the runner attempt the 36 instance builds, apply eligibility, and (only if at
least 30 seeds are eligible) build train rows, train the single model of §5,
build eval replicate rows for eligible seeds, score all three arms,
aggregate, and write the result. Before publishing a `complete` result it
re-checks the clean worktree, the code-manifest hashes, and the on-disk
seal, protocol, and runner files, and records the post-run status. It
records `design_commit` (commit A) and HEAD at execution time as the
execution revision.
