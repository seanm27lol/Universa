# Universa

A research scaffold for **structure-switching architectures**: models that keep
the nuances of each mathematical domain in domain-native data structures and
*switch* between structures internally, transporting information along explicit
maps and projecting onto each structure's consistent subspace.

Sibling of [HOMYMOLY](https://github.com/seanm27lol/HOMYMOLY), which supplies
the motivating evidence and the audit methodology.

**Ten sealed experiments are complete.** 36 of 40 pre-registered claims are
supported; the four that failed are reported as failed, with named seeds and
decision chains. Every experiment ran under a two-commit ceremony — a frozen
protocol and a machine-readable seal pushed before any declared seed was
instantiated — and every result was independently recomputed from its retained
raw rows. `docs/29-writeup.md` is the synthesis;
`results/experiments/` holds the immutable artifacts, including two retained
failures.

## The thesis in one paragraph

Known structure collapses estimation to closed form (HOMYMOLY's sealed v2
result: an exact cycle-subspace constraint beats every learned/soft arm when
the boundary operator is given). Learning is therefore only motivated where
structure is unknown, uncertain, intractable to compute with, or must be
shared across instances. Universa studies the switching layer for exactly
those regimes: a seeded library of structures the model knows, a residual
signal that tells it none of them fit, and a router that jumps between them.

## Design decisions (docs/00-design.md)

- **The complex lives in the computation graph, not the prompt.** Boundary
  operators are instance-constant and extremely sparse; they enter as wiring
  and projection layers, never as tokens.
- **One tensor format.** Every structure compiles to a chain complex: graded
  vector spaces + boundary matrices with `d^2 = 0`. Graphs are 1-complexes,
  cellular complexes are themselves, sheaves are stalks + restriction maps on
  the cells, and categories enter via the Grothendieck nerve
  (`src/universa/nerve.py`).
- **The translator is a chain map plus a projection.** Transport values along
  the map, project onto the target's consistent subspace, and measure what
  had to be removed. This is the atomic move; HOMYMOLY v2 is its audited
  evidence base.
- **Residual misfit is the discovery signal.** If no library structure fits,
  residuals stay high after every projection — a quantitative "my structures
  don't fit" alarm, not a vibe.
- **Routing will be temperature-annealed** (soft early for exploration,
  discrete at inference for exactness and auditability). Not built yet.

## Status

Early build. What exists today:

- `universa.structures` — the chain-complex format with fail-closed `d^2 = 0`
  validation, and chain maps with exact commutation checks.
- `universa.nerve` — finite categories -> nerve -> chain complex.
- `universa.operators` — SVD nullspace bases, orthogonal projectors, the
  transport-and-project atomic move, commutation/misfit residuals.
- `universa.generators` — graph family: random connected graphs with planted
  quotient chain maps (exact integer arithmetic) and decoy structures for
  auditable routing ground truth; SHA-256-derived sub-seeds.
- `universa.complexes2` — 2-complex family: faces attached to deterministic
  fundamental cycles (a Z-basis of the cycle space) and quotient maps lifted
  to degree 2 by exact integer solves.
- `universa.sheaves` — cellular-sheaf family: stalks + restriction maps over
  a graph, block coboundary, planted stalk-isomorphism morphisms with
  exactly-zero naturality residuals.
- `universa.category_instances` — group bar-construction family: finite
  groups as one-object categories, homomorphism-induced nerve chain maps
  (Z/6 -> Z/3, Z/4 -> Z/2, S3 -> Z/2 by sign).
- `examples/quickstart.py` — end-to-end misfit-discrimination demo.

All four families share one contract: planted maps have commutation
residual exactly 0.0, decoys are accepted only if the residual exceeds
1e-9, and instances are deterministic from integer seeds.

Task layer (the regimes where HOMYMOLY showed learning is motivated):

- `universa.multihop` — multi-hop transport through chains of structures
  with verified composition of chain maps and per-hop misfit
  localization: a wrong choice at hop k produces a residual signature at
  exactly hop k (composition alone can hide it — later maps may annihilate
  an earlier defect, so the signal is probed per hop).
- `universa.partial` — partial observation of structure: masked edges and
  corrupted signs with full provenance, nested deterministic degradation,
  ranking studies with measured breakdown fractions.
- `universa.budgets` — sub-identifiability probe budgets: affine feasible
  sets under candidate constraints with certified nullities, and the
  identifiability threshold N* = cycle rank of the target.

Router (v0, the repo's only torch code — import it explicitly with
`from universa.router import StructureRouter`; a plain `import universa`
stays torch-free, and torch is an optional dependency:
`pip install -e ".[router]"`):

- `universa.router` — the temperature-annealed structure router: shared
  per-candidate MLP over certified budget features, soft gates annealed
  from tau = 2.0 to 0.25 during training, strictly discrete argmax with
  straight-through gradients at inference, and a Switch-style
  load-balancing loss against collapse. Trained on one seed block and
  evaluated on a disjoint block, always reported beside the non-learned
  argmin-residual baseline. On the clean regime the baseline is
  essentially perfect and v0's deliverable was the machinery.
- Router v1 (same module): degradation-profile features from
  `universa.partial` observation models, trained on fractions 0.0-0.4 and
  evaluated on disjoint seeds at HELD-OUT fractions 0.5/0.6/0.7. First
  regime where the learned router earns its keep: 1.0000 hard accuracy
  at all three held-out fractions while the polluted argmin oracle
  collapses to 0.3750/0.0625/0.0000 (chance is 0.25). The win is real
  but asymmetric by design — the learned router integrates the whole
  degradation trajectory anchored at the exact fraction-0 residual while
  the oracle reads one polluted column; see `examples/router_v1_demo.py`.
- Router v2 (`universa.router_v2`, no-anchor regime): mask 0.25 plus
  sign-corruption over a grid that excludes 0.0 — no clean column
  anywhere, eligibility bookkeeping never entering features. The learned
  router still beats the polluted oracle decisively, answering the
  strongest criticism of the v1 win. **Second sealed result
  (`docs/06-router-v2-sealed-1-results.md`):** all four claims supported
  at n=36 — margins +0.90/+0.94/+0.99 against the oracle at held-out
  0.6/0.7/0.8 and +0.67 against the non-learned profile heuristic, with
  per-row draw convention replacing the demo's per-instance artifact.
- `universa.partial2` — observation models for 2-complexes: independent
  B1/B2 sign corruption, consistent edge masking (B1 columns with their
  B2 rows), and observed d^2 violation handled by design: the observed
  object is a purported structure, not a valid complex — under
  corruption the true target can fail d^2 while decoys stay valid, which
  is part of why this regime is hard.
- `universa.partial_sheaf` / `universa.partial_group` — observation
  models for cellular sheaves (edge masking plus restriction-entry
  sign corruption, dyadic-exact) and for group nerves (multiplication-
  table corruption bypassing category validation, with the closed-form
  degradation law `sqrt(2 (s/m)^2 round(f m^2))`).

Discovery head (numpy only):

- `universa.discovery` — the seeded-library-plus-discovery mechanism of
  the design doc: certified SVD estimation of the data-supported
  consistent subspace from transported vector observations, an
  annihilator constraint with a certificate residual, fail-closed
  insufficiency (dimensional stabilization or it refuses), and a novelty
  gate so the library admits only genuinely new certified structures.
  End-to-end on the graph-quotient family with the truth withheld:
  full coverage recovered and the planted map's misfit against the
  discovered constraint below 1e-9 on all documented seeds.

1100 tests. **Sealed results so far (all under the full ceremony: frozen
protocol, two-commit seal pushed before any sealed seed was opened,
canonical run, independent recomputation from raw rows, immutable result
artifact):**

1. **router-v1** (`docs/03`): graph-quotient family, sign-corrupted
   observation — learned router beats the polluted oracle at every
   held-out fraction (+0.73/+0.91/+0.99 at 0.5/0.6/0.7) and loses
   nothing at the clean anchor (bounded harm), 4/4 claims, n=36.
2. **router-v2 no-anchor** (`docs/06`): masked edges + corruption, grid
   excluding 0.0 — beats the oracle (+0.90/+0.94/+0.99 at 0.6/0.7/0.8)
   AND the non-learned profile heuristic (+0.67), 4/4 claims, n=36.
3. **router-2complex** (`docs/09`): 2-complex family, two constrained
   degrees, no-anchor dual corruption — beats the oracle
   (+0.82/+0.88/+0.94) and the heuristic (+0.58), 4/4 claims, n=36;
   debuts full-profile raw-row retention (both non-learned arms
   recompute from raw rows alone).
4. **discovery** (`docs/12`): the certified discovery head with the
   truth withheld — full kernel coverage, certification, router-ready
   misfit, and false-discovery refusal ALL unanimous (1.0000) on 36/36
   seeds against pre-registered floors (0.90/0.95/0.95/0.95), 4/4
   claims; no learned model.
5. **router-sheaf** (`docs/15`): cellular-sheaf family, naturality
   residual under masked+corrupted restrictions — beats the oracle
   (+0.88/+0.97/+0.99 at 0.6/0.7/0.8) and the heuristic (+0.61), 4/4
   claims, n=36; learned router at 1.0000 accuracy everywhere.
6. **router-group** (`docs/18`): group-nerve family, corruption of the
   multiplication table — beats the oracle (+0.15/+0.70/+0.95) and the
   heuristic (+0.05), 4/4 claims, n=36; learned router again perfect;
   the true profile matches the protocol's declared closed form
   bit-exactly at all 3456 grid entries. The sweep is complete: all
   four structure families carry the degraded-regime result, 16/16
   claims supported.
7. **router-loop** (`docs/21`): the route-or-discover loop as one
   system — alarm precision, in-library routing, end-to-end
   acquisition on withheld truths, and false-admission refusal ALL
   unanimous (1.0000) on 36/36 sealed seeds, 4/4 claims; no learned
   model. The first attempt failed at seed 130006 (a rank-tolerance
   crash in discovery's stabilization refusal path) and was retained
   immutably; the fault was diagnosed without inspecting successful
   outcomes, fixed behavior-preservingly, the block voided, and the
   experiment re-sealed and completed on a verified-absent replacement
   block — the failure-handling path of the ceremony exercised end to
   end.
8. **router-loop-v2** (`docs/24`): the comparison-grade experiment —
   the degraded-regime loop against a generic no-architecture learned
   model and two ablations. The full architecture wins decisively:
   +0.74 end-to-end vs generic (primary), +0.28 vs routing-only, +0.56
   on in-library routing (3/4 claims, n=36); the generic model fails
   everywhere (0/36 acquisition, 0.42 in-library, 0.17 null refusal).
   The honest boundary: h4 (bounded harm vs always-discovering) NOT
   supported — the learned alarm's selectivity costs exactly one seed
   (160031) of in-library accuracy against discovery-only's perfect
   record, traded against a 44% reduction in discovery invocations
   (61 vs 108). Also the series' first train-block errata: the first
   train block was voided pre-seal by test-side instantiation, caught
   by the consistency audit before commit A, and replaced.
9. **router-loop-v3** (`docs/27`): the alarm redesign — margin
   features, a doubled declared train block, and a train-block-
   calibrated threshold, frozen from the published h4 mechanism. The
   frozen calibration did exactly what it was designed to do: false
   quiets eliminated (out-of-library acquisition 0.833 -> 1.000) at
   the price of a 0.41 train false-alarm rate that materialized as
   14/36 in-library harm. h1 (+0.44 e2e vs generic) and h2 (+0.20 vs
   routing-only) supported; h3 and h4 NOT supported, reported frozen.
   The tradeoff moved along the alarm's Pareto frontier rather than
   closing: the threshold is a one-parameter dial, and a one-sided
   bound is a choice of which error to pay. All 14 harm seeds named,
   each with its decision chain in the artifact.

10. **router-loop-v4** (`docs/32`): the cost-aware calibration — the
    alarm's threshold chosen to minimize false-quiet plus false-alarm
    rate at equal unit costs, replacing v3's binding one-sided bound.
    The protocol declares up front that at equal costs this rule *is*
    unconstrained balanced-accuracy maximization
    (`FQ + FA = 2 − 2·balanced_accuracy`, exactly). 3/4 claims: +49
    pts e2e vs generic, +32 vs routing-only, and **h3 supported (+39
    pts) where loop-v3's failed**. h4 NOT supported, reported frozen —
    exactly one seed (230011), the same mechanism as loop-v2's. The
    resulting point **dominates loop-v2**: identical in-library
    accuracy and identical −1/36 harm with +14 pts of acquisition, at
    98.15% end-to-end. First attempt failed at train seed 200058 on
    the generator's discriminability guard; errata 1 made a train-side
    non-instance a recorded exclusion, both blocks were voided, and
    the experiment was re-sealed and completed on verified-absent
    replacements.

Scope caveats in each results record.

**Synthesis: `docs/29-writeup.md`** — the whole series in one document
(36/40 pre-registered claims supported, the four-family sweep, the
certified discovery head, and the learned alarm's measured Pareto
frontier), quoting the sealed artifacts only.

## Layout

```
src/universa/    the library (numpy only)
tests/           pytest suite
docs/            design record
examples/        runnable demos
```

## Run

```bash
python -m pytest -q
python examples/quickstart.py
```

Any scientific Python 3.11+ environment with numpy and pytest works; no
installation step is required yet.
