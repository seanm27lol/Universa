# Universa

A research scaffold for **structure-switching architectures**: models that keep
the nuances of each mathematical domain in domain-native data structures and
*switch* between structures internally, transporting information along explicit
maps and projecting onto each structure's consistent subspace.

Sibling of [HOMYMOLY](https://github.com/seanm27lol/HOMYMOLY) (private), which
supplies the motivating evidence and the audit methodology. Universa is a
separate, private research repo; nothing here is a result yet.

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

505 tests. **Sealed results so far (all under the full ceremony: frozen
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

Scope caveats in each results record.

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
