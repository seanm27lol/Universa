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
1e-9, and instances are deterministic from integer seeds. 63 tests.

What does not exist yet: any learned model, any router, any experiment, any
claim. When experiments start, they inherit HOMYMOLY's sealed-seed,
frozen-claim, adversarial-audit methodology.

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
