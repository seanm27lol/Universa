# Universa design record

Status: scaffold decisions, 2026-08-25. This file records the design choices
made at founding, with their reasons. It is a working document, not a claim.

## 0. Relation to HOMYMOLY

HOMYMOLY established, under a sealed untouched-seed protocol, that when a
boundary operator `B1` is given exactly, the exact constrained estimator
(least squares restricted to `ker(B1)`) beats the soft penalty, generic
ridge, and a dimension-matched random subspace. The mechanistic reading is
the Pythagoras identity: for any estimate `A` and truth in `S = ker(B1)`,

```
||P_S A - A_*||_F^2 = ||A - A_*||_F^2 - ||(I - P_S) A||_F^2
```

so the off-subspace component is exactly removable. Universa treats that move
— transport along a map, project onto the consistent subspace — as the
*atomic operation* of a larger architecture, and asks the questions
HOMYMOLY deliberately did not: what happens when the structure is not given,
when several structures are in play, and when the model must choose or
discover the right one.

## 1. Where the complex lives: the computation graph, not the prompt

Placement options for structure in a model:

1. **Tokens** — feed boundary matrices as input. Rejected: the operator is
   instance-constant, so every example pays its token cost, and the model
   must re-derive linear algebra that is exact for free.
2. **Wiring** — incidence defines the message-passing graph. Accepted: cost
   is `O(nnz)`, and a graph incidence matrix has two nonzeros per column.
3. **Latent routing** — a router selects among structured internal modules.
   Accepted as the switching layer (section 4).
4. **Output projection** — generic backbone, exact projection onto the
   constraint subspace at readout. Accepted as the exactness mechanism.

Principle: instance-constant, ultra-sparse structure enters as wiring and
projection; tokens carry only cochain coefficients. This is also the answer
to context bloat: the operator is never serialized into a prompt.

## 2. One tensor format

All target structure types compile to chain complexes — graded vector spaces
with boundary matrices satisfying `d^2 = 0`:

- graphs: 1-complexes (`B1` only);
- cellular/simplicial complexes: themselves;
- sheaves: stalks + restriction maps on the cells (a chain complex with
  per-stalk structure);
- categories: the Grothendieck nerve (objects, morphisms, composable pairs,
  ...) yields a simplicial set whose free abelian groups form a chain
  complex. Posets land on simplicial complexes directly (order complex);
  Dold–Kan guarantees the simplicial/chain-complex equivalence. Implemented
  in `src/universa/nerve.py`.

The *translator* between any two structures is a chain map: a tuple of
matrices commuting with the boundaries. Universa's first generators plant
exact chain maps (graph quotients, integer arithmetic) so commutation
residuals are exact zeros, not numerical accidents.

## 3. The atomic move and the misfit signal

Transport values along a chain map, project onto the target's consistent
subspace, measure the removed component. Two residuals carry all the
information:

- **commutation residual** `||d_D f_k - f_{k-1} d_C||_F`: does a candidate
  map respect the two structures? Zero for a true chain map.
- **misfit residual** `||(I - P_S) A||_F`: how much of a lifted quantity the
  target structure had to remove.

If no library structure fits the data, every misfit stays high. That is the
discovery signal — a quantitative "none of my structures explains this",
with no hand-built novelty heuristic. Difficulty escalates by withholding
any explicit "wrong structure" flag: the model sees only residuals
(revisable decision, recorded here).

## 4. Routing: temperature-annealed, discrete at inference

The router jumps between structures. Decided properties:

- soft mixing early in training (stable gradients, hedged exploration),
  annealed toward hard routing as calibration improves, strictly discrete at
  inference;
- a load-balancing pressure against router collapse;
- hard routing at inference because exactness composes only under
  commitment: an average of projectors is not a projector, and HOMYMOLY's
  evidence is that hard constraints beat soft penalties when the structure
  is right. Soft mixing is the *training* mechanism, not the *answer*.

v0 implemented (`router.py`): a shared per-candidate MLP over certified
budget features (identification residual, feasible-set nullity, threshold,
structural dims), soft gates annealed exponentially from tau = 2.0 to 0.25,
one-hot argmax with straight-through gradients at inference, Switch-style
load-balancing loss, train/eval on disjoint seed blocks, reported beside
the non-learned argmin-residual baseline. On the clean regime the baseline
is essentially perfect — v0's deliverable is the annealed-routing
machinery and generalization plumbing, verified working. v1 targets
degraded regimes (partial observation), where exact residual computation
is polluted and an amortized router must earn its keep; that is the first
regime where a learned router can be honestly claimed to matter.

## 5. Seeded library plus discovery

The model is given a small library of structures with their projectors and
translators, and is told — architecturally, via the misfit signal — that the
library is incomplete. Discovery means proposing a new structure (initially
a new low-rank subspace/operator) that reduces the residual, then admitting
it to the library. The synthetic suite controls which structures are in the
seed library and which must be discovered, so discovery decisions are
auditable against ground truth.

## 6. The synthetic multi-structure suite (the main deliverable)

Generator families: graphs, 2-complexes, sheaves-on-complexes, and small
categories (via nerves, so the fourth is nearly free). Tasks plant chain
maps between structures and arrange the answer to be cheap in one structure
and hidden in another, with known ground-truth switch points. All four
families now exist and share one contract — planted maps commute with
residual exactly 0.0, decoys are accepted only above a 1e-9 residual, and
instances are deterministic from integer seeds:

- graphs (`generators.py`): random connected graphs, quotient chain maps;
- 2-complexes (`complexes2.py`): deterministic fundamental-cycle faces (a
  Z-basis of the cycle space), quotient maps lifted to degree 2 by exact
  integer solves over `fractions.Fraction`;
- cellular sheaves (`sheaves.py`): stalks + restriction maps, block
  coboundary compiled to the same format, planted stalk-isomorphism
  morphisms with exactly-zero naturality residuals (dyadic scalings only,
  so no rational arithmetic is ever needed);
- categories (`category_instances.py`): finite groups as one-object
  categories, homomorphism-induced nerve chain maps (the bar construction
  is functorial — commutation is exact at degrees 1 and 2).

Planned escalation, now implemented: multi-hop transport
(`multihop.py` — verified composition, per-hop misfit localization: a
wrong choice at hop k is visible exactly at hop k; composition alone can
hide a defect annihilated by later maps, so residuals are probed per
hop), partial observation of boundary operators (`partial.py` — masked
edges, corrupted signs, nested deterministic degradation, measured
breakdown fractions), and sub-identifiability probe budgets
(`budgets.py` — affine feasible sets under candidate constraints,
certified nullities, threshold N* = target cycle rank). These are the
three regimes where HOMYMOLY showed learning is motivated. A design
finding from the budgets family: decoy structures that share the true
target's kernel (e.g. relabeled complete quotients) are indiscriminable
by any transported quantity, so the generator rejects them — the router
can only be asked to distinguish structures whose constraint sets
actually differ.

## 7. Methodology inheritance

Before any experiment or claim: frozen protocol, sealed disjoint seed
blocks, hard-coded decision rules, adversarial audit before sealing,
immutable result artifacts, independent recomputation from raw rows. No
claim from engineering checks alone. rf-moe and any application work live
elsewhere; this repo is the synthetic foundation.

## 8. What this scaffold does not do

No learned model, no router, no training loop, no experiment, no claim. The
scaffold exists to make the format, the atomic move, and the misfit signal
correct and test-covered before anything is built on them.
