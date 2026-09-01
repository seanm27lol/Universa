# Sealed experiment router-group-sealed-1: results record

Status: complete, independently validated. Universa's sixth sealed
experiment: the degraded-regime structure router on the **group-nerve
family** — the constraint being associativity of the multiplication
table, with the observed object a purported structure that fails the
category axioms while d1=0 keeps d^2=0 trivially invisible. The
immutable machine artifact is
`results/experiments/router-group-sealed-1.json`; the frozen protocol
is `docs/16-sealed-router-group-protocol.md`; the seal is
`docs/17-router-group-seal.json`.

## 0. Lineage

- design commit A: `47e964d067635a1d4cf762336d5cc08d021df911`
- seal commit B: `c075d92…` — pushed before any sealed seed was
  instantiated
- execution at `f1452c1…` (the sheaf result commit; the runner records
  HEAD at execution time — the group seal is an ancestor and its bytes
  are byte-identical there, verified)
- protocol SHA-256 `d974f4b4…`, runner SHA-256 `c1d01a56…`, 17-file
  code manifest
- canonical command: `env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src python
  scripts/run_group_sealed_1.py --output
  results/experiments/router-group-sealed-1.json`

## 1. What was asked

The same routing question on group bar-construction nerves (Z/6 -> Z/3
by the mod-3 homomorphism, K=3 candidates): corruption-only observation
of the multiplication table (no masking for this family) over a profile
grid excluding 0.0. The score is the observed degree-2 nerve
commutation residual; the true candidate's degradation follows the
protocol's declared closed form `sqrt(8 * round(9 g))`.

## 2. Execution accounting

- 36/36 sealed eval seeds eligible (40101..40136); 432 raw rows = 36
  seeds x 3 held-out fractions (0.6, 0.7, 0.8) x 4 paired replicates.
- One model trained inside the sealed run on 800 declared train rows
  (seeds 21001..21200 x fractions 0.2..0.5): final train hard accuracy
  1.0, model state SHA-256 `afab0952…`.
- No outcome-dependent stopping; no rerun; single canonical attempt.

## 3. The four frozen claims (n = 36, critical value 2.3419692993010397,
Bonferroni alpha = 0.0125 one-sided)

| claim | baseline | fraction | estimate | SD | SE | lower bound | supported |
|---|---|---:|---:|---:|---:|---:|---|
| h1-group-0.7-primary | oracle | 0.7 | +0.7014 | 0.2377 | 0.0396 | +0.6086 | **yes** |
| h2-group-0.6 | oracle | 0.6 | +0.1458 | 0.1386 | 0.0231 | +0.0918 | **yes** |
| h3-group-0.8 | oracle | 0.8 | +0.9514 | 0.1312 | 0.0219 | +0.9002 | **yes** |
| h4-group-heuristic-0.7 | heuristic | 0.7 | +0.0486 | 0.1003 | 0.0167 | +0.0094 | **yes** |

Mean arm accuracies: learned **1.0000 at all three held-out fractions**,
oracle 0.8542/0.2986/0.0486, heuristic 0.9722/0.9514/0.9653 at fractions
0.6/0.7/0.8. The group regime is the easiest for the baselines too
(oracle reaches 0.85 at fraction 0.6), so margins are narrower than in
other families — but all four remain supported with lower bounds above
zero, and the learned router is perfect everywhere.

## 4. Independent validation (post-run, pre-commit)

All checks passed. Oracle and heuristic bits each reproduced 432/432
from raw rows alone; all 108 paired summaries bit-exact; all four
claim statistics to 2.8e-17; critical table identical to scipy; the
true candidate's profile matches the protocol's declared closed form
`sqrt(8 * round(9 g))` **bit-exactly at all 3456 grid entries**
(confirming genuine Z/6 -> Z/3 nerve residuals, including the declared
flat 0.4 -> 0.5 step); `mask_fraction` confirmed absent everywhere
(corruption-only design); hashes, commits, environment, and seed
accounting verified against the seal and git history.

## 5. What this adds to the series

All four structure families now carry the degraded-regime result:
graph-quotient (experiments 1-2), 2-complexes (experiment 3), cellular
sheaves (experiment 5), and group nerves (this experiment) — same
router, same ceremony, all 16 claims supported across the sweep. The
family variation is informative in itself: margins are largest where
the baselines are weakest (sheaf) and narrowest where the closed-form
degradation law makes the true profile easy to read heuristically
(group) — and in every case the learned router is at or near perfect.
Nothing beyond the frozen family, sizes, and observation regimes is
claimed.
