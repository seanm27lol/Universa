# Sealed experiment router-sheaf-sealed-1: results record

Status: complete, independently validated. Universa's fifth sealed
experiment: the degraded-regime structure router on the **cellular-sheaf
family** — the constraint living in stalk restriction maps, scored by the
observed naturality residual of the planted morphism. The immutable
machine artifact is `results/experiments/router-sheaf-sealed-1.json`;
the frozen protocol is `docs/13-sealed-router-sheaf-protocol.md`; the
seal is `docs/14-router-sheaf-seal.json`.

## 0. Lineage

- design commit A: `6baccd562ec4c0194614e7fedfe3383d6db91023`
- seal commit B: `e5f0196…` — pushed before any sealed seed was
  instantiated
- execution at `c075d92…` (the group seal commit; the runner records HEAD
  at execution time — the sheaf seal is an ancestor and its bytes are
  byte-identical there, verified)
- result commit: `28d1347…`-series; protocol SHA-256 `7b843450…`, runner
  SHA-256 `ff61b8f5…`, 17-file code manifest
- canonical command: `env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src python
  scripts/run_sheaf_sealed_1.py --output
  results/experiments/router-sheaf-sealed-1.json`

## 1. What was asked

The same routing question on cellular sheaves: masked edges (0.25) plus
sign-corrupted restriction entries over a profile grid excluding 0.0 —
the no-anchor regime with the constraint carried by stalk structure
rather than a plain boundary. Regime note (declared in the protocol):
masking alone never grows the true residual (surviving blocks stay
exact); the discrimination pressure comes from corruption, with masking
reducing coverage.

## 2. Execution accounting

- 36/36 sealed eval seeds eligible (20101..20136); 432 raw rows = 36
  seeds x 3 held-out fractions (0.6, 0.7, 0.8) x 4 paired replicates.
- One model trained inside the sealed run on 800 declared train rows
  (seeds 11001..11200 x fractions 0.2..0.5): final loss 2.52e-05, final
  train hard accuracy 1.0, model state SHA-256 `2dfdbb8d…`.
- No outcome-dependent stopping; no rerun; single canonical attempt.

## 3. The four frozen claims (n = 36, critical value 2.3419692993010397,
Bonferroni alpha = 0.0125 one-sided)

| claim | baseline | fraction | estimate | SD | SE | lower bound | supported |
|---|---|---:|---:|---:|---:|---:|---|
| h1-sheaf-0.7-primary | oracle | 0.7 | +0.9653 | 0.1218 | 0.0203 | +0.9177 | **yes** |
| h2-sheaf-0.6 | oracle | 0.6 | +0.8819 | 0.1741 | 0.0290 | +0.8140 | **yes** |
| h3-sheaf-0.8 | oracle | 0.8 | +0.9931 | 0.0417 | 0.0069 | +0.9768 | **yes** |
| h4-sheaf-heuristic-0.7 | heuristic | 0.7 | +0.6111 | 0.3801 | 0.0634 | +0.4628 | **yes** |

Mean arm accuracies: learned **1.0000 at all three held-out fractions**
(the strongest family result so far), oracle 0.1181/0.0347/0.0069,
heuristic 0.3819/0.3889/0.3542 at fractions 0.6/0.7/0.8.

## 4. Independent validation (post-run, pre-commit)

45/45 checks passed. Oracle bits reproduced 432/432 from retained
operating residuals; heuristic bits reproduced 432/432 from retained
full profiles alone (the retention contract); all 108 paired summaries
bit-exact; all four claim statistics to <1e-12; critical table
identical to scipy; observation/permutation schedules reproduced from
pure seed identifiers (432/432 and 108/108); hashes, commits,
environment, and seed accounting verified against the seal and git
history. The learned arm's bits remain anchored to the retained
hash-pinned model, as the artifact's scope text declares.

## 5. What this adds to the series

Three of four structure families now carry the degraded-regime result
(graph-quotient, 2-complex, sheaf) — same router, same margins'
direction, same ceremony. Nothing beyond the frozen family, sizes, and
observation regime is claimed.
