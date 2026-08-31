# Sealed experiment router-2complex-sealed-1: results record

Status: complete, independently validated. Universa's third sealed
experiment: the degraded-regime structure router on the **2-complex
family** — two constrained degrees, max-degree residual as the frozen
score. The immutable machine artifact is
`results/experiments/router-2complex-sealed-1.json`; the frozen protocol
is `docs/07-sealed-router-2complex-protocol.md`; the seal is
`docs/08-router-2complex-seal.json`.

## 0. Lineage

- design commit A: `ffd189c44e1a21f2f73216ee0682b7c9b7ea15a3`
- seal commit B: `08b9293f2774a705900f0dce8530b5166bf8ddb7` — pushed before
  any sealed seed was instantiated
- pre-execution correction: `d083016073470920b198c55e71f532b044277b8c` —
  preflight's fail-closed design_commit check caught a malformed hash in
  the seal BEFORE any seed was opened; the corrected seal was committed,
  pushed, and is the seal of record (this is the ceremony working as
  designed)
- execution at `d083016`; result commit: `a323bfbc44ca53a990338482d46b429fdc2153f9`
- protocol SHA-256 `792866b6…`, runner SHA-256 `13727136…`, 17-file code
  manifest
- canonical command: `env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src python
  scripts/run_2complex_sealed_1.py --output
  results/experiments/router-2complex-sealed-1.json`

## 1. What was asked

Same routing question as experiments 1-2, on 2-complex targets with two
constrained degrees: masked edges (mask 0.25, B1 columns with their B2
rows) AND dual sign corruption (both boundaries at the grid value) over a
profile grid excluding 0.0 — the no-anchor regime at two degrees. The
frozen score is the max-degree observed commutation residual. H4 keeps
the strengthened non-learned baseline. This experiment also debuts the
retention improvement: raw rows retain the full dual-degree profile
(16 floats per candidate), so BOTH non-learned arms recompute from raw
rows alone — closing experiment 2's declared recompute limit.

## 2. Execution accounting

- 36/36 sealed eval seeds eligible (80101..80136); 432 raw rows = 36
  seeds x 3 held-out fractions (0.6, 0.7, 0.8) x 4 paired replicates.
- One model trained inside the sealed run on 800 declared train rows
  (seeds 70001..70200 x fractions 0.2..0.5): final loss 0.1003, final
  train hard accuracy 0.9663, model state SHA-256 `36a9f946…`.
- No outcome-dependent stopping; no rerun; single canonical attempt.

## 3. The four frozen claims (n = 36, critical value 2.3419692993010397,
Bonferroni alpha = 0.0125 one-sided)

| claim | baseline | fraction | estimate | SD | SE | lower bound | supported |
|---|---|---:|---:|---:|---:|---:|---|
| h1-2complex-0.7-primary | oracle | 0.7 | +0.8750 | 0.2897 | 0.0483 | +0.7619 | **yes** |
| h2-2complex-0.6 | oracle | 0.6 | +0.8194 | 0.2647 | 0.0441 | +0.7161 | **yes** |
| h3-2complex-0.8 | oracle | 0.8 | +0.9444 | 0.1477 | 0.0246 | +0.8868 | **yes** |
| h4-2complex-heuristic-0.7 | heuristic | 0.7 | +0.5833 | 0.3273 | 0.0546 | +0.4556 | **yes** |

Mean arm accuracies: learned 0.9028/0.9236/0.9583, oracle
0.0833/0.0486/0.0139, heuristic 0.3958/0.3403/0.3750 at fractions
0.6/0.7/0.8.

## 4. Independent validation (post-run, pre-commit)

70/70 checks passed (one informational note, the documented
pre-execution seal correction). Oracle bits reproduced 432/432 from
retained operating residuals; **heuristic bits reproduced 432/432 from
retained full profiles alone** (the retention improvement verified
end-to-end); all 108 paired summaries bit-exact; all four claim
statistics to <=5.6e-17; critical table identical to scipy; the stale
`feature_dim=34` string (fixed pre-seal) confirmed absent. The learned
arm's bits remain anchored to the retained hash-pinned model, as the
artifact's scope text declares.

## 5. What this adds to the series

The degraded-regime result now holds on TWO structure families:
graph-quotient 1-complexes (experiments 1-2) and 2-complexes with two
constrained degrees (this experiment) — same router, same machinery,
same ceremony, same direction and magnitude of margins. Nothing beyond
the frozen families, sizes, and observation regimes is claimed.
