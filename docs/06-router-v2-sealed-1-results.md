# Sealed experiment router-v2-sealed-1: results record

Status: complete, independently validated. Universa's second sealed
experiment: the no-anchor degraded regime. The immutable machine artifact
is `results/experiments/router-v2-sealed-1.json`; the frozen protocol is
`docs/04-sealed-router-v2-protocol.md`; the seal is
`docs/05-router-v2-seal.json`.

## 0. Lineage

- design commit A: `6397005fbfbb4db387fde034c1edc800100e4172`
- seal commit B: `e6062061c293ea274023f6c5fc09bdb72d118a0a` — pushed to the
  private remote before any sealed seed was instantiated
- result commit: `81590704581156257010c1538295e36c2028c4c2`
- protocol SHA-256 `2c78f265…`, runner SHA-256 `99055698…`, 15-file code
  manifest — recorded in the seal and re-verified in the result artifact
- canonical command: `env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src python
  scripts/run_router_v2_sealed_1.py --output
  results/experiments/router-v2-sealed-1.json`
- environment: Python 3.12.3, torch 2.13.0+cu130 (CPU, one thread, CUDA
  hidden), numpy 2.5.2

## 1. What was asked

Same question as experiment 1, under the harsher regime that removes its
main caveat: masked edges (mask_fraction 0.25) AND sign corruption over a
profile grid that excludes 0.0, so no clean residual column exists
anywhere — the learned router cannot anchor on any exact value. A new
draw convention makes every feature row-specific (repairing the demo's
per-instance constant-accuracy artifact). A fourth claim strengthens the
baseline side: the learned router must also beat the non-learned
grid-mean profile heuristic, not only the myopic oracle.

## 2. Execution accounting

- 36/36 sealed eval seeds eligible (60101..60136; no build failures, no
  exclusions); 432 raw rows = 36 seeds x 3 held-out fractions
  (0.6, 0.7, 0.8) x 4 paired replicates; one observation seed per row
  drives both mask and corruption draws, shared by both arms.
- One model trained inside the sealed run on 800 declared train rows
  (seeds 50001..50200 x fractions 0.2..0.5, replicate 0): final total
  loss 0.0241, final train hard accuracy 0.9888, model state SHA-256
  `d02e65c0…`.
- No outcome-dependent stopping; no rerun; single canonical attempt.

## 3. The four frozen claims (n = 36, critical value 2.3419692993010397,
Bonferroni alpha = 0.0125 one-sided)

| claim | baseline | fraction | estimate | SD | SE | lower bound | supported |
|---|---|---:|---:|---:|---:|---:|---|
| h1-no-anchor-0.7-primary | oracle | 0.7 | +0.9375 | 0.1250 | 0.0208 | +0.8887089729312283 | **yes** |
| h2-no-anchor-0.6 | oracle | 0.6 | +0.9028 | 0.1612 | 0.0269 | +0.8398488507490467 | **yes** |
| h3-no-anchor-0.8 | oracle | 0.8 | +0.9931 | 0.0417 | 0.0069 | +0.9767918798659650 | **yes** |
| h4-no-anchor-heuristic-0.7 | heuristic | 0.7 | +0.6736 | 0.2451 | 0.0408 | +0.5779453006875759 | **yes** |

Mean arm accuracies on the eval block: learned 0.9861/0.9722/0.9931,
oracle 0.0833/0.0347/0.0000, heuristic 0.2917/0.2986/0.2292 at fractions
0.6/0.7/0.8. The learned accuracies vary by fraction (the per-row draw
convention works as designed); the margins against BOTH non-learned
baselines are wide at every held-out fraction.

## 4. Independent validation (post-run, pre-commit)

92/92 checks passed. Oracle bits reproduced 432/432 from retained row
residuals (first-index tie semantics verified on 85 exact-tie rows); all
108 paired (seed, fraction) summaries bit-exact; all four claim
statistics bit-identical; the critical table equals scipy t.ppf(0.9875,
df) for df 29..35; the observation-seed and permutation schedules
reproduce from pure seed identifiers; hashes, commits, environment, and
seed accounting verified against the seal and git history. Declared
recompute limits, mirrored in the artifact's own scope text: the learned
arm's bits require the retained hash-pinned model; the heuristic arm's
bits require the certified feature machinery over the retained
observation draws (raw rows retain only operating-fraction residuals —
a retention improvement candidate for a future experiment is retaining
the full profile); the eligibility gate's instance-level misfits were
not rebuilt (that would instantiate sealed seeds).

## 5. What the two sealed experiments now establish, and their boundary

On the graph-quotient family at frozen sizes, under two degraded-
observation regimes (corruption-only with a clean anchor, and masked-plus-
corrupted with no anchor), the learned structure router beats the exact
classical reading of the same degraded operator and a non-learned profile
heuristic, by pre-registered margins with multiplicity-controlled one-sided
bounds, on untouched seeds, with adversarially audited ceremony. It loses
nothing where the classical procedure is exact (experiment 1, H4).

Nothing beyond that boundary is claimed: not other families, sizes,
observation models, or real data; not superiority in clean regimes; not
that profile features are the right representation (only that they suffice
here); not discovery, multi-hop, or sheaf/category families — those remain
suite machinery awaiting their own sealed experiments.
