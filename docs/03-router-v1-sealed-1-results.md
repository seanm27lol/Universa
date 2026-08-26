# Sealed experiment router-v1-sealed-1: results record

Status: complete, independently validated. This file is the human-readable
record of Universa's first sealed experiment. The immutable machine artifact
is `results/experiments/router-v1-sealed-1.json`; the frozen protocol is
`docs/01-sealed-router-v1-protocol.md`; the seal is
`docs/02-router-v1-seal.json`.

## 0. Lineage

- design commit A: `26b0687e2c61fc8e085ae255d9fd7f9ec3a0eb7e`
- seal commit B: `a9817c5352286858e48e04922484504f169c2e54` — pushed to the
  private remote before any sealed seed was instantiated
- result commit: `6c4727cb3334e961dbeb560925fa56000ec1486c`
- protocol SHA-256 `53eb06d7…`, runner SHA-256 `cbe4bfda…`, 13-file code
  manifest — all recorded in the seal and re-verified in the result artifact
- canonical command: `env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src python3
  scripts/run_router_v1_sealed_1.py --output
  results/experiments/router-v1-sealed-1.json`
- environment: Python 3.12.3, torch 2.13.0+cu130 (CPU, one thread, CUDA
  hidden), numpy 2.5.2; execution under the HOMYMOLY venv interpreter
  (Universa declares torch only as an optional dependency; the recorded
  environment in the artifact is authoritative)

## 1. What was asked

On the graph-quotient family, with sign-corrupted boundary observation, does
the learned structure router beat the polluted argmin observed-residual
oracle at held-out corruption fractions, and does it avoid harm at the clean
anchor? Declared asymmetry (frozen in the protocol): the learned router reads
the whole degradation profile anchored at the exact fraction-0 residual; the
oracle reads only the polluted operating-fraction column.

## 2. Execution accounting

- 36/36 sealed eval seeds eligible (30101..30136; none disconnected, no
  build failures, no exclusions); 576 raw rows = 36 seeds x 4 fractions
  (0.0, 0.5, 0.6, 0.7) x 4 paired replicates.
- One model trained inside the sealed run on 1000 declared train rows
  (seeds 10001..10200 x fractions 0.0..0.4): final total loss 2.857e-05,
  final train hard accuracy 1.0, model state SHA-256 `323fe288…`.
- No outcome-dependent stopping; no rerun; single canonical attempt.

## 3. The four frozen claims (n = 36, critical value 2.3419692993010397,
Bonferroni alpha = 0.0125 one-sided)

| claim | fraction | estimate (mean paired d) | SD | SE | one-sided lower bound | threshold | supported |
|---|---:|---:|---:|---:|---:|---:|---|
| h1-held-out-0.6-primary | 0.6 | +0.9097222222222222 | 0.1357 | 0.0226 | +0.8567715304654236 | 0 | **yes** |
| h2-held-out-0.5 | 0.5 | +0.7291666666666666 | 0.2631 | 0.0438 | +0.6264899938428429 | 0 | **yes** |
| h3-held-out-0.7 | 0.7 | +0.9930555555555556 | 0.0417 | 0.0069 | +0.9767918798659650 | 0 | **yes** |
| h4-clean-anchor-bounded-harm | 0.0 | +0.0000000000000000 | 0.0000 | 0.0000 | +0.0000000000000000 | -0.05 | **yes** |

Reading: at held-out fractions the learned router beats the polluted oracle
by mean margins of 0.73 to 0.99 accuracy, with lower bounds far above zero;
the margin grows with degradation. At the clean anchor the router matches
the exact oracle on all 144 replicate rows (both arms perfect), so the
bounded-harm claim holds with zero variance.

## 4. Independent validation (post-run, pre-commit)

An independent validator recomputed from the retained raw rows alone:
oracle correctness 576/576 from row residuals; all 144 paired
(seed, fraction) accuracies bit-exact; all four claim estimates/SDs/SEs to
<=1e-12 and lower bounds bit-exact; critical-value table identical to
scipy t.ppf(0.9875, df) for df 29..35; audit block fully recomputable;
hashes, commits, environment, and seed accounting verified against the
seal and git history. The learned arm's per-row bits are checkable only
with the retained hash-pinned model (documented contract); every
downstream aggregation of those bits recomputes bit-exactly.

## 5. Scope and caveats (carried from the protocol)

- Claims hold for the graph-quotient family at the frozen instance sizes
  (8 vertices, 14 edges, 6 quotient classes, K=4) under sign-corruption
  observation. Nothing is claimed for other families, sizes, observation
  models, or real data.
- The comparison is asymmetric by design: the learned router integrates the
  degradation trajectory anchored at the exact fraction-0 residual while
  the oracle reads one polluted column. H4 guards the anchor itself: the
  learned router loses nothing there.
- The win's mechanism is profile-shape learning under a frozen, auditable
  training protocol — not a claim that learned routing is universally
  superior, and not a claim about discovery, multi-hop, or sheaf/category
  families (those remain suite machinery, untested by sealed experiment).
