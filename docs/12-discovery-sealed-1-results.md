# Sealed experiment discovery-sealed-1: results record

Status: complete, independently validated. Universa's fourth sealed
experiment: the certified discovery head under sealed protocol — its
first claim-grade evaluation. The immutable machine artifact is
`results/experiments/discovery-sealed-1.json`; the frozen protocol is
`docs/10-sealed-discovery-protocol.md`; the seal is
`docs/11-discovery-seal.json`.

## 0. Lineage

- design commit A: `9a8b401009abd2a134b9fa82d0034e5c162c2852`
- seal commit B: `5b73fec…` — pushed before any sealed seed was
  instantiated
- pre-execution corrections (both caught by preflight BEFORE any seed
  was opened, both corrected and re-pushed before execution):
  `d083016` (malformed design_commit fingerprint) and `8abfc98`
  (seal `statistic` keys aligned to the frozen definitions)
- execution at `8abfc98`; result commit: `842da88a4e716ebb433a9a6c31fb214e58a58a2a`
- protocol SHA-256 `41c25ee7…`, runner SHA-256 `5c5160e1…`, 17-file code
  manifest
- canonical command: `env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src python
  scripts/run_discovery_sealed_1.py --output
  results/experiments/discovery-sealed-1.json`

## 1. What was asked

With the true target structure withheld from the library (decoys only),
does the certified discovery head recover a constraint from 16
transported vector observations that (a) covers the true kernel, (b)
certifies, (c) would be accepted by the router (planted-map misfit
below 1e-9), and (d) refuses structure-free observations instead of
inventing a false constraint? There is no learned model: claims concern
the deterministic certified procedure over the seed mechanism.

## 2. Execution accounting

- 36/36 sealed eval seeds eligible (90101..90136); 72 raw rows = 36
  structured + 36 null-control; double-execution bit-identity enforced
  per row.
- No training (certified deterministic procedure; the seal's train block
  is the documented empty sentinel).
- No outcome-dependent stopping; no rerun; single canonical attempt.

## 3. The four frozen claims (n = 36, critical value 2.3419692993010397,
Bonferroni alpha = 0.0125 one-sided; SE=0 mechanical case: lower bound
== estimate)

| claim | statistic | threshold | estimate | SD | lower bound | supported |
|---|---|---:|---:|---:|---:|---|
| h1-discovery-coverage-floor (primary) | coverage | 0.90 | 1.0000 | 0.0000 | 1.0000 | **yes** |
| h2-discovery-certification-rate | cert | 0.95 | 1.0000 | 0.0000 | 1.0000 | **yes** |
| h3-discovery-admission-ready | misfit | 0.95 | 1.0000 | 0.0000 | 1.0000 | **yes** |
| h4-discovery-false-discovery-control | refusal | 0.95 | 1.0000 | 0.0000 | 1.0000 | **yes** |

Unanimous across the block: full kernel coverage (coverage_fraction ==
1.0 on all 36 seeds), certification on all 36, router-acceptance misfit
below 1e-9 on all 36, and refusal of all 36 structure-free controls
(each saturating at ambient rank with the certified "trivial
annihilator" refusal). Unanimous results are the strongest form the
claims can take; the pre-registered floors (0.90/0.95) were cleared
exactly, with the SE=0 mechanical case handled per protocol.

## 4. Independent validation (post-run, pre-commit)

All 10 checks passed, plus a discriminating negative control (a
tampered copy fails with 17 precise mismatches). Certificate residuals
recomputed 36/36 bit-exact from retained boundary and observations;
annihilation/orthonormality of the discovered constraints within
1e-10 on all rows; map misfit recomputed 36/36 from the retained
transported cycle images; all claim statistics recomputed exactly; the
H2 success-rate identity holds on all 36 seeds; null trajectories
recomputed from retained null matrices 36/36; hashes, commits, seal
byte-identity, environment, and seed accounting verified against git
history. Fields requiring a generator rebuild (true-kernel coverage,
containment, admission distance) were verified for internal
consistency only — recomputing them would instantiate sealed seeds,
and the artifact's scope text says exactly that.

## 5. Scope and caveats

Claims hold for the graph-quotient family at the frozen sizes with
16 vector observations per seed. Nothing is claimed for other
families, sizes, observation counts, degraded observation, or real
data. Unanimous support at these floors does not imply the procedure
succeeds everywhere — it implies the frozen design's regime is
comfortably inside the procedure's working envelope; harder regimes
(degraded observation, partial coverage by construction) are the
natural next sealed experiments.
