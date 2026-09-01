# Sealed experiment router-loop-sealed-1: results record

Status: complete, independently validated. Universa's seventh sealed
experiment and the flagship integration: the **route-or-discover loop** —
route when the library fits, discover when it must, admit only certified
novel structure, and know the difference. The immutable machine artifact
is `results/experiments/router-loop-sealed-1.json`; the frozen (amended)
protocol is `docs/19-sealed-router-loop-protocol.md`; the seal is
`docs/20-router-loop-seal.json`.

## 0. Lineage

- design commit A: `90ca3a668099306239a9a6876fec80adb3ac1375` (first seal,
  block 130001..130036, seal commit B `db1737c7…`, pushed before
  instantiation)
- FIRST ATTEMPT FAILED at seed 130006 with status `design_failure`:
  `certified loop pass failed: ValueError: need 0 <= prefix_dim <=
  observed_dim`. Diagnosis (without inspecting any successful outcome):
  the stabilization rule in `universa.discovery.discover_constraint`
  estimates prefix and full ranks with matrix-dependent tolerances, and a
  borderline singular value counted in the prefix but not the full matrix
  (`prefix_dim > observed_dim`) made the refusal record itself
  unconstructable — a crash on the refusal path, after the rule had
  already decided to refuse. The failed attempt is retained immutably at
  `results/experiments/failures/router-loop-sealed-1.design_failure.json`;
  the fix is behavior-preserving for every accept/refuse decision
  (per-field nonnegativity replacing the false ordering invariant, with a
  regression test reproducing the exact tolerance inversion); per the
  stop rules the block was voided for claims.
- amended commit A′: `8f8c8808f69cc1e2552bb312ad974c72974b7546` (protocol
  errata + replacement block `140001..140036`, verified absent from every
  dataset construction in history)
- amended seal commit B′: `537be3bdae87bc9f4e110b7e628ff73c2fb5bc57` —
  pushed before any replacement seed was instantiated
- protocol SHA-256 `b9301292…`, runner SHA-256 `c816a854…`, 18-file code
  manifest (including the fixed `discovery.py`)
- canonical command: `env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src python
  scripts/run_loop_sealed_1.py --output
  results/experiments/router-loop-sealed-1.json`

## 1. What was asked

Per sealed eval seed, three deterministic conditions of one graph-quotient
instance: in-library (truth in the library — route, and do NOT alarm),
out-of-library (truth withheld — alarm, discover, admit, re-route), and
null-control (structure-free observations — refuse, admit nothing). No
learned model: the loop's scoring and discovery are certified machinery;
the claims are the system's end-to-end behavior.

## 2. Execution accounting

- 36/36 replacement eval seeds eligible (140001..140036; every undegraded
  audit exact: true misfit 0.0, all decoys > 1e-9); 108 raw rows = 36
  seeds x 3 conditions; every row double-executed bit-identically.
- No training (certified deterministic procedure; empty train sentinel).
- No outcome-dependent stopping; the retry after the errata was executed
  exactly once on the replacement block.

## 3. The four frozen claims (n = 36, critical value 2.3419692993010397,
Bonferroni alpha = 0.0125 one-sided; SE=0 mechanical case: lower bound
== estimate)

| claim | condition | statistic | threshold | estimate | SD | lower bound | supported |
|---|---|---|---:|---:|---:|---:|---|
| h1-loop-acquisition-rate (primary) | out-of-library | acquisition | 0.95 | 1.0000 | 0.0000 | 1.0000 | **yes** |
| h2-loop-false-admission-control | null-control | refusal | 0.95 | 1.0000 | 0.0000 | 1.0000 | **yes** |
| h3-loop-in-library-routing | in-library | routing | 0.95 | 1.0000 | 0.0000 | 1.0000 | **yes** |
| h4-loop-alarm-precision | in-library | alarm_silence | 0.95 | 1.0000 | 0.0000 | 1.0000 | **yes** |

Unanimous across the block: every out-of-library instance alarmed,
discovered, admitted, and acquired (all 36 mode `discover`, map_misfit
<= 1e-9); every null-control admitted nothing; every in-library instance
routed to the true target with no alarm firing. The system routes when it
can, discovers when it must, never discovers when it shouldn't, and never
invents a false structure — each at 100% on the sealed block.

## 4. Independent validation (post-run, pre-commit)

All 10 checks passed with exact agreement. Every decision recomputed
108/108 from raw rows alone (alarm iff best > 1e-9, first-argmin routing,
acquisition conjunction, NOT-admitted refusal, mode/library coherence);
all four claim statistics exact (SE=0 mechanical case honored); critical
table identical to scipy; hashes, commits, seal byte-identity,
environment, and seed accounting verified against git history; the
failure-artifact lineage (seed 130006's crash, the voided block, the
replacement) verified present and consistent across the retained
artifact, the protocol's errata, and the seal's declaration.

## 5. What this adds to the series

The loop is the first system-level claim in the series: not that one
component works (router, discovery — each sealed separately), but that
their composition behaves as a structure-aware system with a misfit
alarm that is exactly discriminating in this regime. Declared scope: the
certified graph-quotient regime at the frozen sizes; the alarm's
precision/recall is exact there by construction (true score exactly 0.0,
decoys > 1e-9). The degraded-observation loop with the learned router
(the regime where routing itself must be learned) is v2 territory, and
the unanimous result marks the envelope's comfortable edge, not a
universal guarantee.
