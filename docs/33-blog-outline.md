# Blog post outline — "Ten sealed experiments"

Working draft, 2026-09-02. Companion to the arXiv version; this file is the
plan, not the prose. Target: ~2,500–3,500 words, one figure, three tables.
Audience: ML researchers and practitioners who have never pre-registered
anything and think of it as a clinical-trials formality.

## The framing decision this outline commits to

**Lead with the method and the failures, not the wins.** The architecture
result is real but contestable — synthetic benchmark, self-designed
baseline, one family for the loop. The methodology and the negative results
are neither contestable nor common:

- 40 pre-registered claims across ten sealed experiments; **four failed and
  are reported as failed**, with named seeds and decision chains.
- Six recorded occasions where the ceremony caught something, including one
  that killed a canonical run mid-flight and one that voided a train block
  before a seal existed.
- One claim (h4) that has now failed three times with an identical
  mechanism, which is a *diagnosis*, not a disappointment.

A reader who finishes should want to pre-register their next experiment.
That is the conversion goal; the architecture is the vehicle.

---

## 1. Hook — the number that should not be impressive (~250 words)

Open on: *"I ran ten experiments. Thirty-six of forty pre-registered claims
were supported. The four that failed are the reason the other thirty-six
mean anything."*

Then the concrete image: a canonical run that stopped dead at train seed
200058 because the data generator refused to build an instance, and the
rule I had frozen said that was a whole-run failure. I could not patch and
rerun. I had to void both seed blocks, publish the failure, amend the
protocol in the open, and re-seal.

Beat: that is what a protocol costs, and it is the only reason the
successes are worth reading.

## 2. What the system is, in one page (~400 words)

Keep it concrete and resist the urge to teach algebraic topology.

- **The premise**: keep each mathematical domain in its native structure and
  *switch* between structures, rather than flattening everything to one
  representation.
- **One format**: everything compiles to a chain complex — graded spaces
  plus boundary matrices with `d² = 0`. Graphs, cellular complexes, sheaves,
  and categories (via the nerve) all land in the same tensor shape.
- **The atomic move**: transport along a chain map, project onto the
  target's consistent subspace, measure what had to be removed.
- **The signal**: if nothing in the library fits, that residual stays high
  everywhere. That is a quantitative "none of my structures explain this" —
  the discovery trigger, not a heuristic.
- **Why learn at all**: the predecessor project established that when the
  structure is *given*, the exact projection wins in closed form (Pythagoras
  — the off-subspace component is exactly removable). So learning is only
  motivated where structure is unknown, uncertain, or shared. That boundary
  is the whole reason this project exists.

Asset: a small prose box for the Pythagoras identity. No figure needed here.

## 3. The ceremony — the part worth stealing (~700 words, the centerpiece)

This is the section people will share. Make it operational, not
philosophical.

Walk the actual steps:

1. Choose seed blocks and **prove they are absent from the entire repo
   history** with a word-boundary scan before writing anything.
2. Freeze a protocol: conditions, features, models, arms, claims with
   pre-registered thresholds, stop rules, retention contract.
3. **Two commits.** Commit A is the design. Commit B is a seal JSON pinning
   the design commit, the protocol and runner hashes, a manifest of every
   source file, both seed blocks, and the claim objects. **Commit B is
   pushed before a single sealed seed is instantiated.**
4. A fail-closed preflight: the runner refuses to execute unless every hash,
   the design commit's ancestry, and a clean worktree all verify.
5. One canonical run. No outcome-dependent stopping. No rerun.
6. Independent recomputation from the retained raw rows by a second
   implementation written from the protocol, not from the runner.

Then the table that makes it real — **the six times it bit**:

| # | what happened | caught by |
|---|---|---|
| 1 | malformed design-commit hash in a seal | preflight, before any seed opened |
| 2 | seal statistic keys misaligned | preflight, before any seed opened |
| 3 | discovery crashed on its *refusal* path at seed 130006 | canonical run; block voided, one retry |
| 4 | a train block instantiated in a test fixture | pre-seal audit, before commit A existed |
| 5 | generator refused to build train seed 200058 | canonical run; both blocks voided, protocol amended |
| 6 | four claims failed | reported frozen, never reinterpreted |

Land the point: **a protocol that only ever confirms is not a protocol.**
Every one of those six is a place where, without the ceremony, I would have
quietly fixed something and moved on — and would never have known whether
the fix was a fix or a rationalization.

## 4. What the architecture actually did (~500 words, two tables)

Move fast here; this is support, not the thesis.

- **The router across four structure families**, 16/16 claims. The honest
  framing of the baseline: the comparison is against the *exact classical
  reading of the same degraded operator*, which is a strong baseline, and
  the asymmetry (profile vs single column) is declared in every protocol.
  The striking bit: the oracle goes *below chance* under corruption,
  because a corrupted operator's residual ordering is actively misleading
  rather than merely noisy.
- **The bounded-harm check that makes the rest meaningful**: at the clean
  anchor, where the classical method is exact, the learned router matched
  it on all 144 replicate rows — zero harm, zero variance. It does not buy
  its degraded-regime win by giving anything up where exactness is free.
- **The certified discovery head**: 100% on coverage, certification,
  router-readiness, and false-discovery refusal, on 36/36 seeds.
- **The loop vs a deliberately architecture-free model**: +74 points
  end-to-end. State plainly that the generic arm cannot acquire *by
  construction*, so that leg is a design constant, not a finding.

**Asset: `docs/figures/arms-by-condition.svg`.** Four arms across three
conditions, with the two structural zeros drawn as explicit labeled zeros.
The caption does the honest work: those zeros are properties of the design,
declared in the protocol before any data — not training failures.

## 5. The frontier — the figure (~600 words)

**Asset: `docs/figures/alarm-frontier.svg`** (PNG fallback beside it).

The loop's one uncertified component is the alarm that decides whether the
library fits at all. It has exactly one knob. Three sealed settings:

- **loop-v2**, frozen threshold: 83.3% acquisition, 97.2% in-library, −1/36.
- **loop-v3**, bound the false-quiet rate at 2%: acquisition goes to 100%
  and in-library *collapses to 61.1%*, −14/36. The bound was binding and
  paid for one error mode entirely with the other.
- **loop-v4**, price both errors equally: 97.2% / 97.2%, −1/36, 98.15%
  end-to-end. **It dominates loop-v2** — same accuracy, same harm, +13.9
  points of acquisition.

**Name the rules — they are textbook, and saying so is what makes the rest
credible** (see `docs/34-related-work.md` §3). loop-v3's rule is
**Neyman–Pearson classification**: minimize one error subject to a hard
bound on the other. loop-v4's is **Chow's rule** — cost-sensitive rejection
at explicit costs. The frontier is a detection-error tradeoff curve, the
same object as an ROC/DET plot. Presenting either as a novel method would
be the fastest way to lose a reader who knows the literature; presenting
them by name and showing what happened when each was *sealed and measured
inside a working system* is the actual contribution.

The lesson generalizes past this project: **a one-sided bound is a choice
about which error to pay, and it is the wrong instrument when your costs are
symmetric.** Say what "symmetric" meant concretely here — a false quiet
costs one out-of-library seed, a false alarm costs one in-library seed, and
the two conditions carry equal weight end-to-end.

Include the identity, because declaring it up front is the honest move and
readers will respect it: at equal unit costs,
`FQ + FA = 2 − 2·balanced_accuracy` exactly — so the "cost-aware" rule *is*
unconstrained balanced-accuracy maximization. The protocol said so before
the run, rather than letting someone discover it later and call it spin.

## 6. The parts that did not work (~500 words)

Do not soften these. This section is why the post is credible.

- **h4 has failed three times with one mechanism.** The alarm fires on an
  in-library instance, certified discovery correctly refuses it as
  non-novel, and the route credit is forfeited. Always-discovering is
  perfect in-library *by construction*, so any learned alarm with a nonzero
  false-alarm rate loses to it there. The claim may be unwinnable as posed —
  which is a statement about the claim's design, and I am not allowed to
  rewrite it after the fact.
- **The train-block confound.** Comparing loop-v3 to loop-v4 changes the
  rule *and* the train block, because a consumed block is never reused. The
  alarm's intrinsic separation moved 78.5% → 94.0% across those two blocks.
  Part of the gain is a better-separated alarm, not only a better rule.
  Attributing the whole margin to the rule would overclaim.
- **It is all synthetic.** One family for the loop, one instance size, no
  real data. The generator is mine. A reader should discount accordingly,
  and the scope sections say so in every results record.

## 7. Takeaway (~250 words)

The ask: pre-register one experiment. Not a whole ceremony — just freeze the
claim and the threshold before the run, and publish the outcome either way.

The argument: the cost is a few hours of writing a protocol; the return is
that you find out which of your results were decisions rather than
discoveries. Four of my forty were decisions I would otherwise have made
after seeing the data.

Close on the h4 diagnosis as the thing that a looser process would have
hidden: three failures with one mechanism is not noise, it is a finding
about the claim itself, and it only becomes visible when you are forbidden
from quietly changing the claim.

---

## Assets and open decisions

**Ready:**
- `docs/figures/alarm-frontier.svg` / `.png` — the money plot, coordinates
  recomputed from the sealed artifacts by `scripts/make_frontier_figure.py`.
- `docs/figures/arms-by-condition.svg` — §4's figure, bar heights recomputed
  from the retained correctness bits by `scripts/make_arms_figure.py`.
- `docs/34-related-work.md` — first-pass survey across all five areas, with
  the positioning consequences spelled out.
- All numbers: `docs/29-writeup.md` (the synthesis) and `docs/01`–`32`.

**Needed before publishing:**
1. **Is the repo going public?** The post's central claim is auditability.
   Linking a private repo undercuts it. Either open `Universa` (and decide
   about `HOMYMOLY`, which supplies the motivating result) or soften every
   "auditable" sentence to "auditable to reviewers on request."
2. **Absorb the related-work consequences** (`docs/34` §3). The alarm
   sections must name Neyman–Pearson and Chow rather than presenting the
   calibration rules as novel. This is a rewrite of §5's framing, already
   applied to this outline but not yet to `docs/29-writeup.md` or the
   results records.
3. **Close the citation gaps** listed at the end of `docs/34`: Chow (1970)
   primary, the foundational neural sheaf diffusion paper, DreamCoder, plus
   unsearched areas (conformal prediction for the admission gate,
   architecture search as an alternative framing for "switching").
4. **A title.** Candidates: "Ten sealed experiments", "What pre-registration
   costs, and what it buys", "The four claims that failed".

## Divergence from the arXiv version

Same spine, different weighting. The paper needs: a formal method section
with definitions and the Pythagoras identity as a stated proposition, a real
related-work section, full per-experiment tables in an appendix, and the
reproduction contract. The blog can compress §2 and §4 hard and spend its
length on §3 and §5, which are the parts a general reader will actually
carry away.
