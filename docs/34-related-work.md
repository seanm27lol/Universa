# Related work — first pass

Literature survey for the paper and blog, 2026-09-02. This is a positioning
document, not a bibliography: each section says what the field already has,
what Universa borrows from it, and what (if anything) is left over as a
contribution. **One finding here materially changes how the alarm arc must
be presented — see §3.**

---

## 1. Structure in the architecture

The closest neighbours, and they are close.

**Sheaf neural networks / neural sheaf diffusion.** Cellular sheaves equip a
graph with a vector space (stalk) per vertex and edge plus linear
restriction maps for each incident pair, inducing a transport-aware
diffusion that handles heterophily and oversmoothing better than isotropic
message passing. This is the same object as `universa.sheaves`, used for the
same reason.

- **Bodnar, Di Giovanni, Chamberlain, Liò & Bronstein, *Neural Sheaf
  Diffusion: A Topological Perspective on Heterophily and Oversmoothing in
  GNNs*, NeurIPS 2022** — https://arxiv.org/abs/2202.04579 (the foundational
  paper; code at https://github.com/twitter-research/neural-sheaf-diffusion).
  Its framing is the one to engage directly: a standard GNN implicitly
  assumes a *trivial* underlying sheaf, and the choice of sheaf is what the
  Laplacian, the diffusion, and the model inherit. Universa's position is the
  next question along — if the sheaf is a choice, which one, and what if none
  of the available ones fit?
- Barbero et al., *Sheaf Neural Networks with Connection Laplacians* —
  https://arxiv.org/pdf/2206.08702
- Bronstein, *Neural Sheaf Diffusion for deep learning on graphs* (overview)
  — https://medium.com/data-science/neural-sheaf-diffusion-for-deep-learning-on-graphs-bfa200e6afa6
- *Polynomial Neural Sheaf Diffusion* — https://arxiv.org/html/2512.00242
- *Cellular Sheaf Neural Operators for Structure-Preserving Surrogate
  Modeling of Constrained PDEs* — https://arxiv.org/pdf/2606.00937
  (uses sheaf structure over a cell complex with incidence and Hodge
  structure as the algebraic backbone — very close to Universa's framing)

**Topological deep learning.** Learning on simplicial, cell, and
combinatorial complexes rather than graphs, to capture higher-order rather
than pairwise relations. Universa's "one tensor format" (everything compiles
to a chain complex) is essentially the TDL premise.

- Hajij et al., *Topological Deep Learning: Going Beyond Graph Data* —
  https://arxiv.org/abs/2206.00606
- Papillon et al., *Architectures of Topological Deep Learning: A Survey of
  Message-Passing Topological Neural Networks* —
  https://arxiv.org/pdf/2304.10031
- *Position: Topological Deep Learning is the New Frontier for Relational
  Learning* — https://pmc.ncbi.nlm.nih.gov/articles/PMC11973457/
- Curated list — https://github.com/lrnzgiusti/awesome-topological-deep-learning
- TopoNetX (tooling over complexes) — referenced throughout the survey

**What is left over.** TDL and SNNs fix *one* structure and learn on it.
Universa keeps a *library* of structures, transports between them along
chain maps, and — the part with no clear neighbour — uses the residual
misfit as a signal that **no library structure fits**, then synthesizes a
new certified one. The switching and the discovery, not the chain complexes,
are the claim. The paper must say this explicitly and early, or a reviewer
who knows TDL will read the format as the contribution and find it derivative.

## 2. Routing

Universa's router is Switch-style and the protocol already says so: a
shared per-candidate scorer, a load-balancing auxiliary loss against
collapse, temperature annealing, and hard argmax at inference with
straight-through gradients.

- Fedus et al., Switch Transformer — top-1 routing with an auxiliary
  load-balancing loss `α·N·Σ f_i·P_i`, minimized when experts receive equal
  token mass. Overview: https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mixture-of-experts
- *Expert Race: A Flexible Routing Strategy…* — https://arxiv.org/pdf/2503.16057
- *Equifinality in Mixture of Experts: Routing Topology Does Not Determine
  Language Modeling Quality* — https://arxiv.org/pdf/2604.14419
  (worth citing as a caution: routing topology mattering less than expected)

**What is left over.** Nothing, on the routing mechanism itself — it is
borrowed and should be presented as borrowed. What differs is *what* is
routed over: MoE routes over undifferentiated learned experts; Universa
routes over structures with certified constraint sets, so a route can be
verified rather than merely trained.

## 3. The alarm — and the finding that changes the framing

**This is the important section.** The alarm decides "does anything in my
library fit?" That is a *learned rejection / abstention gate*, and the
two calibration rules the series compared both have standard names.

**loop-v3's rule is Neyman–Pearson classification.** The NP paradigm
minimizes type II error subject to an upper bound on type I error — which is
exactly "maximize balanced accuracy subject to `false_quiet_rate ≤ 0.02`."

- Tong, Feng & Li, *Neyman-Pearson classification algorithms and NP receiver
  operating characteristics*, Science Advances —
  https://www.science.org/doi/10.1126/sciadv.aao1659
- *Intentional Control of Type I Error… A Neyman–Pearson Approach to Text
  Classification*, JASA —
  https://www.tandfonline.com/doi/full/10.1080/01621459.2020.1740711
- *Neyman-Pearson Classification under Both Null and Alternative
  Distributions Shift* — https://arxiv.org/html/2511.06641v1

**loop-v4's rule is Chow-style cost-sensitive rejection.** Selective
classification originates in Chow's optimal rejection rule: minimize
expected loss where misclassification and abstention carry explicit costs.
Equal costs collapsing to balanced-accuracy maximization is the elementary
special case.

- **C. K. Chow, *On optimum recognition error and reject tradeoff*, IEEE
  Transactions on Information Theory, vol. IT-16, no. 1, pp. 41–46, January
  1970**, doi:10.1109/TIT.1970.1054406 —
  https://ieeexplore.ieee.org/document/1054406/ · dblp:
  https://dblp.org/rec/journals/tit/Chow70.html. The origin of the optimum
  rejection rule and of the **error–reject curve**, which is the standard
  name for what the frontier figure plots. Use that name.
- *Classification with Rejection Based on Cost-sensitive Classification* —
  https://arxiv.org/pdf/2010.11748
- Cortes, DeSalvo & Mohri, *Boosting with Abstention* —
  https://cs.nyu.edu/~mohri/pub/rboost.pdf
- *Predictor-Rejector Multi-Class Abstention* — https://arxiv.org/pdf/2310.14772
- *Binary Classification with Bounded Abstention Rate* — https://arxiv.org/pdf/1905.09561
- *Learning to Reject with a Fixed Predictor* — https://openreview.net/pdf?id=dCHbFDsCZz

**And the "bound the false-quiet rate" scheme is standard OOD practice.**
The TPR-β thresholding scheme picks the threshold on a validation set to hit
a preset false-alarm rate; FPR@95 is the standard reported metric. Universa's
alarm is, mechanically, an OOD detector over library fit.

- *Out-of-Distribution Detection: A Task-Oriented Survey of Recent Advances*
  — https://dl.acm.org/doi/10.1145/3760390
- *Class-wise Thresholding for Robust Out-of-Distribution Detection* —
  https://arxiv.org/pdf/2110.15292
- Hendrycks et al., *Deep Anomaly Detection with Outlier Exposure* —
  https://arxiv.org/pdf/1812.04606

### What this means for the paper — act on this

1. **Retire "cost-aware calibration" as if it were a new method.** It is
   Chow's rule at unit costs. Name it that way and cite Chow. The same for
   loop-v3: call it what it is, a Neyman–Pearson constraint, and cite Tong
   et al. A reviewer who knows this literature will otherwise conclude the
   authors do not.
2. **The contribution is the sealed measurement, not the rule.** What is
   genuinely uncommon: a pre-registered, seed-sealed, independently
   recomputed comparison of an NP-constrained threshold against a
   cost-sensitive one *inside a working system*, with all three operating
   points published including the two that failed their claims. The rules
   are textbook; running them under seal and reporting the failures is not.
3. **The frontier plot is an error–reject curve** (Chow 1970), the standard
   object for exactly this tradeoff, and a cousin of the ROC/DET framing.
   Name it. This strengthens the figure rather than weakening it: readers
   already know how to read one, and the novelty is that these three points
   are *sealed* rather than swept post hoc — a swept curve tells you what
   the tradeoff looks like, three pre-registered points tell you what you
   would have committed to.
4. **The equal-cost identity is elementary.** `FQ + FA = 2 − 2·balanced
   accuracy` should be presented as "declared up front so no one thinks we
   discovered it late," not as an insight. The protocol already frames it
   correctly; keep that framing in the paper.
5. **h4 gains a proper vocabulary.** "A bounded-harm claim against
   always-discovering may be unwinnable" is, in this literature, the
   observation that the full-coverage classifier is the ceiling on the
   accepted subset — the coverage/risk tradeoff. Cite the selective-
   classification framing and the diagnosis becomes a recognizable result
   rather than an anecdote.

## 4. Discovery and library learning

The discovery head — propose a new certified structure, gate it for
novelty, admit it to the library — is library learning with a certificate.

- **Ellis, Wong, Nye, Sablé-Meyer, Morales, Hewitt, Cary, Solar-Lezama &
  Tenenbaum, *DreamCoder: Bootstrapping Inductive Program Synthesis with
  Wake-Sleep Library Learning*, PLDI 2021** — https://arxiv.org/abs/2006.08381
  · https://dl.acm.org/doi/10.1145/3453483.3454080. Extended version:
  *DreamCoder: growing generalizable, interpretable knowledge with
  wake–sleep Bayesian program learning*, Phil. Trans. R. Soc. A 381(2251) —
  https://royalsocietypublishing.org/rsta/article/381/2251/20220050/112456/
  DreamCoder-growing-generalizable-interpretable. The direct analogue: a
  library that grows as new abstractions are discovered and admitted.
- *LILO: Learning Interpretable Libraries by Compressing and Documenting
  Code* — https://arxiv.org/pdf/2310.19791
- *Neurosymbolic Programming* (survey) —
  https://www.researchgate.net/publication/356895111_Neurosymbolic_Programming
- *Towards Modular Algorithm Induction* — https://arxiv.org/pdf/2003.04227
- *Unsupervised Learning of Neurosymbolic Encoders* — https://arxiv.org/pdf/2107.13132

**What is left over.** The admission gate is *certified* — a candidate is
admitted only if its constraint certifies (SVD residual) and is genuinely
novel (projector distance), and the sealed experiment measured 100% refusal
on structure-free controls. DreamCoder-style library learning admits by
description length / Bayesian program-prior compression, not by a
certificate with a refusal guarantee. That contrast is worth a paragraph.

**The strengthening path — conformal novelty detection.** There is now a
literature giving *distribution-free FDR guarantees* on exactly the
admission decision Universa makes with a deterministic certificate. This is
the natural next rigour step, and it should appear in future work rather
than be discovered by a reviewer:

- *Adaptive novelty detection with false discovery rate guarantee* —
  https://arxiv.org/pdf/2208.06685
- *Full-conformal novelty detection* — https://arxiv.org/html/2501.02703
- *Conformal novelty detection with false discovery rate control at the
  boundary* — https://arxiv.org/abs/2601.02610
- *Classification with Reject Option: Distribution-free Error Guarantees via
  Conformal Prediction* — https://arxiv.org/abs/2506.21802 (bridges the
  alarm and the admission gate: turns a conformal predictor into a
  reject-option classifier with a proven error rate, and plots **error–reject
  curves** to set an acceptable error or reject rate — the same object as
  the frontier figure, arrived at from the guarantee side)

The honest comparison: Universa's gate is *exact but distributional-
assumption-free only because it is deterministic* — it certifies an
algebraic property of the candidate, and the sealed experiment measured its
refusal rate at 100% on 36 null controls. Conformal methods instead give a
finite-sample FDR guarantee under exchangeability. These are different kinds
of assurance and the paper should say which it is claiming.

## 5. Preregistration and reproducibility in ML

The methodology section's home literature. This is where the paper's
strongest claim lives, and the field is receptive but thin on worked
examples.

- *Perspectives on Machine Learning from Psychology's Reproducibility
  Crisis* — https://arxiv.org/pdf/2104.08878 (the canonical argument that
  hyperparameter search is ML's p-hacking)
- Semmelrock et al., *Reproducibility in machine-learning-based research:
  Overview, barriers, and drivers*, AI Magazine —
  https://onlinelibrary.wiley.com/doi/10.1002/aaai.70002
  (also arXiv: https://arxiv.org/html/2406.14325v1)
- *Reproducibility in Machine Learning-Driven Research* —
  https://arxiv.org/pdf/2307.10320
- *Preregistration for Experiments with AI Agents* —
  https://arxiv.org/html/2606.11217
- COS, *Simulation Studies Preregistration Template* —
  https://www.cos.io/blog/introducing-the-simulation-studies-preregistration-template
  (**the closest formal analogue to what Universa does** — Universa's
  experiments are simulation studies, and this template is the standard to
  compare the protocol against)
- *Reproducibility: The New Frontier in AI Governance* —
  https://arxiv.org/html/2510.11595v1

**What is left over — and it is the strongest thing here.** The literature
argues for preregistration and offers templates; preregistered-only
workshops exist. What is scarce is a *worked series* at this granularity:
ten sealed experiments, cryptographic seals pushed before seeds are opened,
fail-closed preflight, independent recomputation, four failed claims
reported frozen, and six documented occasions where the protocol caught
something — including two that voided seed blocks. The paper should
position against the simulation-studies template specifically and say what
the ceremony adds beyond it (hash-pinned code manifests, seed-block absence
proofs, and the retained-failure discipline).

## 6. Switching itself — conditional computation

The word "switching" has a home: **dynamic neural networks / conditional
computation**, where the architecture activated depends on the input.
Universa's router is a sample-wise dynamic model in this taxonomy, and the
paper should place it there rather than inventing vocabulary.

- Han, Huang, Song, Yang, Wang & Wang, *Dynamic Neural Networks: A Survey* —
  https://arxiv.org/abs/2102.04906 (the standard taxonomy: sample-wise,
  spatial-wise, temporal-wise; Universa is squarely sample-wise)
- *Conditional computation in neural networks: principles and research
  trends* — https://arxiv.org/html/2403.07965v1
- *Dynamic neural networks: advantages and challenges*, National Science
  Review — https://academic.oup.com/nsr/article/11/8/nwae088/7624214

**What is left over.** Dynamic networks switch among *learned* modules
chosen for efficiency — gating, early exit, adaptive depth. The selection is
justified by the loss, and there is nothing to verify. Universa switches
among *structures with certified constraint sets*, so a switch is checkable
after the fact and a wrong switch has a residual signature. Note also the
motive differs: conditional computation is overwhelmingly an efficiency
literature (allocate compute where needed); Universa's switching is a
correctness mechanism, and the efficiency result (71 discovery calls versus
108) is a side effect rather than the goal. Say that explicitly — a reviewer
from this field will otherwise expect FLOPs tables.

---

## Gaps closed since the first pass

All four citation gaps from the first pass are now closed with primary
sources: Chow 1970 (§3), Bodnar et al. NeurIPS 2022 (§1), DreamCoder PLDI
2021 (§4), and the conformal-novelty thread (§4). The two unsearched areas
are done: conformal prediction for the admission gate (§4) and the framing
for "switching," which turned out to be conditional computation rather than
architecture search (§6).

## Gaps still open

- **No citation has been read in full.** Everything here rests on abstracts,
  overviews, and search summaries. Before any claim of the form "X does not
  do Y" reaches a draft, the primary source needs reading — that class of
  claim is exactly what referees check first.
- **No positioning yet against preregistered-only workshop papers**
  specifically; the preregistration literature (§5) is about the practice,
  not about worked ML examples, and a direct comparison to whatever those
  workshops produced would sharpen the methodology claim.
- **The efficiency framing is unexamined.** If the paper engages the
  conditional-computation literature (§6), it inherits an expectation of
  cost accounting that the sealed experiments never measured — wall times
  are recorded as descriptive metadata only, never claim-tested.
