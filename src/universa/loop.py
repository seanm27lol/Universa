"""The route-or-discover loop: routing and discovery as one closed cycle.

Design doc sections 4 and 5, integrated. The loop holds a seeded library of
K candidate structures. Per instance it scores every candidate by the
certified commutation residual of the instance's planted chain map against
it — exactly :meth:`universa.generators.SwitchInstance.commutation_scores`
semantics (a :class:`universa.structures.ChainMap` probe per candidate, max
residual over degrees; one degree in the graph-quotient family) — and reads
the best (minimum) score:

* ``best <= alarm_tol``: the library explains the instance. ROUTE mode:
  commit to the argmin candidate — hard routing at inference (design
  section 4: exactness composes only under commitment), strictly discrete,
  first-index tie-breaking.
* ``best > alarm_tol``: the misfit alarm FIRES — design section 3's
  quantitative "none of my structures explains this". The loop runs the
  certified discovery head of :mod:`universa.discovery` over vector
  observations transported through the planted map, gates the result
  through :func:`universa.discovery.admit_to_library` against the current
  library, and — only on admission — appends the discovered constraint and
  RE-ROUTES the instance to it (DISCOVER mode). A discovery that cannot
  certify, or that certifies but duplicates an existing library kernel, is
  REFUSED: the library stays as it was and nothing is routed.

The exactness argument is inherited from the generators and the discovery
head, and is what makes the alarm a decision rather than a judgment: the
planted quotient map commutes with exact integer arithmetic, so the true
candidate's score is exactly 0.0 in float64 (asserted, never tolerated),
while the generator accepts decoys only above a 1e-9 residual, and the
discovery residuals on transported cycles are float64 rounding scale
(~1e-15) against 1e-9/1e-10 gates — gaps of orders of magnitude, matching
the operators.py rank convention.

Constants (the frozen exp4 — discovery-sealed-1 — values). ``ALARM_TOL`` is
re-pinned locally at 1e-9, exactly ``universa.router.RESIDUAL_TOL``: this
module is numpy-only by contract and ``universa.router`` imports torch, so
the value is duplicated and documented here rather than imported.
``misfit_tol`` defaults to :data:`universa.operators.CERT_TOL` (1e-10) and
``novelty_tol`` to :data:`universa.discovery.DEFAULT_NOVELTY_TOL` (1e-6),
imported so the frozen values have one source. ``num_observations`` = 16,
the frozen exp4 observation count.

Paired conditions. :func:`make_loop_instance` returns ONE budget instance
and two library views of it: the in-library view ``instance.candidates``
(true target at index 0) and the out-of-library view
``instance.decoy_targets`` (the truth withheld). The two views of the same
instance are the paired conditions of the loop experiment: route mode is
success on the first, discover mode on the second.

Documented deviations and non-features:

* Observation schedule. The discovery path calls
  :func:`universa.discovery.synthesize_observations` directly — the frozen
  exp4 schedule ``subseed(seed, "discovery-observation", j, attempt)`` with
  redraw-on-zero — rather than a parallel "loop-observations" subseed
  family: reuse over reimplementation keeps one canonical observation
  schedule in the repository, and the draws are bit-identical to exp4's
  structured condition. ``run_loop`` requires ``seed == instance.seed`` so
  the provenance record cannot name a different seed than the schedule
  consumed.
* No permutation. The generators place the true target at library index 0
  and this module never permutes; :func:`routing_correct` reads index 0.
  Anti-leak permutation is the experiment runner's concern (the
  ``universa.router`` convention), not the loop's.
* The grown library is not returned. On admission the final library is the
  initial library with the discovered boundary appended; the outcome
  records sizes and the routed index. Retaining matrices (observations,
  discovered boundary) is the runner's concern — exactly the exp4
  retention split — so :class:`LoopOutcome` stays a pure scalar record and
  two runs compare bit-identical with plain equality.
* Re-routing to a discovered structure is certified by the discovery
  machinery's router-acceptance residual ``map_misfit = ||C_disc f1 Z||_F``
  over the certified source cycle basis ``Z`` (the
  :func:`universa.discovery.run_discovery` formula), not by a ChainMap
  probe: the discovered annihilator's row space is not the quotient's
  vertex space, so ``commutation_residuals`` is not defined against it.
  ``map_misfit <= 1e-9`` is the router-acceptance criterion (see
  :func:`acquisition_correct`).
* The loop never reads the withheld truth: no containment/coverage audit
  against the true boundary happens here. That evaluation-only audit
  (:func:`universa.discovery.discovery_quality`) belongs to the experiment
  runner, as in exp4.

Everything is fail-closed: malformed inputs, dimension mismatches, and
provenance mismatches raise; certified-machinery errors from
:mod:`universa.operators` and :mod:`universa.discovery` propagate, never
swallowed; an insufficient discovery is a refused outcome, never a weak
structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from universa.budgets import BudgetInstance, make_budget_instance
from universa.discovery import (
    DEFAULT_NOVELTY_TOL,
    DiscoveredConstraint,
    DiscoveryInsufficient,
    admit_to_library,
    discover_constraint,
    synthesize_observations,
)
from universa.generators import SwitchInstance, subseed
from universa.operators import CERT_TOL, nullspace_basis
from universa.structures import ChainComplex, ChainMap

ALARM_TOL = 1e-9
"""Misfit-alarm threshold, re-pinned at ``universa.router.RESIDUAL_TOL``.

Duplicated rather than imported: this module is numpy-only by contract and
``universa.router`` imports torch at module level. The value is also exp4's
router-acceptance threshold for the discovered-constraint map misfit and
the generators' decoy-acceptance floor, so the alarm separates exactly:
true candidate at exactly 0.0, decoys strictly above.
"""


@dataclass(frozen=True)
class LibraryLoop:
    """The frozen loop configuration (the exp4 constants).

    ``alarm_tol`` is the misfit-alarm threshold (:data:`ALARM_TOL` = 1e-9,
    re-pinned from ``universa.router.RESIDUAL_TOL`` — see the module
    docstring). ``misfit_tol`` is the certification gate consulted by both
    :func:`universa.discovery.discover_constraint` and
    :func:`universa.discovery.admit_to_library`
    (:data:`universa.operators.CERT_TOL` = 1e-10). ``novelty_tol`` is the
    admission novelty gate (:data:`universa.discovery.DEFAULT_NOVELTY_TOL`
    = 1e-6). ``num_observations`` is the frozen exp4 observation count, 16:
    at the frozen sizes the stabilization rule accepts on the documented
    seeds.
    """

    alarm_tol: float = ALARM_TOL
    misfit_tol: float = CERT_TOL
    novelty_tol: float = DEFAULT_NOVELTY_TOL
    num_observations: int = 16

    def __post_init__(self) -> None:
        for name in ("alarm_tol", "misfit_tol", "novelty_tol"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        if int(self.num_observations) < 1:
            raise ValueError("num_observations must be >= 1")
        object.__setattr__(
            self, "num_observations", int(self.num_observations)
        )


@dataclass(frozen=True)
class LoopOutcome:
    """The auditable record of one loop pass over one instance.

    ``mode`` is ``"route"`` (alarm quiet, committed to the argmin
    candidate), ``"discover"`` (alarm fired, discovery certified and
    admitted, re-routed to the appended structure), or ``"refused"``
    (alarm fired, discovery insufficient or admission blocked — nothing
    routed, library unchanged). ``scores`` holds the certified commutation
    scores of the INITIAL library, in library order — the alarm decision
    basis; ``best_score`` is their minimum. ``routed_index`` indexes the
    FINAL library: in discover mode that is exactly
    ``initial_library_size`` (the appended structure's index), in route
    mode an index into the unchanged library.

    Discovery-phase provenance: ``discovery_verdict`` (``"discovered"`` /
    ``"insufficient"``) and ``discovery_reason`` describe the head's
    outcome; ``certificate_residual`` is the discovered constraint's own
    certificate (set for a certified discovery, and also for an
    insufficiency that formed but failed certification). Admission-phase
    provenance: ``admitted``, ``admission_min_distance``,
    ``admission_distances`` (per-library-entry certified projector
    distances) and ``admission_reason``. ``map_misfit`` is the
    router-acceptance residual ``||C_disc f1 Z||_F``, set whenever
    discovery certified. ``seed``, ``initial_library_size``, and
    ``num_observations`` pin the run's provenance.

    Construction is fail-closed: cross-field coherence (mode vs alarm vs
    routed index vs library growth) is validated here, so an incoherent
    record is an error, never a silent state.
    """

    mode: str  # "route" | "discover" | "refused"
    alarm_fired: bool
    best_score: float
    scores: tuple[float, ...]
    routed_index: int | None
    discovery_verdict: str | None
    discovery_reason: str | None
    certificate_residual: float | None
    admitted: bool
    admission_min_distance: float | None
    map_misfit: float | None
    final_library_size: int
    seed: int
    initial_library_size: int
    num_observations: int | None
    admission_distances: tuple[float, ...] | None
    admission_reason: str | None

    def __post_init__(self) -> None:
        if self.mode not in ("route", "discover", "refused"):
            raise ValueError(f"unknown loop mode {self.mode!r}")
        scores = tuple(float(s) for s in self.scores)
        if not scores:
            raise ValueError("scores must be nonempty")
        if any(not np.isfinite(s) or s < 0.0 for s in scores):
            raise ValueError("scores must be nonnegative and finite")
        object.__setattr__(self, "scores", scores)
        best = float(self.best_score)
        if not np.isfinite(best) or best != min(scores):
            raise ValueError("best_score must be the minimum of scores")
        object.__setattr__(self, "best_score", best)
        if self.initial_library_size != len(scores):
            raise ValueError("initial_library_size must equal len(scores)")
        if self.final_library_size < self.initial_library_size:
            raise ValueError("the library never shrinks")
        object.__setattr__(self, "alarm_fired", bool(self.alarm_fired))
        object.__setattr__(self, "admitted", bool(self.admitted))
        object.__setattr__(self, "seed", int(self.seed))
        for name in (
            "certificate_residual",
            "admission_min_distance",
            "map_misfit",
        ):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not np.isfinite(value) or value < 0.0:
                    raise ValueError(f"{name} must be nonnegative and finite")
                object.__setattr__(self, name, value)
        if self.num_observations is not None:
            if int(self.num_observations) < 0:
                raise ValueError("num_observations must be nonnegative")
            object.__setattr__(
                self, "num_observations", int(self.num_observations)
            )
        if self.admission_distances is not None:
            distances = tuple(float(d) for d in self.admission_distances)
            if any(not np.isfinite(d) or d < 0.0 for d in distances):
                raise ValueError(
                    "admission_distances must be nonnegative and finite"
                )
            object.__setattr__(self, "admission_distances", distances)

        if self.mode == "route":
            if self.alarm_fired:
                raise ValueError("route mode requires an unfired alarm")
            if self.routed_index is None or not (
                0 <= self.routed_index < len(scores)
            ):
                raise ValueError(
                    "route mode needs a routed index into the library"
                )
            if scores[self.routed_index] != best:
                raise ValueError(
                    "the routed index must attain the best score"
                )
            if self.admitted:
                raise ValueError("route mode admits nothing")
            if self.final_library_size != self.initial_library_size:
                raise ValueError("route mode leaves the library unchanged")
            for name in (
                "discovery_verdict",
                "discovery_reason",
                "certificate_residual",
                "admission_min_distance",
                "map_misfit",
                "num_observations",
                "admission_distances",
                "admission_reason",
            ):
                if getattr(self, name) is not None:
                    raise ValueError(
                        f"route mode has no discovery phase ({name} is set)"
                    )
        else:
            # discover and refused both come from the alarm path.
            if not self.alarm_fired:
                raise ValueError(
                    f"{self.mode} mode requires a fired alarm"
                )
            if self.num_observations is None:
                raise ValueError(
                    f"{self.mode} mode consumed observations: count required"
                )
            if self.discovery_verdict == "discovered":
                self._check_certified_discovery()
            elif self.discovery_verdict != "insufficient":
                raise ValueError(
                    "the discovery path needs a verdict: "
                    '"discovered" or "insufficient"'
                )
            if self.mode == "discover" and (
                self.discovery_verdict != "discovered"
            ):
                raise ValueError(
                    "discover mode requires a certified discovery"
                )
        if self.mode == "discover":
            if not self.admitted:
                raise ValueError("discover mode requires admission")
            if self.routed_index != self.initial_library_size:
                raise ValueError(
                    "discover mode routes to the appended structure: "
                    "routed_index must equal initial_library_size"
                )
            if self.final_library_size != self.initial_library_size + 1:
                raise ValueError("discover mode grows the library by one")
        if self.mode == "refused":
            if self.routed_index is not None:
                raise ValueError("a refused loop routes nowhere")
            if self.admitted:
                raise ValueError("a refused loop admits nothing")
            if self.final_library_size != self.initial_library_size:
                raise ValueError("a refused loop leaves the library unchanged")
            if self.discovery_verdict == "insufficient":
                if not self.discovery_reason:
                    raise ValueError("an insufficiency carries its reason")
                for name in (
                    "admission_min_distance",
                    "map_misfit",
                    "admission_distances",
                    "admission_reason",
                ):
                    if getattr(self, name) is not None:
                        raise ValueError(
                            "an insufficient discovery never reaches "
                            f"admission ({name} is set)"
                        )

    def _check_certified_discovery(self) -> None:
        """Coherence required whenever the head returned a certificate."""
        if not self.discovery_reason:
            raise ValueError("a certified discovery carries a reason")
        if self.certificate_residual is None:
            raise ValueError("a certified discovery carries its residual")
        if self.map_misfit is None:
            raise ValueError(
                "a certified discovery carries the router-acceptance misfit"
            )
        for name in (
            "admission_min_distance",
            "admission_distances",
            "admission_reason",
        ):
            if getattr(self, name) is None:
                raise ValueError(
                    f"a certified discovery passed through admission ({name}"
                    " required)"
                )
        if len(self.admission_distances) != self.initial_library_size:
            raise ValueError(
                "admission_distances must cover the initial library"
            )
        if min(self.admission_distances) != self.admission_min_distance:
            raise ValueError(
                "admission_min_distance must be the minimum of "
                "admission_distances"
            )


def _library_scores(
    instance: BudgetInstance, library: tuple[ChainComplex, ...]
) -> tuple[float, ...]:
    """Certified commutation scores of the planted map against a library.

    Exactly :meth:`universa.generators.SwitchInstance.commutation_scores`
    semantics — a :class:`universa.structures.ChainMap` probe per
    candidate, max residual over degrees — applied to an arbitrary library
    view rather than to ``instance.candidates``.
    """
    scores = []
    for candidate in library:
        probe = ChainMap(
            instance.chain_map.source, candidate, instance.chain_map.maps
        )
        residuals = probe.commutation_residuals()
        scores.append(max(residuals) if residuals else 0.0)
    return tuple(scores)


def run_loop(
    loop: LibraryLoop,
    instance: BudgetInstance,
    library: tuple[ChainComplex, ...] | list[ChainComplex],
    *,
    seed: int,
    observations: np.ndarray | None = None,
) -> LoopOutcome:
    """One deterministic pass of the route-or-discover loop over an instance.

    Phases, all in certified float64:

    1. SCORE — :func:`_library_scores` over ``library`` (the planted map
       probed against every candidate; exactly the ``commutation_scores``
       semantics).
    2. ALARM — ``best = min(scores)``; the alarm fires iff
       ``best > loop.alarm_tol``.
    3. ROUTE (alarm quiet) — commit to the argmin candidate, first-index
       tie-breaking; the library is untouched.
    4. DISCOVERY (alarm fired) — observations are
       :func:`universa.discovery.synthesize_observations` of a
       :class:`~universa.generators.SwitchInstance` view of the budget
       instance (the same underlying arrays; the budget instance only adds
       the planted quantity and threshold, which the schedule does not
       read), i.e. the frozen exp4 ``"discovery-observation"`` draws keyed
       on the instance seed. When ``observations`` is given it replaces the
       synthesized schedule — the H4-style control channel; it is consulted
       only on the discovery path, never when the alarm stays quiet. The
       head certifies via :func:`universa.discovery.discover_constraint` at
       ``loop.misfit_tol``; an insufficiency is a REFUSED outcome.
    5. ADMISSION + RE-ROUTE — a certified discovery goes through
       :func:`universa.discovery.admit_to_library` against the current
       library boundaries at ``(loop.misfit_tol, loop.novelty_tol)``, and
       the router-acceptance residual ``map_misfit = ||C_disc f1 Z||_F`` is
       computed over the certified source cycle basis (the
       :func:`universa.discovery.run_discovery` formula). On admission the
       library grows by one and the instance re-routes to the appended
       index (DISCOVER); a blocked admission is REFUSED.

    ``seed`` must equal ``instance.seed``: the observation schedule keys on
    the instance seed, so the provenance record and the consumed schedule
    cannot disagree. Every candidate must be a 1-complex sharing the
    instance's target boundary shape. Certified-machinery errors propagate;
    nothing is swallowed.
    """
    if not isinstance(loop, LibraryLoop):
        raise ValueError("loop must be a LibraryLoop")
    if not isinstance(instance, BudgetInstance):
        raise ValueError("instance must be a BudgetInstance")
    if int(seed) != int(instance.seed):
        raise ValueError(
            f"seed {seed} does not match instance.seed {instance.seed}: "
            "the observation schedule keys on the instance seed, so the "
            "record and the instance cannot disagree"
        )
    seed = int(seed)
    library = tuple(library)
    if not library:
        raise ValueError("the library must be nonempty")
    true_boundary = instance.true_target.boundaries[0]
    for candidate in library:
        if not isinstance(candidate, ChainComplex) or (
            candidate.top_degree != 1
        ):
            raise ValueError(
                "library candidates must be 1-complexes (graphs)"
            )
        if candidate.boundaries[0].shape != true_boundary.shape:
            raise ValueError(
                f"candidate boundary has shape "
                f"{candidate.boundaries[0].shape} but the instance's target "
                f"boundary has shape {true_boundary.shape}: the library and "
                "the instance must share the target dimensions"
            )
    if observations is not None:
        if not isinstance(observations, np.ndarray) or (
            observations.ndim != 2
        ):
            raise ValueError("observations must be a 2-D array")
        if not np.all(np.isfinite(observations)):
            raise ValueError("observations must be finite")
        if observations.shape[0] != true_boundary.shape[1]:
            raise ValueError(
                f"observations have ambient dimension "
                f"{observations.shape[0]}, expected {true_boundary.shape[1]}"
            )

    scores = _library_scores(instance, library)
    best_score = min(scores)
    if best_score <= loop.alarm_tol:
        return LoopOutcome(
            mode="route",
            alarm_fired=False,
            best_score=best_score,
            scores=scores,
            routed_index=scores.index(best_score),  # first-index tie-break
            discovery_verdict=None,
            discovery_reason=None,
            certificate_residual=None,
            admitted=False,
            admission_min_distance=None,
            map_misfit=None,
            final_library_size=len(library),
            seed=seed,
            initial_library_size=len(library),
            num_observations=None,
            admission_distances=None,
            admission_reason=None,
        )

    # The misfit alarm fired: the discovery path.
    if observations is None:
        switch_view = SwitchInstance(
            instance.seed,
            instance.source,
            instance.true_target,
            instance.chain_map,
            instance.decoy_targets,
        )
        observations = synthesize_observations(
            switch_view, loop.num_observations
        )
    ambient_dim = int(true_boundary.shape[1])
    result = discover_constraint(
        observations, ambient_dim, tol=loop.misfit_tol, seeds=(seed,)
    )
    shared = dict(
        alarm_fired=True,
        best_score=best_score,
        scores=scores,
        seed=seed,
        initial_library_size=len(library),
        num_observations=int(observations.shape[1]),
    )
    if isinstance(result, DiscoveryInsufficient):
        return LoopOutcome(
            mode="refused",
            routed_index=None,
            discovery_verdict="insufficient",
            discovery_reason=result.reason,
            certificate_residual=result.certificate_residual,
            admitted=False,
            admission_min_distance=None,
            map_misfit=None,
            final_library_size=len(library),
            admission_distances=None,
            admission_reason=None,
            **shared,
        )
    library_boundaries = tuple(c.boundaries[0] for c in library)
    admission = admit_to_library(
        library_boundaries, result, loop.misfit_tol, loop.novelty_tol
    )
    cycle_basis = nullspace_basis(instance.source.boundaries[0]).basis
    map_misfit = float(
        np.linalg.norm(
            result.boundary @ instance.chain_map.maps[1] @ cycle_basis
        )
    )
    discovery_reason = (
        f"certified subspace of dimension {result.coverage} in ambient "
        f"dimension {ambient_dim} (certificate residual "
        f"{result.certificate_residual:.3e})"
    )
    if not admission.admitted:
        return LoopOutcome(
            mode="refused",
            routed_index=None,
            discovery_verdict="discovered",
            discovery_reason=discovery_reason,
            certificate_residual=result.certificate_residual,
            admitted=False,
            admission_min_distance=admission.min_distance,
            map_misfit=map_misfit,
            final_library_size=len(library),
            admission_distances=admission.distances,
            admission_reason=admission.reason,
            **shared,
        )
    return LoopOutcome(
        mode="discover",
        routed_index=len(library),  # the appended structure's index
        discovery_verdict="discovered",
        discovery_reason=discovery_reason,
        certificate_residual=result.certificate_residual,
        admitted=True,
        admission_min_distance=admission.min_distance,
        map_misfit=map_misfit,
        final_library_size=len(library) + 1,
        admission_distances=admission.distances,
        admission_reason=admission.reason,
        **shared,
    )


def make_loop_instance(
    seed: int,
) -> tuple[
    BudgetInstance, tuple[ChainComplex, ...], tuple[ChainComplex, ...]
]:
    """One instance, two library views: the paired loop conditions.

    Reuses :func:`universa.budgets.make_budget_instance` at the frozen exp4
    sizes (8 vertices, 14 edges, 6 classes, 3 decoys — a non-complete
    quotient, so decoys have genuinely different kernels). Returns
    ``(instance, in_library, out_library)`` where ``in_library`` is
    ``instance.candidates`` (the true target sits at index 0 by generator
    convention) and ``out_library`` is ``instance.decoy_targets`` (the
    truth withheld): two views of the SAME instance, the paired in-library
    / out-of-library conditions of the loop experiment.
    """
    instance = make_budget_instance(seed, 8, 14, 6, 3)
    return instance, instance.candidates, instance.decoy_targets


def null_observations(
    seed: int, ambient_dim: int, count: int = 16
) -> np.ndarray:
    """The exp4 H4 control: structure-free i.i.d. Gaussian columns.

    Column ``j`` is ``np.random.default_rng(subseed(seed, "discovery-null",
    str(j))).standard_normal(ambient_dim)`` — one independent generator per
    column, exactly the frozen H4 schedule of discovery-sealed-1, a
    disjoint subseed family from the structured condition's
    ``"discovery-observation"`` draws. Fed through the loop's discovery
    path these must be REFUSED: a false constraint certified from noise is
    a specificity failure.
    """
    if ambient_dim < 1:
        raise ValueError("ambient_dim must be positive")
    if count < 0:
        raise ValueError("count must be nonnegative")
    columns = [
        np.random.default_rng(subseed(seed, "discovery-null", str(j)))
        .standard_normal(ambient_dim)
        for j in range(count)
    ]
    if not columns:
        return np.zeros((ambient_dim, 0))
    return np.column_stack(columns)


def acquisition_correct(
    outcome: LoopOutcome, instance: BudgetInstance
) -> bool:
    """End-to-end success predicate for the out-of-library condition.

    True iff the loop detected the misfit (alarm fired), the discovery head
    certified a constraint (verdict ``"discovered"`` with a certificate
    residual), the novelty gate admitted it, AND the planted map commutes
    with the discovered constraint — ``map_misfit <= ALARM_TOL`` (1e-9),
    the router-acceptance residual: the router would accept the discovered
    structure.

    Fail-closed on mode/route inconsistency: a discover-mode outcome that
    does not route to the appended structure is an error, not a False (a
    real :class:`LoopOutcome` cannot be incoherent — construction validates
    — so this guards foreign records). ``instance`` pins the provenance:
    the outcome's seed must match the instance's. The predicate itself
    reads only certified residuals, never the withheld truth.
    """
    if not isinstance(outcome, LoopOutcome):
        raise ValueError("outcome must be a LoopOutcome")
    if not isinstance(instance, BudgetInstance):
        raise ValueError("instance must be a BudgetInstance")
    if outcome.seed != int(instance.seed):
        raise ValueError("outcome/instance seed mismatch")
    if outcome.mode != "discover":
        if outcome.mode not in ("route", "refused"):
            raise ValueError(f"unknown loop mode {outcome.mode!r}")
        return False
    if (
        outcome.routed_index is None
        or outcome.routed_index != outcome.initial_library_size
        or outcome.final_library_size != outcome.initial_library_size + 1
    ):
        raise ValueError(
            "discover-mode outcome does not route to the appended structure"
        )
    return bool(
        outcome.alarm_fired
        and outcome.discovery_verdict == "discovered"
        and outcome.certificate_residual is not None
        and outcome.admitted
        and outcome.map_misfit is not None
        and outcome.map_misfit <= ALARM_TOL
    )


def routing_correct(outcome: LoopOutcome) -> bool:
    """Success predicate for the in-library condition: routed to the truth.

    The generators place the true target at library index 0 and this module
    never permutes (anti-leak permutation is the experiment runner's
    concern, not the loop's), so correct routing on the in-library
    condition is exactly ``mode == "route" and routed_index == 0``.
    """
    if not isinstance(outcome, LoopOutcome):
        raise ValueError("outcome must be a LoopOutcome")
    return outcome.mode == "route" and outcome.routed_index == 0
