"""Discovery: propose a new constraint structure from transported vectors.

Design doc section 5: the seeded library is architecturally incomplete, so
the discovery head proposes a new structure — here a new low-rank
subspace/operator — that explains the data, certifies it, and only then
admits it to the library. The observation model is the vector-observation
regime, distinct from :mod:`universa.budgets`' scalar probes: across
multiple instances sharing one UNKNOWN target structure we observe
transported vectors

    y_j = f1 a_j,   j = 1..M,

over varying planted source cycles ``a_j``. The planted chain map commutes
with the boundaries in exact integer arithmetic, so every ``y_j`` lies in
``im(f1)``, which is contained in ``ker(B1_true)`` — ``B1' y_j = f0 B1 a_j
= 0`` with residual at float64 rounding scale, never algorithmic error.
The data-supported consistent subspace ``S_disc = span{y_j}`` is therefore
a certified lower approximation of the unknown constraint kernel, and its
annihilator is a candidate constraint operator: a "boundary" whose kernel
is exactly ``S_disc``.

The pipeline, all in certified float64 via :mod:`universa.operators`:

1. :func:`discover_constraint` — certified SVD accumulation. ``S_disc`` is
   the numerical column space of the stacked observations under the
   operators.py rank convention (``max(shape) * eps * sigma_0``; the true
   singular values are O(sigma_0), the rounding residue ~1e-16 * sigma_0,
   so the split is a ~15-orders-of-magnitude gap, not a judgement call).
   The annihilator — an orthonormal basis of ``S_disc^perp``, as rows — is
   computed by :func:`universa.operators.nullspace_basis` applied to the
   support basis, so membership and orthonormality are certified
   independently of the data SVD. The certificate residual
   ``||C_disc Y||_F`` equals the discarded-singular-value energy up to
   float64 rounding (the annihilator rows and the support columns together
   form an orthonormal basis of the ambient space, so Pythagoras splits
   every observation into a kept and a discarded part).
2. FAIL-CLOSED INSUFFICIENCY — never return a weak structure. The rule is
   *dimensional stabilization*: the numerical rank of the first
   ``M - max(1, floor(M/4))`` observations must equal the rank of all
   ``M`` — adding the final 25% must grow the observed dimension by 0.
   Generic transported cycles keep growing the span until the transported
   cycle space is exhausted, so a dimension that is still growing means
   ``M`` is too small to support a stable estimate. Also refused: zero
   observations, an all-zero (rank 0) span, a span that fills the ambient
   space (the annihilator is trivial — there is no constraint to
   discover), and a certificate residual above ``tol``. Every refusal
   returns :class:`DiscoveryInsufficient` with diagnostics; malformed or
   non-finite input raises instead (fail-closed validation).
3. :func:`admit_to_library` — admit only a certified, novel constraint.
   Novelty is the projector distance ``||P_disc - P_i||_F`` between
   ``S_disc`` and every existing library structure's kernel, both
   projectors certified by :mod:`universa.operators` (for equal-dimensional
   subspaces this is ``sqrt(2 * sum sin^2 theta_k)`` over the principal
   angles; it vanishes iff the subspaces coincide). A duplicate or a
   kernel-sharing decoy — the budgets finding: a relabeled boundary has
   the SAME kernel — sits at distance ~1e-16 and is blocked; a genuinely
   new constraint sits at O(1) distance.
4. :func:`discovery_quality` — EVALUATION ONLY, never available to the
   discovery process: principal-angle containment of ``S_disc`` in
   ``ker(B1_true)`` and the coverage fraction
   ``dim(S_disc) / dim(ker(B1_true))`` against the withheld truth.
5. :func:`run_discovery` — end-to-end on the graph-quotient family: the
   true target is dropped from the library (decoys only), observations are
   synthesized deterministically via :func:`universa.generators.subseed`,
   and everything — discovery, admission, quality, and the planted map's
   misfit against the discovered constraint — is returned for audit.

Documented behavior (default sizes, seeds 2026/7/99): the true quotient
target has cycle-rank kernel dimension 6/4/5; ``M = 16`` observations
suffice (the stabilization rule first accepts at ``M = 7`` for seed 2026),
discovery recovers ``ker(B1_true)`` exactly (containment and certificate
residuals ~1e-15, coverage fraction 1.0), admission over the decoy-only
library accepts (novelty distances 1.5-2.2 vs duplicates at 0.0), and the
planted map's misfit against the discovered constraint is ~1e-15, so the
router would now accept the discovered structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from universa.generators import SwitchInstance, make_switch_instance, subseed
from universa.operators import CERT_TOL, nullspace_basis, projector

STABILITY_FRACTION = 0.25
"""Final fraction of observations that must not grow the observed dimension."""

MIN_OBSERVATION_NORM = 1e-9
"""Minimum ``||y_j||`` for a synthesized observation to count as nonzero."""

DEFAULT_NOVELTY_TOL = 1e-6
"""Default novelty threshold: far above duplicate distance (~1e-16), far
below the O(1) distance of a genuinely different subspace."""


@dataclass(frozen=True)
class DiscoveredConstraint:
    """A certified candidate constraint discovered from transported vectors.

    ``boundary`` is the annihilator of the data-supported subspace: an
    orthonormal basis of ``S_disc^perp`` as rows, acting as a boundary on
    the ambient space, so ``ker(boundary) = S_disc`` exactly (up to the
    certificate residual). ``support_basis`` is an orthonormal column basis
    of ``S_disc``; ``certificate_residual`` is ``||boundary @ Y||_F`` over
    the observation set — the discovered constraint annihilates every
    observed vector to at most this norm (per-column norms are bounded by
    the Frobenius norm). ``seeds`` and ``num_observations`` are the
    provenance record.
    """

    boundary: np.ndarray  # annihilator rows: (ambient - coverage, ambient)
    support_basis: np.ndarray  # orthonormal columns spanning S_disc
    certificate_residual: float  # ||boundary @ observations||_F
    num_observations: int
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.boundary, np.ndarray) or (
            self.boundary.ndim != 2
        ):
            raise ValueError("boundary must be a 2-D array")
        if not isinstance(self.support_basis, np.ndarray) or (
            self.support_basis.ndim != 2
        ):
            raise ValueError("support_basis must be a 2-D array")
        ambient = self.support_basis.shape[0]
        if ambient < 1:
            raise ValueError("support_basis must live in a positive dimension")
        if self.coverage < 1 or self.coverage >= ambient:
            raise ValueError("support must be a proper nonzero subspace")
        if self.boundary.shape != (ambient - self.coverage, ambient):
            raise ValueError(
                "boundary must have shape (ambient - coverage, ambient): "
                f"got {self.boundary.shape}, expected "
                f"({ambient - self.coverage}, {ambient})"
            )
        residual = self.certificate_residual
        if not np.isfinite(residual) or residual < 0.0:
            raise ValueError(
                "certificate_residual must be nonnegative and finite"
            )
        if self.num_observations < self.coverage:
            raise ValueError(
                "cannot support a subspace of dimension above the "
                "observation count"
            )
        object.__setattr__(self, "seeds", tuple(int(s) for s in self.seeds))

    @property
    def coverage(self) -> int:
        """``dim(S_disc)``: how much of the kernel the data supports."""
        return int(self.support_basis.shape[1])

    @property
    def ambient_dim(self) -> int:
        return int(self.boundary.shape[1])


@dataclass(frozen=True)
class DiscoveryInsufficient:
    """A refusal: the data cannot support a certified constraint (yet).

    Carries the diagnostics of *why* discovery declined — never a weak
    structure. ``prefix_dim`` / ``observed_dim`` are the numerical ranks
    before and after the final 25% stability holdout;
    ``certificate_residual`` is set when the estimate was formed but failed
    certification, else ``None``.
    """

    reason: str
    num_observations: int
    prefix_dim: int
    observed_dim: int
    certificate_residual: float | None
    tol: float
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("an insufficiency result needs a reason")
        if self.num_observations < 0:
            raise ValueError("num_observations must be nonnegative")
        if not 0 <= self.prefix_dim <= self.observed_dim:
            raise ValueError("need 0 <= prefix_dim <= observed_dim")
        if self.certificate_residual is not None and (
            not np.isfinite(self.certificate_residual)
            or self.certificate_residual < 0.0
        ):
            raise ValueError(
                "certificate_residual must be nonnegative or None"
            )
        if self.tol <= 0.0:
            raise ValueError("tol must be positive")
        object.__setattr__(self, "seeds", tuple(int(s) for s in self.seeds))


DiscoveryResult = DiscoveredConstraint | DiscoveryInsufficient
"""What :func:`discover_constraint` returns: a certificate or a refusal."""


@dataclass(frozen=True)
class AdmissionDecision:
    """The library gate's verdict, with the novelty distances behind it.

    ``distances[i]`` is the certified projector distance between the
    discovered support subspace and the ``i``-th library structure's
    kernel; ``min_distance`` is ``inf`` for an empty library (everything is
    novel there). ``admitted`` requires both certification
    (``certificate_residual <= misfit_tol``) and novelty
    (``min_distance > novelty_tol``); ``reason`` records which gate decided.
    """

    admitted: bool
    distances: tuple[float, ...]
    min_distance: float
    certificate_residual: float
    reason: str


@dataclass(frozen=True)
class DiscoveryQuality:
    """Evaluation-only audit against the withheld true boundary.

    ``containment_residual`` is ``||(I - P_ker) Q_disc||_F`` — the
    Frobenius aggregate of the sines of the principal angles between
    ``S_disc`` and ``ker(B1_true)``; 0 (to float64) iff the discovered
    subspace is contained in the true kernel. ``coverage_fraction`` is
    ``dim(S_disc) / dim(ker(B1_true))``: 1.0 iff discovery recovered the
    whole consistent subspace.
    """

    containment_residual: float
    coverage_fraction: float
    discovered_dim: int
    kernel_dim: int


@dataclass(frozen=True)
class DiscoveryRun:
    """Everything from one end-to-end discovery run, for audit.

    ``switch`` is the full switch instance (the true target is withheld
    from the library — ``library_boundaries`` holds the decoys only).
    ``result`` is the discovery outcome; ``admission``, ``quality``, and
    ``map_misfit`` are ``None`` when discovery refused. ``map_misfit`` is
    ``||C_disc f1 Z||_F`` over the certified source cycle basis ``Z``: the
    residual a router would read when judging the discovered constraint
    against the planted transport.
    """

    seed: int
    switch: SwitchInstance
    library_boundaries: tuple[np.ndarray, ...]
    observations: np.ndarray
    result: DiscoveryResult
    admission: AdmissionDecision | None
    quality: DiscoveryQuality | None
    map_misfit: float | None


def _numerical_rank(matrix: np.ndarray) -> int:
    """Numerical rank under the operators.py SVD tolerance convention."""
    singulars = np.linalg.svd(matrix, compute_uv=False)
    if singulars.size == 0:
        return 0
    rank_tol = max(matrix.shape) * np.finfo(float).eps * singulars[0]
    return int((singulars > rank_tol).sum())


def discover_constraint(
    observations: np.ndarray,
    ambient_dim: int,
    tol: float = CERT_TOL,
    seeds: tuple[int, ...] = (),
) -> DiscoveryResult:
    """Discover a candidate constraint operator from transported vectors.

    ``observations`` is a ``(ambient_dim, M)`` array whose columns are the
    observed vectors ``y_j``. Estimates ``S_disc = span{y_j}`` by certified
    SVD, forms its annihilator as the candidate boundary via
    :func:`universa.operators.nullspace_basis`, and certifies that the
    annihilator kills every observation to ``tol``.

    Fail-closed insufficiency (returns :class:`DiscoveryInsufficient`,
    never a weak structure): no observations; the observed subspace
    dimension grows when adding the final ``max(1, floor(M/4))``
    observations (the estimate has not stabilized — the documented
    sufficiency rule); the span is zero or fills the ambient space (a
    trivial annihilator is not a structure); or the certificate residual
    exceeds ``tol``. The support basis itself is certified at
    ``CERT_TOL``: a basis that cannot be orthonormalized to float64
    precision is a design failure and raises, as does malformed input.
    """
    if not isinstance(observations, np.ndarray) or observations.ndim != 2:
        raise ValueError(
            "observations must be a 2-D array whose columns are the "
            "observed vectors"
        )
    if not isinstance(ambient_dim, int) or ambient_dim < 1:
        raise ValueError("ambient_dim must be a positive integer")
    if observations.shape[0] != ambient_dim:
        raise ValueError(
            f"observations have ambient dimension {observations.shape[0]}, "
            f"expected {ambient_dim}"
        )
    if not np.all(np.isfinite(observations)):
        raise ValueError("observations must be finite")
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    seeds = tuple(int(s) for s in seeds)
    num_observations = observations.shape[1]
    if num_observations == 0:
        return DiscoveryInsufficient(
            "no observations", 0, 0, 0, None, tol, seeds
        )

    u, singulars, _ = np.linalg.svd(observations, full_matrices=True)
    rank_tol = max(observations.shape) * np.finfo(float).eps * singulars[0]
    observed_dim = int((singulars > rank_tol).sum())
    holdout = max(1, int(STABILITY_FRACTION * num_observations))
    prefix_dim = _numerical_rank(observations[:, : num_observations - holdout])
    if prefix_dim != observed_dim:
        return DiscoveryInsufficient(
            f"observed subspace dimension grew from {prefix_dim} to "
            f"{observed_dim} when adding the final {holdout} of "
            f"{num_observations} observations: the estimate has not "
            "stabilized, M is too small",
            num_observations,
            prefix_dim,
            observed_dim,
            None,
            tol,
            seeds,
        )
    if observed_dim == 0:
        return DiscoveryInsufficient(
            "observations carry no nonzero signal",
            num_observations,
            prefix_dim,
            observed_dim,
            None,
            tol,
            seeds,
        )
    if observed_dim == ambient_dim:
        return DiscoveryInsufficient(
            "observations span the whole ambient space: the annihilator is "
            "trivial, there is no constraint to discover",
            num_observations,
            prefix_dim,
            observed_dim,
            None,
            tol,
            seeds,
        )

    basis = u[:, :observed_dim]
    orthonormality = float(
        np.linalg.norm(basis.T @ basis - np.eye(observed_dim))
    )
    if orthonormality > CERT_TOL:
        raise ValueError(
            f"support basis certificate failed: orthonormality "
            f"{orthonormality:.3e} > {CERT_TOL}"
        )
    annihilator = nullspace_basis(basis.T)  # certified basis of S_disc^perp
    boundary = annihilator.basis.T
    certificate_residual = float(np.linalg.norm(boundary @ observations))
    if certificate_residual > tol:
        return DiscoveryInsufficient(
            f"certificate residual {certificate_residual:.3e} exceeds tol "
            f"{tol:.3e}: refusing a weak structure",
            num_observations,
            prefix_dim,
            observed_dim,
            certificate_residual,
            tol,
            seeds,
        )
    return DiscoveredConstraint(
        boundary, basis, certificate_residual, num_observations, seeds
    )


def admit_to_library(
    library_boundaries: tuple[np.ndarray, ...] | list[np.ndarray],
    discovered: DiscoveredConstraint,
    misfit_tol: float,
    novelty_tol: float,
) -> AdmissionDecision:
    """Admit a discovered constraint to the structure library, or refuse.

    Two gates, both fail-closed. Certification: the discovery's own
    certificate residual must pass ``misfit_tol`` (admission never
    re-trusts an uncertified structure). Novelty: the discovered support
    subspace ``S_disc = ker(C_disc)`` must be new relative to EVERY
    existing library structure's kernel — the certified projector distance
    ``||P_disc - P_i||_F`` (both projectors from
    :mod:`universa.operators`) must exceed ``novelty_tol`` for every
    library boundary. This is what blocks re-admitting duplicates and
    kernel-sharing decoys: a relabeled boundary has the same kernel, so its
    projector distance is ~1e-16 and the gate stays shut, while a genuinely
    new consistent subspace sits at O(1) distance. The decision carries the
    distances either way.
    """
    if not isinstance(discovered, DiscoveredConstraint):
        raise ValueError(
            "can only admit a DiscoveredConstraint — an insufficient "
            "discovery is never admissible"
        )
    if misfit_tol <= 0.0 or novelty_tol <= 0.0:
        raise ValueError("tolerances must be positive")
    boundaries = tuple(library_boundaries)
    for boundary in boundaries:
        if not isinstance(boundary, np.ndarray) or boundary.ndim != 2:
            raise ValueError("library boundaries must be 2-D arrays")
        if boundary.shape[1] != discovered.ambient_dim:
            raise ValueError(
                f"library boundary acts on dimension {boundary.shape[1]}, "
                f"but the discovered constraint acts on "
                f"{discovered.ambient_dim}"
            )
    support = projector(nullspace_basis(discovered.boundary))
    distances = tuple(
        float(np.linalg.norm(support - projector(nullspace_basis(boundary))))
        for boundary in boundaries
    )
    min_distance = min(distances, default=float("inf"))
    certified = discovered.certificate_residual <= misfit_tol
    novel = min_distance > novelty_tol
    if not certified:
        reason = (
            f"refused: certificate residual "
            f"{discovered.certificate_residual:.3e} exceeds misfit_tol "
            f"{misfit_tol:.3e}"
        )
    elif not novel:
        reason = (
            f"refused: support subspace duplicates a library kernel (min "
            f"projector distance {min_distance:.3e} <= novelty_tol "
            f"{novelty_tol:.3e})"
        )
    else:
        reason = (
            f"admitted: certified and novel (min projector distance "
            f"{min_distance:.3e} > novelty_tol {novelty_tol:.3e})"
        )
    return AdmissionDecision(
        certified and novel,
        distances,
        min_distance,
        discovered.certificate_residual,
        reason,
    )


def discovery_quality(
    discovered: DiscoveredConstraint, true_boundary: np.ndarray
) -> DiscoveryQuality:
    """Evaluation-only quality of a discovery against the withheld truth.

    NEVER available to the discovery process: ``true_boundary`` is the
    ground-truth target boundary. Reports the principal-angle containment
    of ``S_disc`` in ``ker(B1_true)`` — ``||(I - P_ker) Q_disc||_F`` with
    the certified kernel projector, 0 iff containment is exact — and the
    coverage fraction ``dim(S_disc) / dim(ker(B1_true))``.
    """
    if not isinstance(discovered, DiscoveredConstraint):
        raise ValueError("quality is defined for a DiscoveredConstraint")
    if not isinstance(true_boundary, np.ndarray) or true_boundary.ndim != 2:
        raise ValueError("true_boundary must be a 2-D array")
    if true_boundary.shape[1] != discovered.ambient_dim:
        raise ValueError(
            f"true boundary acts on dimension {true_boundary.shape[1]}, "
            f"but the discovered constraint acts on {discovered.ambient_dim}"
        )
    certificate = nullspace_basis(true_boundary)
    kernel_dim = int(certificate.basis.shape[1])
    if kernel_dim == 0:
        raise ValueError("true boundary has a trivial kernel")
    proj = projector(certificate)
    basis = discovered.support_basis
    containment = float(np.linalg.norm(basis - proj @ basis))
    return DiscoveryQuality(
        containment,
        discovered.coverage / kernel_dim,
        discovered.coverage,
        kernel_dim,
    )


def synthesize_observations(
    instance: SwitchInstance, num_observations: int
) -> np.ndarray:
    """Deterministic transported vectors ``y_j = f1 a_j`` over varying cycles.

    Each ``a_j`` is a certified random source cycle (coefficients over the
    :func:`universa.operators.nullspace_basis` cycle basis, drawn from
    ``subseed(instance.seed, "discovery-observation", j, attempt)``), so
    every column of the result lies in ``im(f1)``, contained in
    ``ker(B1_true)``, with exactness inherited from the planted chain map.
    Draws landing in ``ker(f1)`` (a zero observation) are redrawn with the
    next attempt subseed. Returns an ``(E_target, num_observations)`` array.
    """
    if not isinstance(instance, SwitchInstance):
        raise ValueError("instance must be a SwitchInstance")
    if num_observations < 0:
        raise ValueError("num_observations must be nonnegative")
    cycle_basis = nullspace_basis(instance.source.boundaries[0]).basis
    f1 = instance.chain_map.maps[1]
    ambient_dim = instance.true_target.boundaries[0].shape[1]
    columns = []
    for j in range(num_observations):
        for attempt in range(1, 1001):
            rng = np.random.default_rng(
                subseed(
                    instance.seed,
                    "discovery-observation",
                    str(j),
                    str(attempt),
                )
            )
            cycle = cycle_basis @ rng.standard_normal(cycle_basis.shape[1])
            observed = f1 @ cycle
            if np.linalg.norm(observed) >= MIN_OBSERVATION_NORM:
                columns.append(observed)
                break
        else:
            raise RuntimeError(
                "could not draw a nonzero transported observation"
            )
    if not columns:
        return np.zeros((ambient_dim, 0))
    return np.column_stack(columns)


def run_discovery(
    seed: int,
    num_observations: int = 16,
    num_vertices: int = 8,
    num_edges: int = 14,
    num_classes: int = 6,
    num_decoys: int = 3,
    misfit_tol: float = CERT_TOL,
    novelty_tol: float = DEFAULT_NOVELTY_TOL,
) -> DiscoveryRun:
    """End-to-end discovery on the graph-quotient family.

    Builds the planted switch instance, withholds the true target from the
    library (decoys only — discovery must propose what routing cannot
    select), synthesizes ``num_observations`` transported vectors
    deterministically from ``seed``, discovers, and — when discovery
    succeeds — admits against the decoy library and audits against the
    withheld truth. ``num_classes=6`` (the budgets choice) keeps the
    quotient non-complete so decoys have genuinely different kernels; with
    ``num_observations=16`` the stabilization rule accepts and, for the
    documented seeds, the transported cycles span the whole target kernel.
    """
    switch = make_switch_instance(
        seed, num_vertices, num_edges, num_classes, num_decoys
    )
    library = tuple(d.boundaries[0] for d in switch.decoy_targets)
    observations = synthesize_observations(switch, num_observations)
    ambient_dim = switch.true_target.boundaries[0].shape[1]
    result = discover_constraint(observations, ambient_dim, seeds=(seed,))
    admission = None
    quality = None
    map_misfit = None
    if isinstance(result, DiscoveredConstraint):
        admission = admit_to_library(
            library, result, misfit_tol, novelty_tol
        )
        quality = discovery_quality(result, switch.true_target.boundaries[0])
        cycle_basis = nullspace_basis(switch.source.boundaries[0]).basis
        map_misfit = float(
            np.linalg.norm(
                result.boundary @ switch.chain_map.maps[1] @ cycle_basis
            )
        )
    return DiscoveryRun(
        seed,
        switch,
        library,
        observations,
        result,
        admission,
        quality,
        map_misfit,
    )
