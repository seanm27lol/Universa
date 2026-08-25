"""Sub-identifiability probe budgets: how many probes pin down a transport?

HOMYMOLY v2's scarce-probe regime, lifted to the switch task: a quantity is
transported along the planted chain map, but only ``N`` exact linear probe
pairs ``(x_i, y_i)`` with ``y_i = <x_i, a_*>`` are observed. A candidate
structure contributes its constraint ``C a = 0`` (the candidate boundary —
the quantity must be a cycle of the candidate). Recovery under the candidate
means solving the stacked system ``[X; C] a = [y; 0]``.

The **feasible set** is the affine least-squares solution set of the stacked
system, ``{a : M^T M a = M^T b} = particular + ker(M)`` with
``M = [X; C]``, computed exactly through the certified nullspace/projector
machinery of :mod:`universa.operators` (float64, certificates). When the
candidate admits the truth the stacked system is consistent and the feasible
set is the exact solution set; when it does not, the feasible set is the
best-effort affine set and the truth lies at positive distance from it.

**Identifiability threshold.** Let ``N* = nullity(C)`` be the constraint
nullity. For a connected target graph, ``rank(B1) = V - 1``, so

    N* = nullity(B1) = E - V + 1   (the cycle rank, first Betti number).

Generic probes give ``rank([X_N; C]) = rank(C) + min(N, N*)``, hence

    nullity([X_N; C]) = max(N* - N, 0).

Below ``N*`` the true candidate's feasible set still contains the truth
(residual 0) but is positive-dimensional — the transport is not identified.
At ``N >= N*`` the feasible set is the singleton ``{a_*}``: the true
structure's constraint makes the solution unique, and the probe budget is
what closes the gap. Decoys never admit the truth: the planted quantity is
accepted only when ``||C_decoy a_*|| > tol``, and then
``M^T (M a_* - b) = C^T C a_* + X^T (X a_* - y) != 0`` at every budget, so
the truth is never a least-squares solution under a decoy constraint and
every decoy residual stays positive.

Everything is deterministic from an integer seed: probes are standard
Gaussian rows drawn from ``subseed(seed, "probe-operator")`` and budgets
take nested prefixes, so the curve at budget ``N`` uses exactly the first
``N`` probes of the largest requested budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from universa.generators import make_switch_instance, subseed
from universa.operators import (
    CERT_TOL,
    SubspaceCertificate,
    nullspace_basis,
    projector,
)
from universa.structures import ChainComplex, ChainMap

DECOY_TOL = 1e-6
"""Minimum ``||C_decoy a_*||`` for a planted quantity to discriminate decoys."""

MIN_TRUTH_NORM = 0.1
"""Minimum ``||a_*||`` so the planted transported quantity is nondegenerate."""


@dataclass(frozen=True)
class Probes:
    """``N`` exact linear probe pairs, stacked: ``y = X a_*``.

    ``operator`` has shape ``(N, d)`` with row ``i`` the probe ``x_i^T``;
    ``observations`` has shape ``(N,)`` with entry ``y_i``.
    """

    operator: np.ndarray
    observations: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.operator, np.ndarray) or self.operator.ndim != 2:
            raise ValueError("operator must be a 2-D array of stacked probes")
        obs = self.observations
        if not isinstance(obs, np.ndarray) or obs.ndim != 1:
            raise ValueError("observations must be a 1-D array")
        if self.operator.shape[0] != obs.shape[0]:
            raise ValueError(
                f"probe count mismatch: operator has {self.operator.shape[0]} "
                f"rows but observations has {obs.shape[0]} entries"
            )

    @property
    def count(self) -> int:
        return self.operator.shape[0]

    @property
    def dim(self) -> int:
        return self.operator.shape[1]


@dataclass(frozen=True)
class ProbeInstance:
    """A source quantity recovered under one candidate structure's constraint.

    The feasible set is ``particular + ker([X; C])``: the affine least-squares
    solution set of the stacked probe/constraint system, with ``ker([X; C])``
    certified by :func:`universa.operators.nullspace_basis`. ``nullity`` is
    the dimension of the feasible set — 0 iff the candidate's constraint plus
    the probes identify the quantity uniquely.
    """

    probes: Probes
    constraint: np.ndarray
    particular: np.ndarray
    certificate: SubspaceCertificate
    stacked_residual: float  # ||M particular - b||_F; 0 iff the system is consistent

    def __post_init__(self) -> None:
        if not isinstance(self.constraint, np.ndarray) or self.constraint.ndim != 2:
            raise ValueError("constraint must be a 2-D array")
        if self.constraint.shape[1] != self.probes.dim:
            raise ValueError(
                f"constraint acts on dimension {self.constraint.shape[1]} but "
                f"probes have dimension {self.probes.dim}"
            )
        if self.particular.shape != (self.probes.dim,):
            raise ValueError("particular must be a point of the ambient space")

    @property
    def nullity(self) -> int:
        """Dimension of the feasible set: 0 iff recovery is unique."""
        return self.certificate.basis.shape[1]

    def distance(self, point: np.ndarray) -> float:
        """Distance from ``point`` to the feasible set (0 iff admitted)."""
        if point.shape != (self.probes.dim,):
            raise ValueError("point must be a vector of the ambient dimension")
        proj = projector(self.certificate)
        delta = point - self.particular
        return float(np.linalg.norm(delta - proj @ delta))


def probe_instance(probes: Probes, candidate_constraint: np.ndarray) -> ProbeInstance:
    """Recover the feasible set of ``probes`` under ``candidate_constraint``.

    Builds the stacked operator ``M = [X; C]`` and right-hand side
    ``b = [y; 0]``, certifies ``ker(M)``, and takes ``particular`` as the
    minimum-norm least-squares solution with its (numerically negligible)
    kernel component removed via the certified projector — an exact point of
    the affine feasible set ``{a : M^T M a = M^T b}``.
    """
    if not isinstance(probes, Probes):
        raise ValueError("probes must be a Probes instance")
    if not isinstance(candidate_constraint, np.ndarray) or candidate_constraint.ndim != 2:
        raise ValueError("candidate_constraint must be a 2-D array")
    if candidate_constraint.shape[1] != probes.dim:
        raise ValueError(
            f"constraint acts on dimension {candidate_constraint.shape[1]} "
            f"but probes have dimension {probes.dim}"
        )
    stacked = np.vstack([probes.operator, candidate_constraint])
    rhs = np.concatenate(
        [probes.observations, np.zeros(candidate_constraint.shape[0])]
    )
    certificate = nullspace_basis(stacked)
    proj = projector(certificate)
    (least_squares, *_) = np.linalg.lstsq(stacked, rhs, rcond=None)
    particular = least_squares - proj @ least_squares
    stacked_residual = float(np.linalg.norm(stacked @ particular - rhs))
    return ProbeInstance(
        probes, candidate_constraint, particular, certificate, stacked_residual
    )


def identification_residual(
    probes: Probes, candidate_constraint: np.ndarray, truth: np.ndarray
) -> float:
    """Distance from the planted truth to the candidate's feasible set.

    0 (within float64 tolerance) iff the candidate admits the truth: the
    truth satisfies the probe equations and lies in the candidate's
    constraint kernel. Positive for every decoy at every budget.
    """
    return probe_instance(probes, candidate_constraint).distance(truth)


def probe_operator(seed: int, dim: int, count: int) -> np.ndarray:
    """Deterministic ``(count, dim)`` standard Gaussian probe operator.

    Budgets take nested prefixes of this draw: the first ``N`` rows are the
    budget-``N`` probe set for every ``N <= count``.
    """
    if dim < 1 or count < 0:
        raise ValueError("need dim >= 1 and count >= 0")
    rng = np.random.default_rng(subseed(seed, "probe-operator"))
    return rng.standard_normal((count, dim))


def identifiability_threshold(target: ChainComplex) -> int:
    """Certified constraint nullity of a candidate target structure.

    For a connected target graph this is ``nullity(B1) = E - V + 1`` (the
    cycle rank, first Betti number): the probe budget at which the true
    candidate's feasible set collapses to the singleton truth. The value is
    the certified numerical nullity, not the closed form, so it applies to
    any 1-complex, connected or not.
    """
    if target.top_degree != 1:
        raise ValueError("thresholds are defined for 1-complexes (graphs)")
    certificate = nullspace_basis(target.boundaries[0])
    return int(certificate.basis.shape[1])


@dataclass(frozen=True)
class BudgetInstance:
    """A switch instance with a planted transported quantity and a certified
    identifiability threshold.

    ``source_cycle`` is a planted cycle of the source graph; ``transported``
    is its image under the planted chain map, exactly a cycle of the true
    target and (by the acceptance rule) of no decoy. ``threshold`` is the
    certified constraint nullity shared by every candidate.
    """

    seed: int
    source: ChainComplex
    true_target: ChainComplex
    chain_map: ChainMap
    decoy_targets: tuple[ChainComplex, ...]
    source_cycle: np.ndarray
    transported: np.ndarray
    threshold: int

    @property
    def candidates(self) -> tuple[ChainComplex, ...]:
        return (self.true_target, *self.decoy_targets)


def make_budget_instance(
    seed: int,
    num_vertices: int = 8,
    num_edges: int = 14,
    num_classes: int = 6,
    num_decoys: int = 3,
    decoy_tol: float = DECOY_TOL,
) -> BudgetInstance:
    """The graph-quotient family with a planted transport and a threshold.

    Reuses :func:`universa.generators.make_switch_instance` for the source
    graph, planted quotient chain map, and decoys. The planted quantity is a
    certified random source cycle transported along the chain map; a draw is
    accepted only if it is nondegenerate, exactly a cycle of the true target
    (within ``CERT_TOL`` — the chain map commutes with exact integer
    arithmetic), and outside every decoy's cycle space by ``decoy_tol``.

    **Threshold formula.** ``threshold = nullity(B1_true) = E_t - V_t + 1``
    for the connected quotient target (cycle rank); with generic probes the
    true candidate's feasible set has nullity ``max(threshold - N, 0)`` at
    budget ``N``. Instances whose true target is a tree (threshold 0, no
    sub-identifiability regime) are rejected.

    **Decoy discriminability.** The default ``num_classes=6`` keeps the
    quotient non-complete (``E_t <= num_edges < C(num_classes, 2)``), so
    decoys are genuinely different graphs rather than vertex relabelings.
    This matters: a relabeled boundary ``P B1`` has the *same* kernel as
    ``B1``, so no transported cycle can ever discriminate it — such instances
    are detected structurally (the residual operator ``B1_decoy f1`` vanishes
    on the source cycle space) and rejected with an error rather than
    silently producing decoys that admit the truth.
    """
    switch = make_switch_instance(
        seed, num_vertices, num_edges, num_classes, num_decoys
    )
    cycle_certificate = nullspace_basis(switch.source.boundaries[0])
    source_nullity = cycle_certificate.basis.shape[1]
    if source_nullity == 0:
        raise ValueError("source graph has no cycle to plant")
    threshold = identifiability_threshold(switch.true_target)
    if threshold == 0:
        raise ValueError(
            "true target is a tree: threshold 0, no sub-identifiability regime"
        )
    for candidate in switch.candidates:
        if identifiability_threshold(candidate) != threshold:
            raise ValueError("candidates must share the constraint nullity")

    f1 = switch.chain_map.maps[1]
    true_boundary = switch.true_target.boundaries[0]
    cycle_basis = cycle_certificate.basis
    for decoy in switch.decoy_targets:
        spread = float(np.linalg.norm(decoy.boundaries[0] @ f1 @ cycle_basis))
        if spread <= 1e-9:
            raise ValueError(
                "decoy admits every transported cycle (B1_decoy f1 vanishes "
                "on the source cycle space, e.g. a vertex-relabeled "
                "complete quotient); choose parameters with a non-complete "
                "quotient so decoys have different kernels"
            )
    for attempt in range(1, 1001):
        rng = np.random.default_rng(subseed(seed, "planted", str(attempt)))
        coeffs = rng.standard_normal(source_nullity)
        source_cycle = cycle_certificate.basis @ coeffs
        transported = f1 @ source_cycle
        if np.linalg.norm(transported) < MIN_TRUTH_NORM:
            continue
        if np.linalg.norm(true_boundary @ transported) > CERT_TOL:
            continue  # exactness failed: never accept silently
        if switch.decoy_targets:
            worst = min(
                float(np.linalg.norm(d.boundaries[0] @ transported))
                for d in switch.decoy_targets
            )
            if worst <= decoy_tol:
                continue
        return BudgetInstance(
            seed,
            switch.source,
            switch.true_target,
            switch.chain_map,
            switch.decoy_targets,
            source_cycle,
            transported,
            threshold,
        )
    raise RuntimeError("could not plant a discriminating transported quantity")


@dataclass(frozen=True)
class BudgetCurve:
    """Per-candidate identification residuals and nullities over probe budgets.

    ``residuals[c][i]`` / ``nullities[c][i]`` are for candidate ``c`` (index
    0 is the true target) at budget ``budgets[i]``.
    """

    budgets: tuple[int, ...]
    threshold: int
    residuals: tuple[tuple[float, ...], ...]
    nullities: tuple[tuple[int, ...], ...]


def budget_curve(
    seed: int,
    budgets: tuple[int, ...],
    num_vertices: int = 8,
    num_edges: int = 14,
    num_classes: int = 6,
    num_decoys: int = 3,
) -> BudgetCurve:
    """Identification residuals per candidate over a grid of probe budgets.

    Probes are the nested prefixes of :func:`probe_operator` applied to the
    planted transported quantity, so the curve is deterministic from
    ``(seed, budgets, ...)`` and the budget-``N`` row uses exactly the first
    ``N`` probes of ``max(budgets)``.
    """
    instance = make_budget_instance(
        seed, num_vertices, num_edges, num_classes, num_decoys
    )
    budgets = tuple(int(b) for b in budgets)
    if not budgets or any(b < 0 for b in budgets):
        raise ValueError("budgets must be a nonempty sequence of nonnegative integers")
    dim = instance.true_target.boundaries[0].shape[1]
    operator = probe_operator(seed, dim, max(budgets))
    observations = operator @ instance.transported
    residuals: list[tuple[float, ...]] = []
    nullities: list[tuple[int, ...]] = []
    for candidate in instance.candidates:
        candidate_residuals: list[float] = []
        candidate_nullities: list[int] = []
        for budget in budgets:
            probes = Probes(operator[:budget], observations[:budget])
            recovered = probe_instance(probes, candidate.boundaries[0])
            candidate_residuals.append(recovered.distance(instance.transported))
            candidate_nullities.append(recovered.nullity)
        residuals.append(tuple(candidate_residuals))
        nullities.append(tuple(candidate_nullities))
    return BudgetCurve(
        budgets, instance.threshold, tuple(residuals), tuple(nullities)
    )
