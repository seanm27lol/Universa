"""Learned structure routing, v0: temperature-annealed candidate selection.

This is the repository's first learned (torch) component; everything else is
deliberately numpy-only. The import boundary is strict: ``universa.router``
imports torch, and nothing in the numpy core (including
``universa/__init__.py``) imports this module, so a plain ``import universa``
stays torch-free. All exact suite mathematics (probe residuals, feasible-set
nullities, identifiability thresholds) is computed in float64 by the
certified numpy machinery of :mod:`universa.budgets`; torch sees only the
resulting feature vectors, in float32, on CPU (CUDA is hidden below, before
the torch import).

**Task** (design doc section 4). A switch instance provides a source complex,
``K`` candidate target structures, and ``N`` exact linear probe pairs for a
planted transported quantity. The router scores each candidate and must
select the true target. Ground truth is auditable: the true candidate admits
the truth (identification residual exactly 0.0) and every decoy does not, so
labels are certified at dataset-build time, never assumed.

**Anti-leak discipline.** The generators return the true target at candidate
index 0; :func:`build_dataset` applies a per-instance permutation (drawn from
``subseed(seed, "router-permutation", str(num_probes))``) and records the
permuted true index as the label, so a position heuristic cannot fit.

**Honest expectation.** In this clean regime the exact identification
residual already identifies the true candidate, so the non-learned baseline
``argmin over candidates of the log identification residual`` is essentially
perfect and the learned router is expected only to *roughly match* it. The
v0 deliverable is the annealed-routing machinery and the generalization
plumbing (train on one seed block, evaluate on a disjoint block), not a win
over the residual oracle.

**v1: degraded regimes** (the second half of this module). v0's oracle is
unbeatable only because it reads the *exact* boundary. Under partial
observation (:mod:`universa.partial`) the boundary itself is polluted:
``build_degraded_dataset`` swaps the clean residual for a *degradation
profile* — the commutation misfit of the planted chain map against the
candidate boundary observed through an :class:`ObservationModel`
sign-corruption draw at every fraction of a profile grid (0.0..0.7 by 0.1
by default), plus the profile slopes and v0's structural dims. The
non-learned baseline becomes the *polluted* oracle: argmin over candidates
of the single profile column at the operating fraction. The learned router
reads the whole trajectory — anchored at the exact fraction-0 residual —
so the honest v1 question is whether integrating the trajectory beats the
myopic polluted reading under *held-out* degradation: training on one seed
block over fractions 0.0..0.4, evaluating on a disjoint seed block over
held-out fractions 0.5..0.7 (:func:`evaluate_held_out_regime`). Numbers are
reported per fraction, either way.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only, before the torch import

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from universa.budgets import (
    Probes,
    identifiability_threshold,
    make_budget_instance,
    probe_instance,
    probe_operator,
)
from universa.generators import subseed
from universa.operators import nullspace_basis
from universa.partial import ObservationModel, observed_misfit
from universa.structures import ChainComplex, ChainMap

RESIDUAL_TOL = 1e-9
"""Ground-truth audit tolerance: true residual below, decoy residuals above."""

FEATURE_NAMES = (
    "log1p_identification_residual",
    "feasible_set_nullity",
    "identifiability_threshold",
    "num_probes",
    "probes_over_threshold",
    "num_target_vertices",
    "num_target_edges",
    "boundary_cycle_nullity",
)
"""Per-candidate feature layout, in column order (float64 at build time)."""

FEATURE_DIM = len(FEATURE_NAMES)

DEFAULT_LAMBDA_AUX = 0.01
"""Weight of the load-balancing pressure relative to cross-entropy.

Switch-transformer convention: weak enough that it never distorts the
classification optimum, strong enough to break symmetry against collapse.
"""


def candidate_features(
    probes: Probes, candidate: ChainComplex, truth: np.ndarray
) -> np.ndarray:
    """The fixed-length feature vector for one candidate structure.

    Columns follow :data:`FEATURE_NAMES`: ``log1p`` of the certified
    identification residual (distance from the planted truth to the
    candidate's feasible set, float64), the feasible-set nullity under the
    current probe budget, the candidate's identifiability threshold, the
    probe count ``N`` and the ratio ``N / threshold``, and structural dims
    (``V'``, ``E'``, cycle nullity of the candidate boundary). The two
    certified nullities (threshold and boundary cycle nullity) are computed
    independently and must agree — a disagreement is an error, never a
    warning.
    """
    if not isinstance(probes, Probes):
        raise ValueError("probes must be a Probes instance")
    if candidate.top_degree != 1:
        raise ValueError("routing candidates are 1-complexes (graphs)")
    boundary = candidate.boundaries[0]
    if boundary.shape[1] != probes.dim:
        raise ValueError(
            f"candidate acts on dimension {boundary.shape[1]} but probes "
            f"have dimension {probes.dim}"
        )
    if not isinstance(truth, np.ndarray) or truth.shape != (probes.dim,):
        raise ValueError("truth must be a vector of the probe dimension")
    recovered = probe_instance(probes, boundary)
    residual = recovered.distance(truth)  # == identification_residual(...)
    threshold = identifiability_threshold(candidate)
    if threshold < 1:
        raise ValueError("candidate has threshold 0: nothing to route to")
    cycle_nullity = int(nullspace_basis(boundary).basis.shape[1])
    if cycle_nullity != threshold:
        raise ValueError(
            f"certified nullities disagree: threshold {threshold} vs "
            f"boundary cycle nullity {cycle_nullity}"
        )
    num_probes = probes.count
    num_vertices, num_edges = boundary.shape
    return np.array(
        [
            np.log1p(residual),
            float(recovered.nullity),
            float(threshold),
            float(num_probes),
            num_probes / threshold,
            float(num_vertices),
            float(num_edges),
            float(cycle_nullity),
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class InstanceMetadata:
    """Provenance for one dataset row.

    ``permutation[i]`` is the original candidate index sitting at permuted
    position ``i`` (original index 0 is the generator's true target), so
    ``permutation[true_index] == 0`` always holds and the label can be
    reconstructed from the stored permutation.
    """

    seed: int
    num_probes: int
    threshold: int
    permutation: tuple[int, ...]
    true_index: int

    def __post_init__(self) -> None:
        k = len(self.permutation)
        if k < 2:
            raise ValueError("need at least two candidates")
        if sorted(self.permutation) != list(range(k)):
            raise ValueError("permutation must be a permutation of 0..K-1")
        if not 0 <= self.true_index < k:
            raise ValueError("true_index out of range")
        if self.permutation[self.true_index] != 0:
            raise ValueError("true_index must locate original candidate 0")
        if self.num_probes < 0 or self.threshold < 1:
            raise ValueError("bad probe count or threshold")


def build_dataset(
    seeds,
    num_probes_grid,
    *,
    num_vertices: int = 8,
    num_edges: int = 14,
    num_classes: int = 6,
    num_decoys: int = 3,
) -> tuple[np.ndarray, np.ndarray, tuple[InstanceMetadata, ...]]:
    """Certified routing dataset over a block of generator seeds.

    Returns ``(features, labels, metadata)`` with ``features`` of shape
    ``(M, K, F)`` in float64 — batch-first, matching the torch convention
    and :meth:`StructureRouter.forward` — where ``M = len(seeds) *
    len(num_probes_grid)``, ``K = 1 + num_decoys`` (fixed per dataset for
    batching), and ``F = FEATURE_DIM``. ``labels`` has shape ``(M,)`` and
    holds the permuted true index.

    Every row is audited fail-closed: the true candidate's identification
    residual must be at most ``RESIDUAL_TOL`` and every decoy's strictly
    above it; a violation is an error, never a dropped row. Probes are the
    nested prefixes of :func:`universa.budgets.probe_operator`, so the
    budget-``N`` row uses exactly the first ``N`` probes of
    ``max(num_probes_grid)`` and the whole dataset is deterministic from the
    seed block.
    """
    seeds = tuple(int(s) for s in seeds)
    grid = tuple(int(n) for n in num_probes_grid)
    if not seeds:
        raise ValueError("seeds must be nonempty")
    if not grid or any(n < 0 for n in grid):
        raise ValueError("num_probes_grid must be nonempty and nonnegative")
    if num_decoys < 1:
        raise ValueError("need at least one decoy (K >= 2)")
    k = num_decoys + 1
    features: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[InstanceMetadata] = []
    for seed in seeds:
        instance = make_budget_instance(
            seed, num_vertices, num_edges, num_classes, num_decoys
        )
        dim = instance.true_target.boundaries[0].shape[1]
        operator = probe_operator(seed, dim, max(grid))
        observations = operator @ instance.transported
        for num_probes in grid:
            probes = Probes(
                operator[:num_probes], observations[:num_probes]
            )
            per_candidate = [
                candidate_features(probes, candidate, instance.transported)
                for candidate in instance.candidates
            ]
            rng = np.random.default_rng(
                subseed(seed, "router-permutation", str(num_probes))
            )
            permutation = tuple(int(p) for p in rng.permutation(k))
            true_index = permutation.index(0)
            block = np.stack([per_candidate[p] for p in permutation])
            residuals = np.expm1(block[:, 0])  # undo log1p, float64 exact
            if residuals[true_index] > RESIDUAL_TOL:
                raise RuntimeError(
                    f"seed {seed}, N={num_probes}: true candidate residual "
                    f"{residuals[true_index]:.3e} exceeds {RESIDUAL_TOL}"
                )
            decoy_residuals = np.delete(residuals, true_index)
            if float(decoy_residuals.min()) <= RESIDUAL_TOL:
                raise RuntimeError(
                    f"seed {seed}, N={num_probes}: a decoy residual "
                    f"{float(decoy_residuals.min()):.3e} is at or below "
                    f"{RESIDUAL_TOL}; ground truth is not separable"
                )
            features.append(block)
            labels.append(true_index)
            metadata.append(
                InstanceMetadata(
                    seed=seed,
                    num_probes=num_probes,
                    threshold=instance.threshold,
                    permutation=permutation,
                    true_index=true_index,
                )
            )
    return (
        np.stack(features),
        np.asarray(labels, dtype=np.int64),
        tuple(metadata),
    )


def anneal_temperature(
    epoch: int, total: int, tau_start: float = 2.0, tau_end: float = 0.25
) -> float:
    """Exponential temperature decay from ``tau_start`` to ``tau_end``.

    ``tau(epoch) = tau_start * (tau_end / tau_start) ** frac`` with ``frac``
    linear in the epoch and clamped to ``[0, 1]``: exactly ``tau_start`` at
    epoch 0, exactly ``tau_end`` from epoch ``total - 1`` on, monotone
    non-increasing throughout, and never below ``tau_end``.
    """
    if total < 1:
        raise ValueError("total must be >= 1")
    if not 0.0 < tau_end <= tau_start:
        raise ValueError("need 0 < tau_end <= tau_start")
    frac = min(max(epoch / max(total - 1, 1), 0.0), 1.0)
    return max(tau_end, tau_start * (tau_end / tau_start) ** frac)


def load_balancing_loss(gates: torch.Tensor) -> torch.Tensor:
    """Switch-style auxiliary pressure against router collapse.

    For a batch of soft gates ``(B, K)`` with per-candidate batch means
    ``f_k``: ``K * sum_k f_k^2 - 1``. Since the gates of every instance sum
    to 1, so do the means, and Cauchy-Schwarz gives ``K * sum_k f_k^2 >= 1``
    with equality iff ``f_k = 1/K`` for every ``k`` — so the shifted loss is
    nonnegative and exactly zero iff routing is perfectly uniform across the
    batch (the uniform minimum 1 is subtracted so the floor is 0).
    """
    if not isinstance(gates, torch.Tensor) or gates.ndim != 2:
        raise ValueError("gates must be a (B, K) tensor")
    mean_gates = gates.mean(dim=0)
    return gates.shape[1] * (mean_gates * mean_gates).sum() - 1.0


class StructureRouter(nn.Module):
    """Shared per-candidate MLP encoder producing one logit per candidate.

    Architecture: ``F -> 64 -> 64 -> 1``, applied independently to every
    candidate's feature vector, yielding ``K`` logits. ``forward(features,
    tau, hard=False)`` takes features of shape ``(B, K, F)`` and returns
    gates ``(B, K)``: ``softmax(logits / tau)`` in soft mode; in hard mode
    the one-hot argmax with a straight-through gradient (``one_hot + soft -
    soft.detach()``), so the returned gates are exactly discrete while
    gradients still reach the logits. Inference is always discrete — use
    :func:`hard_predictions`.

    The input standardization (mean/std measured on the training block by
    :func:`train_router`) is stored in buffers so the model is
    self-contained: raw float features in, gates out.
    """

    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = 64):
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1:
            raise ValueError("feature_dim and hidden_dim must be >= 1")
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer("input_mean", torch.zeros(feature_dim))
        self.register_buffer("input_std", torch.ones(feature_dim))

    def logits(self, features: torch.Tensor) -> torch.Tensor:
        """Per-candidate logits, shape ``(B, K)`` from ``(B, K, F)``."""
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features must have shape (B, K, {self.feature_dim})"
            )
        standardized = (features - self.input_mean) / self.input_std
        return self.encoder(standardized).squeeze(-1)

    def forward(
        self, features: torch.Tensor, tau: float = 1.0, hard: bool = False
    ) -> torch.Tensor:
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        logits = self.logits(features)
        soft = torch.softmax(logits / tau, dim=-1)
        if not hard:
            return soft
        index = logits.argmax(dim=-1, keepdim=True)
        one_hot = torch.zeros_like(soft).scatter_(-1, index, 1.0)
        return one_hot + soft - soft.detach()


def _as_float32(features) -> torch.Tensor:
    return torch.as_tensor(np.asarray(features), dtype=torch.float32)


def hard_predictions(model: StructureRouter, features) -> np.ndarray:
    """Strictly discrete argmax over candidate logits — inference is always
    discrete, never sampled or mixed."""
    tensor = _as_float32(features)
    with torch.no_grad():
        logits = model.logits(tensor)
    return logits.argmax(dim=-1).cpu().numpy()


def hard_accuracy(model: StructureRouter, features, labels) -> float:
    """Fraction of instances whose hard (argmax) prediction is the label."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be a 1-D array")
    predictions = hard_predictions(model, features)
    if predictions.shape[0] != labels.shape[0]:
        raise ValueError("features/labels instance count mismatch")
    return float((predictions == labels).mean())


def argmin_residual_accuracy(features, labels) -> float:
    """Non-learned baseline: argmin over candidates of the log residual.

    Uses feature column 0 (``log1p`` of the certified identification
    residual; monotone in the residual, so this is exactly argmin of the
    residual) and no learning. In this clean regime it is essentially
    perfect — the reference floor the learned router is expected to roughly
    match, reported beside every learned number.
    """
    features = np.asarray(features)
    labels = np.asarray(labels)
    if features.ndim != 3 or features.shape[-1] != FEATURE_DIM:
        raise ValueError(f"features must have shape (M, K, {FEATURE_DIM})")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must be one per instance")
    predictions = features[..., 0].argmin(axis=1)
    return float((predictions == labels).mean())


def train_router(
    dataset: tuple[np.ndarray, np.ndarray, tuple[InstanceMetadata, ...]],
    val_dataset: tuple[np.ndarray, np.ndarray, tuple[InstanceMetadata, ...]],
    epochs: int,
    lr: float = 1e-3,
    seed: int = 0,
    lambda_aux: float = DEFAULT_LAMBDA_AUX,
    tau_start: float = 2.0,
    tau_end: float = 0.25,
    hidden_dim: int = 64,
    standardize: bool = True,
    feature_dim: int = FEATURE_DIM,
) -> tuple[StructureRouter, dict]:
    """Train a :class:`StructureRouter`, deterministically, on CPU.

    Full-batch Adam (every epoch is one step, so ``epochs`` is also the step
    count and no shuffling RNG exists); ``torch.manual_seed(seed)`` governs
    initialization. The objective is cross-entropy on the true index at the
    current temperature, ``CE(logits / tau, label)``, plus
    ``lambda_aux * load_balancing_loss(soft gates)`` (default 0.01, see
    :data:`DEFAULT_LAMBDA_AUX`). ``tau`` follows :func:`anneal_temperature`
    over the epochs — soft early for hedged exploration, near-hard late;
    inference is always discrete.

    Input standardization statistics are measured on the *training* block
    only and stored in the model's buffers (the validation block is
    transformed with the training statistics — no leakage).

    ``feature_dim`` is the expected per-candidate feature count — v0's
    :data:`FEATURE_DIM` by default, :func:`degraded_feature_dim` for the
    degraded-regime datasets of v1; datasets built with any other feature
    width are rejected fail-closed.

    History records, per epoch: total loss, cross-entropy, auxiliary loss,
    tau, train/val accuracy under *hard* inference (strictly discrete
    argmax), mean soft-gate entropy on the training batch (nats), and the
    per-candidate usage histogram of hard predictions on the training batch
    (the collapse check). A non-finite loss is an error, never a warning.
    """
    features, labels, _ = dataset
    val_features, val_labels, _ = val_dataset
    features = np.asarray(features)
    labels = np.asarray(labels)
    val_features = np.asarray(val_features)
    val_labels = np.asarray(val_labels)
    if features.ndim != 3 or features.shape[-1] != feature_dim:
        raise ValueError(f"features must have shape (M, K, {feature_dim})")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must be one per instance")
    k = features.shape[1]
    if k < 2:
        raise ValueError("need at least two candidates")
    if labels.size and (labels.min() < 0 or labels.max() >= k):
        raise ValueError("labels out of range")
    if not np.isfinite(features).all():
        raise ValueError("features must be finite")
    if val_features.ndim != 3 or val_features.shape[1:] != (k, feature_dim):
        raise ValueError("validation features must share (K, F) with train")
    if val_labels.ndim != 1 or val_labels.shape[0] != val_features.shape[0]:
        raise ValueError("validation labels must be one per instance")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if lambda_aux < 0.0:
        raise ValueError("lambda_aux must be nonnegative")
    if feature_dim < 1:
        raise ValueError("feature_dim must be >= 1")

    torch.manual_seed(seed)
    model = StructureRouter(feature_dim, hidden_dim)
    x = _as_float32(features)
    y = torch.as_tensor(labels, dtype=torch.long)
    vx = _as_float32(val_features)
    vy = torch.as_tensor(val_labels, dtype=torch.long)
    if standardize:
        flat = x.reshape(-1, x.shape[-1])
        model.input_mean.copy_(flat.mean(dim=0))
        model.input_std.copy_(flat.std(dim=0, unbiased=False).clamp_min(1e-8))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list] = {
        "loss": [],
        "cross_entropy": [],
        "aux_loss": [],
        "tau": [],
        "train_accuracy": [],
        "val_accuracy": [],
        "gate_entropy": [],
        "usage": [],
    }
    for epoch in range(epochs):
        tau = anneal_temperature(epoch, epochs, tau_start, tau_end)
        model.train()
        logits = model.logits(x)
        gates = torch.softmax(logits / tau, dim=-1)
        cross_entropy = F.cross_entropy(logits / tau, y)
        aux = load_balancing_loss(gates)
        loss = cross_entropy + lambda_aux * aux
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
        if not np.isfinite(loss_value):
            raise FloatingPointError(
                f"non-finite training loss at epoch {epoch}"
            )
        model.eval()
        with torch.no_grad():
            train_logits = model.logits(x)
            train_gates = torch.softmax(train_logits / tau, dim=-1)
            train_pred = train_logits.argmax(dim=-1)
            val_pred = model.logits(vx).argmax(dim=-1)
            entropy = -(
                train_gates * train_gates.clamp_min(1e-12).log()
            ).sum(dim=-1).mean()
        history["loss"].append(loss_value)
        history["cross_entropy"].append(float(cross_entropy.detach()))
        history["aux_loss"].append(float(aux.detach()))
        history["tau"].append(tau)
        history["train_accuracy"].append(
            float((train_pred == y).double().mean())
        )
        history["val_accuracy"].append(
            float((val_pred == vy).double().mean())
        )
        history["gate_entropy"].append(float(entropy))
        history["usage"].append(
            np.bincount(train_pred.cpu().numpy(), minlength=k)
        )
    history["usage"] = np.stack(history["usage"])
    return model, history


# ---------------------------------------------------------------------------
# Router v1: degraded regimes — partial observation of the boundary.
#
# v0's argmin-residual oracle is unbeatable only because it reads the exact
# boundary. v1 pollutes that oracle: candidates are observed through
# partial.ObservationModel sign-corruption draws, and the per-candidate
# feature is a DEGRADATION PROFILE — the commutation misfit of the planted
# chain map against the observed candidate boundary at every fraction of a
# profile grid — plus the profile slopes and v0's structural dims. The
# learned router integrates the whole trajectory; the non-learned baseline
# is the myopic polluted oracle reading only the operating fraction's
# column. The protocol trains on one seed block over low fractions and
# evaluates on a disjoint seed block over held-out higher fractions.

DEFAULT_PROFILE_GRID = tuple(i / 10 for i in range(8))
"""Default degradation profile grid: fractions 0.0 to 0.7 by 0.1."""

DEGRADED_STRUCTURAL_NAMES = (
    "num_target_vertices",
    "num_target_edges",
    "boundary_cycle_nullity",
)
"""The structural-dims block carried over from v0's feature layout."""


def _validated_profile_grid(profile_grid) -> tuple[float, ...]:
    """Fail-closed profile-grid validation: nonempty, within [0, 1],
    strictly increasing, and anchored at exactly 0.0 (the fraction-0 column
    is the exact-residual anchor the row audits below rely on)."""
    grid = tuple(float(g) for g in profile_grid)
    if not grid or grid[0] != 0.0:
        raise ValueError("profile grid must be nonempty and start at 0.0")
    if any(not 0.0 <= g <= 1.0 for g in grid):
        raise ValueError("profile grid fractions must lie in [0, 1]")
    if any(b <= a for a, b in zip(grid, grid[1:])):
        raise ValueError("profile grid must be strictly increasing")
    return grid


def _fraction_key(fraction: float) -> str:
    """Deterministic string key for one fraction in subseed components."""
    return f"{fraction:.6f}"


def degraded_feature_names(
    profile_grid=DEFAULT_PROFILE_GRID,
) -> tuple[str, ...]:
    """Per-candidate v1 feature layout, in column order (float64 at build).

    For a grid of ``G`` fractions the layout is ``2G - 1 + 3`` columns:
    ``log1p`` of the observed commutation misfit at every grid fraction,
    then the profile slopes (first differences, ``slope_{i} = profile[i+1]
    - profile[i]``, named by the grid fractions they span), then the
    structural dims :data:`DEGRADED_STRUCTURAL_NAMES`.
    """
    grid = _validated_profile_grid(profile_grid)
    profile = tuple(f"log1p_observed_misfit_fraction_{g:.1f}" for g in grid)
    slopes = tuple(
        f"profile_slope_{a:.1f}_to_{b:.1f}" for a, b in zip(grid, grid[1:])
    )
    return profile + slopes + DEGRADED_STRUCTURAL_NAMES


def degraded_feature_dim(profile_grid=DEFAULT_PROFILE_GRID) -> int:
    """Feature width of a degraded-regime dataset row over ``profile_grid``."""
    return len(degraded_feature_names(profile_grid))


def degraded_candidate_features(
    chain_map: ChainMap,
    candidate: ChainComplex,
    observation_seed: int,
    profile_grid=DEFAULT_PROFILE_GRID,
) -> np.ndarray:
    """Degradation-profile feature vector for one candidate structure.

    Columns follow :func:`degraded_feature_names`. The candidate boundary
    is observed through :class:`ObservationModel` sign-corruption draws
    (``corrupt_fraction = g`` for every grid fraction ``g``, masking off)
    under the shared ``observation_seed`` — one draw per (instance,
    fraction), reused across candidates exactly as
    :func:`universa.partial.ranking_study` does — and the profile entry is
    ``log1p`` of :func:`universa.partial.observed_misfit` of the planted
    chain map against the observed candidate. At grid fraction 0 the
    observation is exact, so the first column is ``log1p`` of the clean
    commutation residual; log1p is strictly increasing, so the stored
    profile keeps the residual's monotonicity (the true candidate's is
    structurally non-decreasing, see :mod:`universa.partial`). The
    certified-nullity cross-check from v0 is kept fail-closed.
    """
    grid = _validated_profile_grid(profile_grid)
    if not isinstance(chain_map, ChainMap):
        raise ValueError("chain_map must be a ChainMap instance")
    if candidate.top_degree != 1:
        raise ValueError("routing candidates are 1-complexes (graphs)")
    boundary = candidate.boundaries[0]
    misfits = [
        observed_misfit(
            chain_map,
            ObservationModel(
                candidate, observation_seed, corrupt_fraction=g
            ).observe(),
        )
        for g in grid
    ]
    profile = np.log1p(np.asarray(misfits, dtype=np.float64))
    slopes = np.diff(profile)
    threshold = identifiability_threshold(candidate)
    cycle_nullity = int(nullspace_basis(boundary).basis.shape[1])
    if cycle_nullity != threshold:
        raise ValueError(
            f"certified nullities disagree: threshold {threshold} vs "
            f"boundary cycle nullity {cycle_nullity}"
        )
    num_vertices, num_edges = boundary.shape
    return np.concatenate(
        [
            profile,
            slopes,
            np.array(
                [float(num_vertices), float(num_edges), float(cycle_nullity)],
                dtype=np.float64,
            ),
        ]
    )


@dataclass(frozen=True)
class DegradedInstanceMetadata:
    """Provenance for one degraded-regime dataset row (instance x fraction).

    Same contract as v0's :class:`InstanceMetadata`: ``permutation[i]`` is
    the original candidate index sitting at permuted position ``i``, so
    ``permutation[true_index] == 0`` always holds.
    """

    seed: int
    fraction: float
    threshold: int
    permutation: tuple[int, ...]
    true_index: int

    def __post_init__(self) -> None:
        k = len(self.permutation)
        if k < 2:
            raise ValueError("need at least two candidates")
        if sorted(self.permutation) != list(range(k)):
            raise ValueError("permutation must be a permutation of 0..K-1")
        if not 0 <= self.true_index < k:
            raise ValueError("true_index out of range")
        if self.permutation[self.true_index] != 0:
            raise ValueError("true_index must locate original candidate 0")
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError("fraction outside [0, 1]")
        if self.threshold < 1:
            raise ValueError("bad threshold")


def build_degraded_dataset(
    seeds,
    fractions,
    *,
    profile_grid=DEFAULT_PROFILE_GRID,
    num_vertices: int = 8,
    num_edges: int = 14,
    num_classes: int = 6,
    num_decoys: int = 3,
) -> tuple[np.ndarray, np.ndarray, tuple[DegradedInstanceMetadata, ...]]:
    """Certified routing dataset over (seed, corruption fraction) rows.

    Returns ``(features, labels, metadata)`` with ``features`` of shape
    ``(M, K, F)`` in float64 where ``M = len(seeds) * len(fractions)``,
    ``K = 1 + num_decoys``, and ``F = degraded_feature_dim(profile_grid)``;
    ``labels`` holds the permuted true index. Every row of ``fractions``
    must be a point of ``profile_grid`` — the row's fraction is what the
    polluted oracle reads, so it must name a profile column.

    Each row shares ONE observation draw across candidates
    (``subseed(seed, "router-v1-observe", fraction)``), exactly as
    :func:`universa.partial.ranking_study` shares its draw, and the profile
    sweeps ``corrupt_fraction`` over the grid (nested draws: a larger
    fraction flips a prefix-extension of the same master permutation). The
    per-row candidate permutation is drawn from
    ``subseed(seed, "router-v1-permutation", fraction)`` and the permuted
    true index is recorded as the audited label, exactly as v0.

    Fail-closed row audits, adapted to degradation: at profile fraction 0
    (the exact observation) the true candidate's misfit must be at most
    ``RESIDUAL_TOL`` and every decoy's strictly above it — a violation is
    an error, never a dropped row. At higher grid fractions the polluted
    values are just recorded, never audit-failed.
    """
    grid = _validated_profile_grid(profile_grid)
    seeds = tuple(int(s) for s in seeds)
    fractions = tuple(float(f) for f in fractions)
    if not seeds:
        raise ValueError("seeds must be nonempty")
    if not fractions:
        raise ValueError("fractions must be nonempty")
    for fraction in fractions:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"fraction {fraction} outside [0, 1]")
        if not any(abs(g - fraction) <= 1e-9 for g in grid):
            raise ValueError(
                f"fraction {fraction} is not a profile grid point; the "
                "row fraction must name a profile column"
            )
    if num_decoys < 1:
        raise ValueError("need at least one decoy (K >= 2)")
    k = num_decoys + 1
    features: list[np.ndarray] = []
    labels: list[int] = []
    metadata: list[DegradedInstanceMetadata] = []
    for seed in seeds:
        instance = make_budget_instance(
            seed, num_vertices, num_edges, num_classes, num_decoys
        )
        for fraction in fractions:
            key = _fraction_key(fraction)
            observation_seed = subseed(seed, "router-v1-observe", key)
            per_candidate = [
                degraded_candidate_features(
                    instance.chain_map, candidate, observation_seed, grid
                )
                for candidate in instance.candidates
            ]
            rng = np.random.default_rng(
                subseed(seed, "router-v1-permutation", key)
            )
            permutation = tuple(int(p) for p in rng.permutation(k))
            true_index = permutation.index(0)
            block = np.stack([per_candidate[p] for p in permutation])
            if not np.isfinite(block).all():
                raise RuntimeError(
                    f"seed {seed}, fraction {fraction}: non-finite features"
                )
            clean = np.expm1(block[:, 0])  # grid fraction 0: exact column
            if clean[true_index] > RESIDUAL_TOL:
                raise RuntimeError(
                    f"seed {seed}, fraction {fraction}: true candidate "
                    f"misfit {clean[true_index]:.3e} at fraction 0 exceeds "
                    f"{RESIDUAL_TOL}"
                )
            decoy_clean = np.delete(clean, true_index)
            if float(decoy_clean.min()) <= RESIDUAL_TOL:
                raise RuntimeError(
                    f"seed {seed}, fraction {fraction}: a decoy misfit "
                    f"{float(decoy_clean.min()):.3e} at fraction 0 is at or "
                    f"below {RESIDUAL_TOL}; ground truth is not separable"
                )
            features.append(block)
            labels.append(true_index)
            metadata.append(
                DegradedInstanceMetadata(
                    seed=seed,
                    fraction=fraction,
                    threshold=instance.threshold,
                    permutation=permutation,
                    true_index=true_index,
                )
            )
    return (
        np.stack(features),
        np.asarray(labels, dtype=np.int64),
        tuple(metadata),
    )


def observed_residual_oracle_accuracy(
    features,
    labels,
    fraction: float,
    *,
    profile_grid=DEFAULT_PROFILE_GRID,
) -> float:
    """The polluted oracle: argmin over candidates of the observed misfit.

    Reads the single profile column at ``fraction`` (log1p of the observed
    commutation misfit; log1p is strictly increasing, so this is exactly
    argmin of the misfit) and predicts its argmin — no learning. At
    fraction 0 this is the exact-residual oracle and is essentially
    perfect; at higher fractions the observation is polluted and the
    reading degrades. This is the reference the learned router is honestly
    compared against at every eval fraction.
    """
    grid = _validated_profile_grid(profile_grid)
    features = np.asarray(features)
    labels = np.asarray(labels)
    expected = degraded_feature_dim(grid)
    if features.ndim != 3 or features.shape[-1] != expected:
        raise ValueError(f"features must have shape (M, K, {expected})")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must be one per instance")
    matches = [i for i, g in enumerate(grid) if abs(g - fraction) <= 1e-9]
    if not matches:
        raise ValueError(f"fraction {fraction} is not a profile grid point")
    predictions = features[..., matches[0]].argmin(axis=1)
    return float((predictions == labels).mean())


@dataclass(frozen=True)
class FractionReport:
    """Learned-vs-oracle accuracies at one eval fraction."""

    fraction: float
    num_instances: int
    learned_accuracy: float
    oracle_accuracy: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError("fraction outside [0, 1]")
        if self.num_instances < 1:
            raise ValueError("num_instances must be >= 1")
        for name, value in (
            ("learned_accuracy", self.learned_accuracy),
            ("oracle_accuracy", self.oracle_accuracy),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} outside [0, 1]")


@dataclass(frozen=True)
class DegradedRegimeReport:
    """Result of a held-out-regime evaluation (router v1).

    ``per_fraction`` carries one :class:`FractionReport` per eval fraction
    — the deliverable: the learned router's hard-inference accuracy beside
    the polluted oracle's, at each held-out degradation level.
    ``history`` is the :func:`train_router` history (excluded from
    equality: it holds numpy arrays).
    """

    train_seeds: tuple[int, ...]
    eval_seeds: tuple[int, ...]
    train_fractions: tuple[float, ...]
    eval_fractions: tuple[float, ...]
    profile_grid: tuple[float, ...]
    final_train_accuracy: float
    final_eval_accuracy: float
    per_fraction: tuple[FractionReport, ...]
    history: dict = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if set(self.train_seeds) & set(self.eval_seeds):
            raise ValueError("train/eval seed blocks overlap")
        if set(self.train_fractions) & set(self.eval_fractions):
            raise ValueError("train/eval fractions overlap")
        if not self.per_fraction:
            raise ValueError("per_fraction must be nonempty")
        for name, value in (
            ("final_train_accuracy", self.final_train_accuracy),
            ("final_eval_accuracy", self.final_eval_accuracy),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} outside [0, 1]")


def evaluate_held_out_regime(
    train_seeds,
    eval_seeds,
    train_fractions,
    eval_fractions,
    *,
    profile_grid=DEFAULT_PROFILE_GRID,
    epochs: int = 150,
    lr: float = 1e-3,
    seed: int = 0,
    lambda_aux: float = DEFAULT_LAMBDA_AUX,
    tau_start: float = 2.0,
    tau_end: float = 0.25,
    hidden_dim: int = 64,
    num_vertices: int = 8,
    num_edges: int = 14,
    num_classes: int = 6,
    num_decoys: int = 3,
) -> tuple[StructureRouter, DegradedRegimeReport]:
    """The v1 protocol: generalization to unseen degradation.

    Trains a :class:`StructureRouter` (:func:`train_router`, annealed,
    load-balanced, deterministic) on the degraded dataset over
    ``train_seeds`` x ``train_fractions`` and evaluates on the DISJOINT
    ``eval_seeds`` x the HELD-OUT ``eval_fractions`` — split hygiene is
    fail-closed: any seed or fraction overlap is an error. At every eval
    fraction the learned router's hard-inference (strictly discrete
    argmax) accuracy is reported beside the polluted oracle's
    (:func:`observed_residual_oracle_accuracy` at that fraction). The
    honest v1 question — does the learned router beat the polluted oracle
    under held-out degradation? — is answered by the numbers in the
    report, whichever way they go.
    """
    train_seeds = tuple(int(s) for s in train_seeds)
    eval_seeds = tuple(int(s) for s in eval_seeds)
    train_fractions = tuple(float(f) for f in train_fractions)
    eval_fractions = tuple(float(f) for f in eval_fractions)
    if not train_seeds or not eval_seeds:
        raise ValueError("train and eval seed blocks must be nonempty")
    if set(train_seeds) & set(eval_seeds):
        raise ValueError("train/eval seed blocks must be disjoint")
    if not train_fractions or not eval_fractions:
        raise ValueError("train and eval fractions must be nonempty")
    if set(train_fractions) & set(eval_fractions):
        raise ValueError("train/eval fractions must be disjoint")
    grid = _validated_profile_grid(profile_grid)
    dim = degraded_feature_dim(grid)
    train_data = build_degraded_dataset(
        train_seeds,
        train_fractions,
        profile_grid=grid,
        num_vertices=num_vertices,
        num_edges=num_edges,
        num_classes=num_classes,
        num_decoys=num_decoys,
    )
    eval_data = build_degraded_dataset(
        eval_seeds,
        eval_fractions,
        profile_grid=grid,
        num_vertices=num_vertices,
        num_edges=num_edges,
        num_classes=num_classes,
        num_decoys=num_decoys,
    )
    model, history = train_router(
        train_data,
        eval_data,
        epochs,
        lr=lr,
        seed=seed,
        lambda_aux=lambda_aux,
        tau_start=tau_start,
        tau_end=tau_end,
        hidden_dim=hidden_dim,
        feature_dim=dim,
    )
    features, labels, metadata = eval_data
    rows: list[FractionReport] = []
    for fraction in eval_fractions:
        mask = np.array([m.fraction == fraction for m in metadata])
        block_features, block_labels = features[mask], labels[mask]
        rows.append(
            FractionReport(
                fraction=fraction,
                num_instances=int(mask.sum()),
                learned_accuracy=hard_accuracy(
                    model, block_features, block_labels
                ),
                oracle_accuracy=observed_residual_oracle_accuracy(
                    block_features, block_labels, fraction, profile_grid=grid
                ),
            )
        )
    report = DegradedRegimeReport(
        train_seeds=train_seeds,
        eval_seeds=eval_seeds,
        train_fractions=train_fractions,
        eval_fractions=eval_fractions,
        profile_grid=grid,
        final_train_accuracy=history["train_accuracy"][-1],
        final_eval_accuracy=history["val_accuracy"][-1],
        per_fraction=tuple(rows),
        history=history,
    )
    return model, report
