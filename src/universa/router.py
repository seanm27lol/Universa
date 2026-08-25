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
over the residual oracle. A v1 targets degraded regimes (partial
observation, :mod:`universa.partial`) where exact residuals are polluted and
learning is motivated.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only, before the torch import

from dataclasses import dataclass

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
from universa.structures import ChainComplex

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
    if features.ndim != 3 or features.shape[-1] != FEATURE_DIM:
        raise ValueError(f"features must have shape (M, K, {FEATURE_DIM})")
    if labels.ndim != 1 or labels.shape[0] != features.shape[0]:
        raise ValueError("labels must be one per instance")
    k = features.shape[1]
    if k < 2:
        raise ValueError("need at least two candidates")
    if labels.size and (labels.min() < 0 or labels.max() >= k):
        raise ValueError("labels out of range")
    if not np.isfinite(features).all():
        raise ValueError("features must be finite")
    if val_features.ndim != 3 or val_features.shape[1:] != (k, FEATURE_DIM):
        raise ValueError("validation features must share (K, F) with train")
    if val_labels.ndim != 1 or val_labels.shape[0] != val_features.shape[0]:
        raise ValueError("validation labels must be one per instance")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    if lr <= 0.0:
        raise ValueError("lr must be positive")
    if lambda_aux < 0.0:
        raise ValueError("lambda_aux must be nonnegative")

    torch.manual_seed(seed)
    model = StructureRouter(features.shape[-1], hidden_dim)
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
