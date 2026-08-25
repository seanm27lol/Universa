import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU-only, before the torch import

import numpy as np
import pytest
import torch

import universa
from universa.budgets import (
    Probes,
    identification_residual,
    make_budget_instance,
    probe_operator,
)
from universa.router import (
    FEATURE_DIM,
    FEATURE_NAMES,
    InstanceMetadata,
    StructureRouter,
    anneal_temperature,
    argmin_residual_accuracy,
    build_dataset,
    candidate_features,
    hard_accuracy,
    hard_predictions,
    load_balancing_loss,
    train_router,
)

BUILD_SEEDS = (11, 12, 13)
GRID = (1, 2, 3)
NUM_DECOYS = 2  # K = 3
K = 1 + NUM_DECOYS


def small_dataset():
    return build_dataset(BUILD_SEEDS, GRID, num_decoys=NUM_DECOYS)


def test_dataset_shapes_finiteness_determinism():
    features, labels, metadata = small_dataset()
    rows = len(BUILD_SEEDS) * len(GRID)
    assert features.shape == (rows, K, FEATURE_DIM)
    assert FEATURE_DIM == len(FEATURE_NAMES) == 8
    assert features.dtype == np.float64
    assert np.isfinite(features).all()
    assert labels.shape == (rows,) and labels.dtype == np.int64
    assert 0 <= labels.min() and labels.max() < K
    assert len(metadata) == rows
    # Deterministic rebuild: identical features, labels, and metadata.
    again = small_dataset()
    assert np.array_equal(features, again[0])
    assert np.array_equal(labels, again[1])
    assert metadata == again[2]


def test_permutation_labeling_reconstructs_true_index():
    features, labels, metadata = small_dataset()
    for i, meta in enumerate(metadata):
        # The stored permutation is a genuine permutation of 0..K-1...
        assert sorted(meta.permutation) == list(range(K))
        # ...and locates the generator's true target (original index 0).
        assert meta.permutation[meta.true_index] == 0
        assert labels[i] == meta.true_index
        # The labeled candidate is exactly the one with residual ~0.
        assert np.expm1(features[i, meta.true_index, 0]) <= 1e-9
        for position in range(K):
            if position != meta.true_index:
                assert np.expm1(features[i, position, 0]) > 1e-9
    # The permutation actually moves the true index off position 0, so a
    # position heuristic cannot fit the labels.
    assert len(set(int(label) for label in labels)) > 1


def test_feature_values_match_certified_residuals():
    features, _, metadata = small_dataset()
    for row, meta in enumerate(metadata):
        instance = make_budget_instance(meta.seed, num_decoys=NUM_DECOYS)
        dim = instance.true_target.boundaries[0].shape[1]
        operator = probe_operator(meta.seed, dim, max(GRID))
        probes = Probes(
            operator[: meta.num_probes],
            (operator @ instance.transported)[: meta.num_probes],
        )
        for position, original in enumerate(meta.permutation):
            candidate = instance.candidates[original]
            residual = identification_residual(
                probes, candidate.boundaries[0], instance.transported
            )
            assert np.isclose(
                np.expm1(features[row, position, 0]), residual, rtol=1e-9
            )
            threshold = instance.threshold
            assert features[row, position, 2] == threshold
            assert features[row, position, 3] == meta.num_probes
            assert features[row, position, 4] == meta.num_probes / threshold
            vertices, edges = candidate.boundaries[0].shape
            assert features[row, position, 5] == vertices
            assert features[row, position, 6] == edges
            assert features[row, position, 7] == threshold


def test_anneal_temperature_monotone_endpoints():
    total = 40
    taus = [anneal_temperature(epoch, total) for epoch in range(total)]
    assert taus[0] == 2.0
    assert taus[-1] == 0.25
    for earlier, later in zip(taus, taus[1:]):
        assert later <= earlier
    # Equal-ratio exponential decay in log space.
    log_ratios = np.diff(np.log(taus[1:-1]))
    assert np.allclose(log_ratios, log_ratios[0])
    # Clamped at tau_end beyond the schedule, and at tau_start before it.
    assert anneal_temperature(total + 100, total) == 0.25
    assert anneal_temperature(0, total) == 2.0
    with pytest.raises(ValueError):
        anneal_temperature(0, 0)  # empty schedule
    with pytest.raises(ValueError):
        anneal_temperature(0, total, tau_start=0.25, tau_end=2.0)  # inverted


def fixed_logit_model():
    torch.manual_seed(0)
    model = StructureRouter()
    torch.manual_seed(1)
    features = torch.randn(2, 4, FEATURE_DIM)
    logits = model.logits(features).detach()
    # Fixture sanity: no near-ties, so low-temperature assertions are exact.
    pairwise = [
        float((logits[..., i] - logits[..., j]).abs().min())
        for i in range(4)
        for j in range(i + 1, 4)
    ]
    assert min(pairwise) > 1e-2
    return model, features


def test_gates_approach_one_hot_as_tau_decreases():
    model, features = fixed_logit_model()
    taus = [4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.01, 0.001]
    entropies = []
    with torch.no_grad():
        for tau in taus:
            gates = model(features, tau=tau)
            entropy = float(
                -(gates * gates.clamp_min(1e-12).log()).sum(-1).mean()
            )
            entropies.append(entropy)
        # Entropy of softmax(logits / tau) is monotone non-increasing in 1/tau.
        for earlier, later in zip(entropies, entropies[1:]):
            assert later <= earlier + 1e-9
        # As tau -> 0 the gates approach one-hot.
        cold = model(features, tau=0.001)
        assert float(cold.max(dim=-1).values.min()) > 1.0 - 1e-3
        assert entropies[-1] < 1e-3


def test_hard_forward_is_one_hot_argmax():
    model, features = fixed_logit_model()
    hard = model(features, tau=0.7, hard=True)
    assert ((hard == 0) | (hard == 1)).all()
    assert torch.allclose(hard.sum(dim=-1), torch.ones(features.shape[0]))
    assert torch.equal(hard.argmax(dim=-1), model.logits(features).argmax(dim=-1))
    with pytest.raises(ValueError):
        model(features, tau=0.0)


def test_straight_through_gradient_reaches_logits():
    model, features = fixed_logit_model()
    weights = torch.tensor([0.3, 1.7, -0.9, 2.2])
    soft = model(features, tau=0.7)
    (soft * weights).sum().backward()
    soft_grads = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    hard = model(features, tau=0.7, hard=True)
    (hard * weights).sum().backward()
    hard_grads = [p.grad.clone() for p in model.parameters()]
    # The straight-through estimator passes the soft gradient exactly.
    for soft_grad, hard_grad in zip(soft_grads, hard_grads):
        assert torch.isfinite(hard_grad).all()
        assert torch.equal(soft_grad, hard_grad)
        assert float(hard_grad.abs().sum()) > 0.0


def test_load_balancing_loss_uniform_collapsed_and_optimization():
    uniform = torch.full((8, 3), 1.0 / 3.0, dtype=torch.float64)
    assert abs(float(load_balancing_loss(uniform))) < 1e-12
    collapsed = torch.eye(3, dtype=torch.float64)[[0] * 8]
    assert float(load_balancing_loss(collapsed)) > 1.9  # K - 1 = 2
    with pytest.raises(ValueError):
        load_balancing_loss(torch.zeros(3))  # not a (B, K) batch

    # Its gradient reduces collapse on a toy parameterization.
    logits = torch.tensor([3.0, -1.5, -1.5], dtype=torch.float64)
    logits.requires_grad_()
    optimizer = torch.optim.Adam([logits], lr=0.1)
    losses = []
    for _ in range(50):
        gates = torch.softmax(logits, dim=-1).expand(8, 3)
        loss = load_balancing_loss(gates)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * 1e-3
    assert losses[-1] < 1e-2


def tiny_dataset():
    return build_dataset((21, 22), (1, 2), num_decoys=NUM_DECOYS)


def test_training_determinism():
    dataset = tiny_dataset()
    model_a, history_a = train_router(dataset, dataset, epochs=50, seed=7)
    model_b, history_b = train_router(dataset, dataset, epochs=50, seed=7)
    for value_a, value_b in zip(
        model_a.state_dict().values(), model_b.state_dict().values()
    ):
        assert torch.equal(value_a, value_b)  # identical after 50 steps
    assert history_a["train_accuracy"] == history_b["train_accuracy"]
    assert history_a["tau"] == history_b["tau"]
    assert not torch.cuda.is_available()  # CPU-only


def test_overfit_sanity():
    dataset = build_dataset((31, 32, 33, 34), (2, 3), num_decoys=NUM_DECOYS)
    assert dataset[0].shape[0] == 8
    model, history = train_router(dataset, dataset, epochs=100, lr=3e-3, seed=3)
    assert history["train_accuracy"][-1] == 1.0
    assert hard_accuracy(model, *dataset[:2]) == 1.0


def test_smoke_generalization():
    # Train on one seed block, evaluate on a disjoint block; K = 3.
    train = build_dataset(range(500, 516), (1, 2, 3, 4), num_decoys=NUM_DECOYS)
    assert train[0].shape[0] == 64
    val = build_dataset(range(700, 712), (1, 2, 3, 4), num_decoys=NUM_DECOYS)
    model, history = train_router(train, val, epochs=150, lr=3e-3, seed=7)
    final = history["val_accuracy"][-1]
    baseline = argmin_residual_accuracy(val[0], val[1])
    assert final > 0.5  # significantly above chance 1/K
    assert final >= baseline - 0.05  # honest v0: roughly match the oracle


def test_no_nans_in_losses_or_gradients():
    dataset = tiny_dataset()
    model, history = train_router(dataset, dataset, epochs=5, seed=11)
    for key in ("loss", "cross_entropy", "aux_loss", "gate_entropy"):
        assert np.isfinite(np.asarray(history[key], dtype=float)).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_fail_closed_validation():
    with pytest.raises(ValueError):
        build_dataset([], GRID, num_decoys=NUM_DECOYS)  # no seeds
    with pytest.raises(ValueError):
        build_dataset(BUILD_SEEDS, (), num_decoys=NUM_DECOYS)  # no grid
    with pytest.raises(ValueError):
        build_dataset(BUILD_SEEDS, (-1,), num_decoys=NUM_DECOYS)  # bad budget
    with pytest.raises(ValueError):
        InstanceMetadata(0, 1, 1, (1, 1), 0)  # not a permutation
    with pytest.raises(ValueError):
        InstanceMetadata(0, 1, 1, (0, 1), 1)  # true_index must locate 0
    dataset = tiny_dataset()
    mismatched = build_dataset((21, 22), (1, 2), num_decoys=3)  # K = 4
    with pytest.raises(ValueError):
        train_router(dataset, mismatched, epochs=1)
    with pytest.raises(ValueError):
        train_router(dataset, dataset, epochs=0)
    model = StructureRouter()
    with pytest.raises(ValueError):
        model.logits(torch.zeros(2, K, FEATURE_DIM + 1))  # wrong F
    probes = Probes(np.zeros((1, 3)), np.zeros(1))
    instance = make_budget_instance(11, num_decoys=NUM_DECOYS)
    with pytest.raises(ValueError):
        candidate_features(probes, instance.true_target, np.zeros(3))
    predictions = hard_predictions(model, dataset[0])
    assert predictions.shape == (dataset[0].shape[0],)


def test_plain_import_of_universa_does_not_import_torch():
    # Run in a fresh interpreter: this test module itself imports torch.
    src = Path(universa.__file__).resolve().parent.parent
    code = (
        "import sys\n"
        "import universa\n"
        "bad = [m for m in sys.modules if m == 'torch' or m.startswith('torch.')]\n"
        "bad += ['universa.router'] if 'universa.router' in sys.modules else []\n"
        "sys.exit(1 if bad else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "PYTHONPATH": str(src)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
