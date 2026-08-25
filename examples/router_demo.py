"""End-to-end router demo: learned candidate selection on disjoint seeds.

Builds a certified routing dataset over seeds 1000..1031 (train) and a
disjoint block 2000..2015 (eval), trains the temperature-annealed
StructureRouter briefly, and reports the final hard-inference (strictly
discrete argmax) accuracy beside the non-learned baseline ``argmin over
candidates of the log identification residual``. In this clean regime the
baseline is essentially perfect; the v0 deliverable is the annealed-routing
machinery and the generalization plumbing, so the assertion is that the
learned router stays within 0.05 of the baseline — a tolerance for the
machinery, not a claim of beating the residual oracle.

Run from the repository root:  python examples/router_demo.py
"""

from universa.router import (
    anneal_temperature,
    argmin_residual_accuracy,
    build_dataset,
    train_router,
)

TRAIN_SEEDS = range(1000, 1032)
EVAL_SEEDS = range(2000, 2016)
NUM_PROBES_GRID = (1, 2, 3, 4)
NUM_DECOYS = 3  # K = 4 candidates
EPOCHS = 60
LR = 3e-3
SEED = 0


def main() -> None:
    train = build_dataset(TRAIN_SEEDS, NUM_PROBES_GRID, num_decoys=NUM_DECOYS)
    evaluation = build_dataset(EVAL_SEEDS, NUM_PROBES_GRID, num_decoys=NUM_DECOYS)
    k = train[0].shape[1]
    print(
        f"dataset: {train[0].shape[0]} train / {evaluation[0].shape[0]} eval "
        f"instances, K={k} candidates, F={train[0].shape[2]} features"
    )

    model, history = train_router(
        train, evaluation, epochs=EPOCHS, lr=LR, seed=SEED
    )

    checkpoints = [0, EPOCHS // 4, EPOCHS // 2, 3 * EPOCHS // 4, EPOCHS - 1]
    print("tau schedule (exponential, soft early -> near-hard late):")
    for epoch in checkpoints:
        expected = anneal_temperature(epoch, EPOCHS)
        assert history["tau"][epoch] == expected
        print(f"  epoch {epoch:>3}: tau = {expected:.4f}")

    train_accuracy = history["train_accuracy"][-1]
    eval_accuracy = history["val_accuracy"][-1]
    baseline = argmin_residual_accuracy(evaluation[0], evaluation[1])
    usage = history["usage"][-1]
    print(f"final train hard accuracy: {train_accuracy:.4f}")
    print(f"final eval  hard accuracy: {eval_accuracy:.4f} (disjoint seed block)")
    print(f"argmin-residual baseline eval accuracy: {baseline:.4f} (no learning)")
    print(f"final gate entropy (nats): {history['gate_entropy'][-1]:.4f}")
    print(f"usage histogram over {k} candidates (collapse check): {usage.tolist()}")

    # Honest v0 check: the learned router roughly matches the non-learned
    # residual oracle; the tolerance covers the machinery, nothing more.
    assert eval_accuracy >= baseline - 0.05
    print(f"eval accuracy within 0.05 of the baseline: OK")


if __name__ == "__main__":
    main()
