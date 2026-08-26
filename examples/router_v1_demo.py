"""Router v1 demo: learned routing vs the polluted oracle, held-out degradation.

The degraded regime: candidates are observed through ObservationModel
sign-corruption draws, so the exact-residual oracle of v0 is polluted. The
learned router reads each candidate's full degradation profile (observed
commutation misfit over fractions 0.0..0.7, slopes, structural dims); the
non-learned oracle reads only the single profile column at the operating
fraction. Training runs on seeds 3000..3031 over fractions 0.0..0.4;
evaluation on the DISJOINT seeds 4000..4015 over the HELD-OUT fractions
0.5, 0.6, 0.7 — generalization to unseen degradation, reported per
fraction, truthfully either way.

Run from the repository root:  python examples/router_v1_demo.py
"""

from universa.router import evaluate_held_out_regime

TRAIN_SEEDS = range(3000, 3032)
EVAL_SEEDS = range(4000, 4016)
TRAIN_FRACTIONS = (0.0, 0.1, 0.2, 0.3, 0.4)
EVAL_FRACTIONS = (0.5, 0.6, 0.7)
NUM_DECOYS = 3  # K = 4 candidates
EPOCHS = 150
LR = 3e-3
SEED = 0


def main() -> None:
    model, report = evaluate_held_out_regime(
        TRAIN_SEEDS,
        EVAL_SEEDS,
        TRAIN_FRACTIONS,
        EVAL_FRACTIONS,
        epochs=EPOCHS,
        lr=LR,
        seed=SEED,
        num_decoys=NUM_DECOYS,
    )
    del model  # the report carries everything this demo prints

    num_train = len(report.train_seeds) * len(report.train_fractions)
    num_eval = len(report.eval_seeds) * len(report.eval_fractions)
    k = 1 + NUM_DECOYS
    print(
        f"dataset: {num_train} train (seeds {report.train_seeds[0]}.."
        f"{report.train_seeds[-1]}, fractions {report.train_fractions}) / "
        f"{num_eval} eval (seeds {report.eval_seeds[0]}.."
        f"{report.eval_seeds[-1]}, fractions {report.eval_fractions}), "
        f"K={k} candidates, F={len(report.profile_grid) * 2 + 2} features"
    )
    print(
        f"final train hard accuracy: {report.final_train_accuracy:.4f}   "
        f"(chance = {1.0 / k:.4f})"
    )
    print(
        f"final eval  hard accuracy: {report.final_eval_accuracy:.4f}   "
        "(pooled over held-out fractions)"
    )
    print()
    print("held-out degradation — learned router vs polluted oracle:")
    print("  (oracle = argmin of the observed-misfit column at that")
    print("   fraction, no learning; learned = hard-inference accuracy)")
    print(f"  {'fraction':>8} | {'n':>3} | {'learned':>7} | {'oracle':>7} | {'delta':>7}")
    print(f"  {'-' * 8}-+-{'-' * 3}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 7}")
    for row in report.per_fraction:
        delta = row.learned_accuracy - row.oracle_accuracy
        print(
            f"  {row.fraction:>8.1f} | {row.num_instances:>3} | "
            f"{row.learned_accuracy:>7.4f} | {row.oracle_accuracy:>7.4f} | "
            f"{delta:>+7.4f}"
        )
    print()
    print(
        "honest read: the oracle is myopic — it reads one polluted column; "
        "the learned router integrates the whole degradation trajectory, "
        "anchored at the exact fraction-0 residual. The table above is the "
        "answer, at every held-out fraction, whichever way it goes."
    )
    for row in report.per_fraction:
        assert 0.0 <= row.learned_accuracy <= 1.0
        assert 0.0 <= row.oracle_accuracy <= 1.0


if __name__ == "__main__":
    main()
