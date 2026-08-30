"""Router v2 demo: learned routing vs the polluted oracle, NO exact anchor.

The no-anchor regime: every candidate boundary is observed through an
ObservationModel with BOTH mask_fraction = 0.25 (edge columns removed) AND
sign-corruption swept over a profile grid that excludes 0.0 (0.2..0.9 by
0.1). No feature is computed against a clean operator — the learned
router's profile has no exact fraction-0 anchor this time (the asymmetry
the sealed v1 win was declared under). Training runs on seeds
50001..50032 over operating fractions {0.2, 0.3, 0.4, 0.5}; evaluation on
the DISJOINT seeds 50101..50116 over the HELD-OUT fractions {0.6, 0.7,
0.8} — generalization to unseen degradation with partially masked edges,
reported per fraction, truthfully either way.

Baselines per fraction: the polluted oracle (argmin of the single
operating-fraction observed-misfit column, polluted at every fraction
here) and, descriptively, the grid-mean profile heuristic (argmin of the
uniform average of the log1p profile, no learning).

Run from the repository root:  python examples/router_v2_demo.py
"""

from universa.router_v2 import evaluate_no_anchor

TRAIN_SEEDS = range(50001, 50033)
EVAL_SEEDS = range(50101, 50117)
TRAIN_FRACTIONS = (0.2, 0.3, 0.4, 0.5)
EVAL_FRACTIONS = (0.6, 0.7, 0.8)
NUM_DECOYS = 3  # K = 4 candidates
EPOCHS = 150
LR = 3e-3
SEED = 0


def main() -> None:
    model, report = evaluate_no_anchor(
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
        f"K={k} candidates, F={len(report.profile_grid) * 2 + 2} features, "
        f"mask_fraction={report.mask_fraction} (no clean anchor)"
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
    print("held-out degradation, no anchor — learned router vs baselines:")
    print("  (oracle = argmin of the observed-misfit column at that")
    print("   fraction, polluted at EVERY fraction here; heuristic = argmin")
    print("   of the grid-mean profile, descriptive; learned = hard")
    print("   inference accuracy)")
    print(
        f"  {'fraction':>8} | {'n':>3} | {'learned':>7} | {'oracle':>7} | "
        f"{'heuristic':>9} | {'l-oracle':>8}"
    )
    print(
        f"  {'-' * 8}-+-{'-' * 3}-+-{'-' * 7}-+-{'-' * 7}-+-"
        f"{'-' * 9}-+-{'-' * 8}"
    )
    for row in report.per_fraction:
        delta = row.learned_accuracy - row.oracle_accuracy
        print(
            f"  {row.fraction:>8.1f} | {row.num_instances:>3} | "
            f"{row.learned_accuracy:>7.4f} | {row.oracle_accuracy:>7.4f} | "
            f"{row.heuristic_accuracy:>9.4f} | {delta:>+8.4f}"
        )
    print()
    print(
        "honest read: v1's learned router read a profile ANCHORED at the"
        " exact fraction-0 residual; here no clean column exists anywhere"
        " — every feature is masked+corrupted. The oracle is myopic (one"
        " polluted column); the learned router integrates the degraded"
        " trajectory. The table above is the answer, at every held-out"
        " fraction, whichever way it goes."
    )
    for row in report.per_fraction:
        assert 0.0 <= row.learned_accuracy <= 1.0
        assert 0.0 <= row.oracle_accuracy <= 1.0
        assert 0.0 <= row.heuristic_accuracy <= 1.0


if __name__ == "__main__":
    main()
