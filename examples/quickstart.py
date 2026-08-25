"""Quickstart: the misfit signal picks the true structure out of a library.

Builds one switch instance — a random source graph, a planted quotient
chain map, and decoy targets — then scores every candidate structure by the
chain-map commutation residual. The true target scores exactly zero; decoys
do not. This is the structure-level version of the routing decision the
learned architecture will eventually make from data alone.

Run from the repository root:  python examples/quickstart.py
"""

from universa import make_switch_instance


def main() -> None:
    instance = make_switch_instance(seed=20260825, num_decoys=4)
    scores = instance.commutation_scores()
    print("candidate commutation residuals (||B1' f1 - f0 B1||_F):")
    for index, score in enumerate(scores):
        label = "true target" if index == 0 else f"decoy {index}"
        print(f"  [{index}] {label:<12} {score:.6e}")
    best = min(range(len(scores)), key=scores.__getitem__)
    assert best == 0 and scores[0] == 0.0
    assert all(score > 1e-6 for score in scores[1:])
    print("misfit signal identifies the true structure: OK")


if __name__ == "__main__":
    main()
