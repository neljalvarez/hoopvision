"""Random-shot sweep in the synthetic room (not a pytest test; takes ~10 s per shot).

    python tests/random_sweep.py --seed 3 --n 36

Shots are aimed at random offsets around the rim with random arcs; the physics simulation
decides the true result, the full pipeline (rendered video -> detector -> tracker -> engine)
decides the prediction. Prints one JSON line per shot and a summary.
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import harness  # noqa: E402
import scenarios as S  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n", type=int, default=36)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    summary: Counter = Counter()
    for i in range(a.n):
        off = (float(rng.uniform(-0.22, 0.22)), float(rng.uniform(-0.05, 0.3)), float(rng.uniform(-0.2, 0.2)))
        arc = float(rng.uniform(0.62, 0.95))
        sim = S.shot(off, arc, seed=i)
        _, ev, _ = harness.run([sim], seed=i + 10)
        truth = "make" if sim.made else "miss"
        out = harness.outcomes(ev)
        got = out[0] if len(out) == 1 else ("not detected" if not out else "duplicate:" + ",".join(out))
        summary[(truth, got)] += 1
        print(json.dumps({"i": i, "aim_offset_m": off, "arc": arc, "truth": truth, "rim_hits": sim.rim_hits, "predicted": out}), flush=True)
    print("summary (truth, predicted):", dict(summary))


if __name__ == "__main__":
    main()
