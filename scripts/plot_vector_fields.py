#!/usr/bin/env python3
"""Reproduce the vector-field figures vf_*.png used in paper.tex
(Figures `fig:equilibria`, `fig:adgt`, `fig:vectorfields1eq`, `fig:vectorfields2`,
`fig:twosteps`, `fig:vectorfields4`).

Clean-script port of MultiAgent/basin_analysis.ipynb (vector-field cell): for each
fixed strategy profile (s1, s2) it runs one Dec-PBQ trajectory, trims it at the
first strategy-boundary crossing, and overlays it on the analytic vector field
of eq:difference_update.

Outputs images/vf_{s1}{s2}.png for all 16 profiles (paper uses 12 of them).

Note: trajectories start from random Q inits, so the exact sample path differs
from the published figure unless the same seed is used; the vector field, basins
and qualitative behaviour are identical. Default seed fixed for reproducibility.

Usage:  python scripts/plot_vector_fields.py [--pairs adgt,acgt] [--seed 7]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "MultiAgent"))

from MultiAgent.MultiAgent_DQ import (  # noqa: E402
    Algorithm,
    QvaluesADvsAD,
    QvaluesGTvsGT,
    QvaluesADvsAGT,
    QvaluesADvsAC,
)
from MultiAgent.basin import VectorFieldAutomatic  # noqa: E402
from utils import convert_dict_to_mean_df  # noqa: E402

IMAGES_DIR = os.path.join(_ROOT, "images")

# Same parameters as basin_analysis.ipynb
T, R, P, S = 1, 0.7, 0.2, 0
EPSILON, D, ALPHA = 0.1, 0.6, 0.3

ALL_PAIRS = [
    ("AD", "AD"), ("AD", "GT"), ("AD", "AGT"), ("AD", "AC"),
    ("GT", "AD"), ("GT", "GT"), ("GT", "AGT"), ("GT", "AC"),
    ("AGT", "AD"), ("AGT", "GT"), ("AGT", "AGT"), ("AGT", "AC"),
    ("AC", "AD"), ("AC", "GT"), ("AC", "AGT"), ("AC", "AC"),
]

FUNC_MAP = {
    "AD": QvaluesADvsAD,
    "GT": QvaluesGTvsGT,
    "AGT": QvaluesADvsAGT,
    "AC": QvaluesADvsAC,
}


def trim_q_by_strategy(q1, q2, s1, s2):
    """Cut both Q trajectories at the first step where either player's
    action ordering leaves its initial strategy region."""
    q1_arr = q1.values if hasattr(q1, "values") else np.asarray(q1)
    q2_arr = q2.values if hasattr(q2, "values") else np.asarray(q2)

    def ordering_sign(Q, strategy):
        delta = Q[1] - Q[0]
        if strategy in ("GT", "AC"):
            delta = -delta
        return np.sign(delta)

    def valid_length(Q, strategy):
        ref_sign = ordering_sign(Q[:2], strategy)
        last_valid = 1
        for k in range(1, Q.shape[0]):
            if np.all(ordering_sign(Q[k - 1:k + 1], strategy) == ref_sign):
                last_valid = k
            else:
                break
        return last_valid + 1

    cut = min(valid_length(q1_arr, s1), valid_length(q2_arr, s2))
    if hasattr(q1, "iloc"):
        return q1.iloc[:cut], q2.iloc[:cut]
    return q1_arr[:cut], q2_arr[:cut]


def make_vf_figure(s1, s2, seed, out_dir):
    np.random.seed(seed)
    algo = Algorithm(
        e=EPSILON, lambda_val=0, alpha=ALPHA, delta=D,
        init_mode="fixed", tolerance=1e-10,
        M=5, strategy1=s1, strategy2=s2,
        t=T, r=R, p=P, s=S, verbose=False,
    )
    q1_hist, q2_hist, _, _, _ = algo.run()
    exp = np.random.randint(0, 5)
    q1 = convert_dict_to_mean_df(q1_hist, mean=False)[exp]
    q2 = convert_dict_to_mean_df(q2_hist, mean=False)[exp]
    q1, q2 = trim_q_by_strategy(q1, q2, s1, s2)

    plotter = VectorFieldAutomatic(
        q1=q1, q2=q2,
        t=T, r=R, p=P, s=S, epsilon=EPSILON, d=D,
        func1=FUNC_MAP[s2], func2=FUNC_MAP[s1],
        strategy1=s1, strategy2=s2,
        alpha=ALPHA, density=15,
        show_trajectory=True, show_basin=True,
    )
    plotter.plot(show_all_quadrants=False)

    path = os.path.join(out_dir, f"vf_{s1.lower()}{s2.lower()}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close("all")
    print("saved", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="all",
                        help="comma-separated profile keys, e.g. 'adgt,acgt' (default: all 16)")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    os.makedirs(IMAGES_DIR, exist_ok=True)
    if args.pairs == "all":
        pairs = ALL_PAIRS
    else:
        wanted = {k.strip().lower() for k in args.pairs.split(",")}
        pairs = [(a, b) for a, b in ALL_PAIRS if f"{a.lower()}{b.lower()}" in wanted]

    for s1, s2 in pairs:
        print(f"--- ({s1}, {s2})")
        make_vf_figure(s1, s2, args.seed, IMAGES_DIR)


if __name__ == "__main__":
    main()
