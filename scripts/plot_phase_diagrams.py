#!/usr/bin/env python3
"""Reproduce Figure `fig:phasediagrams` of paper.tex (IBR-graph phase diagrams).

Python port of the two Mathematica RegionPlots in
mathematica/Similarity_Bellman_eq_responses_GT_Space_Expected_SARSA.nb
(cells `TwoActionPDsimilarityphased7e2` and `TwoActionPDsimilarityphaser8p2`).

The regions follow the closed-form critical curves of Proposition `thm:IBR`
(derived symbolically in the notebook):

    gamma_c1 = (eps(T+S-P-R) - 2(T-R)) / ((1-eps)(eps(T+S-2P) - 2(T-P)))
    gamma_c2 = (eps(T+S-P-R) + 2(P-S)) / ((1-eps)(eps(T+S-2P) + 2(P-S)))

    gamma < gamma_c1              -> G1 (unique NE at (AD,AD))
    gamma_c1 < gamma < gamma_c2   -> G2 ((AD,AD) and (GT,GT) both NE)
    gamma > gamma_c2              -> G3 (unique NE at (AD,AD))

Outputs (300 dpi PNG, matching the filenames referenced in paper.tex):
    images/TwoActionPDsimilarityphased7e2.png        (R-P slice, gamma=0.7, eps=0.2)
    images/TwoActionPDsimilarityphaser8p2-Gamma.png  (eps-gamma slice, R=0.8, P=0.2)

Also dumps the exact numeric grids used by the plots to:
    data/phase_diagram_RP.csv
    data/phase_diagram_eps_gamma.csv

Usage:  python scripts/plot_phase_diagrams.py
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGES_DIR = os.path.join(_ROOT, "images")
DATA_DIR = os.path.join(_ROOT, "data")

# Mathematica ColorData[97] indices 1-3 (default plot theme), as in the notebook.
COLOR_G1 = (0.368417, 0.506779, 0.709798)  # blue
COLOR_G2 = (0.880722, 0.611041, 0.142051)  # orange
COLOR_G3 = (0.560181, 0.691569, 0.194885)  # green


def gamma_c1(eps, T, R, P, S):
    num = eps * (T + S - P - R) - 2.0 * (T - R)
    den = (1.0 - eps) * (eps * (T + S - 2.0 * P) - 2.0 * (T - P))
    return num / den


def gamma_c2(eps, T, R, P, S):
    num = eps * (T + S - P - R) + 2.0 * (P - S)
    den = (1.0 - eps) * (eps * (T + S - 2.0 * P) + 2.0 * (P - S))
    return num / den


def classify(gamma, eps, T, R, P, S):
    """Return region label 1/2/3 per Proposition thm:IBR, 0 where the game
    is not a prisoner's dilemma (ordering T > R > P > S violated)."""
    gamma = np.asarray(gamma, dtype=float)
    pd_valid = (T > R) & (R > P) & (P > S)
    with np.errstate(divide="ignore", invalid="ignore"):
        c1 = gamma_c1(eps, T, R, P, S)
        c2 = gamma_c2(eps, T, R, P, S)
    region = np.where(gamma < c1, 1, np.where(gamma < c2, 2, 3))
    return np.where(pd_valid, region, 0)


def _plot_region_grid(X, Y, region, xlabel, ylabel, out_png):
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    cmap = ListedColormap([(1, 1, 1, 0), COLOR_G1, COLOR_G2, COLOR_G3])
    ax.pcolormesh(X, Y, region, cmap=cmap, vmin=-0.5, vmax=3.5, shading="auto")
    # black region boundaries, as BoundaryStyle -> Black in the notebook
    ax.contour(X, Y, region, levels=[0.5, 1.5, 2.5], colors="black", linewidths=1.2)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    handles = [
        Patch(facecolor=COLOR_G1, edgecolor="black", label=r"$G_{1}$"),
        Patch(facecolor=COLOR_G2, edgecolor="black", label=r"$G_{2}$"),
        Patch(facecolor=COLOR_G3, edgecolor="black", label=r"$G_{3}$"),
    ]
    ax.legend(handles=handles, loc="upper left", framealpha=0.9)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_png)


def main(n_grid=801):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    T, S = 1.0, 0.0

    # --- Left panel: R-P slice at gamma=0.7, eps=0.2 -------------------------
    gamma0, eps0 = 0.7, 0.2
    r_vals = np.linspace(1e-4, 1 - 1e-4, n_grid)
    p_vals = np.linspace(1e-4, 1 - 1e-4, n_grid)
    Rg, Pg = np.meshgrid(r_vals, p_vals)
    region_rp = classify(gamma0, eps0, T, Rg, Pg, S)
    _plot_region_grid(
        Rg, Pg, region_rp, r"$R$", r"$P$",
        os.path.join(IMAGES_DIR, "TwoActionPDsimilarityphased7e2.png"),
    )
    pd.DataFrame({"R": Rg.ravel(), "P": Pg.ravel(), "region": region_rp.ravel()}).to_csv(
        os.path.join(DATA_DIR, "phase_diagram_RP.csv"), index=False
    )

    # --- Right panel: eps-gamma slice at R=0.8, P=0.2 ------------------------
    R0, P0 = 0.8, 0.2
    e_vals = np.linspace(1e-4, 1 - 1e-4, n_grid)
    g_vals = np.linspace(1e-4, 1 - 1e-4, n_grid)
    Eg, Gg = np.meshgrid(e_vals, g_vals)
    region_eg = classify(Gg, Eg, T, R0, P0, S)
    _plot_region_grid(
        Eg, Gg, region_eg, r"$\epsilon$", r"$\gamma$",
        os.path.join(IMAGES_DIR, "TwoActionPDsimilarityphaser8p2-Gamma.png"),
    )
    pd.DataFrame({"epsilon": Eg.ravel(), "gamma": Gg.ravel(), "region": region_eg.ravel()}).to_csv(
        os.path.join(DATA_DIR, "phase_diagram_eps_gamma.csv"), index=False
    )


if __name__ == "__main__":
    main()
