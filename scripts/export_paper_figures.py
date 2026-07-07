#!/usr/bin/env python3
"""Regenerate every simulation-based figure of paper.tex from the CSVs in results/.

Script version of the "Export appendix figures" cell of
MultiAgent/MultiAgent_Analysis.ipynb — no notebook needed.

Figures produced (into images/):
    af0702_adgt.png, af0805_adgt.png            fig:alpha_effect_ADGT   (B-hat)
    af0702_uniform.png, af0805_uniform.png      fig:alpha_effect        (Prop.17 LB/UB)
    af0702_acgt.png, af_0805_acgt.png           fig:alpha_effect_ACGT   (C-hat)
    af0702_acgt_boundary.png,
    af0805_acgt_boundary.png                    fig:alpha_effect_ACGT_boundary
    af0702_acad.png, af0805_acad.png            fig:alpha_effect_ACAD   (D-hat)
    acad_entry_plane.png                        fig:acad-entry

Inputs (results/):
    2000_0702.csv, 2000_0805.csv                       uniform init
    results_vect_paramsfixed_time2000_ADGT.csv         (AD,GT) init  [0805 rows first]
    results_vect_paramsfixed_time2000_ACGT.csv,        (AC,GT) init
    results_vect_paramsfixed_time2000_ACGT0702.csv
    ACGT_Chat_validate_0702.csv, ACGT_Chat_validate_0805.csv
    results_vect_paramsfixed_time2000_ACAD0702.csv,
    results_vect_paramsfixed_time2000_ACAD0805.csv

Usage:  python scripts/export_paper_figures.py
"""

import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "MultiAgent"))

from MultiAgent.basin import (  # noqa: E402
    calculate_B_hat,
    calculate_C_hat,
    calculate_D_hat,
    get_aggregate_bounds_prop17,
    plot_acad_entry_validation,
    sample_acad_entry_points,
)
from utils import plot_multiple_panels  # noqa: E402

IMAGES_DIR = os.path.join(_ROOT, "images")
RESULTS_DIR = os.path.join(_ROOT, "results")

PARAM_SETS = [
    {"delta": 0.5, "e": 0.05},
    {"delta": 0.5, "e": 0.1},
    {"delta": 0.75, "e": 0.05},
    {"delta": 0.75, "e": 0.1},
    {"delta": 0.95, "e": 0.05},
    {"delta": 0.95, "e": 0.1},
]

PLOT_KW = dict(
    param_sets=PARAM_SETS,
    x_var="alpha",
    varying_var="lambda",
    m=2000,
    use_same_scale=True,
    show=False,
)

PAYOFFS = {
    "0702": dict(t=1, r=0.7, p=0.2, s=0, cmap="viridis"),
    "0805": dict(t=1, r=0.8, p=0.5, s=0, cmap="cividis"),
}


def _csv(name):
    return pd.read_csv(os.path.join(RESULTS_DIR, name))


def _hats(fn, t, r, p, s):
    return [fn(t=t, r=r, p=p, s=s, epsilon=ps["e"], d=ps["delta"]) for ps in PARAM_SETS]


def _save(df, filename, **panel_kw):
    path = os.path.join(IMAGES_DIR, filename)
    plot_multiple_panels(df, save_path=path, **panel_kw, **PLOT_KW)
    print("saved", path)


def save_adgt(df, t, r, p, s, cmap, filename):
    b_hat = _hats(calculate_B_hat, t, r, p, s)
    _save(df, filename, y_var="P_GT", cmap=cmap,
          theoretical_basin_AD=b_hat, theoretical_line_label=r"$\hat{B}$",
          y_axis_label=r"$\mathrm{BoA}(\boldsymbol{\pi}^{GT}\mid(AD,GT))$")


def save_uniform(df, t, r, p, s, cmap, filename):
    bounds = [get_aggregate_bounds_prop17(t=t, r=r, p=p, s=s, epsilon=ps["e"], d=ps["delta"])
              for ps in PARAM_SETS]
    _save(df, filename, y_var="P_GT", cmap=cmap,
          theoretical_bounds=bounds, theoretical_bounds_labels=("LB", "UB"),
          y_axis_label=r"$\mathrm{BoA}(\boldsymbol{\pi}^{GT})$")


def save_acgt(df, t, r, p, s, cmap, filename):
    c_hat = _hats(calculate_C_hat, t, r, p, s)
    _save(df, filename, y_var="P_GT", cmap=cmap,
          theoretical_bounds=[(c, 1.0) for c in c_hat],
          theoretical_bounds_labels=(r"$\hat{C}$", "UB"),
          y_axis_label=r"$\mathrm{BoA}(\boldsymbol{\pi}^{GT}\mid(AC,GT))$")


def save_acgt_boundary(df, t, r, p, s, cmap, filename):
    c_hat = _hats(calculate_C_hat, t, r, p, s)
    _save(df, filename, y_var="P_direct_GT", cmap=cmap,
          theoretical_basin_AD=c_hat, theoretical_line_label=r"$\hat{C}$",
          y_axis_label=r"$P(\mathrm{A1D\ first})$")


def save_acad(df, t, r, p, s, cmap, filename):
    d_hat = _hats(calculate_D_hat, t, r, p, s)
    _save(df, filename, y_var="P_GT", cmap=cmap,
          theoretical_basin_AD=d_hat, theoretical_line_label=r"$\hat{D}$",
          y_axis_label=r"$\mathrm{BoA}(\boldsymbol{\pi}^{GT}\mid(AC,AD))$")


def save_entry_plane():
    xh, yh = sample_acad_entry_points(t=1, r=0.7, p=0.2, s=0, epsilon=0.01, d=0.6,
                                      n=300, seed=7)
    fig, _ = plot_acad_entry_validation(
        t=1, r=0.7, p=0.2, s=0, epsilon=0.01, d=0.6,
        entry_points=(xh, yh),
        title=r"$(AC,AD)\to(GT,AD)$ entry plane",
    )
    path = os.path.join(IMAGES_DIR, "acad_entry_plane.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    # (AD,GT): B-hat --------------------------------------------------------
    df_adgt = _csv("results_vect_paramsfixed_time2000_ADGT.csv")
    save_adgt(df_adgt[225:], filename="af0702_adgt.png", **PAYOFFS["0702"])
    save_adgt(df_adgt[:225], filename="af0805_adgt.png", **PAYOFFS["0805"])

    # Uniform init: Prop. 17 aggregate bounds -------------------------------
    save_uniform(_csv("2000_0702.csv"), filename="af0702_uniform.png", **PAYOFFS["0702"])
    save_uniform(_csv("2000_0805.csv"), filename="af0805_uniform.png", **PAYOFFS["0805"])

    # (AC,GT): C-hat ---------------------------------------------------------
    save_acgt(_csv("results_vect_paramsfixed_time2000_ACGT0702.csv"),
              filename="af0702_acgt.png", **PAYOFFS["0702"])
    save_acgt(_csv("results_vect_paramsfixed_time2000_ACGT.csv"),
              filename="af_0805_acgt.png", **PAYOFFS["0805"])

    # (AC,GT) first-crossing validation --------------------------------------
    save_acgt_boundary(_csv("ACGT_Chat_validate_0702.csv"),
                       filename="af0702_acgt_boundary.png", **PAYOFFS["0702"])
    save_acgt_boundary(_csv("ACGT_Chat_validate_0805.csv"),
                       filename="af0805_acgt_boundary.png", **PAYOFFS["0805"])

    # (AC,AD): D-hat ----------------------------------------------------------
    save_acad(_csv("results_vect_paramsfixed_time2000_ACAD0702.csv"),
              filename="af0702_acad.png", **PAYOFFS["0702"])
    save_acad(_csv("results_vect_paramsfixed_time2000_ACAD0805.csv"),
              filename="af0805_acad.png", **PAYOFFS["0805"])

    # (AC,AD) -> (GT,AD) entry plane ------------------------------------------
    save_entry_plane()


if __name__ == "__main__":
    main()
