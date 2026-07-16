# Reproducing the figures in the paper

Everything a reviewer needs is Python. Mathematica is **not** required for any
figure: the two notebook-derived phase diagrams are reproduced by a Python port
of the closed-form critical curves (Proposition `thm:IBR`), and the notebook is
kept only as derivation provenance.

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python ≥ 3.9. All commands below are run from the repository root.

## 2. Regenerate all figures

The paper reads every figure from `images/`. Three scripts regenerate all
code-generated figures:

```bash
# Simulation-result figures (af*.png, acad_entry_plane.png)
# — plots precomputed CSVs from results/, takes ~15 s
python scripts/export_paper_figures.py

# Analytic phase diagrams (fig:phasediagrams) — takes ~5 s
python scripts/plot_phase_diagrams.py

# Vector fields (vf_*.png) — runs one short simulation per profile, ~15 min total
python scripts/plot_vector_fields.py            # all 16 profiles
python scripts/plot_vector_fields.py --pairs adgt,adad   # or a subset
```

See `FIGURE_MANIFEST.md` for the figure-by-figure mapping (paper label → file →
script) and for the list of hand-drawn diagrams that are not code-generated.

## 3. Regenerate the simulation data (optional, slow)

`scripts/export_paper_figures.py` plots precomputed CSVs in `results/`. To
regenerate those CSVs from scratch, run the vectorized simulation driver:

```bash
python MultiAgent/sim_script_vectorized.py
```

(Grid of α, λ, γ, ε per payoff configuration; hours of compute. The shipped
CSVs are the exact data used in the paper.)

The closed-form bounds themselves (B-hat, C-hat, D-hat, Proposition 17
aggregate LB/UB) are implemented in `MultiAgent/basin.py` and validated
numerically by:

```bash
python MultiAgent/validate_dhat_closed.py
```

## 4. Interactive exploration (optional)

`MultiAgent/MultiAgent_Analysis.ipynb` — main analysis notebook (bound
validation tables, all paper plots inline).
`MultiAgent/basin_analysis.ipynb` — vector-field exploration.

## 5. Mathematica provenance (optional, not needed for figures)

`mathematica/Similarity_Bellman_eq_responses_GT_Space_Expected_SARSA.nb`
contains the symbolic derivation behind Proposition `thm:IBR`: it sets up the
Bellman equations for all ε-greedy best responses, enumerates the feasible IBR
graphs (G1–G3), and derives the critical curves

```
gamma_c1 = (eps(T+S-P-R) - 2(T-R)) / ((1-eps)(eps(T+S-2P) - 2(T-P)))
gamma_c2 = (eps(T+S-P-R) + 2(P-S)) / ((1-eps)(eps(T+S-2P) + 2(P-S)))
```

used by `scripts/plot_phase_diagrams.py`. The original notebook RegionPlots
(`TwoActionPDsimilarityphased7e2`, `TwoActionPDsimilarityphaser8p2`) produce the
same diagrams natively.

To re-run the notebook without a Mathematica license, use the free
[Wolfram Engine for Developers](https://www.wolfram.com/engine/) (free for
non-commercial use) together with `wolframscript`:

```bash
wolframscript -file mathematica/Similarity_Bellman_eq_responses_GT_Space_Expected_SARSA.nb
```

Other notebooks in `mathematica/` (`Proof_Steps.nb`, `Q-values.nb`, …) are
supporting symbolic calculations for the proofs; none of them generate paper
figures.

## Known gaps (disclosed)

Six conceptual figures are hand-drawn and their source files are not in the
repository (see `FIGURE_MANIFEST.md`, "ORPHANED" rows): the state-space
illustration, the within/across-batch timelines, the asynchronous best-response
graph, and the best-response tree diagrams. 
