# Soft Decentralized Expected SARSA

Companion code for the paper

**Soft Decentralized Expected SARSA: Equilibrium Selection via Gradual Adaptation**

This repository reproduces the numerical results and code-generated figures associated with that paper: Soft Decentralized Expected SARSA (SDES) simulations, closed-form basin-of-attraction (BoA) calculations, and the scripts that rebuild the paper figures from the shipped data.

## What is included

| Path | Role |
|------|------|
| `MultiAgent/MultiAgent_DQ.py` | SDES / soft inertial ε-BR dynamics (simulation engine) |
| `MultiAgent/basin.py` | Closed-form BoA quantities (B̂, Ĉ, D̂, aggregate bounds) and vector fields |
| `MultiAgent/sim_script_vectorized.py` | Monte Carlo driver that produced `results/*.csv` |
| `MultiAgent/validate_dhat_closed.py` | Numerical check of the closed-form D̂ / related hats |
| `scripts/export_paper_figures.py` | α-effect and related simulation figures from CSVs |
| `scripts/plot_phase_diagrams.py` | IBR phase diagrams (Python port of the critical curves) |
| `scripts/plot_vector_fields.py` | Vector-field figures `vf_*.png` |
| `results/` | Precomputed simulation CSVs used in the paper |
| `data/` | Numeric grids written by the phase-diagram script |
| `images/` | Figure outputs |
| `mathematica/` | Symbolic provenance for the IBR graphs and critical curves |
| `examples/` | Short usage demos |

Mathematica is **not** required to regenerate figures. The notebooks are kept as derivation provenance for Proposition IBR (γ_c1, γ_c2 and graphs G1–G3).

## Setup

Python ≥ 3.9 recommended.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All commands below are run from the **repository root**.

## Reproduce the paper figures

```bash
# Simulation-based figures (reads results/*.csv)  ~15 s
python scripts/export_paper_figures.py

# Analytic IBR phase diagrams  ~5 s
python scripts/plot_phase_diagrams.py

# Vector fields (one short trajectory per profile)  ~15 min
python scripts/plot_vector_fields.py
# optional subset:
python scripts/plot_vector_fields.py --pairs adgt,adad
```

Outputs are written under `images/`.

### Optional checks and full re-simulation

```bash
# Closed-form BoA validation
python MultiAgent/validate_dhat_closed.py

# Regenerate CSVs from scratch (hours of compute)
python MultiAgent/sim_script_vectorized.py
```

The CSVs already in `results/` are the data used for the published plots.

## Figure map (paper label → file → script)

| Paper figure | File(s) in `images/` | Script / source |
|--------------|----------------------|-----------------|
| `fig:alpha_effect_ADGT` | `af0702_adgt.png`, `af0805_adgt.png` | `scripts/export_paper_figures.py` |
| `fig:alpha_effect` | `af0702_uniform.png`, `af0805_uniform.png` | `scripts/export_paper_figures.py` |
| `fig:alpha_effect_ACGT` | `af0702_acgt.png`, `af_0805_acgt.png` | `scripts/export_paper_figures.py` |
| `fig:alpha_effect_ACGT_boundary` | `af0702_acgt_boundary.png`, `af0805_acgt_boundary.png` | `scripts/export_paper_figures.py` |
| `fig:alpha_effect_ACAD` | `af0702_acad.png`, `af0805_acad.png` | `scripts/export_paper_figures.py` |
| `fig:acad-entry` | `acad_entry_plane.png` | `scripts/export_paper_figures.py` |
| `fig:phasediagrams` | `TwoActionPDsimilarityphased7e2.png`, `TwoActionPDsimilarityphaser8p2-Gamma.png` | `scripts/plot_phase_diagrams.py` |
| `fig:IBRgraphs` | `IBRGraph1.pdf`–`IBRGraph3.pdf` | Mathematica notebook (provenance) |
| Vector-field figures | `vf_*.png` | `scripts/plot_vector_fields.py` |

Payoff configurations used in the α-effect panels:

- (R, P) = (0.7, 0.2) with T = 1, S = 0
- (R, P) = (0.8, 0.5) with T = 1, S = 0

## Examples

```bash
python examples/basic_multiagent_example.py
python examples/basin_of_attraction_example.py
python examples/strategy_comparison_example.py
```

## Citation

If you use this code, please cite the paper and this repository:

```bibtex
@misc{sdes_code,
  author       = {Martucci, Giuseppe and Meylahn, Janusz M.},
  title        = {Soft Decentralized Expected {SARSA}: companion code},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/AE452/Soft-Decentralized-Expected-SARSA}
}
```

## License

MIT License. See [LICENSE](LICENSE).
