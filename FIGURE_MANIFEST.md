# Figure Manifest for `paper.tex`

Every `\includegraphics` in `paper.tex`, its generating source, and its reproducibility
status. All paths are relative to the repository root; the paper expects all
figure files in `images/`.

Regeneration commands (see `REPRODUCING.md` for full instructions):

| Command | Regenerates |
|---|---|
| `python scripts/export_paper_figures.py` | all `af*.png`, `acad_entry_plane.png` (from `results/*.csv`) |
| `python scripts/plot_vector_fields.py` | all `vf_*.png` |
| `python scripts/plot_phase_diagrams.py` | both phase diagrams (`TwoActionPD*.png`) |

## Main text

| Label | File | Caption (short) | Section | Source | Status |
|---|---|---|---|---|---|
| `fig:state_abstraction` | `images/state_space_abstraction.pdf` | State spaces S and S-hat | Illustrative Environment | manual (drawn illustration) | **ORPHANED — file missing from repo; recover from Overleaf/authors** |
| `fig:within_across_batches` | `images/within_each_batch.pdf`, `images/across_batches.pdf` | Within/Across batches illustration | Convergence | manual (drawn illustration) | **ORPHANED — files missing from repo; recover from Overleaf/authors** |
| `fig:IBRgraphs` | `images/IBRGraph1.pdf` – `IBRGraph3.pdf` | Possible IBR graphs G1–G3 | Equilibrium Structure | Mathematica: `mathematica/Similarity_Bellman_eq_responses_GT_Space_Expected_SARSA.nb` (GraphPlot cells) | OK — copies of root `Graph1-3.pdf` placed in `images/` |
| `fig:phasediagrams` | `images/TwoActionPDsimilarityphased7e2.png`, `images/TwoActionPDsimilarityphaser8p2-Gamma.png` | Phase diagrams (R–P and ε–γ slices) | Equilibrium Structure | Python: `scripts/plot_phase_diagrams.py` (port of the Mathematica RegionPlots; derivation of the critical curves stays in the `.nb`) | OK — reproducible without Mathematica |
| `fig:equilibria` | `images/vf_adad.png` | Vector fields, (AD,AD) | Influence of hyperparameters | Python: `scripts/plot_vector_fields.py` | OK |
| `fig:adgt` | `images/vf_adgt.png` | Vector fields, (AD,GT) | Influence of hyperparameters | Python: `scripts/plot_vector_fields.py` | OK |
| `fig:alpha_effect_ADGT` | `images/af0702_adgt.png`, `images/af0805_adgt.png` | BoA vs α from (AD,GT), B-hat line | Simulation Results | Python: `scripts/export_paper_figures.py` | OK |
| `fig:alpha_effect` | `images/af0702_uniform.png`, `images/af0805_uniform.png` | BoA vs α, uniform init, Prop. 17 LB/UB | Simulation Results | Python: `scripts/export_paper_figures.py` | OK |

## Appendix

| Label | File | Caption (short) | Section | Source | Status |
|---|---|---|---|---|---|
| `fig:asynchronousBR` | `images/largegraph.pdf` | Asynchronous best-response graph | Proof of Prop. baselineBoASIBRD | manual/unknown (graph diagram) | **ORPHANED — file missing; recover or redraw** |
| `fig:vectorfields1eq` | `images/vf_adad.png`, `images/vf_gtgt.png` | AD vs AD, GT vs GT | Case-by-case analysis | Python: `scripts/plot_vector_fields.py` | OK |
| `fig:eq_tree` | `images/tree_eq.pdf` | Equilibrium trees (AD,AD),(GT,GT) | Case-by-case analysis | manual/unknown (cf. root `alltrees.pdf`) | **ORPHANED — file missing; recover or redraw** |
| `fig:vectorfields2` | `images/vf_gtad.png`, `images/vf_adagt.png` | GT vs AD, AD vs AGT | Case-by-case analysis | Python: `scripts/plot_vector_fields.py` | OK |
| `fig:onestep_tree` | `images/onesteptree.pdf` | One-step trees | Case-by-case analysis | manual/unknown (cf. root `alltrees.pdf`) | **ORPHANED — file missing; recover or redraw** |
| `fig:twosteps` | `images/vf_acgt.png`, `images/vf_acad.png`, `images/vf_agtgt.png`, `images/vf_agtagt.png` | AC vs GT etc. | Case-by-case analysis | Python: `scripts/plot_vector_fields.py` | OK |
| `fig:twostepstree` / `fig:AGT_tree` | `images/twosteps_tree.pdf` | Two-step trees | Case-by-case analysis | manual/unknown | **ORPHANED — file missing; recover or redraw** |
| `fig:vectorfields4` | `images/vf_agtac.png`, `images/vf_acac.png` | AGT vs AC, AC vs AC | Case-by-case analysis | Python: `scripts/plot_vector_fields.py` | OK |
| (unlabeled, l. 1792) | `images/threestepstree.pdf` | Three-step trees | Case-by-case analysis | manual/unknown | **ORPHANED — file missing; recover or redraw** |
| `fig:alpha_effect_ACGT` | `images/af0702_acgt.png`, `images/af_0805_acgt.png` | BoA from (AC,GT), C-hat band | Profile (AC,GT) | Python: `scripts/export_paper_figures.py` | OK |
| `fig:alpha_effect_ACGT_boundary` | `images/af0702_acgt_boundary.png`, `images/af0805_acgt_boundary.png` | First-crossing validation of C-hat | First-crossing validation | Python: `scripts/export_paper_figures.py` | OK |
| `fig:alpha_effect_ACAD` | `images/af0702_acad.png`, `images/af0805_acad.png` | BoA from (AC,AD), D-hat line | Profile (AC,AD) | Python: `scripts/export_paper_figures.py` | OK |
| `fig:acad-entry` | `images/acad_entry_plane.png` | Entry plane (AC,AD)→(GT,AD) | Profile (AC,AD) | Python: `scripts/export_paper_figures.py` | OK |

## Notes

- **Orphaned figures (8 files, 6 figures).** `state_space_abstraction.pdf`,
  `within_each_batch.pdf`, `across_batches.pdf`, `largegraph.pdf`, `tree_eq.pdf`,
  `onesteptree.pdf`, `twosteps_tree.pdf`, `threestepstree.pdf` are referenced by
  `paper.tex` but are not in the repository and have no generating code here.
  They are hand-drawn conceptual diagrams (state-space illustration, batch
  timelines, best-response trees/graphs). **The paper does not compile until
  these are restored** — copy them from the Overleaf project / authors' figure
  folder into `images/`, or redraw them. Root `alltrees.pdf` contains draft
  versions of the tree diagrams; root `Graph1-3.pdf` were matched to
  `IBRGraph1-3.pdf` and copied into `images/`.
- **Mathematica provenance.** The phase diagrams and IBR graphs were originally
  produced by `mathematica/Similarity_Bellman_eq_responses_GT_Space_Expected_SARSA.nb`,
  which also contains the symbolic derivation behind Proposition `thm:IBR`
  (critical curves γ_c1, γ_c2). The phase-diagram *figures* are now reproduced in
  Python (`scripts/plot_phase_diagrams.py`) from those closed-form curves, so
  reviewers do not need Mathematica. See `REPRODUCING.md`.
- **Vector-field trajectories.** Each `vf_*.png` overlays one simulated trajectory
  from a random initialization; regenerated figures show an equivalent but not
  bit-identical trajectory (seed is fixed in the script for internal consistency).
