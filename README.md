# Improved Metabolic Flux Estimations through Compositional Data Analysis

Code to reproduce all results and figures from the paper **"Improved Metabolic Flux Estimations through Compositional Data Analysis"**.

The core idea is to replace the traditional Euclidean MSE on mass isotopologue distributions (MIDs) with an isometric log-ratio (ILR) MSE, respecting the compositional (simplex) geometry of MID data. We demonstrate on a toy network and a TCA cycle model that this reduces estimation bias and yields tighter confidence intervals.

---

## Repository structure

```
cmfa/
├── src/cmfa/                        # Core library
│   ├── functions.py                 # ILR/CLR transforms, profile likelihood, CI calculation
│   ├── toy_network_emu.py           # EMU forward model — toy network
│   └── tca_cycle_emu.py             # EMU forward model — TCA cycle
│
├── scripts/                         # Runnable analysis scripts
│   ├── toy_network.py               # Simulate & estimate fluxes — toy network
│   ├── tca_cycle.py                 # Simulate & estimate fluxes — TCA cycle
│   ├── profile_likelihoods.py       # Profile likelihood curves (both networks)
│   ├── supplementary_data_1.py      # Export Supplementary Data 1 (reference estimates)
│   ├── supplementary_data_2.py      # Export Supplementary Data 2 (sensitivity grid)
│   └── supplementary_fig_c7.py      # Supplementary Figure C7 (permutation invariance)
│
├── notebooks/                       # Figure-generating notebooks
│   ├── ci_plot.ipynb                # Figure 4 — CI comparison (MC / Fisher / profile likelihood)
│   ├── main_plots.ipynb             # Figures 3 & 5 — flux residuals, MSE difference
│   ├── supplementary_fig_b6.ipynb   # Supplementary Figure B6 — cartesian-to-ternary
│   ├── toy_network_notebook.ipynb   # Toy network flux summary table
│   └── tca_cycle_notebook.ipynb     # TCA cycle flux summary table
│
├── figures/                         # Generated figures (PNG / SVG)
│   ├── 3_flux_residuals.*           # Figure 3
│   ├── 4_CI_combined.*              # Figure 4
│   ├── 5_mse_difference.png         # Figure 5
│   ├── b6_cartesian_to_ternary.*    # Supplementary Figure B6
│   └── c7_permutation_invariance.png # Supplementary Figure C7
│
├── data/
│   ├── prepared/                    # Pre-processed flux samples
│   └── results/                     # Output from analysis scripts (generated)
│       ├── toy_example/
│       └── tca_cycle/
│
├── run_analysis.sh                  # Full pipeline (step 1–3)
└── pyproject.toml                   # Dependencies & build config
```

---

## Installation

Requires Python ≥ 3.10. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the project and all dependencies
uv sync
```

---

## Running the analysis

### Full pipeline

```bash
bash run_analysis.sh
```

---

## Citation

> Citation will be added upon publication.

---

## Authors

- Alberte Sloth Fjord (afjsl@dtu.dk)
- Te Chen (techen@dtu.dk)
- Teddy Groves (tedgro@dtu.dk)

---

## Funding

The Novo Nordisk Foundation Biotechnology Research Institute for the Green Transition, Technical University of Denmark (DTU)

Novo Nordisk Foundation
(NNF14OC0009473 and NNF24SA0100980)

EU Horizon project DIGIBIO (101060066)

---

## License

This repository is released under the [MIT License](LICENSE). You are free to use, modify, and redistribute this code provided the copyright notice is retained.
