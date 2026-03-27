#!/bin/bash

set -e # Exit on error

echo "Starting analysis..."

echo "Step 1: Running scripts..."
uv run scripts/simplex_plot.py

echo "Step 2: Executing notebooks..."
# uv run jupyter execute notebooks/alr_ilr_comparison.ipynb # stale dependency on objective_alr_wssr
# uv run jupyter execute notebooks/example_network_simulations.ipynb # fails
uv run jupyter execute notebooks/example_network_simulations_ci.ipynb
uv run jupyter execute notebooks/figure_1_creation.ipynb
# uv run jupyter execute notebooks/flux_analysis.ipynb # fails (missing objective_leave_one_out_wssr)
uv run jupyter execute notebooks/flux_range_TCA.ipynb
uv run jupyter execute notebooks/metabolic_network_TCA.ipynb
# uv run jupyter execute notebooks/simplex_plot.ipynb # fails (ipynb file incorrectly formatted)
uv run jupyter execute notebooks/TCA_OpenFLUX.ipynb
# uv run jupyter execute notebooks/TCA_OpenFLUX_CI.ipynb # fails (SMQError: Too many open files)
uv run jupyter execute notebooks/theoretical_example.ipynb
uv run jupyter execute notebooks/toy_example.ipynb
# uv run jupyter execute notebooks/unconstrained_coda_comparison.ipynb # fails (missing objective_ilr_wssr, objective_leave_one_out_wssr)

echo "Analysis pipeline completed successfully!"

