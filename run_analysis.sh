#!/bin/bash

set -e # Exit on error

echo "Starting analysis..."

echo "Step 1: Running scripts..."
uv run scripts/simplex_plot.py
uv run scripts/permutation_invariance_plot.py
uv run scripts/toy_network.py
uv run scripts/tca_cycle.py
uv run scripts/profile_likelihood.py
uv run scripts/export_flux_residuals_reference.py
uv run scripts/export_mse_diff.py

echo "Step 2: Executing notebooks..."
uv run jupyter execute notebooks/main_plots.ipynb
uv run jupyter execute notebooks/ci_plots.ipynb

echo "Analysis pipeline completed successfully!"

