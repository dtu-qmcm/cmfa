#!/bin/bash

set -e # Exit on error

echo "Starting analysis..."

echo "Step 1: Running analysis..."
uv run scripts/toy_network.py || exit 1
uv run scripts/tca_cycle.py || exit 1
uv run scripts/profile_likelihoods.py || exit 1

echo "Step 2: Making plots..."
uv run jupyter execute notebooks/main_plots.ipynb || exit 1
uv run jupyter execute notebooks/ci_plots.ipynb || exit 1
uv run jupyter execute notebooks/supplementary_fig_b6.ipynb || exit 1
uv run scripts/supplementary_fig_c7.py || exit 1

echo "Step 3: Making tables..."
uv run jypyter execute notebooks/toy_network_notebook.ipynb || exit 1
uv run jypyter execute notebooks/tca_cycle_notebook.ipynb || exit 1
uv run scripts/supplementary_data_1.py || exit 1
uv run scripts/supplementary_data_2.py || exit 1

echo "Analysis pipeline completed successfully!"

