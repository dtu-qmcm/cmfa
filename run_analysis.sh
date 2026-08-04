#!/bin/bash

set -e # Exit on error

echo "Starting analysis..."

echo "Step 1: Running analysis..."
uv run scripts/toy_network.py
uv run scripts/tca_cycle.py
uv run scripts/profile_likelihoods.py

echo "Step 2: Making plots..."
uv run jupyter execute notebooks/main_plots.ipynb
uv run jupyter execute notebooks/ci_plots.ipynb
uv run jupyter execute notebooks/supplementary_fig_b6.ipynb
uv run scripts/supplementary_fig_c7.py

echo "Step 3: Making tables..."
uv run jypyter execute notebooks/toy_network_notebook.ipynb
uv run jypyter execute notebooks/tca_cycle_notebook.ipynb
uv run scripts/supplementary_data_1.py
uv run scripts/supplementary_data_2.py

echo "Analysis pipeline completed successfully!"

