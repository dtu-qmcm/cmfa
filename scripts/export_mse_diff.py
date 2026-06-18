import numpy as np
import jax.numpy as jnp
import pickle
import pandas as pd
from pathlib import Path

RESULTS_DIR_TOY = Path(__file__).parent.parent / "data" / "results" / "toy_example"
RESULTS_DIR_TCA = Path(__file__).parent.parent / "data" / "results" / "tca_cycle"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "results" / "mse_diff_results.xlsx"

# ── Toy Network ───────────────────────────────────────────────────────────────
free_fluxes_toy = np.load(RESULTS_DIR_TOY / "free_fluxes.npy")
with open(RESULTS_DIR_TOY / "estimated_fluxes.pkl", "rb") as f:
    estimated_fluxes_toy = pickle.load(f)

resid_trad_toy = free_fluxes_toy[:, None, :] - estimated_fluxes_toy["trad"][:, :, [4, 5]]
resid_ilr_toy  = free_fluxes_toy[:, None, :] - estimated_fluxes_toy["ilr"][:, :, [4, 5]]

mse_trad_toy = (resid_trad_toy ** 2).mean(axis=(0, 1))
mse_ilr_toy  = (resid_ilr_toy  ** 2).mean(axis=(0, 1))
diff_flux_toy = (resid_trad_toy ** 2).mean(axis=1) - (resid_ilr_toy ** 2).mean(axis=1)

df_toy = pd.DataFrame({
    "free_flux_v3":  free_fluxes_toy[:, 0],
    "free_flux_v5":  free_fluxes_toy[:, 1],
    "mse_trad_v3":   (resid_trad_toy ** 2).mean(axis=1)[:, 0],
    "mse_ilr_v3":    (resid_ilr_toy  ** 2).mean(axis=1)[:, 0],
    "mse_diff_v3":   diff_flux_toy[:, 0],
    "mse_trad_v5":   (resid_trad_toy ** 2).mean(axis=1)[:, 1],
    "mse_ilr_v5":    (resid_ilr_toy  ** 2).mean(axis=1)[:, 1],
    "mse_diff_v5":   diff_flux_toy[:, 1],
})

# ── TCA Cycle ─────────────────────────────────────────────────────────────────
free_fluxes_tca = np.array(jnp.load(RESULTS_DIR_TCA / "free_fluxes.npy", allow_pickle=True))
with open(RESULTS_DIR_TCA / "estimated_fluxes.pkl", "rb") as f:
    estimated_fluxes_tca = pickle.load(f)

resid_trad_tca = free_fluxes_tca[:, None, :] - np.array(estimated_fluxes_tca["trad"])
resid_ilr_tca  = free_fluxes_tca[:, None, :] - np.array(estimated_fluxes_tca["ilr"])

mse_trad_tca = (resid_trad_tca ** 2).mean(axis=(0, 1))
mse_ilr_tca  = (resid_ilr_tca  ** 2).mean(axis=(0, 1))
diff_flux_tca = (resid_trad_tca ** 2).mean(axis=1) - (resid_ilr_tca ** 2).mean(axis=1)

# TCA columns order: v8 (0), v2 (1), v9 (2)
df_tca = pd.DataFrame({
    "free_flux_v8":  free_fluxes_tca[:, 0],
    "free_flux_v2":  free_fluxes_tca[:, 1],
    "free_flux_v9":  free_fluxes_tca[:, 2],
    "mse_trad_v8":   (resid_trad_tca ** 2).mean(axis=1)[:, 0],
    "mse_ilr_v8":    (resid_ilr_tca  ** 2).mean(axis=1)[:, 0],
    "mse_diff_v8":   diff_flux_tca[:, 0],
    "mse_trad_v2":   (resid_trad_tca ** 2).mean(axis=1)[:, 1],
    "mse_ilr_v2":    (resid_ilr_tca  ** 2).mean(axis=1)[:, 1],
    "mse_diff_v2":   diff_flux_tca[:, 1],
    "mse_trad_v9":   (resid_trad_tca ** 2).mean(axis=1)[:, 2],
    "mse_ilr_v9":    (resid_ilr_tca  ** 2).mean(axis=1)[:, 2],
    "mse_diff_v9":   diff_flux_tca[:, 2],
})

# ── Summary ───────────────────────────────────────────────────────────────────
df_summary = pd.DataFrame({
    "Network": ["Toy", "Toy", "TCA", "TCA", "TCA"],
    "Flux":    ["v3",  "v5",  "v8",  "v2",  "v9"],
    "MSE_traditional": [
        float(mse_trad_toy[0]), float(mse_trad_toy[1]),
        float(mse_trad_tca[0]), float(mse_trad_tca[1]), float(mse_trad_tca[2]),
    ],
    "MSE_compositional": [
        float(mse_ilr_toy[0]), float(mse_ilr_toy[1]),
        float(mse_ilr_tca[0]), float(mse_ilr_tca[1]), float(mse_ilr_tca[2]),
    ],
    "Mean_MSE_diff": [
        float(diff_flux_toy[:, 0].mean()), float(diff_flux_toy[:, 1].mean()),
        float(diff_flux_tca[:, 0].mean()), float(diff_flux_tca[:, 1].mean()), float(diff_flux_tca[:, 2].mean()),
    ],
    "Pct_improvement": [
        float(((mse_trad_toy[0] - mse_ilr_toy[0]) / mse_trad_toy[0]) * 100),
        float(((mse_trad_toy[1] - mse_ilr_toy[1]) / mse_trad_toy[1]) * 100),
        float(((mse_trad_tca[0] - mse_ilr_tca[0]) / mse_trad_tca[0]) * 100),
        float(((mse_trad_tca[1] - mse_ilr_tca[1]) / mse_trad_tca[1]) * 100),
        float(((mse_trad_tca[2] - mse_ilr_tca[2]) / mse_trad_tca[2]) * 100),
    ],
})

# ── Write Excel ───────────────────────────────────────────────────────────────
with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
    df_summary.to_excel(writer, sheet_name="Summary", index=False)
    df_toy.to_excel(writer, sheet_name="Toy Network", index=False)
    df_tca.to_excel(writer, sheet_name="TCA Cycle", index=False)

print("Saved to", OUTPUT_PATH)
print(df_summary.to_string(index=False))
