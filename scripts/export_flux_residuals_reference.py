import numpy as np
import pickle
import pandas as pd
from pathlib import Path

RESULTS_DIR_TOY = Path(__file__).parent.parent / "data" / "results" / "toy_example"
RESULTS_DIR_TCA = Path(__file__).parent.parent / "data" / "results" / "tca_cycle"
OUTPUT_PATH = Path(__file__).parent.parent / "data" / "results" / "flux_residuals_reference.xlsx"

# ── Toy Network ───────────────────────────────────────────────────────────────
true_flux_toy = np.load(RESULTS_DIR_TOY / "true_flux.npy")
with open(RESULTS_DIR_TOY / "estimated_fluxes_reference.pkl", "rb") as f:
    est_toy = pickle.load(f)

# true_flux_toy indices [2, 4] correspond to v3 and v5
res_trad_toy = true_flux_toy[None, [2, 4]] - est_toy["trad"][:, [4, 5]]
res_comp_toy = true_flux_toy[None, [2, 4]] - est_toy["ilr"][:, [4, 5]]
mse_trad_toy = (res_trad_toy ** 2).mean(axis=0)
mse_comp_toy = (res_comp_toy ** 2).mean(axis=0)

df_toy = pd.DataFrame({
    "est_trad_v3":      est_toy["trad"][:, 4],
    "residual_trad_v3": res_trad_toy[:, 0],
    "est_ilr_v3":       est_toy["ilr"][:, 4],
    "residual_ilr_v3":  res_comp_toy[:, 0],
    "est_trad_v5":      est_toy["trad"][:, 5],
    "residual_trad_v5": res_trad_toy[:, 1],
    "est_ilr_v5":       est_toy["ilr"][:, 5],
    "residual_ilr_v5":  res_comp_toy[:, 1],
})

# ── TCA Cycle ─────────────────────────────────────────────────────────────────
true_flux_tca = np.load(RESULTS_DIR_TCA / "true_flux.npy", allow_pickle=True).item()
with open(RESULTS_DIR_TCA / "estimated_fluxes_reference.pkl", "rb") as f:
    est_tca = pickle.load(f)

# est_tca columns order: v8, v2, v9
true_ref_tca = np.array([float(true_flux_tca[k]) for k in ["v8", "v2", "v9"]])
res_trad_tca = np.array(est_tca["trad"]) - true_ref_tca[None, :]
res_comp_tca = np.array(est_tca["ilr"])  - true_ref_tca[None, :]
mse_trad_tca = (res_trad_tca ** 2).mean(axis=0)
mse_comp_tca = (res_comp_tca ** 2).mean(axis=0)

df_tca = pd.DataFrame({
    "est_trad_v8":      np.array(est_tca["trad"])[:, 0],
    "residual_trad_v8": res_trad_tca[:, 0],
    "est_ilr_v8":       np.array(est_tca["ilr"])[:, 0],
    "residual_ilr_v8":  res_comp_tca[:, 0],
    "est_trad_v2":      np.array(est_tca["trad"])[:, 1],
    "residual_trad_v2": res_trad_tca[:, 1],
    "est_ilr_v2":       np.array(est_tca["ilr"])[:, 1],
    "residual_ilr_v2":  res_comp_tca[:, 1],
    "est_trad_v9":      np.array(est_tca["trad"])[:, 2],
    "residual_trad_v9": res_trad_tca[:, 2],
    "est_ilr_v9":       np.array(est_tca["ilr"])[:, 2],
    "residual_ilr_v9":  res_comp_tca[:, 2],
})

# ── Summary ───────────────────────────────────────────────────────────────────
df_summary = pd.DataFrame({
    "Network": ["Toy", "Toy", "TCA", "TCA", "TCA"],
    "Flux":    ["v3",  "v5",  "v8",  "v2",  "v9"],
    "True_flux": [
        true_flux_toy[2], true_flux_toy[4],
        float(true_flux_tca["v8"]), float(true_flux_tca["v2"]), float(true_flux_tca["v9"]),
    ],
    "MSE_traditional": [
        float(mse_trad_toy[0]), float(mse_trad_toy[1]),
        float(mse_trad_tca[0]), float(mse_trad_tca[1]), float(mse_trad_tca[2]),
    ],
    "MSE_compositional": [
        float(mse_comp_toy[0]), float(mse_comp_toy[1]),
        float(mse_comp_tca[0]), float(mse_comp_tca[1]), float(mse_comp_tca[2]),
    ],
    "Pct_improvement": [
        float(((mse_trad_toy[0] - mse_comp_toy[0]) / mse_trad_toy[0]) * 100),
        float(((mse_trad_toy[1] - mse_comp_toy[1]) / mse_trad_toy[1]) * 100),
        float(((mse_trad_tca[0] - mse_comp_tca[0]) / mse_trad_tca[0]) * 100),
        float(((mse_trad_tca[1] - mse_comp_tca[1]) / mse_trad_tca[1]) * 100),
        float(((mse_trad_tca[2] - mse_comp_tca[2]) / mse_trad_tca[2]) * 100),
    ],
})

# ── Write Excel ───────────────────────────────────────────────────────────────
with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
    df_summary.to_excel(writer, sheet_name="Summary", index=False)
    df_toy.to_excel(writer, sheet_name="Toy Network", index=False)
    df_tca.to_excel(writer, sheet_name="TCA Cycle", index=False)

print("Saved to", OUTPUT_PATH)
print(df_summary.to_string(index=False))
