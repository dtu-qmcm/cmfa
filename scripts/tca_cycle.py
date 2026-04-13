import pdb
import pickle
from itertools import product
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optimistix
from tqdm import tqdm

from cmfa.emu_example import clr_func, ordered_ilr_func, closure, calculate_ci
from cmfa.tca_openflux import TCA_OpenFLux, flux_vector

jax.config.update("jax_enable_x64", True)

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "results" / "tca_cycle"
# M = jnp.array([[-1, -1, 2], [0, 0, -1], [-1, 0, 1]])
V_FREE = {"v8": 0.215, "v1": 1, "v19": 0.05, "v20": 0.03, "v16": 0.07, "v17": 0.23, "v18": 0.12, "v2": 0.201, "v9": 0.296}
KNOWN_KEYS = ["v1", "v16", "v17", "v18", "v19", "v20"]
KNOWN_FLUXES = {k: V_FREE[k] for k in KNOWN_KEYS}
FREE_KEYS = ["v8", "v2", "v9"]

TRUE_FREE_FLUXES = {k: v for k, v in V_FREE.items() if k not in KNOWN_KEYS}  

# KNOWN_FLUXES = {'v1': 1, 'v16': 0.07, 'v17': 0.23, 'v18': 0.12, 'v19': 0.05, 'v20': 0.03}
# TRUE_FREE_FLUXES = {'v8': 0.215, 'v2': 0.201, 'v9': 0.296}
FREE_FLUX_RANGE = [[0.115, 0.315], [0.101, 0.301], [0.196, 0.396]]
FREE_FLUX_GRID_SIZE = [11, 11, 11]
# IX_INPUT_FLUX = np.array([0])
# IX_FREE_FLUXES = np.array([2, 4])
# IX_FIXED_FLUXES = np.array([1, 3, 5])
TOTAL_MOL = 1e5
SIGMA = 0.1
N_MEASUREMENT_PER_FLUX = 50
INITIAL_GUESS = {'v8': 0.2, 'v2': 0.2, 'v9': 0.2}
SIMULATE_COMPOSITION = partial(TCA_OpenFLux, fixed=KNOWN_FLUXES)
SEED = 1234

def simulate_measurement(key, yhat, sigma):
    key, _ = jax.random.split(key)
    key_tree = dict(zip(yhat.keys(), jax.random.split(key, len(yhat))))
    log_mst = jax.tree.map(
        lambda mean, key: 
            jnp.log(mean)+jax.random.normal(key, shape=mean.shape) * sigma, 
            yhat,
            key_tree)
    return jax.tree.map(lambda x: jnp.exp(x), log_mst)

simulate_measurements = jax.vmap(simulate_measurement, in_axes=(0, None, None))

def wssr(y, args):
    target, weights = args
    output = jax.tree.map(
        lambda y, target, weights:
        jnp.sum(weights * (y - target) ** 2),
        y, target, weights
    )
    return jax.tree.reduce(lambda a, b: a + b, output)


def loss_trad(log_free_flux, args):
    composition = SIMULATE_COMPOSITION(jax.tree.map(lambda x: jnp.exp(x), log_free_flux))
    return wssr(composition, args), composition


def loss_clr(log_free_flux, args):
    composition = SIMULATE_COMPOSITION(jax.tree.map(lambda x: jnp.exp(x), log_free_flux))
    target, weights = args
    return wssr(
        jax.tree.map(lambda x: clr_func(x), composition), 
        (jax.tree.map(lambda x: clr_func(x), target), weights)
        ), composition


def loss_ilr(log_free_flux, args):
    composition = SIMULATE_COMPOSITION(jax.tree.map(lambda x: jnp.exp(x), log_free_flux))
    target, weights = args
    return wssr(
        jax.tree.map(lambda x: ordered_ilr_func(x),composition), 
        (jax.tree.map(lambda x: ordered_ilr_func(x), target), weights)
    ), composition

flux_from_free = partial(
    flux_vector,
    fixed=KNOWN_FLUXES,
)
solver = optimistix.BFGS(rtol=1e-8, atol=1e-8)
true_flux = flux_from_free(TRUE_FREE_FLUXES)

true_id = SIMULATE_COMPOSITION(TRUE_FREE_FLUXES)
true_flux_file = OUTPUT_FOLDER / "true_flux.npy"
n_flux = len(true_flux)
n_comp = len(true_id)

true_composition_file = OUTPUT_FOLDER / "true_composition.npy"
print(f"True flux: {true_flux}")
print(f"True composition: {true_id}")
print(f"Writing {true_flux_file}...")
np.save(true_flux_file, true_flux)
print(f"Writing {true_composition_file}...")
np.save(true_composition_file, true_id)
flux_ranges = [
    np.linspace(range_i[0], range_i[1], grid_size_i)
    for range_i, grid_size_i in zip(FREE_FLUX_RANGE, FREE_FLUX_GRID_SIZE)
]

free_fluxes = jnp.array(list(product(*flux_ranges)))

n_free_flux = len(TRUE_FREE_FLUXES)
key = jax.random.key(seed=SEED)

flux_estimates = {
    model: []
    for model in ("trad", "clr", "ilr")
}

flux_ci = {
    model: []
    for model in ("trad", "clr", "ilr")
}
hessian_matrices = {
    model: []
    for model in ("trad", "clr", "ilr")
}
id_estimates = {
    model: []
    for model in ("trad", "clr", "ilr")
}

flux_ci = {
    model: []
    for model in ("trad", "clr", "ilr")
}

true_ids = []
intensity_msts = []

for i, free_flux in enumerate(tqdm(free_fluxes)):
    key, _ = jax.random.split(key)
    keys = jax.random.split(key, N_MEASUREMENT_PER_FLUX)
    free_flux_dict = {k: float(v) for k, v in zip(FREE_KEYS, free_flux)}
    true_ids.append({"flux_set_index": i, "true_ids": SIMULATE_COMPOSITION(free_flux_dict)})
    
    intensity_msts_i = simulate_measurements(keys, true_ids[i]["true_ids"], SIGMA)
    intensity_msts.append({"flux_set_index": i, "intensity_msts": intensity_msts_i})
    id_msts_i = jax.tree.map(jax.vmap(closure), intensity_msts_i)
    id_sd_trad = jax.tree.map(lambda x: np.std(x, axis=0), id_msts_i)
    id_sd_clr = jax.tree.map(lambda x: np.std(clr_func(x), axis=0), id_msts_i)
    id_sd_ilr = jax.tree.map(lambda x: np.std(ordered_ilr_func(x), axis=0), id_msts_i)
    for j in range(N_MEASUREMENT_PER_FLUX):
        mst_j = jax.tree.map(lambda x: x[j], id_msts_i)
    for j, mst in enumerate(id_msts_i):
        for m, loss, sd_tree in (
            ("trad", loss_trad, id_sd_trad),
            ("clr", loss_clr, id_sd_clr),
            ("ilr", loss_ilr, id_sd_ilr),
        ):
            result = optimistix.minimise(
                loss,
                y0=jax.tree.map(lambda x: jnp.log(x), INITIAL_GUESS),
                args=(mst_j, jax.tree.map(lambda s: 1/s, sd_tree)),
                solver=solver,
                max_steps=10000,
                throw=False,
                has_aux=True,
            )
            flux_est = flux_from_free(jax.tree.map(lambda x: jnp.exp(x), result.value))
            id_est = result.aux
            flux_estimates[m].append({"flux_set_index": i,"sample_index": j, "flux_estimates": flux_est})
            id_estimates[m].append({"flux_set_index": i, "sample_index": j, "id_estimates": id_est})
            flux_ci[m].append({"flux_set_index": i,"sample_index": j, "confidence_interval": calculate_ci(result)})      
            # hessian_matrices[m] = hessian_matrices[m].at[i,j].set(result.state.f_info.hessian_inv.pytree)
flux_estimate_path = OUTPUT_FOLDER / "estimated_fluxes.pkl"
print(f"Writing {flux_estimate_path}...")
with open(flux_estimate_path, "wb") as f:
    pickle.dump(flux_estimates, f)

flux_ci_path = OUTPUT_FOLDER / "estimated_fluxes_ci.pkl"
print(f"Writing {flux_ci_path}...")
with open(flux_ci_path, "wb") as f:
    pickle.dump(flux_ci, f)

# hessian_matrices_path = OUTPUT_FOLDER / "hessian_matrices.pkl"
# print(f"Writing {hessian_matrices_path}...")
# with open(hessian_matrices_path, "wb") as f:
#     pickle.dump(hessian_matrices, f)

id_estimate_path = OUTPUT_FOLDER / "estimated_ids.pkl"
print(f"Writing {id_estimate_path}...")
with open(id_estimate_path, "wb") as f:
    pickle.dump(id_estimates, f)

intensity_msts_path = OUTPUT_FOLDER / "intensity_measurements.npy"
print(f"Writing {intensity_msts_path}...")
with open(intensity_msts_path, "wb") as f:
    pickle.dump(intensity_msts, f)

free_flux_path = OUTPUT_FOLDER / "free_fluxes.npy"
print(f"Writing {free_flux_path}...")
jnp.save(free_flux_path, free_fluxes)

true_ids_path = OUTPUT_FOLDER / "true_isotopologue_distributions.npy"
print(f"Writing {true_ids_path}...")
with open(true_ids_path, "wb") as f:
    pickle.dump(true_ids, f)
