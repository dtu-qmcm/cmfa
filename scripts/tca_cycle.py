import pickle
from functools import partial
from itertools import product
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optimistix
from tqdm import tqdm

from cmfa.functions import calculate_ci, closure, ordered_ilr_func
from cmfa.tca_cycle_emu import TCA_cycle, flux_vector, get_N

jax.config.update("jax_enable_x64", True)

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "results" / "tca_cycle"

TRUE_FLUX_FILE = OUTPUT_FOLDER / "true_flux.npy"
TRUE_COMPOSITION_FILE = OUTPUT_FOLDER / "true_composition.npy"

V_FREE = {"v8": 0.215, "v1": 1, "v19": 0.05, "v20": 0.03, "v16": 0.07, "v17": 0.23, "v18": 0.12, "v2": 0.201, "v9": 0.296}
KNOWN_KEYS = ["v1", "v16", "v17", "v18", "v19", "v20"]
KNOWN_FLUXES = {k: V_FREE[k] for k in KNOWN_KEYS}
FREE_KEYS = ["v8", "v2", "v9"]
N = get_N()
TRUE_FREE_FLUXES_DICT = {k: v for k, v in V_FREE.items() if k not in KNOWN_KEYS}
TRUE_FREE_FLUXES = jnp.array([0.215, 0.201, 0.296])
FREE_FLUX_RANGE = [[0.5 * v, 1.5 * v] for v in TRUE_FREE_FLUXES_DICT.values()]
FREE_FLUX_GRID_SIZE = [5, 5, 5]
SIGMA = 0.1
M_MST_PER_GRID_CELL = 50
N_MST_REF = 10000
INITIAL_GUESS = jnp.array([0.2, 0.2, 0.2])
SIMULATE_COMPOSITION = partial(TCA_cycle, fixed=KNOWN_FLUXES)
SEED = 1234


def simulate_measurement(key, yhat, sigma):
    key, _ = jax.random.split(key)
    key_tree = dict(zip(yhat.keys(), jax.random.split(key, len(yhat))))
    log_mst = jax.tree.map(
        lambda mean, key:
            jnp.log(mean) + jax.random.normal(key, shape=mean.shape) * sigma,
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
    composition = SIMULATE_COMPOSITION(jnp.exp(log_free_flux))
    return wssr(composition, args), composition


def loss_ilr(log_free_flux, args):
    composition = SIMULATE_COMPOSITION(jnp.exp(log_free_flux))
    target, weights = args
    return wssr(
        jax.tree.map(lambda x: ordered_ilr_func(x), composition),
        (jax.tree.map(lambda x: ordered_ilr_func(x), target), weights)
    ), composition


def get_sigma_v(n, inv_hessian):
    n_nofixed = jnp.delete(n, np.array([1, 2, 3, 4, 5, 6]), axis=1)
    return n_nofixed @ inv_hessian @ n_nofixed.T


def get_flux_grid(limits, sizes):
    values = (
        np.linspace(low, high, size) for (low, high), size in zip(limits, sizes)
    )
    return jnp.array(list(product(*values)))


def main():
    print(f"Running TCA cycle\n")
    flux_from_free = partial(
        flux_vector,
        fixed=KNOWN_FLUXES,
    )

    def empties(shape):
        return {model: jnp.empty(shape) for model in ("trad", "ilr")}

    def fit(mst, loss, sd):
        result = optimistix.minimise(
            loss,
            y0=jnp.log(INITIAL_GUESS),
            args=(mst, jax.tree.map(lambda s: 1 / s, sd)),
            solver=solver,
            max_steps=10000,
            throw=False,
            has_aux=True,
        )
        mid_est = result.aux
        free_flux_est_arr = jnp.exp(result.value)
        flux_est = flux_from_free(free_flux_est_arr)
        flux_est_jax = jnp.array([flux_est[k] for k in flux_est.keys()])
        free_flux_est = jax.tree.map(lambda x: flux_est[x].item(), FREE_KEYS)
        inv_hessian = result.state.f_info.hessian_inv.pytree
        sigma_v = get_sigma_v(N, inv_hessian)
        flux_sd = jnp.sqrt(jnp.diag(sigma_v))
        ci = calculate_ci(sigma_v, jnp.log(flux_est_jax))
        fval = float(result.state.f_info.f)
        return free_flux_est, flux_est_jax, mid_est, ci, inv_hessian, fval, flux_sd

    model_to_loss = {"trad": loss_trad, "ilr": loss_ilr}

    solver = optimistix.BFGS(rtol=1e-8, atol=1e-8)
    true_flux = flux_from_free(TRUE_FREE_FLUXES)
    true_mid = SIMULATE_COMPOSITION(TRUE_FREE_FLUXES)
    n_flux = len(true_flux)
    n_free_flux = len(TRUE_FREE_FLUXES)
    print(f"True flux: {true_flux}")
    print(f"True composition: {true_mid}")
    print(f"Writing {TRUE_FLUX_FILE}...")
    np.save(TRUE_FLUX_FILE, true_flux)
    print(f"Writing {TRUE_COMPOSITION_FILE}...")
    np.save(TRUE_COMPOSITION_FILE, true_mid)

    flux_grid = get_flux_grid(FREE_FLUX_RANGE, FREE_FLUX_GRID_SIZE)
    n_grid = len(flux_grid)
    key = jax.random.key(seed=SEED)

    # initialise containers
    flux_est_ref = empties((N_MST_REF, n_free_flux))
    flux_est_full = empties((N_MST_REF, n_flux))
    flux_ci_ref = empties((N_MST_REF, n_flux, 2))
    flux_sd_ref = empties((N_MST_REF, n_flux))
    fval_ref = empties((N_MST_REF,))
    mid_est_ref = {model: [0 for _ in range(N_MST_REF)] for model in ("trad", "ilr")}
    flux_estimates = empties((n_grid, M_MST_PER_GRID_CELL, n_free_flux))
    flux_ci = empties((n_grid, M_MST_PER_GRID_CELL, n_flux, 2))
    mid_estimates = {
        model: [[0 for _ in range(M_MST_PER_GRID_CELL)] for _ in range(n_grid)]
        for model in ("trad", "ilr")
    }
    true_mids = []
    intensity_msts_grid = []

    key_ref, key_grid = jax.random.split(key)

    # Fit the reference flux N_MST_REF times
    keys_ref = jax.random.split(key_ref, N_MST_REF)
    intensity_msts_ref = simulate_measurements(keys_ref, true_mid, SIGMA)
    mid_msts_ref = jax.tree.map(jax.vmap(closure), intensity_msts_ref)
    mid_sds_ref = {
        "trad": jax.tree.map(lambda x: np.std(x, axis=0), mid_msts_ref),
        "ilr": jax.tree.map(lambda x: np.std(ordered_ilr_func(x), axis=0), mid_msts_ref),
    }
    print("Running TCA cycle ground truth flux recovery...")
    for j in tqdm(range(N_MST_REF)):
        mst = jax.tree.map(lambda x: x[j], mid_msts_ref)
        for mod in ("trad", "ilr"):
            free_flux_est, flux_est_jax, mid_est, ci, _, fval, flux_sd = fit(mst, model_to_loss[mod], mid_sds_ref[mod])
            flux_est_ref[mod] = flux_est_ref[mod].at[j].set(free_flux_est)
            flux_est_full[mod] = flux_est_full[mod].at[j].set(flux_est_jax)
            mid_est_ref[mod][j] = mid_est
            flux_ci_ref[mod] = flux_ci_ref[mod].at[j].set(ci)
            flux_sd_ref[mod] = flux_sd_ref[mod].at[j].set(flux_sd)
            fval_ref[mod] = fval_ref[mod].at[j].set(fval)

    # Fit each free flux combination in the grid M_MST_PER_GRID_CELL times
    keys_grid = jax.random.split(key_grid, n_grid)
    print("Running TCA cycle sensitivity analysis...")
    for (i, free_flux), key_i in zip(enumerate(tqdm(flux_grid)), keys_grid):
        keys_i = jax.random.split(key_i, M_MST_PER_GRID_CELL)
        true_mid_i = SIMULATE_COMPOSITION(free_flux)
        intensity_msts_i = simulate_measurements(keys_i, true_mid_i, SIGMA)
        intensity_msts_grid.append(intensity_msts_i)
        true_mids.append(true_mid_i)
        mid_msts_i = jax.tree.map(jax.vmap(closure), intensity_msts_i)
        mid_sds_i = {
            "trad": jax.tree.map(lambda x: np.std(x, axis=0), mid_msts_i),
            "ilr": jax.tree.map(lambda x: np.std(ordered_ilr_func(x), axis=0), mid_msts_i),
        }
        for j in range(M_MST_PER_GRID_CELL):
            mst = jax.tree.map(lambda x: x[j], mid_msts_i)
            for mod in ("trad", "ilr"):
                free_flux_est, _, mid_est, ci, _, _, _ = fit(mst, model_to_loss[mod], mid_sds_i[mod])
                flux_estimates[mod] = flux_estimates[mod].at[i, j].set(free_flux_est)
                mid_estimates[mod][i][j] = mid_est
                flux_ci[mod] = flux_ci[mod].at[i, j].set(ci)

    for obj, filename in (
        (flux_estimates, "estimated_fluxes_grid.pkl"),
        (flux_ci, "estimated_fluxes_ci_grid.pkl"),
        (mid_estimates, "estimated_mids_grid.pkl"),
        (flux_est_ref, "estimated_fluxes_reference.pkl"),
        (flux_est_full, "estimated_fluxes_full.pkl"),
        (flux_ci_ref, "estimated_fluxes_ci_reference.pkl"),
        (mid_est_ref, "estimated_mids_reference.pkl"),
        (fval_ref, "fvalues_reference.pkl"),
        (flux_sd_ref, "estimated_fluxes_sd_reference.pkl"),
        (mid_msts_ref, "mid_simulated_reference.pkl"),
        (intensity_msts_grid, "intensity_measurements_grid.npy"),
        (true_mids, "true_isotopologue_distributions.npy"),
    ):
        path = OUTPUT_FOLDER / filename
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    for arr, filename in (
        (flux_grid, "free_fluxes_grid.npy"),
    ):
        path = OUTPUT_FOLDER / filename
        jnp.save(path, arr)


if __name__ == "__main__":
    main()
