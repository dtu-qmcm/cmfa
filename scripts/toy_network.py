import pickle
from functools import partial
from itertools import product
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optimistix
from tqdm import tqdm

from cmfa.functions import (
    calculate_ci,
    closure,
    ordered_ilr_func,
)
from cmfa.toy_network_emu import toy_network

jax.config.update("jax_enable_x64", True)

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "results" / "toy_example"
TRUE_FLUX_FILE = OUTPUT_FOLDER / "true_flux.npy"
TRUE_COMPOSITION_FILE = OUTPUT_FOLDER / "true_composition.npy"
M = jnp.array([[-1, -1, 2], [0, 0, -1], [-1, 0, 1]])
N = jnp.vstack([-M, jnp.eye(3)])
KNOWN_FLUXES = jnp.array([100.0])
TRUE_FREE_FLUXES = jnp.array([50.0, 20.0])
FREE_FLUX_RANGE = [
    [0.5 * TRUE_FREE_FLUXES[0], 1.5 * TRUE_FREE_FLUXES[0]],
    [0.5 * TRUE_FREE_FLUXES[1], 1.5 * TRUE_FREE_FLUXES[1]],
]
FREE_FLUX_GRID_SIZE = [11, 11]
IX_INPUT_FLUX = np.array([0])
IX_FREE_FLUXES = np.array([2, 4])
IX_FIXED_FLUXES = np.array([1, 3, 5])
SIGMA = 0.1
M_MST_PER_GRID_CELL = 50
N_MST_REF = 10000

INITIAL_GUESS = jnp.array([100, 50])
SIMULATE_COMPOSITION = partial(toy_network, v1=KNOWN_FLUXES[0].item())
SEED = 1234


def get_flux_vector(free, known, ix_known, ix_free, M):
    n = known.shape[0] + free.shape[0] + M.shape[0]
    ix_known_or_free = jnp.concat([ix_free, ix_known])
    ix_fixed = jnp.delete(jnp.arange(n), ix_known_or_free)
    out = jnp.zeros(shape=(n,)).at[ix_known].set(known).at[ix_free].set(free)
    fixed = -M @ out[jnp.concat([ix_known, ix_free])]
    u = jnp.stack([-M, jnp.identity(len(ix_known_or_free))]) @ jnp.concat(
        [known, free]
    )
    return out.at[ix_fixed].set(fixed)


def simulate_measurement(key, yhat, sigma):
    err = jax.random.normal(key, shape=(yhat.shape)) * sigma
    log_mst = jnp.log(yhat) + err
    return jnp.exp(log_mst)


simulate_measurements = jax.vmap(simulate_measurement, in_axes=(0, None, None))


def wssr(y, args):
    target, weights = args
    if target.ndim>1:   
        n_dim = target.shape[0]
        ys = jnp.tile(y, (n_dim, 1))
    else:
        ys=y
    return jnp.sum(weights * (ys - target) ** 2)

def wssr_old(y, args):
    target, weights = args
    return jnp.sum(weights * (y - target) ** 2)


def loss_trad(log_free_flux, args):
    composition = SIMULATE_COMPOSITION(jnp.exp(log_free_flux))
    return wssr(composition, args), composition



def loss_ilr(log_free_flux, args):
    composition = SIMULATE_COMPOSITION(jnp.exp(log_free_flux))
    target, weights = args
    return wssr(
        ordered_ilr_func(composition), (ordered_ilr_func(target), weights)
    ), composition


def get_sigma_v(N, inv_hessian):
    n_nofixed = jnp.delete(N, IX_INPUT_FLUX, axis=1)
    # jax.debug.breakpoint()
    return n_nofixed @ inv_hessian @ n_nofixed.T


def get_flux_grid(limits, sizes):
    values = (
        np.linspace(low, high, size) for (low, high), size in zip(limits, sizes)
    )
    return jnp.array(list(product(*values)))


def main():
    expand_free_fluxes = partial(
        get_flux_vector,
        known=KNOWN_FLUXES,
        ix_known=IX_INPUT_FLUX,
        ix_free=IX_FREE_FLUXES,
        M=M,
    )

    def empties(shape):
        return {model: jnp.empty(shape) for model in ("trad", "ilr")}

    def fit(mst, loss, sd):
        result = optimistix.minimise(
            loss,
            y0=jnp.log(INITIAL_GUESS),
            args=(mst, 1 / sd),
            solver=solver,
            max_steps=10000,
            throw=False,
            has_aux=True,
        )
        log_free_flux_est = result.value
        mid_est = result.aux
        fval = float(result.state.f_info.f)
        free_flux_est = jnp.exp(log_free_flux_est)
        flux_est = expand_free_fluxes(free_flux_est)[corrected_index]
        inv_hessian = result.state.f_info.hessian_inv.pytree
        sigma_v = get_sigma_v(N, inv_hessian)
        flux_sd_est = jnp.sqrt(jnp.diag(sigma_v))
        ci = calculate_ci(sigma_v, jnp.log(flux_est))
        return flux_est, mid_est, ci, inv_hessian, fval, flux_sd_est

    model_to_loss = {"trad": loss_trad, "ilr": loss_ilr}

    solver = optimistix.BFGS(rtol=1e-8, atol=1e-8)
    true_flux = expand_free_fluxes(TRUE_FREE_FLUXES)
    true_mid = SIMULATE_COMPOSITION(TRUE_FREE_FLUXES)
    n_flux = len(true_flux)
    n_comp = len(true_mid)
    print(f"True flux: {true_flux}")
    print(f"True composition: {true_mid}")
    print(f"Writing {TRUE_FLUX_FILE}...")
    np.save(TRUE_FLUX_FILE, true_flux)
    print(f"Writing {TRUE_COMPOSITION_FILE}...")
    np.save(TRUE_COMPOSITION_FILE, true_mid)
    flux_grid = get_flux_grid(FREE_FLUX_RANGE, FREE_FLUX_GRID_SIZE)
    n_grid = len(flux_grid)
    n_free = len(IX_FREE_FLUXES)
    key = jax.random.key(seed=SEED)

    # initialise containers
    flux_est_ref = empties((N_MST_REF, n_flux))
    flux_ci_ref = empties((N_MST_REF, n_flux, 2))
    mid_est_ref = empties((N_MST_REF, n_comp))
    flux_sd_est = empties((N_MST_REF, n_flux))
    flux_est_grid = empties((n_grid, M_MST_PER_GRID_CELL, n_flux))
    flux_ci_grid = empties((n_grid, M_MST_PER_GRID_CELL, n_flux, 2))
    mid_est_grid = empties((n_grid, M_MST_PER_GRID_CELL, n_comp))
    intensity_msts_grid = jnp.empty((n_grid, M_MST_PER_GRID_CELL, n_comp))
    mid_msts_grid = jnp.empty((n_grid, M_MST_PER_GRID_CELL, n_comp))
    true_mids_grid = jnp.empty((n_grid, n_comp))

    # get keys
    key_ref, key_grid = jax.random.split(key)

    # Fit the reference flux N_MEASUREMENT_REF times
    corrected_index = jnp.array([1, 3, 5, 0, 2, 4])
    keys_ref = jax.random.split(key_ref, N_MST_REF)
    intensity_msts_ref = simulate_measurements(keys_ref, true_mid, SIGMA)
    mid_msts_ref = closure(intensity_msts_ref)
    mid_sds_ref = {
        "trad": np.std(mid_msts_ref, axis=0),
        "ilr": np.std(ordered_ilr_func(mid_msts_ref), axis=0),
    }
    for j, mst in enumerate(tqdm(mid_msts_ref)):
        for mod in ("trad", "ilr"):
            loss = model_to_loss[mod]
            composition_sd = mid_sds_ref[mod]
            flux_est, mid_est, ci, _, _, flux_sd = fit(mst, loss, composition_sd)
            flux_est_ref[mod] = flux_est_ref[mod].at[j].set(flux_est)
            mid_est_ref[mod] = mid_est_ref[mod].at[j].set(mid_est)
            flux_ci_ref[mod] = flux_ci_ref[mod].at[j].set(ci)
            flux_sd_est[mod] = flux_sd_est[mod].at[j].set(flux_sd)

    # fit each free flux combination in the grid N_MST_PER_FLUX times
    keys_grid = jax.random.split(key_grid, n_grid)
    for (i, free_flux), key_i in zip(enumerate(tqdm(flux_grid)), keys_grid):
        keys_i = jax.random.split(key_i, M_MST_PER_GRID_CELL)
        true_mid_i = SIMULATE_COMPOSITION(free_flux)
        intensity_msts_i = simulate_measurements(keys_i, true_mid_i, SIGMA)
        mid_msts_i = closure(intensity_msts_i)
        mid_msts_grid = mid_msts_grid.at[i].set(mid_msts_i)
        intensity_msts_grid = intensity_msts_grid.at[i].set(intensity_msts_i)
        true_mids_grid = true_mids_grid.at[i].set(true_mid_i)
        mid_sds_i = {
            "trad": np.std(mid_msts_i, axis=0),
            "ilr": np.std(ordered_ilr_func(mid_msts_i), axis=0),
        }
        for j, mst in enumerate(mid_msts_i):
            for mod in ("trad", "ilr"):
                loss = model_to_loss[mod]
                composition_sd = mid_sds_i[mod]
                flux_est, mid_est, ci, _,_,_ = fit(mst, loss, composition_sd)
                flux_est_grid[mod] = flux_est_grid[mod].at[i, j].set(flux_est)
                mid_est_grid[mod] = mid_est_grid[mod].at[i, j].set(mid_est)
                flux_ci_grid[mod] = flux_ci_grid[mod].at[i, j].set(ci)

    for obj, filename in (
        (flux_est_grid, "estimated_fluxes_grid.pkl"),
        (flux_ci_grid, "estimated_fluxes_ci_grid.pkl"),
        (mid_est_grid, "estimated_mids_grid.pkl"),
        (flux_est_ref, "estimated_fluxes_reference.pkl"),
        (flux_ci_ref, "estimated_fluxes_ci_reference.pkl"),
        (mid_est_ref, "estimated_mids_reference.pkl"),
        (flux_sd_est, "estimated_fluxes_sd_reference.pkl"),
    ):
        path = OUTPUT_FOLDER / filename
        print(f"Writing {path}...")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    for arr, filename in (
        (intensity_msts_grid, "intensity_measurements_grid.npy"),
        (flux_grid, "free_fluxes_grid.npy"),
        (true_mids_grid, "true_isotopologue_distributions.npy"),
        (mid_msts_grid, "mid_simulated_grid.npy"),
        (mid_msts_ref, "mid_simulated_reference.npy"),
    ):
        path = OUTPUT_FOLDER / filename
        print(f"Writing {path}...")
        jnp.save(path, arr)


if __name__ == "__main__":
    main()
