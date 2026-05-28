import pdb
import pickle
from itertools import product
from pathlib import Path
from functools import partial

from scipy.stats import entropy

import jax
import jax.numpy as jnp
import numpy as np
import optimistix
from tqdm import tqdm

from cmfa.emu_example import simple_net

from cmfa.opti_functions import (
    clr_func,
    ordered_ilr_func,
    closure,
    calculate_ci,
    profile_likelihood,
)

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
SIMULATE_COMPOSITION = partial(simple_net, v1=KNOWN_FLUXES[0].item())
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


def loss_clr(log_free_flux, args):
    composition = SIMULATE_COMPOSITION(jnp.exp(log_free_flux))
    target, weights = args
    return wssr(clr_func(composition), (clr_func(target), weights)), composition


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
        return {model: jnp.empty(shape) for model in ("trad", "clr", "ilr")}

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
        id_est = result.aux
        fval = float(result.state.f_info.f)
        free_flux_est = jnp.exp(log_free_flux_est)
        flux_est = expand_free_fluxes(free_flux_est)[corrected_index]
        inv_hessian = result.state.f_info.hessian_inv.pytree
        sigma_v = get_sigma_v(N, inv_hessian)
        flux_sd_est = jnp.sqrt(jnp.diag(sigma_v))
        ci = calculate_ci(sigma_v, jnp.log(flux_est))
        return flux_est, id_est, ci, inv_hessian, fval, flux_sd_est

    model_to_loss = {"trad": loss_trad, "clr": loss_clr, "ilr": loss_ilr}

    solver = optimistix.BFGS(rtol=1e-8, atol=1e-8)
    true_flux = expand_free_fluxes(TRUE_FREE_FLUXES)
    true_id = SIMULATE_COMPOSITION(TRUE_FREE_FLUXES)
    n_flux = len(true_flux)
    n_comp = len(true_id)
    print(f"True flux: {true_flux}")
    print(f"True composition: {true_id}")
    print(f"Writing {TRUE_FLUX_FILE}...")
    np.save(TRUE_FLUX_FILE, true_flux)
    print(f"Writing {TRUE_COMPOSITION_FILE}...")
    np.save(TRUE_COMPOSITION_FILE, true_id)
    flux_grid = get_flux_grid(FREE_FLUX_RANGE, FREE_FLUX_GRID_SIZE)
    n_grid = len(flux_grid)
    n_free = len(IX_FREE_FLUXES)
    key = jax.random.key(seed=SEED)

    # initialise containers
    flux_est_ref = empties((N_MST_REF, n_flux))
    flux_ci_ref = empties((N_MST_REF, n_flux, 2))
    id_est_ref = empties((N_MST_REF, n_comp))
    fval_ref = empties((N_MST_REF))
    flux_sd_est = empties((N_MST_REF, n_flux))
    ci_fisher = empties(())
    ci_profile = empties(())
    flux_est_grid = empties((n_grid, M_MST_PER_GRID_CELL, n_flux))
    flux_ci_grid = empties((n_grid, M_MST_PER_GRID_CELL, n_flux, 2))
    hessian_grid = empties((n_grid, M_MST_PER_GRID_CELL, n_free, n_free))
    id_est_grid = empties((n_grid, M_MST_PER_GRID_CELL, n_comp))
    intensity_msts_grid = jnp.empty((n_grid, M_MST_PER_GRID_CELL, n_comp))
    id_msts_grid = jnp.empty((n_grid, M_MST_PER_GRID_CELL, n_comp))
    true_ids_grid = jnp.empty((n_grid, n_comp))
    entropies_grid = jnp.empty((n_grid, 1))

    # get keys
    key_ref, key_grid = jax.random.split(key)

    # Fit the reference flux N_MEASUREMENT_REF times
    corrected_index = jnp.array([1, 3, 5, 0, 2, 4])
    keys_ref = jax.random.split(key_ref, N_MST_REF)
    intensity_msts_ref = simulate_measurements(keys_ref, true_id, SIGMA)
    id_msts_ref = closure(intensity_msts_ref)
    id_sds_ref = {
        "trad": np.std(id_msts_ref, axis=0),
        "clr": np.std(clr_func(id_msts_ref), axis=0),
        "ilr": np.std(ordered_ilr_func(id_msts_ref), axis=0),
    }
    for mod in ("trad", "clr", "ilr"):
        _,_,ci,_,_,_ = fit(true_id, model_to_loss[mod], id_sds_ref[mod])
        ci_fisher[mod] = ci
    for mod in ("trad", "clr", "ilr"):
        flux_est,_,_,_,fval,_ = fit(true_id, model_to_loss[mod], id_sds_ref[mod])
        flux_est_free = flux_est[np.array([4,5])]
        x_names=["v3", "v5"]
        lb=[-6]*2
        ub=[6]*2
        _, ci_pro = profile_likelihood(model_to_loss[mod],
                                   true_id,
                                   id_sds_ref[mod],
                                   flux_est_free,
                                   fval,
                                   x_names,
                                   lb,
                                   ub,
                                   )
        pdb.set_trace()

    for j, mst in enumerate(tqdm(id_msts_ref)):
        for mod in ("trad", "clr", "ilr"):
            loss = model_to_loss[mod]
            composition_sd = id_sds_ref[mod]
            flux_est, id_est, ci, _, fval, flux_sd = fit(mst, loss, composition_sd)
            flux_est_ref[mod] = flux_est_ref[mod].at[j].set(flux_est)
            id_est_ref[mod] = id_est_ref[mod].at[j].set(id_est)
            flux_ci_ref[mod] = flux_ci_ref[mod].at[j].set(ci)
            fval_ref[mod] = fval_ref[mod].at[j].set(fval)
            flux_sd_est[mod] = flux_sd_est[mod].at[j].set(flux_sd)

    # fit each free flux combination in the grid N_MST_PER_FLUX times
    keys_grid = jax.random.split(key_grid, n_grid)
    for (i, free_flux), key_i in zip(enumerate(tqdm(flux_grid)), keys_grid):
        keys_i = jax.random.split(key_i, M_MST_PER_GRID_CELL)
        true_id_i = SIMULATE_COMPOSITION(free_flux)
        intensity_msts_i = simulate_measurements(keys_i, true_id_i, SIGMA)
        id_msts_i = closure(intensity_msts_i)
        id_msts_grid = id_msts_grid.at[i].set(id_msts_i)
        intensity_msts_grid = intensity_msts_grid.at[i].set(intensity_msts_i)
        true_ids_grid = true_ids_grid.at[i].set(true_id_i)
        entropies_grid = entropies_grid.at[i].set(entropy(pk=true_id_i))
        id_sds_i = {
            "trad": np.std(id_msts_i, axis=0),
            "clr": np.std(clr_func(id_msts_i), axis=0),
            "ilr": np.std(ordered_ilr_func(id_msts_i), axis=0),
        }
        for j, mst in enumerate(id_msts_i):
            for mod in ("trad", "clr", "ilr"):
                loss = model_to_loss[mod]
                composition_sd = id_sds_i[mod]
                flux_est, id_est, ci, inv_hess,_,_ = fit(mst, loss, composition_sd)
                flux_est_grid[mod] = flux_est_grid[mod].at[i, j].set(flux_est)
                id_est_grid[mod] = id_est_grid[mod].at[i, j].set(id_est)
                flux_ci_grid[mod] = flux_ci_grid[mod].at[i, j].set(ci)
                hessian_grid[mod] = hessian_grid[mod].at[i, j].set(inv_hess)

    for obj, filename in (
        (flux_est_grid, "estimated_fluxes.pkl"),
        (flux_ci_grid, "estimated_fluxes_ci.pkl"),
        (hessian_grid, "hessian_matrices.pkl"),
        (id_est_grid, "estimated_ids.pkl"),
        (flux_est_ref, "estimated_fluxes_reference.pkl"),
        (flux_ci_ref, "estimated_fluxes_ci_reference.pkl"),
        (id_est_ref, "estimated_ids_reference.pkl"),
        (fval_ref, "fvalues_reference.pkl"),
        (flux_sd_est, "estimated_fluxes_sd_reference.pkl"),
        (ci_fisher, "fisher_estimated_ci.pkl"),
    ):
        path = OUTPUT_FOLDER / filename
        print(f"Writing {path}...")
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    for arr, filename in (
        (intensity_msts_grid, "intensity_measurements.npy"),
        (flux_grid, "free_fluxes.npy"),
        (true_ids_grid, "true_isotopologue_distributions.npy"),
        (entropies_grid, "entropies.npy"),
        (id_msts_ref, "id_simulated_reference.npy"),
        (id_msts_grid, "id_simulated.npy"),
    ):
        path = OUTPUT_FOLDER / filename
        print(f"Writing {path}...")
        jnp.save(path, arr)


if __name__ == "__main__":
    main()
