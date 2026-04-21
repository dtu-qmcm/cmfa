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

from cmfa.emu_example import clr_func, simple_net, ordered_ilr_func, closure

jax.config.update("jax_enable_x64", True)

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "results" / "toy_example"
M = jnp.array([[-1, -1, 2], [0, 0, -1], [-1, 0, 1]])
KNOWN_FLUXES = jnp.array([100.0])
TRUE_FREE_FLUXES = jnp.array([50.0, 20.0])
FREE_FLUX_RANGE = [[25.0, 75.0], [10.0, 30.0]]
FREE_FLUX_GRID_SIZE = [11, 11]
IX_INPUT_FLUX = np.array([0])
IX_FREE_FLUXES = np.array([2, 4])
IX_FIXED_FLUXES = np.array([1, 3, 5])
TOTAL_MOL = 1e5
SIGMA = 0.1
N_MEASUREMENT_PER_FLUX = 50
INITIAL_GUESS = jnp.array([100, 50])
SIMULATE_COMPOSITION = partial(simple_net, v1=KNOWN_FLUXES[0].item())
SEED = 1234


def get_flux_vector(free, known, ix_known, ix_free, M):
    n = known.shape[0] + free.shape[0] + M.shape[0]
    ix_known_or_free = jnp.concat([ix_free, ix_known])
    ix_fixed = jnp.delete(jnp.arange(n), ix_known_or_free)
    out = jnp.zeros(shape=(n,)).at[ix_known].set(known).at[ix_free].set(free)
    fixed = -M @ out[jnp.concat([ix_known, ix_free])]
    return out.at[ix_fixed].set(fixed)


def simulate_measurement(key, yhat, sigma):
    err = jax.random.normal(key, shape=(yhat.shape)) * sigma
    log_mst = jnp.log(yhat) + err
    return jnp.exp(log_mst)


simulate_measurements = jax.vmap(simulate_measurement, in_axes=(0, None, None))


def wssr(y, args):
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


def main():
    flux_from_free = partial(
        get_flux_vector,
        known=KNOWN_FLUXES,
        ix_known=IX_INPUT_FLUX,
        ix_free=IX_FREE_FLUXES,
        M=M,
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
    key = jax.random.key(seed=SEED)
    flux_estimates = {
        model: jnp.empty((len(free_fluxes), N_MEASUREMENT_PER_FLUX, n_flux))
        for model in ("trad", "clr", "ilr")
    }
    id_estimates = {
        model: jnp.empty((len(free_fluxes), N_MEASUREMENT_PER_FLUX, n_comp))
        for model in ("trad", "clr", "ilr")
    }
    intensity_msts = jnp.empty(
        (len(free_fluxes), N_MEASUREMENT_PER_FLUX, n_comp)
    )
    true_ids = jnp.empty((free_fluxes.shape[0], n_comp))
    for i, free_flux in enumerate(tqdm(free_fluxes)):
        main_key, mc_key = jax.random.split(key)
        keys = jax.random.split(main_key, N_MEASUREMENT_PER_FLUX)
        true_ids = true_ids.at[i].set(SIMULATE_COMPOSITION(free_flux))
        intensity_msts_i = simulate_measurements(keys, true_ids[i], SIGMA)
        intensity_msts = intensity_msts.at[i].set(intensity_msts_i)
        id_msts_i = closure(intensity_msts_i)
        id_sd_trad = np.std(id_msts_i, axis=0)
        id_sd_clr = np.std(clr_func(id_msts_i), axis=0)
        id_sd_ilr = np.std(ordered_ilr_func(id_msts_i), axis=0)
        for j, mst in enumerate(id_msts_i):
            for m, loss, composition_sd in (
                ("trad", loss_trad, id_sd_trad),
                ("clr", loss_clr, id_sd_clr),
                ("ilr", loss_ilr, id_sd_ilr),
            ):
                result = optimistix.minimise(
                    loss,
                    y0=jnp.log(INITIAL_GUESS),
                    args=(mst, 1 / composition_sd),
                    solver=solver,
                    max_steps=10000,
                    throw=False,
                    has_aux=True,
                )
                flux_est = flux_from_free(jnp.exp(result.value))
                id_est = result.aux
                flux_estimates[m] = flux_estimates[m].at[i, j].set(flux_est)
                id_estimates[m] = id_estimates[m].at[i, j].set(id_est)
        if jnp.isclose(free_flux, TRUE_FREE_FLUXES).all():
            N_MONTE_CARLO = 10000
            mc_keys = jax.random.split(mc_key, N_MONTE_CARLO)
            mc_msts = simulate_measurements(mc_keys, true_ids[i], SIGMA)
            mc_id_msts = closure(mc_msts)
            mc_flux_estimates = {
                m: jnp.empty((N_MONTE_CARLO, n_flux))
                for m in ("trad", "clr", "ilr")
            }
            mc_id_estimates = {
                m: jnp.empty((N_MONTE_CARLO, n_comp))
                for m in ("trad", "clr", "ilr")
            }

            for mc_i, mc_id_mst in enumerate(mc_id_msts):
                for m, loss, composition_sd in (
                    ("trad", loss_trad, id_sd_trad),
                    ("clr", loss_clr, id_sd_clr),
                    ("ilr", loss_ilr, id_sd_ilr),
                ):
                    result = optimistix.minimise(
                        loss,
                        y0=jnp.log(INITIAL_GUESS),
                        args=(mc_id_mst, 1 / composition_sd),
                        solver=solver,
                        max_steps=10000,
                        throw=False,
                        has_aux=True,
                    )
                    flux_est = flux_from_free(jnp.exp(result.value))
                    id_est = result.aux
                    mc_flux_estimates[m] = (
                        mc_flux_estimates[m].at[mc_i].set(flux_est)
                    )
                    mc_id_estimates[m] = mc_id_estimates[m].at[mc_i].set(id_est)
    mc_flux_path = OUTPUT_FOLDER / "monte_carlo_estimated_fluxes.pkl"
    print(f"Writing {mc_flux_path}...")
    with open(mc_flux_path, "wb") as f:
        pickle.dump(mc_flux_estimates, f)
    mc_id_path = OUTPUT_FOLDER / "monte_carlo_estimated_ids.pkl"
    print(f"Writing {mc_id_path}...")
    with open(mc_id_path, "wb") as f:
        pickle.dump(mc_id_estimates, f)
    flux_estimate_path = OUTPUT_FOLDER / "estimated_fluxes.pkl"
    print(f"Writing {flux_estimate_path}...")
    with open(flux_estimate_path, "wb") as f:
        pickle.dump(flux_estimates, f)
    id_estimate_path = OUTPUT_FOLDER / "estimated_ids.pkl"
    print(f"Writing {id_estimate_path}...")
    with open(id_estimate_path, "wb") as f:
        pickle.dump(id_estimates, f)
    intensity_msts_path = OUTPUT_FOLDER / "intensity_measurements.npy"
    print(f"Writing {intensity_msts_path}...")
    jnp.save(intensity_msts_path, intensity_msts)
    free_flux_path = OUTPUT_FOLDER / "free_fluxes.npy"
    print(f"Writing {free_flux_path}...")
    jnp.save(free_flux_path, free_fluxes)
    true_ids_path = OUTPUT_FOLDER / "true_isotopologue_distributions.npy"
    print(f"Writing {true_ids_path}...")
    jnp.save(true_ids_path, true_ids)


if __name__ == "__main__":
    main()
