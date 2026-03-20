from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optimistix
from skbio.stats.composition import closure
from tqdm import tqdm

from cmfa.emu_example import clr_func, simple_net, ordered_ilr_func

jax.config.update("jax_enable_x64", True)

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "results" / "toy_example"
M = jnp.array([[-1, -1, 2], [0, 0, -1], [-1, 0, 1]])
KNOWN_FLUXES = jnp.array([100.0])
FREE_FLUXES = jnp.array([50.0, 20.0])
IX_INPUT_FLUX = np.array([0])
IX_FREE_FLUXES = np.array([2, 4])
IX_FIXED_FLUXES = np.array([1, 3, 5])
TOTAL_MOL = 1e5
INITIAL_GUESS = jnp.array([100, 50])
SIMULATE_COMPOSITION = partial(simple_net, v1=KNOWN_FLUXES[0].item())


def get_flux_vector(free, known, ix_known, ix_free, M):
    n = known.shape[0] + free.shape[0] + M.shape[0]
    ix_known_or_free = jnp.concat([ix_free, ix_known])
    ix_fixed = jnp.delete(jnp.arange(n), ix_known_or_free)
    out = jnp.zeros(shape=(n,)).at[ix_known].set(known).at[ix_free].set(free)
    fixed = -M @ out[jnp.concat([ix_known, ix_free])]
    return out.at[ix_fixed].set(fixed)


def simulate_intensity_measurements(
    isotopologue_proportions,
    sigma,
    n=1000,
    total_mol=1e5,
    seed=1234,
    distribution="normal",
):
    k = isotopologue_proportions.shape[0]
    rng = np.random.default_rng(seed=seed)
    yhat = total_mol * isotopologue_proportions
    match distribution:
        case "normal":
            return yhat + rng.normal(0, sigma, size=(n, k))
        case "lognormal":
            return np.exp(np.log(yhat) + rng.normal(0, sigma, size=(n, k)))
        case _:
            raise ValueError('distribution must be "normal" or "lognormal"')


def wssr(y, args):
    target, weights = args
    return jnp.sum(weights * (y - target) ** 2)


def loss_euclidean(log_free_flux, args):
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
    true_flux = flux_from_free(FREE_FLUXES)
    true_composition = SIMULATE_COMPOSITION(FREE_FLUXES)
    true_flux_file = OUTPUT_FOLDER / "true_flux.npy"
    true_composition_file = OUTPUT_FOLDER / "true_composition.npy"
    print(f"True flux: {true_flux}")
    print(f"True composition: {true_composition}")
    print(f"Writing {true_flux_file}...")
    np.save(true_flux_file, true_flux)
    print(f"Writing {true_composition_file}...")
    np.save(true_composition_file, true_composition)
    sigma = 0.1
    intensity_measurements = simulate_intensity_measurements(
        true_composition,
        sigma=sigma,
        distribution="lognormal",
        total_mol=TOTAL_MOL,
    )
    intensity_msts_path = OUTPUT_FOLDER / "intensity_measurements.npy"
    print(f"Writing {intensity_msts_path}...")
    jnp.save(intensity_msts_path, intensity_measurements)
    composition_measurements = closure(intensity_measurements)
    composition_sd = np.std(composition_measurements, axis=0)  # type: ignore
    clr_composition_sd = np.std(clr_func(composition_measurements), axis=0)
    ilr_composition_sd = np.std(
        ordered_ilr_func(composition_measurements), axis=0
    )
    print(f"Standard deviation of simulated compositions:\n\t{composition_sd}")
    print(
        f"Standard deviation of clr transformed simulated compositions:\n\t{clr_composition_sd}"
    )
    estimated_fluxes_euclidean = []
    estimated_fluxes_clr = []
    estimated_fluxes_ilr = []
    estimated_ids_euclidean = []
    estimated_ids_clr = []
    estimated_ids_ilr = []
    for mst in tqdm(composition_measurements):
        result_euclidean = optimistix.minimise(
            loss_euclidean,
            y0=jnp.log(INITIAL_GUESS),
            args=(mst, 1 / composition_sd),
            solver=solver,
            max_steps=10000,
            throw=False,
            has_aux=True,
        )
        result_clr = optimistix.minimise(
            loss_clr,
            y0=jnp.log(INITIAL_GUESS),
            args=(mst, 1 / clr_composition_sd),
            solver=solver,
            max_steps=10000,
            throw=False,
            has_aux=True,
        )
        result_ilr = optimistix.minimise(
            loss_ilr,
            y0=jnp.log(INITIAL_GUESS),
            args=(mst, 1 / ilr_composition_sd),
            solver=solver,
            max_steps=10000,
            throw=False,
            has_aux=True,
        )
        estimated_fluxes_euclidean.append(
            flux_from_free(jnp.exp(result_euclidean.value))
        )
        estimated_ids_euclidean.append(result_euclidean.aux)
        estimated_fluxes_clr.append(flux_from_free(jnp.exp(result_clr.value)))
        estimated_ids_clr.append(result_clr.aux)
        estimated_fluxes_ilr.append(flux_from_free(jnp.exp(result_ilr.value)))
        estimated_ids_ilr.append(result_ilr.aux)
    estimated_fluxes_euclidean = jnp.array(estimated_fluxes_euclidean)
    estimated_ids_euclidean = jnp.array(estimated_ids_euclidean)
    estimated_fluxes_clr = jnp.array(estimated_fluxes_clr)
    estimated_ids_clr = jnp.array(estimated_ids_clr)
    estimated_fluxes_ilr = jnp.array(estimated_fluxes_ilr)
    estimated_ids_ilr = jnp.array(estimated_ids_ilr)
    for filename, arr in (
        ("estimated_fluxes_euclidean.npy", estimated_fluxes_euclidean),
        ("estimated_ids_euclidean.npy", estimated_ids_euclidean),
        ("estimated_fluxes_clr.npy", estimated_fluxes_clr),
        ("estimated_ids_clr.npy", estimated_ids_clr),
        ("estimated_fluxes_ilr.npy", estimated_fluxes_ilr),
        ("estimated_ids_ilr.npy", estimated_ids_ilr),
    ):
        path = OUTPUT_FOLDER / filename
        print(f"Writing {path}...")
        jnp.save(path, arr)


if __name__ == "__main__":
    main()
