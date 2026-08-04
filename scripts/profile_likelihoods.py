import sys

import jax
import jax.numpy as jnp
import numpy as np

from cmfa.functions import ordered_ilr_func, profile_likelihood

sys.path.insert(0, "../scripts")
import pickle
from pathlib import Path

import tca_cycle
import toy_network
from jax import config

config.update("jax_enable_x64", True)

import optimistix

# M = jnp.array([[-1, -1, 2], [0, 0, -1], [-1, 0, 1]])
# N = jnp.vstack([-M, jnp.eye(3)])
# IX_INPUT_FLUX = np.array([0])

PROJECT_ROOT = Path(__file__).parent.parent

OUTPUT_FOLDER_TOY = PROJECT_ROOT / "data" / "results" / "toy_example"
OUTPUT_FOLDER_TCA = PROJECT_ROOT / "data" / "results" / "tca_cycle"


RESULTS_DIR_TOY = PROJECT_ROOT / "data" / "results" / "toy_example"
RESULTS_DIR_TCA =  PROJECT_ROOT / "data" / "results" / "tca_cycle"

key = jax.random.key(seed=1234)
key_toy, key = jax.random.split(key)
key_tca, key = jax.random.split(key)
random_idx_toy = jax.random.randint(key_toy, 1, 0, 10000)
random_idx_tca = jax.random.randint(key_tca, 1, 0, 10000)
print(f"Random index for toy model: {random_idx_toy}")
print(f"Random index for tca cycle: {random_idx_tca}")

# Toy network
true_mid_toy = np.load(RESULTS_DIR_TOY / "true_composition.npy")
true_flux_toy = np.load(RESULTS_DIR_TOY / "true_flux.npy")
simulated_mid_toy = jnp.load(RESULTS_DIR_TOY / "mid_simulated_reference.npy")

with open(RESULTS_DIR_TOY / "estimated_fluxes_reference.pkl", "rb") as f:
    estimated_fluxes_toy = pickle.load(f)

random_sim_toy = simulated_mid_toy[random_idx_toy]

sd_trad_toy = np.std(simulated_mid_toy, axis=0)
sd_ilr_toy = np.std(ordered_ilr_func(simulated_mid_toy), axis=0)

weights_trad_toy = 1/sd_trad_toy
weights_ilr_toy = 1/sd_ilr_toy

# 2. Optimization with Optimistix
init_guess_toy = jnp.array([100.0, 50.0])
solver = optimistix.BFGS(rtol=1e-8, atol=1e-8)

opt_result_trad_toy = optimistix.minimise(
    toy_network.loss_trad,
    y0=jnp.log(init_guess_toy),
    args=(random_sim_toy, weights_trad_toy),
    solver=solver,
    max_steps=10000,
    throw=False,
    has_aux=True,
)

opt_result_ilr_toy = optimistix.minimise(
    toy_network.loss_ilr,
    y0=jnp.log(init_guess_toy),
    args=(random_sim_toy, weights_ilr_toy),
    solver=solver,
    max_steps=10000,
    throw=False,
    has_aux=True,
)

lb_toy=[-7] * 2, 
ub_toy=[7] * 2,  
x_names_toy=['v3', 'v5']

result_trad_toy, ci_trad_profile_toy = profile_likelihood(toy_network.loss_trad, random_sim_toy, sd_trad_toy, np.exp(opt_result_trad_toy.value), opt_result_trad_toy.state.f_info.f, x_names_toy, lb_toy, ub_toy)
result_ilr_toy, ci_ilr_profile_toy = profile_likelihood(toy_network.loss_ilr, random_sim_toy, sd_ilr_toy, np.exp(opt_result_ilr_toy.value), opt_result_ilr_toy.state.f_info.f, x_names_toy, lb_toy, ub_toy)

trad_v3 = jnp.array([result_trad_toy.profile_result.get_profiler_result(0)["x_path"][0], result_trad_toy.profile_result.get_profiler_result(0)["ratio_path"]])
trad_v5 = jnp.array([result_trad_toy.profile_result.get_profiler_result(1)["x_path"][1], result_trad_toy.profile_result.get_profiler_result(1)["ratio_path"]])

ilr_v3 = jnp.array([result_ilr_toy.profile_result.get_profiler_result(0)["x_path"][0], result_ilr_toy.profile_result.get_profiler_result(0)["ratio_path"]])
ilr_v5 = jnp.array([result_ilr_toy.profile_result.get_profiler_result(1)["x_path"][1], result_ilr_toy.profile_result.get_profiler_result(1)["ratio_path"]])

profile_toy = {"trad": {"v3": trad_v3, "v5": trad_v5, "ci": ci_trad_profile_toy}, "ilr": {"v3": ilr_v3, "v5": ilr_v5, "ci": ci_ilr_profile_toy}}

# tca cycle
true_mid_tca = np.load(RESULTS_DIR_TCA / "true_composition.npy", allow_pickle=True).item()
true_flux_tca = np.load(RESULTS_DIR_TCA / "true_flux.npy", allow_pickle=True)
simulated_mid_tca = jnp.load(RESULTS_DIR_TCA / "mid_simulated_reference.pkl", allow_pickle=True)

with open(RESULTS_DIR_TCA / "estimated_fluxes_reference.pkl", "rb") as f:
    estimated_fluxes_tca = pickle.load(f)

random_sim_tca = jax.tree.map(lambda x: x[random_idx_tca], simulated_mid_tca)

sd_trad_tca = jax.tree.map(lambda x: np.std(x, axis=0), simulated_mid_tca)
sd_ilr_tca = jax.tree.map(lambda x: np.std(ordered_ilr_func(x), axis=0), simulated_mid_tca)

weights_trad_tca = jax.tree.map(lambda s: 1/s, sd_trad_tca)
weights_ilr_tca = jax.tree.map(lambda s: 1/s, sd_ilr_tca)

init_guess_tca = np.array([0.2, 0.2, 0.2])
solver = optimistix.BFGS(rtol=1e-8, atol=1e-8)

opt_result_trad_tca = optimistix.minimise(
    tca_cycle.loss_trad,
    y0=jnp.log(init_guess_tca),
    args=(random_sim_tca, weights_trad_tca),
    solver=solver,
    max_steps=10000,
    throw=False,
    has_aux=True,
)

opt_result_ilr_tca = optimistix.minimise(
    tca_cycle.loss_ilr,
    y0=jnp.log(init_guess_tca),
    args=(random_sim_tca, weights_ilr_tca),
    solver=solver,
    max_steps=10000,
    throw=False,
    has_aux=True,
)
lb_tca = [-7] * 3, 
ub_tca = [7] * 3,  
x_names_tca = ['v8', 'v2', "v9"]

result_trad_tca, ci_tca_profile_trad = profile_likelihood(tca_cycle.loss_trad, random_sim_tca, sd_trad_tca, np.exp(opt_result_trad_tca.value), opt_result_trad_tca.state.f_info.f, x_names_tca, lb_tca, ub_tca, use_jax_weights=True)
result_ilr_tca, ci_tca_profile_ilr = profile_likelihood(tca_cycle.loss_ilr, random_sim_tca, sd_ilr_tca, np.exp(opt_result_ilr_tca.value), opt_result_ilr_tca.state.f_info.f, x_names_tca, lb_tca, ub_tca, use_jax_weights=True)

trad_v8 = jnp.array([result_trad_tca.profile_result.get_profiler_result(0)["x_path"][0], result_trad_tca.profile_result.get_profiler_result(0)["ratio_path"]])
trad_v2 = jnp.array([result_trad_tca.profile_result.get_profiler_result(1)["x_path"][1], result_trad_tca.profile_result.get_profiler_result(1)["ratio_path"]])
trad_v9 = jnp.array([result_trad_tca.profile_result.get_profiler_result(2)["x_path"][2], result_trad_tca.profile_result.get_profiler_result(2)["ratio_path"]])

ilr_v8 = jnp.array([result_ilr_tca.profile_result.get_profiler_result(0)["x_path"][0], result_ilr_tca.profile_result.get_profiler_result(0)["ratio_path"]])
ilr_v2 = jnp.array([result_ilr_tca.profile_result.get_profiler_result(1)["x_path"][1], result_ilr_tca.profile_result.get_profiler_result(1)["ratio_path"]])
ilr_v9 = jnp.array([result_ilr_tca.profile_result.get_profiler_result(2)["x_path"][2], result_ilr_tca.profile_result.get_profiler_result(2)["ratio_path"]])

profile_tca = {"trad": {"v8": trad_v8, "v2": trad_v2, "v9": trad_v9, "ci": ci_tca_profile_trad}, "ilr": {"v8": ilr_v8, "v2": ilr_v2, "v9": ilr_v9, "ci": ci_tca_profile_ilr}}

for obj, filename, output_folder in (
    (profile_toy, "profile_likelihood.pkl", OUTPUT_FOLDER_TOY),
    (profile_tca, "profile_likelihood.pkl", OUTPUT_FOLDER_TCA),
):
    path = output_folder / filename
    print(f"Writing {path}...")
    with open(path, "wb") as f:
        pickle.dump(obj, f)