import numpy as np
import jax.numpy as jnp
from functools import partial
import pypesto
import scipy.stats as stats
import optimistix
from pypesto.result import OptimizerResult
import pypesto.profile as profile
import pickle
from pathlib import Path
from cmfa.opti_functions import ordered_ilr_func, profile_likelihood

import tca_cycle

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "tca_cycle"

true_mid = np.load(RESULTS_DIR / "true_composition.npy")
true_flux = np.load(RESULTS_DIR / "true_flux.npy")
simulated_mid = jnp.load(RESULTS_DIR / "id_simulated_reference.npy")


def fisher_ci_toy(opt_result):
    inv_hessian = opt_result.state.f_info.hessian_inv.pytree
    standard_errors = jnp.sqrt(jnp.diag(inv_hessian))
    params = jnp.asarray(opt_result.value)

    alpha = 1 - 0.95
    z = stats.norm.ppf(1 - alpha / 2)
    lower_bounds = jnp.exp(params - z * standard_errors)
    upper_bounds = jnp.exp(params + z * standard_errors)

    ci_range_fisher = jnp.stack([lower_bounds, upper_bounds], axis=1)
    return ci_range_fisher

sd_tree_trad = jax.tree.map(lambda x: np.std(x, axis=0), simulated_mid)
sd_tree_ilr = jax.tree.map(lambda x: np.std(ordered_ilr_func(x), axis=0), simulated_mid)

weights_trad = 1/sd_trad
weights_ilr = 1/sd_ilr

# 2. Optimization with Optimistix
# Note: Ensure loss_trad signature is: loss_trad(params, args)
init_guess = jnp.array([100.0, 50.0])
solver = optimistix.BFGS(rtol=1e-8, atol=1e-8)

opt_result_trad = optimistix.minimise(
    tca_cycle.loss_trad,
    y0=jnp.log(init_guess),
    args=(mst_j, jax.tree.map(lambda s: 1/s, sd_tree)),
    solver=solver,
    max_steps=10000,
    throw=False,
    has_aux=True,
)

opt_result_trad = optimistix.minimise(
    toy_network.loss_trad,
    y0=jnp.log(init_guess),
    args=(true_mid, weights_trad),
    solver=solver,
    max_steps=10000,
    throw=False,
    has_aux=True,
)

opt_result_ilr = optimistix.minimise(
    toy_network.loss_ilr,
    y0=jnp.log(init_guess),
    args=(true_mid, weights_ilr),
    solver=solver,
    max_steps=10000,
    throw=False,
    has_aux=True,
)

ci_trad_fisher = fisher_ci_toy(opt_result_trad)

ci_ilr_fisher = fisher_ci_toy(opt_result_ilr)

print(f"Fisher trad {ci_trad_fisher}")
print(f"Fisher ilr {ci_ilr_fisher}")

ci_trad_plog = profile_likelihood_toy(toy_network.loss_trad, weights_trad, np.exp(opt_result_trad.value), opt_result_trad.state.f_info.f, observation=true_mid)
ci_ilr_plog = profile_likelihood_toy(toy_network.loss_ilr, weights_ilr, np.exp(opt_result_ilr.value), opt_result_ilr.state.f_info.f, observation=true_mid)

print(f"pro-log trad {ci_trad_plog}")
print(f"pro-log ilr {ci_ilr_plog}")
