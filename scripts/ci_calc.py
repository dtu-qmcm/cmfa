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

import toy_network
# M = jnp.array([[-1, -1, 2], [0, 0, -1], [-1, 0, 1]])
# N = jnp.vstack([-M, jnp.eye(3)])
# IX_INPUT_FLUX = np.array([0])

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results" / "toy_example"

true_mid = np.load(RESULTS_DIR / "true_composition.npy")
true_flux = np.load(RESULTS_DIR / "true_flux.npy")
simulated_mid = jnp.load(RESULTS_DIR / "id_simulated_reference.npy")

def profile_likelihood_toy(loss_function, weights, opt_flux, opt_fval, observation=true_mid):
    def my_loss_function(x): 
        return loss_function(x, args=(observation, weights))[0] 

    objective = pypesto.Objective(fun=my_loss_function)

    # 4. Define Problem
    problem = pypesto.Problem(
        objective=objective,
        lb=[-6] * 2, 
        ub=[6] * 2,  
        x_names=['v3', 'v5']
    )

    result = pypesto.Result(problem=problem)
    opt_result_manual = OptimizerResult(
        x=np.array(np.log(opt_flux)),        # Convert JAX to NumPy
        fval=float(np.array(opt_fval)),          # Cast JAX scalar to float
        id="optimistix_bfgs",
    )

    result.optimize_result.append(opt_result_manual)

    result = profile.parameter_profile(
        problem=problem,
        result=result,
        optimizer=pypesto.optimize.ScipyOptimizer(method="L-BFGS-B"),
    )

    result.profile_result.get_profiler_result(0)["x_path"][0] = np.exp(result.profile_result.get_profiler_result(0)["x_path"][0])
    result.profile_result.get_profiler_result(1)["x_path"][1] = np.exp(result.profile_result.get_profiler_result(1)["x_path"][1])

    confidence_ratio = pypesto.profile.chi2_quantile_to_ratio(alpha=0.95, df=1)
    ci_profile_01 = pypesto.profile.calculate_approximate_ci(result.profile_result.get_profiler_result(0)["x_path"][0], result.profile_result.get_profiler_result(0)["ratio_path"], confidence_ratio)
    ci_profile_02 = pypesto.profile.calculate_approximate_ci(result.profile_result.get_profiler_result(1)["x_path"][1], result.profile_result.get_profiler_result(1)["ratio_path"], confidence_ratio)
    ci_profile = np.array([ci_profile_01, ci_profile_02])
    return ci_profile

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



sd_trad = np.std(simulated_mid, axis=0)
sd_ilr = np.std(ordered_ilr_func(simulated_mid), axis=0)

weights_trad = 1/sd_trad
weights_ilr = 1/sd_ilr

# 2. Optimization with Optimistix
# Note: Ensure loss_trad signature is: loss_trad(params, args)
init_guess = jnp.array([100.0, 50.0])
solver = optimistix.BFGS(rtol=1e-8, atol=1e-8)

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

# ci_trad_plog = profile_likelihood_toy(toy_network.loss_trad, weights_trad, np.exp(opt_result_trad.value), opt_result_trad.state.f_info.f, observation=true_mid)
# ci_ilr_plog = profile_likelihood_toy(toy_network.loss_ilr, weights_ilr, np.exp(opt_result_ilr.value), opt_result_ilr.state.f_info.f, observation=true_mid)

lb=[-6] * 2, 
ub=[6] * 2,  
x_names=['v3', 'v5']

_, ci_trad_prolog = profile_likelihood(toy_network.loss_trad, true_mid, sd_trad, np.exp(opt_result_trad.value), opt_result_trad.state.f_info.f, x_names, lb, ub)

print(f"pro-log trad {ci_trad_prolog}")
# print(f"pro-log ilr {ci_ilr_prolog}")
