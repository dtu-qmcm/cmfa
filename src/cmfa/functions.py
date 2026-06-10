import jax
import jax.numpy as jnp
import numpy as np
from jax import config
import sympy
from scipy import stats
from scipy.spatial.distance import euclidean
from skbio.stats.composition import sbp_basis
import pypesto
from pypesto.result import OptimizerResult
import pypesto.profile as profile

config.update("jax_enable_x64", True)

def ordered_sbp_basis(n: int):
    """
    Get an orthonormal basis from an ordered sequential binary partition.

    This function generates a basis where each partition vector `i` (for `i` in
    0, ..., n-2) separates the `i`-th component from all subsequent components
    (`i+1` to `n-1`).

    This specific partition is useful when the components have a natural sequential
    order, making the isometric log-ratios ``ln(x[i] / sum(x[i+1:]))`` meaningful.
    For example, in analyzing isotopologue proportions, this basis isolates the
    ratio of the unlabeled component to the sum of all labeled components.

    Parameters
    ----------
    n : int
        The number of components in the composition (dimension of the space).

    Returns
    -------
    numpy.ndarray
        A `(n-1, n)` matrix representing the orthonormal basis derived from the SBP.
    """

    upper_triangle = np.triu(np.full((n - 1, n), -1), k=1)
    partition = upper_triangle + np.eye(n - 1, n)
    return sbp_basis(partition)


def ordered_ilr_func(mat):
    k = mat.shape[-1]
    basis = ordered_sbp_basis(k)
    return clr_func(mat) @ basis.T


def clr_func(mat):
    lmat = jnp.log(mat)
    k = jnp.mean(lmat, axis=-1, keepdims=True)
    mat_com = lmat - k
    return mat_com


def clr_inv_func(mat_com):
    emat = jnp.exp(mat_com)
    k = jnp.sum(emat, axis=-1, keepdims=True)
    mat = emat / k
    return mat


# closure, _gram_schmidt_basis, _check_orthonality and ilr_func are from skbio.stats.composition and modified to work with jax and otimistix.
def closure(mat):
    mat2d = jnp.atleast_2d(mat)
    mat_sum = jnp.sum(mat2d, axis=1, keepdims=True)
    mat = mat2d / mat_sum
    return jnp.squeeze(mat)


def _gram_schmidt_basis(n):
    basis = jnp.zeros((n, n - 1))
    for j in range(n - 1):
        i = j + 1
        e = jnp.array([(1 / i)] * i + [-1] + [0] * (n - i - 1)) * jnp.sqrt(
            i / (i + 1)
        )
        basis = basis.at[:, j].set(e)
    return basis.T


def _check_orthogonality(basis):
    basis = jnp.atleast_2d(basis)
    if not jnp.allclose(
        basis @ basis.T, jnp.identity(len(basis)), rtol=1e-4, atol=1e-6
    ):
        raise ValueError("Basis is not orthonormal.")


def ilr_func(mat, basis=None, check=True):
    mat = closure(mat)
    if basis is None:
        d = mat.shape[-1]
        basis = _gram_schmidt_basis(d)  # dimension (d-1) x d
    else:
        if len(basis.shape) != 2:
            raise ValueError(
                "Basis needs to be a 2D matrix, "
                "not a %dD matrix." % (len(basis.shape))
            )
        if check:
            _check_orthogonality(basis)

    return clr_func(mat) @ basis.T


def aitchison(x, y):
    return euclidean(clr_func(x), clr_func(y))


def calculate_ci(sigma_v, log_fluxes, confidence_level=0.95):
    """
    Parameters
    ----------
    sigma_v: Covariance matrix for all fluxes
    log_fluxes: Vector of all log(flux)
    confidence_level: The confidence level of the confidence interval. Default 0.95

    Returns
    -------
    ci_range: array
        A `(2, n)` matrix of the cconfidense interval for the free fluxes.
    """
    if type(sigma_v) == dict:
        diag_elements = jnp.array([sigma_v[k][k] for k in sigma_v.keys()])
        standard_errors = jnp.sqrt(diag_elements)
        params = jnp.array([log_fluxes[k] for k in sigma_v.keys()])
    else:
        cov_matrix = jnp.asarray(sigma_v)
        standard_errors = jnp.sqrt(jnp.diag(cov_matrix))
        params = jnp.asarray(log_fluxes)

    alpha = 1 - confidence_level
    z = stats.norm.ppf(1 - alpha / 2)
    lower_bounds = jnp.exp(params - z * standard_errors)
    upper_bounds = jnp.exp(params + z * standard_errors)

    ci_range = jnp.stack([lower_bounds, upper_bounds], axis=1)

    return ci_range

def profile_likelihood(
    loss_function,
    data,
    data_sd,
    opt_flux,
    opt_fval,
    x_names,
    lb,
    ub,
    scipy_method="L-BFGS-B",
    use_jax_weights=False,
):
    """
    Generic profile likelihood wrapper for pyPESTO.

    Parameters
    ----------
    loss_function : callable
        Loss function returning (loss, aux).
    data : any
        Observations / MID data passed to the loss function.
    data_sd : any
        Standard deviations for weighting.
    opt_flux : array-like
        Optimized flux values (linear scale).
    opt_fval : float
        Objective value at optimum.
    x_names : list[str]
        Parameter names.
    lb : list[float]
        Lower bounds.
    ub : list[float]
        Upper bounds.
    scipy_method : str
        Scipy optimizer method.
    use_jax_weights : bool
        If True, compute weights using jax.tree.map.
    """

    # Setup weights
    if use_jax_weights:
        weights = jax.tree.map(lambda s: 1 / s, data_sd)
    else:
        weights = 1 / data_sd

    # pyPESTO objective
    def my_loss_function(x):
        return loss_function(x, args=(data, weights))[0]

    objective = pypesto.Objective(fun=my_loss_function)

    # Define problem
    problem = pypesto.Problem(
        objective=objective,
        lb=lb,
        ub=ub,
        x_names=x_names,
    )

    # Inject optimizer result
    result = pypesto.Result(problem=problem)

    opt_result_manual = OptimizerResult(
        x=np.array(np.log(opt_flux)),
        fval=float(np.array(opt_fval)),
        id="optimistix_bfgs",
    )

    result.optimize_result.append(opt_result_manual)

    # Run profile likelihood
    result = profile.parameter_profile(
        problem=problem,
        result=result,
        optimizer=pypesto.optimize.ScipyOptimizer(method=scipy_method),
    )

    # Convert profile paths back from log-space
    for i in range(len(x_names)):
        result.profile_result.get_profiler_result(i)["x_path"][i] = np.exp(
            result.profile_result.get_profiler_result(i)["x_path"][i]
        )


    # Confidence intervals
    confidence_ratio = pypesto.profile.chi2_quantile_to_ratio(
        alpha=0.95,
        df=1,
    )

    ci_profile = []

    for i in range(len(x_names)):
        profiler_result = result.profile_result.get_profiler_result(i)

        ci = pypesto.profile.calculate_approximate_ci(
            profiler_result["x_path"][i],
            profiler_result["ratio_path"],
            confidence_ratio,
        )

        ci_profile.append(ci)

    ci_profile = np.array(ci_profile)

    return result, ci_profile