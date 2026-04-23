import pdb
import jax
import jax.numpy as jnp
import numpy as np
from jax import config
import sympy
from scipy import stats
from scipy.spatial.distance import euclidean
from skbio.stats.composition import sbp_basis

config.update("jax_enable_x64", True)


def simple_net(parameters: jax.Array, v1: float) -> jax.Array:
    """
    Takes vector of fluxes and return the MID for F123 in a simple metabolic network

    :param parameters: Fluxes to be estimated
    :param v1: Input flux
    :return f123: MID for F123
    """

    v3, v5 = parameters
    # v1 is the input flux and is fixed
    v1 = v1

    # Stoicometric matrix. Rows=(b,c,d), columns = (v2,v4,v6,v1,v3,v5)
    S = np.array([
        [-1, -1, 0, 1, 1, -1],
        [0, 1, 0, 0, 0, -1],
        [1, 0, -1, 0, -1, 1],
    ], dtype=float)

    S_rref, dep_index = sympy.Matrix(S).rref()

    free_index = jnp.array([i for i in range(6) if i not in dep_index])

    M = jnp.array(np.array(S_rref, dtype=float))[:,free_index]
    
    v_free = jnp.array([v1, v3, v5])
    # The fluxes calcualted from M and the free fluxes
    v_full = jnp.vstack([-M, jnp.eye(3)]) @ v_free

    v2, v4, v6 = v_full[jnp.array(dep_index)]

    A1 = jnp.array(
        [
            [-v4, v4, 0, 0, 0],
            [0, -v1 - v3, v3, 0, 0],
            [0, v2, -v2 - v5, v5, 0],
            [0, 0, 0, -v1 - v3, v3],
            [v5, 0, 0, v2, -v2 - v5],
        ],
        dtype=float,
    )

    A2 = jnp.array(
        [
            [-v5 - v2, v2],
            [v3, -v1 - v3],
        ]
    )

    A3 = jnp.array([[-v6, v6, 0], [0, -v5 - v2, v2], [0, v3, -v1 - v3]])

    B1 = jnp.array(
        [
            [0, 0],
            [-v1, 0],
            [0, 0],
            [0, -v1],
            [0, 0],
        ]
    )

    B2 = jnp.array(
        [
            [-v5, 0],
            [0, -v1],
        ]
    )

    B3 = jnp.array(
        [
            [0, 0],
            [-v5, 0],
            [0, -v1],
        ]
    )

    a2 = jnp.array([0, 1])
    a3 = jnp.array([1, 0])
    a23 = jnp.array([0, 1, 0])
    a123 = jnp.array([0, 1, 0, 0])

    Y1 = jnp.array([a2, a3])
    X1 = jnp.linalg.solve(A1, B1 @ Y1)

    c1 = X1[0]
    b2 = X1[1]
    d2 = X1[2]
    b3 = X1[3]
    d3 = X1[4]
    b3c1 = jnp.convolve(b3, c1)

    Y2 = jnp.array([b3c1, a23])
    X2 = jnp.linalg.solve(A2, B2 @ Y2)

    d23 = X2[0]
    b23 = X2[1]
    b23c1 = jnp.convolve(b23, c1)

    Y3 = jnp.array([b23c1, a123])
    X3 = jnp.linalg.solve(A3, B3 @ Y3)

    f123 = X3[0]
    d123 = X3[1]
    b123 = X3[2]

    return f123


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


# def objective_wssr(params, args):
#     if jnp.any(params < 0):
#         out = 1e5
#     else:
#         data_obs, D, v1 = args
#         data_est = simple_net(params, v1)
#         residuals = jnp.array(data_est) - jnp.array(data_obs)
#         out = (residuals.T @ jnp.linalg.solve(D, residuals))
#     return out


def objective_wssr(params, args):
    data_obs, D, v1 = args

    def penalty(_):
        return 1e5

    def compute(_):
        data_est = simple_net(params, v1)
        residuals = data_est - data_obs
        return residuals.T @ jnp.linalg.solve(D, residuals)

    return jax.lax.cond(jnp.any(params < 0), penalty, compute, operand=None)


def objective_clr_wssr(params, args):
    data_obs, D, v1 = args

    def penalty(_):
        return 1e5

    def compute(_):
        data_est = simple_net(parameters=params, v1=v1)
        residuals = jnp.array(clr_func(data_est) - data_obs)
        return residuals.T @ jnp.linalg.solve(D, residuals)

    return jax.lax.cond(jnp.any(params < 0), penalty, compute, operand=None)

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

def gini_coefficient(x):
    """
    Calculate Gini coefficient of a list or array.
    Works for proportions or raw values.
    """
    x = np.array(x, dtype=float)
    
    if np.amin(x) < 0:
        raise ValueError("Values must be non-negative")
    
    if np.sum(x) == 0:
        return 0
    
    # Sort values
    x = np.sort(x)
    n = len(x)
    
    # Compute Gini
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n
    
    return gini