import jax
import jax.numpy as jnp
import numpy as np
import sympy
from jax import config

config.update("jax_enable_x64", True)


def toy_network(parameters: jax.Array, v1: float) -> jax.Array:
    """
    Takes vector of fluxes and return the MID for F123 in a toy metabolic network

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

