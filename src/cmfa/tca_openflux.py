import jax
import jax.numpy as jnp
from scipy import stats

from cmfa.emu_example import clr_func


def flux_vector(parameters: dict, fixed: dict):
    # parameters = {'v8': ..., 'v2': ..., 'v9': ...}
    # fixed = {'v1': ..., 'v16': ..., 'v17': ..., ...}
    v1 = fixed["v1"]
    v16 = fixed["v16"]
    v17 = fixed["v17"]
    v18 = fixed["v18"]
    v19 = fixed["v19"]
    v20 = fixed["v20"]

    v8 = parameters["v8"]
    v2 = parameters["v2"]
    v9 = parameters["v9"]

    v_free = jnp.array([v8, v1, v19, v20, v16, v17, v18, v2, v9])
    M_irr = jnp.array(
        [
            [1, -2, -2, -1, -1, -1, 1, 0],
            [1, -2, -2, -1, -1, -1, 1, 0],
            [1, -2, -2, -1, -1, -1, 1, 0],
            [1, -2, -2, -1, -2, -1, 2, 0],
            [2, -4, -4, -2, -3, -2, 3, 0],
            [0.5, -1, -1, -0.5, -1, -0.5, 1, 0],
            [2.5, -5, -5, -2.5, -4, -2.5, 4, 0],
            [3, -5, -6, -3, -5, -4, 5, 0],
            [10, -20, -21, -10, -17, -11, 17, -1],
            [0, 0, 1, 0, 1, 1, -1, 1],
        ]
    )

    z_M = jnp.column_stack((jnp.zeros(10), M_irr))

    M_rev = jnp.array([[1, -2, -2, -1, -2, -1, 2, 0]])

    i_M = jnp.column_stack((jnp.array([1]), M_rev))
    # Columns correspond to: v8, v1, v19, v20, v16, v17, v18, v2, v9
    NS = jnp.array(jnp.vstack((jnp.identity(9), z_M, i_M)))
    order_values = NS @ v_free
    order_keys = [
        "v8",
        "v1",
        "v19",
        "v20",
        "v16",
        "v17",
        "v18",
        "v2",
        "v9",
        "v3",
        "v4",
        "v5",
        "v6",
        "v11",
        "v12",
        "v13",
        "v14",
        "v15",
        "v10",
        "v7",
    ]
    out = dict(zip(order_keys, order_values))
    return out


def TCA_OpenFLux(parameters: dict, fixed: dict) -> dict:
    v = flux_vector(parameters=parameters, fixed=fixed)
    names = [
        "v1",
        "v2",
        "v3",
        "v4",
        "v5",
        "v6",
        "v7",
        "v8",
        "v9",
        "v10",
        "v11",
        "v12",
        "v13",
        "v14",
        "v15",
        "v16",
        "v17",
        "v18",
        "v19",
        "v20",
    ]
    (
        v1,
        v2,
        v3,
        v4,
        v5,
        v6,
        v7,
        v8,
        v9,
        v10,
        v11,
        v12,
        v13,
        v14,
        v15,
        v16,
        v17,
        v18,
        v19,
        v20,
    ) = [v[n] for n in names]

    A1 = jnp.array(
        [
            [
                -v["v3"] - v5 - v6 - v9 - v19 - v20,
                v3 + v19 + v20,
                v6,
                v9,
                0,
                v5,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [0, -v1 - v9, 0, 0, 0, v9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -v2 - v5, 0, 0, 0, 0, 0, 0, 0, 0, 0, v5, 0, 0, 0, 0],
            [v10, 0, 0, -v7 - v10, 0, 0, v7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -v1 - v9, 0, 0, v9, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, v10, 0, 0, 0, -v7 - v10, 0, 0, v7, 0, 0, 0, 0, 0, 0, 0, 0],
            [
                0,
                0,
                0,
                0.5 * v8,
                0,
                0.5 * v8,
                -v6 - v8,
                0,
                0,
                0.5 * v6,
                0.5 * v6,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [0, 0, 0, 0, v10, 0, 0, -v7 - v10, 0, 0, 0, v7, 0, 0, 0, 0, 0],
            [
                0,
                0,
                0,
                0.5 * v8,
                0,
                0.5 * v8,
                0,
                0,
                -v6 - v8,
                0.5 * v6,
                0.5 * v6,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [0, 0, 0, 0, v5, 0, 0, 0, 0, -v2 - v5, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, v5, 0, 0, 0, 0, 0, 0, -v2 - v5, 0, 0, 0, 0, 0, 0],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.5 * v8,
                0,
                0,
                0,
                -v6 - v8,
                0,
                0.5 * v6,
                0.5 * v6,
                0.5 * v8,
                0,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -v1 - v9, 0, 0, v9, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -v2 - v5, 0, v5, 0],
            [0, 0, 0, 0, 0, 0, 0, v5, 0, 0, 0, 0, 0, 0, -v2 - v5, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, v10, 0, 0, -v7 - v10, v7],
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.5 * v8,
                0,
                0,
                0,
                0,
                0,
                0.5 * v6,
                0.5 * v6,
                0.5 * v8,
                -v6 - v8,
            ],
        ]
    )

    A2 = jnp.array(
        [
            [-v1 - v9, v9, 0, 0],
            [v10, -v7 - v10, v7, 0],
            [0, v8, -v6 - v8, v6],
            [0, v5, 0, -v2 - v5],
        ]
    )

    A3 = jnp.array(
        [
            [-v1 - v9, 0, v9, 0, 0, 0, 0, 0],
            [0, -v7 - v10, 0, v7, 0, 0, 0, 0],
            [v10, 0, -v7 - v10, 0, v7, 0, 0, 0],
            [0, 0.5 * v8, 0.5 * v8, -v6 - v8, 0, 0.5 * v6, 0.5 * v6, 0],
            [0, 0.5 * v8, 0.5 * v8, 0, -v6 - v8, 0.5 * v6, 0.5 * v6, 0],
            [0, 0, 0, 0, 0, -v2 - v5, 0, v5],
            [0, v5, 0, 0, 0, 0, -v2 - v5, 0],
            [0, 0, 0, 0, 0, 0, 0, -v4],
        ]
    )

    A4 = jnp.array(
        [
            [-v7 - v10, v7, 0, 0],
            [v8, -v6 - v8, v6, 0],
            [0, 0, -v2 - v5, v5],
            [0, 0, 0, -v4],
        ]
    )

    B1 = jnp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [-v1, 0, 0, 0, 0, 0, 0, 0],
            [0, -v2, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -v1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, -v2, 0, 0, 0, 0],
            [0, 0, 0, 0, -v2, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -v1, 0, 0],
            [0, 0, 0, 0, 0, 0, -v2, 0],
            [0, 0, 0, 0, 0, 0, 0, -v2],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    B2 = jnp.array(
        [
            [-v1, 0],
            [0, 0],
            [0, 0],
            [0, -v2],
        ]
    )

    B3 = jnp.array(
        [
            [-v1, 0, 0, 0, 0],
            [0, -v10, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, -v2, 0, 0],
            [0, 0, 0, -v2, 0],
            [0, 0, 0, 0, -v4],
        ]
    )

    B4 = jnp.array(
        [
            [-v10, 0, 0],
            [0, 0, 0],
            [0, -v2, 0],
            [0, 0, -v4],
        ]
    )

    # input substrates
    pyr_ex_1 = jnp.array([0.01, 0.99])
    pyr_ex_2 = jnp.array([0.49965, 0.50035])
    pyr_ex_3 = jnp.array([0.49965, 0.50035])

    pyr_ex_12 = jnp.convolve(pyr_ex_1, pyr_ex_2)
    pyr_ex_13 = jnp.convolve(pyr_ex_1, pyr_ex_3)
    pyr_ex_23 = jnp.convolve(pyr_ex_2, pyr_ex_3)

    pyr_ex_123 = jnp.convolve(pyr_ex_12, pyr_ex_3)

    glu_ex_1 = jnp.array([0.01, 0.99])
    glu_ex_2 = glu_ex_3 = glu_ex_4 = glu_ex_5 = jnp.array([0.9893, 0.0107])

    glu_ex_23 = jnp.convolve(glu_ex_2, glu_ex_3)

    glu_ex_123 = jnp.convolve(glu_ex_1, glu_ex_23)
    glu_ex_234 = jnp.convolve(glu_ex_23, glu_ex_4)

    glu_ex_1234 = jnp.convolve(glu_ex_123, glu_ex_4)

    Y1 = jnp.array(
        [
            pyr_ex_1,
            glu_ex_5,
            pyr_ex_3,
            glu_ex_4,
            glu_ex_1,
            pyr_ex_2,
            glu_ex_3,
            glu_ex_2,
        ]
    )
    X1 = jnp.linalg.solve(A1, B1 @ Y1)
    (
        co2_1,
        pyr_1,
        akg_5,
        oaa_4,
        pyr_3,
        oaa_1,
        suc_4,
        oaa_3,
        suc_1,
        akg_4,
        akg_1,
        suc_3,
        pyr_2,
        akg_3,
        akg_2,
        oaa_2,
        suc_2,
    ) = X1

    Y2 = jnp.array([pyr_ex_23, glu_ex_23])

    X2 = jnp.linalg.solve(A2, B2 @ Y2)
    pyr_23, oaa_23, suc_23, akg_23 = X2

    Y3 = jnp.array(
        [
            pyr_ex_123,
            jnp.convolve(pyr_23, co2_1),
            glu_ex_234,
            glu_ex_123,
            jnp.convolve(oaa_23, pyr_3),
        ]
    )

    X3 = jnp.linalg.solve(A3, B3 @ Y3)
    pyr_123, oaa_234, oaa_123, suc_234, suc_123, akg_234, akg_123, ici_234 = X3

    Y4 = jnp.array(
        [
            jnp.convolve(pyr_123, co2_1),
            glu_ex_1234,
            jnp.convolve(oaa_234, pyr_3),
        ]
    )

    X4 = jnp.linalg.solve(A4, B4 @ Y4)
    oaa_1234, suc_1234, akg_1234, ici_1234 = X4

    lysx_123456 = jnp.convolve(oaa_1234, pyr_23)
    valx_12345 = jnp.convolve(pyr_123, pyr_23)
    aspx_1234 = oaa_1234
    return {
        "lysx_123456": lysx_123456,
        "valx_12345": valx_12345,
        "aspx_1234": aspx_1234,
        "suc_1234": suc_1234,
    }


def objective_wssr(params: dict, args):
    data_obs, D_list, fixed = args
    outputs = TCA_OpenFLux(params, fixed)
    residuals = jax.tree.map(lambda e, o: e - o, outputs, data_obs)
    total = sum(
        r.T @ jnp.linalg.inv(D) @ r
        for r, D in zip(jax.tree.leaves(residuals), jax.tree.leaves(D_list))
    )
    return total


def objective_clr_wssr(params: dict, args):
    data_obs, D_list, fixed = args
    outputs = TCA_OpenFLux(params, fixed)
    outputs_clr = jax.tree.map(lambda x: clr_func(x), outputs)
    residuals = jax.tree.map(lambda e, o: e - o, outputs_clr, data_obs)
    total = sum(
        r.T @ jnp.linalg.inv(D) @ r
        for r, D in zip(jax.tree.leaves(residuals), jax.tree.leaves(D_list))
    )
    return total


def calc_ci(opti_solution, confidence_level=0.95) -> dict:
    hess_inv_dict = opti_solution.state.f_info.hessian_inv.pytree
    standard_errors = {
        k: jnp.sqrt(hess_inv_dict[k][k]) for k in hess_inv_dict.keys()
    }

    # Parameters from your optimization
    params = opti_solution.value
    alpha = 1 - confidence_level
    # Critical value (Z-score for 95%)
    z = stats.norm.ppf(1 - alpha / 2)
    ci_results = {}
    for k in params.keys():
        se = standard_errors[k]
        val = params[k]

        ci_results[k] = {
            "mean": val,
            "std_error": se,
            "lower": val - z * se,
            "upper": val + z * se,
        }
    return ci_results
