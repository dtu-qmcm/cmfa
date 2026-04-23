import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

m = jnp.array([[-1, -1, 2], [0, 0, -1], [-1, 0, 1]])
n = jnp.vstack([-m, jnp.eye(3)])

s = jnp.array([
    [-1, -1, 0, 1, 1, -1],
    [0, 1, 0, 0, 0, -1],
    [1, 0, -1, 0, -1, 1],
], dtype=float)

n_nov1 = jnp.delete(n, 0, axis=1)

inv_hess = jnp.array([[435.37893171,  22.66313657],[22.66313657, 1.53734538]])
inv_hess_v1 = jnp.array([[0,0,0], [0, 435.37893171,  22.66313657], [0, 22.66313657,1.53734538]])

print(f"full: {jnp.diag(n@inv_hess_v1@n.T)}")

print(f"without v1: {jnp.diag(n_nov1@inv_hess@n_nov1.T)}")