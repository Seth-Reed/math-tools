from dragonfly.functions.optimization.algorithms.unconstrained import levenberg_marquardt
import jax.numpy as jnp
import jax
import numpy as np
from tqdm import tqdm
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)


def lagrangian_objective(x, z, f, g):
    return f(x).T @ f(x) + z.T @ g(x)


def lagrange_multipliers_algorithm(residuals, constraints, x_0, tol=1e-6, k_max=100):

    lagrangian = partial(lagrangian_objective, f=residuals, g=constraints)
    # dL_dx = jax.grad(lagrangian, 0)
    # dL_dz = jax.grad(lagrangian, 1)
    H_L_x = jax.hessian(lagrangian)
    Df = jax.jacfwd(residuals)
    Dg = jax.jacfwd(constraints)

    x_k = x_0.copy()
    z_k = jnp.zeros(len(constraints(x_0)))

    loop = tqdm(range(k_max), desc="Optimization Progress")
    for k in loop:
        # Compute Lagrangian gradients and hessian
        # dL_dx_k = dL_dx(x_k, z_k)
        # dL_dz_k = dL_dz(x_k, z_k)  # alternate definitions
        dL_dx_k = 2 * Df(x_k).T @ residuals(x_k) + Dg(x_k).T @ z_k
        dL_dz_k = constraints(x_k)
        H_L_x_k = H_L_x(x_k, z_k)

        # Compute step size
        alpha_k = 1 / max(jnp.real(jnp.linalg.eigvals(H_L_x_k)))

        # Update x and z
        x_k -= alpha_k * dL_dx_k
        z_k += alpha_k * dL_dz_k

        loop.set_postfix({'x_k = ': x_k,
                          'z_k = ': z_k,
                          '|dL_dx|': np.linalg.norm(dL_dx_k),
                          '|dL_dz|': np.linalg.norm(dL_dz_k),
                          'alpha_k': alpha_k})

        # Check stopping condition
        if np.linalg.norm(dL_dx_k) < tol and np.linalg.norm(dL_dz_k) < tol:
            print(f"Converged after {k + 1} iterations.")
            break

    return x_k


def penalty_objective(x, f, g, mu):
    return jnp.concatenate((f(x), np.sqrt(mu) * g(x)))


def penalty_algorithm(residuals, constraints, x_0, tol=1e-6, k_max=100, tol_LM=1e-6, k_max_LM=1000):

    penalty = partial(penalty_objective, f=residuals, g=constraints)
    x_k = x_0.copy()
    mu_k = 1

    loop = tqdm(range(k_max), desc="Penalty Algorithm Optimization Progress")
    for i in loop:
        # Solve unconstrained nonlinear least squares problem with composite objective
        x_k = levenberg_marquardt(residuals=partial(penalty, mu=mu_k),
                                  x_0=x_k,
                                  k_max=k_max_LM,
                                  tol=tol_LM,
                                  monitor=True)

        # Update penalty
        mu_k *= 2

        # Show data
        loop.set_postfix({'residual': np.linalg.norm(residuals(x_k)),
                          'constraint': np.linalg.norm(constraints(x_k)),
                          'solution': x_k,
                          'penalty': mu_k})

        # Check stopping condition
        if np.linalg.norm(constraints(x_k)) < tol:
            print(f"Converged after {i + 1} iterations.")
            break

    return x_k


def augmented_lagrangian_objective(x, z, mu, f, g):
    return jnp.hstack((f(x),
                       jnp.sqrt(mu) * g(x) + z / (2 * jnp.sqrt(mu))))


def augmented_lagrangian_algorithm(residuals, constraints, x_0, tol=1e-6, k_max=100, tol_LM=1e-6, k_max_LM=100):
    x_km1 = jnp.array(x_0)
    x_k = jnp.zeros_like(x_0)
    z_k = jnp.zeros(len(constraints(x_0)))
    mu_k = 1.

    loop = tqdm(range(1, k_max), desc="Augmented Lagrangian Algorithm Optimization Progress")
    for k in loop:
        loop.set_postfix({'|g_k|': np.linalg.norm(constraints(x_k)),
                          '|x_k - x_km1|': np.linalg.norm(x_k - x_km1)})

        # Solve unconstrained nonlinear least squares problem with composite objective
        x_k = levenberg_marquardt(residuals=partial(augmented_lagrangian_objective,
                                                    z=z_k, mu=mu_k, f=residuals, g=constraints),
                                  x_0=x_km1,  # warm start
                                  tol=tol_LM,
                                  k_max=k_max_LM,
                                  monitor=True)

        # Update lagrangian
        z_k += 2 * mu_k * constraints(x_k)

        # Update penalty only when ||g(x_k)|| does not sufficiently decrease
        if np.linalg.norm(constraints(x_k)) >= 0.25 * np.linalg.norm(constraints(x_km1)):
            mu_k *= 2

        # Check stopping condition
        if np.linalg.norm(constraints(x_k)) < tol and np.linalg.norm(x_k - x_km1) < tol:
            break

        # Track previous x for penalty update
        x_km1 = x_k

    return x_k


def residuals(x):
    return jnp.array([x[0] - 1,
                      x[1] - 2,
                      x[2] - 3,
                      x[3] - 4])


def constraints(x):
    return jnp.array([x[0] - x[1],
                      x[2] - x[3]])


if __name__ == '__main__':
    x_0 = jnp.array([0., 0., 0., 0.])
    tol = 1e-6
    k_max = 50

    # sol = augmented_lagrangian_algorithm(residuals, constraints, x_0, tol, k_max)
    sol = lagrange_multipliers_algorithm(residuals, constraints, x_0, tol, k_max)
    # sol = penalty_algorithm(residuals, constraints, x_0, tol, k_max)

    print(sol)








