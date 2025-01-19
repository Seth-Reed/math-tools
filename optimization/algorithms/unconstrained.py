import jax
import jax.numpy as jnp
from tqdm import tqdm
from jax.config import config
config.update("jax_enable_x64", True)


def levenberg_marquardt(residuals, x_0: jnp.array, tol=1e-6, k_max=100, monitor=True):
    # Initialize parameters
    x_k = jnp.array(x_0)
    lambda_k = 1

    # Initialize residual and jacobian
    f = residuals
    Df = jax.jacfwd(residuals)
    f_k = f(x_0)
    Df_k = Df(x_0)

    # Begin optimization algorithm
    if monitor:
        loop = tqdm(range(k_max), desc='Levenberg-Marquardt Optimization Progress')
    else:
        loop = range(k_max)
    for k in loop:
        if monitor:
            # time.sleep(0.5)
            loop.set_postfix({'|f_k|': jnp.linalg.norm(f_k),
                              '|Df_k.T @ f_k|': jnp.linalg.norm(Df_k.T @ f_k),
                              'lambda_k': lambda_k})

        # Compute residual and its jacobian
        f_k = residuals(x_k)
        Df_k = Df(x_k)

        # Compute candidate solution
        A = Df_k.T @ Df_k + lambda_k * jnp.eye(len(x_0))
        b = Df_k.T @ f_k
        delta_x = jnp.linalg.solve(A, b)
        x_candidate = x_k - delta_x

        # Accept or reject candidate solution
        if jnp.linalg.norm(f(x_candidate)) < jnp.linalg.norm(f_k):
            x_k = x_candidate
            lambda_k *= 0.8
        else:
            lambda_k *= 2

        # Check stopping conditions
        if jnp.linalg.norm(Df_k.T @ f_k) < tol or jnp.linalg.norm(f_k) < tol or jnp.linalg.norm(delta_x) < tol:
            break
    return x_k


def residual(x: jnp.ndarray):
    res = [x[0] - 1,
           x[1] - 2,
           x[2] - 3,
           x[3] - 4]
    return jnp.array(res)


if __name__ == '__main__':
    x_0 = jnp.array([0., 0., 0., 0.])
    tol = 1e-6
    k_max = 100

    sol = levenberg_marquardt(residual, x_0, tol, k_max)
    print(sol)


