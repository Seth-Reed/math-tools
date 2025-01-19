import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm


def levenberg_marquardt(residual_func, x_0: jnp.ndarray, i_max=1000, tol=1e-6, gauss_newton=False):
    # Initialize parameters
    if gauss_newton:
        lambda_0 = 0.
    else:
        lambda_0 = 1.
    x = x_0.copy()
    lambda_i = lambda_0
    DF_DX = jax.jacfwd(residual_func)

    for i in range(i_max):
        # Compute residual and its jacobian
        F_X_i = np.array(residual_func(x))
        DF_DX_i = np.array(DF_DX(x))

        # Compute candidate solution
        A = DF_DX_i.T @ DF_DX_i + lambda_i * np.eye(len(x))
        b = DF_DX_i.T @ F_X_i
        delta_x = np.linalg.solve(A, b)
        x_candidate = x - delta_x

        # Accept or reject candidate solution
        if gauss_newton:
            x = np.array(x_candidate)
        elif jnp.linalg.norm(residual_func(x_candidate)) < jnp.linalg.norm(residual_func(x)):
            x = np.array(x_candidate)
            lambda_i *= 0.8
        else:
            lambda_i *= 2

        # Check stopping condition
        if jnp.linalg.norm(DF_DX_i.T @ F_X_i) < tol or jnp.linalg.norm(F_X_i) < tol:
            break

    return x
