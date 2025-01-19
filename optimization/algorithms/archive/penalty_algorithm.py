import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from levenberg_marquardt import levenberg_marquardt


def f(x):
    return jnp.array([x[0] + x[1] - 9,
                      x[1] - 2])


def g(x):
    return jnp.array([x[0] - x[1]])


def composite_objective(x):
    return jnp.concatenate((f(x), np.sqrt(mu) * g(x)))


x_0 = np.zeros(2)
x = x_0.copy()
mu = 1
i_max = 10
tol = 1e-6
loop = tqdm(range(i_max), desc="Penalty Algorithm Optimization Progress")
for i in loop:
    # Solve unconstrained nonlinear least squares problem with composite objective
    x = levenberg_marquardt(residual_func=composite_objective,
                            x_0=x,
                            i_max=1000,
                            tol=1e-6)

    # Update penalty
    mu *= 2

    # Show data
    loop.set_postfix({'residual': np.linalg.norm(f(x)),
                      'constraint': np.linalg.norm(g(x)),
                      'solution': x,
                      'penalty': mu})

    # Check stopping condition
    if np.linalg.norm(g(x)) < tol:
        print(f"Converged after {i + 1} iterations.")
        break

# Output Result
print()
print(f'OPTIMAL SOLUTION: {x}')










