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
    return jnp.concatenate((f(x[0:2]), jnp.sqrt(mu)*g(x) + z/(2 * jnp.sqrt(mu))))


x_0 = np.array([3.9, 3.9])
z_0 = np.array([0.])
mu = 1

x = x_0.copy()
z = z_0
solution_path = [x.copy()]

i_max = 10
tol = 1e-6
loop = tqdm(range(i_max), desc="Augmented Lagrangian Algorithm Optimization Progress")
for i in loop:
    # Track previous x for penalty update
    x_prev = x

    # Solve unconstrained nonlinear least squares problem with composite objective
    x = levenberg_marquardt(residual_func=composite_objective,
                            x_0=x,
                            i_max=1000,
                            tol=1e-6)

    # Update lagrangian
    z += 2 * mu * g(x)

    # Update penalty only when ||g(x)|| does not sufficiently decrease
    if np.linalg.norm(g(x)) >= 0.25 * np.linalg.norm(g(x_prev)):
        mu *= 2

    # Save data
    solution_path.append(x.copy())
    loop.set_postfix({'residual': np.linalg.norm(f(x)),
                      'constraint': np.linalg.norm(g(x)),
                      'solution': x,
                      'multiplier': z,
                      'penalty': mu})

    # Check stopping condition
    if np.linalg.norm(g(x)) < tol and np.linalg.norm(x-x_prev) == 0.:
        print(f"Converged after {i + 1} iterations.")
        break

# Output Result
print()
print(f'OPTIMAL SOLUTION: {x}')