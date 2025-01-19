import jax.numpy as jnp
import numpy as np
import jax
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import minimize


def residual(x):
    return [x[0] + x[1] - 9,
            x[1] - 2]


def constraints(x):
    return [x[0] - x[1]]


def lagrangian(x, z):
    lagrangian = 0.
    for i in range(len(residual(x))):
        lagrangian += (residual(x)[i])**2
    for i in range(len(z)):
        lagrangian += z[i] * constraints(x)[i]
    return lagrangian


dL_dx = jax.grad(lagrangian, 0)
dL_dz = jax.grad(lagrangian, 1)
H_L_x = jax.hessian(lagrangian)

x_0 = jnp.array([-2., 1.5])
z_0 = jnp.array([0.])

alpha = 0.01  # Step size
tol = 1e-6  # Convergence tolerance
max_iter = 1000  # Maximum iterations

x = x_0.copy()
z = z_0.copy()
dL_dx_i, dL_dz_i = np.array(dL_dx(x, z)), np.array(dL_dz(x, z))
H_L_x_i = np.array(H_L_x(x, z))

x_path = [x_0.copy()]
loop = tqdm(range(max_iter), desc="Optimization Progress")
for i in loop:
    # Compute Lagrangian gradients and hessian
    dL_dx_i = np.array(dL_dx(x, z))
    dL_dz_i = np.array(dL_dz(x, z))
    H_L_x_i = np.array(H_L_x(x, z))

    # Compute step size
    alpha = 1 / max(np.linalg.eigvals(H_L_x_i))

    # Update x and z
    x -= alpha * dL_dx_i
    z += alpha * dL_dz_i
    x_path.append(x)
    loop.set_postfix({'x = ': x,
                      'z = ': z,
                      '|dL_dx|': np.linalg.norm(dL_dx_i),
                      '|dL_dz|': np.linalg.norm(dL_dz_i),
                      'alpha': alpha,
                      '|x[i]-x[i-1]|': np.linalg.norm(x_path[-1]-x_path[-2])})

    # Check stopping condition
    if (np.linalg.norm(dL_dx_i) < tol and np.linalg.norm(dL_dz_i) < tol) or np.linalg.norm(x_path[-1]-x_path[-2] == 0.):
        print(f"Converged after {i + 1} iterations.")
        break

print()
print(f'OPTIMAL SOLUTION:             {x}')
print()
print(f'OPTIMAL LAGRANGE MULTIPLIERS: {z}')
print(f'OPTIMALITY CONDITIONS: |dL_dx| = {dL_dx_i},')
print(f'                       |dL_dz| = {dL_dz_i}')
print()


def objective(x):
    return np.linalg.norm(residual(x))**2


c = {'type': 'eq', 'fun': constraints}
result = minimize(objective, x0=np.array(x_0), constraints=c)
print(result)

x_path = np.array(x_path)

x1_values = np.linspace(-1, 3, 400)
x2_values = np.linspace(-1, 5, 400)
X1, X2 = np.meshgrid(x1_values, x2_values)
Z = residual(np.array([X1, X2]))

plt.figure()
# x and y axis
plt.axhline(0, color='black', linewidth=1.)
plt.axvline(0, color='black', linewidth=1.)
# objective
contours = plt.contour(X1, X2, Z, levels=20, cmap='viridis')
# constraints
plt.plot(x1_values, x1_values, 'b--', label='Constraint: $x_0 - x_1 = 0$')
# solution progress
plt.plot(x_path[:, 0], x_path[:, 1], '-o', markersize=4, color='blue', label='Path of $x$')
plt.plot(x_path[-1, 0], x_path[-1, 1], 'r*', markersize=10, label='Converged Point')
# plot features
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim([min(x1_values), max(x1_values)])
plt.ylim([min(x2_values), max(x2_values)])
plt.title('Path of $x$ with Objective Contours and Constraint Line')
plt.legend()
plt.show()



