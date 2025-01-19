import numpy as np
import matplotlib.pyplot as plt
from dragonfly.functions.optimization.optimize import OptimizationProblem


def residuals(x):
    return [x[0] + x[1] - 9,
            x[1] - 2]


def constraints(x):
    return [x[0] - x[1]]


if __name__ == '__main__':
    optimization_problem = OptimizationProblem(residuals=residuals, constraints=constraints)

    (solution_path,
     lagrange_path,
     DL_DX_history,
     DL_DZ_history,
     alpha_history,
     residual_history,
     constraint_history) = optimization_problem.method_of_lagrange_multipliers(x_0=np.array([0., 0.5]),
                                                                               z_0=np.array([0.]),
                                                                               tol=1e-6,
                                                                               i_max=1000)

    # PLOTTING RESULTS

    # solution path plot
    pad = 1
    x0_range = np.linspace(min(solution_path[:, 0])-pad, max(solution_path[:, 0])+pad)
    x1_range = np.linspace(min(solution_path[:, 1])-pad, max(solution_path[:, 1]+pad))
    X0, X1 = np.meshgrid(x0_range, x1_range)
    OBJ = optimization_problem.objective(np.array([X0, X1]))

    plt.figure()
    # x and y axis
    plt.axhline(0, color='black', linewidth=1.)
    plt.axvline(0, color='black', linewidth=1.)
    # objective
    contours = plt.contour(X0, X1, OBJ, levels=30, cmap='viridis')
    # constraints
    plt.plot(x0_range, x0_range, 'b--', label='Constraint: $x_0 - x_1 = 0$')
    # solution progress
    plt.plot(solution_path[:, 0], solution_path[:, 1], 'b-o', markersize=3, label='Path of $x$')
    plt.plot(solution_path[-1, 0], solution_path[-1, 1], 'r*', markersize=15, label='Converged Point')
    # plot features
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    plt.xlim([min(x0_range), max(x0_range)])
    plt.ylim([min(x1_range), max(x1_range)])
    plt.grid(which='both')
    plt.legend()
    plt.title('Solution State Path During Optimization')

    # convergence plot
    _, ax = plt.subplots(2, 1, figsize=(8, 8))
    ax[0].plot(np.linalg.norm(DL_DX_history, axis=1), 'b-o', markersize=4)
    ax[1].plot(np.linalg.norm(DL_DZ_history, axis=1), 'b-o', markersize=4)

    for i, axis in enumerate(ax):
        axis.minorticks_on()
        axis.grid(which='minor', axis='both')
        axis.grid(which='major')
        axis.set_xlabel('Iteration, i')
    ax[0].set_yscale('log')
    ax[0].set_ylabel(r'$\left\|\frac{\partial L}{\partial x}\right\|$', fontsize=16)
    ax[1].set_yscale('log')
    ax[1].set_ylabel(r'$\left\|\frac{\partial L}{\partial z}\right\|$', fontsize=16)
    ax[0].set_title('Optimality Conditions vs Iteration')

    plt.show()