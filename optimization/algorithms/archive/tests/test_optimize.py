import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from dragonfly.functions.optimization.optimize import OptimizationProblem


def residuals(x):
    return jnp.array([x[0] + x[1] - 9,
                    x[1] - 2])


def constraints(x):
    return jnp.array([x[0] - x[1]])


if __name__ == '__main__':
    optimization_problem = OptimizationProblem(residuals=residuals, constraints=constraints)
    x_0 = np.array([0., 0.5])

    # METHOD OF LAGRANGE MULTIPLIERS
    (LM_solution_path,
     LM_lagrange_path,
     LM_DL_DX_history,
     LM_DL_DZ_history,
     LM_alpha_history,
     LM_residual_history,
     LM_constraint_history) = optimization_problem.method_of_lagrange_multipliers(x_0=x_0,
                                                                                  tol=1e-6,
                                                                                  i_max=1000)
    #
    # # PENALTY ALGORITHM
    # (PA_solution_path,
    #  PA_penalty_history,
    #  PA_residual_history,
    #  PA_constraint_history) = optimization_problem.penalty_algorithm(x_0=np.array([0., 0.5]),
    #                                                                  tol=1e-6,
    #                                                                  i_max=10)
    #
    # # AUGMENTED LAGRANGIAN ALGORITHM
    # (AL_solution_path,
    #  AL_lagrange_path,
    #  AL_penalty_history,
    #  AL_residual_history,
    #  AL_constraint_history) = optimization_problem.augmented_lagrangian_algorithm(x_0=np.array([0., 0.5]),
    #                                                                               tol=1e-6,
    #                                                                               i_max=10)
    #
    # np.savez('test_save_outputs.npz',
    #          LM_solution_path=LM_solution_path,
    #          LM_lagrange_path=LM_lagrange_path,
    #          LM_DL_DX_history=LM_DL_DX_history,
    #          LM_DL_DZ_history=LM_DL_DZ_history,
    #          LM_alpha_history=LM_alpha_history,
    #          LM_residual_history=LM_residual_history,
    #          LM_constraint_history=LM_constraint_history,
    #          PA_solution_path=PA_solution_path,
    #          PA_penalty_history=PA_penalty_history,
    #          PA_residual_history=PA_residual_history,
    #          PA_constraint_history=PA_constraint_history,
    #          AL_solution_path=AL_solution_path,
    #          AL_lagrange_path=AL_lagrange_path,
    #          AL_penalty_history=AL_penalty_history,
    #          AL_residual_history=AL_residual_history,
    #          AL_constraint_history=AL_constraint_history)

    data = np.load('test_save_outputs.npz')

    LM_solution_path = data['LM_solution_path']
    LM_lagrange_path = data['LM_lagrange_path']
    LM_DL_DX_history = data['LM_DL_DX_history']
    LM_DL_DZ_history = data['LM_DL_DZ_history']
    LM_alpha_history = data['LM_alpha_history']
    LM_residual_history = data['LM_residual_history']
    LM_constraint_history = data['LM_constraint_history']
    PA_solution_path = data['PA_solution_path']
    PA_penalty_history = data['PA_penalty_history']
    PA_residual_history = data['PA_residual_history']
    PA_constraint_history = data['PA_constraint_history']
    AL_solution_path = data['AL_solution_path']
    AL_lagrange_path = data['AL_lagrange_path']
    AL_penalty_history = data['AL_penalty_history']
    AL_residual_history = data['AL_residual_history']
    AL_constraint_history = data['AL_constraint_history']

    # PLOTTING RESULTS

    # solution path plot
    pad = 1
    solution_paths = np.concatenate((LM_solution_path, PA_solution_path, AL_solution_path))
    x0_range = np.linspace(min(solution_paths[:, 0]) - pad, max(solution_paths[:, 0]) + pad)
    x1_range = np.linspace(min(solution_paths[:, 1]) - pad, max(solution_paths[:, 1] + pad))
    X0, X1 = np.meshgrid(x0_range, x1_range)
    OBJ = optimization_problem.objective(np.array([X0, X1]))

    plt.figure()
    # x and y axis
    plt.axhline(0, color='black', linewidth=1.)
    plt.axvline(0, color='black', linewidth=1.)
    # objective
    contours = plt.contourf(X0, X1, OBJ, levels=30, cmap='YlOrRd')
    # constraints
    plt.plot(x0_range, x0_range, 'b--', label='Constraint: $x_0 - x_1 = 0$')
    # solution progress
    plt.plot(LM_solution_path[:, 0], LM_solution_path[:, 1], '-o', markersize=5, label='Lagrange Multipliers')
    plt.plot(PA_solution_path[:, 0], PA_solution_path[:, 1], '-o', markersize=5, label='Penalty Algorithm')
    plt.plot(AL_solution_path[:, 0], AL_solution_path[:, 1], '-o', markersize=5, label='Augmented Lagrangian')
    plt.plot(LM_solution_path[-1, 0], LM_solution_path[-1, 1], 'r*', markersize=15)
    plt.plot(PA_solution_path[-1, 0], PA_solution_path[-1, 1], 'r*', markersize=15)
    plt.plot(AL_solution_path[-1, 0], AL_solution_path[-1, 1], 'r*', markersize=15)
    # plot features
    plt.xlabel('$x_0$')
    plt.ylabel('$x_1$')
    plt.xlim([min(x0_range), max(x0_range)])
    plt.ylim([min(x1_range), max(x1_range)])
    plt.grid(which='both')
    plt.legend()
    plt.title('Solution Optimization Path')
    plt.show()


