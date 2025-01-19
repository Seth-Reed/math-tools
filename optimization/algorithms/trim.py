from scipy.optimize import minimize
import numpy as np


def trim_optimization(initial_conditions, residual_function, constraint_function=None, method='SLSQP', bounds=None, options=None, tol=1e-12):
    if constraint_function is None:
        objective = lambda z: np.sum(residual_function(z) ** 2)
        result = minimize(fun=objective,
                          x0=initial_conditions,
                          method=method,
                          bounds=bounds,
                          options=options,
                          tol=tol)

    else:
        constraint = ({'type': 'eq', 'fun': lambda z: constraint_function(z)})
        objective = lambda z: np.sum(residual_function(z) ** 2)
        result = minimize(fun=objective,
                          x0=initial_conditions,
                          method=method,
                          bounds=bounds,
                          constraints=constraint,
                          options=options,
                          tol=tol)

    return result


def print_optimization_result(result):
    print('OPTIMIZATION RESULTS')
    if result.success:
        print("Optimization succeeded.")
        print("Optimal values:", result.x)
        print("Objective function value at optimum:", result.fun)
        print("Objective function jacobian at optimum:", result.jac)
    else:
        print("Optimization failed.")
        print("Reason:", result.frame)
        print("Last values:", result.x)
        print("Objective function value at last iteration:", result.fun)
        print("Objective function jacobian at last iteration", result.jac)

    print("Number of function evaluations:", result.nfev)
    print("Number of iterations:", result.nit)
    print()
